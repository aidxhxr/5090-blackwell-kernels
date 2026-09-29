// RoPE on q and k plus the K/V cache append, one launch, for a decoder layer's fused
// q|k|v projection output.
//
// In: qkv = [B, S, (H_q + 2 H_kv) * D] bf16, the q, k and v columns of one GEMM, and the
// rotary tables cos, sin = [>= pos0 + S, D] fp32 in the rotate-half layout Llama uses
// (column d pairs with d + D/2 and the tables repeat their first half, so only columns
// 0..D/2-1 are read). Out: q rotated as [B, H_q, S, D] (the head-major layout attention
// takes), k rotated and v copied into the caches [B, H_kv, cap, D] at positions
// pos0..pos0+S-1. In torch this is seven elementwise kernels for the rotation (the fp32
// upcast, the two halves, cat, two muls, an add, the bf16 cast), a transpose copy for q and
// two strided copies for the cache: 17 us of a 315 us decode step and 600 us of a 8.9 ms
// prefill on the RTX 5090. Here every byte is read once and written once.
//
// Work: one thread per 8 elements. A "rope row" is one (token, q or k head): D/16 threads,
// each rotating 8 elements of the first half against the 8 at +D/2 (two 16-byte loads, two
// 16-byte stores, two float4 loads of cos and sin). A "v row" is one (token, v head): D/8
// threads each copying 16 bytes. Rows of a token are consecutive in qkv, so a warp's loads
// cover whole 128-byte segments of the GEMM output.
//
// rope_append_paged_bf16 is the same work for packed tokens and a paged cache
// (docs/design/serving.md): token t rotates at positions[t], its k and v go to slot slots[t]
// of [num_pages, H_kv, page, D] caches, and q comes out token-major as [T, H_q, D].
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {
namespace {

constexpr int kBlock = 256;

struct Params {
    const __nv_bfloat16* qkv;
    const float* cos;
    const float* sin;
    __nv_bfloat16* q;
    __nv_bfloat16* k_cache;
    __nv_bfloat16* v_cache;
    int S, H_q, H_kv, D, pos0, cap;
    int64_t width;       // (H_q + 2 H_kv) * D, the qkv row
    int64_t rope_items;  // (H_q + H_kv) * D / 16 threads per token rotate
    int64_t per_tok;     // rope_items + H_kv * D / 8 (the v copy)
    int64_t total;       // B * S * per_tok
};

__device__ __forceinline__ float4 ld_f4(const float* p) {
    return *reinterpret_cast<const float4*>(p);
}

// Rotates the 8 elements at src against the 8 at src + half with 8 columns of the cos and
// sin tables: o1 = x1 cos - x2 sin, o2 = x2 cos + x1 sin (rotate_half: cat(-x2, x1)).
__device__ __forceinline__ void rotate8(const __nv_bfloat16* src, int half, const float* cs,
                                        const float* sn, bf16x8& o1, bf16x8& o2) {
    const bf16x8 x1 = *reinterpret_cast<const bf16x8*>(src);
    const bf16x8 x2 = *reinterpret_cast<const bf16x8*>(src + half);
    const float4 c0 = ld_f4(cs), c1 = ld_f4(cs + 4);
    const float4 s0 = ld_f4(sn), s1 = ld_f4(sn + 4);
    const float cv[8] = {c0.x, c0.y, c0.z, c0.w, c1.x, c1.y, c1.z, c1.w};
    const float sv[8] = {s0.x, s0.y, s0.z, s0.w, s1.x, s1.y, s1.z, s1.w};
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float2 a = __bfloat1622float2(x1.h[k]);
        const float2 bb = __bfloat1622float2(x2.h[k]);
        float2 r1, r2;
        r1.x = a.x * cv[2 * k] - bb.x * sv[2 * k];
        r1.y = a.y * cv[2 * k + 1] - bb.y * sv[2 * k + 1];
        r2.x = bb.x * cv[2 * k] + a.x * sv[2 * k];
        r2.y = bb.y * cv[2 * k + 1] + a.y * sv[2 * k + 1];
        o1.h[k] = __float22bfloat162_rn(r1);
        o2.h[k] = __float22bfloat162_rn(r2);
    }
}

__global__ void rope_append_kernel(Params p) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= p.total) return;
    const int64_t tok = i / p.per_tok;
    const int64_t j = i - tok * p.per_tok;
    const int b = static_cast<int>(tok / p.S);
    const int s = static_cast<int>(tok - static_cast<int64_t>(b) * p.S);
    const int pos = p.pos0 + s;
    const __nv_bfloat16* row = p.qkv + tok * p.width;

    if (j >= p.rope_items) {  // v: copy 16 bytes into the cache
        const int64_t jv = j - p.rope_items;
        const int per_row = p.D / 8;
        const int vh = static_cast<int>(jv / per_row);
        const int c = static_cast<int>(jv - static_cast<int64_t>(vh) * per_row) * 8;
        const bf16x8 v = *reinterpret_cast<const bf16x8*>(row + (p.H_q + p.H_kv + vh) * p.D + c);
        __nv_bfloat16* dst =
            p.v_cache + ((static_cast<int64_t>(b) * p.H_kv + vh) * p.cap + pos) * p.D + c;
        *reinterpret_cast<bf16x8*>(dst) = v;
        return;
    }

    const int per_row = p.D / 16;
    const int h = static_cast<int>(j / per_row);  // 0..H_q-1: q head; H_q..: k head
    const int c = static_cast<int>(j - static_cast<int64_t>(h) * per_row) * 8;
    const int half = p.D / 2;
    bf16x8 o1, o2;
    rotate8(row + h * p.D + c, half, p.cos + static_cast<int64_t>(pos) * p.D + c,
            p.sin + static_cast<int64_t>(pos) * p.D + c, o1, o2);
    __nv_bfloat16* dst;
    if (h < p.H_q) {
        dst = p.q + ((static_cast<int64_t>(b) * p.H_q + h) * p.S + s) * p.D + c;
    } else {
        const int kh = h - p.H_q;
        dst = p.k_cache + ((static_cast<int64_t>(b) * p.H_kv + kh) * p.cap + pos) * p.D + c;
    }
    *reinterpret_cast<bf16x8*>(dst) = o1;
    *reinterpret_cast<bf16x8*>(dst + half) = o2;
}

// The paged form: token t of a packed batch at position pos[t], its k and v into slot
// slots[t] of the [num_pages, H_kv, page, D] caches, q out as [T, H_q, D]. Same work split
// as above with the token index in place of (b, s).
struct PagedParams {
    const __nv_bfloat16* qkv;
    const float* cos;
    const float* sin;
    const int* pos;
    const int* slots;
    __nv_bfloat16* q;
    __nv_bfloat16* k_cache;
    __nv_bfloat16* v_cache;
    int H_q, H_kv, D, shift;
    int64_t width, rope_items, per_tok, total;
};

// Element offset of (slot, kv head) in a [num_pages, H_kv, page, D] cache.
__device__ __forceinline__ int64_t slot_offset(int slot, int H_kv, int kh, int shift, int D) {
    const int64_t page = slot >> shift, r = slot & ((1 << shift) - 1);
    return (((page * H_kv + kh) << shift) + r) * D;
}

__global__ void rope_append_paged_kernel(PagedParams p) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= p.total) return;
    const int64_t tok = i / p.per_tok;
    const int64_t j = i - tok * p.per_tok;
    const int slot = p.slots[tok];
    const __nv_bfloat16* row = p.qkv + tok * p.width;

    if (j >= p.rope_items) {
        if (slot < 0) return;
        const int64_t jv = j - p.rope_items;
        const int per_row = p.D / 8;
        const int vh = static_cast<int>(jv / per_row);
        const int c = static_cast<int>(jv - static_cast<int64_t>(vh) * per_row) * 8;
        *reinterpret_cast<bf16x8*>(p.v_cache + slot_offset(slot, p.H_kv, vh, p.shift, p.D) + c) =
            *reinterpret_cast<const bf16x8*>(row + (p.H_q + p.H_kv + vh) * p.D + c);
        return;
    }

    const int per_row = p.D / 16;
    const int h = static_cast<int>(j / per_row);
    const int c = static_cast<int>(j - static_cast<int64_t>(h) * per_row) * 8;
    if (h >= p.H_q && slot < 0) return;
    const int half = p.D / 2;
    const int64_t pos = p.pos[tok];
    bf16x8 o1, o2;
    rotate8(row + h * p.D + c, half, p.cos + pos * p.D + c, p.sin + pos * p.D + c, o1, o2);
    __nv_bfloat16* dst = h < p.H_q
                             ? p.q + (tok * p.H_q + h) * p.D + c
                             : p.k_cache + slot_offset(slot, p.H_kv, h - p.H_q, p.shift, p.D) + c;
    *reinterpret_cast<bf16x8*>(dst) = o1;
    *reinterpret_cast<bf16x8*>(dst + half) = o2;
}

}  // namespace

void rope_append_bf16(const __nv_bfloat16* qkv, const float* cos, const float* sin,
                      __nv_bfloat16* q, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, int B,
                      int S, int H_q, int H_kv, int D, int pos0, int cap, cudaStream_t stream) {
    SPARK_REQUIRE(qkv != nullptr && cos != nullptr && sin != nullptr && q != nullptr &&
                      k_cache != nullptr && v_cache != nullptr,
                  "rope_append: null pointer");
    SPARK_REQUIRE(B >= 1 && S >= 1 && H_q >= 1 && H_kv >= 1 && D >= 16 && D % 16 == 0,
                  "rope_append: need B, S, H_q, H_kv >= 1 and D a multiple of 16");
    SPARK_REQUIRE(pos0 >= 0 && cap >= pos0 + S,
                  "rope_append: the cache cannot take S tokens at pos0");
    SPARK_REQUIRE(is_aligned16(qkv) && is_aligned16(cos) && is_aligned16(sin) && is_aligned16(q) &&
                      is_aligned16(k_cache) && is_aligned16(v_cache),
                  "rope_append: needs 16-byte aligned pointers");
    Params p;
    p.qkv = qkv;
    p.cos = cos;
    p.sin = sin;
    p.q = q;
    p.k_cache = k_cache;
    p.v_cache = v_cache;
    p.S = S;
    p.H_q = H_q;
    p.H_kv = H_kv;
    p.D = D;
    p.pos0 = pos0;
    p.cap = cap;
    p.width = static_cast<int64_t>(H_q + 2 * H_kv) * D;
    p.rope_items = static_cast<int64_t>(H_q + H_kv) * (D / 16);
    p.per_tok = p.rope_items + static_cast<int64_t>(H_kv) * (D / 8);
    p.total = static_cast<int64_t>(B) * S * p.per_tok;
    const int64_t blocks = cdiv64(p.total, kBlock);
    SPARK_REQUIRE(blocks < (int64_t{1} << 31), "rope_append: too many tokens for one launch");
    rope_append_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(p);
    SPARK_CHECK_LAUNCH();
}

void rope_append_paged_bf16(const __nv_bfloat16* qkv, const float* cos, const float* sin,
                            const int* positions, const int* slots, __nv_bfloat16* q,
                            __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, int T, int H_q,
                            int H_kv, int D, int page, cudaStream_t stream) {
    SPARK_REQUIRE(qkv != nullptr && cos != nullptr && sin != nullptr && positions != nullptr &&
                      slots != nullptr && q != nullptr && k_cache != nullptr && v_cache != nullptr,
                  "rope_append_paged: null pointer");
    SPARK_REQUIRE(T >= 1 && H_q >= 1 && H_kv >= 1 && D >= 16 && D % 16 == 0,
                  "rope_append_paged: need T, H_q, H_kv >= 1 and D a multiple of 16");
    int shift = 0;
    while ((1 << shift) < page) ++shift;
    SPARK_REQUIRE(page >= 1 && (1 << shift) == page, "rope_append_paged: page a power of two");
    SPARK_REQUIRE(is_aligned16(qkv) && is_aligned16(cos) && is_aligned16(sin) && is_aligned16(q) &&
                      is_aligned16(k_cache) && is_aligned16(v_cache),
                  "rope_append_paged: needs 16-byte aligned pointers");
    PagedParams p;
    p.qkv = qkv;
    p.cos = cos;
    p.sin = sin;
    p.pos = positions;
    p.slots = slots;
    p.q = q;
    p.k_cache = k_cache;
    p.v_cache = v_cache;
    p.H_q = H_q;
    p.H_kv = H_kv;
    p.D = D;
    p.shift = shift;
    p.width = static_cast<int64_t>(H_q + 2 * H_kv) * D;
    p.rope_items = static_cast<int64_t>(H_q + H_kv) * (D / 16);
    p.per_tok = p.rope_items + static_cast<int64_t>(H_kv) * (D / 8);
    p.total = static_cast<int64_t>(T) * p.per_tok;
    const int64_t blocks = cdiv64(p.total, kBlock);
    SPARK_REQUIRE(blocks < (int64_t{1} << 31), "rope_append_paged: too many tokens");
    rope_append_paged_kernel<<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(p);
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
