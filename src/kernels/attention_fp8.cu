// fp8 (e4m3) fused attention forward for Blackwell sm_12x: RTX 5090 (sm_120a), GB10 (sm_121a).
//
//   O[b,h] = softmax(sq sk Q[b,h] K[b,kv(h)]^T / sqrt(D)) sv V[b,kv(h)]
//   Q = [B, H_q, S_q, D], K, V = [B, H_kv, S_kv, D] e4m3, row-major, O [B, H_q, S_q, D] bf16
//
// sq, sk, sv are the descale factors of the three tensors (fp32, on the device: one per
// tensor, or one per (b, head) with `per_head`), so Q = sq * Q8 and so on. Scores, the online
// softmax and the accumulators are fp32; the probabilities are rounded to e4m3 before the
// P V product, which is what lets both products run on the fp8 tensor cores.
//
//   variant 0: one warp per query row, lanes own D/32 columns, keys one at a time, every
//              element dequantized to fp32 and P kept in fp32. The baseline, and the
//              arithmetic the tiled kernel approximates.
//   variant 1: attention variant 5's persistent TMA kernel on e4m3 operands: both products on
//              mma.sync.m16n8k32 (the block-scaled instruction with unit scales, the full fp8
//              rate), P rounded to e4m3 in registers (scaled by 2^8 first, so the small
//              probabilities keep their precision), and V read as it lies in memory, [key][d],
//              through ldmatrix.trans and byte permutes. docs/design/attention.md, "FP8".
//
// This file is built for the architecture-specific target with the fp8 GEMM (CMakeLists.txt),
// and keeps its own copy of the few scheduling pieces it shares with attention.cu (the tail
// split, the combine kernel, the queue counters) so that the two libraries do not depend on
// each other.

#include <cuda_fp8.h>

#include <algorithm>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "attention_internal.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

using attn::bf16;
using attn::ex2;
using attn::kLog2e;
using attn::kv_index;
using fp8 = __nv_fp8_e4m3;

// The descale factors of one launch: element `stride * index` of each array, where index is
// the flattened (b, head) of the tensor (stride 0: one factor per tensor).
struct Scales {
    const float* q;
    const float* k;
    const float* v;
    int stride;
};

// Four floats to four e4m3 bytes (round to nearest even, saturating at 448), x0 in the low
// byte. cvt...e4m3x2 puts its first source in the upper byte of the 16-bit result.
__device__ __forceinline__ unsigned pack_e4m3x4(float x0, float x1, float x2, float x3) {
    unsigned short lo, hi;
    unsigned r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(lo) : "f"(x1), "f"(x0));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(hi) : "f"(x3), "f"(x2));
    asm("mov.b32 %0, {%1, %2};" : "=r"(r) : "h"(lo), "h"(hi));
    return r;
}

// The half-precision path of the softmax (variant 1, kExpH2): two fp32 exponents to one f16x2
// (lo in the low half), 2^x on both halves with one MUFU, the f16x2 sum, and two f16x2 to four
// e4m3 bytes (the low half of h01 in the low byte).
__device__ __forceinline__ unsigned f16x2_of(float lo, float hi) {
    unsigned r;
    asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
    return r;
}
__device__ __forceinline__ unsigned ex2_f16x2(unsigned x) {
    unsigned y;
    asm("ex2.approx.f16x2 %0, %1;" : "=r"(y) : "r"(x));
    return y;
}
__device__ __forceinline__ unsigned add_f16x2(unsigned a, unsigned b) {
    unsigned y;
    asm("add.rn.f16x2 %0, %1, %2;" : "=r"(y) : "r"(a), "r"(b));
    return y;
}
__device__ __forceinline__ unsigned max_f16x2(unsigned a, unsigned b) {
    unsigned y;
    asm("max.f16x2 %0, %1, %2;" : "=r"(y) : "r"(a), "r"(b));
    return y;
}
__device__ __forceinline__ unsigned fma_f16x2(unsigned a, unsigned b, unsigned c) {
    unsigned y;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(y) : "r"(a), "r"(b), "r"(c));
    return y;
}
__device__ __forceinline__ float max_halves_f16x2(unsigned x) {
    unsigned short lo, hi;
    float a, b;
    asm("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(x));
    asm("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(b) : "h"(hi));
    return fmaxf(a, b);
}
__device__ __forceinline__ float sum_f16x2(unsigned x) {
    unsigned short lo, hi;
    float a, b;
    asm("mov.b32 {%0, %1}, %2;" : "=h"(lo), "=h"(hi) : "r"(x));
    asm("cvt.f32.f16 %0, %1;" : "=f"(a) : "h"(lo));
    asm("cvt.f32.f16 %0, %1;" : "=f"(b) : "h"(hi));
    return a + b;
}
__device__ __forceinline__ unsigned pack_e4m3x4_f16(unsigned h01, unsigned h23) {
    unsigned short lo, hi;
    unsigned r;
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(lo) : "r"(h01));
    asm("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(hi) : "r"(h23));
    asm("mov.b32 %0, {%1, %2};" : "=r"(r) : "h"(lo), "h"(hi));
    return r;
}

// ---------------------------------------------------------------------------------------
// Variant 0: attention variant 0 on dequantized e4m3. Lane l owns bytes l*VEC .. l*VEC+VEC-1
// of q, of every key and value row, and of the fp32 output row.
// ---------------------------------------------------------------------------------------
namespace v0 {

constexpr int THREADS = 256;

template <int VEC>
__device__ __forceinline__ void load_fp8(const fp8* p, float (&f)[VEC]) {
    static_assert(VEC == 2 || VEC == 4, "D = 64 or 128");
    unsigned w;
    if constexpr (VEC == 4)
        w = *reinterpret_cast<const unsigned*>(p);
    else
        w = *reinterpret_cast<const unsigned short*>(p);
#pragma unroll
    for (int i = 0; i < VEC; ++i) {
        __nv_fp8_e4m3 e;
        e.__x = static_cast<__nv_fp8_storage_t>((w >> (8 * i)) & 0xFF);
        f[i] = static_cast<float>(e);
    }
}

template <int D>
__global__ void __launch_bounds__(THREADS)
    attention_fp8_v0_kernel(const fp8* __restrict__ Q, const fp8* __restrict__ K,
                            const fp8* __restrict__ V, bf16* __restrict__ O, int H_q, int H_kv,
                            int S_q, int S_kv, float scale_log2, Scales sc, int causal) {
    constexpr int VEC = D / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (THREADS / 32) + (threadIdx.x >> 5);
    const int bh = blockIdx.y;
    const int bkv = kv_index(bh, H_q, H_kv);
    if (row >= S_q) return;
    const fp8* q = Q + (static_cast<size_t>(bh) * S_q + row) * D + lane * VEC;
    const fp8* k = K + static_cast<size_t>(bkv) * S_kv * D + lane * VEC;
    const fp8* v = V + static_cast<size_t>(bkv) * S_kv * D + lane * VEC;
    const float qk = scale_log2 * sc.q[sc.stride * bh] * sc.k[sc.stride * bkv];

    float qr[VEC];
    load_fp8<VEC>(q, qr);
#pragma unroll
    for (int i = 0; i < VEC; ++i) qr[i] *= qk;  // scores come out in the log2 domain

    float o[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) o[i] = 0.f;
    float m = -INFINITY, l = 0.f;

    const int n = causal ? min(S_kv, row + 1) : S_kv;
#pragma unroll 4
    for (int j = 0; j < n; ++j) {
        float kr[VEC], vr[VEC];
        load_fp8<VEC>(k + static_cast<size_t>(j) * D, kr);
        load_fp8<VEC>(v + static_cast<size_t>(j) * D, vr);
        float part = 0.f;
#pragma unroll
        for (int i = 0; i < VEC; ++i) part = fmaf(qr[i], kr[i], part);
        const float s = warp_reduce_sum(part);
        const float m_new = fmaxf(m, s);
        const float alpha = ex2(m - m_new);  // 0 on the first key (m = -inf)
        const float p = ex2(s - m_new);
        l = fmaf(l, alpha, p);
#pragma unroll
        for (int i = 0; i < VEC; ++i) o[i] = fmaf(o[i], alpha, p * vr[i]);
        m = m_new;
    }

    const float inv = sc.v[sc.stride * bkv] / l;
    bf16* out = O + (static_cast<size_t>(bh) * S_q + row) * D + lane * VEC;
#pragma unroll
    for (int i = 0; i < VEC; i += 2)
        *reinterpret_cast<__nv_bfloat162*>(out + i) =
            __floats2bfloat162_rn(o[i] * inv, o[i + 1] * inv);
}

template <int D>
void launch(const fp8* Q, const fp8* K, const fp8* V, bf16* O, int B, int H_q, int H_kv, int S_q,
            int S_kv, float scale_log2, const Scales& sc, bool causal, cudaStream_t stream) {
    const dim3 grid(cdiv(S_q, THREADS / 32), B * H_q);
    attention_fp8_v0_kernel<D><<<grid, THREADS, 0, stream>>>(Q, K, V, O, H_q, H_kv, S_q, S_kv,
                                                             scale_log2, sc, causal ? 1 : 0);
}

}  // namespace v0

// ---------------------------------------------------------------------------------------
// Variant 1: attention variant 5 on e4m3. The block tile, the persistent queue, the item ring,
// the producer warp and the tail split are variant 5's (attention.cu); what changes is the
// data inside a stage and the two products:
//   * a row of Q, K or V is D bytes, one TMA box wide (128-byte swizzle at D = 128, 64-byte at
//     D = 64), so a K + V stage is 16 KB at D = 128 where the bf16 one is 32 KB;
//   * S = Q K^T on mma.sync.m16n8k32: Q's A fragments and K's B fragments load with a plain
//     ldmatrix exactly as the fp8 GEMM's A and Bt do (K is [key][d], d contiguous: the "Bt"
//     layout), four k32 steps at D = 128;
//   * P V wants V's B fragment as four consecutive keys of one d column per register, and V is
//     [key][d]. ldmatrix.x4.trans on 32 keys x 16 d bytes hands lane (g, c) four registers of
//     {key 2c, 2c+1} x {d 2g, 2g+1} for keys +0, +8, +16, +24; two byte permutes per register
//     pair split them into d = 2g and d = 2g+1, each with keys {2c, 2c+1, 8+2c, 9+2c}. So one
//     ldmatrix gives the B fragments of two n8 tiles of O whose columns are the even and the
//     odd d of a 16-byte chunk, and the key order inside a register is permuted;
//   * P's A fragment wants the same four keys per register. The S accumulators of n8 tiles
//     2i and 2i+1 hold exactly keys {2c, 2c+1} and {8+2c, 9+2c} of a 16-key group for lane
//     (g, c), so P is two cvt.e4m3x2 per register in that order and never leaves the lane:
//     the permutation that the transposing load forces on V is the one the accumulator layout
//     already has, and the sum over keys does not care about their order;
//   * the probabilities are 2^(s - m + 8), in (0, 256], before the rounding to e4m3: e4m3's
//     normal range starts at 2^-6, so without the shift every p under 1/64 of the row max would
//     round with fewer than three mantissa bits. The row sum carries the same 2^8 and the
//     final division cancels it;
//   * the O rescale is lazy: a row keeps its running max until a new score beats it by more
//     than 0.8 (log2 units; p then peaks at 2^8.8 = 446, still inside e4m3's 448), and the 64
//     multiplies run only on a tile where some row of the warp moved;
//   * at D = 128 the row sum is one more QMMA per k32 step of P V against an all-ones B, so it
//     is the sum of the rounded P and costs no FADD (kSumMMA below).
// O's n8 tiles 2j and 2j+1 hold d = 16j + 4c + {0, 2} and {1, 3} for lane (g, c): four
// adjacent columns per row, one 8-byte store.
//
// With -DSPARK_ATTN_FP8_EXPERIMENTS and SPARK_ATTN_FP8_VT=1 the producer warp rewrites each V
// tile in shared memory into the consumers' fragment order once (VT), so the consumer warps
// skip the transposing load and the permutes; measured slower (docs/design/attention.md).
// ---------------------------------------------------------------------------------------
namespace v1 {

constexpr int BN = 64;  // keys per K/V tile
constexpr int CONSUMER_WARPS = 8, CONSUMERS = 32 * CONSUMER_WARPS, THREADS = CONSUMERS + 32;
constexpr int BM = 16 * CONSUMER_WARPS;
constexpr int ITEM_DEPTH = 2;
constexpr float kPShift = 8.f;  // P is 2^8 times the probability before the e4m3 rounding
constexpr float kLag = 0.8f;    // a row max may run this far (log2) ahead of the one in use

// Softmax variants, a bit mask (SPARK_ATTN_FP8_MODE under -DSPARK_ATTN_FP8_EXPERIMENTS):
//   kExpH2  the exponentials two at a time, ex2.approx.f16x2, and P from f16x2 to e4m3;
//   kSumMMA the row sum as one more QMMA per k32 step of P V, against an all-ones B;
//   kPipe   Q K^T of tile t+1 issued before the softmax of tile t, two S buffers;
//   kHalfS  (with kExpH2) S itself to f16x2 right after Q K^T, the row max, the scale-and-bias
//           FMA and the exponential all on pairs. The power of two of the score scale goes
//           into the ue8m0 scale of the Q K^T mma, so S lands in f16 range and the FMA
//           multiplies by the mantissa, in [1, 2).
constexpr int kExpH2 = 1, kSumMMA = 2, kPipe = 4, kHalfS = 8;
// The default: the row sum on the tensor pipe at D = 128 (1.5 to 3% faster on the 4096 and
// 8192 shapes); at D = 64, where P V is half the products, its two extra QMMA a tile cost 7
// to 10%. docs/design/attention.md, "Fewer softmax instructions".
template <int D>
constexpr int default_mode() {
    return D == 128 ? kSumMMA : 0;
}

// Four 16 KB stages at D = 128 (a sweep of 3 to 6 moved nothing beyond noise: the tiles come
// from L2 and two in flight already cover its latency), six of 8 KB at D = 64.
template <int D>
constexpr int stages() {
    return D == 128 ? 4 : 6;
}
template <int D>
constexpr int stage_bytes() {
    return 2 * BN * D;  // K tile then V tile
}
template <int D>
constexpr int smem_bytes() {
    return stages<D>() * stage_bytes<D>() + 1024;
}
template <int D>
constexpr int partial_floats() {
    return BM * (D + 2);  // O rows, then m, then l
}

// Byte offset of logical (row, 16-byte chunk) in a tile of D-byte rows written by TMA with the
// swizzle that spans one row: 128-byte rows XOR the chunk with row % 8, 64-byte rows with
// (row / 2) % 4. Eight consecutive rows at one logical chunk land in eight bank groups.
template <int D>
__device__ __forceinline__ int tile_off(int row, int ch) {
    if constexpr (D == 128)
        return row * 128 + ((ch ^ (row & 7)) << 4);
    else
        return row * 64 + ((ch ^ ((row >> 1) & 3)) << 4);
}

// Work assignment: tile = q_rank * bh_count + bh, q_rank heaviest first under the causal mask;
// items [0, dp_tiles) are whole tiles, the rest the `split` key slices of each tail tile.
struct Sched {
    int bh_count;
    int H_q, H_kv;
    int q_tiles;
    int dp_tiles;
    int split;
    float* ws;  // (tiles - dp_tiles) * split partials of partial_floats<D>() floats
};

__device__ __forceinline__ void decode_item(int w, const Sched& s, int& tile, int& slice,
                                            int& split) {
    if (w < s.dp_tiles) {
        tile = w;
        slice = 0;
        split = 1;
    } else {
        const int r = w - s.dp_tiles;
        tile = s.dp_tiles + r / s.split;
        slice = r % s.split;
        split = s.split;
    }
}

template <int D, bool VT, int MODE>
__global__ void __launch_bounds__(THREADS, 1)
    attention_fp8_v1_kernel(const __grid_constant__ CUtensorMap tmQ,
                            const __grid_constant__ CUtensorMap tmK,
                            const __grid_constant__ CUtensorMap tmV, bf16* __restrict__ O, int S_q,
                            int S_kv, float scale_log2, float p_shift, float lag, Scales sc,
                            int causal, Sched sched, int n_items, unsigned* __restrict__ queue) {
    constexpr int STAGES = stages<D>();
    constexpr int KT = D / 32;   // k32 steps of Q K^T
    constexpr int DC = D / 16;   // 16-byte chunks of a row: pairs of n8 tiles of O
    constexpr int DT = D / 8;    // n8 tiles of O
    constexpr int NT = BN / 8;   // n8 tiles of S
    constexpr int PT = BN / 32;  // k32 steps of P V
    constexpr int STAGE = stage_bytes<D>();
    static_assert(BM * D <= STAGE, "the Q tile is staged through a K/V stage");

    extern __shared__ __align__(1024) unsigned char smem[];
    int4* items = reinterpret_cast<int4*>(smem + STAGES * STAGE);
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(items + ITEM_DEPTH);
    uint64_t* empty_bar = full_bar + STAGES;
    uint64_t* item_full = empty_bar + STAGES;
    uint64_t* item_empty = item_full + ITEM_DEPTH;
    uint64_t* vready = item_empty + ITEM_DEPTH;  // VT: the stage's V is in fragment order
    if (smem_u32(smem) % 1024 != 0) __trap();

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c = lane & 3;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full_bar[s], 1);
            mbar_init(&empty_bar[s], CONSUMER_WARPS);
        }
#pragma unroll
        for (int i = 0; i < ITEM_DEPTH; ++i) {
            mbar_init(&item_full[i], 1);
            mbar_init(&item_empty[i], CONSUMER_WARPS);
        }
        if constexpr (VT) {
#pragma unroll
            for (int s = 0; s < STAGES; ++s) mbar_init(&vready[s], 1);
        }
        fence_mbar_init();
    }
    __syncthreads();  // the only block-wide barrier in the kernel

    auto geometry = [&](int tile, int slice, int split, int& bh, int& bkv, int& q0, int& tb,
                        int& te) {
        bh = tile % sched.bh_count;
        bkv = kv_index(bh, sched.H_q, sched.H_kv);
        const int q_rank = tile / sched.bh_count;
        const int q_tile = causal ? sched.q_tiles - 1 - q_rank : q_rank;
        q0 = q_tile * BM;
        const int kv_end = causal ? min(S_kv, q0 + BM) : S_kv;
        const int T = cdiv(kv_end, BN);
        tb = slice * T / split;
        te = (slice + 1) * T / split;
    };

    // V's B fragments, as P V reads them: for k32 step kt and 16-byte chunk j, lane l's four
    // registers are the even-d pair and the odd-d pair of the byte permutes below. With VT
    // the producer warp rewrites each V tile into this order once, in place, and the consumer
    // warps read their four registers with one 16-byte load each; without it every consumer
    // warp does the transposing load and the permutes itself.
    auto v_frag = [&](const unsigned char* vs, int kt, int j, unsigned(&f)[4]) {
        unsigned r[4];
        ldmatrix_x4_trans(r, vs + tile_off<D>(kt * 32 + lane, j));
        f[0] = __byte_perm(r[0], r[1], 0x6420);
        f[1] = __byte_perm(r[2], r[3], 0x6420);
        f[2] = __byte_perm(r[0], r[1], 0x7531);
        f[3] = __byte_perm(r[2], r[3], 0x7531);
    };
    auto frag_off = [&](int kt, int j) { return ((kt * DC + j) * 32 + lane) * 16; };

    // ---- the producer warp with VT: lane 0 runs variant 5's load sequence, and the whole warp
    // turns the V tile of each K/V load into fragment order one load behind the issue (so the
    // copy has landed by then), then arrives on the stage's vready barrier. A Q load needs no
    // rewrite and arrives at once, so vready completes once per use of a stage, like full.
    if constexpr (VT) {
        if (warp == CONSUMER_WARPS) {
            if (lane == 0) {
                prefetch_tensormap(&tmQ);
                prefetch_tensormap(&tmK);
                prefetch_tensormap(&tmV);
            }
            unsigned g_load = 0, n_pub = 0;
            int pend = -1;                         // the K/V load whose V still has to be rewritten
            auto slot = [&]() -> unsigned char* {  // lane 0
                const int s = g_load % STAGES;
                const unsigned use = g_load / STAGES;
                if (use > 0) mbar_wait(&empty_bar[s], (use - 1) & 1);
                return smem + s * STAGE;
            };
            auto rewrite = [&]() {
                if (pend < 0) return;
                const int s = pend % STAGES;
                mbar_wait(&full_bar[s], (pend / STAGES) & 1);
                unsigned char* vs = smem + s * STAGE + BN * D;
                unsigned f[PT][DC][4];
#pragma unroll
                for (int kt = 0; kt < PT; ++kt)
#pragma unroll
                    for (int j = 0; j < DC; ++j) v_frag(vs, kt, j, f[kt][j]);
                __syncwarp();  // the whole tile is in registers before any of it is overwritten
#pragma unroll
                for (int kt = 0; kt < PT; ++kt)
#pragma unroll
                    for (int j = 0; j < DC; ++j)
                        *reinterpret_cast<uint4*>(vs + frag_off(kt, j)) =
                            make_uint4(f[kt][j][0], f[kt][j][1], f[kt][j][2], f[kt][j][3]);
                fence_proxy_async_smem();  // before the copy engine refills the stage
                __syncwarp();
                if (lane == 0) mbar_arrive(&vready[s]);
                pend = -1;
            };
            for (;;) {
                int w = 0;
                if (lane == 0) w = static_cast<int>(atomicAdd(queue, 1u));
                w = __shfl_sync(kFullMask, w, 0);
                if (w >= n_items) {
                    if (lane == 0) {
                        const int r = n_pub % ITEM_DEPTH;
                        const unsigned use = n_pub / ITEM_DEPTH;
                        if (use > 0) mbar_wait(&item_empty[r], (use - 1) & 1);
                        items[r] = make_int4(-1, 0, 0, 0);
                        mbar_arrive(&item_full[r]);
                    }
                    break;
                }
                int tile, slice, split, bh, bkv, q0, tb, te;
                decode_item(w, sched, tile, slice, split);
                geometry(tile, slice, split, bh, bkv, q0, tb, te);
                if (lane == 0) {
                    const int r = n_pub % ITEM_DEPTH;
                    const unsigned use = n_pub / ITEM_DEPTH;
                    if (use > 0) mbar_wait(&item_empty[r], (use - 1) & 1);
                    items[r] = make_int4(w, 0, 0, 0);
                    mbar_arrive(&item_full[r]);
                    unsigned char* dst = slot();
                    uint64_t* bar = &full_bar[g_load % STAGES];
                    mbar_arrive_expect_tx(bar, BM * D);
                    tma_load_3d(dst, &tmQ, bar, 0, q0, bh);
                    mbar_arrive(&vready[g_load % STAGES]);
                }
                ++n_pub;
                ++g_load;
                rewrite();
                for (int t = tb; t < te; ++t) {
                    if (lane == 0) {
                        unsigned char* dst = slot();
                        uint64_t* bar = &full_bar[g_load % STAGES];
                        mbar_arrive_expect_tx(bar, STAGE);
                        tma_load_3d(dst, &tmK, bar, 0, t * BN, bkv);
                        tma_load_3d(dst + BN * D, &tmV, bar, 0, t * BN, bkv);
                    }
                    __syncwarp();
                    rewrite();
                    pend = static_cast<int>(g_load);
                    ++g_load;
                }
            }
            rewrite();
            if (lane == 0) {
                __threadfence();
                if (atomicAdd(queue + 1, 1u) == gridDim.x - 1) {
                    queue[0] = 0u;
                    queue[1] = 0u;
                    __threadfence();
                }
            }
            return;
        }
    }

    // ---- the producer warp: variant 5's, with one box per tile.
    if (warp == CONSUMER_WARPS) {
        if (lane != 0) return;
        prefetch_tensormap(&tmQ);
        prefetch_tensormap(&tmK);
        prefetch_tensormap(&tmV);
        unsigned g_load = 0, n_pub = 0;
        auto publish = [&](int w) {
            const int r = n_pub % ITEM_DEPTH;
            const unsigned use = n_pub / ITEM_DEPTH;
            if (use > 0) mbar_wait(&item_empty[r], (use - 1) & 1);
            items[r] = make_int4(w, 0, 0, 0);
            mbar_arrive(&item_full[r]);
            ++n_pub;
        };
        auto slot = [&]() -> unsigned char* {
            const int s = g_load % STAGES;
            const unsigned use = g_load / STAGES;
            if (use > 0) mbar_wait(&empty_bar[s], (use - 1) & 1);
            return smem + s * STAGE;
        };
        for (;;) {
            const int w = static_cast<int>(atomicAdd(queue, 1u));
            if (w >= n_items) {
                publish(-1);
                break;
            }
            publish(w);
            int tile, slice, split, bh, bkv, q0, tb, te;
            decode_item(w, sched, tile, slice, split);
            geometry(tile, slice, split, bh, bkv, q0, tb, te);
            {
                unsigned char* dst = slot();
                uint64_t* bar = &full_bar[g_load % STAGES];
                mbar_arrive_expect_tx(bar, BM * D);
                tma_load_3d(dst, &tmQ, bar, 0, q0, bh);
                ++g_load;
            }
            for (int t = tb; t < te; ++t) {
                unsigned char* dst = slot();
                uint64_t* bar = &full_bar[g_load % STAGES];
                mbar_arrive_expect_tx(bar, STAGE);
                tma_load_3d(dst, &tmK, bar, 0, t * BN, bkv);
                tma_load_3d(dst + BN * D, &tmV, bar, 0, t * BN, bkv);
                ++g_load;
            }
        }
        // Every block makes its final, failing grab before it counts itself done; the last
        // one done resets the counter for the next launch.
        __threadfence();
        if (atomicAdd(queue + 1, 1u) == gridDim.x - 1) {
            queue[0] = 0u;
            queue[1] = 0u;
            __threadfence();
        }
        return;
    }

    // ---- consumer warps.
    unsigned g_cons = 0, n_item = 0;
    auto stage_of = [&](unsigned u) -> const unsigned char* { return smem + (u % STAGES) * STAGE; };
    auto wait_load = [&](unsigned u) { mbar_wait(&full_bar[u % STAGES], (u / STAGES) & 1); };
    auto release_load = [&](unsigned u) {
        fence_proxy_async_smem();
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty_bar[u % STAGES]);
    };

    unsigned qf[KT][4];
    float o[DT][4];
    float m[2], l[2] = {0.f, 0.f};
    float lsum[4] = {0.f, 0.f, 0.f,
                     0.f};  // kSumMMA: the row sums as one more n8 tile of P V (rows g, g+8)
    float sacc[NT][4];
    float sacc2[NT][4];  // kPipe: the S of the next tile
    unsigned pa[PT][4];
    int q0 = 0;
    float qk_scale = 0.f;
    unsigned qk_sa = 0x7F7F7F7Fu;  // kHalfS: the ue8m0 scale of Q in Q K^T
    unsigned qk_s2 = 0u;           // kHalfS: qk_scale in both halves
    // kSumMMA: an all-ones B fragment (e4m3 1.0 is 0x38): P times it is the row sum of the
    // rounded P in every column of the accumulator.
    const unsigned ones[2] = {0x38383838u, 0x38383838u};

    // S = Q K^T: 16 rows x 64 keys per warp. One ldmatrix.x4 on 16 key rows x 32 bytes gives
    // the B fragments of two n8 key tiles. Lanes 0-7 address keys +0..7 at the first 16 bytes
    // of the k32 step, lanes 8-15 the same keys at the second 16, lanes 16-31 keys +8..15, so
    // each tile's two registers come back adjacent (r0, r1 and r2, r3), the register pair the
    // mma takes. The order the fp8 GEMM uses (matrices 0, 2 and 1, 3) costs 32 MOVs a tile.
    const int k_row = (lane & 7) + ((lane >> 4) << 3), k_ch = (lane >> 3) & 1;
    auto qk = [&](float(&sa)[NT][4], const unsigned char* ks) {
#pragma unroll
        for (int nj = 0; nj < NT; ++nj)
#pragma unroll
            for (int e = 0; e < 4; ++e) sa[nj][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
#pragma unroll
            for (int nj = 0; nj < NT; nj += 2) {
                unsigned r[4];
                ldmatrix_x4(r, ks + tile_off<D>(nj * 8 + k_row, 2 * kk + k_ch));
                const unsigned b0[2] = {r[0], r[1]};
                const unsigned b1[2] = {r[2], r[3]};
                if constexpr ((MODE & kHalfS) != 0) {
                    mma_e4m3_16832_mx<0>(sa[nj], qf[kk], b0, qk_sa, 0x7F7F7F7Fu);
                    mma_e4m3_16832_mx<0>(sa[nj + 1], qf[kk], b1, qk_sa, 0x7F7F7F7Fu);
                } else {
                    mma_e4m3_16832(sa[nj], qf[kk], b0);
                    mma_e4m3_16832(sa[nj + 1], qf[kk], b1);
                }
            }
        }
    };
    // Mask, online softmax on the S accumulators (rows g and g+8), the O rescale, and P as the
    // e4m3 A fragments of P V. The probabilities are 2^(s - m + p_shift).
    auto softmax = [&](float(&sa)[NT][4], int t) {
        const int kv0 = t * BN;
        if (kv0 + BN > S_kv || (causal && kv0 + BN - 1 > q0)) {  // warp-uniform
#pragma unroll
            for (int nj = 0; nj < NT; ++nj)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int i = q0 + warp * 16 + g + (e >> 1) * 8;
                    const int j = kv0 + nj * 8 + 2 * c + (e & 1);
                    if (j >= S_kv || (causal && j > i)) sa[nj][e] = -INFINITY;
                }
        }
        float alpha[2], bias[2];
        unsigned hs_s[NT][2];  // kHalfS: S of row g / g+8 of n8 tile nj, keys 2c, 2c+1
        if constexpr ((MODE & kHalfS) != 0) {
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                hs_s[nj][0] = f16x2_of(sa[nj][0], sa[nj][1]);
                hs_s[nj][1] = f16x2_of(sa[nj][2], sa[nj][3]);
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float mx;
            if constexpr ((MODE & kHalfS) != 0) {
                unsigned hm = max_f16x2(hs_s[0][r], hs_s[1][r]);
#pragma unroll
                for (int nj = 2; nj < NT; nj += 2)
                    hm = max_f16x2(hm, max_f16x2(hs_s[nj][r], hs_s[nj + 1][r]));
                mx = max_halves_f16x2(hm);
            } else {
                mx = fmaxf(sa[0][2 * r], sa[0][2 * r + 1]);
#pragma unroll
                for (int nj = 1; nj < NT; ++nj)
                    mx = fmaxf(mx, fmaxf(sa[nj][2 * r], sa[nj][2 * r + 1]));
            }
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 2));
            // A row keeps its old max while the new one exceeds it by at most `lag` (log2
            // units): its p then reach 2^(p_shift + lag), still under e4m3's 448, and its alpha
            // is exactly 1. (-inf - -inf is NaN, so a row that has seen no key never keeps.)
            float m_new = fmaxf(m[r], mx * qk_scale);
            if (m_new - m[r] <= lag) m_new = m[r];
            const float m_use = m_new == -INFINITY ? 0.f : m_new;
            alpha[r] = ex2(m[r] - m_use);
            bias[r] = p_shift - m_use;
            m[r] = m_new;
        }
        // The 64 multiplies of the O rescale run only when some row of the warp moved its max
        // (a warp-uniform vote); once the maxima settle, most tiles skip them.
        if (lag < 0.f || __any_sync(kFullMask, alpha[0] != 1.f || alpha[1] != 1.f)) {
#pragma unroll
            for (int dj = 0; dj < DT; ++dj) {
                o[dj][0] *= alpha[0];
                o[dj][1] *= alpha[0];
                o[dj][2] *= alpha[1];
                o[dj][3] *= alpha[1];
            }
            if constexpr ((MODE & kSumMMA) != 0) {
                lsum[0] *= alpha[0];
                lsum[1] *= alpha[0];
                lsum[2] *= alpha[1];
                lsum[3] *= alpha[1];
            }
        }
        // pa[kt][0] = row g, keys {2c, 2c+1, 8+2c, 9+2c} of the first 16 of the k32 step: the
        // accumulators of n8 tiles 4kt and 4kt+1; pa[kt][1] the same for row g+8; pa[kt][2],
        // pa[kt][3] the second 16 keys, tiles 4kt+2 and 4kt+3.
        if constexpr ((MODE & kExpH2) != 0) {
            // Two exponentials per MUFU in half precision, and P straight from f16x2 to e4m3:
            // f16 keeps 11 bits where e4m3 keeps 4, and p <= 2^(p_shift + lag) < 448 is far
            // inside its range.
            unsigned hs[2] = {0u, 0u};
            unsigned bias2[2] = {0u, 0u};
            if constexpr ((MODE & kHalfS) != 0) {
                bias2[0] = f16x2_of(bias[0], bias[0]);
                bias2[1] = f16x2_of(bias[1], bias[1]);
            }
#pragma unroll
            for (int kt = 0; kt < PT; ++kt)
#pragma unroll
                for (int q = 0; q < 4; ++q) {
                    const int r = q & 1, t0 = 4 * kt + 2 * (q >> 1);
                    unsigned h01, h23;
                    if constexpr ((MODE & kHalfS) != 0) {
                        h01 = ex2_f16x2(fma_f16x2(hs_s[t0][r], qk_s2, bias2[r]));
                        h23 = ex2_f16x2(fma_f16x2(hs_s[t0 + 1][r], qk_s2, bias2[r]));
                    } else {
                        h01 = ex2_f16x2(f16x2_of(fmaf(sa[t0][2 * r], qk_scale, bias[r]),
                                                 fmaf(sa[t0][2 * r + 1], qk_scale, bias[r])));
                        h23 = ex2_f16x2(f16x2_of(fmaf(sa[t0 + 1][2 * r], qk_scale, bias[r]),
                                                 fmaf(sa[t0 + 1][2 * r + 1], qk_scale, bias[r])));
                    }
                    if constexpr ((MODE & kSumMMA) == 0)
                        hs[r] = add_f16x2(hs[r], add_f16x2(h01, h23));
                    pa[kt][q] = pack_e4m3x4_f16(h01, h23);
                }
            if constexpr ((MODE & kSumMMA) == 0) {
                l[0] = fmaf(l[0], alpha[0], sum_f16x2(hs[0]));
                l[1] = fmaf(l[1], alpha[1], sum_f16x2(hs[1]));
            }
        } else {
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                float rs = 0.f;
#pragma unroll
                for (int nj = 0; nj < NT; ++nj) {
                    const float p0 = ex2(fmaf(sa[nj][2 * r], qk_scale, bias[r]));
                    const float p1 = ex2(fmaf(sa[nj][2 * r + 1], qk_scale, bias[r]));
                    sa[nj][2 * r] = p0;
                    sa[nj][2 * r + 1] = p1;
                    rs += p0 + p1;
                }
                if constexpr ((MODE & kSumMMA) == 0) l[r] = fmaf(l[r], alpha[r], rs);
            }
#pragma unroll
            for (int kt = 0; kt < PT; ++kt) {
                const int t0 = 4 * kt;
                pa[kt][0] = pack_e4m3x4(sa[t0][0], sa[t0][1], sa[t0 + 1][0], sa[t0 + 1][1]);
                pa[kt][1] = pack_e4m3x4(sa[t0][2], sa[t0][3], sa[t0 + 1][2], sa[t0 + 1][3]);
                pa[kt][2] = pack_e4m3x4(sa[t0 + 2][0], sa[t0 + 2][1], sa[t0 + 3][0], sa[t0 + 3][1]);
                pa[kt][3] = pack_e4m3x4(sa[t0 + 2][2], sa[t0 + 2][3], sa[t0 + 3][2], sa[t0 + 3][3]);
            }
        }
        // kSumMMA: the row sum on the tensor pipe, two QMMA a tile in place of 32 FADD, and it
        // is the sum of the P that P V actually multiplies.
        if constexpr ((MODE & kSumMMA) != 0) {
#pragma unroll
            for (int kt = 0; kt < PT; ++kt) mma_e4m3_16832(lsum, pa[kt], ones);
        }
    };
    // O += P V. Lane l addresses key 32kt + l at chunk j, so the four matrices of the
    // transposing load are keys +0..7, +8..15, +16..23, +24..31, and register i holds
    // {key 2c, 2c+1 of matrix i} x {d 16j + 2g, 2g+1} as bytes (k2c d2g, k2c d2g+1, k2c+1 d2g,
    // k2c+1 d2g+1). Selecting bytes 0, 2 of a pair of registers (keys +0 and +8) gives
    // d = 16j + 2g for keys {2c, 2c+1, 8+2c, 9+2c}, the order P's registers are in; bytes 1, 3
    // give d = 16j + 2g + 1.
    auto pv = [&](const unsigned char* vs) {
#pragma unroll
        for (int kt = 0; kt < PT; ++kt) {
#pragma unroll
            for (int j = 0; j < DC; ++j) {
                unsigned f[4];
                if constexpr (VT) {
                    const uint4 u = *reinterpret_cast<const uint4*>(vs + frag_off(kt, j));
                    f[0] = u.x;
                    f[1] = u.y;
                    f[2] = u.z;
                    f[3] = u.w;
                } else {
                    v_frag(vs, kt, j, f);
                }
                const unsigned be[2] = {f[0], f[1]};
                const unsigned bo[2] = {f[2], f[3]};
                mma_e4m3_16832(o[2 * j], pa[kt], be);
                mma_e4m3_16832(o[2 * j + 1], pa[kt], bo);
            }
        }
    };

    for (;;) {
        const int r = n_item % ITEM_DEPTH;
        mbar_wait(&item_full[r], (n_item / ITEM_DEPTH) & 1);
        const int w = items[r].x;
        __syncwarp();
        if (lane == 0) mbar_arrive(&item_empty[r]);
        ++n_item;
        if (w < 0) break;
        int tile, slice, split, bh, bkv, tb, te;
        decode_item(w, sched, tile, slice, split);
        geometry(tile, slice, split, bh, bkv, q0, tb, te);
        qk_scale = scale_log2 * sc.q[sc.stride * bh] * sc.k[sc.stride * bkv];
        if constexpr ((MODE & kHalfS) != 0) {
            // qk_scale = 2^(e - 127) x mantissa: the biased exponent e is the ue8m0 byte that
            // scales Q in the mma, the mantissa is what the softmax multiplies by.
            const unsigned bits = __float_as_uint(qk_scale);
            qk_sa = ((bits >> 23) & 0xFFu) * 0x01010101u;
            qk_scale = __uint_as_float((bits & 0x007FFFFFu) | 0x3F800000u);
            qk_s2 = f16x2_of(qk_scale, qk_scale);
        }

        // Q tile into A fragments: qf[kk] covers d = 32kk .. 32kk+31. Then its stage is free.
        wait_load(g_cons);
        {
            const unsigned char* qs = stage_of(g_cons);
            const int row = warp * 16 + (lane & 15);
#pragma unroll
            for (int kk = 0; kk < KT; ++kk)
                ldmatrix_x4(qf[kk], qs + tile_off<D>(row, 2 * kk + (lane >> 4)));
        }
        release_load(g_cons);
        ++g_cons;

#pragma unroll
        for (int dj = 0; dj < DT; ++dj)
#pragma unroll
            for (int e = 0; e < 4; ++e) o[dj][e] = 0.f;
        m[0] = m[1] = -INFINITY;
        l[0] = l[1] = 0.f;
#pragma unroll
        for (int e = 0; e < 4; ++e) lsum[e] = 0.f;

        if constexpr ((MODE & kPipe) == 0) {
            for (int t = tb; t < te; ++t, ++g_cons) {
                // Under the causal mask the keys of a tile may all follow every row of this
                // warp: every p is 0, and the warp goes on to the next item instead (variant 5).
                const bool dead = causal && t * BN > q0 + warp * 16 + 15;
                wait_load(g_cons);
                if (!dead) {
                    qk(sacc, stage_of(g_cons));
                    softmax(sacc, t);
                    if constexpr (VT) mbar_wait(&vready[g_cons % STAGES], (g_cons / STAGES) & 1);
                    pv(stage_of(g_cons) + BN * D);
                }
                release_load(g_cons);
            }
        } else {
            // kPipe: Q K^T of tile t+1 is issued before the softmax of tile t, into a second S,
            // so the products of one tile run under the softmax of the other inside one warp.
            // The live tiles are a prefix [tb, t_live) under the causal mask; the dead ones
            // after it are waited for and released as before.
            const int t_live = causal ? max(tb, min(te, (q0 + warp * 16 + 15) / BN + 1)) : te;
            auto step = [&](float(&cur)[NT][4], float(&nxt)[NT][4], int t) {
                if (t + 1 < t_live) {
                    wait_load(g_cons + 1);
                    qk(nxt, stage_of(g_cons + 1));
                }
                softmax(cur, t);
                if constexpr (VT) mbar_wait(&vready[g_cons % STAGES], (g_cons / STAGES) & 1);
                pv(stage_of(g_cons) + BN * D);
                release_load(g_cons);
                ++g_cons;
            };
            if (tb < t_live) {
                wait_load(g_cons);
                qk(sacc, stage_of(g_cons));
            }
            for (int t = tb; t < t_live; t += 2) {
                step(sacc, sacc2, t);
                if (t + 1 < t_live) step(sacc2, sacc, t + 1);
            }
            for (int t = t_live; t < te; ++t, ++g_cons) {
                wait_load(g_cons);
                release_load(g_cons);
            }
        }

        // Epilogue: lane (g, c) holds d = 16j + 4c .. 16j + 4c + 3 of rows g and g+8 in tiles
        // 2j (even d) and 2j+1 (odd d). Normalized bf16 rows with the V descale, or the
        // unnormalized fp32 rows (V descale applied) plus (m, l) of a slice to the workspace.
        const float vs = sc.v[sc.stride * bkv];
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            float ls;
            if constexpr ((MODE & kSumMMA) != 0) {
                ls = lsum[2 * rr];
            } else {
                ls = l[rr];
                ls += __shfl_xor_sync(kFullMask, ls, 1);
                ls += __shfl_xor_sync(kFullMask, ls, 2);
            }
            const int lrow = warp * 16 + g + 8 * rr;
            if (split == 1) {
                const int row = q0 + lrow;
                if (row >= S_q) continue;
                const float f = vs / ls;
                bf16* out = O + (static_cast<size_t>(bh) * S_q + row) * D + 4 * c;
#pragma unroll
                for (int j = 0; j < DC; ++j) {
                    uint2 u;
                    u.x = attn::pack_bf16x2(o[2 * j][2 * rr] * f, o[2 * j + 1][2 * rr] * f);
                    u.y = attn::pack_bf16x2(o[2 * j][2 * rr + 1] * f, o[2 * j + 1][2 * rr + 1] * f);
                    *reinterpret_cast<uint2*>(out + 16 * j) = u;
                }
            } else {
                float* part =
                    sched.ws + (static_cast<size_t>(tile - sched.dp_tiles) * split + slice) *
                                   partial_floats<D>();
                float* orow = part + lrow * D + 4 * c;
#pragma unroll
                for (int j = 0; j < DC; ++j)
                    *reinterpret_cast<float4*>(orow + 16 * j) =
                        make_float4(o[2 * j][2 * rr] * vs, o[2 * j + 1][2 * rr] * vs,
                                    o[2 * j][2 * rr + 1] * vs, o[2 * j + 1][2 * rr + 1] * vs);
                if (c == 0) {
                    part[BM * D + lrow] = m[rr];
                    part[BM * D + BM + lrow] = ls;
                }
            }
        }
    }
}

// Merges the `split` partials of every split tile: M = max m_i, L = sum l_i 2^(m_i - M),
// O = sum O_i 2^(m_i - M) / L (attention.cu's combine kernel, on this file's tile).
template <int D>
__global__ void attention_fp8_combine_kernel(bf16* __restrict__ O, int S_q, int causal,
                                             Sched sched) {
    constexpr int CH = D / 8;
    const int tile = sched.dp_tiles + blockIdx.y;
    const int bh = tile % sched.bh_count;
    const int q_rank = tile / sched.bh_count;
    const int q_tile = causal ? sched.q_tiles - 1 - q_rank : q_rank;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BM * CH) return;
    const int lrow = idx / CH, ch = idx % CH;
    const int row = q_tile * BM + lrow;
    if (row >= S_q) return;
    const float* base =
        sched.ws + static_cast<size_t>(blockIdx.y) * sched.split * partial_floats<D>();
    float M = -INFINITY;
    for (int i = 0; i < sched.split; ++i)
        M = fmaxf(M, base[i * partial_floats<D>() + BM * D + lrow]);
    float L = 0.f, acc[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.f;
    for (int i = 0; i < sched.split; ++i) {
        const float* part = base + i * partial_floats<D>();
        const float w = ex2(part[BM * D + lrow] - M);  // 0 for a slice that saw no key of the row
        L = fmaf(part[BM * D + BM + lrow], w, L);
        const float4 a = *reinterpret_cast<const float4*>(part + lrow * D + ch * 8);
        const float4 b = *reinterpret_cast<const float4*>(part + lrow * D + ch * 8 + 4);
        acc[0] = fmaf(a.x, w, acc[0]);
        acc[1] = fmaf(a.y, w, acc[1]);
        acc[2] = fmaf(a.z, w, acc[2]);
        acc[3] = fmaf(a.w, w, acc[3]);
        acc[4] = fmaf(b.x, w, acc[4]);
        acc[5] = fmaf(b.y, w, acc[5]);
        acc[6] = fmaf(b.z, w, acc[6]);
        acc[7] = fmaf(b.w, w, acc[7]);
    }
    const float inv = 1.f / L;
    uint4 u;
    u.x = attn::pack_bf16x2(acc[0] * inv, acc[1] * inv);
    u.y = attn::pack_bf16x2(acc[2] * inv, acc[3] * inv);
    u.z = attn::pack_bf16x2(acc[4] * inv, acc[5] * inv);
    u.w = attn::pack_bf16x2(acc[6] * inv, acc[7] * inv);
    *reinterpret_cast<uint4*>(O + (static_cast<size_t>(bh) * S_q + row) * D + ch * 8) = u;
}

// Knobs, read once, for the tables in docs/design/attention.md: SPARK_ATTN_FP8_SPLIT = 0 / 1
// forces the tail split off / on (-1: the rule in launch), SPARK_ATTN_FP8_PSHIFT = n rounds
// 2^n times the probabilities to e4m3 instead of 2^8 times, SPARK_ATTN_FP8_LAG = x lets a row
// max run x (log2 units, capped so that p stays under 448) ahead of the one in use before the
// O rescale; negative rescales on every tile.
struct Env {
    int split = -1;
    float p_shift = kPShift;
    float lag = kLag;
    int vt = 0;
    int mode = -1;  // -1: default_mode<D>()
};
const Env& env() {
    static Env e;
    static bool read = false;
    if (!read) {
        read = true;
        if (const char* s = std::getenv("SPARK_ATTN_FP8_SPLIT")) e.split = std::atoi(s);
        if (const char* s = std::getenv("SPARK_ATTN_FP8_PSHIFT"))
            e.p_shift = static_cast<float>(std::atoi(s));
        if (const char* s = std::getenv("SPARK_ATTN_FP8_LAG")) e.lag = std::atof(s);
        if (const char* s = std::getenv("SPARK_ATTN_FP8_VT")) e.vt = std::atoi(s);
        if (const char* s = std::getenv("SPARK_ATTN_FP8_MODE")) e.mode = std::atoi(s);
        // p may reach 2^(p_shift + lag), which must stay within e4m3's 448
        e.lag = std::min(e.lag, std::log2(448.f) - e.p_shift);
    }
    return e;
}

// Queue counters, two per stream (a launch must not share its counter with one that may be
// running on another stream); handed out on first sight, zeroed once, left zero by every
// launch. Attention variant 5 keeps its own set.
unsigned* queue_for(cudaStream_t stream) {
    constexpr int kSlots = 64;
    static std::mutex mu;
    static unsigned* base = nullptr;
    static std::unordered_map<cudaStream_t, int> slot;
    std::lock_guard<std::mutex> lock(mu);
    if (base == nullptr) {
        SPARK_CUDA_CHECK(cudaMalloc(&base, kSlots * 2 * sizeof(unsigned)));
        SPARK_CUDA_CHECK(cudaMemset(base, 0, kSlots * 2 * sizeof(unsigned)));
    }
    auto it = slot.find(stream);
    if (it == slot.end()) {
        const int n = static_cast<int>(slot.size());
        it = slot.emplace(stream, n < kSlots ? n : n % kSlots).first;
    }
    return base + 2 * it->second;
}

template <int D, bool VT, int MODE>
int resident_blocks() {
    constexpr int bytes = smem_bytes<D>();
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(attention_fp8_v1_kernel<D, VT, MODE>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, attention_fp8_v1_kernel<D, VT, MODE>, THREADS, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

// Tail partials, one buffer per device, grown on demand, shared by both head sizes and every
// stream: two split launches must not run concurrently on different streams.
float* workspace(size_t floats) {
    static float* ws = nullptr;
    static size_t have = 0;
    if (have < floats) {
        if (ws) SPARK_CUDA_CHECK(cudaFree(ws));
        SPARK_CUDA_CHECK(cudaMalloc(&ws, floats * sizeof(float)));
        have = floats;
    }
    return ws;
}

// The tail split rule of attention variant 5: on for the non-causal shapes and for grids with
// fewer tiles than resident blocks, off under the causal mask once there is a full wave (the
// heaviest-first queue balances it). The tail tiles are the last in the order, so the last
// has the fewest KV tiles; every slice gets at least one.
template <int D, bool VT, int MODE = default_mode<D>()>
void launch(const fp8* Q, const fp8* K, const fp8* V, bf16* O, int B, int H_q, int H_kv, int S_q,
            int S_kv, float scale_log2, const Scales& sc, bool causal, cudaStream_t stream) {
    constexpr int bytes = smem_bytes<D>();
    const int resident = resident_blocks<D, VT, MODE>();
    constexpr CUtensorMapSwizzle swz =
        D == 128 ? CU_TENSOR_MAP_SWIZZLE_128B : CU_TENSOR_MAP_SWIZZLE_64B;
    const uint64_t BHq = static_cast<uint64_t>(B) * H_q;
    const uint64_t BHkv = static_cast<uint64_t>(B) * H_kv;
    const CUtensorMap tmQ = make_tensor_map_3d_u8(Q, D, S_q, BHq, D, BM, swz);
    const CUtensorMap tmK = make_tensor_map_3d_u8(K, D, S_kv, BHkv, D, BN, swz);
    const CUtensorMap tmV = make_tensor_map_3d_u8(V, D, S_kv, BHkv, D, BN, swz);

    Sched s;
    s.bh_count = B * H_q;
    s.H_q = H_q;
    s.H_kv = H_kv;
    s.q_tiles = cdiv(S_q, BM);
    const int tiles = s.bh_count * s.q_tiles;
    s.dp_tiles = tiles;
    s.split = 1;
    s.ws = nullptr;
    const int mode = env().split;
    const bool split_tail = mode < 0 ? !causal || tiles < resident : mode != 0;
    if (split_tail) {
        const int tail = tiles % resident;
        const int q_last = causal ? 0 : s.q_tiles - 1;
        const int T_last = cdiv(causal ? std::min(S_kv, q_last * BM + BM) : S_kv, BN);
        const int split = tail > 0 ? std::min(T_last, resident / tail) : 1;
        if (split > 1) {
            s.dp_tiles = tiles - tail;
            s.split = split;
            s.ws = workspace(static_cast<size_t>(tail) * split * partial_floats<D>());
        }
    }
    const int n_items = s.dp_tiles + (tiles - s.dp_tiles) * s.split;
    const int grid = std::min(resident, n_items);
    attention_fp8_v1_kernel<D, VT, MODE><<<grid, THREADS, bytes, stream>>>(
        tmQ, tmK, tmV, O, S_q, S_kv, scale_log2, env().p_shift, env().lag, sc, causal ? 1 : 0, s,
        n_items, queue_for(stream));
    if (s.split > 1) {
        const dim3 cgrid(cdiv(BM * (D / 8), 256), tiles - s.dp_tiles);
        attention_fp8_combine_kernel<D><<<cgrid, 256, 0, stream>>>(O, S_q, causal ? 1 : 0, s);
    }
}

}  // namespace v1

#ifdef SPARK_ATTN_FP8_EXPERIMENTS
// Variant 1 with the softmax variant `mode` (the kExpH2 / kSumMMA / kPipe bits), for the
// tables in docs/design/attention.md.
template <int MODE>
void launch_mode_t(const fp8* q, const fp8* k, const fp8* v, bf16* O, int B, int H_q, int H_kv,
                   int S_q, int S_kv, int D, float scale_log2, const Scales& sc, bool causal,
                   cudaStream_t stream) {
    if (D == 64)
        v1::launch<64, false, MODE>(q, k, v, O, B, H_q, H_kv, S_q, S_kv, scale_log2, sc, causal,
                                    stream);
    else
        v1::launch<128, false, MODE>(q, k, v, O, B, H_q, H_kv, S_q, S_kv, scale_log2, sc, causal,
                                     stream);
}
void launch_mode(const fp8* q, const fp8* k, const fp8* v, bf16* O, int B, int H_q, int H_kv,
                 int S_q, int S_kv, int D, float scale_log2, const Scales& sc, bool causal,
                 int mode, cudaStream_t stream) {
    if (mode & v1::kHalfS) mode |= v1::kExpH2;
    switch (mode & 15) {
#define SPARK_FP8_MODE_CASE(M)                                                                    \
    case M:                                                                                       \
        launch_mode_t<M>(q, k, v, O, B, H_q, H_kv, S_q, S_kv, D, scale_log2, sc, causal, stream); \
        break;
        SPARK_FP8_MODE_CASE(0)
        SPARK_FP8_MODE_CASE(1)
        SPARK_FP8_MODE_CASE(2)
        SPARK_FP8_MODE_CASE(3)
        SPARK_FP8_MODE_CASE(4)
        SPARK_FP8_MODE_CASE(5)
        SPARK_FP8_MODE_CASE(6)
        SPARK_FP8_MODE_CASE(7)
        SPARK_FP8_MODE_CASE(9)
        SPARK_FP8_MODE_CASE(11)
        SPARK_FP8_MODE_CASE(13)
        SPARK_FP8_MODE_CASE(15)
#undef SPARK_FP8_MODE_CASE
    }
}
#endif

}  // namespace

int attention_fp8_num_variants() {
    return 2;
}

bool attention_fp8_supports(int S_q, int S_kv, int D, int variant) {
    if (variant < 0 || variant >= attention_fp8_num_variants()) return false;
    if (S_q <= 0 || S_kv <= 0) return false;
    return D == 64 || D == 128;
}

void attention_fp8(const __nv_fp8_e4m3* Q, const __nv_fp8_e4m3* K, const __nv_fp8_e4m3* V,
                   __nv_bfloat16* O, int B, int H_q, int H_kv, int S_q, int S_kv, int D,
                   const float* q_scale, const float* k_scale, const float* v_scale, bool per_head,
                   bool causal, int variant, cudaStream_t stream) {
    SPARK_REQUIRE(Q != nullptr && K != nullptr && V != nullptr && O != nullptr,
                  "attention_fp8: null pointer");
    SPARK_REQUIRE(q_scale != nullptr && k_scale != nullptr && v_scale != nullptr,
                  "attention_fp8: null scale pointer");
    SPARK_REQUIRE(B > 0 && H_q > 0 && H_kv > 0 && S_q > 0 && S_kv > 0,
                  "attention_fp8: B, H_q, H_kv, S_q, S_kv must be positive");
    SPARK_REQUIRE(H_q % H_kv == 0, "attention_fp8: H_q must be a multiple of H_kv");
    SPARK_REQUIRE(D == 64 || D == 128, "attention_fp8: D must be 64 or 128");
    SPARK_REQUIRE(variant >= 0 && variant < attention_fp8_num_variants(),
                  "attention_fp8: unknown variant");
    SPARK_REQUIRE(is_aligned16(Q) && is_aligned16(K) && is_aligned16(V) && is_aligned16(O),
                  "attention_fp8: Q, K, V, O must be 16-byte aligned");
    SPARK_REQUIRE(static_cast<int64_t>(B) * H_q * std::max(S_q, S_kv) * D < (int64_t{1} << 40),
                  "attention_fp8: tensor too large");

    const float scale_log2 = kLog2e / std::sqrt(static_cast<float>(D));
    const Scales sc{q_scale, k_scale, v_scale, per_head ? 1 : 0};
    const auto* q = reinterpret_cast<const fp8*>(Q);
    const auto* k = reinterpret_cast<const fp8*>(K);
    const auto* v = reinterpret_cast<const fp8*>(V);
    switch (variant) {
        case 0:
            if (D == 64)
                v0::launch<64>(q, k, v, O, B, H_q, H_kv, S_q, S_kv, scale_log2, sc, causal, stream);
            else
                v0::launch<128>(q, k, v, O, B, H_q, H_kv, S_q, S_kv, scale_log2, sc, causal,
                                stream);
            break;
        case 1:
#ifdef SPARK_ATTN_FP8_EXPERIMENTS
            if (v1::env().vt) {
                if (D == 64)
                    v1::launch<64, true>(q, k, v, O, B, H_q, H_kv, S_q, S_kv, scale_log2, sc,
                                         causal, stream);
                else
                    v1::launch<128, true>(q, k, v, O, B, H_q, H_kv, S_q, S_kv, scale_log2, sc,
                                          causal, stream);
                break;
            }
            if (v1::env().mode >= 0) {
                launch_mode(q, k, v, O, B, H_q, H_kv, S_q, S_kv, D, scale_log2, sc, causal,
                            v1::env().mode, stream);
                break;
            }
#endif
            if (D == 64)
                v1::launch<64, false>(q, k, v, O, B, H_q, H_kv, S_q, S_kv, scale_log2, sc, causal,
                                      stream);
            else
                v1::launch<128, false>(q, k, v, O, B, H_q, H_kv, S_q, S_kv, scale_log2, sc, causal,
                                       stream);
            break;
        default:
            break;
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
