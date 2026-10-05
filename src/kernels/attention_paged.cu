// Attention over a paged K/V cache, the layout a server keeps for a batch of sequences of
// different lengths (vLLM's PagedAttention): K and V live in fixed-size pages of `page`
// tokens, [num_pages, H_kv, page, D] each, and a sequence owns a list of pages, row b of a
// block table [B, max_pages] of int32 page ids. Token j of sequence b is row j % page of page
// block_table[b][j / page]. seq_lens[b] is the number of keys of sequence b. Measurements
// and the arithmetic behind the constants: docs/design/serving.md.
//
// Two ops live here, each a ladder of its own:
//
//   paged decode (one query token per sequence, Q and O [B, H_q, D]):
//     variant 0: one warp per (b, query head), keys one at a time, the page looked up per key.
//     variant 1: flash-decoding with a length-aware split. The keys of every (b, kv head) are
//                16-key slabs, and all the slabs of the batch, (b, kv head) segment after
//                segment, are one flat list cut into equal contiguous ranges, one per warp of
//                a grid of one block per SM. A warp streams its range through a private
//                3-stage cp.async pipeline that does not stop at segment boundaries. A
//                segment one warp covers alone is normalized and written by that warp; the
//                pieces of a segment that crosses warps go to a workspace and a combine
//                kernel merges them. One 32K-token sequence and fifty 500-token ones get the
//                same bytes per warp as fifty-one equal sequences. The partition is computed
//                on the device from seq_lens, so the grid does not depend on the lengths and a
//                CUDA graph of a step stays valid as the sequences grow.
//
//   varlen prefill (packed prompts, Q and O [T, H_q, D], cu_seqlens_q [B + 1]):
//     variant 0: one warp per (token, query head), keys one at a time.
//     variant 1: attention.cu variant 2's tile (128 query rows, 8 warps, mma.sync.m16n8k16,
//                P repacked in registers, 3-stage cp.async on 64-key K/V tiles) with the K/V
//                rows gathered through the block table and the (sequence, q tile) of each
//                block found on the device from cu_seqlens_q. Under the causal mask the new
//                tokens sit at the end of the sequence (bottom-right alignment): query i of a
//                sequence with q_len new tokens and seq_len keys sees keys
//                j <= seq_len - q_len + i, which is the plain causal mask for a fresh prompt
//                and the right one for a chunk appended to an existing context.
//
// Both ops also read an e4m3 cache (paged_decode_fp8, attention_varlen_fp8): the same
// [num_pages, H_kv, page, D] layout at one byte per element, with a per-kv-head scale for K
// and one for V. e4m3 is a subset of bf16, so the kernels convert the bytes to bf16 exactly
// and keep the bf16 mma; the K scale goes into the softmax scale and the V scale into the
// output normalization. Decode variant 1 reads e4m3 slabs from shared memory into registers in
// the fragment layouts (no ldmatrix, no bf16 copy of the slab) and keeps twice the slabs in
// flight; varlen variant 1 converts each 64-key tile to bf16 in shared memory once for its 8
// warps and runs the bf16 tile code unchanged. docs/design/serving.md, "An fp8 K/V cache".
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdlib>

#include "attention_internal.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

using attn::bf16;
using attn::ex2;
using attn::kLog2e;
using attn::pack_bf16x2;
using attn::swz;
using u8 = unsigned char;

// The per-block prefix sums below park one int per sequence in shared memory, next to the
// 96 KB of the decode pipeline, under the 99 KB a block can have on sm_120.
constexpr int kMaxBatch = 512;

// Exclusive prefix over b < B of count(b) into s_pre[0..B] (s_pre[B] is the total). Every
// thread of the block calls it; it ends on a barrier. `scratch` holds 32 ints. Each thread
// sums a contiguous run of sequences, a warp scan and a scan of the warp totals give its
// offset, and it writes its run.
template <typename F>
__device__ void block_prefix(int B, F count, int* s_pre, int* scratch) {
    const int nt = blockDim.x, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5, nw = nt >> 5;
    const int per = (B + nt - 1) / nt;
    const int b0 = min(B, tid * per), b1 = min(B, b0 + per);
    int sum = 0;
    for (int b = b0; b < b1; ++b) sum += count(b);
    int x = sum;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = __shfl_up_sync(kFullMask, x, o);
        if (lane >= o) x += y;
    }
    if (lane == 31) scratch[warp] = x;
    __syncthreads();
    if (warp == 0) {
        int t = lane < nw ? scratch[lane] : 0;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(kFullMask, t, o);
            if (lane >= o) t += y;
        }
        scratch[lane] = t;
    }
    __syncthreads();
    int run = x - sum + (warp > 0 ? scratch[warp - 1] : 0);
    for (int b = b0; b < b1; ++b) {
        s_pre[b] = run;
        run += count(b);
    }
    if (tid == nt - 1) s_pre[B] = scratch[nw - 1];
    __syncthreads();
}

// Largest b in [0, B) with s_pre[b] <= x, for x < s_pre[B]: the sequence that holds item x.
// Sequences with no items share their s_pre with the next one, so they are never the answer.
__device__ __forceinline__ int find_seq(const int* s_pre, int B, int x) {
    int lo = 0, hi = B - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (s_pre[mid] <= x)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}

// Row `j` of kv head `kvh` in the cache: page block_table[j >> shift], row j & (page - 1).
__device__ __forceinline__ size_t cache_row(int page_id, int H_kv, int kvh, int shift, int j) {
    return ((static_cast<size_t>(page_id) * H_kv + kvh) << shift) + (j & ((1 << shift) - 1));
}

// ---- e4m3 helpers --------------------------------------------------------------------------

// The two e4m3 bytes in the low 16 bits of x (the first in the low byte) as a bf16x2 word,
// the first in the low half. cvt.rn.f16x2.e4m3x2 (sm_89+) then f16 -> f32 -> bf16; every
// step is exact, since e4m3 values (2^-9 to 448, 3 mantissa bits) are f16 and bf16 values.
__device__ __forceinline__ unsigned e4m3x2_to_bf16x2(unsigned x) {
    const __half2_raw h =
        __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(x & 0xFFFFu), __NV_E4M3);
    const float2 f = __half22float2(__half2(h));
    return pack_bf16x2(f.x, f.y);
}

// One cache element as a float (the naive rungs).
__device__ __forceinline__ float kv_float(const bf16* p) {
    return __bfloat162float(*p);
}
__device__ __forceinline__ float kv_float(const u8* p) {
    return __half2float(__half(__nv_cvt_fp8_to_halfraw(*p, __NV_E4M3)));
}

// Physical 16-byte chunk of logical (row, chunk) in an e4m3 tile of D-byte rows. D = 128:
// swz's XOR over 8 chunks, which keeps both fragment reads of the decode kernel and the
// conversion reads of the varlen kernel conflict-free. D = 64 (4 chunks a row, two rows per
// 128 bytes): XOR with row / 2, which leaves the decode kernel's 8-byte V reads 2-way
// conflicted instead of 4-way.
template <int D>
__device__ __forceinline__ int swz8(int row, int chunk) {
    static_assert(D == 64 || D == 128, "D = 64 or 128");
    return D == 128 ? chunk ^ (row & 7) : chunk ^ ((row >> 1) & 3);
}

// ---- the naive rungs (variant 0 of both ops) ---------------------------------------------

// One warp computes one query row against keys [0, nkeys) of sequence b, kv head kvh. Lane l
// owns columns l*VEC .. l*VEC+VEC-1; the score of a key is a shuffle reduction.
// `KV` is bf16 or u8 (e4m3); scale_log2 carries the K scale and out_mul the V scale (1 for bf16).
template <int D, typename KV>
__device__ void naive_row(const bf16* q, const KV* K, const KV* V, const int* bt, int H_kv, int kvh,
                          int shift, int nkeys, float scale_log2, float out_mul, bf16* out) {
    constexpr int VEC = D / 32;
    const int lane = threadIdx.x & 31;
    float qf[VEC], acc[VEC];
#pragma unroll
    for (int e = 0; e < VEC; ++e) {
        qf[e] = __bfloat162float(q[lane * VEC + e]);
        acc[e] = 0.f;
    }
    float m = -INFINITY, l = 0.f;
    for (int j = 0; j < nkeys; ++j) {
        const size_t row = cache_row(__ldg(bt + (j >> shift)), H_kv, kvh, shift, j) * D;
        const KV* kr = K + row + lane * VEC;
        float s = 0.f;
#pragma unroll
        for (int e = 0; e < VEC; ++e) s = fmaf(qf[e], kv_float(kr + e), s);
        s = warp_reduce_sum(s) * scale_log2;
        const float m_new = fmaxf(m, s);
        const float alpha = ex2(m - m_new);
        const float pj = ex2(s - m_new);
        l = fmaf(l, alpha, pj);
        const KV* vr = V + row + lane * VEC;
#pragma unroll
        for (int e = 0; e < VEC; ++e) acc[e] = fmaf(acc[e], alpha, pj * kv_float(vr + e));
        m = m_new;
    }
    const float inv = nkeys > 0 ? out_mul / l : 0.f;
#pragma unroll
    for (int e = 0; e < VEC; ++e) out[lane * VEC + e] = __float2bfloat16(acc[e] * inv);
}

// K and V are the caches as bf16 or e4m3 bytes (the kernel's KV); the scales only for e4m3.
struct NaiveDecodeParams {
    const bf16* Q;
    const void* K;
    const void* V;
    bf16* O;
    const int* bt;
    const int* lens;
    int B, H_q, H_kv, max_pages, shift;
    float scale_log2;
    const float* k_scale;
    const float* v_scale;
};

template <int D, typename KV>
__global__ void __launch_bounds__(128) paged_decode_naive_kernel(NaiveDecodeParams p) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);  // (b, h)
    if (row >= p.B * p.H_q) return;
    const int b = row / p.H_q, h = row - b * p.H_q;
    const int kvh = h / (p.H_q / p.H_kv);
    float sl2 = p.scale_log2, om = 1.f;
    if constexpr (sizeof(KV) == 1) {
        sl2 *= p.k_scale[kvh];
        om = p.v_scale[kvh];
    }
    naive_row<D, KV>(p.Q + static_cast<size_t>(row) * D, static_cast<const KV*>(p.K),
                     static_cast<const KV*>(p.V), p.bt + static_cast<size_t>(b) * p.max_pages,
                     p.H_kv, kvh, p.shift, max(0, p.lens[b]), sl2, om,
                     p.O + static_cast<size_t>(row) * D);
}

struct NaiveVarlenParams {
    const bf16* Q;
    const void* K;
    const void* V;
    bf16* O;
    const int* cu_q;
    const int* lens;
    const int* bt;
    int B, T, H_q, H_kv, max_pages, shift, causal;
    float scale_log2;
    const float* k_scale;
    const float* v_scale;
};

template <int D, typename KV>
__global__ void __launch_bounds__(128) varlen_naive_kernel(NaiveVarlenParams p) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);  // (token, h)
    if (row >= p.T * p.H_q) return;
    const int t = row / p.H_q, h = row - t * p.H_q;
    int lo = 0, hi = p.B - 1;  // the sequence whose tokens [cu_q[b], cu_q[b+1]) hold t
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (p.cu_q[mid] <= t)
            lo = mid;
        else
            hi = mid - 1;
    }
    const int b = lo;
    const int q_len = p.cu_q[b + 1] - p.cu_q[b], kv_len = p.lens[b];
    const int nkeys = p.causal ? kv_len - q_len + (t - p.cu_q[b]) + 1 : kv_len;
    const int kvh = h / (p.H_q / p.H_kv);
    float sl2 = p.scale_log2, om = 1.f;
    if constexpr (sizeof(KV) == 1) {
        sl2 *= p.k_scale[kvh];
        om = p.v_scale[kvh];
    }
    naive_row<D, KV>(p.Q + static_cast<size_t>(row) * D, static_cast<const KV*>(p.K),
                     static_cast<const KV*>(p.V), p.bt + static_cast<size_t>(b) * p.max_pages,
                     p.H_kv, kvh, p.shift, nkeys, sl2, om, p.O + static_cast<size_t>(row) * D);
}

// ---- paged flash-decoding (decode variant 1) ---------------------------------------------

namespace dec {

constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;
constexpr int ROWS = 16;  // one m16 tile of query rows: the group's heads
constexpr int SLAB = 16;  // keys per pipeline stage, and the smallest page
constexpr int STAGES = 3;
// An e4m3 stage is half the bytes, so twice the stages fit the same 96 KB: 5 slabs, 20 KB, in
// flight per warp against bf16's 2 slabs, 16 KB. Bytes in flight are what keep DRAM busy
// (the kernel waits on memory and nothing else), so the stages double rather than the slab:
// the slab stays the 16-key unit of the split, the page minimum and the one k16 step of P V.
constexpr int STAGES_FP8 = 6;

template <bool FP8>
constexpr int stages() {
    return FP8 ? STAGES_FP8 : STAGES;
}
// A stage is a K slab then a V slab, 8 KB at D = 128 (4 KB in e4m3).
template <int D>
constexpr int stage_elems() {
    return 2 * SLAB * D;
}
template <int D, bool FP8>
constexpr int stage_bytes() {
    return stage_elems<D>() * (FP8 ? 1 : 2);
}
template <int D, bool FP8>
constexpr int smem_bytes() {
    return WARPS * stages<FP8>() * stage_bytes<D, FP8>();  // 96 KB at D = 128, either format
}

// DT consecutive output columns of accumulator element e, times mul, as bf16 (16-byte stores)
// or fp32 (float4 stores): the e4m3 path's O, whose columns are permuted (see the kernel).
template <int DT>
__device__ __forceinline__ void store_run_bf16(bf16* dst, const float (&o)[DT][4], int e,
                                               float mul) {
    static_assert(DT % 8 == 0, "whole 16-byte stores");
#pragma unroll
    for (int j = 0; j < DT; j += 8) {
        uint4 u;
        u.x = pack_bf16x2(o[j][e] * mul, o[j + 1][e] * mul);
        u.y = pack_bf16x2(o[j + 2][e] * mul, o[j + 3][e] * mul);
        u.z = pack_bf16x2(o[j + 4][e] * mul, o[j + 5][e] * mul);
        u.w = pack_bf16x2(o[j + 6][e] * mul, o[j + 7][e] * mul);
        *reinterpret_cast<uint4*>(dst + j) = u;
    }
}
template <int DT>
__device__ __forceinline__ void store_run_f32(float* dst, const float (&o)[DT][4], int e,
                                              float mul) {
#pragma unroll
    for (int j = 0; j < DT; j += 4)
        *reinterpret_cast<float4*>(dst + j) =
            make_float4(o[j][e] * mul, o[j + 1][e] * mul, o[j + 2][e] * mul, o[j + 3][e] * mul);
}

// N bytes (8 or 16) from shared memory as N / 4 words.
template <int N>
__device__ __forceinline__ void load_words(unsigned (&w)[N / 4], const u8* src) {
    static_assert(N == 8 || N == 16, "one 8- or 16-byte load");
    if constexpr (N == 16) {
        const uint4 u = *reinterpret_cast<const uint4*>(src);
        w[0] = u.x;
        w[1] = u.y;
        w[2] = u.z;
        w[3] = u.w;
    } else {
        const uint2 u = *reinterpret_cast<const uint2*>(src);
        w[0] = u.x;
        w[1] = u.y;
    }
}
// One piece: ROWS unnormalized fp32 O rows, then m, then l.
template <int D>
constexpr int piece_floats() {
    return ROWS * (D + 2);
}

struct Params {
    const bf16* Q;
    const bf16* K;
    const bf16* V;
    bf16* O;
    const int* bt;
    const int* lens;
    int B, H_q, H_kv, group, max_pages, shift;
    int min_slabs;  // slabs per active warp at least: fewer warps on a short batch
    int grid;       // blocks of the decode kernel (the combine kernel needs the same partition)
    float scale_log2;
    float* ws;     // two pieces per warp of the grid
    int* meta;     // the slab prefix s_pre[0..B], published by block 0 for the combine kernel
    const u8* K8;  // the e4m3 caches and their [H_kv] scales (FP8 only)
    const u8* V8;
    const float* k_scale;
    const float* v_scale;
};

// The flat slab list: segment (b, kvh) holds n_b = ceil(len_b / SLAB) slabs and starts at
// s_pre[b] * H_kv + kvh * n_b. Warp w of Weff owns [w T / Weff, (w + 1) T / Weff).
struct Cursor {
    int b, kvh, j, n;  // sequence, kv head, slab within the segment, slabs in the segment
};

__device__ __forceinline__ Cursor locate(const int* s_pre, int B, int H_kv, int x) {
    Cursor c;
    // Segment starts are s_pre[b] * H_kv + kvh * n_b, so the sequence is the last b with
    // s_pre[b] * H_kv <= x.
    int lo = 0, hi = B - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (s_pre[mid] * H_kv <= x)
            lo = mid;
        else
            hi = mid - 1;
    }
    c.b = lo;
    c.n = s_pre[lo + 1] - s_pre[lo];
    const int rem = x - s_pre[lo] * H_kv;
    c.kvh = rem / c.n;
    c.j = rem - c.kvh * c.n;
    return c;
}

__device__ __forceinline__ void advance(Cursor& c, const int* s_pre, int B, int H_kv) {
    if (++c.j < c.n) return;
    c.j = 0;
    if (++c.kvh < H_kv) return;
    c.kvh = 0;
    do {
        ++c.b;
    } while (c.b < B && s_pre[c.b + 1] == s_pre[c.b]);
    c.n = c.b < B ? s_pre[c.b + 1] - s_pre[c.b] : 0;
}

__device__ __forceinline__ int active_warps(int T, int grid, int min_slabs) {
    return min(grid * WARPS, max(1, T / min_slabs));
}
__device__ __forceinline__ int range_start(int w, int T, int weff) {
    return static_cast<int>(static_cast<int64_t>(w) * T / weff);
}
// The warp whose range holds slab x: the largest w with range_start(w) <= x.
__device__ __forceinline__ int warp_of(int x, int T, int weff) {
    return static_cast<int>((static_cast<int64_t>(x + 1) * weff + T - 1) / T) - 1;
}

// FP8: the caches are e4m3 (p.K8, p.V8). A slab is read from shared memory straight into
// the mma fragments, converted to bf16 in registers. K: for its key rows g and g + 8 a lane
// needs, in k16 step kk, d pairs (2c, 2c + 1) and (2c + 8, 2c + 9), 2 bytes out of every 8
// of the row. The dot product does not care which d sits at which k position as long as Q
// and K agree, so lane quad c takes the contiguous bytes [c D/4, (c + 1) D/4) of the row and
// step kk takes bytes 4 kk .. 4 kk + 3 of that run, (2c, 2c + 1) the low pair and (2c + 8,
// 2c + 9) the high pair: two 16-byte loads per row (one at D = 64), and the Q fragments are
// loaded with the same d. V: B of P V wants, for output column n = g of tile dj, keys 2c,
// 2c + 1 (and + 8) of one d. The output columns are permuted instead: column n of tile dj is
// d = n DT + dj (DT = D/8 tiles), so a lane reads DT contiguous bytes of each of its four key
// rows and pairs bytes across rows with byte permutes. Accumulator element (row, 2c + i) of
// tile dj is then d = (2c + i) DT + dj, and a lane writes two runs of DT consecutive columns.
template <int D, bool FP8>
__global__ void __launch_bounds__(THREADS, 1) paged_decode_kernel(Params p) {
    constexpr int CH = D / 8;                   // 16-byte chunks per row
    constexpr int KT = D / 16;                  // k16 steps of Q K^T
    constexpr int DT = D / 8;                   // n8 tiles of O
    constexpr int NT = SLAB / 8;                // n8 tiles of S: 2
    constexpr int KV_ITERS = SLAB * CH / 32;    // chunks per lane per operand per slab: 8, 4
    constexpr int CH8 = D / 16;                 // 16-byte chunks per e4m3 row
    constexpr int KV_ITERS8 = SLAB * CH8 / 32;  // 4, 2
    constexpr int ST = stages<FP8>();
    constexpr int SB = stage_bytes<D, FP8>();
    constexpr int PF = piece_floats<D>();
    static_assert(SLAB == 16, "P V below is one k16 step");
    static_assert(KV_ITERS8 * 32 == SLAB * CH8 && KT % 4 == 0 && DT % 4 == 0, "");

    extern __shared__ __align__(128) unsigned char smem_raw[];
    __shared__ int s_pre[kMaxBatch + 1];
    __shared__ int s_scan[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    [[maybe_unused]] const int t4 = lane & 3;
    const int B = p.B, H_kv = p.H_kv;

    block_prefix(B, [&](int b) { return cdiv(max(0, p.lens[b]), SLAB); }, s_pre, s_scan);
    if (blockIdx.x == 0)
        for (int b = tid; b <= B; b += THREADS) p.meta[b] = s_pre[b];
    const int T = s_pre[B] * H_kv;
    if (T == 0) return;
    const int weff = active_warps(T, p.grid, p.min_slabs);
    // Warp-major numbering, so a short batch spreads its active warps over every SM.
    const int w = warp * gridDim.x + blockIdx.x;
    if (w >= weff) return;  // no block-wide barrier from here on
    const int a = range_start(w, T, weff), e = range_start(w + 1, T, weff);

    // This warp's stages, as bytes (the bf16 path indexes them in elements).
    u8* wsm = smem_raw + warp * ST * SB;

    // Loads run ST - 1 slabs ahead of the compute. `lpage` is the page of the next slab to
    // load, fetched one load early so the address is ready when the load issues.
    Cursor L = locate(s_pre, B, H_kv, a);
    int nl = a;
    auto page_of = [&](const Cursor& c) {
        return __ldg(p.bt + static_cast<size_t>(c.b) * p.max_pages + ((c.j * SLAB) >> p.shift));
    };
    int lpage = page_of(L);
    auto load = [&](int stage) {
        const int len = __ldg(p.lens + L.b);
        const int key0 = L.j * SLAB;
        const size_t base = cache_row(lpage, H_kv, L.kvh, p.shift, key0);
        if constexpr (FP8) {
            u8* ks = wsm + stage * SB;
            u8* vs = ks + SLAB * D;
#pragma unroll
            for (int i = 0; i < KV_ITERS8; ++i) {
                const int c = lane + i * 32;
                const int row = c / CH8, ch = c % CH8;
                const bool ok = key0 + row < len;
                const size_t off = (base + (ok ? row : 0)) * D + ch * 16;
                cp_async_16_zfill(ks + row * D + swz8<D>(row, ch) * 16, p.K8 + off, ok);
                cp_async_16_zfill(vs + row * D + swz8<D>(row, ch) * 16, p.V8 + off, ok);
            }
        } else {
            bf16* ks = reinterpret_cast<bf16*>(wsm) + stage * stage_elems<D>();
            bf16* vs = ks + SLAB * D;
#pragma unroll
            for (int i = 0; i < KV_ITERS; ++i) {
                const int c = lane + i * 32;
                const int row = c / CH, ch = c % CH;
                const bool ok = key0 + row < len;
                const size_t off = (base + (ok ? row : 0)) * D + ch * 8;
                cp_async_16_zfill(ks + row * D + swz(row, ch) * 8, p.K + off, ok);
                cp_async_16_zfill(vs + row * D + swz(row, ch) * 8, p.V + off, ok);
            }
        }
        advance(L, s_pre, B, H_kv);
        if (++nl < e) lpage = page_of(L);
    };
#pragma unroll
    for (int s = 0; s < ST - 1; ++s) {
        if (nl < e) load(s);
        cp_async_commit();
    }

    Cursor C = locate(s_pre, B, H_kv, a);
    bool first_seg = true;
    int piece_j0 = C.j;
    unsigned qf[KT][4];
    float o[DT][4];
    float m[2], l[2];
    float sl2 = p.scale_log2;           // the softmax scale (log2 units), times K's scale in FP8
    [[maybe_unused]] float vmul = 1.f;  // V's scale (FP8)

    for (int x = a, it = 0; x < e; ++x, ++it) {
        cp_async_wait<ST - 2>();  // slab x has landed for this lane
        __syncwarp();             // ... for every lane; and stage (it-1)%ST is free
        if (nl < e) load((it + ST - 1) % ST);
        cp_async_commit();
        const u8* stg = wsm + (it % ST) * SB;

        if (x == a || C.j == 0) {  // a new segment: its Q fragments, a fresh softmax state
            // The A fragments straight from global memory (rows g and g + 8, k pairs c2 and
            // 8 + c2 of each k16 step), rows past the group zero. The rows were written by the
            // RoPE kernel just before, so this is an L2 round trip, taken while the next
            // slabs are already in flight.
            const unsigned* qg = reinterpret_cast<const unsigned*>(
                p.Q +
                (static_cast<size_t>(C.b) * p.H_q + static_cast<size_t>(C.kvh) * p.group) * D);
            const bool r0 = g < p.group, r1 = g + 8 < p.group;
            if constexpr (FP8) {
                // The permuted d of the K fragments: step kk of quad c holds d = c D/4 + 4 kk
                // + {0, 1} in its low pair and + {2, 3} in its high pair (word c D/8 + 2 kk).
#pragma unroll
                for (int kk = 0; kk < KT; ++kk) {
                    const int k0 = t4 * (D / 8) + 2 * kk;
                    qf[kk][0] = r0 ? __ldg(qg + (g * D) / 2 + k0) : 0u;
                    qf[kk][1] = r1 ? __ldg(qg + ((g + 8) * D) / 2 + k0) : 0u;
                    qf[kk][2] = r0 ? __ldg(qg + (g * D) / 2 + k0 + 1) : 0u;
                    qf[kk][3] = r1 ? __ldg(qg + ((g + 8) * D) / 2 + k0 + 1) : 0u;
                }
                sl2 = p.scale_log2 * __ldg(p.k_scale + C.kvh);
                vmul = __ldg(p.v_scale + C.kvh);
            } else {
#pragma unroll
                for (int kk = 0; kk < KT; ++kk) {
                    const int k0 = (kk * 16 + c2) / 2;
                    qf[kk][0] = r0 ? __ldg(qg + (g * D) / 2 + k0) : 0u;
                    qf[kk][1] = r1 ? __ldg(qg + ((g + 8) * D) / 2 + k0) : 0u;
                    qf[kk][2] = r0 ? __ldg(qg + (g * D) / 2 + k0 + 4) : 0u;
                    qf[kk][3] = r1 ? __ldg(qg + ((g + 8) * D) / 2 + k0 + 4) : 0u;
                }
            }
#pragma unroll
            for (int dj = 0; dj < DT; ++dj)
#pragma unroll
                for (int k = 0; k < 4; ++k) o[dj][k] = 0.f;
            m[0] = m[1] = -INFINITY;
            l[0] = l[1] = 0.f;
            piece_j0 = C.j;
        }

        // S = Q K^T: 16 rows x 16 keys.
        float sc[NT][4];
#pragma unroll
        for (int nj = 0; nj < NT; ++nj)
#pragma unroll
            for (int k = 0; k < 4; ++k) sc[nj][k] = 0.f;
        if constexpr (FP8) {
            // Key row nj * 8 + g, bytes [c D/4, (c + 1) D/4): KT words, word kk for step kk.
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                const int row = nj * 8 + g;
                unsigned kw[KT];
#pragma unroll
                for (int h = 0; h < KT / 4; ++h) {
                    const uint4 u = *reinterpret_cast<const uint4*>(
                        stg + row * D + swz8<D>(row, t4 * (KT / 4) + h) * 16);
                    kw[4 * h] = u.x;
                    kw[4 * h + 1] = u.y;
                    kw[4 * h + 2] = u.z;
                    kw[4 * h + 3] = u.w;
                }
#pragma unroll
                for (int kk = 0; kk < KT; ++kk) {
                    const unsigned b[2] = {e4m3x2_to_bf16x2(kw[kk]),
                                           e4m3x2_to_bf16x2(kw[kk] >> 16)};
                    mma_bf16_16816(sc[nj], qf[kk], b);
                }
            }
        } else {
            const bf16* ks = reinterpret_cast<const bf16*>(stg);
#pragma unroll
            for (int kk = 0; kk < KT; ++kk) {
                const int row = lane & 15;
                unsigned r[4];
                ldmatrix_x4(r, ks + row * D + swz(row, 2 * kk + (lane >> 4)) * 8);
                const unsigned b0[2] = {r[0], r[2]};
                const unsigned b1[2] = {r[1], r[3]};
                mma_bf16_16816(sc[0], qf[kk], b0);
                mma_bf16_16816(sc[1], qf[kk], b1);
            }
        }

        const int len = __ldg(p.lens + C.b);
        const int key0 = C.j * SLAB;
        if (key0 + SLAB > len) {  // the segment's last, partial slab (warp-uniform)
#pragma unroll
            for (int nj = 0; nj < NT; ++nj)
#pragma unroll
                for (int k = 0; k < 4; ++k)
                    if (key0 + nj * 8 + c2 + (k & 1) >= len) sc[nj][k] = -INFINITY;
        }

#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float mx =
                fmaxf(fmaxf(sc[0][2 * r], sc[0][2 * r + 1]), fmaxf(sc[1][2 * r], sc[1][2 * r + 1]));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 2));
            const float m_new = fmaxf(m[r], mx * sl2);  // finite: slab has a key
            const float alpha = ex2(m[r] - m_new);
            float rs = 0.f;
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                const float p0 = ex2(fmaf(sc[nj][2 * r], sl2, -m_new));
                const float p1 = ex2(fmaf(sc[nj][2 * r + 1], sl2, -m_new));
                sc[nj][2 * r] = p0;
                sc[nj][2 * r + 1] = p1;
                rs += p0 + p1;
            }
            l[r] = fmaf(l[r], alpha, rs);
            m[r] = m_new;
#pragma unroll
            for (int dj = 0; dj < DT; ++dj) {
                o[dj][2 * r] *= alpha;
                o[dj][2 * r + 1] *= alpha;
            }
        }

        // O += P V, one k16 step: P's A fragment straight from the S accumulators.
        unsigned pa[4];
        pa[0] = pack_bf16x2(sc[0][0], sc[0][1]);
        pa[1] = pack_bf16x2(sc[0][2], sc[0][3]);
        pa[2] = pack_bf16x2(sc[1][0], sc[1][1]);
        pa[3] = pack_bf16x2(sc[1][2], sc[1][3]);
        if constexpr (FP8) {
            // Key rows 2c, 2c + 1, 2c + 8, 2c + 9, bytes [g DT, (g + 1) DT): byte dj of a row
            // is d = g DT + dj, column g of tile dj. __byte_perm pairs byte i of two rows
            // (selector 0x5140: bytes 0 and 1, 0x7362: bytes 2 and 3) into e4m3x2 halves,
            // the lower key in the low byte.
            const u8* vs = stg + SLAB * D;
            unsigned vw[4][DT / 4];
#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const int row = 2 * t4 + (rr & 1) + (rr >> 1) * 8;
                load_words<DT>(vw[rr],
                               vs + row * D + swz8<D>(row, (g * DT) >> 4) * 16 + ((g * DT) & 15));
            }
#pragma unroll
            for (int q4 = 0; q4 < DT / 4; ++q4) {
                const unsigned lo0 = __byte_perm(vw[0][q4], vw[1][q4], 0x5140);
                const unsigned hi0 = __byte_perm(vw[0][q4], vw[1][q4], 0x7362);
                const unsigned lo1 = __byte_perm(vw[2][q4], vw[3][q4], 0x5140);
                const unsigned hi1 = __byte_perm(vw[2][q4], vw[3][q4], 0x7362);
                const unsigned b0[2] = {e4m3x2_to_bf16x2(lo0), e4m3x2_to_bf16x2(lo1)};
                const unsigned b1[2] = {e4m3x2_to_bf16x2(lo0 >> 16), e4m3x2_to_bf16x2(lo1 >> 16)};
                const unsigned b2[2] = {e4m3x2_to_bf16x2(hi0), e4m3x2_to_bf16x2(hi1)};
                const unsigned b3[2] = {e4m3x2_to_bf16x2(hi0 >> 16), e4m3x2_to_bf16x2(hi1 >> 16)};
                mma_bf16_16816(o[4 * q4], pa, b0);
                mma_bf16_16816(o[4 * q4 + 1], pa, b1);
                mma_bf16_16816(o[4 * q4 + 2], pa, b2);
                mma_bf16_16816(o[4 * q4 + 3], pa, b3);
            }
        } else {
            const bf16* vs = reinterpret_cast<const bf16*>(stg) + SLAB * D;
#pragma unroll
            for (int dj = 0; dj < DT; dj += 2) {
                const int row = lane & 15;
                unsigned r[4];
                ldmatrix_x4_trans(r, vs + row * D + swz(row, dj + (lane >> 4)) * 8);
                const unsigned b0[2] = {r[0], r[1]};
                const unsigned b1[2] = {r[2], r[3]};
                mma_bf16_16816(o[dj], pa, b0);
                mma_bf16_16816(o[dj + 1], pa, b1);
            }
        }

        // End of a segment or of the range: this warp's piece of segment (C.b, C.kvh) is done.
        // A whole segment is normalized and written; a piece goes to the warp's slot 0 (the
        // first segment of its range) or 1 (the last) for the combine kernel. In FP8 the
        // columns are the permuted ones above and V's scale is applied here, also to a piece,
        // so the combine is the same for both formats.
        if (C.j == C.n - 1 || x == e - 1) {
            const bool whole = piece_j0 == 0 && C.j == C.n - 1;
            float* part = p.ws + (static_cast<size_t>(w) * 2 + (first_seg ? 0 : 1)) * PF;
            bf16* og =
                p.O + (static_cast<size_t>(C.b) * p.H_q + static_cast<size_t>(C.kvh) * p.group) * D;
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                float ls = l[r];
                ls += __shfl_xor_sync(kFullMask, ls, 1);
                ls += __shfl_xor_sync(kFullMask, ls, 2);
                const int row = g + 8 * r;
                if (row >= p.group) continue;
                if constexpr (FP8) {
                    const int d0 = 2 * t4 * DT, d1 = (2 * t4 + 1) * DT;
                    if (whole) {
                        const float mul = vmul / ls;
                        store_run_bf16<DT>(og + row * D + d0, o, 2 * r, mul);
                        store_run_bf16<DT>(og + row * D + d1, o, 2 * r + 1, mul);
                    } else {
                        store_run_f32<DT>(part + row * D + d0, o, 2 * r, vmul);
                        store_run_f32<DT>(part + row * D + d1, o, 2 * r + 1, vmul);
                        if ((lane & 3) == 0) {
                            part[ROWS * D + row] = m[r];
                            part[ROWS * D + ROWS + row] = ls;
                        }
                    }
                } else if (whole) {
                    const float inv = 1.f / ls;
#pragma unroll
                    for (int dj = 0; dj < DT; ++dj)
                        *reinterpret_cast<__nv_bfloat162*>(og + row * D + dj * 8 + c2) =
                            __floats2bfloat162_rn(o[dj][2 * r] * inv, o[dj][2 * r + 1] * inv);
                } else {
#pragma unroll
                    for (int dj = 0; dj < DT; ++dj)
                        *reinterpret_cast<float2*>(part + row * D + dj * 8 + c2) =
                            make_float2(o[dj][2 * r], o[dj][2 * r + 1]);
                    if ((lane & 3) == 0) {
                        part[ROWS * D + row] = m[r];
                        part[ROWS * D + ROWS + row] = ls;
                    }
                }
            }
            first_seg = false;
        }
        advance(C, s_pre, B, H_kv);
    }
    cp_async_wait<0>();
}

// Online-softmax merge of partial (M, L, acc) into (m, l, x): the running state of a row.
__device__ __forceinline__ void merge_state(float& M, float& L, float (&acc)[8], float m, float l,
                                            const float (&x)[8]) {
    const float mn = fmaxf(M, m);
    // -inf on both sides is an empty state; the guards keep ex2(-inf - -inf) out.
    const float sa = M == -INFINITY ? 0.f : ex2(M - mn);
    const float sb = m == -INFINITY ? 0.f : ex2(m - mn);
    L = L * sa + l * sb;
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = acc[k] * sa + x[k] * sb;
    M = mn;
}

constexpr int COMBINE_THREADS = 128;

// One block per (b, kv head, query row of the group), after the decode kernel. A segment of
// no keys gets zeros; a segment one warp covered was written by it; otherwise the pieces of
// warps warp_of(start) .. warp_of(end - 1) are merged with the online-softmax rule. A long
// segment has a piece per warp that streamed it (85 for one 32K sequence on 680 warps), and
// the merge is all L2 latency, so the 128 threads are D/8 column chunks x 128/(D/8) lanes: lane
// k merges pieces k, k + lanes, ..., and xor shuffles merge the lanes' states. The prefix
// over the lengths comes from `meta`, which the decode kernel's block 0 wrote.
template <int D>
__global__ void __launch_bounds__(COMBINE_THREADS) paged_combine_kernel(Params p) {
    constexpr int CH = D / 8;
    constexpr int LANES = COMBINE_THREADS / CH;  // 8 at D = 128, 16 at D = 64
    constexpr int PF = piece_floats<D>();
    static_assert(LANES <= 32 && 32 % LANES == 0, "the lanes of a chunk share a warp");
    const int seg = blockIdx.x / p.group, row = blockIdx.x - seg * p.group;
    const int b = seg / p.H_kv, kvh = seg - b * p.H_kv;
    const int pre = p.meta[b], n = p.meta[b + 1] - pre;
    const int ch = threadIdx.x / LANES, pl = threadIdx.x % LANES;
    bf16* og = p.O +
               ((static_cast<size_t>(b) * p.H_q + static_cast<size_t>(kvh) * p.group) + row) * D +
               ch * 8;
    if (n == 0) {
        if (pl == 0) *reinterpret_cast<uint4*>(og) = make_uint4(0, 0, 0, 0);
        return;
    }
    const int T = p.meta[p.B] * p.H_kv;
    const int weff = active_warps(T, p.grid, p.min_slabs);
    const int s = pre * p.H_kv + kvh * n;
    const int wf = warp_of(s, T, weff), wl = warp_of(s + n - 1, T, weff);
    if (wf == wl) return;
    // The first warp holds this segment in slot 1 unless its range starts here.
    const int slot_f = range_start(wf, T, weff) < s ? 1 : 0;
    float M = -INFINITY, L = 0.f, acc[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll 4
    for (int w = wf + pl; w <= wl; w += LANES) {
        const float* part = p.ws + (static_cast<size_t>(w) * 2 + (w == wf ? slot_f : 0)) * PF;
        const float4 x0 = __ldcg(reinterpret_cast<const float4*>(part + row * D + ch * 8));
        const float4 x1 = __ldcg(reinterpret_cast<const float4*>(part + row * D + ch * 8 + 4));
        const float x[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
        merge_state(M, L, acc, __ldcg(part + ROWS * D + row), __ldcg(part + ROWS * D + ROWS + row),
                    x);
    }
    if (wl - wf + 1 > 1) {  // block-uniform: the lanes hold more than one state
#pragma unroll
        for (int off = LANES / 2; off > 0; off >>= 1) {
            float x[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) x[k] = __shfl_xor_sync(kFullMask, acc[k], off);
            const float m = __shfl_xor_sync(kFullMask, M, off);
            const float l = __shfl_xor_sync(kFullMask, L, off);
            merge_state(M, L, acc, m, l, x);
        }
    }
    if (pl == 0) {
        const float inv = 1.f / L;
        uint4 u;
        u.x = pack_bf16x2(acc[0] * inv, acc[1] * inv);
        u.y = pack_bf16x2(acc[2] * inv, acc[3] * inv);
        u.z = pack_bf16x2(acc[4] * inv, acc[5] * inv);
        u.w = pack_bf16x2(acc[6] * inv, acc[7] * inv);
        *reinterpret_cast<uint4*>(og) = u;
    }
}

// Pieces and the published prefix, one buffer per process, grown on demand; shared by every
// stream, so two paged decode launches must not run concurrently on different streams.
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
int* meta_buffer() {
    static int* meta = nullptr;
    if (!meta) SPARK_CUDA_CHECK(cudaMalloc(&meta, (kMaxBatch + 1) * sizeof(int)));
    return meta;
}

template <int D, bool FP8>
int resident_blocks() {
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(paged_decode_kernel<D, FP8>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              smem_bytes<D, FP8>()));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, paged_decode_kernel<D, FP8>, THREADS, smem_bytes<D, FP8>()));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

// SPARK_PAGED_MIN_SLABS=n sets the fewest slabs an active warp streams (the sweep in
// docs/design/serving.md); SPARK_PAGED_GRID=n caps the blocks.
int env_int(const char* name, int def) {
    if (const char* e = std::getenv(name)) {
        const int v = std::atoi(e);
        if (v > 0) return v;
    }
    return def;
}

template <int D, bool FP8>
void launch(Params p, cudaStream_t stream) {
    static const int min_slabs = env_int("SPARK_PAGED_MIN_SLABS", 4);
    static const int grid_cap = env_int("SPARK_PAGED_GRID", 1 << 30);
    p.grid = std::min(resident_blocks<D, FP8>(), grid_cap);
    p.min_slabs = min_slabs;
    p.ws = workspace(static_cast<size_t>(p.grid) * WARPS * 2 * piece_floats<D>());
    p.meta = meta_buffer();
    paged_decode_kernel<D, FP8><<<p.grid, THREADS, smem_bytes<D, FP8>(), stream>>>(p);
    SPARK_CHECK_LAUNCH();
    paged_combine_kernel<D><<<p.B * p.H_kv * p.group, COMBINE_THREADS, 0, stream>>>(p);
    SPARK_CHECK_LAUNCH();
}

}  // namespace dec

// ---- varlen prefill (varlen variant 1) ---------------------------------------------------

namespace vl {

constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int BM = 16 * WARPS;  // 128 query rows
constexpr int BN = 64;          // keys per K/V tile
constexpr int STAGES = 3;

template <int D>
constexpr int stage_elems() {
    return 2 * BN * D;  // K tile then V tile
}
// bf16: STAGES bf16 stages, 96 KB at D = 128. FP8: one bf16 tile (the Q staging, then each
// K/V tile converted for the 8 warps), then STAGES e4m3 stages: 32 + 48 = 80 KB at D = 128.
template <int D, bool FP8>
constexpr int smem_bytes() {
    return FP8 ? stage_elems<D>() * 2 + STAGES * stage_elems<D>() : STAGES * stage_elems<D>() * 2;
}

struct Params {
    const bf16* Q;
    const bf16* K;
    const bf16* V;
    bf16* O;
    const int* cu_q;
    const int* lens;
    const int* bt;
    int B, H_q, H_kv, group, max_pages, shift, causal;
    float scale_log2;
    const u8* K8;  // the e4m3 caches and their [H_kv] scales (FP8 only)
    const u8* V8;
    const float* k_scale;
    const float* v_scale;
};

// One ROWS x D e4m3 tile (rows of D bytes, chunks at swz8) to bf16 in the bf16 tiles' layout
// (rows of D elements, chunks at swz), by the block's THREADS_ threads. A thread converts 16
// bytes into two 16-byte chunks; it writes the second one first when (ch / 4) is odd, so the 8
// threads of a store phase (one row at D = 128) land in 8 different bank groups.
template <int D, int ROWS_, int THREADS_>
__device__ __forceinline__ void tile_e4m3_to_bf16(const u8* src, bf16* dst, int tid) {
    constexpr int CH8 = D / 16;
    constexpr int ITERS = ROWS_ * CH8 / THREADS_;
    static_assert(ITERS * THREADS_ == ROWS_ * CH8, "whole chunks per thread");
#pragma unroll
    for (int i = 0; i < ITERS; ++i) {
        const int c = tid + i * THREADS_;
        const int row = c / CH8, ch = c % CH8;
        const uint4 u = *reinterpret_cast<const uint4*>(src + row * D + swz8<D>(row, ch) * 16);
        uint4 lo, hi;  // d 16 ch .. 16 ch + 7, then 16 ch + 8 .. 16 ch + 15
        lo.x = e4m3x2_to_bf16x2(u.x);
        lo.y = e4m3x2_to_bf16x2(u.x >> 16);
        lo.z = e4m3x2_to_bf16x2(u.y);
        lo.w = e4m3x2_to_bf16x2(u.y >> 16);
        hi.x = e4m3x2_to_bf16x2(u.z);
        hi.y = e4m3x2_to_bf16x2(u.z >> 16);
        hi.z = e4m3x2_to_bf16x2(u.w);
        hi.w = e4m3x2_to_bf16x2(u.w >> 16);
        bf16* r = dst + row * D;
        const int f = (ch >> 2) & 1;
        *reinterpret_cast<uint4*>(r + swz(row, 2 * ch + f) * 8) = f ? hi : lo;
        *reinterpret_cast<uint4*>(r + swz(row, 2 * ch + 1 - f) * 8) = f ? lo : hi;
    }
}

// FP8: the K/V tiles arrive as e4m3 in their own stages, past one bf16 tile at the start of
// shared memory. After a tile lands the block converts it into the bf16 tile (each element
// once, where converting in registers would do it once per warp, 8 times) and the compute
// below runs on it unchanged, with K's scale in the softmax scale and V's in the final
// normalization.
template <int D, bool FP8>
__global__ void __launch_bounds__(THREADS, 1) varlen_kernel(Params p) {
    constexpr int CH = D / 8;
    constexpr int KT = D / 16;
    constexpr int DT = D / 8;
    constexpr int NT = BN / 8;
    constexpr int PT = BN / 16;
    constexpr int Q_ITERS = BM * CH / THREADS;
    constexpr int KV_ITERS = BN * CH / THREADS;
    constexpr int CH8 = D / 16;
    constexpr int KV_ITERS8 = BN * CH8 / THREADS;  // 2, 1
    constexpr int SB8 = stage_elems<D>();          // bytes of an e4m3 stage
    static_assert(Q_ITERS * THREADS == BM * CH && KV_ITERS * THREADS == BN * CH, "");
    static_assert(KV_ITERS8 * THREADS == BN * CH8, "");
    static_assert(BM * D <= stage_elems<D>(), "the Q tile is staged through one K/V stage");

    extern __shared__ __align__(128) unsigned char smem_raw[];
    __shared__ int s_pre[kMaxBatch + 1];
    __shared__ int s_scan[32];
    bf16* smem = reinterpret_cast<bf16*>(smem_raw);
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;

    // Q tiles of each sequence, then this block's (sequence, tile, head). Block order is
    // tile-major over the heads, so the heads that share a K/V head are neighbours, and
    // within a sequence the tiles run heaviest first under the causal mask.
    block_prefix(p.B, [&](int b) { return cdiv(p.cu_q[b + 1] - p.cu_q[b], BM); }, s_pre, s_scan);
    const int tile = blockIdx.x / p.H_q, h = blockIdx.x - tile * p.H_q;
    if (tile >= s_pre[p.B]) return;  // the grid is an upper bound on the tile count
    const int b = find_seq(s_pre, p.B, tile);
    const int nq = s_pre[b + 1] - s_pre[b], rank = tile - s_pre[b];
    const int q_tile = p.causal ? nq - 1 - rank : rank;
    const int q0 = q_tile * BM;
    const int tok0 = p.cu_q[b];
    const int q_len = p.cu_q[b + 1] - tok0, kv_len = p.lens[b];
    const int ctx = kv_len - q_len;  // keys before the first new token
    const int kvh = h / p.group;
    const int* btb = p.bt + static_cast<size_t>(b) * p.max_pages;
    const size_t q_stride = static_cast<size_t>(p.H_q) * D;  // between tokens of one head
    const bf16* Qg = p.Q + (static_cast<size_t>(tok0) * p.H_q + h) * D;
    bf16* Og = p.O + (static_cast<size_t>(tok0) * p.H_q + h) * D;

#pragma unroll
    for (int i = 0; i < Q_ITERS; ++i) {
        const int c = tid + i * THREADS;
        const int row = c / CH, ch = c % CH;
        const bool ok = q0 + row < q_len;
        cp_async_16_zfill(smem + row * D + swz(row, ch) * 8,
                          Qg + static_cast<size_t>(ok ? q0 + row : 0) * q_stride + ch * 8, ok);
    }
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    unsigned qf[KT][4];
    {
        const int row = warp * 16 + (lane & 15);
#pragma unroll
        for (int kk = 0; kk < KT; ++kk)
            ldmatrix_x4(qf[kk], smem + row * D + swz(row, 2 * kk + (lane >> 4)) * 8);
    }
    __syncthreads();

    const int kv_end = p.causal ? min(kv_len, ctx + q0 + BM) : kv_len;
    const int nt = cdiv(kv_end, BN);
    // The e4m3 stages, past the bf16 tile (FP8).
    [[maybe_unused]] u8* smem8 = smem_raw + stage_elems<D>() * 2;

    auto load_kv = [&](int stage, int t) {
        if constexpr (FP8) {
            u8* ks = smem8 + stage * SB8;
            u8* vs = ks + BN * D;
#pragma unroll
            for (int i = 0; i < KV_ITERS8; ++i) {
                const int c = tid + i * THREADS;
                const int row = c / CH8, ch = c % CH8;
                const int j = t * BN + row;
                const bool ok = j < kv_len;
                const int jj = ok ? j : 0;
                const size_t off =
                    cache_row(__ldg(btb + (jj >> p.shift)), p.H_kv, kvh, p.shift, jj) * D + ch * 16;
                cp_async_16_zfill(ks + row * D + swz8<D>(row, ch) * 16, p.K8 + off, ok);
                cp_async_16_zfill(vs + row * D + swz8<D>(row, ch) * 16, p.V8 + off, ok);
            }
        } else {
            bf16* ks = smem + stage * stage_elems<D>();
            bf16* vs = ks + BN * D;
#pragma unroll
            for (int i = 0; i < KV_ITERS; ++i) {
                const int c = tid + i * THREADS;
                const int row = c / CH, ch = c % CH;
                const int j = t * BN + row;
                const bool ok = j < kv_len;
                const int jj = ok ? j : 0;
                const size_t off =
                    cache_row(__ldg(btb + (jj >> p.shift)), p.H_kv, kvh, p.shift, jj) * D + ch * 8;
                cp_async_16_zfill(ks + row * D + swz(row, ch) * 8, p.K + off, ok);
                cp_async_16_zfill(vs + row * D + swz(row, ch) * 8, p.V + off, ok);
            }
        }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nt) load_kv(s, s);
        cp_async_commit();
    }

    float o[DT][4];
#pragma unroll
    for (int dj = 0; dj < DT; ++dj)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[dj][e] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    float sl2 = p.scale_log2;           // times K's scale in FP8
    [[maybe_unused]] float vmul = 1.f;  // V's scale (FP8)
    if constexpr (FP8) {
        sl2 = p.scale_log2 * __ldg(p.k_scale + kvh);
        vmul = __ldg(p.v_scale + kvh);
    }

    for (int t = 0; t < nt; ++t) {
        cp_async_wait<STAGES - 2>();
        // Tile t has landed and every warp is done with tile t - 1: in FP8 also with the bf16
        // tile and with the e4m3 stage the next load overwrites.
        __syncthreads();
        {
            const int n2 = t + STAGES - 1;
            if (n2 < nt) load_kv(n2 % STAGES, n2);
            cp_async_commit();
        }
        const bf16* ks;
        if constexpr (FP8) {
            const u8* k8 = smem8 + (t % STAGES) * SB8;
            tile_e4m3_to_bf16<D, BN, THREADS>(k8, smem, tid);
            tile_e4m3_to_bf16<D, BN, THREADS>(k8 + BN * D, smem + BN * D, tid);
            __syncthreads();
            ks = smem;
        } else {
            ks = smem + (t % STAGES) * stage_elems<D>();
        }
        const bf16* vs = ks + BN * D;

        float s[NT][4];
#pragma unroll
        for (int nj = 0; nj < NT; ++nj)
#pragma unroll
            for (int e = 0; e < 4; ++e) s[nj][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
#pragma unroll
            for (int nj = 0; nj < NT; nj += 2) {
                const int row = nj * 8 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4(r, ks + row * D + swz(row, 2 * kk + (lane >> 4)) * 8);
                const unsigned b0[2] = {r[0], r[2]};
                const unsigned b1[2] = {r[1], r[3]};
                mma_bf16_16816(s[nj], qf[kk], b0);
                mma_bf16_16816(s[nj + 1], qf[kk], b1);
            }
        }

        const int kv0 = t * BN;
        if (kv0 + BN > kv_len || (p.causal && kv0 + BN - 1 > ctx + q0)) {  // block-uniform
#pragma unroll
            for (int nj = 0; nj < NT; ++nj)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int i = q0 + warp * 16 + g + (e >> 1) * 8;
                    const int j = kv0 + nj * 8 + c2 + (e & 1);
                    if (j >= kv_len || (p.causal && j > ctx + i)) s[nj][e] = -INFINITY;
                }
        }

#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float mx = fmaxf(s[0][2 * r], s[0][2 * r + 1]);
#pragma unroll
            for (int nj = 1; nj < NT; ++nj) mx = fmaxf(mx, fmaxf(s[nj][2 * r], s[nj][2 * r + 1]));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 2));
            const float m_new = fmaxf(m[r], mx * sl2);
            const float m_use = m_new == -INFINITY ? 0.f : m_new;
            const float alpha = ex2(m[r] - m_use);
            float rs = 0.f;
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                const float p0 = ex2(fmaf(s[nj][2 * r], sl2, -m_use));
                const float p1 = ex2(fmaf(s[nj][2 * r + 1], sl2, -m_use));
                s[nj][2 * r] = p0;
                s[nj][2 * r + 1] = p1;
                rs += p0 + p1;
            }
            l[r] = fmaf(l[r], alpha, rs);
            m[r] = m_new;
#pragma unroll
            for (int dj = 0; dj < DT; ++dj) {
                o[dj][2 * r] *= alpha;
                o[dj][2 * r + 1] *= alpha;
            }
        }

        unsigned pa[PT][4];
#pragma unroll
        for (int kt = 0; kt < PT; ++kt) {
            pa[kt][0] = pack_bf16x2(s[2 * kt][0], s[2 * kt][1]);
            pa[kt][1] = pack_bf16x2(s[2 * kt][2], s[2 * kt][3]);
            pa[kt][2] = pack_bf16x2(s[2 * kt + 1][0], s[2 * kt + 1][1]);
            pa[kt][3] = pack_bf16x2(s[2 * kt + 1][2], s[2 * kt + 1][3]);
        }
#pragma unroll
        for (int kt = 0; kt < PT; ++kt) {
#pragma unroll
            for (int dj = 0; dj < DT; dj += 2) {
                const int row = kt * 16 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4_trans(r, vs + row * D + swz(row, dj + (lane >> 4)) * 8);
                const unsigned b0[2] = {r[0], r[1]};
                const unsigned b1[2] = {r[2], r[3]};
                mma_bf16_16816(o[dj], pa[kt], b0);
                mma_bf16_16816(o[dj + 1], pa[kt], b1);
            }
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float ls = l[r];
        ls += __shfl_xor_sync(kFullMask, ls, 1);
        ls += __shfl_xor_sync(kFullMask, ls, 2);
        const int row = q0 + warp * 16 + g + 8 * r;
        if (row >= q_len) continue;
        float inv;
        if constexpr (FP8)
            inv = vmul / ls;
        else
            inv = 1.f / ls;
        bf16* out = Og + static_cast<size_t>(row) * q_stride + c2;
#pragma unroll
        for (int dj = 0; dj < DT; ++dj)
            *reinterpret_cast<__nv_bfloat162*>(out + dj * 8) =
                __floats2bfloat162_rn(o[dj][2 * r] * inv, o[dj][2 * r + 1] * inv);
    }
}

template <int D, bool FP8>
void launch(const Params& p, int T, cudaStream_t stream) {
    static bool init = false;
    if (!init) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(varlen_kernel<D, FP8>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              smem_bytes<D, FP8>()));
        init = true;
    }
    // sum_b ceil(q_len_b / BM) <= (T + B (BM - 1)) / BM; blocks past the real count exit.
    const int64_t tiles = (static_cast<int64_t>(T) + static_cast<int64_t>(p.B) * (BM - 1)) / BM;
    const int64_t blocks = tiles * p.H_q;
    SPARK_REQUIRE(blocks < (int64_t{1} << 31), "attention_varlen: too many tiles for one launch");
    varlen_kernel<D, FP8>
        <<<static_cast<unsigned>(blocks), THREADS, smem_bytes<D, FP8>(), stream>>>(p);
    SPARK_CHECK_LAUNCH();
}

}  // namespace vl

int log2_exact(int page) {
    int s = 0;
    while ((1 << s) < page) ++s;
    return (1 << s) == page ? s : -1;
}

void check_paged(const void* Q, const void* K, const void* V, const void* O, const int* bt,
                 const int* lens, int B, int H_q, int H_kv, int D, int page, int max_pages,
                 const char* op) {
    const std::string n(op);
    SPARK_REQUIRE(Q && K && V && O && bt && lens, n + ": null pointer");
    SPARK_REQUIRE(B >= 1 && B <= kMaxBatch, n + ": need 1 <= B <= 512");
    SPARK_REQUIRE(H_q >= 1 && H_kv >= 1 && H_q % H_kv == 0, n + ": need H_q a multiple of H_kv");
    SPARK_REQUIRE(D == 64 || D == 128, n + ": D must be 64 or 128");
    SPARK_REQUIRE(page >= 16 && log2_exact(page) >= 0, n + ": page must be a power of two >= 16");
    SPARK_REQUIRE(max_pages >= 1, n + ": need max_pages >= 1");
    SPARK_REQUIRE(is_aligned16(Q) && is_aligned16(K) && is_aligned16(V) && is_aligned16(O),
                  n + ": needs 16-byte aligned Q, K, V, O");
}

// The entry points of both formats: K and V are the caches (bf16, or e4m3 bytes with fp8 and
// their scales).
void decode_impl(const bf16* Q, const void* K, const void* V, const float* k_scale,
                 const float* v_scale, bool fp8, const int* block_table, const int* seq_lens,
                 bf16* O, int B, int H_q, int H_kv, int D, int page, int max_pages, int variant,
                 cudaStream_t stream) {
    check_paged(Q, K, V, O, block_table, seq_lens, B, H_q, H_kv, D, page, max_pages,
                "paged_decode");
    SPARK_REQUIRE(!fp8 || (k_scale && v_scale), "paged_decode: null scale pointer");
    SPARK_REQUIRE(variant >= 0 && variant < paged_decode_num_variants(),
                  "paged_decode: variant out of range");
    const float scale_log2 = kLog2e / std::sqrt(static_cast<float>(D));
    const int shift = log2_exact(page);
    if (variant == 0) {
        NaiveDecodeParams p{Q,   K,    V,         O,     block_table, seq_lens, B,
                            H_q, H_kv, max_pages, shift, scale_log2,  k_scale,  v_scale};
        const int blocks = cdiv(B * H_q, 4);
        if (D == 64 && fp8)
            paged_decode_naive_kernel<64, u8><<<blocks, 128, 0, stream>>>(p);
        else if (D == 64)
            paged_decode_naive_kernel<64, bf16><<<blocks, 128, 0, stream>>>(p);
        else if (fp8)
            paged_decode_naive_kernel<128, u8><<<blocks, 128, 0, stream>>>(p);
        else
            paged_decode_naive_kernel<128, bf16><<<blocks, 128, 0, stream>>>(p);
        SPARK_CHECK_LAUNCH();
        return;
    }
    SPARK_REQUIRE(H_q / H_kv <= dec::ROWS,
                  "paged_decode: variant 1 needs at most 16 query heads per K/V head");
    dec::Params p{};
    if (fp8) {
        p.K8 = static_cast<const u8*>(K);
        p.V8 = static_cast<const u8*>(V);
        p.k_scale = k_scale;
        p.v_scale = v_scale;
    } else {
        p.K = static_cast<const bf16*>(K);
        p.V = static_cast<const bf16*>(V);
    }
    p.Q = Q;
    p.O = O;
    p.bt = block_table;
    p.lens = seq_lens;
    p.B = B;
    p.H_q = H_q;
    p.H_kv = H_kv;
    p.group = H_q / H_kv;
    p.max_pages = max_pages;
    p.shift = shift;
    p.scale_log2 = scale_log2;
    if (D == 64 && fp8)
        dec::launch<64, true>(p, stream);
    else if (D == 64)
        dec::launch<64, false>(p, stream);
    else if (fp8)
        dec::launch<128, true>(p, stream);
    else
        dec::launch<128, false>(p, stream);
}

void varlen_impl(const bf16* Q, const void* K, const void* V, const float* k_scale,
                 const float* v_scale, bool fp8, const int* cu_seqlens_q, const int* seq_lens,
                 const int* block_table, bf16* O, int B, int T, int H_q, int H_kv, int D, int page,
                 int max_pages, bool causal, int variant, cudaStream_t stream) {
    check_paged(Q, K, V, O, block_table, seq_lens, B, H_q, H_kv, D, page, max_pages,
                "attention_varlen");
    SPARK_REQUIRE(cu_seqlens_q != nullptr, "attention_varlen: null pointer");
    SPARK_REQUIRE(!fp8 || (k_scale && v_scale), "attention_varlen: null scale pointer");
    SPARK_REQUIRE(T >= 1, "attention_varlen: need T >= 1");
    SPARK_REQUIRE(variant >= 0 && variant < attention_varlen_num_variants(),
                  "attention_varlen: variant out of range");
    const float scale_log2 = kLog2e / std::sqrt(static_cast<float>(D));
    const int shift = log2_exact(page);
    if (variant == 0) {
        NaiveVarlenParams p{
            Q,   K,    V,         O,     cu_seqlens_q,   seq_lens,   block_table, B,      T,
            H_q, H_kv, max_pages, shift, causal ? 1 : 0, scale_log2, k_scale,     v_scale};
        const int64_t blocks = cdiv64(static_cast<int64_t>(T) * H_q, 4);
        SPARK_REQUIRE(blocks < (int64_t{1} << 31), "attention_varlen: too many rows");
        const unsigned nb = static_cast<unsigned>(blocks);
        if (D == 64 && fp8)
            varlen_naive_kernel<64, u8><<<nb, 128, 0, stream>>>(p);
        else if (D == 64)
            varlen_naive_kernel<64, bf16><<<nb, 128, 0, stream>>>(p);
        else if (fp8)
            varlen_naive_kernel<128, u8><<<nb, 128, 0, stream>>>(p);
        else
            varlen_naive_kernel<128, bf16><<<nb, 128, 0, stream>>>(p);
        SPARK_CHECK_LAUNCH();
        return;
    }
    vl::Params p{};
    if (fp8) {
        p.K8 = static_cast<const u8*>(K);
        p.V8 = static_cast<const u8*>(V);
        p.k_scale = k_scale;
        p.v_scale = v_scale;
    } else {
        p.K = static_cast<const bf16*>(K);
        p.V = static_cast<const bf16*>(V);
    }
    p.Q = Q;
    p.O = O;
    p.cu_q = cu_seqlens_q;
    p.lens = seq_lens;
    p.bt = block_table;
    p.B = B;
    p.H_q = H_q;
    p.H_kv = H_kv;
    p.group = H_q / H_kv;
    p.max_pages = max_pages;
    p.shift = shift;
    p.causal = causal ? 1 : 0;
    p.scale_log2 = scale_log2;
    if (D == 64 && fp8)
        vl::launch<64, true>(p, T, stream);
    else if (D == 64)
        vl::launch<64, false>(p, T, stream);
    else if (fp8)
        vl::launch<128, true>(p, T, stream);
    else
        vl::launch<128, false>(p, T, stream);
}

}  // namespace

void paged_decode_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* k_cache,
                       const __nv_bfloat16* v_cache, const int* block_table, const int* seq_lens,
                       __nv_bfloat16* O, int B, int H_q, int H_kv, int D, int page, int max_pages,
                       int variant, cudaStream_t stream) {
    decode_impl(Q, k_cache, v_cache, nullptr, nullptr, false, block_table, seq_lens, O, B, H_q,
                H_kv, D, page, max_pages, variant, stream);
}

void paged_decode_fp8(const __nv_bfloat16* Q, const __nv_fp8_e4m3* k_cache,
                      const __nv_fp8_e4m3* v_cache, const float* k_scale, const float* v_scale,
                      const int* block_table, const int* seq_lens, __nv_bfloat16* O, int B, int H_q,
                      int H_kv, int D, int page, int max_pages, int variant, cudaStream_t stream) {
    decode_impl(Q, k_cache, v_cache, k_scale, v_scale, true, block_table, seq_lens, O, B, H_q, H_kv,
                D, page, max_pages, variant, stream);
}

int paged_decode_num_variants() {
    return 2;
}

void attention_varlen_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* k_cache,
                           const __nv_bfloat16* v_cache, const int* cu_seqlens_q,
                           const int* seq_lens, const int* block_table, __nv_bfloat16* O, int B,
                           int T, int H_q, int H_kv, int D, int page, int max_pages, bool causal,
                           int variant, cudaStream_t stream) {
    varlen_impl(Q, k_cache, v_cache, nullptr, nullptr, false, cu_seqlens_q, seq_lens, block_table,
                O, B, T, H_q, H_kv, D, page, max_pages, causal, variant, stream);
}

void attention_varlen_fp8(const __nv_bfloat16* Q, const __nv_fp8_e4m3* k_cache,
                          const __nv_fp8_e4m3* v_cache, const float* k_scale, const float* v_scale,
                          const int* cu_seqlens_q, const int* seq_lens, const int* block_table,
                          __nv_bfloat16* O, int B, int T, int H_q, int H_kv, int D, int page,
                          int max_pages, bool causal, int variant, cudaStream_t stream) {
    varlen_impl(Q, k_cache, v_cache, k_scale, v_scale, true, cu_seqlens_q, seq_lens, block_table, O,
                B, T, H_q, H_kv, D, page, max_pages, causal, variant, stream);
}

int attention_varlen_num_variants() {
    return 2;
}

}  // namespace spark
