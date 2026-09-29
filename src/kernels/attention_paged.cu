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

// ---- the naive rungs (variant 0 of both ops) ---------------------------------------------

// One warp computes one query row against keys [0, nkeys) of sequence b, kv head kvh. Lane l
// owns columns l*VEC .. l*VEC+VEC-1; the score of a key is a shuffle reduction.
template <int D>
__device__ void naive_row(const bf16* q, const bf16* K, const bf16* V, const int* bt, int H_kv,
                          int kvh, int shift, int nkeys, float scale_log2, bf16* out) {
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
        const bf16* kr = K + row + lane * VEC;
        float s = 0.f;
#pragma unroll
        for (int e = 0; e < VEC; ++e) s = fmaf(qf[e], __bfloat162float(kr[e]), s);
        s = warp_reduce_sum(s) * scale_log2;
        const float m_new = fmaxf(m, s);
        const float alpha = ex2(m - m_new);
        const float pj = ex2(s - m_new);
        l = fmaf(l, alpha, pj);
        const bf16* vr = V + row + lane * VEC;
#pragma unroll
        for (int e = 0; e < VEC; ++e) acc[e] = fmaf(acc[e], alpha, pj * __bfloat162float(vr[e]));
        m = m_new;
    }
    const float inv = nkeys > 0 ? 1.f / l : 0.f;
#pragma unroll
    for (int e = 0; e < VEC; ++e) out[lane * VEC + e] = __float2bfloat16(acc[e] * inv);
}

struct NaiveDecodeParams {
    const bf16* Q;
    const bf16* K;
    const bf16* V;
    bf16* O;
    const int* bt;
    const int* lens;
    int B, H_q, H_kv, max_pages, shift;
    float scale_log2;
};

template <int D>
__global__ void __launch_bounds__(128) paged_decode_naive_kernel(NaiveDecodeParams p) {
    const int row = blockIdx.x * 4 + (threadIdx.x >> 5);  // (b, h)
    if (row >= p.B * p.H_q) return;
    const int b = row / p.H_q, h = row - b * p.H_q;
    const int kvh = h / (p.H_q / p.H_kv);
    naive_row<D>(p.Q + static_cast<size_t>(row) * D, p.K, p.V,
                 p.bt + static_cast<size_t>(b) * p.max_pages, p.H_kv, kvh, p.shift,
                 max(0, p.lens[b]), p.scale_log2, p.O + static_cast<size_t>(row) * D);
}

struct NaiveVarlenParams {
    const bf16* Q;
    const bf16* K;
    const bf16* V;
    bf16* O;
    const int* cu_q;
    const int* lens;
    const int* bt;
    int B, T, H_q, H_kv, max_pages, shift, causal;
    float scale_log2;
};

template <int D>
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
    naive_row<D>(p.Q + static_cast<size_t>(row) * D, p.K, p.V,
                 p.bt + static_cast<size_t>(b) * p.max_pages, p.H_kv, h / (p.H_q / p.H_kv), p.shift,
                 nkeys, p.scale_log2, p.O + static_cast<size_t>(row) * D);
}

// ---- paged flash-decoding (decode variant 1) ---------------------------------------------

namespace dec {

constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;
constexpr int ROWS = 16;  // one m16 tile of query rows: the group's heads
constexpr int SLAB = 16;  // keys per pipeline stage, and the smallest page
constexpr int STAGES = 3;

// A stage is a K slab then a V slab, 8 KB at D = 128.
template <int D>
constexpr int stage_elems() {
    return 2 * SLAB * D;
}
template <int D>
constexpr int smem_bytes() {
    return WARPS * STAGES * stage_elems<D>() * 2;  // 96 KB at D = 128
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
    float* ws;  // two pieces per warp of the grid
    int* meta;  // the slab prefix s_pre[0..B], published by block 0 for the combine kernel
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

template <int D>
__global__ void __launch_bounds__(THREADS, 1) paged_decode_kernel(Params p) {
    constexpr int CH = D / 8;                 // 16-byte chunks per row
    constexpr int KT = D / 16;                // k16 steps of Q K^T
    constexpr int DT = D / 8;                 // n8 tiles of O
    constexpr int NT = SLAB / 8;              // n8 tiles of S: 2
    constexpr int KV_ITERS = SLAB * CH / 32;  // chunks per lane per operand per slab: 8, 4
    constexpr int PF = piece_floats<D>();
    static_assert(SLAB == 16, "P V below is one k16 step");

    extern __shared__ __align__(128) unsigned char smem_raw[];
    __shared__ int s_pre[kMaxBatch + 1];
    __shared__ int s_scan[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;
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

    bf16* wsm = reinterpret_cast<bf16*>(smem_raw) + warp * STAGES * stage_elems<D>();

    // Loads run STAGES - 1 slabs ahead of the compute. `lpage` is the page of the next slab
    // to load, fetched one load early so the address is ready when the load issues.
    Cursor L = locate(s_pre, B, H_kv, a);
    int nl = a;
    auto page_of = [&](const Cursor& c) {
        return __ldg(p.bt + static_cast<size_t>(c.b) * p.max_pages + ((c.j * SLAB) >> p.shift));
    };
    int lpage = page_of(L);
    auto load = [&](int stage) {
        bf16* ks = wsm + stage * stage_elems<D>();
        bf16* vs = ks + SLAB * D;
        const int len = __ldg(p.lens + L.b);
        const int key0 = L.j * SLAB;
        const size_t base = cache_row(lpage, H_kv, L.kvh, p.shift, key0);
#pragma unroll
        for (int i = 0; i < KV_ITERS; ++i) {
            const int c = lane + i * 32;
            const int row = c / CH, ch = c % CH;
            const bool ok = key0 + row < len;
            const size_t off = (base + (ok ? row : 0)) * D + ch * 8;
            cp_async_16_zfill(ks + row * D + swz(row, ch) * 8, p.K + off, ok);
            cp_async_16_zfill(vs + row * D + swz(row, ch) * 8, p.V + off, ok);
        }
        advance(L, s_pre, B, H_kv);
        if (++nl < e) lpage = page_of(L);
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (nl < e) load(s);
        cp_async_commit();
    }

    Cursor C = locate(s_pre, B, H_kv, a);
    bool first_seg = true;
    int piece_j0 = C.j;
    unsigned qf[KT][4];
    float o[DT][4];
    float m[2], l[2];

    for (int x = a, it = 0; x < e; ++x, ++it) {
        cp_async_wait<STAGES - 2>();  // slab x has landed for this lane
        __syncwarp();                 // ... for every lane; and stage (it-1)%STAGES is free
        if (nl < e) load((it + STAGES - 1) % STAGES);
        cp_async_commit();
        const bf16* ks = wsm + (it % STAGES) * stage_elems<D>();
        const bf16* vs = ks + SLAB * D;

        if (x == a || C.j == 0) {  // a new segment: its Q fragments, a fresh softmax state
            // The A fragments straight from global memory (rows g and g + 8, k pairs c2 and
            // 8 + c2 of each k16 step), rows past the group zero. The rows were written by the
            // RoPE kernel just before, so this is an L2 round trip, taken while the next two
            // slabs are already in flight.
            const unsigned* qg = reinterpret_cast<const unsigned*>(
                p.Q +
                (static_cast<size_t>(C.b) * p.H_q + static_cast<size_t>(C.kvh) * p.group) * D);
            const bool r0 = g < p.group, r1 = g + 8 < p.group;
#pragma unroll
            for (int kk = 0; kk < KT; ++kk) {
                const int k0 = (kk * 16 + c2) / 2;
                qf[kk][0] = r0 ? __ldg(qg + (g * D) / 2 + k0) : 0u;
                qf[kk][1] = r1 ? __ldg(qg + ((g + 8) * D) / 2 + k0) : 0u;
                qf[kk][2] = r0 ? __ldg(qg + (g * D) / 2 + k0 + 4) : 0u;
                qf[kk][3] = r1 ? __ldg(qg + ((g + 8) * D) / 2 + k0 + 4) : 0u;
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
            const float m_new = fmaxf(m[r], mx * p.scale_log2);  // finite: slab has a key
            const float alpha = ex2(m[r] - m_new);
            float rs = 0.f;
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                const float p0 = ex2(fmaf(sc[nj][2 * r], p.scale_log2, -m_new));
                const float p1 = ex2(fmaf(sc[nj][2 * r + 1], p.scale_log2, -m_new));
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

        // End of a segment or of the range: this warp's piece of segment (C.b, C.kvh) is done.
        // A whole segment is normalized and written; a piece goes to the warp's slot 0 (the
        // first segment of its range) or 1 (the last) for the combine kernel.
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
                if (whole) {
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

template <int D>
int resident_blocks() {
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(
            paged_decode_kernel<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<D>()));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, paged_decode_kernel<D>, THREADS, smem_bytes<D>()));
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

template <int D>
void launch(Params p, cudaStream_t stream) {
    static const int min_slabs = env_int("SPARK_PAGED_MIN_SLABS", 4);
    static const int grid_cap = env_int("SPARK_PAGED_GRID", 1 << 30);
    p.grid = std::min(resident_blocks<D>(), grid_cap);
    p.min_slabs = min_slabs;
    p.ws = workspace(static_cast<size_t>(p.grid) * WARPS * 2 * piece_floats<D>());
    p.meta = meta_buffer();
    paged_decode_kernel<D><<<p.grid, THREADS, smem_bytes<D>(), stream>>>(p);
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
template <int D>
constexpr int smem_bytes() {
    return STAGES * stage_elems<D>() * 2;  // 96 KB at D = 128
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
};

template <int D>
__global__ void __launch_bounds__(THREADS, 1) varlen_kernel(Params p) {
    constexpr int CH = D / 8;
    constexpr int KT = D / 16;
    constexpr int DT = D / 8;
    constexpr int NT = BN / 8;
    constexpr int PT = BN / 16;
    constexpr int Q_ITERS = BM * CH / THREADS;
    constexpr int KV_ITERS = BN * CH / THREADS;
    static_assert(Q_ITERS * THREADS == BM * CH && KV_ITERS * THREADS == BN * CH, "");
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

    auto load_kv = [&](int stage, int t) {
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

    for (int t = 0; t < nt; ++t) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            const int n2 = t + STAGES - 1;
            if (n2 < nt) load_kv(n2 % STAGES, n2);
            cp_async_commit();
        }
        const bf16* ks = smem + (t % STAGES) * stage_elems<D>();
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
            const float m_new = fmaxf(m[r], mx * p.scale_log2);
            const float m_use = m_new == -INFINITY ? 0.f : m_new;
            const float alpha = ex2(m[r] - m_use);
            float rs = 0.f;
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                const float p0 = ex2(fmaf(s[nj][2 * r], p.scale_log2, -m_use));
                const float p1 = ex2(fmaf(s[nj][2 * r + 1], p.scale_log2, -m_use));
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
        const float inv = 1.f / ls;
        bf16* out = Og + static_cast<size_t>(row) * q_stride + c2;
#pragma unroll
        for (int dj = 0; dj < DT; ++dj)
            *reinterpret_cast<__nv_bfloat162*>(out + dj * 8) =
                __floats2bfloat162_rn(o[dj][2 * r] * inv, o[dj][2 * r + 1] * inv);
    }
}

template <int D>
void launch(const Params& p, int T, cudaStream_t stream) {
    static bool init = false;
    if (!init) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(
            varlen_kernel<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<D>()));
        init = true;
    }
    // sum_b ceil(q_len_b / BM) <= (T + B (BM - 1)) / BM; blocks past the real count exit.
    const int64_t tiles = (static_cast<int64_t>(T) + static_cast<int64_t>(p.B) * (BM - 1)) / BM;
    const int64_t blocks = tiles * p.H_q;
    SPARK_REQUIRE(blocks < (int64_t{1} << 31), "attention_varlen: too many tiles for one launch");
    varlen_kernel<D><<<static_cast<unsigned>(blocks), THREADS, smem_bytes<D>(), stream>>>(p);
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

}  // namespace

void paged_decode_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* k_cache,
                       const __nv_bfloat16* v_cache, const int* block_table, const int* seq_lens,
                       __nv_bfloat16* O, int B, int H_q, int H_kv, int D, int page, int max_pages,
                       int variant, cudaStream_t stream) {
    check_paged(Q, k_cache, v_cache, O, block_table, seq_lens, B, H_q, H_kv, D, page, max_pages,
                "paged_decode");
    SPARK_REQUIRE(variant >= 0 && variant < paged_decode_num_variants(),
                  "paged_decode: variant out of range");
    const float scale_log2 = kLog2e / std::sqrt(static_cast<float>(D));
    const int shift = log2_exact(page);
    if (variant == 0) {
        NaiveDecodeParams p{Q, k_cache, v_cache, O,         block_table, seq_lens,
                            B, H_q,     H_kv,    max_pages, shift,       scale_log2};
        const int blocks = cdiv(B * H_q, 4);
        if (D == 64)
            paged_decode_naive_kernel<64><<<blocks, 128, 0, stream>>>(p);
        else
            paged_decode_naive_kernel<128><<<blocks, 128, 0, stream>>>(p);
        SPARK_CHECK_LAUNCH();
        return;
    }
    SPARK_REQUIRE(H_q / H_kv <= dec::ROWS,
                  "paged_decode: variant 1 needs at most 16 query heads per K/V head");
    dec::Params p;
    p.Q = Q;
    p.K = k_cache;
    p.V = v_cache;
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
    if (D == 64)
        dec::launch<64>(p, stream);
    else
        dec::launch<128>(p, stream);
}

int paged_decode_num_variants() {
    return 2;
}

void attention_varlen_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* k_cache,
                           const __nv_bfloat16* v_cache, const int* cu_seqlens_q,
                           const int* seq_lens, const int* block_table, __nv_bfloat16* O, int B,
                           int T, int H_q, int H_kv, int D, int page, int max_pages, bool causal,
                           int variant, cudaStream_t stream) {
    check_paged(Q, k_cache, v_cache, O, block_table, seq_lens, B, H_q, H_kv, D, page, max_pages,
                "attention_varlen");
    SPARK_REQUIRE(cu_seqlens_q != nullptr, "attention_varlen: null pointer");
    SPARK_REQUIRE(T >= 1, "attention_varlen: need T >= 1");
    SPARK_REQUIRE(variant >= 0 && variant < attention_varlen_num_variants(),
                  "attention_varlen: variant out of range");
    const float scale_log2 = kLog2e / std::sqrt(static_cast<float>(D));
    const int shift = log2_exact(page);
    if (variant == 0) {
        NaiveVarlenParams p{
            Q, k_cache, v_cache, O,         cu_seqlens_q, seq_lens,       block_table, B,
            T, H_q,     H_kv,    max_pages, shift,        causal ? 1 : 0, scale_log2};
        const int64_t blocks = cdiv64(static_cast<int64_t>(T) * H_q, 4);
        SPARK_REQUIRE(blocks < (int64_t{1} << 31), "attention_varlen: too many rows");
        if (D == 64)
            varlen_naive_kernel<64><<<static_cast<unsigned>(blocks), 128, 0, stream>>>(p);
        else
            varlen_naive_kernel<128><<<static_cast<unsigned>(blocks), 128, 0, stream>>>(p);
        SPARK_CHECK_LAUNCH();
        return;
    }
    vl::Params p;
    p.Q = Q;
    p.K = k_cache;
    p.V = v_cache;
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
    if (D == 64)
        vl::launch<64>(p, T, stream);
    else
        vl::launch<128>(p, T, stream);
}

int attention_varlen_num_variants() {
    return 2;
}

}  // namespace spark
