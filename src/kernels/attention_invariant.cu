// Batch-invariant attention over the paged K/V cache: paged_decode variant 2 and
// attention_varlen variant 2 (docs/design/batch_invariant.md). Reached from
// attention_paged.cu's dispatch; bf16 caches only.
//
// Both ops compute every query row with one arithmetic, fixed by the row's own sequence and
// nothing else:
//
//   * the keys 0 .. n-1 the row sees are cut into chunks of CHUNK = 256 keys at offsets 0,
//     256, 512, ... (the last one shorter);
//   * a chunk is an online softmax from an empty state over its 16-key steps in key order:
//     S = q k^T by mma.sync.m16n8k16 over D in k16 order, the step's max, P = 2^(s - m) on
//     the SFU, P rounded to bf16, O += P V by one k16 mma per n8 tile of O (step16 below);
//   * the chunks are folded left to right into a running (M, L, O) by the merge rule
//     (fold below), and O / L is rounded to bf16 once.
//
// A step of keys the row cannot see (past its length, or past the causal limit) changes
// nothing bit for bit: its max is -inf, the rescale is exactly 1, P is 0. So the varlen
// kernel can run its 64-key tiles past a row's last key, and a chunk can end early, without
// touching the result. What differs between the two ops is only who computes a chunk:
//
//   decode: the chunks of every (b, kv head) of the batch are one flat list, cut into equal
//     contiguous ranges, one per warp of a one-block-per-SM grid (paged decode variant 1's
//     flat split, at chunk granularity: a chunk is never split between warps). A sequence
//     of one chunk is written by its warp; otherwise every chunk's (m, l, O) goes to a
//     workspace slot and a combine kernel folds them in chunk order. The split depends on
//     the batch; the arithmetic does not.
//   varlen: variant 1's 128-row query tile and 64-key K/V tiles; a tile runs as four 16-key
//     steps and every fourth tile ends a chunk, folded in registers.
//
// The rounding-sensitive operations are spelled out (__fmaf_rn, __fmul_rn, __fdiv_rn) so
// the compiler cannot contract them differently in the three places they appear.
#include <algorithm>
#include <cmath>
#include <vector>

#include "attention_internal.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

using attn::bf16;
using attn::ex2;
using attn::kLog2e;
using attn::pack_bf16x2;
using attn::swz;

constexpr int SLAB = 16;                   // keys per step, and the smallest page
constexpr int CHUNK = 256;                 // keys per chunk
constexpr int CHUNK_SLABS = CHUNK / SLAB;  // 16
constexpr int kMaxBatchInv = 512;          // sequences per launch (attention_paged.cu's)
static_assert(CHUNK % 64 == 0, "a chunk is whole 64-key tiles of the varlen kernel");

// ---- helpers from attention_paged.cu (the same prefix scan and cache addressing) ----------

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

__device__ __forceinline__ size_t cache_row(int page_id, int H_kv, int kvh, int shift, int j) {
    return ((static_cast<size_t>(page_id) * H_kv + kvh) << shift) + (j & ((1 << shift) - 1));
}

// ---- the arithmetic both ops share ---------------------------------------------------------

// S = Q K^T for one 16-key step: ks is the step's 16 key rows in shared memory (rows of D
// elements, chunks at swz of the row), qf the m16 tile's Q fragments.
template <int D>
__device__ __forceinline__ void qk16(float (&sc)[2][4], const unsigned (&qf)[D / 16][4],
                                     const bf16* ks, int lane) {
#pragma unroll
    for (int nj = 0; nj < 2; ++nj)
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[nj][e] = 0.f;
#pragma unroll
    for (int kk = 0; kk < D / 16; ++kk) {
        const int row = lane & 15;
        unsigned r[4];
        ldmatrix_x4(r, ks + row * D + swz(row, 2 * kk + (lane >> 4)) * 8);
        const unsigned b0[2] = {r[0], r[2]};
        const unsigned b1[2] = {r[1], r[3]};
        mma_bf16_16816(sc[0], qf[kk], b0);
        mma_bf16_16816(sc[1], qf[kk], b1);
    }
}

// One 16-key step of a chunk's online softmax for rows g and g + 8 of the warp's m16 tile.
// sc holds the step's scores (masked keys -inf) and is packed into pa as P in bf16. l stays a
// per-lane partial (this lane's columns) until the chunk ends. The rescale is exactly 1 when
// the max does not move, so a step of masked keys is a no-op.
template <int DT>
__device__ __forceinline__ void step16(float (&sc)[2][4], float (&m)[2], float (&l)[2],
                                       float (&o)[DT][4], float sl2, unsigned (&pa)[4]) {
    float alphas[2];
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float mx =
            fmaxf(fmaxf(sc[0][2 * r], sc[0][2 * r + 1]), fmaxf(sc[1][2 * r], sc[1][2 * r + 1]));
        mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 2));
        const float m_new = fmaxf(m[r], __fmul_rn(mx, sl2));
        const float m_use = m_new == -INFINITY ? 0.f : m_new;
        const float alpha = m_new == m[r] ? 1.f : ex2(__fsub_rn(m[r], m_use));
        float rs = 0.f;
#pragma unroll
        for (int nj = 0; nj < 2; ++nj) {
            const float p0 = ex2(__fmaf_rn(sc[nj][2 * r], sl2, -m_use));
            const float p1 = ex2(__fmaf_rn(sc[nj][2 * r + 1], sl2, -m_use));
            sc[nj][2 * r] = p0;
            sc[nj][2 * r + 1] = p1;
            rs = __fadd_rn(rs, __fadd_rn(p0, p1));
        }
        l[r] = __fmaf_rn(l[r], alpha, rs);
        m[r] = m_new;
        alphas[r] = alpha;
    }
    // Once the running max settles most steps leave it alone; a rescale by exactly 1 is the
    // identity, so the warp skips it then (one vote, the same bits either way).
    if (!__all_sync(kFullMask, alphas[0] == 1.f && alphas[1] == 1.f)) {
#pragma unroll
        for (int r = 0; r < 2; ++r)
#pragma unroll
            for (int dj = 0; dj < DT; ++dj) {
                o[dj][2 * r] = __fmul_rn(o[dj][2 * r], alphas[r]);
                o[dj][2 * r + 1] = __fmul_rn(o[dj][2 * r + 1], alphas[r]);
            }
    }
    pa[0] = pack_bf16x2(sc[0][0], sc[0][1]);
    pa[1] = pack_bf16x2(sc[0][2], sc[0][3]);
    pa[2] = pack_bf16x2(sc[1][0], sc[1][1]);
    pa[3] = pack_bf16x2(sc[1][2], sc[1][3]);
}

// O += P V for one 16-key step: vs is the step's 16 value rows (the same layout as ks).
template <int D>
__device__ __forceinline__ void pv16(float (&o)[D / 8][4], const unsigned (&pa)[4], const bf16* vs,
                                     int lane) {
#pragma unroll
    for (int dj = 0; dj < D / 8; dj += 2) {
        const int row = lane & 15;
        unsigned r[4];
        ldmatrix_x4_trans(r, vs + row * D + swz(row, dj + (lane >> 4)) * 8);
        const unsigned b0[2] = {r[0], r[1]};
        const unsigned b1[2] = {r[2], r[3]};
        mma_bf16_16816(o[dj], pa, b0);
        mma_bf16_16816(o[dj + 1], pa, b1);
    }
}

// A row's l over the lanes of its quad; every lane of the quad gets the same bits.
__device__ __forceinline__ float quad_sum(float x) {
    x = __fadd_rn(x, __shfl_xor_sync(kFullMask, x, 1));
    return __fadd_rn(x, __shfl_xor_sync(kFullMask, x, 2));
}

// The merge of a finished chunk (max m) into the running state (max M): the scales of the
// two sides. A side that holds the new max gets exactly 1, an empty side (-inf) exactly 0,
// so folding an empty chunk, or a chunk into the empty state, is exact.
struct FoldScales {
    float mn, sa, sb;
};
__device__ __forceinline__ FoldScales fold_scales(float M, float m) {
    FoldScales f;
    f.mn = fmaxf(M, m);
    f.sa = M == f.mn ? 1.f : (M == -INFINITY ? 0.f : ex2(__fsub_rn(M, f.mn)));
    f.sb = m == f.mn ? 1.f : (m == -INFINITY ? 0.f : ex2(__fsub_rn(m, f.mn)));
    return f;
}
__device__ __forceinline__ float fold1(float X, float x, const FoldScales& f) {
    return __fmaf_rn(X, f.sa, __fmul_rn(x, f.sb));
}
// The output of a finished row: O / L, rounded to bf16 by the caller.
__device__ __forceinline__ float finish(float O, float L) {
    return __fmul_rn(O, __fdiv_rn(1.f, L));
}

// ---- decode (paged_decode variant 2) -------------------------------------------------------

namespace idec {

constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;
constexpr int STAGES = 3;

template <int D>
constexpr int stage_elems() {
    return 2 * SLAB * D;  // a K slab then a V slab
}
template <int D>
constexpr int smem_bytes() {
    return WARPS * STAGES * stage_elems<D>() * 2;  // 96 KB at D = 128
}

struct Params {
    const bf16* Q;
    const bf16* K;
    const bf16* V;
    bf16* O;
    const int* bt;
    const int* lens;
    int B, H_q, H_kv, group, max_pages, shift;
    int grid;
    float scale_log2;
    float* ws;  // one piece per chunk: group rows of D fp32 O, then group m, then group l
    int* meta;  // the chunk prefix over the sequences, published by block 0
};

__host__ __device__ __forceinline__ int piece_floats(int D, int group) {
    return group * (D + 2);
}
__device__ __forceinline__ int chunks_of(int len) {
    return (max(0, len) + CHUNK - 1) / CHUNK;
}

// The walk of a warp over its chunks, slab by slab: sequence b, kv head kvh, chunk c of the
// segment's nc, slab j of its ns, and jend the end slab of chunk c.
struct Cursor {
    int b, kvh, c, nc, j, ns, jend;
};

__device__ __forceinline__ void set_seq(Cursor& u, const int* s_pre, const int* lens, int b) {
    u.b = b;
    u.nc = s_pre[b + 1] - s_pre[b];
    u.ns = (max(0, __ldg(lens + b)) + SLAB - 1) / SLAB;
}

// Chunk x of the flat list: segment (b, kvh) holds nc_b chunks from s_pre[b] H_kv + kvh nc_b.
__device__ __forceinline__ Cursor locate(const int* s_pre, const int* lens, int B, int H_kv,
                                         int x) {
    int lo = 0, hi = B - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (s_pre[mid] * H_kv <= x)
            lo = mid;
        else
            hi = mid - 1;
    }
    Cursor u;
    set_seq(u, s_pre, lens, lo);
    const int rem = x - s_pre[lo] * H_kv;
    u.kvh = rem / u.nc;
    u.c = rem - u.kvh * u.nc;
    u.j = u.c * CHUNK_SLABS;
    u.jend = min(u.j + CHUNK_SLABS, u.ns);
    return u;
}

// One slab on; true when that left a chunk behind.
__device__ __forceinline__ bool advance(Cursor& u, const int* s_pre, const int* lens, int B,
                                        int H_kv) {
    if (++u.j < u.jend) return false;
    if (++u.c < u.nc) {
        u.jend = min(u.j + CHUNK_SLABS, u.ns);
        return true;
    }
    u.c = 0;
    u.j = 0;
    if (++u.kvh == H_kv) {
        u.kvh = 0;
        int b = u.b;
        do {
            ++b;
        } while (b < B && s_pre[b + 1] == s_pre[b]);
        if (b < B) {
            set_seq(u, s_pre, lens, b);
        } else {
            u.b = B;
            u.nc = u.ns = 0;
        }
    }
    u.jend = min(CHUNK_SLABS, u.ns);
    return true;
}

__device__ __forceinline__ int range_start(int w, int N, int weff) {
    return static_cast<int>(static_cast<int64_t>(w) * N / weff);
}

template <int D>
__global__ void __launch_bounds__(THREADS, 1) decode_kernel(Params p) {
    constexpr int CH = D / 8;                 // 16-byte chunks per row
    constexpr int KT = D / 16;                // k16 steps of Q K^T
    constexpr int DT = D / 8;                 // n8 tiles of O
    constexpr int KV_ITERS = SLAB * CH / 32;  // chunks per lane per operand per slab
    constexpr int SE = stage_elems<D>();

    extern __shared__ __align__(128) unsigned char smem_raw[];
    __shared__ int s_pre[kMaxBatchInv + 1];
    __shared__ int s_scan[32];
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    const int B = p.B, H_kv = p.H_kv, group = p.group;
    const int PF = piece_floats(D, group);

    block_prefix(
        B, [&](int b) { return chunks_of(p.lens[b]); }, s_pre, s_scan);
    if (blockIdx.x == 0)
        for (int b = tid; b <= B; b += THREADS) p.meta[b] = s_pre[b];
    const int N = s_pre[B] * H_kv;  // chunks in the batch
    if (N == 0) return;
    const int weff = min(p.grid * WARPS, N);
    const int w = warp * gridDim.x + blockIdx.x;  // warp-major: a short batch spans every SM
    if (w >= weff) return;                        // no block-wide barrier from here on
    const int a = range_start(w, N, weff), nchunks = range_start(w + 1, N, weff) - a;

    bf16* wsm = reinterpret_cast<bf16*>(smem_raw) + warp * STAGES * SE;

    // Loads run STAGES - 1 slabs ahead of the compute, across chunk and segment boundaries,
    // and stop after the warp's last chunk.
    Cursor Lc = locate(s_pre, p.lens, B, H_kv, a);
    int lchunks = 0;
    auto load = [&](int stage) {
        const int len = __ldg(p.lens + Lc.b);
        const int key0 = Lc.j * SLAB;
        const int page = __ldg(p.bt + static_cast<size_t>(Lc.b) * p.max_pages + (key0 >> p.shift));
        const size_t base = cache_row(page, H_kv, Lc.kvh, p.shift, key0);
        bf16* ks = wsm + stage * SE;
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
        if (advance(Lc, s_pre, p.lens, B, H_kv)) ++lchunks;
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (lchunks < nchunks) load(s);
        cp_async_commit();
    }

    Cursor C = locate(s_pre, p.lens, B, H_kv, a);
    unsigned qf[KT][4];
    float o[DT][4];
    float m[2], l[2];
    const float sl2 = p.scale_log2;
    bool first = true;

    for (int done = 0, it = 0; done < nchunks; ++it) {
        cp_async_wait<STAGES - 2>();  // slab `it` has landed for this lane
        __syncwarp();                 // ... for every lane; and stage (it-1)%STAGES is free
        if (lchunks < nchunks) load((it + STAGES - 1) % STAGES);
        cp_async_commit();
        const bf16* ks = wsm + (it % STAGES) * SE;
        const bf16* vs = ks + SLAB * D;

        if (first || C.j == 0) {  // a new segment: its Q fragments (rows past the group zero)
            const unsigned* qg = reinterpret_cast<const unsigned*>(
                p.Q + (static_cast<size_t>(C.b) * p.H_q + static_cast<size_t>(C.kvh) * group) * D);
            const bool r0 = g < group, r1 = g + 8 < group;
#pragma unroll
            for (int kk = 0; kk < KT; ++kk) {
                const int k0 = (kk * 16 + c2) / 2;
                qf[kk][0] = r0 ? __ldg(qg + (g * D) / 2 + k0) : 0u;
                qf[kk][1] = r1 ? __ldg(qg + ((g + 8) * D) / 2 + k0) : 0u;
                qf[kk][2] = r0 ? __ldg(qg + (g * D) / 2 + k0 + 4) : 0u;
                qf[kk][3] = r1 ? __ldg(qg + ((g + 8) * D) / 2 + k0 + 4) : 0u;
            }
        }
        if (first || C.j == C.c * CHUNK_SLABS) {  // a new chunk: the empty state
#pragma unroll
            for (int dj = 0; dj < DT; ++dj)
#pragma unroll
                for (int e = 0; e < 4; ++e) o[dj][e] = 0.f;
            m[0] = m[1] = -INFINITY;
            l[0] = l[1] = 0.f;
        }
        first = false;

        float sc[2][4];
        qk16<D>(sc, qf, ks, lane);
        const int len = __ldg(p.lens + C.b);
        const int key0 = C.j * SLAB;
        if (key0 + SLAB > len) {  // the segment's last, partial slab (warp-uniform)
#pragma unroll
            for (int nj = 0; nj < 2; ++nj)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    if (key0 + nj * 8 + c2 + (e & 1) >= len) sc[nj][e] = -INFINITY;
        }
        unsigned pa[4];
        step16<DT>(sc, m, l, o, sl2, pa);
        pv16<D>(o, pa, vs, lane);

        if (C.j == C.jend - 1) {  // the chunk ends: written whole, or as a piece
            const int x = s_pre[C.b] * H_kv + C.kvh * C.nc + C.c;
            float* part = p.ws + static_cast<size_t>(x) * PF;
            bf16* og =
                p.O + (static_cast<size_t>(C.b) * p.H_q + static_cast<size_t>(C.kvh) * group) * D;
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const float ls = quad_sum(l[r]);
                const int row = g + 8 * r;
                if (row >= group) continue;
                if (C.nc == 1) {
                    // One chunk: the fold of it into the empty state is the chunk itself.
#pragma unroll
                    for (int dj = 0; dj < DT; ++dj)
                        *reinterpret_cast<__nv_bfloat162*>(og + row * D + dj * 8 + c2) =
                            __floats2bfloat162_rn(finish(o[dj][2 * r], ls),
                                                  finish(o[dj][2 * r + 1], ls));
                } else {
#pragma unroll
                    for (int dj = 0; dj < DT; ++dj)
                        *reinterpret_cast<float2*>(part + row * D + dj * 8 + c2) =
                            make_float2(o[dj][2 * r], o[dj][2 * r + 1]);
                    if ((lane & 3) == 0) {
                        part[group * D + row] = m[r];
                        part[group * D + group + row] = ls;
                    }
                }
            }
            ++done;
        }
        advance(C, s_pre, p.lens, B, H_kv);
    }
    cp_async_wait<0>();
}

// One block of D threads per (b, kv head, query row of the group): a sequence of no keys gets
// zeros, one of a single chunk was written by the decode kernel, otherwise the chunks' pieces
// are folded left to right, one column per thread.
template <int D>
__global__ void __launch_bounds__(D) combine_kernel(Params p) {
    const int group = p.group;
    const int seg = blockIdx.x / group, row = blockIdx.x - seg * group;
    const int b = seg / p.H_kv, kvh = seg - b * p.H_kv;
    const int nc = p.meta[b + 1] - p.meta[b];
    const int t = threadIdx.x;
    bf16* og =
        p.O + (static_cast<size_t>(b) * p.H_q + static_cast<size_t>(kvh) * group + row) * D + t;
    if (nc == 0) {
        *og = __float2bfloat16_rn(0.f);
        return;
    }
    if (nc == 1) return;
    const int PF = piece_floats(D, group);
    const float* part = p.ws + (static_cast<size_t>(p.meta[b]) * p.H_kv + kvh * nc) * PF;
    float M = -INFINITY, L = 0.f, O = 0.f;
#pragma unroll 4
    for (int c = 0; c < nc; ++c, part += PF) {
        const float mc = __ldcg(part + group * D + row);
        const float lc = __ldcg(part + group * D + group + row);
        const float oc = __ldcg(part + row * D + t);
        const FoldScales f = fold_scales(M, mc);
        L = fold1(L, lc, f);
        O = fold1(O, oc, f);
        M = f.mn;
    }
    *og = __float2bfloat16_rn(finish(O, L));
}

// The pieces: a buffer per process that grows on demand. A grown-out buffer is kept, not
// freed, since a CUDA graph captured earlier may still point at it.
float* workspace(size_t floats) {
    static std::vector<float*> old;
    static float* ws = nullptr;
    static size_t have = 0;
    if (have < floats) {
        if (ws) old.push_back(ws);
        have = std::max(floats, have * 2);
        SPARK_CUDA_CHECK(cudaMalloc(&ws, have * sizeof(float)));
    }
    return ws;
}
int* meta_buffer() {
    static int* meta = nullptr;
    if (!meta) SPARK_CUDA_CHECK(cudaMalloc(&meta, (kMaxBatchInv + 1) * sizeof(int)));
    return meta;
}

template <int D>
int resident_blocks() {
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(
            decode_kernel<D>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes<D>()));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, decode_kernel<D>,
                                                                       THREADS, smem_bytes<D>()));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <int D>
void launch(Params p, cudaStream_t stream) {
    p.grid = resident_blocks<D>();
    // Every sequence's chunks, at most: max_pages * page keys each.
    const int64_t chunks = static_cast<int64_t>(p.B) * p.H_kv *
                           cdiv64(static_cast<int64_t>(p.max_pages) << p.shift, CHUNK);
    p.ws = workspace(static_cast<size_t>(chunks) * piece_floats(D, p.group));
    p.meta = meta_buffer();
    decode_kernel<D><<<p.grid, THREADS, smem_bytes<D>(), stream>>>(p);
    SPARK_CHECK_LAUNCH();
    combine_kernel<D><<<p.B * p.H_kv * p.group, D, 0, stream>>>(p);
    SPARK_CHECK_LAUNCH();
}

}  // namespace idec

// ---- varlen prefill (attention_varlen variant 2) -------------------------------------------

namespace ivl {

constexpr int WARPS = 8;
constexpr int THREADS = WARPS * 32;
constexpr int BM = 16 * WARPS;  // 128 query rows
constexpr int BN = 64;          // keys per K/V tile: four 16-key steps
constexpr int STAGES = 3;
constexpr int TILES_PER_CHUNK = CHUNK / BN;

template <int D>
constexpr int stage_elems() {
    return 2 * BN * D;
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
    constexpr int Q_ITERS = BM * CH / THREADS;
    constexpr int KV_ITERS = BN * CH / THREADS;
    static_assert(Q_ITERS * THREADS == BM * CH && KV_ITERS * THREADS == BN * CH, "");
    static_assert(BM * D <= stage_elems<D>(), "the Q tile is staged through one K/V stage");

    extern __shared__ __align__(128) unsigned char smem_raw[];
    __shared__ int s_pre[kMaxBatchInv + 1];
    __shared__ int s_scan[32];
    bf16* smem = reinterpret_cast<bf16*>(smem_raw);
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;

    block_prefix(
        p.B, [&](int b) { return cdiv(p.cu_q[b + 1] - p.cu_q[b], BM); }, s_pre, s_scan);
    const int tile = blockIdx.x / p.H_q, h = blockIdx.x - tile * p.H_q;
    if (tile >= s_pre[p.B]) return;
    const int b = find_seq(s_pre, p.B, tile);
    const int nq = s_pre[b + 1] - s_pre[b], rank = tile - s_pre[b];
    const int q_tile = p.causal ? nq - 1 - rank : rank;
    const int q0 = q_tile * BM;
    const int tok0 = p.cu_q[b];
    const int q_len = p.cu_q[b + 1] - tok0, kv_len = p.lens[b];
    const int ctx = kv_len - q_len;
    const int kvh = h / p.group;
    const int* btb = p.bt + static_cast<size_t>(b) * p.max_pages;
    const size_t q_stride = static_cast<size_t>(p.H_q) * D;
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

    // The chunk's state and the running state the chunks fold into.
    float o[DT][4], oa[DT][4];
#pragma unroll
    for (int dj = 0; dj < DT; ++dj)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[dj][e] = oa[dj][e] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    float ma[2] = {-INFINITY, -INFINITY}, la[2] = {0.f, 0.f};
    const float sl2 = p.scale_log2;

    for (int t = 0; t < nt; ++t) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();  // tile t has landed and every warp is done with tile t - 1
        {
            const int n2 = t + STAGES - 1;
            if (n2 < nt) load_kv(n2 % STAGES, n2);
            cp_async_commit();
        }
        const bf16* ks = smem + (t % STAGES) * stage_elems<D>();
        const bf16* vs = ks + BN * D;
        const int kv0 = t * BN;
        const bool mask = kv0 + BN > kv_len || (p.causal && kv0 + BN - 1 > ctx + q0);

#pragma unroll
        for (int q = 0; q < BN / SLAB; ++q) {
            float sc[2][4];
            qk16<D>(sc, qf, ks + q * SLAB * D, lane);
            if (mask) {  // block-uniform
#pragma unroll
                for (int nj = 0; nj < 2; ++nj)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int i = q0 + warp * 16 + g + (e >> 1) * 8;
                        const int j = kv0 + q * SLAB + nj * 8 + c2 + (e & 1);
                        if (j >= kv_len || (p.causal && j > ctx + i)) sc[nj][e] = -INFINITY;
                    }
            }
            unsigned pa[4];
            step16<DT>(sc, m, l, o, sl2, pa);
            pv16<D>(o, pa, vs + q * SLAB * D, lane);
        }

        if ((t + 1) % TILES_PER_CHUNK == 0 || t == nt - 1) {  // a chunk ends: fold it
#pragma unroll
            for (int r = 0; r < 2; ++r) {
                const float ls = quad_sum(l[r]);
                const FoldScales f = fold_scales(ma[r], m[r]);
                la[r] = fold1(la[r], ls, f);
#pragma unroll
                for (int dj = 0; dj < DT; ++dj) {
                    oa[dj][2 * r] = fold1(oa[dj][2 * r], o[dj][2 * r], f);
                    oa[dj][2 * r + 1] = fold1(oa[dj][2 * r + 1], o[dj][2 * r + 1], f);
                    o[dj][2 * r] = o[dj][2 * r + 1] = 0.f;
                }
                ma[r] = f.mn;
                m[r] = -INFINITY;
                l[r] = 0.f;
            }
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = q0 + warp * 16 + g + 8 * r;
        if (row >= q_len) continue;
        bf16* out = Og + static_cast<size_t>(row) * q_stride + c2;
#pragma unroll
        for (int dj = 0; dj < DT; ++dj)
            *reinterpret_cast<__nv_bfloat162*>(out + dj * 8) = __floats2bfloat162_rn(
                finish(oa[dj][2 * r], la[r]), finish(oa[dj][2 * r + 1], la[r]));
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
    const int64_t tiles = (static_cast<int64_t>(T) + static_cast<int64_t>(p.B) * (BM - 1)) / BM;
    const int64_t blocks = tiles * p.H_q;
    SPARK_REQUIRE(blocks < (int64_t{1} << 31), "attention_varlen: too many tiles for one launch");
    varlen_kernel<D><<<static_cast<unsigned>(blocks), THREADS, smem_bytes<D>(), stream>>>(p);
    SPARK_CHECK_LAUNCH();
}

}  // namespace ivl

}  // namespace

// Entry points for attention_paged.cu's dispatch, which has checked the arguments.
void paged_decode_invariant_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                                 const __nv_bfloat16* V, const int* block_table,
                                 const int* seq_lens, __nv_bfloat16* O, int B, int H_q, int H_kv,
                                 int D, int shift, int max_pages, cudaStream_t stream) {
    SPARK_REQUIRE(H_q / H_kv <= 16,
                  "paged_decode: variant 2 needs at most 16 query heads per K/V head");
    idec::Params p{};
    p.Q = Q;
    p.K = K;
    p.V = V;
    p.O = O;
    p.bt = block_table;
    p.lens = seq_lens;
    p.B = B;
    p.H_q = H_q;
    p.H_kv = H_kv;
    p.group = H_q / H_kv;
    p.max_pages = max_pages;
    p.shift = shift;
    p.scale_log2 = kLog2e / std::sqrt(static_cast<float>(D));
    if (D == 64)
        idec::launch<64>(p, stream);
    else
        idec::launch<128>(p, stream);
}

void attention_varlen_invariant_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K,
                                     const __nv_bfloat16* V, const int* cu_seqlens_q,
                                     const int* seq_lens, const int* block_table, __nv_bfloat16* O,
                                     int B, int T, int H_q, int H_kv, int D, int shift,
                                     int max_pages, bool causal, cudaStream_t stream) {
    ivl::Params p{};
    p.Q = Q;
    p.K = K;
    p.V = V;
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
    p.scale_log2 = kLog2e / std::sqrt(static_cast<float>(D));
    if (D == 64)
        ivl::launch<64>(p, T, stream);
    else
        ivl::launch<128>(p, T, stream);
}

}  // namespace spark
