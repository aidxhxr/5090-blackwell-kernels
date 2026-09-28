// Fused scaled-dot-product attention forward for Blackwell sm_12x: RTX 5090 (sm_120), GB10.
//
//   O[b,h] = softmax(Q[b,h] K[b,kv(h)]^T / sqrt(D)) V[b,kv(h)]
//   Q, O = [B, H_q, S_q, D], K, V = [B, H_kv, S_kv, D], row-major, kv(h) = h / (H_q / H_kv)
//
// bf16 in and out, fp32 for every score, exponential and accumulator. One query head per
// block; under grouped-query attention (H_kv < H_q) the H_q / H_kv heads of a group read the
// same K/V head, which is a stride on the K/V base pointer and nothing else in these kernels
// (the decode kernel in attention_decode.cu groups them into one tile so K/V are read once).
// The causal mask hides key j from query i when j > i (the top-left alignment torch's
// is_causal uses). Every variant runs the online softmax: a running
// row max m and row sum l, with the O accumulator rescaled by exp(m_old - m_new) whenever a
// new key tile raises the max. Scores are scaled by log2(e)/sqrt(D) and exponentiated with
// ex2, so the exp is one MUFU instruction.
//
//   variant 0: one warp per query row, lanes own D/32 columns, keys one at a time.
//   variant 1: 128-row Q tile, 64-row K/V tiles converted to fp32 in shared memory, CUDA-core
//              FMAs on 4x8 (S) and 4x(D/8) (O) register tiles, the next K/V tile prefetched
//              into registers while the current one is used.
//   variant 2: mma.sync.m16n8k16 + ldmatrix: Q fragments held in registers for the whole KV
//              loop, S = Q K^T and O += P V on the tensor cores, P repacked in registers from
//              the S accumulators, 3-stage cp.async pipeline on K and V.
//   variant 3: variant 2's kernel with a schedule: the tiles of the last partial wave are
//              split along the keys over the idle SMs and merged by a combine kernel; a
//              64-row tile takes S_q <= 64 so a short query does not pay for 128 rows; and
//              decode shapes, where the query rows sharing a K/V head fit 16 rows, run the
//              flash-decoding kernel of attention_decode.cu.
//   variant 4: variant 3's tile with K and V fed by TMA (cp.async.bulk.tensor) into stages
//              guarded by full / empty mbarriers, issued by one lane, so the loop has no
//              __syncthreads: the 8 warps drift out of phase and one warp's softmax runs under
//              the other's mma on each scheduler. Same tail split as variant 3, and
//              the same decode paths.
//   variant 5: variant 4's tile and pipeline on a persistent grid: one block per SM takes
//              (b, h, q-tile) items from a queue in the heaviest-first order, the producer lane
//              publishes each to the warps through a shared-memory ring and issues its Q box
//              and K/V stages as the next loads of one running sequence, so the block prologue
//              is paid once per SM and the next item's Q is in flight during the current one's
//              last tiles. Tail split kept for the non-causal shapes; the same decode paths.

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
using attn::pack_bf16x2;
using attn::swz;

// 8 bf16 (one 16-byte chunk) to 8 floats.
__device__ __forceinline__ void unpack8(const uint4& u, float (&f)[8]) {
    const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 p = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[i]));
        f[2 * i] = p.x;
        f[2 * i + 1] = p.y;
    }
}

// ---------------------------------------------------------------------------------------
// Variant 0: one warp per query row. Lane l owns columns l*VEC .. l*VEC+VEC-1 of q, of every
// key and value row, and of the fp32 output row, so a key row is one coalesced 128/256-byte
// warp read. Each key costs a warp shuffle reduction for its score. Baseline only.
// ---------------------------------------------------------------------------------------
namespace v0 {

constexpr int THREADS = 128;  // 4 rows per block

template <int VEC>
struct Vec;
template <>
struct __align__(4) Vec<2> {
    __nv_bfloat162 h[1];
};
template <>
struct __align__(8) Vec<4> {
    __nv_bfloat162 h[2];
};

template <int VEC>
__device__ __forceinline__ void load_vec(const bf16* p, float (&f)[VEC]) {
    const Vec<VEC> v = *reinterpret_cast<const Vec<VEC>*>(p);
#pragma unroll
    for (int i = 0; i < VEC / 2; ++i) {
        const float2 t = __bfloat1622float2(v.h[i]);
        f[2 * i] = t.x;
        f[2 * i + 1] = t.y;
    }
}

template <int D>
__global__ void __launch_bounds__(THREADS)
    attention_v0_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                        const bf16* __restrict__ V, bf16* __restrict__ O, int H_q, int H_kv,
                        int S_q, int S_kv, float scale_log2, int causal) {
    constexpr int VEC = D / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (THREADS / 32) + (threadIdx.x >> 5);
    const int bh = blockIdx.y;
    const int bkv = kv_index(bh, H_q, H_kv);
    if (row >= S_q) return;
    const bf16* q = Q + (static_cast<size_t>(bh) * S_q + row) * D + lane * VEC;
    const bf16* k = K + static_cast<size_t>(bkv) * S_kv * D + lane * VEC;
    const bf16* v = V + static_cast<size_t>(bkv) * S_kv * D + lane * VEC;

    float qr[VEC];
    load_vec<VEC>(q, qr);
#pragma unroll
    for (int i = 0; i < VEC; ++i) qr[i] *= scale_log2;  // scores come out in the log2 domain

    float o[VEC];
#pragma unroll
    for (int i = 0; i < VEC; ++i) o[i] = 0.f;
    float m = -INFINITY, l = 0.f;

    const int n = causal ? min(S_kv, row + 1) : S_kv;
#pragma unroll 4
    for (int j = 0; j < n; ++j) {
        float kr[VEC], vr[VEC];
        load_vec<VEC>(k + static_cast<size_t>(j) * D, kr);
        load_vec<VEC>(v + static_cast<size_t>(j) * D, vr);
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

    const float inv = 1.f / l;
    bf16* out = O + (static_cast<size_t>(bh) * S_q + row) * D + lane * VEC;
    Vec<VEC> ov;
#pragma unroll
    for (int i = 0; i < VEC / 2; ++i)
        ov.h[i] = __floats2bfloat162_rn(o[2 * i] * inv, o[2 * i + 1] * inv);
    *reinterpret_cast<Vec<VEC>*>(out) = ov;
}

}  // namespace v0

// ---------------------------------------------------------------------------------------
// Variant 1: CUDA-core flash attention. A block owns 128 query rows (8 warps x 16 rows) and
// walks the keys in tiles of 64. The Q tile sits in shared memory as bf16; each K and V tile
// is converted to fp32 on its way into shared memory, so the inner loops are pure LDS + FFMA
// with no conversions (each converted element serves 128 query rows). The tile after the
// current one is loaded into registers before the current one is consumed.
//
// Lane mapping inside a warp (rg = lane / 8, kl = lane % 8):
//   S tile   : rows rg + 4*rr (rr < 4)  x  keys kl + 8*kk (kk < 8)   -> 32 fp32 per lane
//   O tile   : the same rows            x  float4 chunks kl + 8*dd    -> 4*D/8 fp32 per lane
// With the chunk swizzle above, the eight kl lanes read eight rows of K in eight bank groups,
// the four rg groups read four rows of Q in four bank groups, and the eight kl lanes read
// eight consecutive float4 chunks of one V row; nothing conflicts.
// ---------------------------------------------------------------------------------------
namespace v1 {

constexpr int BM = 128, BN = 64, THREADS = 256;

template <int D>
constexpr int smem_bytes() {
    return BM * D * 2 + 2 * BN * D * 4;  // Q bf16, K fp32, V fp32
}

template <int D>
__global__ void __launch_bounds__(THREADS, 1)
    attention_v1_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                        const bf16* __restrict__ V, bf16* __restrict__ O, int H_q, int H_kv,
                        int S_q, int S_kv, float scale_log2, int causal) {
    constexpr int CH = D / 8;                    // 16-byte chunks per bf16 row
    constexpr int DD = D / 32;                   // float4 chunks of O per lane
    constexpr int Q_ITERS = BM * CH / THREADS;   // 8 (D = 128), 4 (D = 64)
    constexpr int KV_ITERS = BN * CH / THREADS;  // 4, 2
    static_assert(Q_ITERS * THREADS == BM * CH && KV_ITERS * THREADS == BN * CH, "");

    extern __shared__ __align__(16) unsigned char smem_raw[];
    bf16* Qs = reinterpret_cast<bf16*>(smem_raw);
    float* Ks = reinterpret_cast<float*>(smem_raw + BM * D * 2);
    float* Vs = Ks + BN * D;

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int rg = lane >> 3, kl = lane & 7;
    const int bh = blockIdx.x;
    // Causal: the heavy tiles (most KV tiles) launch first so the tail of the grid is short.
    const int q_tile = causal ? gridDim.y - 1 - blockIdx.y : blockIdx.y;
    const int q0 = q_tile * BM;
    const int bkv = kv_index(bh, H_q, H_kv);
    const bf16* Qg = Q + static_cast<size_t>(bh) * S_q * D;
    const bf16* Kg = K + static_cast<size_t>(bkv) * S_kv * D;
    const bf16* Vg = V + static_cast<size_t>(bkv) * S_kv * D;
    bf16* Og = O + static_cast<size_t>(bh) * S_q * D;

    // Q tile, rows past S_q zeroed.
#pragma unroll
    for (int i = 0; i < Q_ITERS; ++i) {
        const int c = tid + i * THREADS;
        const int row = c / CH, ch = c % CH;
        uint4 u = make_uint4(0u, 0u, 0u, 0u);
        if (q0 + row < S_q)
            u = *reinterpret_cast<const uint4*>(Qg + static_cast<size_t>(q0 + row) * D + ch * 8);
        *reinterpret_cast<uint4*>(Qs + row * D + swz(row, ch) * 8) = u;
    }

    const int kv_end = causal ? min(S_kv, q0 + BM) : S_kv;
    const int T = cdiv(kv_end, BN);

    uint4 pre[2 * KV_ITERS];
    auto prefetch = [&](int t) {
#pragma unroll
        for (int i = 0; i < KV_ITERS; ++i) {
            const int c = tid + i * THREADS;
            const int row = c / CH, ch = c % CH;
            const int j = t * BN + row;
            pre[i] = make_uint4(0u, 0u, 0u, 0u);
            pre[KV_ITERS + i] = pre[i];
            if (j < S_kv) {
                pre[i] = *reinterpret_cast<const uint4*>(Kg + static_cast<size_t>(j) * D + ch * 8);
                pre[KV_ITERS + i] =
                    *reinterpret_cast<const uint4*>(Vg + static_cast<size_t>(j) * D + ch * 8);
            }
        }
    };
    // bf16 chunk ch of a row becomes fp32 chunks 2ch and 2ch+1, each swizzled. Eight lanes
    // store together, and chunks c and c+8 share banks, so the even chunks 0..14 that lanes
    // 0..7 would write in one instruction collide two ways; lanes with bit 2 of ch set write
    // their odd chunk first instead, and each store instruction then covers chunks 0..7 mod 8.
    auto stage = [&]() {
#pragma unroll
        for (int i = 0; i < KV_ITERS; ++i) {
            const int c = tid + i * THREADS;
            const int row = c / CH, ch = c % CH;
            const int b = (ch >> 2) & 1;
            const int ca = 2 * ch + b, cb = 2 * ch + 1 - b;
            float f[8];
            unpack8(pre[i], f);
            const float4 lo = make_float4(f[0], f[1], f[2], f[3]);
            const float4 hi = make_float4(f[4], f[5], f[6], f[7]);
            *reinterpret_cast<float4*>(Ks + row * D + swz(row, ca) * 4) = b ? hi : lo;
            *reinterpret_cast<float4*>(Ks + row * D + swz(row, cb) * 4) = b ? lo : hi;
            unpack8(pre[KV_ITERS + i], f);
            const float4 vlo = make_float4(f[0], f[1], f[2], f[3]);
            const float4 vhi = make_float4(f[4], f[5], f[6], f[7]);
            *reinterpret_cast<float4*>(Vs + row * D + swz(row, ca) * 4) = b ? vhi : vlo;
            *reinterpret_cast<float4*>(Vs + row * D + swz(row, cb) * 4) = b ? vlo : vhi;
        }
    };

    float o[4][DD][4];
#pragma unroll
    for (int rr = 0; rr < 4; ++rr)
#pragma unroll
        for (int dd = 0; dd < DD; ++dd)
#pragma unroll
            for (int i = 0; i < 4; ++i) o[rr][dd][i] = 0.f;
    float m[4], l[4];
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        m[rr] = -INFINITY;
        l[rr] = 0.f;
    }

    prefetch(0);
    for (int t = 0; t < T; ++t) {
        __syncthreads();  // every warp is done with the previous tile's Ks / Vs
        stage();
        __syncthreads();
        if (t + 1 < T) prefetch(t + 1);  // in flight while this tile is consumed

        // S = Q K^T for the lane's 4 rows x 8 keys, 4 d-values per step.
        float s[4][8];
#pragma unroll
        for (int rr = 0; rr < 4; ++rr)
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) s[rr][kk] = 0.f;
#pragma unroll 4
        for (int d0 = 0; d0 < D; d0 += 4) {
            float q[4][4];
#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const int row = warp * 16 + rg + 4 * rr;
                const uint2 u = *reinterpret_cast<const uint2*>(Qs + row * D +
                                                                swz(row, d0 >> 3) * 8 + (d0 & 4));
                const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
                const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
                q[rr][0] = a.x;
                q[rr][1] = a.y;
                q[rr][2] = b.x;
                q[rr][3] = b.y;
            }
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const int krow = kl + 8 * kk;
                const float4 k =
                    *reinterpret_cast<const float4*>(Ks + krow * D + swz(krow, d0 >> 2) * 4);
#pragma unroll
                for (int rr = 0; rr < 4; ++rr) {
                    s[rr][kk] = fmaf(q[rr][0], k.x, s[rr][kk]);
                    s[rr][kk] = fmaf(q[rr][1], k.y, s[rr][kk]);
                    s[rr][kk] = fmaf(q[rr][2], k.z, s[rr][kk]);
                    s[rr][kk] = fmaf(q[rr][3], k.w, s[rr][kk]);
                }
            }
        }

        const int kv0 = t * BN;
        if (kv0 + BN > S_kv || (causal && kv0 + BN - 1 > q0)) {  // warp-uniform
#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const int i = q0 + warp * 16 + rg + 4 * rr;
#pragma unroll
                for (int kk = 0; kk < 8; ++kk) {
                    const int j = kv0 + kl + 8 * kk;
                    if (j >= S_kv || (causal && j > i)) s[rr][kk] = -INFINITY;
                }
            }
        }

        // Online softmax per row: the row lives on the 8 kl lanes of one rg group.
#pragma unroll
        for (int rr = 0; rr < 4; ++rr) {
            float mx = s[rr][0];
#pragma unroll
            for (int kk = 1; kk < 8; ++kk) mx = fmaxf(mx, s[rr][kk]);
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 2));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 4));
            const float m_new = fmaxf(m[rr], mx * scale_log2);
            const float m_use = m_new == -INFINITY ? 0.f : m_new;  // a fully masked row so far
            const float alpha = ex2(m[rr] - m_use);
            float rs = 0.f;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk) {
                const float p = ex2(fmaf(s[rr][kk], scale_log2, -m_use));
                s[rr][kk] = p;
                rs += p;
            }
            l[rr] = fmaf(l[rr], alpha, rs);  // per-lane partial, merged in the epilogue
            m[rr] = m_new;
#pragma unroll
            for (int dd = 0; dd < DD; ++dd)
#pragma unroll
                for (int i = 0; i < 4; ++i) o[rr][dd][i] *= alpha;
        }

        // O += P V. p for key j sits in lane (rg*8 + j%8), register s[rr][j/8].
#pragma unroll
        for (int j = 0; j < BN; ++j) {
            float4 v[DD];
#pragma unroll
            for (int dd = 0; dd < DD; ++dd)
                v[dd] = *reinterpret_cast<const float4*>(Vs + j * D + swz(j, kl + 8 * dd) * 4);
#pragma unroll
            for (int rr = 0; rr < 4; ++rr) {
                const float p = __shfl_sync(kFullMask, s[rr][j >> 3], (lane & ~7) | (j & 7));
#pragma unroll
                for (int dd = 0; dd < DD; ++dd) {
                    o[rr][dd][0] = fmaf(p, v[dd].x, o[rr][dd][0]);
                    o[rr][dd][1] = fmaf(p, v[dd].y, o[rr][dd][1]);
                    o[rr][dd][2] = fmaf(p, v[dd].z, o[rr][dd][2]);
                    o[rr][dd][3] = fmaf(p, v[dd].w, o[rr][dd][3]);
                }
            }
        }
    }

    // Epilogue: merge the 8 partial row sums, divide, store 4 bf16 per (row, chunk).
#pragma unroll
    for (int rr = 0; rr < 4; ++rr) {
        float ls = l[rr];
        ls += __shfl_xor_sync(kFullMask, ls, 1);
        ls += __shfl_xor_sync(kFullMask, ls, 2);
        ls += __shfl_xor_sync(kFullMask, ls, 4);
        const float inv = 1.f / ls;
        const int row = q0 + warp * 16 + rg + 4 * rr;
        if (row >= S_q) continue;
#pragma unroll
        for (int dd = 0; dd < DD; ++dd) {
            uint2 u;
            u.x = pack_bf16x2(o[rr][dd][0] * inv, o[rr][dd][1] * inv);
            u.y = pack_bf16x2(o[rr][dd][2] * inv, o[rr][dd][3] * inv);
            *reinterpret_cast<uint2*>(Og + static_cast<size_t>(row) * D + (kl + 8 * dd) * 4) = u;
        }
    }
}

}  // namespace v1

// ---------------------------------------------------------------------------------------
// Variant 2: tensor-core flash attention. Same 128 x 64 block tile and 8 warps x 16 rows as
// variant 1, on mma.sync.m16n8k16 (see common.cuh for the fragment layouts):
//   * Q's A fragments for all D/16 k-steps are loaded once with ldmatrix and stay in registers;
//   * S = Q K^T: K is the "col" operand as it lies in memory ([key][d], d contiguous), so a
//     plain ldmatrix.x4 on 16 key rows gives the B fragments of two n8 key tiles;
//   * online softmax on the S accumulators: lane (g, c) owns rows g and g+8, keys 2c, 2c+1 of
//     every n8 tile, so a row max / sum is 16 values per lane plus a 2-step shuffle across the
//     4 lanes that share g;
//   * P: the S accumulator layout (row g / g+8, cols 2c, 2c+1) is the A-fragment layout of the
//     next mma once pairs are packed to bf16x2, so P never leaves the registers;
//   * O += P V: V is [key][d] = [k][n] row-major, the B operand ldmatrix.x4.trans wants;
//   * K and V tiles stream through a 3-stage cp.async pipeline, one __syncthreads per tile.
// ---------------------------------------------------------------------------------------
namespace v2 {

constexpr int BN = 64;

// Work assignment. Tiles are numbered tile = q_rank * bh_count + bh, with q_rank walking the
// Q tiles heaviest first under the causal mask (the most KV tiles first, so the tail of the
// grid is short). Blocks [0, dp_tiles) each own one whole tile; the blocks after them split
// the remaining tiles `split` ways along the keys, write unnormalized fp32 partials (O, m, l)
// to `ws`, and a combine kernel merges them (variant 3; variant 2 has dp_tiles = tiles).
struct Sched {
    int bh_count;   // B * H_q
    int H_q, H_kv;  // tile bh reads K/V head kv_index(bh, H_q, H_kv)
    int q_tiles;    // ceil(S_q / BM)
    int dp_tiles;
    int split;
    float* ws;  // (tiles - dp_tiles) * split partials of BM * (D + 2) floats each
};

template <int D>
constexpr int stage_elems() {
    return 2 * BN * D;  // K tile then V tile
}
template <int D, int STAGES>
constexpr int smem_bytes() {
    return STAGES * stage_elems<D>() * 2;
}
template <int D, int WARPS>
constexpr int partial_floats() {
    return 16 * WARPS * (D + 2);  // O rows, then m, then l
}

template <int D, int WARPS, int STAGES>
__global__ void __launch_bounds__(WARPS * 32, 1)
    attention_v2_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                        const bf16* __restrict__ V, bf16* __restrict__ O, int S_q, int S_kv,
                        float scale_log2, int causal, Sched sched) {
    constexpr int BM = 16 * WARPS;
    constexpr int THREADS = 32 * WARPS;
    constexpr int CH = D / 8;                    // 16-byte chunks per row
    constexpr int KT = D / 16;                   // k16 steps of Q K^T
    constexpr int DT = D / 8;                    // n8 tiles of O
    constexpr int NT = BN / 8;                   // n8 tiles of S
    constexpr int PT = BN / 16;                  // k16 steps of P V
    constexpr int Q_ITERS = BM * CH / THREADS;   // 8 (D = 128), 4 (D = 64)
    constexpr int KV_ITERS = BN * CH / THREADS;  // 2, 1 (4, 2 with 4 warps)
    static_assert(Q_ITERS * THREADS == BM * CH && KV_ITERS * THREADS == BN * CH, "");
    static_assert(BM * D <= stage_elems<D>(), "the Q tile is staged through one K/V stage");

    extern __shared__ __align__(128) unsigned char smem_raw[];
    bf16* smem = reinterpret_cast<bf16*>(smem_raw);

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;

    int tile, slice = 0, split = 1;
    if (static_cast<int>(blockIdx.x) < sched.dp_tiles) {
        tile = blockIdx.x;
    } else {
        const int r = blockIdx.x - sched.dp_tiles;
        tile = sched.dp_tiles + r / sched.split;
        slice = r % sched.split;
        split = sched.split;
    }
    const int bh = tile % sched.bh_count;
    const int q_rank = tile / sched.bh_count;
    const int q_tile = causal ? sched.q_tiles - 1 - q_rank : q_rank;
    const int q0 = q_tile * BM;
    const int bkv = kv_index(bh, sched.H_q, sched.H_kv);
    const bf16* Qg = Q + static_cast<size_t>(bh) * S_q * D;
    const bf16* Kg = K + static_cast<size_t>(bkv) * S_kv * D;
    const bf16* Vg = V + static_cast<size_t>(bkv) * S_kv * D;
    bf16* Og = O + static_cast<size_t>(bh) * S_q * D;

    // Q tile through stage 0 into A fragments: qf[kk] covers d = 16kk .. 16kk+15.
#pragma unroll
    for (int i = 0; i < Q_ITERS; ++i) {
        const int c = tid + i * THREADS;
        const int row = c / CH, ch = c % CH;
        const bool ok = q0 + row < S_q;
        cp_async_16_zfill(smem + row * D + swz(row, ch) * 8,
                          Qg + static_cast<size_t>(ok ? q0 + row : 0) * D + ch * 8, ok);
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
    __syncthreads();  // stage 0 is free again

    // KV tiles [tb, te) of the T this Q tile needs (all of them unless the tile is split).
    const int kv_end = causal ? min(S_kv, q0 + BM) : S_kv;
    const int T = cdiv(kv_end, BN);
    const int tb = slice * T / split;
    const int te = (slice + 1) * T / split;

    auto load_kv = [&](int stage, int t) {
        bf16* ks = smem + stage * stage_elems<D>();
        bf16* vs = ks + BN * D;
#pragma unroll
        for (int i = 0; i < KV_ITERS; ++i) {
            const int c = tid + i * THREADS;
            const int row = c / CH, ch = c % CH;
            const int j = t * BN + row;
            const bool ok = j < S_kv;
            const size_t off = static_cast<size_t>(ok ? j : 0) * D + ch * 8;
            cp_async_16_zfill(ks + row * D + swz(row, ch) * 8, Kg + off, ok);
            cp_async_16_zfill(vs + row * D + swz(row, ch) * 8, Vg + off, ok);
        }
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (tb + s < te) load_kv(s, tb + s);
        cp_async_commit();
    }

    float o[DT][4];
#pragma unroll
    for (int dj = 0; dj < DT; ++dj)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[dj][e] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

    for (int t = tb, it = 0; t < te; ++t, ++it) {
        cp_async_wait<STAGES - 2>();  // tile t has landed (for this thread)
        __syncthreads();              // ... for every thread; and stage (it-1)%STAGES is free
        {
            const int nt = t + STAGES - 1;
            if (nt < te) load_kv((it + STAGES - 1) % STAGES, nt);
            cp_async_commit();
        }
        const bf16* ks = smem + (it % STAGES) * stage_elems<D>();
        const bf16* vs = ks + BN * D;

        // S = Q K^T: 16 rows x 64 keys per warp, 8 n8 accumulators.
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
        if (kv0 + BN > S_kv || (causal && kv0 + BN - 1 > q0)) {  // warp-uniform
#pragma unroll
            for (int nj = 0; nj < NT; ++nj)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int i = q0 + warp * 16 + g + (e >> 1) * 8;
                    const int j = kv0 + nj * 8 + c2 + (e & 1);
                    if (j >= S_kv || (causal && j > i)) s[nj][e] = -INFINITY;
                }
        }

        // Online softmax, rows g (e = 0, 1) and g+8 (e = 2, 3).
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float mx = fmaxf(s[0][2 * r], s[0][2 * r + 1]);
#pragma unroll
            for (int nj = 1; nj < NT; ++nj) mx = fmaxf(mx, fmaxf(s[nj][2 * r], s[nj][2 * r + 1]));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 2));
            const float m_new = fmaxf(m[r], mx * scale_log2);
            const float m_use = m_new == -INFINITY ? 0.f : m_new;
            const float alpha = ex2(m[r] - m_use);
            float rs = 0.f;
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                const float p0 = ex2(fmaf(s[nj][2 * r], scale_log2, -m_use));
                const float p1 = ex2(fmaf(s[nj][2 * r + 1], scale_log2, -m_use));
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

        // P as A fragments: k16 step kt covers keys 16kt .. 16kt+15 = n8 tiles 2kt, 2kt+1.
        unsigned pa[PT][4];
#pragma unroll
        for (int kt = 0; kt < PT; ++kt) {
            pa[kt][0] = pack_bf16x2(s[2 * kt][0], s[2 * kt][1]);
            pa[kt][1] = pack_bf16x2(s[2 * kt][2], s[2 * kt][3]);
            pa[kt][2] = pack_bf16x2(s[2 * kt + 1][0], s[2 * kt + 1][1]);
            pa[kt][3] = pack_bf16x2(s[2 * kt + 1][2], s[2 * kt + 1][3]);
        }

        // O += P V: 16 rows x D per warp, D/8 n8 accumulators.
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

    // Epilogue. Whole tile: merge the 4 partial row sums, divide, two bf16 per store from
    // registers. Split tile: the unnormalized fp32 O rows plus (m, l) go to the workspace.
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float ls = l[r];
        ls += __shfl_xor_sync(kFullMask, ls, 1);
        ls += __shfl_xor_sync(kFullMask, ls, 2);
        const int lrow = warp * 16 + g + 8 * r;  // row within the tile
        if (split == 1) {
            const int row = q0 + lrow;
            if (row >= S_q) continue;
            const float inv = 1.f / ls;
            bf16* out = Og + static_cast<size_t>(row) * D + c2;
#pragma unroll
            for (int dj = 0; dj < DT; ++dj)
                *reinterpret_cast<__nv_bfloat162*>(out + dj * 8) =
                    __floats2bfloat162_rn(o[dj][2 * r] * inv, o[dj][2 * r + 1] * inv);
        } else {
            float* part = sched.ws + (static_cast<size_t>(tile - sched.dp_tiles) * split + slice) *
                                         partial_floats<D, WARPS>();
            float* orow = part + lrow * D + c2;
#pragma unroll
            for (int dj = 0; dj < DT; ++dj)
                *reinterpret_cast<float2*>(orow + dj * 8) =
                    make_float2(o[dj][2 * r], o[dj][2 * r + 1]);
            if ((lane & 3) == 0) {
                part[BM * D + lrow] = m[r];
                part[BM * D + BM + lrow] = ls;
            }
        }
    }
}

// Merges the `split` partials of every split tile: M = max m_i, L = sum l_i 2^(m_i - M),
// O = sum O_i 2^(m_i - M) / L. One thread per (row, 8 columns), so the reads of each partial
// are contiguous.
template <int D, int WARPS>
__global__ void attention_combine_kernel(bf16* __restrict__ O, int S_q, int causal, Sched sched) {
    constexpr int BM = 16 * WARPS;
    constexpr int CH = D / 8;
    const int tile = sched.dp_tiles + blockIdx.y;
    const int bh = tile % sched.bh_count;
    const int q_rank = tile / sched.bh_count;
    const int q_tile = causal ? sched.q_tiles - 1 - q_rank : q_rank;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // (row, chunk) of the tile
    if (idx >= BM * CH) return;
    const int lrow = idx / CH, ch = idx % CH;
    const int row = q_tile * BM + lrow;
    if (row >= S_q) return;
    const float* base =
        sched.ws + static_cast<size_t>(blockIdx.y) * sched.split * partial_floats<D, WARPS>();

    float M = -INFINITY;
    for (int i = 0; i < sched.split; ++i)
        M = fmaxf(M, base[i * partial_floats<D, WARPS>() + BM * D + lrow]);
    float L = 0.f, acc[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[e] = 0.f;
    for (int i = 0; i < sched.split; ++i) {
        const float* part = base + i * partial_floats<D, WARPS>();
        const float w = ex2(part[BM * D + lrow] - M);  // 0 for a slice that saw no key of this row
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
    u.x = pack_bf16x2(acc[0] * inv, acc[1] * inv);
    u.y = pack_bf16x2(acc[2] * inv, acc[3] * inv);
    u.z = pack_bf16x2(acc[4] * inv, acc[5] * inv);
    u.w = pack_bf16x2(acc[6] * inv, acc[7] * inv);
    *reinterpret_cast<uint4*>(O + (static_cast<size_t>(bh) * S_q + row) * D + ch * 8) = u;
}

constexpr int STAGES = 3;

template <int D, int WARPS>
int resident_blocks() {
    constexpr int bytes = smem_bytes<D, STAGES>();
    static int resident = 0;  // blocks resident per GPU; also the > 48 KB smem opt-in
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(attention_v2_kernel<D, WARPS, STAGES>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, attention_v2_kernel<D, WARPS, STAGES>, WARPS * 32, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

// Tail partials, one buffer per device, grown on demand. Shared by every instantiation and
// every stream, so two split launches must not run concurrently on different streams.
struct Workspace {
    float* ws = nullptr;
    size_t floats = 0;
};

// The whole-tile schedule for B * H heads of q_tiles Q tiles each.
template <int WARPS>
Sched whole_tiles(int B, int H_q, int H_kv, int S_q) {
    Sched s;
    s.bh_count = B * H_q;
    s.H_q = H_q;
    s.H_kv = H_kv;
    s.q_tiles = cdiv(S_q, 16 * WARPS);
    s.dp_tiles = s.bh_count * s.q_tiles;
    s.split = 1;
    s.ws = nullptr;
    return s;
}

// Splits the tiles of the last partial wave (variants 3 and 4). The tail tiles are the last in
// the heaviest-first order, so the last one has the fewest KV tiles; every slice must get at
// least one. The workspace is one per (D, WARPS), shared by the variants that use it.
template <int D, int WARPS>
void split_tail_tiles(Sched& s, int resident, int S_kv, bool causal) {
    constexpr int BM = 16 * WARPS;
    const int tiles = s.bh_count * s.q_tiles;
    const int tail = tiles % resident;
    const int q_last = causal ? 0 : s.q_tiles - 1;  // rank tiles - 1
    const int T_last = cdiv(causal ? std::min(S_kv, q_last * BM + BM) : S_kv, BN);
    const int split = tail > 0 ? std::min(T_last, resident / tail) : 1;
    if (split <= 1) return;
    s.dp_tiles = tiles - tail;
    s.split = split;
    static Workspace w;
    const size_t need = static_cast<size_t>(tail) * split * partial_floats<D, WARPS>();
    if (w.floats < need) {
        if (w.ws) SPARK_CUDA_CHECK(cudaFree(w.ws));
        SPARK_CUDA_CHECK(cudaMalloc(&w.ws, need * sizeof(float)));
        w.floats = need;
    }
    s.ws = w.ws;
}

// Blocks in the grid for a schedule: the whole tiles plus `split` slices of each tail tile.
inline int grid_blocks(const Sched& s) {
    return s.dp_tiles + (s.bh_count * s.q_tiles - s.dp_tiles) * s.split;
}

template <int D, int WARPS>
void launch_combine(bf16* O, int S_q, bool causal, const Sched& s, cudaStream_t stream) {
    constexpr int per_tile = 16 * WARPS * (D / 8);
    const dim3 cgrid(cdiv(per_tile, 256), s.bh_count * s.q_tiles - s.dp_tiles);
    attention_combine_kernel<D, WARPS><<<cgrid, 256, 0, stream>>>(O, S_q, causal ? 1 : 0, s);
}

// `split_tail`: variant 3. Variant 2 runs every tile whole on the 8-warp tile.
template <int D, int WARPS>
void launch(const bf16* Q, const bf16* K, const bf16* V, bf16* O, int B, int H_q, int H_kv, int S_q,
            int S_kv, float scale_log2, bool causal, bool split_tail, cudaStream_t stream) {
    constexpr int bytes = smem_bytes<D, STAGES>();
    const int resident = resident_blocks<D, WARPS>();

    Sched s = whole_tiles<WARPS>(B, H_q, H_kv, S_q);
    if (split_tail) split_tail_tiles<D, WARPS>(s, resident, S_kv, causal);
    const int grid = grid_blocks(s);
    attention_v2_kernel<D, WARPS, STAGES>
        <<<grid, WARPS * 32, bytes, stream>>>(Q, K, V, O, S_q, S_kv, scale_log2, causal ? 1 : 0, s);
    if (s.split > 1) launch_combine<D, WARPS>(O, S_q, causal, s, stream);
}

}  // namespace v2

// ---------------------------------------------------------------------------------------
// Variant 4: the mbarrier pipeline. Variant 2's 128 x 64 tile, fragments, softmax and epilogue
// with the operand traffic moved off the warps:
//   * K and V tiles arrive by TMA (cp.async.bulk.tensor) into STAGES stages with a full / empty
//     mbarrier pair each, as in hgemm variant 5, and the Q tile comes the same way through
//     stage 0 before the loop. Lane 0 of warp 0 issues the loads (two or four instructions per
//     tile); the other 255 threads never compute a copy address;
//   * a warp waits on a stage's full barrier before reading it and arrives on its empty barrier
//     after its P V, and the producer waits on empty before refilling. Nothing in the loop is
//     block-wide. Variant 2's __syncthreads per tile kept the two warps of each scheduler in
//     the same phase, so their softmaxes were a hole in the tensor pipe; here the warps drift,
//     and while one is in its softmax the other is usually issuing mma;
//   * the TMA 128-byte swizzle lays a tile out as D/64 boxes of rows x 64 columns, each row
//     128 B with its 16-byte chunks XORed by row % 8. ldmatrix addresses go through box_off().
// The barrier bookkeeping counts the Q tile as load 0 and KV tile t as load t + 1, so load u
// lives in stage u % STAGES and is the (u / STAGES)-th use of it.
//
// The kernel also carries the ping-pong schedules that were tried and measured slower
// (docs/design/attention.md, "Variant 4"): two groups of four warps taking turns on the tensor
// pipe through named barriers, FlashAttention-3 style. They compile only with
// -DSPARK_ATTN_V4_EXPERIMENTS and are picked by SPARK_ATTN_V4_MODE.
// ---------------------------------------------------------------------------------------
namespace v4 {

using v2::BN;
using v2::Sched;

constexpr int WARPS = 8, THREADS = 256, BM = 16 * WARPS;
constexpr int GROUP_THREADS = THREADS / 2;
constexpr int BOX_COLS = 64;   // 128 B, one swizzle span
constexpr int BAR_GROUP0 = 1;  // named barriers 1 and 2: "group g may issue mma"

template <int D>
constexpr int tile_bytes() {
    return BN * D * 2;  // one K or V tile
}
template <int D>
constexpr int stage_bytes() {
    return 2 * tile_bytes<D>();
}
// STAGES stages, then the barriers in a final KB. Stages are whole KB and the region starts on
// a 1 KB boundary (the kernel checks), which the swizzle needs.
template <int D, int STAGES>
constexpr int smem_bytes() {
    return STAGES * stage_bytes<D>() + 1024;
}

// Element offset of logical (row, 16-byte chunk) in a tile of ROWS rows x D columns laid out
// as D/64 TMA boxes of ROWS x 64 with the 128-byte swizzle: box, then row, then the chunk
// XORed with row % 8. Eight consecutive rows at one logical chunk hit eight bank groups.
template <int ROWS>
__device__ __forceinline__ int box_off(int row, int ch) {
    return (ch >> 3) * (ROWS * BOX_COLS) + row * BOX_COLS + ((ch & 7) ^ (row & 7)) * 8;
}

// Schedule knobs for the experiments in docs/design/attention.md; the shipped kernel is
// <D, 3, 0, 0, 0>. MODE: 0 = every warp runs variant 2's loop on the TMA pipeline, 1 = ping-pong
// with two turns per tile (Q K^T, then P V), 2 = one turn per tile (Q K^T of tile t+1 with P V
// of tile t). PAIR: 0 = the groups are warps 0-3 and 4-7 (one warp of each group per
// scheduler), 1 = even and odd warps. OPT bits: 1 = a warp skips a causal tile whose keys all
// follow its 16 rows (the fully masked half of the second diagonal tile), 2 = the O rescale is
// skipped when no row max in the warp moved (warp-uniform branch).
template <int D, int STAGES, int MODE, int PAIR, int OPT>
__global__ void __launch_bounds__(THREADS, 1)
    attention_v4_kernel(const __grid_constant__ CUtensorMap tmQ,
                        const __grid_constant__ CUtensorMap tmK,
                        const __grid_constant__ CUtensorMap tmV, bf16* __restrict__ O, int S_q,
                        int S_kv, float scale_log2, int causal, Sched sched) {
    constexpr int KT = D / 16;   // k16 steps of Q K^T
    constexpr int DT = D / 8;    // n8 tiles of O
    constexpr int NT = BN / 8;   // n8 tiles of S
    constexpr int PT = BN / 16;  // k16 steps of P V
    constexpr int BOXES = D / BOX_COLS;
    constexpr int STAGE = stage_bytes<D>();
    static_assert(BM * D * 2 <= STAGE, "the Q tile is staged through stage 0");
    static_assert(MODE != 2 || STAGES >= 3,
                  "one turn per tile reads K of t+1 while V of t is live");

    extern __shared__ __align__(1024) unsigned char smem[];
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(smem + STAGES * STAGE);
    uint64_t* empty_bar = full_bar + STAGES;
    if (smem_u32(smem) % 1024 != 0) __trap();

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    const int group = PAIR == 0 ? warp >> 2 : warp & 1;
    const int bar_mine = BAR_GROUP0 + group, bar_other = BAR_GROUP0 + 1 - group;

    int tile, slice = 0, split = 1;
    if (static_cast<int>(blockIdx.x) < sched.dp_tiles) {
        tile = blockIdx.x;
    } else {
        const int r = blockIdx.x - sched.dp_tiles;
        tile = sched.dp_tiles + r / sched.split;
        slice = r % sched.split;
        split = sched.split;
    }
    const int bh = tile % sched.bh_count;
    const int bkv = kv_index(bh, sched.H_q, sched.H_kv);  // the K/V head this tile reads
    const int q_rank = tile / sched.bh_count;
    const int q_tile = causal ? sched.q_tiles - 1 - q_rank : q_rank;
    const int q0 = q_tile * BM;
    bf16* Og = O + static_cast<size_t>(bh) * S_q * D;

    const int kv_end = causal ? min(S_kv, q0 + BM) : S_kv;
    const int T = cdiv(kv_end, BN);
    const int tb = slice * T / split;
    const int te = (slice + 1) * T / split;
    const int n = te - tb;    // KV tiles of this block
    const int loads = n + 1;  // plus the Q tile

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full_bar[s], 1);       // the producer's arrive.expect_tx
            mbar_init(&empty_bar[s], WARPS);  // one arrive per warp
        }
        fence_mbar_init();
    }
    __syncthreads();

    // Load u into stage u % STAGES: the Q tile (u = 0) or KV tile tb + u - 1. The n-th use of a
    // stage (n >= 1) waits for the warps' release of the (n-1)-th.
    auto produce = [&](int u) {
        const int s = u % STAGES;
        const int use = u / STAGES;
        if (use > 0) mbar_wait(&empty_bar[s], (use - 1) & 1);
        unsigned char* dst = smem + s * STAGE;
        if (u == 0) {
            mbar_arrive_expect_tx(&full_bar[s], BM * D * 2);
#pragma unroll
            for (int b = 0; b < BOXES; ++b)
                tma_load_3d(dst + b * BM * BOX_COLS * 2, &tmQ, &full_bar[s], b * BOX_COLS, q0, bh);
        } else {
            const int j0 = (tb + u - 1) * BN;
            mbar_arrive_expect_tx(&full_bar[s], STAGE);
#pragma unroll
            for (int b = 0; b < BOXES; ++b) {
                tma_load_3d(dst + b * BN * BOX_COLS * 2, &tmK, &full_bar[s], b * BOX_COLS, j0, bkv);
                tma_load_3d(dst + tile_bytes<D>() + b * BN * BOX_COLS * 2, &tmV, &full_bar[s],
                            b * BOX_COLS, j0, bkv);
            }
        }
    };
    if (tid == 0) {
        prefetch_tensormap(&tmQ);
        prefetch_tensormap(&tmK);
        prefetch_tensormap(&tmV);
        for (int u = 0; u < min(STAGES, loads); ++u) produce(u);
    }
    // Stage and parity of KV tile tb + it (load it + 1).
    auto stage_of = [&](int it) -> const bf16* {
        return reinterpret_cast<const bf16*>(smem + ((it + 1) % STAGES) * STAGE);
    };
    auto wait_tile = [&](int it) {
        mbar_wait(&full_bar[(it + 1) % STAGES], ((it + 1) / STAGES) & 1);
    };
    auto release_tile = [&](int it) {
        fence_proxy_async_smem();
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty_bar[(it + 1) % STAGES]);
    };

    // Q tile into A fragments: qf[kk] covers d = 16kk .. 16kk+15. Then stage 0 is released.
    unsigned qf[KT][4];
    {
        mbar_wait(&full_bar[0], 0);
        const bf16* qs = reinterpret_cast<const bf16*>(smem);
        const int row = warp * 16 + (lane & 15);
#pragma unroll
        for (int kk = 0; kk < KT; ++kk)
            ldmatrix_x4(qf[kk], qs + box_off<BM>(row, 2 * kk + (lane >> 4)));
        fence_proxy_async_smem();
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty_bar[0]);
    }

    float o[DT][4];
#pragma unroll
    for (int dj = 0; dj < DT; ++dj)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[dj][e] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
    float sacc[NT][4];
    unsigned pa[PT][4];

    // S = Q K^T of the tile in `ks`: 16 rows x 64 keys per warp, 8 n8 accumulators.
    auto qk = [&](const bf16* ks) {
#pragma unroll
        for (int nj = 0; nj < NT; ++nj)
#pragma unroll
            for (int e = 0; e < 4; ++e) sacc[nj][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
#pragma unroll
            for (int nj = 0; nj < NT; nj += 2) {
                const int row = nj * 8 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4(r, ks + box_off<BN>(row, 2 * kk + (lane >> 4)));
                const unsigned b0[2] = {r[0], r[2]};
                const unsigned b1[2] = {r[1], r[3]};
                mma_bf16_16816(sacc[nj], qf[kk], b0);
                mma_bf16_16816(sacc[nj + 1], qf[kk], b1);
            }
        }
    };
    // Mask, online softmax on the S accumulators (rows g and g+8, the O rescale included) and
    // P packed as the A fragments of P V: variant 2's code.
    auto softmax = [&](int t) {
        const int kv0 = t * BN;
        if (kv0 + BN > S_kv || (causal && kv0 + BN - 1 > q0)) {  // warp-uniform
#pragma unroll
            for (int nj = 0; nj < NT; ++nj)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int i = q0 + warp * 16 + g + (e >> 1) * 8;
                    const int j = kv0 + nj * 8 + c2 + (e & 1);
                    if (j >= S_kv || (causal && j > i)) sacc[nj][e] = -INFINITY;
                }
        }
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float mx = fmaxf(sacc[0][2 * r], sacc[0][2 * r + 1]);
#pragma unroll
            for (int nj = 1; nj < NT; ++nj)
                mx = fmaxf(mx, fmaxf(sacc[nj][2 * r], sacc[nj][2 * r + 1]));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 2));
            const float m_new = fmaxf(m[r], mx * scale_log2);
            const float m_use = m_new == -INFINITY ? 0.f : m_new;
            alpha[r] = ex2(m[r] - m_use);
            float rs = 0.f;
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                const float p0 = ex2(fmaf(sacc[nj][2 * r], scale_log2, -m_use));
                const float p1 = ex2(fmaf(sacc[nj][2 * r + 1], scale_log2, -m_use));
                sacc[nj][2 * r] = p0;
                sacc[nj][2 * r + 1] = p1;
                rs += p0 + p1;
            }
            l[r] = fmaf(l[r], alpha[r], rs);
            m[r] = m_new;
        }
        // The rescale is 1 for every row of the warp once the maxima have settled; with OPT
        // bit 2 the 64 multiplies are skipped on those tiles (a warp-uniform vote).
        if (!(OPT & 2) || __any_sync(kFullMask, alpha[0] != 1.f || alpha[1] != 1.f)) {
#pragma unroll
            for (int dj = 0; dj < DT; ++dj) {
                o[dj][0] *= alpha[0];
                o[dj][1] *= alpha[0];
                o[dj][2] *= alpha[1];
                o[dj][3] *= alpha[1];
            }
        }
#pragma unroll
        for (int kt = 0; kt < PT; ++kt) {
            pa[kt][0] = pack_bf16x2(sacc[2 * kt][0], sacc[2 * kt][1]);
            pa[kt][1] = pack_bf16x2(sacc[2 * kt][2], sacc[2 * kt][3]);
            pa[kt][2] = pack_bf16x2(sacc[2 * kt + 1][0], sacc[2 * kt + 1][1]);
            pa[kt][3] = pack_bf16x2(sacc[2 * kt + 1][2], sacc[2 * kt + 1][3]);
        }
    };
    // O += P V with the V tile in `vs`: 16 rows x D per warp, D/8 n8 accumulators.
    auto pv = [&](const bf16* vs) {
#pragma unroll
        for (int kt = 0; kt < PT; ++kt) {
#pragma unroll
            for (int dj = 0; dj < DT; dj += 2) {
                const int row = kt * 16 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4_trans(r, vs + box_off<BN>(row, dj + (lane >> 4)));
                const unsigned b0[2] = {r[0], r[1]};
                const unsigned b1[2] = {r[2], r[3]};
                mma_bf16_16816(o[dj], pa[kt], b0);
                mma_bf16_16816(o[dj + 1], pa[kt], b1);
            }
        }
    };
    // Turns. A turn is one bar.sync by the 128 threads that take it and one bar.arrive by the
    // 128 that hand it over, on a 256-thread barrier. Group 0 has the first turn, so group 1
    // arrives once up front and skips its last arrive to leave the barriers balanced.
    auto take_turn = [&]() {
        if (MODE != 0) named_barrier_sync(bar_mine, THREADS);
    };
    auto give_turn = [&](bool last) {
        if (MODE != 0 && !(last && group == 1)) named_barrier_arrive(bar_other, THREADS);
    };
    if (MODE != 0 && group == 1) named_barrier_arrive(BAR_GROUP0, THREADS);

    if constexpr (MODE == 2) {
        // Turn 0: S_0. Then, per tile t: one turn with S_{t+1} and O += P_t V_t, then the
        // softmax of tile t+1 under the other group's turn. Group 0's turn starts when group 1
        // has finished P V of tile t-1, so that tile's stage is free: lane 0 refills it.
        wait_tile(0);
        take_turn();
        qk(stage_of(0));
        give_turn(false);
        softmax(tb);
        for (int it = 0; it < n; ++it) {
            const bool more = it + 1 < n;
            if (more) wait_tile(it + 1);
            take_turn();
            if (tid == 0 && it + STAGES < loads) produce(it + STAGES);
            if (more) qk(stage_of(it + 1));
            pv(stage_of(it) + BN * D);
            release_tile(it);
            give_turn(!more);
            if (more) softmax(tb + it + 1);
        }
    } else {
        for (int it = 0; it < n; ++it) {
            wait_tile(it);
            // Under the causal mask the keys of a tile may all follow every row of this warp
            // (rows q0 + 16 warp .. + 15 against keys from (tb + it) BN): every score is masked,
            // every p is 0, and the tile is a no-op for the warp.
            const bool dead =
                (OPT & 1) && MODE == 0 && causal && (tb + it) * BN > q0 + warp * 16 + 15;
            take_turn();
            if (tid == 0 && it + STAGES < loads) produce(it + STAGES);
            if (!dead) qk(stage_of(it));
            give_turn(false);
            if (!dead) softmax(tb + it);
            take_turn();
            if (!dead) pv(stage_of(it) + BN * D);
            release_tile(it);
            give_turn(it + 1 == n);
        }
    }

    // Epilogue, as variant 2's.
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float ls = l[r];
        ls += __shfl_xor_sync(kFullMask, ls, 1);
        ls += __shfl_xor_sync(kFullMask, ls, 2);
        const int lrow = warp * 16 + g + 8 * r;
        if (split == 1) {
            const int row = q0 + lrow;
            if (row >= S_q) continue;
            const float inv = 1.f / ls;
            bf16* out = Og + static_cast<size_t>(row) * D + c2;
#pragma unroll
            for (int dj = 0; dj < DT; ++dj)
                *reinterpret_cast<__nv_bfloat162*>(out + dj * 8) =
                    __floats2bfloat162_rn(o[dj][2 * r] * inv, o[dj][2 * r + 1] * inv);
        } else {
            float* part = sched.ws + (static_cast<size_t>(tile - sched.dp_tiles) * split + slice) *
                                         v2::partial_floats<D, WARPS>();
            float* orow = part + lrow * D + c2;
#pragma unroll
            for (int dj = 0; dj < DT; ++dj)
                *reinterpret_cast<float2*>(orow + dj * 8) =
                    make_float2(o[dj][2 * r], o[dj][2 * r + 1]);
            if ((lane & 3) == 0) {
                part[BM * D + lrow] = m[r];
                part[BM * D + BM + lrow] = ls;
            }
        }
    }
}

constexpr int STAGES = 3;

template <int D, int MODE, int PAIR, int OPT>
int resident_blocks() {
    constexpr int bytes = smem_bytes<D, STAGES>();
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(attention_v4_kernel<D, STAGES, MODE, PAIR, OPT>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, attention_v4_kernel<D, STAGES, MODE, PAIR, OPT>, THREADS, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <int D, int MODE, int PAIR, int OPT>
void launch_mode(const bf16* Q, const bf16* K, const bf16* V, bf16* O, int B, int H_q, int H_kv,
                 int S_q, int S_kv, float scale_log2, bool causal, cudaStream_t stream) {
    constexpr int bytes = smem_bytes<D, STAGES>();
    const int resident = resident_blocks<D, MODE, PAIR, OPT>();
    // [B*H][S][D] maps with boxes of 64 columns (one 128-byte swizzle span): 128 rows for Q,
    // 64 for K and V. Rows past S_q / S_kv are zero-filled by the copy engine.
    const uint64_t BHq = static_cast<uint64_t>(B) * H_q;
    const uint64_t BHkv = static_cast<uint64_t>(B) * H_kv;  // K/V heads, fewer under GQA
    const CUtensorMap tmQ =
        make_tensor_map_3d_bf16(Q, D, S_q, BHq, BOX_COLS, BM, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap tmK =
        make_tensor_map_3d_bf16(K, D, S_kv, BHkv, BOX_COLS, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap tmV =
        make_tensor_map_3d_bf16(V, D, S_kv, BHkv, BOX_COLS, BN, CU_TENSOR_MAP_SWIZZLE_128B);

    Sched s = v2::whole_tiles<WARPS>(B, H_q, H_kv, S_q);
    v2::split_tail_tiles<D, WARPS>(s, resident, S_kv, causal);
    attention_v4_kernel<D, STAGES, MODE, PAIR, OPT><<<v2::grid_blocks(s), THREADS, bytes, stream>>>(
        tmQ, tmK, tmV, O, S_q, S_kv, scale_log2, causal ? 1 : 0, s);
    if (s.split > 1) v2::launch_combine<D, WARPS>(O, S_q, causal, s, stream);
}

template <int D>
void launch(const bf16* Q, const bf16* K, const bf16* V, bf16* O, int B, int H_q, int H_kv, int S_q,
            int S_kv, float scale_log2, bool causal, cudaStream_t stream) {
#ifdef SPARK_ATTN_V4_EXPERIMENTS
    // SPARK_ATTN_V4_MODE = MODE + 10 * PAIR + 100 * OPT (see the kernel).
    static int mode = -1;
    if (mode < 0) {
        mode = 0;
        if (const char* e = std::getenv("SPARK_ATTN_V4_MODE")) mode = std::atoi(e);
    }
    switch (mode) {
        case 1:
            launch_mode<D, 1, 0, 0>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal,
                                    stream);
            return;
        case 11:
            launch_mode<D, 1, 1, 0>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal,
                                    stream);
            return;
        case 2:
            launch_mode<D, 2, 0, 0>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal,
                                    stream);
            return;
        case 12:
            launch_mode<D, 2, 1, 0>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal,
                                    stream);
            return;
        case 100:
            launch_mode<D, 0, 0, 1>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal,
                                    stream);
            return;
        case 200:
            launch_mode<D, 0, 0, 2>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal,
                                    stream);
            return;
        case 300:
            launch_mode<D, 0, 0, 3>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal,
                                    stream);
            return;
        default:
            break;
    }
#endif
    launch_mode<D, 0, 0, 0>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, stream);
}

}  // namespace v4

// ---------------------------------------------------------------------------------------
// Variant 5: variant 4's tile and pipeline on a persistent grid. One block per SM walks a queue
// of work items (a (b, h, q-tile) tile, or one key slice of a tail tile) in the heaviest-first
// causal order; the producer lane takes the next item from a global counter, publishes it to
// the consumer warps through a two-deep ring of int4 slots in shared memory (full / empty
// mbarrier pair per slot, the hgemm v6 pattern), and issues its Q box and K/V stages as the
// next loads of one running sequence: the Q tile of item i+1 is in flight while the warps run
// the last two K/V tiles of item i, and its first K/V tiles while they run the epilogue. Load
// counters run on across items (load u lives in stage u % STAGES, its u / STAGES-th use), so
// the barrier init, the tensor-map prefetch and the wait for a first box are paid once per
// block instead of once per tile: 1,028 times on the causal 4096 shape became 170.
//
// The producer is lane 0 of a ninth warp (PWARP = 1, the shipped configuration: it issues the
// moment a stage frees and no compute warp ever waits on its behalf, at the price of a
// 168-register cap, three warps sharing one scheduler's 16 K registers) or lane 0 of warp 0
// (PWARP = 0, variant 4's arrangement: one load issued before each load consumed, so the
// pipeline holds STAGES loads including the one being read, and warp 0 waits for the slowest
// warp's release before each issue; 1.6% slower, docs/design/attention.md). A block's last
// grab of the counter fails, it counts itself done, and the last block done resets the counter
// for the next launch.
//
// Under the causal mask a warp skips the second diagonal tile when its 64 keys all follow the
// warp's 16 rows (every p is 0): in variant 4 that freed a pipe the warp's scheduler partner
// could not fill alone, here the warp goes on to the next item's Q K^T instead.
//
// The tail split of variant 3 is kept as an option: the last `tiles mod resident` items of the
// queue become `split` key slices each, merged by the combine kernel. With a queue the causal
// shapes do not need it (the lightest tiles come last and the greedy order evens the blocks out
// to within one small tile); the non-causal ones still do (1,024 equal tiles on 170 blocks is
// 6.02 waves). The default is decided per mask from the measurements in the design doc.
// ---------------------------------------------------------------------------------------
namespace v5 {

using v2::BN;
using v2::Sched;
using v4::BM;
using v4::BOX_COLS;
using v4::box_off;
using v4::stage_bytes;
using v4::tile_bytes;

constexpr int CONSUMER_WARPS = 8, CONSUMERS = 32 * CONSUMER_WARPS;
constexpr int ITEM_DEPTH = 2;  // items the producer may publish ahead of the consumers
constexpr int STAGES = 3;

// Stages, then the item ring and the barriers in a final KB (the ring is 32 bytes, the
// barriers 8 each; the stages need the 1 KB alignment the swizzle keys on).
template <int D>
constexpr int smem_bytes() {
    return STAGES * stage_bytes<D>() + 1024;
}

// Work item w, decoded as variant 4 decodes blockIdx.x: items [0, dp_tiles) are whole tiles,
// the rest are the `split` slices of each tail tile in turn.
__device__ __forceinline__ void decode_item(int w, const Sched& sched, int& tile, int& slice,
                                            int& split) {
    if (w < sched.dp_tiles) {
        tile = w;
        slice = 0;
        split = 1;
    } else {
        const int r = w - sched.dp_tiles;
        tile = sched.dp_tiles + r / sched.split;
        slice = r % sched.split;
        split = sched.split;
    }
}

template <int D, bool PWARP>
__global__ void __launch_bounds__(CONSUMERS + (PWARP ? 32 : 0), 1)
    attention_v5_kernel(const __grid_constant__ CUtensorMap tmQ,
                        const __grid_constant__ CUtensorMap tmK,
                        const __grid_constant__ CUtensorMap tmV, bf16* __restrict__ O, int S_q,
                        int S_kv, float scale_log2, int causal, Sched sched, int n_items,
                        unsigned* __restrict__ queue, int skip_dead) {
    constexpr int KT = D / 16;   // k16 steps of Q K^T
    constexpr int DT = D / 8;    // n8 tiles of O
    constexpr int NT = BN / 8;   // n8 tiles of S
    constexpr int PT = BN / 16;  // k16 steps of P V
    constexpr int BOXES = D / BOX_COLS;
    constexpr int STAGE = stage_bytes<D>();
    static_assert(BM * D * 2 <= STAGE, "the Q tile is staged through a K/V stage");

    extern __shared__ __align__(1024) unsigned char smem[];
    int4* items = reinterpret_cast<int4*>(smem + STAGES * STAGE);
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(items + ITEM_DEPTH);
    uint64_t* empty_bar = full_bar + STAGES;
    uint64_t* item_full = empty_bar + STAGES;
    uint64_t* item_empty = item_full + ITEM_DEPTH;
    if (smem_u32(smem) % 1024 != 0) __trap();

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full_bar[s], 1);                // the producer's arrive.expect_tx
            mbar_init(&empty_bar[s], CONSUMER_WARPS);  // one arrive per consumer warp
        }
#pragma unroll
        for (int i = 0; i < ITEM_DEPTH; ++i) {
            mbar_init(&item_full[i], 1);                // the producer's publish
            mbar_init(&item_empty[i], CONSUMER_WARPS);  // one arrive per warp once it has read
        }
        fence_mbar_init();
    }
    __syncthreads();  // the only block-wide barrier in the kernel

    // The rows and keys of an item: Q rows q0 .. q0+BM-1 of head bh, KV tiles [tb, te) of the
    // T its rows see (all of them unless the tile is split), from K/V head bkv.
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

    // ---- producer state: lives on the producing lane, the rest of the block never reads it.
    unsigned g_load = 0;  // loads issued so far, across items
    unsigned n_pub = 0;   // items published so far
    int p_w = -1, p_bh = 0, p_bkv = 0, p_q0 = 0, p_tb = 0, p_te = 0, p_next = 0;
    bool p_done = false;
    // Ring slot r serves items r, r + DEPTH, ...; its n-th use waits for the consumers' release
    // of the (n-1)-th, exactly like a stage.
    auto publish = [&](int w) {
        const int r = n_pub % ITEM_DEPTH;
        const unsigned use = n_pub / ITEM_DEPTH;
        if (use > 0) mbar_wait(&item_empty[r], (use - 1) & 1);
        items[r] = make_int4(w, 0, 0, 0);
        mbar_arrive(&item_full[r]);
        ++n_pub;
    };
    // Issues the next load of the sequence: the Q box of a new item (taken from the queue and
    // published first) or the K/V boxes of the current item's next tile. False once the queue
    // is empty and the end marker is published.
    auto produce_one = [&]() -> bool {
        if (p_done) return false;
        const int s = g_load % STAGES;
        const unsigned use = g_load / STAGES;
        unsigned char* dst = smem + s * STAGE;
        if (p_w < 0 || p_next == p_te) {
            const int w = static_cast<int>(atomicAdd(queue, 1u));
            if (w >= n_items) {
                publish(-1);
                p_done = true;
                return false;
            }
            publish(w);
            int tile, slice, split;
            decode_item(w, sched, tile, slice, split);
            geometry(tile, slice, split, p_bh, p_bkv, p_q0, p_tb, p_te);
            p_w = w;
            p_next = p_tb;
            if (use > 0) mbar_wait(&empty_bar[s], (use - 1) & 1);
            mbar_arrive_expect_tx(&full_bar[s], BM * D * 2);
#pragma unroll
            for (int b = 0; b < BOXES; ++b)
                tma_load_3d(dst + b * BM * BOX_COLS * 2, &tmQ, &full_bar[s], b * BOX_COLS, p_q0,
                            p_bh);
        } else {
            const int j0 = p_next * BN;
            ++p_next;
            if (use > 0) mbar_wait(&empty_bar[s], (use - 1) & 1);
            mbar_arrive_expect_tx(&full_bar[s], STAGE);
#pragma unroll
            for (int b = 0; b < BOXES; ++b) {
                tma_load_3d(dst + b * BN * BOX_COLS * 2, &tmK, &full_bar[s], b * BOX_COLS, j0,
                            p_bkv);
                tma_load_3d(dst + tile_bytes<D>() + b * BN * BOX_COLS * 2, &tmV, &full_bar[s],
                            b * BOX_COLS, j0, p_bkv);
            }
        }
        ++g_load;
        return true;
    };
    // Every block makes its final, failing grab before it counts itself done; the last one
    // done resets the counter for the next launch.
    auto finish_queue = [&]() {
        __threadfence();
        if (atomicAdd(queue + 1, 1u) == gridDim.x - 1) {
            queue[0] = 0u;
            queue[1] = 0u;
            __threadfence();
        }
    };

    const bool producer = PWARP ? warp == CONSUMER_WARPS && lane == 0 : tid == 0;
    if (producer) {
        prefetch_tensormap(&tmQ);
        prefetch_tensormap(&tmK);
        prefetch_tensormap(&tmV);
    }
    if constexpr (PWARP) {
        if (warp == CONSUMER_WARPS) {
            if (lane == 0) {
                while (produce_one()) {
                }
                finish_queue();
            }
            return;
        }
    } else {
        // STAGES - 1 loads up front; one more before each load consumed keeps STAGES in the
        // pipeline, the one being read included.
        if (tid == 0) {
#pragma unroll
            for (int u = 0; u < STAGES - 1; ++u) produce_one();
        }
    }

    // ---- consumer warps.
    unsigned g_cons = 0;  // loads consumed so far, across items
    unsigned n_item = 0;  // items taken from the ring so far
    auto stage_of = [&](unsigned u) -> const bf16* {
        return reinterpret_cast<const bf16*>(smem + (u % STAGES) * STAGE);
    };
    auto wait_load = [&](unsigned u) { mbar_wait(&full_bar[u % STAGES], (u / STAGES) & 1); };
    auto release_load = [&](unsigned u) {
        fence_proxy_async_smem();
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty_bar[u % STAGES]);
    };

    unsigned qf[KT][4];
    float o[DT][4];
    float m[2], l[2];
    float sacc[NT][4];
    unsigned pa[PT][4];
    int q0 = 0;

    // S = Q K^T of the tile in `ks`: 16 rows x 64 keys per warp, 8 n8 accumulators.
    auto qk = [&](const bf16* ks) {
#pragma unroll
        for (int nj = 0; nj < NT; ++nj)
#pragma unroll
            for (int e = 0; e < 4; ++e) sacc[nj][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
#pragma unroll
            for (int nj = 0; nj < NT; nj += 2) {
                const int row = nj * 8 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4(r, ks + box_off<BN>(row, 2 * kk + (lane >> 4)));
                const unsigned b0[2] = {r[0], r[2]};
                const unsigned b1[2] = {r[1], r[3]};
                mma_bf16_16816(sacc[nj], qf[kk], b0);
                mma_bf16_16816(sacc[nj + 1], qf[kk], b1);
            }
        }
    };
    // Mask, online softmax on the S accumulators (rows g and g+8, the O rescale included) and
    // P packed as the A fragments of P V: variant 4's code.
    auto softmax = [&](int t) {
        const int kv0 = t * BN;
        if (kv0 + BN > S_kv || (causal && kv0 + BN - 1 > q0)) {  // warp-uniform
#pragma unroll
            for (int nj = 0; nj < NT; ++nj)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int i = q0 + warp * 16 + g + (e >> 1) * 8;
                    const int j = kv0 + nj * 8 + c2 + (e & 1);
                    if (j >= S_kv || (causal && j > i)) sacc[nj][e] = -INFINITY;
                }
        }
        float alpha[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float mx = fmaxf(sacc[0][2 * r], sacc[0][2 * r + 1]);
#pragma unroll
            for (int nj = 1; nj < NT; ++nj)
                mx = fmaxf(mx, fmaxf(sacc[nj][2 * r], sacc[nj][2 * r + 1]));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 2));
            const float m_new = fmaxf(m[r], mx * scale_log2);
            const float m_use = m_new == -INFINITY ? 0.f : m_new;
            alpha[r] = ex2(m[r] - m_use);
            float rs = 0.f;
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                const float p0 = ex2(fmaf(sacc[nj][2 * r], scale_log2, -m_use));
                const float p1 = ex2(fmaf(sacc[nj][2 * r + 1], scale_log2, -m_use));
                sacc[nj][2 * r] = p0;
                sacc[nj][2 * r + 1] = p1;
                rs += p0 + p1;
            }
            l[r] = fmaf(l[r], alpha[r], rs);
            m[r] = m_new;
        }
#pragma unroll
        for (int dj = 0; dj < DT; ++dj) {
            o[dj][0] *= alpha[0];
            o[dj][1] *= alpha[0];
            o[dj][2] *= alpha[1];
            o[dj][3] *= alpha[1];
        }
#pragma unroll
        for (int kt = 0; kt < PT; ++kt) {
            pa[kt][0] = pack_bf16x2(sacc[2 * kt][0], sacc[2 * kt][1]);
            pa[kt][1] = pack_bf16x2(sacc[2 * kt][2], sacc[2 * kt][3]);
            pa[kt][2] = pack_bf16x2(sacc[2 * kt + 1][0], sacc[2 * kt + 1][1]);
            pa[kt][3] = pack_bf16x2(sacc[2 * kt + 1][2], sacc[2 * kt + 1][3]);
        }
    };
    // O += P V with the V tile in `vs`: 16 rows x D per warp, D/8 n8 accumulators.
    auto pv = [&](const bf16* vs) {
#pragma unroll
        for (int kt = 0; kt < PT; ++kt) {
#pragma unroll
            for (int dj = 0; dj < DT; dj += 2) {
                const int row = kt * 16 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4_trans(r, vs + box_off<BN>(row, dj + (lane >> 4)));
                const unsigned b0[2] = {r[0], r[1]};
                const unsigned b1[2] = {r[2], r[3]};
                mma_bf16_16816(o[dj], pa[kt], b0);
                mma_bf16_16816(o[dj + 1], pa[kt], b1);
            }
        }
    };

    for (;;) {
        if constexpr (!PWARP) {
            if (tid == 0) produce_one();
        }
        // The next item from the ring; every lane has its copy before the warp hands the slot
        // back. A negative item ends the block.
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
        bf16* Og = O + static_cast<size_t>(bh) * S_q * D;

        // Q tile into A fragments: qf[kk] covers d = 16kk .. 16kk+15. Then its stage is free.
        wait_load(g_cons);
        {
            const bf16* qs = stage_of(g_cons);
            const int row = warp * 16 + (lane & 15);
#pragma unroll
            for (int kk = 0; kk < KT; ++kk)
                ldmatrix_x4(qf[kk], qs + box_off<BM>(row, 2 * kk + (lane >> 4)));
        }
        release_load(g_cons);
        ++g_cons;

#pragma unroll
        for (int dj = 0; dj < DT; ++dj)
#pragma unroll
            for (int e = 0; e < 4; ++e) o[dj][e] = 0.f;
        m[0] = m[1] = -INFINITY;
        l[0] = l[1] = 0.f;

        for (int t = tb; t < te; ++t, ++g_cons) {
            if constexpr (!PWARP) {
                if (tid == 0) produce_one();
            }
            // Under the causal mask the keys of a tile may all follow every row of this warp
            // (the second diagonal tile for warps 0-3): every p is 0 and the tile is a no-op
            // for the warp, which then starts the next item while the other warps finish.
            const bool dead = skip_dead && causal && t * BN > q0 + warp * 16 + 15;
            wait_load(g_cons);
            if (!dead) {
                qk(stage_of(g_cons));
                softmax(t);
                pv(stage_of(g_cons) + BN * D);
            }
            release_load(g_cons);
        }

        // Epilogue, as variant 4's: normalized bf16 rows from registers, or the unnormalized
        // fp32 rows plus (m, l) of a slice to the workspace.
#pragma unroll
        for (int rr = 0; rr < 2; ++rr) {
            float ls = l[rr];
            ls += __shfl_xor_sync(kFullMask, ls, 1);
            ls += __shfl_xor_sync(kFullMask, ls, 2);
            const int lrow = warp * 16 + g + 8 * rr;
            if (split == 1) {
                const int row = q0 + lrow;
                if (row >= S_q) continue;
                const float inv = 1.f / ls;
                bf16* out = Og + static_cast<size_t>(row) * D + c2;
#pragma unroll
                for (int dj = 0; dj < DT; ++dj)
                    *reinterpret_cast<__nv_bfloat162*>(out + dj * 8) =
                        __floats2bfloat162_rn(o[dj][2 * rr] * inv, o[dj][2 * rr + 1] * inv);
            } else {
                float* part =
                    sched.ws + (static_cast<size_t>(tile - sched.dp_tiles) * split + slice) *
                                   v2::partial_floats<D, CONSUMER_WARPS>();
                float* orow = part + lrow * D + c2;
#pragma unroll
                for (int dj = 0; dj < DT; ++dj)
                    *reinterpret_cast<float2*>(orow + dj * 8) =
                        make_float2(o[dj][2 * rr], o[dj][2 * rr + 1]);
                if ((lane & 3) == 0) {
                    part[BM * D + lrow] = m[rr];
                    part[BM * D + BM + lrow] = ls;
                }
            }
        }
    }
    if constexpr (!PWARP) {
        if (tid == 0) finish_queue();
    }
}

// Knobs, read once, for the tables in docs/design/attention.md: SPARK_ATTN_V5_SPLIT = 0 / 1
// forces the tail split off / on (-1: the rule in launch_cfg), SPARK_ATTN_V5_PWARP = 0 runs the
// producer on lane 0 of warp 0 instead of the ninth warp, SPARK_ATTN_V5_SKIP = 0 keeps every
// warp on the fully masked half of the second diagonal tile.
struct Env {
    int split = -1;
    int pwarp = 1;
    int skip_dead = 1;
};
const Env& env() {
    static Env e;
    static bool read = false;
    if (!read) {
        read = true;
        if (const char* s = std::getenv("SPARK_ATTN_V5_SPLIT")) e.split = std::atoi(s);
        if (const char* s = std::getenv("SPARK_ATTN_V5_PWARP")) e.pwarp = std::atoi(s);
        if (const char* s = std::getenv("SPARK_ATTN_V5_SKIP")) e.skip_dead = std::atoi(s);
    }
    return e;
}

// The queue counters, two unsigned per stream: a launch on one stream must not share its
// counter with a launch that may be running on another. Slots are handed out per stream
// handle on first sight and zeroed once; the kernel leaves them zero.
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

template <int D, bool PWARP>
int resident_blocks() {
    constexpr int bytes = smem_bytes<D>();
    constexpr int threads = CONSUMERS + (PWARP ? 32 : 0);
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(attention_v5_kernel<D, PWARP>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, attention_v5_kernel<D, PWARP>, threads, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

// `split_mode`: 0 / 1 forces the tail split off / on, -1 picks it: on for the non-causal
// shapes and for any shape with fewer tiles than resident blocks (the slices are the only way
// to give the idle SMs work); off under the causal mask once the queue has a full wave to
// balance.
template <int D, bool PWARP>
void launch_cfg(const bf16* Q, const bf16* K, const bf16* V, bf16* O, int B, int H_q, int H_kv,
                int S_q, int S_kv, float scale_log2, bool causal, int split_mode,
                cudaStream_t stream) {
    constexpr int bytes = smem_bytes<D>();
    constexpr int threads = CONSUMERS + (PWARP ? 32 : 0);
    const int resident = resident_blocks<D, PWARP>();
    const uint64_t BHq = static_cast<uint64_t>(B) * H_q;
    const uint64_t BHkv = static_cast<uint64_t>(B) * H_kv;
    const CUtensorMap tmQ =
        make_tensor_map_3d_bf16(Q, D, S_q, BHq, BOX_COLS, BM, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap tmK =
        make_tensor_map_3d_bf16(K, D, S_kv, BHkv, BOX_COLS, BN, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap tmV =
        make_tensor_map_3d_bf16(V, D, S_kv, BHkv, BOX_COLS, BN, CU_TENSOR_MAP_SWIZZLE_128B);

    Sched s = v2::whole_tiles<CONSUMER_WARPS>(B, H_q, H_kv, S_q);
    const bool split_tail = split_mode < 0 ? !causal || s.dp_tiles < resident : split_mode != 0;
    if (split_tail) v2::split_tail_tiles<D, CONSUMER_WARPS>(s, resident, S_kv, causal);
    const int n_items = v2::grid_blocks(s);
    const int grid = std::min(resident, n_items);
    attention_v5_kernel<D, PWARP>
        <<<grid, threads, bytes, stream>>>(tmQ, tmK, tmV, O, S_q, S_kv, scale_log2, causal ? 1 : 0,
                                           s, n_items, queue_for(stream), env().skip_dead);
    if (s.split > 1) v2::launch_combine<D, CONSUMER_WARPS>(O, S_q, causal, s, stream);
}

template <int D>
void launch(const bf16* Q, const bf16* K, const bf16* V, bf16* O, int B, int H_q, int H_kv, int S_q,
            int S_kv, float scale_log2, bool causal, cudaStream_t stream) {
    if (env().pwarp)
        launch_cfg<D, true>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, env().split,
                            stream);
    else
        launch_cfg<D, false>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, env().split,
                             stream);
}

}  // namespace v5

template <int D>
void launch_v1(const bf16* Q, const bf16* K, const bf16* V, bf16* O, int B, int H_q, int H_kv,
               int S_q, int S_kv, float scale_log2, bool causal, cudaStream_t stream) {
    constexpr int bytes = v1::smem_bytes<D>();
    static bool opted_in = false;
    if (!opted_in) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(v1::attention_v1_kernel<D>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        opted_in = true;
    }
    const dim3 grid(B * H_q, cdiv(S_q, v1::BM));
    v1::attention_v1_kernel<D><<<grid, v1::THREADS, bytes, stream>>>(
        Q, K, V, O, H_q, H_kv, S_q, S_kv, scale_log2, causal ? 1 : 0);
}

template <int D>
void launch_v0(const bf16* Q, const bf16* K, const bf16* V, bf16* O, int B, int H_q, int H_kv,
               int S_q, int S_kv, float scale_log2, bool causal, cudaStream_t stream) {
    const dim3 grid(cdiv(S_q, v0::THREADS / 32), B * H_q);
    v0::attention_v0_kernel<D><<<grid, v0::THREADS, 0, stream>>>(Q, K, V, O, H_q, H_kv, S_q, S_kv,
                                                                 scale_log2, causal ? 1 : 0);
}

// SPARK_ATTENTION_DECODE=0 keeps decode shapes on the 64-row tile of variant 3 (the path
// before attention_decode.cu existed), for the tile comparison in the design doc.
bool use_decode_kernel() {
    static int v = -1;
    if (v < 0) {
        v = 1;
        if (const char* e = std::getenv("SPARK_ATTENTION_DECODE")) v = std::atoi(e) != 0 ? 1 : 0;
    }
    return v == 1;
}

// Variant 3's dispatch by shape: the flash-decoding kernel when the query rows sharing a
// K/V head fit its 16-row tile, the 64-row tile for other short queries, else 128 rows.
template <int D>
void launch_v3(const bf16* Q, const bf16* K, const bf16* V, bf16* O, int B, int H_q, int H_kv,
               int S_q, int S_kv, float scale_log2, bool causal, cudaStream_t stream) {
    if (use_decode_kernel() && attn::decode_fits(H_q, H_kv, S_q)) {
        attn::decode_launch(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, D, scale_log2, causal, 0, stream);
    } else if (S_q <= 64) {
        v2::launch<D, 4>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, true, stream);
    } else {
        v2::launch<D, 8>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, true, stream);
    }
}

}  // namespace

int attention_num_variants() {
    return 6;
}

bool attention_supports(int S_q, int S_kv, int D, int variant) {
    if (variant < 0 || variant >= attention_num_variants()) return false;
    if (S_q <= 0 || S_kv <= 0) return false;
    return D == 64 || D == 128;
}

void attention_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                    __nv_bfloat16* O, int B, int H_q, int H_kv, int S_q, int S_kv, int D,
                    bool causal, int variant, cudaStream_t stream) {
    SPARK_REQUIRE(Q != nullptr && K != nullptr && V != nullptr && O != nullptr,
                  "attention: null pointer");
    SPARK_REQUIRE(B > 0 && H_q > 0 && H_kv > 0 && S_q > 0 && S_kv > 0,
                  "attention: B, H_q, H_kv, S_q, S_kv must be positive");
    SPARK_REQUIRE(H_q % H_kv == 0, "attention: H_q must be a multiple of H_kv");
    SPARK_REQUIRE(D == 64 || D == 128, "attention: D must be 64 or 128");
    SPARK_REQUIRE(variant >= 0 && variant < attention_num_variants(), "attention: unknown variant");
    SPARK_REQUIRE(is_aligned16(Q) && is_aligned16(K) && is_aligned16(V) && is_aligned16(O),
                  "attention: Q, K, V, O must be 16-byte aligned");
    SPARK_REQUIRE(static_cast<int64_t>(B) * H_q * std::max(S_q, S_kv) * D < (int64_t{1} << 40),
                  "attention: tensor too large");

    const float scale_log2 = kLog2e / std::sqrt(static_cast<float>(D));
    switch (variant) {
        case 0:
            if (D == 64)
                launch_v0<64>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, stream);
            else
                launch_v0<128>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, stream);
            break;
        case 1:
            if (D == 64)
                launch_v1<64>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, stream);
            else
                launch_v1<128>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, stream);
            break;
        case 2:
            if (D == 64)
                v2::launch<64, 8>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, false,
                                  stream);
            else
                v2::launch<128, 8>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, false,
                                   stream);
            break;
        case 3:
            if (D == 64)
                launch_v3<64>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, stream);
            else
                launch_v3<128>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, stream);
            break;
        case 4:
            // Short queries take variant 3's paths (the flash-decoding kernel or the 64-row
            // tile); the TMA kernel takes every 128-row tile.
            if (S_q <= 64) {
                if (D == 64)
                    launch_v3<64>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, stream);
                else
                    launch_v3<128>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, stream);
            } else {
                if (D == 64)
                    v4::launch<64>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, stream);
                else
                    v4::launch<128>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal,
                                    stream);
            }
            break;
        case 5:
            // Variant 4's routing for short queries; the persistent kernel for the rest.
            if (S_q <= 64) {
                if (D == 64)
                    launch_v3<64>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, stream);
                else
                    launch_v3<128>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, stream);
            } else {
                if (D == 64)
                    v5::launch<64>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, stream);
                else
                    v5::launch<128>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal,
                                    stream);
            }
            break;
        default:
            break;
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
