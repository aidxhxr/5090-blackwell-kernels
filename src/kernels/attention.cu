// Fused scaled-dot-product attention forward for Blackwell sm_12x: RTX 5090 (sm_120), GB10.
//
//   O[b,h] = softmax(Q[b,h] K[b,h]^T / sqrt(D)) V[b,h]     Q, K, V, O = [B, H, S, D], row-major
//
// bf16 in and out, fp32 for every score, exponential and accumulator. One head at a time, no
// sharing between heads (MHA). The causal mask hides key j from query i when j > i (the
// top-left alignment torch's is_causal uses). Every variant runs the online softmax: a running
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
//              split along the keys over the idle SMs and merged by a combine kernel, and a
//              64-row tile takes S_q <= 64 (decode) so a single query does not pay for 128.

#include <algorithm>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

using bf16 = __nv_bfloat16;

constexpr float kLog2e = 1.4426950408889634f;

// 2^x on the MUFU pipe. ex2.approx takes -inf to +0, which is what a masked score needs.
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

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

// Physical 16-byte chunk of logical (row, chunk) in a tile whose rows are 128 or 256 bytes:
// the eight rows an ldmatrix (or eight lanes reading eight rows) touch land in eight bank groups.
__device__ __forceinline__ int swz(int row, int chunk) {
    return chunk ^ (row & 7);
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
                        const bf16* __restrict__ V, bf16* __restrict__ O, int S_q, int S_kv,
                        float scale_log2, int causal) {
    constexpr int VEC = D / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (THREADS / 32) + (threadIdx.x >> 5);
    const int bh = blockIdx.y;
    if (row >= S_q) return;
    const bf16* q = Q + (static_cast<size_t>(bh) * S_q + row) * D + lane * VEC;
    const bf16* k = K + static_cast<size_t>(bh) * S_kv * D + lane * VEC;
    const bf16* v = V + static_cast<size_t>(bh) * S_kv * D + lane * VEC;

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
                        const bf16* __restrict__ V, bf16* __restrict__ O, int S_q, int S_kv,
                        float scale_log2, int causal) {
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
    const bf16* Qg = Q + static_cast<size_t>(bh) * S_q * D;
    const bf16* Kg = K + static_cast<size_t>(bh) * S_kv * D;
    const bf16* Vg = V + static_cast<size_t>(bh) * S_kv * D;
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
    int bh_count;  // B * H
    int q_tiles;   // ceil(S_q / BM)
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
    const bf16* Qg = Q + static_cast<size_t>(bh) * S_q * D;
    const bf16* Kg = K + static_cast<size_t>(bh) * S_kv * D;
    const bf16* Vg = V + static_cast<size_t>(bh) * S_kv * D;
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

// `split_tail`: variant 3. Variant 2 runs every tile whole on the 8-warp tile.
template <int D, int WARPS>
void launch(const bf16* Q, const bf16* K, const bf16* V, bf16* O, int B, int H, int S_q, int S_kv,
            float scale_log2, bool causal, bool split_tail, cudaStream_t stream) {
    constexpr int BM = 16 * WARPS;
    constexpr int bytes = smem_bytes<D, STAGES>();
    const int resident = resident_blocks<D, WARPS>();

    Sched s;
    s.bh_count = B * H;
    s.q_tiles = cdiv(S_q, BM);
    const int tiles = s.bh_count * s.q_tiles;
    s.dp_tiles = tiles;
    s.split = 1;
    s.ws = nullptr;
    if (split_tail) {
        // The tail tiles are the last in the heaviest-first order, so the last one has the
        // fewest KV tiles; every slice must get at least one.
        const int tail = tiles % resident;
        const int q_last = causal ? 0 : s.q_tiles - 1;  // rank tiles - 1
        const int T_last = cdiv(causal ? std::min(S_kv, q_last * BM + BM) : S_kv, BN);
        const int split = tail > 0 ? std::min(T_last, resident / tail) : 1;
        if (split > 1) {
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
    }
    const int grid = s.dp_tiles + (tiles - s.dp_tiles) * s.split;
    attention_v2_kernel<D, WARPS, STAGES>
        <<<grid, WARPS * 32, bytes, stream>>>(Q, K, V, O, S_q, S_kv, scale_log2, causal ? 1 : 0, s);
    if (s.split > 1) {
        constexpr int per_tile = BM * (D / 8);
        const dim3 cgrid(cdiv(per_tile, 256), tiles - s.dp_tiles);
        attention_combine_kernel<D, WARPS><<<cgrid, 256, 0, stream>>>(O, S_q, causal ? 1 : 0, s);
    }
}

}  // namespace v2

template <int D>
void launch_v1(const bf16* Q, const bf16* K, const bf16* V, bf16* O, int B, int H, int S_q,
               int S_kv, float scale_log2, bool causal, cudaStream_t stream) {
    constexpr int bytes = v1::smem_bytes<D>();
    static bool opted_in = false;
    if (!opted_in) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(v1::attention_v1_kernel<D>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        opted_in = true;
    }
    const dim3 grid(B * H, cdiv(S_q, v1::BM));
    v1::attention_v1_kernel<D>
        <<<grid, v1::THREADS, bytes, stream>>>(Q, K, V, O, S_q, S_kv, scale_log2, causal ? 1 : 0);
}

template <int D>
void launch_v0(const bf16* Q, const bf16* K, const bf16* V, bf16* O, int B, int H, int S_q,
               int S_kv, float scale_log2, bool causal, cudaStream_t stream) {
    const dim3 grid(cdiv(S_q, v0::THREADS / 32), B * H);
    v0::attention_v0_kernel<D>
        <<<grid, v0::THREADS, 0, stream>>>(Q, K, V, O, S_q, S_kv, scale_log2, causal ? 1 : 0);
}

}  // namespace

int attention_num_variants() {
    return 4;
}

bool attention_supports(int S_q, int S_kv, int D, int variant) {
    if (variant < 0 || variant >= attention_num_variants()) return false;
    if (S_q <= 0 || S_kv <= 0) return false;
    return D == 64 || D == 128;
}

void attention_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                    __nv_bfloat16* O, int B, int H, int S_q, int S_kv, int D, bool causal,
                    int variant, cudaStream_t stream) {
    SPARK_REQUIRE(Q != nullptr && K != nullptr && V != nullptr && O != nullptr,
                  "attention: null pointer");
    SPARK_REQUIRE(B > 0 && H > 0 && S_q > 0 && S_kv > 0,
                  "attention: B, H, S_q, S_kv must be positive");
    SPARK_REQUIRE(D == 64 || D == 128, "attention: D must be 64 or 128");
    SPARK_REQUIRE(variant >= 0 && variant < attention_num_variants(), "attention: unknown variant");
    SPARK_REQUIRE(is_aligned16(Q) && is_aligned16(K) && is_aligned16(V) && is_aligned16(O),
                  "attention: Q, K, V, O must be 16-byte aligned");
    SPARK_REQUIRE(static_cast<int64_t>(B) * H * std::max(S_q, S_kv) * D < (int64_t{1} << 40),
                  "attention: tensor too large");

    const float scale_log2 = kLog2e / std::sqrt(static_cast<float>(D));
    switch (variant) {
        case 0:
            if (D == 64)
                launch_v0<64>(Q, K, V, O, B, H, S_q, S_kv, scale_log2, causal, stream);
            else
                launch_v0<128>(Q, K, V, O, B, H, S_q, S_kv, scale_log2, causal, stream);
            break;
        case 1:
            if (D == 64)
                launch_v1<64>(Q, K, V, O, B, H, S_q, S_kv, scale_log2, causal, stream);
            else
                launch_v1<128>(Q, K, V, O, B, H, S_q, S_kv, scale_log2, causal, stream);
            break;
        case 2:
            if (D == 64)
                v2::launch<64, 8>(Q, K, V, O, B, H, S_q, S_kv, scale_log2, causal, false, stream);
            else
                v2::launch<128, 8>(Q, K, V, O, B, H, S_q, S_kv, scale_log2, causal, false, stream);
            break;
        case 3:
            // A decode step (S_q <= 64) runs the 4-warp, 64-row tile: half the tensor-core
            // work per KV tile on rows that are mostly zero-filled anyway.
            if (S_q <= 64) {
                if (D == 64)
                    v2::launch<64, 4>(Q, K, V, O, B, H, S_q, S_kv, scale_log2, causal, true,
                                      stream);
                else
                    v2::launch<128, 4>(Q, K, V, O, B, H, S_q, S_kv, scale_log2, causal, true,
                                       stream);
            } else {
                if (D == 64)
                    v2::launch<64, 8>(Q, K, V, O, B, H, S_q, S_kv, scale_log2, causal, true,
                                      stream);
                else
                    v2::launch<128, 8>(Q, K, V, O, B, H, S_q, S_kv, scale_log2, causal, true,
                                       stream);
            }
            break;
        default:
            break;
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
