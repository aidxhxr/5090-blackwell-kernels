// Fused attention backward for Blackwell sm_12x: RTX 5090 (sm_120), GB10.
//
// Given the forward's inputs Q, K, V, its output O, the upstream gradient dO and the per-row
// log-sum-exp L the forward saved (natural log, fp32), per (batch, query head):
//
//   P  = exp(Q K^T / sqrt(D) - L)        the forward's softmax, recomputed tile by tile
//   dV = P^T dO
//   dP = dO V^T
//   dS = P * (dP - Dv),   Dv_i = dO_i . O_i  (= sum_j P_ij dP_ij)
//   dQ = dS K / sqrt(D)
//   dK = dS^T Q / sqrt(D)
//
// with dK and dV of a K/V head summed over the H_q / H_kv query heads that read it (GQA).
// Five matrix products of S_q x S_kv x D against the forward's two: the backward is 2.5x the
// forward's FLOPs. As in the forward, P is formed as 2^(s * log2(e)/sqrt(D) - L log2(e)), one
// FFMA and one MUFU ex2 per score.
//
// Every variant first runs a preprocess kernel: Dv per query row, L converted to the log2
// domain, both stored as one float2 per row in a padded workspace (rows past S_q are {0, 0}:
// a zero-filled Q row and dO row then give P = 1 against dO = 0 and dP = 0, which adds nothing
// to any gradient), and the fp32 dQ buffer zeroed when the variant accumulates into it.
//
//   variant 0: one warp per query row for dQ, one warp per key row for dK and dV, shuffle
//              reductions per score. Reference.
//   variant 1: FlashAttention-2's backward on mma.sync: a block owns 64 keys of a K/V head
//              (4 warps x 16 keys), keeps its dK and dV rows in registers, and walks the 64-row
//              Q/dO tiles of every query head of the group. Each warp computes S^T and dP^T
//              for its keys (keys on the mma rows, so P^T and dS^T are A operands straight from
//              the accumulators, like P in the forward), then dV += P^T dO and dK += dS^T Q.
//              dQ needs a sum over the block's keys: dS^T goes to shared memory and the warps
//              compute the tile's dQ = dS K and add it to an fp32 buffer with vector atomics.
//              Loads are cp.async, waited for before the tile is used.
//   variant 2: the same algorithm sized for sm_120's 99 KB: 128 keys per block (8 warps),
//              V's A fragments in registers (V is only ever the A operand of its own warp's
//              dP^T), which frees 32 KB for a double-buffered Q/dO tile of 32 rows (D = 128)
//              or 64 rows (D = 64), so the next tile's loads overlap this tile's products;
//              dS^T double-buffered so the dQ product of tile t - 1 runs after tile t's one
//              barrier (one barrier per tile, not two); under the causal mask a warp whose 16
//              keys all follow the tile's rows skips it (it only zeroes its dS rows); and the
//              grid is split in two levels, every key tile into s1 ranges of its query tiles
//              and the last partial wave again into s2, s1 picked by simulating the block
//              scheduler, the slices adding fp32 partials of dK and dV a small kernel converts.
//   variant 3: variant 2's tile and products without the block barrier: Q, dO and the (L, Dv)
//              rows arrive by TMA into three full / empty mbarrier stages issued by one lane,
//              and the two dS^T buffers are handed between the warps by their own full / empty
//              mbarriers, so the warps drift apart instead of starting every tile in phase.
//
// `deterministic` swaps the dQ atomics for a separate query-tile-outer kernel (forward
// variant 2's structure with Q and dO fragments in registers, K and V streamed) that
// recomputes S and dP for dQ, and runs the key-tile kernel without its dQ product: seven
// matrix products instead of five, no atomics, bitwise reproducible.

#include <algorithm>
#include <array>
#include <cstdlib>
#include <functional>
#include <map>
#include <queue>
#include <type_traits>
#include <vector>

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

// The workspace pads each head's (L, Dv) rows to a multiple of this, so a Q tile of any size
// that divides it never reads past its head.
constexpr int kRowPad = 128;

// ---------------------------------------------------------------------------------------
// Small helpers.
// ---------------------------------------------------------------------------------------
template <int N>
__device__ __forceinline__ void load_bf16(const bf16* p, float (&f)[N]) {
    static_assert(N == 2 || N == 4, "");
    if constexpr (N == 4) {
        const uint2 u = *reinterpret_cast<const uint2*>(p);
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.x));
        const float2 b = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&u.y));
        f[0] = a.x;
        f[1] = a.y;
        f[2] = b.x;
        f[3] = b.y;
    } else {
        const float2 a = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(p));
        f[0] = a.x;
        f[1] = a.y;
    }
}

template <int N>
__device__ __forceinline__ void store_bf16(bf16* p, const float (&f)[N], float scale) {
    static_assert(N == 2 || N == 4, "");
    if constexpr (N == 4) {
        uint2 u;
        u.x = pack_bf16x2(f[0] * scale, f[1] * scale);
        u.y = pack_bf16x2(f[2] * scale, f[3] * scale);
        *reinterpret_cast<uint2*>(p) = u;
    } else {
        *reinterpret_cast<unsigned*>(p) = pack_bf16x2(f[0] * scale, f[1] * scale);
    }
}

// fp32 vector atomic add (RED.E.ADD.F32x2 on sm_90+; the return value is unused).
__device__ __forceinline__ void red_add2(float* p, float a, float b) {
    atomicAdd(reinterpret_cast<float2*>(p), make_float2(a, b));
}

// ---------------------------------------------------------------------------------------
// Preprocess: ld[bh * S_pad + i] = {L_i log2(e), dO_i . O_i}, {0, 0} for the padding rows,
// and the fp32 dQ rows zeroed when dq_acc is not null. One warp per padded row. Zeroing here
// rather than after the previous call has a side effect worth keeping: the zeroed lines are in
// L2 when the key-tile kernel's atomics arrive (moving the zeroing into the dQ conversion of
// the previous call measured 4% slower at 4096 tokens).
// ---------------------------------------------------------------------------------------
template <int D>
__global__ void bwd_preprocess_kernel(const bf16* __restrict__ O, const bf16* __restrict__ dO,
                                      const float* __restrict__ lse, float2* __restrict__ ld,
                                      float* __restrict__ dq_acc, int S_q, int S_pad,
                                      int rows_pad) {
    constexpr int VEC = D / 32;
    const int lane = threadIdx.x & 31;
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (row >= rows_pad) return;
    const int bh = row / S_pad, i = row - bh * S_pad;
    if (i >= S_q) {
        if (lane == 0) ld[row] = make_float2(0.f, 0.f);
        return;
    }
    const size_t r = static_cast<size_t>(bh) * S_q + i;
    float o[VEC], d[VEC];
    load_bf16<VEC>(O + r * D + lane * VEC, o);
    load_bf16<VEC>(dO + r * D + lane * VEC, d);
    float dot = 0.f;
#pragma unroll
    for (int e = 0; e < VEC; ++e) dot = fmaf(o[e], d[e], dot);
    dot = warp_reduce_sum(dot);
    if (lane == 0) ld[row] = make_float2(lse[r] * kLog2e, dot);
    if (dq_acc != nullptr) {
#pragma unroll
        for (int e = 0; e < VEC; ++e) dq_acc[r * D + lane * VEC + e] = 0.f;
    }
}

// dQ = acc * scale in bf16, eight elements per thread.
__global__ void bwd_convert_kernel(const float* __restrict__ acc, bf16* __restrict__ out, size_t n8,
                                   float scale) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n8;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float4 a = reinterpret_cast<const float4*>(acc)[2 * i];
        const float4 b = reinterpret_cast<const float4*>(acc)[2 * i + 1];
        uint4 u;
        u.x = pack_bf16x2(a.x * scale, a.y * scale);
        u.y = pack_bf16x2(a.z * scale, a.w * scale);
        u.z = pack_bf16x2(b.x * scale, b.y * scale);
        u.w = pack_bf16x2(b.z * scale, b.w * scale);
        reinterpret_cast<uint4*>(out)[i] = u;
    }
}

// ---------------------------------------------------------------------------------------
// Variant 0: scalar reference kernels. Lane l owns columns l*VEC .. l*VEC+VEC-1.
// ---------------------------------------------------------------------------------------
namespace v0 {

constexpr int THREADS = 128;  // 4 rows per block

// dQ_i = scale * sum_j P_ij (dP_ij - Dv_i) k_j over the keys row i sees.
template <int D>
__global__ void __launch_bounds__(THREADS)
    bwd_v0_dq_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                     const bf16* __restrict__ V, const bf16* __restrict__ dO,
                     const float2* __restrict__ ld, bf16* __restrict__ dQ, int H_q, int H_kv,
                     int S_q, int S_kv, int S_pad, float scale_log2, float scale, int causal) {
    constexpr int VEC = D / 32;
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (THREADS / 32) + (threadIdx.x >> 5);
    const int bh = blockIdx.y;
    if (row >= S_q) return;
    const int bkv = kv_index(bh, H_q, H_kv);
    const size_t r = static_cast<size_t>(bh) * S_q + row;
    float q[VEC], d[VEC], dq[VEC];
    load_bf16<VEC>(Q + r * D + lane * VEC, q);
    load_bf16<VEC>(dO + r * D + lane * VEC, d);
#pragma unroll
    for (int e = 0; e < VEC; ++e) dq[e] = 0.f;
    const float2 l = ld[static_cast<size_t>(bh) * S_pad + row];
    const bf16* k = K + static_cast<size_t>(bkv) * S_kv * D + lane * VEC;
    const bf16* v = V + static_cast<size_t>(bkv) * S_kv * D + lane * VEC;
    const int n = causal ? min(S_kv, row + 1) : S_kv;
    for (int j = 0; j < n; ++j) {
        float kr[VEC], vr[VEC];
        load_bf16<VEC>(k + static_cast<size_t>(j) * D, kr);
        load_bf16<VEC>(v + static_cast<size_t>(j) * D, vr);
        float s = 0.f, dp = 0.f;
#pragma unroll
        for (int e = 0; e < VEC; ++e) {
            s = fmaf(q[e], kr[e], s);
            dp = fmaf(d[e], vr[e], dp);
        }
        s = warp_reduce_sum(s);
        dp = warp_reduce_sum(dp);
        const float p = ex2(fmaf(s, scale_log2, -l.x));
        const float ds = p * (dp - l.y);
#pragma unroll
        for (int e = 0; e < VEC; ++e) dq[e] = fmaf(ds, kr[e], dq[e]);
    }
    store_bf16<VEC>(dQ + r * D + lane * VEC, dq, scale);
}

// dK_j = scale * sum_i dS_ij q_i and dV_j = sum_i P_ij dO_i over the query rows of every head
// of the group that see key j.
template <int D>
__global__ void __launch_bounds__(THREADS)
    bwd_v0_dkv_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                      const bf16* __restrict__ V, const bf16* __restrict__ dO,
                      const float2* __restrict__ ld, bf16* __restrict__ dK, bf16* __restrict__ dV,
                      int H_q, int H_kv, int S_q, int S_kv, int S_pad, float scale_log2,
                      float scale, int causal) {
    constexpr int VEC = D / 32;
    const int lane = threadIdx.x & 31;
    const int j = blockIdx.x * (THREADS / 32) + (threadIdx.x >> 5);
    const int bkv = blockIdx.y;
    if (j >= S_kv) return;
    const int group = H_q / H_kv;
    const int b = bkv / H_kv, kvh = bkv - b * H_kv;
    const size_t r = static_cast<size_t>(bkv) * S_kv + j;
    float k[VEC], v[VEC], dk[VEC], dv[VEC];
    load_bf16<VEC>(K + r * D + lane * VEC, k);
    load_bf16<VEC>(V + r * D + lane * VEC, v);
#pragma unroll
    for (int e = 0; e < VEC; ++e) dk[e] = dv[e] = 0.f;
    for (int hg = 0; hg < group; ++hg) {
        const int bh = b * H_q + kvh * group + hg;
        const bf16* q = Q + static_cast<size_t>(bh) * S_q * D + lane * VEC;
        const bf16* d = dO + static_cast<size_t>(bh) * S_q * D + lane * VEC;
        const float2* l = ld + static_cast<size_t>(bh) * S_pad;
        for (int i = causal ? j : 0; i < S_q; ++i) {
            float qr[VEC], dr[VEC];
            load_bf16<VEC>(q + static_cast<size_t>(i) * D, qr);
            load_bf16<VEC>(d + static_cast<size_t>(i) * D, dr);
            float s = 0.f, dp = 0.f;
#pragma unroll
            for (int e = 0; e < VEC; ++e) {
                s = fmaf(qr[e], k[e], s);
                dp = fmaf(dr[e], v[e], dp);
            }
            s = warp_reduce_sum(s);
            dp = warp_reduce_sum(dp);
            const float2 li = l[i];
            const float p = ex2(fmaf(s, scale_log2, -li.x));
            const float ds = p * (dp - li.y);
#pragma unroll
            for (int e = 0; e < VEC; ++e) {
                dv[e] = fmaf(p, dr[e], dv[e]);
                dk[e] = fmaf(ds, qr[e], dk[e]);
            }
        }
    }
    store_bf16<VEC>(dK + r * D + lane * VEC, dk, scale);
    store_bf16<VEC>(dV + r * D + lane * VEC, dv, 1.f);
}

}  // namespace v0

// ---------------------------------------------------------------------------------------
// Variants 1 and 2: the key-tile kernel. A block owns BN = 16 * WARPS keys of one K/V head,
// warp w the 16 keys k0 + 16w .. k0 + 16w + 15, and walks the (query head of the group,
// Q tile of BQ rows) pairs [ib, ie) of its slice. Per Q tile, per warp (mma.sync.m16n8k16,
// fragment layouts in common.cuh; lane (g, c) = (lane / 4, lane % 4)):
//
//   S^T  = K_w Q^T      A = K rows from smem (ldmatrix), B = Q as it lies ([q][d] = [n][k])
//   P^T  = 2^(S^T log2(e)/sqrt(D) - L2[q])       L2 and Dv of column q from the tile's ld
//   dP^T = V_w dO^T     A = V fragments (registers in variant 2), B = dO as it lies
//   dV  += P^T dO       A = P^T repacked from the S^T accumulators (the forward's P trick,
//                       k = q), B = dO as [k][n], ldmatrix.trans
//   dS^T = P^T (dP^T - Dv[q])
//   dK  += dS^T Q       A = dS^T repacked, B = Q through ldmatrix.trans
//
// dQ = dS K sums over all BN keys, so it cannot stay in one warp: with DQ each warp writes its
// dS^T rows (bf16) to shared memory, the block syncs, and the BQ x D dQ tile is split over the
// warps (BQ/16 m16 tiles, the D columns over the remaining warp factor), A = dS from the
// k-major dS^T through ldmatrix.trans, B = K through ldmatrix.trans, and each warp adds its
// fp32 tile to the dQ buffer with float2 atomics.
//
// Shared memory: K [BN][D] and, without VREG, V [BN][D], both with the 16-byte chunk swizzle
// of the forward (chunk ^ row % 8); NST stages of Q [BQ][D], dO [BQ][D] and ld [BQ] float2;
// one dS^T [BN][BQ] bf16 buffer, two with DEFER. With VREG the V tile is staged once through
// the stage region and read into A fragments before the loop.
//
// Template knobs (variant 1 is <4 warps, BQ 64, all off>, variant 2 <8 warps, BQ 32 at D = 128
// or 64 at D = 64, all on>): PIPE double-buffers the Q/dO stage with cp.async so the next
// tile's loads overlap this tile's products; VREG keeps V's fragments in registers; DQ runs
// the dQ product (off for the deterministic mode, where the dQ pass does it); SKIP lets a
// warp skip a causal tile its keys cannot see; DEFER runs the dQ product of tile t - 1 after
// tile t's barrier, one barrier per tile.
// ---------------------------------------------------------------------------------------
namespace kv {

struct Sched {
    int bkv_count;  // B * H_kv
    int kv_tiles;   // key tiles per K/V head
    int chunk;      // K/V heads per chunk of the tile order (see tile_coords)
    int H_q, H_kv, group;
    int q_tiles;  // cdiv(S_q, BQ)
    int S_pad;    // ld row stride per query head
    // Work split, two levels. Every tile is s1 items (s1 contiguous ranges of its (head, Q
    // tile) iterations); of the n = tiles * s1 items, the first n - tail are one block each and
    // the last `tail` are s2 blocks each. A block whose tile is split at all adds fp32 partials
    // of dK and dV to the workspace slot `ws_tile0`-relative tile index; others store bf16.
    int s1, s2, n_items, tail;
    int ws_tile0;  // first tile with a workspace slot (0 if s1 > 1, else the first tail tile)
    float* ws;     // per split tile: dK [BN][D] then dV [BN][D], fp32, zeroed
};

// Tile t -> (K/V head bkv, key tile kt). The tiles run in chunks of `chunk` K/V heads, and
// within a chunk key tile by key tile (heaviest first under the causal mask: key tile 0 sees
// every query row) across the chunk's heads. The chunk keeps the blocks in flight on a few
// heads, so their fp32 dQ adds hit the same rows while those rows are in L2; the order within
// it keeps the causal grid's tail short.
__host__ __device__ __forceinline__ void tile_coords(int t, int bkv_count, int kv_tiles, int chunk,
                                                     int& bkv, int& kt) {
    const int per_chunk = chunk * kv_tiles;
    const int c = t / per_chunk, r = t - c * per_chunk;
    const int left = bkv_count - c * chunk;
    const int heads = chunk < left ? chunk : left;
    kt = r / heads;
    bkv = c * chunk + (r - kt * heads);
}

// The block's tile and its [ib, ie) of the `total` iterations of that tile, and whether the
// tile is split (so the block adds to the workspace).
__device__ __forceinline__ void block_work(const Sched& s, int& tile, int& item, int& piece,
                                           int& pieces) {
    const int head = s.n_items - s.tail;
    if (static_cast<int>(blockIdx.x) < head) {
        item = blockIdx.x;
        piece = 0;
        pieces = 1;
    } else {
        const int r = blockIdx.x - head;
        item = head + r / s.s2;
        piece = r % s.s2;
        pieces = s.s2;
    }
    tile = item / s.s1;
}

template <int D, int WARPS, int BQ, bool PIPE, bool VREG, bool DQ, bool DEFER>
struct Cfg {
    static constexpr int BN = 16 * WARPS;
    static constexpr int THREADS = 32 * WARPS;
    static constexpr int NST = PIPE ? 2 : 1;
    static constexpr int K_BYTES = BN * D * 2;
    static constexpr int V_BYTES = VREG ? 0 : BN * D * 2;
    static constexpr int STAGE_BYTES = 2 * BQ * D * 2 + BQ * 8;
    static constexpr int DS_BUFS = DEFER ? 2 : 1;
    static constexpr int DS_BYTES = DQ ? DS_BUFS * BN * BQ * 2 : 0;
    static constexpr int SMEM = K_BYTES + V_BYTES + NST * STAGE_BYTES + DS_BYTES;
    static_assert(!VREG || NST * STAGE_BYTES + DS_BYTES >= BN * D * 2,
                  "V is staged through the stage and dS region");
    static_assert(SMEM <= 99 * 1024, "sm_120 allows 99 KB of shared memory per block");
};

// dS^T [BN][BQ]: rows of 64 bytes (BQ = 32) or 128 bytes (BQ = 64). The 16-byte chunk is
// XORed so that the eight rows an ldmatrix reads at one logical chunk, and the eight rows a
// warp's 32-bit stores hit, land in eight different bank groups: with 64-byte rows two rows
// share a 128-byte line, so the XOR takes row / 2.
template <int BQ>
__device__ __forceinline__ int ds_off(int row, int ch) {
    static_assert(BQ == 32 || BQ == 64, "");
    if constexpr (BQ == 32)
        return row * 32 + ((ch ^ ((row >> 1) & 3)) << 3);
    else
        return row * 64 + ((ch ^ (row & 7)) << 3);
}

template <int D, int WARPS, int BQ, bool PIPE, bool VREG, bool DQ, bool SKIP, bool DEFER>
__global__ void __launch_bounds__(WARPS * 32, 1)
    bwd_kv_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                  const bf16* __restrict__ V, const bf16* __restrict__ dO,
                  const float2* __restrict__ ld, float* __restrict__ dq_acc, bf16* __restrict__ dK,
                  bf16* __restrict__ dV, int S_q, int S_kv, float scale_log2, float scale,
                  int causal, Sched sched) {
    using C = Cfg<D, WARPS, BQ, PIPE, VREG, DQ, DEFER>;
    static_assert(!DEFER || (PIPE && DQ), "the deferred dQ product rides the pipelined loop");
    constexpr int BN = C::BN, THREADS = C::THREADS;
    constexpr int CH = D / 8;    // 16-byte chunks per row
    constexpr int KT = D / 16;   // k16 steps over d
    constexpr int DT = D / 8;    // n8 tiles over d
    constexpr int NQ = BQ / 8;   // n8 tiles of S^T over the tile's rows
    constexpr int QK = BQ / 16;  // k16 steps over the tile's rows

    extern __shared__ __align__(128) unsigned char smem[];
    bf16* Ks = reinterpret_cast<bf16*>(smem);
    const bf16* Vs = reinterpret_cast<const bf16*>(smem + C::K_BYTES);
    unsigned char* stages = smem + C::K_BYTES + C::V_BYTES;
    bf16* dS_base = reinterpret_cast<bf16*>(stages + C::NST * C::STAGE_BYTES);

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;

    int tile, item, piece, pieces;
    block_work(sched, tile, item, piece, pieces);
    int bkv, kt;
    tile_coords(tile, sched.bkv_count, sched.kv_tiles, sched.chunk, bkv, kt);
    const int k0 = kt * BN;
    const int b = bkv / sched.H_kv, kvh = bkv - b * sched.H_kv;
    const int bh0 = b * sched.H_q + kvh * sched.group;
    const bf16* Kg = K + static_cast<size_t>(bkv) * S_kv * D;
    const bf16* Vg = V + static_cast<size_t>(bkv) * S_kv * D;
    // Query tiles that can see a key of this block: all, or those ending at or after k0.
    const int qt0 = causal ? min(k0 / BQ, sched.q_tiles) : 0;
    const int nq = sched.q_tiles - qt0;
    const int total = sched.group * nq;
    // Item sub = item % s1 of the tile, then piece `piece` of `pieces` of that item.
    const int sub = item % sched.s1;
    const int lo = sub * total / sched.s1, hi = (sub + 1) * total / sched.s1;
    const int ib = lo + piece * (hi - lo) / pieces;
    const int ie = lo + (piece + 1) * (hi - lo) / pieces;
    const bool partial = sched.s1 > 1 || pieces > 1;

    // rows x D tile from rows r0.. of src (rows past `limit` zero-filled), swizzled.
    auto load_tile = [&](bf16* dst, const bf16* src, int r0, int rows, int limit) {
        for (int i = tid; i < rows * CH; i += THREADS) {
            const int row = i / CH, ch = i - row * CH;
            const bool ok = r0 + row < limit;
            cp_async_16_zfill(dst + row * D + swz(row, ch) * 8,
                              src + static_cast<size_t>(ok ? r0 + row : 0) * D + ch * 8, ok);
        }
    };
    auto load_q = [&](int it, int st) {
        const int bh = bh0 + it / nq;
        const int q0 = (qt0 + it % nq) * BQ;
        unsigned char* base = stages + st * C::STAGE_BYTES;
        load_tile(reinterpret_cast<bf16*>(base), Q + static_cast<size_t>(bh) * S_q * D, q0, BQ,
                  S_q);
        load_tile(reinterpret_cast<bf16*>(base + BQ * D * 2),
                  dO + static_cast<size_t>(bh) * S_q * D, q0, BQ, S_q);
        const float2* l = ld + static_cast<size_t>(bh) * sched.S_pad + q0;
        for (int i = tid; i < BQ / 2; i += THREADS)
            cp_async_16(base + 2 * BQ * D * 2 + i * 16, l + 2 * i);
    };

    // K (and V) of the block; with VREG, V goes through the stage region into A fragments.
    load_tile(Ks, Kg, k0, BN, S_kv);
    load_tile(VREG ? reinterpret_cast<bf16*>(stages) : const_cast<bf16*>(Vs), Vg, k0, BN, S_kv);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    const int arow = warp * 16 + (lane & 15);  // this lane's ldmatrix row of the warp's keys
    unsigned vf[VREG ? KT : 1][4];
    if constexpr (VREG) {
        const bf16* vs = reinterpret_cast<const bf16*>(stages);
#pragma unroll
        for (int kk = 0; kk < KT; ++kk)
            ldmatrix_x4(vf[kk], vs + arow * D + swz(arow, 2 * kk + (lane >> 4)) * 8);
        __syncthreads();  // the stage region is free again
    }
    if constexpr (PIPE) {
        if (ib < ie) load_q(ib, 0);
        cp_async_commit();
    }

    // dQ tile [BQ][D] += dS [BQ][BN] K [BN][D] of Q tile (bh_, q0_), dS^T in `dsb`, added
    // to the fp32 dQ buffer.
    auto dq_product = [&](const bf16* dsb, int bh_, int q0_) {
        // dQ tile [BQ][D] += dS [BQ][BN] K [BN][D]: warp -> m16 tile mt, n8 tiles
        // nw * DTW .. + DTW - 1.
        constexpr int MQ = BQ / 16;
        constexpr int NW = WARPS / MQ;
        constexpr int DTW = DT / NW;
        static_assert(WARPS % MQ == 0 && DT % NW == 0 && DTW % 2 == 0, "dQ tile split");
        const int mt = warp % MQ, nw = warp / MQ;
        float acc[DTW][4];
#pragma unroll
        for (int dj = 0; dj < DTW; ++dj)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[dj][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < BN / 16; ++kk) {
            // A = dS rows 16mt.., keys 16kk..: matrix i = lane / 8 is keys (i / 2) * 8 ..,
            // rows (i % 2) * 8 .. of the k-major dS^T, transposed by ldmatrix.
            unsigned a[4];
            {
                const int i = lane >> 3;
                const int krow = kk * 16 + (i >> 1) * 8 + (lane & 7);
                ldmatrix_x4_trans(a, dsb + ds_off<BQ>(krow, mt * 2 + (i & 1)));
            }
#pragma unroll
            for (int dj = 0; dj < DTW; dj += 2) {
                const int row = kk * 16 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4_trans(r, Ks + row * D + swz(row, nw * DTW + dj + (lane >> 4)) * 8);
                const unsigned b0[2] = {r[0], r[1]};
                const unsigned b1[2] = {r[2], r[3]};
                mma_bf16_16816(acc[dj], a, b0);
                mma_bf16_16816(acc[dj + 1], a, b1);
            }
        }
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int q = q0_ + mt * 16 + g + h * 8;
            if (q >= S_q) continue;
            float* dst = dq_acc + (static_cast<size_t>(bh_) * S_q + q) * D + nw * DTW * 8 + c2;
#pragma unroll
            for (int dj = 0; dj < DTW; ++dj)
                red_add2(dst + dj * 8, acc[dj][2 * h], acc[dj][2 * h + 1]);
        }
    };

    float dk[DT][4], dv[DT][4];
#pragma unroll
    for (int dj = 0; dj < DT; ++dj)
#pragma unroll
        for (int e = 0; e < 4; ++e) dk[dj][e] = dv[dj][e] = 0.f;

    const int kw0 = k0 + warp * 16;  // the warp's first key
    const bool kv_tail = k0 + BN > S_kv;

    for (int it = ib; it < ie; ++it) {
        const int st = PIPE ? (it - ib) & 1 : 0;
        if constexpr (PIPE) {
            cp_async_wait<0>();  // this tile's loads (this thread's share)
            __syncthreads();     // everyone's; and the other stage and dS are free
            if (it + 1 < ie) load_q(it + 1, st ^ 1);
            cp_async_commit();
        } else {
            __syncthreads();  // the stage and dS of the previous tile are free
            load_q(it, 0);
            cp_async_commit();
            cp_async_wait<0>();
            __syncthreads();
        }
        const int bh = bh0 + it / nq;
        const int q0 = (qt0 + it % nq) * BQ;
        // With DEFER the dQ product of the previous tile runs here, right after the loop's one
        // barrier (which also published that tile's dS^T), and this tile's dS^T goes to the
        // other buffer: one block-wide barrier per tile instead of two.
        bf16* dSs = dS_base + (DEFER ? ((it - ib) & 1) * BN * BQ : 0);
        if constexpr (DEFER) {
            if (it > ib)
                dq_product(dS_base + ((it - 1 - ib) & 1) * BN * BQ, bh0 + (it - 1) / nq,
                           (qt0 + (it - 1) % nq) * BQ);
        }
        const bf16* qs = reinterpret_cast<const bf16*>(stages + st * C::STAGE_BYTES);
        const bf16* dos = qs + BQ * D;
        const float* lds = reinterpret_cast<const float*>(dos + BQ * D);

        // Under the causal mask a warp whose first key follows the tile's last row sees none
        // of it: every P is 0, the tile adds nothing to its dK, dV, and its dS rows are 0.
        const bool dead = SKIP && causal && q0 + BQ - 1 < kw0;
        if (!dead) {
            // S^T = K_w Q^T: 16 keys x BQ rows, NQ n8 accumulators.
            float s[NQ][4];
#pragma unroll
            for (int n = 0; n < NQ; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) s[n][e] = 0.f;
#pragma unroll
            for (int kk = 0; kk < KT; ++kk) {
                unsigned a[4];
                ldmatrix_x4(a, Ks + arow * D + swz(arow, 2 * kk + (lane >> 4)) * 8);
#pragma unroll
                for (int n = 0; n < NQ; n += 2) {
                    const int row = n * 8 + (lane & 15);
                    unsigned r[4];
                    ldmatrix_x4(r, qs + row * D + swz(row, 2 * kk + (lane >> 4)) * 8);
                    const unsigned b0[2] = {r[0], r[2]};
                    const unsigned b1[2] = {r[1], r[3]};
                    mma_bf16_16816(s[n], a, b0);
                    mma_bf16_16816(s[n + 1], a, b1);
                }
            }
            // P^T. Lane (g, c) holds keys g (e = 0, 1) and g + 8 (e = 2, 3) against rows
            // 8n + 2c (e even) and 8n + 2c + 1 (e odd): ld of those two rows is one float4.
            float dvec[NQ][2];
#pragma unroll
            for (int n = 0; n < NQ; ++n) {
                const float4 l4 = *reinterpret_cast<const float4*>(lds + 2 * (n * 8 + c2));
                s[n][0] = ex2(fmaf(s[n][0], scale_log2, -l4.x));
                s[n][1] = ex2(fmaf(s[n][1], scale_log2, -l4.z));
                s[n][2] = ex2(fmaf(s[n][2], scale_log2, -l4.x));
                s[n][3] = ex2(fmaf(s[n][3], scale_log2, -l4.z));
                dvec[n][0] = l4.y;
                dvec[n][1] = l4.w;
            }
            // Masks, warp-uniform tests first: causal (key > row) where the warp's keys reach
            // past the tile's first row, and keys past S_kv, whose zero-filled K gives a finite
            // but meaningless P that must not reach dQ.
            if ((causal && q0 < kw0 + 15) || kv_tail) {
#pragma unroll
                for (int n = 0; n < NQ; ++n)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int key = kw0 + g + (e >> 1) * 8;
                        const int q = q0 + n * 8 + c2 + (e & 1);
                        if ((causal && key > q) || key >= S_kv) s[n][e] = 0.f;
                    }
            }

            // dP^T = V_w dO^T.
            float dp[NQ][4];
#pragma unroll
            for (int n = 0; n < NQ; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) dp[n][e] = 0.f;
#pragma unroll
            for (int kk = 0; kk < KT; ++kk) {
                unsigned a[4];
                if constexpr (VREG) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) a[e] = vf[kk][e];
                } else {
                    ldmatrix_x4(a, Vs + arow * D + swz(arow, 2 * kk + (lane >> 4)) * 8);
                }
#pragma unroll
                for (int n = 0; n < NQ; n += 2) {
                    const int row = n * 8 + (lane & 15);
                    unsigned r[4];
                    ldmatrix_x4(r, dos + row * D + swz(row, 2 * kk + (lane >> 4)) * 8);
                    const unsigned b0[2] = {r[0], r[2]};
                    const unsigned b1[2] = {r[1], r[3]};
                    mma_bf16_16816(dp[n], a, b0);
                    mma_bf16_16816(dp[n + 1], a, b1);
                }
            }

            // dV += P^T dO: k16 step kq covers rows 16kq .. 16kq+15 = n8 tiles 2kq, 2kq+1.
            unsigned pa[QK][4];
#pragma unroll
            for (int kq = 0; kq < QK; ++kq) {
                pa[kq][0] = pack_bf16x2(s[2 * kq][0], s[2 * kq][1]);
                pa[kq][1] = pack_bf16x2(s[2 * kq][2], s[2 * kq][3]);
                pa[kq][2] = pack_bf16x2(s[2 * kq + 1][0], s[2 * kq + 1][1]);
                pa[kq][3] = pack_bf16x2(s[2 * kq + 1][2], s[2 * kq + 1][3]);
            }
#pragma unroll
            for (int kq = 0; kq < QK; ++kq) {
#pragma unroll
                for (int dj = 0; dj < DT; dj += 2) {
                    const int row = kq * 16 + (lane & 15);
                    unsigned r[4];
                    ldmatrix_x4_trans(r, dos + row * D + swz(row, dj + (lane >> 4)) * 8);
                    const unsigned b0[2] = {r[0], r[1]};
                    const unsigned b1[2] = {r[2], r[3]};
                    mma_bf16_16816(dv[dj], pa[kq], b0);
                    mma_bf16_16816(dv[dj + 1], pa[kq], b1);
                }
            }

            // dS^T = P^T (dP^T - Dv), packed as the A fragments of dK += dS^T Q.
#pragma unroll
            for (int n = 0; n < NQ; ++n) {
                s[n][0] *= dp[n][0] - dvec[n][0];
                s[n][1] *= dp[n][1] - dvec[n][1];
                s[n][2] *= dp[n][2] - dvec[n][0];
                s[n][3] *= dp[n][3] - dvec[n][1];
            }
#pragma unroll
            for (int kq = 0; kq < QK; ++kq) {
                pa[kq][0] = pack_bf16x2(s[2 * kq][0], s[2 * kq][1]);
                pa[kq][1] = pack_bf16x2(s[2 * kq][2], s[2 * kq][3]);
                pa[kq][2] = pack_bf16x2(s[2 * kq + 1][0], s[2 * kq + 1][1]);
                pa[kq][3] = pack_bf16x2(s[2 * kq + 1][2], s[2 * kq + 1][3]);
            }
#pragma unroll
            for (int kq = 0; kq < QK; ++kq) {
#pragma unroll
                for (int dj = 0; dj < DT; dj += 2) {
                    const int row = kq * 16 + (lane & 15);
                    unsigned r[4];
                    ldmatrix_x4_trans(r, qs + row * D + swz(row, dj + (lane >> 4)) * 8);
                    const unsigned b0[2] = {r[0], r[1]};
                    const unsigned b1[2] = {r[2], r[3]};
                    mma_bf16_16816(dk[dj], pa[kq], b0);
                    mma_bf16_16816(dk[dj + 1], pa[kq], b1);
                }
            }
            if constexpr (DQ) {
                // dS^T rows g and g + 8 of the warp, columns 16kq + 2c (+ 8): 32-bit stores.
                const int r0 = warp * 16 + g;
#pragma unroll
                for (int kq = 0; kq < QK; ++kq) {
                    *reinterpret_cast<unsigned*>(dSs + ds_off<BQ>(r0, 2 * kq) + c2) = pa[kq][0];
                    *reinterpret_cast<unsigned*>(dSs + ds_off<BQ>(r0 + 8, 2 * kq) + c2) = pa[kq][1];
                    *reinterpret_cast<unsigned*>(dSs + ds_off<BQ>(r0, 2 * kq + 1) + c2) = pa[kq][2];
                    *reinterpret_cast<unsigned*>(dSs + ds_off<BQ>(r0 + 8, 2 * kq + 1) + c2) =
                        pa[kq][3];
                }
            }
        } else if constexpr (DQ) {
            const int r0 = warp * 16 + g;
#pragma unroll
            for (int ch = 0; ch < BQ / 8; ++ch) {
                *reinterpret_cast<unsigned*>(dSs + ds_off<BQ>(r0, ch) + c2) = 0u;
                *reinterpret_cast<unsigned*>(dSs + ds_off<BQ>(r0 + 8, ch) + c2) = 0u;
            }
        }

        if constexpr (DQ && !DEFER) {
            __syncthreads();  // dS^T of every warp is in shared memory
            dq_product(dS_base, bh, q0);
        }
    }
    if constexpr (DEFER) {
        if (ie > ib) {
            __syncthreads();
            dq_product(dS_base + ((ie - 1 - ib) & 1) * BN * BQ, bh0 + (ie - 1) / nq,
                       (qt0 + (ie - 1) % nq) * BQ);
        }
    }
    if constexpr (PIPE) cp_async_wait<0>();

    // Epilogue: rows g and g + 8 of the warp's keys. A whole tile stores bf16 from registers;
    // a slice of a split tile adds fp32 to its tile's workspace.
    if (!partial) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int key = kw0 + g + h * 8;
            if (key >= S_kv) continue;
            bf16* dkr = dK + (static_cast<size_t>(bkv) * S_kv + key) * D + c2;
            bf16* dvr = dV + (static_cast<size_t>(bkv) * S_kv + key) * D + c2;
#pragma unroll
            for (int dj = 0; dj < DT; ++dj) {
                *reinterpret_cast<unsigned*>(dkr + dj * 8) =
                    pack_bf16x2(dk[dj][2 * h] * scale, dk[dj][2 * h + 1] * scale);
                *reinterpret_cast<unsigned*>(dvr + dj * 8) =
                    pack_bf16x2(dv[dj][2 * h], dv[dj][2 * h + 1]);
            }
        }
    } else {
        float* w = sched.ws + static_cast<size_t>(tile - sched.ws_tile0) * 2 * BN * D;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lrow = warp * 16 + g + h * 8;
#pragma unroll
            for (int dj = 0; dj < DT; ++dj) {
                red_add2(w + lrow * D + dj * 8 + c2, dk[dj][2 * h] * scale,
                         dk[dj][2 * h + 1] * scale);
                red_add2(w + BN * D + lrow * D + dj * 8 + c2, dv[dj][2 * h], dv[dj][2 * h + 1]);
            }
        }
    }
}

// dK and dV rows of the split tiles, from their fp32 workspace. blockIdx.y = split tile.
template <int D, int BN>
__global__ void bwd_split_convert_kernel(bf16* __restrict__ dK, bf16* __restrict__ dV, int S_kv,
                                         Sched sched) {
    constexpr int CH = D / 8;
    const int tile = sched.ws_tile0 + blockIdx.y;
    int bkv, kt;
    tile_coords(tile, sched.bkv_count, sched.kv_tiles, sched.chunk, bkv, kt);
    const int k0 = kt * BN;
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // (row, chunk, dK or dV)
    if (idx >= 2 * BN * CH) return;
    const int which = idx / (BN * CH);
    const int lrow = (idx % (BN * CH)) / CH, ch = idx % CH;
    const int key = k0 + lrow;
    if (key >= S_kv) return;
    const float* src = sched.ws + static_cast<size_t>(blockIdx.y) * 2 * BN * D + which * BN * D +
                       lrow * D + ch * 8;
    const float4 a = *reinterpret_cast<const float4*>(src);
    const float4 c = *reinterpret_cast<const float4*>(src + 4);
    uint4 u;
    u.x = pack_bf16x2(a.x, a.y);
    u.y = pack_bf16x2(a.z, a.w);
    u.z = pack_bf16x2(c.x, c.y);
    u.w = pack_bf16x2(c.z, c.w);
    bf16* dst = (which ? dV : dK) + (static_cast<size_t>(bkv) * S_kv + key) * D + ch * 8;
    *reinterpret_cast<uint4*>(dst) = u;
}

}  // namespace kv

// ---------------------------------------------------------------------------------------
// Variant 3: variant 2's tile and products with the block-wide barrier gone, as forward
// variant 4 did for its K/V loop. The Q, dO and (L, Dv) tiles arrive by TMA into three stages
// guarded by full / empty mbarriers, issued by lane 0 of warp 0; the dS^T double buffer is
// handed between the warps by a second full / empty pair per buffer. Per Q tile a warp:
//   waits for the tile's stage, runs S^T, P^T, dP^T, dV, dS^T, dK on it, releases the stage;
//   waits until every warp has finished the dQ product that last read this dS^T buffer,
//   writes its dS^T rows there and arrives on the buffer's "full";
//   waits for the previous tile's buffer to be full (every warp's dS^T of that tile), runs
//   that tile's dQ product and arrives on its "empty";
//   (warp 0, lane 0) issues the loads of the tile two ahead into the stage the previous tile
//   left, which every warp has released by then: the full wait just before implies it.
// Nothing makes the eight warps wait for each other except the dS^T of the previous tile, so
// they drift apart by up to a tile and one warp's exponentials and dS^T run while the other
// warp on its scheduler issues mma, where variant 2's barrier started all eight in phase.
//
// Shared memory: K [BN][D] (cp.async, forward swizzle), three stages of Q and dO as D/64 TMA
// boxes of BQ x 64 with the 128-byte swizzle (box_off), three (L, Dv) tiles by 1-D bulk
// copy, two dS^T buffers, the barriers. V is staged through the stage region once and held
// in registers, as in variant 2.
// ---------------------------------------------------------------------------------------
namespace kvt {

using kv::block_work;
using kv::ds_off;
using kv::Sched;
using kv::tile_coords;

constexpr int WARPS = 8, THREADS = 256, BN = 128, STAGES = 3, BOX = 64;

template <int D, int BQ, bool DQ>
struct Cfg {
    static constexpr int K_BYTES = BN * D * 2;
    static constexpr int QT_BYTES = BQ * D * 2;       // one Q or dO tile
    static constexpr int STAGE_BYTES = 2 * QT_BYTES;  // Q then dO, 1 KB multiples
    static constexpr int LD_BYTES = BQ * 8;
    static constexpr int DS_BYTES = DQ ? 2 * BN * BQ * 2 : 0;
    static constexpr int OFF_STAGES = K_BYTES;
    static constexpr int OFF_LD = OFF_STAGES + STAGES * STAGE_BYTES;
    static constexpr int OFF_DS = OFF_LD + STAGES * LD_BYTES;
    static constexpr int OFF_BAR = OFF_DS + DS_BYTES;
    static constexpr int SMEM = OFF_BAR + 16 * 8;
    static_assert(QT_BYTES % 1024 == 0 && OFF_STAGES % 1024 == 0, "TMA swizzle alignment");
    static_assert(STAGES * STAGE_BYTES >= BN * D * 2, "V is staged through the stages");
    static_assert(SMEM <= 99 * 1024, "sm_120 allows 99 KB of shared memory per block");
};

// Element offset of logical (row, 16-byte chunk) in a ROWS x D tile stored as D/64 TMA boxes
// of ROWS x 64 with the 128-byte swizzle (the forward's variant 4 layout).
template <int ROWS>
__device__ __forceinline__ int box_off(int row, int ch) {
    return (ch >> 3) * (ROWS * BOX) + row * BOX + ((ch & 7) ^ (row & 7)) * 8;
}

// 1-D bulk copy global -> shared, completion counted on `bar` (bytes a multiple of 16).
__device__ __forceinline__ void bulk_load(void* smem_dst, const void* src, unsigned bytes,
                                          uint64_t* bar) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, "
        "[%3];\n" ::"r"(smem_u32(smem_dst)),
        "l"(src), "r"(bytes), "r"(smem_u32(bar))
        : "memory");
}

template <int D, int BQ, bool DQ, bool SKIP>
__global__ void __launch_bounds__(THREADS, 1)
    bwd_kv_tma_kernel(const __grid_constant__ CUtensorMap tmQ,
                      const __grid_constant__ CUtensorMap tmO, const bf16* __restrict__ K,
                      const bf16* __restrict__ V, const float2* __restrict__ ld,
                      float* __restrict__ dq_acc, bf16* __restrict__ dK, bf16* __restrict__ dV,
                      int S_q, int S_kv, float scale_log2, float scale, int causal, Sched sched) {
    using C = Cfg<D, BQ, DQ>;
    constexpr int CH = D / 8, KT = D / 16, DT = D / 8, NQ = BQ / 8, QK = BQ / 16;
    constexpr int BOXES = D / BOX;

    extern __shared__ __align__(1024) unsigned char smem[];
    if (smem_u32(smem) % 1024 != 0) __trap();
    bf16* Ks = reinterpret_cast<bf16*>(smem);
    unsigned char* stages = smem + C::OFF_STAGES;
    float* lds_base = reinterpret_cast<float*>(smem + C::OFF_LD);
    bf16* dS_base = reinterpret_cast<bf16*>(smem + C::OFF_DS);
    uint64_t* full = reinterpret_cast<uint64_t*>(smem + C::OFF_BAR);
    uint64_t* empty = full + STAGES;
    uint64_t* ds_full = empty + STAGES;
    uint64_t* ds_empty = ds_full + 2;

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;

    int tile, item, piece, pieces;
    block_work(sched, tile, item, piece, pieces);
    int bkv, kt;
    tile_coords(tile, sched.bkv_count, sched.kv_tiles, sched.chunk, bkv, kt);
    const int k0 = kt * BN;
    const int b = bkv / sched.H_kv, kvh = bkv - b * sched.H_kv;
    const int bh0 = b * sched.H_q + kvh * sched.group;
    const int qt0 = causal ? min(k0 / BQ, sched.q_tiles) : 0;
    const int nq = sched.q_tiles - qt0;
    const int total = sched.group * nq;
    const int sub = item % sched.s1;
    const int lo = sub * total / sched.s1, hi = (sub + 1) * total / sched.s1;
    const int ib = lo + piece * (hi - lo) / pieces;
    const int ie = lo + (piece + 1) * (hi - lo) / pieces;
    const bool partial = sched.s1 > 1 || pieces > 1;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full[s], 1);       // the producer's arrive.expect_tx
            mbar_init(&empty[s], WARPS);  // lane 0 of each warp, after its reads
        }
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            mbar_init(&ds_full[i], THREADS);   // every thread, after its dS^T stores
            mbar_init(&ds_empty[i], THREADS);  // every thread, after its dQ product reads
        }
        fence_mbar_init();
    }

    // K and V of the block by cp.async; V through the stage region into A fragments.
    {
        const bf16* Kg = K + static_cast<size_t>(bkv) * S_kv * D;
        const bf16* Vg = V + static_cast<size_t>(bkv) * S_kv * D;
        bf16* vstage = reinterpret_cast<bf16*>(stages);
        for (int i = tid; i < BN * CH; i += THREADS) {
            const int row = i / CH, ch = i - row * CH;
            const bool ok = k0 + row < S_kv;
            const size_t off = static_cast<size_t>(ok ? k0 + row : 0) * D + ch * 8;
            cp_async_16_zfill(Ks + row * D + swz(row, ch) * 8, Kg + off, ok);
            cp_async_16_zfill(vstage + row * D + swz(row, ch) * 8, Vg + off, ok);
        }
        cp_async_commit();
        cp_async_wait<0>();
    }
    __syncthreads();
    const int arow = warp * 16 + (lane & 15);
    unsigned vf[KT][4];
    {
        const bf16* vs = reinterpret_cast<const bf16*>(stages);
#pragma unroll
        for (int kk = 0; kk < KT; ++kk)
            ldmatrix_x4(vf[kk], vs + arow * D + swz(arow, 2 * kk + (lane >> 4)) * 8);
    }
    fence_proxy_async_smem();  // the V reads come before the TMA overwrites the stages
    __syncthreads();           // the only block-wide barrier

    // Load of iteration u (ib <= u < ie) into stage (u - ib) % STAGES.
    auto produce = [&](int u) {
        const int j = u - ib, s = j % STAGES, use = j / STAGES;
        if (use > 0) mbar_wait(&empty[s], (use - 1) & 1);
        const int bh = bh0 + u / nq;
        const int q0 = (qt0 + u % nq) * BQ;
        unsigned char* dst = stages + s * C::STAGE_BYTES;
        mbar_arrive_expect_tx(&full[s], C::STAGE_BYTES + C::LD_BYTES);
#pragma unroll
        for (int bx = 0; bx < BOXES; ++bx) {
            tma_load_3d(dst + bx * BQ * BOX * 2, &tmQ, &full[s], bx * BOX, q0, bh);
            tma_load_3d(dst + C::QT_BYTES + bx * BQ * BOX * 2, &tmO, &full[s], bx * BOX, q0, bh);
        }
        bulk_load(lds_base + s * (C::LD_BYTES / 4), ld + static_cast<size_t>(bh) * sched.S_pad + q0,
                  C::LD_BYTES, &full[s]);
    };
    if (tid == 0) {
        prefetch_tensormap(&tmQ);
        prefetch_tensormap(&tmO);
        for (int u = ib; u < min(ie, ib + 2); ++u) produce(u);
    }

    // dQ tile [BQ][D] += dS [BQ][BN] K [BN][D] of iteration u, dS^T in buffer (u - ib) & 1.
    auto dq_product = [&](int u) {
        constexpr int MQ = BQ / 16, NW = WARPS / MQ, DTW = DT / NW;
        static_assert(WARPS % MQ == 0 && DT % NW == 0 && DTW % 2 == 0, "dQ tile split");
        const int j = u - ib, buf = j & 1;
        const bf16* dsb = dS_base + buf * BN * BQ;
        const int bh = bh0 + u / nq;
        const int qt = qt0 + u % nq;
        const int q0 = qt * BQ;
        const int mt = warp % MQ, nw = warp / MQ;
        mbar_wait(&ds_full[buf], (j >> 1) & 1);
        float acc[DTW][4];
#pragma unroll
        for (int dj = 0; dj < DTW; ++dj)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[dj][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < BN / 16; ++kk) {
            unsigned a[4];
            {
                const int i = lane >> 3;
                const int krow = kk * 16 + (i >> 1) * 8 + (lane & 7);
                ldmatrix_x4_trans(a, dsb + ds_off<BQ>(krow, mt * 2 + (i & 1)));
            }
#pragma unroll
            for (int dj = 0; dj < DTW; dj += 2) {
                const int row = kk * 16 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4_trans(r, Ks + row * D + swz(row, nw * DTW + dj + (lane >> 4)) * 8);
                const unsigned b0[2] = {r[0], r[1]};
                const unsigned b1[2] = {r[2], r[3]};
                mma_bf16_16816(acc[dj], a, b0);
                mma_bf16_16816(acc[dj + 1], a, b1);
            }
        }
        mbar_arrive(&ds_empty[buf]);
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int q = q0 + mt * 16 + g + h * 8;
            if (q >= S_q) continue;
            float* dst = dq_acc + (static_cast<size_t>(bh) * S_q + q) * D + nw * DTW * 8 + c2;
#pragma unroll
            for (int dj = 0; dj < DTW; ++dj)
                red_add2(dst + dj * 8, acc[dj][2 * h], acc[dj][2 * h + 1]);
        }
    };

    float dk[DT][4], dv[DT][4];
#pragma unroll
    for (int dj = 0; dj < DT; ++dj)
#pragma unroll
        for (int e = 0; e < 4; ++e) dk[dj][e] = dv[dj][e] = 0.f;

    const int kw0 = k0 + warp * 16;
    const bool kv_tail = k0 + BN > S_kv;

    for (int it = ib; it < ie; ++it) {
        const int j = it - ib, st = j % STAGES;
        const int q0 = (qt0 + it % nq) * BQ;
        const bf16* qs = reinterpret_cast<const bf16*>(stages + st * C::STAGE_BYTES);
        const bf16* dos = qs + BQ * D;
        const float* lds = lds_base + st * (C::LD_BYTES / 4);
        unsigned pa[QK][4];
        mbar_wait(&full[st], (j / STAGES) & 1);

        const bool dead = SKIP && causal && q0 + BQ - 1 < kw0;
        if (!dead) {
            float s[NQ][4];
#pragma unroll
            for (int n = 0; n < NQ; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) s[n][e] = 0.f;
#pragma unroll
            for (int kk = 0; kk < KT; ++kk) {
                unsigned a[4];
                ldmatrix_x4(a, Ks + arow * D + swz(arow, 2 * kk + (lane >> 4)) * 8);
#pragma unroll
                for (int n = 0; n < NQ; n += 2) {
                    unsigned r[4];
                    ldmatrix_x4(r, qs + box_off<BQ>(n * 8 + (lane & 15), 2 * kk + (lane >> 4)));
                    const unsigned b0[2] = {r[0], r[2]};
                    const unsigned b1[2] = {r[1], r[3]};
                    mma_bf16_16816(s[n], a, b0);
                    mma_bf16_16816(s[n + 1], a, b1);
                }
            }
            float dvec[NQ][2];
#pragma unroll
            for (int n = 0; n < NQ; ++n) {
                const float4 l4 = *reinterpret_cast<const float4*>(lds + 2 * (n * 8 + c2));
                s[n][0] = ex2(fmaf(s[n][0], scale_log2, -l4.x));
                s[n][1] = ex2(fmaf(s[n][1], scale_log2, -l4.z));
                s[n][2] = ex2(fmaf(s[n][2], scale_log2, -l4.x));
                s[n][3] = ex2(fmaf(s[n][3], scale_log2, -l4.z));
                dvec[n][0] = l4.y;
                dvec[n][1] = l4.w;
            }
            if ((causal && q0 < kw0 + 15) || kv_tail) {
#pragma unroll
                for (int n = 0; n < NQ; ++n)
#pragma unroll
                    for (int e = 0; e < 4; ++e) {
                        const int key = kw0 + g + (e >> 1) * 8;
                        const int q = q0 + n * 8 + c2 + (e & 1);
                        if ((causal && key > q) || key >= S_kv) s[n][e] = 0.f;
                    }
            }
            float dp[NQ][4];
#pragma unroll
            for (int n = 0; n < NQ; ++n)
#pragma unroll
                for (int e = 0; e < 4; ++e) dp[n][e] = 0.f;
#pragma unroll
            for (int kk = 0; kk < KT; ++kk) {
#pragma unroll
                for (int n = 0; n < NQ; n += 2) {
                    unsigned r[4];
                    ldmatrix_x4(r, dos + box_off<BQ>(n * 8 + (lane & 15), 2 * kk + (lane >> 4)));
                    const unsigned b0[2] = {r[0], r[2]};
                    const unsigned b1[2] = {r[1], r[3]};
                    mma_bf16_16816(dp[n], vf[kk], b0);
                    mma_bf16_16816(dp[n + 1], vf[kk], b1);
                }
            }
#pragma unroll
            for (int kq = 0; kq < QK; ++kq) {
                pa[kq][0] = pack_bf16x2(s[2 * kq][0], s[2 * kq][1]);
                pa[kq][1] = pack_bf16x2(s[2 * kq][2], s[2 * kq][3]);
                pa[kq][2] = pack_bf16x2(s[2 * kq + 1][0], s[2 * kq + 1][1]);
                pa[kq][3] = pack_bf16x2(s[2 * kq + 1][2], s[2 * kq + 1][3]);
            }
#pragma unroll
            for (int kq = 0; kq < QK; ++kq) {
#pragma unroll
                for (int dj = 0; dj < DT; dj += 2) {
                    unsigned r[4];
                    ldmatrix_x4_trans(r,
                                      dos + box_off<BQ>(kq * 16 + (lane & 15), dj + (lane >> 4)));
                    const unsigned b0[2] = {r[0], r[1]};
                    const unsigned b1[2] = {r[2], r[3]};
                    mma_bf16_16816(dv[dj], pa[kq], b0);
                    mma_bf16_16816(dv[dj + 1], pa[kq], b1);
                }
            }
#pragma unroll
            for (int n = 0; n < NQ; ++n) {
                s[n][0] *= dp[n][0] - dvec[n][0];
                s[n][1] *= dp[n][1] - dvec[n][1];
                s[n][2] *= dp[n][2] - dvec[n][0];
                s[n][3] *= dp[n][3] - dvec[n][1];
            }
#pragma unroll
            for (int kq = 0; kq < QK; ++kq) {
                pa[kq][0] = pack_bf16x2(s[2 * kq][0], s[2 * kq][1]);
                pa[kq][1] = pack_bf16x2(s[2 * kq][2], s[2 * kq][3]);
                pa[kq][2] = pack_bf16x2(s[2 * kq + 1][0], s[2 * kq + 1][1]);
                pa[kq][3] = pack_bf16x2(s[2 * kq + 1][2], s[2 * kq + 1][3]);
            }
#pragma unroll
            for (int kq = 0; kq < QK; ++kq) {
#pragma unroll
                for (int dj = 0; dj < DT; dj += 2) {
                    unsigned r[4];
                    ldmatrix_x4_trans(r, qs + box_off<BQ>(kq * 16 + (lane & 15), dj + (lane >> 4)));
                    const unsigned b0[2] = {r[0], r[1]};
                    const unsigned b1[2] = {r[2], r[3]};
                    mma_bf16_16816(dk[dj], pa[kq], b0);
                    mma_bf16_16816(dk[dj + 1], pa[kq], b1);
                }
            }
        } else {
#pragma unroll
            for (int kq = 0; kq < QK; ++kq)
#pragma unroll
                for (int e = 0; e < 4; ++e) pa[kq][e] = 0u;
        }
        // Release the stage to the producer.
        fence_proxy_async_smem();
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty[st]);

        if constexpr (DQ) {
            // dS^T of this tile into buffer j & 1, once every warp is done with its last use.
            const int buf = j & 1;
            if (j >= 2) mbar_wait(&ds_empty[buf], ((j >> 1) - 1) & 1);
            bf16* dSs = dS_base + buf * BN * BQ;
            const int r0 = warp * 16 + g;
#pragma unroll
            for (int kq = 0; kq < QK; ++kq) {
                *reinterpret_cast<unsigned*>(dSs + ds_off<BQ>(r0, 2 * kq) + c2) = pa[kq][0];
                *reinterpret_cast<unsigned*>(dSs + ds_off<BQ>(r0 + 8, 2 * kq) + c2) = pa[kq][1];
                *reinterpret_cast<unsigned*>(dSs + ds_off<BQ>(r0, 2 * kq + 1) + c2) = pa[kq][2];
                *reinterpret_cast<unsigned*>(dSs + ds_off<BQ>(r0 + 8, 2 * kq + 1) + c2) = pa[kq][3];
            }
            mbar_arrive(&ds_full[buf]);
            if (j >= 1) dq_product(it - 1);
        }
        // The stage of iteration it - 1 is free (every warp's dS^T of it - 1 is written, so
        // every warp has released that stage): refill it with iteration it + 2.
        if (tid == 0 && it + 2 < ie) produce(it + 2);
    }
    if constexpr (DQ) {
        if (ie > ib) {
            dq_product(ie - 1);
        }
    }

    if (!partial) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int key = kw0 + g + h * 8;
            if (key >= S_kv) continue;
            bf16* dkr = dK + (static_cast<size_t>(bkv) * S_kv + key) * D + c2;
            bf16* dvr = dV + (static_cast<size_t>(bkv) * S_kv + key) * D + c2;
#pragma unroll
            for (int dj = 0; dj < DT; ++dj) {
                *reinterpret_cast<unsigned*>(dkr + dj * 8) =
                    pack_bf16x2(dk[dj][2 * h] * scale, dk[dj][2 * h + 1] * scale);
                *reinterpret_cast<unsigned*>(dvr + dj * 8) =
                    pack_bf16x2(dv[dj][2 * h], dv[dj][2 * h + 1]);
            }
        }
    } else {
        float* w = sched.ws + static_cast<size_t>(tile - sched.ws_tile0) * 2 * BN * D;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int lrow = warp * 16 + g + h * 8;
#pragma unroll
            for (int dj = 0; dj < DT; ++dj) {
                red_add2(w + lrow * D + dj * 8 + c2, dk[dj][2 * h] * scale,
                         dk[dj][2 * h + 1] * scale);
                red_add2(w + BN * D + lrow * D + dj * 8 + c2, dv[dj][2 * h], dv[dj][2 * h + 1]);
            }
        }
    }
}

}  // namespace kvt

// ---------------------------------------------------------------------------------------
// The deterministic dQ pass: the forward's variant 2 structure with the second product
// changed. A block owns 128 query rows (8 warps x 16) of one query head; Q and dO fragments
// stay in registers; K and V tiles of 64 keys stream through a 3-stage cp.async pipeline.
// Per tile, per warp:
//   S  = Q K^T      B = K as it lies
//   P  = 2^(S log2(e)/sqrt(D) - L2[row])     rows g and g + 8: two constants per lane
//   dP = dO V^T     B = V as it lies ([key][d] = [n][k])
//   dS = P (dP - Dv[row])
//   dQ += dS K      A = dS repacked from the accumulators, B = K through ldmatrix.trans
// ---------------------------------------------------------------------------------------
namespace dqp {

constexpr int WARPS = 8, THREADS = 32 * WARPS, BM = 16 * WARPS, BN = 64, STAGES = 3;

template <int D>
constexpr int smem_bytes() {
    return STAGES * 2 * BN * D * 2;
}

template <int D>
__global__ void __launch_bounds__(THREADS, 1)
    bwd_dq_kernel(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                  const bf16* __restrict__ V, const bf16* __restrict__ dO,
                  const float2* __restrict__ ld, bf16* __restrict__ dQ, int H_q, int H_kv, int S_q,
                  int S_kv, int S_pad, int bh_count, int q_tiles, float scale_log2, float scale,
                  int causal) {
    constexpr int CH = D / 8, KT = D / 16, DT = D / 8, NT = BN / 8, PT = BN / 16;
    constexpr int STAGE = 2 * BN * D;  // elements: K tile then V tile
    static_assert(2 * BM * D <= 2 * STAGE, "Q and dO are staged through stages 0 and 1");
    extern __shared__ __align__(128) unsigned char smem_raw[];
    bf16* smem = reinterpret_cast<bf16*>(smem_raw);

    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    const int bh = blockIdx.x % bh_count;
    const int q_rank = blockIdx.x / bh_count;
    const int q0 = (causal ? q_tiles - 1 - q_rank : q_rank) * BM;
    const int bkv = kv_index(bh, H_q, H_kv);
    const bf16* Kg = K + static_cast<size_t>(bkv) * S_kv * D;
    const bf16* Vg = V + static_cast<size_t>(bkv) * S_kv * D;

    auto load_tile = [&](bf16* dst, const bf16* src, int r0, int rows, int limit) {
        for (int i = tid; i < rows * CH; i += THREADS) {
            const int row = i / CH, ch = i - row * CH;
            const bool ok = r0 + row < limit;
            cp_async_16_zfill(dst + row * D + swz(row, ch) * 8,
                              src + static_cast<size_t>(ok ? r0 + row : 0) * D + ch * 8, ok);
        }
    };
    load_tile(smem, Q + static_cast<size_t>(bh) * S_q * D, q0, BM, S_q);
    load_tile(smem + BM * D, dO + static_cast<size_t>(bh) * S_q * D, q0, BM, S_q);
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
    unsigned qf[KT][4], df[KT][4];
    {
        const int row = warp * 16 + (lane & 15);
#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
            ldmatrix_x4(qf[kk], smem + row * D + swz(row, 2 * kk + (lane >> 4)) * 8);
            ldmatrix_x4(df[kk], smem + BM * D + row * D + swz(row, 2 * kk + (lane >> 4)) * 8);
        }
    }
    const float2 l_lo = ld[static_cast<size_t>(bh) * S_pad + q0 + warp * 16 + g];
    const float2 l_hi = ld[static_cast<size_t>(bh) * S_pad + q0 + warp * 16 + g + 8];
    __syncthreads();  // stages 0 and 1 are free again

    const int kv_end = causal ? min(S_kv, q0 + BM) : S_kv;
    const int T = cdiv(kv_end, BN);
    auto load_kv = [&](int stage, int t) {
        bf16* ks = smem + stage * STAGE;
        load_tile(ks, Kg, t * BN, BN, S_kv);
        load_tile(ks + BN * D, Vg, t * BN, BN, S_kv);
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < T) load_kv(s, s);
        cp_async_commit();
    }

    float dq[DT][4];
#pragma unroll
    for (int dj = 0; dj < DT; ++dj)
#pragma unroll
        for (int e = 0; e < 4; ++e) dq[dj][e] = 0.f;

    for (int t = 0; t < T; ++t) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            const int nt = t + STAGES - 1;
            if (nt < T) load_kv(nt % STAGES, nt);
            cp_async_commit();
        }
        const bf16* ks = smem + (t % STAGES) * STAGE;
        const bf16* vs = ks + BN * D;

        float s[NT][4], dp[NT][4];
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) s[n][e] = dp[n][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
#pragma unroll
            for (int n = 0; n < NT; n += 2) {
                const int row = n * 8 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4(r, ks + row * D + swz(row, 2 * kk + (lane >> 4)) * 8);
                const unsigned b0[2] = {r[0], r[2]};
                const unsigned b1[2] = {r[1], r[3]};
                mma_bf16_16816(s[n], qf[kk], b0);
                mma_bf16_16816(s[n + 1], qf[kk], b1);
                ldmatrix_x4(r, vs + row * D + swz(row, 2 * kk + (lane >> 4)) * 8);
                const unsigned v0[2] = {r[0], r[2]};
                const unsigned v1[2] = {r[1], r[3]};
                mma_bf16_16816(dp[n], df[kk], v0);
                mma_bf16_16816(dp[n + 1], df[kk], v1);
            }
        }
        const int kv0 = t * BN;
        const bool mask = kv0 + BN > S_kv || (causal && kv0 + BN - 1 > q0 + warp * 16);
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 l = (e >> 1) ? l_hi : l_lo;
                float p = ex2(fmaf(s[n][e], scale_log2, -l.x));
                if (mask) {
                    const int i = q0 + warp * 16 + g + (e >> 1) * 8;
                    const int j = kv0 + n * 8 + c2 + (e & 1);
                    if (j >= S_kv || (causal && j > i)) p = 0.f;
                }
                s[n][e] = p * (dp[n][e] - l.y);
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
                ldmatrix_x4_trans(r, ks + row * D + swz(row, dj + (lane >> 4)) * 8);
                const unsigned b0[2] = {r[0], r[1]};
                const unsigned b1[2] = {r[2], r[3]};
                mma_bf16_16816(dq[dj], pa[kt], b0);
                mma_bf16_16816(dq[dj + 1], pa[kt], b1);
            }
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const int row = q0 + warp * 16 + g + 8 * h;
        if (row >= S_q) continue;
        bf16* out = dQ + (static_cast<size_t>(bh) * S_q + row) * D + c2;
#pragma unroll
        for (int dj = 0; dj < DT; ++dj)
            *reinterpret_cast<unsigned*>(out + dj * 8) =
                pack_bf16x2(dq[dj][2 * h] * scale, dq[dj][2 * h + 1] * scale);
    }
}

}  // namespace dqp

// ---------------------------------------------------------------------------------------
// Host side.
// ---------------------------------------------------------------------------------------

// One buffer per purpose per process, grown on demand, shared by every stream (the same
// caveat as the forward's split workspace: two backward launches must not run concurrently
// on different streams).
struct Buffer {
    void* p = nullptr;
    size_t bytes = 0;
};
template <typename T>
T* buffer(Buffer& b, size_t count) {
    const size_t need = std::max<size_t>(count * sizeof(T), 256);
    if (b.bytes < need) {
        if (b.p) SPARK_CUDA_CHECK(cudaFree(b.p));
        SPARK_CUDA_CHECK(cudaMalloc(&b.p, need));
        b.bytes = need;
    }
    return static_cast<T*>(b.p);
}
Buffer g_ld, g_dq, g_dkv;

// Knobs, read once: SPARK_ATTN_BWD_SPLIT = n forces every key tile of variants 1 and 2 into n
// slices (0: the default rule, which splits only the tail of the last wave).
int env_chunk() {
    static int v = -2;
    if (v == -2) {
        v = 0;
        if (const char* e = std::getenv("SPARK_ATTN_BWD_CHUNK")) v = std::atoi(e);
    }
    return v;
}

int env_split() {
    static int v = -2;
    if (v == -2) {
        v = 0;
        if (const char* e = std::getenv("SPARK_ATTN_BWD_SPLIT")) v = std::atoi(e);
    }
    return v;
}

// The block scheduler hands blocks out in index order as SMs free up, so for one resident
// block per SM a grid runs like greedy list scheduling. This simulates it for a split s1 (and
// the s2 cut of the last partial wave) with each block costing its Q tiles plus 0.3 of one
// for the prologue and epilogue, and returns the makespan in Q-tile iterations.
double simulate_grid(int bkv_count, int kv_tiles, int chunk, int group, int q_tiles,
                     int tiles_per_kv, bool causal, int s1, int min_iters, int resident) {
    std::vector<double> items;
    items.reserve(static_cast<size_t>(bkv_count) * kv_tiles * s1);
    for (int t = 0; t < bkv_count * kv_tiles; ++t) {
        int bkv, kt;
        kv::tile_coords(t, bkv_count, kv_tiles, chunk, bkv, kt);
        const int n = group * (q_tiles - (causal ? std::min(kt * tiles_per_kv, q_tiles) : 0));
        for (int sub = 0; sub < s1; ++sub) items.push_back((sub + 1) * n / s1 - sub * n / s1);
    }
    const int n = static_cast<int>(items.size());
    const int tail = n % resident;
    const int s2 = tail ? std::max(1, std::min(min_iters / s1, resident / tail)) : 1;
    std::priority_queue<double, std::vector<double>, std::greater<double>> ends;
    for (int i = 0; i < resident; ++i) ends.push(0.0);
    auto run = [&](double w) {
        const double e = ends.top();
        ends.pop();
        ends.push(e + w + 0.3);
    };
    for (int i = 0; i < n; ++i) {
        if (s2 > 1 && i >= n - tail)
            for (int p = 0; p < s2; ++p) run(items[i] / s2);
        else
            run(items[i]);
    }
    double mk = 0;
    while (!ends.empty()) {
        mk = std::max(mk, ends.top());
        ends.pop();
    }
    return mk;
}

// The split s1 in 1..8 with the shortest estimated time: the simulated grid at about 4 us
// per Q-tile iteration (10 BN BQ D FLOPs at the 1.3 TFLOPS an SM sustains here), plus the
// fp32 round trip of every split tile's dK and dV at 1.5 TB/s: a memset and a conversion read
// of its 2 BN D floats, and one atomic add of them per slice. s1 > 1 splits every tile, so at
// short sequences
// that traffic outweighs any balance it buys; under GQA a key tile carries `group` times the
// work for the same traffic and the split pays. Cached per shape.
int pick_s1(int bkv_count, int kv_tiles, int chunk, int group, int q_tiles, int BN, int BQ, int D,
            bool causal, int min_iters, int resident) {
    static std::map<std::array<int, 10>, int> cache;
    const std::array<int, 10> key = {bkv_count, kv_tiles, chunk, group,          q_tiles,
                                     BN,        BQ,       D,     causal ? 1 : 0, resident};
    auto it = cache.find(key);
    if (it != cache.end()) return it->second;
    const double it_us = 10.0 * BN * BQ * D / 1.3e6;
    const double pass_us = 2.0 * BN * D * 4 / 1.5e6;  // one pass over a tile's dK and dV
    const int tiles = bkv_count * kv_tiles;
    int best_s1 = 1;
    double best = 0;
    for (int c1 = 1; c1 <= std::max(1, std::min(8, min_iters)); ++c1) {
        const double mk = simulate_grid(bkv_count, kv_tiles, chunk, group, q_tiles, BN / BQ, causal,
                                        c1, min_iters, resident);
        const int tail = (tiles * c1) % resident;
        const int c2 = tail ? std::max(1, std::min(min_iters / c1, resident / tail)) : 1;
        const double ws = c1 > 1 ? tiles * (2.0 + c1) : (c2 > 1 ? tail * (2.0 + c2) : 0.0);
        const double t = mk * it_us + ws * pass_us;
        if (c1 == 1 || t < best) {
            best = t;
            best_s1 = c1;
        }
    }
    cache.emplace(key, best_s1);
    return best_s1;
}

struct Args {
    const bf16 *Q, *K, *V, *dO;
    const float2* ld;
    float* dq_acc;
    bf16 *dQ, *dK, *dV;
    int B, H_q, H_kv, S_q, S_kv, S_pad;
    float scale_log2, scale;
    bool causal;
};

// The schedule of a key-tile kernel with BN keys and BQ-row Q tiles, its workspace zeroed on
// `stream`; `ws_tiles` is the number of tiles the split-convert kernel must finish.
kv::Sched build_sched(const Args& a, int BN, int BQ, int D, bool split_tail, cudaStream_t stream,
                      int& ws_tiles, int& grid) {
    kv::Sched s;
    s.bkv_count = a.B * a.H_kv;
    s.H_q = a.H_q;
    s.H_kv = a.H_kv;
    s.group = a.H_q / a.H_kv;
    s.q_tiles = cdiv(a.S_q, BQ);
    s.S_pad = a.S_pad;
    const int kv_tiles = cdiv(a.S_kv, BN);
    const int tiles = s.bkv_count * kv_tiles;
    s.kv_tiles = kv_tiles;
    // Chunks of heads with at least two waves of tiles each (SPARK_ATTN_BWD_CHUNK overrides;
    // a chunk of every head is the plain heaviest-first order).
    {
        const int forced = env_chunk();
        s.chunk = forced > 0 ? forced : s.bkv_count;
        s.chunk = std::max(1, std::min(s.chunk, s.bkv_count));
    }
    // Iterations of the lightest tile (the last key tile under the causal mask): no item or
    // piece may be empty.
    const int last_k0 = (kv_tiles - 1) * BN;
    const int min_iters =
        s.group * (s.q_tiles - (a.causal ? std::min(last_k0 / BQ, s.q_tiles) : 0));
    const int resident = num_sms();
    int s1 = 1;
    if (split_tail) {
        const int forced = env_split();
        if (forced > 0)
            s1 = std::min(forced, std::max(min_iters, 1));
        else
            s1 = pick_s1(s.bkv_count, kv_tiles, s.chunk, s.group, s.q_tiles, BN, BQ, D, a.causal,
                         min_iters, resident);
    }
    s.s1 = s1;
    s.n_items = tiles * s1;
    s.tail = split_tail ? s.n_items % resident : 0;
    s.s2 = s.tail ? std::max(1, std::min(min_iters / s1, resident / s.tail)) : 1;
    if (s.s2 == 1) s.tail = 0;
    s.ws_tile0 = s1 > 1 ? 0 : tiles - s.tail;
    ws_tiles = s1 > 1 ? tiles : s.tail;
    s.ws = nullptr;
    if (ws_tiles > 0) {
        const size_t floats = static_cast<size_t>(ws_tiles) * 2 * BN * D;
        s.ws = buffer<float>(g_dkv, floats);
        SPARK_CUDA_CHECK(cudaMemsetAsync(s.ws, 0, floats * sizeof(float), stream));
    }
    grid = s.n_items - s.tail + s.tail * s.s2;
    return s;
}

template <int D, int BN>
void launch_split_convert(const Args& a, const kv::Sched& s, int ws_tiles, cudaStream_t stream) {
    if (ws_tiles == 0) return;
    const dim3 cgrid(cdiv(2 * BN * (D / 8), 256), ws_tiles);
    kv::bwd_split_convert_kernel<D, BN><<<cgrid, 256, 0, stream>>>(a.dK, a.dV, a.S_kv, s);
    SPARK_CHECK_LAUNCH();
}

template <int D, int WARPS, int BQ, bool PIPE, bool VREG, bool DQ, bool SKIP, bool DEFER>
void launch_kv(const Args& a, bool split_tail, cudaStream_t stream) {
    using C = kv::Cfg<D, WARPS, BQ, PIPE, VREG, DQ, DEFER>;
    auto kernel = kv::bwd_kv_kernel<D, WARPS, BQ, PIPE, VREG, DQ, SKIP, DEFER>;
    static bool opted = false;
    if (!opted) {
        SPARK_CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
        opted = true;
    }
    int ws_tiles = 0, grid = 0;
    const kv::Sched s = build_sched(a, C::BN, BQ, D, split_tail, stream, ws_tiles, grid);
    kernel<<<grid, C::THREADS, C::SMEM, stream>>>(a.Q, a.K, a.V, a.dO, a.ld, a.dq_acc, a.dK, a.dV,
                                                  a.S_q, a.S_kv, a.scale_log2, a.scale,
                                                  a.causal ? 1 : 0, s);
    SPARK_CHECK_LAUNCH();
    launch_split_convert<D, C::BN>(a, s, ws_tiles, stream);
}

template <int D, int BQ, bool DQ>
void launch_kv_tma(const Args& a, bool split_tail, cudaStream_t stream) {
    using C = kvt::Cfg<D, BQ, DQ>;
    auto kernel = kvt::bwd_kv_tma_kernel<D, BQ, DQ, true>;
    static bool opted = false;
    if (!opted) {
        SPARK_CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
        opted = true;
    }
    // [B*H_q][S_q][D] maps with boxes of 64 columns (one 128-byte swizzle span) by BQ rows;
    // rows past S_q are zero-filled by the copy engine.
    const uint64_t BHq = static_cast<uint64_t>(a.B) * a.H_q;
    const CUtensorMap tmQ =
        make_tensor_map_3d_bf16(a.Q, D, a.S_q, BHq, kvt::BOX, BQ, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap tmO =
        make_tensor_map_3d_bf16(a.dO, D, a.S_q, BHq, kvt::BOX, BQ, CU_TENSOR_MAP_SWIZZLE_128B);
    int ws_tiles = 0, grid = 0;
    const kv::Sched s = build_sched(a, kvt::BN, BQ, D, split_tail, stream, ws_tiles, grid);
    kernel<<<grid, kvt::THREADS, C::SMEM, stream>>>(tmQ, tmO, a.K, a.V, a.ld, a.dq_acc, a.dK, a.dV,
                                                    a.S_q, a.S_kv, a.scale_log2, a.scale,
                                                    a.causal ? 1 : 0, s);
    SPARK_CHECK_LAUNCH();
    launch_split_convert<D, kvt::BN>(a, s, ws_tiles, stream);
}

template <int D>
void launch_dq_pass(const Args& a, cudaStream_t stream) {
    constexpr int bytes = dqp::smem_bytes<D>();
    static bool opted = false;
    if (!opted) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(dqp::bwd_dq_kernel<D>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        opted = true;
    }
    const int bh_count = a.B * a.H_q;
    const int q_tiles = cdiv(a.S_q, dqp::BM);
    dqp::bwd_dq_kernel<D><<<bh_count * q_tiles, dqp::THREADS, bytes, stream>>>(
        a.Q, a.K, a.V, a.dO, a.ld, a.dQ, a.H_q, a.H_kv, a.S_q, a.S_kv, a.S_pad, bh_count, q_tiles,
        a.scale_log2, a.scale, a.causal ? 1 : 0);
    SPARK_CHECK_LAUNCH();
}

}  // namespace

int attention_bwd_num_variants() {
    return 4;
}

bool attention_bwd_supports(int S_q, int S_kv, int D, int variant) {
    if (variant < 0 || variant >= attention_bwd_num_variants()) return false;
    if (S_q <= 0 || S_kv <= 0) return false;
    return D == 64 || D == 128;
}

void attention_bwd_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                        const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* lse,
                        __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV, int B, int H_q,
                        int H_kv, int S_q, int S_kv, int D, bool causal, bool deterministic,
                        int variant, cudaStream_t stream) {
    SPARK_REQUIRE(Q && K && V && O && dO && lse && dQ && dK && dV, "attention_bwd: null pointer");
    SPARK_REQUIRE(B > 0 && H_q > 0 && H_kv > 0 && S_q > 0 && S_kv > 0,
                  "attention_bwd: B, H_q, H_kv, S_q, S_kv must be positive");
    SPARK_REQUIRE(H_q % H_kv == 0, "attention_bwd: H_q must be a multiple of H_kv");
    SPARK_REQUIRE(D == 64 || D == 128, "attention_bwd: D must be 64 or 128");
    SPARK_REQUIRE(variant >= 0 && variant < attention_bwd_num_variants(),
                  "attention_bwd: unknown variant");
    SPARK_REQUIRE(is_aligned16(Q) && is_aligned16(K) && is_aligned16(V) && is_aligned16(O) &&
                      is_aligned16(dO) && is_aligned16(dQ) && is_aligned16(dK) && is_aligned16(dV),
                  "attention_bwd: tensors must be 16-byte aligned");
    SPARK_REQUIRE(static_cast<int64_t>(B) * H_q * std::max(S_q, S_kv) * D < (int64_t{1} << 40),
                  "attention_bwd: tensor too large");

    const int S_pad = cdiv(S_q, kRowPad) * kRowPad;
    const int rows_pad = B * H_q * S_pad;
    const size_t n_q = static_cast<size_t>(B) * H_q * S_q * D;
    const bool atomic_dq = variant >= 1 && !deterministic;

    Args a;
    a.Q = Q;
    a.K = K;
    a.V = V;
    a.dO = dO;
    a.dQ = dQ;
    a.dK = dK;
    a.dV = dV;
    a.B = B;
    a.H_q = H_q;
    a.H_kv = H_kv;
    a.S_q = S_q;
    a.S_kv = S_kv;
    a.S_pad = S_pad;
    a.scale = 1.f / std::sqrt(static_cast<float>(D));
    a.scale_log2 = kLog2e * a.scale;
    a.causal = causal;
    float2* ld = buffer<float2>(g_ld, rows_pad);
    a.ld = ld;
    a.dq_acc = atomic_dq ? buffer<float>(g_dq, n_q) : nullptr;

    {
        constexpr int threads = 256;
        const int grid = cdiv(rows_pad, threads / 32);
        if (D == 64)
            bwd_preprocess_kernel<64>
                <<<grid, threads, 0, stream>>>(O, dO, lse, ld, a.dq_acc, S_q, S_pad, rows_pad);
        else
            bwd_preprocess_kernel<128>
                <<<grid, threads, 0, stream>>>(O, dO, lse, ld, a.dq_acc, S_q, S_pad, rows_pad);
        SPARK_CHECK_LAUNCH();
    }

    auto run = [&](auto d) {
        constexpr int DD = decltype(d)::value;
        switch (variant) {
            case 0: {
                const dim3 gq(cdiv(S_q, v0::THREADS / 32), B * H_q);
                v0::bwd_v0_dq_kernel<DD>
                    <<<gq, v0::THREADS, 0, stream>>>(Q, K, V, dO, ld, dQ, H_q, H_kv, S_q, S_kv,
                                                     S_pad, a.scale_log2, a.scale, causal ? 1 : 0);
                SPARK_CHECK_LAUNCH();
                const dim3 gk(cdiv(S_kv, v0::THREADS / 32), B * H_kv);
                v0::bwd_v0_dkv_kernel<DD>
                    <<<gk, v0::THREADS, 0, stream>>>(Q, K, V, dO, ld, dK, dV, H_q, H_kv, S_q, S_kv,
                                                     S_pad, a.scale_log2, a.scale, causal ? 1 : 0);
                SPARK_CHECK_LAUNCH();
                return;
            }
            case 1:
                if (atomic_dq) {
                    launch_kv<DD, 4, 64, false, false, true, false, false>(a, false, stream);
                } else {
                    launch_dq_pass<DD>(a, stream);
                    launch_kv<DD, 4, 64, false, false, false, false, false>(a, false, stream);
                }
                break;
            case 3: {
                constexpr int BQ = DD == 128 ? 32 : 64;
                if (atomic_dq) {
                    launch_kv_tma<DD, BQ, true>(a, true, stream);
                } else {
                    launch_dq_pass<DD>(a, stream);
                    launch_kv_tma<DD, BQ, false>(a, false, stream);
                }
                break;
            }
            case 2: {
                constexpr int BQ = DD == 128 ? 32 : 64;
                if (atomic_dq) {
                    launch_kv<DD, 8, BQ, true, true, true, true, true>(a, true, stream);
                } else {
                    // No tail split either: its dK / dV slices are fp32 atomics too.
                    launch_dq_pass<DD>(a, stream);
                    launch_kv<DD, 8, BQ, true, true, false, true, false>(a, false, stream);
                }
                break;
            }
            default:
                return;
        }
        if (atomic_dq) {
            const size_t n8 = n_q / 8;
            const int grid = static_cast<int>(std::min<size_t>(cdiv64(n8, 256), 4 * num_sms()));
            bwd_convert_kernel<<<grid, 256, 0, stream>>>(a.dq_acc, dQ, n8, a.scale);
            SPARK_CHECK_LAUNCH();
        }
    };
    if (D == 64)
        run(std::integral_constant<int, 64>{});
    else
        run(std::integral_constant<int, 128>{});
}

}  // namespace spark
