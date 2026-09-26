// Flash-decoding attention for Blackwell sm_12x: a few query rows against a long K/V cache,
// where the whole job is to stream K and V from DRAM once at the copy roof. Reached from
// attention variant 3's dispatch (attention.cu) whenever the query rows that share one K/V
// head, (H_q / H_kv) * S_q of them, fit one 16-row tile. Measurements and the arithmetic
// behind the constants: docs/design/attention.md, "Long-context decode".
//
// What is different from the 64-row tile of attention.cu on these shapes:
//   * one block per (b, kv head, key slice). The block's Q tile is the (H_q / H_kv) query
//     heads that read this K/V head, times the S_q tokens, laid out as rows hg * S_q + t;
//     because those heads are adjacent in Q the tile is one contiguous slab. K and V are
//     read once per group of heads instead of once per query head;
//   * a 16-row tile: one m16 mma row block, so the tensor work per key is a quarter of the
//     64-row tile's and a single query (or four GQA heads) pays for 16 rows, not 64;
//   * the four warps of a block split the block's keys, each with its own 3-stage cp.async
//     pipeline in a private 24 KB of shared memory and no block barrier in the loop. Each
//     warp keeps two 8 KB slabs of K+V in flight, 64 KB per SM, against the ~9 KB per SM the
//     card needs (docs/design/hgemm.md, "Bytes in flight");
//   * the keys of a (b, kv head) are split over `split` blocks so the grid fills the SMs at
//     any cache length; the warps' partials (unnormalized O, running max m and sum l) are
//     merged in shared memory, and the block partials in a workspace by the last block of the
//     (b, kv head) to arrive (a per-head counter, the pattern of hgemm_decode.cu), so a call
//     is one launch with no combine kernel and no memset. The merge rule is the online
//     softmax's own: M = max m_i, L = sum l_i 2^(m_i - M), O = sum O_i 2^(m_i - M) / L.
#include <algorithm>
#include <cstdlib>

#include "attention_internal.cuh"
#include "spark/kernels.h"

namespace spark::attn {

namespace {

constexpr int WARPS = 4;
constexpr int THREADS = WARPS * 32;
constexpr int ROWS = 16;            // one m16 tile of query rows per block
constexpr int kTargetBlocks = 128;  // blocks the split aims for (the sweep in the design doc)
constexpr int kMaxSplit = 256;      // the merge parks 3 * split * ROWS floats in shared memory

// A slab is 8 KB of K plus V: 16 keys at D = 128, 32 at D = 64. Three per warp in flight.
template <int D>
constexpr int slab_keys() {
    return 2048 / D;
}
template <int D>
constexpr int slab_elems() {
    return 2 * slab_keys<D>() * D;  // K slab then V slab, bf16 elements
}
constexpr int STAGES = 3;
template <int D>
constexpr int smem_bytes() {
    return WARPS * STAGES * slab_elems<D>() * 2;  // 96 KB
}
// One partial: ROWS unnormalized fp32 O rows, then m, then l.
template <int D>
constexpr int partial_floats() {
    return ROWS * (D + 2);
}

struct Params {
    const bf16* Q;
    const bf16* K;
    const bf16* V;
    bf16* O;
    int H_q, H_kv, S_q, S_kv;
    int rows;    // (H_q / H_kv) * S_q: the live rows of the tile
    int kv_end;  // keys any row can see: S_kv, or S_q under the (top-left) causal mask
    int split;   // blocks per (b, kv head); block = bkv * split + slice
    int causal;
    float scale_log2;
    float* ws;      // B * H_kv * split partials of partial_floats<D>() each (split > 1 only)
    int* counters;  // one per (b, kv head), zero between launches
};

template <int D>
__global__ void __launch_bounds__(THREADS, 1) attention_decode_kernel(Params p) {
    constexpr int BN = slab_keys<D>();
    constexpr int CH = D / 8;                // 16-byte chunks per row
    constexpr int KT = D / 16;               // k16 steps of Q K^T
    constexpr int DT = D / 8;                // n8 tiles of O
    constexpr int NT = BN / 8;               // n8 tiles of S
    constexpr int PT = BN / 16;              // k16 steps of P V
    constexpr int KV_ITERS = BN * CH / 32;   // chunks per lane per operand per slab: 8
    constexpr int Q_ITERS = ROWS * CH / 32;  // chunks per lane of the Q tile: 8, 4
    static_assert(KV_ITERS * 32 == BN * CH && Q_ITERS * 32 == ROWS * CH, "");
    static_assert(ROWS * D <= slab_elems<D>(), "the Q tile is staged through one slab");
    static_assert(partial_floats<D>() * 4 <= STAGES * slab_elems<D>() * 2,
                  "a warp's partial is parked in its own pipeline region");

    extern __shared__ __align__(128) unsigned char smem_raw[];
    __shared__ int s_last;
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    // This warp's private pipeline: STAGES slabs of K+V, nothing else touches it.
    bf16* wsm = reinterpret_cast<bf16*>(smem_raw) + warp * STAGES * slab_elems<D>();

    const int bkv = blockIdx.x / p.split, slice = blockIdx.x - bkv * p.split;
    const int b = bkv / p.H_kv, kv = bkv - b * p.H_kv;
    const int group = p.H_q / p.H_kv;
    // Query heads kv*group .. kv*group+group-1 of batch b are consecutive in Q, so the tile's
    // rows hg * S_q + t are the contiguous slab starting at head kv*group, token 0.
    const size_t q_off =
        (static_cast<size_t>(b) * p.H_q + static_cast<size_t>(kv) * group) * p.S_q * D;
    const bf16* Qg = p.Q + q_off;
    bf16* Og = p.O + q_off;
    const bf16* Kg = p.K + static_cast<size_t>(bkv) * p.S_kv * D;
    const bf16* Vg = p.V + static_cast<size_t>(bkv) * p.S_kv * D;

    // Slabs [sb, se) of this warp: the block's share of the (b, kv head)'s slabs, then this
    // warp's share of the block's. Contiguous ranges, so each warp streams whole rows of K/V.
    const int nslab = cdiv(p.kv_end, BN);
    const int bb = slice * nslab / p.split, be = (slice + 1) * nslab / p.split;
    const int sb = bb + warp * (be - bb) / WARPS, se = bb + (warp + 1) * (be - bb) / WARPS;

    auto load = [&](int stage, int s) {
        bf16* ks = wsm + stage * slab_elems<D>();
        bf16* vs = ks + BN * D;
#pragma unroll
        for (int i = 0; i < KV_ITERS; ++i) {
            const int c = lane + i * 32;
            const int row = c / CH, ch = c % CH;
            const int j = s * BN + row;
            const bool ok = j < p.S_kv;
            const size_t off = static_cast<size_t>(ok ? j : 0) * D + ch * 8;
            cp_async_16_zfill(ks + row * D + swz(row, ch) * 8, Kg + off, ok);
            cp_async_16_zfill(vs + row * D + swz(row, ch) * 8, Vg + off, ok);
        }
    };

    // Prologue: the Q tile (rows past `rows` zero-filled) through the last stage, and the
    // first STAGES-1 slabs into the others, all in flight together, so the first K/V bytes
    // do not wait for Q's DRAM round trip. Three commit groups: Q, slab sb, slab sb+1.
    bf16* qs = wsm + (STAGES - 1) * slab_elems<D>();
#pragma unroll
    for (int i = 0; i < Q_ITERS; ++i) {
        const int c = lane + i * 32;
        const int row = c / CH, ch = c % CH;
        const bool ok = row < p.rows;
        cp_async_16_zfill(qs + row * D + swz(row, ch) * 8,
                          Qg + static_cast<size_t>(ok ? row : 0) * D + ch * 8, ok);
    }
    cp_async_commit();
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (sb + s < se) load(s, sb + s);
        cp_async_commit();
    }
    cp_async_wait<STAGES - 1>();  // Q has landed for this lane
    __syncwarp();                 // ... for every lane
    unsigned qf[KT][4];
    {
        const int row = lane & 15;
#pragma unroll
        for (int kk = 0; kk < KT; ++kk)
            ldmatrix_x4(qf[kk], qs + row * D + swz(row, 2 * kk + (lane >> 4)) * 8);
    }
    __syncwarp();  // the last stage is free for slab sb + STAGES - 1

    float o[DT][4];
#pragma unroll
    for (int dj = 0; dj < DT; ++dj)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[dj][e] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

    for (int s = sb, it = 0; s < se; ++s, ++it) {
        cp_async_wait<STAGES - 2>();  // slab s has landed for this lane
        __syncwarp();                 // ... for every lane; and stage (it-1)%STAGES is free
        {
            const int ns = s + STAGES - 1;
            if (ns < se) load((it + STAGES - 1) % STAGES, ns);
            cp_async_commit();
        }
        const bf16* ks = wsm + (it % STAGES) * slab_elems<D>();
        const bf16* vs = ks + BN * D;

        // S = Q K^T: 16 rows x BN keys, NT n8 accumulators. K is the "col" operand as it lies.
        float sc[NT][4];
#pragma unroll
        for (int nj = 0; nj < NT; ++nj)
#pragma unroll
            for (int e = 0; e < 4; ++e) sc[nj][e] = 0.f;
#pragma unroll
        for (int kk = 0; kk < KT; ++kk) {
#pragma unroll
            for (int nj = 0; nj < NT; nj += 2) {
                const int row = nj * 8 + (lane & 15);
                unsigned r[4];
                ldmatrix_x4(r, ks + row * D + swz(row, 2 * kk + (lane >> 4)) * 8);
                const unsigned b0[2] = {r[0], r[2]};
                const unsigned b1[2] = {r[1], r[3]};
                mma_bf16_16816(sc[nj], qf[kk], b0);
                mma_bf16_16816(sc[nj + 1], qf[kk], b1);
            }
        }

        // Keys past S_kv (zero-filled) and, under the causal mask, keys past the row's token.
        const int kv0 = s * BN;
        if (kv0 + BN > p.S_kv || p.causal) {  // warp-uniform
#pragma unroll
            for (int nj = 0; nj < NT; ++nj)
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const int t = (g + (e >> 1) * 8) % p.S_q;  // row hg * S_q + t
                    const int j = kv0 + nj * 8 + c2 + (e & 1);
                    if (j >= p.S_kv || (p.causal && j > t)) sc[nj][e] = -INFINITY;
                }
        }

        // Online softmax, rows g (e = 0, 1) and g+8 (e = 2, 3), as attention.cu variant 2.
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            float mx = fmaxf(sc[0][2 * r], sc[0][2 * r + 1]);
#pragma unroll
            for (int nj = 1; nj < NT; ++nj) mx = fmaxf(mx, fmaxf(sc[nj][2 * r], sc[nj][2 * r + 1]));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 1));
            mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 2));
            const float m_new = fmaxf(m[r], mx * p.scale_log2);
            const float m_use = m_new == -INFINITY ? 0.f : m_new;  // no key seen yet
            const float alpha = ex2(m[r] - m_use);
            float rs = 0.f;
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                const float p0 = ex2(fmaf(sc[nj][2 * r], p.scale_log2, -m_use));
                const float p1 = ex2(fmaf(sc[nj][2 * r + 1], p.scale_log2, -m_use));
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

        // P as A fragments straight from the S accumulators, then O += P V.
        unsigned pa[PT][4];
#pragma unroll
        for (int kt = 0; kt < PT; ++kt) {
            pa[kt][0] = pack_bf16x2(sc[2 * kt][0], sc[2 * kt][1]);
            pa[kt][1] = pack_bf16x2(sc[2 * kt][2], sc[2 * kt][3]);
            pa[kt][2] = pack_bf16x2(sc[2 * kt + 1][0], sc[2 * kt + 1][1]);
            pa[kt][3] = pack_bf16x2(sc[2 * kt + 1][2], sc[2 * kt + 1][3]);
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
    __syncwarp();

    // Each warp parks its partial (unnormalized O, m, the merged l) at the start of its own
    // region, which its pipeline is done with; then the block merges the four per (row, chunk).
    // The fp32 rows are 512 B (256 B at D = 64), so the eight rows g = 0..7 a store touches
    // would share a bank group: 16-byte chunk c of row r sits at c ^ (r & 7), as in the
    // bf16 tiles. The global partials are plain row-major.
    constexpr int PF = partial_floats<D>();
    constexpr int WSTRIDE = STAGES * slab_elems<D>() / 2;  // floats between warp regions
    float* mine = reinterpret_cast<float*>(wsm);
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float ls = l[r];
        ls += __shfl_xor_sync(kFullMask, ls, 1);
        ls += __shfl_xor_sync(kFullMask, ls, 2);
        const int lrow = g + 8 * r;
#pragma unroll
        for (int dj = 0; dj < DT; ++dj)
            *reinterpret_cast<float2*>(mine + lrow * D + swz(lrow, dj * 2 + (c2 >> 2)) * 4 +
                                       (c2 & 3)) = make_float2(o[dj][2 * r], o[dj][2 * r + 1]);
        if ((lane & 3) == 0) {
            mine[ROWS * D + lrow] = m[r];
            mine[ROWS * D + ROWS + lrow] = ls;
        }
    }
    __syncthreads();

    const float* all = reinterpret_cast<const float*>(smem_raw);
    float* part = p.split > 1 ? p.ws + (static_cast<size_t>(bkv) * p.split + slice) * PF : nullptr;
    for (int i = tid; i < ROWS * CH; i += THREADS) {
        const int row = i / CH, ch = i % CH;
        if (row >= p.rows) continue;
        float M = -INFINITY;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) M = fmaxf(M, all[w * WSTRIDE + ROWS * D + row]);
        float L = 0.f, acc[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) {
            const float* pw = all + w * WSTRIDE;
            const float wt = ex2(pw[ROWS * D + row] - M);  // 0 for a warp that saw no key
            L = fmaf(pw[ROWS * D + ROWS + row], wt, L);
            const float4 a = *reinterpret_cast<const float4*>(pw + row * D + swz(row, 2 * ch) * 4);
            const float4 c =
                *reinterpret_cast<const float4*>(pw + row * D + swz(row, 2 * ch + 1) * 4);
            acc[0] = fmaf(a.x, wt, acc[0]);
            acc[1] = fmaf(a.y, wt, acc[1]);
            acc[2] = fmaf(a.z, wt, acc[2]);
            acc[3] = fmaf(a.w, wt, acc[3]);
            acc[4] = fmaf(c.x, wt, acc[4]);
            acc[5] = fmaf(c.y, wt, acc[5]);
            acc[6] = fmaf(c.z, wt, acc[6]);
            acc[7] = fmaf(c.w, wt, acc[7]);
        }
        if (p.split == 1) {
            const float inv = 1.f / L;
            uint4 u;
            u.x = pack_bf16x2(acc[0] * inv, acc[1] * inv);
            u.y = pack_bf16x2(acc[2] * inv, acc[3] * inv);
            u.z = pack_bf16x2(acc[4] * inv, acc[5] * inv);
            u.w = pack_bf16x2(acc[6] * inv, acc[7] * inv);
            *reinterpret_cast<uint4*>(Og + static_cast<size_t>(row) * D + ch * 8) = u;
        } else {
            *reinterpret_cast<float4*>(part + row * D + ch * 8) =
                make_float4(acc[0], acc[1], acc[2], acc[3]);
            *reinterpret_cast<float4*>(part + row * D + ch * 8 + 4) =
                make_float4(acc[4], acc[5], acc[6], acc[7]);
            if (ch == 0) {
                part[ROWS * D + row] = M;
                part[ROWS * D + ROWS + row] = L;
            }
        }
    }
    if (p.split == 1) return;

    // The fence orders every thread's partial stores before the arrival; the last slice of
    // this (b, kv head) to arrive merges the `split` partials and resets the counter.
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(p.counters + bkv, 1) == p.split - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    // Three passes so the loads of each are independent and in flight together: (m, l) of
    // every (slice, row) into shared memory; per row M, L and the slice weights
    // 2^(m_i - M); then the O rows, eight iterations of loads ahead of the FMAs. One block
    // reads split x 8 KB here (from L2, where the other blocks' stores landed), which is
    // why the split is capped and why this is not a loop of dependent round trips.
    const float* base = p.ws + static_cast<size_t>(bkv) * p.split * PF;
    float* sm = reinterpret_cast<float*>(smem_raw);  // [split][ROWS] m, then l, then weights
    float* sl = sm + p.split * ROWS;
    float* sw = sl + p.split * ROWS;
    float* sL = sw + p.split * ROWS;  // [ROWS]
#pragma unroll 4
    for (int i = tid; i < p.split * ROWS; i += THREADS) {
        const int k = i / ROWS, row = i - k * ROWS;
        sm[i] = __ldcg(base + k * PF + ROWS * D + row);
        sl[i] = __ldcg(base + k * PF + ROWS * D + ROWS + row);
    }
    __syncthreads();
    if (tid < ROWS) {
        float M = -INFINITY;
        for (int k = 0; k < p.split; ++k) M = fmaxf(M, sm[k * ROWS + tid]);
        float L = 0.f;
        for (int k = 0; k < p.split; ++k) {
            const float wt = ex2(sm[k * ROWS + tid] - M);  // 0 for a slice that saw no key
            sw[k * ROWS + tid] = wt;
            L = fmaf(sl[k * ROWS + tid], wt, L);
        }
        sL[tid] = L;
    }
    __syncthreads();
    for (int i = tid; i < ROWS * CH; i += THREADS) {
        const int row = i / CH, ch = i % CH;
        if (row >= p.rows) continue;
        float acc[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) acc[e] = 0.f;
#pragma unroll 8
        for (int k = 0; k < p.split; ++k) {
            const float wt = sw[k * ROWS + row];
            const float* pk = base + k * PF + row * D + ch * 8;
            const float4 a = __ldcg(reinterpret_cast<const float4*>(pk));
            const float4 c = __ldcg(reinterpret_cast<const float4*>(pk + 4));
            acc[0] = fmaf(a.x, wt, acc[0]);
            acc[1] = fmaf(a.y, wt, acc[1]);
            acc[2] = fmaf(a.z, wt, acc[2]);
            acc[3] = fmaf(a.w, wt, acc[3]);
            acc[4] = fmaf(c.x, wt, acc[4]);
            acc[5] = fmaf(c.y, wt, acc[5]);
            acc[6] = fmaf(c.z, wt, acc[6]);
            acc[7] = fmaf(c.w, wt, acc[7]);
        }
        const float inv = 1.f / sL[row];
        uint4 u;
        u.x = pack_bf16x2(acc[0] * inv, acc[1] * inv);
        u.y = pack_bf16x2(acc[2] * inv, acc[3] * inv);
        u.z = pack_bf16x2(acc[4] * inv, acc[5] * inv);
        u.w = pack_bf16x2(acc[6] * inv, acc[7] * inv);
        *reinterpret_cast<uint4*>(Og + static_cast<size_t>(row) * D + ch * 8) = u;
    }
    if (tid == 0) atomicExch(p.counters + bkv, 0);
}

// ---- host side ---------------------------------------------------------------------------

// Partials plus one counter per (b, kv head), zeroed when allocated and left zero by every
// launch. One per process, grown on demand; shared by every stream, so two split launches
// must not run concurrently on different streams (the same rule as variant 3's tail).
struct Workspace {
    float* ws = nullptr;
    int* counters = nullptr;
    size_t floats = 0;
    size_t heads = 0;
};

Workspace& workspace(size_t floats, size_t heads) {
    static Workspace w;
    if (w.floats < floats) {
        if (w.ws) SPARK_CUDA_CHECK(cudaFree(w.ws));
        w.floats = std::max(floats, w.floats * 2);
        SPARK_CUDA_CHECK(cudaMalloc(&w.ws, w.floats * sizeof(float)));
    }
    if (w.heads < heads) {
        if (w.counters) SPARK_CUDA_CHECK(cudaFree(w.counters));
        w.heads = std::max(heads, w.heads * 2);
        SPARK_CUDA_CHECK(cudaMalloc(&w.counters, w.heads * sizeof(int)));
        SPARK_CUDA_CHECK(cudaMemset(w.counters, 0, w.heads * sizeof(int)));
    }
    return w;
}

// Resident blocks per GPU (one per SM at 96 KB of shared memory); also the > 48 KB opt-in.
template <int D>
int resident_blocks() {
    constexpr int bytes = smem_bytes<D>();
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(attention_decode_kernel<D>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, attention_decode_kernel<D>, THREADS, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

// SPARK_ATTENTION_SPLIT=n forces the slices per (b, kv head), for the sweep in the design doc.
int env_split() {
    static int v = -1;
    if (v < 0) {
        v = 0;
        if (const char* e = std::getenv("SPARK_ATTENTION_SPLIT")) v = std::max(0, std::atoi(e));
    }
    return v;
}

template <int D>
void launch(const bf16* Q, const bf16* K, const bf16* V, bf16* O, int B, int H_q, int H_kv, int S_q,
            int S_kv, float scale_log2, bool causal, int split, cudaStream_t stream) {
    constexpr int bytes = smem_bytes<D>();
    const int resident = resident_blocks<D>();
    Params p;
    p.Q = Q;
    p.K = K;
    p.V = V;
    p.O = O;
    p.H_q = H_q;
    p.H_kv = H_kv;
    p.S_q = S_q;
    p.S_kv = S_kv;
    p.rows = (H_q / H_kv) * S_q;
    p.kv_end = causal ? std::min(S_kv, S_q) : S_kv;
    p.causal = causal ? 1 : 0;
    p.scale_log2 = scale_log2;
    const int heads = B * H_kv;
    const int nslab = cdiv(p.kv_end, slab_keys<D>());
    if (split <= 0) split = env_split();
    if (split <= 0) {
        // About 128 blocks over the (b, kv head) pairs, each with at least two slabs per warp.
        // The sweep in the design doc: 64 to 128 blocks read at the same rate, one block per
        // SM (168 at 8 heads) 1 to 4% slower, and past that the per-block fixed cost and the
        // merge of the partials show. Capped at the resident count for a smaller card.
        split = std::max(1, std::min(kTargetBlocks, resident) / heads);
        split = std::min(split, std::max(1, nslab / (2 * WARPS)));
    }
    split = std::max(1, std::min(split, std::min(nslab, kMaxSplit)));
    p.split = split;
    p.ws = nullptr;
    p.counters = nullptr;
    if (split > 1) {
        Workspace& w = workspace(static_cast<size_t>(heads) * split * partial_floats<D>(), heads);
        p.ws = w.ws;
        p.counters = w.counters;
    }
    attention_decode_kernel<D><<<heads * split, THREADS, bytes, stream>>>(p);
}

}  // namespace

bool decode_fits(int H_q, int H_kv, int S_q) {
    return H_kv > 0 && H_q % H_kv == 0 && (H_q / H_kv) * S_q <= ROWS;
}

void decode_launch(const bf16* Q, const bf16* K, const bf16* V, bf16* O, int B, int H_q, int H_kv,
                   int S_q, int S_kv, int D, float scale_log2, bool causal, int split,
                   cudaStream_t stream) {
    SPARK_REQUIRE(decode_fits(H_q, H_kv, S_q), "attention decode: query rows exceed the tile");
    if (D == 64)
        launch<64>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, split, stream);
    else
        launch<128>(Q, K, V, O, B, H_q, H_kv, S_q, S_kv, scale_log2, causal, split, stream);
}

}  // namespace spark::attn
