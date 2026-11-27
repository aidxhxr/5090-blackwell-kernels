// The attention core of Inkling 2 Mini (thinkingmachines/Inkling-2-Mini) for Blackwell sm_12x:
//
//   s_ij = q_i . k_j / sqrt(128) + b_ij,   b_ij = r_i . P[:, d] for d = pos_i - j < extent, else 0
//   O_i  = sum_j exp(s_ij) v_j / (exp(sink_i) + sum_j exp(s_ij))
//
// Q, O = [B, H_q, S_q, 128], K, V = [B, H_kv, S_kv, 128] (GQA, a cache of kv_cap rows per head),
// R = [B, H_q, S_q, 16] the per-query relative states, P = [16, extent] the layer's bias bank
// (one per layer, shared by every head), sinks fp32 [B, H_q, S_q] (one logit per query and head,
// no value). The queries are the last S_q positions: row i sits at pos_i = i + S_kv - S_q. The
// mask is causal, and a window W > 0 keeps the keys with pos_i - j < W (HF's sliding window
// overlay, kv_idx > q_idx - W). This is HF's eager_attention_forward of the model with the
// position bias, the mask and the sink column, in fp32 instead of bf16 logits.
//
// The bias is a Toeplitz product: row i needs the slice of R_i P at distances pos_i - j, a
// different slice per row. For a 64 x 64 tile those distances span 127 values, so each stage
// of the K/V pipeline also stages the P columns of that band (16 x 144 bf16, columns past
// `extent` zero-filled, which is the "zero beyond the extent" rule), each warp multiplies its
// 16 rows of R by the band on the tensor cores (one k16 step, 12 n8 tiles), parks the
// 16 x 96 fp32 result in shared memory and reads back the (i, j) entries of its score tile:
// a skew by one column per row. A tile whose distances all lie past the extent (most tiles
// of a full layer) stages no band and does no bias work. docs/design/inkling_attention.md has
// the arithmetic and the measurements.
//
//   prefill: one block per (b, q head, 64-row Q tile), 4 warps, 64-key tiles, two cp.async
//            stages of K, V and the band, the KV tiles before the first row's window skipped.
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
using attn::sink_norm;
using attn::swz;
using attn::window_lo;

constexpr int D = 128;     // head size
constexpr int DR = 16;     // relative state size, one k16 step
constexpr int CH = D / 8;  // 16-byte chunks per K/V row
constexpr int KT = D / 16;
constexpr int DT = D / 8;

// floor(x / 8) * 8 for negative x too: band bases are 8-aligned so every staged chunk of P is
// wholly inside or wholly outside [0, extent).
__host__ __device__ __forceinline__ int floor8(int x) {
    return x & ~7;
}

struct Args {
    const bf16* Q;
    const bf16* K;
    const bf16* V;
    const bf16* R;
    const bf16* P;
    const float* sinks;  // [B, H_q, S_q] or null
    bf16* O;
    int H_q, H_kv, S_q, S_kv, kv_cap, extent, window, off;
    float scale_log2;
};

// Stages the band of P columns [dbase, dbase + 8 * chunks) as [DR][stride] bf16, zero outside
// [0, extent). `n` threads from index `t`.
__device__ __forceinline__ void load_band(bf16* band, int stride, int chunks, const bf16* P,
                                          int extent, int dbase, int t, int n) {
    for (int c = t; c < DR * chunks; c += n) {
        const int k = c / chunks, ch = c - k * chunks;
        const int d = dbase + ch * 8;
        const bool ok = d >= 0 && d < extent;
        cp_async_16_zfill(band + k * stride + ch * 8,
                          P + static_cast<size_t>(k) * extent + (ok ? d : 0), ok);
    }
}

// R_rows . band for NT n8 tiles from column x0 of the staged band: the warp's 16 rows times
// 8 * NT distances, scaled into the log2 domain and stored to the warp's fp32 scratch
// sb[row][col] (row stride SB). rf is the warp's R tile as an m16n8k16 A fragment.
template <int NT, int SB>
__device__ __forceinline__ void band_bias(float* sb, const unsigned (&rf)[4], const bf16* band,
                                          int stride, int x0, int lane) {
    const int g = lane >> 2, c2 = (lane & 3) * 2;
#pragma unroll
    for (int n = 0; n < NT; n += 2) {
        unsigned r[4];
        ldmatrix_x4_trans(r, band + (lane & 15) * stride + x0 + (n + (lane >> 4)) * 8);
        const unsigned b0[2] = {r[0], r[1]};
        const unsigned b1[2] = {r[2], r[3]};
        float a0[4] = {0.f, 0.f, 0.f, 0.f}, a1[4] = {0.f, 0.f, 0.f, 0.f};
        mma_bf16_16816(a0, rf, b0);
        mma_bf16_16816(a1, rf, b1);
#pragma unroll
        for (int e = 0; e < 4; ++e) {
            const int row = g + (e >> 1) * 8, col = c2 + (e & 1);
            sb[row * SB + n * 8 + col] = a0[e] * kLog2e;
            sb[row * SB + (n + 1) * 8 + col] = a1[e] * kLog2e;
        }
    }
}

// The warp's 16 x 16 R tile (rows past `rows` zero) as an A fragment, straight from global.
__device__ __forceinline__ void load_r(unsigned (&rf)[4], const bf16* Rg, int rows, int lane) {
    const int g = lane >> 2, c2 = (lane & 3) * 2;
#pragma unroll
    for (int e = 0; e < 4; ++e) {
        const int row = g + (e & 1) * 8, k = c2 + (e >> 1) * 8;
        rf[e] = row < rows ? __ldg(reinterpret_cast<const unsigned*>(Rg + row * DR + k)) : 0u;
    }
}

// One row of the online softmax over NT n8 tiles of log2-domain scores (rows g: r = 0,
// g + 8: r = 1). Exponentiates s in place and rescales the O accumulator.
template <int NT>
__device__ __forceinline__ void softmax_row(float (&s)[NT][4], float (&o)[DT][4], float& m,
                                            float& l, int r) {
    float mx = fmaxf(s[0][2 * r], s[0][2 * r + 1]);
#pragma unroll
    for (int nj = 1; nj < NT; ++nj) mx = fmaxf(mx, fmaxf(s[nj][2 * r], s[nj][2 * r + 1]));
    mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 1));
    mx = fmaxf(mx, __shfl_xor_sync(kFullMask, mx, 2));
    const float m_new = fmaxf(m, mx);
    const float m_use = m_new == -INFINITY ? 0.f : m_new;  // no key seen yet
    const float alpha = ex2(m - m_use);
    float rs = 0.f;
#pragma unroll
    for (int nj = 0; nj < NT; ++nj) {
        const float p0 = ex2(s[nj][2 * r] - m_use);
        const float p1 = ex2(s[nj][2 * r + 1] - m_use);
        s[nj][2 * r] = p0;
        s[nj][2 * r + 1] = p1;
        rs += p0 + p1;
    }
    l = fmaf(l, alpha, rs);
    m = m_new;
#pragma unroll
    for (int dj = 0; dj < DT; ++dj) {
        o[dj][2 * r] *= alpha;
        o[dj][2 * r + 1] *= alpha;
    }
}

// S = Q K^T for 16 rows and 8 * NT keys of a swizzled [keys][D] tile.
template <int NT>
__device__ __forceinline__ void qk(float (&s)[NT][4], const unsigned (&qf)[KT][4], const bf16* ks,
                                   int lane) {
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
}

// O += P V for 16 rows, P the exponentiated scores of 8 * NT keys, V a swizzled [keys][D] tile.
template <int NT>
__device__ __forceinline__ void pv(float (&o)[DT][4], const float (&s)[NT][4], const bf16* vs,
                                   int lane) {
    constexpr int PT = NT / 2;
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

// ---- prefill: 64-row Q tiles ---------------------------------------------------------------
namespace pre {

constexpr int WARPS = 4, THREADS = WARPS * 32;
constexpr int BM = 16 * WARPS, BN = 64, STAGES = 2;
constexpr int NT = BN / 8;
// The block's band starts at floor8(dmin), dmin = q0 + off - kv0 - (BN - 1), and warp w's at
// 16 w past it; a warp's 16 rows and 64 keys need at most 79 + 7 distances, 12 n8 tiles.
constexpr int WTILES = 12;
constexpr int BAND = 16 * (WARPS - 1) + 8 * WTILES;  // 144 columns, 18 chunks per channel
// Scratch row stride: lane (g, c2) reads column (row - key) of row g, so word address
// g * (SB + 1) - c2 + const; SB + 1 = 8 (mod 32) makes that two lanes per bank at most.
constexpr int SB = 103;
constexpr int KV_ELEMS = BN * D;
constexpr int STAGE_ELEMS = 2 * KV_ELEMS + DR * BAND;
constexpr int SMEM = STAGES * STAGE_ELEMS * 2 + WARPS * 16 * SB * 4;  // 101120 B
static_assert(BM * D <= STAGE_ELEMS, "the Q tile is staged through one stage");
static_assert(WTILES * 8 <= SB, "");

__global__ void __launch_bounds__(THREADS, 1) kernel(Args a, int bh_count, int q_tiles) {
    constexpr int KV_ITERS = BN * CH / THREADS;  // 8
    constexpr int Q_ITERS = BM * CH / THREADS;   // 8
    extern __shared__ __align__(128) unsigned char smem_raw[];
    bf16* smem = reinterpret_cast<bf16*>(smem_raw);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, c2 = (lane & 3) * 2;
    float* sb = reinterpret_cast<float*>(smem_raw + STAGES * STAGE_ELEMS * 2) + warp * 16 * SB;

    // Heaviest Q tiles first: under the causal mask the last tiles see the most keys.
    const int tile = blockIdx.x;
    const int bh = tile % bh_count;
    const int q0 = (q_tiles - 1 - tile / bh_count) * BM;
    const int bkv = kv_index(bh, a.H_q, a.H_kv);
    const bf16* Qg = a.Q + static_cast<size_t>(bh) * a.S_q * D;
    const bf16* Kg = a.K + static_cast<size_t>(bkv) * a.kv_cap * D;
    const bf16* Vg = a.V + static_cast<size_t>(bkv) * a.kv_cap * D;
    bf16* Og = a.O + static_cast<size_t>(bh) * a.S_q * D;
    const int wrow0 = q0 + warp * 16;  // the warp's first query row

#pragma unroll
    for (int i = 0; i < Q_ITERS; ++i) {
        const int c = tid + i * THREADS;
        const int row = c / CH, ch = c % CH;
        const bool ok = q0 + row < a.S_q;
        cp_async_16_zfill(smem + row * D + swz(row, ch) * 8,
                          Qg + static_cast<size_t>(ok ? q0 + row : 0) * D + ch * 8, ok);
    }
    cp_async_commit();
    unsigned rf[4];
    load_r(rf, a.R + (static_cast<size_t>(bh) * a.S_q + wrow0) * DR, a.S_q - wrow0, lane);
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

    // KV tiles [t0, t1): to the last row's causal bound, from the first row's window edge.
    const int off = a.off;
    const int t1 = cdiv(min(a.S_kv, q0 + BM + off), BN);
    const int t0 = window_lo(q0 + off, a.window) / BN;

    auto load = [&](int stage, int t) {
        bf16* ks = smem + stage * STAGE_ELEMS;
        bf16* vs = ks + KV_ELEMS;
#pragma unroll
        for (int i = 0; i < KV_ITERS; ++i) {
            const int c = tid + i * THREADS;
            const int row = c / CH, ch = c % CH;
            const int j = t * BN + row;
            const bool ok = j < a.S_kv;
            const size_t o = static_cast<size_t>(ok ? j : 0) * D + ch * 8;
            cp_async_16_zfill(ks + row * D + swz(row, ch) * 8, Kg + o, ok);
            cp_async_16_zfill(vs + row * D + swz(row, ch) * 8, Vg + o, ok);
        }
        // No band for a tile whose distances all lie past the extent: its bias is zero.
        const int dmin = q0 + off - t * BN - (BN - 1);
        if (dmin < a.extent)
            load_band(vs + KV_ELEMS, BAND, BAND / 8, a.P, a.extent, floor8(dmin), tid, THREADS);
    };
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (t0 + s < t1) load(s, t0 + s);
        cp_async_commit();
    }

    float o[DT][4];
#pragma unroll
    for (int dj = 0; dj < DT; ++dj)
#pragma unroll
        for (int e = 0; e < 4; ++e) o[dj][e] = 0.f;
    float m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};

    for (int t = t0, it = 0; t < t1; ++t, ++it) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();  // tile t is in for every thread, and the other stage is free
        {
            const int nt = t + STAGES - 1;
            if (nt < t1) load((it + STAGES - 1) % STAGES, nt);
            cp_async_commit();
        }
        const bf16* ks = smem + (it % STAGES) * STAGE_ELEMS;
        const bf16* vs = ks + KV_ELEMS;
        const bf16* band = vs + KV_ELEMS;
        const int kv0 = t * BN;

        float s[NT][4];
        qk<NT>(s, qf, ks, lane);

        // The bias of the warp's 16 x 64 scores: distances from dw = floor8(dmin of the warp),
        // which is 16 * warp columns into the block's band. None when dmin >= extent (most
        // of a full layer's tiles): then the block staged no band either.
        const int dmin = wrow0 + off - kv0 - (BN - 1);
        const int dw = floor8(dmin);
        const bool biased = dmin < a.extent;  // warp-uniform
        if (biased) {
            band_bias<WTILES, SB>(sb, rf, band, BAND, 16 * warp, lane);
            __syncwarp();
        }
        const bool edge = kv0 + BN > a.S_kv || kv0 + BN - 1 > wrow0 + off ||
                          (a.window > 0 && kv0 < wrow0 + 16 + off - a.window);
#pragma unroll
        for (int nj = 0; nj < NT; ++nj)
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int lr = g + (e >> 1) * 8;
                const int pos = wrow0 + lr + off;
                const int j = kv0 + nj * 8 + c2 + (e & 1);
                const float b = biased ? sb[lr * SB + (pos - j - dw)] : 0.f;
                s[nj][e] = fmaf(s[nj][e], a.scale_log2, b);
                if (edge && (j >= a.S_kv || j > pos || (a.window > 0 && j <= pos - a.window)))
                    s[nj][e] = -INFINITY;
            }

        softmax_row<NT>(s, o, m[0], l[0], 0);
        softmax_row<NT>(s, o, m[1], l[1], 1);
        pv<NT>(o, s, vs, lane);
    }
    cp_async_wait<0>();

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        float ls = l[r];
        ls += __shfl_xor_sync(kFullMask, ls, 1);
        ls += __shfl_xor_sync(kFullMask, ls, 2);
        const int row = wrow0 + g + 8 * r;
        if (row >= a.S_q) continue;
        float inv = 1.f / ls;
        if (a.sinks != nullptr)
            inv = sink_norm(m[r], ls,
                            __ldg(a.sinks + static_cast<size_t>(bh) * a.S_q + row) * kLog2e);
        bf16* out = Og + static_cast<size_t>(row) * D + c2;
#pragma unroll
        for (int dj = 0; dj < DT; ++dj)
            *reinterpret_cast<__nv_bfloat162*>(out + dj * 8) =
                __floats2bfloat162_rn(o[dj][2 * r] * inv, o[dj][2 * r + 1] * inv);
    }
}

void launch(const Args& a, int B, cudaStream_t stream) {
    static bool once = false;
    if (!once) {
        SPARK_CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM));
        once = true;
    }
    const int bh_count = B * a.H_q, q_tiles = cdiv(a.S_q, BM);
    kernel<<<bh_count * q_tiles, THREADS, SMEM, stream>>>(a, bh_count, q_tiles);
}

}  // namespace pre

}  // namespace

void inkling_attention_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                            const __nv_bfloat16* R, const __nv_bfloat16* P, const float* sinks,
                            __nv_bfloat16* O, int B, int H_q, int H_kv, int S_q, int S_kv,
                            int kv_cap, int extent, int window, int variant, cudaStream_t stream) {
    SPARK_REQUIRE(B > 0 && H_q > 0 && H_kv > 0 && S_q > 0 && S_kv > 0,
                  "inkling_attention: empty shape");
    SPARK_REQUIRE(H_q % H_kv == 0, "inkling_attention: H_q must be a multiple of H_kv");
    SPARK_REQUIRE(S_q <= S_kv, "inkling_attention: the queries are the last S_q of S_kv keys");
    SPARK_REQUIRE(kv_cap >= S_kv, "inkling_attention: kv_cap must be at least S_kv");
    SPARK_REQUIRE(extent > 0 && extent % 8 == 0,
                  "inkling_attention: extent must be a multiple of 8");
    SPARK_REQUIRE(window >= 0, "inkling_attention: window must be >= 0");
    Args a;
    a.Q = Q;
    a.K = K;
    a.V = V;
    a.R = R;
    a.P = P;
    a.sinks = sinks;
    a.O = O;
    a.H_q = H_q;
    a.H_kv = H_kv;
    a.S_q = S_q;
    a.S_kv = S_kv;
    a.kv_cap = kv_cap;
    a.extent = extent;
    a.window = window < S_kv ? window : 0;  // a window holding every key is no window
    a.off = S_kv - S_q;
    a.scale_log2 = kLog2e / sqrtf(static_cast<float>(D));
    SPARK_REQUIRE(variant == -1 || variant == 0, "inkling_attention: variant is 0");
    pre::launch(a, B, stream);
    SPARK_CUDA_CHECK(cudaGetLastError());
}

}  // namespace spark
