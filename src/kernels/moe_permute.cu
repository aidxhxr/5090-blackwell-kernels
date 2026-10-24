// The two data movements around an MoE layer's grouped GEMMs, both bf16 and 16 bytes per load:
//
//   moe_permute: xs[pos[t, s]] = x[t] for each of token t's k slots, pos[t, s] finished on the
//                way from the routing's (block base + rank) (moe_route.cu). A block takes four
//                tokens, reads each row once and writes it k times, so x is read once;
//   moe_combine: out[t] = sum_s w[t, s] * y[pos[t, s]], the k rows of each token gathered
//                from expert order, the weights applied and summed in fp32 in slot order and
//                rounded once. Each output element is one thread's sum in a fixed order, so
//                the result is the same bits every run; there is no scatter-add and no atomic.
//                A block per token, and each thread's k row loads go out four at a time:
//                four tokens per block with one load in flight per thread cost 11 to 13 us at
//                16 and 128 tokens on Qwen3, a chain of L2 round trips on four or 32 blocks.
//
// D (the row length) must be a multiple of 8 and at most 32 * 256 = 8192 (the per-thread
// vector count is a template parameter), rows 16-byte aligned.

#include <algorithm>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

namespace mp {

constexpr int THREADS = 256;
constexpr int TOK = 4;  // tokens per block
constexpr int MAX_K = 16;

__device__ __forceinline__ void fma8(float (&acc)[8], const bf16x8& v, float w) {
#pragma unroll
    for (int h = 0; h < 4; ++h) {
        const float2 f = __bfloat1622float2(v.h[h]);
        acc[2 * h] = fmaf(w, f.x, acc[2 * h]);
        acc[2 * h + 1] = fmaf(w, f.y, acc[2 * h + 1]);
    }
}

// VPT: 16-byte vectors per thread per row (DV = D / 8 <= VPT * THREADS).
template <int VPT>
__global__ void __launch_bounds__(THREADS)
    permute_kernel(const bf16x8* __restrict__ x, const int32_t* __restrict__ ids,
                   int32_t* __restrict__ pos, const int32_t* __restrict__ base,
                   bf16x8* __restrict__ xs, int T, int k, int E, int tb, int DV) {
    __shared__ int s_pos[TOK * MAX_K];
    const int tid = threadIdx.x;
    const int t0 = blockIdx.x * TOK;
    if (tid < TOK * k) {
        const int t = t0 + tid / k;
        if (t < T) {
            const size_t i = static_cast<size_t>(t0) * k + tid;
            const int p = base[static_cast<size_t>(t / tb) * E + ids[i]] + pos[i];
            pos[i] = p;
            s_pos[tid] = p;
        }
    }
    // The rows first, all of them in flight, then the k stores of each.
    bf16x8 v[TOK][VPT];
#pragma unroll
    for (int tt = 0; tt < TOK; ++tt)
#pragma unroll
        for (int j = 0; j < VPT; ++j) {
            const int c = tid + j * THREADS;
            if (t0 + tt < T && c < DV) v[tt][j] = x[static_cast<size_t>(t0 + tt) * DV + c];
        }
    __syncthreads();
#pragma unroll
    for (int tt = 0; tt < TOK; ++tt) {
        if (t0 + tt >= T) break;
        for (int s = 0; s < k; ++s) {
            bf16x8* dst = xs + static_cast<size_t>(s_pos[tt * k + s]) * DV;
#pragma unroll
            for (int j = 0; j < VPT; ++j) {
                const int c = tid + j * THREADS;
                if (c < DV) dst[c] = v[tt][j];
            }
        }
    }
}

template <int VPT>
__global__ void __launch_bounds__(THREADS)
    combine_kernel(const bf16x8* __restrict__ y, const int32_t* __restrict__ pos,
                   const float* __restrict__ w, bf16x8* __restrict__ out, int T, int k, int DV) {
    constexpr int GROUP = 4;  // slots whose rows are loaded before any is summed
    __shared__ int s_pos[MAX_K];
    __shared__ float s_w[MAX_K];
    const int tid = threadIdx.x;
    const int t = blockIdx.x;
    if (tid < k) {
        const size_t i = static_cast<size_t>(t) * k + tid;
        s_pos[tid] = pos[i];
        s_w[tid] = w[i];
    }
    __syncthreads();
    float acc[VPT][8];
#pragma unroll
    for (int j = 0; j < VPT; ++j)
#pragma unroll
        for (int q = 0; q < 8; ++q) acc[j][q] = 0.f;
    for (int s0 = 0; s0 < k; s0 += GROUP) {
        bf16x8 v[GROUP][VPT];
#pragma unroll
        for (int g = 0; g < GROUP; ++g) {
            if (s0 + g >= k) break;
            const bf16x8* src = y + static_cast<size_t>(s_pos[s0 + g]) * DV;
#pragma unroll
            for (int j = 0; j < VPT; ++j) {
                const int c = tid + j * THREADS;
                if (c < DV) v[g][j] = src[c];
            }
        }
#pragma unroll
        for (int g = 0; g < GROUP; ++g) {  // slot order, so the sum is the same every run
            if (s0 + g >= k) break;
#pragma unroll
            for (int j = 0; j < VPT; ++j)
                if (tid + j * THREADS < DV) fma8(acc[j], v[g][j], s_w[s0 + g]);
        }
    }
#pragma unroll
    for (int j = 0; j < VPT; ++j) {
        const int c = tid + j * THREADS;
        if (c < DV) {
            bf16x8 o;
#pragma unroll
            for (int h = 0; h < 4; ++h)
                o.h[h] = __floats2bfloat162_rn(acc[j][2 * h], acc[j][2 * h + 1]);
            out[static_cast<size_t>(t) * DV + c] = o;
        }
    }
}

}  // namespace mp

}  // namespace

bool moe_permute_supports(int D, int k) {
    return D >= 8 && D % 8 == 0 && D <= 8 * 4 * mp::THREADS && k >= 1 && k <= mp::MAX_K;
}

void moe_permute_bf16(const __nv_bfloat16* x, const int32_t* ids, int32_t* pos,
                      const MoeRoutePlan& plan, __nv_bfloat16* xs, int T, int E, int k, int D,
                      cudaStream_t stream) {
    SPARK_REQUIRE(moe_permute_supports(D, k), "moe_permute: D % 8 == 0, D <= 8192, k <= 16");
    SPARK_REQUIRE(is_aligned16(x) && is_aligned16(xs), "moe_permute: 16-byte aligned rows");
    if (T <= 0) return;
    const int DV = D / 8;
    const int grid = cdiv(T, mp::TOK);
    const auto* xv = reinterpret_cast<const bf16x8*>(x);
    auto* xsv = reinterpret_cast<bf16x8*>(xs);
    if (DV <= mp::THREADS)
        mp::permute_kernel<1>
            <<<grid, mp::THREADS, 0, stream>>>(xv, ids, pos, plan.base, xsv, T, k, E, plan.tb, DV);
    else if (DV <= 2 * mp::THREADS)
        mp::permute_kernel<2>
            <<<grid, mp::THREADS, 0, stream>>>(xv, ids, pos, plan.base, xsv, T, k, E, plan.tb, DV);
    else
        mp::permute_kernel<4>
            <<<grid, mp::THREADS, 0, stream>>>(xv, ids, pos, plan.base, xsv, T, k, E, plan.tb, DV);
    SPARK_CHECK_LAUNCH();
}

void moe_combine_bf16(const __nv_bfloat16* y, const int32_t* pos, const float* weights,
                      __nv_bfloat16* out, int T, int k, int D, cudaStream_t stream) {
    SPARK_REQUIRE(moe_permute_supports(D, k), "moe_combine: D % 8 == 0, D <= 8192, k <= 16");
    SPARK_REQUIRE(is_aligned16(y) && is_aligned16(out), "moe_combine: 16-byte aligned rows");
    if (T <= 0) return;
    const int DV = D / 8;
    const int grid = T;  // a block per token
    const auto* yv = reinterpret_cast<const bf16x8*>(y);
    auto* ov = reinterpret_cast<bf16x8*>(out);
    if (DV <= mp::THREADS)
        mp::combine_kernel<1><<<grid, mp::THREADS, 0, stream>>>(yv, pos, weights, ov, T, k, DV);
    else if (DV <= 2 * mp::THREADS)
        mp::combine_kernel<2><<<grid, mp::THREADS, 0, stream>>>(yv, pos, weights, ov, T, k, DV);
    else
        mp::combine_kernel<4><<<grid, mp::THREADS, 0, stream>>>(yv, pos, weights, ov, T, k, DV);
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
