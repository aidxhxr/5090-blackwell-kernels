// Inkling 2 Mini's block attention residuals
// (docs/design/attn_res.md has the math, the source counts the model reaches and the numbers).
//
// attn_res: for token t the sources are x_s = blocks[t, s, :] for s < nb and x_nb =
//   partial[t, :]. With n(x) = x * rsqrt(mean(x^2) + eps) (no weight) and the learned queries
//   Q [nq, H] fp32,
//       out[q, t, :] = sum_s softmax_s(Q[q] . n(x_s) / sqrt(H)) * n(x_s)
//   One block per token holds its slice of every source in registers. Since n(x_s) = r_s x_s,
//   the logit is r_s (Q[q] . x_s) / sqrt(H): the sums of squares and the raw dots come out of
//   the same block reduction, and the mix is one more pass over the registers. Every source is
//   read once per token, for all the queries. Block states bf16 (the model's dtype), the
//   partial state bf16 or fp32 (HF hands the MLP input's residual over in fp32), fp32 math,
//   fp32 out.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {
namespace {

__device__ __forceinline__ void load8(const float* p, float (&v)[8]) {
    const float4 a = *reinterpret_cast<const float4*>(p);
    const float4 b = *reinterpret_cast<const float4*>(p + 4);
    v[0] = a.x;
    v[1] = a.y;
    v[2] = a.z;
    v[3] = a.w;
    v[4] = b.x;
    v[5] = b.y;
    v[6] = b.z;
    v[7] = b.w;
}

__device__ __forceinline__ void load8(const __nv_bfloat16* p, float (&v)[8]) {
    const bf16x8 b = *reinterpret_cast<const bf16x8*>(p);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(b.h[i]);
        v[2 * i] = f.x;
        v[2 * i + 1] = f.y;
    }
}

__device__ __forceinline__ void store8(float* p, const float (&v)[8]) {
    *reinterpret_cast<float4*>(p) = make_float4(v[0], v[1], v[2], v[3]);
    *reinterpret_cast<float4*>(p + 4) = make_float4(v[4], v[5], v[6], v[7]);
}

// ---------------------------------------------------------------------------
// Block attention residual
// ---------------------------------------------------------------------------
constexpr int kQChunk = 4;  // queries per block reduction
constexpr int kAttnResThreads = 512;

// kS sources (kS - 1 block states, then the partial state). Thread i owns elements
// [8 (i + j blockDim), +8) of every row, j < kChunks.
template <typename TP, int kS, int kChunks>
__global__ void __launch_bounds__(kAttnResThreads)
    attn_res_kernel(const __nv_bfloat16* __restrict__ blocks, int64_t tok_stride,
                    int64_t blk_stride, const TP* __restrict__ partial,
                    const float* __restrict__ query, int nq, int q_per_block,
                    float* __restrict__ out, int T, int H, float eps) {
    constexpr int kNb = kS - 1;
    constexpr int kVals = kS * (kQChunk + 1);  // dots of a chunk + the sums of squares
    __shared__ float warp_part[kAttnResThreads / kWarpSize][kVals];
    __shared__ float total[kVals];
    const int t = blockIdx.x;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;

    float x[kS][kChunks][8];
#pragma unroll
    for (int j = 0; j < kChunks; ++j) {
        const int c = (threadIdx.x + j * blockDim.x) * 8;
#pragma unroll
        for (int s = 0; s < kS; ++s) {
            if (c < H) {
                if (s < kNb) {
                    load8(blocks + t * tok_stride + s * blk_stride + c, x[s][j]);
                } else {
                    load8(partial + static_cast<int64_t>(t) * H + c, x[s][j]);
                }
            } else {
#pragma unroll
                for (int e = 0; e < 8; ++e) x[s][j][e] = 0.0f;
            }
        }
    }

    float ssq[kS];
#pragma unroll
    for (int s = 0; s < kS; ++s) {
        float a = 0.0f;
#pragma unroll
        for (int j = 0; j < kChunks; ++j) {
#pragma unroll
            for (int e = 0; e < 8; ++e) a += x[s][j][e] * x[s][j][e];
        }
        ssq[s] = a;
    }

    const float scale = rsqrtf(static_cast<float>(H));
    float rinv[kS];
    // Block (t, y) serves queries [y q_per_block, (y + 1) q_per_block): all of them unless a
    // few tokens with many queries (the 48 message queries of a loop pass) need more blocks.
    const int q_begin = blockIdx.y * q_per_block;
    const int q_end = min(nq, q_begin + q_per_block);
    for (int q0 = q_begin; q0 < q_end; q0 += kQChunk) {
        const int nqc = min(kQChunk, q_end - q0);
        float dot[kQChunk][kS];
#pragma unroll
        for (int qi = 0; qi < kQChunk; ++qi) {
#pragma unroll
            for (int s = 0; s < kS; ++s) dot[qi][s] = 0.0f;
            if (qi < nqc) {
#pragma unroll
                for (int j = 0; j < kChunks; ++j) {
                    const int c = (threadIdx.x + j * blockDim.x) * 8;
                    if (c < H) {
                        float qv[8];
                        load8(query + static_cast<int64_t>(q0 + qi) * H + c, qv);
#pragma unroll
                        for (int s = 0; s < kS; ++s) {
#pragma unroll
                            for (int e = 0; e < 8; ++e) dot[qi][s] += qv[e] * x[s][j][e];
                        }
                    }
                }
            }
        }

        // One reduction for the chunk: dot (qi, s) at qi * kS + s, the sums of squares (first
        // chunk only) at kQChunk * kS + s.
#pragma unroll
        for (int s = 0; s < kS; ++s) {
#pragma unroll
            for (int qi = 0; qi < kQChunk; ++qi) {
                if (qi < nqc) {
                    const float v = warp_reduce_sum(dot[qi][s]);
                    if (lane == 0) warp_part[warp][qi * kS + s] = v;
                }
            }
            if (q0 == q_begin) {
                const float v = warp_reduce_sum(ssq[s]);
                if (lane == 0) warp_part[warp][kQChunk * kS + s] = v;
            }
        }
        __syncthreads();
        if (threadIdx.x < kVals) {
            float a = 0.0f;
            for (int w = 0; w < nwarps; ++w) a += warp_part[w][threadIdx.x];
            total[threadIdx.x] = a;
        }
        __syncthreads();
        if (q0 == q_begin) {
#pragma unroll
            for (int s = 0; s < kS; ++s) {
                rinv[s] = rsqrtf(total[kQChunk * kS + s] / static_cast<float>(H) + eps);
            }
        }

#pragma unroll
        for (int qi = 0; qi < kQChunk; ++qi) {
            if (qi >= nqc) break;
            float l[kS];
            float mx = -INFINITY;
#pragma unroll
            for (int s = 0; s < kS; ++s) {
                l[s] = total[qi * kS + s] * rinv[s] * scale;
                mx = fmaxf(mx, l[s]);
            }
            float sum = 0.0f;
#pragma unroll
            for (int s = 0; s < kS; ++s) {
                l[s] = expf(l[s] - mx);
                sum += l[s];
            }
            const float inv = 1.0f / sum;
#pragma unroll
            for (int s = 0; s < kS; ++s) l[s] *= inv * rinv[s];  // softmax weight times r_s
#pragma unroll
            for (int j = 0; j < kChunks; ++j) {
                const int c = (threadIdx.x + j * blockDim.x) * 8;
                if (c < H) {
                    float o[8];
#pragma unroll
                    for (int e = 0; e < 8; ++e) o[e] = 0.0f;
#pragma unroll
                    for (int s = 0; s < kS; ++s) {
#pragma unroll
                        for (int e = 0; e < 8; ++e) o[e] += l[s] * x[s][j][e];
                    }
                    store8(out + (static_cast<int64_t>(q0 + qi) * T + t) * H + c, o);
                }
            }
        }
        __syncthreads();  // warp_part and total are reused by the next chunk
    }
}

template <typename TP, int kS>
void attn_res_launch_s(const __nv_bfloat16* blocks, int64_t tok_stride, int64_t blk_stride,
                       const TP* partial, const float* query, int nq, float* out, int T, int H,
                       float eps, cudaStream_t stream) {
    const int vecs = H / 8;
    // One block per token reads each source once for all the queries. With fewer tokens than
    // two blocks per SM and more than one chunk of queries, the chunks get their own blocks
    // (each re-reads the token's sources, from L2): one block streaming 48 queries is slower.
    const int chunks = cdiv(nq, kQChunk);
    int split = cdiv(2 * num_sms(), T);
    split = split < 1 ? 1 : (split > chunks ? chunks : split);
    const int q_per_block = cdiv(chunks, split) * kQChunk;
    const dim3 grid(T, cdiv(nq, q_per_block));
    if (vecs <= kAttnResThreads) {
        const int block = cdiv(vecs, kWarpSize) * kWarpSize;
        attn_res_kernel<TP, kS, 1><<<grid, block, 0, stream>>>(
            blocks, tok_stride, blk_stride, partial, query, nq, q_per_block, out, T, H, eps);
    } else {
        const int block = cdiv(cdiv(vecs, 2), kWarpSize) * kWarpSize;
        attn_res_kernel<TP, kS, 2><<<grid, block, 0, stream>>>(
            blocks, tok_stride, blk_stride, partial, query, nq, q_per_block, out, T, H, eps);
    }
}

template <typename TP>
void attn_res_launch(const __nv_bfloat16* blocks, int64_t tok_stride, int64_t blk_stride, int nb,
                     const TP* partial, const float* query, int nq, float* out, int T, int H,
                     float eps, cudaStream_t stream) {
    switch (nb + 1) {
#define SPARK_ATTN_RES_CASE(S)                                                                  \
    case S:                                                                                     \
        attn_res_launch_s<TP, S>(blocks, tok_stride, blk_stride, partial, query, nq, out, T, H, \
                                 eps, stream);                                                  \
        break;
        SPARK_ATTN_RES_CASE(1)
        SPARK_ATTN_RES_CASE(2)
        SPARK_ATTN_RES_CASE(3)
        SPARK_ATTN_RES_CASE(4)
        SPARK_ATTN_RES_CASE(5)
        SPARK_ATTN_RES_CASE(6)
        SPARK_ATTN_RES_CASE(7)
        SPARK_ATTN_RES_CASE(8)
#undef SPARK_ATTN_RES_CASE
        default:
            break;
    }
}

}  // namespace

bool attn_res_supports(int nb, int H) {
    return nb >= 0 && nb + 1 <= kAttnResMaxSources && H > 0 && H % 8 == 0 &&
           H <= 16 * kAttnResThreads;
}

void attn_res_fwd(const __nv_bfloat16* blocks, int64_t tok_stride, int64_t blk_stride, int nb,
                  const void* partial, bool partial_bf16, const float* query, int nq, float* out,
                  int T, int H, float eps, cudaStream_t stream) {
    SPARK_REQUIRE(attn_res_supports(nb, H),
                  "attn_res: needs 0 <= nb <= 7 and H a multiple of 8, at most 8192");
    SPARK_REQUIRE(nq >= 1 && T >= 0, "attn_res: needs nq >= 1");
    SPARK_REQUIRE(
        partial != nullptr && query != nullptr && out != nullptr && (nb == 0 || blocks != nullptr),
        "attn_res: null pointer");
    if (T == 0) return;
    if (partial_bf16) {
        attn_res_launch(blocks, tok_stride, blk_stride, nb,
                        static_cast<const __nv_bfloat16*>(partial), query, nq, out, T, H, eps,
                        stream);
    } else {
        attn_res_launch(blocks, tok_stride, blk_stride, nb, static_cast<const float*>(partial),
                        query, nq, out, T, H, eps, stream);
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
