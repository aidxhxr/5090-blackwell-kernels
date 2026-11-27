// Inkling 2 Mini's block attention residuals and loop message gates
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
//
// message gates: out = (1 - tanh(Wf z)) * h + tanh(Wg z) * m with z = Wd [n(h); n(m)], Wd
//   [64, 2H] bf16 (nn.Linear layout), Wf and Wg [H, 64] fp32 or bf16 (the checkpoint's dtype;
//   HF upcasts them, which changes no value), h and m fp32. Up to three gates per launch: a
//   loop layer's q, k and v gates run together.
//   decode (T <= 64): one cooperative kernel, three phases split by grid syncs:
//     0. zero the accumulators;
//     1. a block per (gate, 64 columns of [h; m]) dots its slice of Wd with the raw
//        activations and adds into z_h, z_m [T, 64] and the sums of squares (fp32 atomics);
//     2. a block per (gate, 32 output columns) forms z = bf16(r_h z_h + r_m z_m), the two
//        rank-64 up-projections of its columns, the tanh and the mix.
//     HF rounds the normalized activations to bf16 before its bf16 down_proj; this path dots
//     them in fp32 (the norm is factored out), then rounds z to bf16 as down_proj's output is.
//     The atomics make the last bits of z depend on the order the blocks land.
//   prefill: message_gate_prep writes [bf16(n(h)); bf16(n(m))] as the GEMM's A, hgemm does Wd,
//     message_gate_up does the two up-projections, the tanh and the mix on 64 x 64 tiles.

#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace cg = cooperative_groups;

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

__device__ __forceinline__ void store8(__nv_bfloat16* p, const float (&v)[8]) {
    bf16x8 b;
#pragma unroll
    for (int i = 0; i < 4; ++i) b.h[i] = __float22bfloat162_rn(make_float2(v[2 * i], v[2 * i + 1]));
    *reinterpret_cast<bf16x8*>(p) = b;
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

// ---------------------------------------------------------------------------
// Message gates
// ---------------------------------------------------------------------------
constexpr int kR = kGateRank;
constexpr int kGateThreads = 256;
constexpr int kSliceA = 64;  // columns of [h; m] per phase-1 item
constexpr int kSliceB = 32;  // output columns per phase-2 item
constexpr int kTMax = kGateDecodeMaxTokens;
constexpr int kGateSmemFloats = kTMax * kR + 2 * kSliceB * (kR + 1);

__host__ __device__ __forceinline__ int64_t gate_ws_floats_per_gate(int T) {
    return 2LL * T * kR + 2LL * T;  // z_h, z_m [T, 64], then the sums of squares of h, m [T]
}

template <typename TW>
__global__ void __launch_bounds__(kGateThreads)
    message_gate_decode_kernel(MessageGateSet a, float* __restrict__ ws, int T, int H, float eps) {
    cg::grid_group grid = cg::this_grid();
    __shared__ __align__(16) float smem[kGateSmemFloats];
    const int G = a.G;
    const int64_t per_gate = gate_ws_floats_per_gate(T);
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    constexpr int kWarps = kGateThreads / kWarpSize;

    const int slices_half = H / kSliceA;
    const int slices_out = H / kSliceB;
    const int items_a = G * 2 * slices_half;
    const int items_b = G * slices_out;
    float* xs = smem;                       // phase 1: the activation slice [T][64]
    float* zs = smem;                       // phase 2: z [T][64]
    float* wfs = smem + kTMax * kR;         // phase 2: Wf rows c0..c0+31 [32][65]
    float* wgs = wfs + kSliceB * (kR + 1);  // phase 2: Wg rows c0..c0+31 [32][65]
    // Thread (r, qd) holds Wd[r, k0 + 8 qd + 32 jj + (0..7)], jj = 0, 1, of a phase-1 item.
    const int r = threadIdx.x >> 2;
    const int qd = threadIdx.x & 3;
    float w[2][8];
    auto load_wd = [&](int item) {
        const int g = item / (2 * slices_half);
        const int i = item % (2 * slices_half);
        const int half = i / slices_half;
        const int k0 = (i % slices_half) * kSliceA;
        const __nv_bfloat16* wrow =
            a.wd[g] + static_cast<int64_t>(r) * 2 * H + static_cast<int64_t>(half) * H + k0;
        load8(wrow + qd * 8, w[0]);
        load8(wrow + 32 + qd * 8, w[1]);
    };
    auto stage_up = [&](int item) {
        const int g = item / slices_out;
        const int c0 = (item % slices_out) * kSliceB;
        const TW* wf = static_cast<const TW*>(a.wf[g]);
        const TW* wg = static_cast<const TW*>(a.wg[g]);
        for (int v = threadIdx.x; v < kSliceB * (kR / 8); v += kGateThreads) {
            const int c = v / (kR / 8), k8 = v % (kR / 8);
            const int64_t off = static_cast<int64_t>(c0 + c) * kR + k8 * 8;
            float f[8], q[8];
            load8(wf + off, f);
            load8(wg + off, q);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                wfs[c * (kR + 1) + k8 * 8 + e] = f[e];
                wgs[c * (kR + 1) + k8 * 8 + e] = q[e];
            }
        }
    };
    // The weights do not depend on anything the phases compute: the block's first items of
    // both phases start loading before the first grid sync, so their DRAM latency overlaps the
    // zeroing and the syncs. wfs and wgs sit past xs, so phase 1 leaves them alone.
    if (blockIdx.x < items_b) stage_up(blockIdx.x);
    if (blockIdx.x < items_a) load_wd(blockIdx.x);

    // Phase 0: zero the accumulators.
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < G * per_gate;
         i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        ws[i] = 0.0f;
    }
    grid.sync();

    // Phase 1: z_h = Wd[:, :H] h and z_m = Wd[:, H:] m (raw, the norms factored out), and the
    // sums of squares.
    for (int item = blockIdx.x; item < items_a; item += gridDim.x) {
        const int g = item / (2 * slices_half);
        const int i = item % (2 * slices_half);
        const int half = i / slices_half;  // 0: h, 1: m
        const int k0 = (i % slices_half) * kSliceA;
        const float* x = half ? a.m[g] : a.h[g];
        float* zacc = ws + g * per_gate + static_cast<int64_t>(half) * T * kR;
        float* ssq = ws + g * per_gate + 2LL * T * kR + static_cast<int64_t>(half) * T;
        for (int v = threadIdx.x; v < T * (kSliceA / 4); v += kGateThreads) {
            const int tt = v / (kSliceA / 4), c4 = v % (kSliceA / 4);
            *reinterpret_cast<float4*>(xs + tt * kSliceA + c4 * 4) =
                *reinterpret_cast<const float4*>(x + static_cast<int64_t>(tt) * H + k0 + c4 * 4);
        }
        if (item != blockIdx.x) load_wd(item);
        __syncthreads();
        for (int tt = warp; tt < T; tt += kWarps) {
            const float u = xs[tt * kSliceA + lane], v = xs[tt * kSliceA + lane + 32];
            const float s = warp_reduce_sum(u * u + v * v);
            if (lane == 0) atomicAdd(ssq + tt, s);
        }
        for (int tt = 0; tt < T; ++tt) {
            float acc = 0.0f;
#pragma unroll
            for (int jj = 0; jj < 2; ++jj) {
                float xv[8];
                load8(xs + tt * kSliceA + 32 * jj + qd * 8, xv);
#pragma unroll
                for (int e = 0; e < 8; ++e) acc += w[jj][e] * xv[e];
            }
            acc += __shfl_xor_sync(kFullMask, acc, 1);
            acc += __shfl_xor_sync(kFullMask, acc, 2);
            if (qd == 0) atomicAdd(zacc + tt * kR + r, acc);
        }
        __syncthreads();
    }
    grid.sync();

    // Phase 2: z = bf16(r_h z_h + r_m z_m) for every token, then 32 output columns: lane c of
    // warp w does column c0 + c for tokens w, w + 8, ...
    for (int item = blockIdx.x; item < items_b; item += gridDim.x) {
        const int g = item / slices_out;
        const int c0 = (item % slices_out) * kSliceB;
        const float* zacc = ws + g * per_gate;
        const float* ssq = zacc + 2LL * T * kR;
        for (int v = threadIdx.x; v < T * kR; v += kGateThreads) {
            const int tt = v / kR;
            const float rh = rsqrtf(__ldcg(ssq + tt) / static_cast<float>(H) + eps);
            const float rm = rsqrtf(__ldcg(ssq + T + tt) / static_cast<float>(H) + eps);
            const float z = rh * __ldcg(zacc + v) + rm * __ldcg(zacc + T * kR + v);
            zs[v] = __bfloat162float(__float2bfloat16(z));
        }
        if (item != blockIdx.x) stage_up(item);
        __syncthreads();
        for (int tt = warp; tt < T; tt += kWarps) {
            float af = 0.0f, ag = 0.0f;
#pragma unroll 16
            for (int k = 0; k < kR; ++k) {
                const float z = zs[tt * kR + k];
                af += wfs[lane * (kR + 1) + k] * z;
                ag += wgs[lane * (kR + 1) + k] * z;
            }
            const int64_t o = static_cast<int64_t>(tt) * H + c0 + lane;
            a.out[g][o] = (1.0f - tanhf(af)) * a.h[g][o] + tanhf(ag) * a.m[g][o];
        }
        __syncthreads();
    }
}

// Prefill step 1: x[g][t] = [bf16(n(h[g][t])); bf16(n(m[g][t]))], the bf16 A of the Wd GEMM.
__global__ void __launch_bounds__(1024)
    message_gate_prep_kernel(MessageGateSet a, int T, int H, float eps) {
    __shared__ float red[2][32];
    const int t = blockIdx.x, g = blockIdx.y;
    const int c = threadIdx.x * 8;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    float hv[8], mv[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) hv[e] = mv[e] = 0.0f;
    if (c < H) {
        load8(a.h[g] + static_cast<int64_t>(t) * H + c, hv);
        load8(a.m[g] + static_cast<int64_t>(t) * H + c, mv);
    }
    float sh = 0.0f, sm = 0.0f;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        sh += hv[e] * hv[e];
        sm += mv[e] * mv[e];
    }
    sh = warp_reduce_sum(sh);
    sm = warp_reduce_sum(sm);
    if (lane == 0) {
        red[0][warp] = sh;
        red[1][warp] = sm;
    }
    __syncthreads();
    sh = 0.0f;
    sm = 0.0f;
    for (int w = 0; w < nwarps; ++w) {
        sh += red[0][w];
        sm += red[1][w];
    }
    const float rh = rsqrtf(sh / static_cast<float>(H) + eps);
    const float rm = rsqrtf(sm / static_cast<float>(H) + eps);
    if (c < H) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            hv[e] *= rh;
            mv[e] *= rm;
        }
        __nv_bfloat16* xr = a.x[g] + static_cast<int64_t>(t) * 2 * H;
        store8(xr + c, hv);
        store8(xr + H + c, mv);
    }
}

// Prefill step 3: a block per (64 tokens, 64 columns, gate); thread (tx, ty) holds a 4 x 4
// tile of each of Wf z and Wg z, the operands staged k-major in shared memory.
constexpr int kUpTile = 64;
template <typename TW>
__global__ void __launch_bounds__(256) message_gate_up_kernel(MessageGateSet a, int T, int H) {
    __shared__ __align__(16) float zs[kR][kUpTile];
    __shared__ __align__(16) float wfs[kR][kUpTile];
    __shared__ __align__(16) float wgs[kR][kUpTile];
    const int g = blockIdx.z;
    const int t0 = blockIdx.x * kUpTile, c0 = blockIdx.y * kUpTile;
    for (int v = threadIdx.x; v < kUpTile * (kR / 8); v += 256) {
        const int tt = v % kUpTile, k8 = v / kUpTile;
        float zv[8];
        if (t0 + tt < T) {
            load8(a.z[g] + static_cast<int64_t>(t0 + tt) * kR + k8 * 8, zv);
        } else {
#pragma unroll
            for (int e = 0; e < 8; ++e) zv[e] = 0.0f;
        }
#pragma unroll
        for (int e = 0; e < 8; ++e) zs[k8 * 8 + e][tt] = zv[e];
    }
    const TW* wf = static_cast<const TW*>(a.wf[g]);
    const TW* wg = static_cast<const TW*>(a.wg[g]);
    for (int v = threadIdx.x; v < kUpTile * (kR / 8); v += 256) {
        const int cc = v % kUpTile, k8 = v / kUpTile;
        const int64_t off = static_cast<int64_t>(c0 + cc) * kR + k8 * 8;
        float f[8], q[8];
        load8(wf + off, f);
        load8(wg + off, q);
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            wfs[k8 * 8 + e][cc] = f[e];
            wgs[k8 * 8 + e][cc] = q[e];
        }
    }
    __syncthreads();

    const int tx = threadIdx.x & 15, ty = threadIdx.x >> 4;
    float af[4][4], ag[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) af[i][j] = ag[i][j] = 0.0f;
    }
#pragma unroll 8
    for (int k = 0; k < kR; ++k) {
        const float4 zv = *reinterpret_cast<const float4*>(&zs[k][ty * 4]);
        const float4 fv = *reinterpret_cast<const float4*>(&wfs[k][tx * 4]);
        const float4 gv = *reinterpret_cast<const float4*>(&wgs[k][tx * 4]);
        const float zr[4] = {zv.x, zv.y, zv.z, zv.w};
        const float fr[4] = {fv.x, fv.y, fv.z, fv.w};
        const float gr[4] = {gv.x, gv.y, gv.z, gv.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                af[i][j] += zr[i] * fr[j];
                ag[i][j] += zr[i] * gr[j];
            }
        }
    }
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int t = t0 + ty * 4 + i;
        if (t >= T) break;
        const int64_t o = static_cast<int64_t>(t) * H + c0 + tx * 4;
        const float4 hv = *reinterpret_cast<const float4*>(a.h[g] + o);
        const float4 mv = *reinterpret_cast<const float4*>(a.m[g] + o);
        const float hr[4] = {hv.x, hv.y, hv.z, hv.w};
        const float mr[4] = {mv.x, mv.y, mv.z, mv.w};
        float r[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            r[j] = (1.0f - tanhf(af[i][j])) * hr[j] + tanhf(ag[i][j]) * mr[j];
        }
        *reinterpret_cast<float4*>(a.out[g] + o) = make_float4(r[0], r[1], r[2], r[3]);
    }
}

void check_gate_set(const MessageGateSet& a, int T, int H, const char* who) {
    SPARK_REQUIRE(a.G >= 1 && a.G <= kGateMaxGates, std::string(who) + ": 1 to 3 gates per launch");
    SPARK_REQUIRE(T >= 1 && H % 64 == 0 && H <= 8192,
                  std::string(who) + ": needs T >= 1 and H a multiple of 64, at most 8192");
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

int64_t message_gate_decode_workspace_floats(int G, int T) {
    return G * gate_ws_floats_per_gate(T);
}

void message_gate_decode(const MessageGateSet& a, float* workspace, int T, int H, float eps,
                         cudaStream_t stream) {
    check_gate_set(a, T, H, "message_gate_decode");
    SPARK_REQUIRE(T <= kGateDecodeMaxTokens, "message_gate_decode: at most 64 tokens");
    auto* kernel =
        a.up_bf16 ? message_gate_decode_kernel<__nv_bfloat16> : message_gate_decode_kernel<float>;
    static int per_sm[2] = {-1, -1};
    int& fit = per_sm[a.up_bf16 ? 1 : 0];
    if (fit < 0) {
        SPARK_CUDA_CHECK(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&fit, kernel, kGateThreads, 0));
        SPARK_REQUIRE(fit > 0, "message_gate_decode: the kernel does not fit on an SM");
    }
    const int items = a.G * std::max(2 * H / kSliceA, H / kSliceB);
    const int grid = std::min(items, fit * num_sms());
    MessageGateSet ka = a;
    float* ws = workspace;
    int kt = T, kh = H;
    float ke = eps;
    void* args[] = {&ka, &ws, &kt, &kh, &ke};
    SPARK_CUDA_CHECK(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(kernel), dim3(grid),
                                                 dim3(kGateThreads), args, 0, stream));
}

void message_gate_prep(const MessageGateSet& a, int T, int H, float eps, cudaStream_t stream) {
    check_gate_set(a, T, H, "message_gate_prep");
    const int block = cdiv(H / 8, kWarpSize) * kWarpSize;
    message_gate_prep_kernel<<<dim3(T, a.G), block, 0, stream>>>(a, T, H, eps);
    SPARK_CHECK_LAUNCH();
}

void message_gate_up(const MessageGateSet& a, int T, int H, cudaStream_t stream) {
    check_gate_set(a, T, H, "message_gate_up");
    const dim3 grid(cdiv(T, kUpTile), H / kUpTile, a.G);
    if (a.up_bf16) {
        message_gate_up_kernel<__nv_bfloat16><<<grid, 256, 0, stream>>>(a, T, H);
    } else {
        message_gate_up_kernel<float><<<grid, 256, 0, stream>>>(a, T, H);
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
