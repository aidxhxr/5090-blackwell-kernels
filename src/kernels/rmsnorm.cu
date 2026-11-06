// RMSNorm kernels: out[r, c] = x[r, c] * rsqrt(mean_c(x[r, :]^2) + eps) * w[c]
//
// Optimization ladder (see docs/design/rmsnorm.md):
//   variant 0: one thread per row            (naive: uncoalesced, serial reduction)
//   variant 1: one warp per row              (coalesced, shuffle reduction)
//   variant 2: one warp per row, 128-bit loads (4 f32 / 8 bf16 per lane per transaction)
//   variant 3: one 256-thread block per row  (for very wide rows)
//   variant 4: single pass, the row held in registers by a 32..1024-thread group
// Plus the fused residual-add + RMSNorm used in Llama/Qwen decoder blocks (same single-pass
// design), and both norms writing the per-token e4m3 row an fp8 GEMM takes instead of bf16
// (rmsnorm_fp8, add_rmsnorm_fp8: the bytes fp8_quantize's row mode makes of the bf16 output).
//
// All math accumulates in fp32 regardless of the storage type T.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>

#include "fp8quant.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {
namespace {

// ---------------------------------------------------------------------------
// 128-bit vector traits: f32 -> 4 lanes, bf16 -> 8 lanes.
// ---------------------------------------------------------------------------
template <typename T>
struct VecTraits;

template <>
struct VecTraits<float> {
    static constexpr int kWidth = 4;
    __device__ __forceinline__ static void load(const float* p, float (&out)[4]) {
        const f32x4 v = *reinterpret_cast<const f32x4*>(p);
        out[0] = v.v[0];
        out[1] = v.v[1];
        out[2] = v.v[2];
        out[3] = v.v[3];
    }
    __device__ __forceinline__ static void store(float* p, const float (&in)[4]) {
        f32x4 v;
        v.v[0] = in[0];
        v.v[1] = in[1];
        v.v[2] = in[2];
        v.v[3] = in[3];
        *reinterpret_cast<f32x4*>(p) = v;
    }
};

template <>
struct VecTraits<__nv_bfloat16> {
    static constexpr int kWidth = 8;
    __device__ __forceinline__ static void load(const __nv_bfloat16* p, float (&out)[8]) {
        const bf16x8 v = *reinterpret_cast<const bf16x8*>(p);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(v.h[i]);
            out[2 * i] = f.x;
            out[2 * i + 1] = f.y;
        }
    }
    __device__ __forceinline__ static void store(__nv_bfloat16* p, const float (&in)[8]) {
        bf16x8 v;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v.h[i] = __float22bfloat162_rn(make_float2(in[2 * i], in[2 * i + 1]));
        }
        *reinterpret_cast<bf16x8*>(p) = v;
    }
};

// ---------------------------------------------------------------------------
// Variant 0: one thread per row.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void rmsnorm_naive_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                     T* __restrict__ out, int rows, int cols, float eps) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const T* xr = x + static_cast<int64_t>(row) * cols;
    T* orow = out + static_cast<int64_t>(row) * cols;

    float sumsq = 0.0f;
    for (int c = 0; c < cols; ++c) {
        const float v = to_f32(xr[c]);
        sumsq += v * v;
    }
    const float inv = rsqrtf(sumsq / static_cast<float>(cols) + eps);
    for (int c = 0; c < cols; ++c) {
        orow[c] = from_f32<T>(to_f32(xr[c]) * inv * to_f32(w[c]));
    }
}

// ---------------------------------------------------------------------------
// Variant 1: one warp per row, scalar loads strided by lane.
// kWarpsPerBlock rows are processed per block.
// ---------------------------------------------------------------------------
template <typename T, int kWarpsPerBlock>
__global__ void rmsnorm_warp_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                    T* __restrict__ out, int rows, int cols, float eps) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= rows) return;
    const T* xr = x + static_cast<int64_t>(row) * cols;
    T* orow = out + static_cast<int64_t>(row) * cols;

    float sumsq = 0.0f;
    for (int c = lane; c < cols; c += kWarpSize) {
        const float v = to_f32(xr[c]);
        sumsq += v * v;
    }
    sumsq = warp_reduce_sum(sumsq);
    const float inv = rsqrtf(sumsq / static_cast<float>(cols) + eps);
    for (int c = lane; c < cols; c += kWarpSize) {
        orow[c] = from_f32<T>(to_f32(xr[c]) * inv * to_f32(w[c]));
    }
}

// ---------------------------------------------------------------------------
// Variant 2: one warp per row, 128-bit vectorized loads/stores.
// Pass 1 reads the row once from DRAM to compute the sum of squares; pass 2
// re-reads it (row <= 16 KB, served from L2: 24 MB on GB10, TBD on the RTX 5090) and
// writes the output.
// Requires cols % kWidth == 0 and 16-byte aligned x, w, out (with cols a multiple of the
// vector width, every row then starts on a 16-byte boundary too).
// ---------------------------------------------------------------------------
template <typename T, int kWarpsPerBlock>
__global__ void rmsnorm_warp_vec_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                        T* __restrict__ out, int rows, int cols, float eps) {
    using VT = VecTraits<T>;
    constexpr int kW = VT::kWidth;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= rows) return;
    const T* xr = x + static_cast<int64_t>(row) * cols;
    T* orow = out + static_cast<int64_t>(row) * cols;

    float sumsq = 0.0f;
    for (int c = lane * kW; c < cols; c += kWarpSize * kW) {
        float v[kW];
        VT::load(xr + c, v);
#pragma unroll
        for (int i = 0; i < kW; ++i) sumsq += v[i] * v[i];
    }
    sumsq = warp_reduce_sum(sumsq);
    const float inv = rsqrtf(sumsq / static_cast<float>(cols) + eps);

    for (int c = lane * kW; c < cols; c += kWarpSize * kW) {
        float v[kW];
        float wv[kW];
        float o[kW];
        VT::load(xr + c, v);
        VT::load(w + c, wv);
#pragma unroll
        for (int i = 0; i < kW; ++i) o[i] = v[i] * inv * wv[i];
        VT::store(orow + c, o);
    }
}

// ---------------------------------------------------------------------------
// Variant 3: one block per row, block-wide reduction. Consecutive threads read
// consecutive columns so every access is coalesced.
// ---------------------------------------------------------------------------
template <typename T, int kBlockSize>
__global__ void rmsnorm_block_kernel(const T* __restrict__ x, const T* __restrict__ w,
                                     T* __restrict__ out, int rows, int cols, float eps) {
    __shared__ float smem[kWarpSize];
    const int row = blockIdx.x;
    if (row >= rows) return;
    const T* xr = x + static_cast<int64_t>(row) * cols;
    T* orow = out + static_cast<int64_t>(row) * cols;

    float sumsq = 0.0f;
    for (int c = threadIdx.x; c < cols; c += kBlockSize) {
        const float v = to_f32(xr[c]);
        sumsq += v * v;
    }
    sumsq = block_reduce_sum(sumsq, smem);
    const float inv = rsqrtf(sumsq / static_cast<float>(cols) + eps);
    for (int c = threadIdx.x; c < cols; c += kBlockSize) {
        orow[c] = from_f32<T>(to_f32(xr[c]) * inv * to_f32(w[c]));
    }
}

// ---------------------------------------------------------------------------
// Fused residual-add + RMSNorm (bf16), warp-per-row, 128-bit vectors.
//   resid[r, :] += x[r, :];   out[r, :] = rmsnorm(resid[r, :]) * w
// Pass 1 reads x and resid, writes the updated resid, accumulates sum of squares.
// Pass 2 re-reads the fresh resid (L2-resident) and writes out.
// ---------------------------------------------------------------------------
template <int kWarpsPerBlock>
__global__ void add_rmsnorm_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                        __nv_bfloat16* __restrict__ resid,
                                        const __nv_bfloat16* __restrict__ w,
                                        __nv_bfloat16* __restrict__ out, int rows, int cols,
                                        float eps) {
    using VT = VecTraits<__nv_bfloat16>;
    constexpr int kW = VT::kWidth;
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    const int row = blockIdx.x * kWarpsPerBlock + warp;
    if (row >= rows) return;
    const int64_t off = static_cast<int64_t>(row) * cols;
    const __nv_bfloat16* xr = x + off;
    __nv_bfloat16* rr = resid + off;
    __nv_bfloat16* orow = out + off;

    float sumsq = 0.0f;
    for (int c = lane * kW; c < cols; c += kWarpSize * kW) {
        float xv[kW];
        float rv[kW];
        float sv[kW];
        VT::load(xr + c, xv);
        VT::load(rr + c, rv);
#pragma unroll
        for (int i = 0; i < kW; ++i) {
            // Round the residual sum to bf16 first so the stored residual and the
            // value we normalize agree exactly (matches PyTorch's semantics of
            // `resid = resid + x` in bf16 followed by a norm over `resid`).
            const float s = __bfloat162float(__float2bfloat16(xv[i] + rv[i]));
            sv[i] = s;
            sumsq += s * s;
        }
        VT::store(rr + c, sv);
    }
    sumsq = warp_reduce_sum(sumsq);
    const float inv = rsqrtf(sumsq / static_cast<float>(cols) + eps);

    __syncwarp();
    for (int c = lane * kW; c < cols; c += kWarpSize * kW) {
        float rv[kW];
        float wv[kW];
        float o[kW];
        VT::load(rr + c, rv);
        VT::load(w + c, wv);
#pragma unroll
        for (int i = 0; i < kW; ++i) o[i] = rv[i] * inv * wv[i];
        VT::store(orow + c, o);
    }
}

// ---------------------------------------------------------------------------
// Variant 4: single pass, row in registers.
// A group of G threads (32..1024) owns one row and holds all of it in registers, VPT 16-byte
// vectors per thread, so x is read exactly once: the second pass of variants 1-3 (an L2
// re-read that still costs L2 bandwidth and, on the RTX 5090, the 96 MB L2 does not always
// keep) is gone, and traffic is at the floor of one read plus one write. Rows up to 32 K
// elements qualify (1024 threads x 32 elements); wider or unaligned rows fall back to
// variant 3. The group reduction is a warp shuffle plus, for G > 32, one shared-memory
// exchange between the group's warps.
// ---------------------------------------------------------------------------
constexpr int kMaxElemsPerThread = 32;

template <int G>
struct Group {
    static constexpr int kBlock = G < 256 ? 256 : G;
    static constexpr int kRowsPerBlock = kBlock / G;
    static constexpr int kWarps = G / kWarpSize;
    static_assert(G % kWarpSize == 0 && kBlock % G == 0, "G must be a multiple of 32");
};

// Sum `v` over the G threads of the calling thread's group; every thread gets the result.
// Must be reached by all threads of the block (it may __syncthreads).
template <int G>
__device__ __forceinline__ float group_reduce_sum(float v, float* red, int tid) {
    v = warp_reduce_sum(v);
    if constexpr (Group<G>::kWarps > 1) {
        if ((tid & (kWarpSize - 1)) == 0) red[tid >> 5] = v;
        __syncthreads();
        const int first = (tid / G) * Group<G>::kWarps;
        v = 0.0f;
#pragma unroll
        for (int k = 0; k < Group<G>::kWarps; ++k) v += red[first + k];
    }
    return v;
}

// The same for a max of non-negative values (an amax), on its own shared array.
template <int G>
__device__ __forceinline__ float group_reduce_max(float v, float* red, int tid) {
    v = warp_reduce_max(v);
    if constexpr (Group<G>::kWarps > 1) {
        if ((tid & (kWarpSize - 1)) == 0) red[tid >> 5] = v;
        __syncthreads();
        const int first = (tid / G) * Group<G>::kWarps;
        v = 0.0f;
#pragma unroll
        for (int k = 0; k < Group<G>::kWarps; ++k) v = fmaxf(v, red[first + k]);
    }
    return v;
}

template <typename T, int G, int VPT>
__global__ void __launch_bounds__(Group<G>::kBlock)
    rmsnorm_reg_kernel(const T* __restrict__ x, const T* __restrict__ w, T* __restrict__ out,
                       int rows, int cols, float eps) {
    using VT = VecTraits<T>;
    constexpr int kW = VT::kWidth;
    __shared__ float red[Group<G>::kBlock / kWarpSize];

    const int tid = threadIdx.x;
    const int t = tid % G;
    const int row = blockIdx.x * Group<G>::kRowsPerBlock + tid / G;
    const bool active = row < rows;  // no early return: the group reduction may sync
    const int nvec = cols / kW;
    const T* xr = x + static_cast<int64_t>(row) * cols;

    float v[VPT][kW];
    float sumsq = 0.0f;
    if (active) {
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            const int idx = t + i * G;  // consecutive threads, consecutive 16-byte chunks
            if (idx < nvec) {
                VT::load(xr + idx * kW, v[i]);
#pragma unroll
                for (int e = 0; e < kW; ++e) sumsq += v[i][e] * v[i][e];
            }
        }
    }
    sumsq = group_reduce_sum<G>(sumsq, red, tid);
    if (!active) return;
    const float inv = rsqrtf(sumsq / static_cast<float>(cols) + eps);
    T* orow = out + static_cast<int64_t>(row) * cols;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int idx = t + i * G;
        if (idx < nvec) {
            float wv[kW];
            float o[kW];
            VT::load(w + idx * kW, wv);
#pragma unroll
            for (int e = 0; e < kW; ++e) o[e] = v[i][e] * inv * wv[e];
            VT::store(orow + idx * kW, o);
        }
    }
}

// Fused residual-add + RMSNorm with the same single-pass design: x and resid are read once,
// the bf16-rounded sum is stored to resid and kept in registers for the normalize.
template <int G, int VPT>
__global__ void __launch_bounds__(Group<G>::kBlock)
    add_rmsnorm_reg_kernel(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ resid,
                           const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
                           int rows, int cols, float eps) {
    using VT = VecTraits<__nv_bfloat16>;
    constexpr int kW = VT::kWidth;
    __shared__ float red[Group<G>::kBlock / kWarpSize];

    const int tid = threadIdx.x;
    const int t = tid % G;
    const int row = blockIdx.x * Group<G>::kRowsPerBlock + tid / G;
    const bool active = row < rows;
    const int nvec = cols / kW;
    const int64_t off = static_cast<int64_t>(row) * cols;

    float sv[VPT][kW];
    float sumsq = 0.0f;
    if (active) {
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            const int idx = t + i * G;
            if (idx < nvec) {
                float xv[kW];
                float rv[kW];
                VT::load(x + off + idx * kW, xv);
                VT::load(resid + off + idx * kW, rv);
#pragma unroll
                for (int e = 0; e < kW; ++e) {
                    // Round to bf16 first so the stored residual and the value normalized
                    // agree exactly (PyTorch semantics of a bf16 `resid += x` then a norm).
                    const float s = __bfloat162float(__float2bfloat16(xv[e] + rv[e]));
                    sv[i][e] = s;
                    sumsq += s * s;
                }
                VT::store(resid + off + idx * kW, sv[i]);
            }
        }
    }
    sumsq = group_reduce_sum<G>(sumsq, red, tid);
    if (!active) return;
    const float inv = rsqrtf(sumsq / static_cast<float>(cols) + eps);
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int idx = t + i * G;
        if (idx < nvec) {
            float wv[kW];
            float o[kW];
            VT::load(w + idx * kW, wv);
#pragma unroll
            for (int e = 0; e < kW; ++e) o[e] = sv[i][e] * inv * wv[e];
            VT::store(out + off + idx * kW, o);
        }
    }
}

// Both norms into fp8 for a GEMM with per-token activation scales: the row as rmsnorm_reg_kernel
// (kAdd false) or add_rmsnorm_reg_kernel (kAdd true) computes it, rounded to bf16 as they store
// it and kept in registers, then the group's amax of those values, pow2_scale of it and the e4m3
// bytes of fp8quant.cuh. The bf16 row is never written: one launch and one pass over it fewer
// than the norm followed by fp8_quantize's row mode, and the same bits (a max does not depend
// on its order, and the sum of squares is the bf16 kernel's, element for element).
template <bool kAdd, int G, int VPT>
__global__ void __launch_bounds__(Group<G>::kBlock)
    rmsnorm_fp8_reg_kernel(const __nv_bfloat16* __restrict__ x, __nv_bfloat16* __restrict__ resid,
                           const __nv_bfloat16* __restrict__ w, unsigned char* __restrict__ q,
                           float* __restrict__ scales, int rows, int cols, float eps) {
    using VT = VecTraits<__nv_bfloat16>;
    constexpr int kW = VT::kWidth;
    __shared__ float red[Group<G>::kBlock / kWarpSize];
    __shared__ float red_max[Group<G>::kBlock / kWarpSize];

    const int tid = threadIdx.x;
    const int t = tid % G;
    const int row = blockIdx.x * Group<G>::kRowsPerBlock + tid / G;
    const bool active = row < rows;  // no early return before the second group reduction
    const int nvec = cols / kW;
    const int64_t off = static_cast<int64_t>(row) * cols;

    float v[VPT][kW];
    float sumsq = 0.0f;
    if (active) {
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            const int idx = t + i * G;
            if (idx < nvec) {
                if constexpr (kAdd) {
                    float xv[kW];
                    float rv[kW];
                    VT::load(x + off + idx * kW, xv);
                    VT::load(resid + off + idx * kW, rv);
#pragma unroll
                    for (int e = 0; e < kW; ++e) {
                        const float s = __bfloat162float(__float2bfloat16(xv[e] + rv[e]));
                        v[i][e] = s;
                        sumsq += s * s;
                    }
                    VT::store(resid + off + idx * kW, v[i]);
                } else {
                    VT::load(x + off + idx * kW, v[i]);
#pragma unroll
                    for (int e = 0; e < kW; ++e) sumsq += v[i][e] * v[i][e];
                }
            }
        }
    }
    sumsq = group_reduce_sum<G>(sumsq, red, tid);
    const float inv = rsqrtf(sumsq / static_cast<float>(cols) + eps);
    float amax = 0.0f;
    if (active) {
#pragma unroll
        for (int i = 0; i < VPT; ++i) {
            const int idx = t + i * G;
            if (idx < nvec) {
                float wv[kW];
                VT::load(w + idx * kW, wv);
#pragma unroll
                for (int e = 0; e < kW; ++e) {
                    v[i][e] = __bfloat162float(__float2bfloat16(v[i][e] * inv * wv[e]));
                    amax = fmaxf(amax, fabsf(v[i][e]));
                }
            }
        }
    }
    const float scale = pow2_scale(group_reduce_max<G>(amax, red_max, tid));
    if (!active) return;
    if (t == 0) scales[row] = scale;
    const float rs = 1.0f / scale;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int idx = t + i * G;
        if (idx < nvec) store8(q + off + idx * kW, v[i], rs);
    }
}

// Launch helpers: pick the smallest group whose registers hold the row, then the vector
// count. `Launch` is a functor template <G, VPT> that issues the kernel.
template <typename T, int G, template <int, int> class Launch>
bool launch_group(int nvec) {
    constexpr int kMaxVPT = kMaxElemsPerThread / VecTraits<T>::kWidth;  // 8 f32, 4 bf16
    const int need = cdiv(nvec, G);
    if (need > kMaxVPT) return false;
    if (need <= 1) {
        Launch<G, 1>::run();
    } else if (need <= 2) {
        Launch<G, 2>::run();
    } else if (need <= 4) {
        Launch<G, 4>::run();
    } else {
        if constexpr (kMaxVPT >= 8) Launch<G, 8>::run();
    }
    return true;
}

template <typename T, template <int, int> class Launch>
bool launch_reg(int cols) {
    const int nvec = cols / VecTraits<T>::kWidth;
    return launch_group<T, 32, Launch>(nvec) || launch_group<T, 64, Launch>(nvec) ||
           launch_group<T, 128, Launch>(nvec) || launch_group<T, 256, Launch>(nvec) ||
           launch_group<T, 512, Launch>(nvec) || launch_group<T, 1024, Launch>(nvec);
}

// Kernel arguments travel through a static so the functor templates above stay argument-free.
template <typename T>
struct RmsArgs {
    const T* x;
    const T* w;
    T* out;
    int rows, cols;
    float eps;
    cudaStream_t stream;
    static RmsArgs& get() {
        static RmsArgs a;
        return a;
    }
};

template <typename T>
struct RmsLaunch {
    template <int G, int VPT>
    struct L {
        static void run() {
            const RmsArgs<T>& a = RmsArgs<T>::get();
            const int grid = cdiv(a.rows, Group<G>::kRowsPerBlock);
            rmsnorm_reg_kernel<T, G, VPT>
                <<<grid, Group<G>::kBlock, 0, a.stream>>>(a.x, a.w, a.out, a.rows, a.cols, a.eps);
        }
    };
};

struct AddRmsArgs {
    const __nv_bfloat16* x;
    __nv_bfloat16* resid;
    const __nv_bfloat16* w;
    __nv_bfloat16* out;
    int rows, cols;
    float eps;
    cudaStream_t stream;
    static AddRmsArgs& get() {
        static AddRmsArgs a;
        return a;
    }
};

template <int G, int VPT>
struct AddRmsLaunch {
    static void run() {
        const AddRmsArgs& a = AddRmsArgs::get();
        const int grid = cdiv(a.rows, Group<G>::kRowsPerBlock);
        add_rmsnorm_reg_kernel<G, VPT><<<grid, Group<G>::kBlock, 0, a.stream>>>(
            a.x, a.resid, a.w, a.out, a.rows, a.cols, a.eps);
    }
};

struct NormFp8Args {
    const __nv_bfloat16* x;
    __nv_bfloat16* resid;  // null without the residual add
    const __nv_bfloat16* w;
    unsigned char* q;
    float* scales;
    int rows, cols;
    float eps;
    cudaStream_t stream;
    static NormFp8Args& get() {
        static NormFp8Args a;
        return a;
    }
};

template <bool kAdd>
struct NormFp8Launch {
    template <int G, int VPT>
    struct L {
        static void run() {
            const NormFp8Args& a = NormFp8Args::get();
            const int grid = cdiv(a.rows, Group<G>::kRowsPerBlock);
            rmsnorm_fp8_reg_kernel<kAdd, G, VPT><<<grid, Group<G>::kBlock, 0, a.stream>>>(
                a.x, a.resid, a.w, a.q, a.scales, a.rows, a.cols, a.eps);
        }
    };
};

// True if the single-pass kernel can take this input (alignment, vector width, row length).
template <typename T>
bool reg_ok(const void* x, const void* w, const void* out, int cols) {
    return cols % VecTraits<T>::kWidth == 0 && is_aligned16(x) && is_aligned16(w) &&
           is_aligned16(out) && cols <= 1024 * kMaxElemsPerThread;
}

// ---------------------------------------------------------------------------
// Host dispatch shared by both element types.
// ---------------------------------------------------------------------------
template <typename T>
void rmsnorm_dispatch(const T* x, const T* w, T* out, int rows, int cols, float eps, int variant,
                      cudaStream_t stream) {
    SPARK_REQUIRE(rows >= 0 && cols > 0, "rmsnorm: rows must be >= 0 and cols > 0");
    SPARK_REQUIRE(variant >= 0 && variant < rmsnorm_num_variants(),
                  "rmsnorm: variant must be in [0, 4]");
    SPARK_REQUIRE(x != nullptr && w != nullptr && out != nullptr, "rmsnorm: null pointer");
    if (rows == 0) return;

    constexpr int kWarps = 8;  // 256 threads, 8 rows per block for variants 1/2
    switch (variant) {
        case 0: {
            const int block = 256;
            const int grid = cdiv(rows, block);
            rmsnorm_naive_kernel<T><<<grid, block, 0, stream>>>(x, w, out, rows, cols, eps);
            break;
        }
        case 1: {
            const int grid = cdiv(rows, kWarps);
            rmsnorm_warp_kernel<T, kWarps>
                <<<grid, kWarps * kWarpSize, 0, stream>>>(x, w, out, rows, cols, eps);
            break;
        }
        case 2: {
            SPARK_REQUIRE(cols % VecTraits<T>::kWidth == 0,
                          "rmsnorm variant 2: cols must be a multiple of 4 (f32) / 8 (bf16)");
            SPARK_REQUIRE(is_aligned16(x) && is_aligned16(w) && is_aligned16(out),
                          "rmsnorm variant 2: x, w and out must be 16-byte aligned");
            const int grid = cdiv(rows, kWarps);
            rmsnorm_warp_vec_kernel<T, kWarps>
                <<<grid, kWarps * kWarpSize, 0, stream>>>(x, w, out, rows, cols, eps);
            break;
        }
        case 4: {
            if (reg_ok<T>(x, w, out, cols)) {
                RmsArgs<T>::get() = {x, w, out, rows, cols, eps, stream};
                if (launch_reg<T, RmsLaunch<T>::template L>(cols)) break;
            }
            [[fallthrough]];  // unaligned, ragged or > 32 K wide: variant 3 takes anything
        }
        case 3: {
            constexpr int kBlock = 256;
            rmsnorm_block_kernel<T, kBlock>
                <<<rows, kBlock, 0, stream>>>(x, w, out, rows, cols, eps);
            break;
        }
        default:
            break;
    }
    SPARK_CHECK_LAUNCH();
}

// Both fp8 norms: the single-pass kernel only, so a row is reduced in the order the bf16
// kernels reduce it (the bits match theirs only where they run the same kernel).
template <bool kAdd>
void norm_fp8_dispatch(const __nv_bfloat16* x, __nv_bfloat16* resid, const __nv_bfloat16* w,
                       unsigned char* q, float* scales, int rows, int cols, float eps,
                       cudaStream_t stream) {
    const std::string name = kAdd ? "add_rmsnorm_fp8" : "rmsnorm_fp8";
    SPARK_REQUIRE(rows >= 0 && cols > 0, name + ": rows must be >= 0 and cols > 0");
    SPARK_REQUIRE(cols % 8 == 0 && cols <= 1024 * kMaxElemsPerThread,
                  name + ": cols must be a multiple of 8 and at most 32768 (row in registers)");
    SPARK_REQUIRE(x != nullptr && w != nullptr && q != nullptr && scales != nullptr &&
                      (!kAdd || resid != nullptr),
                  name + ": null pointer");
    SPARK_REQUIRE(is_aligned16(x) && is_aligned16(w) && (!kAdd || is_aligned16(resid)),
                  name + ": x, resid and w must be 16-byte aligned");
    SPARK_REQUIRE(reinterpret_cast<uintptr_t>(q) % 8 == 0, name + ": q must be 8-byte aligned");
    if (rows == 0) return;
    NormFp8Args::get() = {x, resid, w, q, scales, rows, cols, eps, stream};
    launch_reg<__nv_bfloat16, NormFp8Launch<kAdd>::template L>(cols);
    SPARK_CHECK_LAUNCH();
}

}  // namespace

int rmsnorm_num_variants() {
    return 5;
}

void rmsnorm_f32(const float* x, const float* w, float* out, int rows, int cols, float eps,
                 int variant, cudaStream_t stream) {
    rmsnorm_dispatch<float>(x, w, out, rows, cols, eps, variant, stream);
}

void rmsnorm_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* out, int rows,
                  int cols, float eps, int variant, cudaStream_t stream) {
    rmsnorm_dispatch<__nv_bfloat16>(x, w, out, rows, cols, eps, variant, stream);
}

void add_rmsnorm_bf16(const __nv_bfloat16* x, __nv_bfloat16* resid, const __nv_bfloat16* w,
                      __nv_bfloat16* out, int rows, int cols, float eps, cudaStream_t stream) {
    SPARK_REQUIRE(rows >= 0 && cols > 0, "add_rmsnorm: rows must be >= 0 and cols > 0");
    SPARK_REQUIRE(cols % 8 == 0, "add_rmsnorm_bf16: cols must be a multiple of 8");
    SPARK_REQUIRE(x != nullptr && resid != nullptr && w != nullptr && out != nullptr,
                  "add_rmsnorm: null pointer");
    SPARK_REQUIRE(is_aligned16(x) && is_aligned16(resid) && is_aligned16(w) && is_aligned16(out),
                  "add_rmsnorm_bf16: x, resid, w and out must be 16-byte aligned");
    if (rows == 0) return;
    if (cols <= 1024 * kMaxElemsPerThread) {  // single pass, row in registers
        AddRmsArgs::get() = {x, resid, w, out, rows, cols, eps, stream};
        if (launch_reg<__nv_bfloat16, AddRmsLaunch>(cols)) {
            SPARK_CHECK_LAUNCH();
            return;
        }
    }
    constexpr int kWarps = 8;  // wider rows: warp per row, two passes
    const int grid = cdiv(rows, kWarps);
    add_rmsnorm_bf16_kernel<kWarps>
        <<<grid, kWarps * kWarpSize, 0, stream>>>(x, resid, w, out, rows, cols, eps);
    SPARK_CHECK_LAUNCH();
}

void rmsnorm_fp8_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w, unsigned char* q,
                      float* scales, int rows, int cols, float eps, cudaStream_t stream) {
    norm_fp8_dispatch<false>(x, nullptr, w, q, scales, rows, cols, eps, stream);
}

void add_rmsnorm_fp8_bf16(const __nv_bfloat16* x, __nv_bfloat16* resid, const __nv_bfloat16* w,
                          unsigned char* q, float* scales, int rows, int cols, float eps,
                          cudaStream_t stream) {
    norm_fp8_dispatch<true>(x, resid, w, q, scales, rows, cols, eps, stream);
}

}  // namespace spark
