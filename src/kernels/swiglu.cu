// Fused SwiGLU gated activation: out[i] = silu(gate[i]) * up[i].
//
// In a Llama/Qwen MLP this sits between the gate/up projections and the down projection.
// PyTorch eager runs it as two kernels (F.silu, then mul) that stream the intermediate
// activation through memory twice; fusing it reads gate and up once and writes out once.
// The kernel is purely memory-bound, so the only levers are (a) bytes moved and (b) how
// efficiently we move them: variant 1 issues one 128-bit load per operand per thread.
//   0: scalar, one element per thread
//   1: 16-byte vectorized (float4 for f32, 8 x bf16 for bf16), grid-stride, in-kernel tail
// swiglu_strided_* is variant 1 over a rows x cols matrix whose rows are `ld` elements apart:
// the two halves of a fused gate|up projection ([rows, 2 * cols] from one GEMM) are read in
// place instead of being copied out to contiguous tensors first.
// swiglu_fp8_bf16 is the strided form writing the per-token e4m3 row an fp8 GEMM takes instead
// of bf16: the bytes and scale fp8_quantize's row mode makes of swiglu's output, in one launch.
#include <cstdint>
#include <cstdlib>
#include <type_traits>

#include "fp8quant.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {
namespace {

constexpr int kBlock = 256;

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

template <typename T>
__global__ void swiglu_scalar_kernel(const T* __restrict__ gate, const T* __restrict__ up,
                                     T* __restrict__ out, int64_t n) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = to_f32(gate[i]);
        const float u = to_f32(up[i]);
        out[i] = from_f32<T>(silu_f(g) * u);
    }
}

// f32: 4 elements per thread via f32x4 (one 128-bit load per operand).
__global__ void swiglu_vec_f32_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                      float* __restrict__ out, int64_t n) {
    constexpr int VEC = 4;
    const int64_t n_vec = n / VEC;
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    const f32x4* gate4 = reinterpret_cast<const f32x4*>(gate);
    const f32x4* up4 = reinterpret_cast<const f32x4*>(up);
    f32x4* out4 = reinterpret_cast<f32x4*>(out);

    for (int64_t i = tid; i < n_vec; i += stride) {
        const f32x4 g = gate4[i];
        const f32x4 u = up4[i];
        f32x4 o;
#pragma unroll
        for (int k = 0; k < VEC; ++k) o.v[k] = silu_f(g.v[k]) * u.v[k];
        out4[i] = o;
    }

    // Scalar tail (n % 4 elements), handled by the first few threads of the grid.
    const int64_t tail_start = n_vec * VEC;
    for (int64_t i = tail_start + tid; i < n; i += stride) {
        out[i] = silu_f(gate[i]) * up[i];
    }
}

// bf16: 8 elements per thread via bf16x8 (4 x __nv_bfloat162 = 16 bytes). Math in fp32.
__global__ void swiglu_vec_bf16_kernel(const __nv_bfloat16* __restrict__ gate,
                                       const __nv_bfloat16* __restrict__ up,
                                       __nv_bfloat16* __restrict__ out, int64_t n) {
    constexpr int VEC = 8;
    const int64_t n_vec = n / VEC;
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;

    const bf16x8* gate8 = reinterpret_cast<const bf16x8*>(gate);
    const bf16x8* up8 = reinterpret_cast<const bf16x8*>(up);
    bf16x8* out8 = reinterpret_cast<bf16x8*>(out);

    for (int64_t i = tid; i < n_vec; i += stride) {
        const bf16x8 g = gate8[i];
        const bf16x8 u = up8[i];
        bf16x8 o;
#pragma unroll
        for (int k = 0; k < VEC / 2; ++k) {
            const float2 gf = __bfloat1622float2(g.h[k]);
            const float2 uf = __bfloat1622float2(u.h[k]);
            float2 of;
            of.x = silu_f(gf.x) * uf.x;
            of.y = silu_f(gf.y) * uf.y;
            o.h[k] = __float22bfloat162_rn(of);
        }
        out8[i] = o;
    }

    const int64_t tail_start = n_vec * VEC;
    for (int64_t i = tail_start + tid; i < n; i += stride) {
        const float g = __bfloat162float(gate[i]);
        const float u = __bfloat162float(up[i]);
        out[i] = __float2bfloat16(silu_f(g) * u);
    }
}

// Grid for the grid-stride vector kernels: enough blocks to fill the machine several times
// over (8 blocks x 256 threads per SM), but never more blocks than there is vector work.
// The factor of 8 was picked for GB10 (48 SMs); re-tune on the RTX 5090 (170 SMs).
unsigned vec_grid(int64_t n_vec) {
    const int64_t want = static_cast<int64_t>(num_sms()) * 8;
    const int64_t need = cdiv64(n_vec, kBlock);
    int64_t g = want < need ? want : need;
    if (g < 1) g = 1;  // n < VEC: a single block still runs the tail loop
    return static_cast<unsigned>(g);
}

template <typename T>
void swiglu_impl(const T* gate, const T* up, T* out, int64_t n, int variant, cudaStream_t stream) {
    SPARK_REQUIRE(gate != nullptr && up != nullptr && out != nullptr, "swiglu: null pointer");
    SPARK_REQUIRE(n >= 0, "swiglu: n must be >= 0");
    SPARK_REQUIRE(variant >= 0 && variant < swiglu_num_variants(), "swiglu: unknown variant");
    if (n == 0) return;

    if (variant == 0) {
        const int64_t blocks = cdiv64(n, kBlock);
        SPARK_REQUIRE(blocks < (int64_t{1} << 31), "swiglu: n too large for variant 0");
        swiglu_scalar_kernel<T>
            <<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(gate, up, out, n);
        SPARK_CHECK_LAUNCH();
        return;
    }

    SPARK_REQUIRE(is_aligned16(gate) && is_aligned16(up) && is_aligned16(out),
                  "swiglu: variant 1 needs 16-byte aligned pointers");
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    const unsigned grid = vec_grid(n / VEC);
    if constexpr (sizeof(T) == 4) {
        swiglu_vec_f32_kernel<<<grid, kBlock, 0, stream>>>(reinterpret_cast<const float*>(gate),
                                                           reinterpret_cast<const float*>(up),
                                                           reinterpret_cast<float*>(out), n);
    } else {
        swiglu_vec_bf16_kernel<<<grid, kBlock, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(gate),
            reinterpret_cast<const __nv_bfloat16*>(up), reinterpret_cast<__nv_bfloat16*>(out), n);
    }
    SPARK_CHECK_LAUNCH();
}

// Strided rows: element (r, c) of gate is gate[r * ld_gate + c], of up is up[r * ld_up + c],
// out is contiguous rows x cols. Vectorized 16-byte loads when every row start and the
// row length are 16-byte multiples, one scalar element per thread otherwise.
template <typename T, int VEC>
__global__ void swiglu_strided_vec_kernel(const T* __restrict__ gate, const T* __restrict__ up,
                                          T* __restrict__ out, int64_t rows, int64_t cols_vec,
                                          int64_t ld_gate, int64_t ld_up) {
    using V = std::conditional_t<sizeof(T) == 4, f32x4, bf16x8>;
    const int64_t n_vec = rows * cols_vec;
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = tid; i < n_vec; i += stride) {
        const int64_t r = i / cols_vec;
        const int64_t c = (i - r * cols_vec) * VEC;
        const V g = *reinterpret_cast<const V*>(gate + r * ld_gate + c);
        const V u = *reinterpret_cast<const V*>(up + r * ld_up + c);
        V o;
        if constexpr (sizeof(T) == 4) {
#pragma unroll
            for (int k = 0; k < VEC; ++k) o.v[k] = silu_f(g.v[k]) * u.v[k];
        } else {
#pragma unroll
            for (int k = 0; k < VEC / 2; ++k) {
                const float2 gf = __bfloat1622float2(g.h[k]);
                const float2 uf = __bfloat1622float2(u.h[k]);
                float2 of;
                of.x = silu_f(gf.x) * uf.x;
                of.y = silu_f(gf.y) * uf.y;
                o.h[k] = __float22bfloat162_rn(of);
            }
        }
        *reinterpret_cast<V*>(out + r * cols_vec * VEC + c) = o;
    }
}

template <typename T>
__global__ void swiglu_strided_scalar_kernel(const T* __restrict__ gate, const T* __restrict__ up,
                                             T* __restrict__ out, int64_t rows, int64_t cols,
                                             int64_t ld_gate, int64_t ld_up) {
    const int64_t n = rows * cols;
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = tid; i < n; i += stride) {
        const int64_t r = i / cols;
        const int64_t c = i - r * cols;
        const float g = to_f32(gate[r * ld_gate + c]);
        const float u = to_f32(up[r * ld_up + c]);
        out[i] = from_f32<T>(silu_f(g) * u);
    }
}

template <typename T>
void swiglu_strided_impl(const T* gate, const T* up, T* out, int64_t rows, int64_t cols,
                         int64_t ld_gate, int64_t ld_up, cudaStream_t stream) {
    SPARK_REQUIRE(gate != nullptr && up != nullptr && out != nullptr, "swiglu: null pointer");
    SPARK_REQUIRE(rows >= 0 && cols >= 0, "swiglu: rows and cols must be >= 0");
    SPARK_REQUIRE(ld_gate >= cols && ld_up >= cols, "swiglu: row stride shorter than the row");
    if (rows == 0 || cols == 0) return;
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    const bool vec = cols % VEC == 0 && ld_gate % VEC == 0 && ld_up % VEC == 0 &&
                     is_aligned16(gate) && is_aligned16(up) && is_aligned16(out);
    if (vec) {
        const unsigned grid = vec_grid(rows * (cols / VEC));
        swiglu_strided_vec_kernel<T, VEC>
            <<<grid, kBlock, 0, stream>>>(gate, up, out, rows, cols / VEC, ld_gate, ld_up);
    } else {
        const unsigned grid = vec_grid(rows * cols);
        swiglu_strided_scalar_kernel<T>
            <<<grid, kBlock, 0, stream>>>(gate, up, out, rows, cols, ld_gate, ld_up);
    }
    SPARK_CHECK_LAUNCH();
}

// The amax a peer block of the cluster left at the same smem offset as p.
__device__ __forceinline__ float ld_cluster_f32(const float* p, unsigned cta) {
    float v;
    asm volatile(
        "{\n"
        ".reg .b32 remote;\n"
        "mapa.shared::cluster.u32 remote, %1, %2;\n"
        "ld.shared::cluster.f32 %0, [remote];\n"
        "}\n"
        : "=f"(v)
        : "r"(smem_u32(p)), "r"(cta)
        : "memory");
    return v;
}

// SwiGLU into fp8 for the down projection, with a scale per token: kC blocks of kT threads per
// row (a thread block cluster when kC > 1) keep the row's products in registers, rounded to
// bf16 as swiglu_strided_vec_kernel stores them, reduce their amax and write the pow2_scale
// of it and the e4m3 bytes of fp8quant.cuh. The bf16 row is never written: one launch and 5
// bytes per element (read 2 + 2 of gate and up, write 1 of q) where swiglu then fp8_quantize's
// row mode take 9, and the same bits (the products are the same fp32 expression rounded the
// same way, and a max does not depend on its order or on how the row is split). Thread t of the
// row's kC * kT handles vectors t, t + kC * kT, ..., VPT of them. With kC > 1 each block's amax
// goes to its shared memory, and after a cluster barrier warp 0 of every block reads all kC of
// them.
template <int kT, int VPT, int kC>
__global__ void __launch_bounds__(kT)
    swiglu_fp8_kernel(const __nv_bfloat16* __restrict__ gate, const __nv_bfloat16* __restrict__ up,
                      unsigned char* __restrict__ q, float* __restrict__ scales, int cols,
                      int64_t ld_gate, int64_t ld_up) {
    __shared__ float smem[32];
    const int64_t row = blockIdx.x / kC;
    const int rank = static_cast<int>(blockIdx.x % kC);
    const int t = rank * kT + static_cast<int>(threadIdx.x);
    const int nvec = cols / 8;
    const __nv_bfloat16* g_row = gate + row * ld_gate;
    const __nv_bfloat16* u_row = up + row * ld_up;
    bf16x8 g[VPT];
    bf16x8 u[VPT];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int idx = t + i * kC * kT;
        if (idx < nvec) {
            g[i] = *reinterpret_cast<const bf16x8*>(g_row + idx * 8);
            u[i] = *reinterpret_cast<const bf16x8*>(u_row + idx * 8);
        }
    }
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int idx = t + i * kC * kT;
        if (idx < nvec) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 gf = __bfloat1622float2(g[i].h[k]);
                const float2 uf = __bfloat1622float2(u[i].h[k]);
                float2 of;
                of.x = silu_f(gf.x) * uf.x;
                of.y = silu_f(gf.y) * uf.y;
                g[i].h[k] = __float22bfloat162_rn(of);  // the product swiglu would store
                const float2 r = __bfloat1622float2(g[i].h[k]);
                amax = fmaxf(amax, fmaxf(fabsf(r.x), fabsf(r.y)));
            }
        }
    }
    amax = block_reduce_max(amax, smem);
    if constexpr (kC > 1) {
        __shared__ float part;
        if (threadIdx.x == 0) part = amax;
        cluster_sync();  // every block's part is written
        if (threadIdx.x < kWarpSize) {
            float m = threadIdx.x < kC ? ld_cluster_f32(&part, threadIdx.x) : 0.f;
            m = warp_reduce_max(m);
            if (threadIdx.x == 0) smem[0] = m;
        }
        __syncthreads();
        amax = smem[0];
    }
    const float s = pow2_scale(amax);
    if (rank == 0 && threadIdx.x == 0) scales[row] = s;
    const float inv = 1.0f / s;
    unsigned char* q_row = q + row * cols;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int idx = t + i * kC * kT;
        if (idx < nvec) {
            float f[8];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 r = __bfloat1622float2(g[i].h[k]);
                f[2 * k] = r.x;
                f[2 * k + 1] = r.y;
            }
            store8(q_row + idx * 8, f, inv);
        }
    }
    if constexpr (kC > 1) cluster_sync();  // no block exits while a peer may still read its part
}

// The cluster form with kR rows to a cluster, for more rows than there are clusters resident.
// The 5090 holds 41 clusters of 8 blocks of 128 threads at once (ncu's Max Active Clusters,
// about two blocks an SM where registers and warps would allow 12), so from 42 rows a cluster a
// row runs in waves, and every wave pays a row's whole latency: loads, the amax, the barrier
// and the read of the peers. Here a block of kT threads is kR groups of kT / kR, group j on
// row c * kR + j of cluster c, each group the block's part of its row as swiglu_fp8_kernel's
// block is, and one exchange carries the kR amaxes: after the barrier warp j of every block
// reads row j's kC of them. Each row's values, products and max are the ones swiglu_fp8_kernel
// computes, so the bits are too.
template <int kT, int VPT, int kC, int kR>
__global__ void __launch_bounds__(kT)
    swiglu_fp8_rows_kernel(const __nv_bfloat16* __restrict__ gate,
                           const __nv_bfloat16* __restrict__ up, unsigned char* __restrict__ q,
                           float* __restrict__ scales, int64_t rows, int cols, int64_t ld_gate,
                           int64_t ld_up) {
    constexpr int kTR = kT / kR;  // threads a row of the block
    constexpr int kW = kTR / kWarpSize;
    static_assert(kC > 1 && kR > 1 && kW >= 1 && kR <= kT / kWarpSize, "a warp per row");
    __shared__ float red[kR][kW];
    __shared__ float part[kR];
    __shared__ float fin[kR];
    const int j = static_cast<int>(threadIdx.x) / kTR;
    const int64_t row = static_cast<int64_t>(blockIdx.x / kC) * kR + j;
    const int rank = static_cast<int>(blockIdx.x % kC);
    const int t = rank * kTR + static_cast<int>(threadIdx.x) % kTR;
    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int wid = static_cast<int>(threadIdx.x) / kWarpSize;
    const int nvec = row < rows ? cols / 8 : 0;  // the last cluster's missing rows do nothing
    const __nv_bfloat16* g_row = gate + row * ld_gate;
    const __nv_bfloat16* u_row = up + row * ld_up;
    bf16x8 g[VPT];
    bf16x8 u[VPT];
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int idx = t + i * kC * kTR;
        if (idx < nvec) {
            g[i] = *reinterpret_cast<const bf16x8*>(g_row + idx * 8);
            u[i] = *reinterpret_cast<const bf16x8*>(u_row + idx * 8);
        }
    }
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int idx = t + i * kC * kTR;
        if (idx < nvec) {
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 gf = __bfloat1622float2(g[i].h[k]);
                const float2 uf = __bfloat1622float2(u[i].h[k]);
                float2 of;
                of.x = silu_f(gf.x) * uf.x;
                of.y = silu_f(gf.y) * uf.y;
                g[i].h[k] = __float22bfloat162_rn(of);
                const float2 r = __bfloat1622float2(g[i].h[k]);
                amax = fmaxf(amax, fmaxf(fabsf(r.x), fabsf(r.y)));
            }
        }
    }
    amax = warp_reduce_max(amax);
    if (lane == 0) red[j][wid % kW] = amax;
    __syncthreads();
    if (wid < kR) {
        float m = lane < kW ? red[wid][lane] : 0.f;
        m = warp_reduce_max(m);
        if (lane == 0) part[wid] = m;
    }
    cluster_sync();  // every block's kR parts are written
    if (wid < kR) {
        float m = lane < kC ? ld_cluster_f32(&part[wid], lane) : 0.f;
        m = warp_reduce_max(m);
        if (lane == 0) fin[wid] = m;
    }
    __syncthreads();
    const float s = pow2_scale(fin[j]);
    if (rank == 0 && threadIdx.x % kTR == 0 && row < rows) scales[row] = s;
    const float inv = 1.0f / s;
    unsigned char* q_row = q + row * cols;
#pragma unroll
    for (int i = 0; i < VPT; ++i) {
        const int idx = t + i * kC * kTR;
        if (idx < nvec) {
            float f[8];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float2 r = __bfloat1622float2(g[i].h[k]);
                f[2 * k] = r.x;
                f[2 * k + 1] = r.y;
            }
            store8(q_row + idx * 8, f, inv);
        }
    }
    cluster_sync();  // no block exits while a peer may still read its parts
}

struct SwigluFp8Args {
    const __nv_bfloat16* gate;
    const __nv_bfloat16* up;
    unsigned char* q;
    float* scales;
    int64_t rows;
    int cols;
    int64_t ld_gate, ld_up;
    cudaStream_t stream;
};

template <int kT, int VPT, int kC, int kR>
void swiglu_fp8_launch(const SwigluFp8Args& a) {
    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = kC;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(static_cast<unsigned>(cdiv64(a.rows, kR) * kC));
    cfg.blockDim = dim3(kT);
    cfg.stream = a.stream;
    cfg.attrs = attr;
    cfg.numAttrs = kC > 1 ? 1 : 0;
    if constexpr (kR == 1) {
        SPARK_CUDA_CHECK(cudaLaunchKernelEx(&cfg, swiglu_fp8_kernel<kT, VPT, kC>, a.gate, a.up, a.q,
                                            a.scales, a.cols, a.ld_gate, a.ld_up));
    } else {
        SPARK_CUDA_CHECK(cudaLaunchKernelEx(&cfg, swiglu_fp8_rows_kernel<kT, VPT, kC, kR>, a.gate,
                                            a.up, a.q, a.scales, a.rows, a.cols, a.ld_gate,
                                            a.ld_up));
    }
}

template <int kT, int kC, int kR>
void swiglu_fp8_vpt(int vpt, const SwigluFp8Args& a) {
    if (vpt == 1) {
        swiglu_fp8_launch<kT, 1, kC, kR>(a);
    } else if (vpt == 2) {
        swiglu_fp8_launch<kT, 2, kC, kR>(a);
    } else {
        swiglu_fp8_launch<kT, 4, kC, kR>(a);
    }
}

template <int kC, int kR>
void swiglu_fp8_block(int kt, int vpt, const SwigluFp8Args& a) {
    if (kt == 128) {
        swiglu_fp8_vpt<128, kC, kR>(vpt, a);
    } else if (kt == 256) {
        swiglu_fp8_vpt<256, kC, kR>(vpt, a);
    } else if (kt == 512) {
        swiglu_fp8_vpt<512, kC, kR>(vpt, a);
    } else {
        swiglu_fp8_vpt<1024, kC, kR>(vpt, a);
    }
}

template <int kC>
void swiglu_fp8_rows(int rpc, int kt, int vpt, const SwigluFp8Args& a) {
    if constexpr (kC == 1) {
        swiglu_fp8_block<1, 1>(kt, vpt, a);
    } else if (rpc == 4) {
        swiglu_fp8_block<kC, 4>(kt, vpt, a);
    } else if (rpc == 2) {
        swiglu_fp8_block<kC, 2>(kt, vpt, a);
    } else {
        swiglu_fp8_block<kC, 1>(kt, vpt, a);
    }
}

// Blocks per row. One block a row puts a whole row's silu (an exp and a division per value) on
// one SM, and at 1 to 8 rows of 14336 that is slower than spreading it: 3.1 us against 2.3 for
// a cluster of 8. A cluster pays only while every cluster of the launch is resident at once:
// the 5090 holds 41 clusters of 8 (ncu's Max Active Clusters, about two blocks an SM; 85 of
// 4), and the 42nd row starts a second wave that pays a row's whole latency again, 2.7 us at
// 41 rows of 14336 and 4.2 at 42. So a cluster of 8 up to 41 rows of 8192 values or more and
// up to 16 rows from 4096 (at 5632 one block of 1024 threads ties at 16 rows and wins from 24),
// a cluster of 4 from 42 to 64 rows of 14336 or more, one block a row otherwise.
// SPARK_SWIGLU_FP8_CLUSTER=1|2|4|8 forces the cluster size. Then the block: up to 128 rows the
// smallest block of 128 to 1024 threads that gives each thread one vector of the block's part,
// since so few rows are latency bound (a few warps an SM, each waiting on its exps and
// divisions), and past that the smallest one at four vectors a thread, which wins from 192
// rows, past the 170 blocks of 1024 threads the SMs hold at once.
// SPARK_SWIGLU_FP8_VPT=1|2|4 forces that target. SPARK_SWIGLU_FP8_ROWS=2|4 gives a cluster
// that many rows (swiglu_fp8_rows_kernel, the target then per row of the block, and fewer rows
// if they would not fit 1024 threads); it fits 64 or 128 rows in one wave but is never the
// fastest, so only the variable picks it. None of this changes a bit of the output.
constexpr int64_t kClusterRows8 = 41;
constexpr int64_t kOneVectorRows = 128;

void swiglu_fp8_impl(const __nv_bfloat16* gate, const __nv_bfloat16* up, unsigned char* q,
                     float* scales, int64_t rows, int64_t cols, int64_t ld_gate, int64_t ld_up,
                     cudaStream_t stream) {
    SPARK_REQUIRE(gate != nullptr && up != nullptr && q != nullptr && scales != nullptr,
                  "swiglu_fp8: null pointer");
    SPARK_REQUIRE(rows >= 0 && rows < (int64_t{1} << 28) && cols > 0,
                  "swiglu_fp8: rows must be in [0, 2^28) and cols > 0");
    SPARK_REQUIRE(cols % 8 == 0 && cols <= 1024 * 32,
                  "swiglu_fp8: cols must be a multiple of 8 and at most 32768 (row in registers)");
    SPARK_REQUIRE(rows <= 1 || (ld_gate >= cols && ld_up >= cols),
                  "swiglu_fp8: row stride shorter than the row");
    SPARK_REQUIRE(ld_gate % 8 == 0 && ld_up % 8 == 0 && is_aligned16(gate) && is_aligned16(up),
                  "swiglu_fp8: gate and up rows must start on 16-byte boundaries");
    SPARK_REQUIRE(reinterpret_cast<uintptr_t>(q) % 8 == 0, "swiglu_fp8: q must be 8-byte aligned");
    if (rows == 0) return;
    static const int forced = [] {
        const char* e = std::getenv("SPARK_SWIGLU_FP8_CLUSTER");
        return e != nullptr ? std::atoi(e) : 0;
    }();
    static const int forced_rows = [] {
        const char* e = std::getenv("SPARK_SWIGLU_FP8_ROWS");
        return e != nullptr ? std::atoi(e) : 0;
    }();
    static const int forced_vpt = [] {
        const char* e = std::getenv("SPARK_SWIGLU_FP8_VPT");
        return e != nullptr ? std::atoi(e) : 0;
    }();
    const SwigluFp8Args a{gate,    up,    q,     scales, rows, static_cast<int>(cols),
                          ld_gate, ld_up, stream};
    const int nvec = a.cols / 8;
    int cl = 1;
    if (forced == 1 || forced == 2 || forced == 4 || forced == 8) {
        cl = forced;
    } else if ((rows <= kClusterRows8 && nvec >= 1024) || (rows <= 16 && nvec >= 512)) {
        cl = 8;
    } else if (rows <= 64 && nvec >= 1792) {
        cl = 4;
    }
    const int per_block = cdiv(nvec, cl);
    int rpc = cl > 1 && (forced_rows == 2 || forced_rows == 4) ? forced_rows : 1;
    while (rpc > 1 && per_block > 4 * 1024 / rpc) rpc /= 2;
    int vt = rows <= kOneVectorRows ? 1 : 4;  // vectors a thread the block is sized for
    if (forced_vpt == 1 || forced_vpt == 2 || forced_vpt == 4) vt = forced_vpt;
    int ktr = 128 / rpc;  // threads a row of the block, kt = rpc * ktr of them in all
    while (ktr < 1024 / rpc && ktr * vt < per_block) ktr *= 2;
    const int kt = rpc * ktr;
    const int need = cdiv(per_block, ktr);
    const int vpt = need <= 1 ? 1 : need <= 2 ? 2 : 4;
    if (cl == 1) {
        swiglu_fp8_rows<1>(rpc, kt, vpt, a);
    } else if (cl == 2) {
        swiglu_fp8_rows<2>(rpc, kt, vpt, a);
    } else if (cl == 4) {
        swiglu_fp8_rows<4>(rpc, kt, vpt, a);
    } else {
        swiglu_fp8_rows<8>(rpc, kt, vpt, a);
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace

int swiglu_num_variants() {
    return 2;
}

void swiglu_strided_f32(const float* gate, const float* up, float* out, int64_t rows, int64_t cols,
                        int64_t ld_gate, int64_t ld_up, cudaStream_t stream) {
    swiglu_strided_impl<float>(gate, up, out, rows, cols, ld_gate, ld_up, stream);
}

void swiglu_strided_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
                         int64_t rows, int64_t cols, int64_t ld_gate, int64_t ld_up,
                         cudaStream_t stream) {
    swiglu_strided_impl<__nv_bfloat16>(gate, up, out, rows, cols, ld_gate, ld_up, stream);
}

void swiglu_fp8_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up, unsigned char* q,
                     float* scales, int64_t rows, int64_t cols, int64_t ld_gate, int64_t ld_up,
                     cudaStream_t stream) {
    swiglu_fp8_impl(gate, up, q, scales, rows, cols, ld_gate, ld_up, stream);
}

void swiglu_f32(const float* gate, const float* up, float* out, int64_t n, int variant,
                cudaStream_t stream) {
    swiglu_impl<float>(gate, up, out, n, variant, stream);
}

void swiglu_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out, int64_t n,
                 int variant, cudaStream_t stream) {
    swiglu_impl<__nv_bfloat16>(gate, up, out, n, variant, stream);
}

}  // namespace spark
