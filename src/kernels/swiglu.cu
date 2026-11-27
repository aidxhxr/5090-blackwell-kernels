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
//
// swiglu_softcap_* is Inkling 2's activation on the same kernels (the activation is a
// template parameter): softcap(g, 4) * sigmoid(g) * softcap(u, 25), softcap(x, c) =
// c * tanh(x / c), the sigmoid on the raw g (Inkling2LoopSoftcappedSwiGLU). The fp32 math
// follows the module op for op: x / c divided (not times 1 / c), tanhf, 1 / (1 + expf(-g)),
// the products in the module's order. torch's sigmoid on the GPU gave the same bits as this
// one on all 32K picks of a 4096-token routing; its tanh is not always tanhf's, so the output
// differs from the module's in the last bit now and then; in bf16, 26 of 4.7M outputs came
// out one bf16 ulp away (4096 x 1152, g ~ N(0, 6), u ~ N(0, 40)).
#include <cstdint>
#include <type_traits>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {
namespace {

constexpr int kBlock = 256;

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + __expf(-x));
}

// The activations: out = act(g, u) in fp32.
struct Silu {
    __device__ __forceinline__ float operator()(float g, float u) const { return silu_f(g) * u; }
};
struct SoftcapSilu {
    float cap_g, cap_u;
    __device__ __forceinline__ float operator()(float g, float u) const {
        const float sg = tanhf(g / cap_g) * cap_g;
        const float su = tanhf(u / cap_u) * cap_u;
        return sg * (1.0f / (1.0f + expf(-g))) * su;
    }
};

template <typename T, class Act = Silu>
__global__ void swiglu_scalar_kernel(const T* __restrict__ gate, const T* __restrict__ up,
                                     T* __restrict__ out, int64_t n, Act act = {}) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        const float g = to_f32(gate[i]);
        const float u = to_f32(up[i]);
        out[i] = from_f32<T>(act(g, u));
    }
}

// f32: 4 elements per thread via f32x4 (one 128-bit load per operand).
template <class Act = Silu>
__global__ void swiglu_vec_f32_kernel(const float* __restrict__ gate, const float* __restrict__ up,
                                      float* __restrict__ out, int64_t n, Act act = {}) {
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
        for (int k = 0; k < VEC; ++k) o.v[k] = act(g.v[k], u.v[k]);
        out4[i] = o;
    }

    // Scalar tail (n % 4 elements), handled by the first few threads of the grid.
    const int64_t tail_start = n_vec * VEC;
    for (int64_t i = tail_start + tid; i < n; i += stride) {
        out[i] = act(gate[i], up[i]);
    }
}

// bf16: 8 elements per thread via bf16x8 (4 x __nv_bfloat162 = 16 bytes). Math in fp32.
template <class Act = Silu>
__global__ void swiglu_vec_bf16_kernel(const __nv_bfloat16* __restrict__ gate,
                                       const __nv_bfloat16* __restrict__ up,
                                       __nv_bfloat16* __restrict__ out, int64_t n, Act act = {}) {
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
            of.x = act(gf.x, uf.x);
            of.y = act(gf.y, uf.y);
            o.h[k] = __float22bfloat162_rn(of);
        }
        out8[i] = o;
    }

    const int64_t tail_start = n_vec * VEC;
    for (int64_t i = tail_start + tid; i < n; i += stride) {
        const float g = __bfloat162float(gate[i]);
        const float u = __bfloat162float(up[i]);
        out[i] = __float2bfloat16(act(g, u));
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

template <typename T, class Act = Silu>
void swiglu_impl(const T* gate, const T* up, T* out, int64_t n, int variant, cudaStream_t stream,
                 Act act = {}) {
    SPARK_REQUIRE(gate != nullptr && up != nullptr && out != nullptr, "swiglu: null pointer");
    SPARK_REQUIRE(n >= 0, "swiglu: n must be >= 0");
    SPARK_REQUIRE(variant >= 0 && variant < swiglu_num_variants(), "swiglu: unknown variant");
    if (n == 0) return;

    if (variant == 0) {
        const int64_t blocks = cdiv64(n, kBlock);
        SPARK_REQUIRE(blocks < (int64_t{1} << 31), "swiglu: n too large for variant 0");
        swiglu_scalar_kernel<T, Act>
            <<<static_cast<unsigned>(blocks), kBlock, 0, stream>>>(gate, up, out, n, act);
        SPARK_CHECK_LAUNCH();
        return;
    }

    SPARK_REQUIRE(is_aligned16(gate) && is_aligned16(up) && is_aligned16(out),
                  "swiglu: variant 1 needs 16-byte aligned pointers");
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    const unsigned grid = vec_grid(n / VEC);
    if constexpr (sizeof(T) == 4) {
        swiglu_vec_f32_kernel<Act><<<grid, kBlock, 0, stream>>>(
            reinterpret_cast<const float*>(gate), reinterpret_cast<const float*>(up),
            reinterpret_cast<float*>(out), n, act);
    } else {
        swiglu_vec_bf16_kernel<Act>
            <<<grid, kBlock, 0, stream>>>(reinterpret_cast<const __nv_bfloat16*>(gate),
                                          reinterpret_cast<const __nv_bfloat16*>(up),
                                          reinterpret_cast<__nv_bfloat16*>(out), n, act);
    }
    SPARK_CHECK_LAUNCH();
}

// Strided rows: element (r, c) of gate is gate[r * ld_gate + c], of up is up[r * ld_up + c],
// out is contiguous rows x cols. Vectorized 16-byte loads when every row start and the
// row length are 16-byte multiples, one scalar element per thread otherwise.
template <typename T, int VEC, class Act = Silu>
__global__ void swiglu_strided_vec_kernel(const T* __restrict__ gate, const T* __restrict__ up,
                                          T* __restrict__ out, int64_t rows, int64_t cols_vec,
                                          int64_t ld_gate, int64_t ld_up, Act act = {}) {
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
            for (int k = 0; k < VEC; ++k) o.v[k] = act(g.v[k], u.v[k]);
        } else {
#pragma unroll
            for (int k = 0; k < VEC / 2; ++k) {
                const float2 gf = __bfloat1622float2(g.h[k]);
                const float2 uf = __bfloat1622float2(u.h[k]);
                float2 of;
                of.x = act(gf.x, uf.x);
                of.y = act(gf.y, uf.y);
                o.h[k] = __float22bfloat162_rn(of);
            }
        }
        *reinterpret_cast<V*>(out + r * cols_vec * VEC + c) = o;
    }
}

template <typename T, class Act = Silu>
__global__ void swiglu_strided_scalar_kernel(const T* __restrict__ gate, const T* __restrict__ up,
                                             T* __restrict__ out, int64_t rows, int64_t cols,
                                             int64_t ld_gate, int64_t ld_up, Act act = {}) {
    const int64_t n = rows * cols;
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = tid; i < n; i += stride) {
        const int64_t r = i / cols;
        const int64_t c = i - r * cols;
        const float g = to_f32(gate[r * ld_gate + c]);
        const float u = to_f32(up[r * ld_up + c]);
        out[i] = from_f32<T>(act(g, u));
    }
}

template <typename T, class Act = Silu>
void swiglu_strided_impl(const T* gate, const T* up, T* out, int64_t rows, int64_t cols,
                         int64_t ld_gate, int64_t ld_up, cudaStream_t stream, Act act = {}) {
    SPARK_REQUIRE(gate != nullptr && up != nullptr && out != nullptr, "swiglu: null pointer");
    SPARK_REQUIRE(rows >= 0 && cols >= 0, "swiglu: rows and cols must be >= 0");
    SPARK_REQUIRE(ld_gate >= cols && ld_up >= cols, "swiglu: row stride shorter than the row");
    if (rows == 0 || cols == 0) return;
    constexpr int VEC = 16 / static_cast<int>(sizeof(T));
    const bool vec = cols % VEC == 0 && ld_gate % VEC == 0 && ld_up % VEC == 0 &&
                     is_aligned16(gate) && is_aligned16(up) && is_aligned16(out);
    if (vec) {
        const unsigned grid = vec_grid(rows * (cols / VEC));
        swiglu_strided_vec_kernel<T, VEC, Act>
            <<<grid, kBlock, 0, stream>>>(gate, up, out, rows, cols / VEC, ld_gate, ld_up, act);
    } else {
        const unsigned grid = vec_grid(rows * cols);
        swiglu_strided_scalar_kernel<T, Act>
            <<<grid, kBlock, 0, stream>>>(gate, up, out, rows, cols, ld_gate, ld_up, act);
    }
    SPARK_CHECK_LAUNCH();
}

// The softcapped form: contiguous rows (ld == cols) run as one flat array through variant 1
// (variant 0 when a pointer is not 16-byte aligned), anything else through the strided kernels.
template <typename T>
void swiglu_softcap_impl(const T* gate, const T* up, T* out, int64_t rows, int64_t cols,
                         int64_t ld_gate, int64_t ld_up, float cap_gate, float cap_up,
                         cudaStream_t stream) {
    SPARK_REQUIRE(cap_gate > 0.f && cap_up > 0.f, "swiglu_softcap: the caps must be > 0");
    const SoftcapSilu act{cap_gate, cap_up};
    if (ld_gate == cols && ld_up == cols) {
        SPARK_REQUIRE(rows >= 0 && cols >= 0, "swiglu: rows and cols must be >= 0");
        const bool vec = is_aligned16(gate) && is_aligned16(up) && is_aligned16(out);
        swiglu_impl<T, SoftcapSilu>(gate, up, out, rows * cols, vec ? 1 : 0, stream, act);
        return;
    }
    swiglu_strided_impl<T, SoftcapSilu>(gate, up, out, rows, cols, ld_gate, ld_up, stream, act);
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

void swiglu_f32(const float* gate, const float* up, float* out, int64_t n, int variant,
                cudaStream_t stream) {
    swiglu_impl<float>(gate, up, out, n, variant, stream);
}

void swiglu_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out, int64_t n,
                 int variant, cudaStream_t stream) {
    swiglu_impl<__nv_bfloat16>(gate, up, out, n, variant, stream);
}

void swiglu_softcap_f32(const float* gate, const float* up, float* out, int64_t rows, int64_t cols,
                        int64_t ld_gate, int64_t ld_up, float cap_gate, float cap_up,
                        cudaStream_t stream) {
    swiglu_softcap_impl<float>(gate, up, out, rows, cols, ld_gate, ld_up, cap_gate, cap_up, stream);
}

void swiglu_softcap_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
                         int64_t rows, int64_t cols, int64_t ld_gate, int64_t ld_up, float cap_gate,
                         float cap_up, cudaStream_t stream) {
    swiglu_softcap_impl<__nv_bfloat16>(gate, up, out, rows, cols, ld_gate, ld_up, cap_gate, cap_up,
                                       stream);
}

}  // namespace spark
