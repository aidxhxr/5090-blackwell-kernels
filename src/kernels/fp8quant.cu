// bf16 -> e4m3 activation quantizer for the fp8 GEMM (kernels.h, fp8_quantize), so a W8A8
// projection quantizes its input in one or two launches instead of a chain of torch ops.
//
// Three scalings:
//   tensor  one scale for the whole [rows][K]: an amax pass (grid-stride, one atomicMax of the
//           float's bits per block into a zeroed cell; the bits of non-negative floats order
//           like the floats) and a quantize pass that reads it. Two launches.
//   row     one scale per row: a block per row takes the row's amax, then quantizes the row,
//           which it reads a second time out of L1/L2. One launch.
//   mx      the OCP MX recipe, a power-of-two scale per 32 consecutive values of a row (e8m0,
//           e = floor(log2(max)) - 8), one thread per block of 32, as fp4quant.cu's MXFP4.
// The tensor and row scales are powers of two too: the smallest 2^e with amax / 2^e <= 448,
// so x * 2^-e is exact in fp32 and the conversion to e4m3 (round to nearest even, saturating)
// is the only rounding. e4m3 is a float format, so the range a power of two gives up (at most
// a factor 2) costs nothing in relative precision above its subnormals.
// spark_kernels.reference.quantize_fp8_pow2 and quantize_mx are the same arithmetic.

#include <cuda_fp8.h>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

constexpr int kThreads = 256;

// The smallest power of two s with amax / s <= 448, e clamped to [-126, 127] (so 1 / s is a
// finite normal float, also for an all-zero input).
__device__ __forceinline__ float pow2_scale(float amax) {
    const float r = fmaxf(amax, 1.17549435e-38f) / 448.0f;
    int p = 0;
    const float m = frexpf(r, &p);  // r = m * 2^p, m in [0.5, 1)
    int e = m == 0.5f ? p - 1 : p;  // ceil(log2(r))
    e = max(-126, min(127, e));
    return ldexpf(1.0f, e);
}

__device__ __forceinline__ void load8(float (&f)[8], const __nv_bfloat16* p) {
    const bf16x8 v = *reinterpret_cast<const bf16x8*>(p);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 t = __bfloat1622float2(v.h[j]);
        f[2 * j] = t.x;
        f[2 * j + 1] = t.y;
    }
}

__device__ __forceinline__ float amax8(const float (&f)[8]) {
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) a = fmaxf(a, fabsf(f[i]));
    return a;
}

// Eight values times inv, converted to e4m3 and stored as 8 bytes.
__device__ __forceinline__ void store8(unsigned char* q, const float (&f)[8], float inv) {
    unsigned w[2] = {0u, 0u};
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const __nv_fp8_e4m3 v(f[i] * inv);
        w[i / 4] |= static_cast<unsigned>(v.__x) << (8 * (i % 4));
    }
    *reinterpret_cast<uint2*>(q) = make_uint2(w[0], w[1]);
}

__global__ void __launch_bounds__(kThreads)
    amax_kernel(const __nv_bfloat16* __restrict__ x, long long chunks,
                unsigned* __restrict__ amax_bits) {
    __shared__ float smem[32];
    float a = 0.f;
    for (long long c = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x; c < chunks;
         c += static_cast<long long>(gridDim.x) * kThreads) {
        float f[8];
        load8(f, x + 8 * c);
        a = fmaxf(a, amax8(f));
    }
    a = block_reduce_max(a, smem);
    if (threadIdx.x == 0) atomicMax(amax_bits, __float_as_uint(a));
}

__global__ void __launch_bounds__(kThreads)
    quantize_tensor_kernel(const __nv_bfloat16* __restrict__ x, unsigned char* __restrict__ q,
                           long long chunks, const unsigned* __restrict__ amax_bits,
                           float* __restrict__ scale) {
    const float s = pow2_scale(__uint_as_float(*amax_bits));
    if (blockIdx.x == 0 && threadIdx.x == 0) *scale = s;
    const long long c = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    if (c >= chunks) return;
    float f[8];
    load8(f, x + 8 * c);
    store8(q + 8 * c, f, 1.0f / s);
}

__global__ void __launch_bounds__(kThreads)
    quantize_row_kernel(const __nv_bfloat16* __restrict__ x, unsigned char* __restrict__ q,
                        int K, float* __restrict__ scales) {
    __shared__ float smem[32];
    const size_t base = static_cast<size_t>(blockIdx.x) * K;
    const int chunks = K / 8;
    float a = 0.f;
    for (int c = threadIdx.x; c < chunks; c += kThreads) {
        float f[8];
        load8(f, x + base + 8 * c);
        a = fmaxf(a, amax8(f));
    }
    const float s = pow2_scale(block_reduce_max(a, smem));
    if (threadIdx.x == 0) scales[blockIdx.x] = s;
    const float inv = 1.0f / s;
    for (int c = threadIdx.x; c < chunks; c += kThreads) {
        float f[8];
        load8(f, x + base + 8 * c);
        store8(q + base + 8 * c, f, inv);
    }
}

__global__ void __launch_bounds__(kThreads)
    quantize_mx_kernel(const __nv_bfloat16* __restrict__ x, unsigned char* __restrict__ q,
                       unsigned char* __restrict__ sf, long long blocks) {
    const long long b = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    if (b >= blocks) return;
    float f[4][8];
    float a = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        load8(f[i], x + 32 * b + 8 * i);
        a = fmaxf(a, amax8(f[i]));
    }
    // floor(log2(amax)) - 8 through frexp, exact at powers of two; e4m3's top binade is 2^8.
    int p = 0;
    frexpf(a, &p);
    int e = a > 0.f ? p - 1 - 8 : -127;
    e = max(-127, min(127, e));
    const float inv = ldexpf(1.0f, -e);
#pragma unroll
    for (int i = 0; i < 4; ++i) store8(q + 32 * b + 8 * i, f[i], inv);
    sf[b] = static_cast<unsigned char>(e + 127);
}

}  // namespace

void fp8_quantize(const __nv_bfloat16* x, unsigned char* q, void* scale, unsigned* work,
                  int rows, int K, int mode, cudaStream_t stream) {
    SPARK_REQUIRE(x != nullptr && q != nullptr && scale != nullptr, "fp8_quantize: null pointer");
    SPARK_REQUIRE(rows > 0 && K > 0, "fp8_quantize: rows and K must be positive");
    SPARK_REQUIRE(K % (mode == FP8Q_MX ? 32 : 8) == 0,
                  "fp8_quantize: K must be a multiple of 8 (32 for mx)");
    SPARK_REQUIRE(is_aligned16(x), "fp8_quantize: x must be 16-byte aligned");
    SPARK_REQUIRE(reinterpret_cast<uintptr_t>(q) % 8 == 0, "fp8_quantize: q must be 8-byte aligned");
    const long long n = static_cast<long long>(rows) * K;
    if (mode == FP8Q_TENSOR) {
        SPARK_REQUIRE(work != nullptr, "fp8_quantize: the tensor mode needs a zeroed work cell");
        const long long chunks = n / 8;
        const int grid_a = static_cast<int>(std::min<long long>(cdiv64(chunks, kThreads),
                                                                4LL * num_sms()));
        amax_kernel<<<grid_a, kThreads, 0, stream>>>(x, chunks, work);
        SPARK_CHECK_LAUNCH();
        quantize_tensor_kernel<<<static_cast<unsigned>(cdiv64(chunks, kThreads)), kThreads, 0,
                                 stream>>>(x, q, chunks, work, static_cast<float*>(scale));
    } else if (mode == FP8Q_ROW) {
        quantize_row_kernel<<<rows, kThreads, 0, stream>>>(x, q, K, static_cast<float*>(scale));
    } else {
        SPARK_REQUIRE(mode == FP8Q_MX, "fp8_quantize: unknown mode");
        const long long blocks = n / 32;
        quantize_mx_kernel<<<static_cast<unsigned>(cdiv64(blocks, kThreads)), kThreads, 0,
                             stream>>>(x, q, static_cast<unsigned char*>(scale), blocks);
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
