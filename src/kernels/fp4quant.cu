// bf16 -> fp4 quantizer for the fp4 GEMM: packed e2m1 elements plus block scales in the
// blocked layout fp4gemm reads (kernels.h, fp4_quantize).
//
// One thread per scale block (16 values for NVFP4, 32 for MXFP4): it reads the block with
// 16-byte loads, takes its max |x|, rounds the block scale, converts the elements and writes
// them packed (8 or 16 bytes) plus one scale byte. Consecutive threads take consecutive blocks
// of a row, so the element loads and stores are contiguous across the warp; the scale bytes
// land four to a 16-byte row of the blocked tile. The kernel is a stream of 2 bytes in and
// 0.5 + 1/16 (1/32) bytes out per value, so it is bound by DRAM, not by the rounding.
//
// The rounding is written out with compares rather than cvt.rn.satfinite.e2m1x2 so that
// spark_kernels.reference can mirror it operation for operation (tests compare the bytes).

#include <cuda_fp8.h>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

constexpr int kThreads = 256;

// e2m1 code of v, round to nearest even, saturating at 6. The codes 0..7 are 0, 0.5, 1, 1.5,
// 2, 3, 4, 6; a tie goes to the even code (0.25 -> 0, 0.75 -> 1, 1.25 -> 1, 1.75 -> 2,
// 2.5 -> 2, 3.5 -> 4, 5 -> 4). The sign is bit 3, set for v < 0.
__device__ __forceinline__ unsigned e2m1_code(float v) {
    const float a = fabsf(v);
    const unsigned c = a <= 0.25f   ? 0u
                       : a < 0.75f  ? 1u
                       : a <= 1.25f ? 2u
                       : a < 1.75f  ? 3u
                       : a <= 2.5f  ? 4u
                       : a < 3.5f   ? 5u
                       : a <= 5.0f  ? 6u
                                    : 7u;
    return c | (v < 0.f ? 8u : 0u);
}

// Blocked-layout byte of scale column j (k-block j) of row r: 4 scale columns per 128-row tile.
__device__ __forceinline__ size_t sf_offset(int r, int j, int cols) {
    const size_t tile = static_cast<size_t>(r / 128) * (cols / 4) + j / 4;
    return tile * 512 + (r % 32) * 16 + ((r / 32) % 4) * 4 + (j % 4);
}

// V values of a block into floats (V / 8 16-byte loads).
template <int V>
__device__ __forceinline__ void load_block(float (&f)[V], const __nv_bfloat16* p) {
#pragma unroll
    for (int i = 0; i < V / 8; ++i) {
        const bf16x8 v = reinterpret_cast<const bf16x8*>(p)[i];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 t = __bfloat1622float2(v.h[j]);
            f[8 * i + 2 * j] = t.x;
            f[8 * i + 2 * j + 1] = t.y;
        }
    }
}

// Packs V codes (V / 8 words, element 2i in the low nibble of byte i) and stores them.
template <int V>
__device__ __forceinline__ void store_codes(unsigned char* q, const unsigned (&c)[V]) {
    unsigned w[V / 8];
#pragma unroll
    for (int i = 0; i < V / 8; ++i) {
        w[i] = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) w[i] |= c[8 * i + j] << (4 * j);
    }
    if constexpr (V == 16) {
        *reinterpret_cast<uint2*>(q) = make_uint2(w[0], w[1]);
    } else {
        *reinterpret_cast<uint4*>(q) = make_uint4(w[0], w[1], w[2], w[3]);
    }
}

// NVFP4: 16 values per block, an e4m3 block scale over the per-tensor decode scale s.
__global__ void __launch_bounds__(kThreads)
    quantize_nvfp4_kernel(const __nv_bfloat16* __restrict__ x, unsigned char* __restrict__ q,
                          unsigned char* __restrict__ sf, int rows, int padded_rows, int K,
                          const float* __restrict__ scale, const unsigned* __restrict__ amax_bits,
                          float* __restrict__ scale_out) {
    const int cols = K / 16;
    const long long idx = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    // With amax_bits the per-tensor scale is the recipe's max|x| / (6 * 448) from the amax
    // pass (absmax_bits), clamped to the smallest normal float, and block 0 publishes it.
    const float s = amax_bits != nullptr
                        ? fmaxf(__uint_as_float(*amax_bits) / (6.0f * 448.0f), 1.17549435e-38f)
                    : scale != nullptr ? *scale
                                       : 1.0f;
    if (amax_bits != nullptr && idx == 0) *scale_out = s;
    if (idx >= static_cast<long long>(padded_rows) * cols) return;
    const int r = static_cast<int>(idx / cols);
    const int j = static_cast<int>(idx % cols);
    if (r >= rows) {
        sf[sf_offset(r, j, cols)] = 0;
        return;
    }
    float f[16];
    load_block<16>(f, x + static_cast<size_t>(r) * K + j * 16);
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(f[i]));
    // The block scale: the e4m3 value nearest max / 6 / s, so the largest element maps to
    // about 6 (the top of e2m1) after dividing by s * scale. The constructor rounds to
    // nearest even and saturates at 448.
    const __nv_fp8_e4m3 sfe(fminf(amax / 6.0f / s, 448.0f));
    const float denom = static_cast<float>(sfe) * s;
    unsigned c[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) c[i] = denom > 0.f ? e2m1_code(f[i] / denom) : 0u;
    store_codes<16>(q + static_cast<size_t>(r) * (K / 2) + j * 8, c);
    sf[sf_offset(r, j, cols)] = sfe.__x;
}

// MXFP4: 32 values per block, a power-of-two scale 2^e with e = floor(log2(max)) - 2.
__global__ void __launch_bounds__(kThreads)
    quantize_mxfp4_kernel(const __nv_bfloat16* __restrict__ x, unsigned char* __restrict__ q,
                          unsigned char* __restrict__ sf, int rows, int padded_rows, int K) {
    const int cols = K / 32;
    const long long idx = static_cast<long long>(blockIdx.x) * kThreads + threadIdx.x;
    if (idx >= static_cast<long long>(padded_rows) * cols) return;
    const int r = static_cast<int>(idx / cols);
    const int j = static_cast<int>(idx % cols);
    if (r >= rows) {
        sf[sf_offset(r, j, cols)] = 0;
        return;
    }
    float f[32];
    load_block<32>(f, x + static_cast<size_t>(r) * K + j * 32);
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(f[i]));
    // floor(log2(amax)) through frexp (amax = m * 2^p, m in [0.5, 1)), exact where a log2
    // would round a value just under a power of two up to it; e2m1's top binade is 2^2.
    int p = 0;
    frexpf(amax, &p);
    int e = amax > 0.f ? p - 1 - 2 : -127;
    e = max(-127, min(127, e));
    unsigned c[32];
#pragma unroll
    for (int i = 0; i < 32; ++i) c[i] = e2m1_code(ldexpf(f[i], -e));
    store_codes<32>(q + static_cast<size_t>(r) * (K / 2) + j * 16, c);
    sf[sf_offset(r, j, cols)] = static_cast<unsigned char>(e + 127);
}

}  // namespace

void fp4_quantize(const __nv_bfloat16* x, unsigned char* q, unsigned char* sf, int rows, int K,
                  const float* scale, int format, cudaStream_t stream, unsigned* work,
                  float* scale_out) {
    SPARK_REQUIRE(x != nullptr && q != nullptr && sf != nullptr, "fp4_quantize: null pointer");
    SPARK_REQUIRE(format == FP4_NVFP4 || format == FP4_MXFP4, "fp4_quantize: unknown format");
    SPARK_REQUIRE(rows > 0 && K > 0, "fp4_quantize: rows and K must be positive");
    SPARK_REQUIRE(K % (format == FP4_MXFP4 ? 128 : 64) == 0,
                  "fp4_quantize: K must be a multiple of 64 (NVFP4) or 128 (MXFP4)");
    SPARK_REQUIRE(is_aligned16(x) && is_aligned16(q) && is_aligned16(sf),
                  "fp4_quantize: x, q, sf must be 16-byte aligned");
    const int padded = cdiv(rows, 128) * 128;
    const int cols = K / (format == FP4_MXFP4 ? 32 : 16);
    const long long threads = static_cast<long long>(padded) * cols;
    const unsigned grid = static_cast<unsigned>((threads + kThreads - 1) / kThreads);
    SPARK_REQUIRE(work == nullptr || (format == FP4_NVFP4 && scale_out != nullptr),
                  "fp4_quantize: a dynamic scale needs NVFP4 and a scale_out");
    if (work != nullptr) absmax_bits(x, static_cast<long long>(rows) * K, work, stream);
    if (format == FP4_NVFP4)
        quantize_nvfp4_kernel<<<grid, kThreads, 0, stream>>>(x, q, sf, rows, padded, K, scale, work,
                                                             scale_out);
    else
        quantize_mxfp4_kernel<<<grid, kThreads, 0, stream>>>(x, q, sf, rows, padded, K);
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
