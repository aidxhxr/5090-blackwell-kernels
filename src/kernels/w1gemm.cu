// W1A16 GEMM: C[M,N] = A[M,K] * dequant(W)[K,N], bf16 activations, 1-bit (sign) or 2-bit
// (ternary) weights with a bf16 scale per 128 k of a column, fp32 accumulation.
// Variants (the ladder of include/spark/kernels.h):
//   0  one thread per output, scalar dequant from the packed words
//   1  one warp per 16-column strip and 8 tokens over the whole K, the packed bits straight
//      into registers, selected into the bf16x2 mma.sync A fragment
//   2  M <= 16: variant 1's step with the block's four warps splitting K, the next groups'
//      weights in flight in registers, the partial sums added through shared memory;
//      M > 16: variant 1's kernel over 8-token chunks (see launch_v2)
// The packed layout, the k order inside a group and the dequant: w1gemm_internal.cuh.
#include <algorithm>

#include "spark/common.cuh"
#include "spark/kernels.h"
#include "w1gemm_internal.cuh"

namespace spark {

namespace {

constexpr int kG = kW1GroupSize;

// 16 bytes of weights, read once per call: no L1 allocation.
__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ unsigned word_of(const uint4& v, int j) {
    return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w;
}

// The dequantized weight of one code, as the kernels and the reference see it.
__device__ __forceinline__ float dequant_scalar(int code, float s, int bits) {
    const float v = bits == 1 ? (code ? s : -s) : s * static_cast<float>(code - 1);
    return __bfloat162float(__float2bfloat16(v));
}

// ---- quantizer ------------------------------------------------------------------------------

// One thread per (group, column): 128 strided reads of W (coalesced across the warp, whose
// threads hold consecutive columns), the mean of |w|, then the codes packed k ascending.
template <int BITS>
__global__ void quantize_kernel(const __nv_bfloat16* __restrict__ W, uint32_t* __restrict__ packed,
                                __nv_bfloat16* __restrict__ scales, int K, int N) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int grp = blockIdx.y;
    if (n >= N) return;
    const int k0 = grp * kG;
    float sum = 0.f;
    for (int k = k0; k < k0 + kG; ++k)
        sum += fabsf(__bfloat162float(W[static_cast<size_t>(k) * N + n]));
    const __nv_bfloat16 sb = __float2bfloat16(sum / static_cast<float>(kG));
    const float s = __bfloat162float(sb);
    scales[static_cast<size_t>(grp) * N + n] = sb;
    const int wpr = w1::words_per_row(K, BITS);
    uint32_t* out = packed + static_cast<size_t>(n) * wpr + grp * (4 * BITS);
    constexpr int per_word = 32 / BITS;
#pragma unroll 1
    for (int w = 0; w < 4 * BITS; ++w) {
        uint32_t word = 0;
        for (int t = 0; t < per_word; ++t) {
            const float x = __bfloat162float(W[static_cast<size_t>(k0 + w * per_word + t) * N + n]);
            uint32_t code;
            if constexpr (BITS == 1) {
                code = x >= 0.f ? 1u : 0u;
            } else {
                const float h = 0.5f * s;
                code = x < -h ? 0u : (x > h ? 2u : 1u);
            }
            word |= code << (BITS * t);
        }
        out[w] = word;
    }
}

// ---- variant 0: one thread per output element -------------------------------------------

__global__ void w1_naive_kernel(const __nv_bfloat16* __restrict__ A,
                                const uint32_t* __restrict__ packed,
                                const __nv_bfloat16* __restrict__ scales,
                                __nv_bfloat16* __restrict__ C, int M, int N, int K, int bits) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const float s = __bfloat162float(scales[static_cast<size_t>(k / kG) * N + n]);
        const float w = dequant_scalar(w1::code_at(packed, n, k, K, bits), s, bits);
        acc = fmaf(__bfloat162float(A[static_cast<size_t>(m) * K + k]), w, acc);
    }
    C[static_cast<size_t>(m) * N + n] = __float2bfloat16(acc);
}

// ---- pieces shared by the tensor core variants ------------------------------------------

// One row's packed codes of one 128-k group: 16 bytes at 1 bit, 32 at 2 bits.
template <int BITS>
struct GroupWords {
    uint4 v[BITS];
    // Load the group `grp` of row `row` (packed already the matrix's base).
    __device__ __forceinline__ void load(const uint32_t* packed, int row, int grp, int wpr) {
        const uint4* p = reinterpret_cast<const uint4*>(packed + static_cast<size_t>(row) * wpr +
                                                        grp * 4 * BITS);
#pragma unroll
        for (int b = 0; b < BITS; ++b) v[b] = ldg_stream(p + b);
    }
    // The lane's 8 codes (8 BITS bits, k ascending) of step pair i for lane column c.
    __device__ __forceinline__ unsigned codes(int i, int c) const {
        if constexpr (BITS == 1) {
            return (word_of(v[0], i) >> (8 * c)) & 0xFFu;
        } else {
            const int wi = 2 * i + (c >> 1);
            return (word_of(v[wi >> 2], wi & 3) >> (16 * (c & 1))) & 0xFFFFu;
        }
    }
};

// The eight k16 steps of one 128-k group for one 16-column block: step j's A fragment comes
// from codes (j / 2) of the two rows (low half for even j, high for odd), its B fragments
// (NT token tiles) from the 16-byte activation load i = j / 2.
template <int BITS, int NT>
__device__ __forceinline__ void group_mma(float (&acc)[NT][4], const GroupWords<BITS>& w_lo,
                                          const GroupWords<BITS>& w_hi, unsigned ss_lo,
                                          unsigned ss_hi, const uint4 (&act)[NT][4], int c) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned q_lo = w_lo.codes(i, c), q_hi = w_hi.codes(i, c);
#pragma unroll
        for (int jj = 0; jj < 2; ++jj) {
            unsigned a[4];
            w1::dequant<BITS>(q_lo >> (4 * BITS * jj), q_hi >> (4 * BITS * jj), ss_lo, ss_hi, a);
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const uint4& v = act[nt][i];
                const unsigned b[2] = {jj == 0 ? v.x : v.z, jj == 0 ? v.y : v.w};
                mma_bf16_16816(acc[nt], a, b);
            }
        }
    }
}

// Store one m16n8 accumulator tile: rows are output columns n0 + g and n0 + g + 8, columns
// tokens m0 + 2c and m0 + 2c + 1, rows at or past m_end dropped.
__device__ __forceinline__ void store_tile(const float (&d)[4], __nv_bfloat16* C, int m_end,
                                           int N, int n0, int m0, int lane) {
    const int g = lane >> 2, c = lane & 3;
    const int m = m0 + 2 * c;
    if (m < m_end) {
        C[static_cast<size_t>(m) * N + n0 + g] = __float2bfloat16(d[0]);
        C[static_cast<size_t>(m) * N + n0 + g + 8] = __float2bfloat16(d[2]);
    }
    if (m + 1 < m_end) {
        C[static_cast<size_t>(m + 1) * N + n0 + g] = __float2bfloat16(d[1]);
        C[static_cast<size_t>(m + 1) * N + n0 + g + 8] = __float2bfloat16(d[3]);
    }
}

// One warp: 16 columns n0.. x the 8 tokens m0..min(m0 + 8, m_end) over the whole K. Per
// 128-k group: one (1 bit) or two (2 bits) 16-byte weight loads per row, two scales, four
// 16-byte activation loads, eight dequants and eight mma.sync. Nothing is loaded ahead.
template <int BITS>
__device__ __forceinline__ void strip_chunk(const __nv_bfloat16* __restrict__ A,
                                            const uint32_t* __restrict__ packed,
                                            const __nv_bfloat16* __restrict__ scales,
                                            __nv_bfloat16* __restrict__ C, int m0, int m_end,
                                            int N, int K, int n0, int lane) {
    const int g = lane >> 2, c = lane & 3;
    const int wpr = w1::words_per_row(K, BITS);
    const bool has_row = m0 + g < m_end;
    const uint4* arow =
        reinterpret_cast<const uint4*>(A + static_cast<size_t>(has_row ? m0 + g : 0) * K) + c;
    const unsigned short* sc = reinterpret_cast<const unsigned short*>(scales);

    float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
    for (int grp = 0; grp < K / kG; ++grp) {
        GroupWords<BITS> w_lo, w_hi;
        w_lo.load(packed, n0 + g, grp, wpr);
        w_hi.load(packed, n0 + g + 8, grp, wpr);
        const size_t srow = static_cast<size_t>(grp) * N + n0;
        const unsigned ss_lo = w1::splat(__ldg(sc + srow + g));
        const unsigned ss_hi = w1::splat(__ldg(sc + srow + g + 8));
        uint4 act[1][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            act[0][i] = has_row ? __ldg(arow + grp * 16 + 4 * i) : make_uint4(0, 0, 0, 0);
        group_mma<BITS, 1>(acc, w_lo, w_hi, ss_lo, ss_hi, act, c);
    }
    store_tile(acc[0], C, m_end, N, n0, m0, lane);
}

// ---- variant 1: registers only, one group at a time ----------------------------------------

// Block of 4 warps, each on its own 16-column strip and one 8-token chunk (blockIdx.y).
template <int BITS>
__global__ void __launch_bounds__(128)
    w1_reg_kernel(const __nv_bfloat16* __restrict__ A, const uint32_t* __restrict__ packed,
                  const __nv_bfloat16* __restrict__ scales, __nv_bfloat16* __restrict__ C, int M,
                  int N, int K) {
    const int t = blockIdx.x * 4 + (threadIdx.x >> 5);  // 16-column strip
    if (t >= N / 16) return;
    strip_chunk<BITS>(A, packed, scales, C, blockIdx.y * 8, M, N, K, t * 16, threadIdx.x & 31);
}

template <int BITS>
void launch_v1(const __nv_bfloat16* A, const uint32_t* packed, const __nv_bfloat16* scales,
               __nv_bfloat16* C, int M, int N, int K, cudaStream_t stream) {
    const dim3 grid(cdiv(N / 16, 4), cdiv(M, 8));
    w1_reg_kernel<BITS><<<grid, 128, 0, stream>>>(A, packed, scales, C, M, N, K);
    SPARK_CHECK_LAUNCH();
}

}  // namespace

void w1_quantize_bf16(const __nv_bfloat16* W, uint32_t* packed, __nv_bfloat16* scales, int K,
                      int N, int bits, cudaStream_t stream) {
    SPARK_REQUIRE(K > 0 && N > 0 && K % kG == 0 && N % 16 == 0,
                  "w1_quantize: needs K % 128 == 0 and N % 16 == 0");
    SPARK_REQUIRE(bits == 1 || bits == 2, "w1_quantize: bits must be 1 or 2");
    SPARK_REQUIRE(W && packed && scales, "w1_quantize: null pointer");
    const dim3 grid(cdiv(N, 128), K / kG);
    if (bits == 1)
        quantize_kernel<1><<<grid, 128, 0, stream>>>(W, packed, scales, K, N);
    else
        quantize_kernel<2><<<grid, 128, 0, stream>>>(W, packed, scales, K, N);
    SPARK_CHECK_LAUNCH();
}

int w1gemm_num_variants() {
    return 3;
}

bool w1gemm_supports(int M, int N, int K, int bits, int variant) {
    return variant >= 0 && variant < w1gemm_num_variants() && (bits == 1 || bits == 2) &&
           M >= 1 && N > 0 && K > 0 && N % 16 == 0 && K % kG == 0;
}

void w1gemm_bf16(const __nv_bfloat16* A, const uint32_t* packed, const __nv_bfloat16* scales,
                 __nv_bfloat16* C, int M, int N, int K, int bits, int variant,
                 cudaStream_t stream) {
    SPARK_REQUIRE(variant >= 0 && variant < w1gemm_num_variants(), "w1gemm: bad variant");
    SPARK_REQUIRE(w1gemm_supports(M, N, K, bits, variant),
                  "w1gemm: needs bits 1 or 2, M >= 1, N % 16 == 0 and K % 128 == 0");
    SPARK_REQUIRE(A && packed && scales && C, "w1gemm: null pointer");
    SPARK_REQUIRE(is_aligned16(A) && is_aligned16(packed) && is_aligned16(scales),
                  "w1gemm: A, packed and scales must be 16-byte aligned");
    switch (variant) {
        case 0: {
            const dim3 grid(cdiv(N, 128), M);
            w1_naive_kernel<<<grid, 128, 0, stream>>>(A, packed, scales, C, M, N, K, bits);
            SPARK_CHECK_LAUNCH();
            break;
        }
        case 1:
            if (bits == 1)
                launch_v1<1>(A, packed, scales, C, M, N, K, stream);
            else
                launch_v1<2>(A, packed, scales, C, M, N, K, stream);
            break;
        default:
            // The K-split rung is not written yet: variant 2 runs variant 1's kernel.
            if (bits == 1)
                launch_v1<1>(A, packed, scales, C, M, N, K, stream);
            else
                launch_v1<2>(A, packed, scales, C, M, N, K, stream);
            break;
    }
}

}  // namespace spark
