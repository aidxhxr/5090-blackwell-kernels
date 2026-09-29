// int4 weight quantization (round to nearest, one scale per 128 k of a column) and the offline
// repack into the layout the w4gemm kernels read. Both run once per weight, at load time.
// The formulas are in include/spark/kernels.h; the layout in docs/design/w4gemm.md.
#include "spark/common.cuh"
#include "spark/kernels.h"
#include "w4gemm_internal.cuh"

namespace spark {

namespace {

// One thread per (group, column): 128 strided reads of W (coalesced across the warp, whose
// threads hold consecutive columns), the scale, then sixteen packed words of 8 nibbles.
template <bool ASYM>
__global__ void quantize_kernel(const __nv_bfloat16* __restrict__ W, int32_t* __restrict__ qw,
                                __nv_bfloat16* __restrict__ scales, uint8_t* __restrict__ zeros,
                                int K, int N) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int grp = blockIdx.y;
    if (n >= N) return;
    const __nv_bfloat16* col = W + static_cast<size_t>(grp) * kW4GroupSize * N + n;
    float lo = 0.f, hi = 0.f, amax = 0.f;
    for (int k = 0; k < kW4GroupSize; ++k) {
        const float w = __bfloat162float(col[static_cast<size_t>(k) * N]);
        lo = fminf(lo, w);
        hi = fmaxf(hi, w);
        amax = fmaxf(amax, fabsf(w));
    }
    // The scale is rounded to bf16 first and the codes computed against the rounded value,
    // so (q - z) * s reconstructs from exactly the numbers the kernel will see.
    const __nv_bfloat16 sb = __float2bfloat16(ASYM ? (hi - lo) / 15.0f : amax / 7.0f);
    const float s = __bfloat162float(sb);
    int z = 8;
    if (ASYM) z = s > 0.f ? static_cast<int>(fminf(fmaxf(rintf(-lo / s), 0.f), 15.f)) : 0;
    const int qmin = ASYM ? 0 : -8, qmax = ASYM ? 15 : 7, off = ASYM ? z : 8;
    for (int k8 = 0; k8 < kW4GroupSize / 8; ++k8) {
        uint32_t word = 0;
        for (int i = 0; i < 8; ++i) {
            const float w = __bfloat162float(col[static_cast<size_t>(k8 * 8 + i) * N]);
            int q = off;
            if (s > 0.f) {
                const float r = rintf(w / s) + static_cast<float>(ASYM ? z : 0);
                q = static_cast<int>(
                        fminf(fmaxf(r, static_cast<float>(qmin)), static_cast<float>(qmax))) +
                    (ASYM ? 0 : 8);
            }
            word |= static_cast<uint32_t>(q) << (4 * i);
        }
        qw[static_cast<size_t>(grp * (kW4GroupSize / 8) + k8) * N + n] = static_cast<int32_t>(word);
    }
    scales[static_cast<size_t>(grp) * N + n] = sb;
    if (ASYM) zeros[static_cast<size_t>(grp) * N + n] = static_cast<uint8_t>(z);
}

// One thread per packed word: the eight nibbles it gathers from qweight are the ones lane
// `lane` feeds to the mma for k16 step `j` of its 64-k block (w4::nibble_coords).
__global__ void repack_kernel(const int32_t* __restrict__ qw, int32_t* __restrict__ packed, int K,
                              int N) {
    const int64_t total = static_cast<int64_t>(N / 16) * (K / 64) * 128;
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int j = static_cast<int>(i & 3);
    const int lane = static_cast<int>((i >> 2) & 31);
    const int64_t blk = i >> 7;  // (t, u) block of 16 columns x 64 k, strip-major
    const int u = static_cast<int>(blk % (K / 64));
    const int t = static_cast<int>(blk / (K / 64));
    uint32_t word = 0;
#pragma unroll
    for (int p = 0; p < 8; ++p) {
        int dn, dk;
        w4::nibble_coords(lane, u, j, p, dn, dk);
        const int n = t * 16 + dn;
        const int k = u * 64 + dk;
        const uint32_t src = static_cast<uint32_t>(qw[static_cast<size_t>(k / 8) * N + n]);
        word |= ((src >> (4 * (k % 8))) & 0xFu) << (4 * p);
    }
    packed[i] = static_cast<int32_t>(word);
}

}  // namespace

void w4_quantize_bf16(const __nv_bfloat16* W, int32_t* qweight, __nv_bfloat16* scales,
                      uint8_t* zeros, int K, int N, cudaStream_t stream) {
    SPARK_REQUIRE(K > 0 && N > 0 && K % kW4GroupSize == 0 && N % 16 == 0,
                  "w4_quantize: needs K % 128 == 0 and N % 16 == 0");
    SPARK_REQUIRE(W && qweight && scales, "w4_quantize: null pointer");
    const dim3 grid(cdiv(N, 128), K / kW4GroupSize);
    if (zeros)
        quantize_kernel<true><<<grid, 128, 0, stream>>>(W, qweight, scales, zeros, K, N);
    else
        quantize_kernel<false><<<grid, 128, 0, stream>>>(W, qweight, scales, zeros, K, N);
    SPARK_CHECK_LAUNCH();
}

void w4_repack(const int32_t* qweight, int32_t* packed, int K, int N, cudaStream_t stream) {
    SPARK_REQUIRE(K > 0 && N > 0 && K % 64 == 0 && N % 16 == 0,
                  "w4_repack: needs K % 64 == 0 and N % 16 == 0");
    SPARK_REQUIRE(qweight && packed && qweight != packed, "w4_repack: bad pointers");
    const int64_t total = static_cast<int64_t>(N / 16) * (K / 64) * 128;
    repack_kernel<<<static_cast<unsigned>(cdiv64(total, 256)), 256, 0, stream>>>(qweight, packed, K,
                                                                                 N);
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
