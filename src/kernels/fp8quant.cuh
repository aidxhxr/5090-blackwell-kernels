// The two pieces of the fp8 activation quantizer (fp8quant.cu) that decide its bits: the
// power-of-two scale of an amax and the e4m3 conversion. The norms that write their output
// in fp8 (rmsnorm.cu, rmsnorm_fp8 and add_rmsnorm_fp8) call the same two functions, so their
// bytes and scales are fp8_quantize's by construction, not by a second copy of the code.
#pragma once

#include <cuda_fp8.h>

#include "spark/common.cuh"

namespace spark {

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

}  // namespace spark
