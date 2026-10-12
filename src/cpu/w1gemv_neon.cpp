// 1-bit and ternary weight GEMV on the CPU, for the GB10's Arm cores.
//
// The question (docs/design/grace.md): a decode step at batch 1 is a weight stream, the
// GB10's 20 Arm cores sit on the same 273 GB/s as its 48 SMs, and DeepSeek V4.1 Flash has
// pieces that are small and serial enough to belong somewhere other than a grouped GPU
// launch: the shared expert (three 5120 x 2304 matrices, 1-bit, 4.4 MB with scales), a cold
// routed expert, the routing itself. This file is the GEMV those pieces would run through
// if they moved to the CPU. It reads the exact packed format of w1_quantize
// (include/spark/kernels.h, src/kernels/w1gemm_internal.cuh): uint32 [N, K bits / 32] with
// row n's k ascending from the lowest bit of its lowest word, and bf16 scales [K / 128, N],
// one per 128 consecutive k of a column. There is no repack: the same bytes the GPU kernel
// streams are what this one streams, so a weight can move between the two sides by changing
// which code reads it.
//
// The ladder:
//   0  scalar: one thread, one code at a time, the reference for the others
//   1  NEON fp32: the activation row widened to fp32 once (bf16 is the top half of fp32, a
//      shift), then per row and 128-k group the codes expanded to lane masks and used to
//      select between a and -a. 16 codes at 1 bit: the 16-bit chunk duplicated into eight
//      16-bit lanes, vtst against {1, 2, .., 128} and against {256, .., 32768} gives two
//      8-lane masks (0xFFFF where the code is 1), which sign-extend to 32-bit lanes with
//      vmovl. Ternary: 8 codes per chunk, one vtst on the even bits ("zero" when the code is
//      1) and one on the odd bits ("positive" when the code is 2), bsl on the odd mask and
//      bic on the even one. fp32 lanes accumulate per group, one fma by the group scale
//      into the row total, so the sum has the same shape as the GPU kernel's (fp32 per group
//      times a bf16 scale) and the products are the exact +-s weights of
//      reference.w1_dequantize, up to order.
//   2  NEON int8 (SDOT): the fp32 path spends about 19 vector instructions per 16 weights
//      because each sign has to become a 32-bit lane. With the activations quantized to int8
//      per group (an fp32 scale per 128), a sign becomes an int8 +1 / -1 in two instructions
//      (vtst, then orr 1 on the inverted mask) and vdot takes 16 products per instruction.
//      Ternary adds a bic for the zero codes. The group sum is an int32 lane sum times the
//      weight scale times the activation scale. Approximate by the activation rounding
//      (about 0.5 / 127 of the group's max |a| per product), not by the weights.
//
// What it is bound by (unmeasured): at K = 5120 a row is 640 bytes of signs. Variant 1
// does about 6,000 vector instructions for those 640 bytes, so one core at a few GB/s is
// nowhere near its memory stream; the GEMV is instruction bound at 1 bit, the opposite of
// the GPU kernel. Variant 2 cuts that by about 6x. The bandwidth probe in route_cpu.cpp
// measures what the cores could stream if they did nothing else; bench_cpu puts the two
// numbers side by side. Rows are split across std::threads in a fixed partition; there is
// no OpenMP and no pool, a thread per call, which is fine for a bench and a reference and
// is the first thing a serving loop would replace.
#include "spark/cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace spark::cpu {
namespace {

inline float bf16_to_f32(bf16 h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// Round to nearest even, NaN kept a NaN (torch's conversion).
inline bf16 f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    if ((u & 0x7F800000u) == 0x7F800000u) {
        return static_cast<bf16>((u >> 16) | ((u & 0xFFFFu) ? 0x40u : 0u));
    }
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<bf16>(u >> 16);
}

inline int words_per_row(int K, int bits) { return K * bits / 32; }

// The code of weight (n, k) from the packed array, the GPU helper's rule.
inline int code_at(const uint32_t* row, int k, int bits) {
    if (bits == 1) return (row[k / 32] >> (k % 32)) & 1;
    return (row[k / 16] >> (2 * (k % 16))) & 3;
}

// ---- variant 0: scalar -------------------------------------------------------------------

void rows_scalar(const float* af, const uint32_t* packed, const bf16* scales, bf16* y, int n0,
                 int n1, int N, int K, int bits) {
    const int wpr = words_per_row(K, bits);
    const int groups = K / kW1GroupSize;
    for (int n = n0; n < n1; ++n) {
        const uint32_t* row = packed + static_cast<size_t>(n) * wpr;
        float total = 0.f;
        for (int g = 0; g < groups; ++g) {
            float acc = 0.f;
            const int k0 = g * kW1GroupSize;
            if (bits == 1) {
                for (int k = k0; k < k0 + kW1GroupSize; ++k) {
                    acc += code_at(row, k, 1) ? af[k] : -af[k];
                }
            } else {
                for (int k = k0; k < k0 + kW1GroupSize; ++k) {
                    const int c = code_at(row, k, 2);
                    if (c == 2) {
                        acc += af[k];
                    } else if (c == 0) {
                        acc -= af[k];
                    }
                }
            }
            total += acc * bf16_to_f32(scales[static_cast<size_t>(g) * N + n]);
        }
        y[n] = f32_to_bf16(total);
    }
}

#if defined(__ARM_NEON)

// ---- variant 1: NEON fp32 ------------------------------------------------------------------

// Bit selectors for vtst on a duplicated 16-bit chunk.
alignas(16) const uint16_t kSignLo[8] = {1, 2, 4, 8, 16, 32, 64, 128};
alignas(16) const uint16_t kSignHi[8] = {256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
alignas(16) const uint16_t kTernEven[8] = {1, 4, 16, 64, 256, 1024, 4096, 16384};
alignas(16) const uint16_t kTernOdd[8] = {2, 8, 32, 128, 512, 2048, 8192, 32768};

// a[0..7] selected by an 8-lane 16-bit mask (0xFFFF = +a, 0 = -a), added to acc.
inline float32x4_t add_signed8(float32x4_t acc, uint16x8_t m16, const float* a) {
    const int16x8_t ms = vreinterpretq_s16_u16(m16);
    const uint32x4_t m0 = vreinterpretq_u32_s32(vmovl_s16(vget_low_s16(ms)));
    const uint32x4_t m1 = vreinterpretq_u32_s32(vmovl_s16(vget_high_s16(ms)));
    const float32x4_t a0 = vld1q_f32(a);
    const float32x4_t a1 = vld1q_f32(a + 4);
    acc = vaddq_f32(acc, vbslq_f32(m0, a0, vnegq_f32(a0)));
    return vaddq_f32(acc, vbslq_f32(m1, a1, vnegq_f32(a1)));
}

// Ternary: `pos` selects +a / -a, `zero` clears.
inline float32x4_t add_ternary8(float32x4_t acc, uint16x8_t pos16, uint16x8_t zero16,
                                const float* a) {
    const int16x8_t ps = vreinterpretq_s16_u16(pos16);
    const int16x8_t zs = vreinterpretq_s16_u16(zero16);
    const uint32x4_t p0 = vreinterpretq_u32_s32(vmovl_s16(vget_low_s16(ps)));
    const uint32x4_t p1 = vreinterpretq_u32_s32(vmovl_s16(vget_high_s16(ps)));
    const uint32x4_t z0 = vreinterpretq_u32_s32(vmovl_s16(vget_low_s16(zs)));
    const uint32x4_t z1 = vreinterpretq_u32_s32(vmovl_s16(vget_high_s16(zs)));
    const float32x4_t a0 = vld1q_f32(a);
    const float32x4_t a1 = vld1q_f32(a + 4);
    const uint32x4_t v0 = vbicq_u32(vreinterpretq_u32_f32(vbslq_f32(p0, a0, vnegq_f32(a0))), z0);
    const uint32x4_t v1 = vbicq_u32(vreinterpretq_u32_f32(vbslq_f32(p1, a1, vnegq_f32(a1))), z1);
    acc = vaddq_f32(acc, vreinterpretq_f32_u32(v0));
    return vaddq_f32(acc, vreinterpretq_f32_u32(v1));
}

void rows_neon_f32(const float* af, const uint32_t* packed, const bf16* scales, bf16* y, int n0,
                   int n1, int N, int K, int bits) {
    const int wpr = words_per_row(K, bits);
    const int groups = K / kW1GroupSize;
    const uint16x8_t sign_lo = vld1q_u16(kSignLo), sign_hi = vld1q_u16(kSignHi);
    const uint16x8_t tern_even = vld1q_u16(kTernEven), tern_odd = vld1q_u16(kTernOdd);
    for (int n = n0; n < n1; ++n) {
        const uint32_t* row = packed + static_cast<size_t>(n) * wpr;
        float32x4_t total = vdupq_n_f32(0.f);
        for (int g = 0; g < groups; ++g) {
            float32x4_t acc = vdupq_n_f32(0.f);
            const float* ag = af + g * kW1GroupSize;
            if (bits == 1) {
                const uint32_t* w = row + g * 4;  // 4 words = 128 signs
                for (int i = 0; i < 4; ++i) {
                    const uint32_t word = w[i];
                    const uint16x8_t lo = vdupq_n_u16(static_cast<uint16_t>(word & 0xFFFFu));
                    const uint16x8_t hi = vdupq_n_u16(static_cast<uint16_t>(word >> 16));
                    const float* a = ag + 32 * i;
                    acc = add_signed8(acc, vtstq_u16(lo, sign_lo), a);
                    acc = add_signed8(acc, vtstq_u16(lo, sign_hi), a + 8);
                    acc = add_signed8(acc, vtstq_u16(hi, sign_lo), a + 16);
                    acc = add_signed8(acc, vtstq_u16(hi, sign_hi), a + 24);
                }
            } else {
                const uint32_t* w = row + g * 8;  // 8 words = 128 codes
                for (int i = 0; i < 8; ++i) {
                    const uint32_t word = w[i];
                    const uint16x8_t lo = vdupq_n_u16(static_cast<uint16_t>(word & 0xFFFFu));
                    const uint16x8_t hi = vdupq_n_u16(static_cast<uint16_t>(word >> 16));
                    const float* a = ag + 16 * i;
                    acc = add_ternary8(acc, vtstq_u16(lo, tern_odd), vtstq_u16(lo, tern_even), a);
                    acc = add_ternary8(acc, vtstq_u16(hi, tern_odd), vtstq_u16(hi, tern_even),
                                       a + 8);
                }
            }
            total = vfmaq_n_f32(total, acc, bf16_to_f32(scales[static_cast<size_t>(g) * N + n]));
        }
        y[n] = f32_to_bf16(vaddvq_f32(total));
    }
}

#if defined(__ARM_FEATURE_DOTPROD)

// ---- variant 2: NEON int8 dot products ------------------------------------------------------

alignas(16) const uint8_t kSign8x2[16] = {1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128};
alignas(16) const uint8_t kTernEven8[16] = {1, 4, 16, 64, 1, 4, 16, 64, 1, 4, 16, 64, 1, 4, 16, 64};
alignas(16) const uint8_t kTernOdd8[16] = {2, 8, 32, 128, 2, 8, 32, 128, 2, 8, 32, 128,
                                           2, 8, 32, 128};
alignas(16) const uint8_t kByteIdx[16] = {0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3};

// The activation row as int8 per group of 128 with one fp32 scale each.
struct QuantA {
    std::vector<int8_t> q;
    std::vector<float> scale;
};

QuantA quantize_a(const float* af, int K) {
    const int groups = K / kW1GroupSize;
    QuantA qa;
    qa.q.resize(K);
    qa.scale.resize(groups);
    for (int g = 0; g < groups; ++g) {
        const float* a = af + g * kW1GroupSize;
        float32x4_t mx = vdupq_n_f32(0.f);
        for (int k = 0; k < kW1GroupSize; k += 4) mx = vmaxq_f32(mx, vabsq_f32(vld1q_f32(a + k)));
        const float amax = vmaxvq_f32(mx);
        const float s = amax > 0.f ? amax / 127.f : 1.f;
        qa.scale[g] = s;
        const float32x4_t inv = vdupq_n_f32(1.f / s);
        for (int k = 0; k < kW1GroupSize; k += 16) {
            const int32x4_t q0 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(a + k), inv));
            const int32x4_t q1 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(a + k + 4), inv));
            const int32x4_t q2 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(a + k + 8), inv));
            const int32x4_t q3 = vcvtnq_s32_f32(vmulq_f32(vld1q_f32(a + k + 12), inv));
            const int16x8_t h0 = vcombine_s16(vqmovn_s32(q0), vqmovn_s32(q1));
            const int16x8_t h1 = vcombine_s16(vqmovn_s32(q2), vqmovn_s32(q3));
            vst1q_s8(qa.q.data() + g * kW1GroupSize + k, vcombine_s8(vqmovn_s16(h0), vqmovn_s16(h1)));
        }
    }
    return qa;
}

// 16 signs (one 16-bit chunk) -> int8 lanes +1 / -1.
inline int8x16_t sign_lanes(uint16_t chunk, uint8x16_t sel) {
    const uint8x16_t bytes = vcombine_u8(vdup_n_u8(static_cast<uint8_t>(chunk & 0xFFu)),
                                         vdup_n_u8(static_cast<uint8_t>(chunk >> 8)));
    const int8x16_t set = vreinterpretq_s8_u8(vtstq_u8(bytes, sel));  // -1 where code 1
    return vorrq_s8(vmvnq_s8(set), vdupq_n_s8(1));                   // 1 where set, -1 where not
}

// 16 ternary codes (one word) -> int8 lanes +1 / 0 / -1.
inline int8x16_t ternary_lanes(uint32_t word, uint8x16_t idx, uint8x16_t even, uint8x16_t odd) {
    const uint8x16_t bytes = vqtbl1q_u8(vreinterpretq_u8_u32(vdupq_n_u32(word)), idx);
    const int8x16_t pos = vreinterpretq_s8_u8(vtstq_u8(bytes, odd));
    const int8x16_t zero = vreinterpretq_s8_u8(vtstq_u8(bytes, even));
    return vbicq_s8(vorrq_s8(vmvnq_s8(pos), vdupq_n_s8(1)), zero);
}

void rows_neon_i8(const QuantA& qa, const uint32_t* packed, const bf16* scales, bf16* y, int n0,
                  int n1, int N, int K, int bits) {
    const int wpr = words_per_row(K, bits);
    const int groups = K / kW1GroupSize;
    const uint8x16_t sel = vld1q_u8(kSign8x2), idx = vld1q_u8(kByteIdx);
    const uint8x16_t even = vld1q_u8(kTernEven8), odd = vld1q_u8(kTernOdd8);
    for (int n = n0; n < n1; ++n) {
        const uint32_t* row = packed + static_cast<size_t>(n) * wpr;
        float total = 0.f;
        for (int g = 0; g < groups; ++g) {
            int32x4_t acc = vdupq_n_s32(0);
            const int8_t* q = qa.q.data() + g * kW1GroupSize;
            if (bits == 1) {
                const uint32_t* w = row + g * 4;
                for (int i = 0; i < 4; ++i) {
                    const uint32_t word = w[i];
                    acc = vdotq_s32(acc, vld1q_s8(q + 32 * i),
                                    sign_lanes(static_cast<uint16_t>(word & 0xFFFFu), sel));
                    acc = vdotq_s32(acc, vld1q_s8(q + 32 * i + 16),
                                    sign_lanes(static_cast<uint16_t>(word >> 16), sel));
                }
            } else {
                const uint32_t* w = row + g * 8;
                for (int i = 0; i < 8; ++i) {
                    acc = vdotq_s32(acc, vld1q_s8(q + 16 * i), ternary_lanes(w[i], idx, even, odd));
                }
            }
            total += static_cast<float>(vaddvq_s32(acc)) * qa.scale[g] *
                     bf16_to_f32(scales[static_cast<size_t>(g) * N + n]);
        }
        y[n] = f32_to_bf16(total);
    }
}

#endif  // __ARM_FEATURE_DOTPROD
#endif  // __ARM_NEON

// Runs fn(n0, n1) over a fixed partition of [0, N) on `threads` threads (the caller is one
// of them when threads <= 1).
template <class F>
void over_rows(int N, int threads, const F& fn) {
    threads = std::max(1, std::min(threads, N));
    if (threads == 1) {
        fn(0, N);
        return;
    }
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (int t = 0; t < threads; ++t) {
        const int n0 = static_cast<int>(static_cast<int64_t>(N) * t / threads);
        const int n1 = static_cast<int>(static_cast<int64_t>(N) * (t + 1) / threads);
        pool.emplace_back([&fn, n0, n1] { fn(n0, n1); });
    }
    for (auto& th : pool) th.join();
}

}  // namespace

int w1gemv_num_variants() { return 3; }

bool w1gemv_supports(int variant) {
    if (variant == 0) return true;
#if defined(__ARM_NEON)
    if (variant == 1) return true;
#if defined(__ARM_FEATURE_DOTPROD)
    if (variant == 2) return true;
#endif
#endif
    return false;
}

size_t w1gemv_bytes(int N, int K, int bits) {
    const size_t n = static_cast<size_t>(N), k = static_cast<size_t>(K);
    return n * (k * bits / 32) * 4 + (k / kW1GroupSize) * n * 2 + k * 2 + n * 2;
}

void w1gemv_bf16(const bf16* a, const uint32_t* packed, const bf16* scales, bf16* y, int N, int K,
                 int bits, int threads, int variant) {
    if (bits != 1 && bits != 2) throw std::invalid_argument("w1gemv: bits must be 1 or 2");
    if (K <= 0 || K % kW1GroupSize != 0) {
        throw std::invalid_argument("w1gemv: K must be a positive multiple of 128");
    }
    if (N <= 0) throw std::invalid_argument("w1gemv: N must be >= 1");
    if (variant < 0) {
        variant = 0;
        for (int v = w1gemv_num_variants() - 1; v > 0; --v) {
            if (w1gemv_supports(v)) {
                variant = v;
                break;
            }
        }
    }
    if (variant >= w1gemv_num_variants() || !w1gemv_supports(variant)) {
        throw std::invalid_argument("w1gemv: variant " + std::to_string(variant) +
                                    " is not built on this host");
    }
    // The activation row in fp32 once, shared by every row of W.
    std::vector<float> af(static_cast<size_t>(K));
    for (int k = 0; k < K; ++k) af[k] = bf16_to_f32(a[k]);

    if (variant == 0) {
        over_rows(N, threads, [&](int n0, int n1) {
            rows_scalar(af.data(), packed, scales, y, n0, n1, N, K, bits);
        });
        return;
    }
#if defined(__ARM_NEON)
    if (variant == 1) {
        over_rows(N, threads, [&](int n0, int n1) {
            rows_neon_f32(af.data(), packed, scales, y, n0, n1, N, K, bits);
        });
        return;
    }
#if defined(__ARM_FEATURE_DOTPROD)
    if (variant == 2) {
        const QuantA qa = quantize_a(af.data(), K);
        over_rows(N, threads, [&](int n0, int n1) {
            rows_neon_i8(qa, packed, scales, y, n0, n1, N, K, bits);
        });
        return;
    }
#endif
#endif
    throw std::logic_error("w1gemv: unreachable variant");
}

}  // namespace spark::cpu
