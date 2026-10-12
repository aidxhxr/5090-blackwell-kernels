// CPU-side pieces of a decode step for the GB10's 20 Arm v9 cores (10 Cortex-X925 + 10
// Cortex-A725), which sit on the same 273 GB/s LPDDR5X as the GPU. Plain C++17, no CUDA
// include anywhere: this header and src/cpu/*.cpp build into the spark_cpu library on any
// host, with the NEON paths compiled in under __ARM_NEON and scalar fallbacks elsewhere.
// bf16 travels as uint16_t (the bit pattern of torch.bfloat16); the conversions live in the
// .cpp files. Nothing here has run on a GB10 (docs/design/grace.md).
#pragma once

#include <cstddef>
#include <cstdint>

namespace spark::cpu {

using bf16 = uint16_t;

// ---- 1-bit / ternary weight GEMV (src/cpu/w1gemv_neon.cpp) ---------------------------------
// y[N] = a[K] . dequant(packed, scales)[K, N] over the exact packed format of w1_quantize
// (include/spark/kernels.h): uint32 [N, K * bits / 32], row n's k ascending from the lowest
// bit of its lowest word (the sign at 1 bit, code 1 = +s; the ternary code 0 / 1 / 2 =
// -s / 0 / +s at 2 bits), bf16 scales [K / 128, N], one per group of 128 k of a column. The
// activations are bf16, the sums fp32 per group then times the group scale, the output bf16,
// so variants 0 and 1 compute the matrix reference.w1_dequantize returns, up to summation
// order. K % 128 == 0, N >= 1, bits in {1, 2}.
// variant 0: scalar reference, one thread
// variant 1: NEON: 16 (or 8 ternary) codes expanded to 32-bit lane masks by vtst on a
//            duplicated 16-bit chunk, sign-extended, selected between a and -a (and cleared
//            where the ternary code is zero), fp32 lanes accumulated per group
// variant 2: NEON int8 dot products (SDOT): the activations quantized to int8 per group with
//            an fp32 scale, the codes expanded to +1 / -1 (0 ternary) int8 lanes with one
//            vtst and one orr, four products per lane per instruction. Approximate: the int8
//            activation rounding adds about 0.5 / 127 of the group's max |a| per product
// Rows are split over `threads` std::threads in a fixed partition (threads <= 1 runs on the
// caller). variant -1 is the top rung the build has (1 on NEON, 2 with the dot product
// extension, 0 elsewhere).
constexpr int kW1GroupSize = 128;
int w1gemv_num_variants();
bool w1gemv_supports(int variant);
void w1gemv_bf16(const bf16* a, const uint32_t* packed, const bf16* scales, bf16* y, int N, int K,
                 int bits, int threads, int variant = -1);
// Bytes one GEMV streams: the packed words and the scales (the activation and the output are
// negligible against them at these shapes and counted anyway).
size_t w1gemv_bytes(int N, int K, int bits);

// ---- Routing and sampling (src/cpu/route_cpu.cpp) ------------------------------------------
// DeepSeek-style sigmoid gating for one token: scores = sigmoid(logits[E]) in fp32, the
// top_k experts of scores + bias (bias may be null; it picks, it does not weight), their
// unbiased scores renormalized to sum `scale`. ids come out in descending pick order, ties
// to the lower index (torch.topk's order on distinct values). A partial selection over E,
// no sort of the 384. 1 <= top_k <= E.
void route_sigmoid_topk(const float* logits, const float* bias, int E, int top_k, float scale,
                        int32_t* ids, float* weights);

// One token from a bf16 logit row of V entries. temperature <= 0: the first index of the
// max. Otherwise p_i = exp((x_i - m) / T) over the kept set: top_k > 0 keeps the top_k
// largest logits (a partial selection, ties at the k-th all kept), top_p < 1 then keeps the
// largest values whose mass before them is under p of the kept mass (tied values together).
// The draw is u from splitmix64(seed, offset), so a (seed, offset) pair is reproducible on
// the CPU; it is not the GPU sampler's Philox stream and does not match it token for token.
int64_t sample_bf16(const bf16* logits, int V, float temperature, int top_k, float top_p,
                    uint64_t seed, uint64_t offset);

// ---- Bandwidth probe (src/cpu/route_cpu.cpp) -----------------------------------------------
// Streams a `bytes`-long private buffer (allocated, first-touched by the threads that read
// it, freed) with 16-byte NEON loads on `threads` threads, `iters` sweeps after one warm
// sweep, and returns the median GB/s (bytes / wall time of the slowest thread). This is the
// number that says what share of the GB10's 273 GB/s the Arm side can take on its own.
double bandwidth_probe(size_t bytes, int threads, int iters = 5);

}  // namespace spark::cpu
