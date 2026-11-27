// Inkling 2 Mini's short causal convolutions and the per-head norms of q, k and v.
//
// Static conv (HF Inkling2LoopShortConvolution): depthwise, causal, K = 4 taps, fp32 weights in
// nn.Conv1d's order ([C, 4], column 3 weights the current token):
//   out[b, t, c] = (sum_{lag < 4} w[c, 3 - lag] * xin[b, t - lag, c] + x[b, t, c]) / sqrt(5)
// Dynamic conv (HF Inkling2LoopDynamicShortConvolution): the taps are a static per-channel part
// (static_weight [4, C], row = lag, row 0 the current token) plus a per-token part shared by
// the C / H channels of each of H heads (taps [B, T, H, 4], lag last: the weight_proj GEMM's
// output, viewed):
//   out[b, t, c] = (sum_lag (sw[lag, c] + taps[b, t, c / (C / H), lag]) * xin[b, t - lag, c]
//                   + x[b, t, c]) / sqrt(9)
// xin is x with the tokens of mask == 0 zeroed (HF's apply_mask_to_padding_states; the residual
// keeps them). Before t = 0 it is the conv state: [B, 3, C] in x's dtype, state[b, j, c] = xin
// at position j - 3 (j = 2 the newest). It is read at the start and rewritten with the last
// three xin of the call at the end, so a prefill in chunks, or decode one token at a time,
// continues it. With slots, row b of the call uses state row slots[b] of a [num_slots, 3, C]
// pool (a serving engine's slot per sequence); a negative or out-of-range slot runs that row
// from zeros and writes nothing (two rows on one slot would race). HF keeps [B, C, 4] fp32
// (column 3 the newest, column 0 never read); ours is hf[:, :, 1:].transpose(1, 2), and bf16 is
// exact for it since it only ever holds inputs.
//
// Epilogues, per head of head_dim channels, on the conv output rounded to x's dtype (HF rounds
// there too: the conv module returns x's dtype and the norm module reads that):
//   SHORT_CONV_EPI_L2CLIP (k): y * min(thr / (|y| + 1e-6), 1)      Inkling2LoopL2Clip
//   SHORT_CONV_EPI_RMS (v):    y * rsqrt(mean(y^2) + eps)           Inkling2LoopUnweightedRMSNorm
//
// head_norm: the same two norms as a standalone op over rows of D, with the q-side extras: a
// learned [H, D] scale before the clip, and the log scaling of the full layers,
// tau = 1 + alpha * log(max((pos + 1) / n_floor, 1)), applied after the clip's rounding
// (Inkling2LoopScaledL2Clip, Inkling2LoopLogScaledL2Clip; D = 128 for q, 16 for the relative
// states).
//
// Layout of the conv kernel: a thread owns 16 bytes of channels (8 bf16 or 4 fp32) and 4
// consecutive tokens, keeps the three previous inputs in registers and slides them. A warp is
// 32 consecutive channel groups of the same tokens, so every row access is 512 contiguous
// bytes and a head (128 channels = 16 or 32 lanes) is reduced with shuffles. The three inputs
// before a chunk are re-read (from L1/L2: the warp of the chunk before, in the same block
// but for the first, loads them too). 4 tokens a thread beat 8, 16 and 32 at an 8192-token
// prefill (docs/design/short_conv.md): longer chunks are fewer threads with their loads in a
// chain, and 1 or 2 re-read too much. Only the thread of chunk 0 touches the state, reading
// it before anything else and writing it last, so the in-place update has no race between
// blocks.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {
namespace {

constexpr int kTaps = 4;
constexpr int kHist = kTaps - 1;    // inputs a step needs from before it
constexpr int kStrip = 32;          // channel groups per block row (one warp)
constexpr int kChunk = 4;           // tokens per thread
constexpr int kChunksPerBlock = 4;  // block = 32 x 4 threads

template <typename T>
struct Vec;

template <>
struct Vec<float> {
    static constexpr int kN = 4;
    __device__ __forceinline__ static void load(const float* p, float (&o)[4]) {
        const f32x4 v = *reinterpret_cast<const f32x4*>(p);
#pragma unroll
        for (int i = 0; i < 4; ++i) o[i] = v.v[i];
    }
    __device__ __forceinline__ static void store(float* p, const float (&in)[4]) {
        f32x4 v;
#pragma unroll
        for (int i = 0; i < 4; ++i) v.v[i] = in[i];
        *reinterpret_cast<f32x4*>(p) = v;
    }
    __device__ __forceinline__ static float round(float v) {
        return v;
    }
};

template <>
struct Vec<__nv_bfloat16> {
    static constexpr int kN = 8;
    __device__ __forceinline__ static void load(const __nv_bfloat16* p, float (&o)[8]) {
        const bf16x8 v = *reinterpret_cast<const bf16x8*>(p);
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float2 f = __bfloat1622float2(v.h[i]);
            o[2 * i] = f.x;
            o[2 * i + 1] = f.y;
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
    __device__ __forceinline__ static float round(float v) {
        return __bfloat162float(__float2bfloat16_rn(v));
    }
};

__device__ __forceinline__ void load_taps(const float* p, float (&d)[kTaps]) {
    const f32x4 v = *reinterpret_cast<const f32x4*>(p);
#pragma unroll
    for (int i = 0; i < kTaps; ++i) d[i] = v.v[i];
}
__device__ __forceinline__ void load_taps(const __nv_bfloat16* p, float (&d)[kTaps]) {
    const uint2 raw = *reinterpret_cast<const uint2*>(p);
    const float2 lo = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.x));
    const float2 hi = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&raw.y));
    d[0] = lo.x;
    d[1] = lo.y;
    d[2] = hi.x;
    d[3] = hi.y;
}

// Sum of v over the `lanes` lanes (a power of two dividing 32) of an aligned lane group.
__device__ __forceinline__ float group_sum(float v, int lanes) {
    for (int o = lanes >> 1; o > 0; o >>= 1) v += __shfl_xor_sync(kFullMask, v, o);
    return v;
}

// ---------------------------------------------------------------------------
// The conv. grid = (strips of 32 channel groups, token chunks / 4, B), block = (32, <= 4).
// kDynamic: w is static_weight [4, C] and taps [B, T, heads, 4] is added per token and head;
// otherwise w is the conv1d weight [C, 4].
// ---------------------------------------------------------------------------
template <typename T, bool kDynamic, typename TapT>
__global__ void __launch_bounds__(kStrip* kChunksPerBlock)
    short_conv_kernel(const T* __restrict__ x, T* __restrict__ out, const float* __restrict__ w,
                      const TapT* __restrict__ taps, T* __restrict__ state,
                      const int32_t* __restrict__ slots, int num_slots,
                      const uint8_t* __restrict__ mask, int T_len, int C, int heads, int epilogue,
                      int head_dim, float param) {
    using V = Vec<T>;
    constexpr int kN = V::kN;
    const int chunk = blockIdx.y * blockDim.y + threadIdx.y;
    const int t0 = chunk * kChunk;
    if (t0 >= T_len) return;  // a warp shares threadIdx.y, so this is warp-uniform
    const int g = blockIdx.x * kStrip + threadIdx.x;
    const bool active = g < C / kN;  // inactive lanes still join the epilogue's shuffles
    const int c0 = active ? g * kN : 0;
    const int b = blockIdx.z;
    const int64_t tok0 = static_cast<int64_t>(b) * T_len;  // first token of this sequence
    const T* xb = x + tok0 * C + c0;
    T* ob = out + tok0 * C + c0;
    const uint8_t* mb = mask != nullptr ? mask + tok0 : nullptr;
    // Only chunk 0 reads or writes the state, so only it looks up the slot.
    T* sb = nullptr;
    if (chunk == 0 && state != nullptr) {
        const int slot = slots != nullptr ? slots[b] : b;
        if (slot >= 0 && slot < num_slots) sb = state + static_cast<int64_t>(slot) * kHist * C + c0;
    }

    // wl[lag][i]: the static tap of channel c0 + i at backward distance lag.
    float wl[kTaps][kN];
    if (active) {
        if constexpr (kDynamic) {
#pragma unroll
            for (int lag = 0; lag < kTaps; ++lag) {
#pragma unroll
                for (int i = 0; i < kN; i += 4) {
                    const f32x4 v = *reinterpret_cast<const f32x4*>(w + lag * C + c0 + i);
#pragma unroll
                    for (int j = 0; j < 4; ++j) wl[lag][i + j] = v.v[j];
                }
            }
        } else {
#pragma unroll
            for (int i = 0; i < kN; ++i) {
                const f32x4 v = *reinterpret_cast<const f32x4*>(w + (c0 + i) * kTaps);
#pragma unroll
                for (int lag = 0; lag < kTaps; ++lag) wl[lag][i] = v.v[kTaps - 1 - lag];
            }
        }
    } else {
#pragma unroll
        for (int lag = 0; lag < kTaps; ++lag) {
#pragma unroll
            for (int i = 0; i < kN; ++i) wl[lag][i] = 0.0f;
        }
    }
    const int dyn_head = kDynamic ? c0 / (C / heads) : 0;

    // h1, h2, h3: xin one, two and three tokens back.
    float h1[kN], h2[kN], h3[kN];
    auto load_in = [&](int t, float(&v)[kN]) {
        if (!active) {
#pragma unroll
            for (int i = 0; i < kN; ++i) v[i] = 0.0f;
        } else if (t >= 0) {
            V::load(xb + static_cast<int64_t>(t) * C, v);
            if (mb != nullptr && mb[t] == 0) {
#pragma unroll
                for (int i = 0; i < kN; ++i) v[i] = 0.0f;
            }
        } else if (sb != nullptr) {
            V::load(sb + (kHist + t) * C, v);
        } else {
#pragma unroll
            for (int i = 0; i < kN; ++i) v[i] = 0.0f;
        }
    };
    load_in(t0 - 1, h1);
    load_in(t0 - 2, h2);
    load_in(t0 - 3, h3);

    const float denom = kDynamic ? 3.0f : 2.2360679774997896f;  // sqrt(2K + 1), sqrt(K + 1)
    const int head_lanes = head_dim / kN;

    // A token's loads (x, its mask byte, its taps) are issued for the whole chunk before the
    // first step: the epilogue's shuffles would otherwise keep the compiler from hoisting the
    // next token's loads above them, and a chunk would be a chain of DRAM round trips.
    struct Tok {
        float x[kN];
        float keep;
        float d[kTaps];
    };
    auto load_tok = [&](int t, Tok& k) {
        if (active) {
            V::load(xb + static_cast<int64_t>(t) * C, k.x);
        } else {
#pragma unroll
            for (int i = 0; i < kN; ++i) k.x[i] = 0.0f;
        }
        k.keep = (mb != nullptr && mb[t] == 0) ? 0.0f : 1.0f;
#pragma unroll
        for (int j = 0; j < kTaps; ++j) k.d[j] = 0.0f;
        if constexpr (kDynamic) {
            if (active) load_taps(taps + ((tok0 + t) * heads + dyn_head) * kTaps, k.d);
        }
    };
    auto step = [&](int t, const Tok& k) {
        float xin[kN];
#pragma unroll
        for (int i = 0; i < kN; ++i) xin[i] = k.x[i] * k.keep;
        float y[kN];
#pragma unroll
        for (int i = 0; i < kN; ++i) {
            float acc = (wl[0][i] + k.d[0]) * xin[i];
            acc += (wl[1][i] + k.d[1]) * h1[i];
            acc += (wl[2][i] + k.d[2]) * h2[i];
            acc += (wl[3][i] + k.d[3]) * h3[i];
            y[i] = (acc + k.x[i]) / denom;
        }
        if (epilogue != SHORT_CONV_EPI_NONE) {
            float ss = 0.0f;
#pragma unroll
            for (int i = 0; i < kN; ++i) {
                y[i] = V::round(y[i]);
                ss += y[i] * y[i];
            }
            ss = group_sum(ss, head_lanes);
            const float f = epilogue == SHORT_CONV_EPI_L2CLIP
                                ? fminf(param / (sqrtf(ss) + 1e-6f), 1.0f)
                                : rsqrtf(ss / static_cast<float>(head_dim) + param);
#pragma unroll
            for (int i = 0; i < kN; ++i) y[i] *= f;
        }
        if (active) V::store(ob + static_cast<int64_t>(t) * C, y);
#pragma unroll
        for (int i = 0; i < kN; ++i) {
            h3[i] = h2[i];
            h2[i] = h1[i];
            h1[i] = xin[i];
        }
    };

    if (t0 + kChunk <= T_len) {
        Tok toks[kChunk];
#pragma unroll
        for (int i = 0; i < kChunk; ++i) load_tok(t0 + i, toks[i]);
#pragma unroll
        for (int i = 0; i < kChunk; ++i) step(t0 + i, toks[i]);
    } else {
        for (int t = t0; t < T_len; ++t) {
            Tok k;
            load_tok(t, k);
            step(t, k);
        }
    }

    if (chunk == 0 && sb != nullptr && active) {
        // The new state is xin at T - 3 .. T - 1. With T <= kChunk this thread walked all of
        // them (the old state included, for T < 3) and holds them; otherwise they are rows of
        // x, never the old state, and are read again.
        if (T_len > kChunk) {
            load_in(T_len - 1, h1);
            load_in(T_len - 2, h2);
            load_in(T_len - 3, h3);
        }
        V::store(sb, h3);
        V::store(sb + C, h2);
        V::store(sb + 2 * C, h1);
    }
}

template <typename T, bool kDynamic, typename TapT>
void launch_conv(const T* x, T* out, const float* w, const TapT* taps, int heads, T* state,
                 const int32_t* slots, int num_slots, const uint8_t* mask, int B, int T_len, int C,
                 int epilogue, int head_dim, float param, cudaStream_t stream) {
    constexpr int kN = Vec<T>::kN;
    const int strips = cdiv(C / kN, kStrip);
    const int chunks = cdiv(T_len, kChunk);
    const int cy = chunks < kChunksPerBlock ? chunks : kChunksPerBlock;
    const dim3 block(kStrip, cy);
    const dim3 grid(strips, cdiv(chunks, cy), B);
    short_conv_kernel<T, kDynamic, TapT><<<grid, block, 0, stream>>>(
        x, out, w, taps, state, slots, num_slots, mask, T_len, C, heads, epilogue, head_dim, param);
    SPARK_CHECK_LAUNCH();
}

template <typename T>
void short_conv_impl(const T* x, T* out, const float* w, const void* taps, bool taps_bf16,
                     int heads, T* state, const int32_t* slots, int num_slots, const uint8_t* mask,
                     int B, int T_len, int C, int epilogue, int head_dim, float param,
                     cudaStream_t stream) {
    constexpr int kN = Vec<T>::kN;
    SPARK_REQUIRE(B > 0 && T_len > 0 && C > 0, "short_conv: empty input");
    SPARK_REQUIRE(B <= 65535, "short_conv: batch above 65535");
    SPARK_REQUIRE(C % 8 == 0, "short_conv: C must be a multiple of 8");
    SPARK_REQUIRE(is_aligned16(x) && is_aligned16(out) && is_aligned16(w) &&
                      (state == nullptr || is_aligned16(state)),
                  "short_conv: x, out, w and state need 16-byte aligned storage");
    if (slots == nullptr) num_slots = B;  // row b is state row b
    SPARK_REQUIRE(state != nullptr || slots == nullptr, "short_conv: slots without a state");
    SPARK_REQUIRE(epilogue >= SHORT_CONV_EPI_NONE && epilogue <= SHORT_CONV_EPI_RMS,
                  "short_conv: unknown epilogue");
    if (epilogue != SHORT_CONV_EPI_NONE) {
        const int lanes = head_dim / kN;
        SPARK_REQUIRE(head_dim > 0 && head_dim % kN == 0 && C % head_dim == 0 && lanes <= 32 &&
                          (lanes & (lanes - 1)) == 0,
                      "short_conv: the epilogue needs head_dim dividing C, a multiple of 8 "
                      "(bf16) or 4 (fp32) with head_dim / that a power of two up to 32");
    }
    if (taps == nullptr) {
        launch_conv<T, false, float>(x, out, w, nullptr, 1, state, slots, num_slots, mask, B, T_len,
                                     C, epilogue, head_dim, param, stream);
        return;
    }
    SPARK_REQUIRE(heads > 0 && C % heads == 0 && (C / heads) % kN == 0,
                  "short_conv: heads must divide C into multiples of 8 channels");
    SPARK_REQUIRE(reinterpret_cast<uintptr_t>(taps) % (taps_bf16 ? 8 : 16) == 0,
                  "short_conv: taps need 8-byte (bf16) or 16-byte (fp32) aligned storage");
    if (taps_bf16) {
        launch_conv<T, true, __nv_bfloat16>(x, out, w, static_cast<const __nv_bfloat16*>(taps),
                                            heads, state, slots, num_slots, mask, B, T_len, C,
                                            epilogue, head_dim, param, stream);
    } else {
        launch_conv<T, true, float>(x, out, w, static_cast<const float*>(taps), heads, state, slots,
                                    num_slots, mask, B, T_len, C, epilogue, head_dim, param,
                                    stream);
    }
}

// ---------------------------------------------------------------------------
// head_norm: `lanes` = D / kN threads per row of D, 256 threads a block.
// ---------------------------------------------------------------------------
template <typename T>
__global__ void __launch_bounds__(256)
    head_norm_kernel(const T* x, T* out, int64_t rows, int H, int D, int lanes, int mode,
                     float param, const float* __restrict__ scale,
                     const void* __restrict__ positions, int position_bytes, float alpha,
                     float n_floor) {
    using V = Vec<T>;
    constexpr int kN = V::kN;
    const int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t row = tid / lanes;
    const int lane = static_cast<int>(tid % lanes);
    const bool active = row < rows;  // the tail's lanes still join the shuffles
    const int64_t off = row * D + lane * kN;
    float v[kN];
    if (active) {
        V::load(x + off, v);
    } else {
#pragma unroll
        for (int i = 0; i < kN; ++i) v[i] = 0.0f;
    }
    if (scale != nullptr && active) {
        const float* s = scale + static_cast<int64_t>(row % H) * D + lane * kN;
#pragma unroll
        for (int i = 0; i < kN; ++i) v[i] *= s[i];
    }
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < kN; ++i) ss += v[i] * v[i];
    ss = group_sum(ss, lanes);
    const float f = mode == SHORT_CONV_EPI_L2CLIP ? fminf(param / (sqrtf(ss) + 1e-6f), 1.0f)
                                                  : rsqrtf(ss / static_cast<float>(D) + param);
#pragma unroll
    for (int i = 0; i < kN; ++i) v[i] *= f;
    if (!active) return;
    if (position_bytes != 0) {
        const int64_t tok = row / H;
        const int64_t pos = position_bytes == 8 ? static_cast<const int64_t*>(positions)[tok]
                                                : static_cast<const int32_t*>(positions)[tok];
        const float tau = 1.0f + alpha * logf(fmaxf(static_cast<float>(pos + 1) / n_floor, 1.0f));
#pragma unroll
        for (int i = 0; i < kN; ++i) v[i] = V::round(v[i]) * tau;
    }
    V::store(out + off, v);
}

template <typename T>
void head_norm_impl(const T* x, T* out, int64_t rows, int H, int D, int mode, float param,
                    const float* scale, const void* positions, int position_bytes, float alpha,
                    float n_floor, cudaStream_t stream) {
    constexpr int kN = Vec<T>::kN;
    const int lanes = D / kN;
    SPARK_REQUIRE(rows > 0 && H > 0 && rows % H == 0, "head_norm: rows must be a multiple of H");
    SPARK_REQUIRE(D % kN == 0 && lanes <= 32 && (lanes & (lanes - 1)) == 0,
                  "head_norm: D must be a multiple of 8 (bf16) or 4 (fp32) with D / that a "
                  "power of two up to 32");
    SPARK_REQUIRE(mode == SHORT_CONV_EPI_L2CLIP || mode == SHORT_CONV_EPI_RMS,
                  "head_norm: mode must be SHORT_CONV_EPI_L2CLIP or SHORT_CONV_EPI_RMS");
    SPARK_REQUIRE(position_bytes == 0 || position_bytes == 4 || position_bytes == 8,
                  "head_norm: positions are int32 or int64");
    SPARK_REQUIRE(is_aligned16(x) && is_aligned16(out),
                  "head_norm: x and out need 16-byte aligned storage");
    constexpr int kThreads = 256;
    const int64_t blocks = cdiv64(rows * lanes, kThreads);
    SPARK_REQUIRE(blocks <= INT32_MAX, "head_norm: too many rows");
    head_norm_kernel<T><<<static_cast<unsigned>(blocks), kThreads, 0, stream>>>(
        x, out, rows, H, D, lanes, mode, param, scale, positions, position_bytes, alpha, n_floor);
    SPARK_CHECK_LAUNCH();
}

}  // namespace

void short_conv_f32(const float* x, float* out, const float* w, const void* taps, bool taps_bf16,
                    int heads, float* state, const int32_t* slots, int num_slots,
                    const uint8_t* mask, int B, int T, int C, int epilogue, int head_dim,
                    float param, cudaStream_t stream) {
    short_conv_impl(x, out, w, taps, taps_bf16, heads, state, slots, num_slots, mask, B, T, C,
                    epilogue, head_dim, param, stream);
}

void short_conv_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, const float* w, const void* taps,
                     bool taps_bf16, int heads, __nv_bfloat16* state, const int32_t* slots,
                     int num_slots, const uint8_t* mask, int B, int T, int C, int epilogue,
                     int head_dim, float param, cudaStream_t stream) {
    short_conv_impl(x, out, w, taps, taps_bf16, heads, state, slots, num_slots, mask, B, T, C,
                    epilogue, head_dim, param, stream);
}

void head_norm_f32(const float* x, float* out, int64_t rows, int H, int D, int mode, float param,
                   const float* scale, const void* positions, int position_bytes, float alpha,
                   float n_floor, cudaStream_t stream) {
    head_norm_impl(x, out, rows, H, D, mode, param, scale, positions, position_bytes, alpha,
                   n_floor, stream);
}

void head_norm_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t rows, int H, int D,
                    int mode, float param, const float* scale, const void* positions,
                    int position_bytes, float alpha, float n_floor, cudaStream_t stream) {
    head_norm_impl(x, out, rows, H, D, mode, param, scale, positions, position_bytes, alpha,
                   n_floor, stream);
}

}  // namespace spark
