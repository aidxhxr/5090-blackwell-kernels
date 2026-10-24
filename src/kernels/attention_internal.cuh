// Shared between attention.cu (the prefill ladder) and attention_decode.cu (the flash-decoding
// kernel): the few device helpers both use, the head-index arithmetic of grouped-query
// attention, and the decode launcher attention.cu dispatches to.
#pragma once

#include <cuda_bf16.h>

#include "spark/common.cuh"

namespace spark::attn {

using bf16 = __nv_bfloat16;

constexpr float kLog2e = 1.4426950408889634f;

// 2^x on the MUFU pipe. ex2.approx takes -inf to +0, which is what a masked score needs.
__device__ __forceinline__ float ex2(float x) {
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {
    const __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const unsigned*>(&h);
}

// Physical 16-byte chunk of logical (row, chunk) in a tile whose rows are 128 or 256 bytes:
// the eight rows an ldmatrix (or eight lanes reading eight rows) touch land in eight bank groups.
__device__ __forceinline__ int swz(int row, int chunk) {
    return chunk ^ (row & 7);
}

// GQA: query head h of batch b reads K/V head h / (H_q / H_kv). For a flattened (b, h) index
// bh = b * H_q + h this is the (b, kv) index into the [B, H_kv, S_kv, D] tensors. With
// H_kv == H_q it is the identity.
__host__ __device__ __forceinline__ int kv_index(int bh, int H_q, int H_kv) {
    const int b = bh / H_q, h = bh - b * H_q;
    return b * H_kv + h / (H_q / H_kv);
}

// Local attention (docs/design/attention.md, "Sliding window and sinks"). Query row i sits at
// key position i + off and sees keys j with j <= i + off under the causal mask and, when
// window > 0, j > i + off - window as well. off is 0 for torch's top-left causal mask and
// S_kv - S_q under a window (the window is aligned to the last key, as a cache is read).
// sinks, when not null, is one fp32 logit per query head added to the softmax denominator
// with no value. All zero / null is plain attention, and the launchers then run the kernels
// they ran before these options existed.
struct Local {
    int window = 0;
    int off = 0;
    const float* sinks = nullptr;
    bool active() const { return window > 0 || off != 0 || sinks != nullptr; }
};

// The first key row `pos` sees under a window of `window` keys (0: no left bound).
__host__ __device__ __forceinline__ int window_lo(int pos, int window) {
    const int lo = pos - window + 1;
    return window > 0 && lo > 0 ? lo : 0;
}

// The factor that normalizes a finished row with a sink in its denominator: O = o * factor
// for the row's unnormalized o, running max m and sum l (the log2 domain of the kernels:
// l = sum_j 2^(s_j - m)) and the sink logit s in the same domain (the logit times log2 e).
// The new max takes the sink in, so neither exponential can overflow. A row that saw no key
// (m = -inf, o = 0) gets 0.
__device__ __forceinline__ float sink_norm(float m, float l, float s) {
    const float mx = fmaxf(m, s);
    const float a = ex2(m - mx);
    return a / fmaf(l, a, ex2(s - mx));
}

// The flash-decoding kernel (attention_decode.cu). Takes every (b, kv head) whose query rows,
// (H_q / H_kv) * S_q of them, fit one 16-row tile; the launcher in attention.cu checks that
// with decode_fits() before calling. `split` > 0 forces the number of key slices per (b, kv
// head); 0 picks it from the SM count. D in {64, 128}. `kv_cap` is the rows each K/V head is
// allocated (attention_bf16's: S_kv when packed). `loc` adds a window and sinks: the kernel
// then streams only the keys some row's window holds.
bool decode_fits(int H_q, int H_kv, int S_q);
void decode_launch(const bf16* Q, const bf16* K, const bf16* V, bf16* O, int B, int H_q, int H_kv,
                   int S_q, int S_kv, int kv_cap, int D, float scale_log2, bool causal, int split,
                   cudaStream_t stream, const Local& loc = Local{});

}  // namespace spark::attn
