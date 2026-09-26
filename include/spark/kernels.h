// Public host API for spark-kernels.
//
// Conventions (apply to every function):
//   * All matrices are row-major and contiguous. GEMMs compute C[M,N] = A[M,K] * B[K,N].
//   * Pointers are device pointers; the caller owns the memory.
//   * `variant` selects an implementation on the optimization ladder documented in
//     docs/DESIGN.md. Variant numbers are stable; the highest number is the fastest and
//     is what the Python bindings use by default.
//   * Launches are asynchronous on `stream`. Host-side validation failures throw
//     std::invalid_argument; CUDA failures throw std::runtime_error.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace spark {

// ---- Memory-bandwidth probe (establishes the roofline ceiling) ----------------------------
// y[i] = x[i] for n floats. variant 0: scalar, 1: float4 vectorized, 2: float4 + grid-stride.
void bandwidth_copy(const float* x, float* y, int64_t n, int variant, cudaStream_t stream);
int bandwidth_num_variants();

// ---- RMSNorm -----------------------------------------------------------------------------
// out[r, c] = x[r, c] * rsqrt(mean_c(x[r, :]^2) + eps) * w[c]
// variant 0: one thread per row (naive)
// variant 1: one warp per row, shuffle reduction
// variant 2: one warp per row, 128-bit vectorized loads (cols % 8 == 0 for bf16, % 4 for f32,
//            and 16-byte aligned pointers)
// variant 3: one block per row (for very wide rows, cols > 8192)
// variant 4: single pass: a 32..1024-thread group holds the whole row in registers, so x is
//            read once (rows up to 32768 elements, same alignment rules as variant 2; other
//            inputs run variant 3's kernel)
void rmsnorm_f32(const float* x, const float* w, float* out, int rows, int cols, float eps,
                 int variant, cudaStream_t stream);
void rmsnorm_bf16(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* out, int rows,
                  int cols, float eps, int variant, cudaStream_t stream);
int rmsnorm_num_variants();

// Fused residual-add + RMSNorm, the decoder-block pattern in Llama/Qwen:
//   resid[r, :] += x[r, :];   out[r, :] = rmsnorm(resid[r, :]) * w
// resid is updated in place. Single pass with the row in registers (variant 4 above) up to
// 32768 columns, the vectorized warp-per-row design (variant 2) beyond; requires cols % 8 == 0
// and 16-byte aligned pointers.
void add_rmsnorm_bf16(const __nv_bfloat16* x, __nv_bfloat16* resid, const __nv_bfloat16* w,
                      __nv_bfloat16* out, int rows, int cols, float eps, cudaStream_t stream);

// ---- SwiGLU (fused gated activation) ---------------------------------------------------
// out[i] = silu(gate[i]) * up[i], n elements. variant 0: scalar, 1: 128-bit vectorized
// (16-byte aligned pointers; any n, the remainder is handled in-kernel).
void swiglu_f32(const float* gate, const float* up, float* out, int64_t n, int variant,
                cudaStream_t stream);
void swiglu_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out, int64_t n,
                 int variant, cudaStream_t stream);
int swiglu_num_variants();

// ---- Row-wise softmax --------------------------------------------------------------------
// out[r, :] = softmax(x[r, :]) computed in fp32.
// variant 0: naive three-pass (max, sum, normalize), one thread per row
// variant 1: one warp per row, online softmax (single pass max/sum), vectorized loads
// variant 2: one block per row, online softmax (for long rows, cols > 4096)
// variant 3: single pass: a 32..1024-thread group holds the row in registers, one read, one
//            exp, one write (cols % 4 (f32) / 8 (bf16) == 0, 16-byte aligned pointers, up to
//            32768 columns; other inputs run variant 2's kernel)
void softmax_f32(const float* x, float* out, int rows, int cols, int variant, cudaStream_t stream);
void softmax_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int rows, int cols, int variant,
                  cudaStream_t stream);
int softmax_num_variants();

// ---- SGEMM (fp32) -------------------------------------------------------------------------
// C = A * B, fp32 in/out, fp32 accumulate. Any M, N, K >= 1 (bounds-checked).
// variant 0: naive, one thread per output element
// variant 1: shared-memory tiled (BM=BN=32, BK=32)
// variant 2: register-tiled, each thread owns an 8x8 micro-tile, float4 global loads
// variant 3: variant 2 + double-buffered shared memory (cp.async)
// variant 4: variant 2 + register-prefetch double buffering, 128x128x16 tile
// variant 5: variant 4 with a 256x128 tile and 16x8 micro-tiles (fewer smem bytes per FMA)
// Variants 4 and 5 split the last partial wave of tiles along K and reduce with fp32 atomics
// into C, so those tiles are not bitwise reproducible run to run.
void sgemm(const float* A, const float* B, float* C, int M, int N, int K, int variant,
           cudaStream_t stream);
int sgemm_num_variants();

// ---- HGEMM (bf16 tensor cores) ------------------------------------------------------------
// C = A * B, bf16 in/out, fp32 accumulate, via mma.sync tensor-core instructions (WMMA API).
// Requires N % 16 == 0, K % 16 == 0, and M % 16 == 0 for variants 0 to 2 (variant 3 takes
// any M >= 1).
// variant 0: one warp per 16x16 output tile straight from global memory (WMMA baseline)
// variant 1: block tile 128x128x32, 8 warps, shared-memory staged, padded to avoid bank conflicts
// variant 2: variant 1 + cp.async double-buffered pipeline (requires M,N % 128 == 0, K % 32 == 0)
// variant 3: raw mma.sync.m16n8k16 + ldmatrix, XOR-swizzled smem, 3-stage cp.async pipeline,
//            split-K over the last partial wave of tiles (fp32 atomics into a per-device
//            workspace, so those tiles are not bitwise reproducible run to run). The tile is
//            picked per call (128x128, 64x128 or 64x64) so small problems fill the card, and
//            decode-sized problems (M <= 64) run a dedicated weight-streaming kernel with a
//            16, 32 or 64-row tile (src/kernels/hgemm_decode.cu). Rows past M are zero-filled,
//            so any M >= 1 works; requires N % 64 == 0 and K % 64 == 0
// variant 4: variant 3's tile on a persistent Stream-K schedule: a grid of resident blocks,
//            L2-aware grouped tile order, whole tiles from a queue and the last wave split
//            along K with a memset-free, deterministic fixup (partials summed in K order
//            through a per-tile fp32 slot, so the output is the same bits every run). Same
//            shape rules as variant 3, and the same decode kernel for M <= 64.
// variant 5: variant 3's 128x128 tile and mma.sync k-loop fed by TMA (cp.async.bulk.tensor)
//            through a warp-specialized mbarrier pipeline: one producer warp issues the
//            loads, eight consumer warps run the tensor cores. Same split-K tail as variant 3.
//            Requires M % 128 == 0, N % 128 == 0, K % 64 == 0 and at least one 128x128 tile
//            per SM; smaller and decode shapes step down to variant 4. On the RTX 5090 it is
//            7 to 9% faster than variant 3 on every shape it takes (2048^3 and up), 102 to
//            109% of cuBLAS
void hgemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                int K, int variant, cudaStream_t stream);
int hgemm_num_variants();
// True if `variant` accepts this shape (the rules above); hgemm_bf16 throws when it is false.
// Lets a caller pick the fastest variant that fits instead of catching the exception.
bool hgemm_supports(int M, int N, int K, int variant);

// ---- Fused attention (scaled dot product, forward) ----------------------------------------
// O = softmax(Q K^T / sqrt(D)) V per (b, h), for Q, O = [B, H, S_q, D] and K, V = [B, H, S_kv, D],
// row-major contiguous, bf16 in/out, fp32 scores / softmax / accumulation. MHA only (the same
// H for Q and K/V), D in {64, 128}, no dropout, no bias. `causal` masks key j from query i
// when j > i (top-left alignment, as torch's is_causal). Any S_q, S_kv >= 1; 16-byte aligned
// pointers.
// variant 0: one warp per query row, lanes own D/32 columns, online softmax key by key
// variant 1: 128-row Q tile, 64-row K/V tiles as fp32 in shared memory, CUDA-core FMAs,
//            online softmax per row with the O accumulator rescaled when the row max moves
// variant 2: mma.sync.m16n8k16 + ldmatrix flash attention: Q fragments in registers, S and
//            P V on the tensor cores, P repacked from the S accumulators without touching
//            shared memory, 3-stage cp.async pipeline on K and V
// variant 3: variant 2 scheduled: the tiles of the last partial wave are split along the keys
//            over the idle SMs and merged by a combine kernel (fp32 partials in a per-device
//            workspace shared by every stream), and S_q <= 64 (decode) runs a 64-row tile
void attention_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                    __nv_bfloat16* O, int B, int H, int S_q, int S_kv, int D, bool causal,
                    int variant, cudaStream_t stream);
int attention_num_variants();
bool attention_supports(int S_q, int S_kv, int D, int variant);

}  // namespace spark
