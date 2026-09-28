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
#include <cuda_fp8.h>
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
// Variant 1 over rows x cols with gate and up rows ld_gate / ld_up elements apart (the two
// halves of a fused [rows, 2 * cols] gate|up projection, read in place), out contiguous.
// Falls back to one element per thread when a row start or the row length is not a 16-byte
// multiple.
void swiglu_strided_f32(const float* gate, const float* up, float* out, int64_t rows, int64_t cols,
                        int64_t ld_gate, int64_t ld_up, cudaStream_t stream);
void swiglu_strided_bf16(const __nv_bfloat16* gate, const __nv_bfloat16* up, __nv_bfloat16* out,
                         int64_t rows, int64_t cols, int64_t ld_gate, int64_t ld_up,
                         cudaStream_t stream);

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
// Requires N % 16 == 0, K % 16 == 0, and M % 16 == 0 for variants 0 to 2 (variants 3, 4 and 6
// take any M >= 1).
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
// variant 6: variant 5's TMA mainloop on variant 4's Stream-K schedule
//            (src/kernels/hgemm_tma_sk.cu): a persistent grid, the grouped tile order, static
//            ranges or the tile queue with geometric K-passes, and the deterministic chain
//            fixup, with the producer lane owning the schedule and publishing each piece to
//            the consumer warps through a shared-memory ring so the pipeline stays full
//            across pieces. Same shape rules as variant 4 (N % 64 == 0, K % 64 == 0, any
//            M >= 1); shapes the 128x128 TMA tile cannot fill run variant 4's tiles, and
//            M <= 64 the decode kernel
void hgemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                int K, int variant, cudaStream_t stream);
int hgemm_num_variants();
// True if `variant` accepts this shape (the rules above); hgemm_bf16 throws when it is false.
// Lets a caller pick the fastest variant that fits instead of catching the exception.
bool hgemm_supports(int M, int N, int K, int variant);

// ---- FP8GEMM (e4m3 tensor cores) ----------------------------------------------------------
// C = scale_a * scale_b * A * Bt^T, with A [M,K] and Bt [N,K] both e4m3 and row-major (K
// contiguous: Bt is B transposed, the "TN" layout cuBLASLt requires for fp8, so a weight matrix
// is stored [out_features, in_features] as PyTorch's nn.Linear already does), scale_a and
// scale_b per-tensor fp32 scalars in device memory, C bf16, fp32 accumulation on
// mma.sync.m16n8k32. Requires N % 64 == 0, K % 64 == 0 and 16-byte aligned A and Bt.
// variant 0: one warp per 16x8 output tile, fragments loaded straight from global memory
//            (requires M % 16 == 0)
// variant 1: 128x128x64 block tile, mma.sync + ldmatrix out of XOR-swizzled shared memory,
//            3-stage cp.async pipeline, split-K over the last partial wave of tiles (fp32
//            atomics into a per-device workspace, so those tiles are not bitwise reproducible
//            run to run), 64x128 / 64x64 tiles for small grids, and one CTA per 16, 32 or
//            64-row strip of Bt for decode shapes (M <= 64). Rows past M are zero-filled, so
//            any M >= 1 works
// variant 2: variant 1's 128x128 tile fed by TMA (cp.async.bulk.tensor) through a
//            warp-specialized mbarrier pipeline: one producer warp, eight consumer warps.
//            Requires M % 128 == 0, N % 128 == 0, K % 128 == 0 and at least one tile per SM;
//            everything else steps down to variant 1
void fp8gemm(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* Bt, __nv_bfloat16* C, int M, int N, int K,
             const float* scale_a, const float* scale_b, int variant, cudaStream_t stream);
int fp8gemm_num_variants();
// True if `variant` accepts this shape (the rules above); fp8gemm throws when it is false.
bool fp8gemm_supports(int M, int N, int K, int variant);

// ---- RoPE + K/V cache append -------------------------------------------------------------
// From qkv = [B, S, (H_q + 2 H_kv) * D] bf16 (a fused q|k|v projection) and rotary tables
// cos, sin = [>= pos0 + S, D] fp32 in the rotate-half layout (column d pairs with d + D/2;
// only columns 0..D/2-1 are read): q rotated into [B, H_q, S, D], k rotated and v copied into
// the caches [B, H_kv, cap, D] at positions pos0..pos0+S-1. One launch, every byte read and
// written once. D a multiple of 16, 16-byte aligned pointers.
void rope_append_bf16(const __nv_bfloat16* qkv, const float* cos, const float* sin,
                      __nv_bfloat16* q, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, int B,
                      int S, int H_q, int H_kv, int D, int pos0, int cap, cudaStream_t stream);

// ---- Fused attention (scaled dot product, forward) ----------------------------------------
// O = softmax(Q K^T / sqrt(D)) V per (b, h), for Q, O = [B, H_q, S_q, D] and
// K, V = [B, H_kv, S_kv, D], row-major contiguous, bf16 in/out, fp32 scores / softmax /
// accumulation. Grouped-query attention: H_q % H_kv == 0 and query head h reads K/V head
// h / (H_q / H_kv) (H_kv == H_q is plain multi-head attention). D in {64, 128}, no dropout,
// no bias. `causal` masks key j from query i when j > i (top-left alignment, as torch's
// is_causal). Any S_q, S_kv >= 1; 16-byte aligned pointers.
// variant 0: one warp per query row, lanes own D/32 columns, online softmax key by key
// variant 1: 128-row Q tile, 64-row K/V tiles as fp32 in shared memory, CUDA-core FMAs,
//            online softmax per row with the O accumulator rescaled when the row max moves
// variant 2: mma.sync.m16n8k16 + ldmatrix flash attention: Q fragments in registers, S and
//            P V on the tensor cores, P repacked from the S accumulators without touching
//            shared memory, 3-stage cp.async pipeline on K and V
// variant 3: variant 2 scheduled: the tiles of the last partial wave are split along the keys
//            over the idle SMs and merged by a combine kernel (fp32 partials in a per-device
//            workspace shared by every stream); S_q <= 64 runs a 64-row tile; and when the
//            (H_q / H_kv) * S_q query rows that share a K/V head fit 16 rows (a decode step)
//            the flash-decoding kernel of src/kernels/attention_decode.cu runs instead: those
//            rows in one 16-row tile so K and V are read once per group, the keys of each
//            (b, kv head) split over blocks that fill the SMs, four warps per block each
//            streaming its own keys, partials merged in shared memory and by the last block
//            to arrive (one launch, no combine kernel)
// variant 4: variant 3's tile and schedule with K and V fed by TMA into a full / empty
//            mbarrier pipeline issued by one lane, so the KV loop has no block-wide barrier and
//            one warp's softmax runs under the other warp's mma on each scheduler. Same tail
//            split and decode paths as variant 3
// variant 5: variant 4's tile and pipeline on a persistent grid of one block per SM: the
//            blocks take (b, h, q-tile) items from a queue in the heaviest-first order, the
//            producer lane publishes each item to the warps through a shared-memory ring and
//            keeps the TMA pipeline running across items, so the barrier init and the wait
//            for a first box are paid once per block and the next item's Q tile is in flight
//            during the current one's last tiles. The tail split stays on for the non-causal
//            shapes, where the queue alone cannot fix 1,024 equal tiles on 170 SMs; same
//            decode paths as variant 3
void attention_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                    __nv_bfloat16* O, int B, int H_q, int H_kv, int S_q, int S_kv, int D,
                    bool causal, int variant, cudaStream_t stream);
int attention_num_variants();
bool attention_supports(int S_q, int S_kv, int D, int variant);

}  // namespace spark
