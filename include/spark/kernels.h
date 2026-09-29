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
// variant 6: tensor cores in TF32: mma.sync.m16n8k8 + ldmatrix on a 128x128x16 block tile
//            with a 3-stage cp.async ring, the operands rounded to tf32 (11 significant bits,
//            cvt.rna) as the fragments are loaded. The precision contract of cuBLAS under
//            CUBLAS_TF32_TENSOR_OP_MATH and of torch with allow_tf32: about 2^-11 relative
//            error per operand, so the result is not fp32-accurate
// variant 7: the same tile in 3xTF32: each operand split into big + small tf32 parts and
//            big*big + big*small + small*big accumulated (three mmas per product), which
//            brings the error back to the fp32 class at a third of variant 6's rate
// Variants 4 to 7 split the last partial wave of tiles along K and reduce with fp32 atomics
// into C, so those tiles are not bitwise reproducible run to run.
void sgemm(const float* A, const float* B, float* C, int M, int N, int K, int variant,
           cudaStream_t stream);
int sgemm_num_variants();
// The variant a caller that wants plain fp32 should run: 5, the top rung on the CUDA cores.
// Variants 6 and 7 compute on the tensor cores in TF32 and are opt-in, as torch's allow_tf32
// is, so the highest variant number is not the default here.
int sgemm_default_variant();

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
//
// Fused epilogue (variants 4 and 6 only, every route inside them: the TMA tile, the Stream-K
// finishing piece, variant 4's tiles and the decode kernel). Applied in fp32 before the one
// rounding to bf16, in this order:
//   C = act(A B + bias) + residual         bias [N] bf16 per column, residual [M][N] bf16
//   residual == C is the in-place accumulate C += act(A B + bias).
//   swiglu: B is [K][N] with column 2j = gate_j and 2j+1 = up_j (interleaved gate/up, see
//   docs/design/hgemm.md, "Fused epilogues"); C is [M][N/2] with
//   C[i][j] = silu(x[i][2j]) * x[i][2j+1] (+ residual[i][j]), x = A B + bias. `act` is
//   ignored. N is still the width of B and keeps the N % 64 == 0 rule.
// The default HgemmEpilogue is the plain store, and every existing call is unchanged. A
// non-default epilogue on variants 0 to 3 or 5 throws std::invalid_argument.
enum HgemmAct { HGEMM_ACT_NONE = 0, HGEMM_ACT_SILU = 1, HGEMM_ACT_GELU = 2, HGEMM_ACT_RELU = 3 };
struct HgemmEpilogue {
    const __nv_bfloat16* bias = nullptr;      // [N], or nullptr
    int act = HGEMM_ACT_NONE;                 // HgemmAct; gelu is the tanh form
    const __nv_bfloat16* residual = nullptr;  // [M][N] ([M][N/2] with swiglu), or nullptr; may be C
    bool swiglu = false;                      // interleaved gate/up columns, N/2 output columns
};
void hgemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                int K, int variant, cudaStream_t stream,
                const HgemmEpilogue& epilogue = HgemmEpilogue());
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
// MX mode (MXFP8): C = scale_a * scale_b * sum_k 2^(sfa[m][k/32] - 127) A[m][k]
// 2^(sfb[n][k/32] - 127) Bt[n][k], the same variants with a ue8m0 scale (the OCP MX E8M0
// byte: exponent + 127, 0xFF is NaN) per row per 32 k on both operands, sfa [M][K/32] and
// sfb [N][K/32] row-major uint8, applied inside the block-scaled mma. scale_a and scale_b
// may both be null (1.0). Requires K % 256 == 0 on top of the per-tensor rules (a row's
// scales for two 128-k stages are one aligned 8-byte chunk) and 16-byte aligned sfa, sfb;
// throws std::invalid_argument on a build without the instruction.
void fp8gemm_mx(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* Bt, __nv_bfloat16* C, int M, int N,
                int K, const float* scale_a, const float* scale_b, const unsigned char* sfa,
                const unsigned char* sfb, int variant, cudaStream_t stream);
bool fp8gemm_mx_supports(int M, int N, int K, int variant);
// True if the fp8 kernels were compiled for sm_120a / sm_121a, where the block-scaled mma
// exists. On a plain sm_120 build the per-tensor mode runs the plain instruction and MX mode
// is refused.
bool fp8gemm_mx_available();

// ---- RoPE + K/V cache append -------------------------------------------------------------
// From qkv = [B, S, (H_q + 2 H_kv) * D] bf16 (a fused q|k|v projection) and rotary tables
// cos, sin = [>= pos0 + S, D] fp32 in the rotate-half layout (column d pairs with d + D/2;
// only columns 0..D/2-1 are read): q rotated into [B, H_q, S, D], k rotated and v copied into
// the caches [B, H_kv, cap, D] at positions pos0..pos0+S-1. One launch, every byte read and
// written once. D a multiple of 16, 16-byte aligned pointers.
void rope_append_bf16(const __nv_bfloat16* qkv, const float* cos, const float* sin,
                      __nv_bfloat16* q, __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, int B,
                      int S, int H_q, int H_kv, int D, int pos0, int cap, cudaStream_t stream);

// Paged form for a batch of sequences in a paged K/V cache (docs/design/serving.md): qkv is
// [T, (H_q + 2 H_kv) * D], one row per token of any sequence (a packed prefill or one token
// per sequence at decode). Token t is rotated at position positions[t] and its k and v go to
// slot slots[t] of the caches [num_pages, H_kv, page, D] (slot = page_id * page + row; a
// negative slot skips the cache write, for padding tokens). q comes out as [T, H_q, D], the
// layout attention_varlen_bf16 and paged_decode_bf16 take. page a power of two.
void rope_append_paged_bf16(const __nv_bfloat16* qkv, const float* cos, const float* sin,
                            const int* positions, const int* slots, __nv_bfloat16* q,
                            __nv_bfloat16* k_cache, __nv_bfloat16* v_cache, int T, int H_q,
                            int H_kv, int D, int page, cudaStream_t stream);

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

// ---- Attention over a paged K/V cache (src/kernels/attention_paged.cu) ------------------
// The cache is k_cache, v_cache = [num_pages, H_kv, page, D] bf16, page a power of two >= 16.
// Sequence b owns pages block_table[b][0..], block_table [B, max_pages] int32 row-major, and
// has seq_lens[b] keys (0 <= seq_lens[b] <= max_pages * page; not checked, they live on the
// device): key j is row j % page of page block_table[b][j / page]. GQA as attention_bf16
// (H_q % H_kv == 0), D in {64, 128}, B <= 512, fp32 softmax and accumulation, 16-byte
// aligned pointers. The lengths are read on the device, so a launch does not depend on them
// and can be replayed from a CUDA graph as they change.
//
// Decode: Q, O = [B, H_q, D], one query token per sequence, no mask (the token's own key is
// already in the cache). A sequence with seq_lens[b] == 0 gets zeros.
// variant 0: one warp per (b, query head), keys one at a time
// variant 1: flash-decoding with a length-aware split: every (b, kv head) cut into 16-key
//            slabs, the slabs of the whole batch divided evenly over the warps of a grid of
//            one block per SM, each warp streaming its range through its own cp.async
//            pipeline across sequence boundaries; pieces of a (b, kv head) that span warps
//            are merged by a second kernel. Needs H_q / H_kv <= 16. Uses a per-process
//            workspace: two launches must not run concurrently on different streams
void paged_decode_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* k_cache,
                       const __nv_bfloat16* v_cache, const int* block_table, const int* seq_lens,
                       __nv_bfloat16* O, int B, int H_q, int H_kv, int D, int page, int max_pages,
                       int variant, cudaStream_t stream);
int paged_decode_num_variants();

// Varlen prefill: Q, O = [T, H_q, D], the new tokens of B sequences packed back to back,
// sequence b's at rows cu_seqlens_q[b] .. cu_seqlens_q[b+1]-1 (cu_seqlens_q [B + 1] int32,
// cu_seqlens_q[0] = 0, cu_seqlens_q[B] = T). Their keys are already in the paged cache, and
// seq_lens[b] >= q_len_b counts them together with the context before them. `causal` aligns
// the mask bottom-right: query i of sequence b sees keys j <= seq_lens[b] - q_len_b + i (the
// plain causal mask when the cache held nothing before; a prompt chunk otherwise).
// variant 0: one warp per (token, query head), keys one at a time
// variant 1: attention variant 2's 128 x 64 mma.sync tile with the K/V rows gathered through
//            the block table and each block's (sequence, q tile) found on the device
void attention_varlen_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* k_cache,
                           const __nv_bfloat16* v_cache, const int* cu_seqlens_q,
                           const int* seq_lens, const int* block_table, __nv_bfloat16* O, int B,
                           int T, int H_q, int H_kv, int D, int page, int max_pages, bool causal,
                           int variant, cudaStream_t stream);
int attention_varlen_num_variants();

}  // namespace spark
