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

#include <cstddef>
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

// ---- Token sampling (src/kernels/sample.cu) ----------------------------------------------
// out[b] (int64) = a token drawn from row b of logits [B, V] bf16, with per-row parameters in
// device memory so a captured CUDA graph does not depend on them: temperature [B] f32, top_k
// [B] int32, top_p [B] f32, seed [B] int64 and offset [B] int64 (the Philox block to draw
// from; every call adds 1 to each row's offset). Per row, with x the logits in fp32 and m
// their max:
//   * temperature <= 0: greedy, the first index of the max (torch.argmax).
//   * weights w_i = floor(exp((x_i - m) / T) * 2^40) as unsigned 64-bit integers (a token
//     under 2^-40 of the top one's probability is never drawn); every sum is an integer sum,
//     so the token depends on nothing but the row and its parameters.
//   * top-k (0 < k): the tokens at or above the k-th largest value among the nonzero weights,
//     ties at the threshold kept. 0 or k >= that count: no top-k.
//   * top-p (p < 1) on what top-k kept (total Z): the tokens at or above the largest value v
//     whose kept weight at or above v is >= ceil(p Z): sorted descending, a token is kept
//     while the mass before it is < p, tied values kept or dropped together. p <= 0 keeps
//     the top value.
//   * the token: the smallest index whose running sum of kept weights exceeds
//     floor(u * Z_f / 2^64), u the first 64 bits of Philox4_32_10(seed) at block offset.
// variant 0: one thread per row, a bisection over the 16-bit value key per threshold
// variant 1: one 1024-thread block per row: the max from one read of the row, each threshold
//            by a two-pass radix select over 256-bin count and weight histograms (the row
//            re-read from L2), the token by a scan of per-thread sums. No sort. Both variants
//            give the same token for the same inputs. V <= 2^23
void sample_bf16(const __nv_bfloat16* logits, int B, int V, const float* temperature,
                 const int* top_k, const float* top_p, const int64_t* seed, int64_t* offset,
                 int64_t* out, int variant, cudaStream_t stream);
int sample_num_variants();

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
// variant 7: batch invariant (docs/design/batch_invariant.md): row i of C has the same bits
//            whatever M is. Variant 6 with no K split anywhere: M <= 64 runs the decode
//            kernel with K unsplit, larger M one TMA tile (128x128, 128x64 or 64x128) with
//            every output tile computed whole by one block. Each output element is then the
//            same chain of k16 mma steps from k = 0 to K on every route, and the epilogue is
//            the same. Opt-in: variant -1 is still 6 (hgemm_default_variant). Same shape
//            rules and fused epilogue as variant 6
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
// The variant -1 runs: 6. Variant 7 trades the K split for batch invariance and is opt-in.
int hgemm_default_variant();
// What variant 7 runs for a shape: the decode kernel or the whole-tile TMA tile it picks.
const char* hgemm_v7_tile_name(int M, int N, int K);
// True if `variant` accepts this shape (the rules above); hgemm_bf16 throws when it is false.
// Lets a caller pick the fastest variant that fits instead of catching the exception.
bool hgemm_supports(int M, int N, int K, int variant);
// Variant 6's tile route for M > 64, for the sweeps (bench_hgemm --models). DEFAULT is the
// per-shape rule of docs/design/hgemm.md, "Model shapes": 128x128 TMA tiles when there are
// enough of them and every block gets 32 k-steps, else a 128x64 or 64x128 TMA tile, else
// variant 4's 64x64 tiles. LEGACY is the route before it: 128x128 when every block gets 32
// k-steps, variant 4 below. n in [0, hgemm_v6_num_tiles()) forces TMA tile n where N tiles
// (1 is 128x256, which no route picks), DEFAULT elsewhere. Process-wide, initial value from
// SPARK_HGEMM_V6_ROUTE; M <= 64 always runs the decode kernel. hgemm_v6_route_name says
// what variant 6 runs for a shape under the current route.
enum HgemmV6Route { HGEMM_V6_ROUTE_DEFAULT = -1, HGEMM_V6_ROUTE_LEGACY = -2 };
void hgemm_v6_set_route(int route);
int hgemm_v6_route();
int hgemm_v6_num_tiles();
const char* hgemm_v6_route_name(int M, int N, int K);
// Variant 6's split-K for M > 64 (docs/design/hgemm.md, "Model shapes"): K cut into slices
// whose fp32 partials the last warp to arrive sums in slice order, so the bits are the same
// every run (and differ from the unsplit kernel's). RULE splits the shapes the rule picks,
// OFF never splits (the route before the split), n >= 2 forces n K-slices on every shape
// with M > 64 (for the sweeps; on the forced tile when it has a split form). Process-wide,
// initial value from SPARK_HGEMM_V6_SPLIT.
enum HgemmV6Split { HGEMM_V6_SPLIT_RULE = -1, HGEMM_V6_SPLIT_OFF = 0 };
void hgemm_v6_set_split(int split);
int hgemm_v6_split();

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
// variant 3: variant 2's mainloop on hgemm's Stream-K schedule (persistent grid, grouped
//            raster, K-ordered fixup, the same bits every run), with a TMA block tile picked
//            per shape (128x256, 128x128, 128x64 or 64x128). Any shape variant 1 takes: decode
//            shapes, K % 128 != 0, MX mode and the shapes the route leaves alone run variant
//            2 where it takes them and variant 1 elsewhere, which is what the default ran
//            before variant 3 (fp8gemm_v3_set_route below)
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
// Variant 3's route (docs/design/fp8gemm.md, "Model shapes"): DEFAULT is the per-shape rule,
// LEGACY sends every shape where the default went before variant 3 (variant 2 if it takes the
// shape, else variant 1), n in [0, fp8gemm_v3_num_tiles()) forces TMA tile n where N and K
// tile (the old route elsewhere). Process-wide, initial value from SPARK_FP8GEMM_V3_ROUTE;
// M <= 64 always runs variant 1's decode kernels. fp8gemm_v3_route_name says what variant 3
// runs for a shape under the current route.
enum Fp8gemmV3Route { FP8GEMM_V3_ROUTE_DEFAULT = -1, FP8GEMM_V3_ROUTE_LEGACY = -2 };
void fp8gemm_v3_set_route(int route);
int fp8gemm_v3_route();
int fp8gemm_v3_num_tiles();
const char* fp8gemm_v3_route_name(int M, int N, int K);

// ---- FP4GEMM (e2m1 tensor cores, NVFP4 and MXFP4) ----------------------------------------
// C = scale_a * scale_b * sum_k (sfa[m][k/V] A[m][k]) (sfb[n][k/V] Bt[n][k]), with A [M,K] and
// Bt [N,K] e2m1 (fp4) packed two values per byte, K contiguous (element k in the low nibble of
// byte k/2 when k is even: the layout of torch.float4_e2m1fn_x2 and cuBLASLt's CUDA_R_4F_E2M1),
// so A is [M][K/2] bytes and Bt [N][K/2]. The block scales:
//   FP4_NVFP4: one e4m3 scale per 16 k (V = 16), as the NVFP4 format; scale_a / scale_b are
//              the per-tensor fp32 second-level scales
//   FP4_MXFP4: one ue8m0 scale (OCP E8M0, exponent + 127) per 32 k (V = 32), as MXFP4
// Both scale tensors are in the blocked layout cuBLASLt's VEC16_UE4M3 / VEC32_UE8M0 modes and
// torch's SWIZZLE_32_4_4 take: tiles of 128 rows by 4 scale bytes (512 bytes), the tiles of a
// 128-row block consecutive along K, and inside a tile the byte for row r and scale column j
// at (r % 32) * 16 + (r / 32) * 4 + j % 4. Rows are padded to a multiple of 128, so a tensor of
// `rows` rows takes fp4_scale_bytes(rows, K, format) bytes. scale_a / scale_b are device fp32
// scalars and may both be null (1.0). C is bf16, accumulation fp32 on
// mma.sync.m16n8k64.kind::mxf4nvf4, which applies the block scales. Requires N % 64 == 0,
// K % 256 == 0, 16-byte aligned A, Bt, sfa, sfb and an sm_120a / sm_121a build.
// variant 0: one warp per 16x8 output tile, fragments and scale words straight from global
//            memory (requires M % 16 == 0)
// variant 1: 128x128 block tile, mma.sync + ldmatrix out of XOR-swizzled shared memory with the
//            scale tiles staged beside the operand tiles, cp.async pipeline, split-K over the
//            last partial wave (fp32 atomics, not bitwise reproducible run to run), 64x128 /
//            64x64 tiles for small grids and one CTA per 32- to 64-row strip of Bt for decode
//            shapes (M <= 64). Any M >= 1
// variant 2: the 128x128 tile fed by TMA through a warp-specialized mbarrier pipeline, the
//            scale tiles by bulk copies on the same barriers. Requires M % 128 == 0,
//            N % 128 == 0 and at least one 128x128 tile per SM
enum Fp4Format { FP4_NVFP4 = 0, FP4_MXFP4 = 1 };
void fp4gemm(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* C, int M, int N, int K,
             const unsigned char* sfa, const unsigned char* sfb, const float* scale_a,
             const float* scale_b, int format, int variant, cudaStream_t stream);
int fp4gemm_num_variants();
bool fp4gemm_supports(int M, int N, int K, int variant);
// True if the fp4 kernels were compiled for sm_120a / sm_121a, where the fp4 mma exists.
bool fp4gemm_available();
// Bytes of a blocked scale tensor for a [rows][K] fp4 matrix (rows padded to 128).
size_t fp4_scale_bytes(int rows, int K, int format);
// Quantizes x [rows][K] bf16 into q [rows][K/2] (packed e2m1) and sf (blocked layout, all
// fp4_scale_bytes of it written, the padding rows as zero scales), K % 64 == 0 (NVFP4) or
// K % 128 == 0 (MXFP4), 16-byte aligned pointers.
//   NVFP4: `scale` is the per-tensor decode scale s (device fp32, x ~ s * sf * q; the NVFP4
//          recipe takes s = max|x| / (6 * 448)). A block's scale is e4m3(max|block| / 6 / s)
//          (round to nearest even, at most 448) and its elements e2m1(x / (s * sf)), round to
//          nearest even, saturating at 6. A null `scale` means s = 1.
//   MXFP4: the OCP recipe, `scale` ignored: a block of 32 gets 2^e with
//          e = floor(log2(max|block|)) - 2 (e2m1's largest exponent), stored as e + 127, and
//          its elements e2m1(x / 2^e), saturating at 6.
// With `work` (NVFP4 only; one device unsigned, zero on entry) s is computed on the device
// instead: max(max|x| / (6 * 448), FLT_MIN) from an amax pass (absmax_bits), written to
// `scale_out`, and `scale` is ignored. Two launches.
void fp4_quantize(const __nv_bfloat16* x, unsigned char* q, unsigned char* sf, int rows, int K,
                  const float* scale, int format, cudaStream_t stream, unsigned* work = nullptr,
                  float* scale_out = nullptr);

// ---- fp8 activation quantizer (src/kernels/fp8quant.cu) -----------------------------------
// x [rows][K] bf16 -> q [rows][K] e4m3 bytes plus scales, x ~ scale * q:
//   FP8Q_TENSOR: one scale, `scale` a device float; `work` one device unsigned that is zero on
//                entry (the amax cell; left holding max|x|'s bits). Two launches.
//   FP8Q_ROW:    one scale per row, `scale` rows device floats. One launch.
//   FP8Q_MX:     the OCP MX recipe, one e8m0 byte per 32 values of a row, `scale` rows * K/32
//                bytes (row-major, the layout fp8gemm's sfa takes). One launch.
// The tensor and row scales are the smallest power of two 2^e (e in [-126, 127]) with
// max|x| / 2^e <= 448; elements are e4m3(x / scale), round to nearest even, saturating.
// K % 8 == 0 (K % 32 == 0 for MX), x 16-byte and q 8-byte aligned.
enum Fp8QuantMode { FP8Q_TENSOR = 0, FP8Q_ROW = 1, FP8Q_MX = 2 };
// atomicMax of max|x| over x [n] bf16 (n % 8 == 0, 16-byte aligned) into *work as float bits
// (non-negative floats order like their bits); *work must hold zero or a smaller amax.
void absmax_bits(const __nv_bfloat16* x, long long n, unsigned* work, cudaStream_t stream);
void fp8_quantize(const __nv_bfloat16* x, unsigned char* q, void* scale, unsigned* work, int rows,
                  int K, int mode, cudaStream_t stream);

// ---- W4A16 GEMM (int4 weights, bf16 activations) ------------------------------------------
// C[M,N] = A[M,K] * dequant(W)[K,N] for bf16 A and C, int4 weights with one bf16 scale (and,
// optionally, one zero point) per group of 128 consecutive k of a column, fp32 accumulation.
// The dequantized weight is w[k][n] = bf16((q[k][n] - z) * s[k / 128][n]), rounded once, with
// z = 8 for symmetric weights and z = zeros[k / 128][n] for asymmetric ones: the kernel's
// weights are bit for bit the ones a dequantize-then-bf16-GEMM reference multiplies.
// Design and measurements: docs/design/w4gemm.md.
constexpr int kW4GroupSize = 128;

// Round-to-nearest quantization of W [K, N] bf16 (row-major, the [K, N] layout hgemm takes),
// per group of 128 k of one column, in fp32:
//   symmetric (zeros == nullptr): s = bf16(max|w| / 7), q = clamp(rint(w / s), -8, 7) + 8
//   asymmetric: lo = min(min w, 0), hi = max(max w, 0), s = bf16((hi - lo) / 15),
//               z = clamp(rint(-lo / s), 0, 15), q = clamp(rint(w / s) + z, 0, 15)
// (an all-zero group gets s = 0 and q = z). Outputs: qweight [K/8, N] int32, eight
// consecutive k per word with the lowest k in the lowest nibble (GPTQ's qweight layout),
// scales [K/128, N] bf16, zeros [K/128, N] uint8 (asymmetric only). K % 128 == 0,
// N % 16 == 0.
void w4_quantize_bf16(const __nv_bfloat16* W, int32_t* qweight, __nv_bfloat16* scales,
                      uint8_t* zeros, int K, int N, cudaStream_t stream);
// Offline repack of qweight into the layout w4gemm reads: int32 [N/16][K/64][32][4], one
// 512-byte block per 16 columns x 64 k in which lane l's 16 bytes are exactly the four
// mma.sync A fragments (16 output columns x 16 k each) it needs for those 64 k, nibbles
// ordered for the lop3 dequant (docs/design/w4gemm.md, "The repacked layout").
void w4_repack(const int32_t* qweight, int32_t* packed, int K, int N, cudaStream_t stream);

// C = A * dequant(packed, scales, zeros), zeros == nullptr for symmetric weights. Requires
// N % 16 == 0, K % 128 == 0, any M >= 1, 16-byte aligned A, packed, scales and zeros.
// variant 0: one thread per output element, scalar dequant from the packed layout
// variant 1: one warp per 16-column strip and 8 tokens over the whole K: each lane's 16 bytes
//            of weights and of activations straight into registers, lop3 dequant into the
//            mma.sync A fragment (weights as the 16-row operand, tokens as the n8 operand)
// variant 2: the block shape picked per call (docs/design/w4gemm.md). M = 1: four warps along
//            K on a multi-stage cp.async pipeline with one activation row staged; M <= 16:
//            independent warps with the weights of the next groups in flight in registers,
//            four or eight along K per 16 or 32 columns, or one per 16 columns when N is wide;
//            M > 16: 64 or 128-column tiles with warps along N and M sharing staged
//            activations, on a Stream-K grid. A K split across blocks goes through an fp32
//            workspace with atomics (those outputs are not bitwise reproducible run to run),
//            which the kernel leaves zeroed: one launch, no memset
void w4gemm_bf16(const __nv_bfloat16* A, const int32_t* packed, const __nv_bfloat16* scales,
                 const uint8_t* zeros, __nv_bfloat16* C, int M, int N, int K, int variant,
                 cudaStream_t stream);
int w4gemm_num_variants();
bool w4gemm_supports(int M, int N, int K, int variant);

// ---- 1-bit / ternary weight GEMM (src/kernels/w1gemm.cu) ----------------------------------
// C[M,N] = A[M,K] * dequant(W)[K,N] for bf16 A and C, 1-bit (sign) or 2-bit (ternary) weights
// with one bf16 scale per group of 128 consecutive k of a column, fp32 accumulation. A decode
// step is a weight stream: a sign bit is 16x fewer bytes than bf16 and 4x fewer than int4.
constexpr int kW1GroupSize = 128;
// Quantize W [K, N] bf16 (row-major, the [K, N] layout hgemm takes) per group of 128 k of
// one column, in fp32. bits == 1: s = bf16(mean|w| over the group), code = w >= 0.
// bits == 2 (ternary): s = bf16(mean|w|), code = 0 for w < -s/2, 1 for |w| <= s/2, 2 for w > s/2
// (an all-zero group gets s = 0). Output packed: uint32 [N, K * bits / 32] with k ascending
// from the lowest bits of the lowest word (bits==1: bit k%32 of word k/32; bits==2: bits
// 2*(k%16) of word k/16), scales [K/128, N] bf16. K % 128 == 0, N % 16 == 0.
void w1_quantize_bf16(const __nv_bfloat16* W, uint32_t* packed, __nv_bfloat16* scales, int K, int N,
                      int bits, cudaStream_t stream);
// C[M, N] = A[M, K] bf16 * dequant(packed, scales), fp32 accumulate, bf16 out. The
// dequantized weight is bf16(s * (2 code - 1)) for bits == 1 and bf16(s * (code - 1)) for
// bits == 2, so it is bit for bit what a dequantize-then-bf16-GEMM reference multiplies.
// Requires N % 16 == 0, K % 128 == 0, any M >= 1, 16-byte aligned pointers.
// variant 0: one thread per output element, scalar dequant
// variant 1: one warp per 16-column strip and 8 tokens over the whole K, the packed bits
//            straight into registers, selected into bf16x2 mma.sync A fragments
// variant 2: M <= 16: variant 1's step with the block's four warps splitting K and the next
//            groups' weights in flight in registers, the partial sums added in shared
//            memory; M > 16: variant 1 over 8-token chunks
// variant 3: M <= 16: variant 2 with one word of each row per lane (its own k order inside a
//            group), 8 groups of words and 3 of activations in flight per warp (4 and 1 at
//            16 tokens), 1 to 16 warps along K from the strip and SM counts (SPARK_W1_WARPS
//            forces it), 8-token tiles for M <= 8; M > 16: variant 2
void w1gemm_bf16(const __nv_bfloat16* A, const uint32_t* packed, const __nv_bfloat16* scales,
                 __nv_bfloat16* C, int M, int N, int K, int bits, int variant, cudaStream_t stream);
int w1gemm_num_variants();
bool w1gemm_supports(int M, int N, int K, int bits, int variant);
// Grouped form over E experts for a mixture-of-experts layer: A is [T, K], its rows sorted
// by expert so expert e owns rows offsets[e]..offsets[e+1]-1 (offsets is int32 [E + 1] on
// the device, offsets[E] == T, empty experts allowed); packed is E stacked [N, K*bits/32]
// weights and scales E stacked [K/128, N]; C is [T, N]. One launch for every expert.
// rows = T when the caller knows it: variant 3 launches min(E, T) block rows, each taking
// the next expert that has rows, instead of one per expert (-1: E rows; E <= 1024).
void w1gemm_moe_bf16(const __nv_bfloat16* A, const uint32_t* packed, const __nv_bfloat16* scales,
                     const int32_t* offsets, __nv_bfloat16* C, int E, int N, int K, int bits,
                     int variant, cudaStream_t stream, int rows = -1);

// ---- Grouped GEMM for MoE prefill (src/kernels/grouped_gemm.cu) --------------------------
// Every expert's GEMM in one persistent launch:
//   C[r, :] = scale_a * scale_b[e] * A[r, :] * W[e]^T   for r in offsets[e] .. offsets[e+1]-1
// A [T, K] has its rows sorted by expert (offsets int32 [E + 1] on the device, offsets[0] ==
// 0, offsets[E] == T, empty experts allowed, the w1gemm_moe convention), W is [E, N, K] (K
// contiguous, nn.Linear's layout per expert), C [T, N] bf16, fp32 accumulate. The tile list
// is built on the device from offsets (no host sync), the last wave is balanced Stream-K
// style, and every tile is fed by TMA. fp8: A and W e4m3, scale_a [1] and scale_b [E] fp32
// on the device (either may be null: 1), through the block-scaled mma with unit block
// scales; needs an sm_120a / sm_121a build. Requires 1 <= E <= 512, N % 128 == 0, K % 64 ==
// 0 (bf16) or K % 128 == 0 (fp8), 16-byte aligned A and W. Stream-K tiles are summed in a
// fixed order, so the output is the same bits every run. Not safe to run concurrently on
// two streams (one workspace per configuration).
void grouped_gemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* W, const int32_t* offsets,
                       __nv_bfloat16* C, int T, int E, int N, int K, cudaStream_t stream);
void grouped_gemm_fp8(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* W, const int32_t* offsets,
                      __nv_bfloat16* C, int T, int E, int N, int K, const float* scale_a,
                      const float* scale_b, cudaStream_t stream);
bool grouped_gemm_supports(int E, int N, int K, bool fp8);
// Both grouped GEMMs pick a tile of 16, 64 or 128 rows from the average rows per expert, T / E,
// in bf16; fp8 keeps 128 (SPARK_GROUPED_BM=16|64|128 forces one for either). grouped_gemm_swiglu is
// the same with the SwiGLU of an MoE expert in the epilogue: W [E, N, K] has its rows alternating
// gate and up (row 2j gate_j, row 2j + 1 up_j), C is [T, N / 2] with C[r, j] = silu(G) * U formed
// in fp32 from the two unrounded sums (scaled first, for fp8) and rounded once. Same rules; the
// default pipeline of each type (SPARK_GROUPED_CONFIG does not apply).
void grouped_gemm_swiglu_bf16(const __nv_bfloat16* A, const __nv_bfloat16* W,
                              const int32_t* offsets, __nv_bfloat16* C, int T, int E, int N, int K,
                              cudaStream_t stream);
void grouped_gemm_swiglu_fp8(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* W, const int32_t* offsets,
                             __nv_bfloat16* C, int T, int E, int N, int K, const float* scale_a,
                             const float* scale_b, cudaStream_t stream);

// ---- Linear cross-entropy (src/kernels/cross_entropy.cu) ---------------------------------
// The pieces of CE(h w^T, y) for a large vocabulary without torch's [T, V] logits; the
// ladder that uses them is linear_cross_entropy in python/spark_kernels/ops.py
// (docs/design/cross_entropy.md). h [T, D] and w [V, D] bf16, both D-contiguous (w is the
// lm_head as nn.Linear stores it), targets int64 [T]; a row "counts" when its target is in
// [0, V) and not ignore_index.
// The fused row kernel on materialized bf16 logits [rows][V]: loss[r] and lse[r] in fp32
// (0 for a row that does not count) and, when scale is not null, the gradient row
// (softmax - one-hot) * scale[0] written over the logits in bf16 (zeros for a row that does
// not count). One 1024-thread block per row on a grid of one block per SM; the row is read
// twice, the second time from L2. V % 8 == 0, 16-byte aligned logits.
void cross_entropy_rows_bf16(__nv_bfloat16* logits, const int64_t* targets, int rows, int V,
                             int ignore_index, const float* scale, float* loss, float* lse,
                             cudaStream_t stream);
// out[c][r] = x[r][c] for r < rows, 0 for rows <= r < ld; x has row stride ldx, out is
// [ncols][ld]. The K-contiguous A operand of dw += g^T h from a slab of the gradient.
void transpose_pad_bf16(const __nv_bfloat16* x, int rows, int ldx, __nv_bfloat16* out, int ncols,
                        int ld, cudaStream_t stream);
// ce_gemm_bf16: x = h w^T in fp32 on a persistent TMA kernel (grouped_gemm's bf16 mainloop
// and Stream-K schedule, one expert), then one of three epilogues on each finished 128x128
// tile:
//   CE_GEMM_LOGITS  out [T][V] = bf16(x)
//   CE_GEMM_STATS   partials [T][V/128][2] fp32 = (max, sum exp(x - max)) of the row over
//                   the tile's columns, target_logit [T] = x[row][y] (rows that count); no
//                   logits are written. ce_combine turns them into lse and loss.
//   CE_GEMM_GRAD    out [T][V] = bf16((exp(x - lse[row]) - [col == y]) * scale[0]), 0 on
//                   rows that do not count: d loss / d logits for a reduction scale.
// Requires V % 128 == 0, D % 64 == 0, 16-byte aligned h and w; any T >= 1. Deterministic.
// Not safe to run concurrently on two streams (one Stream-K workspace).
enum CeGemmMode { CE_GEMM_LOGITS = 0, CE_GEMM_STATS = 1, CE_GEMM_GRAD = 2 };
struct CeGemmEpilogue {
    __nv_bfloat16* out = nullptr;      // LOGITS, GRAD: [T][V]
    float* partials = nullptr;         // STATS: [T][V/128][2]
    float* target_logit = nullptr;     // STATS: [T]
    const int64_t* targets = nullptr;  // STATS, GRAD: [T]
    int ignore_index = -100;
    const float* lse = nullptr;    // GRAD: [T]
    const float* scale = nullptr;  // GRAD: [1] on the device
};
void ce_gemm_bf16(const __nv_bfloat16* h, const __nv_bfloat16* w, int T, int V, int D, int mode,
                  const CeGemmEpilogue& ep, cudaStream_t stream);
bool ce_gemm_supports(int V, int D);
// lse[t] = log sum_v exp(x[t][v]) from the STATS partials, loss[t] = lse[t] - x[t][y] for a
// row that counts, 0 otherwise. One warp per row.
void ce_combine(const float* partials, const float* target_logit, const int64_t* targets, int T,
                int V, int ignore_index, float* lse, float* loss, cudaStream_t stream);

// ---- MoE layer: routing, permute, combine (src/kernels/moe_route.cu, moe_permute.cu) ------
// moe_route: logits [T, E] (fp32 or bf16) -> ids [T, k] int32 (the top k experts of each
// token, largest first, ties to the lower id), weights [T, k] fp32 (the softmax over all E
// at those experts, divided by their sum when `renorm`), offsets [E + 1] int32 (rows per
// expert in expert order, offsets[E] == T * k) and rank [T, k] int32, the (token, slot)'s
// index among its route block's assignments to the same expert. One launch. A (token, slot)'s
// row in expert order is plan.base[(t / plan.tb) * E + e] + rank; rows are sorted by expert,
// then token, then slot (the stable sort of the flat ids). moe_positions adds that base in
// place (rank -> pos); moe_permute does it while moving the rows. The plan points into a
// per-process workspace that the next moe_route call overwrites, and like grouped_gemm the
// routing is not safe on two streams at once. Requires 1 <= k <= min(E, 16), E <= 512.
struct MoeRoutePlan {
    const int32_t* base;  // [nb, E]
    int tb;               // tokens per route block
    int nb;               // route blocks
};
bool moe_route_supports(int E, int k);
MoeRoutePlan moe_route(const void* logits, bool logits_fp32, int T, int E, int k, bool renorm,
                       int32_t* ids, float* weights, int32_t* rank, int32_t* offsets,
                       cudaStream_t stream);
void moe_positions(const int32_t* ids, int32_t* pos, const MoeRoutePlan& plan, int T, int E, int k,
                   cudaStream_t stream);
// moe_permute: pos (in: the rank from moe_route, out: the row in expert order) and xs [T * k,
// D] with xs[pos[t, s]] = x[t]. moe_combine: out [T, D] = sum over s of weights[t, s] *
// y[pos[t, s]], fp32 in slot order, one rounding, no atomics. bf16, D % 8 == 0, D <= 8192,
// k <= 16, 16-byte aligned rows.
bool moe_permute_supports(int D, int k);
void moe_permute_bf16(const __nv_bfloat16* x, const int32_t* ids, int32_t* pos,
                      const MoeRoutePlan& plan, __nv_bfloat16* xs, int T, int E, int k, int D,
                      cudaStream_t stream);
void moe_combine_bf16(const __nv_bfloat16* y, const int32_t* pos, const float* weights,
                      __nv_bfloat16* out, int T, int k, int D, cudaStream_t stream);

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
// The same into an e4m3 cache (docs/design/serving.md, "An fp8 K/V cache"): k_cache and
// v_cache are [num_pages, H_kv, page, D] e4m3, and the element of kv head h is stored as
// e4m3(x / scale[h]), round to nearest even, saturating at +-448, with k_scale and v_scale
// [H_kv] fp32 in device memory (one layer's). k is rotated in fp32 and rounded once, to e4m3;
// q comes out bf16 as above. The attention ops multiply the scales back in.
void rope_append_paged_fp8(const __nv_bfloat16* qkv, const float* cos, const float* sin,
                           const int* positions, const int* slots, __nv_bfloat16* q,
                           __nv_fp8_e4m3* k_cache, __nv_fp8_e4m3* v_cache, const float* k_scale,
                           const float* v_scale, int T, int H_q, int H_kv, int D, int page,
                           cudaStream_t stream);

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
//
// `lse`, when not null, receives the natural-log log-sum-exp of every row of scaled scores,
// fp32 [B, H_q, S_q]: lse_i = log(sum_j exp(q_i . k_j / sqrt(D))), the quantity the backward
// pass recomputes P from (torch calls it logsumexp). Every variant writes it; a decode shape
// that would run the flash-decoding kernel runs variant 3's 64-row tile instead, since that
// kernel does not produce it. With lse null nothing changes.
// `kv_cap`, when not 0, is the number of key rows each K/V head is allocated: K and V are the
// first S_kv rows of every head of a [B, H_kv, kv_cap, D] tensor (a K/V cache with spare
// capacity, read in place instead of copied out), kv_cap >= S_kv. 0 means S_kv, packed.
void attention_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                    __nv_bfloat16* O, int B, int H_q, int H_kv, int S_q, int S_kv, int D,
                    bool causal, int variant, cudaStream_t stream, float* lse = nullptr,
                    int kv_cap = 0);
int attention_num_variants();
bool attention_supports(int S_q, int S_kv, int D, int variant);

// ---- Fused attention backward -------------------------------------------------------------
// Gradients of O = softmax(Q K^T / sqrt(D)) V with respect to Q, K and V, from the forward's
// inputs, its output O, the upstream gradient dO (all bf16, the forward's shapes and GQA
// layout) and its log-sum-exp `lse` (fp32 [B, H_q, S_q], attention_bf16's optional output).
// dQ is [B, H_q, S_q, D]; dK and dV are [B, H_kv, S_kv, D], summed over the H_q / H_kv query
// heads that share each K/V head. fp32 accumulation, bf16 out. Same masks, head sizes and
// shape range as the forward. With P = exp(Q K^T / sqrt(D) - lse) and Dv_i = dO_i . O_i:
//   dV = P^T dO,  dS = P * (dO V^T - Dv),  dQ = dS K / sqrt(D),  dK = dS^T Q / sqrt(D).
// variant 0: one warp per query row for dQ and one per key row for dK, dV, keys (queries)
//            one at a time with shuffle reductions. Reference only
// variant 1: FlashAttention-2's structure on mma.sync: a block owns 64 keys of one K/V head
//            (4 warps x 16), walks the 64-row query tiles of every query head of its group
//            with dK and dV in registers, and adds its share of dQ to an fp32 buffer with
//            atomics; a preprocess kernel computes Dv and a last one converts dQ to bf16
// variant 2: variant 1 rebuilt for sm_120: 128 keys per block (8 warps), V's fragments held
//            in registers so a 32-row (D = 128) or 64-row (D = 64) Q/dO tile can be double
//            buffered by cp.async inside 99 KB, dS double-buffered so the dQ product of the
//            previous tile runs after the loop's one barrier, warps skip query tiles their keys
//            cannot see under the causal mask, and the grid is split (every key tile into
//            ranges of its query tiles, then the last partial wave again) by a simulation of
//            the block scheduler, split tiles adding fp32 partials of dK and dV
// variant 3: variant 2 without the block barrier: Q, dO and the (L, Dv) rows by TMA into
//            three full / empty mbarrier stages issued by one lane, the dS buffers handed
//            between warps by mbarriers, so the warps drift apart instead of starting every
//            tile in phase. The default
// `deterministic` (variants 1 to 3) replaces the dQ atomics with a separate dQ pass, a
// query-tile-outer kernel that recomputes P and dP, and turns the split off: 7 matrix products
// instead of 5, bitwise reproducible results. Uses a per-device workspace (L and Dv, the fp32
// dQ buffer, the split partials) shared by every stream, like the forward's split workspace:
// two backward calls must not run concurrently on different streams.
void attention_bwd_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                        const __nv_bfloat16* O, const __nv_bfloat16* dO, const float* lse,
                        __nv_bfloat16* dQ, __nv_bfloat16* dK, __nv_bfloat16* dV, int B, int H_q,
                        int H_kv, int S_q, int S_kv, int D, bool causal, bool deterministic,
                        int variant, cudaStream_t stream);
int attention_bwd_num_variants();
bool attention_bwd_supports(int S_q, int S_kv, int D, int variant);

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
// variant 2: batch invariant (docs/design/batch_invariant.md): each (b, kv head) cut into
//            chunks of 256 keys at fixed offsets, the chunks of the batch spread over the
//            warps (never one chunk over two), and a sequence's chunks folded in order, so a
//            row's bits depend on its own sequence only, and equal varlen variant 2's for the
//            same row. bf16 caches only; opt-in (paged_decode_default_variant is 1)
void paged_decode_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* k_cache,
                       const __nv_bfloat16* v_cache, const int* block_table, const int* seq_lens,
                       __nv_bfloat16* O, int B, int H_q, int H_kv, int D, int page, int max_pages,
                       int variant, cudaStream_t stream);
int paged_decode_num_variants();
int paged_decode_default_variant();

// Varlen prefill: Q, O = [T, H_q, D], the new tokens of B sequences packed back to back,
// sequence b's at rows cu_seqlens_q[b] .. cu_seqlens_q[b+1]-1 (cu_seqlens_q [B + 1] int32,
// cu_seqlens_q[0] = 0, cu_seqlens_q[B] = T). Their keys are already in the paged cache, and
// seq_lens[b] >= q_len_b counts them together with the context before them. `causal` aligns
// the mask bottom-right: query i of sequence b sees keys j <= seq_lens[b] - q_len_b + i (the
// plain causal mask when the cache held nothing before; a prompt chunk otherwise).
// variant 0: one warp per (token, query head), keys one at a time
// variant 1: attention variant 2's 128 x 64 mma.sync tile with the K/V rows gathered through
//            the block table and each block's (sequence, q tile) found on the device
// variant 2: batch invariant: variant 1's tile with paged decode variant 2's arithmetic (16-key
//            steps, chunks of 256 keys folded in order), so a query row has the same bits in
//            any batch, any chunk of its prompt, and as a decode row. bf16 only; opt-in
void attention_varlen_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* k_cache,
                           const __nv_bfloat16* v_cache, const int* cu_seqlens_q,
                           const int* seq_lens, const int* block_table, __nv_bfloat16* O, int B,
                           int T, int H_q, int H_kv, int D, int page, int max_pages, bool causal,
                           int variant, cudaStream_t stream);
int attention_varlen_num_variants();
int attention_varlen_default_variant();

// Both ops on an e4m3 cache written by rope_append_paged_fp8: key j of kv head h is
// k_scale[h] * k_cache[...], value v_scale[h] * v_cache[...] (k_scale, v_scale [H_kv] fp32 in
// device memory). Q and O stay bf16, the same variants and rules as the bf16 forms. Every
// variant converts e4m3 to bf16 exactly (e4m3 is a subset of bf16) and runs the bf16
// arithmetic; k_scale goes into the softmax scale and v_scale into the output normalization.
// Decode variant 1 reads the slabs into registers in the mma fragment layouts, with the
// head dimension permuted the same way for Q and K and undone when O is written, and keeps
// 6 slabs per warp in its pipeline instead of 3 (the same 96 KB). Varlen variant 1 converts
// each 64-key tile to bf16 in shared memory once for its 8 warps.
void paged_decode_fp8(const __nv_bfloat16* Q, const __nv_fp8_e4m3* k_cache,
                      const __nv_fp8_e4m3* v_cache, const float* k_scale, const float* v_scale,
                      const int* block_table, const int* seq_lens, __nv_bfloat16* O, int B, int H_q,
                      int H_kv, int D, int page, int max_pages, int variant, cudaStream_t stream);
void attention_varlen_fp8(const __nv_bfloat16* Q, const __nv_fp8_e4m3* k_cache,
                          const __nv_fp8_e4m3* v_cache, const float* k_scale, const float* v_scale,
                          const int* cu_seqlens_q, const int* seq_lens, const int* block_table,
                          __nv_bfloat16* O, int B, int T, int H_q, int H_kv, int D, int page,
                          int max_pages, bool causal, int variant, cudaStream_t stream);

// ---- Sliding-window attention and attention sinks ----------------------------------------
// The forward above with the two options local attention needs (docs/design/attention.md,
// "Sliding window and sinks"): the same layouts, variants and rules as attention_bf16, which
// this is when window = 0 and sinks is null. Mistral 7B and Gemma 2 use 4096-key windows,
// Gemma 3 1024, gpt-oss 128 on every other layer and a sink logit per head on every layer.
//
// `window` > 0: each query sees at most the last `window` keys up to its own position, its
// own included. The window is causal and aligned to the last key: query i sits at key
// position p_i = i + S_kv - S_q (S_q <= S_kv) and sees keys p_i - window < j <= p_i,
// whatever `causal` says (with S_q == S_kv that is torch's top-left causal mask with a left
// edge; a decode step sees the last `window` keys of the cache). The kernels skip the K/V
// tiles that lie wholly outside every row's window: a decode step at 128K cached keys with a
// 4096-key window streams 4096 keys.
// `sinks`, when not null, is fp32 [H_q] in device memory: one learned logit per query head
// added to every softmax denominator with no value vector (gpt-oss):
//   O_i = sum_j exp(s_ij - m) v_j / (exp(sink_h - m) + sum_j exp(s_ij - m)),
// with s_ij = q_i . k_j / sqrt(D) and m the max over the row's scores and the sink. A
// forward `lse` then includes the sink: log(exp(sink_h) + sum_j exp(s_ij)).
void attention_window_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                           __nv_bfloat16* O, int B, int H_q, int H_kv, int S_q, int S_kv, int D,
                           bool causal, int window, const float* sinks, int variant,
                           cudaStream_t stream, float* lse = nullptr, int kv_cap = 0);
// The paged decode and the varlen prefill with the same two options (window = 0 and no sinks
// is the plain op), for both cache formats. The decode token of sequence b sees keys
// [seq_lens[b] - window, seq_lens[b]) and the varlen query i keys
// p - window < j <= p for p = seq_lens[b] - q_len_b + i (a window implies `causal`). Both read
// only the slabs and tiles a window holds.
void paged_decode_window_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* k_cache,
                              const __nv_bfloat16* v_cache, const int* block_table,
                              const int* seq_lens, __nv_bfloat16* O, int B, int H_q, int H_kv,
                              int D, int page, int max_pages, int window, const float* sinks,
                              int variant, cudaStream_t stream);
void paged_decode_window_fp8(const __nv_bfloat16* Q, const __nv_fp8_e4m3* k_cache,
                             const __nv_fp8_e4m3* v_cache, const float* k_scale,
                             const float* v_scale, const int* block_table, const int* seq_lens,
                             __nv_bfloat16* O, int B, int H_q, int H_kv, int D, int page,
                             int max_pages, int window, const float* sinks, int variant,
                             cudaStream_t stream);
void attention_varlen_window_bf16(const __nv_bfloat16* Q, const __nv_bfloat16* k_cache,
                                  const __nv_bfloat16* v_cache, const int* cu_seqlens_q,
                                  const int* seq_lens, const int* block_table, __nv_bfloat16* O,
                                  int B, int T, int H_q, int H_kv, int D, int page, int max_pages,
                                  bool causal, int window, const float* sinks, int variant,
                                  cudaStream_t stream);
void attention_varlen_window_fp8(const __nv_bfloat16* Q, const __nv_fp8_e4m3* k_cache,
                                 const __nv_fp8_e4m3* v_cache, const float* k_scale,
                                 const float* v_scale, const int* cu_seqlens_q, const int* seq_lens,
                                 const int* block_table, __nv_bfloat16* O, int B, int T, int H_q,
                                 int H_kv, int D, int page, int max_pages, bool causal, int window,
                                 const float* sinks, int variant, cudaStream_t stream);

// ---- FP8 attention (e4m3 in, bf16 out, forward) ------------------------------------------
// O = softmax(sq sk Q K^T / sqrt(D)) sv V per (b, h), for Q = [B, H_q, S_q, D] and
// K, V = [B, H_kv, S_kv, D] e4m3 (the same layout, heads and mask as attention_bf16), O bf16.
// q_scale, k_scale, v_scale are fp32 descale factors in device memory: one per tensor, or with
// per_head one per (b, head) of that tensor (B * H_q for q, B * H_kv for k and v, indexed by
// b * H + h). Scores, softmax and accumulators are fp32; variant 1 rounds the probabilities
// (times 2^8) to e4m3 for the P V product. D in {64, 128}, any S_q, S_kv >= 1, 16-byte aligned
// pointers.
// variant 0: one warp per query row, every element dequantized to fp32, P in fp32 (baseline)
// variant 1: attention variant 5's persistent TMA + mbarrier kernel on e4m3: both products on
//            mma.sync.m16n8k32 (block-scaled, unit scales), P quantized in registers, V read
//            as it lies ([key][d]) through ldmatrix.trans and byte permutes, the O rescale
//            skipped while no row max moves by more than 0.8 (log2). Same queue and tail split
//            as attention variant 5 (its own workspace, shared by every stream, and its own
//            per-stream counters). Built for sm_120a / sm_121a with the fp8 GEMM; a plain
//            sm_120 build runs the half-rate fp8 instruction
void attention_fp8(const __nv_fp8_e4m3* Q, const __nv_fp8_e4m3* K, const __nv_fp8_e4m3* V,
                   __nv_bfloat16* O, int B, int H_q, int H_kv, int S_q, int S_kv, int D,
                   const float* q_scale, const float* k_scale, const float* v_scale, bool per_head,
                   bool causal, int variant, cudaStream_t stream);
int attention_fp8_num_variants();
bool attention_fp8_supports(int S_q, int S_kv, int D, int variant);

// ---- Unified memory experiments (src/kernels/unified.cu, unified_gemv.cu) ----------------
// On the GB10 the CPU and the GPU share one 128 GB pool and there is no PCIe copy; on the RTX
// 5090 the same calls go over PCIe. These helpers and kernels ask what a kernel reading each
// kind of memory gets, and what the CPU-to-GPU handoff of a few bytes costs. Unmeasured on
// both machines (docs/design/unified.md).
//
// Where a buffer lives. `ptr` is what the GPU dereferences; `host_ptr` what the CPU writes
// (nullptr for device memory, the same address otherwise under unified addressing).
enum class UnifiedKind : int {
    device = 0,       // cudaMalloc
    managed = 1,      // cudaMallocManaged: migrates on demand, takes advice and prefetch
    host_mapped = 2,  // cudaHostAlloc(cudaHostAllocMapped): pinned pages read in place
    system = 3        // posix_memalign: pageable memory. Read as is when the device reports
                      // pageableMemoryAccess (HMM / ATS), else pinned with cudaHostRegister
                      // and `registered` set
};
constexpr int kUnifiedNumKinds = 4;
const char* unified_kind_name(UnifiedKind kind);

struct UnifiedAlloc {
    void* ptr = nullptr;
    void* host_ptr = nullptr;
    size_t bytes = 0;
    UnifiedKind kind = UnifiedKind::device;
    int device = 0;
    bool registered = false;
};
UnifiedAlloc unified_alloc(UnifiedKind kind, size_t bytes);
void unified_free(UnifiedAlloc& a);

// cudaMemAdvise / cudaMemPrefetchAsync on a managed (or, with HMM, system) range. Returns
// false when the runtime refuses the advice for that memory (device, host_mapped, plain pages
// without HMM), which the bench records instead of failing. `device` < 0 means the current one.
enum class UnifiedAdvice : int {
    none = 0,
    preferred_gpu = 1,    // cudaMemAdviseSetPreferredLocation, the GPU
    preferred_cpu = 2,    // cudaMemAdviseSetPreferredLocation, the CPU: pages stay on the host
    accessed_by_gpu = 3,  // cudaMemAdviseSetAccessedBy: a mapping for the GPU, no migration
    read_mostly = 4,      // cudaMemAdviseSetReadMostly: read-duplicated on both sides
    prefetch_gpu = 5,     // cudaMemPrefetchAsync to the GPU, on `stream`
    prefetch_cpu = 6      // cudaMemPrefetchAsync to the CPU, on `stream`
};
constexpr int kUnifiedNumAdvice = 7;
const char* unified_advice_name(UnifiedAdvice advice);
bool unified_advise(void* ptr, size_t bytes, UnifiedAdvice advice, int device, cudaStream_t stream);

// What the device says about unified memory, from cudaDeviceGetAttribute. The GB10 is expected
// to report pageableMemoryAccess, concurrentManagedAccess and
// directManagedMemAccessFromHost; the 5090 reports pageableMemoryAccess only with the open
// kernel modules and HMM.
struct UnifiedDeviceCaps {
    int device = 0;
    int sm_count = 0;
    int unified_addressing = 0;
    int integrated = 0;
    int can_map_host_memory = 0;
    int pageable_memory_access = 0;
    int pageable_memory_access_uses_host_page_tables = 0;
    int concurrent_managed_access = 0;
    int direct_managed_mem_access_from_host = 0;
    int host_native_atomic_supported = 0;
    int can_use_host_pointer_for_registered_mem = 0;
};
UnifiedDeviceCaps unified_device_caps(int device = -1);

// Streams x (n bf16, any kind of memory) into sink[unified_read_sink_floats()] fp32 partial
// sums, one per block, overwritten each launch; the caller adds them. The grid is sized from
// the SM count, so the sink length does not depend on n.
// variant 0: one bf16 per thread per step, grid-stride
// variant 1: one 16-byte load per thread per step, grid-stride (16-byte aligned x)
// variant 2: four independent 16-byte loads in flight per thread per step
void unified_read_bf16(const __nv_bfloat16* x, float* sink, int64_t n, int variant,
                       cudaStream_t stream);
int unified_read_num_variants();
int unified_read_sink_floats();

// Decode GEMV: y[N] = W[N, K] . A[K], bf16 in and out, fp32 accumulation. A and y are device
// memory; W is [N, K] row-major (K contiguous, a Linear's stored layout) in any UnifiedKind.
// K % 8 == 0, 16-byte aligned A and W, any N (src/kernels/unified_gemv.cu).
// variant 0: one warp per row, one 16-byte load per lane per step
// variant 1: one warp per 4 rows, the next step's four loads in flight during the multiplies
// variant 2: variant 1's body on a persistent grid of SM count x resident blocks per SM
//            (cudaOccupancyMaxActiveBlocksPerMultiprocessor), warps striding over row groups
void unified_gemv_bf16(const __nv_bfloat16* A, const __nv_bfloat16* W, __nv_bfloat16* y, int N,
                       int K, int variant, cudaStream_t stream);
int unified_gemv_num_variants();

// The CPU-GPU signal pair, for measuring the handoff of a few bytes without a copy.
// Requirements, all of them:
//   * `flag`, `req`, `ack` and `out_iters` live in memory both sides can access while the
//     kernel runs: host_mapped, system, or managed on a device with concurrentManagedAccess.
//     Device memory is not readable by the host, and managed memory without concurrent
//     access faults on a host touch during a kernel.
//   * The GPU side reads and writes with cuda::atomic_ref<int, cuda::thread_scope_system>
//     (acquire loads, release stores), which bypasses L1 and orders the write of a payload
//     before the flag. The host side must do the same: __atomic_store_n(p, v,
//     __ATOMIC_RELEASE) after writing the payload, __atomic_load_n(p, __ATOMIC_ACQUIRE) in
//     the poll loop. A plain store may sit in a store buffer or a cache line the GPU does not
//     snoop; a plain load may be hoisted out of the loop.
//   * Over PCIe a system-scope load is a round trip to host memory per spin; on the GB10 it
//     is a coherent fabric read. Either way `max_spins` bounds the wait: a flag that never
//     arrives writes -1 to out_iters and returns instead of hanging the device.
//   * One trip is not one kernel launch: the host launches ping_pong once, then does `rounds`
//     trips against it, each trip being the host writing req = i and spinning until ack == i.
// unified_poll_flag: the GPU spins until *flag == expect, then writes the spin count (-1 on
// timeout) to out_iters. unified_set_flag: the GPU writes value to *flag (release), for the
// host to poll. unified_ping_pong: `rounds` trips of (wait req == i, store ack = i); out_iters
// gets the mean spins per trip.
void unified_poll_flag(volatile int* flag, int expect, int* out_iters, long long max_spins,
                       cudaStream_t stream);
void unified_set_flag(volatile int* flag, int value, cudaStream_t stream);
void unified_ping_pong(volatile int* req, volatile int* ack, int rounds, int* out_iters,
                       long long max_spins, cudaStream_t stream);
// ---- fused MoE expert step for decode (src/kernels/w1moe_decode.cu) --------------------------
// For T <= 8 tokens: y[t] = sum over the topk experts e of token t of
//     expert_weights[t, slot] * (silu(x[t] Wgate[e]) * (x[t] Wup[e])) Wdown[e]
// with the routed experts in the w1gemm format: packed_gate_up [E, 2I, K bits / 32] (gate
// columns 0..I-1 then up columns I..2I-1 of a [K, 2I] weight per expert, MoEBlock1Bit's
// layout), scales_gate_up [E, K / 128, 2I]; packed_down [E, K, I bits / 32], scales_down
// [E, I / 128, K]. x [T, K] and y [T, K] bf16, expert_ids int32 [T, topk], expert_weights
// bf16 [T, topk] (the renormalized routing weights). fp32 accumulation; the gate, up and
// SwiGLU values are rounded to bf16 where the sorted w1gemm_moe path rounds them. The shared
// expert and the router are the caller's. Needs 1 <= T <= 8, T * topk <= 64, bits 1 or 2,
// K % 256 == 0, K <= 16384, I % 128 == 0, 16-byte aligned x and packed weights.
// variant 0: three passes through global memory (gate|up GEMV per (token, slot), SwiGLU,
//            down GEMV summing the slots), scalar dequant
// variant 1: one persistent kernel; a work item is (expert, 128 columns of I): the block runs
//            gate and up for every token of the expert, the SwiGLU in shared memory, then its
//            128 rows of Wdown over all K with fp32 atomics into y; the intermediate never
//            touches DRAM. CUDA-core fp32 adds over the sign bits, no mma (see the source)
// variant 2: variant 1 with the routing weight folded into the down epilogue and the next
//            item's first gate|up groups in flight during the SwiGLU and the down pass
// variant 3: the same item on the tensor cores, 16 warps a block: a warp per 16-column gate
//            or up strip on w1gemm variant 3's body, the down pass as one-group strips with
//            the bf16 h slice as the B tile
// Variants 1 to 3 add into y with atomics, so the last bits can differ run to run.
void w1moe_decode_bf16(const __nv_bfloat16* x, const int32_t* expert_ids,
                       const __nv_bfloat16* expert_weights, const uint32_t* packed_gate_up,
                       const __nv_bfloat16* scales_gate_up, const uint32_t* packed_down,
                       const __nv_bfloat16* scales_down, __nv_bfloat16* y, int T, int topk, int E,
                       int K, int I, int bits, int variant, cudaStream_t stream);
int w1moe_decode_num_variants();
bool w1moe_decode_supports(int T, int topk, int K, int I, int bits, int variant);

// ---- Launch configuration: what the decode kernels decided about the device --------------
// (src/kernels/launch_config.cu; the rules live in src/kernels/launch_config.cuh.) Every
// decode kernel (hgemm's decode path, w4gemm's small-M shapes, w1gemm) is a weight stream
// whose grid used to be sized by the RTX 5090's constants (64 CTAs, 128 blocks, 170 SMs).
// They now ask the device: SM count, L2 bytes, DRAM spec bandwidth from the memory clock and
// bus width, and the bytes the device must keep in flight to stream at that bandwidth (the
// 5090's measured 1.5 MB; bandwidth x an assumed 800 ns elsewhere, the GB10's loaded latency
// being unmeasured). SPARK_FORCE_SMS=n (read once per process) replaces the SM count so a
// 5090 can run the 48-SM grids for a correctness pass; device_shape_forced() says it did.
int device_shape_sm_count();
int64_t device_shape_l2_bytes();
double device_shape_spec_gbps();
int64_t device_shape_bytes_in_flight();
bool device_shape_forced();
// The split-K rule and the L2 question on the current device (DECODE_DEVICE_CURRENT) or on a
// named shape without touching CUDA, for the host-side tests: `tiles` CTAs each keeping
// per_cta_in_flight bytes outstanding over a weight of weight_bytes, `slots` CTAs resident at
// once, a split of at most max_split -> the K slices per tile. decode_need_ctas is the CTA
// count the bandwidth rule asks for at that in-flight depth.
enum DecodeDevice { DECODE_DEVICE_CURRENT = 0, DECODE_DEVICE_GB10 = 1, DECODE_DEVICE_RTX5090 = 2 };
int decode_split_k(int64_t weight_bytes, int tiles, int max_split, int64_t per_cta_in_flight,
                   int slots, int device = DECODE_DEVICE_CURRENT);
int decode_need_ctas(int64_t per_cta_in_flight, int device = DECODE_DEVICE_CURRENT);
bool decode_fits_l2(int64_t bytes, int device = DECODE_DEVICE_CURRENT);
// Force the K split of every decode kernel that has one (0 = the rule; also the initial
// value of SPARK_DECODE_SPLIT). bench_decode_spark --sweep-split walks the legal splits with
// it; each kernel clamps it to its own range.
void decode_force_split_k(int split);
int decode_forced_split_k();

}  // namespace spark
