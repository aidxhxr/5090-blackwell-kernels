# FP8GEMM: e4m3 tensor-core GEMM

`C[M,N] = scale_a · scale_b · A[M,K] · Bt[N,K]^T`, A and Bt e4m3 with K contiguous, scales
per-tensor fp32 on the device, C bf16, fp32 accumulation on `mma.sync.m16n8k32`.
Source: `src/kernels/fp8gemm.cu` (variants 0 and 1), `src/kernels/fp8gemm_tma.cu` (variant 2),
the block tile they share in `src/kernels/fp8gemm_tile.cuh`, the two mma helpers in
`include/spark/common.cuh`. Bench: `bench_fp8gemm` (validates every variant against cuBLASLt
and a CPU sample). Python: `sk.fp8gemm(a, b_t, scale_a, scale_b)`, parity test against
`torch._scaled_mm` in `tests/test_fp8gemm.py`. The same kernels take MXFP8 block scales
(`sfa`, `sfb`; the "MX scales" section).

## Why fp8, and what the card does with it

The RTX 5090's fifth-generation tensor cores take 8-bit floats, and e4m3 with a per-tensor
scale is the precision inference ships at. NVIDIA's dense figure for the card is 419 TFLOPS
with fp32 accumulation at the 2.41 GHz spec clock, twice the bf16 number, and 838 with fp16
accumulation. `bench_peak` measures three things, all with register-resident operands:

| instruction | accumulate | measured | at |
|---|---|---|---|
| `mma.sync.m16n8k32...f32.e4m3.e4m3.f32` (the plain fp8 instruction, sm_89+) | fp32 | 517 TFLOPS | 2,975 MHz |
| the same with `.f16` accumulators | fp16 | 1,027 TFLOPS | |
| `mma.sync.m16n8k32.kind::mxf8f6f4.block_scale...f32.e4m3.e4m3.f32.ue8m0`, scales 2^0 (sm_120a) | fp32 | **1,014 TFLOPS** | 2,923 MHz |

So the plain instruction gives 2x bf16 with fp32 accumulation and 4x with fp16, the GeForce
split the Ada cards had. I found the third row by looking at what cuBLASLt runs: its fp8 kernel
(`nvjet_sm120_..._128x128x64_6_64x64x64_tmaAB`) measured 638 TFLOPS at 4096³ against the 517
ceiling of the plain instruction, with output bit-identical to fp32 accumulation, and its SASS
is `QMMA.SF.16832.F32.E4M3.E4M3.E8`: the block-scaled MXFP8 instruction, with the scale factors
set to one. That instruction multiplies each row of A and each column of B by a ue8m0 factor (a
power of two, 8 exponent bits) before the same fp32 accumulation, and with every factor 2^0 it
computes the same bits as the plain one at twice the rate. A one-warp probe on random e4m3
fragments through four accumulated steps returned 128 of 128 outputs identical to the plain
instruction. It is only exposed on the architecture-specific target, so the fp8 kernels and
`bench_peak` are compiled for `sm_120a` (CMake builds them as their own object library with the
"a" form of whatever `ARCH` is; `setup.py` defaults the whole extension to `compute_120a`), and
`mma_e4m3_16832` falls back to the plain instruction on a plain `sm_120` build. The results
scripts take the block-scaled row (`peak_fp8_mma`) as the fp8 roof; the other two rows are
documentation. The fp8 ridge is 1,014 / 1,792 ≈ 566 FLOP/byte, four times the bf16 one.

## Layout: both operands K-contiguous

`mma.sync.m16n8k32` with 8-bit types holds four elements per 32-bit register, lowest k in the
low byte: A is `a[4] = (row g / g+8) × (k 4c..4c+3 / 16+4c..16+4c+3)`, B is
`b[2] = (k 4c..4c+3 / 16+4c..16+4c+3) × col g`, with g = lane/4 and c = lane%4 as for the
bf16 shapes. `ldmatrix` is a 16-bit instruction: one of its 8x8 matrices is 8 rows of 16
bytes, and the word it hands lane (g, c) is b16 elements 2c and 2c+1 of row g. For e4m3 data
that word is bytes 4c..4c+3 of the row, which is exactly the m16n8k32 fragment word for that
lane. So a 16x32 e4m3 A tile is one `ldmatrix.x4` addressed like the bf16 tile's 16x16 (lane l
supplies row l%16, 16-byte chunk l/16), and no repacking exists anywhere.

B is where the layout decision is. The bf16 GEMM stores B as [K][N] and loads its fragments with
`ldmatrix.trans`, which pairs adjacent *n* into a word. The fp8 fragment wants adjacent *k* in a
byte pair, so the transposing load does not produce it; the options are a one-time repack of B
into a k-pair-interleaved layout, or B^T. I took B^T: `Bt[N][K]`, K contiguous, the same layout
as A. Then both operands load with a plain `ldmatrix` (two 8-row by 32-byte blocks of Bt per
`x4` give the fragments of two n8 tiles), both go through the same copy routine and the same
swizzle, and it is the layout cuBLASLt's fp8 path requires anyway (A row-major, B column-major,
its "TN" case), so the reference is timed on the same bytes. It is also how a weight already sits
in memory: `nn.Linear` stores `[out_features, in_features]`, and `torch._scaled_mm` takes the
same thing as `b_t.t()`. The API says so: `fp8gemm(A[M,K], Bt[N,K], C[M,N], scale_a, scale_b)`.

Scaling is per-tensor, the two fp32 scalars read from device memory at kernel start (no host
sync, and a scale computed on the device by the quantizer needs no round trip), multiplied once,
applied to the fp32 accumulators in the epilogue and rounded once to bf16. The split-K tail
sums unscaled partials in the workspace and scales at the conversion.

## The ladder

| variant | idea | what it fixes |
|---|---|---|
| 0 | one warp per 16x8 tile, fragment words loaded straight from global memory: lane (g, c) reads bytes 4c..4c+3 of rows g and g+8 of A and row g of Bt, one aligned 4-byte load per register | baseline: the fragment layout, checked against cuBLASLt |
| 1 | 128x128x64 block tile (64 bytes of K per stage), `ldmatrix` + `mma.sync` out of XOR-swizzled shared memory, 3-stage `cp.async` pipeline, register epilogue with the scale, split-K on the last partial wave, 64x128 and 64x64 tiles for small grids, one CTA per 16/32/64-row strip of Bt for M ≤ 64 (hgemm variant 3 and its decode kernel, on 8-bit rows) | reuse through smem and L2, DRAM latency, wave quantization, decode shapes at the memory floor |
| 2 | the 128x128 tile fed by TMA (`cp.async.bulk.tensor`) through a warp-specialized mbarrier pipeline: a producer warp issues two box loads per stage (128 rows by BK bytes of A and of Bt), the consumer warps run the tensor cores, no `__syncthreads` in the k-loop (hgemm variant 5) | the block barrier and the copy instructions in the consumer warps; at twice the mma rate they cost twice as much per byte |

### The tile

A stage is BM rows of A and BN rows of Bt, every row BK bytes as BK/16 chunks of 16 bytes,
XOR-swizzled by row so that the eight rows an `ldmatrix` touches fall in eight bank groups:
64-byte rows get `chunk ^= (row/2) % 4`, 128-byte rows `chunk ^= row % 8` (32-byte rows,
for the narrowest decode tile, `chunk ^= (row/4) % 2`). Those are the bf16 tile's functions
with the row length in bytes instead of elements, and the 64 B and 128 B forms are what the
TMA unit's `SWIZZLE_64B` and `SWIZZLE_128B` compute, so variant 2 addresses shared memory with
the same code. Per k32 step a warp with an MT x NT grid of accumulator tiles issues MT
`ldmatrix.x4` for A, NT/2 for Bt and MT x NT mma. Rows of A past M are zero-filled with a
`cp.async` of source size 0 and skipped in the epilogue, so any M ≥ 1 works; N % 64 == 0 and
K % 64 == 0 are required (the 128-byte-BK tiles are only picked when K % 128 == 0).

The 128x128 tile runs **2x2 warps of 64x64** (4 x 8 accumulator tiles, 190 registers, 128
threads, two blocks per SM), not hgemm's 2x4 warps of 64x32. At the bf16 rate a 64x32 warp tile
issues 6 `ldmatrix` per 16 mma of 32 pipe cycles each; at the fp8 rate the same 16 mma take 16
cycles each, so the load and address instructions are twice the fraction of the pipe time, and
the 64x64 tile's 8 `ldmatrix` per 32 mma is the way to halve them. cuBLASLt's kernel is 4 warps
of 64x64 at 255 registers for the same reason. Variant 1's sweep on the 128x128 tile,
`--iters=20`, TFLOPS at 4096³ / 8192³ / 4096x11008x4096 (cuBLASLt in the same session 633 /
708 / 621):

| BK, stages, warps | smem | regs | 4096³ | 8192³ | 4096x11008x4096 |
|---|---|---|---|---|---|
| 64, 3, 2x4 of 64x32 | 48 KB | 125 | 602 | 626 | 572 |
| 128, 2, 2x4 | 64 KB | 126 | 597 | 635 | 587 |
| 128, 3, 2x4 | 96 KB | 127 | 597 | 636 | 581 |
| 64, 4, 2x4 | 64 KB | 126 | 539 | 578 | 546 |
| **64, 3, 2x2 of 64x64** | 48 KB | 190 | **612** | **636** | **589** |
| 128, 2, 2x2 | 64 KB | 203 | 567 | 626 | 589 |

`SPARK_FP8GEMM_V1_CONFIG=<index>` forces one of them for a re-sweep. The smaller tiles keep
the 2x4 grid (a 64x64 tile with 2x2 warps would be 32x32 per warp, NT = 4, half the reuse).

### Scheduling

Variant 1's work assignment is hgemm variant 3's: one block per tile, and the tiles past the
last full wave of resident blocks split `min(KT, resident / tail)` ways along K over the idle
blocks, into an fp32 workspace with atomics, memset on the stream before the launch, the last
slice to arrive scaling and converting. Tile selection is the same rule too: 128x128 when its
grid is a full wave, then 64x128, then 64x64 with a 128-byte BK. A 4096³ fp8 GEMM is still
1,024 tiles on 340 slots, 3.01 waves, so the tail split still matters here.

Decode shapes (M ≤ 64) run one CTA per BN-row strip of Bt, which for the transposed layout is
one contiguous run of BN x K bytes: 16x64, 32x32 and 64x32 tiles with a 128-byte BK and four
stages (24 KB of Bt in flight per CTA, the amount the bf16 decode kernel found the card needs
at 64 CTAs), the second configuration of each pair when the strips do not fit in one wave, and
no split-K unless a narrow N leaves fewer than 64 strips.

### Variant 2

The port of hgemm variant 5 is mechanical because the two operands are the same shape: two
tensor maps of rows x K bytes (`CU_TENSOR_MAP_DATA_TYPE_UINT8`, the kernel never interprets
the bytes), two boxes of 128 rows by BK bytes per stage, one `arrive.expect_tx` of the stage's
byte count, the same full/empty mbarrier pairs and parity rule, the same `fence.proxy.async`
before a consumer warp releases a stage, the same named barrier for the tail. The consumer
warp grid is a template parameter, and both grids were swept. TFLOPS at `--iters=20`, 2048³ /
4096³ / 8192³ / 4096x11008x4096 / 4096x4096x11008 (cuBLASLt 410 to 418 / 627 to 639 / 706 to
711 / 595 to 599 / 667 to 673 in the same session):

| BK, stages, warps, blocks/SM | smem | regs | 2048³ | 4096³ | 8192³ | 4096x11008x4096 | 4096x4096x11008 |
|---|---|---|---|---|---|---|---|
| 128, 2, 2x4, 1 | 64 KB | 128 | 510 | 668 | 699 | 623 | 667 |
| **128, 3, 2x4, 1** | 96 KB | 135 | 510 | **675** | **704** | 627 | **680** |
| 64, 3, 2x4, 2 | 48 KB | 96 | 512 | 649 | 668 | 609 | 643 |
| **64, 2, 2x4, 2** | 32 KB | 96 | **545** | 656 | 658 | 605 | 644 |
| 64, 4, 2x4, 1 | 64 KB | 122 | 511 | 662 | 678 | 617 | 653 |
| 128, 3, 2x2, 1 | 96 KB | 168 | 482 | 638 | 694 | 626 | 680 |
| 128, 2, 2x2, 1 | 64 KB | 168 | 510 | 649 | 706 | **638** | 682 |
| 64, 3, 2x2, 2 | 48 KB | 195 | 511 | 649 | 690 | 626 | 660 |
| 64, 2, 2x2, 2 | 32 KB | 195 | 542 | 662 | 693 | 625 | 670 |

Runs repeat to about ±1.5%. Three stages of 128-byte BK at one block per SM is the shipped
configuration for grids past two blocks per SM; under that (2048³ is 256 tiles) the 32 KB
two-block configuration wins by 7%, because 256 tiles on 340 slots is a wave with no tail
where 256 on 170 is 1.5 waves with a split. The 2x2 warp grid that won in variant 1 does not
win here: at one block per SM it leaves 1.25 warps per scheduler to hide the `ldmatrix` and
mbarrier latency, and the instruction savings go into stalls. `SPARK_FP8GEMM_V2_CONFIG=<index>`
forces one.

## Measured

`bench_fp8gemm --iters=50`, the full run, cuBLASLt (`cublasLtMatmul`, `CUDA_R_8F_E4M3` in,
`CUDA_R_16BF` out, `CUBLAS_COMPUTE_32F`, scales through `A_SCALE_POINTER` /
`B_SCALE_POINTER`, a 32 MB workspace and its own heuristic's first algorithm) timed the same
way on the same stream. "%" is cuBLASLt time ÷ our time against the cuBLASLt loop timed next to
that variant. Every row passes both checks: the whole output against cuBLASLt and 4,096 sampled
outputs against a CPU fp32 dot product of the same e4m3-rounded inputs, with the 2% of max|C|
tolerance of `bench_hgemm`. Every max|difference| against cuBLASLt was exactly zero, and that is
arithmetic, not luck: an e4m3 value in [-1, 1] is an integer multiple of 2^-9, so every product
is a multiple of 2^-18, and an fp32 partial sum below 64 in magnitude holds that resolution
exactly, so both sides compute the same sum in any order and round it once.

| M x N x K | cuBLASLt ms / TFLOPS | v0 ms / TFLOPS / % | v1 | v2 |
|---|---|---|---|---|
| 1024³ | 0.0105 / 205.2 | 0.0340 / 63.3 / 30.8% | **0.0092 / 233.8 / 114.1%** | n/a |
| 2048³ | 0.0412 / 416.8 | 0.2405 / 71.4 / 17.1% | 0.0420 / 409.2 / 98.1% | **0.0296 / 580.4 / 139.2%** |
| 4096³ | 0.1946 / 706.2 | 2.0751 / 66.2 / 9.4% | 0.2141 / 642.0 / 90.9% | **0.1953 / 703.8 / 99.5%** |
| 8192³ | 1.4874 / 739.2 | 17.343 / 63.4 / 8.6% | 1.7188 / 639.7 / 86.7% | **1.6185 / 679.4 / 92.0%** |
| 4096x4096x11008 | 0.6356 / 581.2 | 7.5648 / 48.8 / 8.4% | 0.5556 / 664.8 / 114.7% | **0.5101 / 724.1 / 124.9%** |
| 4096x11008x4096 | 0.5507 / 670.8 | 5.7448 / 64.3 / 9.6% | 0.5660 / 652.7 / 97.3% | **0.5474 / 674.8 / 100.7%** |
| 1x4096x4096 | 0.0186 / 1.8 | n/a | **0.0132 / 2.5 / 140.9%** | n/a |
| 16x4096x4096 | 0.0165 / 32.5 | 0.0277 / 19.4 / 61.7% | **0.0132 / 40.8 / 125.0%** | n/a |
| 32x4096x4096 | 0.0165 / 65.0 | 0.0296 / 36.3 / 55.7% | **0.0132 / 81.4 / 125.0%** | n/a |
| 64x4096x4096 | 0.0165 / 130.1 | 0.0378 / 56.8 / 43.7% | **0.0132 / 162.9 / 125.0%** | n/a |
| 16x11008x4096 | 0.0330 / 43.8 | 0.0480 / 30.1 / 68.5% | **0.0295 / 48.9 / 111.9%** | n/a |
| 64x4096x11008 | 0.0360 / 160.2 | 0.0989 / 58.3 / 36.4% | **0.0316 / 182.4 / 113.9%** | n/a |

Variant 2 takes grids of at least one 128x128 tile per SM with K % 128 == 0, so the default
steps down to variant 1 below 2048³ and on the decode shapes. cuBLASLt moved between 706 and
739 TFLOPS at 8192³ across the session (the card is at its power limit, next paragraph); v2 in
a 1,500-iteration loop of that shape, timed in one lock with cuBLASLt, is 663 against 706, 94%.
The decode rows in GB/s (`(MK + KN) + 2MN` bytes ÷ time, the traffic floor): 1,274, 1,291,
1,302, 1,332, 1,541 and 1,464. Against torch: `torch._scaled_mm` with the same bytes, scales
and layout runs 1.12 to 1.29x slower than `sk.fp8gemm` on the four large shapes and 1.5 to
1.6x on the decode ones (`scripts/bench_torch.py`); torch's call is slower than the bench's
direct cuBLASLt call on the same shapes (535 against 706 TFLOPS at 4096³), which is its
heuristic and workspace, not the tensor cores.

### The single-launch floor for 16 MB

fp8 weights are half the bytes, so the question the bf16 decode note asked has a new answer.
A read-only kernel streaming N bytes with 16-byte loads, rotated past L2, timed as the bench
times a GEMM:

| | 16 MB | 32 MB | 44 MB |
|---|---|---|---|
| one launch (680 blocks) | 13.3 µs (1,260 GB/s) | 23.6 µs (1,423) | 30.0 µs (1,539) |
| 20 launches back to back, per launch | 11.3 µs (1,486 GB/s) | 21.3 µs (1,573) | 28.8 µs (1,605) |
| one launch, 64 blocks | 25.8 µs (650 GB/s) | | |

So a lone 16 MB launch has a floor of 13.3 µs the way the bench times it, and the 4096² fp8
decode rows are at 13.2 µs: the weights and nothing else, at M = 1, 16, 32 and 64 alike. The
2 µs of launch and ramp that back-to-back launches hide is a larger share of a 13 µs call than
of a 23 µs one, which is the CUDA-graphs item in the README's list. 64 blocks are not enough
here where they were for the bf16 kernel's strips: a grid-stride read from 64 blocks keeps
half the bytes in flight that 64 strips with four 8 KB stages each do.

### Power

`nvidia-smi --query-gpu=clocks.sm,power.draw -lms 200` during a 1,500-iteration 8192³ loop of
v2 (the bench times cuBLASLt first):

| kernel | SM clock | power | TFLOPS |
|---|---|---|---|
| cuBLASLt | 2,287 to 2,310 MHz | 600 W | 706 |
| v2 | 2,130 to 2,160 MHz | 600 W | 663 |

Both kernels pin the card at 600 W within a second. The fp8 GEMM runs 600 MHz below the bf16
one at the same limit (2.75 GHz for hgemm), so the usable fp8 roof for a long GEMM is about
1,014 x 2.15 / 2.92 ≈ 745 TFLOPS, and cuBLASLt's 706 to 739 is that roof. The 7% clock
difference is the whole gap between the two kernels, as Nsight shows next.

### Nsight Compute at 8192³ (one launch, clocks locked by ncu)

| metric | v1 (64, 3, 2x2) | v2 (128, 3, 2x4) | cuBLASLt |
|---|---|---|---|
| duration, SM clock | 1.73 ms at 2.33 GHz | 1.58 ms at 2.23 GHz | 1.53 ms at 2.31 GHz |
| SM cycles (duration x clock) | 4.03 M | 3.52 M | 3.53 M |
| tensor pipe active (`sm__pipe_tensor_cycles_active`) | 78.2% | 89.5% | 89.3% |
| tensor instructions | 134.2 M | 134.2 M | 134.2 M |
| executed instructions | 398 M | 341 M | 233 M |
| issue slots busy | 14.7% | 14.7% | 10.4% |
| stall: `math_pipe_throttle` (cycles per issue) | 4.8 | 6.5 | 14.8 |
| stall: `wait` | 4.2 | 4.5 | 0.6 |
| stall: `barrier` | 0.8 | 0.0 | 0.0 |
| stall: `long_scoreboard` | 0.6 | 1.0 | 0.4 |
| stall: `mio_throttle` | 0.8 | 0.05 | 0.7 |
| L2 hit rate | 98.1% | 98.0% | 93.9% |
| DRAM throughput | 8.4% | 9.1% | 16.3% |
| shared bank conflicts / load wavefronts | 268 K / 143 M | 0 / 214 M | 2.1 M / 141 M |
| registers, blocks per SM | 190, 2 | 133, 1 | 255, 1 |
| achieved occupancy | 16.5% | 18.6% | 16.6% |

At a fixed clock v2 and cuBLASLt take the same number of cycles (3.52 M against 3.53 M) with
the same 89% tensor-pipe activity and the same 134 M `QMMA` instructions; the bench gap is the
clock under the power cap. What cuBLASLt does with fewer joules is issue less: 233 M
instructions against 341 M for the same tensor work, 0.74 non-mma instructions per mma against
1.54. Its 64x64 warp tile is a quarter of that (8 `ldmatrix` per 32 mma against 6 per 16), and
the rest is a leaner pipeline body; the SASS of v2's k-loop is 64 `QMMA`, 24 `LDSM` and about
20 address instructions per stage with the loads already interleaved into the mma stream by
the compiler, so the next step is the warp tile at one block per SM without losing the latency
hiding, which is what the 2x2 rows of the sweep lost. v1's 78% pipe activity is the block
barrier (0.8 cycles per issue) plus the `cp.async` address generation, the same 10 points TMA
bought the bf16 kernel; its 64x64 warp tile at two blocks per SM keeps the instruction count
close to v2's despite the copies.

## Correctness

Inputs uniform in [-1, 1] rounded to e4m3 (round to nearest even), scales 0.75 and 1.5.
cuBLASLt in the TN layout is the reference for the whole output, and a CPU fp32 dot product of
the same rounded values checks 4,096 sampled outputs, both with `max|C − C_ref| ≤ 0.02·max|C_ref|
+ 1e-3`. `tests/test_fp8gemm.py` runs every tile, the decode paths, the split-K workspaces over
repeated calls and the variant 2 step-down against `torch._scaled_mm`, and checks the bf16 bits
are identical to it at K = 64, where every partial sum is exact.

## MX scales

MXFP8 (the OCP Microscaling format) keeps e4m3 elements and adds one E8M0 scale, a power of
two stored as exponent + 127 in a byte, per 32 consecutive elements. The block-scaled
instruction takes exactly that: a ue8m0 factor per row of A and per column of B per 32 k.
The per-tensor kernels feed it 2^0 everywhere; MX mode feeds it the real scales, so the
whole difference between the two modes is where the scale bytes come from. The API is the
same call with two more tensors, `fp8gemm_mx(A, Bt, C, M, N, K, scale_a, scale_b, sfa, sfb,
variant, stream)` with `sfa[M][K/32]` and `sfb[N][K/32]` row-major uint8, and
`sk.fp8gemm(a, b_t, sfa=sfa, sfb=sfb)` from Python (the per-tensor scales are optional on
top). Every variant takes it; the bench reports it as dtype `mxfp8`. The reference is the
OCP recipe in `spark_kernels.reference.quantize_mx`: a block's exponent is
floor(log2(max|x|)) - 8, so its largest element lands in e4m3's top binade, the elements are
x / 2^e rounded to e4m3 with saturation at 448, and the fp32 product of the dequantized
values is what the kernel must reproduce (a power of two times an e4m3 value is exact in
fp32, so only the summation order differs).

### The operand layout, as measured

The ISA says the scale operands are one `.b32` register per lane per operand plus two
immediates, byte-id and thread-id, and describes the mapping in a figure. I measured it
instead: a probe kernel put distinct ue8m0 bytes in every lane's register (63 + lane + 32 x
byte), ran the instruction on all-ones fragments and decoded log2 of each output. For
`m16n8k32` with `.scale_vec::1X`:

| operand | thread-id 0 | thread-id 1 | byte |
|---|---|---|---|
| A row g (g = lane / 4) | lane 4g | lane 4g + 2 | byte-id of that lane's register |
| A row g + 8 | lane 4g + 1 | lane 4g + 3 | same |
| B column g | lane 4g | lane 4g + 1 | same |

Thread-id is 0 or 1 (ptxas rejects 2 and 3), and the lanes not named are ignored. So with
thread-id 0, lane (g, c) supplies row g + 8 (c & 1) of A and column g of B, and lanes with
c = 2, 3 can hold anything: the kernels give them copies. A row's scales lie along k in
memory, so the 4-byte word at byte 4w of row r holds k-blocks 4w .. 4w+3, which is one
128-byte stage (or two 64-byte ones), and byte-id = kk / 32 selects the k32 step inside the
stage without any repacking: the word goes into the instruction as loaded. Variant 0 reads
its two bytes per k32 step straight from the tensors and is bit-identical to cuBLASLt's
MXFP8 kernel on every shape (max |difference| 0 in the bench), which pins the mapping down
independently of anything the tiled variants do.

### Staging

The tiles come through shared memory; the scales do not. Three stages of 128-byte tiles are
97 KB of the 99 KB a block may take on this card, and the decode configurations fill their
SM at two to four blocks of 24 to 48 KB, so there is no room for a scale ring next to the
tiles without dropping a stage or a block. Instead each consumer warp fetches its own scales
from global memory, one group of stages ahead of the mma, and the question is how.

1. *One 4-byte word per row per lane per stage.* Lane (g, c) loads the word of row g + 8
   (c & 1) of each of its MT A tiles and of column g of each NT Bt tile, one stage ahead.
   Twelve loads per warp per stage, each touching 8 to 16 distinct 32-byte sectors of which
   4 bytes are used: Nsight at 4096³ measured 12 sectors per request, a 61% L1 hit rate, 29%
   more L2 sectors than the tiles themselves, `long_scoreboard` at 6.3 cycles per issue and
   the tensor pipe at 52%. 492 TFLOPS on variant 2 against 704 per-tensor.
2. *One 16-byte chunk per row per 32 lanes, then shuffles.* A 16-byte chunk holds a row's
   scales for four 128-byte stages, and the warp needs the chunks of its 64 rows of A and 32
   columns of Bt: two `uint4` loads for A and one for Bt per lane per group of four stages
   (8-byte chunks and groups of two stages when K % 512 != 0, so that every row stride stays
   aligned; K = 11008 is that case). Each stage then takes its eight words out of the chunks
   with one `__shfl_sync` per tile: lane (g, c) reads word i of the chunk that lane
   (16 mi + g + 8 (c & 1)) % 32 loaded for A tile mi, and of the chunk lane 8 nj + g loaded
   for Bt tile nj. Sector requests drop four times; the four warps that share A rows and the
   two that share Bt columns hit L1 on each other's sectors inside a group (L1 keeps a
   sector for a group, not across groups: 256 lines per group are the whole 32 KB L1 that
   remains next to 97 KB of smem). This is the shipped design.
3. *One chunk per row per lane, no shuffles.* Every lane fetches the chunks of the rows its
   own words stand for (MT + NT `uint4` loads per group). Same sectors, no shuffle traffic,
   but two register sets of (MT + NT) x 4 words: 64 registers for the 2x4 warp grid, 96 for
   the 2x2 one, and the two-blocks-per-SM configurations, whose launch bound caps them at 96
   registers, spill. In one session, both on the 3-stage 128-byte configuration: 631 / 592
   TFLOPS on variant 2 at 4096³ / 8192³ against 582 / 560 for design 2, but 571 against 580
   on variant 1, 15.6 against 15.2 µs on the 16-row decode shape and 37.6 against 34.0 µs on
   64x4096x11008. I kept design 2 and moved variant 2 to the four-stage configuration below,
   which took it to 626 / 627; design 3 on that configuration is the next thing to try.

Two orderings of design 2's loop lost as well, measured back to back (v2, 4096³ / 8192³,
TFLOPS, the card in a slower state that session: per-tensor v2 read 678): copy the next
group's chunks over the current ones, shuffle each stage's words just before its mma, **582 /
560**; the same with the words shuffled one stage ahead, 552 / 513; two register sets used
alternately so nothing is copied, 577 / 538; alternate sets and words ahead, 547 / 515. The
copy is 12 moves per group and the compiler schedules the plain form best, so that is what
ships.

The v2 pipeline configuration is not the per-tensor one either. With the consumers
carrying the fetches and shuffles, a deeper pipeline of 64-byte stages hides more than three
128-byte stages do (`SPARK_FP8GEMM_V2_CONFIG` sweep in MX mode, same slow session, TFLOPS at
4096³ / 8192³ / 4096x11008x4096 / 2048³):

| config | BK, stages, warps, blocks/SM | 4096³ | 8192³ | 4096x11008x4096 | 2048³ |
|---|---|---|---|---|---|
| 0 | 128, 2, 2x4, 1 | 567 | 540 | 518 | 478 |
| 1 (per-tensor default) | 128, 3, 2x4, 1 | 592 | 558 | 526 | 477 |
| 3 (per-tensor small grids) | 64, 2, 2x4, 2 | 475 | 457 | 439 | 389 |
| **4 (MX default)** | 64, 4, 2x4, 1 | **614** | **617** | **563** | **478** |
| 5 | 128, 3, 2x2, 1 | 572 | 557 | 514 | 454 |
| 6 | 128, 2, 2x2, 1 | 572 | 549 | 519 | 477 |

| 9 | 64, 5, 2x4, 1 | 608 | 605 | 566 | 477 |
| 10 | 64, 6, 2x4, 1 | 604 | 604 | 564 | 477 |

(Configurations 9 and 10 were added and measured in a later session, where 4 read 608 /
617 / 564 / 478 and 1 read 582 / 567 / 530 / 477.) Four 64-byte stages is where the curve
flattens, and it is the MX default for every grid: at 2048³ the two-block configuration 3
that the per-tensor mode picks there loses 19% in MX mode.

The two-block configurations (2, 3, 7, 8) lose 20 to 45% in MX mode: their launch bound
caps the kernel at 96 registers and the chunk sets push it into local memory. Variant 1
keeps its 2x2-warp 64x64 tile (`SPARK_FP8GEMM_V1_CONFIG` 0: 571 / 544 / 524 against 443 to
539 for the others).

### Split-K on chunk boundaries

A chunk group is SPC = SFW x 128 / BK stages, and the loop is unrolled by SPC so that the
chunk word and the byte-id are immediates. A split-K slice must therefore start on a group
boundary: the kernels round every slice start down to a multiple of SPC (the last slice
runs to K) and the launchers cap `split` at KT / SPC, so no slice is empty. Everything else
about the schedule, the tail atomics and the decode strips is unchanged.

### Measured

`bench_fp8gemm --iters=50`, the full run, MX rows (dtype `mxfp8`): Gaussian inputs
quantized with the OCP recipe on the CPU, cuBLASLt's MXFP8 kernel (scale mode
`CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0` on both operands, its scales in the 32x4x4 tiled
layout the bench builds from the row-major ones) timed the same way on the same stream and
checked whole-output against, plus the 4,096-sample CPU check of the dequantized products.
"%" is cuBLASLt MXFP8 time ÷ our time; the last column is the fastest MX rung against the
fastest per-tensor rung of the same run. The fp8 kernels' per-tensor rows in this run: v2
at 703 / 682 / 738 / 699 TFLOPS on the four largest square and Llama shapes.

| M x N x K | cuBLASLt MXFP8 ms / TFLOPS | v0 ms / TFLOPS / % | v1 | v2 | best MX / best per-tensor |
|---|---|---|---|---|---|
| 1024³ | 0.0145 / 148.1 | 0.0418 / 51.4 / 34.7% | **0.0089 / 240.5 / 165.6%** | n/a | 102.9% |
| 2048³ | 0.0379 / 453.4 | 0.3263 / 52.7 / 11.6% | 0.0438 / 392.4 / 84.6% | **0.0340 / 506.0 / 110.7%** | 86.8% |
| 4096³ | 0.1997 / 688.2 | 5.1818 / 26.5 / 3.9% | 0.2199 / 624.9 / 90.8% | **0.2196 / 625.7 / 90.9%** | 89.0% |
| 8192³ | 1.6736 / 657.0 | 42.920 / 25.6 / 3.9% | 1.7855 / 615.8 / 93.0% | **1.7544 / 626.7 / 95.2%** | 91.9% |
| 4096x4096x11008 | 0.5390 / 685.2 | 15.152 / 24.4 / 3.6% | **0.6416 / 575.7 / 83.9%** | 0.6477 / 570.3 / 83.2% | 78.0% |
| 4096x11008x4096 | 0.5205 / 709.7 | 14.456 / 25.6 / 3.6% | **0.6252 / 590.8 / 83.3%** | 0.6413 / 576.0 / 80.9% | 84.5% |
| 1x4096x4096 | 0.0298 / 1.1 | n/a | **0.0149 / 2.3 / 200.4%** | n/a | 88.4% |
| 16x4096x4096 | 0.0278 / 19.3 | 0.0500 / 10.7 / 55.6% | **0.0150 / 35.8 / 184.4%** | n/a | 87.8% |
| 32x4096x4096 | 0.0255 / 42.0 | 0.0500 / 21.5 / 51.1% | **0.0151 / 71.2 / 170.9%** | n/a | 87.7% |
| 64x4096x4096 | 0.0338 / 63.6 | 0.0541 / 39.7 / 62.4% | **0.0151 / 141.9 / 223.9%** | n/a | 87.3% |
| 16x11008x4096 | 0.0503 / 28.7 | 0.0897 / 16.1 / 56.1% | **0.0325 / 44.4 / 154.0%** | n/a | 91.0% |
| 64x4096x11008 | 0.0463 / 124.7 | 0.1463 / 39.4 / 31.6% | **0.0339 / 170.1 / 136.3%** | n/a | 93.4% |

Variant 0 is bit-identical to cuBLASLt on every shape (max |difference| 0); the tiled
variants differ from it by at most one bf16 ulp where the split-K order differs. cuBLASLt's
MXFP8 kernel runs 3% under its per-tensor one on the square shapes and 10 to 20% over it
on the Llama ones (685 and 710 against 636 and 671 TFLOPS: a different kernel, with the
tiled scales). Ours costs 8 to 11% on the large shapes and 7 to 14% on the decode rows
(15.0 µs against 13.2 for the 16 MB weight, 1,170 GB/s counting the scale bytes), where
cuBLASLt's MXFP8 decode kernel takes 26 to 50 µs. The 11008-deep shapes (SFW = 2, 8-byte
chunks, groups of two stages) lose the most: twice the fetch instructions per stage of
the 16-byte path.

### Nsight Compute at 8192³ (one launch, clocks locked by ncu)

| metric | v2 per-tensor (128, 3, 2x4) | v2 MX (64, 4, 2x4) |
|---|---|---|
| duration, SM clock | 1.57 ms at 2.23 GHz | 1.72 ms at 2.12 GHz |
| SM cycles | 3.51 M | 3.65 M |
| tensor pipe active | 90.8% | 87.9% |
| executed instructions | 323 M | 402 M |
| issue slots busy | 14.1% | 16.9% |
| global load requests / sectors | 72 K / 101 K | 1.64 M / 50.4 M |
| L1 hit rate | 76.8% | 61.8% |
| L2 sectors, L2 throughput | 277 M, 81% | 298 M, 87% |
| shared-memory wavefronts | 214 M | 263 M |
| stall: `math_pipe_throttle` (cycles per issue) | 6.6 | 3.7 |
| stall: `wait` | 4.8 | 4.8 |
| stall: `long_scoreboard` | 1.1 | 1.1 |
| stall: `mio_throttle` / `short_scoreboard` | 0.1 / 0.3 | 0.6 / 0.6 |
| registers | 128 | 160 |

At a fixed clock the MX kernel takes 3.8% more cycles than the per-tensor one, and the
bench's 9.5% gap is that plus the clock: 24% more instructions (the fetches, the shuffles
and the copy, 79 M over 134 M `QMMA`) at the same 600 W is 5% less clock. The scale traffic
itself is small once the chunks are 16 bytes: 21 M extra L2 sectors on 277 M (7.7%, of
which the scale bytes themselves are 3%), `long_scoreboard` unchanged at 1.1, and the
tensor pipe 3 points lower where design 1 had it at 52%. What remains is on the MIO side:
the shuffles are 49 M more shared-memory wavefronts (23%) and `mio_throttle` plus
`short_scoreboard` rise from 0.4 to 1.2 cycles per issue, which is the cost the tiled scale
layout would remove.

### Accuracy

`scripts/mx_accuracy.py` at 4096³: the fp32 product of the bf16 inputs is the reference,
each scheme quantizes A and B and runs `sk.fp8gemm`, and the table gives the largest error
relative to the largest |C| and the Frobenius norm of the error relative to that of C.
"Floor" is the OCP recipe (block exponent floor(log2(max|x|)) - 8, a block maximum in the
top eighth of its binade saturates at 448); "ceil" is ceil(log2(max|x| / 448)), the
smallest exponent under which nothing saturates. Per-tensor is max|x| / 448.

| input | per-tensor max / Frobenius | MX floor | MX ceil |
|---|---|---|---|
| Gaussian | 4.3e-2 / 3.75e-2 | 4.5e-2 / 4.16e-2 | 4.0e-2 / 3.77e-2 |
| Gaussian, 0.1% of entries x 100 | 4.2e-2 / 3.76e-2 | 9.9e-2 / 5.73e-2 | 4.0e-2 / 3.74e-2 |
| Gaussian, 8 channels of A x 50 | 5.5e-2 / 3.73e-2 | 9.9e-2 / 4.86e-2 | 5.1e-2 / 3.78e-2 |
| Gaussian, 8 channels of A x 2000 | 6.0e-2 / 3.75e-2 | 1.07e-1 / 5.06e-2 | 6.0e-2 / 3.78e-2 |

The honest reading: on these inputs the block scales buy nothing over a per-tensor scale.
e4m3 has three mantissa bits, so every scheme lands at 3.7 to 3.8% relative Frobenius
error, and a per-tensor scale keeps even a 2000x outlier inside e4m3's 2^14.8 of dynamic
range (the Gaussian body of the x2000 tensor sits 2^13 below its maximum, still in e4m3's
normal range after the per-tensor scale). The floor recipe is measurably worse (the block maxima that fall in [448, 512)
saturate), which is why the quantizer takes a `mode` and the bench uses floor as the spec
says. Where MX earns its keep is a tensor whose range exceeds
2^15, or a per-channel range that a per-tensor scale cannot follow at all (activations
with a few 10^4-sized channels next to 10^-2 ones), and in not needing a calibration pass
over the tensor; neither shows up in a Gaussian test. `tests/test_fp8gemm.py` pins the
ceil mode to within 3% of per-tensor and the floor mode between 1x and 1.6x of it, on
Gaussian and on the x2000 input.

## What was done and what remains

Done: the fragment layout and the b16-pair `ldmatrix` for both operands with no transposing
load; the TN layout that matches cuBLASLt and the weight's own storage; the block-scaled
instruction that doubles the fp32-accumulate rate, found in cuBLASLt's SASS and measured at
1,014 TFLOPS; the three rungs, with v2 at 99 to 139% of cuBLASLt on the shapes it takes and
the decode shapes at the 16 MB single-launch floor, 112 to 141% of cuBLASLt; MX mode on
every rung, with the scale operand layout measured and the scales fetched by the consumer
warps.

Open:

- **8192³ at 92 to 94%.** Same cycles as cuBLASLt at a fixed clock, 7% less clock at 600 W.
  The instruction count per mma is the lever: a 64x64 warp tile at one block per SM needs the
  fragment loads for step k+1 issued before step k's mma from a second register set, so that
  1.25 warps per scheduler can keep the pipe fed; the compiler does not do that on its own for
  the 2x2 configurations.
- **Stream-K.** Variant 1 has v3's tail split, not v4's persistent schedule; 4096x11008x4096
  (2,752 tiles, 8.1 waves) and the grouped tile order are the shapes it would help.
- **MX at 90%.** The MX mode costs 8 to 10% on the large shapes against the per-tensor
  kernel where cuBLASLt's MXFP8 kernel costs it 3%: eight shuffles per stage on the MIO
  queue the `ldmatrix` traffic already fills, and the chunk loads. cuBLASLt takes its scales
  in a tiled layout (128 rows by 4 k-blocks in 512 contiguous bytes, the layout torch calls
  `SWIZZLE_32_4_4`) that one TMA box or two `ldmatrix`-shaped shared loads serve; a
  row-major `[rows][K/32]` operand cannot be moved that way. Taking the tiled layout as an
  input option, or repacking the scales once per call, is the way past 90%, and per-channel
  bf16 scales in the epilogue are a smaller rung.
- **The decode launch.** 13.2 µs single launch against 11.3 back to back for 16 MB: the 2 µs
  that CUDA graphs or a persistent decoder kernel would take back.
