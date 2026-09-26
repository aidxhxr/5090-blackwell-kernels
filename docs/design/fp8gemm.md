# FP8GEMM: e4m3 tensor-core GEMM

`C[M,N] = scale_a · scale_b · A[M,K] · Bt[N,K]^T`, A and Bt e4m3 with K contiguous, scales
per-tensor fp32 on the device, C bf16, fp32 accumulation on `mma.sync.m16n8k32`.
Source: `src/kernels/fp8gemm.cu` (variants 0 and 1), `src/kernels/fp8gemm_tma.cu` (variant 2),
the block tile they share in `src/kernels/fp8gemm_tile.cuh`, the two mma helpers in
`include/spark/common.cuh`. Bench: `bench_fp8gemm` (validates every variant against cuBLASLt
and a CPU sample). Python: `sk.fp8gemm(a, b_t, scale_a, scale_b)`, parity test against
`torch._scaled_mm` in `tests/test_fp8gemm.py`.

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

## What was done and what remains

Done: the fragment layout and the b16-pair `ldmatrix` for both operands with no transposing
load; the TN layout that matches cuBLASLt and the weight's own storage; the block-scaled
instruction that doubles the fp32-accumulate rate, found in cuBLASLt's SASS and measured at
1,014 TFLOPS; the three rungs, with v2 at 99 to 139% of cuBLASLt on the shapes it takes and
the decode shapes at the 16 MB single-launch floor, 112 to 141% of cuBLASLt.

Open:

- **8192³ at 92 to 94%.** Same cycles as cuBLASLt at a fixed clock, 7% less clock at 600 W.
  The instruction count per mma is the lever: a 64x64 warp tile at one block per SM needs the
  fragment loads for step k+1 issued before step k's mma from a second register set, so that
  1.25 warps per scheduler can keep the pipe fed; the compiler does not do that on its own for
  the 2x2 configurations.
- **Stream-K.** Variant 1 has v3's tail split, not v4's persistent schedule; 4096x11008x4096
  (2,752 tiles, 8.1 waves) and the grouped tile order are the shapes it would help.
- **Real scales.** Per-tensor scales are what the API takes. The block-scaled instruction takes
  a ue8m0 factor per row of A and column of B per 32 k for free, which is MXFP8; a `[N][K/32]`
  scale operand and its smem staging is the next rung, and per-channel bf16 scales in the
  epilogue are a smaller one.
- **The decode launch.** 13.2 µs single launch against 11.3 back to back for 16 MB: the 2 µs
  that CUDA graphs or a persistent decoder kernel would take back.
