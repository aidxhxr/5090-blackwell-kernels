# SGEMM (fp32): the optimization ladder

`C[M,N] = A[M,K] · B[K,N]`, row-major, fp32 in/out, fp32 accumulate. Every rung accepts arbitrary
`M, N, K ≥ 1`. Source: `src/kernels/sgemm.cu` (rungs 0 to 5, CUDA cores) and
`src/kernels/sgemm_tf32.cu` (variants 6 and 7, the tensor cores in TF32 and 3xTF32, a
different precision contract and opt-in); bench: `src/bench/bench_sgemm.cu`.

The point of this ladder is not to beat cuBLAS. It is to show, one change at a time, *which*
bottleneck each classic GEMM optimization removes, and to measure it on the RTX 5090 (primary
target) and later on the GB10.

## Ceiling

| | RTX 5090 | GB10 |
|---|---|---|
| FP32 lanes | 170 SMs × 128 | 48 SMs × 128 at ≈ 2.42 GHz |
| fp32 peak (CUDA cores, no tensor cores) | **123.4 TFLOPS measured** (`bench_peak`, FMA chains at 2,976 MHz; the 104.8 TFLOPS spec figure assumes 2.41 GHz) | **≈ 31 TFLOPS** |
| tf32 peak (tensor cores, `mma.sync.m16n8k8`, variants 6 and 7 only) | **129.4 TFLOPS measured** at 2,977 MHz, half the bf16 rate | not measured |
| memory | 1,792 GB/s GDDR7 | 273 GB/s LPDDR5X |
| fp32 ridge point | 123.4e12 / 1,792e9 ≈ **69 FLOP/byte** | 31e12 / 273e9 ≈ **114 FLOP/byte** |

A GEMM does `2MNK` FLOPs over `4(MK + KN + MN)` bytes of unique traffic, at 4096³ that is
≈ 683 FLOP/byte, comfortably compute-bound on both *if* the kernel reuses data well enough. Naive
kernels do not, and that is what the rungs fix.

The ceiling that actually binds on sm_120 is not DRAM but **shared-memory bandwidth**: an SM
does 128 FMAs per clock and reads 128 B per clock from shared memory. A thread computing an
8×8 micro-tile loads 8 + 8 floats = 64 B per `k` for 64 FMAs, exactly 1 B per FMA, so the
FMA and LDS pipes have to be perfectly overlapped just to reach peak, and in practice neither
gets there. cuBLAS SGEMM sits at the same wall: 67 TFLOPS is 54% of the measured 123.4.

## The metric

```
% of cuBLAS = cuBLAS median time / our median time × 100
```

cuBLAS is run with `CUBLAS_DEFAULT_MATH` (no TF32) so both sides do real fp32 FMAs. Both sides are
timed identically (10 warmup, median of 100, CUDA events on the same stream).

## Rung 0: naive (one thread per output)

Each thread walks a full row of A and a full column of B from global memory. Per FMA it loads two
floats → **0.25 FLOP/byte** from L1/L2's point of view. Reads of `B[k][col]` across a warp are
coalesced, reads of `A[row][k]` are broadcast, but every value is re-fetched `N` (resp. `M`) times.
Pure latency-bound. (The bench skips it at 4096³ and above; at 2048³ it is 7.6 TFLOPS.)

## Rung 1: shared-memory tiling (32×32×32)

A block loads a 32×32 tile of A and B into shared memory once, then each thread does 32 FMAs from
smem per tile. Global traffic drops by 32× per operand. Arithmetic intensity vs global memory:
`2·32·32·32 / (4·(32·32 + 32·32))` = **8 FLOP/byte**. Still far from the ridge; each FMA also needs
two `LDS` instructions, so the kernel is now bound by shared-memory instruction throughput.
Measured: 9.3 TFLOPS at 4096³, 14% of cuBLAS.

## Rung 2: register tiling (128×128×8, 8×8 micro-tile per thread)

The key idea: **reuse operands in registers**. A thread loads 8 values of A and 8 values of B from
smem per `k`, then does 64 FMAs. That is 64 FMAs per 16 `LDS`, 4 FMAs per smem load, versus 0.5 in
rung 1. Global-memory intensity becomes `2·128·128·8 / (4·(128·8 + 8·128))` = **32 FLOP/byte** and
the smem-instruction bottleneck is gone. Measured: 51.0 TFLOPS at 4096³, 76% of cuBLAS, the
single biggest step on the ladder, 5.5× over rung 1.

Implementation notes:

* **Layout.** 256 threads → a 16×16 grid of 8×8 micro-tiles. Each thread's 8 rows are split
  `{ty·4+i}` and `{64+ty·4+i}` (same for columns). This makes a warp's fragment reads of `Bs[k][·]`
  contiguous 128-bit accesses (16 threads × 16 B = 256 B, conflict-free per 8-thread phase) and
  makes the epilogue two `float4` stores per row instead of eight scalars.
* **A stored transposed** (`As[k][m]`). The inner loop then reads 4 consecutive `m` for a fixed `k`
  as one `float4`. Cost: the tile fill does 4 scalar smem stores per thread with a mild 2-way
  bank conflict (threads with adjacent `k0` land 4·128 floats apart, same bank). It is paid once
  per tile and is dwarfed by the 512 FMAs the tile buys.
* **Registers.** 64 accumulators + 16 fragment registers + addressing ≈ 100 registers/thread. At
  256 threads/block that caps occupancy at 2 blocks/SM (65 536 regs / (256×~110)), i.e. 16 warps ,
  plenty for a kernel whose loop is FMA-dense.
* **Edges.** Any chunk that is not fully in-bounds *and* 16-byte aligned falls back to guarded
  scalar loads with zero fill; stores are guarded the same way. Interior tiles take the vector
  path, so odd shapes cost only a partial-tile penalty.

## Rung 3: cp.async double buffering

Rung 2 serialises "load tile t+1" and "compute tile t": while a tile is being fetched the FMA units
idle. `cp.async` (Ampere+, available on sm_120 and sm_121) copies global → shared *without* passing through
registers, so the next tile's copy can be in flight while the current tile is consumed. Two smem
stages; the loop is

```
issue(t+1 → stage (t+1)&1); commit; wait_group 1; __syncthreads();
compute(stage t&1);            __syncthreads();
```

`cp.async` cannot transpose, so the A tile is stored **untransposed** in this rung (`As[m][k]`).
Fragment reads become 8 scalar `LDS` at stride `BK` instead of two `float4`. Within a warp the two
thread-rows (`ty = 0, 1`) read rows `r` and `r + 4`; with an unpadded 8-float stride those sit
`32` floats apart, the same bank, a 2-way conflict on every fragment load. Padding the row to
12 floats (48 B, still 16-byte aligned for `cp.async`) moves `r + 4` to bank `+16` and removes the
conflict. The trade is 8 `LDS.32` vs 2 `LDS.128` per `k` for A in exchange for full load/compute
overlap.

Shared memory: 2 stages × (128×12 + 8×128) × 4 B = 20.5 KB (static, well under 48 KB).

On the RTX 5090 this rung **loses** to rung 2 at 4096³ (49.0 vs 51.0 TFLOPS) and only wins at
1024³. The overlap it buys is worth little on a 1,792 GB/s bus with a 96 MB L2 (the tile fetch
is short), and the 8 scalar `LDS` per `k` are exactly the wrong direction for a kernel bound by
shared-memory bandwidth. Nsight: 130 registers and a shared-memory block limit of 1, so one
8-warp block per SM (16.7% achieved occupancy), 52.6% SM throughput.

## Rung 4: register-prefetch double buffering, transposed A, BK = 16

Rung 3's problem was that `cp.async` cannot transpose. Prefetching the next tile into
*registers* can: each thread issues its global loads for tile *t+1* before the FMAs of tile *t*
(so they are in flight across the whole compute phase, which is what the overlap needs), and
stores them to shared memory after the FMAs, transposing A on the way. That gives rung 2's inner
loop (4 × `LDS.128` per `k`) with rung 3's latency hiding, at the cost of 16 prefetch registers.
Two other changes:

* **`BK = 16`** halves the number of `__syncthreads` per FLOP (two barriers per tile step: one
  before the tile is overwritten, one after). Shared memory is a single 16 KB stage.
* **XOR swizzle of the transposed A tile.** With `BK = 16` four neighbouring threads store the
  four `k` chunks of one row, and `As[k][m]` rows are 128 floats = 4 × 32 banks, so all four
  stores hit the same bank: a 4-way conflict on every transposed store, 50 M conflicts over
  489 M shared wavefronts in the first profile. Swizzling `m ^= (k/4)·8` puts them in four
  different banks; the XOR operand is a multiple of 8, so the `float4` fragment reads stay
  contiguous and aligned, and `k` is a compile-time constant in the unrolled loop so the XOR
  folds into the address.
* **`__launch_bounds__(256, 2)`.** Left alone the compiler used 153 registers, which fits one
  block per SM: 8 warps, 16.7% achieved occupancy, issue slot active 68% of cycles, FMA pipe
  64%. Capping at 128 registers (no spills) puts two blocks on an SM; the full profile then
  shows 32% occupancy, FMA pipe 68.4%, IPC 2.85, 70% SM throughput.

Measured: 59.3 TFLOPS at 4096³, **88.5% of cuBLAS**, with the tail split-K described under
rung 5 (without it: 54.0 / 80%).

## Rung 5: 256×128 tile, 16×8 micro-tiles

The remaining lever against the shared-memory wall is bytes of smem per FMA. A 16×8 micro-tile
reads 16 + 8 floats = 96 B per `k` for 128 FMAs: **0.75 B per FMA** instead of 1. The block tile
grows to 256×128 (256 threads, 16 × 16 grid of 16×8 micro-tiles; the row groups become four
64-row quarters instead of two halves), the A stage is 256×16 floats and the B stage 16×128, and
the prefetch is 4 + 2 `float4` per thread. Accumulators alone are 128 registers; the kernel
takes 245 with `__launch_bounds__(256, 1)`, no spills, one block per SM. That is 8 warps, but
each of them has 128 independent FMAs per `k` to issue, and the profile says that is enough.

Measured: 63.3 TFLOPS at 2048³ (90.6% of cuBLAS), 56.4 at 4096³ (84%), 56.7–57.3 on the two
Llama MLP shapes (85%). The 4096³ number is *below* rung 4's, for the reason in the next
section; at 2048³ (128 tiles, under one wave) the tile pays off in full.

### Wave quantization and the split-K tail (rungs 4 and 5)

The 5090 has 170 SMs. Rung 4 runs 2 blocks per SM, so 340 tiles are resident at once; a 4096³
GEMM has 32 × 32 = 1,024 tiles = **3.01 waves**, and the 4 tiles of the fourth wave run alone
for a full tile's time: 4 waves of time for 3.01 of work, 75% efficiency. Rung 5 has 16 × 32 =
512 tiles on 170 slots: also 3.01 waves, 2 leftover tiles. 8192-class shapes see 12.05 waves
paid as 13.

Both rungs therefore split the tiles of the last partial wave along K across the idle blocks:
`tail = tiles mod resident`, `split = min(K-tiles, resident / tail)` (85 at 4096³ for both
rungs, 10 for 4096×11008×4096), each slice does `[s·KT/split, (s+1)·KT/split)` of the K loop,
and the extra wave lasts 1/`split` of a tile. For fp32 output no workspace is needed: the tail
tiles of C are zeroed with one `cudaMemset2DAsync` each on the stream, and every slice
`atomicAdd`s its partial sums into C (the adds are performed at L2, so their order does not
matter). Full-wave tiles store directly and are unaffected. `resident` comes from
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` × SM count, computed once per kernel. Those
tail tiles are not bitwise reproducible run to run; the bench's cuBLAS comparison (`1e-3`
relative) never sees the difference.

The fix took rung 4 from 49.6 to 60.6 TFLOPS at 4096³ in the tuning run, but rung 5 only from
55.0 to 56.9: with 245 registers and one block per SM there is no second block to soak up the
issue slots a straggling slice leaves idle, and 4096³ at one block per SM is more sensitive to
per-block latency than the 2048³ shape. I have not profiled the tail phase itself yet.

## Variants 6 and 7: tensor cores (TF32 and 3xTF32)

Everything above runs on the CUDA cores and stops at the shared-memory wall. The tensor
cores take fp32 words too, as TF32: the `mma.sync.m16n8k8.row.col.f32.tf32.tf32.f32`
instruction (sm_80 and up, present on sm_120) reads fp32 registers, ignores the low 13
mantissa bits of each operand and accumulates in fp32. That is a different precision
contract, the one cuBLAS runs under `CUBLAS_TF32_TENSOR_OP_MATH` and torch under
`allow_tf32`, so these two rungs are opt-in: `variant = -1` stays on rung 5, the bench
times both cuBLAS modes next to them, and the results scripts file their rows under the
dtypes `tf32` and `3xtf32`. Source: `src/kernels/sgemm_tf32.cu`.

| | rung 5 (CUDA cores) | variant 6 (TF32) | variant 7 (3xTF32) |
|---|---|---|---|
| operand precision | 24 significant bits | 11 (2^-11 relative per operand) | 22 (2^-22) |
| tensor-core passes per product | none | 1 | 3 |
| roof (`bench_peak`) | 123.4 TFLOPS fp32 FMA | 129.4 TFLOPS tf32 mma | 129.4 / 3 = 43.1 TFLOPS |
| max abs error vs cuBLAS fp32, 4096³ | 3.5e-4 | 3.2e-2 (cuBLAS TF32: 3.2e-2) | 3.6e-3 |
| 4096³, unlocked | 56 to 65 TFLOPS (the limiter decides) | 107.0 | 38.1 |

### The instruction and its roof

`m16n8k8` is 2,048 FLOP per instruction, half of the bf16 `m16n8k16`, and the RTX 5090 issues
it at half the bf16 rate on top: `bench_peak` (`peak_tf32_mma`, register-resident operands,
eight independent accumulator chains per warp) measures **129.4 TFLOPS at 2,977 MHz**,
exactly half of the 258.8 TFLOPS bf16 row and 4.9% above the 123.4 TFLOPS fp32 FMA peak.
So the tensor cores are not a 3x ceiling for fp32 work on this card, as they are on the
datacenter parts; they are a 5% higher ceiling that the ladder can actually reach, because
the mma does not go through the shared-memory wall: a warp reads 3 KB of fragments per 16
mmas (32 KFLOP), 24 B per clock per SM at the tensor rate against 128 B per clock available,
where the 8x8 SIMT micro-tile needed 1 B per FMA. And the mma keeps the clock up: the
sustained fp32 FMA kernels trip the limiter at 1.9 to 2.2 GHz ("The 11008 shapes" above);
a tensor kernel gets 2.7 to 2.9 GHz at the same 600 W.

### Fragments for 32-bit operands

The fragment layout of `m16n8k8` with 32-bit types is one element per register (PTX ISA,
"Matrix Fragments for mma.m16n8k8", .tf32): with g = lane / 4 and c = lane % 4,

```
a0 = A[g][c]      a1 = A[g+8][c]      a2 = A[g][c+4]      a3 = A[g+8][c+4]
b0 = B[c][g]      b1 = B[c+4][g]
d0, d1 = C[g][2c], C[g][2c+1]         d2, d3 = C[g+8][2c], C[g+8][2c+1]
```

`ldmatrix` moves 8x8 matrices of 16-bit elements, 16 bytes per row. Read a 16-byte row as
four fp32 and an 8x8 b16 matrix is 8 rows x 4 tf32; lane l receives the 32-bit word at (row
l/4, word l%4) = element (g, c), exactly a0's position. So `ldmatrix.x4` on the A tile with
matrix 0 = rows 0-7 at k chunk 0 (k 0..3), matrix 1 = rows 8-15 at chunk 0, matrix 2 = rows
0-7 at chunk 1 (k 4..7), matrix 3 = rows 8-15 at chunk 1 lands a0..a3 in order, one
instruction per m16 tile, and the lane address selectors are variant 3 of hgemm's, unchanged
(`lane & 15` for the row, `lane >> 4` for the chunk). The A tile is stored row-major with BK
= 16 fp32 = 64 B rows and the 64-byte XOR swizzle of the bf16 BK = 32 tile (chunk ^= (row/2)
% 4), which is what keeps the eight row addresses of each matrix in eight bank groups.

B cannot use `ldmatrix`: `.trans` transposes 16-bit halves, not 32-bit words, and no layout
that `cp.async` can write (16-byte chunks moved as they are) puts B[c][g] and B[c+4][g] in
one vector. So B is read with plain 32-bit-element loads, and the trick is *which* columns
the warp's four n8 tiles own. The mma does not care what global column its label g is; if
n-tile j, column g of a warp's 32-wide strip is declared to hold global column n0 + 4g + j,
then lane (g, c) reads the `float4` at Bs[c][n0 + 4g .. 4g + 3] and gets b0 of all four
tiles in one `LDS.128`, and the `float4` at row c + 4 gives the four b1. Two 128-bit loads
per k8 step instead of eight 32-bit ones. The B tile is row-major (k rows of 512 B) with
chunk ^= 2 (k % 8): a quarter-warp phase of an `LDS.128` is lanes g = 2q, 2q+1 with c = 0..3,
i.e. two neighbouring chunks at four consecutive k rows, and XOR-ing with {0, 2, 4, 6} sends
those eight accesses to eight bank groups. The permutation comes out in the epilogue for
free: d0 of tile j is column 2c, global n0 + 8c + j, so gathering d0 across the four tiles
gives four consecutive columns and the epilogue is two `float4` stores per row per lane, where
the bf16 tile stores pairs.

### Rounding

A tf32 operand is whatever the register holds with 13 bits ignored. Feeding raw fp32 is
truncation: a relative error of up to 2^-10 per operand, always toward zero, so a dot
product of K terms carries a bias of about 2^-11 of its own value. Rounding to the nearest
tf32 first halves the bound and removes the bias, and it is what cuBLAS does (its TF32 result
has the same 3.2e-2 max error as this kernel's, not the 0.1 that truncation gives at 4096³).
The fragments are rounded as they are loaded, 24 conversions per 16 mmas per warp. Which
instruction does the rounding matters more than I expected: `cvt.rna.tf32.f32` (ties away,
the sm_80 form) has no SASS instruction on sm_120 and expands to three (an FSETP for the
NaN case, an IMAD of 0x1000 and a LOP3 mask), 144 instructions per 32 mmas; `cvt.rn.tf32.f32`
(nearest even, sm_90 and up) is one `F2FP.TF32.F32`. The kernel does the same mmas either way;
what changed is the clock (next section).

### 3xTF32: the split

Variant 7 recovers fp32-class operands with three mmas per product. Each fp32 x is split as

```
big   = tf32(x)             |x - big| <= 2^-11 |x|, and x - big is exact in fp32
small = tf32(x - big)       |x - big - small| <= 2^-11 |x - big| <= 2^-22 |x|
```

so x = big + small + e with |e| <= 2^-22 |x|, and

```
a b = Ab Bb + Ab Bs + As Bb + As Bs + (terms in e)
```

The kernel accumulates the first three (small x big, big x small, then big x big, so the
corrections are not swallowed by a partial sum 2^24 times larger); the dropped As Bs is at
most 2^-22 |ab| and the e terms 2^-22 |ab| each, about 2^-20.4 relative per product against
2^-24 for an fp32 FMA. Both are below the error of accumulating K products in fp32. What is
left is the accumulator: the tensor core's fp32 adder is not an IEEE one (it aligns the
eight products to the largest exponent and truncates), and the measured error grows linearly
with K, 3.6e-3 max abs at 4096³ and 1.1e-2 at 8192³ (3.2e-5 and 6.6e-5 of max |C|), against
3.5e-4 for rung 5 and 3.1e-2 for TF32. It passes the fp32 rows' tolerance with a 20x margin,
and it is a tenth of the way from TF32 to fp32 on a log scale, not fp32 bit for bit.

Three mmas per product puts its roof at a third of the tf32 peak, 43.1 TFLOPS; it measures
38.1 at 4096³ and 38.5 at 8192³ in the 100-iteration sweep (40.1 in 20-iteration runs, before
the clock settles), 88 to 93% of that roof and 57 to 60% of cuBLAS fp32 SGEMM. On this card, where
tf32 runs at half the bf16 rate, 3xTF32 is a precision option, not a speed one: rung 5 is
faster. The bf16 analogue (three bf16 mmas at 258.7 TFLOPS, 16-bit operands, 2^-16 per
product) would sit at 86 TFLOPS, and is the version worth building if fp32-class accuracy
at tensor-core speed is what is wanted.

### Tolerances

The bench checks every variant against cuBLAS fp32 (`CUBLAS_DEFAULT_MATH`). The fp32 rungs
and 3xTF32 keep the existing bound, 2e-5 K + 1e-3 (summation order). TF32 gets 1% of
max |C|: each operand carries up to 2^-11 relative error, a product up to 2^-10 |ab|, and a
sum of K products up to K 2^-10 max|A| max|B| in the worst case (4 at K = 4096 for inputs in
[-1, 1]); the rounding errors are independent, so the typical error is sqrt(K) 2^-11 rms(ab),
1e-2 at K = 4096 against a max |C| of 113, and the measured maximum is 3.15e-2, for this
kernel and for cuBLAS TF32 alike. The 1% check is 35 times the measured error and a hundredth
of the worst case; a wrong k or a dropped chunk misses it by orders of magnitude. The pytest
parity tests use the same reasoning against torch (`fp32_precision = "tf32"` / `"ieee"`),
and additionally check that variant 6 agrees with torch's TF32 result to fp32 summation
noise, i.e. the same rounding contract, and that variant 7 is at least ten times closer to
fp32 than TF32 is.

### The tile, and what was tried

The block tile is hgemm variant 3's with 32-bit elements: 128x128x16, 8 warps as 2 x 4,
64x32 warp tiles (4 x 4 m16n8 accumulators, 64 registers), a 3-stage `cp.async` ring of 16 KB
stages (48 KB, two blocks per SM at 118 registers), split-K over the tiles of the last
partial wave with fp32 atomics into a zeroed C as rungs 4 and 5 do, and a 64x128 tile when
the 128-row one cannot fill a wave. `cp.async` zero-fills rows past M and columns past N,
so any M, N, K works; a problem whose rows are not 16-byte multiples (N or K not a
multiple of 4) takes a guarded register fill instead. Two things are new relative to the
sgemm rungs above:

* **The tail tiles are zeroed by one kernel launch**, not one `cudaMemset2DAsync` each. At
  1024³ the 64x128 tile gives 128 tiles on 340 slots, every tile is a tail tile split two
  ways, and 128 memsets cost 0.16 ms of launch overhead on a 0.05 ms GEMM (0.214 ms before,
  0.053 after). Rungs 4 and 5 pay the same at their small shapes; it is why their 1024³ rows
  read 47% of cuBLAS.
* **Tiles are rastered in bands of 16 tile rows.** At 8192³ an fp32 operand is 256 MB; a
  row-major wave of 340 tiles touches 5.3 tile rows of A (21 MB) and all of B, and re-reads
  B from DRAM every wave, 3.3 GB per GEMM. A 16-row band makes a wave a 16 x 21 rectangle
  (64 + 84 MB). The sweep at 8192³: bands of 1 / 4 / 8 / 16 / 32 / 64 rows give 10.48 / 10.35 /
  10.35 / 10.35 / 10.46 / 10.53 ms. Worth 1.3%; Nsight has both orders at 92% L2 hit rate,
  which is the number that says the row-major order was not far off on a 96 MB L2.

The sweep, TF32, median of 30, unlocked (ms and TFLOPS at 4096³ / 8192³):

| tile | warps | BK | stages | smem | blocks/SM | regs | 4096³ | 8192³ |
|---|---|---|---|---|---|---|---|---|
| **128x128** | **2x4** | **16** | **3** | **48 KB** | **2** | **118** | **1.281 / 107.3** | **10.00 / 110.0** |
| 128x128 | 4x2 | 16 | 3 | 48 KB | 2 | 124 | 1.265 / 108.6 | 10.00 / 109.9 |
| 256x128 | 4x2 | 16 | 3 | 72 KB | 1 | 190 | 1.334 / 103.0 | 10.04 / 109.5 |
| 256x128 | 2x4 | 16 | 3 | 72 KB | 1 | 170 | 1.363 / 100.9 | 10.54 / 104.4 |
| 256x128 | 4x2 | 32 | 2 | 96 KB | 1 | 194 | 1.349 / 101.9 | 10.28 / 107.0 |
| 128x128 | 2x4 | 32 | 2 | 64 KB | 1 | 123 | 1.354 / 101.5 | 10.90 / 100.9 |
| 128x128 | 2x4 | 16 | 4 | 64 KB | 1 | 119 | 1.423 / 96.6 | 11.45 / 96.0 |
| 128x128, register prefetch, rounding once per element | 2x4 | 16 | 2 | 32 KB | 2 | 128 (capped) | 1.297 / 105.9 | 10.26 / 107.2 |

(The first three rows and the last were measured with the one-instruction `cvt.rn`, the
other four with the three-instruction `cvt.rna` of the first build; the ordering did not
change between the two.)

Two of those rows are the interesting negatives. cuBLAS's TF32 kernel on this card is
`cutlass_80_tensorop_s1688gemm_256x128_16x3_nn_align4`: a 256x128 tile with 64x64 warp tiles,
BK = 16, three stages, 74 KB, 217 registers, one block per SM. Its warp tile reads a third
less shared memory per mma and converts a third fewer operands (1 per mma instead of 1.5), so
I expected the 256x128 / 4x2 row to win on energy. It does not: with eight warps per SM and
no fragment double-buffering, the latency the second block used to cover is exposed, and
the two effects cancel. The register-prefetch row is rung 4's idea on this tile, loading
the next slab into registers and rounding it once on the way into shared memory (16
conversions per thread per slab instead of 48, and `ldmatrix` feeding the mma directly as in
the bf16 tile); it loses 2 to 3% because one slab of prefetch distance behind a block barrier
exposes more L2 latency than the 100 fewer instructions save.

### Where the time goes

Nsight Compute at its fixed clock (2.43 to 2.48 GHz), 4096³: tensor pipe 88.1% of peak
sustained active, issue slots busy 25%, 3.75 warps per scheduler with 0.42 eligible, L2 hit
95.4%, DRAM 7.6%, "waiting for the execution pipe" 62% of stall cycles. That is a kernel
bound by the tensor pipe with the block barrier per stage as its overhead, the same 88 to
92% hgemm variant 3 showed before TMA took it to 99%. At 8192³ ours is 10.99 ms against
10.73 for cuBLAS TF32 at the same fixed clock, 2.4% apart, both with a 92% L2 hit rate; ours
moves 259 GB/s against their 217 (two blocks per SM double the L2 working set of a wave).

Unlocked the gap is 6%, and `nvidia-smi` sampled at 250 ms through a 300-iteration run
at 8192³ says why:

| phase | SM clock | board power | limiter |
|---|---|---|---|
| cuBLAS SGEMM, fp32 SIMT | 2,197 MHz | 556 W | SW power cap |
| cuBLAS SGEMM, TF32 | 2,895 to 2,910 MHz | 600 W | SW power cap |
| variant 6, `cvt.rna` build | 2,670 MHz | 600 W | SW power cap |
| variant 6, `cvt.rn` build (shipped) | 2,745 MHz | 600 W | SW power cap |

All three tensor-core rows sit at the 600 W cap; cuBLAS gets 8.6% more clock than the first
build and 5.5% more than the shipped one, which is the 9% and the 6% of time. The instruction
stream is the difference: the `cvt.rna` loop body was 296 SASS instructions per 32 HMMA (48
FSETP, 62 IMAD, 65 MOV); with `cvt.rn` it is 250 (48 F2FP, 66 MOV). The MOVs are the B fragments: b0 comes from the `LDS.128` of k row c and b1
from the one of row c + 4, and HMMA wants the pair in adjacent registers, so two moves per
mma that no `cp.async`-writable layout removes. Swapping the rounding instruction took 8192³
from 105.8 to 110.0 TFLOPS and 4096³ from 105.2 to 107.3 without changing a single mma,
which is the clearest measurement in this repo of the limiter charging for ALU instructions.

### Measured (RTX 5090, default sweep, median of 100, unlocked)

TF32, variant 6, against both cuBLAS math modes:

| shape | cuBLAS fp32 ms | cuBLAS TF32 ms / TFLOPS | v6 ms | v6 TFLOPS | % of tf32 peak | % of cuBLAS fp32 | % of cuBLAS TF32 | max abs err (v6 / cuBLAS TF32) |
|---|---|---|---|---|---|---|---|---|
| 512³ | 0.0146 | 0.0114 / 23.6 | 0.0377 | 7.1 | 5.5 | 38.7 | 30.2 | 9.4e-3 / 9.4e-3 |
| 1024³ | 0.0450 | 0.0296 / 72.5 | 0.0513 | 41.8 | 32.3 | 87.7 | 57.7 | 1.41e-2 / 1.41e-2 |
| 2048³ | 0.2449 | 0.2009 / 85.5 | 0.2181 | 78.8 | 60.9 | 112.3 | 92.1 | 2.18e-2 / 2.18e-2 |
| 4096³ | 2.0537 | 1.3032 / 105.5 | **1.2848** | **107.0** | 82.7 | 159.8 | **101.4** | 3.15e-2 / 3.15e-2 |
| 8192³ | 16.4233 | 9.4470 / 116.4 | **10.0294** | **109.6** | 84.7 | 163.8 | **94.2** | 4.75e-2 / 4.75e-2 |
| 4096×4096×11008 | 5.4995 | 3.4740 / 106.3 | **3.4177** | **108.1** | 83.5 | 160.9 | **101.6** | 5.12e-2 / 5.12e-2 |
| 4096×11008×4096 | 5.4411 | 3.2947 / 112.1 | **3.3357** | **110.7** | 85.6 | 163.1 | **98.8** | 3.18e-2 / 3.18e-2 |

3xTF32, variant 7 (reference: cuBLAS fp32, the contract it claims):

| shape | cuBLAS fp32 ms | v7 ms | v7 TFLOPS | % of tf32 peak / 3 | % of cuBLAS fp32 | max abs err (v7 / rung 5 where measured) |
|---|---|---|---|---|---|---|
| 512³ | 0.0146 | 0.0392 | 6.9 | 15.9 | 37.3 | 1.9e-5 / n/a |
| 1024³ | 0.0451 | 0.0924 | 23.3 | 53.9 | 48.8 | 2.1e-4 / n/a |
| 2048³ | 0.2460 | 0.5930 | 29.0 | 67.2 | 41.5 | 1.3e-3 / n/a |
| 4096³ | 2.0702 | 3.6094 | 38.1 | 88.3 | 57.4 | 3.6e-3 / 3.5e-4 |
| 8192³ | 16.4234 | 28.5668 | 38.5 | 89.3 | 57.5 | 1.1e-2 / n/a |
| 4096×4096×11008 | 5.5598 | 9.6150 | 38.4 | 89.1 | 57.8 | 1.7e-2 / n/a |
| 4096×11008×4096 | 5.4368 | 9.6376 | 38.3 | 88.9 | 56.4 | 3.7e-3 / n/a |

Against torch (`scripts/bench_torch.py --only sgemm`, `torch.matmul` with
`fp32_precision = "tf32"` for the TF32 row and `"ieee"` for the others):

| shape | v6 (TF32) ms | torch TF32 ms | v6 / torch | v7 (3xTF32) ms | torch fp32 ms | v7 / torch |
|---|---|---|---|---|---|---|
| 2048³ | 0.2198 | 0.1911 | 0.87× | 0.5989 | 0.2596 | 0.43× |
| 4096³ | 1.2903 | 1.3132 | 1.02× | 3.6107 | 2.1524 | 0.60× |
| 8192³ | 10.0650 | 9.4945 | 0.94× | 28.5742 | 16.4351 | 0.58× |
| 4096×4096×11008 | 3.4346 | 3.5064 | 1.02× | 9.6169 | 5.6293 | 0.59× |
| 4096×11008×4096 | 3.3489 | 3.3115 | 0.99× | 9.6421 | 5.5765 | 0.58× |

(512³ and 1024³ are 0.35× and 0.63× for TF32: the same under-one-wave and tail-split story as
the SIMT rungs, with the 64x128 tile.)

Targets and where they landed: TF32 at 4096³ and 8192³ is 83 to 85% of the 129.4 TFLOPS
peak measured at 2,977 MHz, and 90 to 92% of the peak at the 2,745 MHz the kernel actually
runs at under the power cap (129.4 x 2,745 / 2,977 = 119.3 TFLOPS); against cuBLAS TF32 it is
101% at 4096³ and 94% at 8192³, and 160% of cuBLAS fp32 on every large shape. 3xTF32 at 40 TFLOPS is 93% of
its own roof and 60% of the SIMT rung, not the 2x the tf32 rate would give on a card where
tf32 is not half rate.

### What remains here

* **TMA and a producer warp** (hgemm variants 5 and 6). The barrier per stage and the
  per-thread copy addressing are the 12% of the tensor pipe still idle at a fixed clock, and
  fewer instructions per mma is the other half of the 6% at 8192³. The A box is a 128x16
  fp32 tile with the 64-byte TMA swizzle, the same bits `swz_a` computes; B needs four 32-column
  boxes of 128 B rows with the 128-byte swizzle, and with that swizzle the column permutation
  has to change to n0 + 4 p(g) + j with p(g) = 4 (g % 2) + g / 2 to keep the `LDS.128` phases
  conflict-free. Everything else transfers.
* **A split accumulator for 3xTF32**: the big x big products in one accumulator and the two
  corrections in another, summed once at the end, so the tensor core's truncation applies to
  each at its own magnitude. 64 more registers, one block per SM.
* **bf16x3** as the fast fp32-class option, at the full-rate 258.7 TFLOPS instruction.

## What Nsight Compute shows

| Rung | Measured limiter (`--set full`, 4096³) |
|---|---|
| 0 | latency-bound, `lg_throttle` / long-scoreboard stalls |
| 1 | `MIO throttle` — shared-memory instruction issue |
| 2 | FMA pipe busy; 2 blocks/SM |
| 3 | 130 registers, smem-limited to 1 block/SM (16.7% occupancy), SM throughput 52.6% |
| 4 | 128 registers, 2 blocks/SM (32% occupancy), FMA pipe 68%, SM throughput 70%, IPC 2.85, dominant stall "waiting for the micro scheduler" |
| 5 | 245 registers, 1 block/SM, FMA-dense inner loop with 128 independent FMAs per `k` |

Profile a rung with `scripts/profile_ncu.sh build` (it runs rungs 3 and 4 at 4096³) or directly
with `ncu --set full ./build/bench_sgemm --variant=<v> --m=4096 --iters=1 --warmup=0`.

## RTX 5090 notes (measured)

The block and register tile sizes (128×128×8, 8×8) were reasoned for 48 SMs and 273 GB/s. They
are correct on the 5090 and rung 2 alone reaches 76% of cuBLAS with them; the last 12 points came
from the memory-system-specific rungs above.

* **The binding ceiling is shared memory, not DRAM.** sm_120 issues 128 FMAs and reads 128 B of
  shared memory per SM per clock; an 8×8 micro-tile needs 1 B per FMA. cuBLAS SGEMM stops at
  the same wall, 67 TFLOPS = 54% of the measured 123.4 TFLOPS fp32 peak, on every shape from
  2048³ up. Rung 5's 16×8 micro-tile is the only way past it without changing the numerics.
* **Rung 3 vs rung 2** was the open question and the answer is "rung 3 loses": the tile fetch
  is short on a 1,792 GB/s bus with a 96 MB L2 (94–96% L2 hit rate, 4–6% DRAM throughput on
  every register-tiled rung), so there is little to hide, and the 8 scalar `LDS` per `k` cost
  more than the overlap gains. Rung 4 keeps the overlap and drops the scalar loads.
* **The fp32 peak is a clock number.** 123.4 TFLOPS is 21,760 lanes × 2 FLOP × 2.84 GHz
  equivalent; the card boosts to ~2.98 GHz for the short FMA loop and sits at 2.72–2.78 GHz
  under a sustained GEMM at its 600 W limit. "% of peak" in the tables uses the measured 123.4.
* **Small shapes underfill 170 SMs.** 16 tiles at 512³ and 64 at 1024³ (32 for rung 5), so
  those rows are occupancy numbers, not kernel-quality numbers: 33% and 47% of cuBLAS at best.
  cuBLAS presumably switches to a smaller tile or split-K there.
* **The Llama MLP shapes read ~10% slower per FLOP than 4096³** (rung 4: 59.3 TFLOPS at
  4096³ vs 51.7–52.6 with N or K = 11008; rung 5 in the re-measurement below: 65.4 vs
  57.0–57.4), while cuBLAS is flat across them. My first guess was the 44 KB row stride and
  the TLB. It is neither: at a fixed clock the kernel runs at the same rate on every shape,
  and the difference is the SM clock the card's limiter allows. The measurements are in
  "The 11008 shapes" below.
* **Occupancy cap** confirmed: the register file is 64 K per SM, `launch__registers_per_thread`
  is 128 for rung 4 (2 blocks) and 245 for rung 5 (1 block).
* `min_ms` and the median agreed to within 1–2% on every row of the first run. That was luck,
  not a property of the kernel: on 4096³ rung 5 has a 1.80 ms minimum and a median anywhere
  between 1.83 and 2.42 depending on when the clock limiter reacts (next section).

### The 11008 shapes

The earlier note above blamed the 44 KB row stride and guessed at the TLB. That was wrong. I
went through the four candidates one at a time, and the loss is not in the kernel at all.

**1. Stride.** `4096×4096×K` and `4096×N×4096` for K, N from 4096 to 12288, rung 5, 20
iterations each. A stride effect would be a dip at 11008 and its neighbours. There is none:
the per-FLOP rate falls from ~75 TFLOPS to ~57 somewhere between K = 8192 and K = 10240 and
stays there, and 10944, 11008, 11072 and 12288 all sit on the same floor. A scratch copy of
the kernel with separate leading dimensions makes it explicit: 4096³ with `lda`, `ldb` and
`ldc` all padded to 11008 (the exact 44,032 B stride of the MLP shapes, same FLOPs, same
tiles) runs at 2.42 ms against 2.39 ms unpadded, and 4112 (a 16 KB + 64 B stride) at 2.38.
The stride is not the variable.

**2. The tail split.** Overriding `split` in the scratch copy at 4096³: 1 (no tail split)
2.45 ms, 2 → 2.43, 8 → 2.38, 32 → 2.41, the launcher's 85 → 2.39. The tail is worth 2–3%
and is the same on every shape; it is not the 10%.

**3. Tile order.** Groups of G tile rows walked column-major, G = 2, 4, 8, 16, at all three
shapes: every result within ±1% of the row-major order (2.36–2.42 ms at 4096³, 6.47–6.50 at
4096×4096×11008, 6.45–6.46 at 4096×11008×4096). Nsight says why there was nothing to gain:
L2 hit rate 90% at 4096³ and at 4096×4096×11008, 80% at 4096×11008×4096, DRAM at 9%, 9% and
19% of peak, `long_scoreboard` stalls under 1% of issue slots on all three. The global loads
are not the limiter, so a better order for them cannot be the fix.

**4. Nsight at a fixed clock.** This is the measurement that settles it. Nsight Compute pins
the SM clock (2.43–2.48 GHz here) for the launch it profiles, and at that clock every shape
runs at the same rate:

| shape | ms | TFLOPS (2.43–2.48 GHz) | FMA pipe active | L2 hit | DRAM |
|---|---|---|---|---|---|
| 4096³ | 1.92 | 71.6 | 73.0% | 90.3% | 9.1% |
| 4096×4096×8192 | 3.73 | 73.7 | 73.7% | 90.5% | 9.3% |
| 4096×4096×11008 | 4.92 | 75.1 | 74.9% | 90.6% | 9.3% |
| 4096×11008×4096 | 5.10 | 72.4 | 74.0% | 80.4% | 19.2% |

Eight consecutive profiled launches of 4096³ agree to 0.5%. The kernel does the same work
per clock on every shape, and 73–75% of the FMA pipe is the whole story on the instruction
side. So the shape dependence in the results table is a clock dependence.

**The clock.** `nvidia-smi` sampled during the bench, unlocked, on 4096×4096×11008:

| phase | SM clock | board power | throttle reason |
|---|---|---|---|
| cuBLAS timing loop | 2,317–2,370 MHz | 580 W | SW power cap |
| rung 5 timing loop | 1,890 MHz | 455 W | SW power cap |

The limiter holds rung 5 at 1.89 GHz while the board draws 145 W less than its 600 W limit,
and gives cuBLAS 2.32 GHz at the limit. It is not reacting to measured power. Locking the
clock at 1.9 GHz (`nvidia-smi -lgc`, under the GPU lock, reset afterwards) reproduces the
"slow" numbers exactly and puts both kernels on the same footing:

| shape, 1.9 GHz locked | cuBLAS ms | rung 5 ms | rung 5 / cuBLAS | rung 5 unlocked |
|---|---|---|---|---|
| 4096³ | 2.594 | 2.493 | 104% | 2.39–2.42 (slow mode) |
| 4096×4096×8192 | 4.979 | 4.900 | 102% | 4.74–4.87 |
| 4096×4096×11008 | 6.874 | 6.504 | 106% | 6.48–6.51 |
| 4096×11008×4096 | 6.552 | 6.602 | 99% | 6.38–6.45 |
| 8192³ | 18.99 | 19.58 | 97% | |
| 2048³ | 0.354 | 0.413 | 86% | 0.27–0.33 |

The cuBLAS kernel behind those rows is `cutlass_80_simt_sgemm_256x128_8x4_nn_align1`: the
same 256×128 block tile as rung 5, 256 threads, 210 registers, one block per SM, 49 KB of
pipelined shared memory, launched as 3,072 blocks (6-way split-K) on 4096×4096×11008.
Nsight puts it at 71.9% FMA pipe active against rung 5's 74.9%, with 10% more LSU
instructions and 32% more shared-memory wavefronts for the same 5.8 G FMA instructions.
By every activity counter it is the same kernel or a slightly busier one.

At an equal clock rung 5 is faster than cuBLAS on the three large shapes, and its unlocked
time on the 11008 shapes is its 1.9 GHz time: the limiter sits at 1.89 GHz for the entire
timing loop there. At the same locked clock the two kernels draw about the same board power
(rung 5 450 W, cuBLAS 440 W, for 2–6% more FLOP/s), so per joule rung 5 is the better
kernel; per second, under this limiter, it is not. Locking at 2.4 GHz does not help: the
limiter pulls rung 5 below it anyway (6.44 ms at 4096×4096×11008, the same number).

**Why 4096³ looked fine and 11008 looked bad.** The limiter takes a few hundred
milliseconds to settle, and it responds to how much of the chip is busy: at 2048³ (128 tiles
on 170 SMs) it lets rung 5 run at 2.4 GHz or more. A 4096³ timing loop is 20–100 launches of
2 ms, and the first part of it runs at 2.4–2.8 GHz before the clock drops to 1.9: `min_ms` 1.80,
median anywhere from 1.83 to 2.42 depending on when the drop lands. Three back-to-back runs
of the default bench gave medians of 2.39, 2.23 and 2.35 ms with a 1.80 min each time. The
11008 shapes run 6.5 ms per launch and are preceded by a longer cuBLAS loop, so the timing
window is always inside the 1.9 GHz state and the median is stable at 6.4–6.5. cuBLAS is at
2.32 GHz throughout on every shape and reads as flat. That is the whole 10%: the same kernel,
at 1.9 GHz instead of at whatever the limiter allowed 4096³ that day. The sentence earlier in
this section about `min_ms` and the median agreeing within 1–2% was true of that one run
and is not true in general for this kernel; for rung 5 on 4096³ they differ by 30%.

Two corrections to the table above follow. Rung 5's 4096³ row (2.4357 ms) is a slow-mode
median; the kernel does 1.92 ms at 2.43 GHz and 1.80 at the clocks it gets before the
limiter reacts, and it does not "trail rung 4" for any reason inside the kernel. And rung 4's
11008 rows (51.7–52.6 TFLOPS) are the same effect one rung down: at a locked 1.9 GHz rung 4
does 7.32 ms on 4096×4096×11008 against its 7.0–7.1 unlocked.

**hgemm does not show it.** The same K and N sweep on `bench_hgemm --variant=3`: the K sweep
is 99–101% of cuBLAS at every K including 11008 (221–224 TFLOPS); the N sweep moves with
the tile count, 97% at N = 8192 (6.02 waves), 95% at 11008 (8.09 waves) and 12288, but 88% at
10944 and 87% at 11072, which are not multiples of the 128-wide tile. No stride signature,
and cuBLAS's bf16 kernel gets faster at N = 11008 (231 vs 223 TFLOPS in this sweep, 239 vs
226 in the results table), which is where the 93.8% in its table comes from. The tensor-core kernel runs at 2.7–2.8 GHz under the same
limiter (see the hgemm note); whatever the limiter keys on, fp32 FMA at 73% of the pipe on
all 170 SMs trips it and `mma.sync` at 95% does not.

**Is there a kernel-side fix?** Not one that survives the limiter. Since it responds to
chip-wide activity, I tried the only lever that leaves: a persistent grid that keeps some SMs
idle on purpose (each block loops over tiles, `grid` blocks in all). On 4096×4096×11008,
170 / 160 / 150 blocks give 6.58 / 6.57 / 6.55 ms, 136 and 128 give 6.70 and 6.69, 112 and
96 give 7.89 and 8.34. Removing a quarter of the SMs costs only 2% of the time, so the clock
rose by roughly that quarter (the limiter holds the product of busy SMs and clock about
constant), but it never comes out ahead. Two more knobs in the same scratch copy, in case
the limiter keys on the blocks marching in lockstep: a block-dependent spin of up to 4 K,
20 K or 100 K cycles at kernel start (6.52 / 6.53 / 6.62 ms against 6.52 without), and
cuBLAS's own schedule of splitting every tile along K with atomics into a zeroed C, 2, 4 and
6 ways (6.53 / 6.49 / 6.48 ms). Neither moves it. The tile order, the split and the stride
were ruled out above; at a fixed clock the kernel is already past cuBLAS. So rung 5 stays as it is, and
the honest statement of its result is: at equal clock, 102–106% of cuBLAS SGEMM on the
large shapes; at the clock this card's limiter gives an fp32 FMA kernel that keeps 73% of
the pipe busy on all 170 SMs, 84–86% on the long shapes and a coin toss between 85% and 98%
on 4096³.

**Re-measured, default sweep, rung 5, 100 iterations, clocks unlocked** (the same run
conditions as the results table, one more time to show the spread):

| shape | cuBLAS ms | rung 5 ms (median / min) | TFLOPS | % of cuBLAS |
|---|---|---|---|---|
| 512³ | 0.0146 | 0.0442 / 0.0432 | 6.07 | 33.0 |
| 1024³ | 0.0443 | 0.0944 / 0.0849 | 22.74 | 46.9 |
| 2048³ | 0.2456 | 0.2733 / 0.2694 | 62.85 | 89.9 |
| 4096³ | 2.0678 | 2.1029 / 1.8120 | 65.36 | 98.3 |
| 4096×4096×11008 | 5.5559 | 6.4310 / 5.6459 | 57.44 | 86.4 |
| 4096×11008×4096 | 5.4594 | 6.4866 / 6.4700 | 56.94 | 84.2 |

The 4096³ row landed on the fast side of the coin this time (98.3%, against 84.2% in the
table above, same binary, same shape). Read the two large-shape rows, or better the
locked-clock table, for the kernel; read the 4096³ row for the limiter. Median and min are
now both in every JSON row, and for this kernel the gap between them is the clock, not
noise.

## Results (RTX 5090, sm_120, CUDA 13.2, driver 595.58)

From `results/sgemm.json`, median of 100 iterations, cuBLAS timed identically. A GB10 table is
added when the Spark has been benchmarked.

4096×4096×4096:

| Variant | ms | TFLOPS | % of cuBLAS |
|---|---|---|---|
| cuBLAS SGEMM | 2.0518 | 66.98 | 100 |
| 1 smem tiled | 14.7207 | 9.34 | 13.9 |
| 2 register tiled | 2.6926 | 51.04 | 76.2 |
| 3 cp.async double buffered | 2.8076 | 48.95 | 73.1 |
| 4 register prefetch, transposed A, BK=16, tail split-K | **2.3189** | **59.27** | **88.5** |
| 5 256×128 tile, 16×8 micro-tiles, tail split-K | 2.4357 | 56.43 | 84.2 |

All shapes, best rung per shape in bold (ms / TFLOPS / % of cuBLAS):

| shape (M×N×K) | cuBLAS ms / TFLOPS | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|
| 512³ | 0.0146 / 18.40 | 0.0585 / 4.59 / 25.0% | 0.0484 / 5.55 / 30.2% | 0.0545 / 4.92 / 26.8% | 0.0443 / 6.06 / 32.9% (v1 is best here: 0.0382 / 7.03 / 38.2%) |
| 1024³ | 0.0443 / 48.45 | 0.1118 / 19.21 / 39.6% | **0.0935 / 22.97 / 47.4%** | 0.1355 / 15.85 / 32.7% | 0.0950 / 22.60 / 46.6% |
| 2048³ | 0.2458 / 69.90 | 0.3228 / 53.22 / 76.1% | 0.3781 / 45.44 / 65.0% | 0.3082 / 55.75 / 79.8% | **0.2714 / 63.30 / 90.6%** |
| 4096³ | 2.0518 / 66.98 | 2.6926 / 51.04 / 76.2% | 2.8076 / 48.95 / 73.1% | **2.3189 / 59.27 / 88.5%** | 2.4357 / 56.43 / 84.2% |
| 4096×4096×11008 | 5.5698 / 66.32 | 7.4008 / 49.91 / 75.3% | 7.6101 / 48.54 / 73.2% | 7.1478 / 51.68 / 77.9% | **6.5172 / 56.68 / 85.5%** |
| 4096×11008×4096 | 5.4645 / 67.59 | 7.3249 / 50.43 / 74.6% | 7.2536 / 50.92 / 75.3% | 7.0252 / 52.58 / 77.8% | **6.4446 / 57.31 / 84.8%** |

Rungs 0 and 1 at the small shapes: v0 5.82 / 7.46 / 7.58 TFLOPS and v1 7.03 / 8.55 / 9.59 at
512³ / 1024³ / 2048³. The Python default (`variant=-1`) is rung 5; the tensor-core variants
6 and 7 have their own tables in "Variants 6 and 7" above (107 to 111 TFLOPS for TF32 on the
large shapes, 38 for 3xTF32) and the 8192³ shape is in the default sweep since they were
added. Against PyTorch eager
(`torch.matmul` in fp32, `results/torch_comparison.json`) that is 0.95× at 2048³, 0.90× at
4096³ and 0.86–0.87× on the MLP shapes; torch's cuBLAS call is a little faster than the bench's
`cublasSgemm` on the large shapes.

## What remains

* **Energy per FLOP is now the lever.** The clock limiter, not the instruction stream, sets
  rung 5's time on the large shapes ("The 11008 shapes" above), and it keys on activity, not
  on the 455 W the board actually draws. Anything that does the same FMAs with less
  switching (fewer `STS` in the transposed A fill, a wider `LDS` for B) is worth trying for
  the clock it might buy, and has to be judged unlocked, over a long enough loop for the
  limiter to settle.
* **Report a locked-clock row.** The bench could take a `--lgc` flag (needs root) or at least
  print `min_ms` next to the median, which it now does; a results table that mixes 1.9 and
  2.6 GHz runs of the same kernel is not a kernel table.
* **Small shapes**: a 64×64 tile or whole-problem split-K for 512³ / 1024³.
* **TF32 tensor cores** are variants 6 and 7 now ("Variants 6 and 7" above). On this card
  they are a 5% higher ceiling than the CUDA cores, not the 3x of the datacenter parts, and
  the win is that the mma neither hits the shared-memory wall nor trips the clock limiter:
  107 to 111 TFLOPS on the large shapes against rung 5's 56 to 65. It is a different
  precision contract (11-bit operands), so it stays opt-in and cuBLAS is timed in both
  math modes.
