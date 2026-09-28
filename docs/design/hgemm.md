# HGEMM: bf16 tensor-core GEMM

`C[M,N] = A[M,K] · B[K,N]`, row-major, bf16 inputs and outputs, fp32 accumulation.
Source: `src/kernels/hgemm.cu` (variants 0 to 3), `src/kernels/hgemm_streamk.cu` (variant 4),
`src/kernels/hgemm_tma.cu` (variant 5) and `src/kernels/hgemm_tma_sk.cu` (variant 6), with the
pieces they share in private headers: the `cp.async` tile of variants 3 and 4 in
`hgemm_tile.cuh`, the Stream-K schedule of variants 4 and 6 in `hgemm_streamk.cuh`, the TMA
tile of variants 5 and 6 in `hgemm_tma_tile.cuh`, and the fused epilogue of variants 4 and 6
(bias, activation, residual, the interleaved gate/up SwiGLU) in `hgemm_epilogue.cuh`. Bench:
`bench_hgemm` (validates every variant against cuBLAS, and the fused rows against the fused
math on cuBLAS's fp32 result).

## Why WMMA / mma.sync on these GPUs

Both targets are the consumer/workstation Blackwell lineage: the RTX 5090 is compute capability
12.0 (sm_120, primary target) and the GB10 is 12.1 (sm_121, secondary). They have
fifth-generation tensor cores driven by the classic `mma.sync` warp-level instruction, which is
what the WMMA C++ API compiles to. The datacenter Blackwell parts (sm_100, B200/GB200) add
`tcgen05` instructions and thread-block clusters; those do not exist on either machine. TMA
(`cp.async.bulk.tensor`) and mbarriers do exist on sm_120, and variant 5 uses them to feed the
same `mma.sync` k-loop. So `mma.sync` is not a compromise, it is the native path, and the same
source builds for both.

A `wmma::fragment` is a warp-distributed register tile. One `mma_sync` on
`fragment<..., 16, 16, 16, __nv_bfloat16, ...>` performs a 16×16×16 matrix multiply-accumulate
(8,192 FLOPs) per warp with fp32 accumulators. Each lane holds a compiler-defined slice of the
tile; `load_matrix_sync` / `store_matrix_sync` move whole tiles between memory and fragments
(pointer must be 32-byte aligned, `ldm` in elements and a multiple of 8 for bf16).

Variant 3 drops below WMMA to the PTX it compiles to, `mma.sync.m16n8k16` with `ldmatrix`, so
the fragment layout is explicit and the kernel controls every shared-memory access itself.

## The ladder

| variant | idea | what it fixes |
|---|---|---|
| 0 | one warp per 16×16 C tile, fragments loaded straight from global memory | baseline: correct use of tensor cores, zero data reuse |
| 1 | 128×128×32 block tile, 8 warps (2×4), each warp owns a 64×32 sub-tile (4×2 fragments), tile staged in shared memory with +8 padding | global traffic ÷ 8 vs v0 for the same FLOPs, no bank conflicts on fragment loads |
| 2 | v1 + two-stage `cp.async` pipeline | overlaps the global→shared copy of tile *k+1* with the tensor-core work on tile *k*; copies bypass registers |
| 3 | raw `mma.sync.m16n8k16` + `ldmatrix`, XOR-swizzled smem, 3-stage `cp.async` pipeline, register-direct epilogue, split-K on the last partial wave, tile picked per call (128×128, 64×128 or 64×64) with zero-filled rows past M | fragment loads in one instruction each, no padding bytes, DRAM latency covered two tiles ahead, no epilogue staging, the wave-quantization tail on 170 SMs, and small / decode shapes (any M % 16 == 0) that a 128-row tile could not fill the card with |
| 4 | the same tile on a persistent Stream-K schedule: a grid of resident blocks, grouped (L2-aware) tile order, equal (tile, k-step) ranges for problems up to a few waves, a tile queue plus geometrically shrinking K-passes for the rest, and a fixup that sums partials in K order through one fp32 slot per tile with no memset, no atomics and the same bits every run | shapes under one wave without dropping to a smaller tile (2048³), the last-wave quantization on long shapes, the per-SM speed spread that a static schedule exposes, DRAM re-reads of B once the operands exceed L2 (8192³: 664 to 165 GB/s), and the two memsets plus counters per call that cost the decode shapes 4 µs each |
| 5 | v3's 128×128 tile and k-loop fed by TMA (`cp.async.bulk.tensor`) through a warp-specialized mbarrier pipeline: one producer warp issues three box loads per stage, eight consumer warps run the tensor cores, no `__syncthreads` in the k-loop (`src/kernels/hgemm_tma.cu`) | the per-K-tile block barrier (8.7 of v3's 30 stall cycles per issue) and the copy address arithmetic; tensor pipe 92% → 99% active, 102 to 109% of cuBLAS on the large shapes. |
| 6 | v5's TMA mainloop driven by v4's Stream-K schedule: a persistent grid of 170 blocks, the producer lane owns the schedule and publishes each (tile, k-range) piece to the consumer warps through a two-deep ring in shared memory, stage counters run on across pieces, static ranges whenever A and B fit in L2, the queue with geometric K-passes otherwise, an L2-footprint raster group, the same deterministic chain fixup (`src/kernels/hgemm_tma_sk.cu`) | v5's row-major tile order (80% L2 hit at 8192³, 40% of DRAM peak) and its one-block-per-tile schedule (1.5 waves at 2048³); v4's per-piece prologue and its 8% slower `cp.async` mainloop. 224 TFLOPS at 2048³ (v4 207), 249 at 4096³ (v5 243), 243.5 at 8192³ (v5 236), 246 at 4096×11008×4096 (v5 239): 103 to 130% of cuBLAS on every large shape, the same bits every run. |

### Arithmetic intensity of the block tile

Per K-step the block loads an A slab of 128×32 and a B slab of 32×128 bf16 = 16 KB, and performs
2·128·128·32 = 1,048,576 FLOPs:

```
AI_block = 2·128·128·32 / ((128·32 + 32·128) · 2 B) = 64 FLOP/byte
```

The RTX 5090's measured bf16 ridge point is

```
258.7 TFLOPS / 1,792 GB/s ≈ 144 FLOP/byte      (GB10: 213 / 273 ≈ 780)
```

so a 128×128 tile fed from DRAM alone would be memory-bound by a factor of ~2.3 on the 5090 and
~12 on the GB10. The reason it still works: neighbouring blocks share A rows and B columns, and
the L2 (96 MB on the 5090, 24 MB on the GB10) serves those re-reads. A 4096² bf16 operand is
32 MB, so on the 5090 both operands of a 4096³ GEMM sit in L2 at once; Nsight Compute reports a
96.5% L2 hit rate and ~5% of DRAM throughput for v3 even at 8192³, where they do not. Per-SM
the tile's smem-side intensity is the number that matters: each fragment loaded from smem is
reused across 4 (A) or 2 (B) `mma_sync` calls in v1/v2, and 4 (A) or 4 (B) `mma.sync` calls in
v3.

### Padding

WMMA loads a 16×16 bf16 sub-tile whose rows are `ldm` elements apart. With `ldm = 32` (64 B)
rows map to the same shared-memory banks every two rows; with `ldm = 40` (80 B, +16 B pad) the
16 row starts hit 16 different bank groups. The same trick with `ldm = 136` for B. Cost: 1.5 KB
of smem. Nsight metric to watch: `l1tex__data_bank_conflicts_pipe_lsu_mem_shared`. Measured on
the 5090: the v2 fragment *loads* are conflict-free; the only conflicts Nsight reports for v2
are on the epilogue's fp32 staging *stores* (2.2 M conflicts over 1.05 M store requests), which
v3 removes altogether.

### cp.async pipeline (v2)

```
issue(tile 0); commit
for t:
    if t+1 < T: issue(tile t+1 into other stage); commit; wait_group 1   # tile t landed
    else:       wait_group 0
    __syncthreads()
    mma over stage t&1
    __syncthreads()          # nobody still reads the stage that t+2 will overwrite
```

`cp.async.cg` copies 16 B global→shared without staging through registers, so the load
instructions do not occupy the warp's issue slots while the tensor cores run. Two stages ×
(10,240 + 8,704) B + an 8 KB fp32 epilogue staging buffer = 46,080 B, under the 48 KB static
limit. v2 requires M, N % 128 == 0 and K % 32 == 0 so every asynchronous copy is in bounds;
v1 zero-fills partial tiles and handles any multiples of 16.

### Epilogue

Accumulators are fp32 and WMMA cannot store bf16 accumulators directly, so each warp stores one
16×16 fp32 fragment to a per-warp 1 KB smem scratch, converts, and writes 8 bf16 (16 B) per lane
with a single vector store.

### Variant 3: mma.sync + ldmatrix, swizzle, three stages, tail split-K

Same 128×128×32 block tile and 2×4 warp grid as v1/v2 (warp tile 64×32, now 4×4 `m16n8`
accumulators per warp) on the large shapes; the kernel is templated on `BM`, `BN`, `BK` and
the stage count, and a 64-row tile takes over for small problems (next subsection). Four things
changed in the tile itself:

* **Raw `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` with `ldmatrix`.** The PTX
  fragment layout is documented, so each lane knows which 32-bit words it holds. One
  `ldmatrix.x4` fills a whole 16×16 A fragment from smem (four 8×8 matrices, one row address
  per lane), and one `ldmatrix.x4.trans` fills the B fragments of two adjacent `n8` tiles from
  the k-major B slab. Per k16 step per warp: 4 + 2 = 6 `ldmatrix` for 16 `mma.sync`, versus 6
  WMMA `load_matrix_sync` (each several `LDS`) for 8 `mma_sync` before.
* **XOR swizzle instead of padding.** The 16-byte chunk index of an A row (64 B rows, 4 chunks)
  is XORed with `(row/2) % 4`; of a B row (256 B rows, 16 chunks) with `row % 8`. The eight row
  addresses an `ldmatrix` touches then fall in eight different bank groups, with zero padding
  bytes: 16 KB per stage instead of 18.9 KB. Measured: 247 K bank conflicts over 403 M
  shared-load wavefronts at 8192³ (0.06%).
* **Three-stage `cp.async` pipeline** in 49 KB of dynamic shared memory (opted in with
  `cudaFuncSetAttribute`): tiles *k+1* and *k+2* are in flight while the tensor cores work on
  tile *k*, one `__syncthreads` per BK step, and the group count stays uniform by always
  committing (possibly empty) groups. A `wait_group STAGES-2` before the barrier is what makes
  the depth a compile-time constant.
* **Register-direct epilogue.** Each lane owns `(row g, cols 2c..2c+1)` and `(row g+8, same)`
  of every 16×8 accumulator, so the output is two `__nv_bfloat162` stores per tile straight
  from registers: no fp32 staging buffer, no barriers after the K loop.

The fourth change is a scheduling one, and it is the one that moved the 4096³ number the most.

#### Wave quantization and the split-K tail

The 5090 has 170 SMs and v3 fits 2 blocks per SM (121–128 registers, 49 KB smem), so 340
tiles run at once. A 4096³ GEMM has 32 × 32 = 1,024 tiles = **3.01 waves**: three full waves,
then 4 tiles run alone for as long as a full wave would. That is 4 waves of time for 3.01 waves
of work, 75% efficiency. 8192³ has 4,096 tiles = 12.05 waves, paid as 13 (93%). Before the tail
fix v3 measured 200 TFLOPS at 4096³ against cuBLAS's 226, and the arithmetic said exactly why.

The fix is split-K on the tail only, in the spirit of Stream-K: the `tiles mod 340` leftover
tiles are each split `split = min(KT, 340 / tail)` ways along K over the otherwise idle blocks
(4 tiles × 85 slices at 4096³, 16 × 21 at 8192³), so the extra wave lasts 1/`split` of a tile.
Each slice accumulates its K-range into an fp32 workspace tile with `atomicAdd` (performed at
L2, order-independent), fences, and bumps a per-tile arrival counter; the last slice to arrive
converts the finished fp32 tile to bf16 in-kernel, reading the workspace with `__ldcg` so it
sees L2 and not a stale L1 line. The workspace (`tail × 64 KB` + counters) is a per-device
static grown on demand and zeroed with `cudaMemsetAsync` on the same stream before the launch;
the full-wave tiles store bf16 directly and never touch it. Two consequences worth stating:
those tail tiles are not bitwise reproducible run to run (fp32 atomics in varying order, then
one bf16 rounding), and the tile order is row-major over N with `blockIdx.x` flattened, so the
DP tiles are exactly the first `tiles - tail` in that order.

Configuration sweep on the 5090 (BN, BK, stages), 4096³ / 8192³ TFLOPS, before the tail fix:
(128, 32, 3) **201 / 221**, shipped; (128, 32, 4) 182 / 201, the compiler dropped to 93
registers and lost throughput; (128, 64, 3) 185 / 205; (256, 32, 3) and (256, 32, 4) 166 / 203
with 186–188 registers and one block per SM; (256, 64, 3) needs 147 KB of smem and does not
fit in the 100 KB configuration.

#### Tile selection and decode shapes

The tail split fixes the *last* wave. It does nothing for a problem that never fills the
*first* one: 1024³ is 8 × 8 = 64 tiles of 128×128 on 340 slots, 0.19 of a wave, and splitting
64 tiles five ways along K leaves every block with six K-steps of work and a three-stage
prologue to amortize over them. v3 measured 52% of cuBLAS there with the 128-row tile. A
decode step is worse: with 16 tokens in flight, `M = 16`, a 128×128 tile could not even be
launched (v2 requires M % 128 == 0) and the default fell back to v1.

So `launch_auto` picks the tile per call, from the same kernel source:

| condition | tile | stages | why |
|---|---|---|---|
| M > 64 and the 128×128 grid is ≥ one full wave (resident blocks from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` × SM count) | 128×128×32 | 3 | the large-shape config above |
| the 64×128 grid is ≥ one full wave | 64×128×32 | 3 | twice the tiles, half the A reuse |
| everything else, M > 64 | 64×64×64 | 3 | 1024³ becomes 256 tiles; `BK = 64` halves the per-step pipeline overhead that dominates when a block owns a dozen K-steps |
| M ≤ 64 (decode) | 16, 32 or 64 rows × 32 or 64 columns, `hgemm_decode.cu` | 3 or 4 | bound by streaming B: one CTA per column strip of B, no split-K and no memsets on the real shapes (next subsection) |

Rows past M are handled without a branch in the inner loop: the A-tile copies for rows
`≥ M - bm` are issued as `cp.async` with a source size of 0 (`cp_async_16_zfill`), which reads
nothing and writes 16 zero bytes to smem, so the `mma.sync` work on those rows is on zeros and
the epilogue and the split-K paths skip them, one row at a time, so any M ≥ 1 works (M = 1 is
the single-token decode); N % 64 == 0 and K % 64 == 0 are required. The split-K tail applies
to every tile size.

Sweep that picked the two small configs, as % of cuBLAS on 16×4096×4096 / 64×4096×4096 /
16×11008×4096 / 1024³:

| tile, BK, stages | 16×4096² | 64×4096² | 16×11008×4096 | 1024³ |
|---|---|---|---|---|
| 64×64, 32, 3 | 97.0 | 95.9 | 96.9 | 101.1 |
| 64×64, 64, 3 | 96.9 | 97.2 | 100.4 | **101.5** |
| 64×64, 32, 4 | 96.6 | 90.8 | 96.9 | 101.5 |
| 64×64, 64, 4 | **104.3** | **104.1** | 97.0 | 90.3 |
| 64×128, 32, 3 | 100.3 | 83.5 | 94.3 | 63.6 |
| 64×128, 64, 3 | 96.6 | 91.2 | 103.9 | 101.1 |

`BK = 64` with four stages won the decode shapes and lost 1024³ to its lower occupancy
(64 KB of smem per block, one block per SM), hence the split by `M ≤ 64`. That was the decode
path of the first run; the kernel in the next subsection replaced it for M ≤ 64.

A decode GEMM is a bandwidth problem: at `M = 16`, `N = K = 4096` there are 2·16·4096·4096 =
537 MFLOP against 32 MB of weights, 16 FLOP/byte, a ninth of the ridge. The kernel's job is to
stream B once at full bandwidth, and the benchmark has to let it. B alone fits in the 96 MB L2,
so timing one B back to back measured 1,730–1,880 GB/s on these shapes, above the DRAM spec,
for cuBLAS and for us alike: the weights never left L2 between iterations. A real decode step
touches every layer's weights once per token, so `bench_hgemm` now rotates through enough
copies of B to exceed L2 (256 MB+) whenever M ≤ 64, and both sides are timed the same way.
Every `hgemm` row now records `gbps = 2(MK + KN + MN) / time`, the traffic floor.

### Decode: streaming the weights

At M ≤ 64 the GEMM reads B once and does almost no math: 16×4096×4096 is 32 MB of weights
for 0.54 GFLOP. The 64×64×64 tile above reached 1,220 GB/s on it (27.7 µs), 80% of the
1,532 GB/s `cudaMemcpy` roof, with cuBLAS at the same place. This subsection is the dedicated
path, `src/kernels/hgemm_decode.cu`, that `launch_auto` takes for every M ≤ 64.

#### What a single launch can do

Before touching the kernel I measured the floor with a scratch kernel that streams a buffer
with 16 B loads, does nothing else, and is timed exactly like `bench_hgemm` times a GEMM:
one launch between two CUDA events, the buffer rotated past L2.

| | 32 MB | 88 MB |
|---|---|---|
| one launch, read only | 23.6 µs (1,419 GB/s) | 58.4 µs (1,581 GB/s) |
| two launches back to back, per launch | 21.6 µs (1,552 GB/s) | 56.3 µs (1,614 GB/s) |
| an empty launch | 1.4 µs | |
| two `cudaMemsetAsync` in front of the launch | +2.1 to 2.9 µs | |

So one 32 MB launch has a floor of about 23.6 µs the way the bench times it, which is
1,430 GB/s in the bench's `2(MK + KN + MN) / time` metric, and 2 µs of that is launch and
ramp that back-to-back launches hide. The old path's 27.7 µs had 4 µs to give: two memsets
and a kernel per call, and a grid of 128 CTAs (64 tiles, tail split two ways) on 170 SMs.

#### Bytes in flight

Bandwidth is bytes in flight over latency. A CTA of the new kernel keeps `STAGES − 1` tiles
of B in flight; at BK = 64, BN = 64 and 4 stages that is 3 × 8 KB = 24 KB. Measured with
one CTA per column strip and no split-K: 32 CTAs (BN = 128) top out at 1,227 GB/s, 38 GB/s
per CTA, which is 24 KB every 630 ns, the loaded DRAM latency; 64 CTAs (BN = 64) reach the
floor, 1,440 GB/s; 128 (BN = 32) the same. So the card needs about 1.5 MB in flight, 64 × 24
KB, or 9 KB per SM if it were spread over all 170. Deeper pipelines do nothing once the CTA
count is right: 6 stages measured the same 1,442 GB/s as 4. The old tile had 128 × 24 KB =
3 MB in flight and lost its time elsewhere.

#### The kernel

A BM × BN × BK block tile with BM ∈ {16, 32, 64} picked from M, so a 16-token step runs one
m16 row block instead of four with three zero-filled. Warps sit in a WM × WN grid over the
tile; each warp owns BM/WM rows by BN/WN columns, `ldmatrix.x4` for A and `ldmatrix.x4.trans`
for B out of the same XOR-swizzled smem layout as v3 (rows of 32 B get `chunk ^= (row/4) % 2`,
64 B `chunk ^= (row/2) % 4`, 128 B and up `chunk ^= row % 8`; Nsight reports zero bank
conflicts). The A slab of a stage (BM × BK, 2 to 16 KB) is shared by all warps and comes
from L2: every column strip re-reads all of A, 64 × 128 KB = 8 MB at M = 16 and 128 × 512
KB = 64 MB at M = 64, next to 32 MB of B from DRAM (`lts__t_sectors_op_read` 42 MB and
101 MB). Same `cp.async` pipeline as v3, one barrier per BK step.

One CTA per column strip of B, the strips of a wave reading the same rows of B so that the
DRAM sees whole contiguous rows. No split-K on any shape that matters. When a narrow N
leaves fewer than 64 strips (N = 1024 has 16 at BN = 64) K is split just enough to reach 64
CTAs, each slice adds its fp32 partial tile into a workspace with `atomicAdd`, and the last
slice to arrive at a strip (a per-strip counter) converts it to bf16, writes zeros back over
the workspace and resets the counter. The workspace is zeroed once, when it is allocated,
and every launch leaves it zero, so a call is one launch, never a memset and a launch.
16×1024×4096 goes from 15.9 µs unsplit to 8.1 µs with split 4.

#### Why not split-K everywhere, and why not persistent CTAs

Every split-K launch measured a flat 2 µs over the unsplit kernel, 2 to 16 slices alike
(25.5 µs against 23.5 at 16×4096×4096). That is the chain at the end of each CTA: the
`RED` adds to L2, a `membar.gl` that waits for them, the counter atomic that returns a
value, and the last arriver's read-back; three L2 round trips that the last wave cannot
hide. Persistent CTAs looping over (strip, slice) items with the pipeline kept flat across
items (the next item's tiles load while the current item's epilogue runs) measured the same
25.5 µs at split 10 and worse beyond (27.5 at split 20, 33.6 at 64). They amortize the
prologue, and the prologue was never the cost; the reduction is.

#### M = 64 is tensor-bound per SM

64×4096×4096 is 4 × 512 × 256 = 524,288 `mma.sync.m16n8k16`. One takes 32 clocks on one
SM sub-partition (that is the 258.7 TFLOPS peak restated: 170 × 4 × 4,096 FLOP / 32 clk ×
2.976 GHz), 16.8 M sub-partition clocks in all. On 64 CTAs that is 65 K clocks per SM, 22 µs
at 2.97 GHz, longer than the 21 µs the weights take to stream, and every 64-CTA layout (4, 8
or 16 warps on a 64×64 tile) measured 36 to 38 µs: the tensor time plus the part of the
stream it does not hide. More warps per CTA cannot help; more SMs can. 128 CTAs of a 64×32
strip (8 warps as 4 × 2) put 11 µs of tensor work per SM under the stream and measure 25.5
µs single-launch; BK = 128 (half the barriers per byte of B, 96 KB of smem, one CTA per SM)
takes it to 22.7 µs back to back. Nsight says the same thing: `sm__pipe_tensor_cycles_active`
60%, top stall `math_pipe_throttle`. 256 CTAs of a 64×16 strip would hide the tensor work
entirely but read 32 B per row of B and lose DRAM efficiency (1,103 GB/s), so 128 it is.

#### Wave fit

172 strips of 64 columns at N = 11008 on 170 SMs with one CTA each is a full wave plus two
stragglers that each run as long as the wave: 70.6 µs instead of 56. So each row tile has
two configurations, and the launcher takes the first whose strip count fits in the resident
slots (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` × SMs), otherwise the second, which
has more slots:

| M | first | slots | second | slots |
|---|---|---|---|---|
| ≤ 16 | 16×64×64, 4 stages, 1×4 warps, 40 KB | 340 | 16×32×64, 4 stages, 1×2 warps, 24 KB | 680 |
| ≤ 32 | 32×32×64, 4 stages, 2×2 warps, 32 KB | 510 | 32×64×64, 4 stages, 2×4 warps, 40 KB | 340 |
| ≤ 64 | 64×32×128, 4 stages, 4×2 warps, 96 KB | 170 | 64×64×64, 3 stages, 2×4 warps, 48 KB | 340 |

64×11008×4096 (344 strips of 32) runs the second: 61.9 µs against 71.3 with the first.

#### The sweep

Single-launch bench medians in GB/s (the bench's metric), B rotated past L2. "split" is
K-slices per strip; "persistent" loops a grid of 340 CTAs over the items. The first row is
the path this kernel replaced.

| tile, stages, warps (WM×WN), split | 16×4096² | 64×4096² | 16×11008×4096 | 64×4096×11008 |
|---|---|---|---|---|
| 64×64×64, 4, 2×4, tail split 2, two memsets (old v3) | 1,219 | 1,165 | 1,448 | 1,469 |
| 16×128×32, 5, 1×8, split 10, no memsets | 1,326 | | 1,552 | |
| 16×64×64, 4, 1×4, split 2 / 4 / 10 / 16 | 1,328 / 1,328 / 1,324 / 1,324 | | | |
| 16×64×64, 4, 1×4, split 10, persistent | 1,324 | | | |
| 16×64×64, 4, 1×4, split 20 / 64, persistent | 1,229 / 1,006 | | | |
| 16×128×64, 3, 1×8, split 1 (32 / 86 CTAs) | 1,227 | | 1,613 | |
| 16×64×64, 4, 1×4, split 1 (64 / 172 CTAs) | **1,442** | | 1,564 | |
| 16×64×64, 6, 1×4, split 1 (one CTA per SM) | 1,442 | | 1,284 | |
| 16×32×64, 4, 1×2, split 1 (128 / 344 CTAs) | 1,442 | | 1,602 | |
| 64×64×64, 4, 1×4, split 1 (64 CTAs) | | 960 | | 1,012 |
| 64×64×64, 4, 2×4, split 1 | | 915 | | 968 |
| 64×64×64, 4, 4×4, split 1 | | 916 | | 948 |
| 64×64×64, 4, 4×4, split 2 (128 CTAs) | | 1,355 | | 1,530 |
| 64×16×64, 4, 4×1, split 1 (256 CTAs) | | 1,103 | | 1,004 |
| 64×32×64, 4, 4×2, split 1 (128 CTAs) | | 1,357 | | 1,473 |
| 64×32×128, 4, 4×2, split 1 | | 1,355 | | **1,532** |
| 64×32×128, 3, 4×2, split 1 | | **1,367** | | 1,528 |

The single-launch medians of this bench fall on a grid of about 2 µs (23.5, 25.5, 27.5 ...
µs; the minimum moves between them), so for the last comparisons I also timed 20 launches
per event pair, which is how a decode step queues its GEMMs anyway. Per launch:

| shape | old v3 | new, single launch (bench) | new, 20 launches per event pair |
|---|---|---|---|
| 1×4096×4096 | (unsupported) | 23.5 µs, 1,427 GB/s | 21.4 µs, 1,569 GB/s |
| 16×4096×4096 | 27.7 µs, 1,219 GB/s | 23.5 µs, 1,438 GB/s | 21.5 µs, 1,574 GB/s |
| 32×4096×4096 | 29.7 µs, 1,146 GB/s | 23.6 µs, 1,447 GB/s | 21.8 µs, 1,566 GB/s |
| 64×4096×4096 | 29.7 µs, 1,165 GB/s | 23.7 µs, 1,459 GB/s | 22.7 µs, 1,527 GB/s |
| 16×11008×4096 | 62.6 µs, 1,448 GB/s | 56.4 µs, 1,607 GB/s | 55.1 µs, 1,646 GB/s |
| 64×4096×11008 | 62.7 µs, 1,469 GB/s | 59.7 µs, 1,543 GB/s | 57.2 µs, 1,610 GB/s |
| 64×11008×4096 | | 62.4 µs, 1,476 GB/s | 60.5 µs, 1,522 GB/s |

The 4096² weights now stream at the single-launch floor (23.5 µs against the 23.6 µs of the
read-only probe) and at 1,552 to 1,574 GB/s back to back, above `cudaMemcpy`'s 1,532: a
read-only stream is a little faster than a copy on this card. M = 1, 16, 32 and 64 cost the
same 23.5 µs, so the decode GEMM is now the weights and nothing else. cuBLAS takes 29.6 to
30.8 µs on the 4096² shapes and 39.9 µs at M = 1 (its gemv path is slower than its GEMM path
for bf16), so the rows read 125 to 170% of cuBLAS.

#### Nsight Compute

`--launch-skip 1 --launch-count 1` on the timing launches (Nsight locks the clocks, so its
durations are longer than the bench's). `dram__bytes_read` is not exposed on this card by
this Nsight, so the DRAM bytes are throughput × duration.

| | 16×4096×4096 | 64×4096×4096 | 64×4096×11008 |
|---|---|---|---|
| kernel | 16×64×64, 4 st, 1×4 warps | 64×32×128, 4 st, 4×2 warps | same |
| grid × block | 64 × 128 | 128 × 256 | 128 × 256 |
| registers, dynamic smem | 56, 41.0 KB | 68, 98.3 KB | 68, 98.3 KB |
| `dram__throughput` % of peak | 83.2 | 77.0 | 87.6 |
| duration under Nsight | 23.0 µs | 25.1 µs | 60.9 µs |
| DRAM bytes from that (B alone) | 34 MB (32 MB) | 35 MB (32 MB) | 96 MB (88 MB) |
| `lts__t_sectors_op_read` | 1.32 M (42 MB) | 3.15 M (101 MB) | 8.46 M (271 MB) |
| shared bank conflicts | 0 | 0 | 0 |
| `sm__warps_active` (achieved occupancy) | 8.3% | 16.6% | 16.7% |
| `sm__pipe_tensor_cycles_active` | 30.9% | 60.1% | 63.7% |
| top stalls (warps per issue) | long_scoreboard 1.7, wait 1.7, short_scoreboard 1.6 | math_pipe_throttle 3.5, wait 2.0, short_scoreboard 2.0 | math_pipe_throttle 3.6, wait 2.0, short_scoreboard 2.0 |

B is read once (the DRAM bytes are within 3 to 8% of B, and the difference includes A, C and
the cold start under the profiler); the L2 read counts are B plus the A re-reads per strip.
The 16-row kernel waits on memory (long scoreboard, at 2 warps per SM); the 64-row kernel
waits on the tensor pipe, which is the M = 64 argument above in one metric.

#### Not tried

A register-only B path (16 B vector loads straight into `mma.sync` operands, no smem for B)
needs k-pairs per lane for the `col` operand, and a row-major B gives a lane 8 consecutive
n at one k; producing the pairs takes a cross-lane transpose per fragment that
`ldmatrix.trans` does for free from smem. With smem bandwidth nowhere near a limit (one
`ldmatrix` per 512 B of B) there was nothing to gain, so the `cp.async` staged version is
the only one.

### Variant 4: Stream-K on a persistent grid

Variant 3 fixes the tail of the last wave and nothing else about scheduling. Three things
it leaves on the table showed up in its own numbers. A problem under one wave of 128×128
tiles (2048³ is 256 tiles on 340 slots) has to drop to the 64×128 tile to fill the card,
and pays for the lower reuse with 96% of cuBLAS. The tile order is row-major over N with
the hardware dispatching blocks in index order, so at 8192³ each wave of 340 tiles spans
5.3 tile rows and all 64 tile columns: A is reused 64 times from L2 but B (128 MB) is read
from DRAM once per wave. Nsight measured 664 GB/s of DRAM traffic and an 80.5% L2 hit rate
for v3 at 8192³ with the caches left warm between replays. And every call with a tail costs
two `cudaMemsetAsync` launches before the kernel, which on a 28 µs decode GEMM is 4 µs.

Variant 4 keeps the tile (the k-loop, the swizzle and the register epilogue moved to
`hgemm_tile.cuh` unchanged; v3 picked up 2 registers and 1 to 2% at the large shapes from
the move, nothing else about it changed) and replaces the block-per-tile launch with a
persistent kernel.

#### The schedule

The grid is `resident` blocks, from `cudaOccupancyMaxActiveBlocksPerMultiprocessor` × SM
count: 340 for the 128×128 tile, 340 for 64×128, 340 for 64×64×64 with three stages and
170 with four. Tiles are numbered in a grouped order (next subsection) and the kernel
picks one of two ways to hand them out per call:

* **Static ranges**, when the problem is at most two waves, or at most four when A and B
  together fit in three quarters of the L2. The flattened (tile, k-step) space of
  `tiles × KT` iterations is cut into `grid` equal contiguous ranges. Every block does the
  same number of k-steps, so there is no quantization at all; a tile is finished by at most
  two or three blocks. The grid is `min(resident, max(tiles, tiles × KT / 32))`: never a
  piece shorter than 32 k-steps (the fixup traffic would outweigh the mma work), and never
  fewer blocks than tiles, so 1024³ runs as 256 whole tiles with no partials, exactly the
  v3 schedule. A block walks its range from the top: the piece that starts a tile is
  computed first and published, the piece that ends one last.
* **A queue** otherwise. Whole tiles are handed out by an `atomicAdd` on a global counter
  (one per tile), so a block on a slow SM simply takes fewer of them. The last wave plus the
  tail (`tail + 340` tiles) are handed out by the same queue in K-passes of geometrically
  shrinking length, half the tile, then a quarter, and so on while the next half would still
  be longer than 8 k-steps, tile-interleaved (every tile's first pass, then every tile's
  second, ...). The last pass is 9 to 16 k-steps, so the kernel ends within that much work
  on every SM, at the cost of one chain link per pass: 4 at K = 4096 (64, 32, 16, 16
  k-steps), 5 at 8192, 6 at 11008. The last block out resets the two queue words, so the
  next launch starts from zero without a memset.

The tile is chosen by the per-block share of k-steps: 128×128 whenever
`tiles × KT / resident ≥ 32`, then 64×128 by the same rule, then 64×64×64; M ≤ 64 takes
the 64×64×64 four-stage tile as in v3. 2048³ therefore stays on the 128×128 tile (256 tiles,
48 k-steps per block) and 1024³ on 64×64 (256 tiles, one each).

#### The fixup

Every Stream-K tile has one fp32 slot of BM×BN in the workspace and one 64-bit flag. A
piece that does not start its tile spins (thread 0, `ld.acquire.gpu`) until the flag reads
its `kt_begin`, then adds the slot into its accumulators. A piece that does not end its tile
writes the running sum to the slot, fences, and publishes its `kt_end` with
`st.release.gpu`. The piece that ends the tile stores bf16 from registers like a whole
tile. So the partials of a tile are always summed in K order, one running sum, whichever
block computed which piece, and the output is the same bits every run; `tests/test_gemm.py`
checks that with `torch.equal` across six calls. The slot is laid out in fragment order
(lane-contiguous `float4`s), so a warp writes and reads 512-byte runs and a lane finds its
own accumulator elements at the same place another block's lane put them; slot reads use
`__ldcg` so they come from L2 and never from a stale L1 line. Rows past M are skipped on
both sides.

I chose this "the next piece waits" form over the "owner waits for every contributor" form
in CUTLASS because it needs one slot per tile instead of one per block, each piece reads
one slot instead of the finisher reading all of them, and in queue mode, where which block
does which piece is decided at run time, there is no per-block bookkeeping to reconstruct.
The flag carries a per-launch epoch in its high word: `(epoch << 32) | kt`. A value from an
earlier launch can never match, so nothing is cleared between launches, and no counter has
to reset itself. The workspace holds two waves of tiles (43 MB for 128×128), grown on
demand, one per tile configuration and device; like v3's it is not safe to share between
streams that run this kernel concurrently.

Why the wait cannot deadlock. A piece only waits for the piece before it in K. In queue
mode that piece was taken from the queue earlier by a block that is therefore running (a
block only holds a queue item while it is resident), and the chain ends at the piece that
starts the tile, which never waits. In static mode the previous piece belongs to the block
with the next lower index, which the hardware dispatched no later than this one; the grid
never exceeds the resident block count, so it is resident, and it computed that piece
first. Cycles are impossible because waits only ever point to earlier K.

#### Rasterization

Tiles are numbered in groups of G tile rows walked column-major. With G = 16 (the largest
power of two whose square is under 340) the 340 tiles in flight cover 16 tile rows and
21.25 tile columns: at 8192³ that is 16 × 2 MB of A and 21.25 × 2 MB of B, 74.5 MB, inside
the 96 MB L2, against 5.3 × 2 MB of A plus all 128 MB of B in row-major order. Nsight, same
method for both (`--set full`, caches left warm, the profiler's fixed 2.55 GHz):

| 8192³ | L2 hit rate | DRAM throughput | DRAM traffic over the kernel | tensor pipe | duration at 2.55 GHz | bench, power-limited clocks |
|---|---|---|---|---|---|---|
| v3 | 80.5% | 664 GB/s (37.7% of peak) | 3.5 GB | 94.6% | 5.34 ms | 4.91 ms |
| v4 | 95.6% | 165 GB/s (9.4%) | 0.9 GB | 92.9% | 5.40 ms | 4.82 ms |

At the profiler's fixed clock v4 is 1% slower (the chain links and the prologue each piece
pays for), in the benchmark it is 1.8% faster: the kernel runs into the 600 W limit either
way and the 2.6 GB of DRAM traffic it no longer moves went into clock. Sweeping G at 8192³
in one session, queue mode: G = 1 4.848 ms, 4 4.838, 8 4.809, 16 4.777, 32 4.773. In
static mode the tiles in flight are strided by the number of waves through the raster, so
G matters less there, and it is one reason static mode is limited to a few waves.

#### What the sweep found (all times in ms, comparisons within one session; the box drifts by up to 2% between sessions)

The first version was the textbook one, contiguous ranges over the whole problem for every
shape. It won 4096³ (0.591 vs v3 0.597) and lost 8192³ (4.94 vs 4.82), and the profile
said why: with 12 tiles per block the tiles in flight are 12 apart in the raster, the L2 hit
rate fell to 68% and DRAM to 1.03 TB/s; with the G = 16 grouping on top of that stride it
fell to 40% and 1.4 TB/s and the kernel took 7.3 ms.

So the multi-wave case became whole tiles plus a Stream-K share of the last wave (CUTLASS's
"two-tile" hybrid). With the whole tiles assigned statically (block c takes tiles c, c+340,
...) it measured 4.97 at 8192³, and per-block timestamps (`%globaltimer` at start and end)
showed the reason: blocks end between 3.93 and 4.93 ms, and the spread is already there
after the first phase, where every block has identical work. Some SMs run this kernel about
20% slower than others, consistently, in TPC pairs (SMs 8/9, 30/31, 52/53, 118/119,
140/141 on this card). v3 never sees this because the hardware dispatcher gives a slow SM
fewer blocks; a static persistent schedule waits for it. Hence the queue.

Then the order of the two phases. Queue first and a static Stream-K share last ran 4.78 at
8192³ but 1.71 at 4096×11008×4096 (bimodal, min 1.61): the queue's phase ends with a spread
of a whole tile-time, and a block that finishes a tile can then wait for a contributor that
took the last queue tile on a slow SM. Stream-K first and the queue last fixed that shape
(1.64) and lost 8192³ (5.00): the queue's own tail, a block that takes the last tile on a
slow SM ends 1.2 tile-times after the rest. The fix for both is for the work at the end to
be fine-grained and dynamic: the last wave in K-chunks from the same queue, which needs a
fixup that does not care which block did which chunk, the K-ordered chain above.

Uniform chunks, tile-interleaved, 4096³ / 8192³ / 4096×11008×4096 / 4096×4096×11008:
16 k-steps 0.616 / 4.769 / 1.627 / 1.641; 32 0.619 / 4.750 / 1.615 / 1.623; 64 0.638 /
4.768 / 1.635 / 1.608; 128 0.689 / 4.797 / 1.674 / 1.657. Handing a tile's chunks out back to
back instead of interleaved put 4096×4096×11008 at 2.07 with 16-step chunks: the chain
serializes and every chunk waits. Geometric passes (half, quarter, ..., last pass 9 to 16
k-steps) beat the best uniform size everywhere: 0.611 to 0.615 / 4.83 to 4.85 (v3 4.91 to
4.93 in that session) / 1.627 to 1.647 (v3 1.625 to 1.637) / 1.624 (v3 1.627). Letting the
passes shrink to 5 to 8 k-steps was no better, stopping at 17 to 32 was worse (0.627 at
4096³).

That left 4096³, where the queue was still 1.5 to 2% behind v3 in three interleaved rounds
(0.611 to 0.615 against 0.599 to 0.604): with four tail tiles split 85 ways, v3's schedule
is already near ideal there, and the queue pays four chain links per tile plus a prologue
per piece. Static ranges tie v3 at 4096³ (0.599 to 0.605) and win by 30% at
4096×2048×4096 (1.5 waves, 0.306 vs 0.398) and by 25% at 2048³, but lose 2% at
4096×4096×11008, where the operands (90 MB each) do not fit in L2 and the strided in-flight
set costs reuse. That is the rule in the code: static ranges up to two waves, or up to
four when A + B fit in three quarters of the L2, the queue beyond.

Smaller knobs. The 32 k-step minimum share in static mode came from the decode shapes:
16×4096×4096 at a minimum of 8, 16, 32, 64 k-steps measured 27.5, 27.5, 25.4, 35.5 µs (fewer
blocks with longer B streams beat more blocks with a fixup each, up to the point where too
few blocks are streaming). The tile rule came from 2048³ (128×128 on Stream-K 0.083, 64×128
0.087, 64×64 0.089, v3's 64×128 without a split 0.1035) and 1024³ (128×128 on Stream-K
0.0236, a 12 k-step share per block, against 0.0174 on 64×64 whole tiles).

Per-block timelines of the final schedule at 8192³: the queue phase ends between 4.33 and
4.46 ms across blocks, the passes bring every block to between 4.79 and 4.82 ms. At 4096³
in queue mode the blocks end between 585 and 612 µs with a mean of 596 against an ideal
572 (3.01 tiles' worth of k-steps per block), which is the 1.5 to 2% the static schedule
gets back.

### Variant 5: TMA and warp specialization

Source: `src/kernels/hgemm_tma.cu`, helpers under "TMA / mbarrier" in `include/spark/common.cuh`.

v3's tensor pipe was 91 to 95% busy in Nsight with `math_pipe_throttle` the top stall, and I
had put the last 5% at 8192³ down to power. Before believing that I wanted to remove the one
thing the consumer warps still do besides `ldmatrix` and `mma.sync`: the copies. In v3 every
thread computes two global addresses and two swizzled shared addresses per stage and issues
two `cp.async` instructions, and the whole block meets at a `__syncthreads` once per K-tile.
The Blackwell TMA unit can do all of that from a single instruction, and an mbarrier pipeline
lets the producer and the consumers stop meeting. So v5 keeps v3's 128×128 tile, its 2×4
warp grid and 64×32 warp tile, its `ldmatrix` + `mma.sync` k-loop body, its register-direct
epilogue and its tail split-K (the `Sched` struct and workspace are copied, not shared, so v3
stays untouched), and replaces only the way a stage gets to shared memory.

**Tensor maps and boxes.** The host encodes one `CUtensorMap` per operand with
`cuTensorMapEncodeTiled` (the entry point comes from `cudaGetDriverEntryPointByVersion`, so
nothing links against libcuda): A is described as a K-innermost matrix with a box of
BK columns × 128 rows, B as an N-innermost matrix with a box of 64 columns × BK rows. Encoding
is 128 bytes of host arithmetic, measured at 27 ns per call, so the maps are rebuilt per call
and passed as `__grid_constant__` kernel parameters. One producer lane issues three
`cp.async.bulk.tensor.2d` per stage: the A box and the two 64-column halves of the B tile.
32 KB of operands (BK = 64) for three instructions; v3 issues 512 `cp.async` for the same
bytes.

**The swizzle is the same XOR.** A TMA box lands in smem row-major with the map's swizzle
applied. The 128-byte swizzle XORs address bits [4:7) (the 16-byte chunk within a 128 B row)
with bits [7:10) (the row within an 8-row group), i.e. `chunk ^= row % 8` for 128 B rows,
which is exactly v3's `swz_b` and the BK = 64 form of `swz_a`. The 64-byte swizzle XORs bits
[4:6) with [7:9), i.e. `chunk ^= (row / 2) % 4` for 64 B rows, which is v3's BK = 32 `swz_a`.
So the box inner dimension is fixed at one swizzle span (64 bf16 = 128 B for B and for A at
BK = 64; 32 bf16 = 64 B for A at BK = 32), the B tile is stored as two 64-column boxes side
by side, and the k-loop body addresses smem with v3's functions unchanged. The only change in
the consumer is that a warp's 32 columns of B live in box `wn / 2` at chunk
`(wn % 2) * 4 + nj + (lane / 16)`. Each `ldmatrix` still touches eight rows whose XORed chunks
are eight different bank groups: Nsight reports 3,658 bank conflicts over 403 M
shared-load wavefronts at 8192³ for the (64, 2, 1) configuration, against v3's 297 K. Both swizzles are keyed on the absolute shared address, so
every stage starts on a 1 KB boundary: the stages sit at the front of the dynamic region
(which starts aligned when the kernel declares no static smem; the kernel traps if not), the
barriers go after them, and the allocation is rounded to a whole KB so the second block on an
SM starts aligned too.

**Two mbarriers per stage, parity.** Stage `s` has a "full" barrier initialized with an
arrival count of 1 and an "empty" barrier initialized with 8. Per K-tile the producer lane
does `mbarrier.arrive.expect_tx` on full[s] with the stage's byte count and issues the three
loads with `.mbarrier::complete_tx::bytes` naming full[s]; the copy engine counts the bytes
down as they land and the phase completes when the arrival and the bytes are both in. The
consumers spin on `mbarrier.try_wait.parity`. A barrier flips between phase 0 and phase 1 each
time it completes, and `try_wait.parity P` returns once the phase with parity P has finished,
so the n-th use of a stage waits on parity `n & 1`: K-tile `kt` uses stage `kt % STAGES` for
the `kt / STAGES`-th time. After its last `mma.sync` on the stage each consumer warp
`__syncwarp`s and lane 0 arrives on empty[s]; before refilling a stage for its n-th use the
producer waits on empty[s] with parity `(n - 1) & 1`. The first STAGES fills skip that wait.
There is no `__syncthreads` after the barrier init: the consumer warps never wait for each
other, only for the data, and the producer only waits for the slowest consumer of the stage it
wants to reuse. The tail split-K path still needs a block-wide sync for its arrival counter;
the producer warp has exited by then, so the consumers use a named barrier (`bar.sync 1, 256`)
instead of `__syncthreads`.

**`fence.proxy.async`, and the bug it fixed.** `ldmatrix` reads shared memory through the
generic proxy; the TMA writes it through the async proxy, and the PTX memory model does not
order the two without a proxy fence. The read-after-write direction is covered for free: a
completed `cp.async.bulk` is followed by an implicit fence, so a consumer that has seen full[s]
complete may `ldmatrix` at once. The write-after-read direction is not: without a
`fence.proxy.async.shared::cta` before the release, the refill of a stage could land while a
lane's `ldmatrix` of it was still outstanding. I first shipped the kernel without that fence.
The one-block-per-SM configurations passed every shape; the two-block configurations returned
wrong tiles, only on grids with a split-K tail, only in tail tiles whose blocks had started
while another block was mid-pipeline on the same SM, and never when I instrumented the
consumer (the extra work moved the timing). One fence per consumer thread per stage, issued
before the `__syncwarp` and the arrive, and every configuration passes every shape.

**Registers and residency.** `setmaxnreg` is sm_90a only, so the producer warp carries the
same register allocation as a consumer: 288 threads × 126 registers at one block per SM, and
`__launch_bounds__(288, 2)` caps the two-block configurations at 96 registers without spills.
The RTX 5090 allows 101,376 B of dynamic smem per block and 102,400 B per SM with 1 KB
reserved per block, not the 227 KB of the datacenter parts, so a 32 KB stage (BK = 64) fits
three deep at one block per SM and two blocks per SM need BK = 32.

#### The sweep

TFLOPS at `--iters=50`, all configurations pass the cuBLAS check on every shape. Runs of the
same configuration repeat to within ±1.5% (the card sits at the 600 W limit and its clock
drifts with temperature; cuBLAS moved 229 to 233 TFLOPS at 8192³ across the session).

| BK, stages, blocks/SM | smem | regs | 2048³ | 4096³ | 8192³ | 4096×4096×11008 | 4096×11008×4096 | 2560²×4096 |
|---|---|---|---|---|---|---|---|---|
| 64, 2, 1 | 64 KB | 126 | 176.7 | 236.9 | **235.3** | **242.0** | **236.9** | 212.5 |
| 64, 3, 1 | 96 KB | 128 | 176.7 | 237.1 | 234.7 | 241.3 | 236.7 | 213.9 |
| 32, 3, 2 | 48 KB | 96 | **180.6** | 241.1 | 230.1 | 237.8 | 232.0 | **219.3** |
| 32, 2, 2 | 32 KB | 96 | **180.7** | **242.9** | 231.1 | 238.8 | 235.4 | **219.4** |
| 32, 3, 1 | 48 KB | 123 | | 236.9 | 232.1 | | | 215.7 |
| 32, 6, 1 | 96 KB | 129 | 176.8 | 238.6 | 234.4 | 238.8 | 234.4 | 213.9 |
| v3 (32, 3, 2 blocks) | 48 KB | 121 | 165.9 | 222.7 | 221.5 | 222.3 | 221.5 | |

Two stages of BK = 64 are enough: a stage is 2·128·128·64 = 2.1 MFLOP, 4,100 tensor-pipe
cycles per SM at the measured rate, about 1.5 µs, and the TMA round trip from L2 is a fraction
of that, so one stage in flight covers it. Depth beyond two buys nothing at one block per SM.
The two-block configurations win by 2% wherever both operands fit in the 96 MB L2 (2048³,
4096³, 2560²×4096: 16 to 64 MB) and lose by 2% wherever they do not (8192³ and the 11008
shapes: 180 to 256 MB). It is not the L2: Nsight at 8192³ has both at an 80% hit rate and,
with the clock locked, within 1% of each other in time (5.05 vs 5.10 ms). The difference only
exists at the 600 W limit, and the one thing that separates the two there is the instruction
stream: 971 M instructions for (32, 3, 2) against 622 M for (64, 2, 1) for the same
`mma.sync` count, since a 32-deep stage pays its waits, arrives and loop overhead twice as
often. My reading is that on the shapes that stream from DRAM the memory system takes its
share of the 600 W and the leaner instruction stream keeps a higher SM clock, while on the
L2-resident shapes there is clock headroom and the second block's latency hiding wins. That is
a reading of five data points, not a measurement; the rule it gives is what ships:
`hgemm_tma` picks (32, 3, 2) when 2(MK + KN) bytes fit in the L2 and (64, 2, 1) otherwise,
and `SPARK_HGEMM_V5_CONFIG=<index>` forces one of the six for a re-sweep.

#### Measured

`bench_hgemm --iters=50`, v3 and v5 back to back, same session:

| M×N×K | cuBLAS TFLOPS | v3 TFLOPS / % | v5 TFLOPS / % | v5 / v3 |
|---|---|---|---|---|
| 2048³ | 172.6 | 165.9 / 96.1% | **180.7 / 104.6%** | 1.09× |
| 4096³ | 222.5 | 224.8 / 101.1% | **240.4 / 107.9%** | 1.07× |
| 8192³ | 229.7 | 218.7 / 95.2% | **235.2 / 102.4%** | 1.08× |
| 4096×4096×11008 | 222.0 | 222.6 / 100.3% | **242.0 / 109.0%** | 1.09× |
| 4096×11008×4096 | 231.4 | 219.9 / 95.1% | **235.0 / 101.5%** | 1.07× |

1024³ and the decode shapes are not v5 shapes (`hgemm_supports` wants a 128×128 grid of at
least one tile per SM, so the default variant steps down to v3 on them; `tests/test_gemm.py`
checks that). The 2048³ row is a 1.5-wave grid, so it says as much about the tail split as
about the pipeline.

#### Power

`nvidia-smi --query-gpu=clocks.sm,power.draw -lms 200` sampled during a 1,500-iteration
8192³ loop of each kernel, both inside one GPU lock, samples from the last five seconds of
each run (the bench times cuBLAS first, then the kernel, so the tail of the trace is ours):

| kernel | TFLOPS (median of 1,500) | SM clock | power | TFLOPS per watt |
|---|---|---|---|---|
| cuBLAS (same runs) | 226.6 | 2,715 to 2,722 MHz | 600 W | 0.378 |
| v3 | 217.2 | 2,760 MHz | 600 W | 0.362 |
| v5 | 234.3 | 2,752 MHz | 600 W | 0.391 |

Every kernel pins the card at its 600 W limit within a second and the clock settles at the
same 2.75 GHz for v3 and v5 (the 25 samples in each window do not move by a MHz), so the
question in the title has a clean answer: TMA did not lower the power, it raised the work
done per joule by 8%. At a fixed clock and a fixed power budget the 8% comes from the tensor
pipe being busier, which is the Nsight result below.

#### Nsight Compute at 8192³ (`--set full`, one launch, clocks locked by ncu at 2.55 GHz)

| metric | v3 | v5 (64, 2, 1) | v5 (32, 3, 2) |
|---|---|---|---|
| duration | 5.49 ms | 5.05 ms | 5.10 ms |
| tensor pipe active (`sm__pipe_tensor_cycles_active`) | 92.2% | 98.7% | 99.4% |
| executed instructions | 1,251 M | 622 M | 971 M |
| issue slots busy | 13.1% | 7.3% | 11.6% |
| stall: `math_pipe_throttle` (cycles per issue) | 14.3 | 20.8 | 26.6 |
| stall: `barrier` | 8.7 | 0.0 | 0.0 |
| stall: `wait` | 2.7 | 5.0 | 3.7 |
| stall: `long_scoreboard` | 0.3 | 1.3 | 2.1 |
| L2 hit rate | 83.1% | 80.1% | 80.4% |
| DRAM throughput | 31.5% | 39.9% | 39.2% |
| shared-memory bank conflicts / load wavefronts | 297 K / 403 M | 3.7 K / 403 M | 1.9 M / 405 M |
| registers per thread | 128 | 126 | 96 |
| achieved occupancy | 33.0% (15.8 warps/SM) | 18.7% (9.0 warps/SM) | 37.4% (17.9 warps/SM) |

The row that explains the result is the `barrier` stall: 8.7 of v3's 30 cycles between issues
were spent at the per-K-tile `__syncthreads`, and the mbarrier pipeline has none. A warp that
finishes its `mma.sync`s on a stage goes straight to the next stage's `try_wait`, which is
already satisfied when the pipeline is healthy, so the tensor pipe stays fed through the stage
boundary; in v3 the eight warps lined up at the barrier and the pipe drained every 32 k. Half
the instructions are gone with the `cp.async` address generation, and the issue-slot number
(7.3%) says how little the consumer warps now do besides feeding the tensor cores: at one
block per SM there are nine warps on the SM, 2.2 per scheduler, and 98.7% tensor-pipe
activity comes out of that. `math_pipe_throttle` grew from 14 to 21 cycles per issue, which
is what a stall breakdown looks like when the pipe is the only thing left to wait for. The L2
hit rate dropped three points and DRAM throughput rose (each SM now works through the K loop
faster, so more tiles are in flight per unit time and the reuse window in L2 is shorter), and
neither cost anything at 8192³; a grouped tile order would claw that back and is the next
experiment. The 3,658 bank conflicts are effectively zero: the 128-byte-row B boxes are a
cleaner swizzle than v3's 256-byte rows.

#### Verdict

TMA plus a producer/consumer mbarrier pipeline is worth 7 to 9% over v3 on every shape it
runs, and it is the first rung to beat cuBLAS on all of the large shapes: 102 to 109%. It did
not lower the power draw, it raised the work per joule by 8% at the same 600 W and the same
2.75 GHz, by taking the tensor pipe from 92% to 99% active: no block barrier in the k-loop,
half the instructions, one lane issuing what 256 threads used to. The remaining gap to the
239 TFLOPS the card sustains at this clock is 2%, and it sits in wave quantization (2048³ is
1.5 waves) and the split-K tail, not the mainloop. v5 ships as the default for M % 128 == 0,
N % 128 == 0, K % 64 == 0 and a grid of at least 170 tiles; everything smaller and every
decode shape stays on v3, which is where the tile heuristics live.

### Variant 6: the TMA mainloop on the Stream-K schedule

Source: `src/kernels/hgemm_tma_sk.cu`; the schedule it shares with v4 in
`src/kernels/hgemm_streamk.cuh`, the tile it shares with v5 in `src/kernels/hgemm_tma_tile.cuh`.

v4 and v5 each fixed one thing and left the other. v5's mainloop keeps the tensor pipe 99%
busy but runs v3's schedule: one block per tile in row-major order, so at 8192³ its L2 hit
rate is 80% and DRAM runs at 40% of peak, and at 2048³ it pays a 1.5-wave grid. v4's schedule
gets a 96% L2 hit rate and no quantization, but its `cp.async` mainloop is the one v5 beat by
8%, and every piece it computes starts with a cold three-stage prologue, which the v4 notes
listed as the open "chain fixup's links" item. Variant 6 is the two put together. Nothing in
either mainloop or schedule changed; what is new is how a producer warp drives a persistent
loop.

#### Factoring

v4's schedule moved out of `hgemm_streamk.cu` into `hgemm_streamk.cuh` as it was: `Params`,
the grouped raster, the geometric passes, the static ranges, the fragment-order slot layout,
and the host `plan()` that fills the parameters and owns the workspace; v4's kernel includes
the header and its constants are the defaults. v5's stage geometry, swizzles, the producer's
three-box issue and the consumer's per-stage `ldmatrix` + `mma.sync` body moved into
`hgemm_tma_tile.cuh`, and v5 calls them. Both kept their numbers, measured before and after
the move in one session at `--iters=20`: v4 207.0 / 228.0 / 227.9 / 225.4 before and 207.1 /
228.0 / 227.7 / 226.8 after at 2048³ / 4096³ / 8192³ / 4096×11008×4096; v5 180.7 / 241.4 /
237.2 / 236.0 before and 180.8 / 241.1 / 237.7 / 236.0 after.

#### The producer owns the schedule

The block is v5's: 288 threads, eight consumer warps in a 2×4 grid over the 128×128 tile and
one producer warp of which lane 0 does all the work. The grid is v4's: `resident` blocks (170
at one block per SM, the configuration that ships), each walking a sequence of (tile,
kt_begin, kt_end) pieces. Both sides of the block have to walk the same sequence, and in queue
mode the sequence is decided at run time by an `atomicAdd`. So the producer lane owns the
schedule. It walks the block's static range from the top, or takes items from the queue, and
before it issues a piece's first stage it publishes the piece to the consumers through a
two-deep ring of `int4` slots in shared memory. Each ring slot has a "full" mbarrier (count 1,
the producer's arrive after the store) and an "empty" mbarrier (count 8, one arrive per
consumer warp once every lane has read the slot), used exactly like the stage barriers: slot
`i % 2` is used for pieces `i`, `i + 2`, ..., its n-th use waits on parity `(n - 1) & 1` of
"empty". The consumers wait on "full", copy the item into registers, `__syncwarp`, arrive on
"empty", and go. A tile of -1 ends the loop on both sides. In queue mode the producer also
does the last-block-out queue reset after its final failing grab, as v4's thread 0 did.

The stage bookkeeping is one running k-tile counter per side, `g_kt`, that never resets: stage
`g_kt % STAGES` is used for the `g_kt / STAGES`-th time whichever piece the k-tile belongs to,
so the parities line up across a piece boundary and the first `STAGES` fills of the launch are
the only ones that skip the "empty" wait. That is what makes the pipeline stay full between
pieces: the producer is bounded by the "empty" barriers and nothing else, so while the
consumers add a slot, store bf16 or publish a partial for piece `i`, it is already filling
stages for piece `i + 1`, up to `STAGES` k-tiles ahead. The consumers never see a prologue
after the first one. `SPARK_HGEMM_V6_SERIAL=1` puts it back for measurement: the producer
waits on a "done" mbarrier that the consumers arrive on after each epilogue before it issues
the next piece's first stage.

The fixup is v4's, on the consumer side, with a named barrier (`bar.sync 1, 256`) where v4 had
`__syncthreads`, because the producer lane is not part of it. Thread 0 spins on the
predecessor's flag with `ld.acquire.gpu`, the barrier broadcasts the fact, every consumer adds
its lane's `float4`s from the slot with `__ldcg`; a piece that does not end its tile writes the
slot, `__threadfence`, barrier, `st.release.gpu` of the flag. Rows past M are skipped by the
slot traffic and the bf16 store, and zero-filled on the way in by the copy engine (a box row
past the end of the tensor map is zero), so v6 takes any M ≥ 1 like v4 does. The deadlock
argument is v4's unchanged: a piece waits only for the piece before it in K, which was handed
out earlier to a block that is resident and processes its items in order; the ring adds no
cross-block waits, only a bound of one piece between a block's producer and its consumers.

**Proxy fences.** Every stage release keeps v5's `fence.proxy.async.shared::cta` before the
arrive, since `ldmatrix` reads the stage through the generic proxy and the refill lands through
the async one. The persistent loop adds no new fence. The slot reads and writes and the bf16
epilogue go from registers to global memory (`__ldcg`, `float4` and `__nv_bfloat162` stores)
and never touch shared memory, so there is no generic-proxy access to a stage other than
`ldmatrix`, and the item ring is written and read through the generic proxy on both sides and
ordered by its own mbarriers. Every configuration passes the cuBLAS check on every shape,
including the two-block ones that exposed the missing fence in v5.

#### Registers and residency

At one block per SM the kernel takes 160 registers (v5: 126) with no spills; the extra live
state across the k-loop is the piece (tile, k-range, `bm`, `bn`, `m_valid`), the two running
counters and the fixup's pointers. The two-block configurations are capped at 96 registers by
the ninth warp, not by `__launch_bounds__`: 18 warps on four sub-partitions puts five on one of
them, and 5 × 32 × R ≤ 16,384 gives 102, rounded down to 96. v5 fits its k-loop in 96; v6
does not, and `cuobjdump` shows `STL`/`LDL` pairs between the `HMMA`s of the (32, 3, 2) and
(32, 2, 2) kernels. That is why those two trail v5's two-block configuration in the sweep
below, and part of why the one-block configuration ships. (The 63 `STL` without an `LDL` that
every v6 kernel carries are the ABI stores around the 64-bit division subroutine
`range_start` and `pass_range` call once per piece; v5 has 15 of them for its slice
arithmetic.) Shared memory is 65 KB: two 32 KB stages, the ring, and nine barriers.

#### The sweep

`--iters=20`, one session, TFLOPS. First the six pipeline configurations with v4's schedule
constants as they were (static up to four waves when A and B fit in L2, the square-root raster
group, which gives G = 8 on 170 blocks):

| BK, stages, blocks/SM | regs | 2048³ | 4096³ | 8192³ | 4096×11008×4096 | 4096×4096×11008 |
|---|---|---|---|---|---|---|
| 64, 2, 1 | 160 | **224.4** | 238.5 | **241.6** | 239.7 | **243.5** |
| 64, 3, 1 | 160 | 224.5 | 236.1 | 241.5 | **240.1** | 243.4 |
| 32, 3, 2 | 96, spills | 208.2 | 236.0 | 234.5 | 231.7 | 236.6 |
| 32, 2, 2 | 96, spills | 212.8 | **239.6** | 239.2 | 235.4 | 237.7 |
| 32, 3, 1 | 158 | 224.3 | 236.9 | 238.2 | 237.4 | 240.7 |
| 32, 6, 1 | 158 | 219.4 | 237.7 | 238.6 | 236.3 | 241.5 |

Two stages of BK = 64 at one block per SM win or tie everywhere, as they did for v5 on the
DRAM shapes, and here also on the L2-resident ones, where v5's two-block configuration had
won by 2%: the second block's latency hiding no longer pays once the pipeline is never
drained between tiles, and the two-block kernels spill. So (64, 2, 1) ships for every shape,
and `SPARK_HGEMM_V6_CONFIG=<index>` forces one of the six for a re-sweep.

Then the schedule, all on (64, 2, 1). With 170 blocks in flight instead of 340 the constants
v4 settled on were worth a second look, and two of them moved.

*Static or queue.* 4096³ is 1,024 tiles, 6.02 waves of 170, so v4's rule (static up to four
waves when the operands fit in L2) sends it to the queue: 238.5. Static ranges: 246.9 to
249.2. The same at twelve waves: 8192×4096×2048 (A 32 MB, B 16 MB) 236.8 queued, 243.5
static; 4096×8192×2048 236.9 and 241.9. And the same answer as v4 where the operands do not
fit: 4096×11008×4096 (124 MB) 239.5 queued, 233.0 static. So v6 uses static ranges whenever
A and B fit in three quarters of the L2, at any wave count, and the queue with geometric
passes beyond that; v4's two-wave rule for the non-fitting shapes stays. The static schedule
still waits for the slowest SM, as v4 found, but with the operands in L2 that costs less than
the queue's four chain links per tile plus its own tail.

*The raster group.* v4's rule, G about the square root of the blocks in flight, gives 8 on
170 blocks. Sweeping G at 8192³ in queue mode: 1 → 238.1, 4 → 238.9, 8 → 241.4, 16 → 243.2,
32 → 244.2, 64 → 236.3; at 4096×11008×4096: 4 → 237.9, 8 → 239.4, 16 → 241.2, 32 → 242.7. The
arithmetic behind the curve: in queue mode consecutive raster indices are in flight, so a
group's G tile rows of A are re-read once per tile column and stay in L2 while the group
runs, and B is read from DRAM once per group, `tiles_m / G` times in all. At 8192³ a tile
row of A is 2 MB: G = 8 reads B eight times (1 GB), G = 32 twice (256 MB), and G = 64 (all
128 MB of A resident at once) no longer fits, so A is read once per column instead. The rule
that ships: the largest power of two whose group of A rows plus the `grid / G` tile columns
of B in flight fit in three quarters of the L2, else the one with the smallest footprint.
That picks G = 16 at 8192³ (54 MB; G = 32 is 76 MB, just over the 72 MB budget, and
measures the same within noise), G = 32 at 4096×11008×4096 (all of A, 38 MB), G = 16 at
4096×4096×11008 (2.75 MB per tile row, nothing fits, 74 MB is the smallest footprint; G = 8,
16, 32 measured 244.0, 244.7, 244.6). In static mode the blocks in flight are strided through
the raster and G stops mattering: at 4096³ G = 8, 16, 32 measured 246.6 / 248.7, 249.2 /
246.5, 249.2 / 246.5 in two rounds, so the same rule applies to both modes.

*Smaller knobs.* The shortest K-pass: 4 / 8 / 16 k-steps of BK = 64 measured 243.6 / 243.6 /
242.8 at 8192³ and 243.1 / 243.3 / 240.5 at 4096×11008×4096, so v4's 8 stays. The 32 k-step
minimum share per block stays too: 1024³ on the TMA tile (64 tiles, 6 k-steps per block if
forced) runs 0.0193 ms against 0.0173 on v4's 64×64 tile, which is what `launch_auto` hands
it to, and 2560²×4096 at the default runs 242.0 (v5: 219.3). Decode shapes (M ≤ 64) go to
`hgemm_decode.cu` as in v3 and v4; N % 128 ≠ 0 goes to v4's 64-column tiles.

*Does the prologue hiding show up.* Same configuration, `SPARK_HGEMM_V6_SERIAL=1` against
the default, one session:

| shape | pieces per block | serialized | overlapped | gain |
|---|---|---|---|---|
| 2048³ | 2 to 3 of ~48 k-steps | 218.3 | 224.3 | 2.7% |
| 4096³ (queue mode) | ~8 | 236.1 | 238.4 | 1.0% |
| 4096×11008×4096 | ~19 | 239.2 | 241.1 | 0.8% |
| 8192³ | ~27 | 240.7 | 241.3 | 0.2% |

A cold prologue is two 32 KB stages from L2, about a microsecond, and a block at 2048³ pays
it two or three times in a 77 µs kernel; at 8192³ it is 27 times in 4.5 ms. The overlap is
worth what that arithmetic says it should be, and it is the whole of v6's margin over v4 at
2048³ beyond the mainloop.

#### Measured

`bench_hgemm --iters=50`, v4, v5, v6 and cuBLAS from one session. TFLOPS, and % is cuBLAS
time over ours from the cuBLAS loop timed next to that variant.

| M×N×K | cuBLAS ms / TFLOPS | v4 ms / TFLOPS / % | v5 | v6 | v6 / v5 | v6 / v4 |
|---|---|---|---|---|---|---|
| 1024³ | 0.0176 / 122.0 | 0.0174 / 123.1 / 101.1% | n/a | 0.0173 / 124.1 / 101.7% | | 1.00× |
| 2048³ | 0.0995 / 172.7 | 0.0829 / 207.3 / 120.0% | 0.0951 / 180.7 / 104.5% | **0.0766 / 224.4 / 129.9%** | 1.24× | 1.08× |
| 4096³ | 0.6090 / 225.7 | 0.5947 / 231.1 / 102.4% | 0.5658 / 242.9 / 107.6% | **0.5517 / 249.1 / 110.4%** | 1.03× | 1.08× |
| 8192³ | 4.7284 / 232.5 | 4.8258 / 227.8 / 98.0% | 4.6540 / 236.3 / 101.6% | **4.5164 / 243.5 / 104.6%** | 1.03× | 1.07× |
| 4096×4096×11008 | 1.6267 / 227.1 | 1.6064 / 229.9 / 101.3% | 1.5116 / 244.4 / 107.5% | **1.5036 / 245.7 / 108.1%** | 1.01× | 1.07× |
| 4096×11008×4096 | 1.5428 / 239.4 | 1.6084 / 229.7 / 95.9% | 1.5449 / 239.1 / 99.9% | **1.5014 / 246.0 / 102.8%** | 1.03× | 1.07× |

The 1024³ row and the six decode rows run the same kernels under v6 as under v4 (v4's 64×64
tile and the weight-streaming kernel) and measure the same within the 2 µs grid the single
launches fall on: 23.5 µs at 1, 16 and 32×4096², 25.2 (v4: 23.7) at 64×4096², 56.3 and 60.3 µs
on the 11008 shapes. 4096×11008×4096 was the shape the v4 notes had left at 95%; it is now
102.8%, from the mainloop (v5 got it to 99.9%) and the grouped order together. All 21 rows
pass the cuBLAS check, the ragged shapes (1000×4096×4096, 2064×11008×2048, 2100×11008×4096)
pass through the same binary, and `tests/test_gemm.py` checks bitwise equality across six
calls on four of the shapes, static and queue mode both.

#### Power

Same method as v5's: `nvidia-smi --query-gpu=clocks.sm,power.draw -lms 200` during a
1,500-iteration 8192³ loop of each kernel, both inside one GPU lock, samples from the last
five seconds of each run:

| kernel | TFLOPS (median of 1,500) | SM clock | power | TFLOPS per watt |
|---|---|---|---|---|
| cuBLAS (same runs) | 226.7 to 227.1 | 2,715 to 2,722 MHz | 600 W | 0.378 |
| v5 | 234.6 | 2,752 MHz | 600 W | 0.391 |
| v6 | 241.7 | 2,805 MHz | 600 W | 0.403 |

Both kernels sit at the 600 W limit, and this time the clock moved: 2,805 MHz for v6 against
2,752 for v5, the 25 samples of each window steady to the MHz. v6 moves 2.8 GB less through
DRAM per 8192³ GEMM (next table), and at a fixed power budget the memory system's share of
the watts went into SM clock instead. That is the reading the v4 notes gave for its own 1.8%
at 8192³, now with the clock measured rather than inferred.

#### Nsight Compute at 8192³ (`--set full`, second launch, caches warm, clocks locked by ncu)

| metric | v5 (64, 2, 1) | v6 (64, 2, 1) |
|---|---|---|
| duration | 5.05 ms | 5.01 ms |
| grid | 4,240 blocks | 170 blocks |
| tensor pipe active (`sm__pipe_tensor_cycles_active`) | 98.7% | 99.3% |
| L2 hit rate | 80.1% | 95.8% |
| DRAM throughput, % of peak | 39.9% | 9.0% |
| DRAM traffic over the kernel (throughput × duration) | 3.6 GB | 0.8 GB |
| L2 read sectors from the SMs | 537 M (17.2 GB) | 538 M (17.2 GB) |
| executed instructions | 622 M | 697 M |
| issue slots busy | 7.4% | 8.5% |
| stall: `math_pipe_throttle` (cycles per issue) | 20.7 | 14.6 |
| stall: `wait` | 5.0 | 4.6 |
| stall: `long_scoreboard` | 1.3 | 2.2 |
| stall: `barrier` | 0.0 | 0.0 |
| shared-memory bank conflicts / load wavefronts | 3.1 K / 403 M | 39 K / 403 M |
| registers per thread | 126 | 160 |
| achieved occupancy | 18.7% (9.0 warps/SM) | 18.7% (9.0 warps/SM) |

The SMs read the same 17.2 GB from L2 either way; the grouped order turns 3.6 GB of DRAM
traffic into 0.8 GB, and the L2 hit rate goes from 80% to 96%, v4's number with v5's
mainloop. At the profiler's fixed clock that is worth 0.9% (5.05 to 5.01 ms); the other 2%
in the benchmark is the 53 MHz of clock above. The tensor pipe is at 99.3%, and
`math_pipe_throttle` fell from 20.7 to 14.6 cycles per issue because the consumers now spend
part of their issue budget on the fixup and the ring (75 M more instructions over 4,654
pieces) rather than because the pipe is less fed. The 39 K bank conflicts are the fixup's
`bar.sync` region and the item ring, 0.01% of the wavefronts. There is still no `barrier`
stall: the only `__syncthreads` is the one after the barrier init.

#### Verdict

The two rungs compose without a compromise. v6 is 7 to 8% over v4 on every large shape and
1 to 3% over v5 on the shapes v5 ran, 24% at 2048³ where v5 paid for a 1.5-wave grid, and
103 to 130% of cuBLAS everywhere from 1024³ up, the same bits every run, no memset, any M.
The mechanism that made it work is small: the producer lane owns the schedule and tells the
consumers what it is loading through a two-slot ring, and the stage counters do not reset.
Everything else is v4 and v5 unchanged, which is what factoring them into headers was for.
The clock told the last part of the story: at 600 W the card runs this kernel 53 MHz faster
than v5 because the DRAM is nearly idle.

### Fused epilogues: bias, activation, residual, and the gate/up pair

Source: `src/kernels/hgemm_epilogue.cuh`, applied in `hgemm_tma_sk.cu` (v6),
`hgemm_tile.cuh` (v4's tiles) and `hgemm_decode.cu`; the options are the `HgemmEpilogue`
argument of `hgemm_bf16` in `include/spark/kernels.h`, and `hgemm(a, b, bias=, act=,
residual=, out=)` and `hgemm_swiglu(a, w_gate_up)` in Python.

Every rung so far stores `C = A B` as bf16 straight from the accumulators, and a decoder
block then runs more kernels over C: a bias add on some models, the residual add after
`o_proj` and `down_proj`, and `silu(gate) * up` between the MLP's two projections and its
third. Each of those is a pass over an activation-sized tensor, and at decode sizes each is
a launch on top of a 23 us GEMM. The accumulators are fp32 and already in registers when
the GEMM ends, so the epilogue can do all of it before the one rounding to bf16:

```
x = acc + bias[col]                      bias: [N] bf16, per column of B
y = act(x)                               none, silu, gelu (tanh form), relu
y = silu(x[2j]) * x[2j+1]                swiglu: instead of act, see below
C = bf16(y + residual[row][col])         residual: [M][N] bf16; residual == C is C += ...
```

The default `HgemmEpilogue` is the plain store and every existing call is unchanged. The
fused options run on variants 4 and 6 (which share the code), and a request for one on
another variant, or on a shape those two refuse, throws `std::invalid_argument`, a
`ValueError` from Python. Variant 6 routes some shapes to variant 4's tiles and the decode
shapes to `hgemm_decode.cu`, so the epilogue lives in each of those and in the Stream-K
fixup, where the piece that ends a tile's chain applies it after adding the slot: the
partials that pass through the slot are plain fp32 sums, so bias, activation and residual
happen once, on the finished sum, and the output is still the same bits every run
(`test_hgemm_epilogue_is_deterministic` checks queue mode). In the decode kernel the storing
thread applies it: the block itself when K is not split, the last K-slice to arrive at a
strip when it is, from the fp32 workspace.

#### The interleaved gate/up layout

HF Llama stores `gate_proj` and `up_proj` as two `[N][K]` weights; a model runs two GEMMs
over the same A and a third kernel over the two results. One GEMM over a `[K][2N]` weight
reads A once, but the tile that owns gate column j has to see up column j in the same
epilogue, and with the two halves side by side those columns are 11008 apart, in another
tile. So the weight is permuted once at load time: column `2j` is `gate_j` and column
`2j+1` is `up_j` (`spark_kernels.interleave_gate_up(w_gate, w_up)`, a `stack` and a
`reshape`; the same helper interleaves a bias). The `mma.sync` accumulator layout gives each
lane columns `(c, c+1)` of an n8 tile, an adjacent pair, so gate and up of the same output
element land in one lane's register pair whatever the block or warp tile, and the epilogue
forms `silu(g) * u` in fp32 without touching shared memory. The output has N columns for a
2N-wide B. The only cost of the pair layout is the store width: each lane ends up with one
output element per row instead of two adjacent ones. The lanes `2q` and `2q+1` of a quad
hold columns `j` and `j+1` of the same two rows, so one `__shfl_xor_sync(1)` per fragment
gives the even lane the odd lane's upper-row value and the odd lane the even lane's
lower-row value, and each stores a whole `bfloat162`: the same 4-byte stores as the plain
epilogue, half as many of them.

#### Shape of the code, and the three things that made it cheap

The first version applied the options per fragment: a runtime `switch` on the activation
and an `if (swiglu)` inside the sixteen unrolled fragments of a warp tile, each fragment's
residual loaded right before its store, and `__frcp_rn` for the sigmoid. It passed every
shape and cost, at 4096^3 on v6, 33 us for `relu` alone, 45 for `gelu`, 81 for `residual`
and 100 for the in-place accumulate, on a 556 us GEMM; the separate elementwise pass it was
meant to replace costs 31 us. The v6 block runs nine warps per SM, so nothing hides latency
behind other warps, and three things were exposed:

* **Instruction fetch.** The per-element switch inlined every activation into every
  fragment, a straight-line epilogue of about 10 K SASS instructions with the taken path
  threaded through all of it. The shipped version makes the activation and the SwiGLU flag
  template parameters of the warp-tile loop (`store_warp_tile_mode<Mode<ACT, SWIGLU>>`) and
  picks the instantiation with one uniform branch per tile (`dispatch`), so the executed
  path is one compact loop. `relu` went from 33 us to 6.
* **One DRAM round trip per fragment.** Each fragment's residual load was followed by its
  store before the next fragment's load. Now the bias of the lane's four column pairs and
  the residual of all sixteen fragments are loaded into registers first (`Pre`, 40
  registers), and in v6 that happens before the k-loop of any piece that will store its
  tile (the piece knows: `ke == KT`), so the round trip hides under the mma work. The
  fused v6 kernel takes 167 registers against the plain one's 160 with no spills, at one
  block per SM where the budget is 224. `residual` went from 81 us to 20 with the loads
  hoisted to the top of the epilogue, and to 13 with them above the k-loop. Variant 4's
  tile cannot do that: it runs two blocks per SM at 121 to 128 registers, and 40 more live
  across its mainloop would cost the second block, so it loads at the store and pays the
  round trip (18 us at 1000x4096x4096 when variant 4 is asked for by name; the default
  route takes that shape through v6 for 2.4 us). The shapes v6 hands to variant 4's tiles
  are the small and the N % 128 != 0 ones, where the tile is a few microseconds.
* **A subroutine per element.** `__frcp_rn` (and a plain `/`) is the correctly rounded
  reciprocal, which ptxas lowers to `MUFU.RCP`, two Newton steps and a slow-path `CALL`;
  sixty-four of those per lane serialized the loop, and `gelu` still cost 57 us after the
  first two fixes. `silu` is `__fdividef(x, 1 + __expf(-x))`, `MUFU.EX2`, `MUFU.RCP` and a
  multiply, and the tanh-form `gelu` is the same function of `2u` with
  `u = sqrt(2/pi) (x + 0.044715 x^3)`, since `0.5 x (1 + tanh(u)) = x sigmoid(2u)`. Both
  are within 2 ulp of fp32, far under the bf16 rounding that follows.

The plain instantiation is a separate template so the plain path is unchanged; its 4096^3,
2048^3 and decode numbers were re-measured after the change (246.5 TFLOPS, 224.3, 23.5 us).

#### Traffic arithmetic

What the fusion takes off the bus, per call, for an `[M][N]` output in bf16:

| form | unfused | fused | removed |
|---|---|---|---|
| bias, act, residual | GEMM writes C (2MN), the pass reads C (2MN) and the residual (2MN), writes out (2MN) | reads the residual (2MN), writes out (2MN) | 4MN bytes, one launch |
| swiglu, from two `[K][N]` weights | two GEMMs read A twice, write gate and up (4MN), swiglu reads both (4MN), writes out (2MN) | reads A once, writes out (2MN) | 8MN bytes + one read of A, two launches |

At 4096x4096 that is 67 MB per call for the first form and 360 MB for the SwiGLU form at
4096x11008 (plus 32 MB of A, mostly from L2). The written C of the unfused sequence is
still in L2 when the pass reads it (32 MB against 96 MB), so on the prefill shapes the
removed bytes are mostly L2 traffic and the saving is the pass's own time; on the decode
shapes the output is a few hundred KB and the saving is the launch.

#### Measured

`bench_hgemm --iters=50` with the epilogue flags, one session, variant 6 (the default
route: the TMA tile for the 4096-row shapes, the decode kernel for the 16-row ones). "plain"
is the same GEMM with the plain store; "unfused ours" is the plain GEMM followed by a
separate elementwise pass (bias, activation, residual) or, for SwiGLU, two GEMMs over the
un-interleaved halves and `swiglu_bf16`; "unfused cuBLAS" is the same sequence with
`cublasGemmEx`. "torch" is eager PyTorch from `scripts/bench_torch.py` (`F.gelu(F.linear(a,
b.t(), bias))`, `torch.addmm(res, a, b)`, `F.silu(a @ g) * (a @ u)`), timed next to the
fused call through the extension. Times in microseconds; "removed" is the DRAM traffic the
fusion takes off the bus per call (the table above).

| M x N x K | epilogue | plain | fused | epilogue cost | unfused ours | unfused cuBLAS | saved vs unfused ours | torch | fused vs torch | removed |
|---|---|---|---|---|---|---|---|---|---|---|
| 4096 x 4096 x 4096 | bias + gelu | 563.8 | 574.0 | +10.2 | 585.3 | 628.4 | 11.3 (2%) | 688.5 | 1.19x | 67 MB |
| 4096 x 4096 x 4096 | residual | 565.3 | 578.0 | +12.7 | 600.4 | 641.2 | 22.4 (4%) | 713.1 | 1.24x | 67 MB |
| 4096 x 4096 x 4096 | accumulate (C += A B) | 565.4 | 574.1 | +8.7 | 598.8 | 641.1 | 24.6 (4%) | | | 67 MB |
| 4096 x 4096 x 4096 | bias + silu + residual | 565.8 | 583.7 | +17.9 | 602.7 | 641.0 | 19.0 (3%) | | | 67 MB |
| 4096 x 11008 x 4096 | bias + gelu | 1521.2 | 1545.1 | +23.9 | 1620.5 | 1647.2 | 75.4 (5%) | 2021.7 | 1.30x | 180 MB |
| 4096 x 11008 x 4096 | residual | 1522.0 | 1545.8 | +23.8 | 1675.9 | 1697.5 | 130.1 (8%) | 1964.3 | 1.26x | 180 MB |
| 4096 x 22016 x 4096 | swiglu, C is [4096][11008] | 3044.5 | 3038.6 | -5.9 | 3221.1 | 3319.4 | 182.5 (6%) | 3481.3 | 1.13x | 361 MB + A |
| 16 x 4096 x 4096 | bias + gelu | 23.5 | 23.5 | 0.0 | 24.4 | 30.7 | 1.0 (4%) | 35.2 | 1.34x | 0.3 MB |
| 16 x 4096 x 4096 | residual | 23.6 | 23.5 | -0.1 | 23.6 | 30.8 | 0.2 (1%) | 33.2 | 1.27x | 0.3 MB |
| 16 x 4096 x 4096 | bias + silu + accumulate | 23.6 | 23.5 | -0.1 | 25.5 | 30.9 | 2.0 (8%) | | | 0.3 MB |
| 16 x 11008 x 4096 | bias + gelu | 56.2 | 56.4 | +0.2 | 58.2 | 61.7 | 1.8 (3%) | 68.1 | 1.15x | 0.7 MB |
| 16 x 11008 x 4096 | residual | 56.3 | 56.4 | +0.1 | 58.4 | 63.4 | 2.0 (4%) | 66.1 | 1.12x | 0.7 MB |
| 16 x 22016 x 4096 | swiglu, C is [16][11008] | 113.5 | 113.3 | -0.2 | 114.5 | 120.2 | 1.2 (1%) | 123.9 | 1.08x | 1.4 MB + A |

Every row passes the check against the fused math on cuBLAS's fp32 result (2% of max|ref|
plus 1e-3, the plain rows' form), including the in-place accumulate, which the bench runs
from a fresh copy of the residual.

What the numbers say:

* On the prefill shapes the epilogue costs 9 to 24 us on the TMA tile, 1.5 to 3% of the
  GEMM, and that is the cost of a heavier store: the residual read is 32 MB from DRAM at
  4096^2 (21 us at the copy rate if nothing hid it; the prefetch above the k-loop hides most
  of it) and the bias and activation are a few instructions per element. The saving is the
  separate pass minus that: 11 to 25 us at 4096^2, 75 to 130 at 4096 x 11008. Against
  eager PyTorch it is 1.2 to 1.3x; `torch.addmm(res, a, b)` is one cuBLAS call with beta = 1,
  but torch copies the residual into the output first, so it moves the same bytes as our
  unfused pair and then some.
* The SwiGLU form is free at prefill: 3038.6 fused against 3044.5 plain is within the run
  to run noise, because the epilogue stores half the bytes of the plain [M][2N] output and
  the 45 M `silu` calls are a few microseconds of SFU work spread over 170 SMs. What it
  saves is the two-GEMM, one-kernel sequence a model runs today: 183 us (6%) against ours,
  281 against cuBLAS's GEMMs, 443 (13%) against eager torch. One read of A instead of two
  is part of that: 32 MB, mostly from L2.
* On the decode shapes the epilogue costs nothing measurable: the kernel is the weight
  stream and the output is a few hundred KB. The saving is the launch that is gone, 1 to 2
  us per call on 23.5 to 113 us (1 to 8%), and 1.1 to 1.3x against eager torch, whose add
  and activation are two launches more. A decoder layer has four of these GEMMs, so a fused
  Llama-7B layer at 16 tokens is about 6 us and four launches shorter per layer, 32 layers
  deep.

## Correctness

Inputs uniform in [-1, 1]. cuBLAS (`cublasGemmEx`, `CUBLAS_COMPUTE_32F`, bf16 in/out) is the
reference. Both results are a single bf16 rounding of an fp32 sum, so the check is
`max|C − C_ref| ≤ 0.02·max|C_ref| + 1e-3`, which admits one bf16 ulp on each side plus fp32
summation-order noise. The split-K tail tiles are covered by the same check (their sum is fp32
throughout, rounded once at the end).

## RTX 5090 notes (measured)

The constants (128×128×32 block tile, 8 warps per tile) were reasoned for 48 SMs, 273 GB/s and
a 24 MB L2. They turned out to be the right tile for the 5090 too; what the 5090 needed on top
was the mma.sync/ldmatrix rung and the tail scheduling.

- **The roof.** `bench_peak` (`results/peak.json`) measures a dense bf16 `mma.sync` peak of
  **258.7 TFLOPS at 2,976 MHz** with register-resident operands. That is the number the "% of
  peak" column uses. But a sustained GEMM hits the card's 600 W power limit within a
  millisecond or two and settles at 2.72–2.78 GHz (`nvidia-smi` sampled during the 8192³
  loop: 600 W, throttle reason 0x4), so the *practical* roof for a long GEMM is ≈ 239 TFLOPS.
  cuBLAS lands there: 233–241 TFLOPS on the large shapes. The remaining gap between v3 and
  cuBLAS at 8192³ (220 vs 233) is at least partly a perf-per-watt gap, not an instruction one:
  Nsight (which runs at a fixed 2.55 GHz) shows v3's tensor pipe **94.9% active** with the
  dominant stall `math_pipe_throttle`, i.e. the tensor cores are the bottleneck and the kernel
  is issuing to them as fast as they take work.
- **Memory side.** 64 FLOP/byte at 1,792 GB/s would allow 115 TFLOPS from DRAM alone; the
  measured 96 MB L2 (96.5% hit rate at 8192³, 81.6% in the `--set full` run with its cold
  replays) is what makes 220+ possible. DRAM throughput during v3 is ~5% of peak. `BK = 64` was
  tried and lost (see the sweep), so the smem-side intensity of `BK = 32` is enough here.
- **What each rung bought at 8192³** (TFLOPS, % of cuBLAS): v0 24.0 (10%), v1 168.1 (72%),
  v2 202.9 (87%), v3 220.0 (95%). The `cp.async` pipeline (v1→v2) was worth 21%, about as much
  as I expected it to be worth on the GB10; the tensor cores got faster by more than the bus
  did, so there was still plenty of copy to hide. In the `--set full` profiles v2 is 88.6%
  tensor-pipe utilized at 124 registers and 33% occupancy; v3 91.2% at 121 registers.
- **Small shapes.** With the 128-row tile 1024³ was 64 tiles on 170 SMs and v3 reached 52% of
  cuBLAS (63 vs 122 TFLOPS). The 64×64×64 tile makes it 256 tiles and **100.9%** (122.5
  TFLOPS). 2048³ is 256 tiles of 128×128, under one wave, and v2 and v3 both sit at 96%; the
  64×128 tile does not beat that (the sweep above), so it stays as is.
- **Decode shapes stream B at 1,430–1,610 GB/s** against a 1,532 GB/s `cudaMemcpy` roof,
  with B rotated past L2, on the dedicated kernel (`hgemm_decode.cu`, the "Decode" subsection):
  16×4096×4096 in 23.5 µs (126.8% of cuBLAS), 32×4096×4096 in 23.6 µs (125.5%), 64×4096×4096
  in 23.7 µs (128.4%), 16×11008×4096 in 56.4 µs (107.5%), 64×4096×11008 in 59.7 µs (106.7%),
  and the single-token 1×4096×4096 in 23.5 µs (169.5%). 23.5 µs is the floor of one 32 MB
  launch timed this way (a read-only probe takes 23.6); back to back the same launches take
  21.5 µs, 1,574 GB/s. The first run's 64×64×64 tile was at 1,160–1,470 GB/s: two memsets
  and a 128-CTA grid on a 28 µs problem.
- **Timing hygiene.** `min_ms` and the median agree to within 1% on every row, so the 300 ms
  clock ramp and per-loop warmup are doing their job; the power-limit clock drop is a
  steady state, not jitter. Nsight Compute needs `sudo` on this GeForce card.

## Results (RTX 5090, sm_120, CUDA 13.2, driver 595.58)

From `results/hgemm.json`, run 2026-09-28 (median of 50 iterations; cuBLAS `cublasGemmEx`
timed identically on the same stream, next to each variant; for M ≤ 64 both stream B from DRAM
through copies that exceed L2). v2 has no rows for the decode shapes (n/a): it requires
M % 128 == 0. v5 takes grids of at least one 128×128 tile per SM, so it has no rows below
2048³ or on the decode shapes. Bold marks the fastest rung per row. A GB10 (sm_121) table is
added when the Spark has been benchmarked.

| shape (M×N×K) | cuBLAS ms / TFLOPS | v0 ms / TFLOPS / % | v1 | v2 | v3 | v4 | v5 | v6 |
|---|---|---|---|---|---|---|---|---|
| 1024³ | 0.0176 / 122.0 | 0.0706 / 30.4 / 25.0% | 0.0443 / 48.5 / 40.0% | 0.0400 / 53.7 / 44.3% | 0.0175 / 122.7 / 101.1% | 0.0175 / 122.5 / 100.9% | n/a | **0.0172 / 124.5 / 102.0%** |
| 2048³ | 0.0995 / 172.7 | 0.6563 / 26.2 / 15.2% | 0.1357 / 126.6 / 73.3% | 0.1035 / 166.0 / 96.1% | 0.1034 / 166.1 / 96.2% | 0.0828 / 207.5 / 120.2% | 0.0950 / 180.8 / 104.7% | **0.0766 / 224.2 / 129.8%** |
| 4096³ | 0.6092 / 225.6 | 5.4155 / 25.4 / 11.2% | 0.8998 / 152.7 / 67.7% | 0.7543 / 182.2 / 80.7% | 0.5946 / 231.1 / 102.1% | 0.5926 / 231.9 / 102.7% | 0.5599 / 245.5 / 108.5% | **0.5496 / 250.1 / 110.8%** |
| 8192³ | 4.7284 / 232.5 | 45.7476 / 24.0 / 10.3% | 6.5091 / 168.9 / 72.6% | 5.4237 / 202.7 / 87.1% | 4.8770 / 225.5 / 97.0% | 4.8271 / 227.8 / 98.0% | 4.6495 / 236.5 / 101.6% | **4.5161 / 243.5 / 104.7%** |
| 4096×4096×11008 | 1.6206 / 227.9 | 15.7426 / 23.5 / 10.3% | 2.3957 / 154.2 / 67.7% | 2.0384 / 181.2 / 79.5% | 1.6132 / 229.0 / 100.6% | 1.6062 / 230.0 / 100.8% | 1.5082 / 244.9 / 107.4% | **1.5034 / 245.7 / 107.8%** |
| 4096×11008×4096 | 1.5428 / 239.4 | 15.2491 / 24.2 / 10.1% | 2.2089 / 167.2 / 69.7% | 1.8457 / 200.1 / 83.7% | 1.6043 / 230.2 / 96.2% | 1.6063 / 229.9 / 96.0% | 1.5363 / 240.4 / 100.4% | **1.4997 / 246.3 / 102.9%** |
| 1×4096×4096 | 0.0400 / 0.8 | n/a | n/a | n/a | **0.0234 / 1.4 / 170.6%** | 0.0234 / 1.4 / 170.9% | n/a | 0.0234 / 1.4 / 170.8% |
| 16×4096×4096 | 0.0286 / 18.7 | 0.1051 / 5.1 / 27.2% | 0.1772 / 3.0 / 16.8% | n/a | **0.0234 / 22.9 / 123.2%** | 0.0234 / 22.9 / 123.4% | n/a | 0.0235 / 22.9 / 122.1% |
| 32×4096×4096 | 0.0298 / 36.0 | 0.1075 / 10.0 / 26.7% | 0.1771 / 6.1 / 16.8% | n/a | 0.0234 / 45.9 / 126.1% | **0.0234 / 46.0 / 123.6%** | n/a | 0.0235 / 45.8 / 127.0% |
| 64×4096×4096 | 0.0308 / 69.6 | 0.1075 / 20.0 / 28.6% | 0.1772 / 12.1 / 17.3% | n/a | **0.0239 / 89.7 / 124.9%** | 0.0245 / 87.6 / 126.0% | n/a | 0.0252 / 85.2 / 122.3% |
| 16×11008×4096 | 0.0606 / 23.8 | 0.1157 / 12.5 / 52.4% | 0.1812 / 8.0 / 33.4% | n/a | **0.0562 / 25.7 / 107.7%** | 0.0563 / 25.6 / 107.7% | n/a | 0.0563 / 25.6 / 107.7% |
| 64×4096×11008 | 0.0643 / 89.8 | 0.2836 / 20.3 / 22.7% | 0.4717 / 12.2 / 13.6% | n/a | **0.0584 / 98.9 / 110.5%** | 0.0585 / 98.7 / 110.5% | n/a | 0.0584 / 98.9 / 110.1% |

"%" is cuBLAS time ÷ our time, against the cuBLAS loop timed next to that variant (the
cuBLAS medians of the decode rows move by a few percent between variants within a run). The
decode rows run the same weight-streaming kernel under v3, v4 and v6, in GB/s
(2(MK + KN + MN) ÷ time) for v6: 1,429, 1,440, 1,455, 1,383, 1,611 and 1,576, against
1,532 GB/s for `cudaMemcpy`. The PyTorch eager comparison (`torch.matmul` in bf16, which
calls cuBLAS/cuBLASLt through its own heuristics) is in `results/torch_comparison.json` and
`docs/RESULTS.md`: with the default variant (v6) 1.12× at 4096³,
1.07× at 8192³, 1.28× at 2048³,
1.25× on the 16-row decode shape, and 0.87× at 1024³, where
the 17 µs kernel is timed through the extension's host path and torch's is not.

## What was done and what remains

Done in v3, from the list I had written before the first run:

- WMMA → raw `mma.sync.m16n8k16` + `ldmatrix`, explicit fragment layout, fewer and wider
  shared-memory instructions.
- Dynamic shared memory with a 3-stage pipeline. Larger `BK` (64) was tried and did not help.
- XOR-swizzled smem instead of padding.
- Split-K, in the tail-only form that the wave arithmetic on 170 SMs actually calls for.
- Small-shape tiles: 64×128 and 64×64×64 picked per call, which took 1024³ from 52% to 101% of
  cuBLAS.
- Decode shapes: zero-filled rows past M, so M = 16..64 ran on the same kernel at 97–105% of
  cuBLAS with the weights streamed from DRAM; then the dedicated decode kernel (one CTA per
  column strip, 16/32/64-row tiles, no memsets, any M ≥ 1) at the single-launch floor,
  107–170% of cuBLAS.

Done in v4:

- Stream-K proper, on a persistent grid, with a memset-free and bitwise-deterministic
  fixup. It did not replace the per-call tile heuristic: a 12 k-step share per block at
  1024³ on the 128×128 tile costs more in fixup than it saves, so the tile is still picked
  per call, now by the per-block share of k-steps rather than by the wave count.
- Grouped tile order: 8192³ from 664 to 165 GB/s of DRAM traffic and from 94.9% to 98.0% of
  cuBLAS.
- Decode shapes at 1,350 to 1,550 GB/s: the two memsets and the launch between them were
  4 µs of a 28 µs call.

Done in v6:

- The TMA mainloop on the Stream-K schedule, with the producer lane owning the schedule and
  the stage counters running across pieces: 8192³ from 80% to 96% L2 hit with the 99% tensor
  pipe, 2048³ from 207 to 224 TFLOPS, 4096³ from 243 to 249, 4096×11008×4096 from 95% (v4) and
  100% (v5) of cuBLAS to 103%.
- The prologue-hiding item from the v4 list: measured at 2.7% at 2048³, 1% at 4096³ and 0.2%
  at 8192³ against a serialized producer. The slot traffic stays, as expected.
- Two schedule constants re-swept for 170 blocks in flight: static ranges at any wave count
  when the operands fit in L2, and the L2-footprint raster group.

Done after v6, on the same kernels ("Fused epilogues" above):

- Bias, activation, residual (or in-place accumulate) and the interleaved gate/up SwiGLU
  applied in fp32 from the accumulators, in the TMA tile, the Stream-K finisher, variant 4's
  tiles and the decode kernel, with the plain instantiations untouched. The separate passes
  and their launches are gone from the decoder block's GEMMs.

Still open:

- **The two-block TMA configurations spill.** A ninth warp caps two blocks per SM at 96
  registers, and the persistent loop's extra live state (the piece, two counters, the fixup's
  pointers) pushes the k-loop over it. A 4-consumer-warp layout with a 64×64 warp tile would
  free registers and halve the `ldmatrix` count per `mma.sync`, and it is cheaper to try with
  the producer warp in place than it was in v3.
- **A 64-row TMA tile.** v6 hands 1024³, ragged small grids and N % 128 ≠ 0 to v4's 64-row
  `cp.async` tiles; the TMA tile forced onto 1024³ measured 111 TFLOPS against 124, because a
  6 k-step share per block is all fixup. A 64×64 TMA box on the same ring would let the
  Stream-K TMA kernel take those shapes on its own terms.
- **The 2 µs of launch and ramp** that a single decode launch pays over the back-to-back
  rate (23.5 against 21.5 µs at 32 MB). It is not in the kernel; CUDA graphs or a persistent
  megakernel across the decoder block are the ways at it.
- **Perf per watt** on the long shapes. v5 raised the work per joule 8% from the tensor pipe,
  v6 another 3% from the DRAM traffic (2,805 against 2,752 MHz at 600 W). What is left at
  8192³ is a 99.3% busy tensor pipe at whatever clock 600 W buys; the 64×64 warp tile above
  is the remaining instruction-side lever.
