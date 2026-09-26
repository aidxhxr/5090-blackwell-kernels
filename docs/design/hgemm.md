# HGEMM: bf16 tensor-core GEMM

`C[M,N] = A[M,K] · B[K,N]`, row-major, bf16 inputs and outputs, fp32 accumulation.
Source: `src/kernels/hgemm.cu`. Bench: `bench_hgemm` (validates every variant against cuBLAS).

## Why WMMA / mma.sync on these GPUs

Both targets are the consumer/workstation Blackwell lineage: the RTX 5090 is compute capability
12.0 (sm_120, primary target) and the GB10 is 12.1 (sm_121, secondary). They have
fifth-generation tensor cores driven by the classic `mma.sync` warp-level instruction, which is
what the WMMA C++ API compiles to. The datacenter Blackwell parts (sm_100, B200/GB200) add
`tcgen05` instructions and thread-block clusters; those do not exist on either machine (TMA,
`cp.async.bulk.tensor`, does exist on sm_120, but with `cp.async` already hiding the copies
behind the tensor cores it is not used here). So `mma.sync` is not a compromise, it is the
native path, and the same source builds for both.

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
| 3 | raw `mma.sync.m16n8k16` + `ldmatrix`, XOR-swizzled smem, 3-stage `cp.async` pipeline, register-direct epilogue, split-K on the last partial wave, tile picked per call (128×128, 64×128 or 64×64) with zero-filled rows past M, and a dedicated weight-streaming kernel with a 16, 32 or 64-row tile for M ≤ 64 (`hgemm_decode.cu`) | fragment loads in one instruction each, no padding bytes, DRAM latency covered two tiles ahead, no epilogue staging, the wave-quantization tail on 170 SMs, and small / decode shapes (any M % 16 == 0) that a 128-row tile could not fill the card with |

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

From `results/hgemm.json` (median of 50 iterations; cuBLAS `cublasGemmEx` timed identically on
the same stream; for M ≤ 64 both stream B from DRAM through copies that exceed L2). v2 has no
rows for the decode shapes (n/a): it requires M % 128 == 0. A GB10 (sm_121) table is added when the
Spark has been benchmarked.

| shape (M×N×K) | cuBLAS ms / TFLOPS | v0 ms / TFLOPS / % | v1 | v2 | v3 |
|---|---|---|---|---|---|
| 1024³ | 0.0177 / 121.6 | 0.0707 / 30.4 / 25.0% | 0.0442 / 48.6 / 40.1% | 0.0400 / 53.7 / 44.2% | **0.0175 / 122.5 / 100.9%** |
| 2048³ | 0.0996 / 172.6 | 0.6581 / 26.1 / 15.1% | 0.1344 / 127.9 / 74.0% | 0.1035 / 166.0 / 96.2% | **0.1035 / 166.1 / 96.3%** |
| 4096³ | 0.6091 / 225.7 | 5.4217 / 25.4 / 11.2% | 0.9091 / 151.2 / 67.0% | 0.7564 / 181.7 / 80.5% | **0.6111 / 224.9 / 99.7%** |
| 8192³ | 4.7306 / 232.4 | 45.7073 / 24.1 / 10.3% | 6.5359 / 168.2 / 72.5% | 5.4208 / 202.8 / 87.3% | **4.9877 / 220.4 / 94.9%** |
| 4096×4096×11008 | 1.6267 / 227.1 | 15.7354 / 23.5 / 10.3% | 2.3978 / 154.0 / 67.8% | 2.0425 / 180.8 / 79.7% | **1.6433 / 224.8 / 98.8%** |
| 4096×11008×4096 | 1.5427 / 239.4 | 15.2551 / 24.2 / 10.1% | 2.2055 / 167.5 / 70.0% | 1.8501 / 199.6 / 83.5% | **1.6453 / 224.5 / 93.8%** |
| 1×4096×4096 | 0.0399 / 0.8 | n/a | n/a | n/a | **0.0235 / 1.4 / 169.5%** |
| 16×4096×4096 | 0.0298 / 18.0 | 0.1035 / 5.2 / 28.8% | 0.1794 / 3.0 / 16.0% | n/a | **0.0235 / 22.8 / 126.8%** |
| 32×4096×4096 | 0.0296 / 36.3 | n/a | n/a | n/a | **0.0236 / 45.6 / 125.5%** |
| 64×4096×4096 | 0.0304 / 70.6 | 0.1058 / 20.3 / 29.0% | 0.1815 / 11.8 / 16.5% | n/a | **0.0237 / 90.7 / 128.4%** |
| 16×11008×4096 | 0.0606 / 23.8 | 0.1156 / 12.5 / 52.5% | 0.1812 / 8.0 / 33.5% | n/a | **0.0564 / 25.6 / 107.5%** |
| 64×4096×11008 | 0.0637 / 90.6 | 0.2815 / 20.5 / 23.6% | 0.4822 / 12.0 / 13.2% | n/a | **0.0597 / 96.7 / 106.7%** |

"%" is cuBLAS time ÷ our time. The decode rows (v3 from the second run, on the dedicated
kernel; v0 and v1 from the first run, and none for M = 1 or 32, which the first run did not
have) in GB/s (2(MK + KN + MN) ÷ time): 1,427, 1,438, 1,447, 1,459, 1,607 and 1,543
respectively, against 1,532 GB/s for `cudaMemcpy`; the first run's 64×64×64 tile had 1,220,
1,163, 1,449 and 1,469 on the four shapes it ran. The PyTorch eager
comparison (`torch.matmul` in bf16, which calls cuBLAS/cuBLASLt through its own heuristics) is
in `results/torch_comparison.json` and `docs/RESULTS.md`; on the earlier run of the same
kernel v3 was 1.01× at 4096³, 1.02× at 4096×4096×11008 and 0.97× at 8192³.

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

Still open:

- **Stream-K proper**, i.e. persistent blocks each owning an equal share of the flattened
  (tile, k-step) space, would handle every shape uniformly instead of only the last partial
  wave, and would replace the per-call tile heuristic with one schedule.
- **The 2 µs of launch and ramp** that a single decode launch pays over the back-to-back
  rate (23.5 against 21.5 µs at 32 MB). It is not in the kernel; CUDA graphs or a persistent
  megakernel across the decoder block are the ways at it.
- **Perf per watt** on the long shapes: a 64×64 warp tile (4 warps per 128×128 block, or 8 warps
  on 128×256) halves the `ldmatrix` count per `mma.sync`. My 128×256 attempt lost with the
  current 2×4 warp grid; a 2×2 grid with 4 warps and 2–3 blocks per SM is the untested layout.
  8192³ at 95% of cuBLAS is the row it would move.
- **Persistent scheduling with L2-aware rasterization** (grouped tile order), which matters once
  operands exceed the 96 MB L2 (8192³ and up).
