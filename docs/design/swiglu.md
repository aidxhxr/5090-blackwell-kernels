# SwiGLU (`swiglu_f32`, `swiglu_bf16`, `swiglu_fp8_bf16`)

`out[i] = silu(gate[i]) * up[i]`, where `silu(x) = x * sigmoid(x) = x / (1 + e^-x)`.

This is the gated activation between the gate/up projections and the down projection in every
Llama- and Qwen-style MLP. For a 4096-token prefill through a 14336-wide intermediate in bf16
the tensors are 117 MB each, so the op is purely memory-bound and the whole game is bytes.

## The fusion win

PyTorch eager evaluates `F.silu(gate) * up` as two kernels:

| step | reads | writes | passes over an activation-sized tensor |
|------|-------|--------|----------------------------------------|
| `F.silu(gate)` | gate | tmp | 2 |
| `tmp * up` | tmp, up | out | 3 |
| **eager total** | | | **5** (+ 2 launches, + a temporary allocation) |
| **fused** | gate, up | out | **3** (1 launch, no temporary) |

Ideal speedup from traffic alone is 5/3 = 1.67x; in practice the gap is larger because the two
eager kernels each pay launch latency and neither is guaranteed to be at the bandwidth roofline
(`torch.compile` fuses this too, which is why the Python bench compares against both eager and
compiled).

Traffic formula used for GB/s in the bench: `3 * n * sizeof(T) / time`.

## Variants

| # | Design | Notes |
|---|--------|-------|
| 0 | scalar, one element per thread, `cdiv(n, 256)` blocks | Baseline. bf16 lanes load 2 bytes each: a warp moves 64 B per load instruction, far below what the LSU pipe can issue, so the kernel is instruction-bound before it is DRAM-bound. |
| 1 | 16-byte vectorized, grid-stride, tail in-kernel | `f32x4` (4 floats) or `bf16x8` (4 x `__nv_bfloat162`) per thread per operand, one 128-bit load each. Math is done in fp32 via `__bfloat1622float2` / `__float22bfloat162_rn`. The grid is fixed at 8 blocks x 256 threads per SM and each thread strides over the array; the `n % VEC` remainder is handled by a second loop in the same kernel so there is no extra launch. On the RTX 5090 both variants saturate the bus on the large shapes; variant 1's win is on the shapes that fit in L2 (see the notes). |
| fp8 | `swiglu_fp8`: the strided form writing the e4m3 row and its per-token scale, a cluster of 8 blocks a row up to 32 rows | removes `fp8_quantize` and the bf16 row in front of fp8-tok's down projection |

`silu` uses `__expf`, the fast SFU exponential. Its error is well under bf16 resolution, and
for f32 it is within 2 ulp over the activation range that matters here. For `x -> -inf`,
`__expf(-x) -> inf` and `x / inf -> -0`, which is the correct limit.

Variant 1 requires 16-byte-aligned pointers (`cudaMalloc` and PyTorch allocations satisfy this;
sliced tensors may not, and the host wrapper throws `std::invalid_argument` rather than
silently misreading).

## Into fp8 for the down projection

On the engine's fp8-tok path the down projection wants e4m3 rows with one scale per token. As
two ops, `swiglu` writes the bf16 product, then `fp8_quantize(h, "row")` reads it back, takes
its amax and writes the bytes and the scale: two launches and a bf16 row that lives for one
kernel. `swiglu_fp8(gate, up)` is one launch. It reads the two halves of the gate|up output in
place, as the strided kernel does, keeps the products in registers rounded to bf16, reduces the
row's amax, and writes the power-of-two scale and the e4m3 bytes. Per element that is 5 bytes
instead of 9 (read gate and up, write q; the two ops also write h and read it back).

The output is the two ops' bits:

- each product is the same fp32 expression, `silu_f(g) * u` with `__expf` and an IEEE division,
  in the same file under the same flags, rounded to nearest even as the strided kernel stores it;
- the scale and the conversion are `pow2_scale` and `store8` from `src/kernels/fp8quant.cuh`,
  the code `fp8_quantize` and the fp8 norms run;
- the amax is a max, so neither the block size nor splitting a row across blocks changes it.

The tests compare bytes and scales with `torch.equal` at every block size, on ragged rows
(2,056 and 11,008 columns), the widest row it takes (32,768), an all-zero row, an outlier and
`up` scaled by 2^-100 and 2^100, at 1 to 257 rows (both geometries below), and I ran them
again with each cluster size forced. Rows wider than 32,768, unaligned rows or other dtypes
raise; there is no fallback.

### One block a row, then a cluster for decode

The first version was the rmsnorm recipe: one block of 128 to 1,024 threads per row, at most
four `bf16x8` vectors a thread, 512 threads at 14,336 columns. It won 1.88x at 4096 x 14336 and
lost at the shape a decode step runs: 3.78 us against 3.58 for the two ops at 1 x 14336, in a
graph. `swiglu` alone spreads that row over 7 blocks. One block a row puts all 14,336 exps and
divisions, and all 57 KB of loads, on one SM.

So up to 32 rows a row goes to a thread block cluster of 8 blocks of 128 threads. Each block
takes an eighth of the row (vector `t` of the row's 1,024 threads, then `t + 1024`, ...),
reduces its amax, leaves it in its shared memory, and after a cluster barrier warp 0 of every
block reads the eight amaxes over distributed shared memory (`mapa` then `ld.shared::cluster`).
All eight blocks then hold the same scale and write their part. A second barrier at the end
keeps a block's shared memory alive until its peers have read it. Block 0 writes the scale.

Cluster size forced with `SPARK_SWIGLU_FP8_CLUSTER`, 1 x 14336 to 512 x 14336, us per call in a
CUDA graph (`scripts/bench_swiglu_fp8.py --part cluster`, the untagged `kind: cluster_sweep`
rows in `results/swiglu_fp8.jsonl`, 30 replays; with today's dispatch `--vpts 4` gives these
blocks):

| rows | 1 block | cluster of 2 | of 4 | of 8 |
|---|---|---|---|---|
| 1 | 3.78 | 3.62 | 3.28 | **2.56** |
| 8 | 3.88 | 3.68 | 3.37 | **2.66** |
| 32 | 3.91 | 3.78 | 3.47 | **2.96** |
| 48 | 3.79 | **3.72** | 3.89 | 5.21 |
| 64 | **3.89** | 3.94 | 4.11 | 5.34 |
| 128 | **4.40** | 4.86 | 7.36 | 9.41 |
| 512 | **9.82** | 13.82 | 20.55 | 29.48 |

Past 32 rows the clusters cost more than they spread. The cliff sits between 256 and 384 blocks
in clusters of 8 (32 and 48 rows), and past it the cluster form takes 9.41 us at 128 rows and
29.48 at 512, where one block a row takes 4.40 and 9.82 for the same bytes. So the first rule
was a cluster of 8 up to 32 rows when the row has 4,096 values or more, one block a row
otherwise; at 2,048 columns one block a row was the fastest at every row count I swept (2 to
48; an eighth of that row is 32 vectors for 128 threads). That left 33 to 128 rows, small
batch decode and prefill chunks, with a whole 14,336 row's silu on one SM: 1.01 to 1.16x the
two ops at 33 to 64 rows of 14336 (the same hold as the last table below).

### Past 41 rows: the clusters the GPU holds

Nsight Compute names the cliff. For a cluster of 8 blocks of 128 threads it reports 41 Max
Active Clusters: 328 blocks resident, about two an SM, where registers and warps would allow
12 such blocks an SM. A cluster of 4 gets 85, a cluster of 2 gets 170. So from the 42nd row a
cluster a row runs in waves, and every wave pays a row's whole latency again: the loads, the
amax, the barrier, the read of the peers, the stores. It is not the exchange, which every row
pays from 1 up, where the clusters win. A sweep across the edge pins it: a cluster of 8 takes
3.06 us at 41 rows of 14336 and 4.81 at 42.

ncu at 14,336 columns, one call under ncu's clock control (so the durations are not the graph
times below), the geometries of the first rule's blocks:

| rows | geometry | grid | Max Active Clusters | waves of clusters | duration | warps an SM | eligible warps a scheduler |
|---|---|---|---|---|---|---|---|
| 64 | cluster of 8, 128 threads | 512 | 41 | 2 | 8.83 us | 7.3 | 0.14 |
| 64 | cluster of 4, 128 threads | 256 | 85 | 1 | 7.07 us | 6.1 | 0.13 |
| 64 | cluster of 2, 256 threads | 128 | 170 | 1 | 7.36 us | 8.3 | 0.18 |
| 64 | one block, 512 threads | 64 | | | 6.62 us | 15.5 | 0.51 |
| 128 | cluster of 8, 128 threads | 1024 | 41 | 4 | 13.73 us | 7.6 | 0.16 |
| 128 | cluster of 4, 128 threads | 512 | 85 | 2 | 12.00 us | 6.9 | 0.16 |
| 128 | cluster of 2, 256 threads | 256 | 170 | 1 | 9.63 us | 11.3 | 0.23 |
| 128 | one block, 512 threads | 128 | | | 8.99 us | 16.5 | 0.36 |

The last two columns matter as much. Even in one wave every geometry has a few warps an SM and
a scheduler with less than one warp ready to issue: each thread waits on its exps and
divisions. A few rows are latency bound, not byte bound. The first rule sized the block the
rmsnorm way, the fewest threads that hold the part at four vectors a thread. Sized for one vector a
thread instead (a second ncu run, the two side by side), 32 rows in clusters of 8 go from 128
to 256 threads a block, 5.9 to 11.4 warps an SM, 0.11 to 0.19 eligible warps and 5.06 to 4.77
us; 64 rows one block a row go from 512 to 1,024 threads, 16.1 to 30.1 warps, 0.53 to 1.00
eligible and 6.62 to 6.08 us.

What I tried at 14,336 columns, us per call in a graph (`kind: cluster_sweep`, `tag:
2026-11-08`, 30 replays). "4 vec" is the first rule's block, "1 vec" one vector a thread;
"2 rows" and "4 rows" are `swiglu_fp8_rows_kernel`, a cluster of 8 with that many rows, each
row of a block its own group of threads:

| rows | cluster of 8, 4 vec | of 8, 1 vec | of 4, 1 vec | 1 block, 4 vec | 1 block, 1 vec | of 8, 2 rows | of 8, 4 rows | two ops |
|---|---|---|---|---|---|---|---|---|
| 16 | 2.76 | **2.45** | 2.69 | 3.88 | 3.28 | 2.66 | 3.36 | 3.81 |
| 32 | 2.96 | **2.65** | 2.76 | 3.92 | 3.38 | 2.86 | 3.37 | 4.09 |
| 41 | 3.06 | **2.66** | 2.86 | 3.97 | 3.37 | 3.17 | 3.37 | 4.19 |
| 42 | 4.81 | 4.19 | **2.96** | 3.92 | 3.38 | 3.07 | 3.37 | 3.89 |
| 48 | 5.11 | 4.40 | **3.17** | 3.92 | 3.38 | 3.18 | 3.58 | 3.99 |
| 64 | 5.22 | 4.50 | **3.06** | 3.80 | 3.27 | 3.07 | 3.48 | 4.40 |
| 96 | 7.37 | 6.44 | 4.91 | 3.98 | **3.47** | 5.11 | 6.24 | 4.81 |
| 128 | 9.21 | 8.09 | 5.42 | 4.41 | **3.88** | 5.77 | 6.65 | 5.73 |

- **A smaller cluster** moves the wall. A cluster of 4 of 512 threads holds a row in one wave
  to 85 rows and is the fastest from 42 to 64; at 96 it is in waves again.
- **Several rows to a cluster** does what it is for: 64 rows in 32 clusters of 8 are one wave,
  3.07 us against 4.50 for a cluster a row. But it only ties the cluster of 4, and at 128 rows
  four rows a cluster need blocks of 1,024 threads, for which ncu gives 19 Max Active Clusters,
  so its 32 clusters are two waves again (6.65 us). It runs more instructions too: 1.58 M at 64
  rows against 1.19 M for one block a row of 1,024 threads. It stays behind
  `SPARK_SWIGLU_FP8_ROWS`.
- **Bigger blocks** are most of the win and need no exchange at all: one block a row of 1,024
  threads beats the first rule's 512 at every row count here, and from 96 rows it is the
  fastest of all.

One vector a thread stops paying past 128 rows. One block of 1,024 threads fits an SM, so the
170 SMs take 170 rows at once. One block a row, 128 then 192 rows of 14336: 3.88 and 6.65 us at
one vector a thread, 4.40 and 6.26 at four, and 12.39 against 9.93 at 512 rows. (At 2,048
columns one vector a thread wins to 1,024 rows, 3.31 against 3.61 us, and at 4096 x 14336 it is
2% ahead, 171.64 against 175.02; the rule leaves both on the table.)

The rule now: a cluster of 8 up to 41 rows of 8,192 values or more and up to 16 rows from
4,096 (at 5,632 one block of 1,024 threads ties at 16 rows and wins from 24), a cluster of 4
from 42 to 64 rows of 14,336 or more, one block a row otherwise; the block sized for one
vector a thread up to 128 rows, four past that. Against the first rule in the same hold, 1 to
4096 rows of the four widths twice each (`tag: 2026-11-06-rule` and `2026-11-08-rule`), in a
graph, the faster of the two runs:

| rows x 14336 | first rule | now | | rows x 14336 | first rule | now |
|---|---|---|---|---|---|---|
| 1 | 2.45 | 2.24 | | 48 | 3.81 | 3.07 |
| 8 | 2.65 | 2.25 | | 64 | 3.78 | 3.07 |
| 32 | 2.86 | 2.47 | | 96 | 3.93 | 3.47 |
| 33 | 3.81 | 2.55 | | 128 | 4.24 | 3.69 |
| 41 | 3.85 | 2.56 | | 192 | 6.24 | 6.14 |
| 42 | 3.78 | 2.86 | | 512 | 9.85 | 9.81 |

It is never slower by more than 2% at the 60 shapes, and at 64 rows the other widths go from
2.03 to 1.73 us (2,048), 2.66 to 2.15 (5,632) and 3.17 to 2.97 (11,008). Eager calls barely
move, 3.9 to 4.4 us either way up to 96 rows: that is the host's launch path.

It is batch invariant: a row's bits don't depend on how many rows came with it or on which
geometry ran it. `scripts/check_invariance.py` has it: the same bits at 1, 8, 32 and 256 rows
of 4096, the first two in clusters of 8 and the others one block a row.
`test_swiglu_fp8_rows_do_not_depend_on_the_batch` crosses the rule's edges at 14336 (41 and 42,
64 and 65, 128 and 129 rows), and every geometry forced runs the parity test.
`SparkModel.fuse_swiglu_fp8 = False` runs the two ops; with the batch-invariant rungs the
engine's logits are `torch.equal` both ways.

## Validation

Both variants are checked against a CPU reference computed from the *rounded* inputs (the host
converts to bf16 first, so the reference sees exactly the values the GPU sees). Pass criterion
is `|got - ref| <= atol + rtol * |ref|` with `1e-5 / 1e-5` for f32 and `2e-2 / 2e-2` for bf16.

## Nsight Compute

```
ncu --set full --kernel-name regex:swiglu_ ./build/bench_swiglu --n=58720256 --iters=3
```

Look at `dram__throughput.avg.pct_of_peak_sustained_elapsed` (should approach the bandwidth
probe's number for variant 1) and `smsp__inst_executed.sum` (should fall ~8x for bf16 between
variants 0 and 1). If `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` is more than
`3 * n * sizeof(T) / 32`, loads are not coalesced.

## RTX 5090 notes (measured)

- **The kernel sits on the bus.** On the DRAM-bound shapes (4096×5632 and up; 4096×2048 is
  33–67 MB of traffic and lives in the 96 MB L2, which is why it reports 3–6 TB/s) both
  variants land at 1,500–1,600 GB/s by the `3 · n · sizeof(T)` formula: 98–104% of the
  1,532 GB/s copy probe and 84–89% of the 1,792 GB/s spec. Above the copy figure is possible
  because a copy pays one read for one write while SwiGLU pays two reads for one write, and
  reads are cheaper than read/write turnaround on GDDR7.
- **Vectorization stopped mattering once the bus was saturated.** Variant 1 wins clearly where
  the shape is L2-resident (1.4× at 4096×2048 f32, 1.5× bf16) and on the 4096×5632 bf16 shape
  (1,801 vs 1,524 GB/s, part of it L2-resident); on 4096×11008 and 4096×14336 the scalar
  variant is 2–3% *faster*. With 16 GB of bytes per second arriving, a warp's 64 B bf16 loads
  are issued fast enough, and the `cdiv(n, 256)` grid of variant 0 (917,504 blocks at
  4096×14336 bf16) tails better than the fixed 8 × 170 blocks of variant 1. The `8 blocks × 256
  threads per SM` constant was left as is; it is not the limiter.
- **Fusion against eager is 1.6× on the large shapes, 1.3–3.8× on the small ones.** The 5/3
  traffic ratio predicts 1.67× and that is exactly what the DRAM-bound rows show (1.61× bf16,
  1.64–1.66× f32); the small shapes gain more because eager's two launches and temporary
  dominate when the bytes are cheap (3.8× at 4096×2048 f32).
- **`__expf` is invisible.** The SFU never shows up: the kernel is bandwidth-bound at every
  shape that leaves L2. `torch.compile` fuses the same two ops; the eager number is the one
  quoted because it is what `F.silu(gate) * up` costs in a model that is not compiled.
- `min_ms` is within 1–2% of the median on every row.

## Results (RTX 5090)

From `results/swiglu.json` (median of 100 iterations, CUDA 13.2, driver 595.58, no display) and
`results/torch_comparison.json` (PyTorch 2.14 eager `F.silu(gate) * up`, shown for the variant
the Python binding runs, variant 1). `% of probe` is against the 1,532 GB/s `cudaMemcpy` figure
from `bench_bandwidth`; rows far above 100% are L2-resident (4096×2048: 33 MB f32 / 17 MB bf16
per operand). A GB10 table is added when the Spark has been benchmarked.

| dtype | shape | variant | median ms | GB/s | % of probe | % of 1,792 | vs PyTorch eager |
|-------|-------|---------|-----------|------|------------|------------|------------------|
| f32  | 4096×2048  | 0 | 0.0218 | 4626 | 302% (L2) | 258% | — |
| f32  | 4096×2048  | 1 | 0.0156 | 6446 | 421% (L2) | 360% | 3.80× |
| f32  | 4096×5632  | 0 | 0.1730 | 1601 | 104% | 89.3% | — |
| f32  | 4096×5632  | 1 | 0.1802 | 1536 | 100% | 85.7% | 1.64× |
| f32  | 4096×11008 | 0 | 0.3428 | 1578 | 103% | 88.1% | — |
| f32  | 4096×11008 | 1 | 0.3512 | 1541 | 101% | 86.0% | 1.65× |
| f32  | 4096×14336 | 0 | 0.4454 | 1582 | 103% | 88.3% | — |
| f32  | 4096×14336 | 1 | 0.4536 | 1553 | 101% | 86.7% | 1.66× |
| bf16 | 4096×2048  | 0 | 0.0175 | 2881 | 188% (L2) | 161% | — |
| bf16 | 4096×2048  | 1 | 0.0115 | 4381 | 286% (L2) | 245% | 1.32× |
| bf16 | 4096×5632  | 0 | 0.0908 | 1524 | 99% | 85.0% | — |
| bf16 | 4096×5632  | 1 | 0.0768 | 1801 | 118% | 100.5% | 1.49× |
| bf16 | 4096×11008 | 0 | 0.1729 | 1564 | 102% | 87.3% | — |
| bf16 | 4096×11008 | 1 | 0.1803 | 1500 | 98% | 83.7% | 1.61× |
| bf16 | 4096×14336 | 0 | 0.2241 | 1572 | 103% | 87.7% | — |
| bf16 | 4096×14336 | 1 | 0.2324 | 1516 | 99% | 84.6% | 1.61× |

| swiglu_fp8 | rows x cols | `swiglu_fp8` us | `swiglu` + `fp8_quantize` us | speedup | eager speedup |
|---|---|---|---|---|---|
| | 1 x 14336 | 2.24 | 3.38 | 1.51x | 1.73x |
| | 8 x 14336 | 2.29 | 3.58 | 1.56x | 1.70x |
| | 32 x 14336 | 2.49 | 3.88 | 1.56x | 1.76x |
| | 64 x 14336 | 3.07 | 4.39 | 1.43x | 1.72x |
| | 512 x 14336 | 9.81 | 13.52 | 1.38x | 1.38x |
| | 4096 x 14336 | 176.27 | 330.96 | 1.88x | 1.88x |
| | 4096 x 11008 | 125.67 | 247.85 | 1.97x | 1.98x |
| | 4096 x 5632 | 59.48 | 90.57 | 1.52x | 1.54x |
| | 4096 x 2048 | 8.19 | 15.64 | 1.91x | 1.91x |

From `results/swiglu_fp8.jsonl` (`scripts/bench_swiglu_fp8.py --part op`, `tag:
2026-11-08-rule`, the first of its two runs; every row of 2048, 5632, 11008 and 14336 at 1 to
4096 rows is in the file, and the untagged rows are the first rule's): gate and up are the
halves of one [rows, 2 x cols] tensor, 20 calls captured in one CUDA graph, the median of 50
replays divided by 20. Eager is 20 calls queued between two events. The inputs are not
rotated, as in the engine. It wins at all 60 shapes, 1.28 to 1.62x at the 1, 8, 32, 64 and 512
rows the first table had and 1.52 to 1.97x at 4096.

- **1 to 512 rows is launches.** At 1 x 14336 the fused op takes 2.24 us, `swiglu` alone 1.23
  and `fp8_quantize` alone 2.24. In a graph the fusion saves 1.1 us a call; eager, where each
  op also pays the host's launch path, it is 4.15 against 7.17 us.
- **33 to 128 rows was the weak spot.** Under the first rule 64 x 14336 was 1.18x: past the
  clusters, 64 blocks on 170 SMs each did a whole row's silu. Now a cluster of 4 takes it,
  1.43x (the section on the clusters the GPU holds).
- **4096 rows is bytes.** At 4096 x 14336 the fused op moves 294 MB in 176.3 us, 1,666 GB/s by
  the 5-byte model, above the 1,532 GB/s copy figure because it reads four bytes for every one
  it writes. It is faster than `swiglu` alone (232.2 us), which writes two bytes an element.
  At 5632 the gain is smaller (1.52x): by the 9-byte model the two ops run at 2,292 GB/s there,
  over the copy figure, so part of their traffic is L2 hits (h is 46 MB, the L2 96 MB).

The engine rows below are the first rule's, from 2026-11-06; I did not run the engine again.

| fp8-tok engine, 8 layers | batch | decode ms/step, two ops | fused | prefill tok/s, two ops | fused |
|---|---|---|---|---|---|
| | 1 | 2.068 | 2.061 | 119,337 | 121,282 |
| | 8 | 2.214 | 2.206 | 135,453 | 141,896 |
| | 32 | 2.462 | 2.451 | 133,860 | 140,136 |
| | 64 | 2.989 | 2.981 | 131,279 | 136,515 |

`scripts/bench_swiglu_fp8.py --part engine`: bench_serve's static batches (prompts log-uniform
in 64 to 2048 tokens, 64 new tokens each, greedy), random weights, bf16 cache, CUDA graphs, the
fp8 norms on in both, the two modes alternated three times on one model and the median of each
kept. The three runs of a mode agreed within 4 us a step. Decode drops 7 to 11 us a step (0.3
to 0.4%): 8 down projections at about 1 us each, what the graph rows say. Prefill gains 1.6% at
884 tokens and 4.0 to 4.8% from 6,073 up, where the op moves bytes. fp8-tok still runs
`fp8_quantize` in front of o, whose input comes out of attention, and a row-scale multiply after
every projection.
