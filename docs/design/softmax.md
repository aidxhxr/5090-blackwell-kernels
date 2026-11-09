# Softmax

`softmax_f32` / `softmax_bf16` in `src/kernels/softmax.cu`. Row-wise softmax over a `rows x cols`
row-major matrix, fp32 math regardless of storage type.

## Why softmax matters here

Softmax is the normalization inside every attention head: for a query the scores against
`cols = sequence length` keys are exponentiated and normalized. Numerically it must be computed as
`exp(x - max) / sum(exp(x - max))`, otherwise `exp` overflows in fp32 past `x ≈ 88`. That subtraction
is what makes a naive implementation *three* passes over the row: max, sum, normalize.

## The optimization ladder

| Variant | Mapping | Passes over the row | `exp` per element | Loads |
|---|---|---|---|---|
| 0 naive | one thread per row | 3 (max, sum, write) | 2 | scalar, uncoalesced |
| 1 warp | one warp per row | 2 (online (m, s), write) | 2 | 16-byte vectors |
| 2 block | 256 threads per row | 2 (online (m, s), write) | 2 | 16-byte vectors |
| 3 registers | 32–1024-thread group per row, row in registers | 1 | 1 | 16-byte vectors |

### Variant 0: three-pass, one thread per row

The reference implementation. Adjacent threads read addresses `cols` elements apart, so a warp's
32 loads hit 32 different cache lines: every byte of DRAM traffic is amplified by up to 32x, and
each thread walks its row three times. It exists only to be measured against; on the RTX 5090
it moves 60 GB/s in bf16 and ~150 GB/s in f32.

### Variant 1: online softmax, one warp per row

The trick from Milakov & Gimelshein, *Online normalizer calculation for softmax* (2018): keep a
running max `m` and a running sum `s = Σ exp(x_i − m)` and update both in a single sweep. When a
new max `m'` arrives, the old sum is rescaled instead of recomputed:

```
s ← s · exp(m − m') + exp(x − m')
m ← m'
```

The same rule merges two partial states, which is what makes the algorithm parallel:

```
merge((m1, s1), (m2, s2)):
    m = max(m1, m2)
    s = s1 · exp(m1 − m) + s2 · exp(m2 − m)
```

`merge` is associative and commutative. Each lane of the warp accumulates a private `(m, s)` over
a strided slice of the row, then a five-step `__shfl_xor_sync` butterfly merges the 32 lane states
so every lane holds the row result. Lanes read 16 bytes at a time (`float4` or 8 x bf16), so one
warp instruction moves 512 contiguous bytes: full coalescing and the fewest possible load
instructions per byte, which is what a memory-bound kernel needs on either target's bus
(1,792 GB/s on the RTX 5090, 273 GB/s on the GB10).

The write pass re-reads the row. That second read is meant to be served from L2 (a 16384-wide
f32 row is 64 KB against 24 MB of L2 on the GB10 and 96 MB on the 5090), so DRAM would see
roughly one read and one write of the matrix. On the 5090 Nsight Compute puts the L2 hit rate
of the block-per-row kernel on 4096×16384 f32 at 9.6%: with 4,096 rows in flight across
170 SMs the re-read mostly misses, and the kernel really moves about three passes' worth of
bytes. That is what variant 3 removes.

**Edge case**: merging two `-inf` maxima gives `exp(-inf − (-inf)) = NaN`. The merge returns
`s = 0` when the merged max is `-inf`, which is the mathematically consistent answer for "no
finite elements seen yet" and what masked attention rows produce.

This is exactly the primitive FlashAttention builds on: because `(m, s)` partial states merge, the
attention softmax can be computed block by block over the key dimension without ever
materializing the full `rows x cols` score matrix. Variant 1 is that inner loop, isolated.

### Variant 2: online softmax, one block per row

A single warp can only keep so many 16-byte loads in flight. For long rows (attention over long
contexts, `cols ≥ 4096`) 8 warps share one row, each warp reduces its own `(m, s)` with shuffles,
lane 0 of each warp writes the pair to shared memory, and warp 0 merges the 8 pairs with the same
butterfly. Two `__syncthreads()` and 64 floats of shared memory are the whole cost. For short rows
this mapping wastes threads (256 threads for a 128-element row): 0.0091 ms against variant 1's
0.0033 ms at 4096×128 on the 5090.

### Variant 3: single pass, row in registers

The same design as `rmsnorm` variant 4. A group of `G ∈ {32, 64, …, 1024}` threads owns one
row and holds all of it in registers, `VPT ∈ {1, 2, 4, 8}` (f32) or `{1, 2, 4}` (bf16) 16-byte
vectors per thread, at most 32 elements per thread; the host picks the smallest `G` whose
registers hold the row, then the smallest `VPT`. So a 128-wide row is one warp with one
`float4` per lane and 8 rows per 256-thread block; a 16384-wide f32 row is 512 threads with
8 `float4` each. Thread `t` loads vectors `t, t + G, …`, so a warp's load is still 512
contiguous bytes.

The row is then processed entirely from registers: group-reduce the max (warp shuffle, plus one
shared-memory exchange between the group's warps when `G > 32`), replace every value by
`exp(x − m)` in place and accumulate the sum, group-reduce the sum, scale and store. One read,
one `exp` per element, one write. The online merge is no longer needed because the true max is
known before any exponential is taken. The block is `max(G, 256)` threads and inactive threads
(past the last row) stay for the barriers with neutral `(−inf, 0)` contributions.

Rows wider than 32,768 elements, misaligned storage, or `cols` not a multiple of the vector
width run variant 2's kernel inside the same entry point, so the Python default (the top rung)
takes any input.

## Memory traffic and the roofline

Softmax does ~5 flops per element, so it is purely bandwidth-bound. Minimum traffic is one read and
one write:

```
bytes = 2 · rows · cols · sizeof(T)
GB/s  = bytes / time
```

The bench reports `gbps` against that formula, so the number to compare with is the RTX 5090's
1,792 GB/s peak (see `docs/RTX5090.md`; 273 GB/s on the GB10, `docs/GB10.md`), or better, the
achieved copy bandwidth from `bench_bandwidth`: 1,532 GB/s on the 5090.
bf16 moves half the bytes of f32 for the same shape, so at the same GB/s it is twice as fast.
Variant 3 on the 4096×16384 shapes sits at 1,521 GB/s (f32) and 1,542 GB/s (bf16) by this
formula: 99–101% of the copy probe. The variants that re-read the row can only reach that
figure if the re-read is free, and on this card it is not.

## What to look at in Nsight Compute

```
ncu --set full --kernel-name regex:softmax_ ./build/bench_softmax --cols=16384 --iters=3
```

Measured on the RTX 5090, f32 4096×16384 (`results/ncu_softmax_v2.txt`, `_v3.txt`; ncu locks
the SM clock at 2.55 GHz, so durations are longer than in the bench):

- **Memory Workload Analysis → DRAM Throughput (% of peak)**: 86.8% for variant 2 and 87.1% for
  variant 3, at 453 µs and 313 µs respectively. Both saturate the bus; variant 2 just moves
  ~45% more bytes through it, which is the second read of the row.
- **L2 hit rate**: 9.6% for variant 2 (the re-read of 4,096 × 64 KB rows mostly misses),
  0.06% for variant 3 (there is nothing to hit; every byte is touched once).
- **Memory Workload Analysis → L1/TEX sectors per request**: 32 for variant 0 (one sector per
  lane), 4 for the vectorized variants (a warp's 512 bytes are 16 sectors of 32 B over 4 requests).
- **Occupancy → Achieved**: variant 2 runs at 90% with 40 registers; variant 3 at 60% (48
  registers, 512-thread blocks, 2/3 theoretical). The lower occupancy costs nothing here because
  each thread has 8 independent 16-byte loads in flight.
- **Compute Workload Analysis → Pipe utilization (XU)**: `__expf` runs on the SFU. "Compute
  (SM) Throughput" is 7.6% for variant 2 and 5.9% for variant 3 at full DRAM throughput, so
  the exponential never became a limiter on this card, even in f32; halving the `exp` count in
  variant 3 is a byte-count story, not a compute one.
- **Warp State → Stall Long Scoreboard**: memory latency. Variant 2 exists to hide it on long
  rows by putting more loads in flight per row; variant 3 does the same with 8 vectors per
  thread instead of 8 warps per row.

## RTX 5090 notes (measured)

- **The L2 is 96 MB and most of the bench shapes fit in it.** 4096×1024 f32 is 16 MB and
  4096×4096 f32 is 64 MB of input; those rows stay resident across the timing loop and report
  2–4 TB/s, which is L2 bandwidth, not DRAM. Only the 4096×16384 rows (256 MB f32, 128 MB bf16
  of input) are DRAM-bound, and they are the ones to judge the kernel by.
- **The `cols > 4096` crossover from variant 1 to variant 2 did move down, as expected**, then
  variant 3 made it moot: at 1024 columns variant 2 is already within 20% of variant 1 in f32
  and variant 3 beats both at every width (0.0093 vs 0.0095 ms at 4096×1024 f32; 0.0054 vs
  0.0072 at 4096×1024 bf16), and at 128 columns it matches variant 1 (0.0033 ms) where variant 2
  wastes 7/8 of its threads. The Python default is variant 3 everywhere.
- **The second pass is not free on GDDR7.** f32 4096×16384 went from 1,104 GB/s (variant 2)
  to 1,521 GB/s (variant 3), a 1.38× speedup from removing one read and one `exp` per element;
  bf16 from 1,434 to 1,542 GB/s. Nsight puts the re-read's L2 hit rate at 9.6%.
- **Against PyTorch**: `torch.softmax` is already a single-pass register kernel for these
  widths, so variant 3 ties it on the DRAM-bound shapes (1.00× f32, 1.01× bf16 at 4096×16384)
  and beats it where PyTorch picks a worse configuration: 1.08× at 4096×4096 f32, 1.21× at
  4096×1024 bf16 and 2.32× at 4096×4096 bf16. Variant 2 lost to it on both 4096×16384 shapes
  (0.72× and 0.94×) before variant 3 existed.
- **Few rows underfill the card.** Variant 1 launches `rows / 4` blocks, variant 3 `rows / 8`
  for narrow rows: 4096 rows is enough for 170 SMs, decode-time shapes with a few dozen rows are
  not, and the 4096×128 rows (3.3 µs, 4 MB) are launch-bound rather than bandwidth-bound
  regardless of variant.
- `__expf` on the SFU is not a limiter at 1.5 TB/s: SM throughput stays under 8% in every
  profiled variant.
- `min_ms` sat within 2–4% of the median on every row of the run.

## Timing

The rows used to be single launches, the median of 100 (of 10 for v0). On 2026-11-08 I ran
w1gemm's two-run test ([w1gemm](w1gemm.md#timing)) on this bench: two runs back to back in
two lock holds, rows more than x% apart (32 rows), twice for each:

| rows | pair | > 2% | > 3% | > 5% | largest | largest ref_ms |
|---|---|---|---|---|---|---|
| single launches | first | 1 | 1 | 1 | 18.0% | 1.6% |
| single launches | second | 3 | 3 | 3 | 9.7% | 1.5% |
| chains of 20 | first | 2 | 0 | 0 | 2.6% | 0.1% |
| chains of 20 | second | 0 | 0 | 0 | 1.7% | 0.0% |

In the first pair one row moved, but by a whole step: bf16 v3 at 4096 x 1024, 5.34 µs in one
run and 6.30 in the other, and the other 31 were within 2%. The second pair moved the same row
(5.60 and 6.14) and two more, f32 v1 at 4096 x 1024 and f32 v2 at 4096 x 128. Any two of the
four chain runs are within 3.0%; any two of the four single-launch runs have 0 to 3 rows past
5%. So the rows are chains now: the median and min per launch of 20 launches between one event
pair over 20 pairs (`spark::bench::time_chained`), v0 too, so ref_ms is a chain as well. The
single-launch medians are on stderr and `--chain=1` gives the old rows. The two chain rows past
2% in the first pair are the fastest ones, 4096 x 128 on v3 at 1.55 to 1.69 µs a launch, near the 1.3 µs below
which the host sets the pace (w1gemm's note). Against the single-launch rows of the same
evening 16 of 32 are more than 5% faster and none slower: 4096 x 128 goes from 3.23 µs alone to
1.55 (bf16) and 1.65 (f32) a launch, 4096 x 1024 bf16 v3 from 5.34 to 4.37, and the rows over
100 µs are within 1%. The tables below are the older single-launch run.

## Results (RTX 5090)

From `results/softmax.json` before it held chains (median of 100 single launches, CUDA 13.2,
driver 595.58, no display) and `results/torch_comparison.json` (PyTorch 2.14 eager
`torch.softmax`). `% of 1,792 GB/s` above 100% means the shape is L2-resident; the 4096×16384
rows are DRAM-bound. A GB10 table (% of 273 GB/s) is added when the Spark has been benchmarked.

| dtype | rows×cols | v0 ms | v1 ms | v2 ms | v3 ms | best | GB/s | % of 1,792 GB/s | best / v0 | vs torch |
|---|---|---|---|---|---|---|---|---|---|---|
| f32 | 4096×128 | 0.0310 | 0.0033 | 0.0091 | 0.0033 | v1 = v3 | 1285 | 71.7% (launch-bound) | 9.5× | 1.04× |
| f32 | 4096×1024 | 0.2122 | 0.0095 | 0.0112 | 0.0093 | v3 | 3591 | 200.4% (L2) | 22.7× | 1.00× |
| f32 | 4096×4096 | 0.9077 | 0.1034 | 0.0764 | 0.0712 | v3 | 1885 | 105.2% (L2) | 12.7× | 1.08× |
| f32 | 4096×16384 | 3.8183 | 0.5129 | 0.4862 | 0.3531 | v3 | 1521 | 84.9% | 10.8× | 1.00× |
| bf16 | 4096×128 | 0.0363 | 0.0033 | 0.0093 | 0.0033 | v1 = v3 | 643 | 35.9% (launch-bound) | 11.1× | 1.05× |
| bf16 | 4096×1024 | 0.2735 | 0.0072 | 0.0094 | 0.0054 | v3 | 3102 | 173.1% (L2) | 50.6× | 1.21× |
| bf16 | 4096×4096 | 1.0865 | 0.0216 | 0.0175 | 0.0175 | v2 = v3 | 3841 | 214.3% (L2) | 62.2× | 2.32× |
| bf16 | 4096×16384 | 4.4184 | 0.2610 | 0.1872 | 0.1741 | v3 | 1542 | 86.1% | 25.4× | 1.01× |
