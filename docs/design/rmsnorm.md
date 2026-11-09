# RMSNorm and fused add+RMSNorm

`out[r, c] = x[r, c] · rsqrt(mean_c(x[r, :]²) + ε) · w[c]`

RMSNorm runs twice per decoder layer in Llama/Qwen-style models, and it is a pure
bandwidth problem: every element is read, squared, reduced, and written back.
The arithmetic intensity is ~1 FLOP/byte, far below the fp32 ridge point of either
target (≈ 58.5 FLOP/byte on the RTX 5090 at 1,792 GB/s GDDR7, 114 on the GB10 at
273 GB/s LPDDR5X), so the only lever is **moving bytes at peak bandwidth**. The ladder below is about
removing everything that stops the memory system from streaming, and then about reading the
input exactly once.

## Optimization ladder

| Variant | Mapping | What it fixes |
|---|---|---|
| 0 | one thread per row | baseline |
| 1 | one warp per row, lane-strided scalar loads | coalescing + parallel reduction |
| 2 | one warp per row, 128-bit vector loads | fewer, wider memory instructions |
| 3 | one 256-thread block per row | parallelism for very wide rows |
| 4 | 32–1024-thread group per row, row held in registers | single pass: `x` is read once |
| fused | add+RMSNorm, same single-pass design as variant 4 | removes a full pass over the activations |
| fp8 | both norms writing the e4m3 row and its per-token scale | removes `fp8_quantize` and the bf16 row on the fp8-tok path |

### Variant 0: one thread per row (why it is slow)

Thread `t` walks row `t` from column 0 to `cols-1`. Adjacent threads in a warp
therefore touch addresses `cols · sizeof(T)` bytes apart on every iteration, so a
32-lane warp issues 32 separate cache-line requests per load instruction instead of
one. Each thread also performs a serial `cols`-length reduction, and with only
`rows / 256` blocks the GPU is badly under-occupied for typical `rows = 4096`
(16 blocks for 48 SMs on the GB10, for 170 on the RTX 5090). Nsight Compute shows this as low
`dram__throughput`, a huge `l1tex__t_sectors_pipe_lsu_mem_global_op_ld` to
requests ratio, and `sm__warps_active` near the floor. On the 5090 it manages 36 GB/s in bf16
and ~105 GB/s in f32, i.e. 2–6% of the bus.

### Variant 1: one warp per row

Lane `l` reads columns `l, l+32, l+64, …`. Consecutive lanes now read consecutive
elements, so every warp-wide load is a contiguous 128 B (f32) or 64 B (bf16)
request. The sum of squares is reduced with `__shfl_xor_sync` in five steps
(`warp_reduce_sum` from `common.cuh`) instead of a serial loop. Eight warps per
block gives `rows / 8` blocks, plenty of parallelism.

### Variant 2: 128-bit vectorized loads

A warp-wide load of 4-byte scalars is 128 B; the LSU can issue 16 B per lane per
instruction, i.e. 512 B per warp. Loading `float4` / eight packed `bf16` per lane
cuts load instruction count 4–8× and lets the memory pipeline keep more bytes in
flight, which matters on LPDDR5X where latency is high relative to bandwidth
(GB10 reasoning; for what GDDR7 did with it see the RTX 5090 notes below).
This requires `cols % 4 == 0` (f32) or `cols % 8 == 0` (bf16); every real
hidden size satisfies this.

The kernel makes two passes over the row: pass 1 reads to compute the sum of
squares, pass 2 re-reads and writes. A row is at most 32 KB for the shapes here,
and both L2s (24 MB on the GB10, 96 MB on the 5090) hold the rows in flight, so the second
read is meant to be an L2 hit and DRAM traffic to stay at `2·rows·cols·sizeof(T)`. Nsight
Compute on the 5090 says that only partly holds: on the 16384×8192 f32 shape the block-per-row
kernel (same two-pass structure) sees a 31% L2 hit rate, so a good share of the re-read does
go back to DRAM. That is the motivation for variant 4.

### Variant 3: one block per row

For rows wider than ~8k elements a single warp cannot keep enough loads in flight.
A 256-thread block per row, with `block_reduce_sum` (warp shuffle + one shared
memory exchange), spreads the row over 8 warps. For narrow rows this is worse than
variant 2 because each block does one `__syncthreads`-heavy reduction for little
work. On the 5090 it was in fact never worse: it wins or ties variant 2 at every width from
1024 up, so the "wide rows only" reasoning from the GB10 did not carry over (see the notes).

### Variant 4: single pass, row in registers

Variants 1–3 read the row twice. Variant 4 reads it once: a *group* of `G` threads owns a row
and keeps all of it in registers, `VPT` 16-byte vectors per thread (`VPT ∈ {1, 2, 4, 8}` for f32,
`{1, 2, 4}` for bf16, at most 32 elements per thread). The host picks the smallest
`G ∈ {32, 64, 128, 256, 512, 1024}` whose registers hold the row, then the smallest `VPT` that
covers it, so a 1024-wide f32 row is one warp with 8 `float4` per lane, an 8192-wide row is
256 threads with 8 per lane, and a bf16 row of 8192 is 256 threads with 4 `bf16x8` per lane.
Thread `t` of the group loads vectors `t, t + G, t + 2G, …`, so consecutive threads still read
consecutive 16-byte chunks and every warp load is 512 contiguous bytes.

Launch geometry: the block is `max(G, 256)` threads, so a small group does not waste a block
(`256 / G` rows per block, 8 rows per block for `G = 32`). The reduction is a warp shuffle
followed, when the group spans several warps, by one shared-memory exchange between the
group's warps: every thread of the group ends up with the sum, then normalizes the values it
already holds and writes them. Inactive threads (past the last row) stay in the block for the
barrier and contribute zero.

Rows wider than 32,768 elements, misaligned storage, or `cols` not a multiple of the vector
width fall back to variant 3's kernel inside the same entry point, which is why the Python
default (`variant=-1`, the top rung) still accepts any input.

### Fused residual add + RMSNorm

The decoder block computes `resid = resid + x; h = rmsnorm(resid) · w`. Unfused, that is: read x,
read resid, write resid (add), then read resid, write h (norm), five passes over an `rows × cols`
tensor. The fused kernel reads `x` and `resid` once, writes the updated `resid` and `h`: four
passes. It uses the variant 4 design: the bf16-rounded sum is stored to `resid` and *kept in
registers* for the normalize, so the four passes are the true DRAM traffic and nothing is
re-read. On a 273 GB/s machine (GB10) a saved pass over a `4096 × 4096` bf16 tensor is ~32 MB ≈
0.12 ms, which is more than the whole norm costs. At the RTX 5090's 1,792 GB/s spec the same pass
is ≈ 0.018 ms: still a whole pass and a whole launch removed, but 6.5× less wall time. The sum is
rounded to bf16 before it is squared so the stored residual and the normalized value agree
bit-for-bit, matching PyTorch's `(resid + x).rms_norm(...)` in bf16. Rows wider than 32,768
elements use the older two-pass warp-per-row kernel.

### Into fp8 for the next GEMM

On the engine's fp8-tok path every norm feeds a projection that wants e4m3 rows with one scale
per token. As two ops, `add_rmsnorm_` writes the bf16 row, then `fp8_quantize(h, "row")` reads
it back, takes its amax and writes the e4m3 bytes and the scale: two launches and a bf16 row
that lives for one kernel. `add_rmsnorm_fp8_` and `rmsnorm_fp8` are the fused add and variant 4
with the quantizer moved into the tail of the kernel. After the normalize each thread rounds
its values to bf16, exactly as the bf16 kernel stores them, and keeps them in registers. The
group reduces their amax (a second reduction on its own shared array, so the two need no
barrier between them), thread 0 of the group writes the power-of-two scale, and every thread
converts its values to e4m3 and stores 8 bytes per vector. Per element that is 7 bytes instead
of 11 for the add form (read x and resid, write resid and q; the two ops also write h and read
it back) and 3 instead of 7 for the plain norm.

The output is the two ops' bits, not something close to them:

- the sum of squares is the bf16 kernel's, element for element: the same group size (picked
  from `cols` only), the same loop order, the same fp32 FMAs and `rsqrtf`;
- the values quantized are the bf16 values that kernel would have stored (round to nearest
  even), widened back to fp32, which is exact;
- the scale and the e4m3 conversion are the same two functions. `pow2_scale` and `store8`
  moved out of `fp8quant.cu` into `src/kernels/fp8quant.cuh`, which both files include. The
  amax is a max, so a different reduction tree (a group of 32 to 1,024 threads instead of
  `fp8_quantize`'s 256-thread block) still gives the same number.

The tests compare the bytes, the scales and the updated residual with `torch.equal` on every
group size, a ragged row (2,056 columns, 257 vectors over 128 threads), the widest row the
registers take (32,768), an all-zero row and weight scales of 2^-100 and 2^60. Only the
single-pass kernel exists in fp8: a row over 32,768 elements or unaligned storage raises
instead of falling back, because the fallback kernels reduce in another order and would not
match.

It is batch invariant for the same reason the norm is: one group per row, sized from `cols`,
so a row's bits don't depend on how many rows came with it. `scripts/check_invariance.py`
has both ops (same bits at 1, 8, 32 and 256 rows), and the batch-invariant engine keeps the
fusion on. `SparkModel.fuse_norm_fp8 = False` runs the two ops instead; with the
batch-invariant rungs (none of them splits K) the engine's logits are `torch.equal`
both ways, at prefill and at decode.

## Memory-traffic formulas (used for the GB/s column)

```
rmsnorm      bytes = 2 · rows · cols · sizeof(T) + cols · sizeof(T)
add_rmsnorm  bytes = 4 · rows · cols · sizeof(T) + cols · sizeof(T)
add_rmsnorm_fp8_  bytes = 7 · rows · cols   (two ops: 11 · rows · cols)
rmsnorm_fp8       bytes = 3 · rows · cols   (two ops:  7 · rows · cols)
GB/s         = bytes / (median_ms · 1e-3) / 1e9
% of peak    = GB/s / 1792     (RTX 5090; 273 on the GB10)
```

The bench validates every variant against a double-precision CPU reference
(tolerance 1e-4 for f32, 2e-2 for bf16 outputs) before timing it, and exits
non-zero if any check fails.

## What to look at in Nsight Compute

```
ncu --set full -k regex:rmsnorm ./build/bench_rmsnorm --rows=16384 --cols=8192 --iters=3
```

| Metric | Meaning | Measured, RTX 5090, f32 16384×8192 (`results/ncu_rmsnorm_v3.txt`, `_v4.txt`) |
|---|---|---|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | share of peak DRAM bandwidth actually used | v3 82.8% → v4 85.7% (1.46 → 1.51 TB/s at ncu's locked 2.5 GHz) |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | achieved occupancy | v3 93.6% (40 registers) → v4 64.1% (62 registers, 2/3 theoretical): enough, the bus is the limit either way |
| `lts__t_sector_hit_rate` (L2 hit rate) | how much of the second pass is served by L2 | v3 31.0% (part of the re-read goes to DRAM) → v4 0.65% (there is no re-read) |
| `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` / `…_requests` | sectors per request (coalescing) | 32 → 4 from v0 to any vectorized variant |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | shared-memory bank conflicts | ~0 for v3 and v4 (32 floats of smem per reduction) |

## RTX 5090 notes (measured)

- **The L2 is 96 MB, and it hides in the small shapes.** Any bench shape whose input fits
  (4096×1024 bf16 is 8 MB, 4096×4096 f32 is 64 MB) stays resident across the 100 timing
  iterations, so those rows report L2 bandwidth: 2–4 TB/s, above the 1,792 GB/s DRAM spec. That
  is why the 16384×8192 shape (256 MB of bf16 input, 512 MB of f32) was added and why it is the
  headline row: everything at or above 4096×8192 f32 is DRAM-bound and lands at 1.4–1.5 TB/s.
- **The realistic roof is 1,532 GB/s**, the `bandwidth_copy` / `cudaMemcpy` figure (85.5% of
  spec). Against that, variant 4 is at 99% (bf16, 1,529 GB/s) and 99% (f32, 1,518 GB/s) on
  16384×8192; the fused kernel at 98.5% (1,509 GB/s).
- **Variant 0 got relatively worse, as expected.** 36 GB/s in bf16 is 2% of the bus; the
  best-variant speedup over it is 10–105× depending on how much of the shape sits in L2.
- **The `cols > 8192` crossover to variant 3 did not survive.** Variant 3 wins or ties
  variant 2 at every width, including 1024: with ≈ 10.5 GB/s of bus per SM, one warp per row
  runs out of loads in flight much earlier than on LPDDR5X, and the block-per-row reduction is
  cheap. Variant 4 then takes over wherever the shape leaves L2: f32 4096×8192 goes from 1,424
  (v3) to 1,523 GB/s (v4), bf16 4096×8192 from 1,621 to 1,800 GB/s (a partly L2-resident
  shape: 64 MB in, 64 MB out), and the fused kernel on 4096×8192 from 0.2037 to 0.1791 ms.
  On the fully DRAM-bound 16384×8192 bf16 shape the gap closes to 1,516 vs 1,529 GB/s: once
  the second read is cheap enough to overlap, both designs sit on the bus limit.
- **Fusion pays in launches as much as in bytes here.** Against PyTorch eager
  (`resid.add_(x)` then `F.rms_norm`), the fused kernel is 1.24× at 16384×8192 and 1.34× at
  4096×1024, and about even (0.99–1.03×) on the L2-resident middle shapes where eager's extra
  pass costs nothing. Plain `rmsnorm` beats `F.rms_norm` by 1.28–1.40× in f32 and 1.06–1.56×
  in bf16; the f32 gap is the single pass, the bf16 gap at 1024 columns is launch geometry.
- **Nsight Compute confirmed the story**: on the f32 16384×8192 shape the two-pass kernel
  shows a 31% L2 hit rate (the re-read is not free on a card that streams 256 MB per pass) and
  the single-pass kernel 0.65%, with DRAM throughput 82.8% → 85.7% of ncu's peak.
- `min_ms` sat within 2–3% of the median on every row of the run; the 300 ms clock ramp is
  enough for these short kernels, which never reach the 600 W limit.

## Timing

The rows used to be single launches, the median of 100. On 2026-11-08 I ran w1gemm's two-run
test ([w1gemm](w1gemm.md#timing)) on this bench: two runs back to back in two lock holds,
rows more than x% apart (55 rows), twice for each:

| rows | pair | > 2% | > 3% | > 5% | largest | largest ref_ms |
|---|---|---|---|---|---|---|
| single launches | first | 8 | 6 | 6 | 12.5% | 0.6% |
| single launches | second | 4 | 4 | 4 | 9.5% | 0.1% |
| chains of 20 | first | 0 | 0 | 0 | 0.7% | 0.6% |
| chains of 20 | second | 0 | 0 | 0 | 1.7% | 0.1% |

Any two of the four chain runs are within 1.7% on every row; any two of the four single-launch
runs have 2 to 6 rows past 5%, all of them rows under 20 µs. The six of the first pair are 4096
x 1024 and 4096 x 2048, 7 to 11 µs: bf16 v1 at 4096 x 1024 was 7.17 µs in one run and 8.06 in
the other, bf16 v3 at 4096 x 2048 10.78 and 9.47, f32 v2 at 4096 x 1024 9.50 and 10.50. Each is
a step of 0.9 to 1.3 µs on the median, not a drift, like the lone launches in w1gemm's note. So
the rows are chains now: the median and min per launch of 20 launches between one event pair
over 20 pairs (`spark::bench::time_chained`), v0 and the fused add+RMSNorm too. The
single-launch medians are on stderr and `--chain=1` gives the old rows. Against the
single-launch rows of the same evening 23 of 55 chain rows are more than 5% faster and none
slower: the rows under 20 µs lose 0.9 to 2.5 µs (bf16 v4 at 4096 x 1024 is 4.93 µs a launch in
the chain, 7.26 alone), the rows over 100 µs are within 2.6%. The tables below are the older
single-launch run.

## Results (RTX 5090)

From `results/rmsnorm.json` before it held chains (median of 100 single launches, CUDA 13.2,
driver 595.58, no display) and `results/torch_comparison.json` (same shapes, PyTorch 2.14
eager). `% peak` is relative to 1,792 GB/s; rows above 100% are L2-resident (input smaller than
the 96 MB L2), the DRAM-bound rows are the last one per dtype. A GB10 table (relative to 273
GB/s) is added when the Spark has been benchmarked.

| dtype | rows×cols | v0 ms | v1 ms | v2 ms | v3 ms | v4 ms | best | GB/s | % of 1,792 | best / v0 | vs torch |
|---|---|---|---|---|---|---|---|---|---|---|---|
| f32 | 4096×1024 | 0.3244 | 0.0111 | 0.0104 | 0.0094 | 0.0094 | v3 | 3579 | 199.7% (L2) | 34.6× | 1.28× |
| f32 | 4096×2048 | 0.6439 | 0.0176 | 0.0175 | 0.0154 | 0.0156 | v3 | 4351 | 242.8% (L2) | 41.7× | 1.29× |
| f32 | 4096×4096 | 1.2604 | 0.1113 | 0.1074 | 0.0767 | 0.0706 | v4 | 1901 | 106.1% (L2) | 17.8× | 1.10× |
| f32 | 4096×8192 | 2.4841 | 0.2651 | 0.2589 | 0.1873 | 0.1764 | v4 | 1522 | 84.9% | 14.1× | 1.40× |
| f32 | 16384×8192 | 2.4716 | 1.0553 | 1.0289 | 0.7564 | 0.7072 | v4 | 1518 | 84.7% | 3.5× | 1.37× |
| bf16 | 4096×1024 | 0.4599 | 0.0072 | 0.0082 | 0.0075 | 0.0073 | v1 | 2341 | 130.6% (L2) | 64.2× | 1.56× |
| bf16 | 4096×2048 | 0.9248 | 0.0113 | 0.0135 | 0.0102 | 0.0115 | v3 | 3277 | 182.9% (L2) | 90.3× | 1.07× |
| bf16 | 4096×4096 | 1.8483 | 0.0195 | 0.0236 | 0.0176 | 0.0176 | v3 | 3820 | 213.2% (L2) | 105.2× | 1.16× |
| bf16 | 4096×8192 | 3.6534 | 0.1133 | 0.1095 | 0.0828 | 0.0746 | v4 | 1800 | 100.4% (part L2) | 49.0× | 1.06× |
| bf16 | 16384×8192 | 3.6453 | 0.5149 | 0.5069 | 0.3545 | 0.3511 | v4 | 1529 | 85.3% | 10.4× | 1.07× |

The v0 time for 16384×8192 is lower than a 4× scaling of 4096×8192 would give because the
naive kernel is occupancy-bound, not bandwidth-bound: 64 blocks of it fill more of the 170 SMs
than 16 do.

| fused add+RMSNorm (bf16) | rows×cols | ms | GB/s (4 passes) | % of 1,792 | vs torch eager (`add_` + `rms_norm`) |
|---|---|---|---|---|---|
| | 4096×1024 | 0.0092 | 3641 | 203.2% (L2) | 1.34× |
| | 4096×2048 | 0.0175 | 3841 | 214.4% (L2) | 0.99× |
| | 4096×4096 | 0.0316 | 4254 | 237.4% (L2) | 1.03× |
| | 4096×8192 | 0.1791 | 1499 | 83.6% | 1.02× |
| | 16384×8192 | 0.7116 | 1509 | 84.2% | 1.24× |

| fp8 norms, hidden 4096 | rows | `add_rmsnorm_fp8_` us | `add_rmsnorm_` + `fp8_quantize` us | speedup | `rmsnorm_fp8` us | `rmsnorm` + `fp8_quantize` us | speedup |
|---|---|---|---|---|---|---|---|
| | 1 | 2.81 | 3.88 | 1.38x | 2.35 | 3.48 | 1.48x |
| | 8 | 3.27 | 4.50 | 1.38x | 2.66 | 3.88 | 1.46x |
| | 64 | 3.41 | 4.69 | 1.38x | 2.77 | 3.99 | 1.44x |
| | 512 | 4.42 | 6.49 | 1.47x | 3.59 | 5.67 | 1.58x |
| | 2048 | 14.40 | 21.11 | 1.47x | 7.96 | 15.28 | 1.92x |
| | 8192 | 153.60 | 226.97 | 1.48x | 22.33 | 106.00 | 4.75x |

From `results/norm_fp8.jsonl` (`scripts/bench_norm_fp8.py --part op`): 20 calls captured in one
CUDA graph, the median of 50 replays divided by 20, so a row is what a graphed decode step pays
per call. The inputs are not rotated, as in the engine: up to 2048 rows everything stays in L2.

- **1 to 64 rows is a launch and two reductions, not bytes.** The fused add costs 0.16 to 0.28
  us more than `add_rmsnorm_` alone (2.81 against 2.65 us at one row) and saves the whole
  `fp8_quantize` launch, 1.53 to 1.63 us in a graph. Eager, where each op also pays the host's
  launch path, the two ops are 7.15 to 7.32 us and the fused one 4.17 to 4.30, 1.70 to 1.72x.
- **From 2048 rows it is bytes.** The fused add at 2048 rows (14.40 us) is faster than
  `add_rmsnorm_` alone (15.77): it writes one byte per element where that writes two. At 8192
  rows its 235 MB take 153.6 us, 1,529 GB/s, the copy roof.
- **The 4.75x at 8192 rows is the L2.** `rmsnorm_fp8` only reads x (64 MB), which stays in the
  96 MB L2 between calls, so it runs at 4,508 GB/s by the traffic model; the two ops also write
  and read back a 64 MB h, which pushes x out. The engine runs `rmsnorm_fp8` once per forward
  (layer 0's attention norm, on the embeddings); the add form is the one it runs everywhere
  else.

| fp8-tok engine, 8 layers | batch | decode ms/step, two ops | fused | prefill tok/s, two ops | fused |
|---|---|---|---|---|---|
| | 1 | 2.086 | 2.069 | 119,418 | 119,884 |
| | 8 | 2.234 | 2.215 | 134,043 | 135,930 |
| | 32 | 2.481 | 2.463 | 132,092 | 135,010 |
| | 64 | 3.007 | 2.987 | 129,569 | 132,135 |

`scripts/bench_norm_fp8.py --part engine`: bench_serve's static batches (prompts log-uniform
in 64 to 2048 tokens, 64 new tokens each, greedy), random weights, bf16 cache, CUDA graphs,
the two modes alternated three times on one model and the median of each kept. The three runs
of a mode agreed within 3 us a step. Decode drops 17 to 19 us a step (0.6 to 0.8%): 16 norms
at 1.1 to 1.2 us each, what the graph rows above say. Prefill gains 0.4% at 884 tokens and 1.4
to 2.2% from 6,073 up, where the norms move bytes. An fp8-tok step still runs `fp8_quantize`
before o and down and a row-scale multiply after every projection; those are not part of this.

