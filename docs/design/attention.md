# Attention: fused scaled-dot-product forward

`O = softmax(Q K^T / sqrt(D)) V` per (batch, head), for `Q, O = [B, H, S_q, D]` and
`K, V = [B, H, S_kv, D]`, row-major with `D` contiguous. bf16 in and out, fp32 for every score,
exponential and accumulator. MHA only (one K/V head per Q head), `D` in {64, 128}, optional
causal mask, no dropout, no bias, forward only. Source: `src/kernels/attention.cu`. Bench:
`bench_attention` (validates every variant against a CPU double-precision reference). The
library comparison is `torch.nn.functional.scaled_dot_product_attention` in
`scripts/bench_torch.py`, which on this card and this torch picks the FlashAttention-2 kernel
for every shape below.

The causal mask hides key `j` from query `i` when `j > i`, the top-left alignment
`is_causal=True` uses in torch. A decode step is `S_q = 1` against a cache of `S_kv` keys with
no mask.

## Why a fused kernel

The unfused version materializes `S = Q K^T` (`S_q x S_kv` fp32 per head: 64 MB per head at
4096 tokens, 2 GB for 32 heads), runs a softmax over it, and multiplies by `V`. Three kernels,
and the middle one is memory-bound on a matrix that never needed to exist. The flash-attention
formulation keeps a running softmax per query row and walks the keys in tiles, so the scores of
a tile live in registers between the two matrix products and the only DRAM traffic is Q, K, V
once and O once.

## What the numbers mean

FLOPs, as FlashAttention counts them: two matrix products of `S_q x S_kv x D`,

```
FLOP = 4 · B · H · S_q · S_kv · D          (halved under the causal mask)
```

The halving is the convention, not what the kernel does: with a 128-row Q tile and 64-key KV
tiles, the tiles on the diagonal are computed whole and masked, so at `S = 4096` the kernel
does 1,056 of 2,048 tile products, 51.6%. The TFLOPS column undercounts causal work by 3%.

Bytes, the traffic floor: `2 · (Q + O + K + V)` bf16 elements,

```
bytes = 2 · 2 · B · H · (S_q + S_kv) · D
```

At `B = 1, H = 32, S = 4096, D = 128` that is 128 MB against 275 GFLOP: 2,048 FLOP/byte,
fourteen times the card's 144 FLOP/byte bf16 ridge. Prefill attention is compute-bound by a
wide margin, and the only question is how close to the tensor-core peak the softmax and the
tile loads let it get. A decode step is the opposite: `S_q = 1` against 4096 keys is 32 MFLOP
per head over 2 MB of K and V, 16 FLOP/byte, so it is bound by streaming the cache.

## The online softmax

Per query row the kernel keeps a running max `m`, a running sum `l` and an unnormalized output
row `o`. For a new tile of scores `s_j`:

```
m' = max(m, max_j s_j)
o  = o · exp(m − m') + Σ_j exp(s_j − m') · v_j
l  = l · exp(m − m') + Σ_j exp(s_j − m')
m  = m'
```

and at the end `O = o / l`. The rescale `exp(m − m')` is what makes it exact: every earlier
contribution was scaled relative to the old max and is brought to the new one. It is 1 when the
max does not move, which after the first few tiles is most of the time, and the multiply is
cheap enough (64 FMULs per lane per tile in variant 2) that nothing checks for that.

Two details that are the same in every variant. Scores are scaled by `log2(e) / sqrt(D)` rather
than `1 / sqrt(D)`, so the exponential is one `ex2.approx` (a MUFU instruction) instead of a
multiply plus `ex2`; `m` and `l` live in that scaled domain and the final division cancels it.
And a masked score is `-inf`: `ex2(-inf − m') = 0` with no branch, and the guard `m_use = 0` when
`m' = -inf` keeps a row that has seen no key yet from computing `-inf − (-inf)`. That case
cannot arise for a real row (key 0 is visible to every query and tile 0 is always first), but
rows past `S_q` in a zero-filled tile go through the same code.

## The ladder

| variant | idea | what it fixes |
|---|---|---|
| 0 | one warp per query row, lanes own `D/32` columns of q, k, v and o, keys one at a time with a shuffle reduction per score | baseline: correct online softmax, no reuse of K or V between rows |
| 1 | 128-row Q tile, 64-key K/V tiles converted to fp32 in shared memory, 8 warps x 16 rows, CUDA-core FMAs on 4x8 (scores) and 4xD/8 (output) register tiles, the next tile prefetched into registers | every K/V element loaded once per 128 rows; FlashAttention-2 without tensor cores |
| 2 | `mma.sync.m16n8k16` + `ldmatrix`: Q fragments in registers for the whole KV loop, S and PV on the tensor cores, P repacked from the S accumulators, 3-stage `cp.async` pipeline on K and V | the tensor cores, and no shared-memory round trip for P |
| 3 | variant 2's kernel with a schedule: the tiles of the last partial wave are split along the keys over the idle SMs and merged by a combine kernel; `S_q <= 64` runs a 64-row tile | wave quantization on 170 SMs, and a decode step that no longer pays for 128 rows |

### Variant 0

`4 · S_kv · D / 32` FMAs per lane per row and one five-step shuffle reduction per key. Every
query row re-reads all of K and V from L2: 275 GB of L2 traffic for the 4096 causal shape. It
runs at 10 TFLOPS and exists to be validated against and to validate the rest.

### Variant 1: CUDA cores

The block owns 128 query rows (8 warps of 16) and walks the keys in tiles of 64. Q stays in
shared memory as bf16 (32 KB); each K and V tile is converted to fp32 on its way into shared
memory (32 KB each), so the inner loops are `LDS` and `FFMA` with no conversions: a converted
element serves 128 rows, where converting in the loop would cost one instruction per FMA.
The next tile's 32 KB is loaded into 32 registers per thread before the current tile is
consumed, which hides the L2 latency behind the 5 µs of FMAs a tile takes.

Lane mapping, with `rg = lane / 8` and `kl = lane % 8`: the score micro-tile is rows
`rg + 4·rr` (four rows) by keys `kl + 8·kk` (eight keys), so per four d-values a lane does 4
`LDS.64` of Q, 8 `LDS.128` of K and 128 FMAs. For `O += P V` the same lane owns the same four
rows by `D/8` float4 chunks of the output, the score `p` for key `j` comes from lane
`rg·8 + j%8` by shuffle, and per key it is 4 shuffles, `D/32` `LDS.128` and `D/2` FMAs. Bank
conflicts: 16-byte chunks are XOR-swizzled by `row % 8`, so the eight `kl` lanes read eight
rows of K in eight bank groups, the four `rg` groups read four rows of Q in four, and the eight
`kl` lanes read eight consecutive chunks of one V row. The staging stores needed one more
trick: eight lanes store together, fp32 chunks `c` and `c + 8` share banks, and the even chunks
0..14 a lane group would write in one instruction collide two ways, so lanes with bit 2 of
their chunk index set write their odd chunk first. Nsight went from 17.3 M to 5 K conflicts.

255 registers, no spills, one block of 8 warps per SM (96 KB of shared memory). It is
issue-bound: Nsight shows the issue slots 72% busy with the FMA pipe 64% active and shared
loads at 67% of the wavefront rate, and 2.86 G instructions for the 4096 causal shape where
variant 3 issues 334 M. That is the CUDA-core wall, at 49 TFLOPS, 40% of the 123 TFLOPS FMA
peak. Fixing the store conflicts did not move the time, which says the same thing.

### Variant 2: tensor cores

Same 128 x 64 block tile and 8 warps of 16 rows. The tile fits `mma.sync.m16n8k16` naturally:
a warp's 16 rows are one `m16`, the 64 keys are eight `n8` tiles of S, and the `D` columns of
O are `D/8` `n8` tiles. Per KV tile per warp: 64 `mma.sync` for `S = Q K^T` (8 n-tiles x
`D/16` k-steps), 64 for `O += P V` (`D/8` n-tiles x 4 k-steps), 64 `ldmatrix.x4`.

**Q in registers.** The A fragments of Q for all `D/16` k-steps (32 registers at `D = 128`)
are loaded once with `ldmatrix.x4` through the first pipeline stage, and the KV loop never
touches Q again.

**K as it lies.** `mma.sync ... row.col` wants B as `[n][k]` with `k` contiguous. K in memory is
`[key][d]` with `d` contiguous, and `n = key`, `k = d`, so a plain `ldmatrix.x4` (no
transpose) on 16 key rows gives the B fragments of two `n8` key tiles: matrices 0 and 2 (keys
0-7, d 0-7 and 8-15) are `b0, b1` of the first, matrices 1 and 3 are the second. V is the
other way round: `[key][d] = [k][n]`, the k-major layout `ldmatrix.x4.trans` was made for, as
in `hgemm` v3.

**The fragment-layout trick.** After `S = Q K^T` lane `(g = lane/4, c = lane%4)` holds, for
every `n8` tile `nj`, `S[g][8nj + 2c]`, `S[g][8nj + 2c + 1]`, `S[g+8][8nj + 2c]` and
`S[g+8][8nj + 2c + 1]`. The A fragment of the next `mma` (`P[16 rows][16 keys]`) wants lane
`(g, c)` to hold `P[g][2c..2c+1]`, `P[g+8][2c..2c+1]`, `P[g][2c+8..2c+9]` and
`P[g+8][2c+8..2c+9]`, which is exactly the accumulator of n-tiles `2kt` and `2kt + 1` packed
to `bf16x2` pairs. So `P` is four `cvt` per k-step and never leaves the registers: no shared
memory, no barrier, between the softmax and the second product.

**Softmax on the accumulators.** A row is 16 values per lane (two per n-tile) plus the three
other lanes that share `g`, so the row max is 15 `fmax` and two `shfl.xor` (1 and 2), and the
row sum is kept as a per-lane partial that is merged the same way once, in the epilogue. The
rescale touches the 64 O accumulators.

**Pipeline.** K and V tiles (16 KB each at `D = 128`) stream through three `cp.async` stages,
96 KB of dynamic shared memory, one `__syncthreads` per tile, the same structure as `hgemm` v3.
Rows past `S_kv` are zero-filled with `cp.async` src-size 0 and masked to `-inf` in the
scores. Rows past `S_q` are zero-filled in Q and skipped in the epilogue, so any `S_q` and
`S_kv` work. Causal: a Q tile at rows `q0..q0+127` needs KV tiles up to `q0 + 127`, so the
tiles above the diagonal are never loaded, and only tiles whose last key exceeds `q0` run the
mask code (two per Q tile).

**Order.** Under the causal mask the Q tiles are launched heaviest first (the last Q tile of
the sequence needs 64 KV tiles, the first needs 2), so the tail of the grid is short.

248 registers, no spills, one block per SM: 8 warps, two per scheduler. Each `mma.sync`
occupies the tensor pipe of its scheduler for 32 cycles on this card (258.7 TFLOPS over
170 SMs at 2.98 GHz is 512 FLOP per SM-clock; an `m16n8k16` is 4,096 FLOP), so the ~400
non-`mma` instructions a warp issues per tile fit inside the 4,096 cycles its 128 `mma` take,
if the other warp on the scheduler keeps the pipe fed meanwhile. Mostly it does: see the
Nsight numbers below.

### Variant 3: the schedule

Same kernel, two scheduling changes.

**Wave quantization.** Non-causal 4096 is 32 Q tiles x 32 heads = 1,024 equal blocks on 170
SMs at one block per SM: 6.02 waves, paid as 7. Variant 2 measures 1.41 ms there, and 6.02/7
of that is 1.21 ms. The fix is the one `hgemm` v3 uses on its last partial wave: the
`tiles mod 170` tail tiles are split `split = min(T, 170 / tail)` ways along the keys (4 tiles
x 42 slices at 4096), each slice writes its unnormalized fp32 `O` rows plus `(m, l)` to a
workspace, and a combine kernel merges the slices per row with the same rule as the online
softmax:

```
M = max_i m_i;   L = Σ_i l_i · 2^(m_i − M);   O = Σ_i O_i · 2^(m_i − M) / L
```

The extra wave now lasts one slice instead of a whole tile. Measured: 1.224 ms, 224.5 TFLOPS,
13% faster than variant 2 on that shape. Under the causal mask the heaviest-first order already
evens the waves out and the tail tiles are the lightest ones (two KV tiles each), so the split
does nothing there, within noise. Slices of a split tile are merged in fp32 in a fixed order,
so the result is reproducible; the workspace is one per device and shared across streams.

**Decode.** With `S_q = 1` the 128-row tile computes 127 zero rows per KV tile, 8,192
tensor-pipe cycles per tile per block, 64 tiles per head: 0.20 ms for a 64 MB cache the DRAM
could stream in 43 µs. `S_q <= 64` therefore runs the same kernel instantiated with 4 warps
(a 64-row tile), which halves the wasted work, and the split above turns 32 blocks into 160
(32 heads x 5 slices of 13 tiles) so the whole card streams the cache. Measured with K and V
rotated through copies that exceed the 96 MB L2, as `bench_hgemm` does for the weights of a
decode GEMM: 49 µs, 1,369 GB/s, 89% of the 1,532 GB/s `cudaMemcpy` roof, against 74 µs for
torch's split-KV flash kernel under the same rotation. A 16-row tile would cut the remaining
compute four times more; the number says it is not needed at this cache size.

## Correctness

`bench_attention` checks every output row of the small shapes against a CPU double-precision
reference computed from the same bf16-rounded inputs, and 64 sampled `(b, h, row)` triples
(always including the first and last row of the first head, the two extremes of the causal
mask) on the large ones. Q and K are uniform in [-2, 2] and V in [-1, 1], so the scaled scores
have a spread of a few units and the running max actually moves. Tolerance:
`max|O − O_ref| <= 0.02 · max|O_ref| + 1e-3`, one bf16 rounding of an fp32 result on our side
plus the bf16 rounding of P before the P V product in variants 2 and 3, the same form as
`bench_hgemm`. Measured errors are 1e-3 to 2.6e-3 against a tolerance of 2e-2 or more. The
pytest parity test compares every variant with `F.scaled_dot_product_attention` in fp32 on
the same bf16 inputs, including `S = 200`, `S = 1000`, `S_q = 1` and `S_q = 7` against a
longer cache, both masks and both head sizes.

## RTX 5090 notes (measured)

Driver 595.58, CUDA 13.2, medians of 50 launches after the clock ramp, `bench_attention`
defaults. Nsight Compute runs at a fixed 2.53 GHz; the timed runs boost higher.

- **The roof.** 258.7 TFLOPS of dense bf16 `mma.sync` at 2,976 MHz (`bench_peak`), 239 at the
  2.75 GHz a sustained GEMM throttles to. Variant 3 reaches 224.5 TFLOPS on the non-causal
  4096 shape with the tensor pipe 90.1% active in Nsight, which puts the clock during the timed
  run at about 2.86 GHz: attention throttles less than a GEMM at the same pipe utilization.
  87% of the measured peak, 94% of the throttled roof.
- **What limits variant 2 / 3.** Nsight on 4096 causal: tensor pipe **89.3% active** (90.1%
  non-causal), issue slots 13.6% busy, 0.55 instructions per cycle, dominant stall
  `math_pipe_throttle` at 9.7 of the 14.5 cycles between a warp's issues, then `wait` (2.1,
  fixed-latency dependencies), `barrier` 0.3, `long_scoreboard` 0.3, `mio_throttle` 0.3.
  Shared-memory bank conflicts: 0 (16 on the non-causal run). Shared load wavefronts 22% of
  peak. L2 hit rate 90.4% (94.7% non-causal), DRAM throughput 8.5% of peak. So the kernel is
  issuing to the tensor pipe as fast as it takes work, the loads are hidden, and the 10% of
  idle pipe is the softmax gaps where both warps of a scheduler are between products, plus
  the per-block prologue and epilogue. Occupancy is 16.7% (8 warps), fixed by the 248
  registers and the 96 KB tile; a second block per SM would need both halved.
- **The causal diagonal.** 215 TFLOPS by the halved count is 220 by the tiles actually
  computed. A 64-row Q tile would waste half as much on the diagonal at twice the K/V traffic
  per FLOP; not tried.
- **Variant 1** is issue-bound on the `LDS` + `FFMA` mix, 49 TFLOPS, 40% of the 123 TFLOPS FMA
  peak, 8.6x the instruction count of variant 3 for the same work. The fp32 staging of K and
  V in shared memory (32 KB each) is what keeps conversions out of the loop; register
  prefetch of the next tile costs 32 registers and takes it to 255.
- **Against torch** (`F.scaled_dot_product_attention`, which picked its FlashAttention-2
  kernel on every shape; cuDNN's SDPA forced through `sdpa_kernel` is 3 to 7% faster than
  that on the prefill shapes and still behind variant 3):

| shape | v3 ms | v3 TFLOPS | torch ms | torch TFLOPS | v3 / torch |
|---|---|---|---|---|---|
| b1 h32 s4096 d128 | 1.246 | 220.5 | 1.458 | 188.6 | 1.17x |
| b1 h32 s4096 d128 causal | 0.644 | 213.3 | 0.789 | 174.1 | 1.23x |
| b1 h32 s8192 d128 causal | 2.498 | 220.1 | 2.776 | 198.0 | 1.11x |
| b4 h32 s2048 d128 causal | 0.705 | 194.9 | 0.762 | 180.3 | 1.08x |
| b1 h32 s4096 d64 causal | 0.341 | 201.4 | 0.417 | 165.0 | 1.22x |
| b1 h32 sq1 skv4096 d128 (decode) | 0.054 | 1,250 GB/s | 0.074 | 907 GB/s | 1.38x |

  (`scripts/bench_torch.py` timings, 50 iterations, K and V rotated past L2 for the decode
  row; the C++ bench, which allocates less between launches, gets 49 µs there.)

## Results (RTX 5090, sm_120, CUDA 13.2, driver 595.58)

From the default `bench_attention` sweep (median of 50). TFLOPS by the halved causal count;
the decode row in GB/s of Q, K, V and O once.

| shape | v0 ms / TFLOPS | v1 | v2 | v3 |
|---|---|---|---|---|
| b1 h4 s512 d128 | 0.0810 / 6.6 | 0.0851 / 6.3 | 0.0298 / 18.0 | **0.0124 / 43.5** |
| b1 h4 s512 d128 causal | 0.0788 / 3.4 | 0.0850 / 3.2 | 0.0298 / 9.0 | **0.0185 / 14.5** |
| b1 h4 s512 d64 | 0.0564 / 4.8 | 0.0441 / 6.1 | 0.0155 / 17.4 | **0.0084 / 31.8** |
| b1 h4 s200 d128 causal | 0.0298 / 1.4 | 0.0441 / 0.9 | 0.0155 / 2.7 | **0.0124 / 3.3** |
| b1 h32 s4096 d128 | 26.28 / 10.5 | 5.096 / 53.9 | 1.414 / 194.5 | **1.224 / 224.5** |
| b1 h32 s4096 d128 causal | 13.76 / 10.0 | 2.799 / 49.1 | **0.639 / 215.2** | 0.640 / 214.8 |
| b1 h32 s8192 d128 causal | 57.06 / 9.6 | 11.24 / 48.9 | **2.486 / 221.1** | 2.487 / 221.1 |
| b4 h32 s2048 d128 causal | 13.10 / 10.5 | 2.897 / 47.4 | 0.687 / 200.1 | **0.686 / 200.4** |
| b1 h32 s4096 d64 causal | 9.389 / 7.3 | 1.150 / 59.8 | **0.329 / 209.1** | 0.331 / 207.8 |
| b1 h32 sq1 skv4096 d128 | 1.258 / 53 GB/s | 0.668 / 100 GB/s | 0.200 / 335 GB/s | **0.049 / 1,369 GB/s** |

The small shapes are where the tail split matters most: 16 tiles on 170 SMs become 128
slices, 2.4x on the 512-token shape. Variant 1's D = 64 time moved between 1.15 and 1.57 ms
across runs on a card other jobs were also heating; the other rows repeat to within 1%.

## What remains

- **Warp specialization.** The 10% of idle tensor pipe is the softmax. FlashAttention-3's
  answer is producer/consumer warps, or two consumer warp groups that ping-pong so one is in
  its softmax while the other issues `mma`. On this card that means fitting two blocks per
  SM (registers under 128, tiles under 48 KB) or a 16-warp block with a hand-scheduled
  barrier pattern.
- **GQA.** `H_kv < H_q` is a stride on the K/V pointer and a change to the parity test; the
  kernel does not care.
- **Backward**, which needs the log-sum-exp saved from the forward (`m + log2(l)` per row,
  free here) and two more kernels.
- **A 16-row decode tile** for caches past 16K tokens, where the 64-row tile's compute would
  start to show against the DRAM time again.
