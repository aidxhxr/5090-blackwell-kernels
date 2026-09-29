# Attention backward

The gradients of `O = softmax(Q K^T / sqrt(D)) V` with respect to Q, K and V, from the
forward's inputs, its output O, the upstream gradient dO and the log-sum-exp L the forward
saves per query row. Same shapes as the forward: `Q, O, dO, dQ = [B, H_q, S_q, D]`,
`K, V, dK, dV = [B, H_kv, S_kv, D]`, bf16 in and out, fp32 for every score and accumulator,
`D` in {64, 128}, optional causal mask (top-left, as `is_causal`), grouped-query attention with
dK and dV summed over the `H_q / H_kv` query heads that read each K/V head. Source:
`src/kernels/attention_bwd.cu`; the forward's `lse` output is in `src/kernels/attention.cu`
(the forward note has the details). Bench: `bench_attention_bwd` (validates every variant and
both dQ modes against a CPU double-precision reference). The library comparison is torch
autograd through `F.scaled_dot_product_attention` in `scripts/bench_torch.py --only
attention_bwd`, which on this card runs FlashAttention-2's backward unless cuDNN is forced.
Python: `attention_fwd` (O and L), `attention_bwd`, and `attention_with_grad`, a
`torch.autograd.Function` over the two.

## The math

With `P = exp(Q K^T / sqrt(D) − L)` recomputed from the saved L:

```
dV = P^T dO
dP = dO V^T
dS = P ∘ (dP − Dv),    Dv_i = Σ_j P_ij dP_ij = dO_i · O_i
dQ = dS K / sqrt(D)
dK = dS^T Q / sqrt(D)
```

Five products of `S_q x S_kv x D` where the forward has two (S and dP are recomputed or new,
then dV, dQ, dK). FLOPs as FlashAttention counts them:

```
FLOP = 10 · B · H_q · S_q · S_kv · D          (halved under the causal mask)
```

2.5 times the forward. Bytes, the floor: Q, O, dO read and dQ written per query head, K, V
read and dK, dV written per K/V head, `2 · 4 · B · (H_q · S_q + H_kv · S_kv) · D`. At
`B = 1, H = 32, S = 4096, D = 128` that is 268 MB against 344 GFLOP causal, 1,280 FLOP per
byte: compute-bound, like the forward, and the question is again how close to the tensor-core
peak the non-`mma` work lets it run.

`Dv_i = dO_i · O_i` is the identity FlashAttention-2 uses so the row sum `Σ_j P_ij dP_ij`
never needs the whole row at once: it is computed up front from O, which the forward already
wrote. As in the forward, `P` is formed in the log2 domain, `2^(s · log2(e)/sqrt(D) − L ·
log2(e))`, one FFMA and one `MUFU.EX2` per score.

## Every variant: the preprocess

One warp per query row writes `{L_i · log2(e), Dv_i}` as a float2 into a workspace whose rows
are padded per head to a multiple of 128, `{0, 0}` in the padding. A tile then never needs a
bounds check on those rows: a query row past `S_q` has zero-filled Q and dO, so its P is
`2^(0 − 0) = 1` against `dO = 0` and `dP = 0`, and it adds nothing to dV, dS, dK or dQ. The
same kernel zeroes the fp32 dQ buffer the atomic variants add into. Zeroing it here, right
before the main kernel, has a side effect I found by moving it: the zeroed lines are still in
L2 when the first atomics arrive. Doing the zeroing in the previous call's dQ conversion
instead (the buffer left zero between calls) measured 4% slower at 4096 tokens, with the main
kernel unchanged and the preprocess 11 µs faster.

## The ladder

| variant | idea | what it fixes |
|---|---|---|
| 0 | one warp per query row for dQ and one per key row for dK and dV, a key (query) at a time, two shuffle reductions per score | baseline, and the check the others are measured against |
| 1 | FlashAttention-2's backward on `mma.sync`: a block owns 64 keys of one K/V head (4 warps x 16), keeps its dK and dV rows in registers over every query tile of every query head of the group, and adds dQ to an fp32 buffer with atomics | the tensor cores, K and V read once per key tile, no dK / dV reduction across blocks, GQA summed in registers |
| 2 | variant 1 sized for 99 KB: 128 keys per block, V's fragments in registers so the Q/dO tile can be double-buffered by `cp.async`, dQ of the previous tile deferred behind the loop's one barrier, causal tile skipping per warp, a two-level split of the grid chosen by simulating the block scheduler | the load latency, half the barriers, the partial waves (GQA's 256 key tiles on 170 SMs), the diagonal |
| 3 | variant 2's tile and products with the block barrier gone: Q, dO and (L, Dv) by TMA into three mbarrier stages issued by one lane, and the dS double buffer handed between warps by full / empty mbarriers | the barrier that started all eight warps in phase, so both warps of a scheduler reached their exponentials and dS at once |

`deterministic=True` (variants 1 to 3) replaces the dQ atomics with a separate pass; its own
section is below.

### Variant 0

The dQ kernel is the forward's variant 0 with the second product changed: per key, `s = q ·
k_j` and `dp = dO_i · v_j` by shuffle reductions, `ds = p (dp − Dv)`, `dq += ds k_j`. The dK /
dV kernel is the transpose, one warp per key row walking every query row of every head of the
group that sees it. Every row re-reads all of the other side from L2. It exists to be checked
against.

### Variant 1: FlashAttention-2's loop on `mma.sync`

The structure is FlashAttention-2's: the outer loop is over key tiles (one per block) and the
inner loop over query tiles, so dK and dV accumulate in registers for the whole block and only
dQ, a sum over key tiles, crosses blocks. The keys sit on the `m16` rows of the `mma`: a warp
owns 16 keys and computes the transposed scores, `S^T = K_w Q^T`, with the tile's query rows
on the `n8` columns. Then the forward's register trick works twice:

- the S^T accumulators, exponentiated in place, are the A operand of `dV += P^T dO` once packed
  to bf16 pairs (lane `(g, c)` holds keys `g, g+8` against rows `2c, 2c+1` of every n8 tile,
  which is exactly `m16k16` A layout with `k` = query row), and
- `dS^T = P^T ∘ (dP^T − Dv)` is formed on the same accumulators (dP^T = `V_w dO^T` has the same
  layout) and is the A operand of `dK += dS^T Q`.

So P and dS never go through shared memory for dK and dV. The B operands are Q and dO as they
lie: `[q][d]` is the `[n][k]` layout a plain `ldmatrix` gives for S^T and dP^T, and the
`[k][n]` layout `ldmatrix.trans` gives for dV and dK. L and Dv are per query row, so per
column here: a lane needs them for rows `8n + 2c` and `8n + 2c + 1`, which with the padded
float2 layout is one 16-byte shared load per n8 tile.

dQ is the part that does not fit. `dQ = dS K` sums over the block's keys, spread over its
warps, so each warp stores its `dS^T` rows (bf16, 32-bit stores) to shared memory, the block
syncs, and the `BQ x D` dQ tile is split over the warps: A is dS read from the key-major dS^T
by `ldmatrix.trans`, B is K through `ldmatrix.trans`, and each warp adds its fp32 piece to a
`[B, H_q, S_q, D]` buffer with `RED.E.ADD.F32x2` (the sm_90+ vector atomic, from
`atomicAdd(float2*)`). A last kernel converts the buffer to bf16 with the `1/sqrt(D)` scale.

The tile is 64 keys by 64 query rows, 4 warps, loads by `cp.async` waited for at once, two
barriers per query tile. dS^T rows are 128 bytes at `BQ = 64` (64 bytes at variant 2's
`BQ = 32`), and their 16-byte chunks are XORed with `row / 2` or `row` so both the 32-bit stores of a warp and the
eight rows an `ldmatrix` reads fall in eight bank groups (Nsight counts no shared-memory
bank conflicts in variant 1). Registers: 255 at `D = 128` with 36 bytes of spill (dK and dV
are 128 of them), 218 at `D = 64`. 73.5 KB of shared memory, so one block per SM: four warps,
one per scheduler, and the forward's variant 4 section already showed that one `mma.sync`
warp does not keep a scheduler's tensor pipe full. Nsight puts it at 68% busy on the causal
4096 shape, with `long_scoreboard` (the loads waited for in place) at 1.3 cycles per issue.
157 to 162 TFLOPS from 4096 tokens up.

### Variant 2: sized for 99 KB

Per 16 keys and per query row, the five products are the same work whatever the tile, so the
choices are about latency and overhead. The first constraint is the register file. A warp
holds dK and dV for its 16 keys over all `D` columns: 128 fp32 registers at `D = 128`, before
any of the per-tile state. That fixes 16 keys per warp and makes 8 warps, 128 keys, the
largest block that fits one per SM.

The second is shared memory. 128 keys of K and V are 64 KB, and a double-buffered 64-row Q and
dO tile another 64 KB. V, though, is only ever the A operand of its own warp's `dP^T = V_w
dO^T`: 16 keys by `D`, 32 registers, the same size as the Q fragments the forward keeps. So V
is staged once through the stage region, read into fragments, and never stored in shared
memory again. That leaves 32 KB of K, two stages of a 32-row Q and dO tile (16.6 KB each with
the L / Dv rows) and dS^T. At `D = 64` the Q tile is 64 rows. The `mma` count per query tile
per warp at `D = 128` is 32 for each of S^T, dP^T, dV, dK and dQ, 160 in all.

The first version of this rung kept variant 1's two barriers per query tile (one before the
tile is read, one between the dS^T stores and the dQ product) and measured 202 TFLOPS on the
4096 shape without the mask and 191 with it. The second barrier goes away by deferring the
dQ product one tile: dS^T is double-buffered, and the product for tile `t − 1` runs right
after the loop's barrier at the top of tile `t`, which is also what publishes tile `t − 1`'s
dS^T. One barrier per tile, 16 KB more shared memory (82.4 KB in all), 208 and 195 TFLOPS.

Under the causal mask a key tile starts at the first query tile that ends at or after its
first key, and a warp whose 16 keys all follow a tile's last row skips its products for that
tile (it still zeroes its dS^T rows, which the dQ product reads).

254 registers at `D = 128`, no spills; 250 at `D = 64`. Nsight: tensor pipe 84.7% busy on
the causal 4096 shape (86.8% without the mask), the barrier 0.9 cycles per issue.

#### The schedule

A block owns a whole key tile, so the grid has `B · H_kv · S_kv / 128` blocks, one per SM at a
time. Two shapes go wrong. Without the mask, 1,024 equal tiles at 4096 tokens are 6.02 waves
of 170; with GQA at 32/8 heads there are 256 tiles, four times the work each, and 256 on 170
SMs is two waves with the second half empty: the first run measured 159 TFLOPS there against
208 for the multi-head shape.

The fix is a split in two levels. Every tile is cut into `s1` items, contiguous ranges of its
(query head, query tile) iterations; the last partial wave of items is cut again into `s2`
pieces over the idle SMs; and a block whose tile is cut at all adds fp32 partials of dK and
dV to a workspace (zeroed by `cudaMemsetAsync`) that a small kernel converts, where a whole
tile stores bf16 from registers. `s2` follows the forward's tail rule, `min(iterations,
170 / tail)`. `s1` I first picked from a wave count that assumed equal tiles, and it was
wrong for the causal shapes, where key tile 0 sees every query row and the last one sees
four. So the host now simulates the block scheduler: blocks go out in index order to
whichever SM frees first, each costs its query tiles plus 0.3 of one for the prologue, and
the grid time is the makespan. The simulation alone then picked `s1 = 2` for the 1K and 2K
multi-head causal shapes, and they got slower (143 to 121 and 166 to 132 TFLOPS): a split
tile's dK and dV make a round trip through fp32 memory, a memset, one atomic add per slice
and a conversion read, which at 2K tokens is a third of the kernel. So the choice minimizes
the simulated time at about 4 µs per query-tile iteration plus that traffic at 1.5 TB/s, and
the answer is cached per shape. Under GQA a key tile carries `group` times the work for the
same traffic, and the split pays.

Forced splits against the rule (`SPARK_ATTN_BWD_SPLIT=s1`, TFLOPS, variant 2):

| shape | s1 = 1 | 2 | 4 | 8 | rule |
|---|---|---|---|---|---|
| b1 hq32 hkv8 s4096 d128 | 164.5 | **207.4** | 203.4 | 198.5 | 207.6 |
| b1 hq32 hkv8 s4096 d128 causal | 160.2 | 181.3 | **190.4** | 183.2 | 190.6 |
| b1 hq32 hkv8 s8192 d128 causal | 193.1 | **204.6** | 201.8 | 195.2 | 204.7 |
| b1 h32 s1024 d128 causal | **145.9** | 120.9 | 103.0 | 102.5 | 147.3 |

The rule lands on the best forced value on every row.

I also tried ordering the tiles in chunks of a few K/V heads (all key tiles of a chunk before
the next chunk) so the blocks in flight add to the dQ rows of fewer heads while those rows
are in L2. It lost 1% at 4096 tokens with a chunk of 11 heads and 8% with 4 (heaviest first
across all heads matters more), and moved the batch of four, the shape it was meant for, by
under 1%, so the order stays key tile major across all heads (`SPARK_ATTN_BWD_CHUNK` keeps
the knob).

### Variant 3: no block barrier

This is what the forward's variant 4 did for its K/V loop, applied to the Q/dO loop. Nsight on
variant 2 showed the tensor pipe 84% busy, with one `BRA` (the barrier's) holding 7% of all
stall samples and the rest of the idle time in the phases where both warps of a scheduler are
out of `mma` together, which the barrier makes likely by starting all eight warps in phase
every tile. The loop needs the barrier for three hand-offs, and each gets its own mbarrier:

- **Stages.** Q, dO and the (L, Dv) rows come by TMA: two 3-D tensor maps
  (`[B·H_q][S_q][D]`, boxes of 64 columns by `BQ` rows with the 128-byte swizzle, rows past
  `S_q` zero-filled by the copy engine) and a 1-D bulk copy for the float2 rows, three stages
  with a full / empty barrier pair each. Lane 0 of warp 0 issues them.
- **dS^T, written.** A "full" barrier per dS buffer that every thread arrives on after its
  stores; the dQ product of that tile waits on it.
- **dS^T, read.** An "empty" barrier per buffer that every thread arrives on after its dQ
  product; a warp waits on it before writing that buffer again two tiles later.

Per query tile a warp waits for its stage, runs S^T through dK, releases the stage, writes
dS^T into buffer `t mod 2`, then runs the dQ product of tile `t − 1`. Lane 0 of warp 0 then
refills the stage tile `t − 1` used with tile `t + 2`; every warp has released it, because
the full wait of the dQ product just before implies every warp wrote tile `t − 1`'s dS^T, and
each warp releases a stage before it writes dS^T. So the producer never waits. The warps can
drift up to a tile apart.

Shared memory is 96.9 KB at `D = 128` (32 KB of K, three 16 KB stages, three 256-byte L / Dv
rows, two 8 KB dS^T buffers, the barriers), 252 registers with no spills.

Nsight Compute, 4096 tokens, `D = 128`, the key-tile kernel alone, one launch at the clock
Nsight locks (2.53 GHz):

| | v1 causal | v2 causal | v3 causal | v2 | v3 |
|---|---|---|---|---|---|
| duration | 2.30 ms | 1.86 ms | 1.76 ms | 3.57 ms | 3.41 ms |
| tensor pipe (bf16 `mma` ops, % of peak) | 68.5% | 84.7% | 89.5% | 86.8% | 91.0% |
| issue slots busy | 9.2% | 13.5% | 13.5% | 13.5% | 13.4% |
| instructions executed | 365.0 M | 427.6 M | 400.0 M | 833.4 M | 775.0 M |
| stall cycles per issue: `math_pipe_throttle` | 4.23 | 8.64 | 8.82 | 8.83 | 9.13 |
| `wait` | 2.50 | 2.74 | 2.75 | 2.75 | 2.77 |
| `barrier` | 0.09 | 0.89 | 0.00 | 0.74 | 0.00 |
| `long_scoreboard` | 1.29 | 0.30 | 0.64 | 0.23 | 0.51 |
| `short_scoreboard` | 0.72 | 0.55 | 0.27 | 0.55 | 0.27 |
| `branch_resolving` | 0.04 | 0.09 | 0.68 | 0.09 | 0.58 |
| registers | 255 | 254 | 252 | 254 | 252 |
| L2 hit rate | 86.8% | 83.0% | 83.1% | 80.2% | 80.2% |
| DRAM throughput | 23.5% | 19.0% | 19.9% | 21.1% | 22.1% |
| shared bank conflicts | 0 | 0.79 M | 6.9 M | 1.46 M | 13.3 M |

(The "v2" and "v3" columns are the shape without the mask.) The barrier stall is gone and
the pipe gains 4.8 points causal, 4.2 without the mask; 6.5% fewer instructions, the
`cp.async` address arithmetic of the Q/dO loads that TMA now does. What the mbarriers cost
shows as `branch_resolving` and `long_scoreboard`: the `try_wait` loop. Source-level sampling
on the causal shape puts 3.4% of all stall samples on the wait for a dS^T buffer's "empty"
barrier, a warp that got a full tile ahead of the slowest one, and 2.6% on the branch around
it; the other warp on its scheduler keeps issuing while it waits. The bank conflicts are the
same waits: 100 per block per query tile, and the deterministic build of the same kernel,
which has no dS^T barriers, counts 123 K for the whole launch. 72% of the stall samples sit
on `HMMA` instructions waiting for the pipe (`math_pipe_throttle`), which is where a
tensor-bound kernel should wait. The 9 to 10% of pipe left idle is the tiles where both warps
of a scheduler are in their exponentials or dS^T at once, the per-block prologue (K and V,
V's fragments, the barriers, 1,024 blocks on this shape) and the dS^T handoff. A third dS^T
buffer would give the warps more room to drift, but at `D = 128` it only fits with two Q/dO
stages instead of three.

#### Tried and dropped: finishing dQ in the kernel

The dQ conversion is a separate pass over a 64 MB fp32 buffer at 4096 tokens, 43 µs of a
1.7 ms backward, and the zeroing in the preprocess another 30 µs. Variant 3 has no block
barrier, so I tried letting the last warp to add into a (head, query tile) of dQ convert that
tile and re-zero it: a counter per tile, `WARPS` adds per key tile that sees it, a
`__threadfence()` between a warp's atomics and its count. It was correct and 20% slower (163
against 203 TFLOPS causal). `__threadfence()` compiles to `ERRBAR` plus an L1 invalidate
(`CCTL`), 4.7% of all stall samples, and the warp waited on the counter's returned value.
Moving each fence one tile after the adds it orders and reading each counter's result one
tile after the atomic brought it to 191. That version zeroes the buffer at the end of the
previous call, which loses the L2 effect of the preprocess section; the separate pass with
its zeroing moved the same way measured 194. So the in-kernel finish came out 1.5% behind in
equal conditions and 6% behind the shipped arrangement, and the pass stays.

## Deterministic dQ

The atomics make the last bits of dQ depend on the order blocks finish in. With
`deterministic=True` a separate kernel computes dQ with the query tile outer: the forward's
variant 2 structure, a block of 128 query rows with Q and dO fragments in registers, K and V
tiles of 64 keys through a three-stage `cp.async` pipeline, and per KV tile S = Q K^T, P,
dP = dO V^T (B = V as it lies), dS, and `dQ += dS K` with dS repacked from the accumulators
as the forward repacks P. The key-tile kernel then runs without its dQ product and without
the split (whose dK / dV slices are atomics too). Seven products instead of five, no atomics
anywhere; two runs agree bit for bit (the bench checks it on every shape).

Measured (`--deterministic=1`, medians of 30, TFLOPS by the five-product count so the
columns compare):

| shape | v2 ms / TFLOPS | v3 ms / TFLOPS | v3, atomics | deterministic / atomics |
|---|---|---|---|---|
| b1 h32 s4096 d128 | 5.040 / 136.4 | 4.832 / 142.2 | 3.070 / 223.8 | 1.57x |
| b1 h32 s4096 d128 causal | 2.292 / 149.9 | 2.260 / 152.0 | 1.643 / 209.1 | 1.38x |
| b1 h32 s8192 d128 causal | 8.847 / 155.3 | 8.701 / 158.0 | 6.369 / 215.8 | 1.37x |
| b1 hq32 hkv8 s4096 d128 causal | 2.628 / 130.7 | 2.596 / 132.4 | 1.698 / 202.3 | 1.53x |
| b1 h32 s4096 d64 causal | 1.159 / 148.3 | 1.114 / 154.3 | 0.792 / 216.8 | 1.41x |

The causal shapes cost 1.37 to 1.41x, the 7 / 5 of the extra products. The other two lose
the split as well: 1,024 equal tiles run as seven waves where the tail split made it 6.02,
and GQA's 256 tiles as two half-empty waves. Nsight on the causal 4096 shape: the dQ pass
1.11 ms at 87.6% of the tensor peak (248 registers, 96 KB of shared memory), the key-tile
kernel without its dQ product 1.47 ms at 85.3%, the preprocess 46 µs.

## Against torch

`scripts/bench_torch.py --only attention_bwd`: torch's backward is `torch.autograd.grad` of
the SDPA output (the same kernels `loss.backward()` launches: FlashAttention-2's dQ / dK / dV
kernel with its own `dot(dO, O)` preprocess and dQ conversion, and for GQA the sum of its
per-query-head dK and dV), ours is `attention_bwd` from the saved O and L, both timed by CUDA
events around the Python call, medians of 50. The v3 columns are the C++ bench's medians
from the sweep below; the ratio is the script's own, both sides in the same loop, where our
Python call adds 5 to 20 µs. The cuDNN column forces cuDNN's SDPA with
`--sdpa-backend cudnn`, which on this card is slower than flash on every backward shape.

| shape | v3 ms | v3 TFLOPS | flash ms | flash TFLOPS | v3 / flash | cuDNN ms | v3 / cuDNN |
|---|---|---|---|---|---|---|---|
| b1 h32 s1024 d128 | 0.266 | 161.4 | 0.311 | 138.0 | 1.13x | 0.346 | 1.26x |
| b1 h32 s1024 d128 causal | 0.139 | 154.3 | 0.207 | 103.7 | 1.41x | 0.229 | 1.57x |
| b1 h32 s2048 d128 | 0.781 | 220.0 | 0.985 | 174.4 | 1.25x | 1.095 | 1.38x |
| b1 h32 s2048 d128 causal | 0.468 | 183.4 | 0.573 | 149.9 | 1.19x | 0.640 | 1.34x |
| b1 h32 s4096 d128 | 3.070 | 223.8 | 3.512 | 195.7 | 1.11x | 3.879 | 1.24x |
| b1 h32 s4096 d128 causal | 1.643 | 209.1 | 1.936 | 177.4 | 1.14x | 2.110 | 1.25x |
| b1 h32 s8192 d128 | 12.605 | 218.1 | 13.034 | 210.9 | 1.02x | 14.194 | 1.11x |
| b1 h32 s8192 d128 causal | 6.369 | 215.8 | 6.794 | 202.3 | 1.04x | 7.253 | 1.12x |
| b4 h32 s2048 d128 causal | 2.170 | 158.3 | 2.211 | 155.4 | 1.02x | 2.360 | 1.08x |
| b1 h32 s4096 d64 causal | 0.792 | 216.8 | 1.037 | 165.6 | 1.27x | 1.206 | 1.50x |
| b1 hq32 hkv8 s4096 d128 | 3.109 | 221.0 | 3.533 | 194.5 | 1.11x | 3.908 | 1.23x |
| b1 hq32 hkv8 s4096 d128 causal | 1.698 | 202.3 | 1.967 | 174.7 | 1.12x | 2.160 | 1.24x |
| b1 hq32 hkv8 s8192 d128 causal | 6.397 | 214.9 | 6.889 | 199.5 | 1.05x | 7.353 | 1.13x |

Ahead on every shape, by 2 to 5% at 8192 tokens and on the batch of four, and by 11 to 41%
below that. The torch profiler's kernel names say what it runs on the causal 4096 shape:
`flash_bwd_dq_dk_dv_loop_seqk_parallel_kernel<Flash_bwd_kernel_traits<128, 64, 64, 8, 4, 2,
2, true, false>>`, 64-key by 64-row tiles with 8 warps and V in registers too, one block per
(key tile, head), 1.615 ms; before it two elementwise copies of a Q-sized tensor (47 and 49
µs) and its `dot(dO, O)` kernel (57 µs), after it the dQ conversion (43 µs), and under GQA two
more reductions (27 and 28 µs) that sum its per-query-head dK and dV. So on that shape about
half of the lead is the main kernel (ours about 1.55 ms: the 1.643 ms total less about 90 µs
of preprocess and conversion, against 1.615) and half the 194 µs torch spends around it. At 1K and 2K tokens the split is the rest:
torch's grid has no answer to the partial wave, the way our variant 2 had none before the
split rule. The batch of four is the one shape where neither side moves: its dQ accumulator
is 128 MB, past the 96 MB L2, and our key-tile kernel runs the DRAM at 79% of peak with 69%
of its atomic sectors missing L2 (the 8192-token shape, with the same 128 MB buffer but its
blocks in flight spread over 32 heads instead of 128, misses 18%).

## Results (RTX 5090, sm_120, CUDA 13.2, driver 595.58)

From the default `bench_attention_bwd` sweep, run 2026-09-29 (medians of 30, `ms / TFLOPS`,
TFLOPS by the five-product count halved under the mask). Bold marks the fastest rung per row.

| shape | v0 | v1 | v2 | v3 |
|---|---|---|---|---|
| b1 h4 s512 d128 | 0.2518 / 5.3 | 0.0511 / 26.2 | **0.0296 / 45.3** | 0.0296 / 45.3 |
| b1 h4 s512 d128 causal | 0.2434 / 2.8 | 0.0512 / 13.1 | 0.0317 / 21.2 | **0.0297 / 22.6** |
| b1 h4 s512 d64 | 0.2109 / 3.2 | 0.0286 / 23.4 | 0.0194 / 34.5 | **0.0193 / 34.7** |
| b1 h4 s512 d64 causal | 0.2046 / 1.6 | 0.0286 / 11.7 | **0.0256 / 13.1** | 0.0256 / 13.1 |
| b1 h4 s200 d128 causal | 0.0961 / 1.1 | 0.0286 / 3.6 | 0.0236 / 4.3 | **0.0235 / 4.4** |
| b1 h3 sq150 skv333 d64 | 0.0963 / 1.0 | **0.0143 / 6.7** | 0.0153 / 6.3 | 0.0153 / 6.3 |
| b2 hq8 hkv2 s300 d128 causal | 0.3910 / 2.4 | 0.1126 / 8.2 | **0.0336 / 27.4** | 0.0337 / 27.4 |
| b1 hq8 hkv2 s512 d64 | 0.5569 / 2.4 | 0.0901 / 14.9 | 0.0235 / 57.1 | **0.0235 / 57.2** |
| b1 h32 s1024 d128 | 4.1984 / 10.2 | 0.3376 / 127.2 | 0.2825 / 152.0 | **0.2662 / 161.4** |
| b1 h32 s1024 d128 causal | 2.1881 / 9.8 | 0.1718 / 125.0 | 0.1452 / 147.8 | **0.1391 / 154.3** |
| b1 h32 s2048 d128 | 17.1022 / 10.0 | 1.1361 / 151.2 | 0.8300 / 207.0 | **0.7808 / 220.0** |
| b1 h32 s2048 d128 causal | 8.6436 / 9.9 | 0.5577 / 154.0 | 0.4975 / 172.7 | **0.4685 / 183.4** |
| b1 h32 s4096 d128 | 72.7596 / 9.4 | 4.3851 / 156.7 | 3.2341 / 212.5 | **3.0702 / 223.8** |
| b1 h32 s4096 d128 causal | 33.7732 / 10.2 | 2.1265 / 161.6 | 1.7179 / 200.0 | **1.6432 / 209.1** |
| b1 h32 s8192 d128 | 284.5309 / 9.7 | 17.0455 / 161.3 | 12.8969 / 213.1 | **12.6048 / 218.1** |
| b1 h32 s8192 d128 causal | 141.4181 / 9.7 | 8.4761 / 162.1 | 6.5832 / 208.8 | **6.3693 / 215.8** |
| b4 h32 s2048 d128 causal | 33.8298 / 10.2 | 3.8519 / 89.2 | 2.2084 / 155.6 | **2.1702 / 158.3** |
| b1 h32 s4096 d64 causal | 29.4097 / 5.8 | 0.8290 / 207.2 | 0.8432 / 203.7 | **0.7924 / 216.8** |
| b1 hq32 hkv8 s4096 d128 | 75.9389 / 9.0 | 5.2298 / 131.4 | 3.2795 / 209.5 | **3.1092 / 221.0** |
| b1 hq32 hkv8 s4096 d128 causal | 34.8790 / 9.9 | 2.2204 / 154.7 | 1.7740 / 193.7 | **1.6984 / 202.3** |
| b1 hq32 hkv8 s8192 d128 causal | 147.0304 / 9.3 | 8.0347 / 171.1 | 6.6644 / 206.2 | **6.3969 / 214.9** |

Against the 258.7 TFLOPS the card's `mma.sync` peaks at (bench_peak, 2.98 GHz), variant 3
reaches 86.5% at 4096 tokens without the mask and 81 to 83% with it at 4096 and 8192, next to
the forward's 92%. The backward's non-`mma` work per tile is larger (two exponent-and-mask passes' worth of
elementwise work, a dS^T round trip through shared memory, atomics), and every number
includes the preprocess and the dQ conversion, about 90 µs of the 1.64 ms causal 4096 run.
Variant 1 on the batch of four is the outlier in its column (89 TFLOPS): its 64-key tiles
make twice the dQ atomics per FLOP of the 128-key rungs, into the 128 MB buffer that does not
fit L2. At `D = 64` variant 2 is no faster than variant 1 (203.7 against 207.2 TFLOPS causal)
and only variant 3 moves ahead; I have not profiled why. The small shapes are launch-bound
(three or four kernels, 20 to 30 µs).

## What the forward pays for L

The forward with and without the `lse` output, variant 5 through the extension, medians of
50: 1.158 against 1.159 ms at 4096 tokens without the mask, 2.349 against 2.366 ms at 8192
causal (0.7%). The causal 4096 pair came out 0.631 against 0.595 ms, the first measurement of
the process, with the output the faster one, so read it as noise: the cost is under 1%, the
`MUFU.LG2`, two multiplies and one 4-byte store per row in the epilogue, 16 rows per warp
against 33 KV tiles of `mma` work.

## Correctness

`bench_attention_bwd` checks dQ, dK and dV against a CPU double-precision reference built from
the same bf16 inputs, the GPU's O and the GPU's L: every row of the small shapes, 32 sampled
rows of each gradient on the large ones (the first and last rows of the first head always
among them). L itself is checked against the reference log-sum-exp on the checked query rows
(error at most 1e-3). Tolerance per tensor: `max|x − x_ref| <= 0.02 · max|x_ref| + 1e-3`, the
form of the forward's check; the kernels round P and dS to bf16 before their products and
round the outputs to bf16. Variants 1 to 3 are also run in deterministic mode on every shape,
checked the same way, and run twice to check the two results are identical. The pytest file
`tests/test_attention_bwd.py` compares every variant with torch autograd through
`F.scaled_dot_product_attention` in fp32 on the same bf16 inputs (both masks, both head
sizes, lengths 128 to 1000 that are not tile multiples, `S_q` shorter and longer than `S_kv`,
GQA groups of 4 and 8), the forward's L against `torch.logsumexp` for every forward rung
including the decode shapes, the autograd Function end to end, GQA against the same inputs
with K and V repeated, and repeated calls on two streams.

## What remains

- **The dQ round trip.** The preprocess (46 µs at 4096 tokens, mostly the zeroing) and the
  conversion (43 µs) are 5 to 6% of a causal 4096 backward. Finishing dQ inside the kernel
  lost to `__threadfence()`; what has not been tried is a release-ordered counter
  (`red.release.gpu`) or a persistent grid where a block that finished a key tile can finish
  the dQ tiles whose last contributor it was without a fence per tile.
- **Deterministic with the split.** The deterministic mode turns the split off because the
  slices add. Slices that each write their own fp32 partial, summed in a fixed order by the
  conversion kernel, would keep it bitwise reproducible and bring the non-causal and GQA rows
  from 1.5x back to about 1.4x of the atomic version.
- **The batch of four.** A 128 MB dQ accumulator does not fit the 96 MB L2. Heaviest-first
  across all heads is still the best order measured; splitting the launch per batch element,
  or accumulating dQ for a group of heads at a time, has not been tried.
- **A persistent grid**, the forward's variant 5: 1,024 blocks each load their K and V tile,
  read V's fragments and initialize their barriers before the first `mma`.
- **Scope.** `D` in {64, 128}, no dropout, no attention bias, no sliding window, and sm_120
  only as measured (the GB10 build compiles the same source and is untested).
