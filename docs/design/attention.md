# Attention: fused scaled-dot-product forward

`O = softmax(Q K^T / sqrt(D)) V` per (batch, head), for `Q, O = [B, H_q, S_q, D]` and
`K, V = [B, H_kv, S_kv, D]`, row-major with `D` contiguous. bf16 in and out, fp32 for every
score, exponential and accumulator. `H_q % H_kv == 0`: query head `h` reads K/V head
`h / (H_q / H_kv)`, grouped-query attention, with `H_kv == H_q` the plain multi-head case.
`D` in {64, 128}, optional causal mask, no dropout, no bias. This note is the forward; the
backward pass has its own, [attention_bwd.md](attention_bwd.md). Source:
`src/kernels/attention.cu` (the ladder) and `src/kernels/attention_decode.cu` (the
flash-decoding kernel variant 3 runs on decode shapes); the e4m3 forward, `attention_fp8`,
has its own section ("FP8") and source `src/kernels/attention_fp8.cu`. Bench: `bench_attention` (validates
every variant against a CPU double-precision reference). The library comparison is
`torch.nn.functional.scaled_dot_product_attention` in `scripts/bench_torch.py`, which on this
card and this torch picks the FlashAttention-2 kernel for every shape below.

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

Bytes, the traffic floor: `2 · (Q + O + K + V)` bf16 elements, with K and V counted once per
K/V head,

```
bytes = 2 · 2 · B · (H_q · S_q + H_kv · S_kv) · D
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
| 3 | variant 2's kernel with a schedule: the tiles of the last partial wave are split along the keys over the idle SMs and merged by a combine kernel; `S_q <= 64` runs a 64-row tile; decode shapes (the query rows sharing a K/V head fit 16 rows) run the flash-decoding kernel of the "Long-context decode" section | wave quantization on 170 SMs, a short query that no longer pays for 128 rows, and a decode step that streams each K/V head once for its whole group of query heads |
| 4 | variant 3's tile with K and V fed by TMA into full / empty mbarrier stages issued by one lane, no block-wide barrier in the KV loop | the per-tile `__syncthreads` that kept both warps of a scheduler in the same phase, so their softmaxes were a hole in the tensor pipe |
| 5 | variant 4's tile and pipeline on a persistent grid: 170 resident blocks take (b, h, q-tile) items from a queue in the heaviest-first order, a producer warp publishes each item to the consumer warps through a shared-memory ring and keeps the TMA loads running across items; the split-KV tail stays on for the non-causal shapes | the per-block prologue (barrier init, the Q box and the first K/V box before any `mma`), 1,028 times on the causal 4096 shape, and the per-SM spread a static assignment of six blocks per SM leaves |

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
torch's split-KV flash kernel under the same rotation. That was the decode path until the
flash-decoding kernel below replaced it for every shape whose rows fit its 16-row tile; the
64-row tile still takes `S_q <= 64` queries that do not (`S_q = 32` in MHA, say).

## The log-sum-exp output

Training needs one more number per query row: the backward recomputes P from Q, K and the
row's log-sum-exp `L_i = log Σ_j exp(q_i · k_j / sqrt(D))` instead of storing the `S_q x S_kv`
probabilities. Every variant already holds it at the end of the row: `m` is the row max in the
`log2(e) / sqrt(D)` domain and `l` the row sum of `2^(s − m)`, so `L = (m + log2 l) · ln 2`,
one `MUFU.LG2` and two multiplies per row in the epilogue. `attention_bf16` takes an optional
`float* lse` (`[B, H_q, S_q]`, natural log, torch's `logsumexp` convention) and every rung
writes it when it is not null: variants 0 and 1 from their row state, variants 2, 4 and 5 from
the lanes with `c = 0` of each row, and the split-KV combine kernel from its merged `(M, L)`.
The flash-decoding kernel has no such output, so a call that asks for one on a decode shape
runs variant 3's 64-row tile instead. With the pointer null nothing changes: the branch is on
a kernel argument, taken once per row in the epilogue. Measured cost of asking for it is in
the backward note.

## GQA

Grouped-query attention shares one K/V head between `group = H_q / H_kv` query heads:
Llama-3-8B is 32 query heads over 8 K/V heads, Llama-3-70B 64 over 8, Qwen2-7B 28 over 4.
The math per query head is unchanged; what changes is which K/V head it reads and, in decode,
how many times the cache is read.

**Pointer arithmetic.** The kernels index Q and O by the flattened `bh = b · H_q + h` and K
and V by

```
kv_index(bh) = (bh / H_q) · H_kv + (bh mod H_q) / group
```

(`kv_index` in `attention_internal.cuh`). In variants 0 to 3 that is the only change: the
block for query head `h` sets its K/V base pointer to head `h / group` and runs as before, so
the four variants take GQA with the same code and give the same bits as before on MHA inputs
(except the decode shapes variant 3 now hands to the flash-decoding kernel). In prefill the `group` blocks that share a K/V head are adjacent in the tile order (tiles are
numbered `q_rank · B · H_q + bh`), run in the same wave and pull the head through L2 once
from DRAM; prefill is compute-bound, so the extra L2 traffic costs nothing measurable:

| shape (v3, RTX 5090) | ms | TFLOPS |
|---|---|---|
| b1 h32 s4096 d128 causal | 0.649 | 211.8 |
| b1 hq32 hkv8 s4096 d128 causal | 0.646 | 212.8 |
| b1 h32 s4096 d128 | 1.241 | 221.6 |
| b1 hq32 hkv8 s4096 d128 | 1.236 | 222.5 |
| b1 hq28 hkv4 s4096 d128 causal | 0.579 | 207.6 |

(The 28-head shape has 28 x 32 = 896 tiles, 5.27 waves of 170: a tail of 46 tiles split 3
ways, where 1,024 tiles are 6.02 waves with a 4-tile tail split 42 ways. The longer tail is
the 2%.)

**Head grouping in decode.** A decode step reads the whole cache for a few query rows, and
with the per-head layout above each of the `group` query heads would stream the same K/V
head: 4x the DRAM traffic at 32/8 if L2 does not catch the sharing, and it only partly does
(the 64-row path measures 772 GB/s of floor traffic at 128K tokens, below). The decode kernel
of the next section instead puts the `group` heads of one K/V head into the rows of one
16-row Q tile, rows `hg · S_q + t` for head `hg` of the group and token `t`. Because the
query heads of a group are consecutive in Q, those rows are one contiguous
`[group · S_q, D]` slab starting at `Q + (b · H_q + kv · group) · S_q · D`, and so are the
output rows. K and V are read once per K/V head, and a single MHA query (one live row) pays
for 16 rows instead of 64.

**Traffic.** The bench and the results scripts count the floor as Q and O once per query
head and K and V once per K/V head, `2 · 2 · B · (H_q · S_q + H_kv · S_kv) · D` bytes, so a
GB/s figure under GQA says how close the kernel gets to reading each K/V head once for the
whole group. FLOPs follow the query heads. The shape string spells the heads apart only when
they differ (`b1_hq32_hkv8_sq1_skv4096_d128`), so the MHA rows keep their old keys.

**Torch.** `F.scaled_dot_product_attention(..., enable_gqa=True)` (torch 2.5+) takes the
same `[B, H_kv, S_kv, D]` K and V; its dispatcher picked the FlashAttention-2 kernel for
every GQA shape here, prefill and decode, as it does for MHA.

## Long-context decode

Source: `src/kernels/attention_decode.cu`, reached from variant 3 whenever
`group · S_q <= 16` (`decode_fits`). The job at `S_q = 1` against 4K to 128K keys is
streaming K and V once per K/V head at the DRAM roof: at 128K tokens, 8 K/V heads and
`D = 128` that is 512 MB per step, 350 µs at 1,532 GB/s, against 34 MFLOP of tensor work per
head.

**The kernel.** One block per (b, kv head, key slice). The Q tile is 16 rows (the group's
heads x tokens, above); its A fragments are loaded once and stay in registers. The block's
keys are divided among its four warps in contiguous ranges of 16-key slabs (32 keys at
`D = 64`), and each warp runs its own three-stage `cp.async` pipeline in a private 24 KB of
shared memory: 8 KB of K plus V per slab, two slabs in flight while the third is consumed,
`__syncwarp` where variant 2 has `__syncthreads`, and no block barrier in the loop. Per slab
a warp does 8 `ldmatrix.x4` and 16 `mma.sync` for `S = Q K^T` (2 n8 key tiles x 8 k-steps),
the online softmax on 4 scores per lane, 8 `ldmatrix.x4.trans` and 16 `mma.sync` for
`O += P V`, with P repacked from the S accumulators exactly as in variant 2. 190 registers,
96 KB of shared memory, one block per SM.

**Why this shape.** Two numbers from the design docs set it. Bandwidth is bytes in flight
over latency: the hgemm decode kernel needs about 1.5 MB in flight across the card (64 CTAs x
24 KB), and this kernel has 64 KB per SM (4 warps x 2 slabs), 8 MB at 128 blocks, so the
DRAM queue is never the limit. And the tensor work must hide under the stream: an `mma.sync`
holds a scheduler's tensor pipe for 32 cycles, so a warp spends 32 x 32 = 1,024 cycles per
8 KB slab; at the roof an SM's share of DRAM is 9 GB/s, 3.2 bytes per cycle at 2.8 GHz, so
the four warps' slabs arrive every 2,560 cycles each while they are consumed in 1,024
across four schedulers: the tensor pipe is 16% busy (Nsight, below) and the softmax, at 4
scores per lane per slab, is not on the critical path. The 64-row tile of variant 3 has the
same 4 warps each doing 128 `mma` per 32 KB tile, 4,096 cycles per tile per SM against
10,240 cycles of DRAM time: 40% of the pipe, which shows up as the 89% of roof it reached and
as its 2x loss under GQA, where it also reads each K/V head four times.

**The split.** The keys of every (b, kv head) are cut into `split` slices so the grid has
work for the card at any cache length: `split = min(128 / (B · H_kv), nslab / 8)`, so about
128 blocks with at least two slabs per warp (the constants from the sweep below; forced with
`SPARK_ATTENTION_SPLIT=n` for the sweep). Each block merges its four warps' partials
(unnormalized O rows, `m`, `l`) in shared memory, then either normalizes and writes O
(`split = 1`) or writes the block partial, 16 x (D + 2) floats, to a workspace and
increments a per-head counter. The last block to arrive merges the `split` partials with the
online-softmax rule, `M = max m_i`, `L = Σ l_i · 2^(m_i − M)`, `O = Σ O_i · 2^(m_i − M) / L`,
writes O and resets the counter, so a step is one launch: no combine kernel and no memset
(the pattern of `hgemm_decode.cu`). The merge is three passes with independent loads in
each: every slice's `(m, l)` into shared memory, per-row `M`, `L` and slice weights, then the
O rows eight loads ahead of the FMAs; at `split = 32` one block reads 256 KB of partials from
L2 here, and the first version, which chained those reads per element four at a time, was
1 µs slower on the 28/4-head shape (30.5 to 29.6 µs).

**Split sweep** (GB/s of the K+V floor, `SPARK_ATTENTION_SPLIT`, `bench_attention` medians
of 30, K and V rotated past L2; the last column is the 64-row tile of variant 3 on the same
inputs, `SPARK_ATTENTION_DECODE=0`):

| shape | split 1 | 2 | 4 / 5 | 8 | 16 | 21 | 32 | 43 | 64 | 128 | 64-row tile |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b1 hq32 hkv8 sq1 skv4096 | 420 | 712 | 941 | 986 | 886 | 963 | 866 | 782 | 735 | 498 | 569 |
| b1 h32 sq1 skv4096 | 1,459 | 1,455 | 1,456 | 1,457 | 1,451 | 1,395 | 1,394 | 1,342 | 1,337 | 950 | 1,337 |
| b1 hq32 hkv8 sq1 skv16384 | 459 | 874 | 1,454 | 1,456 | 1,452 | 1,395 | 1,348 | 1,286 | 1,285 | 1,154 | 719 |
| b1 h32 sq1 skv16384 | 1,610 | 1,609 | 1,630 | 1,611 | 1,590 | 1,575 | 1,572 | 1,536 | 1,508 | 1,403 | 1,518 |
| b1 hq32 hkv8 sq1 skv65536 | | 927 | 1,575 | 1,591 | 1,592 | 1,572 | 1,571 | 1,434 | 1,499 | 1,404 | 763 |
| b1 h32 sq1 skv65536 | | 1,667 | 1,684 | 1,668 | 1,648 | 1,648 | 1,663 | 1,636 | 1,647 | 1,612 | 1,619 |
| b1 hq32 hkv8 sq1 skv131072 | | 939 | 1,626 | 1,633 | 1,645 | 1,644 | 1,629 | 1,556 | 1,570 | 1,539 | 772 |
| b1 h32 sq1 skv131072 | | 1,682 | 1,693 | 1,685 | 1,677 | 1,675 | 1,682 | 1,627 | 1,671 | 1,658 | 1,636 |
| b8 hq32 hkv8 sq1 skv4096 | 1,583 | 1,546 | 1,546 | 1,521 | 1,478 | 1,445 | 1,413 | 1,327 | 1,253 | 806 | 611 |

(The 4 / 5 column is split 4 for the shapes up to 16K and 5 beyond.) What it says: at 8 K/V
heads the kernel needs 32 or more blocks (split 4 and up) to pull the full rate, and 64 to
128 blocks (split 8 to 16) are the plateau; one block per SM (split 21, 168 blocks) is 1 to
4% slower and finer splits pay for the partials. The MHA rows are flat from 32 blocks on,
and the batch-of-eight row is fastest with no split at all, 64 blocks. The 64-row tile loses
half its bandwidth under GQA at every length and 2 to 3% on MHA past 16K; on the 4K shapes
the plateau is 40 to 63% of roof because a 16 MB (GQA) or 67 MB (MHA) stream is short next
to the fixed costs: about 2 µs of launch, a DRAM round trip before the first slab is
consumed (Q and the first two slabs are requested together), and the drain and merge at the
end, 5 to 6 µs in all that a longer cache amortizes.

**Measured** (`bench_attention --variant=3`, medians of 50, K and V rotated past L2 where a
copy fits it; torch is `F.scaled_dot_product_attention` with `enable_gqa` from
`scripts/bench_torch.py`, whose loop adds 2 to 4 µs of Python per launch to both sides; the
roof is the 1,532 GB/s of `cudaMemcpy`):

| shape | K+V MB | v3 µs | GB/s | % roof | torch µs | torch GB/s | ours / torch |
|---|---|---|---|---|---|---|---|
| b1 hq32 hkv8 sq1 skv4096 | 16.8 | 17.3 | 968 | 63% | 37.3 | 451 | 1.80x |
| b1 h32 sq1 skv4096 | 67.1 | 46.1 | 1,456 | 95% | 64.1 | 1,047 | 1.35x |
| b1 hq32 hkv8 sq1 skv16384 | 67.1 | 46.1 | 1,457 | 95% | 67.8 | 990 | 1.33x |
| b1 h32 sq1 skv16384 | 268 | 164.7 | 1,630 | 106% | 187.2 | 1,434 | 1.11x |
| b1 hq32 hkv8 sq1 skv65536 | 268 | 168.4 | 1,594 | 104% | 194.1 | 1,383 | 1.14x |
| b1 h32 sq1 skv65536 | 1,074 | 639.6 | 1,679 | 110% | 665.2 | 1,614 | 1.03x |
| b1 hq32 hkv8 sq1 skv131072 | 537 | 324.4 | 1,655 | 108% | 357.3 | 1,503 | 1.09x |
| b1 h32 sq1 skv131072 | 2,147 | 1,268 | 1,693 | 111% | 1,298 | 1,655 | 1.02x |
| b8 hq32 hkv8 sq1 skv4096 | 134 | 86.8 | 1,548 | 101% | 103.6 | 1,297 | 1.15x |
| b8 h32 sq1 skv4096 | 537 | 320.5 | 1,675 | 109% | | | |
| b1 hq64 hkv8 sq1 skv16384 | 67.1 | 48.0 | 1,399 | 91% | | | |
| b1 hq28 hkv4 sq1 skv16384 | 33.6 | 29.6 | 1,133 | 74% | | | |
| b1 hq32 hkv8 sq4 skv16384 | 67.1 | 50.1 | 1,340 | 87% | | | |
| b1 hq32 hkv8 sq8 skv16384 (64-row tile) | 67.1 | 93.3 | 721 | 47% | | | |

(The ratio is the Python harness's own, ours against torch in the same loop; the C++ bench
times in the v3 column are 2 to 4 µs lower than what that loop sees for us.)

Above 100% of the copy roof is real: a read-only stream does not pay the read/write
turnaround `cudaMemcpy` does (the hgemm decode note measured a bare read stream at 1,581 to
1,614 GB/s), and Nsight puts the DRAM at 92 to 95% of its 1,792 GB/s peak on the 128K
shapes, so the read roof on this card is about 1,700 GB/s and the 128K rows are at it. From
16K tokens up both head layouts are within 5% of that, GQA at 32/8 reads each K/V head once
(the L2 sector count below), and torch's split-KV flash kernel, which is also at the roof by
128K, is 1.3x behind at 4K to 16K where its fixed costs are larger than ours. The 64-row
shape in the last row (8 tokens x 4 heads = 32 rows per K/V head, over the 16-row tile) is
what the second m16 tile per warp in "What remains" would fix.

**Nsight Compute** on the decode kernel (`--metrics`, one launch after warmup, fixed 2.53 GHz;
`dram__bytes_read` is not exposed on this card, so the bytes are L2 read sectors x 32 B):

| shape | DRAM % of peak | L2 read | K+V floor | duration | tensor pipe | issue active | warps active | conflicts |
|---|---|---|---|---|---|---|---|---|
| b1 hq32 hkv8 sq1 skv131072 | 92.6% | 538 MB | 537 MB | 333 µs | 16.5% | 5.8% | 8.33% | 41 K |
| b1 h32 sq1 skv131072 | 94.6% | 2,148 MB | 2,147 MB | 1.29 ms | 16.8% | 5.8% | 8.33% | 41 K |
| b1 hq32 hkv8 sq1 skv4096 | 53.7% | 18.2 MB | 16.8 MB | 17.8 µs | 12.8% | 6.5% | 8.25% | 41 K |
| b1 h32 sq1 skv4096 | 83.9% | 67.7 MB | 67.1 MB | 45.4 µs | 15.5% | 5.9% | 8.30% | 41 K |

128 blocks of 4 warps, 190 registers, 98.3 KB of dynamic shared memory, occupancy limited to
one block per SM by the shared memory (8.33% warps active is 4 of 48). The L2 read bytes
equal the K+V floor to within the Q, partials and the 1.4 MB the 4K GQA shape spends on its
partials: each K/V head crosses L2 once for its four query heads. The bank conflicts are
constant per block and sit in the epilogue (the parked partials; the swizzle on them took
the count from 107 K to 41 K), 0.1 µs per block. The tensor pipe at 16% and issue slots at
6% say the kernel is waiting on DRAM and nothing else, which is what a decode kernel should
be doing.

### Variant 4: the mbarrier pipeline

The plan for this rung was FlashAttention-3's ping-pong: two groups of four warps taking turns
on the tensor pipe through named barriers, so that one group's softmax always runs under the
other group's `mma`. Building it needed the K/V pipeline to stop using `__syncthreads`, because
a block-wide barrier once per tile would put the two groups back in phase. So the first step
was the pipeline, the second the turns. The pipeline alone gave the whole gain, and every
version of the turns gave some of it back. Both halves are below, with the numbers.

**The pipeline.** K and V tiles arrive by TMA (`cp.async.bulk.tensor`, the same machinery as
`hgemm` v5) into three 32 KB stages, each with a "full" mbarrier the copy engine completes and
an "empty" mbarrier every warp arrives on after its `P V`. The Q tile comes the same way
through stage 0 before the loop, and the bookkeeping counts it as load 0 and KV tile `t` as
load `t + 1`, so load `u` sits in stage `u mod 3` and is that stage's `u / 3`-th use: the
producer waits on "empty" with parity `(u/3 - 1) & 1` before refilling, a warp waits on "full"
with parity `(u/3) & 1` before reading. The tensor maps are 3-D, `[B·H][S][D]` with a box of
`1 x rows x 64` (128 bytes, one swizzle span; 128 rows for Q, 64 for K and V), so a box that
hangs off the end of a head is zero-filled by the copy engine instead of reading the next head,
and a `D = 128` tile is two boxes side by side. The 128-byte swizzle puts logical chunk `c` of
row `r` at `c XOR (r mod 8)` within the row's 128 bytes, and `ldmatrix` addresses go through
one function, `box_off(row, chunk)`, that adds the box offset and the XOR.

The loads are issued by lane 0 of warp 0, not by a producer warp. A ninth warp would have cut
the register budget to 224 per thread (65,536 / 288), and there is nowhere in this kernel for
a load to wait: at the top of iteration `it` the stage being refilled held tile `it - 1`, whose
`P V` every warp finished before it could reach this tile's "full" wait, so the producer's
"empty" wait completes at once and the cost is four TMA instructions on one lane per tile. The
prefetch distance is two tiles, about 16,000 cycles at the rate the pipe consumes them; a 32 KB
tile from L2 needs a few hundred.

What the pipeline changes for the warps: no `cp.async` addresses to compute (the instruction
count for the 4096 causal shape drops from 175.6 M to 164.5 M), no `__syncthreads` per tile,
and, the part that matters, no moment when all eight warps are at the same point. In variant 3
the two warps of each scheduler leave the barrier together, issue their 64 `Q K^T` `mma`
together, and then sit in their softmaxes together while the pipe has nothing queued. Without
the barrier the two warps drift apart within a few tiles, and while one is in its softmax the
other is usually in one of its two products. Nsight (below) puts the pipe at 92.8% active on
the causal shape and 94.1% non-causal, from 87.4% and 90.1%.

**The turns.** With the pipeline in place I added the ping-pong as compile-time schedules
(`-DSPARK_ATTN_V4_EXPERIMENTS`, `SPARK_ATTN_V4_MODE`), and measured them on the 4096 shapes
with `--iters=20`. Registers and shared memory are `ptxas -v` and the dynamic allocation; all
of them run one block per SM. Times are from the same session as the variant 3 row, which was
4 TFLOPS under the morning's full run on a card the other agents were also heating.

| schedule | regs | smem | causal TFLOPS | non-causal TFLOPS |
|---|---|---|---|---|
| variant 3 (`cp.async`, `__syncthreads` per tile) | 248 | 96 KB | 211.8 | 221.5 |
| **TMA + mbarrier pipeline, no turns (variant 4)** | **204** | **99.3 KB** | **225.0** | **235.4** |
| ping-pong, two turns per tile (`Q K^T`, then `P V`), groups = warps 0-3 / 4-7 | 206 | 99.3 KB | 198.6 | 207.5 |
| ping-pong, one turn per tile (`Q K^T` of t+1 with `P V` of t, FlashAttention-3's order) | 216 | 99.3 KB | 206.7 | 216.1 |
| ping-pong, two turns, groups = even / odd warps | 206 | 99.3 KB | 114.0 | 118.7 |
| ping-pong, one turn, groups = even / odd warps | 216 | 99.3 KB | 117.4 | 122.2 |
| pipeline + a warp skips a causal tile whose keys all follow its rows | 212 | 99.3 KB | 225.1 | 235.4 |
| pipeline + the O rescale skipped on a warp-uniform "no max moved" vote | 190 | 99.3 KB | 220.9 | 231.5 |
| pipeline + both | 214 | 99.3 KB | 227.6 | not measured |

Two things the table says. The even/odd rows are the proof of where warps live: at half speed,
every turn had one group's four warps on two of the four schedulers and nothing on the other
two, which is only possible if warps `w` and `w + 4` share a scheduler. So the 0-3 / 4-7
grouping is the right one, and it still lost 8% to the pipeline without turns. The reason is
`mma.sync` itself. A turn hands the pipe of each scheduler to one warp, and one warp does not
keep it full: each `ldmatrix` it needs has a 30-cycle shared-memory latency, each `mma`
depends on the one before it on the same accumulator, and every such gap idles the pipe, where
a second warp in the same phase would have an `mma` ready. Nsight on the one-turn schedule
(same shape, same session) shows the pipe at 84.4% against the pipeline's 92.8%, with
`math_pipe_throttle` down from 9.91 to 4.07 cycles per issue, `barrier` up from nothing to
6.53, `short_scoreboard` doubled to 0.38 and 15.14 cycles between a warp's issues: the group
holding the turn was not blocked by a busy pipe, it was waiting on its own `ldmatrix` results,
and the other group was waiting on it. FlashAttention-3 gets its ping-pong gain on Hopper because `wgmma` reads its operands
from shared memory asynchronously and one warpgroup's instruction stream does saturate the
pipe; `mma.sync` on this card wants two warps issuing at once, and the schedule that lets them
drift gives it that.

The two micro-optimizations are neutral or worse. Skipping the fully masked half of the second
diagonal tile removes 1.5% of the causal tile products but does not move the time: the tile
is the block's last, the warp that skips it frees the pipe for the warp it shares a scheduler
with, and that warp alone cannot use it. The rescale skip trades 64 `FMUL` that were hidden
under the pipe for a `__any_sync` vote and a branch that are not. The row with both is a real
1% on one shape and a loss on `D = 64` (210.7 against 213.0), so neither shipped.

**Two blocks per SM** was the other route to overlapping softmax with `mma`, and the arithmetic
says no at `D = 128`: 128 registers per thread with O (64), Q (32) and S (32) alone at 128, and
49 KB of shared memory per block against a 32 KB Q tile plus at least two 16 KB K/V stages.
Moving Q to shared memory frees the 32 registers but costs 32 KB of the 49. `D = 64` would fit
(Q 16 KB, two 16 KB stages, about 110 registers); it is the minor shape and was not tried.

**Nsight, 4096 causal, same session** (`--set full`, one launch, 2.51 to 2.53 GHz under the
profiler):

| | variant 3 | variant 4 |
|---|---|---|
| tensor pipe active | 87.4% | 92.8% (94.1% non-causal) |
| issue slots busy | 13.6% | 13.7% |
| cycles per issued instruction | 14.41 | 14.24 |
| of which `math_pipe_throttle` | 9.63 | 9.91 |
| `wait` | 2.04 | 2.25 |
| `not_selected` | 0.40 | 0.33 |
| `mio_throttle` | 0.31 | 0.03 |
| `long_scoreboard` | 0.28 | 0.20 |
| `short_scoreboard` | 0.19 | 0.19 |
| instructions executed | 175.6 M | 164.5 M |
| registers | 248 | 204 |
| dynamic shared memory | 96 KB | 99.3 KB |
| L2 hit rate | 90.5% | 88.7% |
| DRAM throughput | 8.4% | 9.0% |
| shared bank conflicts | 6 | 34,730 (34 per block, the barrier words) |

The stall mix barely moves: the kernel was and is throttled by the tensor pipe, and what
changed is how often the pipe had nothing queued. `mio_throttle` is the `cp.async` issue queue,
gone with the copies. The bank conflicts are the mbarrier arrivals and polls on the same eight
bytes, 34 per block, nothing in the tiles.

**What variant 4 is for.** `S_q <= 64` still runs variant 3's 64-row tile with the split,
which is bound by streaming the cache and gains nothing from the pipe. Everything else runs the
TMA kernel, including the tail split and the combine kernel, which are shared with variant 3
(`v2::split_tail_tiles`, `v2::launch_combine`). Every launch encodes three tensor maps on the
host, 27 ns each.

### Variant 5: the persistent grid

Variant 4 launches one block per Q tile: 1,028 of them on the causal 4096 shape, six per SM one
after the other, and each starts from nothing. It initializes its barriers, prefetches three
tensor maps, waits for its Q box and its first K/V box, and only then issues an `mma`; at the
other end its warps write their rows and exit, and the SM waits for the block scheduler to
put the next one in. Variant 5 launches 170 blocks that stay resident and walk a queue of
tiles, so the prologue is paid once per SM and the loads of one tile are in flight while the
warps finish the last one. The kernel is variant 4's tile, pipeline and epilogue with three
things added: a work queue, an item ring, and a producer warp.

**The queue.** The items are variant 4's tiles in variant 4's order, `tile = q_rank · B·H_q +
bh` with `q_rank` walking the Q tiles heaviest first under the causal mask (the last Q tile
of the sequence needs 64 KV tiles at 4096 tokens, the first needs 2), the `group` query heads
of a K/V head adjacent so they run together and pull the head through L2 once. A block takes
the next item with one `atomicAdd` on a counter in global memory, one atomic per 2 to 64 KV
tiles of work, and the last block to find the counter exhausted resets it for the next launch
(the `hgemm` v6 pattern, no memset). The counter is one per stream, handed out on first sight,
so a launch on a second stream cannot take items from one still running on the first. Greedy
heaviest-first on a queue is the LPT rule, and it evens the blocks out to within one small
item: simulated on the causal shapes, the longest block's share of KV tiles against the mean
is 200 against 198.8 at 4096 (1,024 items), 786 against 783.1 at 8192 (2,048 items) and 206
against 204.8 for the batch of four 2048-token sequences, so the causal tail is 0.4 to 0.6%
with no split at all. The split-KV tail of variant 3 stays as an option: the last `tiles mod
170` items become `split` slices each, merged by the combine kernel. It is on for the
non-causal shapes, where 1,024 equal tiles on 170 blocks are still 6.02 waves and four blocks
would otherwise run a seventh item (the table below), and for any shape with fewer tiles than
blocks, where the slices are the only work the idle SMs can get; it is off under the causal
mask once there is a full wave to balance.

**The item ring.** The producer publishes each item to the eight consumer warps through two
`int4` slots in shared memory with a full / empty mbarrier pair each, exactly like a pipeline
stage: it waits on the slot's "empty" (one arrive per warp, made once the warp has read the
slot), writes the item, arrives on "full"; a warp waits on "full" with the parity of the slot's
use count, reads, and arrives on "empty". Item `-1` ends the block. Both sides then run
variant 4's bookkeeping on one running load counter that never resets: load `u` is the Q
tile of an item or one of its KV tiles, lives in stage `u mod 3` and is that stage's
`u / 3`-th use, whichever item it belongs to. The producer issues the Q box of item `i + 1`
as soon as the stage that held the third-to-last tile of item `i` is released, so the Q wait
that opened every block of variant 4 is over before the warps get there, and item `i + 1`'s
first two K/V tiles land while they run item `i`'s epilogue. Per item a warp now pays a ring
read, one `mbar_wait` on a Q box that has already landed, eight `ldmatrix`, and the
epilogue stores.

**The producer warp.** Variant 4 issues its loads from lane 0 of warp 0 because a ninth warp
would cost registers. I built variant 5 both ways (`SPARK_ATTN_V5_PWARP`) and the ninth warp
won:

| producer | regs | causal TFLOPS | non-causal TFLOPS (tail split on) |
|---|---|---|---|
| lane 0 of warp 0, one load issued before each load consumed | 228 | 226.7 | 234.9 |
| **lane 0 of a ninth warp, runs ahead until the empty barriers stop it** | **168** | **230.4** | **239.4** |
| variant 4, same session | 204 | 224.7 | 235.7 |

(`--iters=30`, 4096 tokens, `D = 128`, before the dead-tile skip below.) The inline lane's
wait on "empty" is warp 0's wait: at the top of tile `t` it refills the stage of tile `t - 1`,
and whenever warp 0 is not the slowest warp it sits there until the slowest one arrives on
the barrier, then issues the load; variant 4 has the same wait, and the section above, which
called it free, was only right about the warps it does not delay. The producer warp
issues the moment a stage frees, whichever warp freed it, and warp 0 computes like the other
seven. The price is the register file: nine warps put three on one of the four schedulers,
whose 16,384 registers then allow 168 per thread (16,384 / 3 / 32, rounded down to a multiple
of 8; `ptxas` applies exactly this), so the kernel that needed 204 in variant 4 is compiled
to 168 with 40 bytes of spill, all of it in the producer's own path (the SASS of the K/V loop
has no `LDL` / `STL`). Executed instructions rise 8.5% (165.0 M to 179.1 M on the causal
shape), which is the producer's `try_wait` loop: one iteration per 63 cycles per SM, three
instructions each, 5% of one scheduler's issue slots, and the stall reasons it adds
(`branch_resolving`, `short_scoreboard`) are its own, one warp in nine of the average.

**The dead tile, again.** Variant 4 tried skipping the second diagonal tile for warps 0-3
(under the causal mask its 64 keys all follow their rows, every `p` is 0) and measured
nothing: the tile was the block's last, and a warp that skips it frees a pipe its scheduler
partner cannot fill alone. In the persistent kernel the warp that skips goes on to the next
item and issues that item's `Q K^T` while its partner finishes this one, so the skip is worth
having (`SPARK_ATTN_V5_SKIP=0` turns it off):

| shape | skip off | skip on |
|---|---|---|
| b1 h32 s4096 d128 causal | 228.9 | 231.3 |
| b1 h32 s8192 d128 causal | 232.9 | 234.1 |
| b4 h32 s2048 d128 causal | 212.2 | 216.4 |
| b1 h32 s4096 d64 causal | 226.2 | 227.2 |
| b1 h32 s4096 d128 (no mask, the skip never fires) | 237.9 | 237.8 |

(TFLOPS, `--iters=30`, a warmer card than the table above.) The batch of four gains most:
its Q tiles are 16 KV tiles long on average, so the skipped half-tile is a larger share.

**With and without the tail split** (TFLOPS, `--iters=30`, `SPARK_ATTN_V5_SPLIT`, the
producer-warp kernel; the default rule picks the bold column):

| shape | items | waves of 170 | split off | split on |
|---|---|---|---|---|
| b1 h32 s4096 d128 causal | 1,024 | 6.02 | **230.4** | 229.6 |
| b1 h32 s4096 d128 | 1,024 | 6.02 | 209.3 | **239.4** |
| b1 h4 s512 d128 causal | 16 | 0.09 | 9.8 (16 blocks) | **14.0** (128 slices) |

Under the causal mask the four tail items are the four lightest (two KV tiles each), and
splitting them in two buys nothing the queue had not already balanced, at the price of a
combine launch. Without the mask the blocks that draw a seventh item run 7 / 6.02 = 16%
longer than the rest, which is the 209 against 239.

**Nsight Compute, 4096 tokens, `D = 128`, one launch each, 2.52 GHz under the profiler**
(the variant 4 column is the same session, so it differs slightly from the table in the
variant 4 section; the variant 5 columns are the shipped kernel, producer warp, dead-tile
skip on, tail split by the default rule):

| | v4 causal | v5 causal | v4 non-causal | v5 non-causal |
|---|---|---|---|---|
| duration | 705 µs | 683 µs | 1.34 ms | 1.32 ms |
| grid | 1,028 blocks | 170 | 1,188 | 170 |
| tensor pipe active, mean over SMs | 94.2% | 96.1% (96.8% before the skip) | 95.5% | 97.5% |
| tensor pipe active, slowest SM | 92.9% | 94.7% (95.4%) | 95.1% | 97.1% |
| issue slots busy | 14.0% | 15.9% | 13.9% | 15.5% |
| instructions executed | 165.0 M | 180.6 M | 312.2 M | 340.0 M |
| cycles per issued instruction | 14.17 | 14.09 | 14.39 | 14.53 |
| of which `math_pipe_throttle` | 9.87 | 8.03 | 10.13 | 8.60 |
| `wait` | 2.24 | 2.31 | 2.28 | 2.29 |
| `branch_resolving` | 0.14 | 0.81 | 0.10 | 0.78 |
| `short_scoreboard` | 0.19 | 0.77 | 0.18 | 0.67 |
| `long_scoreboard` | 0.19 | 0.60 | 0.16 | 0.59 |
| registers | 204 | 168 | 204 | 168 |
| L2 hit rate | 88.6% | 88.6% | 93.7% | 93.5% |
| shared bank conflicts | 35 K | 290 K | 15 K | 918 K |

The pipe gains 2.6 points causal and 2.0 non-causal, and the slowest SM gains more than the
mean: the queue is also evening out the per-SM spread that a static assignment of six blocks
per SM left. The bank conflicts are the producer warp polling the barrier words while the
consumers arrive on them, 3 K per SM over 1.7 M cycles. The prologue share, measured rather
than estimated: at the same 2.52 GHz the causal launch went from 705 to 687 µs before the
dead-tile skip, 18 µs, which over 1,028 blocks on 170 SMs is 3.0 µs per block of prologue
plus drain that variant 4 paid and variant 5 does not, 2.6% of the run.

**What is left** in the 3% of idle pipe: the per-item boundary (about a thousand cycles of
ring read, `ldmatrix` and epilogue per item against 33 tiles of 8,192 cycles, 0.4%), the LPT
tail (0.6% on the causal shapes), and the tiles where both warps of a scheduler still meet in
their softmaxes, which is what the 96.8% is mostly made of and what a 128-key tile would
halve.

**Shapes.** Variant 5 routes exactly as variant 4 does: `S_q <= 64` to variant 3's paths
(the flash-decoding kernel, or the 64-row tile), every 128-row shape to the persistent kernel.
GQA is the same `kv_index` stride on the K/V head; rows past `S_q` and keys past `S_kv` are
zero-filled by the copy engine as before. The split workspace stays one per device, shared
with variants 3 and 4 (the same caveat about concurrent split launches on two streams); the
queue counters are per stream.

## FP8: `attention_fp8`

Variant 5 runs its tensor pipe 96 to 97% busy, so the bf16 kernel is done: 239 TFLOPS against
a 258.7 TFLOPS roof, and the roof is the bf16 `mma.sync` rate. The fp8 GEMM measured the
block-scaled e4m3 `mma.sync` at 1,014 TFLOPS on this card ([fp8gemm](fp8gemm.md)), 3.9 times
the bf16 instruction. `attention_fp8` is the same forward with Q, K and V in e4m3 and both
products on that instruction, FlashAttention-3's fp8 recipe on `mma.sync`.

`O = softmax(sq sk Q K^T / sqrt(D)) sv V`, where `Q = sq Q8` and so on: e4m3 inputs with fp32
descale factors on the device, one per tensor or one per (b, head) (`per_head`), bf16 out.
Heads, GQA, masks, `D` in {64, 128} and any `S_q`, `S_kv` are the bf16 op's. Source
`src/kernels/attention_fp8.cu`, built with the fp8 GEMM for `sm_120a`; bench
`bench_attention_fp8`; Python `sk.attention_fp8(q8, k8, v8, sq, sk, sv, causal)` with
`sk.quantize_fp8(x, per_head=False)` making the e4m3 tensor and its scale. It is its own op with
its own ladder rather than variants 6 and up of `attention`, because the inputs are a
different type with a different contract (scales, an accuracy bound), the way `fp8gemm` sits
next to `hgemm`.

| variant | idea | what it fixes |
|---|---|---|
| 0 | one warp per query row, every element dequantized to fp32, P in fp32 | baseline: the arithmetic of the quantized inputs, with no rounding of P |
| 1 | variant 5's persistent TMA kernel on e4m3: both products on `mma.sync.m16n8k32` (block-scaled, unit scales), P rounded to e4m3 in registers, V transposed by `ldmatrix.trans` and byte permutes, the O rescale skipped while no row max moves | the bf16 `mma` roof |

### The three problems

**V is the wrong way round.** The fp8 B fragment of `m16n8k32` holds four consecutive k of
one column n per register. For `P V`, k is the key and n is d, so a register wants four keys
of one d column, and V in memory is `[key][d]`: four keys of one column are four rows apart.
The bf16 kernel had the same problem and `ldmatrix.trans` solved it, but `ldmatrix` is a
16-bit instruction: its transpose moves pairs of bytes, so it delivers two keys of two
adjacent d columns, `{key 2c, 2c+1} x {d 2g, 2g+1}` in lane `(g, c)`. The fix is two byte
permutes per register pair. Lane l addresses key `32kt + l` at 16-byte chunk j, so the four
matrices of one `ldmatrix.x4.trans` are keys +0..7, +8..15, +16..23 and +24..31;
`prmt(r0, r1, 0x6420)` takes bytes 0 and 2 of the first two (keys 2c, 2c+1, 8+2c, 9+2c at
`d = 16j + 2g`) and `0x7531` takes bytes 1 and 3 (the same keys at `d = 16j + 2g + 1`). One
`ldmatrix` and four `PRMT` give the B fragments of two n8 tiles of O, one holding the even d of
the chunk and one the odd d. The keys inside a register come out in the order
`{2c, 2c+1, 8+2c, 9+2c}`, not `4c .. 4c+3`, which is the next problem's answer.

**The S accumulator is not the A fragment.** In bf16, the accumulator of `S = Q K^T` was
exactly the A fragment of `P V` (the section on variant 2). In fp8 the A fragment wants four
consecutive k per register and lane `(g, c)` holds keys `2c, 2c+1` of each n8 tile of S: the
layouts disagree, and the usual fix is a round of shuffles. But the sum over keys does not
care about their order, only that P and V agree on it. Lane `(g, c)` holds keys
`{2c, 2c+1}` of n8 tile 2i and `{8+2c, 9+2c}` of tile 2i+1, which is exactly the key order
the permuted V registers carry. So P is two `cvt.rn.satfinite.e4m3x2.f32` per register, in
that order, and it never leaves the lane. The transpose V needs and the layout S already has
are the same permutation. Two further consequences: O's n8 tiles 2j and 2j+1 hold
`d = 16j + 4c + {0, 2}` and `{1, 3}`, four adjacent columns per row, so the epilogue writes 8
bytes per row per chunk; and K is `[key][d]` with d contiguous, the "Bt" layout, so its
fragments load with a plain `ldmatrix` as the fp8 GEMM's do.

**P's dynamic range.** e4m3's normal range starts at 2^-6. Rounded as they are, every
probability under 1/64 of the row max would keep fewer than three mantissa bits, down to
2^-9 where it becomes 0. The kernel computes `p = 2^(s - m + 8)` instead, in (0, 256], which
costs nothing (the 8 goes into the bias of the `ex2`'s FFMA) and moves the subnormal edge to
2^-14 of the row max. The row sum carries the same 2^8 and the final division cancels it.
The accuracy section says how much it buys: less than I expected.

### Two cheap fixes

The first profile of the kernel (4096 non-causal, one launch at the 2.3 GHz Nsight locks the
clock to) counted 548 instructions per warp per 64-key tile, 64 of them `QMMA`. Two fixes
came from the SASS:

- **32 MOVs.** The fp8 GEMM's `ldmatrix` addressing (lanes 0-15 on rows 0-15 at the first 16
  bytes, lanes 16-31 at the second) returns the two registers of an n8 tile as r0, r2 and the
  next tile's as r1, r3, and `QMMA` wants its B operand in an aligned register pair, so ptxas
  moved them, 32 `MOV` a tile. Addressing lanes 8-15 at the second 16 bytes of rows 0-7 and
  lanes 16-31 at rows 8-15 returns them as r0, r1 and r2, r3. 287 M to 268 M instructions.
- **The lazy rescale.** The O rescale is 64 `FMUL` per tile per warp, and once the running
  maxima settle it multiplies by 1. A row now keeps its max until a score beats it by more
  than 0.8 (log2 units): p then peaks at `2^8.8 = 446`, still inside e4m3's 448, and its alpha
  is exactly 1, and the multiplies run under a warp vote only when some row moved.
  FlashAttention-4 does the same with a larger threshold; here the P shift and e4m3's range
  set it. 268 M to 246 M instructions, 2.3% fewer cycles.

| `SPARK_ATTN_FP8_LAG` | instructions | SM cycles | tensor pipe |
|---|---|---|---|
| -1 (rescale every tile) | 268.4 M | 1,208 K | 67.5% |
| 0 (skip only when no max moved) | 253.2 M | 1,193 K | 68.4% |
| **0.8** | **246.3 M** | **1,181 K** | **69.0%** |

(Nsight, `--metrics`, 4096 tokens, 32 heads, no mask, one launch.)

### What limits it

Nsight at 4096 tokens, one launch each, both kernels at the clock Nsight locks:

| | bf16 v5, no mask | fp8 v1, no mask | fp8 v1, causal |
|---|---|---|---|
| SM cycles | 3,407 K | 1,173 K | 613 K |
| tensor pipe active | 96.7% | 69.0% | 67.3% |
| instructions | 349.1 M | 246.2 M | 130.3 M |
| issue slots busy | 15.7% | 32.1% | 32.7% |
| registers | 168 | 168 (8 bytes of stack) | 168 |
| cycles between a warp's issues | | 7.0 | |
| of which `wait` / `math_pipe_throttle` / `mio_throttle` / `branch_resolving` / `long_scoreboard` / `short_scoreboard` | | 2.09 / 1.45 / 0.50 / 0.52 / 0.45 / 0.36 | |
| XU (`MUFU`) / ALU / FMA pipes | | 18.6% / 12.4% / 7.0% | |
| L2 hit rate | | 91.8% | |

2.9 times fewer cycles than the bf16 kernel, where the instruction's rate says up to 3.9. The
arithmetic of where the rest goes: a warp's 64-key tile is 64 `QMMA` of 16 cycles each on its
scheduler's pipe, 1,024 cycles, where bf16's 128 `HMMA` of 32 cycles were 4,096. The softmax,
the conversions, the `ldmatrix`s and the permutes are the same 400-odd instructions per tile
in both kernels (the fp8 one: 32 `LDSM`, 64 `PRMT`, 34 `FFMA` and 34 `MUFU` for the
exponentials, 36 `FMNMX`, 36 `FADD`, 16 `F2FP` for P, the rescale when it runs), issued at one
every 7 cycles by a warp that is waiting on its own dependencies, about 2,800 cycles of
which only the mma part overlaps the other warp's. In bf16 that chain hid under 4,096 cycles
of the other warp's products; in fp8 it is longer than them. Two warps per scheduler at 2 x
1,024 pipe cycles out of about 2,900 is 70%, which is what Nsight shows.

So the work is in the softmax, and I tried four ways to hide it:

| change | SM cycles, no mask | causal | instructions | verdict |
|---|---|---|---|---|
| **variant 1 as shipped** | **1,173 K** | **613 K** | **246 M** | |
| row max and row sum as trees, not chains | 1,180 K | 616 K | 246 M | same |
| softmax per 32-key half tile, `Q K^T` of one half issued before the other half's softmax (FlashAttention-3's intra-warp overlap without a second S) | 1,229 K | 647 K | 282 M | spills 32 bytes, slower |
| the same half tiles, not pipelined | 1,205 K | 628 K | 256 M | the extra max and vote cost 2% |
| 12 compute warps, three per scheduler, 192-row tile, the producer on a lane of warp 0 (a thirteenth warp would cap registers at 128) | 1,425 K | 762 K | 274 M | slower |
| softmax turns between the two warps of a scheduler (named barriers, one softmax at a time per scheduler), against the same build without them | 1,257 K against 1,244 K | 657 K against 654 K | 263 M | slower |
| V rewritten into fragment order once per tile by the producer warp (next section) | 1,244 K | 643 K | 216 M | slower |

The half-tile pipelining is the idea that works for FlashAttention-3 on Hopper, where
`wgmma` is asynchronous. With `mma.sync` the overlap depends on ptxas interleaving the
softmax of one half into the `QMMA` stream of the other, and at 168 registers it spilled
instead. The twelve-warp tile gives each scheduler a third warp to fill the softmax gaps, but
the inline producer paces every warp to warp 0 (the ring cannot run ahead of it) and the Q
tile needs its own 24 KB buffer; that version measured 1,217 K cycles with eight warps too,
against 1,179 K for Q through a K/V stage, so a third of its loss is the buffer. Code
placement moves these numbers by several percent on its own: the same kernel with the loop
body split into three basic blocks measured 1,307 K. None of the rows above is within noise
of a gain, so variant 1 stayed as it is.

### V in fragment order: what a transposed V layout would buy

Every consumer warp transposes the same V tile: 8 warps x (16 `LDSM.T` + 64 `PRMT`) per tile,
a quarter of the non-mma instructions. With `-DSPARK_ATTN_FP8_EXPERIMENTS` and
`SPARK_ATTN_FP8_VT=1` the producer warp, whose other 31 lanes were idle, does it once: after
issuing load u it waits for load u - 1 to land, reads its V tile into registers with the same
`ldmatrix.trans` and permutes, and writes the fragments back in place, 512 bytes per
(k32 step, 16-byte chunk) with lane l's four registers at offset `16 l`; a per-stage `vready`
mbarrier tells the consumers, who then read their four registers with one 16-byte `LDS` and
no permutes. Instructions drop 12% (246 M to 216 M), and the kernel gets 6% slower: 1,244 K
cycles. Nsight puts `math_pipe_throttle` up from 1.45 to 2.27 and `wait` from 2.09 to 2.43:
without the permutes between them, each warp's `P V` products issue as one dense burst, and
the two warps of a scheduler collide on the pipe more often. The producer's rewrite also lands
on scheduler 0 only.

The same build with the rewrite removed (wrong output, timing only) is the best a V stored
transposed in memory could do, since the consumers then see exactly that layout: 1,208 K
cycles at Nsight's clock, still 3% more than variant 1, and 3% faster in the timed loop
(584 against 565 TFLOPS, 4096 no mask), where fewer instructions buy clock under the power
cap (below). A transpose pass over V per call would cost more than that: V is 16 MB at 32
heads x 4096 x 128 bytes, 32 MB of traffic, about 22 us against a 440 us kernel, 5%. So the
kernel reads V as it lies. A K/V cache that stores V in fragment order as it is appended is
the one layout that would gain, at most 3%.

### Power

`nvidia-smi` at 250 ms during 8,000 back-to-back launches of the 4096 no-mask shape:

| kernel | SM clock | power | throttle reason |
|---|---|---|---|
| fp8 v1 | 2,362 to 2,422 MHz | 567 to 577 W | software power cap |
| bf16 v5 | 2,872 MHz | 600 W | software power cap |

The fp8 kernel runs 17% below the bf16 one's clock. At 2.4 GHz the instruction's roof is
1,014 x 2.4 / 2.92 = 833 TFLOPS, and 69% of it is 575, which is what a long loop measures; the
bench's 625 to 635 at 4096 come from shorter bursts on a cooler card. The 16K shapes run
7.8 ms a launch and settle to the capped clock inside the timing loop, hence their 561 and
601. It is the fp8 GEMM's story (2.13 to 2.16 GHz at 600 W there): at this rate fewer
instructions per FLOP are also fewer joules per FLOP, which is why the transposed-V timing
above gained where the fixed-clock one lost.

### Measured

`bench_attention_fp8`, medians of 50, uniform inputs quantized per tensor (the bf16 bench's:
Q, K in [-2, 2], V in [-1, 1]). The bf16 column is attention variant 5 on the unquantized
inputs rounded to bf16, timed in the same process right before; "% of peak" is against the
1,014 TFLOPS block-scaled roof.

| shape | bf16 v5 ms / TFLOPS | fp8 v1 ms / TFLOPS | fp8 / bf16 | % of fp8 peak |
|---|---|---|---|---|
| b1 h32 s1024 d128 | 0.0992 / 173.2 | 0.0418 / 411.4 | 2.37x | 41% |
| b1 h32 s1024 d128 causal | 0.0520 / 165.2 | 0.0213 / 403.1 | 2.44x | 40% |
| b1 h32 s2048 d128 | 0.2927 / 234.8 | 0.1321 / 520.1 | 2.22x | 51% |
| b1 h32 s2048 d128 causal | 0.1646 / 208.8 | 0.0626 / 549.2 | 2.63x | 54% |
| b1 h32 s4096 d128 | 1.1453 / 240.0 | 0.4328 / 635.2 | 2.65x | 63% |
| b1 h32 s4096 d128 causal | 0.5819 / 236.2 | 0.2199 / 625.1 | 2.65x | 62% |
| b1 h32 s8192 d128 | 4.5460 / 241.9 | 1.7441 / 630.4 | 2.61x | 62% |
| b1 h32 s8192 d128 causal | 2.2993 / 239.1 | 0.8648 / 635.7 | 2.66x | 63% |
| b1 h32 s16384 d128 | 18.438 / 238.5 | 7.8379 / 561.1 | 2.35x | 55% |
| b1 h32 s16384 d128 causal | 9.2820 / 236.9 | 3.6606 / 600.7 | 2.54x | 59% |
| b1 hq32 hkv8 s4096 d128 | 1.1495 / 239.1 | 0.4351 / 631.7 | 2.64x | 62% |
| b1 hq32 hkv8 s4096 d128 causal | 0.5818 / 236.2 | 0.2218 / 619.7 | 2.62x | 61% |
| b4 h32 s2048 d128 causal | 0.6029 / 228.0 | 0.2422 / 567.4 | 2.49x | 56% |
| b1 h32 s4096 d64 causal | 0.2924 / 235.0 | 0.1342 / 512.2 | 2.18x | 51% |

Variant 0 runs at 7.5 to 12.6 TFLOPS on these shapes, like the bf16 variant 0. The short
sequences pay a larger share for the per-item boundary (one Q box, eight `ldmatrix`, the
epilogue) against 16 or 32 tiles of 1,024 pipe cycles; the batch of four has short causal
items for the same reason. A decode step (`S_q = 1`) runs the 128-row tile and is slower than
the bf16 op, which hands it to the flash-decoding kernel (8.9 against 7.2 us at 700 keys):
decode is bound by streaming the cache, where e4m3 halves the bytes, but that needs its own
kernel.

Against torch (`scripts/bench_torch.py --only attention_fp8`, `F.scaled_dot_product_attention`
in bf16 on the same shapes, 100 iterations; torch has no fp8 attention on this card, so this
is the kernel a model would otherwise call; the loop runs launches back to back, so the fp8
times are nearer the power-capped clock than the C++ bench's):

| shape | fp8 v1 ms | flash ms | fp8 / flash | cuDNN ms | fp8 / cuDNN |
|---|---|---|---|---|---|
| b1 h32 s1024 d128 | 0.0407 | 0.1183 | 2.91x | 0.1202 | 2.95x |
| b1 h32 s1024 d128 causal | 0.0255 | 0.0950 | 3.72x | 0.1022 | 4.07x |
| b1 h32 s2048 d128 | 0.1191 | 0.4308 | 3.62x | 0.4063 | 3.41x |
| b1 h32 s2048 d128 causal | 0.0659 | 0.2490 | 3.78x | 0.2506 | 3.80x |
| b1 h32 s4096 d128 | 0.4451 | 1.4647 | 3.29x | 1.3617 | 3.07x |
| b1 h32 s4096 d128 causal | 0.2303 | 0.7951 | 3.45x | 0.7833 | 3.41x |
| b1 h32 s8192 d128 | 1.8924 | 5.3578 | 2.83x | 5.0300 | 2.64x |
| b1 h32 s8192 d128 causal | 0.9097 | 2.7774 | 3.05x | 2.7156 | 2.96x |
| b1 h32 s16384 d128 | 8.1749 | 20.459 | 2.50x | 19.440 | 2.38x |
| b1 h32 s16384 d128 causal | 3.9326 | 10.379 | 2.64x | 10.118 | 2.56x |
| b1 hq32 hkv8 s4096 d128 | 0.4497 | 1.4630 | 3.25x | 1.3557 | 2.99x |
| b1 hq32 hkv8 s4096 d128 causal | 0.2297 | 0.7892 | 3.44x | 0.7769 | 3.38x |
| b4 h32 s2048 d128 causal | 0.2486 | 0.7644 | 3.07x | 0.7538 | 3.03x |
| b1 h32 s4096 d64 causal | 0.1396 | 0.4177 | 2.99x | 0.3956 | 2.83x |

(The cuDNN column's fp8 times are the same kernel in the second run of the script.)

### Accuracy

`scripts/attention_fp8_accuracy.py`: B = 1, 8 heads, 4096 tokens, D = 128, q, k, v drawn in
fp32 and quantized by `sk.quantize_fp8`; the reference is `F.scaled_dot_product_attention` in
fp64 on the unquantized values. Each cell is max / mean |O - O_ref|, absolute. "v0" is the
fp8 baseline, whose only error is the rounding of q, k and v; "no shift" is variant 1 with
P rounded as 2^(s - m) (`SPARK_ATTN_FP8_PSHIFT=0`).

| input | mask | bf16 kernel | fp8 v0 (P in fp32) | fp8 v1 | v1 per-head scales | v1 no shift | v1 against the bf16 kernel | max / mean \|O_ref\| |
|---|---|---|---|---|---|---|---|---|
| Gaussian | none | 8.4e-4 / 7.6e-5 | 1.5e-2 / 9.5e-4 | 1.6e-2 / 1.09e-3 | 1.4e-2 / 1.09e-3 | 1.7e-2 / 1.09e-3 | 1.6e-2 / 1.09e-3 | 0.18 / 0.020 |
| Gaussian | causal | 1.3e-2 / 1.4e-4 | 1.3e-1 / 1.8e-3 | 1.4e-1 / 2.0e-3 | 1.4e-1 / 2.0e-3 | 1.4e-1 / 2.0e-3 | 1.4e-1 / 2.0e-3 | 3.20 / 0.040 |
| Gaussian, q x 4 (peaked softmax) | none | 4.4e-2 / 2.0e-3 | 6.6e-1 / 3.1e-2 | 6.2e-1 / 3.1e-2 | 6.7e-1 / 3.1e-2 | 6.2e-1 / 3.1e-2 | 6.4e-1 / 3.1e-2 | 4.46 / 0.325 |
| Gaussian, q x 4 | causal | 4.9e-2 / 2.1e-3 | 7.1e-1 / 3.2e-2 | 6.8e-1 / 3.2e-2 | 8.2e-1 / 3.2e-2 | 6.8e-1 / 3.2e-2 | 7.0e-1 / 3.2e-2 | 4.58 / 0.371 |
| 0.1% of entries x 20 | none | 1.0e0 / 2.9e-3 | 1.8e1 / 4.4e-2 | 1.8e1 / 4.4e-2 | 1.8e1 / 4.4e-2 | 1.8e1 / 4.4e-2 | 1.8e1 / 4.4e-2 | 52.3 / 0.377 |
| 0.1% of entries x 20 | causal | 5.2e-1 / 2.2e-3 | 1.8e1 / 3.4e-2 | 1.8e1 / 3.4e-2 | 1.8e1 / 3.4e-2 | 1.8e1 / 3.4e-2 | 1.8e1 / 3.4e-2 | 52.3 / 0.324 |
| 4 key channels of 128 x 20 | none | 1.3e-1 / 2.5e-3 | 1.5e0 / 4.0e-2 | 1.5e0 / 4.0e-2 | 1.8e0 / 3.9e-2 | 1.5e0 / 4.0e-2 | 1.5e0 / 4.0e-2 | 3.93 / 0.250 |
| 4 key channels x 20 | causal | 1.1e-1 / 2.5e-3 | 1.5e0 / 4.1e-2 | 1.4e0 / 4.1e-2 | 1.7e0 / 4.0e-2 | 1.4e0 / 4.1e-2 | 1.5e0 / 4.1e-2 | 4.13 / 0.293 |
| 4 value channels x 20 | none | 1.3e-2 / 1.2e-4 | 1.5e-1 / 1.5e-3 | 1.8e-1 / 1.7e-3 | 1.8e-1 / 1.8e-3 | 1.7e-1 / 1.7e-3 | 1.7e-1 / 1.7e-3 | 2.49 / 0.033 |
| 4 value channels x 20 | causal | 2.0e-1 / 2.3e-4 | 1.7e0 / 2.9e-3 | 2.4e0 / 3.2e-3 | 2.2e0 / 3.2e-3 | 2.4e0 / 3.2e-3 | 2.3e0 / 3.2e-3 | 62.3 / 0.064 |

What it says:

- **The error is the inputs, not P.** Variant 0, which keeps P in fp32, is within 15% of
  variant 1's mean error on every row. e4m3 carries 3 mantissa bits, 2^-4 relative per
  element, against bf16's 7, and every row of the table sits at 10 to 16 times the bf16
  kernel's mean error. The last column shows the fp8 kernel against the bf16 kernel is the
  same number as against the fp64 reference: the bf16 error is noise next to it.
- **Scores amplify it.** A score is a sum of 128 products each off by up to 2^-4, and the
  softmax exponentiates the absolute score error, so anything that makes the scores large
  makes the output error large: a peaked softmax (q x 4) or a few large key channels move the
  mean error from 5% to 10 to 16% of mean |O|. That is the case FlashAttention-3's
  incoherent processing (a random Hadamard rotation of q and k before the rounding) is for;
  it is not implemented here.
- **The P shift is nearly free and nearly useless at this size.** Without the 2^8 the
  probabilities under 2^-6 of the row max go subnormal, and the max error on Gaussian inputs
  rises from 1.56e-2 to 1.71e-2; the mean does not move. Those probabilities carry little of a
  row's mass, and their rounding errors are unbiased. I kept the shift because it costs no
  instruction.
- **Per-head scales change nothing for heads drawn alike.** e4m3 is a float format: a finer
  scale only keeps small values out of the subnormal range, which a per-tensor scale already
  does unless one head is 2^15 smaller than another. `tests/test_attention_fp8.py` runs heads
  spread over three decades with per-head scales to check the indexing, not the benefit.

In the bench's own terms (uniform inputs, the dequantized reference, so only the kernel's
rounding counts) the largest error is 2.1e-2 against max |O| = 1 on the causal shapes, where
the first rows attend to a handful of keys and a rounded p is not averaged away; the check is
`max|O - O_ref| <= 0.04 max|O_ref| + 4e-3`. The pytest parity test uses the same bound
against SDPA on the dequantized inputs plus a mean error under 3% of mean |O| (measured 1.5 to
2% on its short sequences), on every shape of `test_attention.py`, both masks, GQA groups of
4 and 8, per-head scales, the split tail, the queue with more items than blocks, and a second
stream.

### What remains for fp8

- **The softmax chain.** 69% of the pipe. The rows above are the ways that did not work; the
  one left is ptxas-proof interleaving: writing the half-tile pipeline so that the `QMMA`s of
  one half and the softmax instructions of the other alternate in the source, one `QMMA`
  per handful of softmax instructions, and freeing the 32 registers it needs by keeping Q in
  shared memory (4 `ldmatrix` per tile per warp).
- **Decode.** A flash-decoding kernel on e4m3 K and V would halve the bytes a decode step
  streams, the one place fp8 attention helps at the memory roof.
- **Incoherent processing** for inputs with outlier channels, and per-block scales for K and V
  (the MX form the fp8 GEMM already takes) as the other answer to them.

## Correctness

`bench_attention` checks every output row of the small shapes against a CPU double-precision
reference computed from the same bf16-rounded inputs, and 64 sampled `(b, h, row)` triples
(always including the first and last row of the first head, the two extremes of the causal
mask) on the large ones. Q and K are uniform in [-2, 2] and V in [-1, 1], so the scaled scores
have a spread of a few units and the running max actually moves. Tolerance:
`max|O − O_ref| <= 0.02 · max|O_ref| + 1e-3`, one bf16 rounding of an fp32 result on our side
plus the bf16 rounding of P before the P V product in variants 2 to 5, the same form as
`bench_hgemm`. Measured errors are 1e-3 to 2.6e-3 against a tolerance of 2e-2 or more. The
pytest parity test compares every variant with `F.scaled_dot_product_attention` in fp32 on
the same bf16 inputs, including `S = 200`, `S = 1000`, `S_q = 1` and `S_q = 7` against a
longer cache, both masks and both head sizes. Variants 4 and 5 are also checked for K/V boxes
that hang off the end of a head (a bleed from the next head would show), for repeated calls and
a second stream, and variant 5 with 256 items on 170 blocks, two launches back to back and one
on another stream, so the queue counter's reset by the last block is exercised.

## RTX 5090 notes (measured)

Driver 595.58, CUDA 13.2, medians of 50 launches after the clock ramp, `bench_attention`
defaults. Nsight Compute runs at a fixed 2.53 GHz; the timed runs boost higher.

- **The roof.** 258.7 TFLOPS of dense bf16 `mma.sync` at 2,976 MHz (`bench_peak`), 239 at the
  2.75 GHz a sustained GEMM throttles to. Variant 4 reaches 239.3 TFLOPS on the non-causal
  4096 shape with the tensor pipe 94.1% active in Nsight, which puts the clock during the timed
  run at about 2.92 GHz: attention throttles less than a GEMM at the same pipe utilization
  (variant 3, at 224.5 TFLOPS and 90.1%, ran at about 2.86 GHz). 92.5% of the measured peak,
  and above the throttled GEMM roof.
- **What limits variant 2 / 3.** Nsight on 4096 causal: tensor pipe **89.3% active** (90.1%
  non-causal; 87.4% in the later session that profiled variant 4), issue slots 13.6% busy,
  0.55 instructions per cycle, dominant stall `math_pipe_throttle` at 9.7 of the 14.5 cycles
  between a warp's issues, then `wait` (2.1, fixed-latency dependencies), `barrier` 0.3,
  `long_scoreboard` 0.3, `mio_throttle` 0.3. Shared-memory bank conflicts: 0 (16 on the
  non-causal run). Shared load wavefronts 22% of peak. L2 hit rate 90.4% (94.7% non-causal),
  DRAM throughput 8.5% of peak. So the kernel is issuing to the tensor pipe as fast as it takes
  work, the loads are hidden, and the 10% of idle pipe is the softmax gaps where both warps of
  a scheduler are between products, plus the per-block prologue and epilogue. Occupancy is
  16.7% (8 warps), fixed by the 248 registers and the 96 KB tile; a second block per SM would
  need both halved.
- **What limits variant 4.** Tensor pipe 92.8% active causal, 94.1% non-causal, the same
  stall mix as variant 3 (the table in the variant 4 section). The remaining 6 to 7% is the
  per-block prologue (barrier init, the Q tile and the first K/V tile arriving before any `mma`
  can issue, 1,028 blocks of 33 tiles on the causal shape) plus the tiles where both warps of a
  scheduler still land in their softmaxes together. Forcing them apart with named barriers
  measured slower every way it was tried; the numbers and the reason are in that section.
- **What limits variant 5.** Tensor pipe 96.1% active causal (96.8% before the dead-tile skip
  took 1.5% of the products out), 97.5% non-causal, from 94.2% and 95.5% for variant 4 in the
  same session. What is gone is the per-block prologue and drain, 3.0 µs per block, 2.6% of
  the causal run; what is left is the per-item boundary (0.4%), the greedy queue's tail (0.6%)
  and the softmax coincidences. The variant 5 section has the tables.
- **The causal diagonal.** 215 TFLOPS by the halved count is 220 by the tiles actually
  computed. A 64-row Q tile would waste half as much on the diagonal at twice the K/V traffic
  per FLOP; not tried.
- **Variant 1** is issue-bound on the `LDS` + `FFMA` mix, 49 TFLOPS, 40% of the 123 TFLOPS FMA
  peak, 8.6x the instruction count of variant 3 for the same work. The fp32 staging of K and
  V in shared memory (32 KB each) is what keeps conversions out of the loop; register
  prefetch of the next tile costs 32 registers and takes it to 255.
- **Against torch** (`F.scaled_dot_product_attention`, which picks its FlashAttention-2
  kernel on every shape when left to choose; the cuDNN column forces cuDNN's SDPA with
  `scripts/bench_torch.py --only attention --sdpa-backend cudnn`, and it is 2 to 7% faster
  than flash on the prefill shapes):

| shape | v4 ms | v4 TFLOPS | flash ms | v4 / flash | cuDNN ms | v4 / cuDNN |
|---|---|---|---|---|---|---|
| b1 h32 s4096 d128 | 1.171 | 234.8 | 1.453 | 1.24x | 1.354 | 1.16x |
| b1 h32 s4096 d128 causal | 0.609 | 225.8 | 0.791 | 1.30x | 0.774 | 1.28x |
| b1 h32 s8192 d128 causal | 2.363 | 232.7 | 2.776 | 1.17x | 2.716 | 1.15x |
| b4 h32 s2048 d128 causal | 0.664 | 207.1 | 0.762 | 1.15x | 0.752 | 1.13x |
| b1 h32 s4096 d64 causal | 0.328 | 209.7 | 0.416 | 1.27x | 0.394 | 1.21x |
| b1 h32 sq1 skv4096 d128 (decode) | 0.054 | 1,247 GB/s | 0.071 | 1.31x | 0.069 | 1.29x |

  (`scripts/bench_torch.py` timings, 100 iterations, K and V rotated past L2 for the decode
  row, where variant 4 runs variant 3's kernel; the C++ bench, which allocates less between
  launches, gets 50 µs there. Variant 3 on the same script was 1.17x, 1.23x, 1.11x, 1.08x,
  1.22x and 1.38x over flash. The GQA and long-context decode rows are in the "Long-context
  decode" section.)

  Variant 5 on the same script (2026-09-28, the default rung, so `sk.attention` with no
  `variant`; the TFLOPS are the C++ sweep's medians of 50 from the same session):

| shape | v5 ms | v5 TFLOPS | flash ms | v5 / flash | cuDNN ms | v5 / cuDNN |
|---|---|---|---|---|---|---|
| b1 h32 s4096 d128 | 1.169 | 242.2 | 1.458 | 1.25x | 1.356 | 1.16x |
| b1 h32 s4096 d128 causal | 0.589 | 236.1 | 0.791 | 1.34x | 0.775 | 1.32x |
| b1 h32 s8192 d128 causal | 2.356 | 239.1 | 2.776 | 1.18x | 2.723 | 1.16x |
| b4 h32 s2048 d128 causal | 0.623 | 227.3 | 0.763 | 1.22x | 0.752 | 1.21x |
| b1 h32 s4096 d64 causal | 0.303 | 235.3 | 0.418 | 1.38x | 0.394 | 1.30x |
| b1 hq32 hkv8 s4096 d128 causal | 0.594 | 237.1 | 0.788 | 1.33x | 0.774 | 1.32x |
| b1 h32 sq1 skv4096 d128 (decode, variant 3's kernel) | 0.050 | 1,462 GB/s | 0.071 | 1.43x | 0.069 | 1.39x |

## Results (RTX 5090, sm_120, CUDA 13.2, driver 595.58)

From the default `bench_attention` sweep, run 2026-09-28 (median of 50); the variant 5 column
is the same sweep run 2026-09-28, with variant 4 rerun in that session for the comparison in
its section (it measured 228.9 causal and 239.8 non-causal at 4096 that day, so the two
sessions agree to within 1%). TFLOPS by the halved causal count; the decode rows in GB/s of
Q, K, V and O once (K and V once per K/V head). The decode rows of variants 0 to 2 read each
K/V head once per query head, so under GQA they land at a quarter of their MHA figure;
variants 3 to 5 all run the flash-decoding kernel there, so their decode rows are the same
kernel timed three times. Bold marks the fastest rung per row.

| shape | v0 ms / TFLOPS | v1 | v2 | v3 | v4 | v5 |
|---|---|---|---|---|---|---|
| b1 h4 s512 d128 | 0.0809 / 6.6 | 0.0829 / 6.5 | 0.0297 / 18.1 | **0.0123 / 43.6** | 0.0131 / 41.0 | 0.0131 / 41.0 |
| b1 h4 s512 d128 causal | 0.0788 / 3.4 | 0.0810 / 3.3 | 0.0297 / 9.0 | **0.0185 / 14.5** | 0.0189 / 14.2 | 0.0189 / 14.2 |
| b1 h4 s512 d64 | 0.0563 / 4.8 | 0.0460 / 5.8 | 0.0153 / 17.5 | **0.0083 / 32.5** | 0.0090 / 30.0 | 0.0090 / 29.9 |
| b1 h4 s512 d64 causal | 0.0542 / 2.5 | 0.0460 / 2.9 | 0.0169 / 8.0 | 0.0122 / 11.0 | **0.0112 / 12.0** | 0.0114 / 11.7 |
| b1 h4 s200 d128 | 0.0317 / 2.6 | 0.0420 / 2.0 | 0.0173 / 4.7 | 0.0101 / 8.1 | **0.0099 / 8.3** | 0.0109 / 7.5 |
| b1 h4 s200 d128 causal | 0.0297 / 1.4 | 0.0420 / 1.0 | 0.0154 / 2.7 | **0.0124 / 3.3** | 0.0129 / 3.2 | 0.0128 / 3.2 |
| b1 hq8 hkv2 s512 d128 causal | 0.0849 / 6.3 | 0.0829 / 6.5 | 0.0296 / 18.2 | **0.0191 / 28.1** | 0.0192 / 28.0 | 0.0191 / 28.1 |
| b1 h32 s4096 d128 | 26.2875 / 10.5 | 4.9853 / 55.1 | 1.4137 / 194.4 | 1.2346 / 222.6 | 1.1574 / 237.5 | **1.1515 / 238.7** |
| b1 h32 s4096 d128 causal | 13.8511 / 9.9 | 2.5308 / 54.3 | 0.6377 / 215.5 | 0.6406 / 214.6 | 0.6019 / 228.3 | **0.5802 / 236.9** |
| b1 h32 s8192 d128 causal | 57.0156 / 9.6 | 11.0320 / 49.8 | 2.4839 / 221.3 | 2.4859 / 221.1 | 2.3342 / 235.5 | **2.2994 / 239.1** |
| b4 h32 s2048 d128 causal | 13.0545 / 10.5 | 2.5555 / 53.8 | 0.6927 / 198.4 | 0.6858 / 200.4 | 0.6448 / 213.2 | **0.6049 / 227.2** |
| b1 h32 s4096 d64 causal | 9.6376 / 7.1 | 1.3280 / 51.7 | 0.3302 / 208.1 | 0.3307 / 207.8 | 0.3179 / 216.2 | **0.2919 / 235.4** |
| b1 h32 sq1 skv4096 d128 | 1.2599 / 53 GB/s | 0.6448 / 104 GB/s | 0.1995 / 336 GB/s | **0.0459 / 1,462 GB/s** | 0.0460 / 1,459 GB/s | 0.0460 / 1,461 GB/s |
| b1 hq32 hkv8 s4096 d128 causal | 13.9496 / 9.9 | 2.4756 / 55.5 | 0.6417 / 214.2 | 0.6357 / 216.2 | 0.5980 / 229.8 | **0.5796 / 237.1** |
| b1 hq32 hkv8 sq1 skv4096 d128 | 1.2452 / 13 GB/s | 0.6459 / 26 GB/s | 0.1995 / 84 GB/s | 0.0173 / 972 GB/s | 0.0173 / 972 GB/s | **0.0172 / 975 GB/s** |
| b1 h32 sq1 skv131072 d128 | 40.1835 / 53 GB/s | 20.5778 / 104 GB/s | 6.2703 / 342 GB/s | **1.2672 / 1,695 GB/s** | 1.2678 / 1,694 GB/s | 1.2672 / 1,695 GB/s |
| b1 hq32 hkv8 sq1 skv131072 d128 | 39.5534 / 14 GB/s | 20.5738 / 26 GB/s | 6.2655 / 86 GB/s | **0.3273 / 1,640 GB/s** | 0.3284 / 1,635 GB/s | 0.3284 / 1,635 GB/s |
| b8 hq32 hkv8 sq1 skv4096 d128 | 1.2172 / 110 GB/s | 1.3380 / 100 GB/s | 0.3981 / 337 GB/s | 0.0867 / 1,550 GB/s | 0.0866 / 1,551 GB/s | **0.0866 / 1,552 GB/s** |

The small shapes are where the tail split matters most: 16 tiles on 170 SMs become 128
slices, 2.4× on the 512-token shape; variant 4's block prologue costs it a microsecond there,
and variant 5, which keeps the split on those shapes, matches it. On every 4096-token-and-up
prefill shape variant 4 is 4 to 6% ahead of variant 3 and variant 5 another 1 to 9% ahead of
variant 4 (most on the batch of four and on `D = 64`, whose tiles are shortest and whose
prologue share was therefore largest), and the GQA prefill row matches its MHA twin, as it
should (same math, a stride on the K/V pointer).

## The K/V head stride

`attention_bf16` takes `kv_cap`, the rows each K/V head is allocated, and reads the first
`S_kv` of them: head `bkv` starts at `bkv * kv_cap * D`. The pointer variants and the
flash-decoding kernel put it in the head offset; variants 4 and 5 put it in the outer
stride of their K/V tensor maps, where the box past `S_kv` stays out of range and
zero-filled as before. The binding reads it off the strides of `cache[:, :, :length]`, so
a layer's cache with room to grow is read where it is, and `bench_attention --kvcap=N`
times the strided read against the packed one (four such rows are in the default sweep).

## What remains

- **The last 3%.** The tensor pipe is 96 to 97.5% active in variant 5; the block prologue
  is gone. What is left is mostly the tiles where both warps of a scheduler meet in their
  softmaxes. A 128-key tile would halve how often that can happen per FLOP, at 32 more
  registers for S and 64 KB per stage, which the 168-register budget of the nine-warp block
  and the 99 KB of shared memory do not have; it would need the producer back on a consumer
  lane (the inline scheme, 1.6% slower here) or a two-stage pipeline.
- **The producer's register cost.** Nine warps cap the kernel at 168 registers per thread
  because three warps share one scheduler's 16 K registers. The inline producer keeps 8 warps
  and 228 registers but stalls warp 0 on the slowest warp's release. A producer that polls the
  barriers without blocking (`mbarrier.test_wait`) from lane 0 of warp 0, issuing whenever a
  stage happens to be free and blocking only when the load is about to be needed, has not
  been tried; it would give the persistent kernel variant 4's register budget back.
- **Ping-pong, revisited.** The named-barrier schedules lost because one `mma.sync` warp
  does not fill the pipe alone. A version that keeps two warps per scheduler in the `mma`
  phase and only forbids two softmaxes at once (a barrier a warp takes before its softmax and
  releases after, with one slot per scheduler) has not been tried.
- **Backward.** Done: the forward writes the log-sum-exp on request, and the backward pass
  is its own ladder in [attention_bwd.md](attention_bwd.md).
- **GQA prefill with grouped tiles.** The prefill kernels read a K/V head once per query head
  and let L2 serve the group; a 128-row tile holding 4 heads x 32 tokens would read it once
  from L2 too. Prefill is compute-bound, so this is a few percent at most.
- **Decode with more rows per K/V head** (8 heads x 8 tokens, speculative decoding) on the
  flash-decoding kernel: two or four m16 tiles per warp instead of one, which the register
  budget (64 accumulators per m16 tile at `D = 128`) allows twice over.
