# Inkling 2 Mini attention: the relative position bias in the kernel

Inkling 2 Mini (thinkingmachines/Inkling-2-Mini) has no RoPE. Position enters attention as an
additive bias on the logits, computed per query and per head from a 16-dim "relative state"
and a learned bank of bias-vs-distance profiles. Every one of its 48 layer applications per
token runs this, so the bias has to live in the attention kernel: HF builds it as a
[B, H, T, S] tensor, which at 16K tokens is 12.9 GB in bf16 for one layer.
`src/kernels/inkling_attention.cu`, in Python `sk.inkling_attention`.

## What it computes

From `eager_attention_forward` and `Inkling2LoopRelativePositionBias` in HF's
`modeling_inkling2_loop.py` (transformers 5.24.1):

```
s_ij = q_i . k_j / sqrt(128) + b_ij
b_ij = r_i . P[:, d]   for d = pos_i - j, 0 <= d < extent;   0 otherwise
O_i  = sum_j exp(s_ij - m) v_j / (exp(sink_i - m) + sum_j exp(s_ij - m))
```

q is [B, 24, T, 128] (clipped and scaled by the q projection before it gets here), k and v
are [B, 8, S, 128] (GQA 3:1), r is [B, 24, T, 16] (the relative states, already clipped), P is
`relative_position_bias.proj`, [16, 1024], one per layer and shared by all heads, and the sink
is one logit per query and head ([B, 24, T], from `sink_proj(hidden)`) with a zero value. The
queries are the last T of S positions, `pos_i = i + S - T`, which is what
`get_relative_distance` gives with a cache. The mask is causal; the 20 sliding layers add
HF's `sliding_window_overlay`, `kv_idx > q_idx - 1024`, so distances 0 to 1023 are attended
and 1024 is not. In those layers the window equals the extent, so every attended key has a
bias. The 4 full layers attend everything and only the last 1024 keys get one.

HF rounds the logits to bf16 (the matmul and the bias add are bf16, only the softmax is
fp32). The kernel keeps every logit in fp32, so it matches an fp32 version of HF's math to one
bf16 rounding of the output, not HF's bf16 eager bit for bit.

## Folding the bias into a flash tile

The bias is a Toeplitz product: `b_ij` is row i of `R P` read at column `pos_i - j`, a
different column per row. For a 64 x 64 tile (rows `q0 ..`, keys `kv0 ..`) the distances run
from `q0 + off - kv0 - 63` to `q0 + off - kv0 + 63`: 127 columns of P out of 1024. So:

1. **Stage the band with K and V.** Each stage of the cp.async pipeline holds the K tile, the V
   tile and P's columns `[floor8(dmin), floor8(dmin) + 144)` as [16][144] bf16 (4.6 KB, next to
   the 32 KB of K and V). The base is 8-aligned so every 16-byte chunk of P is wholly inside or
   outside `[0, extent)`, and the chunks outside are zero-filled by the copy. That zero fill is
   the "bias 0 past the extent" rule of the full layers; negative distances are masked anyway.
2. **Multiply on the tensor cores.** Each warp holds its 16 rows of R as one m16n8k16 A
   fragment (loaded once from global, 4 registers) and multiplies it by its 96 columns of the
   band (warp w's distances start 16 w columns in): 12 mma per KV tile, against the 128 of
   `Q K^T` and `P V`. The result, `R P` for 16 rows by 96 distances, is fp32.
3. **Skew through shared memory.** The warp parks that 16 x 96 block (times log2 e, the
   kernel's exponent domain) in a private scratch and each lane reads back the entries of its
   score fragment, `row * SB + (pos - j - dw)`. The row stride is SB = 103: lane (g, c2) reads
   word `g (SB + 1) - c2 + const`, and SB + 1 = 8 (mod 32) puts at most two lanes on a bank.
   (No stride makes it conflict-free: c2 is even, so the 8 rows would need offsets that cover
   both parities at spacing 8, which an arithmetic progression mod 32 cannot do.)
4. `s = fmaf(qk, log2e / sqrt(128), bias)`, then the masks, then the online softmax of
   `attention.cu` variant 2 unchanged. The sink joins the denominator where the row is
   normalized, `sink_norm` of `attention_internal.cuh`, read per query instead of per head.

Per KV tile and warp this adds 12 mma, 6 `ldmatrix.x4.trans`, 48 scratch stores and 32 scratch
loads to the 128 mma and 64 `ldmatrix.x4` of the tile. The cost is shared memory: the scratch
(26 KB for 4 warps) and the band push the block to 101,120 B, so the pipeline has two stages
where variant 2 has three, and one block per SM.

**Skips.** The KV tiles run from the one holding the first row's window edge to the one holding
the last row's causal bound (`kv_tiles` of `attention.cu`), so a sliding layer at 16K tokens
reads 17 KV tiles per Q tile, not up to 256. The masks run only on tiles a warp's rows cross
an edge of. And a tile whose smallest distance is past the extent (in a full layer, all but
the last 1024 keys of every row) stages no band and skips the bias work: its bias is zero.

## Decode

T = 1 (and up to 5 tokens for speculative decoding) runs the flash-decoding layout of
`attention_decode.cu`: one block per (b, kv head, key slice), the 3 T rows of the K/V head's
query heads in one 16-row tile, each of the four warps streaming its own 16-key slabs through a
private pipeline, the partials merged in shared memory and across slices by the last block to
arrive. The band of a 16-key slab is 16 + T - 1 distances, staged as 32 columns and multiplied
with 4 mma. The extra shared memory (band and scratch) cuts the per-warp pipeline from three
slabs to two; at 8 KB a slab that is still 32 KB in flight per SM.

## Measured

`scripts/bench_inkling_attention.py`, RTX PRO 6000 (sm_120), CUDA 13.2, under the GPU lock,
medians of 30 (prefill) and 100 (decode) launches; rows in
`results/inkling_attention_torch.jsonl`. Inkling's heads: 24 query, 8 K/V, D = 128, a
[16, 1024] bank, per-query sinks. Columns: this kernel; this kernel with an 8-column bank, so
only the diagonal tiles do any bias work; `sk.attention` with the same window and causal mask
and no bias or sinks (its default, variant 5, and variant 2, the cp.async + mma.sync tile this
kernel is built from); and `F.scaled_dot_product_attention` with the whole op in one float
mask (bias, mask, and the sink as a zero key with its logit in the mask; K and V repeated to
24 heads; the mask built once and not timed).

Prefill, B = 1, ms (TFLOP/s by the pairs the mask keeps, 4 x 128 FLOP a pair):

| tokens | layer | ours | 8-column bank | attention v5, no bias | v2, no bias | SDPA + mask |
|---|---|---|---|---|---|---|
| 4096 | sliding | 0.244 (184) | 0.241 | 0.171 (264) | 0.189 | 2.25 |
| 4096 | full | 0.481 (214) | 0.478 | 0.303 (340) | 0.331 | 2.25 |
| 16384 | sliding | 1.032 (194) | 1.025 | 0.653 (306) | 0.732 | 31.8 |
| 16384 | full | 7.05 (234) | 7.13 | 4.58 (360) | 4.90 | 32.0 |

Decode, T = 1, ms (GB/s of the K/V rows read). K and V rotate through copies of the cache so
every launch reads from DRAM; at S = 1024 the window holds every key, so both layer kinds read
the same rows:

| cached keys | B | layer | ours | flash-decoding, no bias | SDPA + mask |
|---|---|---|---|---|---|
| 1024 | 1 | sliding | 0.0198 (213) | 0.0180 | 0.088 |
| 1024 | 1 | full | 0.0179 (236) | 0.0155 | 0.088 |
| 1024 | 8 | sliding | 0.0366 (920) | 0.0354 | 0.140 |
| 1024 | 8 | full | 0.0359 (938) | 0.0344 | 0.140 |
| 32768 | 1 | sliding | 0.0167 (252) | 0.0149 | 1.76 |
| 32768 | 1 | full | 0.104 (1292) | 0.103 | 1.76 |
| 32768 | 8 | sliding | 0.0359 (938) | 0.0348 | 3.82 |
| 32768 | 8 | full | 0.717 (1497) | 0.715 | 3.82 |

What the numbers say:

- **The bias arithmetic is close to free.** The 8-column bank does no bias work off the
  diagonal and is within 1.5% of the real bank on every prefill row (the 16K full row is
  slower with it, which is noise). The 12 extra mma, the scratch round trip and the band's
  4.6 KB per stage hide under the 128 mma of the tile.
- **The tile is what costs.** Against `sk.attention` without the bias the prefill is 1.4x to
  1.6x slower, and against variant 2 (the same cp.async + mma.sync structure) 1.29x to 1.44x.
  Variant 2 runs 8 warps (a 128-row Q tile) and three stages in 96 KB. Here the band and the
  scratch take 30 KB of the 99 KB a block can have, which leaves room for 4 warps and two
  stages: one warp per SM sub-partition, so nothing hides one warp's ldmatrix and exp
  latencies. The bias's real price is shared memory, paid as a smaller tile.
- **An 8-warp tile does not win either.** I tried one that fits: 128 rows, the bias done in
  two halves of 32 keys (8 n8 tiles each, a 208-column band) into an fp16 scratch with row
  stride 71 halves, which is conflict-free for the gather (the eight rows land 4 banks apart),
  97,024 B in all. Its base got faster (the 8-column bank: 0.979 ms at 16K sliding, 6.28 at
  16K full, against 1.025 and 7.13), but the bias work went from at most 1.5% of the time to
  up to 16% on the sliding rows: twice the band mma, 64 two-byte scratch stores a tile, the
  halves' extra warp syncs, and every warp walking the 18 KV tiles of the whole 128-row tile.
  Net, 0.287 / 0.472 / 1.125 / 6.42 ms for the four prefill rows above (same session, same
  bench; that code is not in the tree): worse on the sliding layers, which are 20 of the 24.
  So the 4-warp tile stays. The way forward is probably variant 5's TMA pipeline, where a
  producer warp could stage the band and the consumers keep 8 warps.
- **SDPA with an explicit bias is 4.5x to 31x slower** on prefill (the most on the sliding
  layers, where its dense mask cannot skip anything) and needs the [B, 24, T, S] mask (12.9 GB
  at 16K tokens, before the temporaries that build it). HF's eager path materializes the same
  tensor; I did not time it.
- **Decode reads at the no-bias kernel's rate.** At 32K cached keys and B = 8 the full layer
  streams 1.07 GB in 0.717 ms, 1.50 TB/s, 0.3% behind the flash-decoding kernel without the
  bias; the other rows are 1% to 16% behind it. The small rows (a sliding layer reads 1024
  keys, 4 MB at B = 1) are a launch's fixed cost, 15 to 20 us, not bandwidth.

Parity (`tests/test_inkling_attention.py`, on the same card): 27 cases against an fp32 torch
copy of HF's math, both layer kinds, T in {1, 3, 5, 17, 64, 1000, 2048} with caches up to 4100
keys (so distances past the extent are read), B = 2, both kernels on the decode shapes, k and v
as the leading rows of a bigger cache, and a decode step's rows against the same rows of a
2000-token prefill. On five of those shapes the max error against the reference is 0.001 to
0.008, while dropping the bias moves the reference by 0.48 to 3.4 and dropping the sinks by
0.04 to 3.4, so the test sees both.

## What is not done

- No backward, no fp8 K/V, no paged cache. The decode kernel reads a contiguous cache
  (`k[:, :, :S]` of a bigger one is fine).
- T between 6 and 16 runs the 64-row prefill tile per query head, which reads K and V three
  times (once per head of the group).
