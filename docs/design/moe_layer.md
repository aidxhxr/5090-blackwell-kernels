# A whole MoE layer

The grouped GEMM runs every expert of an MoE layer in one launch, but a layer is more than
its two expert GEMMs. This note puts the rest of it on the device and times the whole layer
against what people run: the transformers loop, torch's own grouped matmul, and vLLM's
`fused_moe`. Source: `python/spark_kernels/moe_layer.py` (the layer, the two torch layers,
the configs), `src/kernels/moe_route.cu` (routing), `src/kernels/moe_permute.cu` (permute
and combine), the SwiGLU epilogue and the small-row tiles in `src/kernels/grouped_gemm.cu`.
Bench: `scripts/bench_moe_layer.py`, results in `results/moe_layer.json`. Tests:
`tests/test_moe_layer.py`.

## The layer

Two configurations, bf16, softmax routing with the top-k weights renormalized to sum 1 (both
models do that: Mixtral always, Qwen3-MoE through `norm_topk_prob`):

| config | experts | top-k | hidden | inter | one expert | the layer's experts |
|---|---|---|---|---|---|---|
| Mixtral 8x7B | 8 | 2 | 4096 | 14336 | 352 MB | 2.82 GB |
| Qwen3-30B-A3B | 128 | 8 | 2048 | 768 | 9.4 MB | 1.21 GB |

What a layer does for T tokens, in the order it runs:

```
logits = x @ router.T                    [T, E] bf16 (F.linear, cuBLAS)
ids, w = topk(softmax(logits), k)        renormalized: w / w.sum(-1)
sort the T * k (token, slot) pairs by expert; offsets per expert
xs     = x[token of each sorted row]     [T * k, D]
h      = silu(xs @ w_gate[e].T) * (xs @ w_up[e].T)     per expert, [T * k, I]
y      = h @ w_down[e].T                 [T * k, D]
out[t] = sum over s of w[t, s] * y[row of (t, s)]
```

In transformers this is a loop over the experts that got a token, with a `torch.where` per
expert to find its rows (a device to host sync each), then a gather, two GEMMs, three
elementwise kernels and an `index_add_` per expert. vLLM, from its source: the router GEMM,
a top-k softmax kernel, `moe_align_block_size` (the sort, padded to its block size; one
small-batch path skips it), the first Triton GEMM with the gather folded into its A loads,
the activation (`silu_and_mul`), the second Triton GEMM with the routing weight folded in,
and `moe_sum`.

## Ours

Six launches, none of which waits on the host:

```
logits = F.linear(x, router)                       cuBLAS
xs, ids, w, offsets, pos = moe_dispatch(x, logits, k)    route_kernel + permute_kernel
h = grouped_gemm_swiglu(xs, w13, offsets)          gate|up with silu(g) * u in the epilogue
y = grouped_gemm(h, w2, offsets)                   down
out = moe_combine(y, pos, w)                       combine_kernel
```

**Routing, one launch.** `route_kernel` takes the logits (bf16 or fp32, E <= 512, k <= 16)
and gives the top-k ids and weights, the per-expert offsets and, for every (token, slot),
its rank among its block's assignments to the same expert:

- 512-thread blocks, a warp per token: the warp holds the token's E logits in registers (4
  per lane for 128 experts, the next token's loaded before this one's picks),
  computes the softmax max and sum in fp32, then picks k experts. A pick is two `redux.sync`
  instructions: the max over lanes of each lane's best untaken logit, as an order-preserving
  u32 key, then the min expert id among the lanes that hold it. So the picks are exact on
  the logits, largest first, ties to the lower id, which is torch.topk's order. The weights
  are exp(l - max) / sum over all E, divided by their own sum;
- the block's assignments in (token, slot) order are ranked within their experts by one warp,
  32 at a time, with `__match_any_sync` (the lanes going to the same expert find each other,
  a popcount of the lower ones is the rank, the highest one adds the group to the block's
  count for that expert);
- the last block to finish (an integer atomic on a counter it resets) loads every block's
  counts into shared memory with coalesced loads, scans each expert's column over the
  blocks and the totals over the experts, and writes `offsets[E + 1]` and `base[b][e]`, the
  first row block b's assignments to expert e take. A routing that fits one block (16 tokens
  or fewer) skips the counter and the fences.

A row's place in expert order is `base[block][e] + rank`, which sorts by expert, then token,
then slot: the stable sort of the flattened ids, the same order `moe_offsets` produces. The
blocks are at least 16 tokens and at most 128 of them, and at most 8192 / E so the counts
fit the 32 KB the block already holds for its assignments' experts.

**Permute.** The addition is done by the next kernel: `permute_kernel` takes four tokens per
256-thread block, finishes their k positions, loads each token's row once (16 bytes per
lane) and stores it k times into `xs`. x is read once and `xs` written once.

**Gate/up with the SwiGLU in the epilogue.** `interleave_experts` stores each expert's gate
and up rows alternating, row 2j gate_j and row 2j + 1 up_j, once at load time. The mma
accumulator gives a lane columns 2c and 2c + 1 of every n8 tile, so gate_j and up_j land in
the same lane's register pair and the epilogue forms silu(g) * u in fp32 from the unrounded
sums. Lanes 2q and 2q + 1 hold output columns j and j + 1 of the same two rows, so one
shuffle lets each store a bf16 pair: hgemm's SwiGLU epilogue, on the grouped tile. The
Stream-K fixup runs on the accumulators before it, unchanged. The gate/up output is never
written: at 16384 tokens on Qwen3 that is 403 MB written and read again by the swiglu kernel.

**Tiles of 16 and 64 rows.** The grouped GEMM's tile was 128 rows. An expert with 8 rows
(Qwen3 at 128 tokens) then costs 16 times its MMA work. A 128-row stage over 16 KB of weight
is 512 m16n8k16, about 1.4 us at the tensor-core peak, so one block streams at most about
12 GB/s of weight. The 1.6 TB/s read needs 9.4 GB/s from each of the 170 SMs, which leaves
little room, and with fewer tiles than SMs it cannot be reached at all: at one token Qwen3's
gate/up is 96 tiles, 96 blocks, at most 1.1 TB/s (the first run of the layer measured its two
GEMMs at 0.9 TB/s there). The kernel is now templated on the tile height: 16 rows (eight warps
side by side, each 16 x 16, four 18 KB stages, and Stream-K ranges down to 8 k-steps since a
16 x 128 fp32 slot is only 8 KB) and 64 rows (2 x 4 warps of 32 x 32, three 24 KB stages),
beside the 128-row tile. The host picks from the average rows per expert, T * k / E, which it
knows without the counts: up to 24 rows the 16-row tile, up to 512 the 64-row tile, above
that 128. That is for bf16; fp8 keeps the 128-row tile (the next section but one says why).
`SPARK_GROUPED_BM=16|64|128` forces one for either type. The plain `grouped_gemm` takes the
same rule.

**Combine.** `combine_kernel` gives each token its k rows of `y` times their weights, summed
in fp32 in slot order and rounded once, a block per token with each thread's k row loads
issued four at a time (four tokens per block with one load in flight cost 11 to 13 us at 16
and 128 tokens on Qwen3; this is 3 to 4). Every output element is one thread's sum in a fixed
order: no scatter, no atomics, the same bits every run (the tests check that, and that the
whole layer is bitwise repeatable).

## Results

Measured on the RTX 5090 on 2026-10-23: driver 595.58, CUDA 13.2, torch 2.14+cu130 for ours
and the torch layers; vLLM 0.30.0 from its own environment (torch 2.13, Triton 3.7.1) with
its default `fused_moe` config, since vLLM ships no tuned file for these shapes on this card
(it says so in a warning). Other tenants held about 18 GB of the card; nvidia-smi showed it
idle before every run but the tile sweep's, whose rows matched the later runs within 1%. `scripts/bench_moe_layer.py`: median of 50 calls, CUDA events around each call and
no sync between calls, after a 300 ms clock ramp; the input cycles through a pool of token
batches so consecutive calls route to different experts and the weights come from DRAM (8
Qwen3 experts are 75 MB and would otherwise sit in the 96 MB L2). Every implementation
includes the router GEMM, the same `F.linear` in all of them. Every row was checked first
against the transformers layer with fp32 accumulation: ours is within 8.6e-3 of max |ref| on
every row, every implementation within 1.04e-2 (the bf16 transformers layer is the farthest),
and the routing kernel picked the same logits as torch.topk for every token of each row's
first input.

ms per layer:

| config, tokens | ours | vLLM fused_moe | torch._grouped_mm | transformers loop | vs vLLM | vs the loop |
|---|---|---|---|---|---|---|
| Mixtral 8x7B, 1 | 0.440 | 0.440 | 0.647 | 0.624 | 1.00x | 1.42x |
| Mixtral 8x7B, 16 | 1.716 | 1.725 | 2.043 | 2.455 | 1.01x | 1.43x |
| Mixtral 8x7B, 128 | 1.747 | 1.766 | 2.075 | 2.459 | 1.01x | 1.41x |
| Mixtral 8x7B, 1024 | 3.852 | 4.413 | 4.370 | 4.583 | 1.15x | 1.19x |
| Mixtral 8x7B, 4096 | 13.01 | 14.14 | 14.98 | 14.85 | 1.09x | 1.14x |
| Mixtral 8x7B, 16384 | 49.12 | 53.61 | 57.20 | 55.78 | 1.09x | 1.14x |
| Qwen3-30B-A3B, 1 | 0.064 | 0.199 | 0.761 | 1.236 | 3.10x | 19.2x |
| Qwen3-30B-A3B, 16 | 0.480 | 0.474 | 1.647 | 10.92 | 0.99x | 22.8x |
| Qwen3-30B-A3B, 128 | 0.750 | 0.759 | 2.708 | 17.42 | 1.01x | 23.2x |
| Qwen3-30B-A3B, 1024 | 0.820 | 0.886 | 2.477 | 18.53 | 1.08x | 22.6x |
| Qwen3-30B-A3B, 4096 | 1.794 | 1.971 | 4.104 | 18.31 | 1.10x | 10.2x |
| Qwen3-30B-A3B, 16384 | 6.394 | 6.907 | 14.27 | 18.81 | 1.08x | 2.94x |

What the call has to move and compute. The experts touched are counted on the first input
of the pool; the GB/s column is those experts' bf16 weights over the call's time, which is
the floor up to 128 tokens, and TFLOPS counts the 6 D I FLOPs of each of the T * k rows.

| config, tokens | experts touched | their weights | ours GB/s | vLLM GB/s | ours TFLOPS | vLLM TFLOPS |
|---|---|---|---|---|---|---|
| Mixtral 8x7B, 1 | 2 | 0.70 GB | 1,602 | 1,603 | 1.6 | 1.6 |
| Mixtral 8x7B, 16 | 8 | 2.82 GB | 1,643 | 1,634 | 6.6 | 6.5 |
| Mixtral 8x7B, 128 | 8 | 2.82 GB | 1,613 | 1,596 | 51.6 | 51.1 |
| Mixtral 8x7B, 1024 | 8 | 2.82 GB | 732 | 639 | 187.3 | 163.5 |
| Mixtral 8x7B, 16384 | 8 | 2.82 GB | 57 | 53 | 235.0 | 215.3 |
| Qwen3-30B-A3B, 1 | 8 | 0.08 GB | 1,176 | 380 | 1.2 | 0.4 |
| Qwen3-30B-A3B, 16 | 86 | 0.81 GB | 1,692 | 1,711 | 2.5 | 2.5 |
| Qwen3-30B-A3B, 128 | 128 | 1.21 GB | 1,610 | 1,593 | 12.9 | 12.7 |
| Qwen3-30B-A3B, 1024 | 128 | 1.21 GB | 1,473 | 1,363 | 94.3 | 87.2 |
| Qwen3-30B-A3B, 16384 | 128 | 1.21 GB | 189 | 175 | 193.4 | 179.1 |

Where our call goes, kernel microseconds per call from the torch profiler (10 calls):

| config, tokens | router GEMM | route | permute | gate/up + SwiGLU | down | combine | kernels | call |
|---|---|---|---|---|---|---|---|---|
| Mixtral 8x7B, 1 | 3.4 | 3.2 | 2.1 | 281 | 147 | 2.0 | 439 | 440 |
| Mixtral 8x7B, 16 | 23.9 | 2.8 | 3.2 | 1,108 | 574 | 2.0 | 1,714 | 1,716 |
| Mixtral 8x7B, 128 | 4.5 | 3.9 | 3.3 | 1,128 | 602 | 2.2 | 1,743 | 1,747 |
| Mixtral 8x7B, 1024 | 8.5 | 4.3 | 8.0 | 2,442 | 1,346 | 10.5 | 3,819 | 3,852 |
| Mixtral 8x7B, 4096 | 26.3 | 5.6 | 30.0 | 8,537 | 4,300 | 39.4 | 12,939 | 13,005 |
| Mixtral 8x7B, 16384 | 87.6 | 11.5 | 269 | 32,172 | 16,487 | 218 | 49,245 | 49,117 |
| Qwen3-30B-A3B, 1 | 2.4 | 3.5 | 1.6 | 30.8 | 17.3 | 2.1 | 57.7 | 64.2 |
| Qwen3-30B-A3B, 16 | 4.3 | 4.4 | 4.3 | 308 | 156 | 3.3 | 480 | 480 |
| Qwen3-30B-A3B, 128 | 4.6 | 5.5 | 4.3 | 485 | 245 | 4.1 | 748 | 750 |
| Qwen3-30B-A3B, 1024 | 8.2 | 7.0 | 13.6 | 507 | 271 | 11.3 | 818 | 820 |
| Qwen3-30B-A3B, 4096 | 22.1 | 14.6 | 75.6 | 1,060 | 542 | 75.0 | 1,790 | 1,794 |
| Qwen3-30B-A3B, 16384 | 54.9 | 44.1 | 402 | 3,667 | 1,863 | 352 | 6,384 | 6,394 |

What the numbers say:

* Up to 128 tokens the layer is a weight read and both we and vLLM are at it: 1.59 to 1.71
  TB/s on the weights the routing touched, above the 1,532 GB/s `cudaMemcpy` does (a copy
  writes as much as it reads; this only reads). The two are within about 1% of each other on
  all of those rows, and vLLM is ahead on one of them: Qwen3 at 16 tokens, by 1.1% (0.474
  against 0.480 ms).
  Our two GEMMs there read their 0.81 GB at 1.75 TB/s; the 16 us of small kernels around
  them (router 4.3, route 4.4, permute 4.3, combine 3.3) is the difference. vLLM folds the
  gather into its first GEMM's loads, so it has no permute.
* Qwen3 at one token: 0.064 ms against vLLM's 0.199. Our kernels are 58 us, the two GEMMs 48
  us of it for 75 MB. vLLM's call takes longer than its weights do to read; I did not profile
  where its time goes.
* From 1024 tokens up the layer is compute and our GEMMs are faster than vLLM's Triton
  kernels at their default config: 1.09x to 1.15x on Mixtral (235 against 215 TFLOPS at
  16384 tokens; the grouped GEMM alone did 240 at 8192) and 1.08x to 1.10x on Qwen3.
* Against the transformers loop: 19x to 23x on Qwen3 up to 1024 tokens, where the loop pays
  a `torch.where` and a sync for each of up to 128 experts, 10x at 4096 and 2.9x at 16384;
  1.14x to 1.43x on Mixtral, which has 8 experts. `torch._grouped_mm` runs on sm_120 in torch
  2.14 but is 2.2x to 3.6x slower than ours on Qwen3 from 16 tokens up (12x at one), and
  slower than the loop on Mixtral from 4096 tokens up; I did not look at what it dispatches
  to.
* The router GEMM on Mixtral at 16 tokens is 23.9 us, against 4.5 at 128: cuBLAS through
  `F.linear` on [16, 4096] x [4096, 8]. Every implementation makes the same call, so it is in
  every row; I did not chase it.

## The tile height

The whole layer with the tile forced (`SPARK_GROUPED_BM`), ms, 30 calls, from before the
combine and the routing were last changed (they move these rows by a few microseconds at most):

| config, tokens | rows per expert | 16-row tile | 64-row tile | 128-row tile |
|---|---|---|---|---|
| Mixtral 8x7B, 1 | 0.25 | 0.441 | 0.449 | 0.561 |
| Mixtral 8x7B, 16 | 4 | 1.722 | 1.746 | 1.793 |
| Mixtral 8x7B, 128 | 32 | 1.952 | 1.749 | 1.809 |
| Mixtral 8x7B, 1024 | 256 | 7.779 | 3.842 | 3.974 |
| Mixtral 8x7B, 4096 | 1024 | 29.91 | 13.30 | 12.91 |
| Qwen3-30B-A3B, 1 | 0.06 | 0.064 | 0.067 | 0.096 |
| Qwen3-30B-A3B, 16 | 1 | 0.488 | 0.494 | 0.504 |
| Qwen3-30B-A3B, 128 | 8 | 0.761 | 0.769 | 0.784 |
| Qwen3-30B-A3B, 1024 | 64 | 0.956 | 0.829 | 0.842 |
| Qwen3-30B-A3B, 4096 | 256 | 3.480 | 1.793 | 1.864 |

The 16-row tile wins below a few rows per expert, most at one token (Qwen3 0.096 to 0.064
ms, Mixtral 0.561 to 0.441), where the 128-row tile kept only 96 SMs busy on mostly padded
MMAs. It loses badly past a few dozen rows: at 256 rows an expert is 16 of its m-tiles per
weight strip, each on a 16 x 16 warp tile (two ldmatrix per two mma), and the layer runs at
half the 64-row tile's speed. The 64-row tile beat the 128-row one
at 256 rows per expert (3 to 4%) and lost at 1024 (3%), so the rule is 16 rows up to 24 per
expert, 64 up to 512, then 128.

## The grouped GEMM's own shapes

The rule changes what the plain `grouped_gemm` runs too, so `scripts/bench_grouped.py`
(the expert projections alone, its routing, 20 calls) ran again with the rule and with the
128-row tile forced, ms:

| shape | rows per expert | tile the rule picks | bf16, 128 rows | bf16, the rule | fp8, 128 rows | fp8, 64 rows |
|---|---|---|---|---|---|---|
| Mixtral, 512 tokens | 128 | 64 | 0.876 | 0.844 | 0.360 | 0.399 |
| Mixtral, 2048 | 512 | 64 | 2.212 | 2.227 | 0.865 | 1.005 |
| Mixtral, 8192 | 2048 | 128 | 8.251 | 8.216 | 3.224 | |
| Qwen3-30B-A3B, 512 | 32 | 64 | 0.2295 | 0.2205 | 0.1193 | 0.1171 |
| Qwen3-30B-A3B, 2048 | 128 | 64 | 0.3597 | 0.3418 | 0.1583 | 0.1693 |
| Qwen3-30B-A3B, 8192 | 512 | 64 | 0.9645 | 0.9396 | 0.3624 | 0.4363 |

In bf16 the 64-row tile is 3 to 5% faster on four of the five shapes it takes and 0.7%
slower on the fifth (Mixtral at 2048 tokens, the 512-row edge of the rule). In fp8 it lost:
7 to 20% slower on four of the five, 2% faster on Qwen3 at 512 tokens. The block-scaled fp8
mma runs at four times the bf16 rate, so the padded rows of a 128-row tile cost it a quarter
as much, and what the 64-row tile gives up (half the rows to share each weight stage, and a
second set of Stream-K tiles) is not paid back. So fp8 keeps the 128-row tile. The 16-row
tile was not tried in fp8. The tables in [grouped_gemm.md](grouped_gemm.md) are the 128-row
tile's.

## The fused SwiGLU

The same layer with the plain grouped GEMM on gate|up and the swiglu kernel after it
(`MoELayer(fuse_swiglu=False)`, the `ours_unfused` rows) is level or slower on every row:
0 to 1% up to 128 tokens, where the gate/up output is small (3% on Qwen3 at one token), 1.9
to 3.5% on Mixtral and 3.5 to 5.5% on Qwen3 from 1024 tokens up (6.394 against 6.769 ms at 16384 tokens, where the unfused path
writes and reads back 403 MB of gate|up).

## Not done

* The gather in the GEMM's A loads. The permute and the combine are at the copy rate but at
  16384 tokens on Qwen3 they are 754 us of 6.39 ms, 12%. vLLM's Triton kernel loads A rows
  through the sorted token ids instead; here the A operand comes in as one 2-D TMA box per
  stage, which cannot gather rows, so it would mean cp.async loads for A or a TMA box per row.
* fp8 weights. `grouped_gemm_swiglu` has the e4m3 form (tested), but the layer would need
  the activations quantized on the way into each GEMM, a scale per tensor computed on the
  device, and I did not build that.
* CUDA graphs. The grouped GEMM's Stream-K flags are tagged with an epoch the host passes
  per launch, so a captured graph would replay one epoch; as it stands the layer is not
  graph-safe, and the eager call is what is measured.
* The routing kernel at 16384 tokens is 44 us on Qwen3: 64 blocks of 256 tokens, each warp
  16 tokens in a row. The block count is capped so the last block's scan fits shared memory;
  more blocks would need the scan to change.
