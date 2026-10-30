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

or, where the rule in "The gather" below says so, the permute replaced by the routing's
inverse permutation and the gate/up GEMM reading x through it:

```
ids, w, offsets, pos, rows = moe_route_rows(logits, k)   route_kernel + positions_kernel
h = grouped_gemm_swiglu(x, w13, offsets, rows=rows)      the A tile gathered in the loads
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

## The gather

vLLM's Triton kernel loads its A rows through the sorted token ids and has no permute; ours
read A as one TMA box per stage, which cannot gather. `grouped_gemm_bf16` now takes a row
table: `positions_kernel` writes the inverse permutation as it finishes the positions
(`rows[pos[t, s]] = t`, `moe_route_rows`), and in the GATHER instantiation of the kernel
the producer warp loads the A tile with cp.async instead of a box: 8 lanes take the 8
16-byte chunks of a row's 128-byte segment (one request per row), each lane holds the
table entries of its rows for the piece, the copies land in the same swizzled layout the box
would (`fp8gemm_tile::swz`, row & 7), and each lane's copies arrive on the stage's barrier
through `cp.async.mbarrier.arrive.noinc`, the barrier initialized for 33 arrivals: the 32
lanes and lane 0's expect_tx for the weight box, which stays TMA. A row past the table (the
padded rows of a tile) is zero-filled. The stage holds the same bytes and the mma order does
not change, so the gathered GEMM gives the TMA path's bits (the tests check it on every
SwiGLU shape); the layer's outputs are the same on either path.

What it costs and what it saves, the gate/up GEMM alone (its [T * k, D] rows at the layer's
tile height), ms, median of 30 calls: the TMA form on the permuted copy, the gathered form
on the tokens, and the copy itself (`index_select`, what the permute kernel moves):

| shape (E, K, 2I) | rows | rows per expert | TMA GEMM | gathered GEMM | permute | gathered / (TMA + permute) |
|---|---|---|---|---|---|---|
| Mixtral 8, 4096, 28672 | 128 | 16 | 1.117 | 1.117 | 0.015 | 0.99 |
| Mixtral | 2048 | 256 | 2.372 | 2.573 | 0.017 | 1.08 |
| Mixtral | 16384 | 2048 | 16.33 | 16.98 | 0.093 | 1.03 |
| Qwen3 128, 2048, 1536 | 128 | 1 | 0.297 | 0.296 | 0.015 | 0.95 |
| Qwen3 | 2048 | 16 | 0.488 | 0.485 | 0.015 | 0.96 |
| Qwen3 | 16384 | 128 | 0.592 | 0.610 | 0.026 | 0.99 |
| 8, 4096, 1536 | 2048 | 256 | 0.125 | 0.127 | 0.027 | 0.84 |
| 8, 4096, 1536 | 16384 | 2048 | 0.835 | 0.855 | 0.092 | 0.92 |
| 8, 4096, 4096 | 2048 | 256 | 0.331 | 0.338 | 0.030 | 0.93 |
| 8, 4096, 4096 | 16384 | 2048 | 2.268 | 2.366 | 0.092 | 1.00 |
| 8, 4096, 8192 | 2048 | 256 | 0.704 | 0.717 | 0.027 | 0.98 |
| 8, 4096, 8192 | 16384 | 2048 | 4.574 | 4.778 | 0.092 | 1.02 |
| 8, 4096, 14336 | 16384 | 2048 | 8.112 | 8.379 | 0.092 | 1.02 |
| 8, 2048, 28672 | 16384 | 2048 | 8.232 | 8.526 | 0.035 | 1.03 |
| 128, 2048, 4096 | 2048 | 16 | 1.285 | 1.281 | 0.027 | 0.98 |
| 128, 2048, 4096 | 16384 | 128 | 1.555 | 1.588 | 0.047 | 0.99 |

The gathered GEMM is level with the TMA one on the 16-row tile and 1.5 to 4% slower on the
64- and 128-row tiles (8.5% on Mixtral at 256 rows per expert): a row is gathered again for
every n-tile of its expert, 224 of them on Mixtral's 28672 gate|up columns against 12 on
Qwen3's 1536, and 32 cp.async per lane per stage from one warp do not keep up with a box
where the mainloop is at its compute peak. The permute is paid once per row. So the gather
wins where the permute is a large share of the GEMM: narrow experts, or few rows. The rule
in `MoELayer.gathers` is the two lines the table draws: gather at every token count when
the experts have up to 2048 inter columns (2I <= 4096: Qwen3, DeepSeek V3's 2048), and at
any width while the GEMM runs its 16-row tile, up to 24 rows per expert on average (Mixtral
up to 96 tokens). `MoELayer(gather=True/False)` forces one; the bench's `ours_gather` and
`ours_permute` rows are the two forced on every row.

## CUDA graphs

Nothing in the forward waits on the host, and no launch takes a shape from the routing:
the grouped GEMM's grid is its resident blocks and its tile list is built on the device from
the offsets, the routing's grid comes from T, the combine's from T. Two things still kept it
out of a graph. The grouped GEMM tagged its Stream-K flags with an epoch the host bumped per
launch and passed as a kernel parameter, which a replay would run again: the flag now counts
k-steps and the piece that finishes a tile (the last to read its flag) writes 0 back, so a
launch leaves the flags as it found them and its parameters never change. And the routing's
per-block counts lived in a workspace that grew by free and reallocate, so a graph captured
at a decode-sized T pointed at memory a later prefill freed: it is now one allocation of
the largest size (128 blocks x 512 experts, 256 KB). `MoEGraph(layer, T)` runs the layer
twice on a side stream (every workspace exists before the capture, the way the engine's
`_capture_all` warms its buckets) and captures the forward on a static input; the test
captures at 1, 16, 64 and 300 tokens, on both paths, and checks each replay against the
eager bits with a 4096-token eager call in between.

What the replay buys at the layer level, Qwen3-30B-A3B, eager against the replay with the
input copy inside both, the host's issue time per call measured without a sync:

| tokens | eager, ms | host issue, eager | replay, ms | host issue, replay |
|---|---|---|---|---|
| 1 | 0.0652 | 0.029 | 0.0652 | 0.005 |
| 4 | 0.1949 | 0.052 | 0.1942 | 0.005 |
| 16 | 0.4862 | 0.173 | 0.4852 | 0.005 |
| 64 | 0.7305 | 0.259 | 0.7305 | 0.005 |

The replay takes 24 to 254 us of host work per call off the step, but the layer alone is
GPU-bound at every T here (65 us of kernels at one token against 29 us of issue), so its
time does not move; the same check on a loaded host (another tenant's engine busy) gave 82
to 67 us at one token, 1.22x. Mixtral is 0.44 ms at one token and does not move either.
The gain is a model's: a 48-layer Qwen3 decode step issues 48 times these launches plus
attention, and that is where the engine's graph buckets (GRAPH_MAX = 64) earn their keep
on the dense model. Wiring an MoE model into the engine would be: an MoE block in
`layer.py` on `MoELayer`, its interleaved w13 built at load, `warm_workspaces` running the
layer at the largest bucket first (the routing's workspace is fixed now, the grouped GEMM's
Stream-K slots are sized per tile configuration on first use), then `_capture_all` as it
is. The engine has no MoE checkpoint today, so this is not done.

## Results

Measured on the RTX 5090 on 2026-10-30 (every row; the first round was 2026-10-23): driver
595.58, CUDA 13.2, torch 2.14+cu130 for ours and the torch layers; vLLM 0.30.0 from its own
environment (torch 2.13, Triton 3.7.1) with its default `fused_moe` config, since vLLM ships
no tuned file for these shapes on this card (it says so in a warning). Other tenants held
about 17 GB of the card and the runs held a lock the other benches on the card respect.
`scripts/bench_moe_layer.py`: median of 50 calls, CUDA events around each call and
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
| Mixtral 8x7B, 1 | 0.4405 | 0.4397 | 0.573 | 0.589 | 1.00x | 1.34x |
| Mixtral 8x7B, 16 | 1.715 | 1.725 | 2.018 | 2.371 | 1.01x | 1.38x |
| Mixtral 8x7B, 128 | 1.749 | 1.766 | 2.054 | 2.380 | 1.01x | 1.36x |
| Mixtral 8x7B, 1024 | 3.845 | 4.413 | 4.359 | 4.497 | 1.15x | 1.17x |
| Mixtral 8x7B, 4096 | 12.94 | 14.14 | 15.02 | 14.79 | 1.09x | 1.14x |
| Mixtral 8x7B, 16384 | 48.87 | 53.61 | 57.17 | 55.37 | 1.10x | 1.13x |
| Qwen3-30B-A3B, 1 | 0.0631 | 0.1989 | 0.766 | 1.250 | 3.15x | 19.8x |
| Qwen3-30B-A3B, 16 | 0.4755 | 0.4744 | 1.662 | 11.10 | 1.00x | 23.3x |
| Qwen3-30B-A3B, 128 | 0.7437 | 0.7585 | 2.726 | 17.57 | 1.02x | 23.6x |
| Qwen3-30B-A3B, 1024 | 0.8016 | 0.8862 | 2.498 | 18.57 | 1.11x | 23.2x |
| Qwen3-30B-A3B, 4096 | 1.739 | 1.971 | 4.115 | 18.24 | 1.13x | 10.5x |
| Qwen3-30B-A3B, 16384 | 6.137 | 6.907 | 14.26 | 18.82 | 1.13x | 3.07x |

The two paths and the graph, ms (`ours_gather`, `ours_permute`, `ours_graph`; the graph
rows stop at 1024 tokens, a graph being for decode-sized steps, and include the copy of the
input into the graph's static buffer):

| config, tokens | rows per expert | the rule picks | gathered | permuted copy | graph replay |
|---|---|---|---|---|---|
| Mixtral 8x7B, 1 | 1 | gather | 0.4398 | 0.4397 | 0.4420 |
| Mixtral 8x7B, 16 | 4 | gather | 1.715 | 1.717 | 1.717 |
| Mixtral 8x7B, 128 | 32 | copy | 1.748 | 1.749 | 1.754 |
| Mixtral 8x7B, 1024 | 256 | copy | 3.901 | 3.854 | 3.854 |
| Mixtral 8x7B, 4096 | 1024 | copy | 13.18 | 12.95 | |
| Mixtral 8x7B, 16384 | 4096 | copy | 49.61 | 48.89 | |
| Qwen3-30B-A3B, 1 | 1 | gather | 0.0627 | 0.0642 | 0.0648 |
| Qwen3-30B-A3B, 16 | 1 | gather | 0.4755 | 0.4799 | 0.4774 |
| Qwen3-30B-A3B, 128 | 8 | gather | 0.7438 | 0.7502 | 0.7461 |
| Qwen3-30B-A3B, 1024 | 64 | gather | 0.8035 | 0.8219 | 0.8029 |
| Qwen3-30B-A3B, 4096 | 256 | gather | 1.745 | 1.795 | |
| Qwen3-30B-A3B, 16384 | 1024 | gather | 6.145 | 6.385 | |

What the call has to move and compute. The experts touched are counted on the first input
of the pool; the GB/s column is those experts' bf16 weights over the call's time, which is
the floor up to 128 tokens, and TFLOPS counts the 6 D I FLOPs of each of the T * k rows.

| config, tokens | experts touched | their weights | ours GB/s | vLLM GB/s | ours TFLOPS | vLLM TFLOPS |
|---|---|---|---|---|---|---|
| Mixtral 8x7B, 1 | 2 | 0.70 GB | 1,600 | 1,603 | 1.6 | 1.6 |
| Mixtral 8x7B, 16 | 8 | 2.82 GB | 1,644 | 1,634 | 6.6 | 6.5 |
| Mixtral 8x7B, 128 | 8 | 2.82 GB | 1,611 | 1,596 | 51.6 | 51.1 |
| Mixtral 8x7B, 1024 | 8 | 2.82 GB | 733 | 639 | 187.7 | 163.5 |
| Mixtral 8x7B, 4096 | 8 | 2.82 GB | 218 | 199 | 223.0 | 204.1 |
| Mixtral 8x7B, 16384 | 8 | 2.82 GB | 58 | 53 | 236.2 | 215.3 |
| Qwen3-30B-A3B, 1 | 8 | 0.08 GB | 1,196 | 380 | 1.2 | 0.4 |
| Qwen3-30B-A3B, 16 | 86 | 0.81 GB | 1,707 | 1,711 | 2.5 | 2.5 |
| Qwen3-30B-A3B, 128 | 128 | 1.21 GB | 1,624 | 1,593 | 13.0 | 12.7 |
| Qwen3-30B-A3B, 1024 | 128 | 1.21 GB | 1,507 | 1,363 | 96.4 | 87.2 |
| Qwen3-30B-A3B, 4096 | 128 | 1.21 GB | 695 | 613 | 177.9 | 156.9 |
| Qwen3-30B-A3B, 16384 | 128 | 1.21 GB | 197 | 175 | 201.6 | 179.1 |

Where our call goes, kernel microseconds per call from the torch profiler (10 calls):

| config, tokens | router GEMM | route | permute, or positions | gate/up + SwiGLU | down | combine | kernels | call |
|---|---|---|---|---|---|---|---|---|
| Mixtral 8x7B, 1 | 3.5 | 3.2 | 1.6 | 282 | 147 | 2.1 | 439 | 441 |
| Mixtral 8x7B, 16 | 24.2 | 2.8 | 1.4 | 1,108 | 575 | 2.0 | 1,713 | 1,715 |
| Mixtral 8x7B, 128 | 4.5 | 3.8 | 3.2 | 1,130 | 603 | 2.2 | 1,746 | 1,749 |
| Mixtral 8x7B, 1024 | 8.4 | 4.3 | 7.7 | 2,451 | 1,351 | 10.5 | 3,833 | 3,845 |
| Mixtral 8x7B, 4096 | 26.1 | 5.6 | 29.0 | 8,514 | 4,282 | 39.4 | 12,896 | 12,942 |
| Mixtral 8x7B, 16384 | 87.3 | 11.4 | 269 | 31,920 | 16,365 | 217 | 48,870 | 48,869 |
| Qwen3-30B-A3B, 1 | 2.5 | 3.4 | 1.0 | 30.6 | 17.4 | 2.0 | 57.0 | 63.1 |
| Qwen3-30B-A3B, 16 | 4.3 | 4.3 | 1.4 | 307 | 156 | 3.4 | 476 | 475 |
| Qwen3-30B-A3B, 128 | 4.5 | 5.5 | 1.4 | 481 | 245 | 4.1 | 741 | 744 |
| Qwen3-30B-A3B, 1024 | 8.0 | 7.0 | 1.5 | 511 | 263 | 10.4 | 801 | 802 |
| Qwen3-30B-A3B, 4096 | 22.9 | 14.6 | 1.5 | 1,080 | 540 | 74.9 | 1,734 | 1,739 |
| Qwen3-30B-A3B, 16384 | 54.4 | 44.3 | 2.4 | 3,784 | 1,869 | 353 | 6,107 | 6,137 |

What the numbers say:

* Up to 128 tokens the layer is a weight read and both we and vLLM are at it: 1.59 to 1.71
  TB/s on the weights the routing touched, above the 1,532 GB/s `cudaMemcpy` does (a copy
  writes as much as it reads; this only reads). The two are within about 1% of each other on
  all of those rows: vLLM is ahead by 0.2% on two of them (Mixtral at 1 token, Qwen3 at 16),
  we are ahead by 0.6 to 2% on the other four. On Qwen3 at 16 tokens our two GEMMs read
  their 0.81 GB at 1.75 TB/s and the 13 us of small kernels around them (router 4.3, route
  4.3, positions 1.4, combine 3.4) is the difference; the permute used to be 4.3 of it, and
  with the gather the row table costs 1.4.
* Qwen3 at one token: 0.063 ms against vLLM's 0.199. Our kernels are 57 us, the two GEMMs 48
  us of it for 75 MB. vLLM's call takes longer than its weights do to read; I did not profile
  where its time goes.
* From 1024 tokens up the layer is compute and our GEMMs are faster than vLLM's Triton
  kernels at their default config: 1.09x to 1.15x on Mixtral (236 against 215 TFLOPS at
  16384 tokens; the grouped GEMM alone did 240 at 8192) and 1.11x to 1.13x on Qwen3, where
  the gather took the permute's 14 to 402 us out of the call (0.820 to 0.802 ms at 1024
  tokens, 6.394 to 6.137 at 16384 against the first round).
* Against the transformers loop: 20x to 24x on Qwen3 up to 1024 tokens, where the loop pays
  a `torch.where` and a sync for each of up to 128 experts, 10x at 4096 and 3.1x at 16384;
  1.13x to 1.38x on Mixtral, which has 8 experts. `torch._grouped_mm` runs on sm_120 in torch
  2.14 but is 2.3x to 3.7x slower than ours on Qwen3 from 16 tokens up (12x at one), and
  slower than the loop on Mixtral from 4096 tokens up; I did not look at what it dispatches
  to.
* The router GEMM on Mixtral at 16 tokens is 24 us, against 4.5 at 128: cuBLAS through
  `F.linear` on [16, 4096] x [4096, 8]. Every implementation makes the same call, so it is in
  every row; I did not chase it.
* The graph replay is within 0.5% of the eager call on every row up to 1024 tokens, the
  input copy included: the host is never the bottleneck of one layer here ("CUDA graphs"
  above has the issue times).

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
0 to 1% up to 128 tokens, where the gate/up output is small (11% on Qwen3 at one token),
2.1 to 3.7% on Mixtral and 3.7 to 5.4% on Qwen3 from 1024 tokens up (6.137 against 6.468
ms at 16384 tokens, where the unfused path writes and reads back 403 MB of gate|up).

## Not done

* The combine fused into the down GEMM's epilogue. A token's k rows land in one tile only
  when they fall in the same expert, which they never do, so the epilogue would have to
  scatter-add across tiles in fp32 and lose the fixed-order sum; the combine stays a kernel
  (352 us of 6.1 ms at 16384 tokens on Qwen3, 5.8%).
* The gathered GEMM on the wide tiles. The loss is the producer warp issuing 32 cp.async per
  lane per stage; spreading the gather over the consumer warps, or a TMA box per row
  (a 1-row box per table entry, 128 per stage), was not tried.
* fp8 weights. `grouped_gemm_swiglu` has the e4m3 form (tested), but the layer would need
  the activations quantized on the way into each GEMM, a scale per tensor computed on the
  device, and I did not build that; the gather is bf16 only.
* The routing kernel at 16384 tokens is 44 us on Qwen3: 64 blocks of 256 tokens, each warp
  16 tokens in a row. The block count is capped so the last block's scan fits shared memory;
  more blocks would need the scan to change.
