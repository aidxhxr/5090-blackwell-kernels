# DeepSeek V4.1 Flash on the GB10's 128 GB: the footprint, the step and the fused expert kernel

What a decode step of the whole model costs on the DGX Spark once the routed experts are at
1 bit, and the kernel written for it: `w1moe_decode` (`src/kernels/w1moe_decode.cu`), the
six active experts of a token in one persistent launch with the SwiGLU intermediate in
shared memory. Bench: `bench_w1moe_decode` (one layer's 384 experts, random codes, every
variant against a CPU reference, the floors at 273 and 1,532 GB/s) and
`scripts/bench_deepseek_spark.py` (the footprint table and a 40-layer step through
`MoEBlock1Bit.decode`). Python: `sk.w1moe_decode(x, expert_ids, expert_weights, experts,
down)`, `MoEBlock1Bit.decode`, `MoEBlock1Bit.random_bits`. The format, the quantizer and the
sorted prefill path are in [w1gemm](w1gemm.md).

**Measured on the RTX 5090, not on the Spark.** This note, the kernel, the bench and the
script were written before the Spark arrived; on 2026-10-23 they ran on the 5090, the bench and
`tests/test_w1moe_decode.py` pass, and the 5090 columns below are measurements. The footprint
tables are byte counts, and every number at 273 GB/s is still a floor, not a time. The first
things to run on the Spark are at the end.

## The model, as assumed here

552B parameters, 40 layers, hidden 5120, 384 routed experts and one shared expert per layer,
each a SwiGLU MLP 5120 -> 2304 -> 5120 (35.39M weights), 6 routed experts per token, 64
query heads over one K/V head of dimension 512, vocabulary 129,280. The routed experts are
40 x 384 x 35.39M = 543.6B of the 552B. The shared experts are 1.42B, the embedding and the
head 0.66B each, and what is left, about 5.5B or 140M per layer, is the attention
projections, the norms and the routers. That 140M is a subtraction, not a config value; the
footprint tables carry it as one line, and the two numbers it feeds (the attention weights
at bf16 and at int4) move with it. The K/V cache is taken as K and V of 512 values each per
token per layer; a latent that serves as both would halve it.

## The footprint on 128 GB

Byte counts from `scripts/bench_deepseek_spark.py --footprint`, `MoEConfig.bytes_per_layer`
behind them. "g128" is the w1gemm / w4gemm format: one bf16 scale per 128 weights of a
column, 0.1406 bytes per weight at 1 bit, 0.2656 ternary, 0.5156 int4.

| part | bf16 | int4 g128 | ternary g128 | 1 bit g128 |
|---|---|---|---|---|
| routed experts, 40 layers | 1,087 GB | 280 GB | **144.4 GB** | **76.4 GB** |
| shared experts, 40 layers | 2.83 GB | 0.73 GB | 0.38 GB | 0.20 GB |
| attention, norms, routers (140M per layer, derived) | 11.2 GB | 2.9 GB | | |
| embedding + head | 2.65 GB | 0.68 GB | | |
| K/V cache, 32K context | 2.68 GB (e4m3: 1.34 GB) | | | |
| K/V cache, 128K context | 10.7 GB (e4m3: 5.4 GB) | | | |

Three sums:

| configuration | GB | headroom on 128 GB |
|---|---|---|
| 1-bit experts, 1-bit shared, int4 attention, bf16 embedding + head, no cache | 82.2 | 45.8 |
| the same with a 32K bf16 cache | 84.9 | 43.1 |
| the same with a 128K bf16 cache | 92.9 | 35.1 |
| the same with bf16 attention and bf16 shared experts, 128K bf16 cache | 103.9 | 24.1 |
| **ternary experts, anything else** | **144.4 + ...** | **does not fit** |

So at 1 bit the model fits with room for the whole 128K cache in bf16 and for the attention
weights in bf16 if that is what the quality numbers want. Ternary does not: the routed
experts alone are 144 GB. What does fit is a mix, ternary in some layers and 1 bit in the
rest: with about 110 GB for the experts (the remaining 18 GB for the attention, the
embeddings, a 32K cache and the OS, which shares this memory) the budget allows 19 ternary
layers and 21 at 1 bit. Which layers would take the ternary is a quality question for
`scripts/eval_ppl.py`, not a kernel question; the footprint table in the script prints
both widths so the answer can be read off.

The 5090 holds 15 to 17 layers at 1 bit (w1gemm.md), so the whole-model number is the
Spark's alone.

## The step: bytes and the floor at 273 GB/s

A batch-1 decode step reads, per layer, the six routed experts, the shared expert, the
router, the attention weights and the K/V cache so far, and once per token the embedding
row and the head. Bytes, with every weight at the width the row says:

| per layer | bytes |
|---|---|
| 6 routed experts, 1 bit (4.98 MB each: 4.42 MB of codes, 0.55 MB of scales) | 29.86 MB |
| shared expert, 1 bit | 4.98 MB |
| router, bf16 | 3.93 MB |
| attention + norms, int4 (bf16) | 72 MB (280 MB) |
| K/V at 32K, e4m3 (bf16) | 33.6 MB (67.1 MB) |
| K/V at 128K, e4m3 (bf16) | 134 MB (268 MB) |

| per token, 40 layers | GB | floor at 273 GB/s | tokens/s that floor allows |
|---|---|---|---|
| routed experts only | 1.194 | 4.38 ms | 229 |
| + shared, router, int4 attention, int4 head | 4.44 + 0.34 | 17.5 ms | 57 |
| + K/V at 32K e4m3 | 6.12 | 22.4 ms | 45 |
| + K/V at 32K bf16 | 7.46 | 27.3 ms | 37 |
| + K/V at 128K e4m3 | 10.15 | 37.2 ms | 27 |
| + K/V at 128K bf16 | 15.52 | 56.8 ms | 18 |

These are bounds, not claims: the floor is the bytes over the spec bandwidth with nothing
else in the way, and `bench_bandwidth` on the Spark will say what fraction of 273 GB/s a
copy actually gets (GB10.md guesses 75 to 90%). Two things the table says that are worth
having before anything runs. First, once the experts are at 1 bit they are no longer the
largest reader in a step: 1.2 GB against 2.9 GB of int4 attention weights (11 GB at bf16)
and 1.3 to 10.7 GB of cache. The expert kernel's job is to stay at the floor; the step's
tokens/s is decided by the attention weights' width and the cache's. Second, the expert
floor per layer is 109 µs here against 19.5 µs on the 5090 (w1gemm.md's floor column at
1,532 GB/s): the same launch overhead that eats a 20 µs floor is a tenth of a 109 µs one,
which is why a kernel that is launch-bound on the 5090 can be bandwidth-bound on the Spark.

## The fused kernel

`w1moe_decode_bf16` takes `x [T, K]`, the `[T, topk]` expert ids and routing weights, the
gate|up and down experts in the w1gemm format, and writes `y [T, K]`, the weighted sum over
each token's experts of `silu(x Wgate) * (x Wup) Wdown`, for `T <= 8`. The shared expert and
the routing are the caller's (`MoEBlock1Bit.decode`). Three variants:

| variant | idea |
|---|---|
| 0 | three passes: a gate|up GEMV per (token, slot) into global memory, a SwiGLU pass, a down GEMV that sums the slots. Scalar. The baseline |
| 1 | one persistent kernel. A work item is (expert, 128 columns of the intermediate). The block runs the gate and up GEMVs of its slice for every token routed to that expert, the SwiGLU in shared memory, then its 128 rows of the down weight over all 5120 outputs, added into an fp32 `y` with atomics |
| 2 | variant 1 with the routing weight folded into the down epilogue and the first groups of the next item's gate|up rows loaded into registers before the SwiGLU and the down pass |

**Why fuse.** Not for the intermediate's bytes. At batch 1 the gate|up output per expert is
9.2 KB and the SwiGLU result 4.6 KB; through DRAM and back that is 170 KB per layer against
a 30 MB weight stream, 0.6%. The reasons are the launches and the tails. The sorted path
is, per layer, a gather, two grouped GEMMs, a SwiGLU, a weighted scatter and the shared
expert's three ops: eight or nine launches, each with a ramp and a tail where the DRAM
queue drains, at 2 to 5 µs each on a 109 µs floor. The fused kernel is one launch (plus a
memset and a bf16 convert), and the down pass of one item overlaps the gate|up stream of
another, so the DRAM sees one continuous read of 30 MB. The second reason is reuse: a work
item is an expert, not a (token, expert) pair, so at T = 8 an expert two tokens share is
streamed once (`w1gemm_moe` does that too, through the sort).

**Why CUDA cores.** At `T <= 8` every weight byte is used T times and never again, and a
sign bit needs no multiply: the dequantized weight times the activation is the activation
with its sign flipped or not. The inner loop is one bit test and T predicated fp32 adds per
weight, the sign applied as `2 * sum over set bits - sum over the group` with the group sums
of x computed once per token when the activations are staged in shared memory as fp32
`[k][T]`, so one 16-byte read feeds four tokens; a group's partial sum is then multiplied by
its scale once. About `1 + 1.25 T` lane instructions per weight at 1 bit. The budget: at the
DRAM floor an SM of the GB10 receives 273 GB/s / 48 = 5.7 GB/s = 45 G weights/s at 1 bit and
issues about 4 x 32 x 2.4 G = 307 G lane instructions/s, 6.7 per weight. T = 1 uses a third
of that, T = 4 all of it, T = 8 would need 1.6x more issue than there is, on paper. On the
5090 (1,532 GB/s over 170 SMs, 4.3 instructions per weight) the line is at T = 2. Ternary
doubles both the bytes and the instructions per weight, so the lines do not move. Inside
those lines the tensor-core strip of `w1gemm` would add fragment plumbing for weights used
once; past them it is the next rung (w1gemm's `strip_chunk` on the staged activations, the
same work decomposition). Nsight will say whether the predicated adds are what the compiler
emits.

**The active set and L2.** One layer's active set at batch 1 is 29.9 MB of routed experts
plus 5 MB of shared expert plus the router: about 35 MB, against a 24 MB L2 (96 MB on the
5090). Nothing in it is reused across layers, since every layer has its own experts, and
within a layer the fused kernel reads each byte once, so L2 residency is not something the
kernel relies on. Where L2 matters is inside the kernel: the down weight is read as 16-byte
pieces of 288-byte rows, half of a 32-byte sector per block, the other half belonging to the
neighbouring slice's block; the slices of one expert are consecutive items that run at the
same time, so the second read of a sector should be an L2 hit, and if it is not the down
pass costs twice its bytes. `--repeat-routing` in the bench launches the same six experts
twice back to back: on the 5090 the second launch should run from L2 entirely (31 MB in
96 MB), on the Spark it should not (31 MB in 24 MB, streamed in order, so the lines that
survive are the last 24 MB and the kernel asks for the first ones first). The gap between
the two machines' repeat rows is the L2 effect, and the gap between a repeat row and a cold
row on the Spark is how much a 24 MB L2 helps a 31 MB working set, which should be close to
nothing. Across layers the question is whether anything else fits: the activations (10 KB),
the router (4 MB per layer, read once), the attention weights of the next layer (72 MB at
int4) do not. On this machine the L2 is a bandwidth amplifier for the kernel's own
re-reads (the down sectors, the activation stage), not a cache for the model.

**Shared memory.** 44 KB static per block: the 32 KB activation stage (fp32, 8192 values:
the whole K at T = 1, a 1024-k chunk at T = 8), the 128 x 8 intermediate, the up values
handed from the up threads to the gate threads, and the per-group sums. Two blocks per SM
by `__launch_bounds__`, 256 threads each; the grid is the resident block count from the
occupancy API times the SM count (`cudaDevAttrMultiProcessorCount`, 48 here, 170 on the
5090), and a block loops over items. At T = 1 there are 6 x 18 = 108 items for 96 resident
blocks on the Spark, which is one uneven wave; smaller slices (64 columns, 216 items) are
the first knob if the tail shows.

## What the 5090 said, and variant 3

Two things, one of them not in the kernel.

**The scratch allocation.** The launcher takes the fp32 `y` it adds into from a
stream-ordered allocation. The device's default pool hands its memory back to the driver
whenever the device synchronizes (release threshold 0), so after every sync the next
`cudaMallocAsync` maps pages again: Nsight Systems put it at 68 us of host time per call, in
front of a 57 us kernel, and the bench's events counted both, 130 us. The scratch now comes
from a pool of the file's own whose release threshold is "never", so the allocation is a pool
lookup. The occupancy query of the persistent grid is cached for the same reason.

**The kernel.** Variant 2 alone is 57 us at T = 1 on the 5090 against 19.5 us of bytes. Nsight
Compute: 108 blocks (6 experts x 18 slices) of 8 warps on 170 SMs, so 62 SMs idle and 2 warps
per scheduler on the rest; one instruction issued every 10 cycles per warp, DRAM at 27% of
peak, and the long-scoreboard stall half the samples. 12.6 million warp instructions for 212
million weights is the 1.9 lane instructions per weight the "why CUDA cores" paragraph
counted, so the count was right; what it left out is that each thread runs a whole 5,120-k
row as one dependent chain of about 15,000 of them. The instruction budget only holds when
something else hides each chain's latency, and 8 warps per SM do not.

Variant 3 keeps the work item and moves it to the tensor cores with 16 warps a block, on
`w1gemm` variant 3's lane-word body (one word of the row per lane, an 8-deep ring of words, a
4-deep ring of activations; [w1gemm](w1gemm.md)):

| phase | variant 2 | variant 3 |
|---|---|---|
| gate and up | 256 threads, each a whole row on the CUDA cores, 4 groups in flight per thread | 16 warps, each a 16-column strip (8 gate, 8 up) over the whole K, `mma.sync` with the item's tokens as the 8-column B tile |
| SwiGLU | gate threads read the up threads' values from shared memory | the same from the strips' accumulators, h stored as the bf16 [token][128] tile |
| down | thread j owns outputs j, j + 256, ...; 16 bytes of each row | 320 strips of 16 outputs, one 128-k group each, 20 per warp with the next 8 in flight, h from shared memory as B |
| y | fp32 atomics, routing weight in the epilogue | the same |

At T = 1 a tensor-core step does the same work per weight as the CUDA-core loop (one m16n8k16
for 256 weights with 7 of its 8 token columns empty, about 17 instructions counted from the
source), so the gain is
not fewer instructions but 16 independent strips per block where variant 2 had 256 chains
of 15,000 instructions on 8 warps; at T = 8 the mma does 8 tokens for the same instructions
where the CUDA cores do 8 times the adds.

## Results

`bench_w1moe_decode --chain=20` on the RTX 5090, 2026-10-23 (rows in
`results/w1moe_decode.json`, with the `--repeat-routing` rows). Every variant checks against
the CPU reference; `tests/test_w1moe_decode.py` passes. Consecutive calls take consecutive
routings, so each meets its experts cold. GB/s is the active bytes (the distinct experts'
codes and scales, x and y) over the time. Single calls first (the JSON rows), then 20 calls
back to back, us per call; the repeat column is the same six experts twice in a row, the
second call meeting what the first left in the 96 MB L2. The Spark's columns are unmeasured.

| T | bits | active MB (experts) | floor at 273 GB/s | floor at 1,532 GB/s | v0 | v1 | v2 | v3 | v3, 20 back to back | GB/s | repeat, v1 / v3 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 29.9 (6) | 109.5 | 19.5 | 543 | 61.2 | 62.2 | **49.2** | **42.9** | 697 | 37.2 / 38.9 |
| 4 | 1 | 117 (23.6) | 430 | 76.6 | 863 | 165 | 164 | **142** | **139** | 843 | 143 / 120 |
| 8 | 1 | 227 (45.6) | 832 | 148 | 1,728 | 293 | 290 | **243** | **246** | 924 | 296 / 252 |
| 1 | 2 | 56.4 (6) | 207 | 36.8 | 591 | 109 | 110 | **87.8** | **83.0** | 680 | |
| 4 | 2 | 222 (23.6) | 812 | 145 | 904 | 377 | 376 | **303** | **298** | 743 | |
| 8 | 2 | 429 (45.6) | 1,572 | 280 | 1,916 | 707 | 700 | **545** | **543** | 790 | |

Variant 1 and 2's single calls were 130 us at T = 1 before the scratch fix; the kernel did not
change. Variant 3 is 1.15 to 1.4x variant 2 at every T and both widths, and still 2.2x the
copy-roof floor at T = 1 (1.7x at T = 8, 1.9x at ternary T = 8). The repeat column says why:
with all six experts in L2, T = 1 takes 37 to 39 us on both variants, so the T = 1 step is not
waiting on DRAM but on its own latency, 108 items on 170 SMs with the same instruction
latency per group as `w1gemm` variant 3 (Nsight on variant 3 at T = 1: 51 us under locked
clocks, DRAM at 33% of peak, 16 warps per SM capped by 121 registers, 3.7 of 9.3 cycles per
instruction waiting on L1TEX). At ternary variant 3 has a 32-byte spill (`cuobjdump`);
variants 1 and 2 at ternary have 320 and 336 bytes of stack at their 128-register cap, the
register pressure the earlier round flagged (at 1 bit neither spills).

On the 5090 the fused step is level with the sorted path's GEMMs alone: the two grouped
`w1gemm_moe` launches of one layer at T = 1 take 27.8 and 17.3 us back to back
([w1gemm](w1gemm.md)), 45 us, against 42.9 for the fused kernel, which also does the SwiGLU,
the routing weights and the sum. Against the 109.5 us floor at the Spark's 273 GB/s, the
5090's 42.9 us says the kernel's own latency is well under the Spark's bytes, which is the
case this kernel was written for; whether it streams at 273 GB/s there is the Spark's number.

`scripts/bench_deepseek_spark.py` on the 5090: 14.3 GB free holds one layer next to the
12 GB reserve, so one random layer is resident and re-used 40 times with a different
activation (so a different routing) per layer; the step is `MoEBlock1Bit.decode` per layer:
the torch router, the fused kernel (variant 3) and the shared expert on `w1gemm`. `--check`
at T = 4: max abs error 1.6e-2 against a max |y| of 3.0.

| T | step ms | GB per step | GB/s | tokens/s (experts only) | floor ms at 273 GB/s | floor ms at 1,532 GB/s |
|---|---|---|---|---|---|---|
| 1 | 5.54 | 1.39 | 252 | 181 | 5.10 | 0.91 |
| 4 | 7.40 | 4.48 | 605 | 541 | 16.4 | 2.92 |
| 8 | 10.18 | 7.54 | 741 | 786 | 27.6 | 4.92 |

At T = 1 a layer is 138 us, of which the fused kernel is 43: the rest is the router and the
shared expert as a handful of torch and `w1gemm` launches, about 95 us a layer of host-bound
work, which is what a CUDA graph of the layer is for. Not a Spark number either way: the
whole model is the Spark's, and so is its 273 GB/s.

`scripts/bench_deepseek_flash.py` (the sorted `forward` path, two `w1gemm_moe` launches and
the torch gather, SwiGLU and scatter around them): 0.31 ms a layer at T = 1, 0.33 at 4, 0.58 at
16 and 1.44 at 64, with `--check` at T = 8 at 7.8e-3 against a max |y| of 3.0. Also mostly
launches at T = 1: the two grouped GEMMs inside it are 45 us.

## What to run first

Done on the 5090: the bench with checks, `tests/test_w1moe_decode.py`, the repeat rows, Nsight
on variant 3 at T = 1, and `bench_deepseek_spark.py` with one layer resident. On the Spark:

1. `bench_w1moe_decode` and `pytest tests/test_w1moe_decode.py`, the first time on sm_121,
   then with `--repeat-routing`, and `bench_bandwidth` next to it for the real roof. The v3
   row at T = 1 against the 109 us floor is the number this note is about; the v0 row says
   what three launches cost there.
2. `bench_deepseek_spark.py` with all 40 layers resident: the step time against the 4.4 ms
   expert floor, and `--check` once.
3. Nsight on the T = 1 and T = 8 rows: whether the 16 warps a block that are short on the
   5090 are enough at a sixth of its bandwidth per SM.
4. The real checkpoint through `deepseek_flash.load_moe_layer` for every layer (the loader
   has never met the checkpoint, its key names are an assumption), and `scripts/eval_ppl.py`
   at 1 bit against the MXFP4 it ships in. Nothing above matters if that number is bad.
