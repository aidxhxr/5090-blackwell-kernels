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

**Not measured.** This note, the kernel, the bench and the script were written on a machine
without a GPU or nvcc, before the Spark arrived. Every number below is a byte count from the
model's shapes or a floor from a bandwidth figure, and none of them is a time. The 5090 can
run everything here except the whole-model footprint, so `bench_w1moe_decode` runs there
first; the first thing to run on the Spark is at the end.

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

## Results

`bench_w1moe_decode`, every variant, median of 20 launches on rotating routings (cold L2),
plus the `--repeat-routing` rows. GB/s is the active bytes (the distinct experts' codes and
scales, x, y) over the time. The 5090 columns fill first; the Spark's when it arrives.

| T | bits | active MB | floor µs at 273 GB/s | floor µs at 1,532 GB/s | v0 µs | v1 µs | v2 µs | GB/s | repeat v2 µs |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 1 | 29.9 | 109 | 19.5 | not measured | not measured | not measured | not measured | not measured |
| 4 | 1 | about 116 (23 distinct experts) | 425 | 76 | not measured | not measured | not measured | not measured | not measured |
| 8 | 1 | about 225 (45 distinct experts) | 824 | 147 | not measured | not measured | not measured | not measured | not measured |
| 1 | 2 | 56.4 | 207 | 36.8 | not measured | not measured | not measured | not measured | not measured |
| 4 | 2 | about 219 | 802 | 143 | not measured | not measured | not measured | not measured | not measured |
| 8 | 2 | about 425 | 1,557 | 277 | not measured | not measured | not measured | not measured | not measured |

The distinct-expert counts at T = 4 and 8 are the expectation for 6 draws per token out of
384 (the bench prints the actual ones). `scripts/bench_deepseek_spark.py`, 40 layers
resident, T = 1, 4, 8:

| T | step ms | GB per step | GB/s | tokens/s (experts only) | floor ms at 273 GB/s |
|---|---|---|---|---|---|
| 1 | not measured | 1.39 | not measured | not measured | 5.1 |
| 4 | not measured | about 4.8 | not measured | not measured | about 18 |
| 8 | not measured | about 9.2 | not measured | not measured | about 34 |

(1.39 GB per step at T = 1 is the routed experts plus the shared expert, 34.8 MB per layer.)

## What to run first

1. `bench_w1moe_decode --bits=1 --t=1` on the 5090, checks on. The CPU reference takes a few
   seconds per T; a mismatch in variant 1 or 2 with variant 0 passing points at the item
   ownership scan or the slice bookkeeping, a mismatch in all three at the bit order.
2. `pytest tests/test_w1moe_decode.py`: decode against the sorted path and against the
   dense reference at T = 1, 3, 8, both widths.
3. The same bench on the Spark, then with `--repeat-routing`, and `bench_bandwidth` next
   to it for the real roof. The v2 row at T = 1 against the 109 µs floor is the number this
   note is about; the v0 row says what three launches cost here.
4. `bench_deepseek_spark.py` with all 40 layers resident: the step time against the 4.4 ms
   expert floor, and `--check` once.
5. Nsight on the T = 1 and T = 8 rows: issue slot utilization and `dram__throughput`
   decide whether the CUDA-core argument above held and where T = 8 needs the mma rung.
6. The real checkpoint through `deepseek_flash.load_moe_layer` for every layer (the loader
   has never met the checkpoint, its key names are an assumption), and `scripts/eval_ppl.py`
   at 1 bit against the MXFP4 it ships in. Nothing above matters if that number is bad.
