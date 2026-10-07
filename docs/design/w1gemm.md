# W1A16: 1-bit and ternary weights, bf16 activations

`C[M,N] = A[M,K] · dequant(W)[K,N]`, row-major, bf16 activations and output, weights at one
bit (sign) or two bits (ternary) with one bf16 scale per 128 consecutive k of a column, fp32
accumulation. Source: `src/kernels/w1gemm.cu` (the three rungs, the launcher, and the grouped
MoE form `w1gemm_moe`). Bench: `bench_w1gemm` (checks every variant against cuBLAS on the
dequantized weights and times the bf16 `hgemm` and the int4 `w4gemm` it replaces). Python:
`sk.W1Weight.quantize(w, bits=1)`, `sk.w1gemm(a, w1)`, `sk.W1ExpertWeights`,
`sk.w1gemm_moe(a_sorted, experts, offsets)`, `reference.w1_quantize`,
`reference.w1_dequantize`, and the MoE block in `python/spark_kernels/moe.py`
(`MoEConfig`, `DEEPSEEK_V41_FLASH`, `route`, `sort_by_expert`, `MoEBlock1Bit`) with the
checkpoint loader in `python/spark_kernels/deepseek_flash.py`.

**Not measured yet.** Everything in this note was written on a machine without a GPU or
nvcc. The kernels have not been compiled, `tests/test_w1gemm.py` has not run, and
`bench_w1gemm` has no rows. The numbers below are byte counts from the model's shapes and
floors from the measured copy roof; none of them is a time. `bench_w1gemm` and the tests are
the first things to run on the card.

## Why

[w4gemm](w4gemm.md) made the point for a dense model: at batch 1 a decode step is a weight
stream, the bf16 kernel already reads at the copy roof, and the only thing left is fewer
bytes per weight. A mixture of experts makes the same point harder. DeepSeek V4.1 Flash has
552B parameters, 40 layers, hidden 5120, 384 routed experts and one shared expert per layer
with a SwiGLU intermediate of 2304, 6 routed experts active per token, 64 query heads over 1
K/V head of dimension 512, a vocabulary of 129,280. A decode step at batch 1 touches the
6 chosen experts (plus the shared one) per layer, and nothing else of the 384. So the bytes
per step are the bytes of 6 x 3 matrices of 5120 x 2304, 40 times: 212 million weights per
layer, 8.5 billion per step. At bf16 that is 17 GB per token, 11 ms at the copy roof, before
the attention weights and the cache; at 1 bit plus scales it is 1.2 GB and 0.8 ms. The
whole model's routed experts do not fit one 32 GB card in any of these formats (the
footprint table below), so the question this kernel asks is narrower: on one layer's
experts, how close to the 1-bit byte floor does a decode step get, and what does the format
cost in perplexity. The second question is `scripts/eval_ppl.py`'s, not this note's.

The format is BitNet b1.58's absmean recipe (a per-group mean of |w| as the scale, the sign
or the ternary code as the weight) applied after training to a checkpoint that was trained
in higher precision. That is lossy, and nobody should read this note as a claim about the
quality of 1-bit DeepSeek. It is a kernel and a format for measuring what 1-bit inference
costs and gains on this card, with the quality measured separately on text.

## The format

**Quantizer** (`w1_quantize`, `reference.w1_quantize`): per group of 128 k of one column, in
fp32.

| bits | scale | code | weight |
|---|---|---|---|
| 1 | `s = bf16(mean|w|)` over the group | `code = w >= 0` | `bf16(s (2 code - 1))`, that is `-s` or `+s` |
| 2 (ternary) | the same `s` | `0` for `w < -s/2`, `1` for `|w| <= s/2`, `2` for `w > s/2` | `bf16(s (code - 1))`, that is `-s`, `0` or `+s` |

The scale is rounded to bf16 before the codes are chosen, so the ternary thresholds are at
half the scale the kernel will use. Output: `packed` uint32 `[N, K bits / 32]`, one column's
bits contiguous with k ascending from the lowest bit (32 k per word at 1 bit, 16 at 2 bits,
two bits per code), and `scales` bf16 `[K/128, N]`. There is no zero point: the code is
symmetric around zero by construction, and no repack: a 32-bit word already holds a run of
one column's k, which is what the fragment selection below wants.

**The weight the kernel multiplies** is `-s`, `0` or `+s` exactly. Unlike int4, where the
dequant forms `(q - z) s` and rounds once, nothing here rounds at all: `s` is bf16, its
negation is bf16, and zero is zero. So every variant multiplies exactly the matrix
`reference.w1_dequantize` returns, and the check against cuBLAS on that matrix only sees
summation order, as with `w4gemm`.

Bytes per weight: 1 bit plus 16 bits per 128 weights is 0.1406 bytes, 14.2x fewer than
bf16 and 3.67x fewer than int4's 0.5156; the scales are an eighth of the packed bits.
Ternary is 0.2656 bytes, 7.5x fewer than bf16. The MXFP4 the checkpoint ships its routed
experts in is 0.5313 (4 bits plus a ue8m0 scale per 32).

## Why select, not look up

The int4 dequant in `w4gemm` is a `lop3` into a bf16 magic number, a subtract and a
multiply per two weights. At 1 bit there is less to do than that. The two values a weight can
take are `+s` and `-s`, and in bf16 those differ in one bit, the sign. So the A fragment of an
`mma.sync.m16n8k16` (two bf16 weights of one column in a 32-bit word, see
[w4gemm](w4gemm.md), "Weights as the 16-row operand") is the scale splatted into both halves,
XORed with the two code bits placed at bits 15 and 31, inverted, since code 1 means `+s`:

```
sel  = ((~bits) & 1) << 15 | ((~bits) & 2) << 30    // sign bits of the pair
frag = s_splat ^ sel                                 // {±s, ±s} as bf16x2
```

Two or three integer ops per pair and no multiply: the scale is applied by being the
operand, not by an `HMUL2`. A lookup table does the same selection the slow way. The table
has to be `{-s, +s}` for the current group, so it is rebuilt every 128 k; indexing it is a
`PRMT` or a `SEL` chain on the code bits, or a shared-memory load, which puts the weight
stream through the LSU twice. The XOR form is a `LOP3` and a shift per pair with the scale
word hoisted once per group. Ternary adds one
mask: `code - 1` is `-1`, `0` or `+1`, so the sign comes from `code >> 1` and a zero mask
from `code == 1`, two `LOP3` per pair, still no multiply and still exact.

What the dequant costs no longer matters at M = 1 in any case: the kernel is a weight stream
whose bytes are one seventh of int4's, and the arithmetic per byte went up seven times. That
is the tension the ladder is about: with so few bytes per weight, a warp has to keep many
more weights in flight to cover the same latency, and the per-weight work (one `mma` per 16
weights per token, as before) is now the larger share of the SM's time.

## Bytes per step, DeepSeek V4.1 Flash

Per layer, the routed experts are 384 x 3 x 5120 x 2304 = 13.59 billion weights. A decode
step at batch 1 reads 6 of the 384, 212 million weights per layer. Byte counts, not
measurements:

| format | bytes per weight | one layer's experts | 40 layers | 6 active experts, one layer | 6 active, 40 layers (one token) | floor per layer at 1,532 GB/s |
|---|---|---|---|---|---|---|
| bf16 | 2 | 27.18 GB | 1,087 GB | 424.7 MB | 16.99 GB | 277 µs |
| MXFP4 (the checkpoint) | 0.5313 | 7.22 GB | 289 GB | 112.8 MB | 4.51 GB | 73.6 µs |
| int4 g128 (`w4gemm`) | 0.5156 | 7.01 GB | 280 GB | 109.5 MB | 4.38 GB | 71.5 µs |
| ternary g128 | 0.2656 | 3.61 GB | 144 GB | 56.4 MB | 2.26 GB | 36.8 µs |
| 1 bit g128 | 0.1406 | 1.91 GB | 76.4 GB | 29.9 MB | 1.19 GB | 19.5 µs |

The 1-bit row splits into 1.70 GB of packed bits and 0.21 GB of scales per layer, 68.0 and
8.5 GB for the model. 32 GB holds 16.7 layers of 1-bit experts and nothing else; with the
attention weights, the shared expert, the embedding and the activations next to them it is
about 15 to 17 layers, and at ternary 8. So no bench on this card runs the model; the bench
and `scripts/bench_deepseek_flash.py` run one layer's experts (and as many layers as fit, to
rotate the weights past the 96 MB L2 the way `bench_layer` does with four copies). The
floor column is the active bytes over the copy roof, the number the measured kernel will be
judged against; it does not include the shared expert, the attention projections, the norms
or the routing, which are read at whatever width they are stored in.

The per-expert GEMMs at batch 1 are small: `1 x 2304 x 5120` for gate and up, `1 x 5120 x
2304` for down, 1.66 MB each at 1 bit, 5.0 MB per expert, 29.9 MB for the six. At
`w4gemm`'s measured fixed cost of 1 to 3 µs per launch, eighteen separate launches per
layer would cost as much as the bytes. That is why the MoE form exists.

## The ladder

| variant | idea | what it fixes |
|---|---|---|
| 0 | one thread per output, scalar: read the bit, add or subtract the activation, multiply by the scale per group, fp32 | baseline, and the reference the tests check the others against |
| 1 | one warp per 16-column strip and 8 tokens over the whole K; a lane loads its two columns' words, selects its four A fragments per k16 step by the XOR above, loads the activations straight into registers, 8 `mma.sync` per 128-k group | the tensor cores, 128-bit loads, nothing staged |
| 2 | M <= 16: warps split K, a block of 4 or 8 warps on one strip takes alternate groups, partial sums added in shared memory at the end | bytes in flight per SM at small M, where one warp per strip cannot cover the latency |

### Variant 0: one thread per output

Thread `(m, n)` walks K, reading one bit of column `n` per k and adding `a[m, k]` to a
group's running sum with the sign the bit says, then multiplies the group's sum by its scale
and accumulates. One fp32 FMA per group instead of one per weight, which is the one thing
this format gives the scalar kernel for free. It reads every word of a column once per token
and the activations `N` times. Its role is the same as `w4gemm` variant 0: the thing every
other variant is checked against, and the one that shows how far the bytes are from the time
when nothing is in flight.

### Variant 1: a warp per strip, the bits selected into fragments

The warp owns 16 output columns and 8 tokens, the `C^T = W^T A^T` arrangement of `w4gemm`:
the weights are the 16-row A operand, the tokens the 8 columns of B. Per 32 k a lane needs
the words of its two columns (`g` and `g + 8`), one 32-bit load each at 1 bit, two at
ternary; from them it selects the four bf16x2 fragments of two k16 steps by shift and XOR
against the group's splatted scale. The activations come straight from global memory into
the B fragments, 16 bytes per lane per 32 k, the way `w4gemm` loads them after its k
permutation; here no permutation is needed, because a lane's bits are contiguous in its
column's word and the fragment's k pairs `(2c, 2c + 1)` are two adjacent bits.

Per 128-k group and lane: 8 bytes of weights at 1 bit (16 at ternary), 64 bytes of
activations for 8 tokens, 8 `mma.sync`. The activations outweigh the weights eight to one
in a lane's loads, and they are the same for every strip, so they come from L2 after the
first strip reads them. What the warp cannot do is keep enough weight bytes in flight: 8
bytes per lane per group is one small load, and a 128-k group is 8 `mma` of latency to
cover with it. `w4gemm`'s variant 1 had the same shape and was latency-bound unless there
were thousands of strips; with an eighth of the bytes per load this one will be more so.
The 2304-column expert matrices have 144 strips, far under the 4 x SMs rule, so this variant
is not the one that should win at batch 1. Where it should be fine is wide N at larger M.

### Variant 2: warps split K for M <= 16

The `w4_warp_kernel` idea: a block of 4 or 8 warps on one 16-column strip, warp `wk` taking
groups `wk, wk + WK, ...`, each warp with the words of its next groups in a register ring,
no barrier until the partial sums are added in shared memory and written once. With 144
strips and 8 warps per strip the expert GEMM has 1,152 warps on 170 SMs, and each warp's
loads are the words of 4 groups ahead, 32 bytes per lane at 1 bit. For `down` (K = 5120, 40
groups) a 4-warp block gives 10 groups per warp; for gate and up (K = 2304, 18 groups) 8
warps give 2 or 3, which is where the K split stops paying and the launcher should fall back
to 4. The rules `w4gemm`'s `pick()` found (fit one wave, wide N needs no split, long K wants
more warps along K) are the starting point; which of them hold at a seventh of the bytes is
the first thing the sweep will say.

There is no M > 16 rung. At 1 bit the experts' decode shapes are the point, and a
prefill-shaped MoE GEMM (hundreds of tokens per expert) would be tensor-bound with a
fragment selection cheaper than int4's; `w4gemm`'s Stream-K tile with the selection swapped
in is the obvious way to get one, not done.

### Nsight

To be filled from `make ncu` on the first run. The metrics that decide between the rungs
are the ones that did for `w4gemm`:

| metric | v1, 1 x 2304 x 5120 | v2, 1 x 2304 x 5120 | v2, 1 x 5120 x 2304 | v2 moe, 6 of 384 |
|---|---|---|---|---|
| `dram__throughput` | unmeasured | unmeasured | unmeasured | unmeasured |
| SM active cycles / elapsed | unmeasured | unmeasured | unmeasured | unmeasured |
| achieved occupancy | unmeasured | unmeasured | unmeasured | unmeasured |
| top stall | unmeasured | unmeasured | unmeasured | unmeasured |
| registers per thread | unmeasured | unmeasured | unmeasured | unmeasured |

## The MoE launch

`w1gemm_moe` is the grouped form: `E` experts' weights in one `W1ExpertWeights` (packed
`[E, N, K bits / 32]`, scales `[E, K/128, N]`), the activation rows sorted by expert, and
int32 `offsets [E + 1]` with expert `e`'s rows at `offsets[e] .. offsets[e + 1]`. One launch
serves every expert: the grid is strips x experts, a block reads its row range from the
offsets, skips if it is empty (378 of the 384 are, at batch 1), and runs the variant 2 body
on its expert's strip over those rows. At batch 1 that is 6 experts x 144 strips of live
blocks per matrix and three matrices per layer, three launches instead of eighteen; the
gate and up projections can share a launch on interleaved columns the way `hgemm_swiglu`
does, which is on the list, not done.

Routing is DeepSeek-V3's: a sigmoid over the gate logits, the top 6 of 384 chosen on the
score plus a per-expert selection bias (the bias picks, it does not weight), the 6 chosen
scores renormalized to sum to one. `route` in `moe.py` does that in torch, `sort_by_expert`
gathers the token rows in expert order and builds the offsets, and `MoEBlock1Bit` runs the
three grouped GEMMs, the SwiGLU, the scatter back by the renormalized weights, and the
shared expert in bf16. The sort is a torch `argsort` over 6 x tokens ids per layer; at batch
1 that is a sort of six numbers, and it is still a handful of launches per layer that a
CUDA graph will have to absorb.

## Results

`bench_w1gemm`, RTX 5090, the DeepSeek V4.1 Flash expert projections at M = 1 to 64 and a
4096² reference shape, bits 1 and 2, every variant, weights rotated past L2, median of 50
single launches and 20 launches back to back, as in `bench_w4gemm`. GB/s is the traffic
floor (activations, packed bits, scales, output, once each) over the time, and the roof is
`cudaMemcpy`'s 1,532 GB/s. The bf16 column is `hgemm`'s decode kernel on the dequantized
weights, the int4 column `w4gemm` variant 2 on the same weights requantized to int4.

Unmeasured. The rows are the shapes the bench covers; the time columns fill in when it runs.

| shape | bits | M | floor µs at 1,532 GB/s | v0 µs | v1 µs | v2 µs | GB/s | % of roof | bf16 `hgemm` µs | int4 `w4gemm` µs |
|---|---|---|---|---|---|---|---|---|---|---|
| gate/up, 5120 to 2304 | 1 | 1 | 1.1 | | | | | | | |
| | 1 | 16 | 1.2 | | | | | | | |
| | 1 | 64 | 1.7 | | | | | | | |
| | 2 | 1 | 2.0 | | | | | | | |
| | 2 | 16 | 2.2 | | | | | | | |
| down, 2304 to 5120 | 1 | 1 | 1.1 | | | | | | | |
| | 1 | 16 | 1.2 | | | | | | | |
| | 1 | 64 | 1.7 | | | | | | | |
| | 2 | 1 | 2.0 | | | | | | | |
| | 2 | 16 | 2.2 | | | | | | | |
| 4096 x 4096 | 1 | 1 | 1.5 | | | | | | | |
| | 1 | 16 | 1.7 | | | | | | | |
| | 2 | 1 | 2.9 | | | | | | | |
| moe, 6 of 384 experts, gate/up | 1 | 1 token | 6.5 | | | | | | | |
| moe, 6 of 384 experts, down | 1 | 1 token | 6.5 | | | | | | | |
| moe, 6 of 384 experts, gate/up | 1 | 64 tokens | 6.5 to 416, by experts touched | | | | | | | |

The floors are the shape's bytes over the copy roof and say what a perfect kernel would
take; `w4gemm` measured 1 to 3 µs of launch and ramp on top of its floors at these sizes,
and a 1.1 µs floor is under that. The single-expert rows will therefore be launch-bound and
the MoE rows are the ones that can show the format's gain; a per-layer decode step from
`scripts/bench_deepseek_flash.py` (three grouped launches, routing, SwiGLU, the shared
expert) is the number to compare with the 19.5 µs per layer of the bytes table.

## What to measure first

In this order, once the card is up:

1. `bench_w1gemm` with the checks on: the quantizer against the torch copy bit for bit,
   every variant against cuBLAS on `w1_dequantize`'s matrix. Nothing else means anything
   until that passes.
2. The MoE row at 1 token against the 29.9 MB floor: 19.5 µs is the byte floor per matrix
   set, and the gap to it is launch, ramp and the empty-block skip. If the gap is launch, the
   gate/up merge and a CUDA graph of the layer are next; if the kernel is slow with its
   bytes in flight, the register ring depth and warps per strip.
3. The same step at 16 and 64 tokens, where the experts' rows stop being one each and the
   `mma` per weight start to matter.
4. `scripts/eval_ppl.py` on the real checkpoint at 1 bit and ternary, routed experts only,
   against the MXFP4 it ships in. That is the number that decides whether any of the above
   is worth a layer of a real model, and it is not a kernel number.
5. `scripts/bench_deepseek_flash.py` with as many layers as fit, rotating, for a tokens/s
   per layer that a 40-layer model would extrapolate from, with the caveat that 40 layers do
   not fit this card at any of these widths.
