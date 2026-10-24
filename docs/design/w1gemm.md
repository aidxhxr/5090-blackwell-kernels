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

**Measured on the RTX 5090** (2026-10-23): the quantizer, every variant and the grouped
form check out against the CPU and `tests/test_w1gemm.py` passes; the times are in the
results section. The byte counts and floors before it were written first and still hold.
A fourth rung came out of the measurements.

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
| 3 | M <= 16: lane c of a quad takes k 32c .. 32c + 31 of a group, so it loads one word of each row instead of the row's 16 bytes; an 8-deep ring of words and a 4-deep ring of activations per warp; 1 to 16 warps along K from the strip and SM counts; 8-token tiles for M <= 8 | what variant 2 measured: 2 KB of distinct weight bytes in flight per block and every group waiting on its activations |

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

### Variant 3: one word per lane, deeper rings, warps from the SM count

Variant 2 measured 13.3 us at 1 x 2304 x 5120 and 35 us at 1 x 4096 x 14336, against 1.1
and 5.4 us of bytes at the copy roof. Three things in it were short:

* **Bytes in flight.** The four lanes of a quad load the same 16 bytes of a row and each uses
  4 of them, so a warp's two 16-byte loads per group are 256 distinct bytes held in 8
  registers per lane. A block of 4 warps with a 2-deep ring has 2 KB in flight, and the
  144-strip expert puts at most one block on each of 144 SMs. Little asks for about 9 KB per
  SM at this card's measured latency.
* **The activations.** They were loaded in front of the mma that needs them. A warp reads
  1 / WK of a token's activations, so most of those loads miss L1 and wait on L2, every
  group.
* **The grid.** N / 16 blocks of 4 warps regardless of the device.

The k order inside a 128-k group is free as long as both operands follow it. Variant 3 gives
lane `(g, c)` the k `32c .. 32c + 31` and step pair `i` the 8 of them at `32c + 8i`, so its
weights are word `c` of each of its two rows (words `2c`, `2c + 1` at 2 bits): one 4-byte
load per row, and the quad's four lanes read the row's 16 bytes once between them. Its
activations for step pair `i` are the 16 bytes at `32c + 8i`, one load as before. That makes a
ring slot 2 registers instead of 8, so the ring is 8 groups deep at 8 tokens (4 at 16), and
the activations get a ring of their own, 3 groups ahead at 8 tokens. The fragment selection is
the same XOR, only the byte it starts from moved.

One more thing the first build of this rung did wrong, and variants 1 and 2 still do: the
activation load was `row valid ? load : 0`. That compiles to a predicated load whose other
lanes write zeros into the same registers, and since a warp has one scoreboard, not one per
lane, the zeroing waits for the load: every group waited on its activations whatever the
ring depth. The token columns of an mma are independent, so a lane past the last row now
reads the last row and its sums land in outputs nobody stores. That took 1 x 2304 x 5120 at
one warp along K from 23.6 to 21.5 us; it is not most of the gap, see below.

The warps along K (`stream_warps`) come from this sweep, `SPARK_W1_WARPS` forcing each of
1, 2, 4, 8 and 16, 1 bit, 20 launches back to back, us per launch:

| shape (M x N x K) | strips | 1 | 2 | 4 | 8 | 16 | picked |
|---|---|---|---|---|---|---|---|
| 1 x 2304 x 5120 | 144 | 16.5 | 10.0 | 5.2 | 4.2 | 4.3 | 16 |
| 16 x 2304 x 5120 | 144 | 21.9 | 15.0 | 10.4 | 9.2 | **8.0** | 16 |
| 1 x 5120 x 2304 | 320 | 7.7 | 4.9 | 4.0 | **3.9** | 4.8 | 4 |
| 16 x 5120 x 2304 | 320 | 10.9 | 7.4 | **6.1** | 6.4 | 8.1 | 4 |
| 1 x 4096 x 4096 | 256 | 13.2 | 8.0 | 5.3 | **5.2** | 5.9 | 4 |
| 16 x 4096 x 4096 | 256 | 17.6 | 11.4 | **7.6** | 9.0 | 10.3 | 4 |
| 1 x 6144 x 4096 | 384 | 13.4 | 9.3 | 6.9 | **6.4** | 8.1 | 4 |
| 1 x 4096 x 14336 | 256 | 46.8 | 28.4 | 16.7 | **13.0** | 14.2 | 8 |
| 16 x 4096 x 14336 | 256 | 59.8 | 40.3 | 29.5 | **29.4** | 32.5 | 8 |
| 1 x 28672 x 4096 | 1,792 | 19.8 | 20.2 | 17.3 | **16.3** | 26.4 | 4 |
| 16 x 28672 x 4096 | 1,792 | **27.5** | 29.4 | 31.0 | 43.2 | 45.4 | 4 |

So: fewer strips than SMs takes 16 warps, everything else 4, doubled until a warp has at
most 16 groups (K = 14336 takes 8), and never more than one warp per two groups. It is a
compromise at the 28672-wide weight, where M = 1 wants 8 and M = 16 wants 1; 4 is within 6%
and 13% of each. The rule is a function of N and K only, on purpose: the partial sums are
added in shared memory in a fixed order, so the result is deterministic and depends on the
split, and the grouped form below has to pick the same split per expert to stay bit for bit
equal to `w1gemm` on that expert's rows.

What is still slow. Nsight on 1 x 2304 x 5120 with one warp along K: a lone warp per SM issues
an instruction every 5.9 cycles, 3.4 of them waiting on L1TEX, and a group is about 170
instructions (8 mma and the selection), so it costs about 0.4 us however deep the ring is;
16 warps along K (the default there) bring it to 5.4 us under Nsight with DRAM at 18% of peak
and the occupancy capped at 16 warps per SM by 125 registers a thread. The kernel is short of
warps, not of bytes in flight. The next rung would cut registers (the activation ring is 64 of
them) or stage the words through shared memory with `cp.async`, so more warps fit; not done.

There is no M > 16 rung. At 1 bit the experts' decode shapes are the point, and a
prefill-shaped MoE GEMM (hundreds of tokens per expert) would be tensor-bound with a
fragment selection cheaper than int4's; `w4gemm`'s Stream-K tile with the selection swapped
in is the obvious way to get one, not done.

### Nsight

`ncu` (clocks locked by Nsight, so the durations are a little longer than the bench's), 1 bit:

| metric | v2, 1 x 2304 x 5120 | v3, 1 x 2304 x 5120 | v3, 1 warp along K | v3, 1 x 28672 x 4096 |
|---|---|---|---|---|
| duration | 23.9 us | 5.4 us | 22.2 us | 20.2 us |
| block, grid | 128 x 144 | 512 x 144 | 32 x 144 | 128 x 1,792 |
| `dram__throughput` | 4.0% | 18.0% | 4.4% | 46.5% |
| achieved occupancy | 8.3% | 33.3% | 2.1% | 30.6% |
| warp cycles per issued instruction | 21.5 | 13.7 | 5.9 | 9.2 |
| of which waiting on L1TEX | 18.0 | 4.3 | 3.4 | 2.9 |
| registers per thread | 73 | 125 | 115 | 123 |

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

Variant 3 of the grouped form changes the grid, not only the body. Variants 1 and 2 launch a
block row per expert, so at batch 1 a gate|up launch is 384 x 72 blocks of which 378 rows
exit at once. Variant 3 launches `min(E, T)` block rows (`T` the row count the binding
passes; `-1` falls back to E), and block row `j` finds the `j`-th expert that has rows with a
ballot scan of the offsets staged in shared memory (E <= 1024). An expert with at most 16 rows
runs variant 3's block, with the warps along K `w1gemm` would pick for that weight alone; an
expert with more runs variant 1's warp body over 8-row chunks spread across the block's
warps, which is what `w1gemm` does above 16 rows. Every expert's rows therefore come out bit
for bit as `w1gemm` on those rows, which `tests/test_w1gemm.py` checks.

## Results

RTX 5090, 2026-10-23, `bench_w1gemm --chain=20` (rows in `results/w1gemm.json`): random
weights quantized on the GPU and checked bit for bit against the CPU quantizer, every
variant checked against a CPU product on the dequantized matrix (all pass), the weights
rotated past L2. The floor is the shape's bytes (packed bits, scales, activations, output)
over the 1,532 GB/s copy roof. The bf16 column is `hgemm` variant 6 on the dequantized
weights, the int4 column `w4gemm` variant 2 on the same weights at int4.

Single launches, median of 50, us. The JSON rows are these.

| shape | bits | M | floor | v0 | v1 | v2 | v3 | bf16 `hgemm` | int4 `w4gemm` |
|---|---|---|---|---|---|---|---|---|---|
| gate/up, 5120 to 2304 | 1 | 1 | 1.1 | 134.2 | 29.5 | 13.3 | **5.2** | 17.4 | 7.0 |
|  | 1 | 2 | 1.1 | 134.1 | 29.7 | 13.3 | **7.3** | 17.4 | 8.9 |
|  | 1 | 4 | 1.1 | 130.2 | 29.7 | 15.3 | **7.3** | 17.6 | 9.2 |
|  | 1 | 8 | 1.2 | 129.9 | 29.7 | 15.4 | **7.3** | 17.6 | 9.1 |
|  | 1 | 16 | 1.2 | 179.1 | 29.6 | 17.5 | **9.3** | 19.2 | 13.2 |
|  | 1 | 64 | 1.7 | - | 27.6 | 25.5 | **25.6** | 23.5 | 17.4 |
|  | 2 | 1 | 2.1 | 166.8 | 44.1 | 11.2 | **7.2** | 17.3 | 7.1 |
|  | 2 | 2 | 2.1 | 164.9 | 44.1 | 11.3 | **7.2** | 17.4 | 8.4 |
|  | 2 | 4 | 2.1 | 164.8 | 46.0 | 11.3 | **7.7** | 17.4 | 9.1 |
|  | 2 | 8 | 2.1 | 164.5 | 46.0 | 13.2 | **9.3** | 17.4 | 9.1 |
|  | 2 | 16 | 2.2 | 205.4 | 45.2 | 15.2 | **9.3** | 19.3 | 13.2 |
|  | 2 | 64 | 2.7 | - | 43.9 | 44.0 | **43.9** | 23.5 | 17.3 |
| down, 2304 to 5120 | 1 | 1 | 1.1 | 54.2 | 14.2 | 7.3 | **5.2** | 17.3 | 7.2 |
|  | 1 | 2 | 1.1 | 53.7 | 15.2 | 8.9 | **5.2** | 17.2 | 8.9 |
|  | 1 | 4 | 1.1 | 52.2 | 15.3 | 9.1 | **7.2** | 17.2 | 9.1 |
|  | 1 | 8 | 1.2 | 54.5 | 15.4 | 9.2 | **7.2** | 17.3 | 9.2 |
|  | 1 | 16 | 1.2 | 97.2 | 15.2 | 9.3 | **9.1** | 17.3 | 9.2 |
|  | 1 | 64 | 1.7 | - | 19.4 | 18.0 | **17.5** | 19.3 | 19.4 |
|  | 2 | 1 | 2.1 | 56.4 | 21.4 | 7.2 | **7.2** | 17.3 | 7.2 |
|  | 2 | 2 | 2.1 | 56.3 | 21.5 | 8.4 | **7.2** | 17.3 | 8.5 |
|  | 2 | 4 | 2.1 | 56.0 | 21.5 | 9.2 | **7.3** | 17.3 | 9.1 |
|  | 2 | 8 | 2.1 | 70.7 | 21.5 | 9.2 | **7.2** | 17.2 | 9.9 |
|  | 2 | 16 | 2.2 | 111.6 | 21.5 | 9.3 | **9.3** | 17.6 | 9.2 |
|  | 2 | 64 | 2.7 | - | 27.7 | 27.7 | **27.8** | 19.4 | 19.4 |
| 4096 x 4096 | 1 | 1 | 1.6 | 105.5 | 23.5 | 11.3 | **7.2** | 23.4 | 9.2 |
|  | 1 | 16 | 1.7 | 228.3 | 23.6 | 11.4 | **9.3** | 23.4 | 11.2 |
|  | 2 | 1 | 2.9 | 122.0 | 35.8 | 11.2 | **9.3** | 23.5 | 9.2 |
|  | 2 | 16 | 3.1 | 250.4 | 35.9 | 11.2 | **11.3** | 23.5 | 11.2 |
| gate/up, 4096 to 28672 | 1 | 1 | 10.8 | 132.1 | 21.5 | 25.6 | **19.4** | 146.3 | 39.8 |
|  | 1 | 16 | 11.5 | 3825.3 | 29.7 | 29.6 | **31.8** | 148.3 | 41.9 |
|  | 2 | 1 | 20.4 | 144.4 | 42.0 | 39.6 | **27.7** | 147.8 | 39.8 |
|  | 2 | 16 | 21.0 | 3851.9 | 52.0 | 46.0 | **36.7** | 148.3 | 41.9 |

Twenty launches between one event pair, us per launch: what a decode step's launches queued
back to back see, without the 1 to 3 us a lone launch spends being scheduled (the single
launches above also come in steps of about 2 us on this machine, which the chains do not).

| shape | bits | M | floor | v1 | v2 | v3 | v3 GB/s | % of roof | bf16 `hgemm` | int4 `w4gemm` | v3 vs int4 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gate/up, 5120 to 2304 | 1 | 1 | 1.1 | 23.4 | 11.7 | **4.3** | 389 | 25% | 16.5 | 6.0 | 1.40x |
|  | 1 | 2 | 1.1 | 23.7 | 11.9 | **4.8** | 350 | 23% | 16.5 | 6.4 | 1.34x |
|  | 1 | 4 | 1.1 | 24.1 | 12.8 | **5.3** | 327 | 21% | 16.5 | 6.9 | 1.31x |
|  | 1 | 8 | 1.2 | 24.4 | 14.1 | **6.2** | 287 | 19% | 16.6 | 7.3 | 1.17x |
|  | 1 | 16 | 1.2 | 23.9 | 15.9 | **7.1** | 266 | 17% | 16.8 | 10.8 | 1.51x |
|  | 1 | 64 | 1.7 | 23.7 | 23.6 | **23.6** | 110 | 7% | 21.7 | 16.2 | 0.68x |
|  | 2 | 1 | 2.1 | 42.6 | 9.3 | **5.6** | 558 | 36% | 16.5 | 6.0 | 1.07x |
|  | 2 | 2 | 2.1 | 43.1 | 9.3 | **6.1** | 519 | 34% | 16.5 | 6.4 | 1.06x |
|  | 2 | 4 | 2.1 | 43.5 | 10.1 | **6.6** | 483 | 32% | 16.5 | 6.9 | 1.04x |
|  | 2 | 8 | 2.1 | 44.0 | 11.1 | **7.3** | 444 | 29% | 16.5 | 7.2 | 0.99x |
|  | 2 | 16 | 2.2 | 43.1 | 12.8 | **7.9** | 425 | 28% | 16.8 | 10.8 | 1.36x |
|  | 2 | 64 | 2.7 | 41.8 | 41.9 | **41.9** | 98 | 6% | 21.8 | 16.2 | 0.39x |
| down, 2304 to 5120 | 1 | 1 | 1.1 | 10.9 | 6.4 | **3.9** | 429 | 28% | 15.6 | 6.4 | 1.64x |
|  | 1 | 2 | 1.1 | 11.2 | 6.6 | **4.0** | 422 | 28% | 15.6 | 6.5 | 1.62x |
|  | 1 | 4 | 1.1 | 11.3 | 7.1 | **4.6** | 373 | 24% | 15.5 | 6.8 | 1.47x |
|  | 1 | 8 | 1.2 | 11.5 | 7.1 | **5.2** | 341 | 22% | 15.6 | 7.2 | 1.38x |
|  | 1 | 16 | 1.2 | 11.3 | 8.2 | **6.6** | 287 | 19% | 15.6 | 8.1 | 1.23x |
|  | 1 | 64 | 1.7 | 16.2 | 16.2 | **16.2** | 161 | 10% | 17.3 | 18.1 | 1.11x |
|  | 2 | 1 | 2.1 | 19.2 | 6.4 | **5.3** | 597 | 39% | 15.5 | 6.4 | 1.21x |
|  | 2 | 2 | 2.1 | 19.7 | 6.4 | **5.3** | 595 | 39% | 15.6 | 6.5 | 1.23x |
|  | 2 | 4 | 2.1 | 19.9 | 7.0 | **5.7** | 557 | 36% | 15.6 | 6.8 | 1.18x |
|  | 2 | 8 | 2.1 | 20.1 | 6.9 | **6.2** | 525 | 34% | 15.6 | 7.2 | 1.15x |
|  | 2 | 16 | 2.2 | 19.6 | 8.0 | **7.7** | 437 | 29% | 15.6 | 8.1 | 1.05x |
|  | 2 | 64 | 2.7 | 26.7 | 26.7 | **26.8** | 152 | 10% | 17.2 | 18.1 | 0.67x |
| 4096 x 4096 | 1 | 1 | 1.6 | 18.5 | 9.6 | **5.3** | 450 | 29% | 21.4 | 7.5 | 1.42x |
|  | 1 | 16 | 1.7 | 19.0 | 10.3 | **7.6** | 344 | 22% | 21.4 | 9.7 | 1.28x |
|  | 2 | 1 | 2.9 | 34.2 | 8.9 | **7.2** | 620 | 40% | 21.4 | 7.5 | 1.04x |
|  | 2 | 16 | 3.1 | 34.5 | 9.6 | **9.4** | 504 | 33% | 21.4 | 9.7 | 1.04x |
| gate/up, 4096 to 28672 | 1 | 1 | 10.8 | 20.2 | 23.6 | **17.2** | 962 | 63% | 145.2 | 37.9 | 2.20x |
|  | 1 | 16 | 11.5 | 28.6 | 27.6 | **31.1** | 564 | 37% | 146.8 | 40.0 | 1.28x |
|  | 2 | 1 | 20.4 | 40.2 | 37.6 | **26.7** | 1,170 | 76% | 145.8 | 37.9 | 1.42x |
|  | 2 | 16 | 21.0 | 50.6 | 44.6 | **35.9** | 897 | 59% | 147.2 | 40.0 | 1.11x |

What the tables say:

* **Variant 3 is the fastest rung at M <= 16 on every shape but one**, 1.2 to 2.7x over
  variant 2 at M = 1 in the chains (1 x 2304 x 5120: 11.7 to 4.3 us). The one loss is
  16 x 28672 x 4096 at 1 bit, 31.1 us against variant 2's 27.6, where the 1,792 strips would
  want one warp along K (27.5) and the rule picks 4.
* **It is still far from the floor at the expert shapes**: 4 to 8 us for 1.1 to 2.2 us of
  bytes, 17 to 39% of the roof. The 28672-wide weight, long enough to amortize the ramp,
  reaches 962 GB/s at 1 bit and 1,170 at 2 bits (63% and 76% of the roof).
* **Against int4, 1.4 to 2.2x at M = 1 at 1 bit, where the bytes allow 3.67x, and 1.0 to
  1.4x at ternary, where they allow 1.94x.** The int4 kernel is closer to its floor (6.0 us
  for 4.0 us of bytes on the expert): it moves four times the bytes per mma, so the same
  instruction latency covers four times the bytes, and that latency on few warps (Nsight,
  above) is what limits the 1-bit kernel.
* **Ternary pays little for twice the bytes**: 5.3 to 5.6 us against 3.9 to 4.3 at M = 1 on
  the experts, since neither is at its floor.
* **M = 64 has no rung of its own** and loses to both bf16 and int4 (64 x 2304 x 5120: 23.6
  us against 21.7 and 16.2); prefill-sized expert GEMMs want `w4gemm`'s Stream-K tile with
  the selection swapped in, not done.

The grouped form, DeepSeek V4.1 Flash's two expert GEMMs over 384 experts of random codes,
T tokens routed to 6 distinct experts each, the T x 6 rows sorted, chains of 20 (us per
launch); the floor is the distinct experts' bytes at the copy roof, the loop one `w1gemm`
launch per routed expert (single-launch timing):

| GEMM | bits | T | experts | floor | v1 | v2 | v3 | per-expert loop |
|---|---|---|---|---|---|---|---|---|
| gate\|up, 5120 to 4608 | 1 | 1 | 6 | 13.0 | 40.8 | 40.9 | **27.8** | 42.0 |
| | 1 | 4 | 23.4 | 50.6 | 114.0 | 113.7 | **85.1** | 160.7 |
| | 1 | 16 | 84.5 | 183.0 | 428.3 | 427.8 | **307.5** | 571.3 |
| | 1 | 64 | 243 | 526.3 | 1,386 | 1,386 | **994** | 1,697 |
| | 2 | 1 | 6 | 24.5 | 69.3 | 69.5 | **37.0** | 57.3 |
| | 2 | 64 | 243 | 994.0 | 3,424 | 3,430 | **1,351** | 2,309 |
| down, 2304 to 5120 | 1 | 1 | 6 | 6.5 | 25.0 | 24.9 | **17.3** | 24.6 |
| | 1 | 4 | 23.4 | 25.3 | **36.7** | 36.9 | 54.9 | 94.0 |
| | 1 | 16 | 84.5 | 91.5 | 108.0 | **105.9** | 198.5 | 335.3 |
| | 1 | 64 | 243 | 263.1 | 442.4 | **438.6** | 669.1 | 973.5 |
| | 2 | 1 | 6 | 12.3 | 37.8 | 37.7 | **21.8** | 32.6 |
| | 2 | 64 | 243 | 497.0 | 891.3 | 888.5 | **798.4** | 1,281 |

At batch 1 the grouped variant 3 is 1.4 to 1.9x variant 2, and a layer's two grouped GEMMs
take 45 us at 1 bit against 66. The down GEMM at 1 bit with 4 or more tokens is a loss, 1.5
to 1.9x slower than variant 2: its K is 2,304 (18 groups), so a block is one strip of 4
warps with 4 or 5 groups each, the grid is 320 strips x one row per routed expert (7,680
blocks at T = 4), and at 154 registers 3 blocks fit an SM: 15 waves of short blocks that each
pay the offsets scan and the ring's fill. The gate|up GEMM has 40 groups a strip and wins at
every T by more than the down GEMM loses, so a layer's routed path is faster on variant 3 at
every T measured, and the default stays the top rung. Fixing the down shape means more
strips a block (the scan paid once) or fewer registers; not done.

## What is next

Measured above: the quantizer, every variant against the CPU, the grouped form, Nsight on the
decode shapes. Left, in order:

1. More warps per SM for variant 3: the activation ring is 64 of its 125 registers; a
   shallower one, or the words through shared memory by `cp.async` as `w4gemm` does, should
   let 32 warps fit an SM. Then the down shape of the grouped form with several strips a
   block.
2. `scripts/eval_ppl.py` on the real checkpoint at 1 bit and ternary, routed experts only,
   against the MXFP4 it ships in. That is the number that decides whether any of the above
   is worth a layer of a real model, and it is not a kernel number.
3. The real checkpoint through `deepseek_flash.load_moe_layer`, whose key names are still an
   assumption.
4. A prefill rung (M > 16) on `w4gemm`'s Stream-K tile.
