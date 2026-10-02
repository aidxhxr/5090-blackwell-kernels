# Low-precision weights on a real Llama-3-8B

The int4, fp8 and fp4 GEMMs ([w4gemm](w4gemm.md), [fp8gemm](fp8gemm.md), [fp4gemm](fp4gemm.md))
were only ever checked on Gaussian matrices. This note puts them under the engine
([serving](serving.md)) with the weights of Llama-3-8B-Instruct and measures two things per
format: what it costs in quality (wikitext-2 perplexity, and KL divergence against bf16) and
what it buys in speed (graphed decode, prefill) and memory.

Pieces:

- `python/spark_kernels/quant.py`: one class per format that quantizes a `[K, N]` weight once
  and multiplies by it, with the activation quantizer its GEMM needs.
- `SparkModel(weights, rope, weights_format=..., head_format=...)` in `engine.py`: quantizes
  the four projections of every layer (and optionally lm_head) at construction and frees the
  bf16 copies.
- `src/kernels/fp8quant.cu`: `fp8_quantize`, a bf16 to e4m3 activation quantizer (per
  tensor, per row, MX), and the amax pass the NVFP4 quantizer now uses for its tensor scale.
- `python/spark_kernels/awq.py` and `scripts/awq_search.py`: activation-aware int4 scales.
- `scripts/eval_ppl.py --format --head-format --awq --save-ref --ref`, and
  `scripts/bench_quant.py`.
- Results: `results/llm_ppl.jsonl` (one row per perplexity run), `results/llm_quant.json`
  (speed, memory and the kernel breakdown per format, with its perplexity rows).
- Tests: `tests/test_quant.py`, `tests/test_awq.py`, the `fp8_quantize` tests in
  `tests/test_fp8gemm.py`, a graph regression test in `tests/test_w4gemm.py`.

Every number is from the RTX 5090 (torch 2.14, CUDA 13), on a GPU shared with other jobs but
never at the same time (one process at a time under a lock).

## The formats

| format | weights | activations | GEMM | bits per weight |
|---|---|---|---|---|
| bf16 | bf16 | bf16 | hgemm | 16 |
| int4 | int4, a bf16 scale per 128 k | bf16 | w4gemm | 4.125 |
| int4-asym | int4, a bf16 scale and a uint8 zero per 128 k | bf16 | w4gemm | 4.19 |
| fp8 | e4m3, one fp32 scale | e4m3, one scale per call | fp8gemm | 8 |
| fp8-tok | e4m3, one fp32 scale | e4m3, one scale per token | fp8gemm | 8 |
| mxfp8 | e4m3, an e8m0 scale per 32 k | the same | fp8gemm (MX) | 8.25 |
| nvfp4 | e2m1, an e4m3 scale per 16 k, one fp32 scale | the same, scales per call | fp4gemm | 4.5 |
| mxfp4 | e2m1, an e8m0 scale per 32 k | the same | fp4gemm | 4.25 |

All weight quantizers are round to nearest, computed once from the bf16 checkpoint. Only the
four projections of each layer change format; the embedding, the norms, the K/V cache and
attention stay bf16, and lm_head stays bf16 unless `head_format` says otherwise.

The fp8 activation scales are powers of two: the smallest 2^e with max|x| / 2^e <= 448. The
division is then exact and the e4m3 rounding is the only one. e4m3 is a float format, so a
scale that is up to 2x too large keeps 3 mantissa bits on every normal value and only moves
the subnormal floor.

### What the quantized layers lose: the fused epilogues

The bf16 layer is eight launches because hgemm fuses the residual add into the o and down
GEMMs and the SwiGLU into the gate/up GEMM. fp8gemm, fp4gemm and w4gemm have no epilogue
options, so the quantized layer does this instead:

- The two residual adds go into the norms that follow them: `add_rmsnorm_` adds the o output
  to the residual stream and normalizes for the MLP, and the down output is carried to the next
  layer's attention norm (and added once after the last layer). No extra launch.
- The SwiGLU is its own launch (`swiglu` on the two halves of the gate/up output; the
  interleaved gate/up weight is put back to `[gate | up]` before it is quantized, so the
  halves are strided views the kernel reads in place).
- The activation quantizer: one or two launches per projection.

What that costs, from the profiles below: the SwiGLU is 2.9 to 3.3 ms of a 2048-token prefill
(7.7% of the nvfp4 prefill) and 0.04 ms of a decode step; the adds inside the norms put
0.5 ms on the prefill's norms (1.3 to 1.4 ms against 0.8 ms in bf16) and nothing measurable
on decode.

## Quality

Perplexity on the whole wikitext-2 test set: 141 windows of 2048 tokens, 288,627 scored
tokens, the tokenizer's BOS at the start of the text only. KL is the mean KL(bf16 || format)
per token in nats, and top-1 the fraction of positions whose argmax agrees with bf16, both on
the first 4 windows (8,192 positions) against the saved logits of the bf16 engine.

| format | ppl | vs bf16 | KL | top-1 |
|---|---|---|---|---|
| bf16 (this engine) | 8.287 | | | |
| bf16, plain PyTorch | 8.287 | +0.000 | 0.0006 | 98.9% |
| fp8 | 8.320 | +0.033 | 0.0097 | 95.4% |
| fp8-tok | 8.319 | +0.031 | 0.0087 | 95.8% |
| fp8, lm_head fp8 too | 8.314 | +0.027 | 0.0119 | 94.8% |
| mxfp8 | 8.399 | +0.112 | 0.0138 | 94.7% |
| int4 | 9.136 | +0.849 | 0.121 | 85.0% |
| int4-asym | 8.854 | +0.567 | 0.079 | 88.1% |
| int4 + AWQ scales | 8.775 | +0.487 | 0.090 | 87.3% |
| int4-asym + AWQ scales | 8.562 | +0.274 | 0.057 | 90.6% |
| int4-asym + AWQ, lm_head int4-asym | 8.664 | +0.376 | 0.071 | 87.9% |
| nvfp4 | 8.976 | +0.689 | 0.095 | 87.3% |
| nvfp4, lm_head nvfp4 | 9.051 | +0.764 | 0.111 | 85.2% |
| mxfp4 | 12.167 | +3.880 | 0.413 | 72.5% |

The PyTorch row is the noise floor: the same weights through `F.scaled_dot_product_attention`
and cuBLAS agree with our kernels on 98.9% of argmaxes. fp8 is close to free. The rest:

- **fp8 does not need per-token scales on this model.** One scale over the whole
  `[8192, K]` activation of a packed prefill, outlier tokens and all, gives 8.320; one per
  token gives 8.319. e4m3 has about 15 binades of normal values below 448, so when an
  outlier token sets the scale, the ordinary tokens still keep 3 mantissa bits.
- **MXFP8 is worse than per-tensor fp8**, which I did not expect from a format with 32x
  finer scales. The OCP "floor" recipe puts the largest element of a block in [256, 512) and
  saturates anything in [448, 512) to 448, in every block. A probe with the "ceil"
  recipe (`reference.quantize_mx(mode="ceil")` for both operands, no saturation) gave 8.348
  against 8.400: most of the gap is that saturation. The rest is the power-of-two block scale.
- **NVFP4 (W4A4) beats round-to-nearest int4 (W4A16).** 8.98 against 9.14 with the
  activations in 4 bits too. A scale per 16 values in e4m3 is a much finer grid than a bf16
  scale per 128, and that matters more than the activation bits. The symmetric int4 code
  also wastes one of its 16 levels: s = max|w| / 7 maps every weight into [-7, 7], so -8 is
  never used.
- **MXFP4 is not usable as is**: +3.9. A power-of-two scale per 32 with e2m1 elements loses
  up to a factor 2 of a 6-step grid to the scale rounding.
- **int4 lm_head costs 0.10** on top of AWQ int4-asym; fp8 lm_head costs nothing.

Run-to-run noise: the split-K GEMMs add fp32 partials in arrival order, so two runs differ in
the last bits, and with activations rounded to 4 bits that flips some codes. nvfp4 measured
8.989 and 8.976 in two runs; the fp8 and int4 rows repeated to three decimals.

### AWQ scales for int4

Round to nearest gives a group of 128 one step. The few input channels with large
activations then lose as much relative precision as any other, while their errors weigh
more. `awq.search` does AWQ's scale search: for each projection, s = mean|x|^alpha over
calibration tokens, alpha on a grid of 20 in [0, 1), keeping the alpha that minimizes
`||x W - (x / s) Q(W s)||^2`. The activation side folds into the op before the projection,
so nothing runs at decode: into `attn_norm` for qkv, into `mlp_norm` for gate/up, into the
up columns for down, and into the v columns for o (one scale per K/V channel, shared by the
four query heads that read it). Calibration is 32 windows of 512 tokens from the wikitext-2
*train* split; the search takes 20 s for the 32 layers.

It closes about half of the gap: int4-asym from +0.567 to +0.274, int4 from +0.849 to +0.487.
The largest single wins are on down projections (the SwiGLU output has a few huge channels:
layer 1's down error, symmetric codes, dropped from 8.3e-4 to 3.1e-5 of the output energy) and on the o
projections of the last layers. I stopped there. AWQ's second step, a clipping search, and
GPTQ are the obvious next ones; both fold into the bf16 weights the same way.

## Speed and memory

`scripts/bench_quant.py`, one format per process. Prefill is prompts of 2048 random tokens,
one alone and four packed into one prefill, median of 5 after a warm-up. Decode is B prompts
of 512 tokens, then 128 graphed decode steps (the context grows from 512 to 640). Weights is
everything the model holds; the bf16 embedding and lm_head are 1.96 GiB of it.

| format | weights GiB | prefill 1x2048 tok/s | prefill 4x2048 | decode B=1 tok/s | B=8 | B=32 |
|---|---|---|---|---|---|---|
| bf16 | 14.96 | 15,992 | 16,196 | 102 | 789 | 2,799 |
| int4 | 5.31 | 12,955 | 12,526 | 271 | 1,917 | 5,008 |
| int4-asym | 5.36 | 9,564 | 9,577 | 268 | 1,902 | 4,890 |
| int4-asym, lm_head int4-asym | 4.64 | 9,594 | 9,586 | 312 | 2,173 | 5,218 |
| fp8 | 8.46 | 26,579 | 26,221 | 165 | 1,242 | 4,183 |
| fp8-tok | 8.46 | 23,980 | 23,776 | 159 | 1,203 | 4,070 |
| fp8, lm_head fp8 | 7.97 | 26,697 | 26,291 | 175 | 1,316 | 4,357 |
| mxfp8 | 8.66 | 26,842 | 26,819 | 166 | 1,238 | 4,165 |
| nvfp4 | 5.61 | 52,608 | 50,809 | 224 | 1,658 | 5,297 |
| nvfp4, lm_head nvfp4 | 4.91 | 53,324 | 50,988 | 252 | 1,849 | 5,709 |
| mxfp4 | 5.41 | 55,374 | 54,057 | 239 | 1,741 | 5,524 |

Against bf16:

| format | decode B=1 | B=8 | B=32 | prefill 4x2048 |
|---|---|---|---|---|
| int4 | 2.64x | 2.43x | 1.79x | 0.77x |
| int4-asym, int4 lm_head | 3.04x | 2.75x | 1.86x | 0.59x |
| fp8 | 1.61x | 1.57x | 1.49x | 1.62x |
| nvfp4 | 2.19x | 2.10x | 1.89x | 3.14x |
| nvfp4, nvfp4 lm_head | 2.46x | 2.34x | 2.04x | 3.15x |

### Is int4 decode faster end to end? Yes, 2.6x at batch 1

A bf16 decode step is 9.77 ms, a symmetric int4 one 3.70 ms. The step at batch 1, from
`--profile` (one eager step, GPU time per kernel family, ms):

| | bf16 | int4 | fp8 | nvfp4 |
|---|---|---|---|---|
| projection GEMMs | 8.63 | 2.46 | 4.46 | 2.58 |
| lm_head (bf16 hgemm) | 0.67 | 0.68 | 0.67 | 0.68 |
| activation quantize | | | 0.28 + 0.19 | 0.53 + 0.16 |
| norms, attention, RoPE, SwiGLU | 0.57 | 0.57 | 0.58 | 0.53 |
| total | 9.87 | 3.70 | 6.18 | 4.48 |

(The bf16 projection row is the hgemm total less lm_head; the second quantize number is the
fill kernel that zeroes the amax cell, which torch reports as elementwise.)

Every projection GEMM here streams its weights at the copy roof: 3.60 GB of int4 codes and
scales in 2.46 ms is 1.47 TB/s, 6.98 GB of fp8 in 4.46 ms is 1.57 TB/s, 3.93 GB of NVFP4 in
2.58 ms is 1.52 TB/s, against 1.53 TB/s for `cudaMemcpy`. So decode speed is the weight bytes,
and the formats line up by bits per weight: int4 (4.125) ahead of nvfp4 (4.5 plus its
activation quantizer) ahead of fp8 (8). After the projections, lm_head is the next largest
item at 0.67 ms, 18% of the int4 step. In int4-asym it is 0.17 ms, and the int4-asym step
drops from 3.73 to 3.21 ms.

The int4 lead shrinks with the batch, 2.64x at B=1 to 1.79x at B=32: w4gemm's M = 32 block
shape is slower per weight byte than its M = 1 one (the crossover section of
[w4gemm](w4gemm.md)). The nvfp4 step grows less with the batch (4.46 to 6.04 ms from B=1 to
B=32, against 3.70 to 6.39 for int4), and at B=32 nvfp4 passes int4 (5,297 against 5,008
tok/s).

### Where fp8 and nvfp4 prefill land

| 2048-token prefill, ms | bf16 | int4 | fp8 | nvfp4 |
|---|---|---|---|---|
| projection GEMMs | 118.2 | 144.6 | 60.9 | 23.3 |
| as TFLOPS (28.6 TFLOP of projections) | 242 | 198 | 470 | 1,227 |
| lm_head (bf16, the last row) | 0.7 | 0.7 | 0.7 | 0.7 |
| attention | 6.2 | 6.2 | 6.3 | 6.2 |
| SwiGLU | (fused) | 3.3 | 3.0 | 2.9 |
| activation quantize | | | 1.9 | 2.8 |
| norms, RoPE, rest | 1.3 | 2.0 | 2.1 | 2.0 |
| total | 126.4 | 156.7 | 74.9 | 37.9 |

fp8 halves the GEMM time and the prefill runs 1.62x faster end to end. NVFP4 cuts the GEMMs
5.1x, and the prefill runs 3.1 to 3.3x faster; at that point the GEMMs are 61% of the time and
attention (still bf16) is 16%, the SwiGLU launch 8% and the quantizer 7.5%. The next prefill
wins for nvfp4 are not in the GEMM: an fp8 attention prefill, and the SwiGLU and the next
quantization fused into fp4gemm's epilogue.

int4 prefill is slower than bf16 (0.77x) and asymmetric int4 slower still (0.59x): w4gemm is a
decode kernel, behind hgemm from M = 128 up, and the zero points cost the large-M tiles more.
A server would dequantize the int4 weights of a layer into a bf16 scratch buffer and run hgemm
for prefill chunks above the crossover; I did not build it.

### The activation quantizers matter at decode

My first fp8 path quantized activations with torch ops: an amax reduction, then the scale and
the cast, about ten launches per projection. At decode that was 1.8 ms of a 7.5 ms step, a
quarter of it, for 128 tiny activations. `fp8_quantize` does it in two launches per tensor
scale (an amax pass with one atomicMax per block, then the conversion) or one per row or MX
block. The measured change, same runs otherwise:

| | decode B=1 ms/step, before | after | prefill 4x2048 tok/s, before | after |
|---|---|---|---|---|
| fp8 | 7.35 | 6.07 | 24,139 | 26,221 |
| fp8-tok | 7.71 | 6.27 | 21,829 | 23,776 |
| mxfp8 | 8.04 | 6.04 | 22,015 | 26,819 |
| nvfp4 (tensor scale on the device) | 4.90 | 4.46 | 46,295 | 50,809 |

The NVFP4 quantizer was already a kernel, but its per-tensor scale was five torch ops in the
binding. It now comes from the same amax pass, and the quantize kernel derives the scale from
it (same arithmetic, the parity tests still compare bytes).

fp8-tok keeps a cost: fp8gemm takes per-tensor scales only, so the row scales multiply the
bf16 output afterwards, 8.8 ms of torch elementwise on a 2048-token prefill. For a 0.002
perplexity gain I would not pay it; a per-row scale in fp8gemm's epilogue would make it free.

### A CUDA graph bug the engine found

The first int4 benchmark crashed with an illegal address at the B=32 decode. w4gemm keeps
one split-K workspace per process, `M x N` fp32, grown on demand, and growing it freed the old
buffer. The engine captures its decode graphs first (small workspace), then a prefill grows it
to `8192 x 28672` floats, and every captured graph still launches with the freed pointer.
B=1 and B=8 happened to survive because nothing reused that memory yet. The fix keeps an
outgrown buffer alive (growth doubles, so what is kept is at most the size of the live
buffer). `test_w4gemm_graph_survives_workspace_growth` reproduces it: on the old build it hits
the illegal address, on the new one it passes. The prefill-sized workspace is 0.94 GB that
the int4 model holds next to its weights; a workspace sized to the split tiles only would
remove it.

## Recommendations from these numbers

- fp8 is the default to ship on this card: +0.03 perplexity, 1.6x prefill, 1.6x decode,
  43% less weight memory.
- For memory or batch-1 latency, int4-asym with AWQ scales: +0.27 perplexity, 2.6x decode,
  but prefill should run on dequantized weights.
- NVFP4 is the prefill format: 3.1x, and 1.9 to 2.2x decode, at +0.69 with round-to-nearest
  weights. The same activation-aware scaling, or a calibrated NVFP4 weight recipe, is the
  next thing to try on it.

## Reproduce

```
python scripts/eval_ppl.py ~/models/llama3-8b-instruct --save-ref ref_bf16.pt
python scripts/eval_ppl.py MODEL --format int4 --ref ref_bf16.pt
python scripts/awq_search.py MODEL --text wikitext2_train.txt --asym --out awq_asym.pt
python scripts/eval_ppl.py MODEL --format int4-asym --awq awq_asym.pt --ref ref_bf16.pt
python scripts/bench_quant.py MODEL --format nvfp4 --head-format nvfp4 --profile
```
