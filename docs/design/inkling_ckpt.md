# Inkling 2 Mini checkpoint: names, formats, and the experts in fp4gemm / fp8gemm layouts

Thinking Machines ship Inkling 2 Mini (`thinkingmachines/Inkling-2-Mini`) as one 19.5 GB
safetensors file in its training-side names (`model.llm.*`, `model.visual.*`), with the MoE
experts already quantized: NVFP4 in layers 0-19, MXFP8 in layers 20-23. This note is what is
in the file, how its names map to transformers' `Inkling2LoopForCausalLM`, and how the
experts become operands of this repo's block-scaled GEMMs without changing a code or a scale.

Source: `scripts/inkling_inspect.py` (the header, from a file or a URL),
`python/spark_kernels/inkling_ckpt.py` (name map, reader, expert repack),
`scripts/inkling_ckpt_check.py` (the real-file check and GEMM errors below),
`tests/test_inkling_ckpt.py`.

## What the file holds

The header is 148,408 bytes; `inkling_inspect.py` reads it with two HTTP Range requests (the
8-byte length, then the JSON), so the table below came off the Hub without downloading the
file:

    python scripts/inkling_inspect.py \
        https://huggingface.co/thinkingmachines/Inkling-2-Mini/resolve/main/model-00001-of-00001.safetensors

| format | tensors | bytes | share |
|---|---|---|---|
| bf16 | 994 | 3.854 GB | 19.7 % |
| fp32 | 74 | 0.97 MB | 0.0 % |
| NVFP4 experts (layers 0-19) | 160 | 11.466 GB | 58.7 % |
| MXFP8 experts (layers 20-23) | 16 | 4.204 GB | 21.5 % |
| all | 1244 | 19.526 GB | |

The bf16 part is the attention (1.308 GB over 24 layers), the embedding and the unembedding
(0.926 GB each, [201024, 2304]), the vision tower (0.637 GB), the message gates of the core
layers (42 MB), the routers (14 MB) and the norms, conv taps and queries. fp32 is the block
attention residual queries, the message queries and the router correction biases.

Per expert projection, E = 128 experts, H = 2304, I = 1152:

| tensor | NVFP4 (layers 0-19) | MXFP8 (layers 20-23) |
|---|---|---|
| `w13_weight` | U8 [128, 2304, 1152], two e2m1 a byte | F8_E4M3 [128, 2304, 2304] |
| `w13_weight_scale` | F8_E4M3 [128, 2304, 144], per 16 k | U8 [128, 2304, 72], E8M0 per 32 k |
| `w2_weight` | U8 [128, 2304, 576] | F8_E4M3 [128, 2304, 1152] |
| `w2_weight_scale` | F8_E4M3 [128, 2304, 72] | U8 [128, 2304, 36] |
| `*_weight_amax` | F32 [128], one per expert | |
| `*_weight_outer_scale_denominator` | F32 [1], 2688 = 6 x 448 | |
| bytes per layer | 573 MB (0.5625 B per weight) | 1051 MB (1.03125 B per weight) |

w13 is [2I, H] per expert, gate and up stacked; w2 is [H, I]. Both are nn.Linear's
[out, in], K contiguous, which is the `b_t` layout fp4gemm and fp8gemm take.

The layer map the script prints shows the rest of the model's structure in the names:
full-attention layers (5, 11, 17, 23) carry `k_sconv.static_w` / `weight_proj` and the
q/k/v input norms in place of `attn_norm`; core layers (6-17) carry the `controllers.*`
message gates. `inkling_ckpt.raw_names()` rebuilds the 822 `model.llm.*` names from those
rules, and on the real file it equals the header's set.

## The name map

`inkling_ckpt.hf_name(raw)` is transformers 5.24's renaming chain for this model (model types
`inkling2_loop_text` then `inkling2_text` in `conversion_mapping.py`) written out as one table.
The test runs every raw name through transformers' own `rename_source_key` with the model's
conversion list and checks the two agree, and that the mapped set is exactly the meta model's
state dict. Inside `model.llm.layers.L.` (HF `model.layers.L.`):

| raw | HF |
|---|---|
| `attn.wq_du`, `wk_dv`, `wv_dv`, `wo_ud` | `self_attn.{q,k,v,o}_proj.linear` |
| `attn.gate_attn` | `self_attn.o_proj.gate_proj` |
| `attn.attn_sink_proj` | `self_attn.sink_proj` |
| `attn.q_l2_clip.scale` | `self_attn.q_proj.norm.scale` |
| `attn.wr_du`, `rel_l2_clip.scale`, `rel_logits_proj.proj` | `self_attn.relative_position_bias.{linear, norm.scale, proj}` |
| `attn.{k,v}_sconv.weight` (sliding) | `self_attn.{k,v}_proj.sconv.conv1d.weight` |
| `attn.{k,v}_sconv.static_w`, `weight_proj` (full) | `self_attn.{k,v}_proj.sconv.static_weight`, `weight_proj.weight` |
| `attn_norm`, `mlp_norm` | `input_layernorm`, `post_attention_layernorm` |
| `{q,k,v}_input_norm` (full) | `{q,k,v}_input_layernorm` |
| `attn_sconv`, `mlp_sconv` | `post_attention_convolution.conv1d`, `post_mlp_convolution.conv1d` |
| `controllers.{attn,k,v,mlp}.{down,gain,forget}_weight` | `{q,k,v,mlp}_message_gate.{down_proj, message_gate_proj, forget_gate_proj}.weight` |
| `mlp.gate.weight`, `mlp.gate.bias` | `mlp.gate.weight`, `mlp.gate.e_score_correction_bias` |
| `mlp.experts.w13_weight*` | `mlp.experts.gate_up_proj` |
| `mlp.experts.w2_weight*` | `mlp.experts.down_proj` |

Outside the layers: `embed` -> `model.embed_tokens`, `embed_norm` ->
`model.embed_tokens.embed_norm`, `norm` -> `model.norm`, `unembed` -> `lm_head`,
`attn_res.attn_queries.i` -> `model.layers.{i+1}.attention_input.query` (layer 0's attention
input is the embedding alone, so there are 23 of them), `attn_res.mlp_queries.i` ->
`model.layers.i.mlp_input.query`, `attn_res.output_query` -> `model.final_input.query`. The
vision tower maps to nothing in the text model.

Four HF parameters are not a copy of one raw tensor: `lm_head.weight` is the first 200058 of
the 201024 rows; `model.message_input.query` [12, 4, 2304] merges
`middle_cycle_cross_pass_queries` (query and MLP query of each core layer, interleaved) and
`..._kv_queries` (key and value) in the order q, k, v, mlp; and the two expert projections are
dequantized, w13 also deinterleaved. The file's w13 rows alternate gate and up (gate_j at row
2j, up_j at 2j + 1); transformers' `Interleave(dim=1)` turns that into [gate; up]. That
alternating order is the layout `ops.interleave_experts` builds for `grouped_gemm_swiglu`, so
a fused SwiGLU GEMM takes the file's rows as they are, and `gate_up_split` gives the halves
(on codes and block scales too, since every scale belongs to one row).

## NVFP4 experts: the file's format is already fp4gemm's

The "amax_denominator" scale format decodes as transformers' `Inkling2LoopDequantizeExperts`
does it:

    w[e, n, k] = e2m1(code) * e4m3(scale[e, n, k / 16]) * (amax[e] / denominator)

in fp32, in that order, then cast to bf16. The codes are two per byte with element 2k in the
low nibble, the same packing as `torch.float4_e2m1fn_x2` and this repo's `fp4_quantize`.
`denominator` is 2688 = 6 x 448 in all 20 layers, so `amax / 2688` is exactly the per-tensor
NVFP4 scale `fp4_quantize` computes (`max|x| / (6 * 448)`), here one per expert.
"fouroversix_mae" is how the quantizer chose each block scale (map the block maximum to 4 or
to 6, whichever gives the smaller error); it does not enter the decode. In expert 0 of layer 0
every block of w13 and w2 has a largest code of magnitude 6.

`repack_nvfp4` therefore changes no values: `b_t` is the packed bytes as stored, `sfb` is the
same e4m3 bytes put in the blocked layout by `reference.to_blocked` (one call per expert;
2304 rows and 144 or 80 scale columns need no padding), and `scale_b` is `amax[e] / 2688`
computed with the same fp32 division transformers does. The outer factor folds into
`scale_b` exactly, because fp4gemm multiplies the fp32 sum by `scale_a * scale_b` and
transformers multiplies each element by that same fp32 number.

Lossless, precisely: `dequantize_nvfp4` (codes times block scales from the blocked layout,
times `scale_b`, the way fp4gemm reads them) equals transformers' fp32 values bit for bit, and
so its bf16 cast equals what transformers loads. The one rounding in the format, the fp32
division `amax / 2688`, is in the format's definition, and both sides do it the same way.
The GEMM is not transformers' arithmetic: transformers rounds every weight to bf16 and runs a
bf16 matmul, fp4gemm keeps the exact e2m1 x e4m3 products and applies the factor once to the
fp32 sum. That difference is in the GEMM numbers below (about 2e-3), and it is the bf16
path that loses precision, not the repack.

## MXFP8 experts: the stored scales are already 1 x 32

`hf_quant_config.json` lists the MXFP8 group with `weight_block_shape [32, 32]`, so I expected
one scale per 32 x 32 tile to broadcast over its 32 rows. The file does not store that: the
scale tensor is [E, N, K / 32], one E8M0 byte per row per 32 k, and the rows differ (in expert
0 of layer 20, all 2304 scale rows of w13 are distinct, 2297 of w2's; no two adjacent rows
of w13 are equal, in the interleaved or the deinterleaved order). transformers decodes it the same
way, `e4m3 * 2 ** (scale - 127)` per 32 consecutive k of a row. That is fp8gemm's MX operand
(`sfb` [N, K / 32] uint8, row-major), so `repack_mxfp8` copies the bytes, and
`dequantize_mxfp8` (`reference.dequantize_mx`) equals transformers' values bit for bit. The
products are exact in fp32 (a power of two times an e4m3 value) and exact in bf16 too.

The block maxima of layer 20's w13, expert 0, range from 240 to 448 in units of the block
scale (8.4 % of blocks below 256, 11.7 % at 448). That fits the ceil rule
(`quantize_mx(mode="ceil")`, the largest element in (224, 448]), not the OCP floor rule,
which would put every block maximum at 256 or above.

## Shapes against the GEMMs

| projection | N | K | fp4gemm / fp8gemm MX (N % 64, K % 256) |
|---|---|---|---|
| w13 | 2304 | 2304 | takes it |
| w2 | 2304 | 1152 | K padded to 1280 |

The w2 padding appends zero codes (and, for NVFP4, zero block scales; for MXFP8 scale byte 127,
2 ** 0, so the zeros never meet an infinite scale). Zero times anything adds an exact zero to
the fp32 sum, and the activation is padded with zeros to match, so the padded GEMM is the
unpadded one. It costs 21 MB per NVFP4 layer and 39 MB per MXFP8 layer, 580 MB over the
model.

The grouped GEMM does not take either format. `grouped_gemm` and `grouped_gemm_swiglu` take
bf16, or e4m3 with one scale per expert. NVFP4 has no grouped path at all. For MXFP8 I checked
whether each expert fits one scale (rescale every block to the expert's largest block scale,
round to e4m3, compare): it does not. An expert's block scales span 2 to 5 binades, and
1.0e5 to 1.7e5 of the 6.8e8 (w13) or 3.4e8 (w2) nonzero elements per layer fall into e4m3's
subnormals and lose bits (layers 20 and 23). The lossless options today are a per-expert loop
of `fp4gemm` / `fp8gemm`, or dequantizing to bf16 for `grouped_gemm`. The bf16 copy is
48.9 GB for all experts, so it does not fit the 32 GB card this model is meant for.

## Measured

On the real file (RTX PRO 6000 (sm_120), `scripts/inkling_ckpt_check.py`), against
transformers' loader ops (`Inkling2LoopDequantizeExperts`, then `Interleave(dim=1)` for w13)
run on the same raw tensors:

| layer | format | projection | bf16 elements differing from transformers |
|---|---|---|---|
| 0 | NVFP4 | w13, all 128 experts | 0 of 679,477,248 |
| 0 | NVFP4 | w2 | 0 of 339,738,624 |
| 20 | MXFP8 | w13 | 0 of 679,477,248 |
| 20 | MXFP8 | w2 | 0 of 339,738,624 |
| 23 | MXFP8 | w13, w2 | 0 of 679,477,248, 0 of 339,738,624 |

The reader's bytes equal `safetensors.safe_open`'s on the first 16 experts of each.

One expert's GEMM, expert 0, M = 512 rows of Gaussian bf16 activations, the activation
quantized by this repo's kernels (`fp4_quantize`, one per-tensor scale; `fp8_quantize`
"mx"). Relative Frobenius error, then the largest error over the largest |reference|:

| GEMM | vs reference GEMM, same operands | vs quantized x (fp32) @ W_hf.T | vs bf16 x @ W_hf.T |
|---|---|---|---|
| fp4gemm, layer 0 w13 (K 2304) | 0 (bit equal) | 2.32e-3 / 3.47e-3 | 9.50e-2 / 1.00e-1 |
| fp4gemm, layer 0 w2 (K 1152 -> 1280) | 0 (bit equal) | 2.54e-3 / 3.65e-3 | 9.54e-2 / 9.76e-2 |
| fp8gemm MX, layer 20 w13 | 2.92e-5 / 3.01e-3 | 1.66e-3 / 2.94e-3 | 2.95e-2 / 2.86e-2 |
| fp8gemm MX, layer 20 w2 (K -> 1280) | 1.85e-5 / 3.31e-3 | 1.66e-3 / 3.31e-3 | 2.94e-2 / 3.12e-2 |

The first column is the kernel: fp4gemm gives the reference's bits, fp8gemm differs by the
bf16 roundings a different summation order flips. The second is the weight path: the bf16 rounding of
the output, plus for NVFP4 that of transformers' bf16 weight. The third is what a user of the
repacked weights sees against a bf16 matmul, and nearly all of it is the activation's
quantization: e2m1 on Gaussian values with a per-tensor scale, e4m3 for MXFP8.

## Gaps

- The model quantizes activations to NVFP4 per token (`amax_type per_token`,
  `fouroversix_mae`). `fp4gemm` takes one fp32 `scale_a` per tensor and `fp4_quantize` has
  neither the per-token scale nor the 4-or-6 block rule, so the 9.5e-2 above is the repo's
  activation recipe, not the model's.
- A block-scaled grouped GEMM (NVFP4 and MXFP8 operands, `offsets` like `grouped_gemm`) is
  what the MoE layer needs; the per-expert loop works now, one launch per active expert.
- The comparison with transformers runs its conversion ops on the raw tensors, the code its
  loader calls; I did not run `from_pretrained` on the whole model.
