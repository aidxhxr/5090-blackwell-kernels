# Inkling 2 Mini: the model as the kernels see it, and the reference they check against

Inkling 2 Mini (Thinking Machines, `thinkingmachines/Inkling-2-Mini`, Apache 2.0) is a
26.4B-total looped MoE: 24 stored layers, of which layers 6-17 run three times, so a token
goes through 48 layer applications. This note is the per-token op list with shapes, what in
it is new to this repo and where it lives, and how the pure-torch reference
(`python/spark_kernels/inkling.py`) compares with HF transformers on the real checkpoint.

Sources: transformers 5.24.1 `modeling_inkling2_loop.py` (the generated file, read in full),
the checkpoint's `config.json`, `hf_quant_config.json` and safetensors headers.

## The reference

`inkling.py` needs only torch. `InklingConfig.from_hf` reads `config.json`
(`text_config` and its `loop_config`); `InklingForCausalLM` has HF's state_dict keys, so
`load_state_dict(hf.state_dict(), strict=True)` works both ways; `InklingCache` holds the 48
cache layers for `decode_step`. One `DecoderLayer` class covers HF's four (sliding or full, in
the loop or not), and the casts follow HF's: the residual mixes, the convs, the router, the
message gates and every norm compute in fp32, the stream between layers is bf16.

Parameters (`InklingConfig.param_counts()`, from the shapes; HF's module on the meta device
gives the same total):

| part | parameters |
|---|---|
| experts, 24 layers x 128 x (2304 x 2304 + 2304 x 1152) | 24,461,180,928 |
| embedding (201024 x 2304, padded) + its norm | 463,161,600 |
| lm_head (200058 x 2304) | 460,933,632 |
| everything else (attention, convs, mixes, routers, message gates) | 682,880,256 |
| text model total | 26,068,156,416 |
| active per token (8 of 128 experts in each stored layer) | 3,135,799,296 |
| vision tower (HF module), not in the text model | 318,256,128 |
| MTP head (`mtp.safetensors`, HF does not load it) | 317,897,872 |

Text plus vision is 26,386,412,544, the published 26.4B. The published 3.4B active is the
text model's 3.14B plus about 0.32B, which is the size of the vision tower and, to 0.1%, of
the MTP head; the numbers do not say which. The loop adds no weights but a token routes 48
times, so a decode step can touch 48 x 8 expert slots, 3.06B expert parameters, if every pass
picks new experts.

## One token through the model

Shapes for one decode token, H = 2304. "mix" is the block attention residual: a softmax over
depth, per token, with learned queries (the same at every position), keys and values the
weightless-RMS-normalized states, scale 1/sqrt(2304), fp32. A block is 6 layers. The states
are the finished ones (the embedding, then the sum of each finished block: at most 4 of them,
in layers 19-23) plus the current partial block (the running sum of this block's layer
outputs).

### A sliding layer (20 of the 24: all but 5, 11, 17, 23)

| # | op | shape | notes |
|---|---|---|---|
| 1 | attention input mix | [n + 1, 2304] -> [2304] | 1 query, n finished blocks + the partial one; layer 0 skips it |
| 2 | RMSNorm (`input_layernorm`) | [2304] | weighted, fp32 -> bf16 |
| 3 | q = linear, x scale, L2 clip | 2304 -> 3072 = 24 x 128 | per-head scale [24, 128], clip to norm 22.627 |
| 4 | k = linear, short conv, L2 clip | 2304 -> 1024 = 8 x 128 | depthwise causal conv, 4 static taps per channel, (conv + x) / sqrt(5), fp32; state 3 x 1024 |
| 5 | v = linear, short conv, RMS norm | 2304 -> 1024 | same conv; norm per head without weight |
| 6 | relative state | 2304 -> 384 = 24 x 16 | x scale, L2 clip to 8 per head |
| 7 | bias per distance | [24, 16] x [16, 1024] -> [24, 1024] | one bias per backward distance 0..1023 |
| 8 | sinks | 2304 -> 24 | one extra logit per head with value 0 |
| 9 | attention | 24 q heads, 8 kv heads, 1024 keys | logits / sqrt(128) + bias[distance], window 1024 (distances 0..1023), sink in the softmax, no RoPE |
| 10 | output gate, o_proj | sigmoid(2304 -> 3072) x attn, 3072 -> 2304 | the gate reads the query input |
| 11 | post-attention conv | depthwise, 2304 channels, k = 4 | (conv + x) / sqrt(5), fp32; state 3 x 2304 |
| 12 | partial block | + | or, at a block boundary (layers 0, 6, 12, 18), the layer input becomes a finished block and the partial restarts here |
| 13 | MLP input mix | [n + 1, 2304] -> [2304] | 1 query |
| 14 | RMSNorm (`post_attention_layernorm`) | [2304] | |
| 15 | router | 2304 -> 128, fp32 | sigmoid; top 8 of score + bias; weights = the 8 scores (no bias) / their sum x 8.0 |
| 16 | 8 experts | 8 x (2304 -> 2 x 1152, 1152 -> 2304) | softcap(g, 4) x sigmoid(g) x softcap(u, 25), weighted sum in fp32 |
| 17 | post-MLP conv | depthwise, 2304 channels, k = 4 | (conv + x) / sqrt(5); state 3 x 2304 |
| 18 | layer output | partial + MLP | bf16 |

### A full layer (5, 11, 17, 23)

The same, with:

- the attention input mix has 3 queries and gives separate q, k and v inputs, each with its
  own RMSNorm (`q_`, `k_`, `v_input_layernorm`);
- q and the relative state are scaled by tau = 1 + 0.1 log(max((pos + 1) / 8192, 1)) after
  the clip (exactly 1 up to position 8191);
- the k and v convs add per-token taps: `weight_proj` 2304 -> 32 (4 taps for each of the 8
  kv heads, shared by the head's 128 channels) on top of the static [4, 1024] taps,
  (conv + x) / sqrt(9);
- attention sees every earlier position, and the relative bias covers distances 0..1023
  (zero beyond).

### The loop (layers 6-17, three passes)

Pass 1 is the layers as above. Passes 2 and 3 restart from the prelude's output (the partial
block and block states after layer 5) and use their own 12 cache layers each. Before each
loop layer j of a pass:

| op | shape | notes |
|---|---|---|
| message read | [2, 2304] -> [4, 2304] | a mix with 4 queries (row j of `message_input.query` [12, 4, 2304]) over the previous pass's last finished block and its final partial block; the same two states for all 12 layers, so one [48, 2304] x [2304, 2] product per pass |
| 4 message gates (q, k, v, MLP inputs) | per gate: [2304; 2304] -> 64 -> 2 x 2304 | z = W_d [rms(h); rms(m)]; out = (1 - tanh(W_f z)) h + tanh(W_g z) m, fp32 |

In a sliding loop layer the q, k and v gates all read the one mixed input; in a full loop
layer each reads its own. The MLP gate goes after the MLP input mix.

### The logits

A last mix (1 query) over 5 states: the embedding, the blocks of layers 0-5, 6-11 and 12-17
(from pass 3) and the partial block of layers 18-23; RMSNorm; lm_head 2304 -> 200058 (bf16;
the embedding has 201024 rows, the logits do not); softcap 30 (fp32).

### What a sequence keeps (`InklingCache`)

48 cache layers: 40 sliding (keep the last 1024 positions) and 8 full (4 full layers plus
the pass 2 and 3 copies of 11 and 17). K and V after the conv and the clip or norm, bf16,
8 x 128 each: the full layers grow by 8 x 2 x 1024 x 2 B = 32 KiB per token, the sliding
ones stop at 40 x 2 x 1024 x 1024 x 2 B = 160 MiB per sequence. Conv states: 48 x 3 x
(1024 + 1024 + 2304 + 2304) x 4 B = 3.8 MB per sequence, fp32. Nothing else: the mixes and
the messages are over depth within one token, so the block states are rebuilt every step.

## New ops and where they live

| op | new? | where |
|---|---|---|
| attention with an additive relative bias by distance, per-query sinks from a projection, window 1024 | the window and per-head constant sinks exist (`ops.attention(window=, sinks=)`); the bias and per-query sinks are new | `src/kernels/inkling_attention.cu`, [inkling_attention.md](inkling_attention.md) |
| short causal depthwise conv, static taps and static + per-token taps per kv head, the (conv + x) / sqrt scale | new | `src/kernels/short_conv.cu`, [short_conv.md](short_conv.md) |
| q / k scaled L2 clip, log scaling tau, value RMS norm without weight | new (epilogues) | `src/kernels/short_conv.cu` |
| block attention residual mix (1 or 3 queries), the message read (4 queries), the rank-64 message gate | new | `src/kernels/attn_res.cu`, [attn_res.md](attn_res.md) |
| sigmoid router with a selection-only bias, renormalized x 8, top 8 of 128 | the top-k MoE exists (`moe_layer`); the router math is new | `src/kernels/moe_route.cu` |
| softcapped SwiGLU in the experts | SwiGLU exists; the two softcaps are new | `src/kernels/swiglu.cu`, `src/kernels/grouped_gemm.cu` |
| expert GEMMs, 128 experts, 2304 x 2304 and 1152 x 2304 | grouped GEMM exists; the shapes are new | `src/bench/model_shapes.hpp`, [inkling_decode.md](inkling_decode.md) |
| NVFP4 (e2m1, 16-blocks, e4m3 scales, amax / denominator) and MXFP8 expert weights | fp4gemm and fp8 MX exist; the checkpoint layout is new | `python/spark_kernels/inkling_ckpt.py`, [inkling_ckpt.md](inkling_ckpt.md) |
| logit softcap 30 in the loss and the sampler, 200058 logits | new | `src/kernels/cross_entropy.cu`, `src/kernels/sample.cu` |
| the loop: 48 applications, 12 layers' weights read 3 times | new to the cost model | `scripts/inkling_roofline.py`, [inkling_decode.md](inkling_decode.md) |
| dense GEMMs (q, k, v, gate, o, rel, sinks, router, lm_head), RMSNorm | exist | |

## The reference against HF on the real checkpoint

`scripts/inkling_ref_check.py` loads `Inkling2LoopForCausalLM` from the checkpoint in bf16
on the GPU, builds the reference on the meta device and gives it HF's own tensors
(`load_state_dict(strict=True, assign=True)`: every key matches and both run on the same
storage), then runs one prompt through both. HF's loader dequantizes the NVFP4 (layers 0-19)
and MXFP8 (layers 20-23) experts to bf16 on load (`Inkling2LoopDequantizeExperts` in
`core_model_loading.py`, "until these formats can be loaded quantized"), so both sides use
the same dequantized bf16 experts; nothing here tests the quantized formats.

Run on the RTX PRO 6000 (sm_120), torch 2.14+cu130, transformers 5.24.1, on a 31-token
English prompt (the default in the script). HF's load takes 110-135 s (on the CPU, then moved:
`device_map` needs accelerate) and holds 48.6 GiB of weights on the GPU.

Prefill logits over the 31 positions (fp32 logits, after the softcap; the reference side of
each row is the second one named):

| a vs b | max abs diff | mean abs diff | rel err median | top-1 agree | KL mean | KL max |
|---|---|---|---|---|---|---|
| reference vs HF (default: grouped_mm experts) | 4.69 | 0.205 | 4.75% | 93.5% (29/31) | 4.9e-3 | 1.6e-2 |
| HF eager experts vs HF grouped_mm | 4.69 | 0.205 | 4.75% | 93.5% (29/31) | 4.9e-3 | 1.6e-2 |
| reference vs HF eager experts | 0 | 0 | 0 | 100% | 0 | 0 |

The reference is bit-identical to HF with its per-expert loop (`set_experts_implementation(
"eager")`, the same GEMMs the reference runs): zero difference in the logits and in the
output of every one of the 48 applications (`--layers`). The whole difference against HF's
default comes from HF's own two expert paths, which disagree with each other by exactly as
much. 4.75% relative error and 2 of 31 argmax flips is a lot next to Llama 3 8B in bf16 (1.4%
and 99.2% between two bf16 stacks, `llm_parity.md`). The expert picks say why (`--routes`):
between HF's two paths, 262 of the 1488 (token, application) pairs pick a different set of 8
experts. The first one is at layer 5, and from the second pass of the loop on it is 3 to 12
of the 31 tokens in every application. Each flip changes that token's input to every later
layer, and there are 48 routed applications per token, so two bf16 stacks drift further apart
here than in a dense model. Which of the two is closer to an fp32 run I have not measured
(the 26B model in fp32 does not fit next to the other jobs on this GPU). Greedy decoding from
the prompt gives the same next token on every side.

Decode (`--decode 16`): 16 greedy tokens after the prompt (" want this, and why it is not
the same as a causal convolution over the"), the logits of those 16 positions four ways. Each
side decodes through its own cache after a prefill of the prompt into it (the reference's
`InklingCache`, HF's DynamicCache), and each prefills the whole 47-token sequence:

| a vs b | max abs diff | mean abs diff | rel err median | top-1 agree | KL mean |
|---|---|---|---|---|---|
| reference decode vs HF decode | 0 | 0 | 0 | 100% (16/16) | 0 |
| reference decode vs reference prefill | 1.39 | 0.174 | 3.71% | 100% | 3.5e-3 |
| HF decode vs HF prefill | 1.39 | 0.174 | 3.71% | 100% | 3.5e-3 |

The reference's decode is bit-identical to HF's, so the 48 cache layers, the conv states and
the sliding window line up with HF's cache exactly. Decode against prefill differs by the same
amount on both sides: one token at a time and 47 at once round differently in bf16. The cache
held 12.5 MiB after the 47 tokens (48 x 2 x 8 x 128 x 2 B = 192 KiB of K and V per token,
plus 3.8 MB of conv states).

Wall times, after a warm-up, under the GPU lock (the GPU is shared, so other jobs' untimed
work can still overlap), batch 1. Prefill of the 31-token prompt in three runs: HF
(grouped_mm) 330, 403 and 352 ms, the reference 1021, 1028 and 1074 ms. One decode step at
31-47 tokens, one run: HF with its eager experts 395 ms, the reference 325 ms. Both are eager
PyTorch with a few thousand small launches per token (the decode cost model is
[inkling_decode.md](inkling_decode.md)); the reference is not meant to be fast.

## Tests

`tests/test_inkling_ref.py`, on the CPU (the Mac and loon): a 12-layer random model in fp32
that keeps every feature (blocks of 3, the loop over layers 3-8 three times, full layers in
the prelude, in the loop and after it, a window of 8 and a full-layer bias range of 6 below
the 21 test tokens, log scaling from position 3, clips that clip), random weights everywhere.
The prefill matches HF (eager experts) within 1e-4, decoding through our cache matches HF
decoding through its DynamicCache within 1e-4, and our decode, token by token and after a
prefill into the cache, matches our prefill within 2e-4 (fp32 rounding at logits of about
10). The real config's layout and counts are checked from the published config.json values.
The HF tests skip without a transformers that has Inkling2Loop (CI has none).

## What is not done

- No fp32 run of the real model, so which bf16 path is closer to the exact logits is open.
- Padding and attention masks are not supported; batches must be the same length.
- The quantized experts are not exercised: both sides use HF's dequantized bf16 copies.
- The MTP head and the vision tower are not modeled.
