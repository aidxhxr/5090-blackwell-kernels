# Inkling 2 Mini decode: what a token costs on the cards this repo targets

What one decode step of Thinking Machines' Inkling 2 Mini reads from DRAM, the tokens/s that
puts as a ceiling on the RTX 5090, the RTX PRO 6000 and the GB10 (DGX Spark) in three weight
formats, and what its dense GEMMs measure on the RTX PRO 6000 with the kernels already in this
repo. The cost model is `scripts/inkling_roofline.py` (`--help`; tests in
`tests/test_inkling_roofline.py`), the GEMM shapes are the `inkling2-mini` rows of
`src/bench/model_shapes.hpp`, and the measured rows are `results/inkling_rtxpro6000.jsonl`.

**What is measured and what is not.** The bandwidth of the RTX PRO 6000 and every GEMM row
below were measured on loon (RTX PRO 6000 Blackwell Server Edition, sm_120, 188 SMs, 128 MB L2,
ECC on) on 2026-11-26. The ceilings are byte counts over a bandwidth: nothing here ran the
model. The RTX 5090 bandwidth is the one measured earlier (docs/RTX5090.md); the 5090 was down
this round. The GB10 has no measured bandwidth in this repo, so its rows use the 273 GB/s spec
and are optimistic.

## The model in bytes

Hidden 2304, 24 stored layers, 24 query heads and 8 K/V heads of 128 in every layer, 128
experts of 2304 -> 1152 -> 2304 with 8 picked per token, a 201024-row embedding and a
200058-row lm_head. Layers 6-17 run three times per token on shared weights, so a token goes
through 48 layer applications, and the K/V cache has 48 layers (layers 11 and 17, full
attention, are three of them each: 8 full cache layers, 40 sliding ones).

| piece | elements | bf16 | checkpoint | all fp4 |
|---|---|---|---|---|
| one expert (gate_up 2304 x 2304, down 2304 x 1152) | 7.96M | 15.9 MB | 4.48 MB (NVFP4), 7.97 MB (MXFP8, layers 20-23) | 4.48 MB |
| dense weights of a sliding layer (q, k, v, gate, o, sinks, relative state, router, convs, norms) | 27.2M | 54.5 MB | 54.5 MB | 15.4 MB |
| message gates of a loop layer (4 x (64 x 4608 + 2 x 2304 x 64)) | 2.36M | 4.7 MB | 4.7 MB | 1.3 MB |
| lm_head | 461M | 922 MB | 922 MB | 259 MB |
| text decoder, stored once | 26.07B | 52.1 GB | 18.8 GB | 15.3 GB |
| K/V per sequence at 4K / 32K / 128K (bf16) | | 0.30 / 1.24 / 4.46 GB | | |

NVFP4 is counted at 0.5625 bytes a weight (e2m1 and an e4m3 scale per 16) and MXFP8 at 1 + 1/1024
(a ue8m0 scale per 32 x 32 block). The checkpoint row adds up to 18.8 GB of text decoder; with
the 318M-parameter vision encoder in bf16 that is 19.4 GB, within 0.7% of the 19.53 GB the
safetensors index gives (I did not chase the difference).

The model card's numbers, reconciled. 26.4B total is the text decoder's 26,068,156,416 (the
reference model's own count, which the script reproduces exactly) plus the 318M of the vision
encoder. 3.4B active is the usual total minus the unpicked experts, 26.39B - 22.93B = 3.45B,
vision included; the text decoder alone is 3.14B. 5.4B "weight activations per token to account
for looping" is the text decoder's 3.14B plus 24 more layer applications of 93.3M each (dense
weights, message gates and 8 experts): 5.376B. That count includes the message gates in all three passes; they
only run in passes 2 and 3, which is 28M fewer weights.

## The 614 GB/s check

The blog plots "a theoretical ceiling at 614 GB/s with BF16 weights and batch size one,
including repeated reads across loops" (it does not say which device has 614 GB/s). 614 GB/s
over 5.4B x 2 bytes is **56.9 tokens/s**, and over the exact 5.376B it is 57.1. The script
reproduces that (`--blog`). The bytes a step actually reads at B = 1 are lower, 9.78 GB, because
the embedding is a gather of one 4.6 KB row and not a 926 MB stream: that is 62.8 tokens/s at
614 GB/s. Everything else in the 5.4B (the loop's re-reads of 24 layer applications, lm_head)
is real traffic.

## One step, by part

B sequences at context L. Experts per layer application are the expected distinct experts
under uniform routing, 128 (1 - (1 - 8/128)^B): 8 at B = 1, 51.6 at B = 8, 125.9 at B = 64.
Each pass routes on its own hidden state, so the passes of a loop layer read their own sets.
At B = 1, L = 4096, in MB:

| format | dense | loop re-reads | experts | K/V | conv states | block states | total |
|---|---|---|---|---|---|---|---|
| bf16 | 2,288 | 1,366 | 6,115 | 302 | 8 | 4 | 10,083 |
| checkpoint | 2,288 | 1,366 | 1,832 | 302 | 8 | 4 | 5,799 |
| all fp4 | 646 | 386 | 1,720 | 302 | 8 | 4 | 3,066 |

"dense" is every non-expert weight read once (lm_head is 922 MB of it in bf16), "loop re-reads"
is passes 2 and 3 of layers 6-17 reading their dense weights again. In the checkpoint format,
the bf16 dense weights and their re-reads are 63% of the step: the NVFP4 experts are already
the smaller part. The conv states (6,656 channels x 3 positions x fp32, read and written, per
application) and the block-state reads of the attention residuals (380 vectors of 2304 fp32 per
token) are activations; they scale with B and stay under 1.5% of the step at B = 64.

## Ceilings per card and format

Tokens/s = B / (bytes / bandwidth). Bandwidth is bench_bandwidth's copy roof: 1,532 GB/s on the
RTX 5090, 1,456 GB/s on the RTX PRO 6000 (measured 2026-11-26, ECC on), 273 GB/s spec on the
GB10. "no fit" means the weights plus B sequences of K/V do not fit in the card's memory (32,
96 and 128 GB).

| card (GB/s) | format | B=1, L=4K | B=1, L=32K | B=1, L=128K | B=8, L=4K | B=8, L=32K | B=64, L=4K | B=64, L=32K |
|---|---|---|---|---|---|---|---|---|
| RTX 5090 (1,532) | bf16 | no fit | no fit | no fit | no fit | no fit | no fit | no fit |
| RTX 5090 (1,532) | checkpoint | 264 | 227 | 154 | 682 | 481 | 1,866 (no fit) | 870 (no fit) |
| RTX 5090 (1,532) | all fp4 | 500 | 383 | 212 | 837 | 553 | 2,036 (no fit) | 905 (no fit) |
| RTX PRO 6000 (1,456) | bf16 | 144 | 132 | 102 | 255 | 219 | 777 | 517 (no fit) |
| RTX PRO 6000 (1,456) | checkpoint | 251 | 216 | 146 | 648 | 457 | 1,773 | 827 (no fit) |
| RTX PRO 6000 (1,456) | all fp4 | 475 | 364 | 201 | 796 | 526 | 1,935 | 860 |
| GB10 (273, spec) | bf16 | 27 | 25 | 19 | 48 | 41 | 146 | 97 (no fit) |
| GB10 (273, spec) | checkpoint | 47 | 41 | 27 | 121 | 86 | 333 | 155 |
| GB10 (273, spec) | all fp4 | 89 | 68 | 38 | 149 | 99 | 363 | 161 |

`python scripts/inkling_roofline.py` prints the same table with the bytes by part and the
memory each row needs. Three things it says:

- At B = 1 and short context the weights decide. Going from the checkpoint to all fp4 is 1.9x
  on every card, nearly all of it from the dense weights and lm_head, which the checkpoint keeps
  in bf16.
- Past 32K the K/V cache of the 8 full-attention cache layers takes over: at 128K one sequence
  reads 4.46 GB of K/V per step, more than all the fp4 weights. The sliding layers are capped at
  1024 positions and stop growing at L = 1024.
- At B = 64 nearly every expert is touched in every application (125.9 of 128), so the expert
  bytes stop growing with B and tokens/s rises almost linearly until the K/V cache fills
  memory. On the 5090 that happens before B = 64 at 4K in the checkpoint format.

## L2: what fits, and what pinning buys

The banner of every bench prints the L2: 96 MB on the 5090, 128 MB on the RTX PRO 6000, 24 MB
on the GB10. The three passes of a loop layer are 11 layer applications apart, and one
application reads 182 MB (bf16), 90 MB (checkpoint) or 51 MB (fp4) of weights at B = 1, so by
itself nothing stays in L2 between passes. With an L2 persistence window holding the dense
weights of some loop layers (`--pin`, 3/4 of L2 assumed usable), passes 2 and 3 of those layers
read them from L2:

| card | bf16 or checkpoint (59 MB a layer) | all fp4 (16.7 MB a layer) |
|---|---|---|
| RTX 5090 (72 MB usable) | 1 layer: 1,366 -> 1,252 MB of re-reads, 264 -> 269 tokens/s | 4 layers: 386 -> 258 MB, 500 -> 522 |
| RTX PRO 6000 (96 MB usable) | 1 layer: 251 -> 256 tokens/s | 6 layers: 386 -> 193 MB, 475 -> 507 |
| GB10 (18 MB usable) | none | 1 layer: 89 -> 90 tokens/s |

(checkpoint and fp4 at B = 1, L = 4K.) A few percent at most. Pinning experts does not help
either: at B = 1 the three passes of a layer pick 24 experts and share only 1.5 of them in
expectation (22.5 distinct), and at B = 64, where the passes do share almost everything, a
layer's 128 experts are 573 MB in fp4. The loop's re-reads are real DRAM traffic on all three
cards.

## Measured: Inkling's GEMMs on the RTX PRO 6000

`bench_hgemm --models=inkling --ms=1,8,64,4096`, `bench_fp8gemm --models=inkling` at the same
Ms, and `bench_fp4gemm --format=nvfp4 --m --n --k` per shape (it has no model route), each
under the GPU lock. Every row is checked against cuBLAS or cuBLASLt on the device and all
pass; the fp4 rows are bit-identical to cuBLASLt. Decode rows (M <= 64) rotate the weight over
copies that exceed L2, so they stream from DRAM. Times are medians of single launches between
two events. bf16 is variant 6 (the default), fp8 variant 3 (the default), fp4 the best variant
(v1 at decode). "n/a": no variant runs the shape (N below 64 or not a multiple of 64 for fp8
and fp4, N not a multiple of 16 for bf16, K = 1152 not a multiple of 256 for fp4). The cuBLAS
times of the shapes no variant runs come from the bench's stderr line, which times cuBLAS on
every shape.

M = 1, microseconds:

| weight (N x K) | bf16 v6 | cuBLAS | fp8 v3 | cuBLASLt fp8 | NVFP4 | cuBLASLt NVFP4 |
|---|---|---|---|---|---|---|
| q_proj / gate_proj (3072 x 2304) | 18.6 | 22.7 | 17.9 | 15.8 | 13.1 | 24.5 |
| k_proj / v_proj (1024 x 2304) | 14.1 | 17.3 | 13.8 | 13.4 | 8.9 | 23.8 |
| o_proj (2304 x 3072) | 20.5 | 22.5 | 17.8 | 15.8 | 10.6 | 26.1 |
| relative state (384 x 2304) | 11.4 | 15.3 | 13.7 | 12.3 | 8.6 | 22.9 |
| sink_proj (24 x 2304) | n/a | 14.2 | n/a | n/a | n/a | n/a |
| dynamic conv taps (32 x 2304) | n/a | 13.4 | n/a | n/a | n/a | n/a |
| message-gate down (64 x 4608) | 10.2 | 15.4 | 10.6 | 12.0 | 8.7 | 30.3 |
| expert gate_up (2304 x 2304) | 18.3 | 20.1 | 17.1 | 14.3 | 10.6 | 23.9 |
| expert down (2304 x 1152) | 14.3 | 17.4 | 15.2 | 12.1 | n/a | n/a |
| lm_head (200058 x 2304) | n/a | 851 | n/a | n/a | n/a | n/a |
| lm_head padded (201024 x 2304) | 691 | 633 | 311 | 352 | 178 | 266 |

M = 64, microseconds:

| weight (N x K) | bf16 v6 | cuBLAS | fp8 v3 | cuBLASLt fp8 | NVFP4 | cuBLASLt NVFP4 |
|---|---|---|---|---|---|---|
| q_proj / gate_proj (3072 x 2304) | 18.5 | 21.4 | 13.0 | 15.0 | 10.6 | 23.7 |
| k_proj / v_proj (1024 x 2304) | 15.5 | 17.2 | 17.2 | 13.0 | 11.1 | 23.8 |
| o_proj (2304 x 3072) | 20.4 | 24.4 | 13.6 | 16.6 | 13.2 | 27.6 |
| relative state (384 x 2304) | 12.6 | 15.4 | 15.0 | 12.0 | 10.7 | 23.0 |
| message-gate down (64 x 4608) | 12.3 | 15.4 | 15.0 | 12.0 | 10.1 | 28.7 |
| expert gate_up (2304 x 2304) | 18.2 | 21.5 | 11.9 | 16.1 | 10.7 | 23.9 |
| expert down (2304 x 1152) | 14.2 | 18.3 | 9.9 | 12.9 | n/a | n/a |
| lm_head padded (201024 x 2304) | 657 | 670 | 340 | 348 | 204 | 214 |

M = 8 is within 2 us of M = 1 on every weight (in the results file).

M = 4096 (prefill), TFLOPS:

| weight (N x K) | bf16 v6 | cuBLAS | fp8 v3 | cuBLASLt fp8 | NVFP4 | cuBLASLt NVFP4 |
|---|---|---|---|---|---|---|
| q_proj / gate_proj (3072 x 2304) | 358 | 349 | 612 | 606 | 818 (v2) | 918 |
| k_proj / v_proj (1024 x 2304) | 310 | 314 | 501 | 514 | 654 (v2) | 561 |
| o_proj (2304 x 3072) | 356 | 347 | 624 | 612 | 893 (v2) | 889 |
| relative state (384 x 2304) | 197 | 204 | 275 | 315 | 316 (v1) | 315 |
| message-gate down (64 x 4608) | 102 | 98 | 122 | 129 | 163 (v1) | 78 |
| expert gate_up (2304 x 2304) | 345 | 332 | 584 | 584 | 770 (v1) | 800 |
| expert down (2304 x 1152) | 296 | 304 | 456 | 475 | n/a | n/a |
| lm_head (200058 x 2304) | n/a | 364 | n/a | n/a | n/a | n/a |
| lm_head padded (201024 x 2304) | 298 | 404 | 575 | 721 | 307 (v1) | 1,290 |

Fused, the projections that share an input, at M = 1 / 8 / 64 (microseconds, `bench_hgemm
--variant=6` and `bench_fp4gemm --variant=1`):

| fused weight (N x K) | bf16 v6 | NVFP4 v1 | the separate pieces at M = 1 (bf16, NVFP4) |
|---|---|---|---|
| q, k, v, gate, relative state, sinks (8640 x 2304, 8600 padded to 64) | 33.9 / 33.4 / 35.3 | 16.0 / 14.2 / 14.8 | 91.0, 52.6 + sinks on cuBLAS |
| q, gate, relative state, sinks (6592 x 2304) | 27.3 / 27.2 / 28.4 | 12.6 / 12.7 / 13.0 | 62.8, 34.8 + sinks |

What the rows say:

- **At decode M, a single small GEMM is a launch, not a weight.** The 14 MB bf16 q_proj takes
  18.6 us, 763 GB/s, half the copy roof; the 1.8 MB relative state takes 11.4 us; nothing
  below 8.6 us was measured in any format. In NVFP4 the q_proj weight is 4 MB, a 2.7 us read at
  1,456 GB/s, and it takes 13.1 us. Only lm_head is big enough to reach the roof (178 us, 1,463
  GB/s in NVFP4; 311 us, 1,491 GB/s in fp8).
- **Fusing the projections that share an input pays.** q, k, v, gate, relative state and sinks
  all read the same normed hidden state in the sliding layers and in pass 1 of the loop: one
  8640 x 2304 bf16 GEMM is 33.9 us, 1,176 GB/s, against 91 us for the six launches. In passes 2
  and 3, k and v get their own message-gated inputs, and the 6592-row fusion without them is
  27.3 us.
- **Pad lm_head to 201024.** cuBLAS takes 851 us on the 200058-row bf16 lm_head and 633 us on
  the padded one (the embedding is already 201024 rows), and no variant here runs 200058 rows.
  The fp4 TMA variant also wants N a multiple of its 128-column tile, which 201024 is not
  (201088 is): our v1 reaches 307 TFLOPS at M = 4096 against cuBLASLt's 1,290.
- **Expert down needs K = 1152 in fp4.** fp4gemm wants K a multiple of 256; 1152 is 4.5 of
  them. The checkpoint's experts are NVFP4 in 20 of 24 layers, so the MoE kernel needs either
  a K tail or padding to 1280 (11% more bytes).
- At M = 4096 bf16 and fp8 are at cuBLAS on the projection and expert shapes, as on the 5090,
  but not on lm_head (298 against 404 TFLOPS in bf16, 575 against 721 in fp8); NVFP4 v2 is at
  89 to 117% of cuBLASLt where its tiles fit.

## Where the time per token should go

Take the checkpoint format at B = 1, L = 4K on the RTX PRO 6000: the byte floor is 3.98 ms
(251 tokens/s). Of that, 2.51 ms is the bf16 dense weights with the loop's re-reads, 1.26 ms
the NVFP4 and MXFP8 experts, 0.21 ms the K/V cache. Measured against that:

- Run as separate launches at the M = 1 times above, the projections of one application (q,
  gate, k, v, o, relative state, sinks) take 111.5 us: 5.4 ms over 48 applications, against
  1.8 ms for their bytes, and more than the whole step's floor before the router, the message
  gates (24 x 4 down projections at 10 us each, plus their up projections) and lm_head (633 to
  851 us). One fused projection plus o_proj is 54 us where q, k and v share an input and 76 us
  where k and v are apart (passes 2 and 3, and the full layers, whose q, k and v come from three
  residual mixes): 20 x 54 + 28 x 76 us = 3.2 ms.
- So the first job is not a faster GEMM but fewer launches: one fused input projection per
  application, the router and the conv taps folded in, the attention-residual mix, the message
  gates and the norms fused into the kernels next to them, and the step in a CUDA graph. With
  about ten kernels per application and 48 applications, 5 us of fixed cost per kernel is 2.4
  ms, more than half the floor. Nothing here measured a graph; the single-launch times above
  include launch overhead a graph partly hides.
- The second job is the width of the dense weights. The checkpoint keeps 3.65 GB of dense
  weights and re-reads per step in bf16, 63% of its bytes. fp8 dense weights would take 1.8 GB
  off (about 365 tokens/s on this card), fp4 2.6 GB (the all-fp4 row, 475). Quantizing lm_head
  alone saves 0.46 GB (fp8) or 0.66 GB (fp4) per step.
- At long context the 8 full-attention cache layers dominate: at 128K the K/V read is 4.46 GB
  per sequence, and an fp8 cache halves it. At B = 64 the experts are read whole every
  application (125.9 of 128), so the grouped expert GEMM's bandwidth at M = 64 / 128 is what
  matters, and the K/V capacity decides how far B can go.
- The GB10 is 5.3x slower than the PRO 6000 on bytes alone and has a 24 MB L2, which holds one
  fp4 loop layer's dense weights and nothing else. There the launch overhead is a smaller share
  of a 21 ms (checkpoint) step, and the weight width is everything.

## Reproducing

    python scripts/inkling_roofline.py                       # every card, format, B and L
    python scripts/inkling_roofline.py --blog                # the 614 GB/s check
    python scripts/inkling_roofline.py --card pro6000 --format ckpt --batch 1 --context 4096 --pin
    ./bench_bandwidth
    ./bench_hgemm --models=inkling --ms=1,8,64,4096
    ./bench_fp8gemm --models=inkling --ms=1,8,64,4096
    ./bench_fp4gemm --m=1 --n=3072 --k=2304 --format=nvfp4 --quant=0   # one shape per run

Not done here: the router (128 x 2304) and the message-gate up projections (2304 x 64) were
not benched, the grouped expert GEMM was not run on Inkling's shapes (the inkling-moe lane), no
GEMM ran on the 5090 or the GB10 this round, and the ceilings ignore attention math, the
softmaxes of the residual mixes and the sampling, which are small in bytes but are kernels.
