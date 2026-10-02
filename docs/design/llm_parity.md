# Parity on real weights: the engine against transformers and an fp32 reference

The kernel tests check each op against a reference on random inputs. That says nothing about
whether 32 layers of them, on real Llama-3-8B-Instruct weights, produce the same model. This
note answers two questions with measurements:

1. How close is the engine (`engine.SparkModel`, bf16) to Hugging Face transformers (bf16,
   sdpa attention), the implementation people actually run?
2. Is the engine as close to the exact answer as transformers is? Two bf16 stacks always
   disagree a little, so the useful comparison is each of them against a higher-precision
   reference.

Short answer: the engine is closer to the fp32 reference than transformers is, on the logits,
at every layer and in perplexity. Greedy generations agree for a while and then split at a
near tie, and every split I found sits on a gap below one bf16 step of the logits.

Script: `scripts/llm_parity.py` (stages `hf`, `spark`, `torch`, `ref`, `report`, `ppl`),
`scripts/eval_ppl.py` for the perplexities. Results: `results/llm_parity.json`,
`results/llm_ppl.jsonl`. All on the RTX 5090, torch 2.14+cu130, transformers 5.18, model
`llama3-8b-instruct`.

## The four backends

| name | what | precision |
|---|---|---|
| spark | `engine.SparkModel`: this package's kernels through the engine's packed prefill | bf16 storage, fp32 accumulate and epilogues |
| torch | `engine.TorchModel`: the same model in plain PyTorch on the same weights | bf16 |
| hf | `LlamaForCausalLM`, `attn_implementation="sdpa"` | bf16 |
| ref | the same math in fp32 on the bf16 weights, TF32 off, SDPA's math backend | fp32 |

The reference upcasts the checkpoint's bf16 weights to fp32 and keeps everything in fp32 from
the embedding to the logits. It is the exact answer for these weights up to fp32 rounding,
which is about four orders of magnitude below the bf16 errors measured here. A full fp32 model
is 32 GB and does not fit next to anything, so `ref_forward` streams the weights one layer at
a time: the whole set of sequences goes through layer 0, then layer 1, and so on. A pass over
5,780 tokens takes 14 s.

Only one model is ever on the GPU. Each stage writes its logits and its residual stream after
every layer to `.npy` files (1.5 GB each in bf16, twice that for the reference), and the
report stage compares them on the CPU.

To get the engine's residual stream I added a `trace` list to `SparkModel` and `TorchModel`:
when it is a list, each layer appends its output. It is `None` by default and outside the
captured decode graph, so serving does not see it. `tests/test_engine.py` checks that turning
it on changes no logit and that the last entry gives back the logits.

## The inputs

The logit set is eight sequences, 5,780 positions:

- six chat prompts through the chat template, each followed by transformers' own greedy reply
  of 256 tokens. Only the reply positions are scored (1,542 of them), so this is the model
  predicting its own output, mostly with high confidence (median top-2 margin 4.9 logits).
- two wikitext-2 test windows of 2,048 tokens, window 0 (which starts with BOS) and the middle
  window, every position scored (median top-2 margin 1.7).

Every backend sees the same token ids. The engine prefills all eight in one packed batch;
transformers runs them one at a time.

## Logits

Per position: the relative error `||a - b|| / ||b||` of the logit vector, whether the argmax
agrees, and `KL(b || a)` of the next-token distributions in fp32. `b` is the reference side.
All positions pooled (5,638 scored):

| a vs b | rel err median | rel err p99 | top-1 agree | KL mean | KL p99 | KL max |
|---|---|---|---|---|---|---|
| spark vs ref | 0.98% | 2.83% | 98.86% | 3.29e-4 | 2.2e-3 | 0.030 |
| torch vs ref | 1.06% | 2.72% | 98.90% | 3.38e-4 | 2.1e-3 | 0.014 |
| hf vs ref | 1.18% | 3.10% | 99.08% | 3.91e-4 | 3.0e-3 | 0.014 |
| spark vs hf | 1.41% | 4.15% | 99.20% | 6.36e-4 | 5.1e-3 | 0.034 |

Split by subset:

| a vs b | subset | positions | rel err median | top-1 agree (misses) | KL mean | ppl a | ppl b |
|---|---|---|---|---|---|---|---|
| spark vs ref | chat | 1,542 | 0.94% | 99.09% (14) | 1.98e-4 | 1.1778 | 1.1778 |
| spark vs ref | wiki | 4,096 | 1.00% | 98.78% (50) | 3.78e-4 | 8.4701 | 8.4699 |
| torch vs ref | chat | 1,542 | 1.05% | 99.29% (11) | 2.25e-4 | 1.1768 | 1.1778 |
| torch vs ref | wiki | 4,096 | 1.07% | 98.75% (51) | 3.80e-4 | 8.4743 | 8.4699 |
| hf vs ref | chat | 1,542 | 1.13% | 99.35% (10) | 2.60e-4 | 1.1782 | 1.1778 |
| hf vs ref | wiki | 4,096 | 1.21% | 98.97% (42) | 4.40e-4 | 8.4731 | 8.4699 |
| spark vs hf | chat | 1,542 | 1.35% | 99.35% (10) | 3.98e-4 | 1.1778 | 1.1782 |
| spark vs hf | wiki | 4,096 | 1.45% | 99.15% (35) | 7.25e-4 | 8.4701 | 8.4731 |

What I read from these:

- On the error that measures the whole distribution (relative error and mean KL) the order is
  spark, then torch, then hf, in both subsets. The engine is about 17% closer to the
  reference than transformers by logit error and 16% by mean KL.
- The engine's perplexity on the two windows is 8.4701 against the reference's 8.4699. hf is
  at 8.4731 and torch at 8.4743. Shifts of 0.003 are well below anything a benchmark
  resolves; the full wikitext run below is the better test.
- Top-1 agreement is the one column where hf looks better (99.08% against 98.86%, 52 misses
  against 64). Every miss of every backend is at a near tie in the reference: the largest
  reference top-2 margin at a spark miss is 0.114 logits, at an hf miss 0.223, and the median
  is 0.04 for both. A difference of 12 coin flips out of 5,638 is noise, not a trend.
- spark and hf are further from each other (1.41%) than either is from the reference, by
  about the root sum of squares of their two errors (1.53%). Their errors are mostly
  independent. That is why "spark vs hf" alone cannot say which one is wrong.

One detail behind the near ties: the logits are bf16, and at the values the top logits take
(16 to 32) one bf16 step is 0.125. At 2.0% of the positions the engine's top two logits are
bit-for-bit equal (1.8% for hf), so the argmax is decided by token id order. Two logits
that round to the same bf16 value are less than one step apart, so a miss there is a coin
flip, not an error in the kernel.

## Where the bf16 stacks lose precision

I did not isolate each rounding, but counting where each implementation rounds to bf16 in a
layer explains the order:

| step | spark | torch | hf |
|---|---|---|---|
| RMSNorm | one rounding (fp32 normalize and scale, `rmsnorm.cu`) | one, if `F.rms_norm`'s fused kernel scales in fp32 (I did not check) | two: normalized x to bf16, then times the bf16 weight in bf16 |
| RoPE | fp32 math, one rounding | fp32 math, one rounding | cos and sin rounded to bf16, the products in bf16 |
| o projection + residual | one: the add is in the GEMM epilogue in fp32 | two: the GEMM output, then the add | two |
| gate, up, SiLU, product | one: SwiGLU in the GEMM epilogue on fp32 gate and up | four | four |
| down projection + residual | one, fused | two | two |

The engine rounds less often than either, and plain PyTorch with fp32 RoPE and a fused
`rms_norm` rounds less often than transformers. The fused epilogues were written for speed
([layer.md](layer.md)); they also happen to drop roundings.

## Error growth through the layers

The residual stream after every layer, per token, relative to the reference. The first token
of every sequence is left out of the medians: from layer 1 to layer 30 it carries the
attention-sink activation, a vector of norm about 388 against a median of 1 to 46 for the
other tokens, and it would dominate any pooled number.

| after layer | spark vs ref | torch vs ref | hf vs ref | spark vs hf | median norm (ref) |
|---|---|---|---|---|---|
| 0 | 0.33% | 0.40% | 0.46% | 0.54% | 0.9 |
| 1 | 0.53% | 0.62% | 0.73% | 0.88% | 1.3 |
| 2 | 0.66% | 0.75% | 0.89% | 1.05% | 1.8 |
| 4 | 0.84% | 0.92% | 1.15% | 1.35% | 3.0 |
| 8 | 0.99% | 1.09% | 1.33% | 1.58% | 4.9 |
| 11 | 1.06% | 1.16% | 1.37% | 1.63% | 5.9 |
| 16 | 0.88% | 0.96% | 1.13% | 1.35% | 9.3 |
| 20 | 0.83% | 0.91% | 1.03% | 1.23% | 13.8 |
| 24 | 0.85% | 0.92% | 1.02% | 1.22% | 20.7 |
| 28 | 0.92% | 0.99% | 1.10% | 1.31% | 30.3 |
| 31 | 1.10% | 1.19% | 1.33% | 1.58% | 45.9 |

The error does not compound. It grows over the first ten layers, to about 1%, then shrinks to
0.8% by layer 20 and ends at 1.1%. The shape is the same for all three bf16 stacks, so it is
the model, not a kernel. My reading: the residual stream's norm grows about 50x from layer 0
to layer 31, and each layer's new rounding error scales with what that layer adds, not with
the whole stream. While the stream grows faster than the error, the ratio falls. The
ranking (spark, torch, hf) holds at every one of the 32 layers.

The sink token behaves differently. Its error relative to the reference is 0.2% to 0.5% for
the engine through layer 30, 0.2% to 0.75% for hf, and 3.5% to 4.4% for all three after
layer 31, where the last layer cancels most of the sink activation (its norm drops from 387
to 44) and the absolute error of the large vector is left over. The pooled error over all
tokens follows the sink and climbs to 1.8%, which is why I report per-token medians.

## Greedy generation

Sixteen chat prompts, 256 new tokens each, greedy. The engine runs them as it serves: one
packed prefill of all sixteen, then CUDA-graph decode steps over the batch. transformers runs
`generate(do_sample=False)` one prompt at a time. None of the sixteen replies ended before 256
tokens on either side.

- 4 of 16 replies are identical for all 256 tokens.
- 12 diverge. The first divergence is at token 0, 9, 14, 23, 83, 101, 108, 143, 163, 180,
  201 and 246; the median is 104.5.

For each divergence I scored the common prefix with the engine (one prefill) and with the
fp32 reference, and read transformers' own logits at that step. "gap" is the logit of hf's
token minus the logit of the engine's token: positive means that backend prefers hf's token.

| prompt | first divergence | hf token / spark token | hf gap | spark gap | ref gap | ref picks |
|---|---|---|---|---|---|---|
| CPU cache hierarchy | 23 | ` breakdown` / ` detailed` | 0.125 | 0 | 0.010 | hf |
| lighthouse keeper story | 9 | `,` / ` of` | 0 | -0.125 | -0.090 | spark |
| French Revolution | 180 | ` enjoyed` / `,` | 0.125 | 0 | 0.119 | hf |
| CSV parser | 201 | ` flag` / ` boolean` | 0.125 | 0.125 | 0.036 | hf |
| TCP and UDP | 101 | ` ensures` / ` is` | 0.125 | 0 | 0.083 | hf |
| photosynthesis | 163 | ` transferred` / ` passed` | 0.125 | 0 | 0.114 | hf |
| meal plan | 14 | `Day` / `Monday` | 0 | -0.125 | -0.120 | spark |
| Hamlet | 83 | `.\n\n` / `.` | 0.125 | 0 | -0.049 | spark |
| RSA | 108 | ` Compute` / ` Calculate` | 0.125 | 0 | 0.012 | hf |
| cover letter | 143 | ` SQL` / ` Python` | 0 | -0.125 | -0.075 | spark |
| gradient descent | 0 | `Imagine` / `Gradient` | 0.125 | 0.125 | -0.048 | spark |
| French translation | 246 | `j` / `si` | 0 | 0 | -0.108 | spark |

Every divergence is a near tie:

- transformers' own gap at the step where it went the other way is 0.125 (one bf16 step) or
  0, an exact tie in bf16, in all twelve cases.
- the reference gap is at most 0.120 logits in absolute value, median 0.079. Below one bf16
  step every time.
- the reference sides with hf six times and with the engine six times.

So neither implementation is making worse choices; at a tie each one is a coin flip, and after
the first different token the two continuations are different texts. A typical split
(TCP and UDP, token 101):

> common: *...This connection is maintained throughout the duration of the communication
> session. TCP*
>
> hf: *ensures that data is delivered reliably and in the correct order, making it suitable
> for applications that require guaranteed delivery...*
>
> spark: *is a reliable protocol, ensuring that data is delivered in the correct order and
> retransmitted if errors occur...*

The spark column needs a caveat. In nine rows the rescored engine logits show a tie or
even favor hf's token (0.125 for "CSV parser" and "gradient descent"), although the engine
chose the other token when it generated. The rescoring is a fresh prefill of the prefix in a
batch of twelve prefixes, not the decode step that made the choice, and the two run different
GEMM schedules. At a gap this small that is enough to move the logit by one bf16 step. Which
leads to the next point.

### The engine is not batch invariant

I ran the same sixteen prompts through the engine again, one at a time. 9 of 16 differ from
the batched run, first at tokens 5, 28, 74, 95, 101, 143, 201, 246 and 254. `hgemm` picks its
schedule by M: a dedicated weight-streaming kernel for M up to 64 and a Stream-K split of K
across SMs above that, so the same row summed in a different batch is summed in a different
order and can round differently. At a near tie that flips a token. The splits are the same kind
of event as the hf ones above. transformers on cuBLAS is not batch invariant either, but I only
ran it with batch 1, so I have not measured it.

If someone needs reproducible greedy output across batch sizes, two changes would do it: a
fixed reduction order in the GEMMs regardless of M, and fp32 logits out of the last GEMM, so
the argmax stops seeing exact ties at 2% of positions. Neither would make the model more
accurate.

## Perplexity, all of wikitext-2

`scripts/eval_ppl.py`, wikitext-2 raw test, tokenized once with BOS, cut into non-overlapping
windows, every token after the first of a window scored.

| ctx | windows | tokens scored | ref (fp32) | spark | torch | hf |
|---|---|---|---|---|---|---|
| 2048 | 141 | 288,627 | 8.2840 | 8.2873 | 8.2870 | 8.2889 |
| 4096 | 70 | 286,650 | 7.7395 | 7.7420 | | 7.7429 |
| 8192 | 35 | 286,685 | | 7.4624 | | 7.4641 |

The same at ctx 2048 as mean NLL in nats above the reference's 2.11433: torch +3.6e-4,
spark +3.9e-4, hf +5.8e-4.

- Every bf16 stack scores a little worse than the fp32 reference. Rounding noise on the logits
  raises the cross entropy on average, so this is the expected sign.
- At 2048 the engine and plain PyTorch are tied (8.2873 and 8.2870, a gap I would not read
  anything into) and both are closer to the reference than transformers. At 4096 the engine is 0.0025 above the
  reference and transformers 0.0034. At 8192 the engine is 0.0017 below transformers; the
  fp32 reference does not fit there, because SDPA's math backend needs the full 8192 x 8192
  score matrix of 32 heads in fp32 (8.6 GB, twice with the softmax).
- The differences are in the fourth significant digit. For anyone quoting a perplexity the
  engine and transformers give the same number; the reference only shows which side of the
  noise each one is on.
- Speed of the whole run at 2048: 19.5 s on the engine, 23.5 s on the torch model, 24.7 s on
  transformers, scoring included.

The torch model being as close as the engine over all of wikitext, while it was further on
the logit set above, says the engine's edge over plain PyTorch is small. The edge over
transformers shows up in every measurement.

## Reproducing

```
M=~/models/llama3-8b-instruct
python scripts/llm_parity.py $M hf          # 16 generations + logits and hidden states
python scripts/llm_parity.py $M spark       # the same on the engine, plus margins
python scripts/llm_parity.py $M torch
python scripts/llm_parity.py $M ref         # fp32, streamed per layer
python scripts/llm_parity.py $M report      # CPU only, writes results/llm_parity.json
python scripts/eval_ppl.py $M --backend spark --ctx 2048 --batch 4
python scripts/llm_parity.py $M ppl --ctx 2048      # the fp32 reference perplexity
```

Each GPU stage loads one model and takes one to two minutes, the reference perplexity about
two. The dumps in `parity_work/` take about 13 GB of disk.
