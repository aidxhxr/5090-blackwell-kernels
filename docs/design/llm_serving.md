# Serving a real Llama-3-8B: this engine, vLLM and transformers

serving.md measured the engine on random weights against the same model written in PyTorch.
This note puts it next to the servers people actually use. I loaded Llama-3-8B-Instruct and
ran the same requests, as the same token ids, through three stacks:

- **spark**: `engine.Engine` on `SparkModel` (this package's kernels), CUDA-graph decode,
  256 slots, an 8.5 GiB paged cache (69,632 tokens), an 8,192-token prefill budget.
- **vLLM 0.30.0** (its own venv, torch 2.13): offline `LLM.generate`, bf16, FlashAttention 2
  backend (its pick on sm_120), chunked prefill with 8,192 batched tokens, 256 sequences,
  `gpu_memory_utilization=0.85` (8.91 GiB of KV, 72,960 tokens), prefix caching off.
- **transformers 5.18** (torch 2.14): `generate` with sdpa and the dynamic cache (`hf`), with
  the static cache and the decode step it compiles itself (`hf_static`), and its own
  continuous-batching manager (`hf_cb`).

Every request is greedy and generates exactly its output length (vLLM `ignore_eos=True`,
transformers with EOS disabled, our engine without stop ids). Each backend runs in its own
process, one at a time on the GPU. RTX 5090, 600 W cap, driver 595.58.

`scripts/bench_llm.py` builds the workloads once (`workload`), runs one backend (`run`) and
merges the rows (`report`) into `results/llm_serve.json`. The timing is the same for every
backend: a warm-up of the same shape, then the batch with one output token (its wall time is
the time to first token, TTFT), then the batch with the full output. ITL, the time per decoded
token, is (full - TTFT) / (out - 1). All of it is wall clock around a blocking call, so the
host side of each server counts.

## Static batches, single stream and prefill

Prompts are 512-token windows of wikitext-2 (a BOS and 511 text tokens), 256 tokens out.

| workload | spark | vLLM | hf | hf_static |
|---|---|---|---|---|
| B=1, 128 in, 1024 out | **101.8 tok/s**, ITL 9.82 ms | 100.7 tok/s, ITL 9.92 ms | 85.1 tok/s, ITL 11.75 ms | 76.8 tok/s, ITL 13.01 ms |
| B=1, 512 in, 256 out | **100.9 tok/s**, ITL 9.82 ms, TTFT 33 ms | 99.7, ITL 9.92, TTFT 38 | 83.4, ITL 11.87, TTFT 43 | 80.8, ITL 12.25, TTFT 45 |
| B=8 | **716 tok/s**, ITL 10.27 ms, TTFT 243 ms | 652, ITL 11.28, TTFT 265 | 505, ITL 14.74, TTFT 293 | 431, ITL 17.49, TTFT 294 |
| B=32 | **2,064 tok/s**, ITL 11.74 ms, TTFT 977 ms | 1,938, ITL 12.42, TTFT 1,060 | 1,507, ITL 16.56, TTFT 1,212 | 852, ITL 32.98, TTFT 1,209 |
| B=64 | **2,923 tok/s**, ITL 14.31 ms, TTFT 1,955 ms | 2,853, **ITL 14.16**, TTFT 2,132 | 1,959, ITL 23.36, TTFT 2,404 | out of memory |
| prefill 2,048 tokens, TTFT | **126.9 ms** | 144.5 ms | 154.0 ms | |
| prefill 8,000 tokens, TTFT | **553.2 ms** | 606.9 ms | 665.5 ms | |

**Bandwidth at B = 1.** A decode step reads the 32 layers (13.96 GB) and the lm_head
(1.05 GB): 15.01 GB. Ours does it in 9.82 ms, 1,528 GB/s, 99.8% of the 1,532 GB/s copy roof
(bandwidth.md). vLLM: 9.92 ms, 1,512 GB/s, 98.7%. transformers eager: 11.75 ms, 1,277 GB/s,
83%. A single stream on this card is a memory benchmark, and both servers are at the roof.

What I read from the table:

- At B = 1 and B = 64 the two servers are within about 1%. At B = 8 and 32 ours takes 9% and
  5% less time per token; the profile below shows where.
- Prefill is where ours wins clearly: 9 to 12% lower TTFT on one long prompt, 8 to 13% on the
  packed batches.
- transformers eager spends about 2 ms more per step than either server at B = 1, and its
  step grows faster with the batch (23.4 ms at B = 64). I did not profile it; 2 ms is about
  the cost of launching an eager step's kernels one by one from Python.
- The static cache with the compiled step was slower than the dynamic cache at every batch
  here, and at B = 64 it ran out of memory. I report it as measured and did not dig further.

## Continuous batching

256 chat requests ("Summarize this passage in a few sentences." over a wikitext excerpt,
through the chat template), excerpts of 32 to 1,536 tokens and outputs of 16 to 768 tokens,
both log-uniform with a fixed seed: 99,074 prompt tokens and 47,875 output tokens in all,
submitted at once. TTFT and the end-to-end time are per request from the moment all are
submitted; ITL is per request, (last token - first token) / (out - 1).

| | spark | vLLM | hf_cb |
|---|---|---|---|
| wall time | 19.32 s | **18.70 s** | 132.2 s |
| output tok/s | 2,478 | **2,561** | 362 |
| TTFT p50 / p90 | 3.88 / 9.95 s | **3.81 / 6.84 s** | |
| end to end p50 / p90 | **9.26 / 14.79 s** | 9.67 / 15.78 s | |
| ITL mean / p90 | **32.2 / 55.3 ms** | 68.3 / 165.2 ms | |

vLLM finishes 3.3% sooner. The two schedulers make different trades: vLLM admits more
requests early (its log shows 207 running at 99.8% of its KV cache, and it preempted 9 of
them), so its TTFT tail is shorter, but every chunked-prefill step it runs stretches the
decode of everything already running, so its per-request ITL is twice ours. Ours keeps the
running batch moving and makes new requests wait.

The engine on main ran this workload in 20.48 s (2,337 tok/s). Three engine changes on this
branch took it to 19.32 s; the next section has what they are.

## Correctness

Same six chat prompts, 128 greedy tokens each. The first position where two backends differ:

| prompt | spark vs vLLM | spark vs hf | hf vs vLLM |
|---|---|---|---|
| L2 cache | 19 | 19 | 93 |
| haiku | 2 | 14 | 2 |
| Canberra | 61 | 61 | 100 |
| TCP vs UDP | 25 | identical | 25 |
| French and German | identical | identical | identical |
| ice | 27 | 27 | 79 |

At every divergence I recomputed our logits for the shared prefix (`Engine.prompt_logits`):
the two candidate tokens are tied in bf16 (' ice' and ' water' both at 27.625, ',' and
' wings' both at 14.75) or one bf16 step apart at that magnitude (21.375 against 21.25). Each
pair of stacks splits on a different tie, and transformers disagrees with vLLM as often as we
do, so the three run the same model and the comparison is apples to apples.

## Where the time goes

`scripts/profile_llm.py` runs torch.profiler over our decode steps and prefills and times
every forward of the continuous run. `scripts/profile_vllm.py` runs vLLM's static batches
under nsys (engine core in process, CUDA graph nodes traced) and splits its decode steps the
same way. Both decode at a context of 512 + 32 tokens. Times are GPU time per step, ms.

| decode step | ours GEMM | attention | rest | kernels | vLLM GEMM | attention | rest | kernels |
|---|---|---|---|---|---|---|---|---|
| B=1 | 9.28 | 0.27 | 0.28 | 9.83 | 9.20 | 0.45 | 0.17 | 9.83 |
| B=8 | **9.36** | 0.55 | 0.29 | **10.20** | 10.08 | 0.66 | 0.32 | 11.05 |
| B=64 | 10.33 | 3.19 | 0.30 | 13.80 | **9.86** | **2.91** | 0.42 | **13.20** |

- **Host overhead is gone on both.** Our step is 296 launches with 0.03 to 0.04 ms of GPU
  idle time per step. vLLM's is 370 to 528 launches and 0.1 to 0.2 ms idle. Neither is
  host bound.
- **B = 1.** Both GEMMs stream the weights at about 1.62 TB/s, above the copy roof (a read
  stream has no write turnaround). cuBLAS runs a GEMV here. Our attention is faster and our
  norms slower (vLLM's are Triton kernels with the residual add fused in). The totals are
  equal.
- **B = 8.** cuBLAS switches to a CUTLASS sm80 wmma kernel with 16 x 16 tiles, and the GEMMs
  take 10.08 ms, 10% more than its GEMV at M = 1. Our weight-streaming kernel costs 0.08 ms
  more at M = 8 than at M = 1. That is the whole B = 8 and B = 32 lead.
- **B = 64.** Here vLLM is ahead per step. Its M = 64 GEMMs take 9.86 ms against our 10.33,
  and FlashAttention 2's split-KV decode reads the 4.6 GB of K/V in 2.91 ms against our
  3.19 (1.57 against 1.43 TB/s). serving.md already says our paged decode is 5 to 7% behind
  the contiguous kernel, all of it the separate combine launch; this is that gap.
- **Prefill.** One 8,000-token prompt is 553 ms of kernels for us: 466 ms of GEMM (111.7
  TFLOP, 240 TFLOPS), 79 ms of attention, the rest under 1%. vLLM's trace shows its prefill
  GEMMs are CUTLASS sm80 kernels chosen by cuBLAS; ours is hgemm v6 on the TMA pipeline. Its
  TTFT on that prompt is 10% longer.

**The continuous run, by phase** (every forward synchronized and timed, so the wall times
are a little longer than the table above):

| engine | wall | prefill forwards | decode steps | steps with 16 or fewer live |
|---|---|---|---|---|
| main | 20.60 s | 6.48 s in 89 | 14.09 s in 1,021 | 27 steps |
| + slot compaction | 19.92 s | 6.49 s in 89 | 13.40 s in 1,021 | 388 steps, 4.07 s |
| + mixed steps | 19.47 s | 7.26 s in 85, with 7,656 decode rows | 12.18 s in 941 | 390 steps, 4.12 s |

The kernels are not what separates us from vLLM here. A fifth of the run is the tail: a few
long requests, admitted late, decoding at batch 16 or less for 4 s. What admits them late is
the scheduler. Ours reserves pages for the prompt plus every requested token at admission,
so 69,632 tokens of cache hold about 90 running requests in the busy phase, and it never
preempts. vLLM allocates pages as tokens arrive and preempts when it runs out, so the same
amount of memory held 207 requests at its peak. Changing only our cache size confirms it:

| our cache | tokens | wall | output tok/s | TTFT p50 |
|---|---|---|---|---|
| 6 GiB | 49,152 | 21.26 s | 2,252 | 6.60 s |
| 8.5 GiB (the table above) | 69,632 | 19.32 s | 2,478 | 3.88 s |
| 10 GiB | 81,920 | 18.49 s | **2,589** | 3.27 s |

With 12% more cache tokens than vLLM had, ours finishes first. With the same budget vLLM's
on-demand allocation wins. 10 GiB is also the most I can give it: the 8,192-token prefill's
activations then hit the caching allocator's limit (it frees and retries, nothing fails),
while vLLM sizes its cache after profiling the peak activation.

## What I changed in the engine

Four commits, each with a test in `tests/test_engine.py`:

- **A graph that kept a freed buffer (fix).** The kernels keep their split-K workspaces in
  one buffer per process and free and reallocate it when a launch needs more. A CUDA graph
  captured before the largest launch keeps the old address. With 256 slots the 128 and 256
  buckets ran the Stream-K GEMM schedule, whose workspace a later prefill grew, and the
  continuous run died with an illegal address on replay. With 64 slots it had been silent.
  The engine now runs every GEMM shape at M = 64 down to 1 before capturing, and graphs
  only the buckets up to 64 (the M <= 64 kernel). Buckets 128 and 256 run eagerly; the graph
  saves under 1% of a step (serving.md). The test runs 128 slots with long prompts, graphs
  against eager; on main it gives different tokens.
- **Slot compaction.** A sequence keeps its slot, and a step runs the power-of-two bucket over
  the highest occupied slot, so six sequences left in high slots ran a 64-row step. Before a
  decode step the engine now moves the sequences above the bucket their count needs into
  free low slots (two device copies: the next token id and the block table row). 20.48 to
  19.92 s.
- **Mixed steps.** A step that admits prompts used to run the prefill and then a separate
  decode step over the running batch, two passes over the weights. Now the running
  sequences ride in the prefill forward as one-token rows after their context: one pass, the
  GEMMs over prompts and decode rows together, the attention split between `attention_varlen`
  for the prompts and `paged_decode` for the decode rows. My first version sent the decode
  rows through `attention_varlen` too, and the run got slower (20.40 s): each decode row
  costs that kernel a whole 128-row query tile. With the split, 19.92 to 19.32 s. Prefill is
  compute bound, so a decode row there costs its share of GEMM flops (about 0.06 ms per row)
  where a separate step would have cost a full weight read; the saving is 0.6 s, not more.
- **Stop ids.** `Engine(stop_ids=...)` ends a request at its first stop token. Each forward's
  new tokens are copied to pinned memory behind an event, and a step reads only the copies
  whose event has passed, so the host never waits for the GPU. The price is that a stopped
  sequence may run a few extra tokens before it retires; `outputs()` cuts them. `generate.py`
  now stops at end of turn instead of printing up to `--new` tokens. The benchmark keeps
  them off, so it is not in the numbers above.

## What limits each

- **Ours.** At B = 1 the step is the weight read at the roof; nothing is left there without
  smaller weights (int4, w4gemm.md). At B = 64 the M = 64 GEMM and the paged decode combine
  cost about 0.75 ms per step against vLLM. Under load the limit was the scheduler:
  whole-request page reservation with no preemption, so fewer requests ran at once and long
  requests finished in a thin tail. The engine now takes pages as the tokens arrive and
  preempts the youngest running sequence when they run out (`Engine(preempt=True)`, the
  default; the tokens it has so far go back to the queue with its prompt, from a per-slot
  buffer the decode step writes on the device), which is the policy that let vLLM run 207
  requests where this engine ran 90. The table above is the reservation scheduler; the run
  with preemption is not in it yet. Chunked prefill is still missing too, so an 8K prompt
  holds the running batch for 0.55 s.
- **vLLM.** Its decode kernels are at the roof like ours. It loses where cuBLAS picks a weak
  kernel for small M (B = 8, 32) and on prefill GEMMs. Its scheduler favors admission, which
  buys throughput and TTFT at the cost of a 68 ms mean ITL under load.
- **transformers.** Eager `generate` pays per-op launch overhead (2 ms a step at B = 1) and
  attention that grows with the batch. The continuous-batching manager ran 7x behind vLLM on
  the mixed workload, with out-of-memory retries at its default memory share; a lower share
  (0.6) did not change the result (127 s).

## Prefix caching (not measured yet)

Written with the box down. The code and its tests are in, the GPU tests have not run, and
nothing in this section is a measurement.

Requests often start with the same tokens: a chat system prompt, few-shot examples, one
document asked several questions. Without a prefix cache the engine prefills that prefix
once per request. With `Engine(prefix_cache=True)`, the default for `SparkModel`, a request
takes the cache pages of the prefix an earlier request already wrote and prefills only the
rest. The design is vLLM's automatic prefix caching, on our 16-token pages:

- **Only full pages are shared.** Page i of a prompt is named by sha256 over the name of
  page i - 1 and the page's 16 token ids, so a name stands for the whole prompt up to the
  end of that page, not for 16 tokens alone. `PageTable` (`python/spark_kernels/pages.py`,
  plain Python) maps names to page ids and keeps a hold count per page.
- **Admission.** The prompt's names are computed at submit, from the host copy of its ids.
  Admission walks them, takes the pages of the longest named prefix (one hold each), and
  allocates the rest. It stops at the last full page before the prompt's last token, so at
  least one token is computed and its logits exist; a prompt of exactly k pages that is all
  cached still prefills its last page. The prefill then starts after the matched tokens: a
  chunk after a cached context, the path chunked prefill already uses (`attention_varlen`
  with the bottom-right causal mask). Matched tokens do not count against the prefill
  budget. A matched page nobody held was counted as available, so the page check subtracts
  it once it is taken.
- **Registration.** After each prefill forward is queued, the full pages it wrote that hold
  submitted prompt tokens get their names. Queued is enough: anything that reads them later
  is a later launch on the same stream.
- **Generated tokens are not registered.** vLLM registers them. Here their ids live on the
  device (the decode loop never copies a token to the host), and hashing them would make the
  host wait for the GPU every 16 steps. What it costs: a preempted sequence finds its own
  prompt pages again on resume (unless they were evicted) but recomputes what it generated,
  and a multi-turn chat reuses the earlier turns' prompt pages but not the pages of the
  answers.
- **Release and eviction.** When a sequence finishes or is preempted each of its pages loses
  a hold. A page nobody holds goes back to the free list, or, if it has a name, to an LRU set
  of evictable pages that stay findable. `alloc` takes free pages first, then the least
  recently released cached ones, and drops their names. A sequence's pages are released last
  to first, so its deeper pages are evicted before its first ones: page i can only be
  reached through pages 0 to i - 1.
- **Nothing writes a shared page.** Only full pages get names, and a sequence writes only
  past the pages it matched, so no forward writes a named page or a page held twice. The
  engine asserts this for every prefill piece and decode row (`PageTable.writable`).
- **Preemption and reservation.** A preempted sequence drops one hold per page, so a shared
  page stays with its other holders. With `preempt=False` the matched pages count toward the
  reservation of prompt plus requested tokens.
- **What it does not do.** Two requests with the same prefix admitted in the same step both
  prefill it, since the names appear only after the forward; the second one's copies are
  never named and go back to the free list. Holding the second request back one step would
  save that work; it is not done. `prompt_logits` neither reuses nor registers pages (it
  wants every position's logits). `TorchModel` cannot start a prefill after a cached
  context, so it gets no prefix cache.

**What it should buy.** Time to first token on shared-prefix workloads, and the GPU time of
the prefill it skips. `bench_llm.py` has a `shared` workload for it: 8 wikitext passages of
1,536 tokens, 8 questions each, the question after the passage, 64 requests in a seeded
random order. Each passage's requests share about 1,550 tokens; the rest is the question
and the end of the chat template, plus up to 15 tokens before the next page boundary. If
every passage's later requests come after its first one's forward, the prefill drops from
about 64 x 1,580 = 101K tokens to about 8 x 1,580 + 56 x 50 = 15K. At the 8,000-token
prefill rate in the first table (14.5K tokens/s) that is about 7 s of prefill against 1 s.
The same-step case above makes the real saving smaller. On the `continuous` workload the
shared part is the chat header and the one-line instruction, about 15 tokens, under one
page, so I expect no hits there.

The comparison against vLLM in the tables above ran with prefix caching off on both sides,
and `bench_llm.py` keeps it off by default so those rows stay comparable. `--prefix-cache`
turns it on for both, and each backend's cache is reset before every timed call, so the
warm-up never counts as a hit. Our `continuous` and `shared` rows report the prompt tokens
looked up and the hit rate (`Stats.prefix_queries`, `Stats.prefix_hit_tokens`).

Host costs I have not measured: hashing at submit (one sha256 per 16 tokens, about 6,000
for a 100K-token prompt) and a hold-count check per decode row per step.

Tests: `tests/test_pages.py` runs the bookkeeping on the CPU (the hash chain, holds, the
LRU order, eviction dropping names, and a random workload against a small table with the
invariants checked after every operation). `tests/test_engine.py` runs requests built from
a few long shared prefixes, in waves, with and without the cache, and checks the same tokens
for every request, a smaller prefill by exactly the hit tokens, every page back on the free
list or the evictable set, eviction under a small cache, preemption with chunked prefill,
mixed steps and stop ids, and the reservation mode.

## Surprises

- The graph bug. Nothing in the kernels or the engine looked wrong on its own: a per-process
  workspace that grows is fine for eager launches and fatal for a graph. It only showed with
  more than 64 slots, a size the random-weight benchmark never used.
- Batch 8 is where a hand-written decode GEMM beats cuBLAS by the most, not batch 1.
- transformers' static cache with its own compiled step was slower than its dynamic cache.
- Mixing decode into prefill helps less than I expected, for the reason in the table above:
  the prefill GEMM is compute bound, so the decode rows are not free there.

## Speculative decoding by prompt lookup (not measured yet)

Nothing in this section has run on the GPU yet. The code and its tests are in, the numbers
below are arithmetic from the measurements above, and the table they lead to is still empty.

**Why.** At B = 1 a decode step is the 15.01 GB weight read at the copy roof, 9.82 ms, and
the GEMMs at M <= 64 are bandwidth bound: the weight-streaming kernel costs 0.08 ms more at
M = 8 than at M = 1 (the profile above). A forward over 5 or 9 rows of one sequence costs
about what a forward over 1 row costs. If some of those rows are guesses of the next tokens
and the model agrees with them, one weight read yields several tokens.

**Drafting** (`spark_kernels/spec.py`). No draft model: the last 3 tokens of the context
(prompt and tokens so far) are looked up earlier in the same context, then the last 2, then
the last 1, and the up to k tokens that followed the most recent match are the draft. This is
prompt lookup decoding. It pays off when the output copies from the input: a summary quoting
its passage, an answer citing a document, an edit of code given in the prompt. On a chat reply
with nothing to copy the drafts mostly miss. The drafter keeps an index from every n-gram to
the last place it ends, so a draft is three dictionary lookups and a new token adds three
entries, whatever the context length.

**Verification** (`Engine(speculative=k)`). Each running sequence contributes the rows
[next token, d1, .., dk] at positions length .. length + k, as a chunk after its cached
context: the prefill path, `attention_varlen` with its bottom-right causal mask, every row's
logits. Row i's argmax is the greedy token after row i. `spec.accept` keeps d1 .. da while
each equals the argmax of the row before it, then adds the argmax of row a (the bonus token):
between 1 and k + 1 tokens per sequence per forward, and they are the tokens greedy decoding
would give one at a time, because each kept row saw exactly the keys a plain step would have.
A sequence with no draft is a one-row chunk; those rows go last and through `paged_decode`, as
the decode rows of a mixed step do. A step in which no sequence has a draft runs the ordinary
graphed decode step. A draft is cut so it never takes a sequence past its max_new.

**Rollback.** The forward writes the K/V of all k + 1 rows into the cache. The sequence's
length then grows by a + 1 only, so the rejected rows sit past the length, where no kernel
reads them (both attention kernels stop at `seq_lens`), and the next step writes its own rows
over them. Nothing is copied or cleared.

**Pages.** Before the forward each sequence takes the pages its k + 1 rows need (`_grow` with
a row count per sequence; it may preempt the youngest sequence as a plain step would). Pages
that end up past the new length are kept for the next step rather than freed: with k < 16 that
is at most one page per sequence more than plain decoding holds, and it goes back at retire or
preemption with the rest. The cost is that a draft can preempt a sequence a plain step would
have left alone, one page earlier.

**The host copy.** The engine keeps generated tokens on the device (the `ids` buffer and the
`gen` record the graph writes) and never waits for the GPU. Drafting needs the tokens of the
previous step on the host, so speculative mode reads every forward's argmax back (one
`.tolist()`, a synchronize) and writes the kept tokens, the next input and the record columns
back with two small copies. The transfer is tens of microseconds. The larger cost is that the
GPU idles while the host drafts, builds the next batch and launches its first kernels, which I
estimate at 0.1 to 0.3 ms against a 10 ms step (1 to 3%, not measured). With the tokens on the
host, stop ids are checked exactly in this mode: a stop inside a run of kept drafts cuts the
rest.

**Expected gain.** If a verify forward yields A tokens on average and costs (1 + c) of a plain
step, single-stream throughput goes from 100 tok/s to about 100 A / (1 + c). At B = 1 with k = 4
to 8, c is the extra rows in the GEMMs (about 1%), the lm_head over k + 1 rows, the 128-row
query tile `attention_varlen` spends on a chunk of 5 to 9 rows (small next to the weight read
at a few thousand keys of context), the forward not being graphed (0.07 ms, serving.md) and the
host gap above: c of 3 to 5% is my guess. A = 2 on summarization prompts would be about 1.9x;
A = 1.1 on open chat would be close to a wash. These are the numbers to fill in.

**Costs and limits.**

- Batch size. At B = 8 with k = 7 the GEMM M is 64, the last size the weight-streaming kernel
  takes; above it the Stream-K schedule runs and every rejected row costs real flops. On a full
  continuous batch speculation should cost throughput. There is no policy yet that turns it off
  per step by batch size or acceptance rate.
- An admitting step prefills the new prompts in their own forward and then runs the verify
  forward (no mixed step in this mode), two weight reads where the plain engine does one.
- The verify forward is not captured into a CUDA graph; its shape changes every step.
- Greedy only. Sampling would need the rejection rule of speculative sampling.
- Same tokens up to near ties. The verify forward computes a row through `attention_varlen`
  and GEMMs of a different M than the plain decode step, and the engine is not batch invariant
  (llm_parity.md), so at a near tie in bf16 a token can flip, the same kind of split the mixed
  step can make.

**Tests.** `tests/test_spec.py` runs on the CPU: the indexed drafter against a plain scan on
growing random contexts, the acceptance rule, the stop cut, and whole speculative decodes on a
fake model whose cache keeps rejected rows past the length, against plain greedy.
`tests/test_engine.py` (GPU, two random layers, 1,024-token vocabulary) compares the engine
with and without speculation token for token: on prompts that repeat a segment, with drafts
that are the true next tokens with some changed (so partial, full and zero acceptance and the
rollback all happen whatever the random model does), and in a cache small enough to preempt,
under stop ids.

**To measure** (`scripts/bench_llm.py`, the `spark_spec` backend):

    python scripts/bench_llm.py run spark MODEL --only decode,parity,summarize --out spark.jsonl
    python scripts/bench_llm.py run spark_spec MODEL --spec 4 --only decode,parity,summarize --out spec.jsonl
    python scripts/bench_llm.py report spark.jsonl spec.jsonl

The decode row is a wikitext window continued for 1,024 tokens (little to copy), the summarize
rows are the first chat requests of the continuous workload ("Summarize this passage") at
B = 1, and report prints the first differing token of each parity prompt between the two.
`python scripts/generate.py MODEL --spec 4 "..."` prints the share of drafts kept.

## Reproducing

On the box, from the repository root (one model on the GPU at a time):

    python scripts/bench_llm.py workload ~/models/llama3-8b-instruct
    python scripts/bench_llm.py run spark ~/models/llama3-8b-instruct --out spark.jsonl
    ~/vllm-env/bin/python scripts/bench_llm.py run vllm ~/models/llama3-8b-instruct --out vllm.jsonl
    python scripts/bench_llm.py run hf ~/models/llama3-8b-instruct --only static,decode,prefill,parity --out hf.jsonl
    python scripts/bench_llm.py run hf_static ~/models/llama3-8b-instruct --only static --batches 32 --out hf_static.jsonl
    python scripts/bench_llm.py run hf_cb ~/models/llama3-8b-instruct --only continuous --out hf_cb.jsonl
    python scripts/bench_llm.py report spark.jsonl vllm.jsonl hf.jsonl hf_static.jsonl hf_cb.jsonl

With prefix caching (the `shared` workload needs a workload file written after it was added):

    python scripts/bench_llm.py run spark ~/models/llama3-8b-instruct --only shared,continuous --prefix-cache --out spark_pc.jsonl
    ~/vllm-env/bin/python scripts/bench_llm.py run vllm ~/models/llama3-8b-instruct --only shared,continuous --prefix-cache --out vllm_pc.jsonl

`hf_static` compiles a step per batch shape and keeps it, so each batch size gets its own
process.
