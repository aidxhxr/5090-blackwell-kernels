# Batch-invariant inference

The engine has always been deterministic: the same requests on the same schedule give the
same bits every run. It was not batch invariant. A request's logits changed with what else
was in its steps, so the same prompt could take a different token at a near tie depending on
the batch around it. llm_parity.md measured that on the real checkpoint: sixteen prompts run
one at a time against the same sixteen batched, 9 of 16 replies differ, first at tokens 5 to
254. Thinking Machines' "Defeating Nondeterminism in LLM Inference" (2025) made the point
that this, not GPU nondeterminism, is why a server gives different answers to the same
prompt: every kernel is deterministic, but the reduction order of a row follows the batch.

This note covers `Engine(batch_invariant=True)` (or `SPARK_BATCH_INVARIANT=1`): a mode in
which a request's logits are the same bits whatever shares its steps, how many slots the
decode graph has, how its prompt is chunked, whether it is preempted and resumed, which slot
it lands in, whether its prefix came from the cache, and whether speculative decoding
verified its tokens. It is three kernel rungs and a flag:

- `hgemm` variant 7: no route splits K;
- `paged_decode` variant 2 and `attention_varlen` variant 2: one arithmetic per query row,
  fixed by the row's own sequence, in both ops;
- `SparkModel(batch_invariant=...)` / `Engine(batch_invariant=...)`, which picks them.

The default mode is untouched: the new rungs are opt-in, `-1` never picks them, and the
default engine runs the same kernels as before.

Tests: `tests/test_batch_invariant.py`. Bench: `scripts/bench_invariant.py`, rows in
`results/batch_invariant.jsonl`.

## What varied

A row's bits depend on the order its sums are taken in. Everything below is deterministic
for a fixed input; the question is whether the order for one row moves when the other rows
change.

| op | what picks the order | depends on |
|---|---|---|
| hgemm (every projection, lm_head) | the kernel (decode kernel for M <= 64, a TMA tile or variant 4 above), and the K split: Stream-K static ranges or K-passes, the decode kernel's split when a narrow N leaves fewer than 64 strips | M, the tokens in the forward |
| paged_decode variant 1 | the flat split: the slabs of the whole batch cut into equal ranges per warp, so a sequence is cut wherever the batch puts the boundaries; the combine merges the pieces as a tree over the lanes | every other sequence's length |
| a decode row through attention_varlen | variant 1 runs the online softmax in 64-key steps, paged decode in 16-key steps with a split | which op the row lands in: a resumed sequence's tokens (prefilled again), a verify row of speculative decoding, the K/V of tokens a prefix hit or a preemption recomputes |
| a mixed step's decode rows | paged_decode over the decode rows only, so the split sees a different batch than the decode step would | the prompts admitted in that step |

What does not depend on the batch: `rmsnorm` (its kernel is picked by the row width only),
`rope_append_paged_` and the embedding (per token), the SwiGLU and residual adds (in the GEMM
epilogue, per element), argmax, and `sk.sample`, which already draws from each request's own
Philox stream with integer sums (sample.md).

Three default-mode tests show it. The first rows of a 300 x 4096 x 4096 GEMM computed as a
batch of their own (1, 64 or 200 rows) do not match the same rows of the full GEMM, one
4,100-key sequence decoded alone does not match the same sequence decoded next to seven
others, and ten requests served one at a time get other logits than in a batch of 8.

## The GEMM: variant 7

The tensor cores compute each output element of `mma.sync.m16n8k16` from its A row, its B
column and its accumulator, nothing else. So if every kernel reduces a row the same way, from
a zero accumulator through the k16 steps of K in order, the tile shape does not matter: a
64-row decode tile, a 128 x 64 TMA tile and a 128 x 128 one give the same bits for the same
element. Every route of variants 3 to 6 already walks K that way inside a tile. What breaks
it is the K split: Stream-K starts a second piece of a tile from zero and adds the first
piece's partial at the end, and the decode kernel's split adds slices with atomics.

Variant 7 is variant 6 with the split taken out:

- M <= 64: the decode kernel with K unsplit (`hgemm_decode::launch_unsplit`). On the Llama
  shapes the default never split there anyway (every projection has at least 64 strips), so
  nothing changes but the guarantee: variant 6 and 7 give the same bits and the same time
  at M = 1 to 64 on all five projections (the bench checks it).
- M = 65 to 128 on a narrow N: the decode kernel again, with a 128-row tile, when its
  128 x 32 strips fit in one wave (N up to 5,440). Without a K split a TMA tile gives the o
  and down projections 64 tiles for 170 SMs at 128 tokens; 128 strips of B with A in L2 do
  better: o 51 to 37 us, down 163 to 112 to 134 us (it moved between runs). On qkv, gate_up
  and lm_head the decode tile has no more CTAs than the TMA tile and lost (qkv 51 to 60 us,
  gate_up 164 to 176), so they keep the TMA tile. `SPARK_HGEMM_V7_DECODE_M=64` turns it off.
- Otherwise one TMA tile with every output tile computed whole by one block
  (`Knobs::whole_tiles` in hgemm_streamk.cuh): one block per tile when they fit in a wave,
  whole tiles from the queue otherwise. The tile is the one with the fewest waves times tile
  area, a 64-wide tile counted at 0.9 of the 128 x 128 rate.

Variant 7 never goes through variant 6's per-shape route or its split switches: it is its
own entry point (`hgemm_tma_sk_invariant`) and launches the whole-tile plan directly. Any
later split-K route added to variant 6 has to stay out of it.

The epilogue (bias, activation, residual, the SwiGLU product) is the same code on every
route, applied once to the finished fp32 sum. `test_hgemm_invariant_rows_do_not_depend_on_m`
checks rows of 15 values of M from 1 to 513, from the start and the middle of the batch, on
four shapes, and the fused residual and SwiGLU forms, with `torch.equal`.

## Attention: variant 2 of both paged ops

A query row of sequence b at position p sees keys 0..p. Both ops now compute it the same way:

1. The keys are cut into chunks of 256 at offsets 0, 256, 512, ... (the last one short).
2. A chunk is an online softmax from an empty state over its 16-key steps in key order:
   `S = q k^T` by mma over D in k16 order, the step's max, `P = 2^(s - m)`, P rounded to
   bf16, `O += P V` (`step16` in attention_invariant.cu, the same function in both kernels).
3. The chunks are folded left to right into a running (M, L, O), and O / L is rounded once.

Two details make it exact rather than nearly so. A step of keys the row cannot see changes
nothing bit for bit: its max is -inf, the rescale is set to exactly 1 when the max does not
move (instead of `2^(m - m)`, which I did not want to depend on), and P is 0. The fold gives a
side that holds the new max a scale of exactly 1 and an empty side exactly 0, so folding an
empty chunk, or folding a chunk into the empty state, is exact. That lets the varlen kernel
run its 64-key tiles past a row's last key, and lets a chunk end early, without changing a
bit. And the multiply-adds that round are written out (`__fmaf_rn`, `__fmul_rn`), so the
compiler cannot contract them one way in one kernel and another way in the other.

Who computes a chunk is the only difference between the ops:

- **decode** keeps variant 1's flat split, at chunk granularity: the chunks of every (b, kv
  head) of the batch are one list cut into equal contiguous ranges, one per warp of a
  one-block-per-SM grid, and a chunk never spans two warps. A sequence of one chunk is
  normalized and written by its warp. Otherwise every chunk's (m, l, O) goes to a workspace
  slot and a combine kernel folds them in chunk order, one column per thread. Which warp
  does what follows the batch; what is computed does not.
- **varlen** keeps variant 1's 128-row query tile and 64-key K/V tiles. A tile runs as four
  16-key steps and every fourth tile ends a chunk, folded in registers. The second O
  accumulator takes the kernel to 255 registers with a 12-byte spill at D = 128.

The decode workspace is one slot per chunk (2 KB per 256 keys of a kv head at GQA 4), sized
for B x max_pages of keys and grown on demand. A grown-out buffer is kept rather than freed:
a CUDA graph captured before the growth still points at it.

`test_paged_decode_invariant_to_the_batch` decodes eight sequences (1 to 4,100 keys) alone,
in pairs, reversed, with repeats and as a batch of 64, and gets the same bits for each.
`test_varlen_invariant_to_chunking_and_matches_decode` prefills the same sequences whole, in
chunks of 1, 5, 64, 129 and 1,000 rows after their context, packed next to other rows, and
checks every chunk row against the whole prompt's row and every last row against the decode
kernel's, with `torch.equal`. Both ops' variant 2 also run `tests/test_paged.py`'s cases
against SDPA (the fp8 cases skip it).

## The engine

`SparkModel(batch_invariant=True)` runs hgemm 7 for every GEMM (prefill, decode, lm_head) and
variant 2 of both attention ops (decode steps, prefill, both halves of a mixed step, verify
forwards). `Engine(batch_invariant=...)` sets it on its model before the decode graphs are
captured; None keeps the model's own setting, which defaults to `SPARK_BATCH_INVARIANT`.
`Engine(log_logits=True)` keeps every generated token's logits row per request, which is what
the tests compare.

`test_engine_logits_are_batch_invariant` runs ten requests (1 to 300-token prompts, three of
them sharing a 40-token prefix, one sampled with its own seed) through five schedules: one
slot with no graphs and no prefix cache (the reference), 8 slots with graphs, 4 slots with a
64-token prefill budget (chunked prompts, mixed steps), 16 eager slots without mixed steps or
compaction, and a 24-page cache that preempts. Every request's logits, token by token, are
`torch.equal` to the reference, and so are the tokens.
`test_speculative_logits_are_batch_invariant` does the same with `speculative=4`: a verify forward computes its rows through
attention_varlen and GEMMs of a different M, and with the invariant kernels they are the plain
decode steps' bits, so speculation gives greedy decoding's logits exactly, not up to near
ties.

## What it costs

All on the RTX 5090 with the other tenants of the card idle at the start of each run
(nvidia-smi before and after each hold), `scripts/bench_invariant.py`, rows in
`results/batch_invariant.jsonl`.

**The engine**, default against `batch_invariant=True` on the same random weights, 8 of
Llama-3-8B's 32 layers (the 32-layer model does not fit next to the card's other tenants in
the 10 GB I had), bench_serve.py's static batches (prompts log-uniform in 64..2048, 32 new
tokens each, greedy, CUDA graphs on), 8,192-token prefill forwards:

| batch | prompt tokens | decode ms/step, default | invariant | cost | prefill tok/s, default | invariant | cost |
|---|---|---|---|---|---|---|---|
| 1 | 884 | 2.894 | 2.942 | +1.7% | 60,990 | 55,462 | 9.1% |
| 8 | 6,073 | 3.040 | 3.014 | -0.9% | 65,477 | 62,982 | 3.8% |
| 32 | 17,986 | 3.302 | 3.355 | +1.6% | 65,089 | 63,138 | 3.0% |
| 64 | 38,822 | 3.934 | 3.970 | +0.9% | 64,172 | 63,467 | 1.1% |

A second run of the same table, earlier the same day and before the 128-row decode tile, was
within 1% of every cell. Continuous batching, 128 requests through 64 slots (2,015 decode rows
inside mixed steps): 3,212 generated tok/s default, 3,139 invariant, 2.3% less.

Decode costs at most 2%, and all of it is attention: at M <= 64 the GEMMs are the same
kernels.
Prefill costs what the GEMMs lose without Stream-K, which shrinks as the forward grows,
since a long forward is many waves of whole tiles and the last partial wave matters less.
The B = 1 row is one 884-token prompt; at 512 tokens the qkv, o and down GEMMs lose 24 to
27% (below), at 2,048 10 to 16%.

**The GEMMs**, variant 6 against 7 on Llama-3-8B's projections, us per call (ten calls back
to back between events, median of 50 samples at M <= 64, 10 above; B rotated past L2 at M <=
64):

| M | qkv 4096 x 6144 | o 4096 x 4096 | gate_up 4096 x 28672 | down 14336 x 4096 | lm_head 4096 x 128256 |
|---|---|---|---|---|---|
| 1 | 31.6 / 31.6 | 21.7 / 21.8 | 146.7 / 146.6 | 72.1 / 72.2 | 666.1 / 666.5 |
| 64 | 34.6 / 34.1 | 23.1 / 23.1 | 166.7 / 166.6 | 77.1 / 77.0 | 655.5 / 655.8 |
| 128 | 33.4 / 48.3 | 24.8 / 33.8 | 153.9 / 164.1 | 82.7 / 133.6 | 698.5 / 684.5 |
| 256 | 59.2 / 93.3 | 41.9 / 48.8 | 277.3 / 286.8 | 138.2 / 168.5 | 1,204 / 1,198 |
| 512 | 110.4 / 140.3 | 76.4 / 94.7 | 538.4 / 572.8 | 259.8 / 323.5 | 2,343 / 2,351 |
| 2048 | 422.3 / 464.6 | 283.1 / 330.0 | 2,036 / 2,086 | 1,036 / 1,204 | 9,091 / 9,030 |
| 4096 | 871.9 / 943.1 | 568.7 / 656.8 | 4,026 / 4,089 | 2,047 / 2,287 | |

Variant 6 / variant 7. At M = 1, 8, 32 and 64 every projection is within 1.5% and gives the
same bits. From 128 tokens up the cost is the wave quantization Stream-K was built to remove:
at 4096 x 4096 x 4096, 1,024 whole 128 x 128 tiles on 170 SMs are 6.02 waves run as 7, and
o costs 15.5%. gate_up has enough tiles that the last wave matters little (1.6 to 6.6%), and
lm_head is even or slightly ahead. The worst rows are down at 128 tokens, 1.62x in this run
(1.36 to 1.43x in the sweep that chose its tile), and qkv at 256, 1.58x: 96 whole 128 x 128
tiles or 192 smaller ones on 170 SMs, so most of the card idles for one tile's time or two
waves run.

**Attention**, variant 1 against 2, the same timing, the page pool rotated past L2:

| decode batch | K+V MB | v1 us | v2 us | cost |
|---|---|---|---|---|
| 1 x 4096 | 16.8 | 16.6 | 17.9 | 8.2% |
| 1 x 32768 | 134.2 | 91.4 | 95.6 | 4.5% |
| 8 x 4096 | 134.2 | 90.4 | 90.1 | -0.4% |
| 32 x 2048 | 268.4 | 172.8 | 182.0 | 5.3% |
| 64 x 1024 | 268.4 | 173.0 | 178.8 | 3.3% |
| 64 x 512 | 134.2 | 89.9 | 89.1 | -0.8% |
| 1 x 32768 + 50 x 500 | 236.6 | 157.0 | 169.1 | 7.7% |

| prefill | v1 us | TFLOPS | v2 us | TFLOPS | cost |
|---|---|---|---|---|---|
| 1 x 4096 | 659.6 | 208 | 714.4 | 192 | 8.3% |
| 4 x 2048 | 712.5 | 193 | 766.3 | 179 | 7.6% |
| 16 x U(64, 1024) | 352.0 | 136 | 374.1 | 128 | 6.3% |

`bench_paged` (C++, checked against its CPU double-precision reference) on its own sweep
agrees: decode v2 is 4 to 8% behind v1 on six of its eight batches and level or ahead on
8 x 4096 and 32 x U(100, 2000), and varlen v2 is 5 to 6% behind on all four prompt batches.
The decode cost has three parts: a chunk is never split between warps, so the work comes in
256-key pieces (the 1 x 4096 row has 128 of them for 680 warps); every chunk of a sequence
longer than 256 keys writes 2 KB of partials and reads them back; and the fold runs in a
second kernel for every such sequence. The varlen cost is four softmax updates per 64-key
tile instead of one and the fold every fourth tile. Both kernels skip the O rescale when no
row's max moved (a vote per step; a multiply by exactly 1 changes no bit); I did not time
them without it.

## What is still not invariant

- **Quantized weights.** I did not check the fp8, MXFP8, int4, NVFP4 or MXFP4 GEMMs. The
  fp8 GEMM picks its tile and schedule by M the way hgemm does, so I assume none of them is
  invariant, and the engine refuses `batch_invariant` with a quantized format rather than
  half-doing it. Each would need its own whole-K route and a test like hgemm's.
- **The fp8 K/V cache.** The e4m3 decode path permutes D in its fragments and the varlen path
  converts tiles to bf16 in shared memory, so the two compute S in different orders. Variant 2
  is bf16 only and the engine refuses `kv_format="fp8"` in this mode.
- **`prefill_variant`.** The invariant GEMM runs the TMA tiles at prefill; the variant-4
  escape hatch for the TMA kernels' first-launch memory (llm_models.md) is not used.
- **TorchModel** has no such mode (cuBLAS and SDPA pick their kernels by shape).
- **Other GPUs.** Nothing in variant 7 or variant 2 depends on the SM count for its bits (the
  tile and the warp split only move work around), so the same build on a GB10 should give the
  same bits as on the 5090 if the two tensor cores round alike. Not measured.
- **Real weights.** Everything here ran on random weights. The property does not depend on
  the weights, but I did not rerun llm_parity.md's sixteen prompts batched and one at a time
  on the checkpoint: the 32-layer model does not fit next to the card's other tenants today.
- **Long contexts.** The decode combine folds a sequence's chunks one after another, 512 of
  them at 128K tokens, and the partials workspace is sized for B x max_pages keys. Neither
  is measured past the 32K rows above.

It does not make the model more accurate. A near tie is still decided by one bf16 step; it is
decided the same way every time.
