# Serving: a paged K/V cache, varlen prefill and a batching engine

The kernels so far ran one decoder layer for one batch of equal-length sequences with one
contiguous cache each ([layer.md](layer.md)). A server has neither: it holds dozens of
sequences of different lengths, admits new ones while old ones are still generating, and
cannot reserve `max_len` of cache per sequence up front. This note covers the four pieces that
turn the kernels into something that serves a batch the way vLLM does:

1. a paged K/V cache and a RoPE + append kernel that writes new tokens into their pages
   (`rope_append_paged_bf16`, `src/kernels/rope.cu`);
2. a paged flash-decoding kernel whose work split follows the lengths of the batch
   (`paged_decode_bf16`, `src/kernels/attention_paged.cu`);
3. a varlen causal prefill for packed prompts (`attention_varlen_bf16`, same file);
4. an engine that runs all 32 layers of Llama-3-8B with batched prefill, continuous batching
   and CUDA-graph decode (`python/spark_kernels/engine.py`), and `scripts/bench_serve.py`.

Benches: `bench_paged` (both attention ladders, CPU double-precision reference, the contiguous
kernel as `ref_ms`), `bench_rope` (a `rope_paged` row per shape), `scripts/bench_paged_torch.py`
(the kernels against PyTorch), `scripts/bench_serve.py` (the whole model against the same model
in PyTorch). Tests: `tests/test_paged.py`, `tests/test_engine.py`. Every number below is from
the RTX 5090 (sm_120, CUDA 13.2, driver 595.58, torch 2.14+cu130). The roof for the
memory-bound rows is 1,532 GB/s, the `cudaMemcpy` rate; a read-only stream can beat it by up to
10% (the decode section of [attention.md](attention.md)).

## The cache layout

K and V are two tensors of `[num_pages, H_kv, page, D]` bf16 per layer. A sequence owns a list
of page ids, row `b` of a block table `[B, max_pages]` int32, and `seq_lens[b]` says how many of
its keys are valid. Key `j` of sequence `b` is row `j % page` of page `block_table[b][j / page]`.
The page is a power of two and at least 16 tokens; the engine uses 16, vLLM's default.

I put the head dimension outside the page (FlashInfer's HND order, not vLLM's
`[pages, page, H_kv, D]`) because the decode kernel reads one K/V head at a time: with HND,
16 keys of one head at `D = 128` are one contiguous 4 KB run, and a 16-key slab of the
pipeline below is exactly one page. With the head inside, those 16 keys are 16 runs of 256 B,
2 KB apart. The loads are 16-byte `cp.async` requests either way; HND makes a slab one page
lookup and one base address. I did not measure the other order.

A token's cache position is its slot, `page_id * page + row`. The prefill and the decode step
both hand the RoPE kernel one slot per token, so the kernel never walks a block table. A
negative slot skips the write; the engine instead points padding tokens at a reserved scratch
page, which keeps the captured decode graph free of branches on the host side.

Memory: one 16-token page holds 2 x 8 x 16 x 128 x 2 B = 64 KB per layer, 2 MB across the 32
layers. The engine's 7 GB cache is 3,584 pages, 57,344 tokens, next to 16.06 GB of weights.

## RoPE + paged append

`rope_append_paged_bf16` is `rope_append_bf16` ([layer.md](layer.md)) with the token index in
place of `(b, s)`: token `t` rotates at `positions[t]` and writes k and v to `slots[t]`. q comes
out as `[T, H_q, D]`, token-major, which is the layout both attention ops take and also what the
o projection wants, so the attention output needs no transpose (the 25 to 80 us torch copy of
layer.md is gone). The rotation is the same `rotate8` helper in both kernels.

| shape (`bench_rope`, medians of 50) | contiguous us | paged us |
|---|---|---|
| 4096 tokens | 25.6 | 29.7 |
| 8192 tokens | 138.1 | 137.9 |
| 1 token at 4095 | 3.1 | 3.1 |
| 8 tokens at 4095 | 3.2 | 3.1 |

The 4096-token row is L2-resident (100 MB moved against 96 MB of L2) and pays for the scattered
16-token pages there; at 8192 tokens both run at 1,460 GB/s.

## Paged decode

The contiguous flash-decoding kernel splits the keys of each `(b, kv head)` into the same
number of slices, sized for the longest shape. With one 32K-token sequence and fifty 500-token
ones in a batch that split is wrong for every sequence: slices of the long one are 64 times the
work of slices of the short ones, and the short ones finish while the long one's blocks are
still streaming. The paged kernel does not split per sequence at all.

**The flat split.** Every `(b, kv head)` is a segment of `n_b = ceil(len_b / 16)` slabs of 16
keys. Laid end to end the batch has `T = H_kv * sum_b n_b` slabs. The grid is one block per SM
(170 blocks of 4 warps, 680 warps), and warp `w` of `W_eff` streams slabs
`[w T / W_eff, (w + 1) T / W_eff)`, whatever segments they belong to. `W_eff = min(680, T / 4)`,
so a small batch does not spread 2 slabs over each of 680 warps; the warps are numbered
warp-major (`w = warp * 170 + block`), so whatever `W_eff` is, the active warps are spread over
every SM. For 1 x 32768 + 50 x 500 that is 29,184 slabs, 42 or 43 per warp: the long sequence
gets 382 warps and the fifty short ones share the other 298, the same bytes per warp as any
other batch of the same total.

Each block computes the prefix of `n_b` over the batch itself (a block scan of `seq_lens`, 512
sequences at most), so the partition lives on the device. The grid depends only on the SM
count, which is what lets one CUDA graph serve every step of a growing batch.

**One pipeline across segments.** A warp's range usually crosses a few segment boundaries. The
warp runs the same 3-stage `cp.async` pipeline as the contiguous kernel (8 KB of K+V per slab,
two slabs in flight while the third is consumed, 24 KB per warp, 96 KB per block) and does not
drain it at a boundary: the load cursor and the compute cursor each walk `(b, kv head, slab)`
and the loads run two slabs ahead across the boundary. The page id of the next slab is fetched
one load early, so the address is ready when the `cp.async` issues. At a segment's first slab
the warp resets its softmax state and loads the segment's Q fragments.

Those Q fragments come straight from global memory into registers (rows `g` and `g + 8`, k pairs
`c2` and `8 + c2` of each k16 step, the A-fragment layout of `mma.m16n8k16`), not through
shared memory. My first version staged Q in each stage next to its slab: 12 KB stages, 144 KB
per block, which the launch refused with `invalid argument`, since an sm_120 block gets at most
99 KB. With Q in registers the block is 96 KB of pipeline plus 2 KB for the prefix. The Q rows
were written by the RoPE kernel just before and sit in L2, and the round trip overlaps the two
slabs already in flight.

**Pieces and the combine.** When a warp finishes a segment it either covered the whole segment,
and normalizes and writes O itself, or it holds a piece: unnormalized O rows plus `m` and `l`,
written to a workspace slot. A warp has at most two pieces, its first segment (slot 0) and its
last (slot 1); every segment in between is whole. The combine kernel runs next, one block per
`(b, kv head, query row)`, finds the warps that touched its segment with the same arithmetic
(`warp_of(x) = ceil((x + 1) W_eff / T) - 1`), and merges their pieces with the online-softmax
rule. Merging is deterministic (fixed order), so a replayed step gives the same bits
(`test_paged_decode_is_deterministic`).

The combine took three tries. Timings are nsys kernel medians on the equal-length shapes:

| combine | 1 x 4096 | 1 x 32768 | 64 x 1024 |
|---|---|---|---|
| one thread per (row, 8 columns), pieces merged serially, one block per (b, kv head) | | 40 us (ncu, caches flushed) | |
| 16 lanes per (row, 8 columns), 512-thread blocks that each rescan the lengths | 8.3 us | 10.9 us | 8.5 us |
| one 128-thread block per query row, lengths prefix published by the decode kernel | 3.0 us | 6.3 us | 4.3 us |

The first version put a 32K sequence's 85 pieces on one thread each for 64 (row, chunk) items, 8
SMs busy, every piece an L2 round trip after the last. Spreading the pieces over lanes and
reducing with `shfl_xor` fixed the latency chain, but Nsight then showed 6.6 M instructions for
a kernel with almost nothing to merge on the 64-sequence batch: each of 512 blocks of 512
threads rescanned the lengths and ran two passes of 16-lane shuffles for segments with two
pieces. The third version has block 0 of the decode kernel write the prefix to a 2 KB buffer,
and gives each query row its own small block.

I also tried programmatic dependent launch (`griddepcontrol.launch_dependents` at the top of the
decode kernel, `griddepcontrol.wait` in the combine) to hide the second launch. It was slower:
22.7 to 29.7 us at 1 x 4096 and 96 to 108 us at 1 x 32768, and I reverted it. The early combine
blocks sit on the SMs next to the decode blocks for the whole kernel.

**The minimum slabs per warp** (`SPARK_PAGED_MIN_SLABS`, us, `bench_paged` medians of 50):

| batch | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| 1 x 4096 | 19.6 | 17.6 | 18.4 | 18.5 | 28.7 |
| 1 x 32768 | 92.2 | 92.1 | 92.3 | 92.2 | 92.0 |
| 8 x 4096 | 91.8 | 90.9 | 91.8 | 91.0 | 89.8 |
| 64 x 1024 | 175.9 | 175.9 | 176.0 | 175.9 | 175.9 |
| 1 x 32768 + 50 x 500 | 155.6 | 157.1 | 156.4 | 156.4 | 156.5 |

Only the short batch cares: at 1 x 4096 (2,048 slabs), 4 slabs per warp is 512 warps, 3 per SM,
and 32 per warp is 64 warps on 64 SMs, too few bytes in flight. Everything bigger has more
than 4 slabs per warp anyway. The default is 4.

**Measured** (`bench_paged`, medians of 50, the page pool rotated through copies that exceed L2;
the contiguous column is `attention_bf16` variant 3, the flash-decoding kernel of
[attention.md](attention.md), on the same keys copied into `[B, H_kv, L, D]`):

| decode batch (GQA 32/8, D 128, page 16) | K+V MB | v1 us | GB/s | % of 1,532 | contiguous us | contiguous / paged | v0 us |
|---|---|---|---|---|---|---|---|
| 1 x 4096 | 16.8 | 18.4 | 913 | 60% | 17.2 | 0.93 | 2,545 |
| 1 x 32768 | 134.2 | 92.2 | 1,455 | 95% | 87.0 | 0.94 | 20,279 |
| 8 x 4096 | 134.2 | 91.9 | 1,461 | 95% | 86.7 | 0.94 | 2,494 |
| 32 x 2048 | 268.4 | 172.7 | 1,554 | 101% | 164.0 | 0.95 | 1,205 |
| 64 x 1024 | 268.4 | 176.0 | 1,526 | 100% | 164.7 | 0.94 | 605 |
| 1 x 32768 + 50 x 500 | 236.6 | 157.6 | 1,501 | 98% | | | 19,826 |
| 64 x U(1, 8192), 300,104 keys | 1,229 | 755.5 | 1,627 | 106% | | | 4,664 |
| 32 x U(100, 2000), 32,593 keys | 133.5 | 92.1 | 1,450 | 95% | | | 1,197 |

The mixed batch is the one the design is for: one 32K sequence among fifty short ones reads at
98% of the copy roof, the same rate as equal lengths, and the 64 uniform lengths up to 8K run at
106%, the read roof. Against the contiguous kernel on equal lengths the paged one is 5 to 7%
behind, and nsys says where: the decode kernels themselves are even (12.6 against 14.7 us at
1 x 4096, 83.9 against 85.3 at 1 x 32768, 167.6 against 163.0 at 64 x 1024), and the difference is
the combine's 3 to 6 us plus a second launch. The contiguous kernel merges in its last block
instead, which works there because its slices of a head are few and known on the host.

**Nsight Compute** on the decode kernel (one launch after warmup, clocks locked by ncu; the L2
read bytes are the sector count x 32 B):

| batch | duration | DRAM % of peak | L2 read | K+V floor | issue active | warps active |
|---|---|---|---|---|---|---|
| 1 x 32768 | 88.0 us | 89.2% | 134.9 MB | 134.2 MB | 5.4% | 8.3% |
| 64 x 1024 | 175.3 us | 88.8% | 269.8 MB | 268.4 MB | 5.2% | 8.2% |
| 1 x 32768 + 50 x 500 | 154.9 us | 88.6% | 237.9 MB | 236.6 MB | 5.3% | 8.2% |

K and V cross L2 once (the extra 0.5% is Q, the block table and the pieces), DRAM is at 89% of
its 1,792 GB/s peak, and the SMs issue on 5% of cycles: the kernel waits on memory and nothing
else. 216 registers, no spills, one block per SM by shared memory.

**Against PyTorch** (`scripts/bench_paged_torch.py`, 20 back-to-back calls between CUDA events,
median of five; no L2 rotation here, so the 1 x 4096 row is L2-resident for everyone). PyTorch
has no paged attention, so the torch column gathers each sequence's pages to the longest length
in the batch (one copy) and runs `F.scaled_dot_product_attention` with a length mask. The
gathered column passes each K/V head's four query heads as four query rows of that head, plain
multi-head attention with a mask; the `enable_gqa` column is the same gather with the query
heads as heads and the GQA flag, 1.7 to 2.5x slower on the batched rows, so the engine's torch
model uses the first form. The contiguous column is SDPA over caches that are already contiguous per sequence, which only
exists for equal lengths.

| decode batch | ours us | torch gathered us | gathered, enable_gqa us | torch contiguous us | ours vs gathered |
|---|---|---|---|---|---|
| 1 x 4096 | 9.8 | 278.3 | 283.4 | 17.8 | 28.4x |
| 1 x 32768 | 92.1 | 2,350 | 2,440 | 91.8 | 25.5x |
| 8 x 4096 | 90.5 | 500.2 | 829.3 | 88.3 | 5.5x |
| 32 x 2048 | 172.9 | 732.0 | 1,591 | 172.3 | 4.2x |
| 64 x 1024 | 173.2 | 726.7 | 1,535 | 212.4 | 4.2x |
| 51: 1 x 32768 + 50 x 500 | 159.2 | 13,991 | 34,587 | | 87.9x |
| 64: U(1, 8192), 300,104 keys | 752.1 | 5,335 | 11,889 | | 7.1x |
| 32: U(100, 2000), 32,593 keys | 90.7 | 637.4 | 1,490 | | 7.0x |

The gather is what costs torch: it writes and reads the padded cache once more, and on the
mixed batch it pads fifty 500-token sequences to 32K. Where torch does not have to gather
(contiguous, equal lengths) its flash-decoding kernel is level with ours from 2K keys up and
behind at 64 x 1024, 212 against 173 us.

## Varlen prefill

`attention_varlen_bf16` takes the new tokens of a batch packed back to back, `Q = [T, H_q, D]`
with `cu_seqlens_q [B + 1]`, and reads their K/V from the paged cache, where the RoPE kernel
has just put them. Reading the cache instead of the packed k and v makes the same kernel serve a
prompt chunk appended to a context already in the cache: the causal mask is aligned
bottom-right, query `i` of a sequence with `q_len` new tokens and `seq_len` keys sees keys
`j <= seq_len - q_len + i`, which is the ordinary causal mask when the cache held nothing
before. `tests/test_paged.py` covers both.

Variant 1 is attention variant 2's tile ([attention.md](attention.md)): 128 query rows, 8 warps,
`mma.sync.m16n8k16` for `Q K^T` and `P V`, P repacked from the S accumulators in registers,
K and V in 64-key tiles through a 3-stage `cp.async` pipeline. What changes is where things are:
a K/V row is looked up through the block table (one `__ldg` per row per tile, L1 hits after the
first), a query row is `H_q * D` elements from the next one of its head, and each block finds
its `(sequence, q tile, head)` on the device from `cu_seqlens_q`. The grid is an upper bound on
the tile count, `(T + B * 127) / 128` tiles times `H_q`, and the blocks past the real count exit
after the prefix scan. Blocks are tile-major over the heads, so the four query heads of a K/V
head run next to each other and share its tiles in L2, and within a sequence the tiles run
heaviest first under the causal mask.

**Measured** (`bench_paged`, medians of 50; causal, TFLOPS by the halved count per prompt; the
reference is the dense ladder's top rung, attention variant 5, launched once per prompt on
contiguous copies, which is what a server without a varlen kernel runs short of padding):

| prompts | v1 us | TFLOPS | dense v5 per prompt us | TFLOPS | v5 / v1 time |
|---|---|---|---|---|---|
| 1 x 4096 | 651.5 | 211.0 | 576.4 | 238.5 | 0.88 |
| 4 x 2048 | 701.1 | 196.1 | 663.3 | 207.2 | 0.95 |
| 3000, 1500, 700, 300, 200, 100, 50, 20 | 523.1 | 186.1 | 523.0 | 186.1 | 1.00 |
| 16 x U(64, 1024), 8,353 tokens | 359.0 | 135.1 | 490.0 | 99.0 | 1.36 |

On a single long prompt variant 1 is where variant 2 is on the dense shape (215.5 TFLOPS at
4096 causal), so the gather through the block table costs about 2%; the 12% to variant 5 is the
TMA pipeline and the persistent grid of variants 4 and 5, which variant 1 does not have. On a
batch of short prompts one launch beats sixteen: 1.36x, because the dense kernel's per-launch
cost and its partial last wave are paid once per prompt. Against torch's
`F.scaled_dot_product_attention(is_causal=True, enable_gqa=True)` once per prompt
(`bench_paged_torch.py`), variant 1 is 1.17x on 1 x 4096, 1.33x on 4 x 2048, 1.39x on the
eight mixed prompts and 1.85x on the sixteen short ones.

## The engine

`python/spark_kernels/engine.py`, about 500 lines:

- `ModelWeights.random()` draws all 32 layers on the GPU (14 GB of layer weights in seconds;
  `LayerWeights.random` draws on the CPU). `w_gate_up` is stored interleaved, gate_j at column
  2j and up_j at 2j + 1, the layout `hgemm_swiglu` takes.
- `SparkModel` runs a layer as eight launches: `rmsnorm`, the qkv `hgemm`,
  `rope_append_paged_`, `attention_varlen` (prefill) or `paged_decode` (decode), the o `hgemm`
  with the residual add in its epilogue (`residual=x, out=x`), `rmsnorm`, `hgemm_swiglu`, and the
  down `hgemm` with the residual add. layer.md's layer had nine at decode and ten at prefill,
  with the adds inside the norms, a separate SwiGLU kernel and, at prefill, a transpose copy.
- `Engine` keeps a queue of requests, a fixed number of batch slots and the page free list. A
  step retires finished sequences (their pages go back at once), admits waiting requests into
  free slots while pages and a prefill token budget last (16,384 tokens), prefills the admitted
  prompts as one packed batch, and then runs one decode step over every occupied slot. Pages are
  reserved for the prompt plus the requested tokens at admission, so a running sequence never
  needs a page it cannot get and there is no preemption.
- A sequence keeps its slot from admission to the end, so the decode inputs are per-slot device
  buffers: token id, position, cache slot, length, block table row. A step uses the first `Bp`
  slots, `Bp` the next power of two over the highest occupied slot; empty slots have length 0
  (`paged_decode` writes zeros for them) and write their K/V to the scratch page. One CUDA graph
  per bucket is captured at construction with every slot empty and replayed every step after the
  host copies the positions, slots and lengths of the step (a ring of four pinned buffers, so
  the host can run ahead of the GPU). The next token is the argmax of the logits, written into
  the token-id buffer inside the graph, so the decode loop never reads a token back to the host.

Layer.md's graph was per cache length because the contiguous kernel takes `S_kv` from the host.
Here nothing in the captured step depends on a length: the paged kernel reads `seq_lens` on the
device and the RoPE kernel reads positions and slots from buffers. `test_graph_decode_matches_eager`
runs ten requests through four slots with graphs and without, and gets the same tokens.

**Measured** (`scripts/bench_serve.py --eager`, the whole Llama-3-8B, random bf16 weights, 16.06 GB;
7 GB of cache; greedy). A static row submits B requests at once with prompt lengths drawn
log-uniformly between 64 and 2048 (the same lengths for every backend) and generates 128 tokens
each: prefill is every prompt admitted and prefilled with no decode in between, decode is the
127 steps after it, total is both. Torch is `TorchModel` on the same weights and the same paged
cache: `F.rms_norm`, `@`, the RoPE in fp32, SDPA per prompt at prefill and over the gathered
pages (the grouped-rows form above) at decode; "compiled" runs each decode layer's dense parts
under `torch.compile(mode="max-autotune-no-cudagraphs")`, one compilation per bucket.

| B | prompt tokens | ours prefill tok/s | torch | ours decode ms/step | tok/s | torch ms/step | compiled ms/step | ours vs compiled, decode | total tok/s ours / torch / compiled |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 884 | 16,192 | 13,730 | 9.80 | 101 | 13.77 | 12.11 | 1.24x | 98 / 70 / 79 |
| 8 | 6,073 | 16,427 | 13,718 | 10.36 | 766 | 17.58 | 16.34 | 1.58x | 604 / 380 / 404 |
| 32 | 17,986 | 16,501 | 13,318 | 11.56 | 2,746 | 29.06 | 27.51 | 2.38x | 1,594 / 808 / 840 |
| 64 | 38,822 | 16,401 | 13,299 | 14.26 | 4,453 | 51.79 | 50.21 | 3.52x | 1,954 / 858 / 876 |

Continuous batching, 256 requests (prompts log-uniform in 64..2048, 143,998 tokens; 16 to 255
new tokens each, 36,459 in all) through 64 slots: 730 decode steps with 49.6 sequences each on
average and 165 prefill batches. Ours finishes in 19.54 s, 1,866 generated tokens/s; torch in
46.82 s (779/s), compiled torch in 45.80 s (796/s): 2.4x.

**Where the decode step is.** A step streams every layer's weights (32 x 436.2 MB), the lm_head
(1.05 GB) and every sequence's K/V (131 KB per token across the 32 layers). At B = 1 that is
15.01 + 0.12 GB in 9.80 ms, 1,544 GB/s, 101% of the copy roof. At B = 64 the context in the
middle of the decode phase is 38,822 + 64 x 64 = 42,918 tokens, 5.63 GB, so a step moves
20.6 GB in 14.26 ms, 1,447 GB/s, 94%. The step's cost grows with the K/V bytes and nothing else:
64 sequences cost 1.46x one sequence for 64x the tokens. Torch's step grows 3.8x over the same
range, and the growth is the gathered attention (the table above: 4 to 7x our kernel on these
batches). At B = 1, where attention is a small part of the step, compiled torch is 1.24x
behind. At B = 32 and 64 the gather also pushed torch's caching allocator into OOM warnings
(it frees its cache and retries; no run failed).

**Prefill** runs at 16.4K tokens/s whatever the batch: 14.0 GFLOP per token in the GEMMs (32 x
218.1 M weights x 2) plus about 1% in attention at these lengths, 231 TFLOPS, where hgemm v6
runs alone. Torch is 1.18 to 1.24x behind on the same packed batch, its per-prompt attention
loop included.

**Graphs.** Replaying the captured step saves 0.07 ms at B = 1 (9.80 against 9.87) and 0.06 ms
at B = 64: 0.4 to 0.7%. It is layer.md's finding at model scale: the host queues a step's
roughly 260 launches faster than the GPU runs them, so the queue never drains and eager already runs at
the back-to-back rate. The graph is kept because it costs nothing and removes the host from the
critical path, which matters on a slower host or with a smaller model.

llm_serving.md has what changed after this: graphs only for buckets up to 64 (a larger one
could keep a kernel workspace that a prefill had freed), slot compaction, prefill and decode
rows in one forward, stop ids, and the engine against vLLM and transformers on the real
checkpoint.

## Correctness

- `tests/test_paged.py`: `paged_decode` (both variants) against SDPA in fp32 per sequence on the
  gathered keys, six batches: empty sequences, one key, partial pages and slabs, one 5000-token
  sequence among short ones, GQA groups 1, 4 and 8, `D` 64 and 128, pages of 16, 32 and 64, the
  cache past each sequence's length filled with NaN (so a key read past the length would show).
  Plus bitwise determinism across calls, agreement with the contiguous kernel on equal lengths,
  lengths changed in place between two launches, and bad input rejected. `attention_varlen`
  (both variants, causal and not) on five batches including empty prompts and prompts appended
  to a context. `rope_append_paged_` against the torch rotation and slot writes, with a skipped
  slot.
- `tests/test_engine.py`: `SparkModel` against `TorchModel` on a two-layer model (packed prefill
  of four prompts, then three teacher-forced decode steps, logits within 2% relative error),
  and graph against eager decode token for token under continuous batching, with every page
  back on the free list at the end.
- `bench_paged` validates every variant against a CPU double-precision reference: every row of
  up to eight sequences per decode batch and 256 sampled rows per prefill batch.

## What remains

- **The combine launch.** The paged decode is 5 to 7% behind the contiguous kernel on equal
  lengths, all of it the combine and its launch. Merging in the last warp to finish a segment
  (a counter per segment, the contiguous kernel's scheme) would remove the launch, but a long
  segment has one piece per warp that streamed it, 85 for a 32K sequence, and a single warp
  merging 85 pieces would put that latency chain on the kernel's tail. A two-level merge,
  pieces of the same segment in one block first, would need block-contiguous warp numbering,
  which gives up spreading a short batch over every SM.
- **The prefill on the TMA pipeline.** Variant 1 is variant 2's tile; variant 5's TMA pipeline
  and persistent grid are 12% faster on a single long prompt. A TMA box cannot gather through a
  block table in one piece, but a 64-key tile is four 16-token pages at `D = 128`: four boxes of
  a 4-D tensor map over `[pages, H_kv, page, D]`, issued by the producer lane.
- **Chunked prefill.** The kernel takes a chunk after a context; the engine does not use it yet.
  It admits a prompt whole, so a prompt larger than the prefill budget waits until the budget is
  empty, and a long prefill delays the decode step of the running batch. Mixing a prompt chunk
  into each decode step (vLLM's chunked prefill) needs a kernel that takes decode rows and
  prefill rows in one launch.
- **Preemption and a smaller page reservation.** Pages for all requested tokens are taken at
  admission. A server that reserves as it goes needs to preempt or swap when the pool runs out.
- **Sampling** is greedy argmax; the logits of a 64-sequence step are 16 MB and a top-p sampler
  would read them once more.
