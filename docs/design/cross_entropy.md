# Linear cross-entropy

The loss at the top of a language model, `F.cross_entropy(h @ w.T, y)`, for a 128K
vocabulary, without the `[T, V]` logits: `src/kernels/cross_entropy.cu`,
`sk.linear_cross_entropy(h, w, targets, ignore_index=-100, reduction="mean", variant=-1,
chunk=None)` in `python/spark_kernels/ops.py`. Bench: `scripts/bench_cross_entropy.py`
(the whole op against torch eager and `torch.compile`, and `--pieces` for the kernels one
at a time), rows in `results/cross_entropy.json`. Tests: `tests/test_cross_entropy.py`.

Measured on the RTX 5090 on 2026-10-23 (driver 595.58, CUDA 13.2, torch 2.14+cu130) with
the card shared: other tenants held about 18 GB, so the bench caps its own process at 10 GB
(`torch.cuda.set_per_process_memory_fraction`) and records a method that would need more as
not fitting. Every timed method's first call was checked first, its loss and both gradients
against the first method that fit at that shape (torch, `torch.compile` where torch did
not fit, variant 1 where neither did), and the 47 tests in `tests/test_cross_entropy.py` pass.

## The problem

h is `[T, D]` bf16, the last hidden states, and w is the lm_head as `nn.Linear` stores it,
`[V, D]`. Torch's spelling makes the logits in bf16, then a float32 copy for the loss, and
`log_softmax` saves its float32 output for the backward. The backward then makes a float32
gradient of the same size, and its bf16 cast feeds the two products. For Llama 3
(V = 128256, D = 4096):

| T | bf16 logits | one fp32 copy |
|---|---|---|
| 2048 | 0.53 GB | 1.05 GB |
| 4096 | 1.05 GB | 2.10 GB |
| 8192 | 2.10 GB | 4.20 GB |
| 16384 | 4.20 GB | 8.41 GB |

w and dw are 1.05 GB each whatever T is, and Qwen 3's 151936 is 18% more of everything.
Two or three of the fp32 copies are alive at once, so at 8192 tokens torch's loss does not
fit in 10 GB even without a backward (measured below). Two published fixes: Liger Kernel's
fused linear cross-entropy computes the logits a chunk of rows at a time and does the
whole backward inside the forward, so only one chunk is ever alive; Apple's Cut
Cross-Entropy reduces each logits tile inside the GEMM, never stores the logits at all,
and recomputes them in the backward.

## The ladder

| variant | forward | backward | held between the passes | extra memory at the peak |
|---|---|---|---|---|
| 0 | torch: bf16 logits, `F.cross_entropy` in fp32 | torch autograd | the fp32 log-softmax | several `[T, V]` |
| 1 | per chunk: logits GEMM, the fused row kernel (loss and gradient row in place), dh and dw by `hgemm` | a scale by the upstream gradient | dh and dw | one chunk of logits and one transposed slab |
| 2 | one GEMM whose epilogue keeps (max, sum of exp) per row per 128-column tile, a combine | per chunk: the GEMM again with softmax minus one-hot in its epilogue, dh and dw by `hgemm` | the `[T]` log-sum-exp | forward: the `[T, V/128]` partials; backward: as variant 1 |

Variant 1 does three GEMMs of 2TVD flops each, the same as torch. Variant 2 does four: the
price of holding nothing between the passes is computing the logits twice. `variant=-1`
takes 1 when a gradient is needed and 2 when not (an eval loss).

## The GEMM

`ce_gemm_bf16` is the bf16 kernel of `grouped_gemm` with one expert: `x = h w^T` with both
operands K-contiguous, which is the layout the lm_head already has (hgemm takes `B` as
`[K, N]`, so it would need a transposed copy of w, 1 GB). A persistent grid of one
288-thread block per SM, one producer lane issuing two TMA boxes per 64-k stage, eight
consumer warps on `mma.sync.m16n8k16`, the last partial wave plus one full wave split along
K Stream-K style and summed through hgemm's deterministic chain fixup. Tiles are numbered
m-tile fastest, so the 170 tiles in flight share a few 128-row strips of w and walk h, the
smaller operand, out of L2. Nothing in the mainloop is new; the three modes differ only in
what the consumer warps do with a finished tile, and that runs once per tile on the full
fp32 sums (the piece that ends a tile's K chain), so every mode gives the same bits every
run.

**LOGITS** stores bf16, the plain epilogue.

**STATS** never stores the tile. Lane (g, t) of a warp holds 8 values of each of its 8 rows
(columns `2t, 2t+1` of four n8 tiles). The row's max over the warp's 32 columns is two
`__shfl_xor` away (the four lanes of a quad share a row), then the sum of
`exp(x - max)` the same way. The four warps that share the tile's rows meet in 8 KB of
shared memory set aside next to the stages (two buffers, alternating per tile, so one named
barrier per tile is enough), and one thread per row merges the four pairs with the online
softmax rule and writes one `(max, sum)` float2 to `partials[row][tile_n]`. The lane whose
columns hold the row's target writes that logit to `target_logit[row]`; exactly one lane in
the grid does, so there is no race. `ce_combine` is a warp per row over the V/128 pairs:
`lse = max + log(sum)`, `loss = lse - x_y`. At V = 128256 the partials are 1002 pairs per
row, 8 KB, 0.4% of the bf16 row they stand for.

**GRAD** takes the forward's lse and a device scale (1/rows-that-count for "mean", times the
upstream gradient, so the backward has no host sync) and stores
`bf16((exp(x - lse) - [col == y]) * scale)`, zero on rows that do not count. That is
variant 1's row kernel folded into the GEMM: the gradient chunk comes out of the GEMM
without the logits ever being written.

## Variant 1: the row kernel and the chunk

`cross_entropy_rows_bf16` is softmax variant 2's shape for rows too wide for registers: a
1024-thread block per row, an online (max, sum) over 16-byte vectors, a block merge, then a
second pass that writes `(exp(x - lse) - [col == y]) * scale` over the logits in place. A
128256-wide bf16 row is 256 KB, so softmax variant 3's row-in-registers cannot hold it. The
grid is one block per SM, persistent over the rows: 170 rows in flight are 44 MB, so the
second read of a row comes from L2 and DRAM sees one read and one write. Thread 0 reads the
target's logit before the block merge, because the second pass overwrites it. A row whose
target is `ignore_index` is zeroed without being read.

dh for the chunk is `hgemm(g, w)`: the gradient `[C, V]` is K-contiguous and w `[V, D]` is
hgemm's `[K, N]`. dw is the awkward one: `dw += g^T h_c` needs `g^T` as a K-contiguous
`[V, C]`, and hgemm has no transposed-A mode. So the gradient goes through a padded
transpose (`transpose_pad_bf16`, 64 x 64 tiles through shared memory) one 16384-row slab of
the vocabulary at a time into a 64 MB buffer, and each slab is one `hgemm(g_slab^T, h_c,
residual=dw[v0:v1], out=dw[v0:v1])`, the fused in-place accumulate. The chunk's rows are
padded to a multiple of 64 with zeros in the slab and in a copy of the last chunk of h,
since hgemm wants K % 64 == 0. dw is summed in bf16 across chunks, one rounding per chunk,
as Liger does.

## The pieces, one at a time

`bench_cross_entropy.py --pieces`, one chunk of 2048 rows, D = 4096, medians of 50
launches, all in one session at 2.8 GHz and the 600 W cap. TFLOPS count 2 x 2048 x V x D;
GB/s count the bytes the kernel must move (one read of the chunk, or a read and a write).

| piece | V = 128256 | | V = 151936 | |
|---|---|---|---|---|
| torch `h @ w.T` (cuBLAS) | 9.160 ms | 234.9 TFLOPS | 10.939 ms | 233.0 TFLOPS |
| `ce_gemm` LOGITS | 8.850 ms | 243.1 TFLOPS | 10.495 ms | 242.9 TFLOPS |
| `ce_gemm` STATS + `ce_combine` | 8.959 ms | 240.2 TFLOPS | 10.634 ms | 239.7 TFLOPS |
| `ce_gemm` GRAD | 8.966 ms | 240.0 TFLOPS | 10.638 ms | 239.6 TFLOPS |
| `ce_rows`, loss only | 0.317 ms | 1,657 GB/s | 0.376 ms | 1,653 GB/s |
| `ce_rows`, loss and gradient | 0.741 ms | 1,419 GB/s | 0.893 ms | 1,394 GB/s |
| transpose, one 16384-row slab | 0.080 ms | 1,688 GB/s | 0.078 ms | 1,732 GB/s |
| `hgemm` dh = g w | 8.669 ms | 248.2 TFLOPS | 10.371 ms | 245.8 TFLOPS |
| `hgemm` dw[slab] += g^T h | 1.146 ms | 240.0 TFLOPS | 1.147 ms | 239.6 TFLOPS |

And the whole forward of variant 2 at T = 8192, against torch's product alone: STATS +
combine 35.26 ms (244.1 TFLOPS) against 36.52 ms (235.7) at V = 128256, 41.82 ms against
43.28 ms at V = 151936.

What that says:

* The lm_head GEMM is 3.4 to 4.2% ahead of cuBLAS on this shape.
* The STATS epilogue costs 1.2% over the plain store (8.959 against 8.850 ms) and stores
  nothing: 64 exponentials and a few shuffles per lane per tile on a tile that is 64 k-steps
  of mma. GRAD costs the same. The forward without the logits is cheaper than torch's
  product alone.
* The row kernel's loss-only pass reads the chunk at 1,657 GB/s, a read-only stream above
  the 1,532 GB/s copy rate, as the decode kernels do. With the gradient it reads the row
  twice and writes it once and runs at 93% of the copy rate counting one read and one
  write, so the second read is coming from L2.
* Per chunk at V = 128256, variant 1's step is about 8.85 + 0.74 + 8.67 + 7.83 x (0.08 +
  1.15) = 27.9 ms, 95% of it in the three GEMMs. The transposes are 2.2%.

## Measured

```
python scripts/bench_cross_entropy.py --vocab 128256 --tokens 8192        # one shape per GPU hold
python scripts/bench_cross_entropy.py --pieces --vocab 128256 151936 --tokens 8192
python scripts/bench_cross_entropy.py --vocab 128256 --tokens 8192 --methods v1 v2 --chunk 1024
```

D = 4096, 5% of the targets ignored, medians of 50 launches.
"Peak" is `torch.cuda.max_memory_allocated` during the call minus what was allocated
before it (h, w and the targets), so for the forward and backward it includes the returned
dh and dw, and dw alone is V x D bf16: 1.05 GB for Llama 3, 1.24 GB for Qwen 3. torch is
`F.cross_entropy((h @ w.T).float(), y)` in eager mode, torch.compile is the same function
under `torch.compile` (default mode), the backward is `torch.autograd.grad` of the loss for
h and w. Variants 1 and 2 at their default 2048-row chunk.

The forward alone (under `no_grad`, the loss):

| V | T | torch | torch.compile | v1 | v2 | v2 peak |
|---|---|---|---|---|---|---|
| 128256 | 2048 | 12.43 ms, 2.10 GB | 9.54 ms, 0.53 GB | 9.15 ms, 0.53 GB | **8.99 ms** | 0.02 GB |
| 128256 | 4096 | 24.79 ms, 4.20 GB | 18.94 ms, 1.05 GB | 18.39 ms, 0.53 GB | **17.97 ms** | 0.03 GB |
| 128256 | 8192 | does not fit in 10 GB | 37.92 ms, 2.10 GB | 36.72 ms, 0.53 GB | **35.79 ms** | 0.07 GB |
| 128256 | 16384 | does not fit in 10 GB | 78.09 ms, 4.20 GB | 73.70 ms, 0.53 GB | **71.74 ms** | 0.07 GB |
| 151936 | 2048 | 14.82 ms, 2.49 GB | 11.30 ms, 0.62 GB | 10.88 ms, 0.62 GB | **10.67 ms** | 0.02 GB |
| 151936 | 4096 | 29.60 ms, 4.98 GB | 22.42 ms, 1.25 GB | 21.79 ms, 0.62 GB | **21.28 ms** | 0.04 GB |
| 151936 | 8192 | does not fit in 10 GB | 44.73 ms, 2.49 GB | 43.60 ms, 0.62 GB | **42.46 ms** | 0.08 GB |
| 151936 | 16384 | does not fit in 10 GB | 92.68 ms, 4.98 GB | 87.15 ms, 0.62 GB | **84.95 ms** | 0.08 GB |

The forward and the backward:

| V | T | torch | torch.compile | v1 | v2 |
|---|---|---|---|---|---|
| 128256 | 2048 | 35.10 ms, 3.15 GB | 29.17 ms, 1.59 GB | **28.29 ms**, 1.66 GB | 36.67 ms, 1.66 GB |
| 128256 | 4096 | does not fit in 10 GB | 58.16 ms, 2.13 GB | **56.76 ms**, 1.68 GB | 73.50 ms, 1.68 GB |
| 128256 | 8192 | does not fit in 10 GB | 115.39 ms, 3.22 GB | **113.50 ms**, 1.71 GB | 146.91 ms, 1.71 GB |
| 128256 | 16384 | does not fit in 10 GB | 228.57 ms, 5.39 GB | **227.99 ms**, 1.78 GB | 295.37 ms, 1.78 GB |
| 151936 | 2048 | 41.67 ms, 3.74 GB | 34.61 ms, 1.89 GB | **33.62 ms**, 1.95 GB | 43.48 ms, 1.95 GB |
| 151936 | 4096 | does not fit in 10 GB | 68.52 ms, 2.52 GB | **67.43 ms**, 1.97 GB | 87.07 ms, 1.97 GB |
| 151936 | 8192 | does not fit in 10 GB | **134.67 ms**, 3.80 GB | 134.79 ms, 2.00 GB | 174.38 ms, 2.00 GB |
| 151936 | 16384 | does not fit in 10 GB | does not fit in 10 GB | **269.76 ms**, 2.07 GB | 349.29 ms, 2.07 GB |

How to read it:

* **Memory is the result.** Our peak is flat in T: the gradients every method returns plus
  one chunk of logits (0.53 or 0.62 GB) and one 64 MB slab. Torch eager does not fit a
  4096-token backward in 10 GB, nor a forward at 8192. `torch.compile` is much better than
  eager (it fuses the loss and keeps only the bf16 logits), but it still holds them, so its
  peak grows by the logits, 0.26 GB per 1024 tokens of Llama 3's vocabulary: 3.22 GB against
  our 1.71 at 8192 tokens, 5.39 against 1.78 at 16384, and Qwen 3 at 16384 does not fit. At
  2048 tokens its peak is 3 to 4% below ours: one chunk is all of T there, and it keeps no
  transposed slab.
* **Speed is a tie with `torch.compile` on the backward.** Variant 1 is 0.3 to 3.1% ahead,
  and 0.1% behind at Qwen 3 and 8192 tokens. Every method runs the same three GEMMs, which
  are 95% of our step, and our lm_head product is only 3.4 to 4.2% ahead of cuBLAS's.
  Against eager torch variant 1 is 1.24x at 2048 tokens.
* **The forward without logits is the fastest forward.** Variant 2 is 5.3 to 9.1% ahead of
  `torch.compile`, 1.38 to 1.39x ahead of eager, and holds 16 to 78 MB where compile holds
  the logits. For an eval loss that is the right call, and `variant=-1` makes it.
* **Variant 2's backward lost, as the flop count says it must.** It runs four GEMMs to
  variant 1's three, and at every shape it is 1.29 to 1.30x slower than variant 1 for the
  same peak memory: that peak is set by the backward's chunk, which both share.
* **v2's forward over all of T lost at 16384 tokens.** As one STATS GEMM it measured
  75.0 ms (229.5 TFLOPS) against 71.7 ms (240.0) in 8192-row pieces: past 8192 rows h is
  128 MB, more than the L2 holds while w streams through. It runs in pieces now.

The chunk, variant 1 at V = 128256, T = 8192:

| chunk | forward | peak | forward and backward | peak |
|---|---|---|---|---|
| 512 | 37.65 ms | 0.13 GB | 120.96 ms | 1.27 GB |
| 1024 | 37.11 ms | 0.26 GB | 116.42 ms | 1.41 GB |
| **2048** | 36.72 ms | 0.53 GB | 113.50 ms | 1.71 GB |
| 4096 | 36.62 ms | 1.05 GB | 112.84 ms | 2.30 GB |
| 8192 | 36.52 ms | 2.10 GB | 112.56 ms | 3.49 GB |

A smaller chunk costs time in the dw accumulate, which reads and writes all of dw once per
chunk (2.1 GB at Llama 3's size) and does less math per pass, and in the dh GEMM's shorter
M. 2048 is where it stops paying: 0.6% slower than 4096 for 0.6 GB less. Variant 2's
backward follows the same curve, 154.5 ms at 512 to 146.2 at 8192.

## Correctness

`tests/test_cross_entropy.py`, 47 tests, all pass on the card: the loss and both gradients
of variants 0, 1 and 2 against torch autograd in fp32 on the same bf16 inputs, on four
shapes (T = 77 to 1000, two not a multiple of 64; D = 64 to 512; V = 384 to 4096) with 15%
of the rows ignored, at a chunk of 64 rows (several chunks, a ragged last one) and of 4096
(one chunk); "sum" with an upstream gradient of 0.37; a custom ignore_index with only h
requiring a gradient; the no-grad forward; every row ignored (nan, as torch); the shape
rules; variant 2's forward in STATS pieces of 128 rows. And the pieces on their own: the
row kernel against `F.cross_entropy` on bf16 logits at V up to 151936, the STATS
epilogue's lse and loss and the GRAD epilogue against fp32 torch, the LOGITS epilogue
against the fp32 product, two runs of a Stream-K shape bit for bit, and the padded
transpose. The loss is within 2e-3 of fp32; the gradients within 2%
of their largest entry, the same bound the attention backward uses.

## What is left

**A v2 backward with no chunk.** Cut Cross-Entropy's backward recomputes a logits tile,
turns it into the gradient tile in registers and feeds it straight into both products,
`dh[t-tile] += G w[v-tile]` and `dw[v-tile] += G^T h[t-tile]`, so not even a chunk is
stored. I did not build it. The arithmetic is why it is a project and not an epilogue: a
block that owns a 128 x 128 tile of G contributes a `[128, D]` partial to dh and another to
dw, so every element of dh receives one partial from each of the V/128 vocabulary tiles and
every element of dw one from each of the T/128 token tiles: 33.6 G fp32 additions into each
at T = 8192, V = 128256. Those only work as atomics that stay in L2, which means a tile
order that keeps a vocabulary strip of dw (2 MB in fp32) resident while its token tiles run, and I
have not measured this card's L2 atomic rate. Cut Cross-Entropy also skips the tiles whose
softmax is below bf16's resolution everywhere, which for a trained model's peaked
distribution is most of them. On the random weights these benches use the softmax is not
sparse, so that filter cannot be judged here; it needs real activations and a real lm_head.

**hgemm with a transposed A.** dw goes through a transposed copy of the gradient because
hgemm takes A only K-contiguous. A mode that loads the A tile from `[K, M]` (two 64-column
TMA boxes and `ldmatrix.trans`, the way the B tile already loads) would remove the
transpose and its 64 MB slab; on the pieces above that is about 2% of a step.

**The transposed gradient straight from the epilogue.** In variant 2 the GRAD epilogue
could write `g^T` next to `g` (the accumulator layout gives each lane two rows of two
columns; one `__shfl_xor` turns that into 4-byte stores of two rows of one column), which
would save the transpose's read. I left it for the transposed-A mode, which removes the
whole pass.

**dw in fp32.** dw is summed in bf16 across chunks, one rounding per chunk, as Liger does;
with 2048-row chunks that is four roundings at 8192 tokens. An fp32 accumulator would cost
2.1 GB more for Llama 3's lm_head.
