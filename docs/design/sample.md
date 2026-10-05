# Sampling

`sample_bf16` in `src/kernels/sample.cu`, `sk.sample` in Python, and the engine's sampled
decode step (`SamplingParams` in `python/spark_kernels/engine.py`).

**Status: not measured yet.** The kernel was written on a machine without a GPU or nvcc; it
has not been compiled, `tests/test_sample.py` has not run, and `scripts/bench_sample.py` has
no rows. The selection logic of both variants was checked against each other and against the
float64 rule in a NumPy model of the kernel (same fixed-point weights, same radix passes, same
chunked scan), which is all this note can claim until the box runs it.

## The rule

Per row of logits `x` (bf16, read as fp32), with `m = max(x)` and per-row `T`, `k`, `p`:

1. `T <= 0` is greedy: the first index of the max, what `torch.argmax` returns.
2. Weights `w_i = floor(exp((x_i - m) / T) * 2^40)` as unsigned 64-bit integers. A token whose
   probability is under 2^-40 of the top token's gets weight 0 and is never drawn (128K such
   tokens together are under 2^-23 of the top one's mass).
3. top-k (`k > 0`): keep the tokens whose value is at least the k-th largest among the tokens of
   nonzero weight. Tokens tied with the k-th are all kept.
4. top-p (`p < 1`), on what top-k kept (total `Z`): keep the tokens whose value is at least the
   largest `v` with `W(>= v) >= ceil(p Z)`, `W(>= v)` the kept weight at or above `v`. This is
   the usual rule (sort descending, keep a token while the probability before it is under `p`)
   with tied values kept or dropped together, where a sort keeps an arbitrary subset of them.
5. The token is the inverse CDF in index order at `r = floor(u * Z_f / 2^64)`: the smallest
   index whose running sum of kept weights exceeds `r`, with `Z_f` the kept total and `u` the
   first 64 bits of Philox4_32_10 (curand) block `offset` of stream `seed`.
6. Each call adds 1 to every row's offset in device memory.

The weights are integers so that every sum is exact: the token is a function of the row and
its five parameters, whatever order a block's reductions and atomics run in and whatever else
is in the batch. Floating-point sums in a different order would move a threshold or the
point `r` by an ulp now and then, and a request would not reproduce from its seed.

## Why on the device

The engine's decode step is a CUDA graph that ends by writing each slot's next token into the
id buffer it reads at the next replay; the host never sees a token on the way. Sampling has to
fit in that graph: the parameters are per-slot device tensors (`temperature`, `top_k`, `top_p`,
`seeds`, `offsets`), so a replay reads whatever the scheduler wrote there, and the kernel
advances the offsets itself, so the next replay draws a fresh number.

The usual PyTorch path (`scripts/bench_sample.py`, `torch_sample`) is a `topk` or a full
descending sort of the 128256-wide row, a softmax, a cumsum, a mask and `torch.multinomial`,
each its own launch, and for top-p without top-k a sort of 128K keys per row.

## Why the row is not held in registers

A 128256-wide row is 501 KB in fp32 and 251 KB in bf16. An SM's register file is 64K 32-bit
registers, 256 KB, and its shared memory at most 99 KB per block on sm_120. At 1024 threads a
thread may use 64 registers, so the 125 values per thread a register-resident row needs do not
fit, in either precision, next to anything else (softmax variant 3 holds rows up to 32K
elements, 32 fp32 values per thread at 1024 threads). The first pass reads the row from DRAM;
the later passes read it again and find it in L2: a batch of 64 rows is 16 MB of the 5090's
96 MB.

## The ladder

| Variant | Mapping | Thresholds | Reads of the row |
|---|---|---|---|
| 0 | one thread per row | bisection over the 16-bit key, one pass per probe | 37 for top-k + top-p |
| 1 | one 1024-thread block per row | two 8-bit radix passes per threshold over 256-bin histograms | 1 to 5 (below) |

Both give the same token for the same inputs (`test_variants_give_the_same_token`): the
thresholds are exact integer searches over the same weights, and variant 1's scan visits the
tokens in index order, like variant 0's walk.

### Variant 1, pass by pass

The key is the bf16 bit pattern mapped so an unsigned compare orders it like the value (flip
all bits of a negative, set the sign bit of a positive). bf16 has 16 bits, so a radix select
over 8-bit digits takes two passes.

| Pass | Reads | Does |
|---|---|---|
| A | DRAM | max and first argmax (coalesced 16-byte loads, four in flight per thread); greedy rows end here |
| H1 | L2 | high-byte histogram of the nonzero-weight tokens: a count and a weight per bin |
| K2 | L2 | top-k: the low-byte histogram inside the bin where the k-th count falls |
| P2 | L2 | top-p: the low-byte histogram inside the bin where the `ceil(p Z)` weight falls (skipped when it is K2's bin) |
| S | L2 | each thread's kept weight over a contiguous chunk, a block scan, then the one thread whose range holds `r` walks its chunk again |

So greedy is one read, plain temperature two (A, S), top-k or top-p four, both five (four when
the two thresholds fall in the same high-byte bin). H1 serves both thresholds, and top-p's
target needs the top-k total, which K2's histogram gives without another pass.

Logits crowd into a few exponents, so most of a row lands in a handful of high-byte bins and a
plain shared-memory histogram would serialize on those addresses. The lanes of a warp that
share a digit are grouped with `__match_any_sync`; their weights are summed with two
`__reduce_add_sync` (the weight split into 21 + 20 bits so 32 lanes fit 32 bits) and one lane
does the two atomics for the group. The token pass uses contiguous chunks (thread t owns
elements `t * 128 ...` for V = 128256), so the exclusive scan over threads is a scan in index
order.

## The engine

`Engine.submit(prompt, max_new, SamplingParams(temperature, top_k, top_p, seed))`. At
admission the parameters go into the slot's device tensors and the offset is set to the
sequence's `generated` count: the token at output index j is always drawn from block j of the
request's stream. `_compact` moves the parameters with the slot, a preempted request resumes
at its own index, and the prefill path (a prompt's first token, mixed-step decode rows) samples
through the same kernel with the offsets gathered and written back. So a seeded request gives
the same tokens whatever max_batch, bucket, slot or neighbours it runs with, provided its
logits are the same bits (the engine tests already require that of the greedy tokens).

Each decode bucket is captured twice, ending in argmax and in `sk.sample`. A step replays the
sampling graph only while some running request samples; a run with only greedy requests
replays the argmax graphs it replayed before, bit for bit.

## What to measure

- `python scripts/bench_sample.py`: B in 1, 8, 64 at V = 128256 for greedy, temperature,
  top-k = 50, top-p = 0.9 and both, against `torch_sample`. The floor is pass A: 251 KB per
  row from DRAM.
- B = 1 runs on one SM. If the L2 passes dominate, a split of the row over a few blocks (a
  cluster, or a second kernel that merges the histograms) is the next rung.
- Contention of the shared-memory atomics in H1 on real logits (Nsight Compute,
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_atom`).
- After H1 the top-k candidates for a small k usually fit in shared memory; compacting them
  there would turn K2, P2 and S into shared-memory passes.
