# The Grace side: which parts of a decode step belong on the GB10's Arm cores

The GB10 has 20 Arm v9 cores (10 Cortex-X925, 10 Cortex-A725) on the same LPDDR5X as its
48 SMs: one 273 GB/s pool, one address space, no PCIe between them. Source: `include/spark/cpu.h`,
`src/cpu/w1gemv_neon.cpp` (the 1-bit GEMV), `src/cpu/route_cpu.cpp` (routing, sampling, the
bandwidth probe), `src/cpu/bench_cpu.cpp` (the bench, no CUDA), `python/csrc/cpu_bindings.cpp`
and the `cpu_*` ops in `ops.py`. Tests: `tests/test_cpu.py`.

**Not measured.** Nothing in this note has run on a GB10. The code was compiled and
exercised on an Apple Silicon Mac (NEON, the dot-product extension) to shake out the build and
check the rungs against the scalar reference and the torch reference; none of those numbers
is a result and none is quoted here. Byte counts and floors from 273 GB/s are arithmetic; the
CPU's own bandwidth is left as a variable, `B_cpu`, until `bench_cpu` prints it.

## The question

A decode step at batch 1 is a weight stream, and [w1gemm](w1gemm.md) puts DeepSeek V4.1
Flash's six routed experts per layer at 29.9 MB of 1-bit weights and scales, 109 us at the
roof. Around that stream sit pieces that are small, serial, or both: the shared expert (one
more expert's worth of bytes, every token, no routing), the router (a 5120 x 384 bf16 GEMV
and a top-6 of 384), the sort of six ids, the sampler over 129,280 logits, the K/V page
bookkeeping. On the GPU each is a launch or three, each a few microseconds of latency with
almost no bytes behind it, and a CUDA graph absorbs the launches but not the dependencies
(the routing has to finish before the grouped GEMM can start; the sample has to finish before
the next token's embedding lookup).

The cores can read the same bytes. So: which of these pieces should they take, and does
running them concurrently with the GPU's expert stream gain anything?

## The bandwidth argument

Both sides draw from the same 273 GB/s. If the GPU streams the six routed experts (29.9 MB)
while the cores stream the shared expert (5.0 MB), the layer still moves 34.8 MB, and its
floor is still 34.8 MB / 273 GB/s = 127 us, whoever reads what. Concurrency cannot beat the
roof; it can only split it. The GPU's own kernel slows by exactly the bytes the cores took,
if both are at the roof.

What concurrency can buy is the time the GPU leaves idle: the routing's dependency stall,
the gaps between the three grouped launches, the sampler at the end of the step, the host-side
bookkeeping between steps. If the cores do those while the GPU streams, the step gets shorter
by the idle time, not by any bandwidth. That is a latency-hiding gain, worth tens of
microseconds per layer at most, and it is the honest upper bound on what this experiment can
show.

There is a second, smaller point. The GPU's grouped kernel reads 6 experts in 144 strips each;
at 48 SMs and 1-bit weights the blocks are short and the tail is a real fraction of the time
([w1gemm](w1gemm.md), "The ladder"). The shared expert on the GPU is one more small
launch with the same tail. On the cores it is a 2304-row loop with no launch at all. Whether
that is faster in wall time depends on `B_cpu` and is the thing to measure.

## The bytes

DeepSeek V4.1 Flash, hidden 5120, expert intermediate 2304, group 128, bf16 scales. One
projection is 11,796,480 weights. The shared expert is taken to have a routed expert's shape,
as `moe.py` assumes.

| piece | 1 bit (packed + scales) | 2 bits | at 273 GB/s (1 bit) |
|---|---|---|---|
| one projection, 5120 x 2304 | 1,474,560 + 184,320 = 1.66 MB | 3.13 MB | 6.1 us |
| one expert (gate, up, down) | 4.98 MB | 9.40 MB | 18.2 us |
| the shared expert, per layer | 4.98 MB | 9.40 MB | 18.2 us |
| 6 routed experts, per layer | 29.9 MB | 56.4 MB | 109 us |
| routed + shared, per layer | 34.8 MB | 65.8 MB | 127 us |
| the router, 5120 x 384 bf16 | 3.93 MB | same | 14.4 us |
| shared expert, 40 layers | 199 MB | 376 MB | 0.73 ms |
| 6 routed experts, 40 layers | 1.19 GB | 2.26 GB | 4.37 ms |
| the logit row, 129,280 bf16 | 0.26 MB | same | 0.9 us |

The cores streaming the shared expert at `B_cpu` take 4.98 MB / `B_cpu` per layer. For that
to hide behind the GPU's 109 us of routed experts, `B_cpu` has to be at least 45.7 GB/s, and
then the GPU's stream is slower by the same 4.98 MB. Below that `B_cpu`, the shared expert
on the cores is the long pole of the layer and the experiment is over.

## What the GEMV is bound by

The GPU kernel is a byte stream: one 16-byte load covers 128 sign bits and a few `lop3`s
select them into tensor-core fragments. The core has no such instruction. Rung 1 of
`cpu_w1gemv` (NEON fp32) turns 16 signs into sixteen 32-bit lane masks with about 19 vector
instructions (a vtst, two sign-extending widens, four selects, four negates, four adds, the
loads), so a row of K = 5120 is about 6,100 instructions for 640 bytes of weights. With `w`
vector instructions per cycle at `f` GHz, one core streams at most 2 `w f` / 19 GB/s on rung 1:
whatever `w` and `f` turn out to be on the X925, that is single-digit GB/s per core, and ten
cores sit far below `B_cpu`'s roof. The GEMV at 1 bit is instruction bound on a CPU, the
opposite of the GPU.

Rung 2 (NEON int8, SDOT) is the answer to that: the activations go to int8 per group of 128
with an fp32 scale, a sign becomes an int8 +1 / -1 in two instructions (`vtst`, then `orr 1`
on the inverted mask) and `vdot` takes 16 products per instruction, so 16 weights cost about
3 instructions, 2 `w f` / 3 GB/s per core. That is in the range where ten cores could reach
`B_cpu` if `B_cpu` is a modest fraction of 273. It is approximate: the activation rounding
adds about 0.5 / 127 of the group's max |a| per product; the weights are exact. SVE2 would
let the same loop run at the vector length the core has, which on the X925 is 128 bits, so
NEON is not leaving width on the table there; what SVE2 adds is predication for the tails,
which these shapes do not have.

The ladder, in `w1gemv_neon.cpp`:

| rung | what | bound by |
|---|---|---|
| 0 | scalar, one code at a time | the reference |
| 1 | NEON fp32: vtst on a duplicated 16-bit chunk, sign-extend, bsl between a and -a, fp32 lanes per group times the scale; ternary adds a bic | instructions, about 19 per 16 weights |
| 2 | NEON int8: activations int8 per group, codes to +-1 lanes, vdot | instructions, about 3 per 16 weights, or memory if `B_cpu` is low |

Rows are split across std::threads in a fixed partition, one thread per call; the spawn is
tens of microseconds and belongs in a pool before any of this runs in a loop.

## The serial pieces

**Routing** (`route_sigmoid_topk`): 384 sigmoids and an insertion into a sorted list of six.
It touches 1.5 KB and should take single-digit microseconds; the GPU version is a few
launches with a dependency on the gate GEMV before it and the sort after it. Taking the
router GEMV too (3.9 MB of bf16, a plain `hgemm` decode shape) would move the whole routing
decision to the cores and let the GPU's expert stream start from a host-written offsets array,
which is the experiment's most plausible gain: it removes a device-to-host round trip or a
graph-side dependency per layer. Not built: the bf16 router GEMV on the cores.

**Sampling** (`sample_bf16`): the row read once, bf16 widened to fp32 and the max found with
NEON; temperature 0 is argmax; top-k by `std::nth_element` over an index array; top-p sorts
only what top-k kept. With top-k off and top-p on, that is a sort of 129,280, which is the
slow case and the reason the GPU sampler ([sample](sample.md)) is a radix select. The draw is
splitmix64 of (seed, offset), not the GPU's Philox, so the two samplers do not agree token
for token on the same inputs; a serving loop picks one and stays with it.

**The bandwidth probe** (`bandwidth_probe`): `threads` threads each sweep a slice of a private
buffer with 16-byte NEON loads behind a spin barrier; the time is the slowest thread's. Over
1, 2, 4, 8, 10 and 20 threads it says how `B_cpu` grows with cores, and whether the ten A725
cores add anything past the ten X925.

## What to run first

1. `make bench-cpu` (or `cmake --build build --target bench_cpu && build/bench_cpu >
   results/cpu.json`) on the Spark alone. Read the `cpu_bandwidth` rows first: that is
   `B_cpu` by thread count. Then `cpu_w1gemv` rung 2 at N 2304, K 5120 over threads: if its
   GB/s tracks `B_cpu` the GEMV is memory bound and the cores can be given bytes; if it stays
   flat the instruction bound holds and only rung 2's instruction count matters.
2. `build/bench_w1gemm` alone, for the GPU's GB/s on the same shapes (the number in
   [w1gemm](w1gemm.md)'s table).
3. Both at once: `build/bench_w1gemm --iters=500 & build/bench_cpu --threads=10`. If the
   GPU's GB/s plus the probe's GB/s is about what either reached alone, the two sides split
   the pool and the only gain left is latency hiding. If the sum is higher than either alone,
   one of them was not at the roof by itself, and that is where the shared expert goes.
4. `pytest tests/test_cpu.py` for the contracts, then `scripts/bench_deepseek_flash.py` with
   the shared expert on `cpu_w1gemv` against the GPU path (not wired: the block runs the
   shared expert on `w1gemm` today).

## Results

Unmeasured. The rows are what `bench_cpu` prints; the columns fill in on the Spark.

| kernel | shape | threads | median_ms | GB/s |
|---|---|---|---|---|
| cpu_bandwidth | 1 GiB | 1 | not measured | not measured |
| cpu_bandwidth | 1 GiB | 2 | not measured | not measured |
| cpu_bandwidth | 1 GiB | 4 | not measured | not measured |
| cpu_bandwidth | 1 GiB | 8 | not measured | not measured |
| cpu_bandwidth | 1 GiB | 10 | not measured | not measured |
| cpu_bandwidth | 1 GiB | 20 | not measured | not measured |
| cpu_w1gemv w1 rung 1 | N 2304, K 5120 | 1 | not measured | not measured |
| cpu_w1gemv w1 rung 1 | N 2304, K 5120 | 10 | not measured | not measured |
| cpu_w1gemv w1 rung 1 | N 2304, K 5120 | 20 | not measured | not measured |
| cpu_w1gemv w1 rung 2 | N 2304, K 5120 | 1 | not measured | not measured |
| cpu_w1gemv w1 rung 2 | N 2304, K 5120 | 10 | not measured | not measured |
| cpu_w1gemv w1 rung 2 | N 2304, K 5120 | 20 | not measured | not measured |
| cpu_w1gemv w2 rung 2 | N 2304, K 5120 | 10 | not measured | not measured |
| cpu_expert w1 rung 2 | 3 x (5120 x 2304) | 10 | not measured | not measured |
| cpu_route | E 384, k 6 | 1 | not measured | |
| cpu_sample | V 129,280, greedy | 1 | not measured | not measured |
| cpu_sample | V 129,280, k 50, p 0.9 | 1 | not measured | not measured |
| cpu_sample | V 129,280, p 0.95 | 1 | not measured | not measured |
| bench_w1gemm alone, w1 M 1 | N 2304, K 5120 | GPU | not measured | not measured |
| bench_w1gemm with the probe on 10 threads | N 2304, K 5120 | GPU + 10 | not measured | not measured |
| the probe with bench_w1gemm running | 1 GiB | 10 | not measured | not measured |
