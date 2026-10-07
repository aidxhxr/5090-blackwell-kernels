# 5090-blackwell-kernels

I have an RTX 5090 and a copy of cuBLAS, and I wanted to know how close I could get to it by
hand. So I wrote the pieces of a Llama-style decoder block from scratch: RMSNorm, SwiGLU,
softmax, an fp32 GEMM, a bf16 tensor-core GEMM, fp8 and fp4 tensor-core GEMMs, GEMMs for
int4 and for 1-bit weights, and fused attention with grouped-query heads, in bf16 and fp8, with its
backward pass and a paged K/V cache. Each kernel is a ladder. Variant 0 is the naive
version. Every rung after it changes one thing, gets benchmarked against cuBLAS, cuBLASLt or
PyTorch on the same shapes in the same timing loop, and gets profiled in Nsight Compute so
the ladder says what each change bought.

Where it ended up: the bf16 GEMM is ahead of cuBLAS on every shape from 2048 cubed up and at
the memory floor on decode shapes. The fp8 GEMM is level with cuBLASLt at 4096 cubed, and the
fp4 GEMM gives cuBLASLt's NVFP4 bits at 96% of its speed on the big squares. The
attention kernel is 18 to 38% ahead of PyTorch's FlashAttention-2 on prefill and streams a
128K-token cache faster than `cudaMemcpy` copies it, and its backward pass is 1 to 40% ahead
of FlashAttention-2's backward. The fp8 attention runs at 630 TFLOPS, 2.6 times the bf16
kernel. With int4 weights a batch-1 decode GEMM is 3.7 times the bf16 one, because a decode
step is a weight stream and the weights got four times smaller. Put together on a paged
cache, the kernels run the whole 32-layer Llama-3-8B for a batch of mixed-length requests,
and a decode step at 64 sequences is 3.5 times `torch.compile`'s. On real Llama-3-8B
weights that engine is closer to an fp32 reference than transformers is, and it matches or
beats vLLM on every static batch, decode and prefill row I measured. And four things about
this card turned out to be different from the spec sheet, which is the part I'd read first.

This started as spark-kernels, for a DGX Spark that hasn't shipped, and the Python package still
carries that name. The 5090 is what the numbers come from. The same source builds for the GB10
with `ARCH=121`.

## numbers

Measured 2026-09-29 on one card. Driver 595.58, CUDA 13.2, PyTorch 2.14+cu130; the GPU state
during the run is in `results/env.txt`. Every row is the median of 100 launches (50 for the
GEMMs and attention) between CUDA events, after a 300 ms clock ramp, and every variant is
checked against a reference before it is timed: CPU double precision for the row kernels,
cuBLAS or cuBLASLt for the GEMMs. The bench exits non-zero if anything is off. The "vs
PyTorch" column is eager PyTorch on the same shapes through the extension.

| kernel | shape | best | time | achieved | vs the library | vs PyTorch |
|---|---|---|---|---|---|---|
| bf16 GEMM | 4096 x 4096 x 4096 | v6 | 0.549 ms | 250 TFLOPS | 111.0% of cuBLAS | 1.12x |
| bf16 GEMM | 8192 x 8192 x 8192 | v6 | 4.51 ms | 244 TFLOPS | 104.8% of cuBLAS | 1.07x |
| bf16 GEMM, decode | 16 x 4096 x 4096 | v6 | 23.4 us | 1,446 GB/s | 123% of cuBLAS | 1.21x |
| fp8 GEMM | 4096 x 4096 x 4096 | v2 | 0.195 ms | 703 TFLOPS | 99.6% of cuBLASLt | 1.35x |
| fp8 GEMM | 8192 x 8192 x 8192 | v2 | 1.59 ms | 693 TFLOPS | 93.9% of cuBLASLt | 1.03x |
| fp8 GEMM, decode | 16 x 4096 x 4096 | v1 | 13.2 us | 1,291 GB/s | 126% of cuBLASLt | 1.62x |
| fp4 GEMM, NVFP4 | 8192 x 8192 x 8192 | v2 | 0.803 ms | 1,369 TFLOPS | 96.2% of cuBLASLt | 1.10x |
| fp4 GEMM, NVFP4 decode | 16 x 4096 x 4096 | v1 | 9.1 us | 1,061 GB/s | 228% of cuBLASLt | 3.78x |
| int4 weights, decode | 1 x 28672 x 4096 | v2 | 39.7 us | 1,525 GB/s | 3.70x over bf16 `hgemm` | |
| attention | 32 heads, 4096 x 128, causal | v5 | 0.582 ms | 236 TFLOPS | | 1.34x over flash |
| attention | 32 heads, 8192 x 128, causal | v5 | 2.30 ms | 240 TFLOPS | | 1.18x over flash |
| attention, fp8 | 32 heads, 8192 x 128, causal | v1 | 0.873 ms | 630 TFLOPS | 2.63x over bf16 v5 | 3.06x over flash |
| attention, backward | 32 heads, 4096 x 128, causal | v3 | 1.64 ms | 209 TFLOPS | | 1.13x over flash |
| attention, decode | 1 query, 4096 keys, 32 heads | v3 | 46 us | 1,461 GB/s | | 1.43x over flash |
| attention, GQA decode | 1 query, 128K keys, 32/8 heads | v3 | 324 us | 1,655 GB/s | | 1.09x over flash |
| paged decode | 64 sequences of 1 to 8K keys, 32/8 heads | v1 | 755 us | 1,627 GB/s | | |
| fp32 GEMM, TF32 | 4096 x 4096 x 4096 | v6 | 1.29 ms | 107 TFLOPS | 101.0% of cuBLAS TF32 | 1.02x |
| fp32 GEMM | 4096 x 11008 x 4096 | v5 | 6.40 ms | 58 TFLOPS | 86.5% of cuBLAS | 0.86x |
| rmsnorm bf16 | 16384 x 8192 | v4 | 0.349 ms | 1,538 GB/s | 10.4x over naive | 1.07x |
| add + rmsnorm bf16 | 16384 x 8192 | fused | 0.713 ms | 1,505 GB/s | | 1.25x |
| softmax bf16 | 4096 x 16384 | v3 | 0.173 ms | 1,551 GB/s | 25.6x over naive | 1.00x |
| swiglu bf16 | 4096 x 14336 | v0 | 0.224 ms | 1,572 GB/s | | 1.61x |

`cudaMemcpy` device to device does 1,532 GB/s on this card. The spec sheet says 1,792. The
memory-bound rows are at the copy number, and nothing reaches the spec.

Full tables for every variant and shape are in [docs/RESULTS.md](docs/RESULTS.md), the
roofline is `results/roofline.png`, and each kernel has a design note with the Nsight numbers
behind each rung.

## four things the spec sheet gets wrong

**The tensor-core roof is the power limit.** `bench_peak` sustains 258.7 TFLOPS of dense bf16
`mma.sync` on register-resident operands at 2,976 MHz. A real 8192 cubed GEMM hits the 600 W
cap within a second and the clock settles between 2.72 and 2.81 GHz, which scales that roof
to about 240 TFLOPS. cuBLAS lands there, 233 to 240 on the large shapes. Every percent past
it is joules, not instructions: variant 6 runs at 2.81 GHz where variant 5 ran at 2.75, at the
same 600 W, because it stopped moving 2.6 GB of DRAM traffic per GEMM.

**170 is 2 x 5 x 17.** No power-of-two grid divides into whole waves on 170 SMs. A 4096 square
output is 1,024 tiles of 128 x 128 on 340 resident blocks, 3.01 waves, and the last four tiles
used to run alone for as long as a full wave. That cost 12% at 4096 cubed on a kernel whose
profile was clean. Splitting the tail along K fixed that shape; Stream-K is the general
fix and is what the GEMM runs now.

**fp8 with fp32 accumulation is half rate, unless you ask for the block-scaled instruction.**
The plain e4m3 `mma.sync` with fp32 accumulate measures 517 TFLOPS, twice bf16 and half of the
fp16-accumulate rate, the same split the Ada GeForce parts have. cuBLASLt beat that ceiling
with bit-exact fp32 results, and its SASS says how: it issues the MXFP8 block-scaled
instruction (`mma.sync.kind::mxf8f6f4`, `sm_120a` only) with the scale factors set to one.
That runs fp32 accumulation at 1,014 TFLOPS and gives the same bits. Every fp8 kernel here uses
it, and the build targets `sm_120a` because of it.

**fp32 FMA trips a limiter that `mma.sync` does not.** The fp32 GEMM reads 84% of cuBLAS on the
11008-wide shapes and I spent a while looking for it in the kernel. Nsight at a fixed clock has
it doing the same work per clock on every shape and 2 to 6% ahead of cuBLAS. Unlocked, the
card holds a sustained fp32 FMA kernel at 1.89 GHz and 455 W, well under the cap, while cuBLAS's
SGEMM gets 2.32 GHz. With the clock locked at 1.9 GHz the two are within a few percent on every
large shape. Whatever the limiter keys on, 73% of the FMA pipe busy on all 170 SMs sets it off
and a 95% busy tensor pipe does not.

## the bf16 GEMM

C = A B in bf16 with fp32 accumulation, row-major. The ladder, one change per rung:

| rung | what changed | 4096 cubed |
|---|---|---|
| v0 | WMMA, fragments straight from global memory | 25 TFLOPS |
| v1 | 128 x 128 x 32 tile staged through padded shared memory | 151 |
| v2 | two-stage `cp.async` pipeline | 183 |
| v3 | raw `mma.sync.m16n8k16` + `ldmatrix`, XOR swizzle, three stages, register epilogue, split-K on the last wave | 230 |
| v4 | Stream-K: a persistent grid, grouped tile order, a deterministic memset-free fixup | 231 |
| v5 | TMA: one producer warp, eight consumer warps on mbarriers, no barrier in the k-loop | 246 |
| v6 | v5's mainloop on v4's schedule | **250** |

Variant 3 is the tile everything after it shares: a 128 x 128 block of raw `mma.sync` with
`ldmatrix` out of XOR-swizzled shared memory, bf16 stored straight from the accumulators. It
took 4096 cubed to cuBLAS speed. What the next three rungs found was that the tile was not
the limit anymore.

Variant 4 is Stream-K. A persistent grid of 340 blocks walks the tiles in a grouped order so
the ones in flight share A rows and B columns in L2. Up to a few waves the flattened
(tile, k-step) space is cut into equal ranges, one per block; past that, whole tiles come from
a queue and the last wave is handed out in K-passes of shrinking length. Every partial goes
through one fp32 slot per tile guarded by a flag with the k-steps done and a per-launch epoch,
so the pieces are summed in K order whichever block computed them. No memset, no atomics on
data, the same bits every run. At 8192 cubed the grouped order takes the L2 hit rate from 80%
to 96% and DRAM traffic down four times. At 2048 cubed, 256 tiles on 340 slots, variant 3 had
to drop to a 64 x 128 tile to fill the card; Stream-K keeps the big tile and spreads the
k-steps, and that shape goes from 96% of cuBLAS to 130%.

Variant 5 moves the copies to the TMA unit. In variant 3 every thread computes addresses and
issues two `cp.async` per stage and the block meets at a `__syncthreads` once per K tile. Here
one producer warp issues three `cp.async.bulk.tensor` boxes per stage against tensor maps
whose 128-byte swizzle is bit for bit the XOR pattern the tile already used, and eight
consumer warps wait on a full barrier, run the tensor cores, and release the stage on an empty
one. Nsight had variant 3's tensor pipe at 92% active with the top stall on the pipe itself,
so I had put the last few percent down to power. Taking the barrier and the address
arithmetic out of the consumer warps moved the pipe to 99% at the same 600 W and the same
clock. Same joules, 8% more work. It also found a real race: without a `fence.proxy.async`
before each release, the TMA refill of a stage can land before the `ldmatrix` reads of its
last use are done.

Variant 6 is the two put together. The producer lane owns the schedule: it takes the pieces,
publishes each one to the consumers through a two-deep ring in shared memory, and issues its
stages, and the stage counters run on across pieces, so it is loading the next piece while the
consumers finish the last one's epilogue. It is the fastest rung on every shape it takes and
the Python default.

The epilogue then took on the kernels that used to follow the GEMM. The accumulators are fp32
and in registers when a tile ends, so bias, an activation (silu, gelu, relu), the residual add
(or `C += A B` in place) and the SwiGLU product are applied there, before the one rounding to
bf16: `hgemm(a, b, bias=, act=, residual=)` and `hgemm_swiglu(a, w)`. The SwiGLU form takes
one `[K, 2N]` weight with gate and up interleaved column by column
(`interleave_gate_up(w_gate, w_up)`, once at load time), so gate_j and up_j land in the same
lane's register pair whatever the tile, and one GEMM reads A once where a model runs two GEMMs
and a third kernel. The same epilogue runs in the Stream-K finishing piece, in variant 4's
tiles and in the decode kernel, so every shape the default variant takes gets it.

| M x N x K | epilogue | plain GEMM | fused | unfused, ours | torch eager | vs torch |
|---|---|---|---|---|---|---|
| 4096 x 4096 x 4096 | bias + gelu | 564 us | 574 us | 585 us | 689 us | 1.19x |
| 4096 x 4096 x 4096 | residual | 565 us | 578 us | 600 us | 713 us | 1.24x |
| 4096 x 11008 x 4096 | residual | 1522 us | 1546 us | 1676 us | 1964 us | 1.26x |
| 4096 x 22016 x 4096 | swiglu, C is [4096][11008] | 3045 us | 3039 us | 3221 us | 3481 us | 1.13x |
| 16 x 4096 x 4096 | bias + gelu | 23.5 us | 23.5 us | 24.4 us | 35.2 us | 1.34x |
| 16 x 22016 x 4096 | swiglu, C is [16][11008] | 113.5 us | 113.3 us | 114.5 us | 123.9 us | 1.08x |

This table is from the epilogue's own measurement session in the design note; the fused rows
of the full run are in docs/RESULTS.md. "unfused, ours" is the plain GEMM followed by a
separate pass, or two GEMMs and the swiglu kernel; "torch eager" is `F.gelu(F.linear(...))`,
`torch.addmm` and `F.silu(a @ g) * (a @ u)`.
The epilogue costs 1.5 to 3% of a prefill GEMM (the residual read and the activation) and
nothing measurable on a decode shape, where the saving is the launch that is gone. The
SwiGLU form is free at prefill and saves the second GEMM's read of A and the two intermediate
tensors: 183 us at 4096 tokens. The design note has the full table, and how the first version
of this epilogue cost 33 us for a `relu` (instruction fetch on nine warps per SM) before it
was restructured.

| M x N x K | v0 | v1 | v2 | v3 | v4 | v5 | v6 | cuBLAS | v6 / cuBLAS | vs torch |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 x 1024 x 1024 | 30 | 49 | 54 | 122 | 123 |  | **124** | 121 | 102.3% | 1.00x |
| 2048 x 2048 x 2048 | 26 | 127 | 166 | 166 | 207 | 181 | **224** | 172 | 129.9% | 1.28x |
| 4096 x 4096 x 4096 | 25 | 151 | 183 | 230 | 231 | 246 | **250** | 225 | 111.0% | 1.12x |
| 8192 x 8192 x 8192 | 24 | 169 | 203 | 224 | 228 | 237 | **244** | 233 | 104.8% | 1.07x |
| 4096 x 4096 x 11008 | 23 | 154 | 181 | 230 | 229 | 246 | **246** | 227 | 108.3% | 1.10x |
| 4096 x 11008 x 4096 | 24 | 167 | 200 | 231 | 228 | 242 | **248** | 239 | 103.6% | 1.06x |

TFLOPS, all from one run. Variant 5 only takes grids of at least one 128 x 128 tile per SM;
at 1024 cubed variant 6 runs variant 4's smaller tiles. On a 17 us kernel the extension's
host path is a visible share of the torch column; the C++ bench has it at 102.3% of cuBLAS.

### decode

With 1 to 64 tokens against a 4096 x 4096 weight matrix the GEMM is 32 MB of weights and
almost no math. The bench rotates through enough copies of B to get past the 96 MB L2,
otherwise both sides read out of cache and report numbers the memory cannot do; cuBLAS is
timed the same way. Variants 3, 4 and 6 all run one kernel here: a 16, 32 or 64-row tile
picked from M, one CTA per column strip of B, no split-K and no memsets, so a call is one
launch with nothing waiting on a reduction at the end.

| M x N x K | time | ours | cuBLAS | ours / cuBLAS | vs torch |
|---|---|---|---|---|---|
| 1 x 4096 x 4096 | 23.5 us | 1,429 GB/s | 839 GB/s | 170% | 1.68x |
| 16 x 4096 x 4096 | 23.4 us | 1,446 GB/s | 1,173 GB/s | 123% | 1.21x |
| 32 x 4096 x 4096 | 23.4 us | 1,457 GB/s | 1,181 GB/s | 123% | 1.26x |
| 64 x 4096 x 4096 | 25.2 us | 1,374 GB/s | 1,121 GB/s | 123% | 1.23x |
| 16 x 11008 x 4096 | 56.2 us | 1,612 GB/s | 1,497 GB/s | 108% | 1.07x |
| 64 x 4096 x 11008 | 58.3 us | 1,581 GB/s | 1,435 GB/s | 110% | 1.08x |

A read-only kernel that streams 32 MB and does nothing else takes 23.6 us timed this way, so
the 4096-wide rows are at the floor of a single launch. Queued back to back the same launches
take 21.5 us, 1,574 GB/s. Split-K was the cost, not the cure: every split-K version measured a
flat 2 us over the unsplit kernel for the atomics, fence, counter and last-arriver chain, and
the two memsets in front of the old kernel were another 2 to 3.

## the fp8 GEMM

C = scale_a * scale_b * A B^T with A as [M, K] and B^T as [N, K] in e4m3, both K-contiguous,
per-tensor fp32 scales, bf16 out. That is the layout cuBLASLt requires for fp8 and the one
`nn.Linear` stores its weight in, and it makes the bf16 tile carry over with no transpose:
`ldmatrix` moves 16-bit words, and a word of two adjacent k in e4m3 is exactly what
`mma.sync.m16n8k32` wants in a fragment. Variant 1 is the bf16 variant 3 design on 8-bit rows
with a 64 x 64 warp tile, since at twice the mma rate the `ldmatrix` count per mma is twice
the fraction of pipe time. Variant 2 is the TMA pipeline of bf16 variant 5. Both run on the
block-scaled instruction from the section above.

| M x N x K | v0 | v1 | v2 | cuBLASLt | best / cuBLASLt | vs torch |
|---|---|---|---|---|---|---|
| 1024 x 1024 x 1024 | 63 | 234 |  | 205 | 114.1% | 1.54x |
| 2048 x 2048 x 2048 | 73 | 410 | 582 | 410 | 142.0% | 1.25x |
| 4096 x 4096 x 4096 | 65 | 638 | 703 | 706 | 99.6% | 1.35x |
| 8192 x 8192 x 8192 | 63 | 639 | 693 | 738 | 93.9% | 1.03x |
| 4096 x 4096 x 11008 | 49 | 665 | 738 | 580 | 127.2% | 1.29x |
| 4096 x 11008 x 4096 | 64 | 653 | 698 | 672 | 103.9% | 1.21x |

TFLOPS. Every row is bit-identical to cuBLASLt: e4m3 products are multiples of 2^-18 and the
fp32 partial sums stay exact. At 8192 cubed Nsight has variant 2 and cuBLASLt at the same
cycle count and the same 89% tensor-pipe activity, and the gap is clock: both at 600 W,
cuBLASLt at 2.3 GHz and variant 2 at 2.15, because it issues 341 M instructions to cuBLASLt's
233 M for the same 134 M `QMMA`. The "vs torch" column is `torch._scaled_mm`, whose own call
path lands well under a direct cuBLASLt call.

| M x N x K | time | ours | ours / cuBLASLt | vs torch |
|---|---|---|---|---|
| 1 x 4096 x 4096 | 13.2 us | 1,267 GB/s | 140% | 1.63x |
| 16 x 4096 x 4096 | 13.2 us | 1,291 GB/s | 126% | 1.62x |
| 32 x 4096 x 4096 | 13.1 us | 1,309 GB/s | 126% | 1.62x |
| 64 x 4096 x 4096 | 13.2 us | 1,332 GB/s | 126% | 1.62x |
| 16 x 11008 x 4096 | 29.6 us | 1,537 GB/s | 111% | 1.28x |
| 64 x 4096 x 11008 | 31.6 us | 1,464 GB/s | 117% | 1.51x |

fp8 weights are half the bytes, so a 16 MB launch has a 13.3 us read-only floor and the
4096-wide rows sit on it.

The same kernels run MXFP8: a ue8m0 scale per 32 consecutive k of every row of A and of
B^T (`sfa[M, K/32]`, `sfb[N, K/32]`, uint8), fed to the block-scaled instruction through the
scale operands it already had at 2^0. I measured the operand layout with a probe (with
thread-id 0 the instruction reads row g of A from lane 4g, row g+8 from lane 4g+1 and
column g of B from lane 4g, byte-id selecting the k32 step inside a 4-byte word of a row's
scales), so a row-major scale tensor feeds it with no repacking. The scales do not fit in
shared memory next to the tiles (the 3-stage pipeline is at 97 of 99 KB), so the consumer
warps fetch 16-byte chunks of them, four stages' worth, a group ahead and shuffle each
stage's words into place. `sk.fp8gemm(a, b_t, sfa=sfa, sfb=sfb)`, and
`sk.reference.quantize_mx` produces the inputs.

| M x N x K | MX v1 | MX v2 | cuBLASLt MXFP8 | best / cuBLASLt | MX / per-tensor |
|---|---|---|---|---|---|
| 2048 x 2048 x 2048 | 388 | 506 | 453 | 111.8% | 86.9% |
| 4096 x 4096 x 4096 | 625 | 630 | 688 | 91.6% | 89.6% |
| 8192 x 8192 x 8192 | 623 | 628 | 655 | 95.9% | 90.6% |
| 4096 x 4096 x 11008 | 553 | 560 | 688 | 81.4% | 75.9% |
| 4096 x 11008 x 4096 | 593 | 573 | 709 | 83.7% | 85.0% |
| 16 x 4096 x 4096 | 14.8 us, 1,184 GB/s | | 27.8 us | 188% | 89.2% |

TFLOPS. Variant 0 is bit-identical to cuBLASLt's MXFP8 kernel. The MX mode costs 9 to 10%
against the per-tensor kernel on the large squares: at a fixed clock it is 3.8% more cycles
(the shuffles on the MIO queue, 23% more shared-memory wavefronts) and the rest is clock, 24%
more instructions at the same 600 W. On the outlier inputs where MX is supposed to help, the
GEMM error of both schemes is the 3-bit mantissa, 3.7 to 3.8% relative; the table and the
reasons are in the design note.

## the fp4 GEMM

fp4 is the densest format the tensor cores here take, and NVFP4 is the one Blackwell inference
stacks ship weights in: e2m1 values (a sign, two exponent bits, one mantissa bit) packed two to
a byte, an e4m3 scale per 16 of them, and an fp32 scale per tensor on top. MXFP4 is the OCP
version, a power-of-two scale per 32. On this card both run through one instruction,
`mma.sync.m16n8k64.kind::mxf4nvf4.block_scale`, which exists only on `sm_120a`, like the
block-scaled fp8 one. `bench_peak` measures it at 2,029 TFLOPS: the fp8 instruction's issue
rate with twice the multiply-adds in it. I found the fragment and scale-register layouts with a
probe, as for MXFP8, and took cuBLASLt's blocked scale layout as the API's, because one 4-byte
word of it turned out to be exactly one lane's scale register for one k64 step.

| rung | what changed |
|---|---|
| v0 | one warp per 16 x 8 tile, fragment and scale words straight from global memory |
| v1 | the fp8 GEMM's variant 1 on fp4 bytes: a 128 x 128 swizzled tile, the scale tiles copied behind the operand tiles by the same `cp.async` pipeline, a self-cleaning split-K workspace for the last wave, strips of Bt for decode |
| v2 | TMA boxes for the operands and one bulk copy per stage for the scales, a warp-specialized mbarrier pipeline, the consumer loop unrolled by the stage count with no proxy fence per stage |

| M x N x K | v0 | v1 | v2 | cuBLASLt | v2 / cuBLASLt | MXFP4 v2 | vs torch |
|---|---|---|---|---|---|---|---|
| 2048 x 2048 x 2048 | 141 | 621 | **807** | 695 | 116.2% | 876 | 1.98x |
| 4096 x 4096 x 4096 | 155 | 1,171 | **1,236** | 1,278 | 96.7% | 1,277 | 1.13x |
| 8192 x 8192 x 8192 | 138 | 1,299 | **1,369** | 1,423 | 96.2% | 1,407 | 1.10x |
| 4096 x 6144 x 4096 | 155 | 1,180 | **1,222** | 1,317 | 92.8% | 1,265 | 1.11x |
| 4096 x 28672 x 4096 | 154 | 1,222 | **1,272** | 1,381 | 92.1% | 1,286 | 1.03x |
| 4096 x 4096 x 14336 | 105 | 1,324 | **1,348** | 1,346 | 100.2% | 1,427 | 1.18x |

TFLOPS, NVFP4 unless marked. Every NVFP4 output of every rung is bit-identical to cuBLASLt's
NVFP4 matmul; cuBLASLt and torch have no MXFP4 kernel on this card, so that column has no
reference. The last three rows are Llama-3-8B's qkv, gate/up and down projections at 4096
tokens. The gap to cuBLASLt is clock, not work: at 8192 cubed my kernel takes 1.4% fewer SM
cycles than cuBLASLt, but at 600 W it runs at 2.15 GHz to cuBLASLt's 2.21. The full run
reaches qkv and gate/up after a minute at the power cap; one shape per process, v2 reads 96 to
98% of cuBLASLt on the squares and 96 to 97% on those two. MXFP4 is 1 to 6% faster than NVFP4
on the large shapes, because its scale words cover twice as many k and the consumers issue
half the scale loads. Against the fp8 GEMM the 8192 cube is 2x, the instruction ratio.

| M x N x K | time | GB/s | cuBLASLt | ours / cuBLASLt | vs torch |
|---|---|---|---|---|---|
| 1 x 4096 x 4096 | 9.0 us | 1,051 | 21.6 us | 241% | 3.85x |
| 16 x 4096 x 4096 | 9.1 us | 1,061 | 20.6 us | 228% | 3.78x |
| 64 x 4096 x 4096 | 10.9 us | 924 | 20.7 us | 189% | 3.75x |
| 16 x 28672 x 4096 | 41.9 us | 1,600 | 61.6 us | 147% | 1.95x |
| 16 x 4096 x 14336 | 24.8 us | 1,344 | 40.1 us | 162% | 2.61x |

Decode is v1's strips of Bt. A 4096 square NVFP4 weight is 9.4 MB with its scales, and it
reads in 9.1 us where the fp8 kernel takes 13.2 us for its 16 MB. One mantissa bit is not
free: on Gaussian inputs the GEMM's relative error is 13.4% for NVFP4 against 3.75% for
per-tensor e4m3, and 16.2% for MXFP4, whose power-of-two scale wastes part of e2m1's small
range. The e4m3 block scale is what keeps NVFP4 there with outlier channels in the input. The
probe, the layouts, the power readings and what did not help are in
[the design note](docs/design/fp4gemm.md).

## int4 weights for decode

At batch 1 a decode step is a weight stream, and the bf16 decode kernel already reads its
weights at the copy roof. The only way left to go faster is fewer bytes per weight. `w4gemm`
takes bf16 activations and int4 weights with one bf16 scale (and optionally a zero point) per
128 k of a column, GPTQ's packing, and multiplies exactly the bf16 matrix a
dequantize-then-GEMM reference would: 0.516 bytes per weight instead of 2, so 3.88x is the
ceiling at M = 1. The weights are repacked once so that a lane's 16 bytes are its four
tensor-core fragments for 64 k, dequantized in registers with the lop3 magic-number trick, and
k is permuted inside each group on both operands so a lane's activations are 16 contiguous
bytes too.

| rung | what changed |
|---|---|
| v0 | one thread per output, scalar dequant |
| v1 | one warp per 16-column strip over the whole K, repacked weights and activations straight into registers, `mma.sync` |
| v2 | the block shape picked per call: four warps along K on a `cp.async` pipeline for M = 1, independent warps with the next four groups in flight for M <= 16, Stream-K tiles above that |

| projection | M | int4 | GB/s | bf16 `hgemm` | speedup |
|---|---|---|---|---|---|
| qkv, 4096 to 6144 | 1 | 11.2 us | 1,157 | 33.7 us | 3.00x |
| | 16 | 13.3 us | 1,002 | 33.7 us | 2.53x |
| | 64 | 27.6 us | | 35.7 us | 1.29x |
| | 256 | 74.7 us | | 60.4 us | 0.81x |
| o, 4096 to 4096 | 1 | 9.1 us | 957 | 23.5 us | 2.59x |
| | 16 | 11.1 us | 805 | 23.4 us | 2.11x |
| | 64 | 21.5 us | | 25.1 us | 1.17x |
| | 256 | 52.1 us | | 50.0 us | 0.96x |
| gate/up, 4096 to 28672 | 1 | 39.7 us | 1,525 | 147.2 us | **3.70x** |
| | 16 | 41.8 us | 1,473 | 148.2 us | 3.54x |
| | 64 | 90.2 us | | 166.7 us | 1.85x |
| | 256 | 311.3 us | | 266.9 us | 0.86x |
| down, 14336 to 4096 | 1 | 23.3 us | 1,303 | 72.4 us | 3.11x |
| | 16 | 25.5 us | 1,210 | 74.4 us | 2.92x |
| | 64 | 50.0 us | | 79.7 us | 1.59x |
| | 256 | 156.5 us | | 131.7 us | 0.84x |

Weights rotated past L2, single launches, GB/s over the traffic floor. gate/up reads its
weights at the copy roof and gets 3.70x of the 3.88x the bytes allow. The smaller matrices
stop at 2.6 to 3.1x because a 9 to 23 us launch carries the same 1 to 3 us of launch and
ramp as a 150 us one: a read-only probe of the same bytes needs 8.3 us single launch. The
int4 GEMM is ahead of `hgemm` up to 96 tokens, level at 128, and behind at 256, where both are
tensor-bound and the int4 tile does more work per weight.

In the decoder layer (`SparkLayer(..., int4=True)`, weights quantized once), the int4 layer
takes 0.093 ms per decode step at 4,096 cached tokens against 0.298 for bf16, 3.2x, or 335
tokens/s for 32 layers against 105. At 131,072 cached tokens attention is most of the step and
the gain drops to 1.47x. Round-to-nearest int4 has the format's error, up to half a step per
weight; GPTQ or AWQ change the codes and scales, not the kernel. Details in
[the design note](docs/design/w4gemm.md).

## 1-bit experts for DeepSeek V4.1 Flash

The int4 result says a batch-1 decode step is bytes of weights over the copy roof, and a
mixture of experts makes it more so: a step reads only the experts the router picked. DeepSeek
V4.1 Flash is 552B parameters in 40 layers, hidden 5120, 384 routed experts of SwiGLU
intermediate 2304 plus one shared, 6 active per token. So `w1gemm` takes bf16 activations and
weights at one bit (the sign) or two (ternary) with one bf16 scale per 128 k of a column, the
BitNet b1.58 absmean recipe applied after training: `s` is the group's mean |w|, the weight is
`-s` or `+s` (or `0` at two bits), and the kernel's dequant is an XOR of the code bit into the
sign of a splatted scale, no multiply, exact. `w1gemm_moe` is the grouped form: tokens sorted
by expert, int32 offsets, one launch for all 384 experts. `moe.py` has the sigmoid routing with
the selection bias, top-6 of 384 renormalized, and `deepseek_flash.py` loads the checkpoint,
whose routed experts ship in MXFP4.

| rung | what changed |
|---|---|
| v0 | one thread per output, one sign-add per weight, one FMA per group |
| v1 | one warp per 16-column strip and 8 tokens, the bits selected into bf16x2 `mma.sync` fragments, activations straight into registers |
| v2 | warps split K for M <= 16, a register ring of the next groups' words, partial sums added in shared memory |

What the format buys is a byte count, and byte counts are all these are: nothing in this
section has run on the card. The routed experts of one layer are 13.6 billion weights.

| format | bytes per weight | one layer's experts | 40 layers | 6 active experts, one layer | 6 active, 40 layers, per token | decode step on one layer |
|---|---|---|---|---|---|---|
| bf16 | 2 | 27.2 GB | 1,087 GB | 425 MB | 17.0 GB | unmeasured |
| MXFP4 (the checkpoint) | 0.531 | 7.2 GB | 289 GB | 113 MB | 4.5 GB | unmeasured |
| int4 g128 (`w4gemm`) | 0.516 | 7.0 GB | 280 GB | 109 MB | 4.4 GB | unmeasured |
| ternary g128 (`w1gemm`, bits=2) | 0.266 | 3.6 GB | 144 GB | 56 MB | 2.3 GB | unmeasured |
| 1 bit g128 (`w1gemm`, bits=1) | 0.141 | 1.9 GB | 76 GB | 30 MB | 1.2 GB | unmeasured |

At 1 bit the model's experts are 68 GB of packed bits and 8.5 GB of scales, which is not a 32
GB card; about 15 to 17 layers of them fit next to the rest, so the bench
(`scripts/bench_deepseek_flash.py`, `bench_w1gemm`) runs per layer and rotates what fits past
L2. The per-layer floor for the six experts' bytes at the copy roof is 19.5 us at 1 bit against
277 at bf16 and 72 at int4. Quality is the other half and it is not a kernel question: sign
weights with a mean-abs scale are lossy on a checkpoint trained in higher precision, and
`scripts/eval_ppl.py` on the real weights is where that gets measured. Details, the floors per
shape and what to measure first in [the design note](docs/design/w1gemm.md).

## attention

Fused scaled-dot-product attention, forward, bf16 in and out, fp32 math. Head sizes 64 and
128, optional causal mask, any sequence length, multi-head or grouped-query (K and V with
fewer heads than Q, as Llama-3's 32 query heads over 8 K/V heads). The comparison is
`F.scaled_dot_product_attention`, which picks its FlashAttention-2 kernel on every shape here
(`enable_gqa=True` for the grouped ones). cuDNN's attention forced through `sdpa_kernel` is 2
to 7% faster than flash on prefill and still 16 to 32% behind.

| rung | what changed |
|---|---|
| v0 | one warp per query row, online softmax key by key |
| v1 | 128-row Q tile, 64-key K/V tiles as fp32 in shared memory, CUDA-core FMAs |
| v2 | `mma.sync` flash attention: Q in registers for the whole KV loop, P repacked from the S accumulators straight into the next mma, three-stage `cp.async` on K and V |
| v3 | the last partial wave split along the keys and merged by a combine kernel; a 64-row tile for short queries; a flash-decoding kernel for decode |
| v4 | K and V through TMA into mbarrier stages issued by one lane, no barrier in the KV loop |
| v5 | a persistent grid: 170 resident blocks take Q tiles from a queue, heaviest first, and a producer warp keeps the TMA pipeline running from one tile into the next |

The trick variant 2 is built on is FlashAttention-2's: after S = Q K^T each lane holds exactly
the pairs of S the next mma wants for its A operand, so P is four conversions and never
touches shared memory. Variant 4's gain came from somewhere I did not expect. The per-tile
`__syncthreads` in variant 2 kept both warps of a scheduler in the same phase, so their
softmaxes were a hole in the tensor pipe. Take the barrier out and they drift apart, one in
its softmax while the other issues mma, and the pipe goes from 87% to 93% active. I built
FlashAttention-3's ping-pong on top of that and measured it slower every way, because one
`mma.sync` warp does not fill the pipe alone on this card; those schedules stay in the source
behind a build flag so the table in the design note is reproducible. Variant 5 is the
persistent grid I had on the list: every one of variant 4's 1,028 blocks on the causal 4096
shape initialized its barriers and waited for its Q tile before its first mma, 3 us a block.
Now 170 blocks stay resident and take tiles from a queue, a ninth warp owns the loads and
publishes each tile to the eight compute warps through a shared-memory ring, so the next
tile's Q is in flight while the current one finishes, and the tensor pipe went from 94 to
96% active. The ninth warp costs registers (three warps on one scheduler cap it at 168), and
it still beat the version that issues from a compute warp's lane by 1.6%, because that lane
waits for the slowest warp before every load.

`attention_fp8` is the same forward on e4m3 Q, K and V with fp32 descale factors, both
products on the block-scaled fp8 `mma.sync`: 613 to 630 TFLOPS at 4K and 8K tokens, 2.6 times
the bf16 kernel and 3 to 3.7 times torch's bf16 SDPA. The two layout problems cancel: the fp8
B operand wants four keys per register and V is stored by key, so `ldmatrix.trans` plus two
byte permutes per register pair deliver V with its keys in the order {2c, 2c+1, 8+2c, 9+2c},
which is exactly the order the S accumulators already hold P in, so P goes from the first
product to the second without a shuffle. It runs the tensor pipe 69% busy: at four times the
bf16 rate the softmax no longer hides under the other warp's products, and the card drops to
2.4 GHz at its power cap. The error is that of rounding q, k and v to 3 mantissa bits, 10 to
16 times the bf16 kernel's; the design note has the tables and the five things that did not
make it faster.

| shape (B x H x S x D) | bf16 v5 | fp8 v1 | fp8 / bf16 | flash, bf16 | fp8 / flash |
|---|---|---|---|---|---|
| 1 x 32 x 1024 x 128 | 173 | 412 | 2.38x | 141 | 2.95x |
| 1 x 32 x 2048 x 128, causal | 209 | 503 | 2.41x | 138 | 3.72x |
| 1 x 32 x 4096 x 128 | 242 | 626 | 2.58x | 188 | 3.27x |
| 1 x 32 x 4096 x 128, causal | 237 | 613 | 2.59x | 173 | 3.46x |
| 1 x 32 x 8192 x 128, causal | 239 | **630** | 2.63x | 198 | 3.06x |
| 1 x 32 x 16384 x 128, causal | 235 | 576 | 2.45x | 212 | 2.64x |
| 1 x 32/8 x 4096 x 128, causal | 237 | 614 | 2.59x | 174 | 3.44x |
| 4 x 32 x 2048 x 128, causal | 226 | 561 | 2.48x | 180 | 3.08x |
| 1 x 32 x 4096 x 64, causal | 234 | 498 | 2.13x | 164 | 3.01x |

TFLOPS. The bf16 column is variant 5 timed in the same process on the same shape; torch has no
fp8 attention for this card, so the flash column is its bf16 kernel. A sustained fp8 loop holds
the card at its power cap near 2.4 GHz, and the 16K rows, 4 to 8 ms a launch, are long enough
to show it.

| shape (B x H x S x D) | v0 | v1 | v2 | v3 | v4 | v5 | flash | v5 / flash |
|---|---|---|---|---|---|---|---|---|
| 1 x 32 x 4096 x 128 | 10 | 50 | 195 | 222 | 236 | **239** | 188 | 1.26x |
| 1 x 32 x 4096 x 128, causal | 10 | 56 | 216 | 214 | 228 | **236** | 174 | 1.34x |
| 1 x 32 x 8192 x 128, causal | 10 | 49 | 222 | 222 | 235 | **240** | 198 | 1.18x |
| 4 x 32 x 2048 x 128, causal | 11 | 53 | 200 | 201 | 213 | **227** | 180 | 1.22x |
| 1 x 32 x 4096 x 64, causal | 7 | 51 | 204 | 203 | 216 | **234** | 165 | 1.38x |
| 1 x 32/8 x 4096 x 128, causal | 10 | 55 | 214 | 217 | 230 | **237** | 174 | 1.33x |

TFLOPS, causal counted as half the products the way FlashAttention reports it. The 32/8 row is
grouped-query attention and matches its multi-head twin, as it should: it is the same math
with a stride on the K/V pointer.

### decode

A decode step is one query against the cache, and the job is to stream K and V once per K/V
head at the copy roof. The flash-decoding kernel puts the query heads that share a K/V head
into one 16-row tile so the cache is read once per group, splits the keys of each head over
about 128 blocks whose four warps each stream their own slice through a private pipeline, and
has the last block to finish merge the partials, so a step is one launch. K and V are rotated
past L2 when they fit in it; at 128K tokens they do not.

| shape | v2 | v3 | achieved | flash | v3 / flash |
|---|---|---|---|---|---|
| 1 query, 4096 keys, 32 heads | 200 us | **46 us** | 1,461 GB/s | 71 us | 1.43x |
| 1 query, 4096 keys, 32/8 heads | 200 us | **17 us** | 974 GB/s | 38 us | 1.82x |
| 1 query, 128K keys, 32 heads | 6251 us | **1275 us** | 1,685 GB/s | 1299 us | 1.02x |
| 1 query, 128K keys, 32/8 heads | 6249 us | **324 us** | 1,655 GB/s | 358 us | 1.09x |
| batch 8, 4096 keys, 32/8 heads | 398 us | **87 us** | 1,550 GB/s | 105 us | 1.16x |

GB/s counts each K/V head once for its whole group of query heads. Past 16K tokens both
layouts run at 1,600 to 1,700 GB/s, above the 1,532 GB/s of `cudaMemcpy`: a read-only stream
does not pay the copy's write turnaround, and Nsight puts the DRAM at 93 to 95% of its
1,792 GB/s peak there. At 4K tokens a step is 17 to 46 us and launch plus pipeline fill are a
visible share of it.

### backward

`attention_bwd` computes dQ, dK and dV from Q, K, V, the forward's output and the
log-sum-exp the forward writes on request (`attention_fwd`), and `attention_with_grad` wraps
the pair as a `torch.autograd.Function`. Under GQA dK and dV are summed over each K/V head's
query heads inside the kernel. The design is FlashAttention-2's: a block owns a tile of keys,
keeps their dK and dV in registers while it walks the query tiles, and adds its share of dQ
to an fp32 buffer with atomics. What made it fast on this card:

| rung | what changed |
|---|---|
| v0 | a warp per query row for dQ and per key row for dK, dV |
| v1 | FlashAttention-2's key-tile loop on `mma.sync`, keys on the mma rows so P and dS feed dV and dK straight from the accumulators |
| v2 | 128 keys per block with V's fragments in registers, which frees the shared memory for a double-buffered Q/dO tile; one barrier per tile; the grid split by simulating the block scheduler |
| v3 | the barrier gone: Q and dO by TMA into mbarrier stages, the dS buffers handed between warps by mbarriers |

| shape (B x H x S x D) | v1 | v2 | v3 | flash | v3 / flash |
|---|---|---|---|---|---|
| 1 x 32 x 1024 x 128, causal | 125 | 148 | **155** | 104 | 1.40x |
| 1 x 32 x 4096 x 128 | 156 | 212 | **223** | 195 | 1.10x |
| 1 x 32 x 4096 x 128, causal | 162 | 198 | **209** | 178 | 1.13x |
| 1 x 32 x 8192 x 128, causal | 164 | 206 | **214** | 202 | 1.04x |
| 1 x 32/8 x 4096 x 128, causal | 154 | 193 | **201** | 173 | 1.12x |
| 1 x 32 x 4096 x 64, causal | 207 | 204 | **217** | 169 | 1.24x |

TFLOPS with the backward counted as 2.5 times the forward's FLOPs, halved under the mask. The
split mattered most for GQA: 256 key tiles on 170 SMs is two waves with the second half
empty, 159 TFLOPS before the split and 208 after. Taking the barrier out bought the same thing
it bought the forward: the tensor pipe went from 85% to 90% busy. The atomics make the last
bits of dQ run-to-run dependent; `deterministic=True` computes dQ in a separate pass instead,
1.4x the time under the causal mask. The numbers, the profiles and the experiments that lost
are in [the design note](docs/design/attention_bwd.md).

## a whole layer

The kernels run together as one Llama-3-8B decoder layer in `spark_kernels.layer`: two
`add_rmsnorm_` (each residual add fused into the norm after it), four `hgemm` on fused
weights (q|k|v as one [4096, 6144], gate|up as one [4096, 28672]), a `rope_append_` kernel
that rotates q and k and writes k and v into the cache in one launch, `attention` with
grouped-query heads, and `swiglu` reading the two halves of the gate|up output in place.
`scripts/bench_layer.py` times it against the same layer in plain PyTorch, eager and
`torch.compile` (`max-autotune-no-cudagraphs` for prefill, `reduce-overhead`, which is CUDA
graphs, for decode), and captures our decode step into a `torch.cuda.CUDAGraph`. One layer's
weights are 436 MB in bf16, so a decode step streams them from DRAM with no help from L2.

| shape | ours | torch eager | torch compiled | ours, graph | tokens/s, 32 layers |
|---|---|---|---|---|---|
| prefill B=1, S=4096 | 8.42 ms | 9.63 | 8.96 | | 15,200 |
| prefill B=1, S=8192 | 18.11 ms | 20.78 | 18.84 | | 14,130 |
| prefill B=4, S=2048 | 16.40 ms | 18.86 | 16.89 | | 15,610 |
| decode B=1, L=4096 | 0.299 ms | 0.350 | 0.324 | 0.298 | 105 |
| decode B=1, L=16384 | 0.329 ms | 0.370 | 0.345 | 0.328 | 95 |
| decode B=1, L=131072 | 0.606 ms | 0.681 | 0.649 | 0.605 | 52 |
| decode B=8, L=4096 | 0.370 ms | 0.417 | 0.377 | 0.369 | 675 |

ms per layer; a decode row is one token per sequence against a cache of L - 1 tokens,
appended in place; tokens/s is 32 copies of this layer and nothing else. Where a decode step
goes, kernel time from the torch profiler:

| stage | B=1, L=4096 | B=1, L=131072 | B=8, L=4096 |
|---|---|---|---|
| qkv GEMM, 50 MB | 31 us | 31 | 31 |
| o GEMM, 34 MB | 21 us | 21 | 21 |
| gate/up GEMM, 235 MB | 146 us | 146 | 147 |
| down GEMM, 117 MB | 72 us | 71 | 71 |
| attention | 16 us | 325 | 84 |
| two norms, RoPE + append, swiglu | 12 us | 12 | 13 |
| total | 299 us | 606 | 367 |

The four GEMMs are 91% of a 4K-context step and run at 1,575 to 1,650 GB/s, the
back-to-back rate of the decode kernel; the whole layer moves its 436 MB at 96% of the
`cudaMemcpy` rate. At 128K tokens the cache is 512 MB and attention is more than half the
step. The graph buys 1 us per layer: the ten launches are queued back to back by a host that
issues a step in 52 us while the GPU runs it in 298, so there is no launch gap left for a
graph to close. Prefill is 88% GEMM at 246 to 254 TFLOPS. Torch eager is 12 to 17% behind on
every row and compiled torch 2 to 8%. The reading of it, and the honest caveats, are in
[docs/design/layer.md](docs/design/layer.md). One of them is gone: the attention kernels
take the K/V head stride now, so a cache with spare capacity is read where it is instead
of being copied out before every step (19 us at 4K tokens, 0.68 ms at 128K, measured
before the stride).

## serving a batch

The same kernels on a paged K/V cache run the whole 32-layer model for a batch of sequences of
different lengths (`spark_kernels.engine`, random bf16 weights). The paged decode kernel cuts
the keys of the whole batch into equal ranges per warp, so one 32K-token sequence among fifty
500-token ones reads at 1,501 GB/s, 98% of the copy roof, the rate of equal lengths; the varlen
prefill takes packed prompts in one launch.

| batch | K+V lengths | paged v1 | GB/s | % of copy roof | contiguous kernel |
|---|---|---|---|---|---|
| 1 | 4,096 | 18.4 us | 913 | 60% | 17.3 us |
| 1 | 32,768 | 92.1 us | 1,458 | 95% | 86.9 us |
| 8 | 4,096 each | 91.9 us | 1,460 | 95% | 85.0 us |
| 64 | 1,024 each | 174.8 us | 1,536 | 100% | 164.8 us |
| 51 | one of 32,768, fifty of 500 | 157.6 us | 1,502 | 98% | |
| 64 | uniform in 1 to 8,192 | 755.4 us | 1,627 | 106% | |
| 32 | uniform in 100 to 2,000 | 92.6 us | 1,441 | 94% | |

GQA 32/8, D = 128, pages of 16 tokens. The contiguous kernel is attention variant 3's
flash-decoding on a cache of equal lengths; the 6 to 8% it keeps is the paged kernel's second
launch, which merges the pieces of a sequence that span warps. Past 1,532 GB/s the reads are
the same read-only stream the decode section describes.

Llama-3-8B at a 64 to 2048-token prompt mix, 128 new tokens each:

| batch | decode ms/step | decode tok/s | vs compiled torch | prefill tok/s | vs torch |
|---|---|---|---|---|---|
| 1 | 9.78 | 101 | 1.24x | 16,053 | 1.17x |
| 8 | 10.34 | 768 | 1.58x | 16,368 | 1.19x |
| 32 | 11.57 | 2,744 | 2.37x | 16,442 | 1.24x |
| 64 | 14.27 | 4,448 | 3.51x | 16,340 | 1.23x |

A step at 64 sequences moves 20.6 GB (weights plus every sequence's K/V) at 94% of the copy
roof. Torch runs the same model on the same paged cache and has to gather the pages for its
attention, which is where most of the gap at large batches comes from. With continuous
batching, 256 requests through 64 slots, the engine generates 1,862 tokens/s against 797 for
compiled torch on the same schedule, 2.3x. Details and what is not done yet (preemption,
a TMA prefill) are in [docs/design/serving.md](docs/design/serving.md).

The cache can also be e4m3 (`Engine(kv_format="fp8")`): K and V at one byte with a scale per
kv head, converted to bf16 exactly inside the decode and prefill kernels. At 128K tokens the
K/V read is more than half a decode step, so halving it should make that step about 1.36x
faster, and the same bytes hold twice the tokens. That figure is arithmetic: the kernels and
their tests (against a reference on the same e4m3 bytes) are written but have not run on the
card yet (the fp8 section of serving.md).

## real models

Everything above ran on random weights. So I loaded real checkpoints into the engine
(`spark_kernels.hf`, any Hugging Face Llama-3-8B-shaped model) and checked three things: is
it right, is it fast next to the stacks people actually use, and what do the low-precision
GEMMs cost on weights that aren't Gaussian. Llama-3-8B-Instruct, Llama-3.1-8B-Instruct and
Mistral-7B-Instruct-v0.3, all measured 2026-10-02 on the same card.

```
python scripts/generate.py ~/models/llama3-8b-instruct "What is 17 * 23?"   # 391
python scripts/eval_ppl.py ~/models/llama3-8b-instruct --format nvfp4        # wikitext-2
python scripts/bench_llm.py run spark ~/models/llama3-8b-instruct --out rows.jsonl
python scripts/needle.py ~/models/llama3.1-8b-instruct --lengths 8192 32768   # long context
```

**Is it right.** I compared the engine and transformers (both bf16) against an fp32 run of
the same weights. The engine is the closer of the two, on the logits, at every one of the 32
layers, and in perplexity. It rounds to bf16 less often: RMSNorm and RoPE stay in fp32, and
the residual adds and SwiGLU happen in the fp32 GEMM epilogue.

| wikitext-2, ctx 2048 | ppl | logit error vs fp32 (median) | KL vs fp32 |
|---|---|---|---|
| fp32 reference | 8.2840 | | |
| this engine | 8.2873 | 0.98% | 3.3e-4 |
| transformers | 8.2889 | 1.18% | 3.9e-4 |

Greedy outputs still split from transformers', in 12 of 16 replies within 256 tokens. Every
split I looked at was a tie or one bf16 step between the top two logits, and the fp32
reference sided with each implementation six times. The engine is also not batch invariant:
hgemm picks its schedule by M, so the same prompt can take a different near-tie at a
different batch size. Details in [docs/design/llm_parity.md](docs/design/llm_parity.md).

**Is it fast.** I ran the same token ids through vLLM 0.30 and transformers 5.18, all bf16,
greedy, no early stop.

| Llama-3-8B-Instruct | this engine | vLLM 0.30 | transformers |
|---|---|---|---|
| batch 1 decode, tok/s | 101.8 | 100.7 | 85.1 |
| batch 1, weight stream, % of copy roof | 99.8% | 98.7% | 83% |
| batch 8, 512 in / 256 out, tok/s | 716 | 652 | 505 |
| batch 32, tok/s | 2,064 | 1,938 | 1,507 |
| batch 64, tok/s | 2,923 | 2,853 | 1,959 |
| TTFT, 2K prompt | 127 ms | 144 ms | 154 ms |
| TTFT, 8K prompt | 553 ms | 607 ms | 666 ms |
| 256 chat requests, continuous batching | 2,478 tok/s | 2,561 tok/s | 362 tok/s |

At batch 1, both engines stream the weights at the copy roof, so there is nothing left there.
Batch 8 is the biggest gap, because cuBLAS leaves its GEMV there for an sm80 CUTLASS kernel that is
10% slower than our decode GEMM. Continuous batching is the one row I lose, by 3%, and it's the
scheduler. I reserve every page a request could need at admission and never preempt, so
about 90 requests run at once against vLLM's 207. Give the engine 10 GiB of cache instead of
8.5 and it does 2,589. Getting there also fixed a real bug: a CUDA graph captured before a
prefill kept the address of a split-K workspace that the prefill then freed. Details in
[docs/design/llm_serving.md](docs/design/llm_serving.md).

**Low-precision weights on real weights.** Full wikitext-2, ctx 2048, the four projections of
every layer in the format, embedding and lm_head in bf16:

| format | weights | ppl | decode B=1 | decode B=32 | prefill 4x2048 |
|---|---|---|---|---|---|
| bf16 | 14.96 GiB | 8.287 | 102 tok/s | 2,799 tok/s | 16,196 tok/s |
| fp8 e4m3, per-tensor | 8.46 GiB | 8.320 | 165 | 4,183 | 26,221 |
| MXFP8 | 8.66 GiB | 8.399 | 166 | 4,165 | 26,819 |
| int4 g128, round to nearest | 5.31 GiB | 9.136 | 271 | 5,008 | 12,526 |
| int4 g128 asym + AWQ scales | 5.31 GiB | 8.562 | 271 | 5,008 | 12,526 |
| NVFP4 (W4A4) | 5.61 GiB | 8.976 | 224 | 5,297 | 50,809 |
| MXFP4 (W4A4) | 5.41 GiB | 12.17 | 239 | 5,524 | 54,057 |

fp8 costs 0.03 perplexity and is 1.6x faster at both ends. int4 is 2.6x faster at batch 1,
since a decode step is a weight stream and every quantized GEMM reads at the copy roof.
Round to nearest costs 0.85 perplexity, which a 20-second activation-aware scale search on
calibration text cuts to 0.27. NVFP4 is the prefill format: the fp4 GEMM runs a 2048-token
prompt at 1,227 TFLOPS, 3.1x bf16's prefill throughput, and it is closer to bf16 than plain
int4 even with 4-bit activations. MXFP4's power-of-two scales lose 3.9 perplexity. Int4 prefill is
slower than bf16, because w4gemm is a decode kernel. Details in
[docs/design/llm_quant.md](docs/design/llm_quant.md).

**Other models and long context.** Llama 3.1 needed its rope scaling. Mistral needed nothing
but its own stop token. On all three models the engine's perplexity is within 0.02% of
transformers', a hair lower each time.

| model | engine | transformers |
|---|---|---|
| Llama-3-8B-Instruct | 8.5298 | 8.5311 |
| Llama-3.1-8B-Instruct | 7.4370 | 7.4386 |
| Mistral-7B-Instruct-v0.3 | 5.5235 | 5.5240 |

(wikitext-2, the first 16 windows of 2048.) Llama 3.1 with a chunked prefill went out to 100K
tokens next to the weights on one 32 GB card. The engine found a planted number in the
haystack 3 of 3 times at every length up to 100K. transformers' `generate` ran out of memory
from 64K on.

| context | TTFT, engine | TTFT, transformers | decode B=1, engine | decode B=1, transformers |
|---|---|---|---|---|
| 2K | 0.133 s | 0.157 s | 99.7 tok/s | 83.1 tok/s |
| 8K | 0.583 s | 0.680 s | 94.1 | 75.9 |
| 32K | 3.32 s | 3.66 s | 79.9 | 52.5 |
| 64K | 9.33 s | 9.83 s | 66.5 | out of memory |
| 100K | 20.6 s | 19.1 s | 56.5 | out of memory |

100K is the one loss, and it has a cause I haven't found yet. The first launch of the TMA
GEMMs takes 3.4 GiB of device memory outside torch's allocator. Next to 15 GiB of weights and
a 12 GiB cache, that doesn't fit, so the 100K run uses the Stream-K GEMM for prefill. Details in
[docs/design/llm_models.md](docs/design/llm_models.md).

## the memory-bound kernels

RMSNorm, SwiGLU and softmax move bytes and do almost no math, so the only question is whether
each byte crosses DRAM once. The ladders are the usual ones: thread per row, warp per row with
a shuffle reduction, 128-bit loads, a block per row for wide rows. Each of those reads the row
twice, once for the statistic and once for the output. The top rung of rmsnorm and softmax
holds the row in registers: a group of 32 to 1,024 threads owns one row, sized so no thread
holds more than 32 elements, and the row is read once, exponentiated once and written once.
That is 1,538 GB/s for rmsnorm and 1,551 for softmax on inputs bigger than L2, the copy number.
The fused residual add plus rmsnorm is 1.25x PyTorch running the two ops and swiglu is 1.6x
`F.silu(g) * u`, because PyTorch's version is two kernels.

One thing the 96 MB L2 does to you: a 4096 x 1024 bf16 input is 8 MB and never leaves L2
between timing iterations, so those rows in the full tables read 2 to 4 TB/s. They are L2
numbers. The headline shapes are 256 MB and up for that reason.

## the fp32 GEMM

Variant 5 uses a 256 x 128 tile with 16 x 8 micro-tiles to get under a byte of shared memory
per FMA, because an SM on this part does 128 FMAs per clock and reads 128 bytes per clock from
shared memory, and cuBLAS SGEMM is stuck at the same wall at 54% of the measured FMA peak. It
reads 84 to 90% of cuBLAS, and at a locked clock it is 99 to 106%; the difference is the limiter
described above, not the kernel.

Variants 6 and 7 take the fp32 GEMM to the tensor cores. `mma.sync.m16n8k8` reads fp32
registers as TF32 (11 significant bits) and measures 129.4 TFLOPS on this card, half the bf16
rate and only 5% above the fp32 FMA peak; what it buys is that it needs 24 bytes of shared
memory per clock instead of 128 and runs at 2.7 to 2.9 GHz where the FMA kernels are held at
1.9 to 2.2. Variant 6 rounds the operands to TF32 as the fragments are loaded (`cvt.rn`, one
instruction; the sm_80 `cvt.rna` is three here and cost 4% of clock) and does 107 to 111
TFLOPS on the large shapes, 160% of cuBLAS SGEMM and 94 to 102% of cuBLAS in its own TF32
mode, with the same 3e-2 error against fp32 that cuBLAS TF32 has. Variant 7 splits each
operand into two TF32 parts and does three passes (3xTF32), which brings the error back to
4e-3 at a third of the rate, 38 TFLOPS. Both are opt-in, as torch's `allow_tf32` is: the
default stays on the CUDA cores.

## what's here

| kernel | ladder | reference |
|---|---|---|
| `bandwidth_copy` | scalar, float4, grid-stride | `cudaMemcpy` D2D |
| `rmsnorm`, `add_rmsnorm` | thread per row, warp per row, 128-bit loads, block per row, row in registers | PyTorch eager |
| `swiglu` | scalar, 128-bit vectorized | PyTorch eager, two kernels |
| `softmax` | three pass, warp online softmax, block online softmax, row in registers | `torch.softmax` |
| `sgemm` fp32 | naive, smem tile, 8x8 register tile, cp.async, register prefetch with swizzle, 256x128 tile; TF32 and 3xTF32 on the tensor cores (opt-in) | cuBLAS SGEMM, in fp32 and in TF32 math mode |
| `hgemm` bf16 | WMMA, smem tile, cp.async, `mma.sync` + `ldmatrix` with swizzle and split-K, Stream-K, TMA, TMA on Stream-K; a weight-streaming kernel for decode; fused bias / activation / residual / SwiGLU epilogues | cuBLAS GemmEx; eager `addmm`, `F.gelu`, `F.silu * up` |
| `fp8gemm` e4m3 | naive `mma.sync.m16n8k32`, the swizzled cp.async tile with a 64x64 warp tile and decode configs, TMA; all on the block-scaled instruction, with per-tensor scales or MXFP8 block scales | cuBLASLt, `torch._scaled_mm`, `F.scaled_mm` |
| `fp4gemm` NVFP4, MXFP4 | naive `mma.sync.m16n8k64.kind::mxf4nvf4`, the swizzled cp.async tile with the block scales staged beside it, TMA with bulk-copied scales; a bf16 -> fp4 quantizer | cuBLASLt NVFP4 (bit-identical), `F.scaled_mm` |
| `w4gemm` W4A16 | thread per output; a warp per 16 columns with the repacked int4 fragments and the activations straight into registers; warps along K on a cp.async pipeline for M <= 8 and independent warps for M <= 16, Stream-K tiles above; round-to-nearest quantizer and the offline repack | the bf16 `hgemm` on the dequantized weights, cuBLAS |
| `w1gemm`, `w1gemm_moe` W1A16 | thread per output; a warp per 16 columns with the sign bits XORed into bf16x2 fragments and the activations straight into registers; warps along K for M <= 16; the grouped form over experts from sorted rows and offsets; the absmean quantizer at 1 and 2 bits | the bf16 `hgemm` on the dequantized weights, cuBLAS; unmeasured |
| `attention` bf16 | warp per row, CUDA-core flash attention, `mma.sync` flash attention, split-KV tail, TMA mbarrier pipeline, persistent tile queue with a producer warp; GQA and a flash-decoding kernel | `F.scaled_dot_product_attention` |
| `attention_fp8` e4m3 | warp per row on dequantized inputs, variant 5's persistent TMA kernel with both products on the block-scaled fp8 `mma.sync`, P rounded to e4m3 in registers, V transposed by `ldmatrix.trans` and byte permutes | the bf16 kernel, `F.scaled_dot_product_attention` in bf16 |
| `attention_bwd` bf16 | warp per row, FlashAttention-2's key-tile loop on `mma.sync`, V in registers with a double-buffered Q/dO tile and a simulated split, TMA and mbarriers with no block barrier; GQA, a deterministic dQ pass, an autograd Function | torch autograd through `F.scaled_dot_product_attention` (flash and cuDNN) |
| `rope_append` bf16 | RoPE on q and k plus the K/V cache append from a fused q\|k\|v projection, one launch | the torch spelling, ten kernels |
| `layer` | one Llama-3-8B decoder layer from the kernels above, prefill and decode with a K/V cache, our decode step as a CUDA graph | the same layer in PyTorch, eager and `torch.compile` |
| `engine` | the 32-layer model on a paged cache: packed and chunked prefill, mixed prefill and decode steps, CUDA-graph decode, stop ids; bf16, fp8, MXFP8, int4 (with AWQ scales), NVFP4 and MXFP4 weights; Hugging Face checkpoints through `spark_kernels.hf` | vLLM 0.30, transformers 5.18, the same model in PyTorch |
| `bench_peak` | | the card's real `mma.sync` (bf16, fp8 plain and block-scaled, fp4, tf32) and FMA peaks and the clock they ran at |

Every kernel takes a `variant` argument so each rung can be run, timed and tested on its own.
All of them are exposed to PyTorch through a C++ extension, with parity tests for every
variant and every shape rule.

## running it

You need CUDA 13, CMake 3.24 and, for the extension, a PyTorch with cu130 wheels (2.9 or
newer). The extension builds as C++20 because the torch headers ask for it.

```bash
make build            # sm_120 (the fp8 and fp4 kernels as sm_120a); ARCH=121 for the GB10
make bench            # every bench_* binary, validates each variant, writes results/*.json
make results          # docs/RESULTS.md, results/headline.md, results/roofline.png

pip install -e . --no-build-isolation
pytest -q tests
python scripts/bench_torch.py
python scripts/bench_layer.py     # one decoder layer against PyTorch, results/layer.json

make ncu              # Nsight Compute on the top rungs of every ladder, needs root on GeForce
```

`./scripts/run_all.sh` does all of it in order. Individual benches take flags:

```bash
./build/bench_hgemm --m=16 --n=4096 --k=4096 --variant=6
./build/bench_fp8gemm --m=4096 --n=4096 --k=4096
./build/bench_attention --b=1 --hq=32 --hkv=8 --sq=1 --skv=131072 --d=128
```

## from python

```python
import torch, spark_kernels as sk

x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
w = torch.ones(4096, device="cuda", dtype=torch.bfloat16)

y = sk.rmsnorm(x, w, eps=1e-6)             # fastest variant
y0 = sk.rmsnorm(x, w, eps=1e-6, variant=0) # naive, for comparison
out = sk.add_rmsnorm_(x, resid, w)         # resid += x, then norm, in place
h = sk.swiglu(gate, up)
p = sk.softmax(scores)
c = sk.hgemm(a_bf16, b_bf16)               # bf16 tensor-core GEMM
c = sk.hgemm(a_bf16, b_bf16, bias=bias, act="gelu", residual=r)  # fused epilogue, one kernel
w = sk.interleave_gate_up(w_gate, w_up)    # [K, 2N], once at load time
h = sk.hgemm_swiglu(a_bf16, w)             # silu(a @ w_gate) * (a @ w_up), one GEMM, A read once
c = sk.fp8gemm(a_e4m3, w_e4m3, sa, sb)     # sa * sb * a @ w.T, w is [N, K] as nn.Linear stores it
a_mx, sfa = sk.reference.quantize_mx(a_bf16)   # MXFP8: e4m3 plus a ue8m0 scale per 32 elements
c = sk.fp8gemm(a_mx, w_mx, sfa=sfa, sfb=sfw)   # the scales go into the tensor-core instruction
a4, sfa4, sa4 = sk.fp4_quantize(a_bf16)    # NVFP4: packed e2m1, e4m3 scale per 16, fp32 per tensor
c = sk.fp4gemm(a4, w4, sfa4, sfw4, sa4, sw4)  # fmt="mxfp4" for a ue8m0 scale per 32
wq = sk.W4Weight.quantize(w_bf16)          # int4, one bf16 scale per 128 k of a column, repacked once
c = sk.w4gemm(a_bf16, wq)                  # a @ w with 4x fewer weight bytes than hgemm streams
w1 = sk.W1Weight.quantize(w_bf16, bits=1)  # sign weights, one bf16 mean-abs scale per 128 k; bits=2 for ternary
c = sk.w1gemm(a_bf16, w1)                  # a @ (±s), 14x fewer weight bytes than hgemm streams
ex = sk.W1ExpertWeights.quantize(w_experts, bits=1)       # [E, K, N], every expert in one tensor
c = sk.w1gemm_moe(a_sorted, ex, offsets)   # rows sorted by expert, int32 offsets [E+1], one launch
o = sk.attention(q, k, v, causal=True)     # q is [B, H, S, D] bf16; k and v may have fewer heads
(q8, sq), (k8, sk8), (v8, sv) = (sk.quantize_fp8(x) for x in (q, k, v))   # e4m3 + descale
o = sk.attention_fp8(q8, k8, v8, sq, sk8, sv, causal=True)   # both products on the fp8 tensor cores
q = sk.rope_append_(qkv, cos, sin, k_cache, v_cache, pos, 32, 8)  # RoPE; q head-major; k, v into the cache

from spark_kernels import layer as L        # one Llama-3-8B decoder layer
lyr = L.SparkLayer(L.LayerWeights.random(), L.RoPE(8192))   # int4=True: W4A16 projections
cache = L.KVCache(batch=1, capacity=8192)
x, delta = lyr.prefill(x, cache)           # x is [B, S, 4096]; returns the residual stream and the pending MLP output
x, delta = lyr.decode(x_next, cache, delta)
```

## notes

One document per kernel with what each rung changes, the traffic and FLOP formulas behind the
numbers, the sweeps that picked the constants, and which Nsight metric moved:

- [docs/DESIGN.md](docs/DESIGN.md), conventions and how things are measured
- [docs/RTX5090.md](docs/RTX5090.md), the card, measured
- [docs/GB10.md](docs/GB10.md), the card that hasn't arrived
- [bandwidth](docs/design/bandwidth.md), [rmsnorm](docs/design/rmsnorm.md),
  [swiglu](docs/design/swiglu.md), [softmax](docs/design/softmax.md),
  [sgemm](docs/design/sgemm.md), [hgemm](docs/design/hgemm.md),
  [fp8gemm](docs/design/fp8gemm.md), [fp4gemm](docs/design/fp4gemm.md),
  [w4gemm](docs/design/w4gemm.md), [w1gemm](docs/design/w1gemm.md),
  [attention](docs/design/attention.md),
  [layer](docs/design/layer.md), [serving](docs/design/serving.md)

Things I'd still like to do: a 128-key tile for attention, which would halve how often the
two warps of a scheduler land in their softmaxes together (the 3% of tensor pipe still idle);
the MX GEMM's last 10%, which is the scale operand's layout (cuBLASLt takes its scales in a
tiled layout that one TMA box serves; a row-major one costs the consumer warps eight shuffles
per stage); and a GB10 run when the Spark arrives. Both
cards are consumer Blackwell, so `mma.sync`, `cp.async` and TMA are there and `tcgen05` isn't.

```
include/spark/     public API and shared device helpers
src/kernels/       one .cu per kernel, all variants inside
src/bench/         one bench binary per kernel, JSON lines out
python/            the PyTorch extension
tests/             pytest parity tests
scripts/           run_all.sh, results tables, roofline, ncu, torch comparison
docs/              hardware sheets, design notes, generated results
```

MIT.
