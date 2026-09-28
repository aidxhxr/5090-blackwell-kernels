# 5090-blackwell-kernels

I have an RTX 5090 and a copy of cuBLAS, and I wanted to know how close I could get to it by
hand. So I wrote the pieces of a Llama-style decoder block from scratch: RMSNorm, SwiGLU,
softmax, an fp32 GEMM, a bf16 tensor-core GEMM, an fp8 tensor-core GEMM and fused attention
with grouped-query heads. Each kernel is a ladder. Variant 0 is the naive version. Every rung
after it changes one thing, gets benchmarked against cuBLAS, cuBLASLt or PyTorch on the same
shapes in the same timing loop, and gets profiled in Nsight Compute so the ladder says what
each change bought.

Where it ended up: the bf16 GEMM is ahead of cuBLAS on every shape from 2048 cubed up and at
the memory floor on decode shapes. The fp8 GEMM is level with cuBLASLt at 4096 cubed. The
attention kernel is 18 to 38% ahead of PyTorch's FlashAttention-2 on prefill and streams a
128K-token cache faster than `cudaMemcpy` copies it. And four things about this card turned out
to be different from the spec sheet, which is the part I'd read first.

This started as spark-kernels, for a DGX Spark that hasn't shipped, and the Python package still
carries that name. The 5090 is what the numbers come from. The same source builds for the GB10
with `ARCH=121`.

## numbers

Measured 2026-09-26 on one card. Driver 595.58, CUDA 13.2, PyTorch 2.14+cu130; the GPU state
during the run is in `results/env.txt`. Every row is the median of 100 launches (50 for the
GEMMs and attention) between CUDA events, after a 300 ms clock ramp, and every variant is
checked against a reference before it is timed: CPU double precision for the row kernels,
cuBLAS or cuBLASLt for the GEMMs. The bench exits non-zero if anything is off. The "vs
PyTorch" column is eager PyTorch on the same shapes through the extension.

| kernel | shape | best | time | achieved | vs the library | vs PyTorch |
|---|---|---|---|---|---|---|
| bf16 GEMM | 4096 x 4096 x 4096 | v6 | 0.549 ms | 250 TFLOPS | 110.9% of cuBLAS | 1.12x |
| bf16 GEMM | 8192 x 8192 x 8192 | v6 | 4.51 ms | 244 TFLOPS | 104.7% of cuBLAS | 1.07x |
| bf16 GEMM, decode | 16 x 4096 x 4096 | v6 | 23.5 us | 1,440 GB/s | 123% of cuBLAS | 1.25x |
| fp8 GEMM | 4096 x 4096 x 4096 | v2 | 0.195 ms | 704 TFLOPS | 100.1% of cuBLASLt | 1.36x |
| fp8 GEMM | 8192 x 8192 x 8192 | v2 | 1.62 ms | 678 TFLOPS | 91.5% of cuBLASLt | 1.03x |
| fp8 GEMM, decode | 16 x 4096 x 4096 | v1 | 13.2 us | 1,284 GB/s | 125% of cuBLASLt | 1.64x |
| attention | 32 heads, 4096 x 128, causal | v5 | 0.582 ms | 236 TFLOPS | | 1.34x over flash |
| attention | 32 heads, 8192 x 128, causal | v5 | 2.30 ms | 239 TFLOPS | | 1.18x over flash |
| attention, decode | 1 query, 4096 keys, 32 heads | v3 | 46 us | 1,461 GB/s | | 1.43x over flash |
| attention, GQA decode | 1 query, 128K keys, 32/8 heads | v3 | 328 us | 1,635 GB/s | | 1.09x over flash |
| fp32 GEMM | 4096 x 11008 x 4096 | v5 | 6.45 ms | 57 TFLOPS | 84.5% of cuBLAS | 0.86x |
| rmsnorm bf16 | 16384 x 8192 | v4 | 0.351 ms | 1,529 GB/s | 10.4x over naive | 1.07x |
| add + rmsnorm bf16 | 16384 x 8192 | fused | 0.712 ms | 1,509 GB/s | | 1.24x |
| softmax bf16 | 4096 x 16384 | v3 | 0.175 ms | 1,536 GB/s | 25.3x over naive | 1.00x |
| swiglu bf16 | 4096 x 14336 | v0 | 0.224 ms | 1,572 GB/s | | 1.60x |

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
| v1 | 128 x 128 x 32 tile staged through padded shared memory | 153 |
| v2 | two-stage `cp.async` pipeline | 182 |
| v3 | raw `mma.sync.m16n8k16` + `ldmatrix`, XOR swizzle, three stages, register epilogue, split-K on the last wave | 231 |
| v4 | Stream-K: a persistent grid, grouped tile order, a deterministic memset-free fixup | 232 |
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

| M x N x K | v0 | v1 | v2 | v3 | v4 | v5 | v6 | cuBLAS | v6 / cuBLAS | vs torch |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 x 1024 x 1024 | 30 | 49 | 54 | 123 | 123 |  | **124** | 122 | 101.9% | 0.87x |
| 2048 x 2048 x 2048 | 26 | 127 | 166 | 166 | 207 | 181 | **224** | 173 | 129.6% | 1.28x |
| 4096 x 4096 x 4096 | 25 | 153 | 182 | 231 | 232 | 246 | **250** | 226 | 110.9% | 1.12x |
| 8192 x 8192 x 8192 | 24 | 169 | 203 | 224 | 228 | 237 | **244** | 233 | 104.7% | 1.07x |
| 4096 x 4096 x 11008 | 23 | 154 | 181 | 231 | 230 | 246 | **246** | 228 | 107.9% | 1.10x |
| 4096 x 11008 x 4096 | 24 | 169 | 200 | 230 | 229 | 241 | **247** | 240 | 102.6% | 1.06x |

TFLOPS, all from one run. Variant 5 only takes grids of at least one 128 x 128 tile per SM;
at 1024 cubed variant 6 runs variant 4's smaller tiles. The 0.87x there is the extension's
host path on a 17 us kernel, not the kernel: the C++ bench has it at 101.9% of cuBLAS.

### decode

With 1 to 64 tokens against a 4096 x 4096 weight matrix the GEMM is 32 MB of weights and
almost no math. The bench rotates through enough copies of B to get past the 96 MB L2,
otherwise both sides read out of cache and report numbers the memory cannot do; cuBLAS is
timed the same way. Variants 3, 4 and 6 all run one kernel here: a 16, 32 or 64-row tile
picked from M, one CTA per column strip of B, no split-K and no memsets, so a call is one
launch with nothing waiting on a reduction at the end.

| M x N x K | time | ours | cuBLAS | ours / cuBLAS | vs torch |
|---|---|---|---|---|---|
| 1 x 4096 x 4096 | 23.5 us | 1,429 GB/s | 837 GB/s | 171% | 1.69x |
| 16 x 4096 x 4096 | 23.5 us | 1,440 GB/s | 1,173 GB/s | 123% | 1.25x |
| 32 x 4096 x 4096 | 23.4 us | 1,455 GB/s | 1,181 GB/s | 123% | 1.28x |
| 64 x 4096 x 4096 | 25.0 us | 1,383 GB/s | 1,157 GB/s | 120% | 1.23x |
| 16 x 11008 x 4096 | 56.3 us | 1,611 GB/s | 1,496 GB/s | 108% | 1.08x |
| 64 x 4096 x 11008 | 58.4 us | 1,576 GB/s | 1,428 GB/s | 110% | 1.08x |

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
| 1024 x 1024 x 1024 | 63 | 235 |  | 206 | 114.4% | 1.50x |
| 2048 x 2048 x 2048 | 72 | 411 | 582 | 418 | 139.2% | 1.24x |
| 4096 x 4096 x 4096 | 66 | 642 | 704 | 703 | 100.1% | 1.36x |
| 8192 x 8192 x 8192 | 63 | 637 | 678 | 740 | 91.5% | 1.03x |
| 4096 x 4096 x 11008 | 49 | 667 | 722 | 581 | 124.2% | 1.28x |
| 4096 x 11008 x 4096 | 64 | 655 | 695 | 671 | 103.6% | 1.25x |

TFLOPS. Every row is bit-identical to cuBLASLt: e4m3 products are multiples of 2^-18 and the
fp32 partial sums stay exact. At 8192 cubed Nsight has variant 2 and cuBLASLt at the same
cycle count and the same 89% tensor-pipe activity, and the gap is clock: both at 600 W,
cuBLASLt at 2.3 GHz and variant 2 at 2.15, because it issues 341 M instructions to cuBLASLt's
233 M for the same 134 M `QMMA`. The "vs torch" column is `torch._scaled_mm`, whose own call
path lands well under a direct cuBLASLt call.

| M x N x K | time | ours | ours / cuBLASLt | vs torch |
|---|---|---|---|---|
| 1 x 4096 x 4096 | 13.2 us | 1,270 GB/s | 141% | 1.69x |
| 16 x 4096 x 4096 | 13.2 us | 1,284 GB/s | 125% | 1.64x |
| 32 x 4096 x 4096 | 13.2 us | 1,302 GB/s | 125% | 1.67x |
| 64 x 4096 x 4096 | 13.2 us | 1,332 GB/s | 127% | 1.62x |
| 16 x 11008 x 4096 | 29.6 us | 1,539 GB/s | 111% | 1.28x |
| 64 x 4096 x 11008 | 31.7 us | 1,462 GB/s | 120% | 1.47x |

fp8 weights are half the bytes, so a 16 MB launch has a 13.3 us read-only floor and the
4096-wide rows sit on it.

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

| shape (B x H x S x D) | v0 | v1 | v2 | v3 | v4 | v5 | flash | v5 / flash |
|---|---|---|---|---|---|---|---|---|
| 1 x 32 x 4096 x 128 | 10 | 51 | 195 | 223 | 236 | **242** | 189 | 1.25x |
| 1 x 32 x 4096 x 128, causal | 10 | 56 | 214 | 215 | 229 | **236** | 174 | 1.34x |
| 1 x 32 x 8192 x 128, causal | 10 | 50 | 221 | 221 | 235 | **239** | 198 | 1.18x |
| 4 x 32 x 2048 x 128, causal | 10 | 54 | 200 | 200 | 213 | **227** | 180 | 1.22x |
| 1 x 32 x 4096 x 64, causal | 7 | 46 | 208 | 208 | 217 | **235** | 164 | 1.38x |
| 1 x 32/8 x 4096 x 128, causal | 10 | 56 | 216 | 216 | 230 | **237** | 174 | 1.33x |

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
| 1 query, 4096 keys, 32/8 heads | 200 us | **17 us** | 970 GB/s | 38 us | 1.83x |
| 1 query, 128K keys, 32 heads | 6272 us | **1266 us** | 1,696 GB/s | 1297 us | 1.02x |
| 1 query, 128K keys, 32/8 heads | 6271 us | **328 us** | 1,635 GB/s | 358 us | 1.09x |
| batch 8, 4096 keys, 32/8 heads | 398 us | **87 us** | 1,553 GB/s | 105 us | 1.16x |

GB/s counts each K/V head once for its whole group of query heads. Past 16K tokens both
layouts run at 1,600 to 1,700 GB/s, above the 1,532 GB/s of `cudaMemcpy`: a read-only stream
does not pay the copy's write turnaround, and Nsight puts the DRAM at 93 to 95% of its
1,792 GB/s peak there. At 4K tokens a step is 17 to 46 us and launch plus pipeline fill are a
visible share of it.

## the memory-bound kernels

RMSNorm, SwiGLU and softmax move bytes and do almost no math, so the only question is whether
each byte crosses DRAM once. The ladders are the usual ones: thread per row, warp per row with
a shuffle reduction, 128-bit loads, a block per row for wide rows. Each of those reads the row
twice, once for the statistic and once for the output. The top rung of rmsnorm and softmax
holds the row in registers: a group of 32 to 1,024 threads owns one row, sized so no thread
holds more than 32 elements, and the row is read once, exponentiated once and written once.
That is 1,529 GB/s for rmsnorm and 1,536 for softmax on inputs bigger than L2, the copy number.
The fused residual add plus rmsnorm is 1.24x PyTorch running the two ops and swiglu is 1.6x
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

## what's here

| kernel | ladder | reference |
|---|---|---|
| `bandwidth_copy` | scalar, float4, grid-stride | `cudaMemcpy` D2D |
| `rmsnorm`, `add_rmsnorm` | thread per row, warp per row, 128-bit loads, block per row, row in registers | PyTorch eager |
| `swiglu` | scalar, 128-bit vectorized | PyTorch eager, two kernels |
| `softmax` | three pass, warp online softmax, block online softmax, row in registers | `torch.softmax` |
| `sgemm` fp32 | naive, smem tile, 8x8 register tile, cp.async, register prefetch with swizzle, 256x128 tile | cuBLAS SGEMM |
| `hgemm` bf16 | WMMA, smem tile, cp.async, `mma.sync` + `ldmatrix` with swizzle and split-K, Stream-K, TMA, TMA on Stream-K; a weight-streaming kernel for decode | cuBLAS GemmEx |
| `fp8gemm` e4m3 | naive `mma.sync.m16n8k32`, the swizzled cp.async tile with a 64x64 warp tile and decode configs, TMA; all on the block-scaled instruction | cuBLASLt, `torch._scaled_mm` |
| `attention` bf16 | warp per row, CUDA-core flash attention, `mma.sync` flash attention, split-KV tail, TMA mbarrier pipeline, persistent tile queue with a producer warp; GQA and a flash-decoding kernel | `F.scaled_dot_product_attention` |
| `bench_peak` | | the card's real `mma.sync` (bf16, fp8 plain and block-scaled) and FMA peaks and the clock they ran at |

Every kernel takes a `variant` argument so each rung can be run, timed and tested on its own.
All of them are exposed to PyTorch through a C++ extension, with parity tests for every
variant and every shape rule.

## running it

You need CUDA 13, CMake 3.24 and, for the extension, a PyTorch with cu130 wheels (2.9 or
newer). The extension builds as C++20 because the torch headers ask for it.

```bash
make build            # sm_120 (the fp8 kernels as sm_120a); ARCH=121 for the GB10
make bench            # every bench_* binary, validates each variant, writes results/*.json
make results          # docs/RESULTS.md, results/headline.md, results/roofline.png

pip install -e . --no-build-isolation
pytest -q tests
python scripts/bench_torch.py

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
c = sk.fp8gemm(a_e4m3, w_e4m3, sa, sb)     # sa * sb * a @ w.T, w is [N, K] as nn.Linear stores it
o = sk.attention(q, k, v, causal=True)     # q is [B, H, S, D] bf16; k and v may have fewer heads
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
  [fp8gemm](docs/design/fp8gemm.md), [attention](docs/design/attention.md)

Things I'd still like to do: a 128-key tile for attention, which would halve how often the
two warps of a scheduler land in their softmaxes together (the 3% of tensor pipe still idle);
per-block scales on the fp8 GEMM, which the instruction already takes; CUDA graphs for the
2 us a lone decode launch pays over the back-to-back rate; and a GB10 run when the Spark
arrives. Both
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
