# 5090-blackwell-kernels

I have an RTX 5090 and a copy of cuBLAS, and I wanted to know how close I could get to it by
hand. So I wrote the pieces of a Llama-style decoder block from scratch: RMSNorm, SwiGLU,
softmax, an fp32 GEMM, a bf16 tensor-core GEMM, fp8 and fp4 tensor-core GEMMs and fused
attention with grouped-query heads. Each kernel is a ladder. Variant 0 is the naive version. Every rung
after it changes one thing, gets benchmarked against cuBLAS, cuBLASLt or PyTorch on the same
shapes in the same timing loop, and gets profiled in Nsight Compute so the ladder says what
each change bought.

Where it ended up: the bf16 GEMM is ahead of cuBLAS on every shape from 2048 cubed up and at
the memory floor on decode shapes. The fp8 GEMM is level with cuBLASLt at 4096 cubed, and the
fp4 GEMM gives cuBLASLt's NVFP4 bits at 96 to 98% of its speed on the big squares. The
attention kernel is 18 to 38% ahead of PyTorch's FlashAttention-2 on prefill and streams a
128K-token cache faster than `cudaMemcpy` copies it, and its backward pass is 2 to 41% ahead
of FlashAttention-2's backward. And four things about this card turned out
to be different from the spec sheet, which is the part I'd read first.

This started as spark-kernels, for a DGX Spark that hasn't shipped, and the Python package still
carries that name. The 5090 is what the numbers come from. The same source builds for the GB10
with `ARCH=121`.

## numbers

Measured 2026-09-28 on one card. Driver 595.58, CUDA 13.2, PyTorch 2.14+cu130; the GPU state
during the run is in `results/env.txt`. Every row is the median of 100 launches (50 for the
GEMMs and attention) between CUDA events, after a 300 ms clock ramp, and every variant is
checked against a reference before it is timed: CPU double precision for the row kernels,
cuBLAS or cuBLASLt for the GEMMs. The bench exits non-zero if anything is off. The "vs
PyTorch" column is eager PyTorch on the same shapes through the extension.

| kernel | shape | best | time | achieved | vs the library | vs PyTorch |
|---|---|---|---|---|---|---|
| bf16 GEMM | 4096 x 4096 x 4096 | v6 | 0.550 ms | 250 TFLOPS | 110.8% of cuBLAS | 1.12x |
| bf16 GEMM | 8192 x 8192 x 8192 | v6 | 4.52 ms | 243 TFLOPS | 104.7% of cuBLAS | 1.07x |
| bf16 GEMM, decode | 16 x 4096 x 4096 | v6 | 23.5 us | 1,442 GB/s | 122% of cuBLAS | 1.28x |
| fp8 GEMM | 4096 x 4096 x 4096 | v2 | 0.195 ms | 704 TFLOPS | 99.6% of cuBLASLt | 1.29x |
| fp8 GEMM | 8192 x 8192 x 8192 | v2 | 1.61 ms | 685 TFLOPS | 92.6% of cuBLASLt | 1.02x |
| fp8 GEMM, decode | 16 x 4096 x 4096 | v1 | 13.3 us | 1,278 GB/s | 125% of cuBLASLt | 1.63x |
| attention | 32 heads, 4096 x 128, causal | v5 | 0.580 ms | 237 TFLOPS | | 1.33x over flash |
| attention | 32 heads, 8192 x 128, causal | v5 | 2.30 ms | 239 TFLOPS | | 1.18x over flash |
| attention, decode | 1 query, 4096 keys, 32 heads | v3 | 46 us | 1,462 GB/s | | 1.43x over flash |
| attention, GQA decode | 1 query, 128K keys, 32/8 heads | v3 | 327 us | 1,640 GB/s | | 1.08x over flash |
| fp32 GEMM, TF32 | 4096 x 4096 x 4096 | v6 | 1.29 ms | 107 TFLOPS | 101.5% of cuBLAS TF32 | 1.02x |
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

"unfused, ours" is the plain GEMM followed by a separate pass, or two GEMMs and the swiglu
kernel; "torch eager" is `F.gelu(F.linear(...))`, `torch.addmm` and `F.silu(a @ g) * (a @ u)`.
The epilogue costs 1.5 to 3% of a prefill GEMM (the residual read and the activation) and
nothing measurable on a decode shape, where the saving is the launch that is gone. The
SwiGLU form is free at prefill and saves the second GEMM's read of A and the two intermediate
tensors: 183 us at 4096 tokens. The design note has the full table, and how the first version
of this epilogue cost 33 us for a `relu` (instruction fetch on nine warps per SM) before it
was restructured.

| M x N x K | v0 | v1 | v2 | v3 | v4 | v5 | v6 | cuBLAS | v6 / cuBLAS | vs torch |
|---|---|---|---|---|---|---|---|---|---|---|
| 1024 x 1024 x 1024 | 30 | 49 | 54 | 123 | 122 |  | **125** | 122 | 102.0% | 1.05x |
| 2048 x 2048 x 2048 | 26 | 127 | 166 | 166 | 208 | 181 | **224** | 173 | 129.8% | 1.29x |
| 4096 x 4096 x 4096 | 25 | 153 | 182 | 231 | 232 | 245 | **250** | 226 | 110.8% | 1.12x |
| 8192 x 8192 x 8192 | 24 | 169 | 203 | 225 | 228 | 236 | **243** | 233 | 104.7% | 1.07x |
| 4096 x 4096 x 11008 | 23 | 154 | 181 | 229 | 230 | 245 | **246** | 228 | 107.8% | 1.10x |
| 4096 x 11008 x 4096 | 24 | 167 | 200 | 230 | 230 | 240 | **246** | 239 | 102.9% | 1.06x |

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
| 1 x 4096 x 4096 | 23.4 us | 1,433 GB/s | 839 GB/s | 171% | 1.69x |
| 16 x 4096 x 4096 | 23.5 us | 1,442 GB/s | 1,181 GB/s | 122% | 1.28x |
| 32 x 4096 x 4096 | 23.5 us | 1,453 GB/s | 1,144 GB/s | 127% | 1.28x |
| 64 x 4096 x 4096 | 25.2 us | 1,372 GB/s | 1,122 GB/s | 122% | 1.23x |
| 16 x 11008 x 4096 | 56.3 us | 1,612 GB/s | 1,497 GB/s | 108% | 1.08x |
| 64 x 4096 x 11008 | 58.4 us | 1,578 GB/s | 1,433 GB/s | 110% | 1.09x |

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
| 1024 x 1024 x 1024 | 63 | 235 |  | 205 | 114.3% | 1.57x |
| 2048 x 2048 x 2048 | 73 | 410 | 583 | 418 | 139.5% | 1.24x |
| 4096 x 4096 x 4096 | 65 | 642 | 704 | 707 | 99.6% | 1.29x |
| 8192 x 8192 x 8192 | 63 | 638 | 685 | 739 | 92.6% | 1.02x |
| 4096 x 4096 x 11008 | 49 | 665 | 735 | 579 | 126.9% | 1.28x |
| 4096 x 11008 x 4096 | 64 | 653 | 688 | 670 | 102.7% | 1.21x |

TFLOPS. Every row is bit-identical to cuBLASLt: e4m3 products are multiples of 2^-18 and the
fp32 partial sums stay exact. At 8192 cubed Nsight has variant 2 and cuBLASLt at the same
cycle count and the same 89% tensor-pipe activity, and the gap is clock: both at 600 W,
cuBLASLt at 2.3 GHz and variant 2 at 2.15, because it issues 341 M instructions to cuBLASLt's
233 M for the same 134 M `QMMA`. The "vs torch" column is `torch._scaled_mm`, whose own call
path lands well under a direct cuBLASLt call.

| M x N x K | time | ours | ours / cuBLASLt | vs torch |
|---|---|---|---|---|
| 1 x 4096 x 4096 | 13.2 us | 1,277 GB/s | 141% | 1.69x |
| 16 x 4096 x 4096 | 13.3 us | 1,278 GB/s | 125% | 1.63x |
| 32 x 4096 x 4096 | 13.2 us | 1,306 GB/s | 126% | 1.62x |
| 64 x 4096 x 4096 | 13.2 us | 1,329 GB/s | 125% | 1.56x |
| 16 x 11008 x 4096 | 29.5 us | 1,541 GB/s | 111% | 1.27x |
| 64 x 4096 x 11008 | 31.7 us | 1,462 GB/s | 119% | 1.50x |

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
| 2048 x 2048 x 2048 | 392 | 506 | 453 | 110.7% | 86.8% |
| 4096 x 4096 x 4096 | 625 | 626 | 688 | 90.9% | 89.0% |
| 8192 x 8192 x 8192 | 616 | 627 | 657 | 95.2% | 91.9% |
| 4096 x 4096 x 11008 | 576 | 570 | 685 | 83.9% | 78.0% |
| 4096 x 11008 x 4096 | 591 | 576 | 710 | 83.3% | 84.5% |
| 16 x 4096 x 4096 | 15.0 us, 1,169 GB/s | | 27.8 us | 184% | 87.8% |

TFLOPS. Variant 0 is bit-identical to cuBLASLt's MXFP8 kernel. The MX mode costs 8 to 11%
against the per-tensor kernel on the large shapes: at a fixed clock it is 3.8% more cycles
(the shuffles on the MIO queue, 23% more shared-memory wavefronts) and the rest is clock, 24%
more instructions at the same 600 W. On the outlier inputs where MX is supposed to help, the
GEMM error of both schemes is the 3-bit mantissa, 3.7 to 3.8% relative; the table and the
reasons are in the design note.

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
products on the block-scaled fp8 `mma.sync`: 620 to 636 TFLOPS at 4K and 8K tokens, 2.6 times
the bf16 kernel and 3 to 3.8 times torch's bf16 SDPA. The two layout problems cancel: the fp8
B operand wants four keys per register and V is stored by key, so `ldmatrix.trans` plus two
byte permutes per register pair deliver V with its keys in the order {2c, 2c+1, 8+2c, 9+2c},
which is exactly the order the S accumulators already hold P in, so P goes from the first
product to the second without a shuffle. It runs the tensor pipe 69% busy: at four times the
bf16 rate the softmax no longer hides under the other warp's products, and the card drops to
2.4 GHz at its power cap. The error is that of rounding q, k and v to 3 mantissa bits, 10 to
16 times the bf16 kernel's; the design note has the tables and the five things that did not
make it faster.

| shape (B x H x S x D) | v0 | v1 | v2 | v3 | v4 | v5 | flash | v5 / flash |
|---|---|---|---|---|---|---|---|---|
| 1 x 32 x 4096 x 128 | 10 | 55 | 194 | 223 | 237 | **239** | 189 | 1.26x |
| 1 x 32 x 4096 x 128, causal | 10 | 54 | 216 | 215 | 228 | **237** | 174 | 1.33x |
| 1 x 32 x 8192 x 128, causal | 10 | 50 | 221 | 221 | 236 | **239** | 198 | 1.18x |
| 4 x 32 x 2048 x 128, causal | 11 | 54 | 198 | 200 | 213 | **227** | 180 | 1.23x |
| 1 x 32 x 4096 x 64, causal | 7 | 52 | 208 | 208 | 216 | **235** | 164 | 1.38x |
| 1 x 32/8 x 4096 x 128, causal | 10 | 56 | 214 | 216 | 230 | **237** | 174 | 1.34x |

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
| 1 query, 4096 keys, 32 heads | 199 us | **46 us** | 1,462 GB/s | 71 us | 1.43x |
| 1 query, 4096 keys, 32/8 heads | 200 us | **17 us** | 972 GB/s | 38 us | 1.83x |
| 1 query, 128K keys, 32 heads | 6270 us | **1267 us** | 1,695 GB/s | 1298 us | 1.02x |
| 1 query, 128K keys, 32/8 heads | 6265 us | **327 us** | 1,640 GB/s | 357 us | 1.08x |
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
| 1 x 32 x 1024 x 128, causal | 125 | 148 | **154** | 104 | 1.41x |
| 1 x 32 x 4096 x 128 | 157 | 212 | **224** | 196 | 1.11x |
| 1 x 32 x 4096 x 128, causal | 162 | 200 | **209** | 177 | 1.14x |
| 1 x 32 x 8192 x 128, causal | 162 | 209 | **216** | 202 | 1.04x |
| 1 x 32/8 x 4096 x 128, causal | 155 | 194 | **202** | 175 | 1.12x |
| 1 x 32 x 4096 x 64, causal | 207 | 204 | **217** | 166 | 1.27x |

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
| prefill B=1, S=4096 | 8.43 ms | 9.59 | 8.93 | | 15,190 |
| prefill B=1, S=8192 | 18.11 ms | 20.73 | 18.80 | | 14,140 |
| prefill B=4, S=2048 | 16.41 ms | 18.80 | 16.86 | | 15,610 |
| decode B=1, L=4096 | 0.298 ms | 0.352 | 0.326 | 0.297 | 105 |
| decode B=1, L=16384 | 0.328 ms | 0.371 | 0.344 | 0.326 | 96 |
| decode B=1, L=131072 | 0.604 ms | 0.682 | 0.650 | 0.603 | 52 |
| decode B=8, L=4096 | 0.370 ms | 0.417 | 0.376 | 0.369 | 678 |

ms per layer; a decode row is one token per sequence against a cache of L - 1 tokens,
appended in place; tokens/s is 32 copies of this layer and nothing else. Where a decode step
goes, kernel time from the torch profiler:

| stage | B=1, L=4096 | B=1, L=131072 | B=8, L=4096 |
|---|---|---|---|
| qkv GEMM, 50 MB | 31 us | 31 | 31 |
| o GEMM, 34 MB | 21 us | 22 | 22 |
| gate/up GEMM, 235 MB | 145 us | 145 | 146 |
| down GEMM, 117 MB | 71 us | 71 | 73 |
| attention | 16 us | 324 | 84 |
| two norms, RoPE + append, swiglu | 12 us | 12 | 13 |
| total | 296 us | 605 | 370 |

The four GEMMs are 91% of a 4K-context step and run at 1,575 to 1,650 GB/s, the
back-to-back rate of the decode kernel; the whole layer moves its 436 MB at 96% of the
`cudaMemcpy` rate. At 128K tokens the cache is 512 MB and attention is more than half the
step. The graph buys 1 us per layer: the ten launches are queued back to back by a host that
issues a step in 52 us while the GPU runs it in 297, so there is no launch gap left for a
graph to close. Prefill is 88% GEMM at 246 to 254 TFLOPS. Torch eager is 14 to 18% behind on
every row and compiled torch 2 to 10%. The reading of it, and the honest caveats (no head
stride in the attention kernels, so a cache with spare capacity is copied before the kernel
reads it: 19 us at 4K tokens, 0.68 ms at 128K), are in
[docs/design/layer.md](docs/design/layer.md).

## serving a batch

The same kernels on a paged K/V cache run the whole 32-layer model for a batch of sequences of
different lengths (`spark_kernels.engine`, random bf16 weights). The paged decode kernel cuts
the keys of the whole batch into equal ranges per warp, so one 32K-token sequence among fifty
500-token ones reads at 1,501 GB/s, 98% of the copy roof, the rate of equal lengths; the varlen
prefill takes packed prompts in one launch. Llama-3-8B at a 64 to 2048-token prompt mix, 128
new tokens each:

| batch | decode ms/step | decode tok/s | vs compiled torch | prefill tok/s | vs torch |
|---|---|---|---|---|---|
| 1 | 9.80 | 101 | 1.24x | 16,192 | 1.18x |
| 8 | 10.36 | 766 | 1.58x | 16,427 | 1.20x |
| 32 | 11.56 | 2,746 | 2.38x | 16,501 | 1.24x |
| 64 | 14.26 | 4,453 | 3.52x | 16,401 | 1.23x |

A step at 64 sequences moves 20.6 GB (weights plus every sequence's K/V) at 94% of the copy
roof. Torch runs the same model on the same paged cache and has to gather the pages for its
attention, which is where most of the gap at large batches comes from. Details, the
continuous-batching run and what is not done yet (chunked prefill, preemption, a TMA prefill)
are in [docs/design/serving.md](docs/design/serving.md).

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
| `attention` bf16 | warp per row, CUDA-core flash attention, `mma.sync` flash attention, split-KV tail, TMA mbarrier pipeline, persistent tile queue with a producer warp; GQA and a flash-decoding kernel | `F.scaled_dot_product_attention` |
| `attention_fp8` e4m3 | warp per row on dequantized inputs, variant 5's persistent TMA kernel with both products on the block-scaled fp8 `mma.sync`, P rounded to e4m3 in registers, V transposed by `ldmatrix.trans` and byte permutes | the bf16 kernel, `F.scaled_dot_product_attention` in bf16 |
| `attention_bwd` bf16 | warp per row, FlashAttention-2's key-tile loop on `mma.sync`, V in registers with a double-buffered Q/dO tile and a simulated split, TMA and mbarriers with no block barrier; GQA, a deterministic dQ pass, an autograd Function | torch autograd through `F.scaled_dot_product_attention` (flash and cuDNN) |
| `rope_append` bf16 | RoPE on q and k plus the K/V cache append from a fused q\|k\|v projection, one launch | the torch spelling, ten kernels |
| `layer` | one Llama-3-8B decoder layer from the kernels above, prefill and decode with a K/V cache, our decode step as a CUDA graph | the same layer in PyTorch, eager and `torch.compile` |
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
  [w4gemm](docs/design/w4gemm.md), [attention](docs/design/attention.md),
  [layer](docs/design/layer.md), [serving](docs/design/serving.md)

Things I'd still like to do: a 128-key tile for attention, which would halve how often the
two warps of a scheduler land in their softmaxes together (the 3% of tensor pipe still idle);
the MX GEMM's last 10%, which is the scale operand's layout (cuBLASLt takes its scales in a
tiled layout that one TMA box serves; a row-major one costs the consumer warps eight shuffles
per stage); a K/V head stride in the attention kernels so a growing cache is read where it is;
and a GB10 run when the Spark arrives. Both
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
