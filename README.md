# spark-kernels

CUDA kernels I wrote by hand for the parts of a Llama-style decoder block, run on an RTX 5090.
RMSNorm, SwiGLU, softmax, an fp32 GEMM, a bf16 tensor-core GEMM and fused attention. Each one
is a ladder: a naive version first, then one change at a time, with every rung benchmarked
against cuBLAS or PyTorch and profiled in Nsight Compute. The bf16 GEMM ends up ahead of cuBLAS
on every shape from 2048 cubed up and at the memory floor on the decode shapes, and the
attention kernel ahead of PyTorch's FlashAttention-2.

The name is left over from when this was going to run on a DGX Spark. The Spark hasn't shipped,
the 5090 has, so the numbers below are from the 5090. The same source builds for the GB10 with
`ARCH=121`.

## numbers

Measured 2026-09-26. Driver 595.58, CUDA 13.2, PyTorch 2.14+cu130. The GPU state during the run
is in `results/env.txt`. Every row is the median of 100 launches (50 for the GEMMs and
attention) with CUDA events, after a clock ramp.

Best rung of each ladder on its largest shape:

| kernel | shape | best | time | achieved | vs cuBLAS or naive | vs PyTorch |
|---|---|---|---|---|---|---|
| bf16 GEMM | 4096 x 4096 x 4096 | v5 | 0.560 ms | 245 TFLOPS | 108.7% of cuBLAS | 1.11x |
| bf16 GEMM | 8192 x 8192 x 8192 | v5 | 4.66 ms | 236 TFLOPS | 101.6% of cuBLAS | 1.04x |
| bf16 GEMM, decode | 16 x 4096 x 4096 | v4 | 23.5 us | 1,442 GB/s | 122% of cuBLAS | 1.24x |
| attention | 32 heads, 4096 x 128, causal | v3 | 0.640 ms | 215 TFLOPS | | 1.21x over flash |
| attention | 32 heads, 8192 x 128, causal | v3 | 2.49 ms | 221 TFLOPS | | 1.11x over flash |
| attention, decode | 1 query, 4096 keys, 32 x 128 | v3 | 49 us | 1,366 GB/s | | 1.30x over flash |
| fp32 GEMM | 4096 x 11008 x 4096 | v5 | 6.48 ms | 57 TFLOPS | 84.3% of cuBLAS | 0.86x |
| rmsnorm bf16 | 16384 x 8192 | v4 | 0.351 ms | 1,529 GB/s | 10.4x over naive | 1.05x |
| add + rmsnorm bf16 | 16384 x 8192 | fused | 0.713 ms | 1,506 GB/s | | 1.25x |
| softmax bf16 | 4096 x 16384 | v3 | 0.174 ms | 1,542 GB/s | 25.4x over naive | 1.00x |
| swiglu bf16 | 4096 x 14336 | v0 | 0.224 ms | 1,572 GB/s | | 1.61x |

`cudaMemcpy` device to device gets 1,532 GB/s on this card, so the memory-bound rows are at
the copy roof. The spec sheet says 1,792 GB/s. Nothing reaches that.

### the bf16 GEMM against cuBLAS

Three rungs share one tile: a 128x128 block of raw `mma.sync.m16n8k16` with `ldmatrix` out of
XOR-swizzled shared memory, fp32 accumulators, bf16 stored straight from registers.

Variant 3 feeds it with a three-stage `cp.async` pipeline, one block per tile, and splits the
last partial wave of tiles along K. Variant 4 is Stream-K: a persistent grid of 340 blocks,
tiles in a grouped order so the ones in flight share A rows and B columns in L2, equal
(tile, k-step) ranges per block up to a few waves and a tile queue above that, and a fixup that
sums the pieces of a tile in K order through one fp32 slot. No memset, no atomics on data, the
same bits every run. Variant 5 keeps variant 3's schedule and moves the copies to the TMA unit:
one producer warp issues three `cp.async.bulk.tensor` boxes per stage, eight consumer warps
wait on an mbarrier and run the tensor cores, and nothing in the k-loop is a block-wide barrier.

| M x N x K | v3 | v4 Stream-K | v5 TMA | cuBLAS | best / cuBLAS |
|---|---|---|---|---|---|
| 1024 x 1024 x 1024 | 122.9 | 123.1 | | 121.6 | 101.5% |
| 2048 x 2048 x 2048 | 166.1 | 207.4 | 180.6 | 172.6 | 120.2% |
| 4096 x 4096 x 4096 | 229.5 | 231.1 | 245.4 | 225.7 | 108.7% |
| 8192 x 8192 x 8192 | 223.0 | 227.5 | 235.9 | 232.1 | 101.6% |
| 4096 x 4096 x 11008 | 228.2 | 228.8 | 244.3 | 227.1 | 107.6% |
| 4096 x 11008 x 4096 | 229.2 | 227.3 | 240.1 | 239.1 | 100.3% |

TFLOPS. Variant 5 only runs on grids of at least one 128x128 tile per SM; below that the
default steps down to variant 4. The Python default is the highest variant that takes the shape.

The TMA rung surprised me. Nsight had variant 3's tensor pipe at 92% active with the top stall
on the pipe itself, so I had put the last few percent down to power. Taking the copies out of
the consumer warps' instruction stream and the `__syncthreads` out of the k-loop moved the pipe
to 99% at the same 600 W and the same 2.75 GHz. Same joules, 8% more work. Variant 4's win is
the 2048 cubed shape, 256 tiles on 340 slots, where variant 3 had to drop to a 64x128 tile:
Stream-K keeps the big tile and spreads the k-steps evenly. At 8192 cubed the grouped tile
order takes the L2 hit rate from 80% to 96% and DRAM traffic down four times.

Decode shapes, where the whole thing is streaming the weight matrix once. The bench rotates
through enough copies of B to get past the 96 MB L2, otherwise both sides read out of cache
and report numbers above what the memory can do. Variants 3 and 4 both run a dedicated kernel
here: one CTA per column strip of B, a 16, 32 or 64-row tile picked from M, no split-K and no
memsets, so a call is one launch and nothing waits on a reduction at the end.

| M x N x K | ours | cuBLAS | ours / cuBLAS |
|---|---|---|---|
| 1 x 4096 x 4096 | 1,429 GB/s | 837 GB/s | 170.8% |
| 16 x 4096 x 4096 | 1,442 GB/s | 1,180 GB/s | 122.2% |
| 32 x 4096 x 4096 | 1,453 GB/s | 1,180 GB/s | 123.1% |
| 64 x 4096 x 4096 | 1,404 GB/s | 1,121 GB/s | 125.2% |
| 16 x 11008 x 4096 | 1,612 GB/s | 1,497 GB/s | 107.7% |
| 64 x 4096 x 11008 | 1,577 GB/s | 1,428 GB/s | 110.4% |

A read-only kernel that streams 32 MB and does nothing else takes 23.6 us timed the same way,
so the 4096-wide rows are at the floor of a single launch. Queued back to back the same
launches take 21.5 us, 1,574 GB/s. M = 1 to 64 all cost the same 23.5 us. Split-K was the cost
here, not the cure: every split-K version measured a flat 2 us over the unsplit kernel for the
atomics, fence, counter and last-arriver chain, and the two memsets in front of the old kernel
were another 2 to 3 us.

Two things I didn't expect on the first run still hold. `bench_peak` measures 258.7 TFLOPS of
dense bf16 `mma.sync` at 2,976 MHz, but a real 8192 cubed GEMM hits the 600 W power limit
within a second and the clock settles around 2.75 GHz, so the usable roof is near 239 TFLOPS.
And 170 SMs is 2 x 5 x 17, so no power-of-two grid divides into whole waves. A 4096 square
output has 1,024 tiles for 340 resident blocks, 3.01 waves, and the last four tiles used to run
alone for as long as a full wave. Splitting the tail along K took 4096 cubed from 88% to 100%
of cuBLAS; Stream-K is the general form of that fix.

### attention

Fused scaled-dot-product attention, forward, bf16 in and out, fp32 math, head sizes 64 and
128, optional causal mask, any sequence length. Variant 2 is the tensor-core flash attention:
Q fragments stay in registers for the whole KV loop, `S = Q K^T` and `O += P V` run on
`mma.sync`, and P is repacked from the S accumulators straight into the next `mma`'s A
operand without touching shared memory, with a three-stage `cp.async` pipeline on K and V.
Variant 3 splits the tiles of the last partial wave along the keys and merges them in a
combine kernel, and runs a 64-row tile for short queries. The comparison is
`F.scaled_dot_product_attention`, which picked its FlashAttention-2 kernel on every shape.

| shape | ours | TFLOPS | torch flash | ours / torch |
|---|---|---|---|---|
| 1 x 32 x 4096 x 128 | 1.22 ms | 225 | 1.46 ms | 1.17x |
| 1 x 32 x 4096 x 128, causal | 0.640 ms | 215 | 0.792 ms | 1.21x |
| 1 x 32 x 8192 x 128, causal | 2.49 ms | 221 | 2.78 ms | 1.11x |
| 4 x 32 x 2048 x 128, causal | 0.685 ms | 201 | 0.763 ms | 1.08x |
| 1 x 32 x 4096 x 64, causal | 0.331 ms | 208 | 0.420 ms | 1.23x |
| decode: 1 query, 4096 keys, 32 x 128 | 49 us | 1,366 GB/s | 71 us | 1.30x |

Causal TFLOPS use the halved FLOP count FlashAttention reports. Nsight puts the tensor pipe at
89 to 90% active with `math_pipe_throttle` on top; the rest is the softmax between the two
products, which is what warp specialization would take next. Decode is K and V streamed once,
at 89% of the copy roof.

### fp32 GEMM

Variant 5 uses a 256x128 tile with 16x8 micro-tiles to get under a byte of shared memory per
FMA, since on sm_120 an SM does 128 FMAs per clock and reads 128 bytes per clock from shared
memory. It reads 84 to 90% of cuBLAS SGEMM, and the missing 10% turned out not to be in the
kernel. Nsight at a fixed clock has it doing the same work per clock on every shape and 2 to 6%
ahead of cuBLAS. Unlocked, the card's limiter holds the sustained fp32 FMA kernel at 1.89 GHz
and 455 W, well under the 600 W cap, while cuBLAS's kernel gets 2.32 GHz. With the clock locked
at 1.9 GHz variant 5 is 99 to 106% of cuBLAS on every large shape. Whatever the limiter keys on,
73% of the FMA pipe busy on all 170 SMs trips it and `mma.sync` at 95% does not.

Full tables in [docs/RESULTS.md](docs/RESULTS.md). Roofline in `results/roofline.png`. Nsight
numbers are quoted in each design note.

## what's here

| kernel | ladder | reference |
|---|---|---|
| `bandwidth_copy` | scalar, float4, grid-stride | `cudaMemcpy` D2D |
| `rmsnorm`, `add_rmsnorm` | thread per row, warp per row, 128-bit loads, block per row, single pass with the row in registers | PyTorch eager |
| `swiglu` | scalar, 128-bit vectorized | PyTorch eager, two kernels |
| `softmax` | three pass, warp online softmax, block online softmax, single pass with the row in registers | `torch.softmax` |
| `sgemm` fp32 | naive, smem tile, 8x8 register tile, cp.async, register prefetch with swizzle, 256x128 tile | cuBLAS SGEMM |
| `hgemm` bf16 | WMMA, smem tile, cp.async, `mma.sync` + `ldmatrix` with swizzle, 3 stages and split-K, persistent Stream-K, TMA with a producer warp, a weight-streaming kernel for decode | cuBLAS GemmEx |
| `attention` bf16 | warp per query row, CUDA-core flash attention, `mma.sync` + `ldmatrix` flash attention, split-KV tail and decode | `F.scaled_dot_product_attention` |
| `bench_peak` | | measures the card's real `mma.sync` and FMA peaks and the clock they run at |

Every kernel takes a `variant` argument so each rung can be run, timed and tested on its own.
They're all exposed to PyTorch through a C++ extension, with parity tests for every variant.

## running it

You need CUDA 13, CMake 3.24 and, for the extension, a PyTorch with cu130 wheels (2.9 or
newer). The extension builds as C++20 because the torch headers ask for it.

```bash
make build            # sm_120 by default; ARCH=121 for the GB10
make bench            # every bench_* binary, validates each variant, writes results/*.json
make results          # docs/RESULTS.md, results/headline.md, results/roofline.png

pip install -e . --no-build-isolation
pytest -q tests
python scripts/bench_torch.py

make ncu              # Nsight Compute on the top two rungs of every ladder, needs root on GeForce
```

`./scripts/run_all.sh` does all of it in order. Individual benches take flags:

```bash
./build/bench_hgemm --m=16 --n=4096 --k=4096 --variant=3
./build/bench_rmsnorm --rows=16384 --cols=8192
./build/bench_attention --b=1 --h=32 --s=4096 --d=128 --causal=1 --variant=3
```

Each bench checks every variant against a reference, CPU double precision for the row kernels
and cuBLAS for the GEMMs, and exits non-zero if anything is off.

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
c = sk.hgemm(a_bf16, b_bf16)               # tensor-core GEMM
o = sk.attention(q, k, v, causal=True)     # fused attention, q/k/v are [B, H, S, D] bf16
```

## notes

One document per kernel with what each rung changes, the traffic and FLOP formulas behind the
numbers, and which Nsight metric moved:

- [docs/DESIGN.md](docs/DESIGN.md), conventions and how things are measured
- [docs/RTX5090.md](docs/RTX5090.md), the card, measured
- [docs/GB10.md](docs/GB10.md), the card that hasn't arrived
- [bandwidth](docs/design/bandwidth.md), [rmsnorm](docs/design/rmsnorm.md),
  [swiglu](docs/design/swiglu.md), [softmax](docs/design/softmax.md),
  [sgemm](docs/design/sgemm.md), [hgemm](docs/design/hgemm.md),
  [attention](docs/design/attention.md)

Things I'd still like to do: a TMA producer warp on the Stream-K schedule, warp specialization
for the softmax gap in attention, CUDA graphs for the 2 us a lone decode launch pays over the
back-to-back rate, and a GB10 run when the Spark arrives. Both cards
are consumer Blackwell, so `mma.sync`, `cp.async` and TMA are available and `tcgen05` isn't.

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
