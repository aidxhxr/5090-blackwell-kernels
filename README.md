# spark-kernels

CUDA kernels I wrote by hand for the parts of a Llama-style decoder block, run on an RTX 5090.
RMSNorm, SwiGLU, softmax, an fp32 GEMM and a bf16 tensor-core GEMM. Each one is a ladder: a
naive version first, then one change at a time, with every rung benchmarked against cuBLAS or
PyTorch and profiled in Nsight Compute. The bf16 GEMM ends up at cuBLAS speed on most shapes,
including the small-batch ones that matter for decoding.

The name is left over from when this was going to run on a DGX Spark. The Spark hasn't shipped,
the 5090 has, so the numbers below are from the 5090. The same source builds for the GB10 with
`ARCH=121`.

## numbers

Measured 2026-09-24. Driver 595.58, CUDA 13.2, PyTorch 2.14+cu130. The GPU state during the run
is in `results/env.txt`. Every row is the median of 100 launches (50 for the GEMMs) with CUDA
events, after a clock ramp.

Best rung of each ladder on its largest shape:

| kernel | shape | best | time | achieved | vs cuBLAS or naive | vs PyTorch |
|---|---|---|---|---|---|---|
| bf16 GEMM | 4096 x 4096 x 4096 | v3 | 0.611 ms | 225 TFLOPS | 99.7% of cuBLAS | 1.01x |
| bf16 GEMM | 8192 x 8192 x 8192 | v3 | 4.99 ms | 220 TFLOPS | 94.9% of cuBLAS | 0.96x |
| bf16 GEMM, decode | 16 x 4096 x 4096 | v3 | 23.5 us | 1,438 GB/s | 126.8% of cuBLAS | |
| fp32 GEMM | 4096 x 4096 x 11008 | v5 | 6.45 ms | 57 TFLOPS | 86.2% of cuBLAS | 0.87x |
| rmsnorm bf16 | 16384 x 8192 | v4 | 0.351 ms | 1,529 GB/s | 10.4x over naive | 1.06x |
| add + rmsnorm bf16 | 16384 x 8192 | fused | 0.711 ms | 1,509 GB/s | | 1.24x |
| softmax bf16 | 4096 x 16384 | v3 | 0.174 ms | 1,544 GB/s | 25.4x over naive | 1.00x |
| swiglu bf16 | 4096 x 14336 | v0 | 0.224 ms | 1,572 GB/s | | 1.61x |

`cudaMemcpy` device to device gets 1,532 GB/s on this card, so the memory-bound rows are at
the copy roof. The spec sheet says 1,792 GB/s. Nothing reaches that.

### the bf16 GEMM against cuBLAS

Variant 3 is raw `mma.sync.m16n8k16` with `ldmatrix`, an XOR-swizzled shared-memory tile, a
three-stage `cp.async` pipeline, and split-K on the last partial wave of tiles. It picks a
128x128, 64x128 or 64x64 tile per call depending on whether the grid fills the card, and for
M <= 64 it runs a separate weight-streaming kernel with a 16, 32 or 64-row tile.

| M x N x K | v3 TFLOPS | cuBLAS TFLOPS | v3 / cuBLAS |
|---|---|---|---|
| 1024 x 1024 x 1024 | 122.5 | 121.4 | 100.9% |
| 2048 x 2048 x 2048 | 166.1 | 172.5 | 96.3% |
| 4096 x 4096 x 4096 | 224.9 | 225.7 | 99.7% |
| 8192 x 8192 x 8192 | 220.4 | 232.2 | 94.9% |
| 4096 x 4096 x 11008 | 224.8 | 227.6 | 98.8% |
| 4096 x 11008 x 4096 | 224.5 | 239.4 | 93.8% |

Decode shapes, where the whole thing is streaming the weight matrix once. The bench rotates
through enough copies of B to get past the 96 MB L2, otherwise both sides read out of cache
and report numbers above what the memory can do. These run a dedicated kernel: one CTA per
column strip of B, a 16, 32 or 64-row tile, no split-K and no memsets, so a call is one
launch and nothing waits on a reduction at the end.

| M x N x K | v3 | cuBLAS | v3 / cuBLAS |
|---|---|---|---|
| 1 x 4096 x 4096 | 1,427 GB/s | 841 GB/s | 169.5% |
| 16 x 4096 x 4096 | 1,438 GB/s | 1,135 GB/s | 126.8% |
| 32 x 4096 x 4096 | 1,447 GB/s | 1,152 GB/s | 125.5% |
| 64 x 4096 x 4096 | 1,459 GB/s | 1,138 GB/s | 128.4% |
| 16 x 11008 x 4096 | 1,607 GB/s | 1,497 GB/s | 107.5% |
| 64 x 4096 x 11008 | 1,543 GB/s | 1,446 GB/s | 106.7% |

A read-only kernel that streams 32 MB and does nothing else takes 23.6 us timed the same way,
so the 4096-wide rows are at the floor of a single launch; queued back to back the same
launches take 21.5 us, 1,574 GB/s. M = 1 to 64 all cost the same 23.5 us.

Two things I didn't expect. `bench_peak` measures 258.7 TFLOPS of dense bf16 `mma.sync` at
2,976 MHz, but a real 8192 cubed GEMM hits the 600 W power limit within a second and the clock
settles around 2.75 GHz. That puts the usable roof near 239 TFLOPS, which is where cuBLAS sits
too. And 170 SMs is 2 x 5 x 17, so no power-of-two grid divides into whole waves. A 4096 square
output has 1,024 tiles for 340 resident blocks, 3.01 waves, and the last four tiles used to run
alone for as long as a full wave. Splitting the tail along K took 4096 cubed from 88% to 100%
of cuBLAS.

fp32 GEMM is a different story. On sm_120 an SM does 128 FMAs per clock and reads 128 bytes
per clock from shared memory, and an 8x8 register tile needs a byte per FMA. cuBLAS SGEMM is
stuck at the same wall, at 54% of the measured 123 TFLOPS. Variant 5 uses a 16x8 tile to get
under a byte per FMA and lands at 84 to 90% of cuBLAS.

Full tables in [docs/RESULTS.md](docs/RESULTS.md). Roofline in `results/roofline.png`. Nsight
dumps in `results/ncu_*.txt`.

## what's here

| kernel | ladder | reference |
|---|---|---|
| `bandwidth_copy` | scalar, float4, grid-stride | `cudaMemcpy` D2D |
| `rmsnorm`, `add_rmsnorm` | thread per row, warp per row, 128-bit loads, block per row, single pass with the row in registers | PyTorch eager |
| `swiglu` | scalar, 128-bit vectorized | PyTorch eager, two kernels |
| `softmax` | three pass, warp online softmax, block online softmax, single pass with the row in registers | `torch.softmax` |
| `sgemm` fp32 | naive, smem tile, 8x8 register tile, cp.async, register prefetch with swizzle, 256x128 tile | cuBLAS SGEMM |
| `hgemm` bf16 | WMMA, smem tile, cp.async, `mma.sync` + `ldmatrix` with swizzle, 3 stages and split-K, a weight-streaming kernel for decode | cuBLAS GemmEx |
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

Things I'd still like to do: Stream-K proper instead of only splitting the tail, CUDA graphs
for the 2 us a lone decode launch pays over the back-to-back rate, and figuring out why the
11008-wide fp32 shapes lose 10% per FLOP. Both cards
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
