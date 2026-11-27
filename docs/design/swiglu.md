# SwiGLU (`swiglu_f32`, `swiglu_bf16`)

`out[i] = silu(gate[i]) * up[i]`, where `silu(x) = x * sigmoid(x) = x / (1 + e^-x)`.

This is the gated activation between the gate/up projections and the down projection in every
Llama- and Qwen-style MLP. For a 4096-token prefill through a 14336-wide intermediate in bf16
the tensors are 117 MB each, so the op is purely memory-bound and the whole game is bytes.

## The fusion win

PyTorch eager evaluates `F.silu(gate) * up` as two kernels:

| step | reads | writes | passes over an activation-sized tensor |
|------|-------|--------|----------------------------------------|
| `F.silu(gate)` | gate | tmp | 2 |
| `tmp * up` | tmp, up | out | 3 |
| **eager total** | | | **5** (+ 2 launches, + a temporary allocation) |
| **fused** | gate, up | out | **3** (1 launch, no temporary) |

Ideal speedup from traffic alone is 5/3 = 1.67x; in practice the gap is larger because the two
eager kernels each pay launch latency and neither is guaranteed to be at the bandwidth roofline
(`torch.compile` fuses this too, which is why the Python bench compares against both eager and
compiled).

Traffic formula used for GB/s in the bench: `3 * n * sizeof(T) / time`.

## Variants

| # | Design | Notes |
|---|--------|-------|
| 0 | scalar, one element per thread, `cdiv(n, 256)` blocks | Baseline. bf16 lanes load 2 bytes each: a warp moves 64 B per load instruction, far below what the LSU pipe can issue, so the kernel is instruction-bound before it is DRAM-bound. |
| 1 | 16-byte vectorized, grid-stride, tail in-kernel | `f32x4` (4 floats) or `bf16x8` (4 x `__nv_bfloat162`) per thread per operand, one 128-bit load each. Math is done in fp32 via `__bfloat1622float2` / `__float22bfloat162_rn`. The grid is fixed at 8 blocks x 256 threads per SM and each thread strides over the array; the `n % VEC` remainder is handled by a second loop in the same kernel so there is no extra launch. On the RTX 5090 both variants saturate the bus on the large shapes; variant 1's win is on the shapes that fit in L2 (see the notes). |

`silu` uses `__expf`, the fast SFU exponential. Its error is well under bf16 resolution, and
for f32 it is within 2 ulp over the activation range that matters here. For `x -> -inf`,
`__expf(-x) -> inf` and `x / inf -> -0`, which is the correct limit.

Variant 1 requires 16-byte-aligned pointers (`cudaMalloc` and PyTorch allocations satisfy this;
sliced tensors may not, and the host wrapper throws `std::invalid_argument` rather than
silently misreading).

## Validation

Both variants are checked against a CPU reference computed from the *rounded* inputs (the host
converts to bf16 first, so the reference sees exactly the values the GPU sees). Pass criterion
is `|got - ref| <= atol + rtol * |ref|` with `1e-5 / 1e-5` for f32 and `2e-2 / 2e-2` for bf16.

## Nsight Compute

```
ncu --set full --kernel-name regex:swiglu_ ./build/bench_swiglu --n=58720256 --iters=3
```

Look at `dram__throughput.avg.pct_of_peak_sustained_elapsed` (should approach the bandwidth
probe's number for variant 1) and `smsp__inst_executed.sum` (should fall ~8x for bf16 between
variants 0 and 1). If `l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum` is more than
`3 * n * sizeof(T) / 32`, loads are not coalesced.

## RTX 5090 notes (measured)

- **The kernel sits on the bus.** On the DRAM-bound shapes (4096×5632 and up; 4096×2048 is
  33–67 MB of traffic and lives in the 96 MB L2, which is why it reports 3–6 TB/s) both
  variants land at 1,500–1,600 GB/s by the `3 · n · sizeof(T)` formula: 98–104% of the
  1,532 GB/s copy probe and 84–89% of the 1,792 GB/s spec. Above the copy figure is possible
  because a copy pays one read for one write while SwiGLU pays two reads for one write, and
  reads are cheaper than read/write turnaround on GDDR7.
- **Vectorization stopped mattering once the bus was saturated.** Variant 1 wins clearly where
  the shape is L2-resident (1.4× at 4096×2048 f32, 1.5× bf16) and on the 4096×5632 bf16 shape
  (1,801 vs 1,524 GB/s, part of it L2-resident); on 4096×11008 and 4096×14336 the scalar
  variant is 2–3% *faster*. With 16 GB of bytes per second arriving, a warp's 64 B bf16 loads
  are issued fast enough, and the `cdiv(n, 256)` grid of variant 0 (917,504 blocks at
  4096×14336 bf16) tails better than the fixed 8 × 170 blocks of variant 1. The `8 blocks × 256
  threads per SM` constant was left as is; it is not the limiter.
- **Fusion against eager is 1.6× on the large shapes, 1.3–3.8× on the small ones.** The 5/3
  traffic ratio predicts 1.67× and that is exactly what the DRAM-bound rows show (1.61× bf16,
  1.64–1.66× f32); the small shapes gain more because eager's two launches and temporary
  dominate when the bytes are cheap (3.8× at 4096×2048 f32).
- **`__expf` is invisible.** The SFU never shows up: the kernel is bandwidth-bound at every
  shape that leaves L2. `torch.compile` fuses the same two ops; the eager number is the one
  quoted because it is what `F.silu(gate) * up` costs in a model that is not compiled.
- `min_ms` is within 1–2% of the median on every row.

## Results (RTX 5090)

From `results/swiglu.json` (median of 100 iterations, CUDA 13.2, driver 595.58, no display) and
`results/torch_comparison.json` (PyTorch 2.14 eager `F.silu(gate) * up`, shown for the variant
the Python binding runs, variant 1). `% of probe` is against the 1,532 GB/s `cudaMemcpy` figure
from `bench_bandwidth`; rows far above 100% are L2-resident (4096×2048: 33 MB f32 / 17 MB bf16
per operand). A GB10 table is added when the Spark has been benchmarked.

| dtype | shape | variant | median ms | GB/s | % of probe | % of 1,792 | vs PyTorch eager |
|-------|-------|---------|-----------|------|------------|------------|------------------|
| f32  | 4096×2048  | 0 | 0.0218 | 4626 | 302% (L2) | 258% | — |
| f32  | 4096×2048  | 1 | 0.0156 | 6446 | 421% (L2) | 360% | 3.80× |
| f32  | 4096×5632  | 0 | 0.1730 | 1601 | 104% | 89.3% | — |
| f32  | 4096×5632  | 1 | 0.1802 | 1536 | 100% | 85.7% | 1.64× |
| f32  | 4096×11008 | 0 | 0.3428 | 1578 | 103% | 88.1% | — |
| f32  | 4096×11008 | 1 | 0.3512 | 1541 | 101% | 86.0% | 1.65× |
| f32  | 4096×14336 | 0 | 0.4454 | 1582 | 103% | 88.3% | — |
| f32  | 4096×14336 | 1 | 0.4536 | 1553 | 101% | 86.7% | 1.66× |
| bf16 | 4096×2048  | 0 | 0.0175 | 2881 | 188% (L2) | 161% | — |
| bf16 | 4096×2048  | 1 | 0.0115 | 4381 | 286% (L2) | 245% | 1.32× |
| bf16 | 4096×5632  | 0 | 0.0908 | 1524 | 99% | 85.0% | — |
| bf16 | 4096×5632  | 1 | 0.0768 | 1801 | 118% | 100.5% | 1.49× |
| bf16 | 4096×11008 | 0 | 0.1729 | 1564 | 102% | 87.3% | — |
| bf16 | 4096×11008 | 1 | 0.1803 | 1500 | 98% | 83.7% | 1.61× |
| bf16 | 4096×14336 | 0 | 0.2241 | 1572 | 103% | 87.7% | — |
| bf16 | 4096×14336 | 1 | 0.2324 | 1516 | 99% | 84.6% | 1.61× |

## The softcapped form (Inkling 2)

`swiglu(gate, up, softcap=(4.0, 25.0))` is Inkling 2 Mini's activation, softcap(g, 4) *
sigmoid(g) * softcap(u, 25) with softcap(x, c) = c * tanh(x / c), on the same kernels: the
activation is a template parameter of each of them, and the softcapped one does the torch
module's fp32 math op for op (`tanhf`, `expf`, IEEE divisions). I did not time it on its
own: an MoE layer runs the same activation in the grouped GEMM's epilogue instead, on the
fast forms, and that and its cost are in [moe_layer.md](moe_layer.md), "Inkling 2 Mini".
