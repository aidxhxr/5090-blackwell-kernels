# Design overview

## Conventions

- **Row-major everywhere.** GEMMs compute `C[M,N] = A[M,K] · B[K,N]`. Row ops treat the input as
  `rows × cols` with `cols` contiguous.
- **`variant` is the ladder.** Variant 0 is always the naive baseline. Numbers are stable so the
  benchmarks, tests, and design docs can refer to them. The Python bindings default to the
  highest variant.
- **fp32 accumulation** for every reduction and every GEMM, regardless of I/O dtype.
- **Host validation throws** `std::invalid_argument`; CUDA errors throw `std::runtime_error`.
  The PyTorch bindings surface these as Python exceptions.
- **One 128-bit transaction per thread** is the target memory access pattern for the
  memory-bound kernels (`bf16x8`, `f32x4` in `common.cuh`).

## Targets

Primary: RTX 5090 (`sm_120`, [RTX5090.md](RTX5090.md)). Secondary: GB10 / DGX Spark (`sm_121`,
[GB10.md](GB10.md)), to be benchmarked when the machine arrives. Same source for both; only the
arch flag and the tuning constants differ. What the two machines mean for kernel design:

| Fact | RTX 5090 | GB10 | Consequence |
|---|---|---|---|
| Memory bandwidth | 1,792 GB/s GDDR7, discrete | 273 GB/s LPDDR5X, unified | Memory-bound kernels are judged as % of this. Fusion removes whole passes and launches on both; the time saved per pass is ~6.5× larger on the GB10, where it matters more than on HBM parts. GDDR7 is much closer to HBM-class. |
| fp32 peak | 123.4 TFLOPS measured (`bench_peak`; 104.8 spec at the spec clock) | ≈ 31 TFLOPS | fp32 ridge ≈ 69 vs 114 FLOP/byte. A 4096³ SGEMM (≈ 683 FLOP/byte) is compute-bound on both, with more margin on the 5090. In practice `sgemm` is bound by shared-memory bandwidth on sm_120, not FMAs (see [sgemm](design/sgemm.md)). |
| bf16 dense peak | 258.7 TFLOPS measured at 2,976 MHz; ≈ 239 at the 2.72–2.78 GHz a sustained GEMM runs at under the 600 W power limit | ~213 TFLOPS (community measurement) | Ridges ≈ 144 vs 780 FLOP/byte; a GEMM block tile must reuse operands heavily through smem and L2. The results scripts read the measured peaks from `results/peak.json`. The GB10 number is not scaled. |
| L2 | 96 MB | 24 MB | GB10: a 4096×4096 bf16 operand (32 MB) does *not* fit, so tile order matters. On the 5090 both operands of a 4096³ GEMM fit, and any row-kernel bench shape under ~32 MB per operand measures L2, not DRAM, so the headline shapes are 256 MB+. |
| `mma.sync`, `cp.async` and TMA yes; `tcgen05` / `wgmma` no | same | same | Tensor-core GEMMs use `mma.sync` (WMMA for the early rungs) and `cp.async`, not CUTLASS 3.x SM100 pipelines. TMA (`cp.async.bulk.tensor`) exists on sm_120 but is not used here. |
| SMs | 170 | 48 | Grid sizes for grid-stride kernels are set from `multiProcessorCount` at runtime. Fixed-size launches (one block per GEMM tile, one block per row) need 3.5× more blocks to fill the 5090. |

The tile sizes and crossovers in the first rungs (`hgemm` 128×128×32 with +8 padding and 8
warps per tile, `sgemm` 128×128×8 with an 8×8 register tile, `cols > 8192` / `cols > 4096` for
the block-per-row rmsnorm / softmax) were reasoned for 48 SMs and 273 GB/s. The top rungs were
tuned on the 5090: `hgemm` v3 keeps the 128×128×32 tile (a sweep of BK = 64, 4 stages and
128×256 was slower) but moves to raw `mma.sync` + `ldmatrix`, an XOR swizzle, a 3-stage
pipeline and split-K on the last partial wave of tiles; `sgemm` v4/v5 use register prefetch
and a 256×128 / 16×8 tile against the shared-memory bandwidth limit; rmsnorm v4 and softmax v3
keep the row in registers and pick the thread group from the row length. Each per-kernel note
has an "RTX 5090 notes" section with what was measured.

One more thing the SM count does: 170 = 2 × 5 × 17, so no power-of-two tile grid divides into
whole waves. With 2 resident blocks per SM a 4096² output has 1,024 tiles = 3.01 waves, and
the last 4 tiles cost a whole wave; the GEMMs split those tail tiles along K over the idle SMs.

## Measuring

- CUDA events around each launch, 5–10 warmup launches per timing loop (`--warmup=N`
  overrides all of them), median of 100 (`--iters=N`). The first kernel of each bench process is
  also spun for ~300 ms so a cold 5090 has reached its boost clocks; `--warmup=0` disables that
  too. The minimum is recorded next to the median (`min_ms`): the 5090 boosts and
  throttles, and a median far above the min means the clocks moved during the run.
- Every JSON row carries a `device` field. The results scripts pick that device's ceilings
  from `scripts/shape_utils.py` and refuse a `results/` directory that mixes machines.
- Achieved bandwidth = bytes that *must* cross DRAM for the op (documented per kernel) ÷ time.
  This is a lower bound on true traffic and therefore a conservative efficiency number.
- Achieved TFLOPS = `2·M·N·K ÷ time`.
- `% of cuBLAS` = `cuBLAS time ÷ our time`, same shape, same stream, same timing loop.
- Reference correctness: CPU double precision for row ops, cuBLAS for GEMMs. Tolerances are
  stated in each bench source.

## Profiling

`scripts/profile_ncu.sh` runs Nsight Compute with `--set full` on the top two rungs of every
ladder and dumps the details page to `results/ncu_*.txt`. The metrics referenced in the design
docs:

| Metric | Tells you |
|---|---|
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | how close to the memory roofline |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | how close to the compute roofline |
| `sm__warps_active.avg.pct_of_peak_sustained_active` | achieved occupancy |
| `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | shared-memory bank conflicts |
| `smsp__inst_executed_pipe_tensor.sum` | tensor-core utilization |
| `launch__registers_per_thread` | register pressure (occupancy limiter for register-tiled GEMM) |

## Per-kernel notes

- hardware: [RTX 5090](RTX5090.md), [GB10](GB10.md)
- [bandwidth](design/bandwidth.md)
- [rmsnorm / add_rmsnorm](design/rmsnorm.md)
- [swiglu](design/swiglu.md)
- [softmax](design/softmax.md)
- [sgemm](design/sgemm.md)
- [hgemm](design/hgemm.md)
- [attention](design/attention.md)
