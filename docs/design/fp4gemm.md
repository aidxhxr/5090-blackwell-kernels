# FP4GEMM: e2m1 block-scaled tensor-core GEMM (NVFP4 and MXFP4)

`C[M,N] = scale_a · scale_b · Σ_k (sfa[m][k/V] · A[m][k]) (sfb[n][k/V] · Bt[n][k])`, A and Bt
e2m1 (4-bit floats) packed two to a byte with K contiguous, one block scale per V values of a
row on both operands, C bf16, fp32 accumulation inside `mma.sync.m16n8k64.kind::mxf4nvf4`.
Two formats: **NVFP4** (V = 16, e4m3 block scales, plus the per-tensor fp32 `scale_a`,
`scale_b`) and **MXFP4** (V = 32, ue8m0 power-of-two block scales, the OCP format).
Source: `src/kernels/fp4gemm.cu` (variants 0 and 1), `src/kernels/fp4gemm_tma.cu` (variant 2),
the tile and scale staging they share in `src/kernels/fp4gemm_tile.cuh` (built on
`fp8gemm_tile.cuh`), the quantizer in `src/kernels/fp4quant.cu`, the two mma helpers in
`include/spark/common.cuh`. Bench: `bench_fp4gemm` (validates every variant against
cuBLASLt's NVFP4 matmul and a CPU sample, and the quantizer's bytes against a CPU copy of it).
Python: `sk.fp4_quantize(x, fmt)` and `sk.fp4gemm(a, b_t, sfa, sfb, scale_a, scale_b, fmt)`,
parity tests in `tests/test_fp4gemm.py`, bit for bit against `F.scaled_mm` for NVFP4.

## Why fp4, and what the card does with it

fp4 is the densest tensor-core format the RTX 5090 has, and NVFP4 is the one the Blackwell
inference stacks ship weights in. `bench_peak` measures the instruction with register-resident
operands and unit scales:

| instruction | measured | at |
|---|---|---|
| `mma.sync.m16n8k32.kind::mxf8f6f4...e4m3.e4m3.f32.ue8m0` (the fp8 GEMM's) | 1,014 TFLOPS | 2,923 MHz |
| `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale.scale_vec::4X...e2m1.e2m1.f32.ue4m3` (NVFP4) | **2,029 TFLOPS** | 2,924 MHz |
| the same with `.scale_vec::2X...ue8m0` (MXFP4) | 2,028 TFLOPS | 2,924 MHz |

So the fp4 instruction issues at exactly the fp8 one's rate, one m16n8 mma per 16 cycles per
scheduler, and does twice the multiply-adds in it: 16,384 FLOP against 8,192. The results
scripts take the NVFP4 row (`peak_fp4_mma`) as the fp4 roof. The ridge is 2,029 / 1,792 ≈ 1,132
FLOP/byte, and a 128x128 tile streams exactly the fp8 tile's bytes per mma (32 bytes of K per
row, 64 fp4 values instead of 32 fp8 ones), so it needs the fp8 kernel's L2 bandwidth to run
the mma twice as fast. That turns out to be the limit (the Nsight section).

The instruction exists only on the architecture-specific target, like the block-scaled fp8
one: `fp4*.cu` join the fp8 sources in the CMake library that is built for the "a" form of
`ARCH` (120a), and a plain `sm_120` build compiles the helpers to a trap and the host refuses
(`fp4gemm_available()`).

## The instruction, as measured

CUDA 13.2 ships no PTX ISA document on the box, and CUTLASS's `SM120_16x8x64_TN_VS` atom (in
the flashinfer wheel) gives the operand counts but not the layouts. A throwaway probe kernel
settled both, as the fp8 MX work did: one mma per warp, 4,096 warps of different inputs per
run.

**Fragments.** I guessed the m16n8k32 fp8 fragments with two e2m1 values per byte and checked
it on random codes with unit scales: 524,288 outputs of 524,288 equal to a CPU sum under that
layout. Lane (g, c) = (lane / 4, lane % 4) holds `a[0]` = row g, k 8c .. 8c+7 (value 8c + j in
nibble j, low nibble first), `a[1]` = row g + 8, `a[2]` and `a[3]` the same rows at k + 32,
`b[0]` = column g, k 8c .. 8c+7, `b[1]` = k + 32. Byte for byte that is the fp8 layout, so the
fp8 tile's `ldmatrix` addressing loads it unchanged: an m16 x k64 block of A is one
`ldmatrix.x4` over 16 rows by 32 bytes.

**Scales.** Every lane's scale register held 1.0 in every byte except one byte of one lane set
to 2.0, and A was nonzero in one k-block only; the doubled output rows or columns name the
(lane, byte) → (row, k-block) map:

| selector | A row g | A row g + 8 | B column g | byte i |
|---|---|---|---|---|
| thread-id 0 | lane 4g | lane 4g + 1 | lane 4g | |
| thread-id 1 | lane 4g + 2 | lane 4g + 3 | lane 4g + 1 | |
| `scale_vec::4X`, ue4m3 (NVFP4) | | | | k 16i .. 16i + 15, byte-id must be 0 |
| `scale_vec::2X`, ue8m0 (MXFP4) | | | | byte byte-id + i scales k 32i .. 32i + 31, byte-id 0 or 2 |

The fp8 kernels only use thread-id 0 and give the other lanes copies. Here thread-id 1 earns
its place: one register carries the scales of two A tiles (one tile's rows in lanes 0 and 1 of
each quad, the other's in lanes 2 and 3) or of two B tiles (lanes 0 and 1), and the kernels use
that to halve their scale loads.

## The scale layout: cuBLASLt's blocked one

The fp8 MX mode takes row-major scales (`[rows][K/32]`) and spends a shuffle per tile per
stage to get each word into the right lane; its design note ends with the blocked layout as
the way past that cost. For fp4 I took the blocked layout as the API: tiles of 128 rows by 4
scale bytes (512 bytes), the tiles of a 128-row block consecutive along K, and inside a tile
the 4 bytes of row r at `(r % 32) * 16 + (r / 32) * 4`. It is what cuBLASLt's
`VEC16_UE4M3` / `VEC32_UE8M0` modes and torch's `SWIZZLE_32_4_4` take, so the bench and the
tests hand cuBLASLt and `F.scaled_mm` the same bytes, and it fits the instruction exactly:

- For NVFP4 the 4 bytes of a row in one tile are the scales of 64 consecutive k, one k64 step:
  the lane's scale register is one aligned 4-byte word of the tile, used as loaded. For MXFP4
  the word covers two k64 steps, and the step selects its half with byte-id 0 or 2.
- A stage of BK bytes of K is BK / 32 k64 steps, which is BK / 32 NVFP4 tiles (BK / 64 MXFP4
  tiles) per 128-row block, consecutive in memory: one contiguous run of 1 KB (BK = 64) or 2 KB
  (BK = 128) per operand per stage, which `cp.async` or a single bulk copy moves as is.
- Rows r and r + 32 of a tile are adjacent words. The A tiles 0 and 2 of a 64-row warp tile
  are rows 32 apart, so one 8-byte shared load fills the registers of both pairs of A tiles.

`fp4_scale_bytes(rows, K, format)` is the tensor size (rows padded to 128), and
`sk.reference.to_blocked` / `from_blocked` convert from and to `[rows][K/V]`. The padding
matters for ragged M: variant 1 reads the scale rows of A past M (their A rows are zero-filled
and their outputs dropped), which exist because of it.

## The ladder

| variant | idea | what it fixes |
|---|---|---|
| 0 | one warp per 16x8 tile, fragment words and scale words loaded straight from global memory (4-byte loads, the lanes the instruction reads with thread-id 0) | baseline, and the check of the fragment and scale maps against cuBLASLt: every output bit-identical |
| 1 | fp8gemm variant 1 on fp4 bytes: 128x128 block tile, `ldmatrix` + mma out of XOR-swizzled smem, each stage's scale tiles copied behind its operand tiles by the same `cp.async` pipeline, split-K on the last partial wave with a workspace that cleans up after itself, 64x128 / 64x64 tiles for small grids, one CTA per 16/32/64-row strip of Bt for M ≤ 64 | reuse through smem, DRAM latency, wave quantization, decode shapes at the weight-streaming floor |
| 2 | the 128x128 tile fed by TMA through a warp-specialized mbarrier pipeline: per stage one producer lane issues two tensor-map boxes (A and Bt) and one bulk copy per operand for the scale tiles on the same "full" barrier; the consumer k-loop is unrolled by the ring depth and hands stages back without a proxy fence | the block barrier and the copy instructions in the consumer warps, then a fifth of the instructions (loop overhead, barrier spins and a `MEMBAR` per stage): 96 to 98% of cuBLASLt at 4096³ and 8192³ |

### The tile

`fp4gemm_tile::Cfg` extends the fp8 tile's configuration (BM, BN, BK bytes, stages, warp grid)
with the scale region after the two operand tiles of a stage: per operand, `[128-row block]
[scale tile][512 bytes]`, or for 16-row decode tiles only the 16 16-byte rows of the tile they
use. Per k64 step a warp:

- loads A with MT `ldmatrix.x4` and Bt with NT / 2, exactly as `fp8gemm_tile::mma_step`;
- loads its scale registers from the step's tile (once per step for NVFP4, once per two steps
  for MXFP4): lane (g, c) holds row g + 8 (c & 1) of A tile 2p + (c >> 1), and column g of B tile
  2p + (c & 1). For a 64x32 warp tile that is one 8-byte load for A (tiles 0, 1 in the low word,
  2, 3 in the high one) and two 4-byte loads for B;
- issues MT x NT mma with thread-ids (mi & 1, nj & 1) and the step's byte-id as immediates.

So a 64x32 warp tile issues 6 `ldmatrix`, 3 scale loads and 16 mma per k64 step, the fp8
tile's 6 per 16 plus the scales.

### Variant 1

The configuration sweep of the 128x128 tile (`SPARK_FP4GEMM_V1_CONFIG`), NVFP4, `--iters=50`,
TFLOPS / % of cuBLASLt in the same process, 4096³ / 8192³ / 4096x28672x4096 /
4096x4096x14336 / 4096x6144x4096 / 2048³:

| config | BK, stages, warps, blocks/SM | smem | 4096³ | 8192³ | gate/up | down | qkv | 2048³ |
|---|---|---|---|---|---|---|---|---|
| **0** | 64, 2, 2x2 of 64x64, 2 | 36 KB | **1131 / 93%** | **1192 / 92%** | **1105 / 91%** | **1274 / 101%** | **1100 / 91%** | 622 / 90% |
| 1 | 64, 2, 2x4 of 64x32, 2 | 36 KB | 1149 / 94% | 1184 / 91% | 1101 / 90% | 1282 / 102% | 1100 / 91% | 620 / 89% |
| 2 | 64, 4, 2x4, 1 | 72 KB | 1010 / 83% | 1082 / 83% | 1001 / 82% | 1122 / 90% | 1002 / 83% | 621 / 90% |
| 3 | 128, 2, 2x4, 1 | 72 KB | 1075 / 88% | 1149 / 88% | 1055 / 87% | 1209 / 97% | 1046 / 87% | 606 / 87% |
| 4 | 64, 3, 2x4, 1 | 54 KB | 967 / 79% | 1046 / 81% | 990 / 81% | 1066 / 85% | 946 / 78% | 622 / 90% |
| 5 | 64, 4, 2x2, 1 | 72 KB | 953 / 78% | 1022 / 79% | 1001 / 82% | 1016 / 81% | 912 / 75% | 585 / 84% |

(An NVFP4 stage of BK = 64 bytes is 16 KB of operands and 2 KB of scales.) Two blocks per SM
beat every one-block configuration by 10% or more, as for fp8 variant 1: with a block barrier
in every k-tile, the second block's warps are what keeps the tensor pipe busy across it. The
2x2 and 2x4 warp grids are level; I kept fp8 variant 1's 2x2 of 64x64.

### Variant 2

The port of fp8gemm variant 2 adds two 1-D bulk copies (`cp.async.bulk.shared::cluster.global`,
new in `common.cuh` as `bulk_load`) per stage for the scale runs, counted on the stage's full
barrier with the two boxes: `arrive.expect_tx` of the whole stage's bytes, tiles plus scales.
The first version kept fp8 v2's consumer loop: wait on full[kt % STAGES], run the stage,
`fence.proxy.async`, arrive on empty. Nsight at 8192³ and the SASS showed where its
instructions went:

| per consumer warp and 64-byte stage (2 k64 steps, 32 mma) | instructions |
|---|---|
| mma | 32 |
| `ldmatrix` + scale loads | 12 + 6 |
| stage address, barrier parity and loop control, recomputed from kt every stage (`S2R SR_CgaCtaId`, `LEA`, 8 `IADD`, ...) | 31 |
| `fence.proxy.async`: 4 predicated-off `LDS`, `MEMBAR.ALL.CTA`, `FENCE.VIEW.ASYNC.S` | 6 |
| mbarrier `try_wait` spins: 11.2 M executions for 2.1 M stage waits, 5.3 per wait | 16 |

Two changes removed most of it. The k-loop is unrolled by the ring depth (`static_for` over the
stages inside a loop over rounds), so every stage address is the smem base plus an immediate
and the parity is one register flipped per round. And the fence is gone: the consumers only
*read* a stage before handing it back, the arrive has release semantics and the producer's wait
on it has acquire semantics, and CUTLASS's TMA pipelines release a stage with the arrive alone.
The fence compiled to a `MEMBAR.ALL.CTA` per stage per warp. Measured in one ncu session at
8192³:

| metric | v2 as ported | v2 unrolled, no fence | cuBLASLt |
|---|---|---|---|
| duration, SM clock | 891 us at 2.12 GHz | 869 us at 2.11 GHz | 857 us at 2.17 GHz |
| SM cycles | 1.912 M | **1.849 M** | 1.876 M |
| tensor pipe active | 84.9% | **88.0%** | 86.9% |
| executed instructions | 229.2 M | **180.0 M** | 179.9 M |
| issue slots busy | 18.5% | 15.2% | 14.7% |
| shared wavefronts / bank conflicts | 164 M / 34 M | 152 M / 34 M | 136 M / 8.7 M |
| L2 hit rate, L2 throughput | | 97.7%, 86.0% | 95.8%, 84.1% |
| L2 requests (64 B vs 128 B boxes) | | 78.2 M | 38.8 M |
| registers | 125 | 130 | 168 |

After the change the kernel takes 1.4% *fewer* cycles than cuBLASLt's
(`cutlass3x_sm120_bstensorop_s16864gemm_block_scaled_ue4m3xe2m1..._128x128x256`, 8 math warps
plus a producer warpgroup, 2 stages of 128-byte K, 88 KB of smem, a one-block-per-tile grid),
with the same instruction count and a busier tensor pipe. The time it loses is the clock
(next section).

The configuration sweep (`SPARK_FP4GEMM_V2_CONFIG`), after both changes, NVFP4, `--iters=50`,
TFLOPS / % of cuBLASLt, same shapes as the variant 1 table:

| config | tile, BK, stages, warps, blocks/SM | smem | 4096³ | 8192³ | gate/up | down | qkv | 2048³ |
|---|---|---|---|---|---|---|---|---|
| 0 | 128x128, 128, 2, 2x4, 1 | 72 KB | 1169 / 96% | 1222 / 94% | 1115 / 92% | 1327 / 106% | 1140 / 95% | 806 / 119% |
| **1** | 128x128, 64, 4, 2x4, 1 | 72 KB | **1191 / 97%** | **1263 / 97%** | **1165 / 96%** | **1379 / 109%** | **1166 / 97%** | 806 / 116% |
| 2 | 128x128, 64, 5, 2x4, 1 | 90 KB | 1191 / 98% | 1264 / 98% | 1171 / 96% | 1371 / 109% | 1167 / 96% | 807 / 116% |
| **3** | 128x128, 64, 2, 2x4, 2 | 36 KB | 1027 / 83% | 1102 / 85% | 1003 / 82% | 1203 / 96% | 1025 / 85% | **809 / 116%** |
| 4 | 128x128, 64, 4, 2x2 of 64x64, 1 | 72 KB | 1113 / 91% | 1225 / 94% | 1126 / 93% | 1288 / 103% | 1090 / 90% | 736 / 106% |
| 5 | 128x128, 128, 2, 2x2, 1 | 72 KB | 1150 / 93% | 1254 / 97% | 1142 / 94% | 1348 / 107% | 1114 / 93% | 805 / 116% |
| 6 | 128x128, 64, 3, 2x4, 1 | 54 KB | 1169 / 96% | 1260 / 98% | 1156 / 95% | 1356 / 108% | 1153 / 96% | 806 / 116% |
| 7 | 128x256, 64, 3, 2x4 of 64x64, 1 | 81 KB | 1094 / 90% | 1217 / 94% | 1124 / 93% | 1234 / 98% | 1112 / 92% | 802 / 116% |
| 8 | 128x256, 64, 2, 2x4 of 64x64, 1 | 54 KB | 1093 / 90% | 1239 / 96% | 1136 / 93% | 1261 / 101% | 1114 / 92% | 807 / 116% |
| 9 | 256x128, 64, 3, 4x2 of 64x64, 1 | 81 KB | 1011 / 82% | 1142 / 88% | 1039 / 85% | 1185 / 94% | 1036 / 87% | 807 / 116% |

Four stages of 64 bytes (config 1) is the default for grids past two blocks per SM, config 3
below that. At 2048³ every configuration lands at 806 to 809: 256 tiles are 1.5 waves at one
block per SM and 0.75 at two, and in both cases 86 SMs run two tiles, so the time is two tiles
whatever the pipeline (a Stream-K schedule is the fix; cuBLASLt is at 692 there). The larger
tiles (7 to 9) were the attempt to cut the L2 traffic by a quarter; they lose 2 to 15%, and
the 64x64 warp tiles lose on the 128x128 tile too (4, 5): with 4 or 8 of them per block the
schedulers have 1 to 2 warps each to hide the `ldmatrix` latency, the fp8 kernel's finding.

The split-K tail was fp8 v2's: `split = min(KT, resident / tail)` slices per tail tile, a
workspace and counters zeroed with two `cudaMemsetAsync` before the launch. At 4096³ that is 4
tail tiles split 32 ways, each slice one 64-byte k-tile plus a prologue and 64 KB of atomics,
and two memset launches in front of a 115 µs kernel. Now the last slice to arrive writes zeros
back over the tile it converts and resets its counter (`fp4gemm_tile::Workspace`: zeroed once
at allocation, no memset per call), and `split` is capped at 8 (`SPARK_FP4GEMM_MAX_SPLIT`
overrides it). Measured on v2, % of cuBLASLt:

| max split | 4096³ (4 tail tiles) | 4096x6144x4096 (6) | 8192³ (16) | 4096x4096x14336 (4) | 4096x28672x4096 (28) |
|---|---|---|---|---|---|
| none (32, 28, 10, 42, 6), memset-free | 94.3% | 93.8% | 96.9% | 105.8% | 96.1% |
| 2 | | | 97.1% | 102.4% | 96.4% |
| 4 | 96.8% | 96.7% | 97.4% | 107.6% | 96.5% |
| **8** | 96.8% | 97.0% | 97.8% | 107.7% | 96.0% |
| 16 | 95.0% | 95.0% | | | |

(The memsets themselves were worth about 3 points at 4096³: 91% with them and 94.3% without,
both at the uncapped split, in back-to-back sessions.)

## Measured

`bench_fp4gemm --iters=50`, the full default run in one process (so the shapes that come late
run on a hot card), ms / TFLOPS / % of cuBLASLt. cuBLASLt is `cublasLtMatmul` with
`CUDA_R_4F_E2M1` operands, `VEC16_UE4M3` scale mode on both, the same blocked scale tensors,
the per-tensor scales in alpha, bf16 out, a 32 MB workspace and its heuristic's first
algorithm, timed on the same stream next to each variant. It has no MXFP4 kernel for sm_120
(the heuristic returns none, and torch says "MXFP4 scaling only supported in CUDA for
B200/B300"), so the MXFP4 rows have no reference column; compare them with the NVFP4 rows.
Every NVFP4 row, all three variants, is **bit-identical to cuBLASLt on every output**, and the
MXFP4 rows pass the CPU check of the dequantized products.

| M x N x K | cuBLASLt NVFP4 ms / TFLOPS | v0 | v1 | v2 |
|---|---|---|---|---|
| 1024³ | 0.0125 / 171.6 | 0.0236 / 90.9 / 53.0% | **0.0084 / 254.2 / 148.1%** | n/a |
| 2048³ | 0.0248 / 691.8 | 0.1217 / 141.1 / 20.4% | 0.0276 / 622.8 / 90.0% | **0.0212 / 808.5 / 116.4%** |
| 4096³ | 0.1075 / 1279.0 | 0.8876 / 154.8 / 12.1% | 0.1157 / 1188.1 / 93.0% | **0.1115 / 1232.8 / 96.4%** |
| 8192³ | 0.7666 / 1434.2 | 7.9742 / 137.9 / 9.6% | 0.8393 / 1310.1 / 91.3% | **0.7953 / 1382.6 / 96.4%** |
| 4096 x 6144 x 4096 (qkv) | 0.1547 / 1332.7 | 1.3238 / 155.7 / 11.7% | 0.1730 / 1191.7 / 89.9% | **0.1685 / 1223.6 / 91.8%** |
| 4096 x 28672 x 4096 (gate/up) | 0.6929 / 1388.5 | 6.2470 / 154.0 / 11.1% | 0.7811 / 1231.8 / 88.7% | **0.7621 / 1262.4 / 90.9%** |
| 4096 x 4096 x 14336 (down) | 0.3572 / 1346.9 | 4.5637 / 105.4 / 7.8% | 0.3632 / 1324.3 / 98.3% | **0.3564 / 1349.6 / 100.2%** |
| 1 x 4096 x 4096 | 0.0217 / 1.5 | n/a | **0.0091 / 3.7 / 238.4%** | n/a |
| 16 x 4096 x 4096 | 0.0213 / 25.2 | 0.0255 / 21.1 / 83.8% | **0.0091 / 59.1 / 227.1%** | n/a |
| 64 x 4096 x 4096 | 0.0212 / 101.2 | 0.0276 / 77.9 / 77.0% | **0.0107 / 200.9 / 193.1%** | n/a |
| 16 x 28672 x 4096 | 0.0616 / 61.0 | 0.1032 / 36.4 / 59.5% | **0.0419 / 89.7 / 147.0%** | n/a |
| 16 x 4096 x 14336 | 0.0391 / 48.1 | 0.0993 / 18.9 / 39.3% | **0.0242 / 77.7 / 161.5%** | n/a |

The Llama-3-8B shapes are its projections at 4096 tokens: qkv 4096 → 6144 (32 query and 8 K/V
heads of 128), gate/up 4096 → 2 x 14336, down 14336 → 4096; o is the 4096 square. Timed one
shape per process, where the card starts cool for each, v2 reads 96.5 to 98.5% of cuBLASLt at
4096³ and 8192³, 96.0 to 96.6% on gate/up, 96.7 to 97.0% on qkv and 100 to 108% on down (the
split and order checks above and below); the full run's 91 to 92% on qkv and gate/up is the
same kernels on a card that has been at 600 W for a minute. The decode rows in GB/s of the
traffic floor (weights plus their scales plus the activations): 1,043, 1,057, 946, 1,602 and
1,376; the 9.4 MB of a 4096² NVFP4 weight is read in 9.1 µs where cuBLASLt's NVFP4 kernel
takes 21 µs, and where the fp8 kernel reads its 16 MB in 13.2 µs.

MXFP4, same run, ms / TFLOPS:

| M x N x K | v0 | v1 | v2 | best MXFP4 rung / best NVFP4 rung (speed) |
|---|---|---|---|---|
| 1024³ | 0.0234 / 91.7 | **0.0073 / 294.3** | n/a | 115% (v1) |
| 2048³ | 0.1196 / 143.6 | 0.0274 / 627.9 | **0.0196 / 875.8** | 108% |
| 4096³ | 0.9083 / 151.3 | 0.1117 / 1229.9 | **0.1075 / 1278.3** | 104% |
| 8192³ | 9.9414 / 110.6 | 0.8075 / 1361.6 | **0.7792 / 1411.1** | 102% |
| 4096 x 6144 x 4096 | 1.3483 / 152.9 | 0.1675 / 1230.9 | **0.1627 / 1267.0** | 104% |
| 4096 x 28672 x 4096 | 6.4198 / 149.9 | 0.7721 / 1246.0 | **0.7449 / 1291.6** | 102% |
| 4096 x 4096 x 14336 | 6.0299 / 79.8 | 0.3468 / 1387.0 | **0.3358 / 1432.6** | 106% |
| 1 x 4096 x 4096 | n/a | **0.0091 / 3.7** | n/a | 100% |
| 16 x 4096 x 4096 | 0.0212 / 25.3 | **0.0108 / 49.9** | n/a | 84% |
| 64 x 4096 x 4096 | 0.0234 / 91.9 | **0.0090 / 238.8** | n/a | 119% |
| 16 x 28672 x 4096 | 0.0805 / 46.7 | **0.0426 / 88.2** | n/a | 98% |
| 16 x 4096 x 14336 | 0.0765 / 24.5 | **0.0236 / 79.6** | n/a | 103% |

MXFP4 is 2 to 8% faster than NVFP4 on the large shapes: its scale words cover two k64 steps,
so the consumers issue half the scale loads, and there is half the scale data to copy (1/32
of a byte per value instead of 1/16). The decode rows are within their ±1.5 µs run-to-run
spread of each other.

Against torch (`scripts/bench_torch.py --only fp4gemm`): `F.scaled_mm` with the two NVFP4
scale levels (`BlockWise1x16` + `TensorWise`, `SWIZZLE_32_4_4`) on the same bytes is cuBLASLt
with torch's heuristic, and it returns the same bits. `sk.fp4gemm` is 1.09x faster at 8192³,
1.13x at 4096³, 1.02 to 1.20x on the Llama shapes, 1.95x at 2048³ and 1.9 to 3.8x on the
decode rows.

Against the fp8 GEMM on the same shapes: v2 at 8192³ is 1,383 TFLOPS against fp8 v2's 679
(docs/design/fp8gemm.md), 2.04x, the instruction ratio; at 4096³ 1,233 against 704.

### Power

At 8192³ both kernels pin the card at 600 W within a second, and the clock is what separates
them. `nvidia-smi --query-gpu=clocks.sm,power.draw -lms 100` during a 2,500-iteration loop of
each (the bench times cuBLASLt first):

| kernel | SM clock | power | TFLOPS |
|---|---|---|---|
| cuBLASLt NVFP4 | 2,205 to 2,212 MHz | 600 W | 1,293 |
| v2 | 2,145 to 2,160 MHz | 600 W | 1,251 |

The 2.7% clock gap is the 3.3% time gap, and the order does not explain it: timing v2 first
(`--ours-first=1`) moves the ratio by 0.3 to 0.5 points. The fp8 kernel was in the same place
and got there by issuing fewer instructions; this one issues the same number as cuBLASLt and
still draws more per unit of work. What differs in the counters: 16 M more shared-memory
wavefronts (the scale loads, below), twice the L2 requests (TMA boxes of 64-byte rows against
cuBLASLt's 128-byte ones, the same sectors), and 3% more L2 sectors, which the epilogue's
half-sector bf16 stores would account for (cuBLASLt stages its tile through `stmatrix` and a
TMA store). The usable fp4 roof for a
long GEMM on this card is about 2,029 x 2.2 / 2.92 ≈ 1,530 TFLOPS, and cuBLASLt's 1,430 is 93%
of it, ours 90%.

### What did not work

- **Removing the scale-load bank conflicts.** The 34 M conflict wavefronts are all the scale
  loads: rows r and r + 8 of an m16 tile (lanes 4g and 4g + 1) sit 128 bytes apart in the tile,
  in the same banks, so the 8-byte A load takes 8 wavefronts where 2 would do and each B load 2.
  A build that pointed those loads at conflict-free (wrong) addresses ran at the same speed,
  96.3% against 96.5% of cuBLASLt at 8192³, so the TMA-compatible swizzle that would fix it (the
  copy engine's swizzles cannot, they XOR the chunk with bits that are zero here) is not worth a
  per-lane copy loop in the producer.
- **Splitting a warp's 32 columns into two 16-column halves 32 columns apart,** so that one
  8-byte load fills both B scale registers (2 scale loads per step instead of 3): 29.6 M
  conflicts instead of 33.9 M, 0.7% fewer cycles, a lower clock, the same time. Reverted.
- **Larger tiles:** configs 7 to 9 above.

## The quantizer

`fp4_quantize(x, q, sf, rows, K, scale, format)` turns bf16 rows into the packed codes and the
blocked scales, one thread per scale block: two or four 16-byte loads, the block's max |x|,
the scale, 16 or 32 codes packed into one 8- or 16-byte store, one scale byte.

- **NVFP4**: with s the per-tensor decode scale (the caller's, or max|x| / (6 x 448) from
  `sk.fp4_quantize`), the block scale is the e4m3 value nearest max|block| / 6 / s (at most
  448), and each value is e2m1(x / (s · scale)), round to nearest even, saturating at 6.
- **MXFP4**: the OCP recipe, 2^e with e = floor(log2(max|block|)) − 2 (e2m1's top binade),
  values e2m1(x · 2^−e).

The rounding is written as seven compares rather than `cvt.rn.satfinite.e2m1x2.f32`, so that
`spark_kernels.reference.quantize_nvfp4` / `quantize_mxfp4` can do the same operations in
torch; the tests compare every byte, on ragged rows, an all-zero block, an outlier and a given
scale that saturates. `bench_fp4gemm` times it (`fp4quant` rows) and checks its bytes against
the CPU copy: 19.4 µs for 4096 x 4096 (2,213 GB/s of the traffic floor, the 32 MB input served
from L2) and 76.9 µs for 4096 x 14336 (1,957 GB/s; the timing loop reuses one 117 MB input, so
part of it still comes from L2).

## Correctness

The bench quantizes Gaussian rows on the CPU with the quantizer's arithmetic, checks the whole
output against cuBLASLt (NVFP4) and 4,096 sampled outputs against a CPU fp32 dot product of
the dequantized values, both with the 2% of max|C| tolerance of the other GEMM benches. On
these inputs the tolerance is not needed for NVFP4: every output of every row is the same bf16
as cuBLASLt's. A dequantized product (an e2m1 code times an e4m3 scale, twice) has at most 12
significant bits, and a double-precision check on a 1024 x 1024 x 4096 sample found every one
of the 1 M sums exactly representable in fp32, for both formats, so both kernels round the
same exact sum once. `tests/test_fp4gemm.py` runs every variant, tile, decode configuration and
split-K path of both formats against `reference.fp4gemm`; checks every variant bit for bit on
inputs built so that every sum is exact, with block scales that differ per block (a scale word
sent to the wrong row, column or k-block changes the bits); checks the NVFP4 output of every
variant bit for bit against `F.scaled_mm`; and covers the torch fp4 / fp8 dtypes as inputs and
the error paths.

## Accuracy

`scripts/fp4_accuracy.py` at 4096³: the fp32 product of the bf16 inputs is the reference, each
scheme quantizes A and B and runs the GEMM (`sk.fp4gemm`, `sk.fp8gemm`), and each cell is max
|error| / max |C|, then ||error|| / ||C||. The inputs are the MX fp8 study's.

| input | NVFP4 | MXFP4 | e4m3 per-tensor | MXFP8 |
|---|---|---|---|---|
| Gaussian | 1.45e-1 / 1.34e-1 | 1.69e-1 / 1.62e-1 | 4.25e-2 / 3.75e-2 | 4.51e-2 / 4.16e-2 |
| 0.1% of entries x 100 | 4.26e-2 / 7.04e-2 | 2.39e-1 / 2.00e-1 | 4.16e-2 / 3.76e-2 | 9.93e-2 / 5.73e-2 |
| 8 channels of A x 50 | 1.35e-1 / 1.18e-1 | 2.35e-1 / 1.86e-1 | 5.51e-2 / 3.73e-2 | 9.88e-2 / 4.86e-2 |
| 8 channels of A x 2000 | 1.35e-1 / 9.95e-2 | 2.56e-1 / 1.75e-1 | 6.02e-2 / 3.75e-2 | 1.07e-1 / 5.06e-2 |

e2m1 has one mantissa bit: 0.5, 1, 1.5, 2, 3, 4, 6. With the block maximum mapped to 6, a
Gaussian block's values fall mostly in (1, 4), where the step is 0.5 to 1, so each operand
carries about 9.5% relative error and the product about 13%: NVFP4's 13.4% Frobenius error is
3.6 times e4m3's 3.75%. The block scale is what keeps it there under outliers. NVFP4's e4m3
scale per 16 values has three mantissa bits of its own, so it lands a block's maximum within
6% of 6, and a block that holds an outlier does not coarsen the others: the error stays at 10
to 12% (lower than Gaussian, because the outlier channels, quantized in their own blocks,
dominate ||C||). MXFP4's power-of-two scale per 32 lands the maximum anywhere in [4, 8), wastes
up to a binade of the tiny e2m1 range and saturates the top of it (a maximum in [6, 8) clips
to 6): 16 to 20%, 1.2 to 1.8 times NVFP4. The per-tensor NVFP4 scale does not suffer the way a
per-tensor fp8 scale would: it only sets where the block scales sit in e4m3's range, which is
about 2^18 wide, so a 2000x outlier still leaves the other blocks' scales in e4m3's normal
range.

`tests/test_fp4gemm.py::test_fp4_accuracy_against_bf16` pins NVFP4 between 12 and 15% on
Gaussian and under 12% with 8 outlier channels, and MXFP4 at least 1.1 and 1.5 times NVFP4
respectively.

## What was done and what remains

Done: the fp4 instruction's fragment and scale-register maps, measured, including the
thread-id 1 selection that lets one register serve two tiles; the blocked scale layout as the
API, which turns every scale register into one aligned shared-memory word and every stage's
scales into one bulk copy; the three rungs for NVFP4 and MXFP4 on one code path, bit-identical
to cuBLASLt's NVFP4 kernel on every output; v2 at 96 to 98% of cuBLASLt on 4096³ and 8192³ and
91 to 108% on the Llama-3-8B shapes, 116% at 2048³, the decode rows at 147 to 238%; MXFP4,
which cuBLASLt and torch do not offer on this card, 2 to 8% faster still; a quantizer that the
Python reference reproduces byte for byte.

Open:

- **The clock.** At a fixed clock v2 needs 1.4% fewer cycles than cuBLASLt; at 600 W it runs
  2.7% slower clocks. The candidates are the ones in the Power section: 128-byte TMA rows
  (BK = 128 needs 108 KB for three stages; the 2-stage configuration 0 loses 1 to 3 points),
  the epilogue through `stmatrix` and a TMA store, and conflict-free scale loads.
- **Stream-K.** 2048³ (1.5 waves) and the 91 to 97% shapes with a tail are hgemm variant 6's
  problem; its schedule on this mainloop is the next rung.
- **Scale recipe.** NVFP4 maps each block maximum to 6. Choosing the block scale that minimizes
  the block's squared error instead (a search over a few e4m3 neighbours) is known to cut fp4
  error by a tenth or more, and needs nothing from the GEMM.
- **Decode.** 9.1 µs for 9.4 MB is 1,060 GB/s; the fp8 note's single-launch floor for 16 MB was
  13.3 µs, so a lone launch of this size is near its floor, and CUDA graphs are the lever.
