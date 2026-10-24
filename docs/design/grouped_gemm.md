# Grouped GEMM for MoE prefill

Every expert's GEMM of an MoE layer in one persistent launch, bf16 or e4m3:
`src/kernels/grouped_gemm.cu`, `sk.grouped_gemm(a, w, offsets, scale_a=None, scale_b=None)`,
`sk.moe_offsets(expert_ids, E)` to sort a routing, `reference.grouped_gemm` (the loop over
experts) and `reference.moe_route` (the seeded router the benches use). Bench:
`bench_grouped_gemm` (against cuBLAS grouped and the per-expert cuBLAS loop) and
`scripts/bench_grouped.py` (against the torch loop). Tests: `tests/test_grouped_gemm.py`.

Measured on the RTX 5090 (CUDA 13, torch 2.14+cu130) on 2026-10-13, with other tenants on
the card holding about 18 GB, so the DeepSeek fp8 rows did not fit and are not measured.
Every row below passed its correctness check first: `bench_grouped_gemm` against cuBLAS
grouped to 1% of max|C|, `scripts/bench_grouped.py` against the torch loop, and the 15 tests
in `tests/test_grouped_gemm.py`. The kernel compiled and passed on its first run.

## The problem

A prefill batch of T tokens routed top-k over E experts is E GEMMs of different heights:
expert e gets the rows the router sent it, `rows_e` of them, and multiplies them by its own
`[N, K]` weight. What most code does is the loop:

```python
counts = torch.bincount(ids.flatten(), minlength=E).tolist()   # a device -> host sync
for e in range(E):
    out[lo:hi] = a[lo:hi] @ w[e].T
```

Three costs, all independent of the math: the sync for the counts, one launch per expert
(128 or 256 of them for Qwen3 or DeepSeek), and the tail of every small GEMM, where a 16-row
expert gets a grid of 16 blocks on a 170-SM card and the other 154 SMs wait for it. cuBLAS
grouped GEMM (`cublasGemmGroupedBatchedEx`, CUDA 12.5+) removes the launches but still wants
the per-group sizes on the host, so it keeps the sync.

## Inputs

The `w1gemm_moe` convention: `a [T, K]` sorted by expert, expert e owning rows
`offsets[e] .. offsets[e+1]-1`, `offsets` int32 `[E + 1]` on the device with `offsets[E] == T`,
empty experts allowed; `w [E, N, K]`, each expert's weight as `nn.Linear` stores it, so both
operands are K-contiguous; `C [T, N]` bf16. fp8 takes `scale_a [1]` and `scale_b [E]`, the
per-tensor activation scale and one weight scale per expert that fp8 MoE checkpoints ship.
Rules: `1 <= E <= 512`, `N % 128 == 0`, `K % 64 == 0` (bf16) or `K % 128 == 0` (fp8).

## The design

**The tile list on the device.** Each block's producer warp reads `offsets`, takes
`ceil(rows_e / 128)` m-tiles per expert and scans them 32 experts at a time with
`__shfl_up_sync` into a prefix `tp[E + 1]` in shared memory, before the block's one
`__syncthreads`. Expert e then owns tiles `[tp[e] * N/128, tp[e+1] * N/128)`, m-tile fastest
within the expert. Every block computes the same prefix from the same 1 KB of offsets, so
every block derives the same schedule with no global queue, no atomics and no host
involvement. A tile index goes back to its expert with a binary search over `tp` (at most 9
steps for 512 experts), once per piece, in the producer; the expert rides along in the item
the consumers read.

M-tile fastest matters for L2: consecutive tiles share the expert's weight columns, so the
170 tiles of one wave on Mixtral at 8192 tokens (about 17 m-tiles per expert) are 10 weight
column strips (10 MB) and one expert's rows (16 MB), and each byte of W is read from DRAM
about once.

**The work split.** hgemm variant 4/6's hybrid, computed per block from the tile count:
with G resident blocks, the last partial wave plus one full wave (`tiles % G + G` tiles) are
cut into G equal ranges of (tile, k-step) work, and every tile before them is whole and goes
to block `tile % G`. A Stream-K tile's pieces are summed through hgemm_sk's chain fixup: one
fp32 slot and one epoch-tagged flag per Stream-K tile, each piece waiting for the flag to
say everything before its first k-step is in the slot, so the output is the same bits every
run and nothing is memset per call (the flags are zeroed once at allocation and the epoch
moves on). The ranges are walked from the top, the piece that starts a tile first, as in
variant 6. When there are fewer tiles than blocks, the number of ranges is capped at one
per 32 k-steps (hgemm_sk's `kMinShare`) but never below the tile count.

The point for MoE: the work is counted in k-steps of 128-row tiles, so a 3-row expert costs
one tile's k-steps on one SM and a 2,000-row expert sixteen tiles' worth spread over the
grid; nothing waits on the largest expert's GEMM to drain the way a per-expert grid does.

**The mainloop.** fp8gemm variant 2's stage, unchanged for both types. Both operands are
K-contiguous, so a stage is two TMA boxes of 128 rows by BKB bytes (64 or 128), swizzled by
the copy engine over the row, and `fp8gemm_tile::swz` is the matching read address. A comes
from one 2-D map over `[T, K]` at row `offsets[e] + 128 * m-tile`; W from one 3-D map over
`[E][N][K]` with the expert as the outer coordinate, so a box never crosses from one expert
into the next and no per-expert map is encoded. Both maps move raw bytes: a bf16 row of K
values is 2K bytes, and the swizzle is defined on bytes.

The consumers run `fp8gemm_tile::mma_stage` for e4m3 (the block-scaled
`kind::mxf8f6f4` instruction with every block scale 2^0, which is fp8gemm's full-rate path)
and, for bf16, `bf16_stage`: the same ldmatrix addressing on the same 32-byte k-steps, one
`m16n8k16` per fragment pair instead of one `m16n8k32`. The fragment words line up because
ldmatrix is a 16-bit instruction that does not know the element type: lane (g, t) gets bf16
k = 2t, 2t+1 of row g, which is the m16n8k16 A and B fragment word. The producer lane and the
consumers hand pieces over through hgemm variant 6's two-deep int4 ring, so the pipeline
stays full across a piece boundary and across experts.

A tile's 128 rows may run past its expert into the next one's rows (or past T, where the
copy engine zero-fills); those rows are multiplied and never stored. The epilogue is
`fp8gemm_tile::store_bf16` with the expert's last row as the bound, the scale
`scale_a * scale_b[e]` applied in fp32 before the one rounding.

Shared memory per block: the stages, 32 B of ring, 80 B of barriers and 2 KB of prefix. The
fp8 default (three 32 KB stages) comes to 100,468 B, rounded to 99 KB, the most a block may
take; that is what caps E at 512.

## Configurations

`SPARK_GROUPED_CONFIG=<i>` forces one: 0 = two 128-byte stages (64 KB, the bf16 default,
hgemm variant 6's winner), 1 = three (96 KB, the fp8 default, fp8gemm variant 2's
many-wave winner), 2 = four 64-byte stages, 3 = six. Swept on Qwen3 and Mixtral at 512 and
8192 tokens: all four within 2% of each other on every row, and the defaults best on six of
the eight rows and within 1% on the other two. The 64-byte stages are up to 5% slower on
fp8. Defaults kept.

## Results

`bench_grouped_gemm`, medians of 10, ms. Routing from the bench's seeded router with a
per-expert bias of N(0, 1), which leaves Qwen3 with 26, 12 and 6 empty experts and DeepSeek
with 89 and 31 at 512 and 8192 tokens. cuBLAS grouped is `cublasGemmGroupedBatchedEx` with
the counts already on the host; "+ sync" adds the offsets copy, the stream sync and the
pointer arrays it needs; the loop is one `cublasGemmEx` per non-empty expert. fp8 is
compared with cuBLAS grouped in bf16, since cuBLAS has no grouped fp8 path.

| shape | grouped bf16 | cuBLAS grouped | + sync | cuBLAS loop | grouped fp8 | bf16 vs cuBLAS grouped | fp8 vs cuBLAS grouped |
|---|---|---|---|---|---|---|---|
| Mixtral, 512 tokens | 0.865 | 1.038 | 1.047 | 0.882 | 0.347 | 1.20x | 2.99x |
| Mixtral, 2048 | 2.195 | 2.497 | 2.523 | 2.333 | 0.842 | 1.14x | 2.97x |
| Mixtral, 8192 | 8.024 (240 TFLOPS) | 9.231 | 9.258 | 8.541 | 3.123 (616 TFLOPS) | 1.15x | 2.96x |
| Qwen3-30B-A3B, 512 | 0.222 | 0.219 | 0.228 | 0.595 | 0.114 | 0.99x | 1.92x |
| Qwen3-30B-A3B, 2048 | 0.341 | 0.328 | 0.337 | 0.876 | 0.150 | 0.96x | 2.19x |
| Qwen3-30B-A3B, 8192 | 0.932 | 0.999 | 1.011 | 1.701 | 0.353 | 1.07x | 2.83x |
| DeepSeek V3-like, 512 | 3.184 | 3.164 | 3.170 | 5.271 | out of memory | 0.99x | |
| DeepSeek V3-like, 8192 | 10.247 | 10.030 | 10.038 | 13.326 | out of memory | 0.98x | |

`scripts/bench_grouped.py`, the torch loop most MoE code runs (`a[lo:hi] @ w[e].T` per
expert, counts on the host; fp8 is `torch._scaled_mm` per expert with rows padded to 16),
medians of 10, ms. Its router is `reference.moe_route`, a different draw from the C++
bench's, so the two tables agree in shape but not row for row.

| shape | torch loop | grouped bf16 | speedup | torch fp8 loop | grouped fp8 | speedup over the fp8 loop |
|---|---|---|---|---|---|---|
| Mixtral, 512 | 0.998 | 0.866 | 1.15x | 0.489 | 0.352 | 1.39x |
| Mixtral, 2048 | 2.275 | 2.158 | 1.05x | 1.106 | 0.805 | 1.37x |
| Mixtral, 8192 | 8.237 | 8.001 | 1.03x | 3.991 | 2.990 | 1.33x |
| Qwen3-30B-A3B, 512 | 1.074 | 0.228 | 4.71x | 2.258 | 0.119 | 19.0x |
| Qwen3-30B-A3B, 2048 | 1.224 | 0.358 | 3.42x | 2.572 | 0.158 | 16.3x |
| Qwen3-30B-A3B, 8192 | 1.724 | 0.952 | 1.81x | 2.561 | 0.357 | 7.2x |
| DeepSeek V3-like, 512 | 5.780 | 3.642 | 1.59x | not run | not run | |
| DeepSeek V3-like, 2048 | 7.390 | 4.744 | 1.56x | not run | not run | |

The loop with the `offsets.tolist()` sync it needs is 1 to 2% slower again than the loop
above (0.98x to 1.00x in the raw results).

What the numbers say:

* The win against what code actually runs is large where experts are many and small:
  3.4x to 4.7x over the bf16 torch loop on Qwen3 below 8192 tokens, 1.6x on DeepSeek, and
  7x to 19x for fp8, where `torch._scaled_mm` per expert is slower than the bf16 loop. The
  per-expert cuBLAS loop in C++ shows the same gap without Python (2.6x on Qwen3 at 512).
* Against cuBLAS grouped, bf16 is 14 to 20% faster on Mixtral's few large experts, 7%
  faster on Qwen3 at 8192, and at parity (0.96x to 0.99x) where the experts are small. On
  those shapes both are at the byte floor: DeepSeek at 512 tokens reads only the 167
  non-empty experts' weights, 4.90 GB, which in 3.18 ms is 1.54 TB/s, the copy rate; Qwen3
  at 512 reads 102 experts' 321 MB in 0.222 ms, 1.45 TB/s. Nothing grouped in bf16 can go
  under that, so the host sync cuBLAS needs (about 0.01 ms here) is the only difference
  left. The table above counts every expert's bytes; the real floor counts only the
  experts the router used.
* The padded rows of 128-row tiles did not cost anything visible: DeepSeek at 512 tokens
  (16 rows per expert, about 8x padded rows) runs at the copy rate. A 64-row tile is not
  worth adding for these shapes.
* fp8 halves the bytes and runs the block-scaled instruction: 2.0x to 3.0x over cuBLAS
  grouped in bf16, 616 to 643 TFLOPS on Mixtral at 8192 tokens.
* Mixtral bf16 at 8192 tokens is 240 TFLOPS, the plain GEMM's sustained rate under the power
  cap (hgemm variant 6: 243 at 8192^3), so the device-built schedule costs nothing there.

## What the shapes ask for

FLOPs and bytes per call (one expert projection, the bench's shapes), and two floors that
are arithmetic, not measurements: all bytes once at the measured 1,532 GB/s copy rate, and
the FLOPs at the mma.sync peak (258.7 TFLOPS bf16 at 2,976 MHz, 1,014 fp8). Real GEMMs run
at 2.72 to 2.81 GHz under the power cap, so the compute floor is optimistic by about 6% or
more.

| shape | rows | rows per expert | GFLOP | W bf16 | bytes floor bf16 | FLOP floor bf16 | bytes floor fp8 | FLOP floor fp8 |
|---|---|---|---|---|---|---|---|---|
| Mixtral, 512 tokens | 1,024 | 128 | 120 | 940 MB | 0.64 ms | 0.47 ms | 0.33 ms | 0.12 ms |
| Mixtral, 2048 | 4,096 | 512 | 481 | 940 MB | 0.71 ms | 1.86 ms | 0.39 ms | 0.47 ms |
| Mixtral, 8192 | 16,384 | 2,048 | 1,924 | 940 MB | 1.01 ms | 7.44 ms | 0.66 ms | 1.90 ms |
| Qwen3-30B-A3B, 512 | 4,096 | 32 | 13 | 403 MB | 0.28 ms | 0.05 ms | 0.14 ms | 0.01 ms |
| Qwen3-30B-A3B, 2048 | 16,384 | 128 | 52 | 403 MB | 0.32 ms | 0.20 ms | 0.17 ms | 0.05 ms |
| Qwen3-30B-A3B, 8192 | 65,536 | 512 | 206 | 403 MB | 0.50 ms | 0.80 ms | 0.29 ms | 0.20 ms |
| DeepSeek V3-like, 512 | 4,096 | 16 | 120 | 7.5 GB | 4.96 ms | 0.47 ms | 2.48 ms | 0.12 ms |
| DeepSeek V3-like, 2048 | 16,384 | 64 | 481 | 7.5 GB | 5.10 ms | 1.86 ms | 2.57 ms | 0.47 ms |
| DeepSeek V3-like, 8192 | 65,536 | 256 | 1,924 | 7.5 GB | 5.70 ms | 7.44 ms | 2.94 ms | 1.90 ms |

Before the runs this table read the shapes as weight-bound at 512 and 2048 tokens, and
the results bear that out (the Results section). The padded-rows worry for DeepSeek at 512
tokens (about 8x the useful rows in the 128-row tiles, 3.7 ms at the bf16 peak) did not
show: that row runs at the copy rate on the experts it reads.

## Not measured

DeepSeek V3-like in fp8, in both benches, and DeepSeek at 2048 tokens in the C++ bench and
8192 in the torch one: with about 14 GB free on the shared card they ran out of memory (the
bf16 weight, its e4m3 copy and the reference outputs together). They need the card to
themselves.

## Later: smaller tiles

The whole-layer work ([moe_layer.md](moe_layer.md)) made the tile height a template
parameter and added 16- and 64-row tiles, which bf16 now picks from the average rows per
expert (16 up to 24 rows, 64 up to 512). Every table above was measured with the 128-row
tile. Rerun through `scripts/bench_grouped.py`, the 64-row tile is 3 to 5% faster in bf16 on
four of the five shapes it takes (Qwen3 at 2048 tokens: 0.342 against 0.360 ms) and lost in
fp8, which keeps 128. The line above about a 64-row tile not being worth it held for DeepSeek
at 512 tokens, which runs at the copy rate; it did not hold at 32 to 512 rows per expert.
`bench_grouped_gemm` (the C++ bench against cuBLAS grouped) was not rerun.

