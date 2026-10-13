# Grouped GEMM for MoE prefill

Every expert's GEMM of an MoE layer in one persistent launch, bf16 or e4m3:
`src/kernels/grouped_gemm.cu`, `sk.grouped_gemm(a, w, offsets, scale_a=None, scale_b=None)`,
`sk.moe_offsets(expert_ids, E)` to sort a routing, `reference.grouped_gemm` (the loop over
experts) and `reference.moe_route` (the seeded router the benches use). Bench:
`bench_grouped_gemm` (against cuBLAS grouped and the per-expert cuBLAS loop) and
`scripts/bench_grouped.py` (against the torch loop). Tests: `tests/test_grouped_gemm.py`.

**Not measured.** The GPU box was down for the whole session this was written in: the
kernel has not been compiled, the tests have not run, and every number below is a byte or
FLOP count from the shapes, not a time. What to run first is at the end.

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
many-wave winner), 2 = four 64-byte stages, 3 = six. The defaults are borrowed from the
plain GEMMs, not swept on grouped shapes.

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

Two things this table says before anything runs. Most of these shapes are bound by the
weight bytes, not the FLOPs: at 512 and 2048 tokens every model reads more bytes than it
does math at either precision, which is where fp8 halves the floor and where the per-expert
loop's launches and tails cost the most. And the 128-row tile wastes rows when experts are
small: DeepSeek at 512 tokens has 16 rows per expert on average, so the tiles carry about 8x
the useful rows, 0.96 TFLOP of mma for 0.12 of math, about 3.7 ms at the bf16 peak against
the 5.0 ms byte floor. It still fits under the bytes on paper, but with no slack; a 64-row
tile for experts under 64 rows is the first change to try if that shape comes out
compute-bound.

## What to run first

1. `cmake --build build --target bench_grouped_gemm spark_kernels` and
   `pytest tests/test_grouped_gemm.py`: the scan, the 3-D W map and the bf16 fragment
   mapping are the new parts and the tests hit each (one-row and empty experts, 40 and 128
   experts across scan rounds, a one-tile long-K shape that runs a Stream-K chain).
2. `bench_grouped_gemm` for the three models at 512, 2048 and 8192 tokens, then the four
   configurations through `SPARK_GROUPED_CONFIG` on the bf16 and fp8 rows.
3. `scripts/bench_grouped.py` for the torch loop.
4. If DeepSeek at 512 tokens sits above its byte floor by more than the other shapes, ncu on
   it for tensor-pipe utilization: that is the padded-rows question above.
