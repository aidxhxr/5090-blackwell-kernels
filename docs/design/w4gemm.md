# W4A16: int4 weights, bf16 activations

`C[M,N] = A[M,K] · dequant(W)[K,N]`, row-major, bf16 activations and output, int4 weights
with one bf16 scale (and optionally one zero point) per 128 consecutive k of a column, fp32
accumulation. Source: `src/kernels/w4gemm.cu` (the four rungs and the launcher),
`src/kernels/w4gemm_quant.cu` (quantizer and repack), `src/kernels/w4gemm_internal.cuh`
(the packed layout, the k order, the dequant). Bench: `bench_w4gemm` (checks the quantizer
bit for bit against a CPU copy, every variant against cuBLAS on the dequantized weights,
and times the bf16 `hgemm` it replaces). Python: `sk.w4_quantize`, `sk.w4_repack`,
`sk.W4Weight.quantize(w)`, `sk.w4gemm(a, w4)`, and `SparkLayer(..., int4=True)`.

## Why

A decode step is a weight stream. `hgemm`'s decode kernel already reads bf16 weights at
the copy roof: 16×4096×4096 is 32 MB in 23.5 µs single launch, 21.5 µs back to back
([hgemm](hgemm.md), "Decode"). Nothing left to schedule, so the only way to go faster is
fewer bytes per weight. At 4 bits plus a 2-byte scale per 128 weights a weight costs
0.516 bytes instead of 2: 3.88x fewer bytes, and that ratio is the ceiling on the speedup
at M = 1.

## The format

**Quantizer** (`w4_quantize_bf16`, `reference.w4_quantize`): round to nearest per group of
128 k of one column, in fp32.

| | scale | code | weight |
|---|---|---|---|
| symmetric | `s = bf16(max|w| / 7)` | `q = clamp(rint(w / s), -8, 7) + 8` | `(q - 8) s` |
| asymmetric | `s = bf16((hi - lo) / 15)`, `lo = min(min w, 0)`, `hi = max(max w, 0)` | `z = clamp(rint(-lo / s), 0, 15)`, `q = clamp(rint(w / s) + z, 0, 15)` | `(q - z) s` |

The scale is rounded to bf16 before the codes are computed, so the codes are the best ones
for the scale the kernel will actually use. Output: `qweight [K/8, N]` int32 with eight
consecutive k per word, lowest k in the lowest nibble (GPTQ's packing), `scales [K/128, N]`
bf16, `zeros [K/128, N]` uint8. The CPU copy in the bench and the torch copy in
`reference.py` use the same fp32 operations and match the GPU bit for bit (checked on every
bench weight and in `tests/test_w4gemm.py`).

**The weight the kernel multiplies** is `bf16((q - z) s)`, rounded once. The dequant forms
`q - z` exactly (a difference of two integers in [128, 143], below) and rounds the product
once, which is what fp32 `(q - z) * s` rounded to bf16 gives. So every variant multiplies
exactly the matrix `reference.w4_dequantize` returns, and the check against cuBLAS on that
matrix only sees summation order. `test_w4gemm_weight_bits_are_the_reference` multiplies
an identity and compares the output to the dequantized matrix with `torch.equal`.

Scales are bf16, not fp16: they multiply bf16 in `mul.rn.bf16x2`. A GPTQ or AWQ checkpoint
with fp16 scales converts at load (8 significant bits instead of 11). The group size is fixed
at 128.

## The ladder

| variant | idea | what it fixes |
|---|---|---|
| 0 | one thread per output, scalar dequant from the packed layout (`w4::locate`), fp32 FMA | baseline |
| 1 | one warp per 16-column strip and 8 tokens over the whole K; per 128-k group a lane loads 2 x 16 bytes of weights and 4 x 16 bytes of activations straight into registers, 8 lop3 dequants, 8 `mma.sync` | the tensor cores and 128-bit loads; nothing staged, nothing loaded ahead |
| 2 | the block shape picked per call: M = 1, four warps along K on a 6-stage `cp.async` pipeline with one activation row staged; M <= 16, independent warps with the weights of the next 4 groups in flight in registers, four or eight along K, or one per strip when N is wide; M > 16, 64 or 128-column tiles with warps along N and M sharing staged activations on a Stream-K grid; K split across blocks through a self-resetting fp32 workspace | bytes in flight, SM fill, wave fit, and the tensor-bound regime |
| 3 | batch invariant: the four-warp shapes up to 16 rows, variant 2's wide tiles in the class order above, K cut only between the four chains | a row's bits do not depend on M ([below](#batch-invariant-variant-3)) |

## The repacked layout

### Weights as the 16-row operand

`mma.sync.m16n8k16` multiplies a 16x16 A by a 16x8 B. The kernels compute `C^T = W^T A^T`:
16 output columns of W are the 16 rows of A, and 8 tokens are the 8 columns of B. At M = 1
that wastes 7 of the 8 token columns rather than 15 of the 16 rows the other way round, and
the lane's A fragment of a 16x16 block is exactly 8 weights: `a0 = (row g, k 2c, 2c+1)`,
`a1 = (row g+8, ...)`, `a2 = (row g, k 2c+8, 2c+9)`, `a3 = (row g+8, ...)`, with
`g = lane / 4`, `c = lane % 4`. Eight nibbles, one 32-bit word.

For each 16 columns x 64 k the repack writes 32 lanes x 4 words, one word per k16 step: a
lane's 16 bytes are the four A fragments of those 64 k, and a warp's 512 bytes are one
contiguous block. Blocks run along K inside a 16-column strip, `int32 [N/16][K/64][32][4]`,
so a block reading its strip reads one contiguous run of memory.

A K-major block order (all strips' block for k 0..63, then the next 64) puts every block of
a wave on neighbouring addresses, the way a linear copy reads. Measured with the M = 1
kernel: 7.40 µs per launch back to back on 4096², against 7.30 for strip-major, and 39.5
against 39.2 µs on 28672x4096. Strip-major stayed.

### Nibble order for the dequant

Nibble `p` of a word holds `p = 0: (g, 2c)`, `4: (g, 2c+1)`, `1: (g+8, 2c)`, `5: (g+8,
2c+1)`, `2: (g, 2c+8)`, `6: (g, 2c+9)`, `3: (g+8, 2c+8)`, `7: (g+8, 2c+9)`. The dequant
takes nibbles `p` and `p + 4` together:

```
t = lop3(q >> 4p, 0x000F000F, 0x43004300)   // (a & b) | c: two bf16 lanes 128 + q
a = mul.rn.bf16x2(sub.rn.bf16x2(t, 128 + z), s)
```

`0x4300` is bf16 128.0, whose mantissa unit is 1, so OR-ing a nibble into its low mantissa
bits gives exactly `128 + q`. Subtracting `128 + z` (`0x4308` = 136 for the symmetric code)
is exact, and the multiply rounds once. Per 8 weights: 4 `LOP3`, 3 shifts, 4
`HADD2.BF16_V2`, 4 `HMUL2.BF16_V2`, all native on sm_120 (read off the SASS). bf16 has 7
mantissa bits, so the fp16 trick of masking the high nibble in place and folding a 1/16
into the multiply does not apply: the fourth bit would land in the exponent. Hence the
shifts.

### The k order inside a group

A lane's B fragment (activations) is `b0 = (token g, k 2c, 2c+1)`, `b1 = (token g, k 2c+8,
2c+9)`: 4-byte pieces, 16 bytes apart. The first version loaded them with `ldmatrix` from a
shared-memory copy of the activations. A dot product does not care in which order it visits
k, so the kernels instead feed k16 step `j` of a 128-k group, at mma index `kappa`, the
matrix k

```
k(j, kappa) = 32 (j / 2) + 8 c + 4 (j % 2) + 2 (kappa / 8) + kappa % 2,   c = (kappa % 8) / 2
```

on both operands (the repack permutes the weights to match). Lane `(g, c)`'s activations for
steps `2i` and `2i + 1` are then the eight consecutive k `32i + 8c .. 32i + 8c + 7` of token
`g`: one 16-byte load fills `b0` and `b1` of both steps, and the four lanes of a row read 64
contiguous bytes. That lets the small-M kernel load activations straight from global memory
into registers, and the large-M kernel read them from shared memory with `LDS.128` instead
of `ldmatrix` (bank-conflict-free with the chunk index XORed by `4 (row & 1)`). Scales stay
per 128 matrix k: the permutation never leaves a group.

## What a single launch can do

As in [hgemm](hgemm.md), first the floor: a read-only kernel that streams a buffer with 16 B
loads, timed like the bench times a GEMM, rotated past L2. Best of four grid shapes:

| bytes | one launch | 20 launches per event pair, per launch |
|---|---|---|
| 4.4 MB | 5.0 µs (880 GB/s) | 3.8 µs (1,159 GB/s) |
| 8.7 MB (int4 4096²) | 8.3 µs (1,050 GB/s) | 6.3 µs (1,380 GB/s) |
| 13.0 MB (int4 qkv) | 11.0 µs (1,178 GB/s) | 8.9 µs (1,468 GB/s) |
| 29.4 MB (int4 down) | 21.2 µs (1,386 GB/s) | 18.6 µs (1,580 GB/s) |
| 58.7 MB (int4 gate/up) | 37.6 µs (1,563 GB/s) | 36.0 µs (1,633 GB/s) |

Against the 1,633 GB/s these reach on large buffers, the fixed cost of a launch is about
3 µs single and 1 µs back to back, a third of an int4 4096² GEMM. The speedup over bf16 at
M = 1 cannot reach 3.88x on the small matrices: 23.5 µs against 8.3 µs is 2.8x at best on
4096² single launch, 21.4 against 6.3 µs 3.4x back to back. On gate/up (235 MB bf16
against 60.6 MB int4) it can, and does.

A per-block timeline (a `%globaltimer` stamp at block start, first stage landed and end,
scratch build) of the M = 8 kernel on 4096² makes the same point from the inside: the first
bytes land 0.74 µs (median) after the block starts, blocks end between 4.4 and 6.0 µs, and
the kernel's device span is 6.0 µs for 8.7 MB, 1.45 TB/s. The bench's 9.1 µs is that plus
the launch.

## M <= 16: keeping the weights moving

### The first version, and what it was waiting on

The first variant 2 was an `hgemm_decode`-style block: warps along K sharing a `cp.async`
pipeline of weights, scales and an 8-row activation tile read with `ldmatrix`. At M = 1 on
4096² it took 9.1 µs single, 7.66 µs back to back. Nsight (`--launch-skip 5`):

| metric | value |
|---|---|
| grid, smem per block | 128 blocks, 66.6 KB (one block per SM) |
| `dram__throughput` | 55.8% |
| SM active cycles / elapsed | 13.4 K / 22.0 K |
| achieved occupancy | 8.3% (4 warps per SM) |

42 SMs had nothing to do and the rest had one block each, because 8 KB per stage of the
66.6 KB was an activation tile of which seven rows were zeros. Making the activation rows per
block a template parameter (1, 4 or 8 rows; the `ldmatrix` rows past them read a zero
chunk, later the lanes past them feed zeros) let the one-token block stage one row: 5.2 KB
per stage, 6 stages, 3 blocks per SM, 256 blocks of 16 columns. 7.26 µs back to back.

Two things that did not move it. Two accumulator sets for alternate k16 steps (the 8 `mma`
of a group otherwise form one dependent chain): 7.36 µs, noise. And the math is not free,
but it is not the chain either: with the dequant and `mma` compiled out the pipeline alone
runs at 6.70 µs, 0.4 µs above the probe's 6.3. At K = 4096 a block has only 8 stages; the
prologue's five land nearly together and their math runs as a burst the loads cannot cover.

### M = 16: the activations were four times the weights

In the staged design every warp needs its group's activations in shared memory:
`16 tokens x 128 k x 2 B = 4 KB` per group against 1 KB of weights for a 16-column block.
The stage size capped the blocks per SM at two or three and the warps per SM at 4 to 8, and
at M = 16 each warp's work is a long serial chain (8 dequants, 16 `mma`, 8 `ldmatrix` per
group). Nsight on the Stream-K version (128x16 tiles): `sm__warps_active` 8.3%, SM active
cycles 63% of elapsed, DRAM 37%. Best staged configuration: 11.3 µs, against 9.1 at M = 8.

The k order above removed the activation tile. The M <= 16 kernel (`w4_warp_kernel`) is
independent warps: a block is 4 or 8 warps on one strip splitting its groups (warp `wk`
takes groups `wk, wk + WK, ...`, so the block reads consecutive 1 KB blocks at any moment),
each warp keeping the weights and scales of its next 4 groups in a register ring and the
activations of its next group in a second buffer, and no barrier until the warps' partial
sums are added in shared memory at the end. 144 registers at M = 16, no spills.

I also tried the weights through a per-warp `cp.async` ring in shared memory (each lane
copies its own 16-byte chunks, `cp.async.wait_group` then `__syncwarp`), on the argument
that `cp.async` groups complete in order and a ring of register loads may make the warp
wait for all of them. It was not better: 10.0 µs against 9.7 back to back on 4096² at
M = 16, 54.1 against 45.5 µs on gate/up. The register ring stayed.

### Which shape, by M and N

The sweeps (`SPARK_W4_CFG=entry[,split]` forces an entry of the launcher's table; single
launch / back to back, µs):

| M | block | 4096x4096 | 6144x4096 | 28672x4096 | 4096x14336 |
|---|---|---|---|---|---|
| 1 | pipelined, 4 warps along K, 1 row, 6 stages | **9.1 / 7.6** | **11.2 / 10.2** | 41.8 / 39.7 | **23.3 / 21.0** |
| 1 | independent warps, 4 along K, 8 tokens | 9.2 / 8.0 | 12.3 / 10.4 | 39.9 / 38.1 | 25.5 / 23.5 |
| 1 | independent warps, one per strip | 21.3 / 19.1 | 21.5 / 19.5 | **39.9 / 37.8** | 66.5 / 64.9 |
| 4 | pipelined, 4 along K, 8 rows | **9.2 / 8.0** | 15.1 / 12.9 | 42.0 / 40.8 | **23.5 / 22.2** |
| 4 | independent warps, 4 along K, 8 tokens | 9.2 / 8.0 | **12.7 / 10.4** | **39.8 / 38.2** | 25.3 / 23.2 |
| 16 | independent, 4 along K, 16 columns | **11.1 / 9.8** | **15.1 / 12.7** | 45.7 / 43.4 | 33.5 / 31.1 |
| 16 | independent, 8 along K, 32 columns | 11.1 / 9.0 | 19.0 / 15.9 | 48.0 / 46.8 | **25.7 / 24.8** |
| 16 | independent, one per strip | 25.5 / 23.3 | 25.4 / 23.8 | **42.0 / 39.9** | 80.9 / 79.6 |

Three rules come out of it, and `pick()` in `w4gemm.cu` is those rules:

* **Fit one wave.** The pipelined 8-row block holds 340 blocks at once; the qkv projection
  has 384 strips of 16 columns, so 44 of them ran a second wave (15.1 µs against 12.7 for the
  register kernel, which fits). A shape is taken only when its blocks fit the resident slots
  (`cudaOccupancyMaxActiveBlocksPerMultiprocessor` x SMs), as in `hgemm_decode`.
* **Wide N needs no K split.** With 1,792 strips (gate/up) one warp per strip over the whole
  K already keeps enough loads in flight, and splitting K only adds the reduction: 39.9 µs
  against 41.8. The rule: at least 4 x SMs strips (N >= 10,880). Variant 1 is the same idea
  without the ring and is as fast there at M = 1 and 4, and at M = 16 5% faster (39.8 against
  41.9 µs) with its 8-token chunks, which read the weights twice (the second time from L2);
  the 8-token one-warp shape of variant 2 took 48.1 µs there, so the 16-token one stayed.
* **Long K wants more warps along K.** The down projection (K = 14336, 112 groups) gives a
  4-warp block 28 groups per warp; 8 warps on 32 columns halve that: 25.7 against 33.5 µs at
  M = 16.

A K split across blocks exists for narrow N (a 256-column matrix has 16 strips): each slice
adds its fp32 sums into a workspace with atomics, and the block whose groups complete the
tile's count converts it to bf16 and zeroes the workspace and the counter, so the next launch
starts clean without a memset (the `hgemm_decode` scheme, counting groups instead of
slices). Split tiles are not bitwise reproducible run to run. None of the Llama shapes split.

## M > 16: the tensor cores

Above 16 tokens the dequant is shared by more `mma` (a fragment feeds `M / 8` of them) and the
kernel turns tensor-bound: 64x4096x4096 is 524,288 `mma.sync.m16n8k16`, 8.3 µs at the
258.7 TFLOPS peak and more at a sustained clock, against 5.6 µs of int4 weights at the roof.
The block is a 64 or 128-column tile of 32 tokens (warps along N, each 16 or 32 columns, all
reading the same activation stage with `LDS.128`), one 128-k group per stage, three or four
stages, on a Stream-K grid: every resident block takes an equal share of the tiles x groups
iterations in tile-major order, and tiles no single block covers go through the same
workspace. The best one-tile-per-block shape took 27.6 µs on 64x4096x4096; Stream-K 21.4.

What limits it, from Nsight on an earlier build of the 128x64 tile at M = 64: SM active
cycles 77% of elapsed (a block's first stages wait on DRAM, and 170 blocks with 6 iterations
each end at different times), tensor pipe active 54% of the active cycles, top stall
`math_pipe_throttle` then `wait`. So the kernel is not dequant-bound (the dequant is 15
instructions per fragment against `M / 8` `mma` of 32 clocks each); it is a GEMM tile that
has less reuse and a shorter pipeline than `hgemm`'s, and it needs more work than I gave it
to reach `hgemm`'s 90% of peak.

## Results

`bench_w4gemm`, RTX 5090, the four Llama-3-8B projections (`shape_utils.LLAMA3_8B_PROJECTIONS`)
at M = 1 to 256, symmetric weights, variant 2. Weights (with their scales) rotated past L2,
median of 50 single launches; "back to back" is 20 launches per event pair, how a decode step
or a CUDA graph queues its GEMMs. GB/s is the traffic floor (activations, packed weights,
scales, output, once each) over the time; the roof is `cudaMemcpy`'s 1,532 GB/s. The bf16
path is `hgemm` variant 6 on the dequantized weights (its decode kernel for M <= 64), timed
the same way.

| projection | M | w4gemm µs | GB/s | % of roof | bf16 hgemm µs | speedup | back to back µs | GB/s | speedup |
|---|---|---|---|---|---|---|---|---|---|
| qkv 4096 → 6144 | 1 | 11.2 | 1,160 | 76% | 33.8 | 3.01x | 10.1 | 1,289 | 3.10x |
| | 4 | 11.3 | 1,159 | 76% | 33.7 | 2.99x | 10.4 | 1,259 | 3.01x |
| | 16 | 13.2 | 1,004 | 66% | 33.7 | 2.54x | 12.0 | 1,108 | 2.61x |
| | 64 | 27.6 | 519 | 34% | 35.8 | 1.30x | 25.9 | 552 | 1.31x |
| | 256 | 74.7 | 244 | 16% | 60.5 | 0.81x | 74.3 | 245 | 0.80x |
| o 4096 → 4096 | 1 | 9.0 | 960 | 63% | 23.4 | 2.59x | 7.4 | 1,169 | 2.89x |
| | 4 | 9.2 | 946 | 62% | 23.5 | 2.55x | 7.9 | 1,099 | 2.70x |
| | 16 | 11.0 | 810 | 53% | 23.5 | 2.13x | 9.7 | 917 | 2.20x |
| | 64 | 21.4 | 452 | 30% | 25.3 | 1.18x | 20.4 | 476 | 1.12x |
| | 256 | 52.2 | 246 | 16% | 50.0 | 0.96x | 50.8 | 253 | 0.96x |
| gate/up 4096 → 28672 | 1 | 39.8 | 1,522 | 99% | 148.1 | 3.72x | 37.9 | 1,599 | 3.84x |
| | 4 | 39.8 | 1,529 | 100% | 148.1 | 3.72x | 37.9 | 1,606 | 3.86x |
| | 16 | 41.9 | 1,471 | 96% | 148.2 | 3.54x | 40.3 | 1,530 | 3.64x |
| | 64 | 91.8 | 706 | 46% | 168.5 | 1.84x | 91.0 | 711 | 1.81x |
| | 256 | 315.6 | 245 | 16% | 270.7 | 0.86x | 319.9 | 242 | 0.87x |
| down 14336 → 4096 | 1 | 23.3 | 1,303 | 85% | 72.5 | 3.12x | 20.8 | 1,457 | 3.43x |
| | 4 | 23.6 | 1,290 | 84% | 72.6 | 3.08x | 22.2 | 1,370 | 3.22x |
| | 16 | 25.5 | 1,210 | 79% | 74.3 | 2.91x | 24.3 | 1,270 | 2.96x |
| | 64 | 50.0 | 652 | 43% | 80.5 | 1.61x | 48.3 | 676 | 1.61x |
| | 256 | 160.2 | 248 | 16% | 131.8 | 0.82x | 160.4 | 248 | 0.82x |

At M = 1 the int4 weights stream at 63 to 99% of the copy roof, and the speedups sit where
the probe says they can: 3.72x (3.84x back to back) on gate/up against the 3.88x byte ratio,
2.6x to 3.1x on the three smaller matrices, whose 9 to 23 µs launches carry the same 1 to
3 µs of launch and ramp as a 150 µs one. Asymmetric weights (one more byte per 128 weights)
measure within 0.5 µs back to back: 37.6 µs on gate/up at M = 1, 7.5 on o, 21.3 on down.
(Single-launch medians of the small shapes move by 1 to 1.5 µs between runs, as in
[hgemm](hgemm.md): qkv at M = 4 measured 11.3 and 12.7 µs in two runs, 10.4 back to back in
both. The back-to-back column is the steadier one.)

The ladder at M = 1 and 16 (µs, single launch):

| shape | v0 | v1 | v2 |
|---|---|---|---|
| 1x4096x4096 | 171.1 | 33.8 | 9.0 |
| 16x4096x4096 | 238.5 | 33.8 | 11.0 |
| 1x28672x4096 | 185.2 | 39.8 | 39.8 |
| 16x28672x4096 | 1,174 | 39.8 | 41.9 |
| 1x4096x14336 | 592.5 | 113.3 | 23.3 |

Variant 1 has one 16-byte weight load in flight per warp and nothing else, so it is bound by
latency unless there are thousands of warps; with 1,792 strips (gate/up) there are.

### Where it crosses over with hgemm

| M | 4096x4096 | 6144x4096 | 28672x4096 | 4096x14336 |
|---|---|---|---|---|
| 32 | 1.54x | 1.94x | 2.76x | 2.58x |
| 64 | 1.18x | 1.30x | 1.84x | 1.61x |
| 96 | 1.16x | 1.40x | 1.25x | 1.31x |
| 128 | 1.00x | 1.15x | 0.95x | 1.07x |
| 192 | 0.96x | 1.03x | 1.16x | 1.10x |
| 256 | 0.96x | 0.81x | 0.86x | 0.82x |

(speedup over the bf16 `hgemm`, single launch). The int4 GEMM is ahead up to M = 96 on every
shape, level at M = 128 (0.95 to 1.15x) and behind at M = 256 (0.81 to 0.96x): the crossover
is at about 128 tokens, where the tensor time of both passes the time to stream bf16 weights
and the int4 tile's lower efficiency decides. (`hgemm` itself steps from its decode kernel to the 128x128 tiles
above M = 64 and is uneven at 192: cuBLAS takes 209 µs for 192x28672x4096 where `hgemm`
takes 269.)

Quantize and repack run once per weight, cold: 0.04 to 0.24 ms and 0.012 to 0.078 ms per
matrix.

## One decoder layer

`scripts/bench_layer.py` times a Llama-3-8B decode step with `SparkLayer(int4=True)`, whose
four projections are `w4gemm` on weights quantized once at construction. One layer's int4
weights are 107 MiB, which would mostly stay in the 96 MB L2 from one step to the next of a
single-layer benchmark (the o projection measured 4.7 µs that way, 7.3 with its weights in
DRAM). A 32-layer model never sees that, so the int4 steps rotate over four separately
quantized copies (428 MiB) and the CUDA graph holds one step per copy. bf16 weights are 416
MiB and never fit. Per layer, from the CUDA graph:

| decode | bf16 layer | int4 layer | speedup | projections, bf16 → int4 | 32 layers, tok/s |
|---|---|---|---|---|---|
| b1, 4,096 cached | 0.298 ms | 0.093 ms | 3.21x | 269 → 76 µs | 105 → 337 |
| b1, 16,384 | 0.329 ms | 0.134 ms | 2.46x | 271 → 77 µs | 95 → 234 |
| b1, 131,072 | 0.607 ms | 0.413 ms | 1.47x | 269 → 77 µs | 51 → 76 |
| b8, 4,096 | 0.370 ms | 0.180 ms | 2.05x | 270 → 81 µs | 676 → 1,389 |

At 4,096 cached tokens the projections are 90% of the bf16 step and still 82% of the int4
one. At 16,384 attention is a third of the int4 step and at 131,072 three quarters, and int4
weights do nothing for it; at b8 it is already half.
`tests/test_layer.py` checks the int4 layer against the torch layer on the dequantized weights
(`LayerWeights.w4_dequantized`) to the same tolerance as the bf16 decode.

## Accuracy

Round-to-nearest int4 with a scale per 128 weights has a relative error per weight of up to
half a step, about 7% of the group's largest weight for the symmetric code. That is the
format's cost, not the kernel's: the kernel adds nothing to it (its weights are bit for bit
the reference's) and sums in fp32. Better codes for the same format (GPTQ, AWQ) change the
codes and scales, not the kernel; their `qweight` packing is the one `w4_repack` takes. The
tests hold the quantizer to half a step plus the bf16 rounding of the product (0.53 of a
step), 0.6 for the asymmetric code, whose rounded zero point and rounded-down scale can clip
the top code by a further 0.06.

## Batch invariant: variant 3

`Engine(batch_invariant=True)` needs a row of C to come out with the same bits whatever M is
([batch_invariant](batch_invariant.md)). So the order an output element is summed in can't
depend on M, on the tile M picks, or on how K is cut up. Variant 2 cuts K twice: across the
warps of a block (WK picked by M and N) and across blocks (atomics into the workspace, or
Stream-K pieces).

The first variant 3 fixed the order by running variant 2's shapes with four warps along K at
every M, with no K split across blocks. Warp w accumulates the groups w, w + 4, w + 8, ...
from zero and the block adds the four partials in warp order. Up to 16 rows those are the
shapes variant 2 picks anyway. Above that they are the wrong shapes: a block with four warps
along K has a quarter of its warps along N, and from 32 rows on they cost 1.02 to 1.68x of
variant 2 (the "four-warp" column below).

### The class order

That order belongs to the element, not to the warps. Call the groups w, w + 4, w + 8, ...
class w. An element is four chains, one per class, each a run of k16 steps from a zero
accumulator, added `(((0 + p0) + p1) + p2) + p3`. One warp can compute exactly that: walk a
tile's groups class by class (0, 4, 8, ..., then 1, 5, 9, ..., and so on), and at the end of
each class add the accumulators into a running sum and clear them. That costs one more set of
accumulators, not four. With it, any of variant 2's wide tiles with one warp along K gives
the bits of the four-warp shapes, at any M.

In the code it is a template flag on the pipelined kernel (`CLS`): the stage loader maps a
position in the tile's order to its group (`class_group`), and the compute loop folds the
accumulators at each class end. Entries 15 to 19 of the table are variant 2's Stream-K tiles
(entries 8 to 12) in the class order. The 128 x 32 one takes 168 registers in the symmetric
build and 128 with zero points, no spills.

### A K split that keeps the bits

Whole tiles leave the last wave partly empty, which is what Stream-K is there to fix. The
class order allows one K split: cut a tile into its four classes. Each piece runs one class
into its own fp32 slot (four BM x BN slots per cut tile), and the block that completes the
tile's group count adds the four slots in class order from zero and converts
(`combine_classes`). Those are the adds the four-warp block does in shared memory. The piece
of class w stores `0 + p_w`, which is `p_w` unless `p_w` is -0, and a running sum is never -0
in round to nearest (x + y is -0 only when both are), so adding `0 + p_w` gives the bits of
adding `p_w`. Every slot is written before it is read, so only the tile's counter has to go
back to zero.

The plan (`class_plan`): whole tiles for the full waves of the resident blocks, the last
partial wave cut into pieces. Which tiles are whole only changes when an output is finished,
never its bits, so the schedule can follow the SM count or anything else.

### Which tile, which schedule

`bench_w4gemm --sweep=invariant` runs v2, v3 on both of its routes, and each class-order tile
under each schedule (every tile whole, every tile in four pieces, the plan) on the four Llama
projections, and compares every output with the four-warp route's bits. Single launches,
weights rotated past L2, median of 30 (20 from 512 rows), at 32 to 4096 rows. All 420
class-order rows of the final sweep carry the four-warp bits, and so did all 420 of an
earlier one with the first plan. Rows in `results/w4gemm_invariant.jsonl`. Two projections, us,
in bold what the rule runs:

| projection, M | four-warp | v2 | 64 x 32, plan | 128 x 32, whole | 128 x 32, pieces | 128 x 32, plan |
|---|---|---|---|---|---|---|
| o, 32 | 15.3 | 15.2 | **13.3** | 41.9 | 15.3 | 16.0 |
| o, 64 | 27.6 | 21.4 | 23.5 | 43.9 | 21.3 | **21.4** |
| o, 128 | 41.8 | 31.6 | 37.8 | 43.9 | 37.7 | **37.6** |
| o, 256 | 78.8 | 53.2 | 62.0 | 62.4 | 60.3 | **60.3** |
| o, 512 | 136.1 | 95.9 | 109.6 | 119.8 | 105.2 | **101.3** |
| down, 32 | 48.2 | 29.8 | **35.7** | 140.3 | 41.8 | 41.8 |
| down, 64 | 81.9 | 50.2 | 66.5 | 146.2 | 58.3 | **58.2** |
| down, 128 | 130.0 | 85.0 | 115.3 | 146.3 | 109.3 | **109.2** |
| down, 256 | 255.0 | 160.5 | 201.5 | 209.8 | 189.3 | **189.3** |
| down, 512 | 443.3 | 306.5 | 379.8 | 412.6 | 334.7 | **330.7** |

What it says:

* **Whole tiles only pay with many waves.** At 32 to 128 rows o has 32 to 128 tiles of
  128 x 32 for 170 SMs, and one whole tile of K = 4096 takes 42 to 44 us however few of
  them run. Cut into four, o at 64 rows is 21.3 us, level with v2.
* **The plan is the fastest of the three** wherever it differs from both others, at every
  point of the sweep from 32 to 4096 rows on all four projections. Where it cuts every tile it
  is the four-pieces launch, and the two runs of that launch differ by up to 0.6 us (15.3
  and 16.0 on o at 32 rows). The first plan cut the last wave only when that took fewer
  quarter waves than running it whole. o and down at 256 rows, 256 tiles on 340 slots (86
  SMs with two), then ran whole: 62.4 and 209.8 us against 60.3 and 189.3 cut.
* **64 x 32 where 128 x 32's pieces do not reach the SMs.** o and down at 32 rows have 32
  tiles of 128 x 32, 128 pieces for 170 SMs: 16.0 against 13.3 on o, 41.8 against 35.7 on
  down. Everywhere else 128 x 32 is as fast or faster. The 8-warp 128 x 32 and the 64-row
  tiles (entries 15, 18 and 19) were never the fastest.

So `pick_invariant`: from 17 rows the class-order 128 x 32 tile on the plan, or 64 x 32 when
four pieces per 128 x 32 tile come to fewer blocks than SMs; up to 16 rows, and on an N that
64 does not divide, the four-warp shapes. `w4gemm_v3_set_route(-2)` (or
`SPARK_W4_V3_ROUTE=-2`) puts the four-warp shapes back at every M and changes no bit, and
`w4gemm_v3_route_name(M, N, K)` says what runs. `tests/test_batch_invariant.py` checks that on
four shapes, symmetric and asymmetric: the rule, the four-warp route and every class-order
tile under every schedule give the same bits, and an int4 engine on the four-warp route alone
gets the logits of one on the rule in 32 slots. `tests/test_w4gemm.py` runs every class-order
tile under every schedule against the reference, including K = 384 (three groups, never cut)
and K = 1152 (classes of 3, 2, 2 and 2 groups).

The symmetric 128 x 32 class tile fits two blocks per SM; the asymmetric one, with 128 more
bytes of zero points per stage, fits one, as variant 2's Stream-K tile of the same shape
does. The plan counts the asymmetric tile's slots on its own (at 512 x 4096 x 4096: 340 whole
tiles and 172 cut symmetric, 510 and 2 asymmetric). I did not time the asymmetric rows.

### Results

`scripts/bench_invariant.py --part w4`, us per call (ten launches back to back between
events, median of 50 samples at M <= 64 and of 10 above; weights rotated past L2 at
M <= 64), v2 / v3 on the four-warp route / v3:

| M | qkv 4096 x 6144 | o 4096 x 4096 | gate_up 4096 x 28672 | down 14336 x 4096 | lm_head 4096 x 128256 |
|---|---|---|---|---|---|
| 1 | 10.5 / 10.5 / 10.5 | 8.1 / 7.9 / 8.1 | 36.9 / 40.3 / 40.3 | 21.3 / 21.2 / 21.2 | 158.1 / 163.9 / 163.9 |
| 8 | 10.7 / 10.7 / 10.7 | 8.5 / 8.3 / 8.3 | 38.2 / 38.9 / 38.9 | 24.9 / 24.2 / 24.2 | 158.3 / 159.3 / 159.4 |
| 32 | 16.8 / 17.1 / 20.3 | 13.5 / 14.6 / 12.5 | 53.2 / 71.3 / 52.1 | 29.0 / 48.8 / 34.1 | 217.3 / 327.0 / 202.8 |
| 64 | 26.4 / 37.7 / 29.1 | 20.9 / 25.7 / 20.0 | 91.7 / 140.5 / 89.0 | 49.4 / 81.2 / 57.4 | 383.3 / 618.0 / 380.9 |
| 128 | 41.1 / 60.1 / 44.8 | 30.1 / 40.8 / 35.9 | 163.2 / 216.2 / 167.9 | 84.1 / 130.3 / 108.2 | 742.1 / 952.8 / 730.1 |
| 256 | 74.3 / 99.7 / 78.1 | 52.1 / 78.3 / 58.9 | 317.8 / 425.8 / 319.0 | 160.3 / 257.9 / 187.7 | 1,475 / 1,875 / 1,426 |
| 512 | 139.6 / 193.9 / 142.1 | 95.9 / 136.0 / 100.2 | 632.6 / 835.3 / 636.9 | 308.8 / 447.3 / 332.9 | 2,938 / 3,720 / 2,817 |
| 2048 | 535.6 / 715.3 / 534.7 | 357.7 / 482.8 / 359.9 | 2,520 / 3,318 / 2,499 | 1,229 / 1,605 / 1,247 | 11,857 / 14,802 / 11,169 |
| 4096 | 1,080 / 1,426 / 1,072 | 718.8 / 951.5 / 715.4 | 5,081 / 6,620 / 4,986 | 2,585 / 3,572 / 2,625 | |

Up to 8 rows nothing changed (the same shapes; gate_up and lm_head at one row keep the 4 to
9% the four-warp shape costs there). From 512 rows v3 is 0.94 to 1.08x of v2, where the
four-warp route is 1.25 to 1.45x, and on lm_head it is ahead of v2 at every M from 32. At 32
and 64 rows, the decode steps of 32 and 64 slots, the four projections add up to 119.0 and
195.6 us against v2's 112.4 and 188.3: 5.9% and 3.8% more, where the four-warp route was 35%
and 51% more. What still costs is qkv at 32 rows (20.3 against 16.8, and the four-warp shape
does 17.1 there), down at 32 and 64 (1.18 and 1.16x), and o and down at 128 and 256
(1.13 to 1.29x). Four pieces is the finest cut the class order allows, and Stream-K cuts
finer.

The engine, `bench_invariant.py --part engine --weights-format int4`, 8 of the 32 layers,
the same random weights and static batches as batch_invariant.md, default / invariant /
invariant on the four-warp route:

| cache, batch | decode ms/step | prefill tok/s |
|---|---|---|
| bf16, 1 | 1.425 / 1.493 / 1.494 | 48,421 / 48,539 / 37,711 |
| bf16, 8 | 1.587 / 1.578 / 1.578 | 50,619 / 51,511 / 35,930 |
| bf16, 32 | 2.083 / 2.172 / 2.412 | 49,473 / 48,758 / 29,772 |
| bf16, 64 | 3.157 / 3.173 / 3.869 | 48,980 / 48,496 / 30,175 |
| fp8, 1 | 1.425 / 1.528 / 1.527 | 48,733 / 48,288 / 37,828 |
| fp8, 8 | 1.541 / 1.586 / 1.584 | 50,778 / 51,087 / 35,854 |
| fp8, 32 | 1.919 / 2.055 / 2.290 | 49,422 / 48,106 / 29,760 |
| fp8, 64 | 2.806 / 2.862 / 3.564 | 48,437 / 47,842 / 30,119 |

On the bf16 cache invariance cost 16 and 23% of a decode step at 32 and 64 slots on the
four-warp route and costs 4.3 and 0.5% now; prefill went from 22 to 40% fewer tokens per
second to within 2% of the default either way. At one slot both routes run the same shapes,
so the 5 to 7% there is the invariant attention and the one-row GEMMs of the table above.

## Not done

* Variant 3 at 128 to 256 rows on o and down, 1.13 to 1.29x of variant 2. Four class pieces
  is the finest cut that keeps the bits; a finer one needs more chains per element, which
  means other bits at every M and eight warps along K at decode.
* The fused epilogues of `hgemm` (bias, activation, residual, SwiGLU on interleaved gate/up
  columns). The int4 layer still runs `swiglu` and the residual adds as separate kernels.
* The tensor-bound regime (M >= 64): 16% of the roof and 0.8 to 0.96x of `hgemm` at M = 256.
  A larger tile with dequantized weights shared through shared memory across warps along M,
  and a deeper pipeline, is the obvious next step.
* The single-launch floor: at 4096² a third of the time is the launch. Programmatic
  dependent launch, which lets the next kernel's prologue overlap this one's tail, is where
  the rest of the M = 1 speedup on the small matrices is.
* Group sizes other than 128, fp16 scales, and a loader for GPTQ or AWQ checkpoints.
