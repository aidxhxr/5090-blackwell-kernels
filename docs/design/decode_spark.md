# Decode grids on the GB10: bytes in flight, split-K and the 24 MB L2

The decode shapes (M <= 16 rows of activations against a weight read once) of `hgemm`'s
decode path (`src/kernels/hgemm_decode.cu`), `w4gemm` and `w1gemm`, re-derived for 48 SMs at
273 GB/s with a 24 MB L2 from the constants that were measured on 170 SMs at 1.53 TB/s with
96 MB. Source: `src/kernels/launch_config.cuh` (the rules), `launch_config.cu` (the device
query), the call sites in the three kernels. Bench: `bench_decode_spark`. Python:
`spark_kernels.ops.device_shape()`, `ops.split_k_for(...)`, `ops.fits_l2(...)`, test
`tests/test_launch_config.py`.

**Not measured.** This note was written before a GB10 was in reach and on a machine without
nvcc. Nothing in it has run, on either card. Every number below is a byte count, a count of
CTAs, or a division by 273 GB/s; none is a time. The one physical quantity the whole page
turns on, the loaded DRAM latency of the GB10's LPDDR5X behind its unified fabric, has not
been published by anyone, so it is a variable here and the first thing to measure.

## What a decode kernel is on either card

At M = 16, 16 x 4096 x 4096 is 32 MB of bf16 weights for 0.54 GFLOP. On the 5090 that is
21 us at the copy roof and 2 us of tensor work; on the GB10 it is 123 us at spec and 2.5 us of
tensor work at the 213 TFLOPS others have measured. The kernel's only job is to keep enough
bytes outstanding that DRAM never idles, and then to not pay for anything else: not a second
wave of CTAs that runs alone, not a fixup across a K split, not a memset in front.

Three things were constants in the kernels because the 5090 made them so:

| constant | where | what it stood for |
|---|---|---|
| `kMinCtas = 64` | `hgemm_decode.cu` | CTAs with 24 KB of B in flight each that stream at the 5090's roof (hgemm.md, "Bytes in flight": 32 CTAs reach 1,227 GB/s, 64 reach 1,440, 128 the same) |
| `kMinBlocks = 128` | `w4gemm.cu` | blocks the swept small-M grids were brought to on the 5090 (w4gemm.md) |
| `4 * num_sms()` strips | `w4gemm.cu` `pick()` | "wide N": one warp per strip already streams, no split |

The first two were SM-independent numbers that only make sense at 1.5 MB in flight; the
third already read the SM count but only moves with `SPARK_FORCE_SMS` now. `w1gemm` had no
such constant: its grid is N / 16 blocks and K is split inside the block.

## Little's law, with the latency left open

A memory stream delivers bandwidth B only if the bytes outstanding across the device are at
least

    bytes in flight = B x L,    L the latency a load sees under load.

On the 5090 this was measured, not computed: 64 CTAs x 24 KB = 1,572,864 B stream at the roof
and 32 do not, which at the 1,792 GB/s spec is L = 878 ns (1.03 us at the measured 1.53 TB/s
roof, queueing included). Per SM that is 9,252 B.

For the GB10 the formula with 273 GB/s and L unknown:

| L (ns) | bytes in flight | per SM (48) | CTAs at 24 KB (hgemm_decode 16x64x64) | blocks at 12 KB (w4gemm) | blocks at 8 KB (w1gemm) |
|---|---|---|---|---|---|
| 400 | 109,200 (107 KB) | 2,275 | 5 | 9 | 14 |
| 600 | 163,800 (160 KB) | 3,413 | 7 | 14 | 20 |
| 800 | 218,400 (213 KB) | 4,550 | 9 | 18 | 27 |
| 1,000 | 273,000 (267 KB) | 5,688 | 12 | 23 | 34 |
| 5090, measured | 1,572,864 (1.5 MB) | 9,252 | 64 | 128 | 192 |

`launch_config.cuh` takes 800 ns (`kGB10LoadedLatencyNs`) as the working assumption, so
`kBytesInFlightPerSM` is 4,550 B. Two things follow if the assumption is anywhere near right:

* The GB10 is far easier to keep fed than the 5090 in bytes-in-flight terms: one 24 KB CTA
  per SM is five times what Little asks for. Pipeline depth is never the question there.
* The question that replaces it is whether 9 CTAs on 48 SMs really stream 273 GB/s, or
  whether one CTA per SM is needed for the load issue rate. On the 5090 one CTA sustained
  38 GB/s (24 KB per 630 ns); the GB10 needs 5.7 GB/s per SM. If the LPDDR5X controller
  behaves like a DRAM, 9 CTAs are enough. If the unified fabric adds latency under load
  the table's rows move down and the CTA count up. `bench_decode_spark --sweep-split` times
  every legal split so the answer falls out as the split that reaches the roof.

The latency itself: a pointer chase over a buffer past the L2 with one load in flight
(`bench_bandwidth`-style, not written yet) gives the unloaded number; the loaded one is what
the sweep measures indirectly as the bytes in flight at which the roof is reached.

## The split-K rule

`split_k_for(weight_bytes, tiles, max_split, per_cta_in_flight, slots, device)`:

1. **Bytes in flight.** `need = cdiv(device.bytes_in_flight, per_cta_in_flight)` CTAs;
   `split = clamp(cdiv(need, tiles), 1, max_split)`. On 170 SMs at 24 KB per CTA, `need` is
   64 and the split is `cdiv(64, strips)`: the old `kMinCtas` line. At 12 KB it is 128: the
   old `kMinBlocks`. On 48 SMs at 800 ns it is 9 and 18.
2. **Waves**, only when `need > slots` (the bandwidth rule itself asks for more than one wave
   of resident CTAs): then a slightly larger split whose last wave is at least 7/8 full is
   taken, up to twice the split. A grid that is wide because N is wide is left alone; the
   tile choice in `hgemm_decode` already keeps those under one wave. On 170 SMs `need` is 64
   or 128 against at least 170 slots, so this clause never fires and the 5090 launches are
   bit for bit the old ones. On 48 SMs at 800 ns it does not fire either.
3. **Forced**: `decode_force_split_k(s)` or `SPARK_DECODE_SPLIT=s` replaces the rule,
   clamped to the kernel's range. The sweep uses it.

What changes on the GB10 under the rule, per Llama-3-8B shape at M <= 16 (hgemm's 16x64x64
tile, 64-column strips; w4gemm's 16-column strips):

| projection | K x N | strips (hgemm) | 5090 split | GB10 split (800 ns) | tiles (w4gemm) | 5090 split | GB10 split |
|---|---|---|---|---|---|---|---|
| q|k|v | 4096 x 6144 | 96 | 1 | 1 | 384 | 1 | 1 |
| o | 4096 x 4096 | 64 | 1 | 1 | 256 | 1 | 1 |
| gate|up | 4096 x 28672 | 448 | 1 | 1 | 1,792 (wide) | 1 | 1 |
| down | 14336 x 4096 | 64 | 1 | 1 | 256 | 1 | 1 |
| k or v alone | 4096 x 1024 | 16 | 4 | 1 | 64 | 2 | 1 |
| expert up | 5120 x 2304 | 36 | 2 | 1 | 144 | 1 | 1 |
| expert down | 2304 x 5120 | 80 | 1 | 1 | 320 | 1 | 1 |

So the rule's only visible effect on the Spark is to stop splitting the narrow shapes, which
saves the flat fixup cost (2 us on the 5090: atomics, fence, counter, last arriver; unknown
here) on weights whose whole stream is 7 to 30 us. If the sweep shows that 16 or 36 CTAs on
48 SMs do not reach the roof, the fix is `kGB10LoadedLatencyNs`, not the kernel.

## 48 is 2^4 x 3

The README's "170 is 2 x 5 x 17" section says no power-of-two tile grid divides into whole
waves on the 5090. 48 has the same property one factor cheaper: 2^k mod 48 is 16 or 32 for
every k >= 4, so a power-of-two CTA count always leaves one third or two thirds of the last
wave empty. Whole waves need a multiple of 3 in the tile count.

| CTAs | waves at 1 per SM (48) | at 2 per SM (96) | at 3 per SM (144) | at 4 per SM (192) |
|---|---|---|---|---|
| 36 (expert up, 64-col strips) | 0.75 | 0.375 | 0.25 | 0.19 |
| 64 (o, down; k/v at 16 cols) | 1.33 | 0.67 | 0.44 | 0.33 |
| 80 (expert down) | 1.67 | 0.83 | 0.56 | 0.42 |
| 96 (q|k|v) | **2.00** | **1.00** | 0.67 | 0.50 |
| 128 (o at 32-col strips) | 2.67 | 1.33 | 0.89 | 0.67 |
| 144 (expert up at 16 cols) | **3.00** | 1.50 | **1.00** | 0.75 |
| 192 | **4.00** | **2.00** | 1.33 | **1.00** |
| 256 (o at 16 cols) | 5.33 | 2.67 | 1.78 | 1.33 |
| 384 (q|k|v at 16 cols) | **8.00** | **4.00** | 2.67 | **2.00** |
| 448 (gate|up, 64-col strips) | 9.33 | 4.67 | 3.11 | 2.33 |
| 896 (gate|up at 32 cols) | 18.67 | 9.33 | 6.22 | 4.67 |

`hgemm_decode` keeps the rule that a tile configuration is taken only if its strips fit one
wave of resident CTAs, and the slot counts now come from the device: 96 slots for the
16x64x64 tile (2 per SM) means `o` and `down` (64 strips) run 0.67 of a wave with 32 SMs
holding one CTA and 16 holding two, and gate|up (448) falls through to the 32-column tile
with 192 slots and 4.67 waves. Whether a 0.67-wave launch streams at the roof is the same
question as the split rule's. If it does not, the next thing to try is not a new tile but
`SPARK_FORCE_SMS`-style emulation in reverse: pick BN so the strips land on a multiple of 48.
For N = 4096 that is BN = 32 (128 strips, 2.67 waves at 1 per SM, 0.67 at 4 per SM) or a
3-way K split of the 64 strips (192 CTAs, exactly one wave of 4 per SM, and a fixup).

## The 24 MB L2 and the second call

A decode step runs every layer's weights once. On the 5090 the 96 MB L2 holds a whole
4096 x 4096 bf16 weight (32 MB) with room to spare, but the next layer's weight evicts it
before the next step, so the bench rotates copies past L2 and that is the honest number. On
the GB10 the L2 is 24 MB, and `fits_l2` asks for three quarters of it (18 MB; the share
`hgemm_streamk.cuh` uses for the GEMM operands) so the activations, output and the other
operands that pass through do not evict the weight:

| weight | bf16 | int4 + scales | 1 bit + scales |
|---|---|---|---|
| 4096 x 4096 (o) | 33.55 MB, does not fit | 8.65 MB, fits | 2.36 MB, fits |
| 4096 x 6144 (q|k|v) | 50.33 MB, no | 12.98 MB, fits | 3.54 MB, fits |
| 4096 x 28672 (gate|up) | 234.9 MB, no | 60.56 MB, no | 16.52 MB, fits |
| 14336 x 4096 (down) | 234.9 MB, no | 59.64 MB, no | 15.60 MB, fits |
| 5120 x 2304 (expert up) | 23.59 MB, no | 6.08 MB, fits | 1.66 MB, fits |
| 2304 x 5120 (expert down) | 23.59 MB, no | 6.08 MB, fits | 1.66 MB, fits |
| floor at 273 GB/s, 4096 x 4096 | 123 us | 32 us | 8.6 us |

Fitting is necessary, not sufficient. Between two decode steps of a 32-layer model the L2
sees 31 other layers' weights, so a single weight is L2-resident across steps only if the
whole step's weights fit, which none of these formats manage for Llama-3-8B (Llama-3-8B's 32
layers are 3.6 GB at int4 and 0.98 GB at 1 bit). What *can* be resident is within a step:
the two matrices a fused projection reads in the same layer, or the 6 active experts of a
DeepSeek V4.1 Flash layer at 1 bit (6 x 3 x 1.66 MB = 30 MB, which does not fit either;
three experts' 15 MB do). The realistic L2 win on the GB10 is therefore the second call of the *same*
weight in a step, which is what `--l2-resident` measures: the same copy twice back to back,
events around the second call only, against the rotated cold time. Expect the bf16 32 MB
weight to show no gain (it cannot be resident), and the int4 and 1-bit weights to show
whatever the L2 gives over LPDDR5X; that ratio is the number that says whether a tile order
that keeps a layer's weight in L2 for a speculative-decoding verify pass is worth writing.

A 32 MB bf16 weight that does not fit should not be tiled to pretend otherwise: the second
decode call of a step is the next layer, a different weight, and tiling the first weight so
that half of it is in L2 when the kernel ends only helps a caller that re-reads the same
weight immediately, which the engine never does at bf16. The format that fits is the one
that could be resident, and the design choice is the format, not the tile.

## What to run first on the Spark

1. `bench_bandwidth` for the copy roof, then `bench_decode_spark --roof=<that>`: the
   correctness column for every shape and format (the first time any of these kernels run
   on sm_121), the GB/s against spec, and the plan lines on stderr.
2. `bench_decode_spark --sweep-split --format=bf16 --m=16 --n=1024 --k=4096`: the 16-strip
   shape where the rule changed its mind (split 1 instead of 4). If split 1 is at the roof,
   800 ns was conservative or right; if split 4 or 8 wins, read the latency off the split
   that first reaches the roof (bytes in flight = split x 16 x 24 KB) and set
   `kGB10LoadedLatencyNs` from it.
3. The same sweep on `--format=int4 --n=1024`, and on the expert shapes.
4. `bench_decode_spark --l2-resident`: the second-call ratio per format.
5. `SPARK_FORCE_SMS=48 bench_decode_spark` on the 5090 before any of this, for the
   correctness column alone: it runs the 48-SM grids on a card that has 170 SMs, so the
   times are meaningless and the ok column is the point.

## Results

Not measured. Columns to be filled by `bench_decode_spark` on the GB10; the floor is the
weight's bytes at 273 GB/s.

| shape (M x N x K) | format | bytes (MB) | floor at 273 GB/s (us) | measured (us) | GB/s | % of spec | split chosen | best split (sweep) | 2nd call (us) |
|---|---|---|---|---|---|---|---|---|---|
| 1 x 6144 x 4096 | bf16 | 50.33 | 184 | not measured | | | 1 | | |
| 1 x 6144 x 4096 | int4 | 12.98 | 48 | not measured | | | 1 | | |
| 1 x 6144 x 4096 | 1 bit | 3.54 | 13 | not measured | | | in-block | | |
| 1 x 4096 x 4096 | bf16 | 33.55 | 123 | not measured | | | 1 | | |
| 1 x 4096 x 4096 | int4 | 8.65 | 32 | not measured | | | 1 | | |
| 1 x 4096 x 4096 | 1 bit | 2.36 | 8.6 | not measured | | | in-block | | |
| 1 x 28672 x 4096 | bf16 | 234.9 | 860 | not measured | | | 1 | | |
| 1 x 28672 x 4096 | int4 | 60.56 | 222 | not measured | | | 1 | | |
| 1 x 28672 x 4096 | 1 bit | 16.52 | 61 | not measured | | | in-block | | |
| 1 x 4096 x 14336 | bf16 | 234.9 | 860 | not measured | | | 1 | | |
| 1 x 4096 x 14336 | int4 | 59.64 | 218 | not measured | | | 1 | | |
| 1 x 4096 x 14336 | 1 bit | 15.60 | 57 | not measured | | | in-block | | |
| 1 x 2304 x 5120 | bf16 | 23.59 | 86 | not measured | | | 1 | | |
| 1 x 2304 x 5120 | int4 | 6.08 | 22 | not measured | | | 1 | | |
| 1 x 2304 x 5120 | 1 bit | 1.66 | 6.1 | not measured | | | in-block | | |
| 1 x 5120 x 2304 | bf16 | 23.59 | 86 | not measured | | | 1 | | |
| 1 x 5120 x 2304 | int4 | 6.08 | 22 | not measured | | | 1 | | |
| 1 x 5120 x 2304 | 1 bit | 1.66 | 6.1 | not measured | | | in-block | | |
| 16 x (each of the above) | all | same | same | not measured | | | same | | |

Per decode step of Llama-3-8B (32 layers' projections only, no lm_head, no attention, no
cache): 13.96 GB at bf16, 51 ms at 273 GB/s; 3.60 GB at int4 with scales, 13.2 ms; 0.98 GB
at 1 bit with scales, 3.6 ms. Those are the floors the step can be judged against once it
runs; the kernels' share of the gap is what this bench will show.
