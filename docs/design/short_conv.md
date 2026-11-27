# Inkling 2 Mini's short convolutions and head norms (`short_conv`, `head_norm`)

Thinking Machines' Inkling 2 Mini runs a causal depthwise convolution of 4 taps in four places
per layer application: on k and v (sliding layers: static taps; full layers: static plus
per-token taps), on the attention output and on the MLP output (static). With 48 layer
applications per token that is 192 convolutions a token, each a pass over an activation, and
around every k and v one more pass for its per-head norm. Each is a few bytes per channel of
arithmetic and the rest is memory traffic, so the work here is: do it in one pass, carry the
state without extra copies, and fold the norm that follows into the same kernel.

Source: `src/kernels/short_conv.cu`. Ops: `short_conv`, `dynamic_short_conv`,
`short_conv_state`, `head_l2_clip`, `head_rms_norm` (python/spark_kernels/ops.py). Tests:
`tests/test_short_conv.py`. Bench: `scripts/bench_short_conv.py`.

## The math, from HF's modeling code

Ground truth is transformers 5.24.1, `models/inkling2_loop/modeling_inkling2_loop.py`. K = 4.
`xin` is x with the padded tokens zeroed (`apply_mask_to_padding_states` with `conv_mask`); the
residual uses the unmasked x.

**Static** (`Inkling2LoopShortConvolution`, attn/mlp out with C = 2304, k and v of the
sliding layers with C = 1024). The weight is `nn.Conv1d(C, C, 4, groups=C).weight`, [C, 1, 4],
fp32 (the module is cast to fp32). `F.conv1d` is a cross-correlation over the input padded by
three zeros on the left, so column 3 weights the current token and column 0 the token three
back:

    out[b, t, c] = (sum_{lag=0..3} w[c, 3 - lag] * xin[b, t - lag, c] + x[b, t, c]) / sqrt(5)

**Dynamic** (`Inkling2LoopDynamicShortConvolution`, k and v of the full layers, C = 1024, 8 kv
heads of 128 channels). `static_weight` is [4, C] with the row the backward distance (its
comment: `static_weight[0]` weights the current token). `weight_proj(h)` gives 8 x 4 taps per
token, `.view(B, T, 8, 1, 4)`, shared by the 128 channels of a head. HF builds the windows as
`context.unfold(-1, 4, 1).flip(-1)`: unfold gives `[x[t-3], x[t-2], x[t-1], x[t]]`, the flip
puts the current token at index 0, so index = lag on both sides:

    out[b, t, c] = (sum_lag (sw[lag, c] + taps[b, t, c / 128, lag]) * xin[b, t - lag, c]
                    + x[b, t, c]) / sqrt(9)

I checked both orders numerically against HF's modules (the formulas above, written as loops
over lag, match `Inkling2LoopShortConvolution` and `Inkling2LoopDynamicShortConvolution` to
0 and 2.4e-7 in fp32).

**Norms.** k: `Inkling2LoopL2Clip(22.627)`, `y * min(22.627 / (|y| + 1e-6), 1)`. v:
`Inkling2LoopUnweightedRMSNorm(1e-6)`, `y * rsqrt(mean(y^2) + 1e-6)`. Both per head of 128 and
both on the conv output after it was rounded to bf16 (the conv module returns its input dtype,
the norm reads that), so the fused epilogue rounds there too before it norms. q (and the
16-dim relative states, clip 8.0): `Inkling2LoopScaledL2Clip`, the clip after a learned
[H, D] scale; in the full layers `Inkling2LoopLogScaledL2Clip`, which rounds the clipped value
to bf16 and then multiplies by `tau = 1 + 0.1 * log(max((pos + 1) / 8192, 1))` and rounds
again. `head_l2_clip` does the same two roundings.

## The state

HF's cache (`LinearAttentionLayer.update_conv_state`) keeps `[B, C, 4]` fp32 per conv: the
last four inputs, column 3 the newest. Only three of them are ever read (the kernel has 4 taps,
one is the current token). Ours is `[B, 3, C]` in x's dtype: `state[b, j] = xin at position
j - 3`, `j = 2` the newest. The two are related by

    ours = hf[:, :, 1:].transpose(1, 2).to(x.dtype)

and the bf16 copy is exact, because the state only ever holds inputs, and the inputs are bf16
(cast to fp32 by HF; the mask multiplies by 0 or 1). The channel-contiguous layout is what the
kernel reads anyway, and bf16 with three entries is 6 bytes per channel per direction instead
of HF's 16. For one decode step of a layer's four convs (6,656 channels) at B = 64 that is
5.1 MB of state traffic instead of 13.6 MB.

Semantics: the kernel reads the state as the three inputs before t = 0 and, at the end,
overwrites it with the last three `xin` of the call. For T >= 3 those are rows of x; for
T < 3 they are a mix of the old state and x, as `cat([state, x])[-3:]` would be. So:

- a prefill split into chunks with the state carried gives the same bits as one prefill;
- a decode step is the same kernel at T = 1, updated in place;
- `state=None` means zeros before t = 0 and nothing written (HF's first prefill pads with
  zeros the same way).

Padding: a `[B, T]` mask (bool or integer, 0 = padding) zeroes those inputs for the conv and
for the state, and the residual keeps them, as HF's `conv_mask` does. With left padding the
first real token sees zeros before it, as from a fresh state; with right padding the state
ends on zeros, which is also what HF's cache holds. HF's `seq_idx` (several sequences
packed in one row, the dynamic conv cutting its window at a boundary) is not supported: pack
by batch row, or split the call at the boundary and start the next sequence from a zero state.

## The kernel

One thread owns 16 bytes of channels (8 bf16 or 4 fp32) and 4 consecutive tokens. It loads
its 4 taps per channel once (a float4 per channel from the [C, 4] conv weight, or four float4
rows of the [4, C] static weight), loads the three inputs before its first token (from x, or
from the state at t < 0) and the four tokens of its chunk (x, the mask byte, in the dynamic
case 8 or 16 bytes of taps per head, the same address for the 16 lanes of a head, so a
broadcast), then walks them with the three previous inputs in registers: four FMAs per
channel and one 16-byte store per token. A warp is 32 consecutive channel groups at the same
tokens, so every row access is a contiguous 512 bytes. The grid is (C / 256 channel strips,
T / 16, B) with 32 x 4 threads a block (bf16).

The chunk length is the knob. The three inputs before a chunk are re-read: they are the last
three rows of the chunk before, in the same block for three chunks out of four, so they come
from L1 or L2, but they are requests. Long chunks re-read little but are few threads, each a
chain of steps. At the 8192-token prefill, L2 flushed, C = 2304 / C = 1024 with the clip /
C = 1024 dynamic with the RMS norm (us, from two sweeps, the box shared by other jobs, so
+- 5%):

| tokens a thread | 1 | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|
| attn/mlp out | 82.4 | 65.5 | 62.0 | 66.6 | 78.8 | 96.2 |
| k + clip | 45.1 | 36.9 | 38.9 | 41.0 | 39.9 | 69.6 |
| v dynamic + RMS | 45.1 | 45.1 | 46.1 | 44.0 | 49.2 | 89.1 |

4 is the best for the C = 2304 conv and within the noise of the best for the other two, so 4
it is. Loading the whole chunk before the first step (rather than each token in its step)
took the two epilogue rows from 37.9 / 43.0 to 35.8 / 41.0 us: the epilogue's shuffles kept
the compiler from hoisting the next token's load above them. Wider blocks (64 x 2, 128 x 1,
128 x 2, 64 x 4, 32 x 8 threads) were no better than 32 x 4.

The epilogue is a reduction over a head: 128 channels are 16 lanes in bf16 (32 in fp32), so
the sum of squares is four (five) `__shfl_xor_sync`. Every lane of a warp walks the same
tokens, so the shuffles stay convergent; lanes past C still take part with zeros.

The in-place state: the thread of chunk 0 is the only one that reads the state and the only
one that writes it, and it reads before it writes, so there is no race between blocks. When
T <= 4 it walked every token and holds the new state in its window registers already; for a
longer call it re-reads x[T - 3 .. T - 1] (never the old state, since T >= 3). out must not
alias x (the halo of the next chunk would read outputs).

`head_norm` is the same reduction as its own kernel, D / 8 lanes a row (16 for q's 128, two for
the 16-dim relative states), so it covers q's scaled clip with the log scaling (positions
int32 or int64, one per token) and serves as the separate op in the fused-vs-separate
comparison.

## Tests

`tests/test_short_conv.py`, references copied from HF (the conv functions, the cache's
`update_conv_state`, `causal_conv1d_update`, the norm modules), run on the CPU in fp32:

- prefill against HF for both convs, fp32 and bf16, B up to 3, T from 1 to 1000, C 1024 and
  2304, and the state against HF's cache (`hf[:, :, 1:].transpose(1, 2)`, exact);
- a prefill in two chunks with the state carried, split at 1, 2, 3, 4, 5, 15, 16, 17, 40 and
  63 of 64 tokens, starting from a nonzero state: bit-identical to one prefill, same final
  state;
- prefill then six decode steps at B = 1, 8, 64: bit-identical to the prefill, and against HF's
  decode path (`causal_conv1d_update` for the static conv) step by step, state included;
- the padding mask (left, right and scattered) against HF's `conv_mask`, bool and int masks;
- bf16 per-token taps (what the weight_proj GEMM gives) against HF's `.float()` of them;
- both epilogues fused into both convs against HF's conv then norm, on inputs scaled so the
  head norms straddle 22.627, and against our conv then the standalone op;
- the clip on vectors of norm 22.627 x (1 +- 1e-1 .. 1e-4) and exactly 22.627 (shorter vectors
  come back unchanged);
- `head_l2_clip` against `Inkling2LoopScaledL2Clip` and `Inkling2LoopLogScaledL2Clip` for
  D = 128 (clip 22.627) and D = 16 (clip 8.0), positions around 8192 and up to 100K, int32 and
  int64; `head_rms_norm` and the in-place form.

## Numbers

RTX PRO 6000 Blackwell Server Edition (sm_120, 188 SMs, 128 MB L2), CUDA 13.2, driver 580.196,
600 W, `scripts/bench_short_conv.py` under the GPU lock, rows in `results/short_conv.jsonl`.
A 256 MB device-to-device copy ran at 1,435 GB/s in the same run. All bf16, B = 1 for prefill.
113 tests pass on the same card.

**Prefill, 8192 tokens, L2 overwritten before every call** (what a cold read from DRAM gives;
"copy" is `y.copy_(x)` of a tensor the size of x, the floor at that size; GB/s counts x, out,
the state both ways, the weights and the taps):

| conv | sk | GB/s | copy of x | sk conv + separate op | torch (HF code) |
|---|---|---|---|---|---|
| attn/mlp out, C = 2304 | 63.5 us | 1,190 | 57.3 us | | 805 us |
| k sliding, C = 1024, clip | 35.8 us | 937 | 27.7 us | 43.0 us | 412 us |
| v full, C = 1024, dynamic, RMS | 41.0 us | 833 | 27.7 us | 47.1 us | 1,093 us |
| q full, 24 x 128, scale + clip + log scale | 73.7 us | 1,366 | 71.7 us | | 833 us |

The C = 2304 conv is at 90% of the same-size copy and q's head norm at 97%. The two C = 1024
rows are at 77% and 67% of it; I have not found what the rest is (the chunk sweep above and the
block shapes did not move it, and there is no ncu on this box). Against HF's code the
convs are 13x (static), 11x (k with the clip) and 27x (v: HF's unfold, flip, reshape and the
sum with the broadcast taps materialize several [B, T, 8, 128, 4] fp32 tensors) faster, q 11x.
The fused epilogue saves 7.2 and 6.1 us over conv then the separate op (17% and 13%): one write
and one read of the [T, 1024] output.

L2-warm (the call before left x and out in the 128 MB L2; a real prefill's conv reads what the
GEMM before it just wrote, so somewhere between): 41.4 / 26.8 / 26.9 / 39.1 us for the four
rows, the same-size copy 22.3 / 14.3 / 15.0 / 24.9 us. There the conv is further from the copy
(about 55%): the re-read halo and the 7 loads for 4 tokens cost L1 and L2 requests that DRAM
time used to hide.

**Decode, T = 1, state updated in place, CUDA graph of 50 calls** (GPU time a call, L2-warm:
x, the state and the weights stay in L2 across the calls, which a real step would not give
the state):

| conv | B = 1 | B = 8 | B = 64 | torch (HF code), B = 1 / 8 / 64 |
|---|---|---|---|---|
| attn/mlp out, C = 2304 | 2.44 us | 2.61 us | 3.62 us | 10.7 / 10.8 / 15.3 us |
| k sliding, clip (fused) | 2.72 us | 2.84 us | 3.64 us | 20.6 / 20.8 / 22.9 us |
| k sliding, conv + separate op | 3.75 us | 4.09 us | 4.99 us | |
| v full, dynamic, RMS (fused) | 2.55 us | 2.61 us | 2.72 us | 26.1 / 27.3 / 31.0 us |
| v full, conv + separate op | 3.65 us | 3.87 us | 4.08 us | |
| q full, scale + clip + log scale | 2.08 us | 2.23 us | 2.48 us | 23.0 / 23.5 / 24.5 us |

Decode is a launch and one round trip: 74 KB at B = 1 for the C = 2304 conv (of which 37 KB is
the fp32 weights), 2.4 MB at B = 64, so 30 GB/s at B = 1 and 660 GB/s at B = 64. The fused
epilogue saves one launch, 1.0 to 1.4 us a call. A token runs 192 of these convs and 96 head
norms (q and the relative states at every one of the 48 layer applications), so at B = 1 the
launch is the cost, and the next step is fusing them into the GEMMs around them (below).

## Not done

- `seq_idx` (packed sequences in one row) is not supported; the padding mask is.
- The state has no slot indirection: row b of the state is sequence b of the call. A serving
  engine with a slot per sequence would want `state_slots[b]`, which is one more index load in
  the thread of chunk 0.
- No CUDA graph or fusion with the GEMM before the conv (the conv on o_proj's output or the k/v
  projections could be the GEMM's epilogue, which would remove the conv's read of x).
- Only measured on sm_120. It builds for sm_121 (the DGX Spark); not run there.
