# A whole decoder layer

Every kernel in this repo was benchmarked alone. This note runs them together as one
Llama-3-8B decoder layer, prefill and decode, against the same layer in PyTorch, and says
where the time goes. Source: `python/spark_kernels/layer.py` (the layer on our kernels, the
same layer in plain torch, the K/V cache), `src/kernels/rope.cu` (the one kernel written for
it), `scripts/bench_layer.py` (the timings and the breakdown), `tests/test_layer.py`.

## The layer

Llama-3-8B: hidden 4096, 32 query heads over 8 K/V heads of 128, a 14336-wide SwiGLU MLP,
RMSNorm with eps 1e-5, RoPE with theta 500000. bf16 tensors, fp32 math inside every kernel.

```
h    = add_rmsnorm_(delta, x)          x += delta (the previous layer's MLP output), then norm
qkv  = hgemm(h, W_qkv)                 [tokens, 6144]: q | k | v from one weight
q    = rope_append_(qkv, cos, sin, K, V, pos)   q rotated into [B, 32, S, 128];
                                       k rotated and v copied into the cache at pos..pos+S-1
o    = attention(q, K, V, causal)      GQA, the flash-decoding kernel when S = 1
attn = hgemm(o^T, W_o)                 o^T is the one torch copy left (heads back to tokens)
h2   = add_rmsnorm_(attn, x)           x += attn, then norm
gu   = hgemm(h2, W_gate_up)            [tokens, 28672]: gate | up from one weight
a    = swiglu(gu[:, :14336], gu[:, 14336:])   the two halves read in place
down = hgemm(a, W_down)                returned as the next layer's delta
```

What is fused where:

- The two residual adds are inside the norm kernels. A stack of layers runs each layer with
  the previous layer's MLP output still unadded, so `add_rmsnorm_` does `x += delta` and the
  norm in one pass, and the attention output's add goes into the second norm the same way.
  A layer therefore returns `(x, down)` and the model adds the last `down` once at the end
  (`finish`). With `delta=None` the first norm is a plain `rmsnorm`.
- q, k and v come from one [4096, 6144] weight and gate and up from one [4096, 28672]
  weight, so a decode step streams four matrices, not seven, and each once.
- RoPE on q and k, q's move to the head-major layout attention takes, and the K/V cache
  append are one kernel, `rope_append_`. In torch that is seven elementwise kernels for the
  rotation (the fp32 upcast, the two halves, the cat, two muls, an add, the bf16 cast), a
  transpose copy for q and two strided copies into the cache: 17.7 us of a 315 us decode step
  and 600 us of an 8.9 ms prefill, measured before the kernel existed. The kernel reads the
  qkv rows once and writes q and the two cache slots once: 1.7 us at decode, 44 us for a
  4096-token prefill (100 MB moved, mostly out of L2 since the GEMM just wrote it).
- SwiGLU reads the two halves of the fused gate|up output where they are. The elementwise
  kernel wanted contiguous inputs, and `gu[:, :14336]` is not one, so `swiglu` grew a
  strided entry (`swiglu_strided_*`, the same 16-byte vector kernel with a row stride) that
  the binding takes for 2-D inputs whose rows are contiguous but not adjacent.
- Not fused: the q/k/v column split (a view, free), the attention output's transpose back
  to [tokens, 4096] (a torch copy, 25 us at 4096 tokens, nothing at decode where [B, 32, 1,
  128] already is [B, 32, 128]), and the attention over a cache that is not full (below).

The K/V cache is [B, 8, capacity, 128] per layer, the layout the attention kernel reads. The
kernel takes a contiguous [B, H_kv, S_kv, D] tensor and has no head stride, so the filled
part of a cache with spare capacity is a strided slice, and the layer copies it out
contiguous before the kernel runs (`KVCache.kv()` says when). A cache whose capacity equals
its length after the append is read in place. The decode rows below are the in-place case
(the step fills the last slot of a cache of capacity L); the copied case is measured next
to it as the cost of not having a strided kernel.

## The RoPE + append kernel

`rope_append_bf16` in `src/kernels/rope.cu`, bench `bench_rope`, validated against a CPU
double-precision reference. One thread per 8 elements: a (token, q or k head) row is D/16
threads each rotating 8 elements of the first half against the 8 at +D/2 (two 16-byte loads,
two float4 loads of cos and sin, two 16-byte stores), a (token, v head) row is D/8 threads
copying 16 bytes. The rows of a token are consecutive in the qkv output, so a warp's loads
cover whole 128-byte segments of it. The tables are the rotate-half layout Llama uses (column
d pairs with d + D/2 and the second half of a row repeats the first), so only D/2 columns of
cos and sin are read per position.

| shape | time | moved | GB/s |
|---|---|---|---|
| 4096 tokens, 32/8 heads, D 128 | 25.8 us | 100 MB | 3,903 (the 50 MB input sits in L2) |
| 8192 tokens | 138 us | 201 MB | 1,458 |
| 1 token at position 4095 | 3.1 us | 25 KB | |
| 8 sequences, 1 token each | 3.2 us | 197 KB | |

GB/s counts the qkv rows read once and q, k and v written once. In the layer the kernel
follows the qkv GEMM, whose output is still in L2, so it runs at the 4096-token rate there
(44 us for the 4096-token prefill row, which the profiler puts at the end of a GEMM that
also left the residual stream and the norm weight in cache). A decode step's 3 us is the
launch: 25 KB of work.

## Parity

`tests/test_layer.py` runs ours and the torch layer on the same random weights (N(0, 1/sqrt(K))
projections, norms near 1) for prefill S in {1, 7, 128, 1000} and decode at cache lengths
{1, 500, 4096}, each at B in {1, 4}, with and without a pending delta, and with the cache
read in place and copied. Both are bf16 pipelines with different rounding points (our GEMMs
against cuBLAS, the fused norms, P in bf16 inside attention), so the check is against an
fp32 run of the torch layer on the same weights: our result must land within 1.5x of bf16
torch's own distance from it, and the two bf16 results must agree to 6e-2 absolute, 3e-2
relative. The relative errors against fp32 come out at 3 to 6e-3 for both. A prefill of 300
tokens followed by two decode steps is checked the same way, and the strided `swiglu` and
`rope_append_` have their own parity tests.

## Timing

`scripts/bench_layer.py` on the RTX 5090 (driver 595.58, CUDA 13.2, torch 2.14+cu130), one
layer, ms per step, the median over five rounds of 20 back-to-back steps between CUDA events
after the 300 ms clock ramp. Every step is the steady-state form with a pending delta. The
torch layer is `F.rms_norm`, `@` on the same fused weights, the same RoPE as fp32 torch ops,
`F.scaled_dot_product_attention(enable_gqa=True)` (FlashAttention-2 on every shape here),
`F.silu(g) * u`; "compiled" is `torch.compile(mode="max-autotune-no-cudagraphs")` for
prefill and `mode="reduce-overhead"` (CUDA graphs) for decode, with the weights, tables and
cache marked as static addresses. One layer's weights are 2 x (4096 x 6144 + 4096 x 4096 +
4096 x 28672 + 14336 x 4096) = 436 MB in bf16, more than the 96 MB L2 on their own, so no
rotation is needed to keep the decode weights out of cache; the K/V cache at 4K tokens is 16
MB and would fit, but 436 MB of weights pass through L2 between two reads of it.

### Prefill

B sequences of S tokens from an empty cache, causal attention.

| shape | ours | torch eager | torch compiled | vs eager | vs compiled | tokens/s, 32 layers |
|---|---|---|---|---|---|---|
| B=1, S=4096 | 8.43 ms | 9.59 | 8.93 | 1.14x | 1.06x | 15,190 |
| B=1, S=8192 | 18.11 ms | 20.73 | 18.80 | 1.15x | 1.04x | 14,140 |
| B=4, S=2048 | 16.41 ms | 18.80 | 16.86 | 1.15x | 1.03x | 15,610 |

Kernel time per stage of our step from the torch profiler (us, B=1, S=4096):

| stage | us | share | what it is |
|---|---|---|---|
| norm 1 | 38 | 0.5% | `add_rmsnorm` |
| qkv GEMM | 839 | 10.4% | 4096 x 6144 x 4096, 246 TFLOPS |
| RoPE + append | 44 | 0.5% | `rope_append` |
| attention | 596 | 7.4% | v4, causal, 231 TFLOPS, plus a 2 us combine |
| o transpose | 25 | 0.3% | torch copy |
| o GEMM | 546 | 6.8% | 4096 x 4096 x 4096, 252 TFLOPS |
| norm 2 | 64 | 0.8% | `add_rmsnorm` |
| gate/up GEMM | 3,784 | 46.9% | 4096 x 28672 x 4096, 254 TFLOPS |
| swiglu | 226 | 2.8% | strided, 352 MB at 1,560 GB/s |
| down GEMM | 1,914 | 23.7% | 4096 x 4096 x 14336, 251 TFLOPS |
| sum | 8,075 | | |

88% of a prefill layer is the four GEMMs, at 246 to 254 TFLOPS, which is where `hgemm` v6
runs alone (docs/design/hgemm.md). Attention is 7%; at 8192 tokens it is 13%, since it grows with
S^2 while the rest grows with S. Everything else is under 5%. The sum of kernel times is 350
us under the wall time here and 130 us under it at the two larger shapes; the profiled steps
are three iterations at a fresher clock than the hundred timed ones under the 600 W cap, so
that gap is clock, not launch gaps (the decode rows, memory-bound and clock-insensitive,
have wall and kernel sum equal).

Compiled torch is 5% behind at S=4096: it fuses the RoPE, the norms and the SwiGLU into a
few Triton kernels, which we match with `rope_append`, `add_rmsnorm` and `swiglu`, and its
GEMMs are cuBLAS (`max-autotune` tried the Triton templates and picked cuBLAS on every
shape), which v6 is ahead of by 3 to 11% on these sizes.

### Decode

One token per sequence against a cache of L - 1 tokens, appended in place. "graph" is our
forward captured once into a `torch.cuda.CUDAGraph` and replayed.

| shape | ours | ours, graph | torch eager | torch compiled | graph vs torch compiled | tokens/s, 32 layers |
|---|---|---|---|---|---|---|
| B=1, L=4096 | 0.298 ms | 0.297 | 0.352 | 0.326 | 1.10x | 105 |
| B=1, L=16384 | 0.328 ms | 0.326 | 0.371 | 0.344 | 1.06x | 96 |
| B=1, L=131072 | 0.604 ms | 0.603 | 0.682 | 0.650 | 1.08x | 52 |
| B=8, L=4096 | 0.370 ms | 0.369 | 0.417 | 0.376 | 1.02x | 678 |

Kernel time per stage (us):

| stage | B=1, L=4096 | B=1, L=16384 | B=1, L=131072 | B=8, L=4096 |
|---|---|---|---|---|
| norm 1 | 4.8 | 3.8 | 3.8 | 5.0 |
| qkv GEMM (50 MB) | 31.2 | 31.0 | 31.2 | 31.4 |
| RoPE + append | 1.7 | 1.6 | 1.7 | 1.8 |
| attention | 15.6 | 45.4 | 323.9 | 84.4 |
| o GEMM (34 MB) | 21.3 | 21.4 | 21.5 | 21.8 |
| norm 2 | 3.7 | 4.6 | 4.9 | 4.6 |
| gate/up GEMM (235 MB) | 144.9 | 145.1 | 144.9 | 145.9 |
| swiglu | 1.6 | 1.6 | 1.7 | 1.9 |
| down GEMM (117 MB) | 71.1 | 70.9 | 71.1 | 73.0 |
| sum | 296 | 325 | 605 | 370 |

The four GEMMs are 269 us of a 296 us step at 4K tokens, 91%, and they run at the
back-to-back rate the hgemm note measured for the decode kernel: 50 MB in 31.2 us is 1,612
GB/s, 235 MB in 145 us is 1,620, 117 MB in 71 us is 1,651, 34 MB in 21.3 us is 1,575. The
whole 436 MB layer takes 296 us, 1,473 GB/s counting only the weights, 96% of the 1,532 GB/s
`cudaMemcpy` rate. Attention is 5% at 4K tokens and 54% at 128K, where its 512 MB of cache is
more than the weights; the flash-decoding kernel streams it at 1,580 GB/s here. The norms,
the RoPE and the SwiGLU are 12 us together, 4%, and they are 4 to 5 us each because a
[1, 4096] row is a single block's work with a launch around it.

**The graph measurement.** The hgemm note's open item was the 2 us a lone decode launch
pays over the back-to-back rate (23.5 against 21.5 us at 32 MB) and CUDA graphs as the way
at it. Here the graph buys 1 to 2 us per layer, 0.3%: eager and graph are 0.298 and 0.297
ms at 4K tokens, and the kernel-time sum is 0.296. The reason is that the layer's ten
launches are already queued back to back: the host issues a step in 52 us of Python and
pybind (measured with no synchronization in the loop) and the GPU takes 297, so the queue
never drains and every kernel starts as the one before it ends, which is the back-to-back
rate. The 2 us is the cost of an empty queue, and an eager layer whose host is ahead does not
have one; with a `torch.cuda.synchronize()` after every step, so that each step starts from
an empty queue, the same loop takes 310 us. Before `rope_append` existed the
step had eighteen launches, eight of them torch elementwise kernels of 2 to 7 us, and the
graph was worth 9 us; the host was still ahead, but each of those launches paid its own
front-end and drain. A graph would matter with a slower host, a smaller batch of work per
launch, or a model where Python is between the kernels, which is what `reduce-overhead`
is for in torch: there it takes the torch layer from 0.352 to 0.326 ms.

**Graph safety.** The capture is torch's default global mode, in which a `cudaMalloc` or any
other capture-unsafe call inside a launch path fails the capture instead of being recorded.
Every kernel here is captured after three warm-up steps on a side stream, so the per-device
workspaces are already grown: `hgemm`'s decode kernel needs none at these widths (every
strip count is over 64, so K is never split), attention's flash-decoding kernel grows its
partials and per-head counters on the first call at each cache length, and the counters are
left at zero by every launch, which is what makes a replay the same launch as the capture.
Three replays reproduce the eager step bit for bit at every shape (`spark_graph_ok` in
`results/layer.json`). A graph is per cache length, because S_kv is a launch parameter of
the attention kernel and the write position is baked into the captured RoPE and append; a
serving loop would capture one graph per length bucket, or take a position from a device
tensor, neither of which is done here.

**A 32-layer step.** 32 x 0.297 ms is 9.5 ms per token at a 4K context, 105 tokens/s, for the
layers alone; the lm_head is another 1.05 GB of weight (128256 x 4096 in bf16), about 0.65
ms at the same rate, so a whole Llama-3-8B step lands near 10.2 ms, 98 tokens/s, with the
14 GB of layer weights streamed at 96% of the copy rate. At 128K tokens it is 19.3 ms and 52
tokens/s, and 16 GB of K/V cache is being read per step next to the 14 GB of weights. Eight
sequences at 4K cost 0.37 ms per layer, 25% more than one, for eight times the tokens: 678
tokens/s. Torch eager is 15 to 18% behind at every shape and compiled torch 2 to 10%.

**The cache-append cost.** The same step against a cache with 256 spare slots, where the
filled part is copied out contiguous before the attention kernel reads it:

| shape | in place | copied | the copy |
|---|---|---|---|
| B=1, L=4096 | 0.298 ms | 0.317 | +19 us (16 MB) |
| B=1, L=16384 | 0.328 ms | 0.387 | +59 us (64 MB) |
| B=1, L=131072 | 0.604 ms | 1.287 | +683 us (512 MB) |
| B=8, L=4096 | 0.370 ms | 0.516 | +146 us (128 MB) |

That is the cache read and written once more per step, at 1,500 GB/s, and at 128K tokens it
doubles the step. A decode loop grows its cache by one token a step, so without a fix every
step but the one that fills the buffer pays this. The fix is a K/V head stride in the
attention kernels (head `h` of batch `b` at `(b H_kv + h) x capacity x D` instead of
`x S_kv x D`), which is a pointer computation in each variant and the outer stride of
variant 4's tensor maps; it is not done here because the attention source is being changed
elsewhere at the same time.

## Caveats

The paged K/V cache, the varlen prefill and the 32-layer engine of
[serving.md](serving.md) remove the first three caveats below for a serving loop: the paged
decode kernel reads a cache in place at any length, the attention output is token-major so the
transpose copy is gone, and the positions, slots and lengths are device buffers, so one captured
graph serves every step. This layer is kept as it is for the one-layer measurements.

- No strided K/V in attention, so a growing cache pays the copy above; the headline decode
  rows are the in-place case.
- The attention output's transpose is a torch copy (25 to 80 us at prefill, none at decode).
- The layer takes `pos` and the cache length from the host: a captured graph is for one
  cache length and one write position.
- The parity test compares against an fp32 torch run, not against a Llama checkpoint; the
  weights are random with the right shapes and scales. RoPE is the plain theta 500000
  rotation without Llama 3.1's frequency scaling.
- The tokens/s figures are 32 copies of this layer and nothing else: no embedding, no final
  norm, no lm_head, no sampling, one stream, no scheduler.
