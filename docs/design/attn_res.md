# Block attention residuals and loop message gates (`attn_res`, `message_gates`)

Inkling 2 Mini (HF `thinkingmachines/Inkling-2-Mini`, transformers 5.24.1
`modeling_inkling2_loop.py`) replaces the plain residual stream with two small mixing ops,
and this note is about both. Neither is big. What makes them matter is how often they run:
one decoded token goes through 96 attention residuals, 24 message inputs and 96 message gates.
Source: `src/kernels/attn_res.cu`, wrappers `sk.attn_res`, `sk.message_gates`, tests
`tests/test_attn_res.py`, bench `scripts/bench_attn_res.py`.

## The math

**Block attention residual** (`Inkling2LoopBlockAttentionResidual`, after Kimi's attention
residuals, arXiv 2603.15031). Every 6 layers form a block. For each token the sources are the
nb block states `b_0 .. b_{nb-1}` (each the input of a block-boundary layer) plus the partial
state `p` (the running residual of the current block). With the unweighted RMSNorm
`n(x) = x * rsqrt(mean(x^2) + 1e-6)` and nq learned fp32 queries `q_i` of width H = 2304:

    w_i = softmax_s( q_i . n(x_s) / sqrt(H) )       over the nb + 1 sources x_s
    out_i = sum_s w_i[s] * n(x_s)

fp32 throughout (HF keeps `attention_input`, `mlp_input`, `final_input` and `message_input` in
fp32 through `_keep_in_fp32_modules_strict`). nq is 1 for most inputs, 3 for a full-attention
layer's attention input (one mix each for q, k and v), 4 for `message_input`
(`Inkling2LoopMessageAttentionResidual`: the q, k, v and MLP messages of a loop layer, read from
the previous pass's block states).

**Message gate** (`Inkling2LoopMessageGate`), in the loop layers' passes 2 and 3:

    z   = Wd [n(h); n(m)]                 Wd [64, 4608] bf16, z rounded to bf16 (HF's down_proj)
    out = (1 - tanh(Wf z)) * h + tanh(Wg z) * m     Wf, Wg [2304, 64], fp32 in HF

h is the attention residual's output for that input, m the message. HF upcasts Wf and Wg to
fp32 (`forget_gate_proj`, `message_gate_proj` are in `_keep_in_fp32_modules_strict`), but the
checkpoint stores them in bf16 (`controllers.*.forget_weight`, `gain_weight`: BF16 [2304, 64]
in the safetensors header), so every fp32 value is a bf16 value. The kernels take either; bf16
gives the same result from two thirds of the gate's bytes (1.18 MB instead of 1.77 MB).

## How many sources the model reaches

From the forward (`Inkling2LoopTextModel.forward` and the four decoder layer classes): block
boundaries are layers 0, 6, 12 and 18. At a boundary the layer's input joins the block states
before its MLP input (inside the layer) and for the layers after it (in the model loop), and
the partial state restarts from the attention output. Passes 2 and 3 of the loop restart from
the prelude's block states. Counting block states nb (sources = nb + 1):

| layers | attention input nb | MLP input nb |
|--------|-------------------:|-------------:|
| 0 | (none) | 1 |
| 1-5 | 1 | 1 |
| 6, each pass | 1 | 2 |
| 7-11, each pass | 2 | 2 |
| 12, each pass | 2 | 3 |
| 13-17, each pass | 3 | 3 |
| 18 | 3 | 4 |
| 19-23 | 4 | 4 |
| `final_input` | | 4 |
| `message_input`, passes 2 and 3 | 1 (the previous pass's layer-12 input), with that pass's output as the partial | |

So the largest nb is **4, five sources**, at layers 18-23 and the final input. The kernel takes
up to 8 sources (nb <= 7). The partial state is bf16 for attention inputs and fp32 for MLP inputs
(HF adds the attention output to the block residual in fp32 before `mlp_input`); the block
states are bf16, the model's dtype.

## Attention residual kernel

One block per token, `H / 8` threads (288 for Inkling), each owning 8 consecutive elements of
every source in registers. The trick is that `n(x_s) = r_s x_s` with a scalar `r_s`, so the
logit is `r_s (q_i . x_s) / sqrt(H)`: the sums of squares and the raw dots `q_i . x_s` come out
of one block reduction (warp shuffles, then a cross-warp sum in shared memory), the softmax is
done redundantly by every thread on the reduced values, and the output is one more pass over
the registers with the weights `w_i[s] r_s`. Queries go four per reduction, and every source is
read once per token for all of them, except when a few tokens bring many queries: with fewer
tokens than two blocks per SM, each chunk of four queries gets its own block, which re-reads the
token's sources from L2 (one block streaming 48 queries took 29 us at one token, the split takes
4.8). The source count is a template parameter (1..8), so nothing is guarded at run time;
Inkling's shapes compile with no spills.

The twelve `message_input` calls of a loop pass read the same two sources (the previous pass's
layer-12 input and output) with different queries, so they can run as one call with nq = 48;
the bench measures both.

## Message gate kernels

Each gate reads 0.59 MB of Wd and 1.18 MB (fp32) or 0.59 MB (bf16) of Wf and Wg, against 9 KB
of h and of m per token: at decode it is a weight stream with a rank-64 bottleneck in the
middle, which needs all of z before any output column can be computed.

**Decode (T <= 64, the default up to 16 tokens): one cooperative kernel.** Up to three gates per launch: a loop layer's q, k
and v gates read different weights and messages but are independent, so they share the grid
(in sliding layers they also share h). Three phases with two grid syncs:

0. zero the accumulators;
1. a block per (gate, 64 columns of [h; m]): `Wd[:, slice] . x[t, slice]` for every token with
   the raw activations, plus the slice's sums of squares, added into z_h, z_m [T, 64] and the
   norms with fp32 atomics;
2. a block per (gate, 32 output columns): `z = bf16(r_h z_h + r_m z_m)`, then its 32 rows of Wf
   and Wg against z, the tanh and the mix.

Each block's first Wd slice (registers) and first Wf/Wg slice (shared memory) are loaded before
the first grid sync, since the weights depend on nothing, so their DRAM latency overlaps the
zeroing and the syncs. Two differences from HF: the normalized activations are not rounded to
bf16 before the Wd dot (the norm is factored out of it; HF's bf16 down_proj rounds them), and
the atomics make the last bits of z depend on the order the blocks land, so a z value next to a
bf16 rounding boundary can round either way between runs.

**Prefill: three steps.** `message_gate_prep` writes `[bf16(n(h)); bf16(n(m))]` [T, 4608]
(exactly HF's down_proj input), `hgemm` multiplies by Wd^T (the repo's bf16 GEMM, bf16 out,
which is HF's rounding of z), and `message_gate_up` does the two rank-64 up-projections on 64
tokens x 64 columns tiles (each thread a 4 x 4 tile of Wf z and of Wg z, fp32 FMAs, operands
staged k-major in shared memory), the tanh and the mix. The prep and up kernels take all three
q/k/v gates in one launch; the GEMM runs once per gate (`grouped_gemm` wants N a multiple of
128; here N = 64).

## Numbers

Measured on the RTX PRO 6000 (sm_120, 188 SMs) with `scripts/bench_attn_res.py` under the GPU
lock; rows in `results/attn_res.json`. Every op runs 64 times (8 at 4096 tokens) inside one
CUDA graph, cycling over enough copies of its inputs and weights to read about three L2s of
cold memory, and the time is per call. "torch" is the HF code's own torch ops for the same call
(copied into the script), replayed the same way: no Python overhead, so it is a lower bound on
HF eager. The reference bandwidth is a 1 GiB device-to-device copy, 1453 GB/s.

### Attention residual (us per call)

| sources | nq | partial | T = 1 | T = 8 | T = 64 | T = 4096 | torch, T = 1 | torch, T = 4096 | GB/s at 4096 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 1 | bf16 | 2.3 | 2.7 | 2.9 | 29.0 | 15.7 | 295 | 2606 |
| 5 | 1 | bf16 | 3.7 | 4.0 | 4.1 | 85.8 | 16.7 | 1211 | 1541 |
| 2 | 1 | f32 | 2.5 | 2.8 | 2.9 | 43.9 | 15.6 | 320 | 2147 |
| 3 | 1 | f32 | 2.8 | 3.0 | 3.2 | 56.7 | 15.9 | 589 | 1996 |
| 4 | 1 | f32 | 3.2 | 3.4 | 3.6 | 80.3 | 16.3 | 855 | 1645 |
| 5 | 1 | f32 | 3.7 | 4.0 | 4.2 | 106.2 | 16.5 | 1114 | 1423 |
| 2 | 3 | bf16 | 3.4 | 4.1 | 4.4 | 105.1 | 19.4 | 433 | 1436 |
| 3 | 3 | bf16 | 3.9 | 4.5 | 4.7 | 118.1 | 19.6 | 751 | 1438 |
| 4 | 3 | bf16 | 4.6 | 5.1 | 5.4 | 131.2 | 19.8 | 1022 | 1439 |
| 5 | 3 | bf16 | 5.4 | 5.6 | 5.8 | 144.1 | 20.1 | 1282 | 1440 |
| 2 | 4 | bf16 | 3.9 | 4.8 | 5.1 | 133.8 | 19.5 | 439 | 1411 |
| 2 | 48 | bf16 | 4.8 | 4.9 | 19.4 | | 25.6 | | |

GB/s counts the sources, the output and the queries once. At 4096 tokens the multi-query rows
sit at 97-99% of the copy bandwidth; the single-query rows read more than DRAM can deliver
because their output (38 MB) is the same buffer every call and stays in the L2. At one to 64
tokens every row is a launch and a few DRAM round trips: 2.3 to 5.8 us, 3.7 to 7 times less than
the torch ops (up to 19 times for nq = 3 at 8 and 64 tokens, where torch's batched matmul is
slow). The 48-query call costs what one 4-query call does at 1 and 8 tokens, because a token
with more than one chunk of queries and fewer tokens than two blocks per SM gets a block per
chunk (each re-reads the token's few sources from L2); before that split one block streamed all
48 queries and took 29 us.

### Message gates (us per call)

| gates | path | wf, wg | T = 1 | T = 8 | T = 64 | T = 4096 | torch, T = 1 | torch, T = 4096 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 1 | decode | bf16 | 5.6 | 8.0 | 29.0 | | 33.1 | |
| 1 | decode | f32 | 6.0 | 8.5 | 29.7 | | 31.8 | |
| 1 | prefill | bf16 | 10.9 | 12.2 | 15.6 | 172.0 | 32.6 | 475 |
| 1 | prefill | f32 | 10.4 | 12.0 | 15.3 | 166.5 | 31.9 | 474 |
| 3 (q, k, v) | decode | bf16 | 8.4 | 11.9 | 41.0 | | 112.2 | |
| 3 (q, k, v) | decode | f32 | 9.9 | 13.3 | 42.8 | | 112.0 | |
| 3 (q, k, v) | prefill | bf16 | 19.9 | 21.9 | 26.7 | 534.2 | 112.0 | 1418 |
| 3 (q, k, v) | prefill | f32 | 20.2 | 22.3 | 27.3 | 529.4 | 112.1 | 1418 |

The three q/k/v gates in one decode launch cost 8.4 us at one token, 1.5 times one gate, and 13
times the torch ops for the three. The decode kernel is still far from its weights' roofline:
3.5 MB of weights (bf16 wf, wg) in 8.4 us is 430 GB/s, 30% of the copy; each phase is one
round of 8 KB loads per block and the two grid syncs and the launch are most of the time. A
separate sweep of the two paths (same harness) put the crossover between 16 and 32 tokens
(T = 16: decode 11.4 us, prefill 12.8 us for one gate; T = 32: 18.3 and 13.8), so
`message_gates` takes the decode kernel up to 16 tokens: its token loops are serial per
block, and at 64 tokens the prefill path is twice as fast. At 4096 tokens the prefill path moves
about 265 MB per gate (the [T, 4608] bf16 GEMM input is written and read back) in 172 us,
close to the copy bandwidth for what it does; the unique bytes alone would take 79 us, which
is what fusing the norm into the GEMM's A load would buy.

### What one decoded token spends here

From the forward's schedule (`decode_schedule` in the bench) and the measured rows, one decode
step runs 96 attention residuals (47 attention inputs, 48 MLP inputs, `final_input`), 24
message inputs and 96 gates (24 loop-layer applications x q, k, v, MLP, in 48 launches):

| batch | residuals | message inputs as 2 calls of nq = 48 | gates | total per step | per token | torch per token |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 392 us | 309 us | 337 us (decode) | 646 us | 646 us | 5533 us |
| 8 | 439 us | 335 us | 477 us (decode) | 812 us | 101 us | 1019 us |
| 64 | 464 us | 381 us | 1015 us (prefill) | 1396 us | 21.8 us | 152 us |

"residuals" counts the 24 message inputs one call each; the next column replaces them with one
nq = 48 call per loop pass, which is what the total uses. These are sums of ops timed one at a
time from cold memory, not a measured model step. At batch 1 that is 0.65 ms a token in ops
that are almost all launch latency: 216 small ops in 146 launches. The bytes they must move are
small next to the time: the gate weights are 113 MB a step with bf16 wf, wg (170 MB with HF's
fp32 copies, 1.77 MB a gate, the same 48 gates read in both later passes), 78 us at the copy
bandwidth, and the residuals' queries, sources and outputs about 5 MB. The next step is fewer launches,
not faster ones: the attention residual folded into the RMSNorm that always follows it, and the
whole loop layer's small ops in one persistent kernel.

## Validation

`tests/test_attn_res.py` copies HF's `Inkling2LoopUnweightedRMSNorm`, the residual's forward
with the eager attention it calls, and `Inkling2LoopMessageGate`, and checks:

* `attn_res` at H = 2304 against the HF forward for nb 1..4, nq 1, 3, 4, T 1, 7, 256, bf16 and
  fp32 partial states (atol = rtol = 1e-4 in fp32), plus zero queries (HF's init, a plain mean),
  sharp queries (logits ~ +-25), HF's [B, S, nb, H] layout, a strided slice of a bigger buffer,
  an [nb, T, H] view, no block states, and nq = 48;
* the gates on both paths for T 1, 7, 64 (decode) and up to 256 (prefill), fp32 and bf16
  Wf/Wg: the error against the exact fp64 math must be at most twice HF's own error against
  it, and the q/k/v launch must match each gate run alone, for shared and separate h.

96 passed, 2 skipped on the RTX PRO 6000. The residual matches HF to 1.4e-6 (nb = 4, nq = 3,
256 tokens). For one gate (h ~ 1.5, m ~ 1 in scale) the largest errors against exact math:

| T | path | ours | HF | ours vs HF |
|---:|---|---:|---:|---:|
| 1 | decode | 1.4e-2 | 2.3e-2 | 2.8e-2 |
| 1 | prefill | 2.3e-2 | 2.3e-2 | 9.5e-7 |
| 64 | decode | 2.5e-2 | 3.3e-2 | 4.2e-2 |
| 64 | prefill | 3.3e-2 | 3.3e-2 | 9.6e-3 |
| 256 | prefill | 4.1e-2 | 4.5e-2 | 4.5e-2 |

The prefill path rounds where HF rounds, so at one token it is HF to the last bit or two (the
GEMM sums in another order, and at more tokens a z next to a bf16 boundary can round the other
way). The decode path skips HF's rounding of the normalized activations, so it is closer to
exact math than HF is, and further from HF.
