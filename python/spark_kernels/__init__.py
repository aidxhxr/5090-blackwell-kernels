"""5090-blackwell-kernels: hand-written CUDA kernels for LLM inference on the RTX 5090 (sm_120).

Also builds for the NVIDIA DGX Spark (GB10, sm_121) with TORCH_CUDA_ARCH_LIST="12.1a"; the "a"
is where the block-scaled fp8 mma.sync lives (see setup.py).

Ops (all run on the current CUDA stream, all accept float32 or bfloat16 unless noted):

    rmsnorm(x, w, eps=1e-6, variant=-1)      RMSNorm over the last dim
    add_rmsnorm_(x, resid, w, eps=1e-6)      resid += x; return rmsnorm(resid)  (bf16, in place)
    swiglu(gate, up, variant=-1)             silu(gate) * up
    softmax(x, variant=-1)                   softmax over the last dim, fp32 math
    sample(logits, temperature, top_k, top_p, seed, offset, variant=-1)
                                             one token per row of bf16 logits (temperature,
                                             top-k, top-p, Philox), parameters on the device
    sgemm(a, b, variant=-1)                  fp32 GEMM (a @ b)
    hgemm(a, b, variant=-1, bias=, act=, residual=, out=)
                                             bf16 tensor-core GEMM (a @ b), fp32 accumulate,
                                             with a fused act(a @ b + bias) + residual epilogue
    w4gemm(a, W4Weight.quantize(w))          bf16 activations times an int4 weight (one scale
                                             per 128 k), fp32 accumulate
    hgemm_swiglu(a, w_gate_up, ...)          silu(a @ w_gate) * (a @ w_up) in one GEMM from the
                                             interleaved weight of interleave_gate_up(g, u)
    fp8gemm(a, b_t, scale_a, scale_b, variant=-1, sfa=None, sfb=None)
                                             e4m3 GEMM (a @ b_t.T, scaled), bf16 out; with
                                             sfa, sfb: MXFP8 (a ue8m0 scale per 32 k)
    fp8_quantize(x, mode="tensor")           bf16 -> e4m3 with a power-of-two scale per tensor
                                             or row, or MX scales per 32 ("mx")
    fp4_quantize(x, fmt="nvfp4", scale=None)   bf16 -> packed e2m1 + blocked block scales
    fp4gemm(a, b_t, sfa, sfb, scale_a=None, scale_b=None, fmt="nvfp4", variant=-1)
                                             fp4 GEMM (NVFP4 or MXFP4), bf16 out
    attention(q, k, v, causal=False, variant=-1)  fused attention over [B, H, S, D] (bf16)
    attention_fp8(q, k, v, q_scale, k_scale, v_scale, causal=False, variant=-1)
                                             the same over e4m3 q, k, v with descale factors,
                                             bf16 out; quantize_fp8(x, per_head=False) makes them
    attention_fwd(q, k, v, causal=False)     (out, lse): the forward plus its log-sum-exp
    attention_bwd(q, k, v, out, d_out, lse, causal=False, deterministic=False)
                                             (dq, dk, dv), the backward pass
    attention_with_grad(q, k, v, causal=False)  attention as a torch.autograd.Function
    rope_append_(qkv, cos, sin, k_cache, v_cache, pos0, H_q, H_kv)
                                             RoPE on q and k, k and v into the caches; returns q
    rope_append_paged_(qkv, cos, sin, positions, slots, k_cache, v_cache, H_q, H_kv)
                                             the same into a paged K/V cache, packed tokens
    paged_decode(q, k_cache, v_cache, block_table, seq_lens, variant=-1)
                                             a decode step of a batch of any lengths
    attention_varlen(q, k_cache, v_cache, cu_seqlens_q, seq_lens, block_table, causal=True)
                                             packed prompts of any lengths, no padding
                                             (the three paged ops also take float8_e4m3fn
                                             caches with k_scale=, v_scale= per kv head;
                                             quantize_kv_cache is the same rounding in
                                             torch, dequantize_kv_cache its values)
    num_variants(name)                       how many implementations exist for `name`

`variant` selects a rung on the optimization ladder described in docs/DESIGN.md; -1 picks
the fastest one that accepts the input (see ops.py for the two ladders where that is not the
top rung). `spark_kernels.reference` holds plain-PyTorch implementations used by tests.
`spark_kernels.layer` is a Llama-3-8B decoder layer built from these ops (prefill and decode
with a K/V cache) next to the same layer in plain PyTorch. `spark_kernels.engine` runs the
whole 32-layer model on a paged K/V cache with batched prefill and continuous batching.
"""

from importlib.metadata import PackageNotFoundError, version

from . import engine, layer, reference
from .ops import (
    AttentionFunction,
    W4Weight,
    add_rmsnorm_,
    attention,
    attention_bwd,
    attention_fp8,
    attention_fwd,
    attention_varlen,
    attention_with_grad,
    dequantize_kv_cache,
    fp4_quantize,
    fp4gemm,
    fp8_quantize,
    fp8gemm,
    hgemm,
    hgemm_swiglu,
    interleave_gate_up,
    num_variants,
    paged_decode,
    quantize_fp8,
    quantize_kv_cache,
    rmsnorm,
    rope_append_,
    rope_append_paged_,
    sample,
    sgemm,
    softmax,
    swiglu,
    w4_quantize,
    w4_repack,
    w4gemm,
)

__all__ = [
    "AttentionFunction",
    "W4Weight",
    "add_rmsnorm_",
    "attention",
    "attention_bwd",
    "attention_fp8",
    "attention_fwd",
    "attention_varlen",
    "attention_with_grad",
    "dequantize_kv_cache",
    "engine",
    "fp4_quantize",
    "fp8_quantize",
    "fp4gemm",
    "fp8gemm",
    "hgemm",
    "hgemm_swiglu",
    "interleave_gate_up",
    "layer",
    "num_variants",
    "paged_decode",
    "quantize_fp8",
    "quantize_kv_cache",
    "reference",
    "rmsnorm",
    "rope_append_",
    "rope_append_paged_",
    "sample",
    "sgemm",
    "softmax",
    "swiglu",
    "w4_quantize",
    "w4_repack",
    "w4gemm",
]

try:
    __version__ = version("5090-blackwell-kernels")  # the one copy lives in pyproject.toml
except PackageNotFoundError:  # imported from a checkout that was never pip-installed
    __version__ = "0.0.0+unknown"
