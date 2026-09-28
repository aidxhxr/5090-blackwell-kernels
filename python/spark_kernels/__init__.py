"""5090-blackwell-kernels: hand-written CUDA kernels for LLM inference on the RTX 5090 (sm_120).

Also builds for the NVIDIA DGX Spark (GB10, sm_121) with TORCH_CUDA_ARCH_LIST="12.1a"; the "a"
is where the block-scaled fp8 mma.sync lives (see setup.py).

Ops (all run on the current CUDA stream, all accept float32 or bfloat16 unless noted):

    rmsnorm(x, w, eps=1e-6, variant=-1)      RMSNorm over the last dim
    add_rmsnorm_(x, resid, w, eps=1e-6)      resid += x; return rmsnorm(resid)  (bf16, in place)
    swiglu(gate, up, variant=-1)             silu(gate) * up
    softmax(x, variant=-1)                   softmax over the last dim, fp32 math
    sgemm(a, b, variant=-1)                  fp32 GEMM (a @ b)
    hgemm(a, b, variant=-1, bias=, act=, residual=, out=)
                                             bf16 tensor-core GEMM (a @ b), fp32 accumulate,
                                             with a fused act(a @ b + bias) + residual epilogue
    hgemm_swiglu(a, w_gate_up, ...)          silu(a @ w_gate) * (a @ w_up) in one GEMM from the
                                             interleaved weight of interleave_gate_up(g, u)
    fp8gemm(a, b_t, scale_a, scale_b, variant=-1)  e4m3 GEMM (a @ b_t.T, scaled), bf16 out
    attention(q, k, v, causal=False, variant=-1)  fused attention over [B, H, S, D] (bf16)
    rope_append_(qkv, cos, sin, k_cache, v_cache, pos0, H_q, H_kv)
                                             RoPE on q and k, k and v into the caches; returns q
    num_variants(name)                       how many implementations exist for `name`

`variant` selects a rung on the optimization ladder described in docs/DESIGN.md; -1 picks
the fastest one that accepts the input (see ops.py for the two ladders where that is not the
top rung). `spark_kernels.reference` holds plain-PyTorch implementations used by tests.
`spark_kernels.layer` is a Llama-3-8B decoder layer built from these ops (prefill and decode
with a K/V cache) next to the same layer in plain PyTorch.
"""

from importlib.metadata import PackageNotFoundError, version

from . import layer, reference
from .ops import (
    add_rmsnorm_,
    attention,
    fp8gemm,
    hgemm,
    hgemm_swiglu,
    interleave_gate_up,
    num_variants,
    rmsnorm,
    rope_append_,
    sgemm,
    softmax,
    swiglu,
)

__all__ = [
    "add_rmsnorm_",
    "attention",
    "fp8gemm",
    "hgemm",
    "hgemm_swiglu",
    "interleave_gate_up",
    "layer",
    "num_variants",
    "reference",
    "rmsnorm",
    "rope_append_",
    "sgemm",
    "softmax",
    "swiglu",
]

try:
    __version__ = version("5090-blackwell-kernels")  # the one copy lives in pyproject.toml
except PackageNotFoundError:  # imported from a checkout that was never pip-installed
    __version__ = "0.0.0+unknown"
