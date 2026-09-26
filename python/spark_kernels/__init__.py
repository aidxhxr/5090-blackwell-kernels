"""spark-kernels: hand-written CUDA kernels for LLM inference on Blackwell RTX 5090 (sm_120).

Also builds for the NVIDIA DGX Spark (GB10, sm_121) with TORCH_CUDA_ARCH_LIST="12.1".

Ops (all run on the current CUDA stream, all accept float32 or bfloat16 unless noted):

    rmsnorm(x, w, eps=1e-6, variant=-1)      RMSNorm over the last dim
    add_rmsnorm_(x, resid, w, eps=1e-6)      resid += x; return rmsnorm(resid)  (bf16, in place)
    swiglu(gate, up, variant=-1)             silu(gate) * up
    softmax(x, variant=-1)                   softmax over the last dim, fp32 math
    sgemm(a, b, variant=-1)                  fp32 GEMM (a @ b)
    hgemm(a, b, variant=-1)                  bf16 tensor-core GEMM (a @ b), fp32 accumulate
    attention(q, k, v, causal=False, variant=-1)  fused attention over [B, H, S, D] (bf16)
    num_variants(name)                       how many implementations exist for `name`

`variant` selects a rung on the optimization ladder described in docs/DESIGN.md; -1 picks
the fastest one that accepts the input (see ops.py for the two ladders where that is not the
top rung). `spark_kernels.reference` holds plain-PyTorch implementations used by tests.
"""

from importlib.metadata import PackageNotFoundError, version

from . import reference
from .ops import (
    add_rmsnorm_,
    attention,
    hgemm,
    num_variants,
    rmsnorm,
    sgemm,
    softmax,
    swiglu,
)

__all__ = [
    "add_rmsnorm_",
    "attention",
    "hgemm",
    "num_variants",
    "reference",
    "rmsnorm",
    "sgemm",
    "softmax",
    "swiglu",
]

try:
    __version__ = version("spark-kernels")  # the one copy lives in pyproject.toml
except PackageNotFoundError:  # imported from a checkout that was never pip-installed
    __version__ = "0.0.0+unknown"
