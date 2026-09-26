"""Thin Python wrappers over the compiled extension (spark_kernels._C).

All validation and dispatch happens in python/csrc/bindings.cpp: row-wise ops flatten every
leading dim into `rows`, and the output always has the input's shape and dtype. These wrappers
exist for the docstrings and type hints.

`variant=-1` (the default) runs the fastest implementation that accepts the input. Two
ladders have a top rung with extra requirements, and there the default steps down one rung
instead of failing: `swiglu` needs 16-byte aligned storage for its vectorized variant, and
`hgemm` needs N and K multiples of 64 for its top rungs (variants 3, 4 and 6). An explicit
variant is never substituted; it raises ValueError if it cannot take the input.
"""

from __future__ import annotations

import torch

from . import _C

KERNELS = ("bandwidth", "rmsnorm", "swiglu", "softmax", "sgemm", "hgemm", "attention")


def num_variants(name: str) -> int:
    """Number of implementation variants for kernel `name` (see docs/DESIGN.md)."""
    if name not in KERNELS:
        raise ValueError(f"unknown kernel {name!r}; expected one of {KERNELS}")
    return _C.num_variants(name)


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6, variant: int = -1) -> torch.Tensor:
    """out = x * rsqrt(mean(x**2, dim=-1) + eps) * w, computed in fp32.

    Args:
        x: [..., cols] float32 or bfloat16 CUDA tensor (contiguous).
        w: [cols] weight, same dtype as x.
        eps: numerical epsilon inside the rsqrt.
        variant: implementation index; -1 = fastest (variant 4; it runs variant 3's kernel on
            rows it cannot hold in registers, so any cols works).
            Variant 2 needs cols % 4 == 0 (float32) or cols % 8 == 0 (bfloat16) and
            16-byte aligned x, w and out.
    """
    return _C.rmsnorm(x, w, eps, variant)


def add_rmsnorm_(
    x: torch.Tensor, resid: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Fused decoder-block prologue: `resid += x` in place, then return rmsnorm(resid) * w.

    bfloat16 only; cols must be a multiple of 8. The residual stream is updated in place so
    the next block can consume it without another kernel launch.
    """
    return _C.add_rmsnorm_(x, resid, w, eps)


def swiglu(gate: torch.Tensor, up: torch.Tensor, variant: int = -1) -> torch.Tensor:
    """silu(gate) * up, elementwise, fp32 math. Any shape; gate and up must match.

    Variant 1 (128-bit loads) needs gate, up and the output to be 16-byte aligned; the default
    variant falls back to the scalar kernel on a tensor sliced to an odd storage offset.
    """
    return _C.swiglu(gate, up, variant)


def softmax(x: torch.Tensor, variant: int = -1) -> torch.Tensor:
    """Softmax over the last dim with fp32 max/sum (online softmax for variants >= 1)."""
    return _C.softmax(x, variant)


def sgemm(a: torch.Tensor, b: torch.Tensor, variant: int = -1) -> torch.Tensor:
    """fp32 GEMM: a[M,K] @ b[K,N] -> [M,N]. Any M, N, K."""
    return _C.sgemm(a, b, variant)


def hgemm(a: torch.Tensor, b: torch.Tensor, variant: int = -1) -> torch.Tensor:
    """bf16 tensor-core GEMM with fp32 accumulation: a[M,K] @ b[K,N] -> [M,N] (bf16).

    Requires N and K multiples of 16. Variant 6 (the default: the TMA mainloop on the
    Stream-K schedule) takes any M >= 1 when N and K are multiples of 64, like variants 3 and
    4, and runs a dedicated weight-streaming kernel for M <= 64; variant 5 needs M, N
    multiples of 128 and a grid of at least one 128x128 tile per SM; variants 0 to 2 need M a
    multiple of 16, and variant 2 M, N multiples of 128 and K a multiple of 32. The default
    steps down to the highest variant that accepts the shape.
    """
    return _C.hgemm(a, b, variant)


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, variant: int = -1
) -> torch.Tensor:
    """Fused attention forward: softmax(q @ k^T / sqrt(D)) @ v, fp32 math, bf16 out.

    q is [B, H_q, S_q, D], k and v are [B, H_kv, S_kv, D], all bfloat16, contiguous, with D 64
    or 128 and H_q a multiple of H_kv: grouped-query attention, where query head h reads k/v
    head h // (H_q // H_kv), the same as `F.scaled_dot_product_attention(enable_gqa=True)`;
    H_kv == H_q is plain multi-head attention. `causal` hides key j from query i when j > i,
    the same top-left convention as `F.scaled_dot_product_attention(is_causal=True)`. Any S_q
    and S_kv; a decode step is S_q = 1 against the cache, and the default variant runs a
    flash-decoding kernel for it that reads each k/v head once for all the query heads that
    share it. Returns [B, H_q, S_q, D] bfloat16.
    """
    return _C.attention(q, k, v, causal, variant)
