"""Plain-PyTorch reference implementations. Used by tests and by scripts/bench_torch.py.

They deliberately mirror the numerics of the CUDA kernels (fp32 math, cast at the end).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv * w.float()).to(x.dtype)


def add_rmsnorm(
    x: torch.Tensor, resid: torch.Tensor, w: torch.Tensor, eps: float = 1e-6
) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (new_resid, out) without mutating inputs. Matches the fused kernel's rounding:
    the residual is rounded to the storage dtype before normalization."""
    new_resid = (resid.float() + x.float()).to(x.dtype)
    return new_resid, rmsnorm(new_resid, w, eps)


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return (F.silu(gate.float()) * up.float()).to(gate.dtype)


def softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x.float(), dim=-1).to(x.dtype)


def gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a.float() @ b.float()).to(a.dtype)


# ---- MXFP8 (OCP Microscaling): e4m3 elements with one E8M0 scale per 32-element block ----

MX_BLOCK = 32
E4M3_MAX = 448.0
E4M3_MAX_EXP = 8  # 448 = 1.75 * 2**8


def quantize_mx(
    x: torch.Tensor, block: int = MX_BLOCK, mode: str = "floor"
) -> tuple[torch.Tensor, torch.Tensor]:
    """OCP MX quantization of the rows of x along the last dim: (e4m3 elements, ue8m0 scales).

    Each block of `block` consecutive elements shares one power-of-two scale 2**e. With
    mode="floor" (the OCP recipe) e = floor(log2(max|x|)) - 8, the exponent that puts the
    block's largest element in the top binade of e4m3, [256, 512); the elements are x / 2**e
    rounded to e4m3, and a largest element above 448 (the top eighth of that binade)
    saturates to 448, which the recipe accepts. With mode="ceil", e = ceil(log2(max|x| / 448)),
    the smallest exponent under which nothing saturates: the largest element lands in
    (224, 448]. The scale is returned as the E8M0 byte e + 127 (a uint8 tensor of shape
    x.shape[:-1] + [K // block]), clamped to [0, 254] (255 is NaN); an all-zero block gets 0.
    The result is what `fp8gemm(..., sfa=, sfb=)` and cuBLASLt's VEC32_UE8M0 mode consume
    (cuBLASLt in its own tiled layout, see tests/test_fp8gemm.py).
    """
    K = x.shape[-1]
    if K % block != 0:
        raise ValueError(f"last dim {K} is not a multiple of the block size {block}")
    if mode not in ("floor", "ceil"):
        raise ValueError(f"mode must be 'floor' or 'ceil', got {mode!r}")
    xb = x.float().reshape(*x.shape[:-1], K // block, block)
    amax = xb.abs().amax(dim=-1, keepdim=True)
    if mode == "floor":
        # floor(log2(amax)) through frexp (amax = m * 2^p with m in [0.5, 1)), exact where
        # log2 would round a value just under a power of two up to it.
        p = torch.frexp(amax).exponent.float() - 1 - E4M3_MAX_EXP
    else:
        p = torch.ceil(torch.log2(amax / E4M3_MAX))
    # frexp(0) and log2(0) do not give -inf alike; a zero block goes to the smallest scale.
    p = torch.where(amax > 0, p, torch.full_like(p, -1000.0))
    e = p.clamp(-127, 127)
    q = (xb / torch.exp2(e)).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    sf = (e + 127).to(torch.uint8)
    return q.reshape(x.shape), sf.reshape(*x.shape[:-1], K // block)


def dequantize_mx(q: torch.Tensor, sf: torch.Tensor, block: int = MX_BLOCK) -> torch.Tensor:
    """Inverse of quantize_mx in float32 (exact: a power of two times an e4m3 value)."""
    scale = torch.exp2(sf.float() - 127).repeat_interleave(block, dim=-1)
    return q.float() * scale


def quantize_per_tensor(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor e4m3 quantization: (e4m3 elements, float32 scale) with scale = max|x| / 448."""
    scale = (x.float().abs().amax() / E4M3_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    q = (x.float() / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale


def fp8gemm_mx(
    a: torch.Tensor, sfa: torch.Tensor, b_t: torch.Tensor, sfb: torch.Tensor
) -> torch.Tensor:
    """The MX GEMM as fp8gemm computes it: fp32 sum of the dequantized products, one bf16 round."""
    return (dequantize_mx(a, sfa) @ dequantize_mx(b_t, sfb).t()).to(torch.bfloat16)
