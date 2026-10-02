"""Quantized projections for the engine: one weight matrix of a layer in a low-precision
format, with the GEMM that multiplies by it and the activation quantizer that GEMM needs.

Every class takes the [K, N] bf16 weight of `hgemm` (activations @ weight), quantizes it once
at construction and keeps nothing else, and is called as `lin(a)` with a [M, K] bf16
activation to give the [M, N] bf16 product. `dequantize()` returns the bf16 [K, N] weight the
format stands for (what a reference model multiplies).

    format      weights                                activations            GEMM
    bf16        bf16                                   bf16                   hgemm
    int4        int4, a bf16 scale per 128 k           bf16                   w4gemm
    int4-asym   int4, a bf16 scale and zero per 128 k  bf16                   w4gemm
    fp8         e4m3, one fp32 scale                   e4m3, one dynamic fp32 scale per call
    fp8-tok     e4m3, one fp32 scale                   e4m3, a dynamic scale per token
    mxfp8       e4m3, an e8m0 scale per 32 k           e4m3, an e8m0 scale per 32 k
    nvfp4       e2m1, an e4m3 scale per 16 k + fp32    e2m1, the same, per call  fp4gemm
    mxfp4       e2m1, an e8m0 scale per 32 k           e2m1, an e8m0 scale per 32 k

fp8, fp8-tok and mxfp8 go through `fp8gemm`, nvfp4 and mxfp4 through `fp4gemm`. Those two
take the weight K-contiguous ([N, K], nn.Linear's layout), so the weight is transposed once
here. The activation quantizers of the fp8 formats are a few torch launches each (an amax
reduction, the scale, a multiply, a clamp and the cast), with power-of-two scales so the bf16
multiply is exact; fp4 activations go through the
`fp4_quantize` kernel. fp8-tok cannot hand fp8gemm a per-row scale (its scales are per
tensor), so it multiplies the bf16 output by the row scales afterwards: one more launch, and
one more bf16 rounding.
"""

from __future__ import annotations

import torch

from . import ops as sk
from . import reference as ref

FORMATS = ("bf16", "int4", "int4-asym", "fp8", "fp8-tok", "mxfp8", "nvfp4", "mxfp4")
E4M3_MAX = 448.0
TINY = torch.finfo(torch.float32).tiny


def _nbytes(*ts: torch.Tensor | None) -> int:
    return sum(t.numel() * t.element_size() for t in ts if t is not None)


class BF16Linear:
    def __init__(self, w: torch.Tensor):
        self.w = w

    def __call__(self, a: torch.Tensor) -> torch.Tensor:
        return sk.hgemm(a, self.w)

    def nbytes(self) -> int:
        return _nbytes(self.w)

    def dequantize(self) -> torch.Tensor:
        return self.w


class Int4Linear:
    """W4A16: int4 weights with a bf16 scale (and with asym a zero point) per 128 k of a
    column, round to nearest, multiplied by `w4gemm` against bf16 activations."""

    def __init__(self, w: torch.Tensor, asym: bool = False):
        self.w4 = sk.W4Weight.quantize(w, asym)

    def __call__(self, a: torch.Tensor) -> torch.Tensor:
        return sk.w4gemm(a, self.w4)

    def nbytes(self) -> int:
        return self.w4.nbytes()

    def dequantize(self) -> torch.Tensor:
        return ref.w4_dequantize(ref.w4_unpack(self.w4.packed), self.w4.scales, self.w4.zeros)


def _fp8_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[K, N] bf16 -> ([N, K] e4m3, fp32 scale) with the scale max|w| / 448."""
    wt = w.t()
    s = (wt.float().abs().amax() / E4M3_MAX).clamp(min=TINY)
    q = (wt.float() / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).contiguous()
    return q, s.reshape(1)


def pow2_scale(amax: torch.Tensor) -> torch.Tensor:
    """The smallest power of two s with amax / s <= 448, as fp32. A power-of-two scale makes
    the multiply by 1 / s exact in bf16, so the cast to e4m3 is the only rounding; e4m3 is a
    float format, so the at most 2x of range it gives up costs no relative precision above
    its subnormals."""
    return torch.exp2(torch.ceil(torch.log2(amax.float().clamp(min=TINY) / E4M3_MAX)))


def quantize_fp8_tensor(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-tensor e4m3 quantization of a bf16 activation: (e4m3, fp32 scale [1]),
    the scale `pow2_scale(max|a|)`."""
    s = pow2_scale(torch.linalg.vector_norm(a, float("inf")))
    q = torch.mul(a, (1.0 / s).to(a.dtype)).clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, s.reshape(1)


def quantize_fp8_rows(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-row e4m3 quantization of a bf16 activation: (e4m3, fp32 scales [M, 1]),
    each `pow2_scale` of its row's max|a|."""
    s = pow2_scale(torch.linalg.vector_norm(a, float("inf"), dim=1, keepdim=True))
    q = torch.mul(a, (1.0 / s).to(a.dtype)).clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, s


def quantize_mx_bf16(a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """`reference.quantize_mx` (the OCP floor recipe) for a bf16 [M, K] activation without
    the fp32 copy: the block scales are powers of two, so the multiply by 2^-e is exact in
    bf16 (outside its subnormals) and the cast to e4m3 is the only rounding."""
    M, K = a.shape
    ab = a.view(M, K // 32, 32)
    amax = torch.linalg.vector_norm(ab, float("inf"), dim=-1).float()
    e = torch.frexp(amax).exponent.float() - 1 - ref.E4M3_MAX_EXP
    e = torch.where(amax > 0, e, torch.full_like(e, -127.0)).clamp_(-127, 127)
    q = torch.mul(ab, torch.exp2(-e).to(a.dtype)[..., None])
    q = q.clamp_(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).view(M, K)
    return q, (e + 127).to(torch.uint8)


class FP8Linear:
    """W8A8 e4m3 with a per-tensor weight scale. `per_token=False`: one dynamic activation
    scale per call (max|a| / 448 over the whole [M, K]). `per_token=True`: one per row,
    applied to the output rows after the GEMM."""

    def __init__(self, w: torch.Tensor, per_token: bool = False):
        self.wt, self.sw = _fp8_weight(w)
        self.per_token = per_token
        self.one = torch.ones(1, device=w.device, dtype=torch.float32)

    def __call__(self, a: torch.Tensor) -> torch.Tensor:
        if self.per_token:
            q, s = quantize_fp8_rows(a)
            return sk.fp8gemm(q, self.wt, self.one, self.sw).mul_(s)
        q, s = quantize_fp8_tensor(a)
        return sk.fp8gemm(q, self.wt, s, self.sw)

    def nbytes(self) -> int:
        return _nbytes(self.wt, self.sw)

    def dequantize(self) -> torch.Tensor:
        return (self.wt.float() * self.sw).t().to(torch.bfloat16)


class MXFP8Linear:
    """W8A8 MXFP8: e4m3 with a power-of-two scale per 32 k on both operands, applied inside
    the block-scaled mma. K must be a multiple of 256."""

    def __init__(self, w: torch.Tensor):
        self.wt, self.sfb = ref.quantize_mx(w.t().contiguous())

    def __call__(self, a: torch.Tensor) -> torch.Tensor:
        q, sfa = quantize_mx_bf16(a)
        return sk.fp8gemm(q, self.wt, sfa=sfa, sfb=self.sfb)

    def nbytes(self) -> int:
        return _nbytes(self.wt, self.sfb)

    def dequantize(self) -> torch.Tensor:
        return ref.dequantize_mx(self.wt, self.sfb).t().to(torch.bfloat16)


class FP4Linear:
    """W4A4 NVFP4 (e2m1, an e4m3 scale per 16 k and a per-tensor fp32 scale) or MXFP4 (e2m1,
    an e8m0 scale per 32 k). The activations are quantized per call by `fp4_quantize`, with
    the NVFP4 per-tensor scale computed on the device. K must be a multiple of 256."""

    def __init__(self, w: torch.Tensor, fmt: str = "nvfp4"):
        self.fmt = fmt
        self.wt, self.sfb, self.sw = sk.fp4_quantize(w.t().contiguous(), fmt)
        self.n = w.shape[1]

    def __call__(self, a: torch.Tensor) -> torch.Tensor:
        q, sfa, s = sk.fp4_quantize(a, self.fmt)
        return sk.fp4gemm(q, self.wt, sfa, self.sfb, s, self.sw, fmt=self.fmt)

    def nbytes(self) -> int:
        return _nbytes(self.wt, self.sfb, self.sw)

    def dequantize(self) -> torch.Tensor:
        return ref.dequantize_fp4(self.wt, self.sfb, self.fmt, self.sw).t().to(torch.bfloat16)


def make_linear(w: torch.Tensor, fmt: str):
    """The [K, N] bf16 weight `w` in format `fmt` (one of FORMATS)."""
    if fmt == "bf16":
        return BF16Linear(w)
    if fmt in ("int4", "int4-asym"):
        return Int4Linear(w, asym=fmt == "int4-asym")
    if fmt in ("fp8", "fp8-tok"):
        return FP8Linear(w, per_token=fmt == "fp8-tok")
    if fmt == "mxfp8":
        return MXFP8Linear(w)
    if fmt in ("nvfp4", "mxfp4"):
        return FP4Linear(w, fmt)
    raise ValueError(f"unknown weight format {fmt!r}; expected one of {FORMATS}")
