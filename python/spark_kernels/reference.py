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


# ---- fp4 (e2m1): NVFP4 and MXFP4 ---------------------------------------------------------
# An e2m1 value is a sign, two exponent bits and one mantissa bit: 0, 0.5, 1, 1.5, 2, 3, 4, 6
# and their negatives, stored as a 4-bit code (bit 3 the sign) two to a byte, the element with
# the lower index in the low nibble (torch.float4_e2m1fn_x2). NVFP4 adds an e4m3 scale per 16
# consecutive values and a per-tensor fp32 scale; MXFP4 a ue8m0 (power of two) scale per 32.
# The functions below are the quantizer of src/kernels/fp4quant.cu operation for operation
# (the tests compare the bytes), and the GEMM as fp4gemm computes it.

E2M1_MAX = 6.0
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
NVFP4_BLOCK = 16
MXFP4_BLOCK = 32
E2M1_MAX_EXP = 2  # 6 = 1.5 * 2**2


def e2m1_codes(v: torch.Tensor) -> torch.Tensor:
    """4-bit e2m1 codes of float32 v, round to nearest even, saturating at 6 (uint8 0..15).

    The magnitude code counts the rounding boundaries below |v|; a tie goes to the even code
    (0.25 -> 0, 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5 -> 4), which is why
    the comparisons alternate between > and >=. The sign bit is set for v < 0.
    """
    a = v.abs()
    code = ((a > 0.25).to(torch.uint8) + (a >= 0.75) + (a > 1.25) + (a >= 1.75) + (a > 2.5)
            + (a >= 3.5) + (a > 5.0))
    return code | ((v < 0).to(torch.uint8) << 3)


def e2m1_values(codes: torch.Tensor) -> torch.Tensor:
    """float32 values of 4-bit e2m1 codes."""
    mag = torch.tensor(E2M1_VALUES, device=codes.device)[(codes & 7).long()]
    return torch.where((codes & 8) != 0, -mag, mag)


def pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    """[..., K] codes -> [..., K // 2] bytes, element 2i in the low nibble of byte i."""
    return codes[..., 0::2] | (codes[..., 1::2] << 4)


def unpack_e2m1(q: torch.Tensor) -> torch.Tensor:
    """[..., K // 2] bytes (uint8 or float4_e2m1fn_x2) -> [..., K] codes."""
    q = q.view(torch.uint8)
    return torch.stack((q & 15, q >> 4), dim=-1).reshape(*q.shape[:-1], 2 * q.shape[-1])


def to_blocked(sf: torch.Tensor) -> torch.Tensor:
    """The blocked layout of a [rows, cols] scale tensor, flat: what fp4gemm, cuBLASLt's
    VEC16_UE4M3 / VEC32_UE8M0 modes and torch's SWIZZLE_32_4_4 take. Rows are padded to a
    multiple of 128 and columns to a multiple of 4 (with zeros); tiles of 128 rows by 4
    columns are 512 bytes, row-major over tiles, and inside a tile the byte of (row r, column
    c) is at (r % 32) * 16 + (r // 32) * 4 + c % 4."""
    rows, cols = sf.shape
    rp, cp = -(-rows // 128) * 128, -(-cols // 4) * 4
    padded = torch.zeros(rp, cp, dtype=sf.dtype, device=sf.device)
    padded[:rows, :cols] = sf
    t = padded.view(rp // 128, 128, cp // 4, 4).permute(0, 2, 1, 3)
    return t.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()


def from_blocked(flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Inverse of to_blocked: the [rows, cols] scale tensor."""
    rp, cp = -(-rows // 128) * 128, -(-cols // 4) * 4
    t = flat.reshape(rp // 128, cp // 4, 32, 4, 4).transpose(2, 3).reshape(rp // 128, cp // 4,
                                                                            128, 4)
    return t.permute(0, 2, 1, 3).reshape(rp, cp)[:rows, :cols]


def quantize_nvfp4(
    x: torch.Tensor, scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """NVFP4 quantization of the rows of x [rows, K]: (q [rows, K // 2] uint8, blocked sf
    uint8, per-tensor decode scale s as a 0-dim float32 tensor).

    s defaults to max|x| / (6 * 448), which maps the largest element to 6 * 448 in units of s.
    A block of 16 gets the e4m3 scale nearest max|block| / 6 / s (at most 448), and its values
    are x / (s * scale) rounded to e2m1 (nearest even, saturating at 6); a block whose scale
    rounds to 0 is all zeros.
    """
    rows, K = x.shape
    if K % 64 != 0:
        raise ValueError(f"K = {K} must be a multiple of 64")
    xf = x.float()
    if scale is None:
        s = (xf.abs().amax() / (6.0 * 448.0)).clamp(min=torch.finfo(torch.float32).tiny)
    else:
        s = scale.float().reshape(())
    xb = xf.view(rows, K // NVFP4_BLOCK, NVFP4_BLOCK)
    amax = xb.abs().amax(dim=-1)
    sf = (amax / 6.0 / s).clamp(max=448.0).to(torch.float8_e4m3fn)
    denom = (sf.float() * s).unsqueeze(-1)
    codes = torch.where(denom > 0, e2m1_codes(xb / denom), torch.zeros_like(xb, dtype=torch.uint8))
    return pack_e2m1(codes.view(rows, K)), to_blocked(sf.view(torch.uint8)), s


def quantize_mxfp4(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP4 (OCP) quantization of the rows of x [rows, K]: (q [rows, K // 2] uint8, blocked
    sf uint8). A block of 32 gets 2**e with e = floor(log2(max|block|)) - 2 (e2m1's largest
    exponent), stored as e + 127 and clamped to [0, 254]; its values are x / 2**e rounded to
    e2m1, saturating at 6 (a block maximum in [6, 8) saturates, as the recipe accepts). An
    all-zero block gets the byte 0."""
    rows, K = x.shape
    if K % 128 != 0:
        raise ValueError(f"K = {K} must be a multiple of 128")
    xb = x.float().view(rows, K // MXFP4_BLOCK, MXFP4_BLOCK)
    amax = xb.abs().amax(dim=-1)
    # floor(log2(amax)) through frexp (amax = m * 2^p, m in [0.5, 1)), exact at powers of two.
    p = torch.frexp(amax).exponent - 1 - E2M1_MAX_EXP
    e = torch.where(amax > 0, p, torch.full_like(p, -127)).clamp(-127, 127)
    codes = e2m1_codes(xb * torch.exp2(-e.float()).unsqueeze(-1))
    return pack_e2m1(codes.view(rows, K)), to_blocked((e + 127).to(torch.uint8))


def dequantize_fp4(
    q: torch.Tensor, sf: torch.Tensor, fmt: str = "nvfp4", scale: torch.Tensor | None = None
) -> torch.Tensor:
    """float32 [rows, K] from packed codes and blocked block scales (times `scale` if given)."""
    rows, K = q.shape[0], 2 * q.shape[1]
    block = NVFP4_BLOCK if fmt == "nvfp4" else MXFP4_BLOCK
    sfr = from_blocked(sf.view(torch.uint8), rows, K // block)
    if fmt == "nvfp4":
        bs = sfr.view(torch.float8_e4m3fn).float()
    elif fmt == "mxfp4":
        bs = torch.exp2(sfr.float() - 127)
    else:
        raise ValueError(f"fmt must be 'nvfp4' or 'mxfp4', got {fmt!r}")
    out = e2m1_values(unpack_e2m1(q)) * bs.repeat_interleave(block, dim=-1)
    return out * scale.float() if scale is not None else out


def fp4gemm(
    a: torch.Tensor, sfa: torch.Tensor, b_t: torch.Tensor, sfb: torch.Tensor,
    fmt: str = "nvfp4", scale_a: torch.Tensor | None = None, scale_b: torch.Tensor | None = None,
) -> torch.Tensor:
    """The fp4 GEMM as fp4gemm computes it: the fp32 sum of the dequantized products (exact
    in fp32), times scale_a * scale_b, rounded once to bf16."""
    c = dequantize_fp4(a, sfa, fmt) @ dequantize_fp4(b_t, sfb, fmt).t()
    if scale_a is not None:
        c = c * (scale_a.float() * scale_b.float())
    return c.to(torch.bfloat16)
