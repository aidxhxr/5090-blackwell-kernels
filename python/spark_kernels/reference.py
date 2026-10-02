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


def quantize_fp8_pow2(x: torch.Tensor, per_row: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """`spark_kernels.fp8_quantize` in modes "tensor" and "row": e4m3 elements and the
    smallest power-of-two scale 2^e, e in [-126, 127], with max|x| / 2^e <= 448, per tensor
    ([1]) or per row ([rows, 1]), float32."""
    xf = x.float()
    amax = xf.abs().amax(dim=1, keepdim=True) if per_row else xf.abs().amax().reshape(1)
    m, p = torch.frexp(amax.clamp(min=torch.finfo(torch.float32).tiny) / E4M3_MAX)
    e = torch.where(m == 0.5, p - 1, p).clamp(-126, 127)
    s = torch.exp2(e.float())
    q = (xf / s).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, s


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


# ---- W4A16: int4 weights, one scale (and zero point) per 128 k of a column ----------------

W4_GROUP = 128


def w4_quantize(w: torch.Tensor, asym: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The round-to-nearest quantizer of `spark_kernels.w4_quantize` in plain PyTorch, same
    fp32 operations: (codes [K, N] uint8 in 0..15, scales [K/128, N] bf16, zeros or None).
    Codes are unpacked here; `w4_pack_gptq` packs them."""
    K, N = w.shape
    wf = w.float().reshape(K // W4_GROUP, W4_GROUP, N)
    if asym:
        lo = wf.amin(dim=1).clamp(max=0.0)
        hi = wf.amax(dim=1).clamp(min=0.0)
        s = ((hi - lo) / 15.0).to(torch.bfloat16)
    else:
        s = (wf.abs().amax(dim=1) / 7.0).to(torch.bfloat16)
    sf = s.float()
    safe = torch.where(sf > 0, sf, torch.ones_like(sf))
    if asym:
        z = torch.where(sf > 0, torch.round(-lo / safe).clamp(0, 15), torch.zeros_like(sf))
        q = (torch.round(wf / safe[:, None]) + z[:, None]).clamp(0, 15)
        q = torch.where(sf[:, None] > 0, q, z[:, None].expand_as(q))
    else:
        z = None
        q = torch.round(wf / safe[:, None]).clamp(-8, 7) + 8
        q = torch.where(sf[:, None] > 0, q, torch.full_like(q, 8.0))
    zeros = z.to(torch.uint8) if z is not None else None
    return q.reshape(K, N).to(torch.uint8), s, zeros


def w4_dequantize(q: torch.Tensor, scales: torch.Tensor, zeros: torch.Tensor | None = None
                  ) -> torch.Tensor:
    """bf16((q - z) * s) for codes [K, N], the weight w4gemm multiplies (z = 8 if no zeros)."""
    K, N = q.shape
    z = zeros.float() if zeros is not None else torch.full_like(scales, 8.0, dtype=torch.float32)
    qf = q.float().reshape(K // W4_GROUP, W4_GROUP, N)
    w = (qf - z[:, None]) * scales.float()[:, None]
    return w.reshape(K, N).to(torch.bfloat16)


def w4_pack_gptq(q: torch.Tensor) -> torch.Tensor:
    """Codes [K, N] -> qweight [K/8, N] int32, eight consecutive k per word, lowest k lowest."""
    K, N = q.shape
    words = torch.zeros(K // 8, N, dtype=torch.int64, device=q.device)
    qv = q.to(torch.int64).reshape(K // 8, 8, N)
    for i in range(8):
        words |= qv[:, i] << (4 * i)
    return _as_int32(words)


def w4_unpack_gptq(qweight: torch.Tensor) -> torch.Tensor:
    """qweight [K/8, N] int32 -> codes [K, N] uint8."""
    words = qweight.to(torch.int64) & 0xFFFFFFFF
    shifts = torch.arange(8, device=qweight.device, dtype=torch.int64) * 4
    q = (words[:, None, :] >> shifts[None, :, None]) & 0xF
    return q.reshape(-1, qweight.shape[1]).to(torch.uint8)


def _as_int32(words: torch.Tensor) -> torch.Tensor:
    """uint32 values held in int64 -> the same bits as int32."""
    words = words & 0xFFFFFFFF
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def w4_group_k(j: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """Matrix k inside a 128-k group that mma step j (0..7) feeds at mma index kappa (0..15):
    the k order of src/kernels/w4gemm_internal.cuh, which makes each lane's activations for
    two steps eight consecutive k."""
    c = (kappa % 8) // 2
    return 32 * (j // 2) + 8 * c + 4 * (j % 2) + 2 * (kappa // 8) + kappa % 2


def _w4_nibble_coords(device) -> tuple[torch.Tensor, torch.Tensor]:
    """(row within 16 columns, k within the 128-k group) of nibble p of word j of lane in the
    unit of parity h (h = 0, 1 are the two 64-k units of a group), as [2, 32, 4, 8] tensors."""
    h = torch.arange(2, device=device)[:, None, None, None]
    lane = torch.arange(32, device=device)[None, :, None, None]
    j = torch.arange(4, device=device)[None, None, :, None]
    p = torch.arange(8, device=device)[None, None, None, :]
    g, c = lane // 4, lane % 4
    dn = (g + 8 * (p & 1) + 0 * j + 0 * h).expand(2, 32, 4, 8)
    dk = w4_group_k(4 * h + j, 8 * ((p >> 1) & 1) + 2 * c + (p >> 2)).expand(2, 32, 4, 8)
    return dn, dk


def w4_repack(qweight: torch.Tensor) -> torch.Tensor:
    """The permutation of `spark_kernels.w4_repack`: qweight [K/8, N] -> int32 [N/16, 2K],
    blocks of 16 columns x 64 k in strip-major order, 32 lanes x 4 words per block."""
    q = w4_unpack_gptq(qweight)
    K, N = q.shape
    dn, dk = _w4_nibble_coords(q.device)
    groups = q.reshape(K // 128, 128, N // 16, 16).permute(2, 0, 1, 3)  # [N/16, K/128, 128, 16]
    vals = groups[:, :, dk, dn].to(torch.int64)  # [N/16, K/128, 2, 32, 4, 8]
    shifts = torch.arange(8, device=q.device, dtype=torch.int64) * 4
    words = (vals << shifts).sum(dim=-1)
    return _as_int32(words).reshape(N // 16, 2 * K)


def w4_unpack(packed: torch.Tensor) -> torch.Tensor:
    """Inverse of w4_repack: packed [N/16, 2K] -> codes [K, N] uint8."""
    T, twoK = packed.shape
    K, N = twoK // 2, T * 16
    words = packed.to(torch.int64).reshape(T, K // 128, 2, 32, 4) & 0xFFFFFFFF
    shifts = torch.arange(8, device=packed.device, dtype=torch.int64) * 4
    vals = (words[..., None] >> shifts) & 0xF  # [T, K/128, 2, 32, 4, 8]
    dn, dk = _w4_nibble_coords(packed.device)
    groups = torch.zeros(T, K // 128, 128, 16, dtype=torch.int64, device=packed.device)
    groups[:, :, dk, dn] = vals
    return groups.permute(1, 2, 0, 3).reshape(K, N).to(torch.uint8)


def w4gemm(a: torch.Tensor, q: torch.Tensor, scales: torch.Tensor,
           zeros: torch.Tensor | None = None) -> torch.Tensor:
    """a @ the dequantized weight, fp32 sum, one bf16 rounding."""
    return (a.float() @ w4_dequantize(q, scales, zeros).float()).to(torch.bfloat16)
