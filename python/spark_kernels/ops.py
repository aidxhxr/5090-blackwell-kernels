"""Thin Python wrappers over the compiled extension (spark_kernels._C).

All validation and dispatch happens in python/csrc/bindings.cpp: row-wise ops flatten every
leading dim into `rows`, and the output always has the input's shape and dtype. These wrappers
exist for the docstrings and type hints.

`variant=-1` (the default) runs the fastest implementation that accepts the input. Two
ladders have a top rung with extra requirements, and there the default steps down one rung
instead of failing: `swiglu` needs 16-byte aligned storage for its vectorized variant, and
`hgemm` needs N and K multiples of 64 for its top rungs (variants 3, 4 and 6). `sgemm`'s
default is its top fp32 rung (variant 5); its TF32 tensor-core variants 6 and 7 are a
different precision contract and have to be asked for. An explicit variant is never
substituted; it raises ValueError if it cannot take the input.
"""

from __future__ import annotations

import torch

from . import _C

KERNELS = ("bandwidth", "rmsnorm", "swiglu", "softmax", "sgemm", "hgemm", "fp8gemm", "fp4gemm",
           "w4gemm", "attention", "attention_bwd", "paged_decode", "attention_varlen",
           "attention_fp8", "sample")


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
    With the default variant, 2-D gate and up whose rows are contiguous but not adjacent
    (the two halves `gu[:, :I]` and `gu[:, I:]` of a fused gate|up projection) are read in
    place by a strided kernel; the output is a new contiguous tensor.
    """
    return _C.swiglu(gate, up, variant)


def softmax(x: torch.Tensor, variant: int = -1) -> torch.Tensor:
    """Softmax over the last dim with fp32 max/sum (online softmax for variants >= 1)."""
    return _C.softmax(x, variant)


def sample(
    logits: torch.Tensor,
    temperature: torch.Tensor,
    top_k: torch.Tensor,
    top_p: torch.Tensor,
    seed: torch.Tensor,
    offset: torch.Tensor,
    variant: int = -1,
) -> torch.Tensor:
    """One token per row of `logits` [B, V] bfloat16, returned as int64 [B]. The parameters
    are [B] device tensors, so a CUDA graph that captures the call replays with whatever they
    hold: temperature and top_p float32, top_k int32, seed and offset int64. Every call adds
    one to each row's offset in place, so replays draw fresh numbers; the engine sets a
    request's offset to the index of the token it is about to generate.

    Per row, with x the logits in fp32 and m their max:

    * temperature <= 0 is greedy: the first index of the max, as torch.argmax.
    * The weights are w_i = floor(exp((x_i - m) / T) * 2^40) as 64-bit integers; a token
      under 2^-40 of the top token's probability has weight 0 and is never drawn. All sums
      are integer sums, so the token depends on the row and its parameters only, not on the
      batch around it or the order of the kernel's reductions.
    * top_k > 0: keep the tokens whose value is at least the k-th largest among the nonzero
      weights; tokens tied with the k-th are all kept. 0 (or k at least that count) disables.
    * top_p < 1, applied after top-k to what it kept (total weight Z): keep the tokens whose
      value is at least the largest v such that the kept weight at or above v is at least
      ceil(top_p Z). That is the usual rule (sorted descending, a token is kept while the
      probability before it is < top_p), with tied values kept or dropped together. top_p <= 0
      keeps the top value only.
    * The token is the inverse CDF in index order at floor(u Z_f / 2^64), Z_f the kept total
      and u the first 64 bits of Philox4_32_10 block `offset` of stream `seed`.

    Variant 0 is one thread per row (the reference); variant 1, the default, one block per row
    with a radix select instead of a sort. Both give the same token for the same inputs.
    """
    return _C.sample(logits, temperature, top_k, top_p, seed, offset, variant)


def sgemm(a: torch.Tensor, b: torch.Tensor, variant: int = -1) -> torch.Tensor:
    """fp32 GEMM: a[M,K] @ b[K,N] -> [M,N]. Any M, N, K.

    The default (variant 5) is plain fp32 on the CUDA cores. Variants 6 and 7 run on the
    tensor cores: 6 rounds the operands to TF32 (the contract of torch's
    ``allow_tf32``, about 2^-11 relative error per operand) and 7 does three TF32 passes
    (3xTF32) for fp32-class accuracy at a third of the rate. Both are opt-in: -1 never
    picks them.
    """
    return _C.sgemm(a, b, variant)


def hgemm(
    a: torch.Tensor,
    b: torch.Tensor,
    variant: int = -1,
    *,
    bias: torch.Tensor | None = None,
    act: str | None = None,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """bf16 tensor-core GEMM with fp32 accumulation: a[M,K] @ b[K,N] -> [M,N] (bf16), with an
    optional fused epilogue: act(a @ b + bias) + residual, applied in fp32 in the kernel
    before the one rounding to bf16.

    Requires N and K multiples of 16. Variant 6 (the default: the TMA mainloop on the
    Stream-K schedule) takes any M >= 1 when N and K are multiples of 64, like variants 3 and
    4, and runs a dedicated weight-streaming kernel for M <= 64; variant 5 needs M, N
    multiples of 128 and a grid of at least one 128x128 tile per SM; variants 0 to 2 need M a
    multiple of 16, and variant 2 M, N multiples of 128 and K a multiple of 32. The default
    steps down to the highest variant that accepts the shape.

    Args:
        bias: [N] bfloat16, added to every row before the activation.
        act: None, "silu", "gelu" (the tanh form, as F.gelu(approximate="tanh")) or "relu".
        residual: [M, N] bfloat16, added after the activation.
        out: [M, N] bfloat16 to write into instead of allocating; `out=residual` is the
            in-place accumulate `residual += act(a @ b + bias)`.
    The fused options run on variants 4 and 6 only (N and K multiples of 64); asking for one
    on another variant, or on a shape those two refuse, raises ValueError.
    """
    return _C.hgemm(a, b, variant, bias, act, residual, False, out)


def interleave_gate_up(w_gate: torch.Tensor, w_up: torch.Tensor) -> torch.Tensor:
    """The gate/up weight layout `hgemm_swiglu` takes: [..., 2N] with gate_j in column 2j and
    up_j in column 2j+1. Done once at load time, for the [K, N] weights as `hgemm` takes them
    (w_gate = gate_proj.weight.t() for an HF Llama layer) and for a [N] bias alike."""
    if w_gate.shape != w_up.shape:
        raise ValueError(f"gate and up must have the same shape, got {tuple(w_gate.shape)} and "
                         f"{tuple(w_up.shape)}")
    return torch.stack((w_gate, w_up), dim=-1).reshape(*w_gate.shape[:-1], 2 * w_gate.shape[-1])


def hgemm_swiglu(
    a: torch.Tensor,
    w_gate_up: torch.Tensor,
    variant: int = -1,
    *,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """silu(a @ w_gate) * (a @ w_up) in one GEMM: a[M,K] @ w_gate_up[K,2N] -> [M,N] (bf16).

    `w_gate_up` is the interleaved layout from `interleave_gate_up` (gate_j at column 2j, up_j
    at 2j+1), so both halves of every output element land in the same lane of the epilogue,
    which forms the product in fp32 and rounds once. One read of `a` and one kernel where
    eager PyTorch runs two GEMMs, a silu and a multiply. `bias` is [2N] in the same
    interleaved layout, added before the gate; `residual` is [M, N], added to the product.
    Same shape and variant rules as `hgemm` with N = 2N (a multiple of 64).
    """
    return _C.hgemm(a, w_gate_up, variant, bias, None, residual, True, out)


def fp8gemm(
    a: torch.Tensor,
    b_t: torch.Tensor,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
    variant: int = -1,
    sfa: torch.Tensor | None = None,
    sfb: torch.Tensor | None = None,
) -> torch.Tensor:
    """fp8 tensor-core GEMM: (scale_a * scale_b) * (a @ b_t.T) -> [M, N] bfloat16, fp32 accumulate.

    a is [M, K] and b_t is [N, K], both float8_e4m3fn, both K-contiguous: b_t is the weight
    as nn.Linear stores it ([out_features, in_features]), and the same layout torch._scaled_mm
    wants for its second operand as b_t.t(). scale_a and scale_b are per-tensor float32 CUDA
    scalars (0-dim or one element), applied in fp32 before the single rounding to bf16.
    Requires N and K multiples of 64 and 16-byte aligned storage. Variant 1 (the default for
    most shapes) takes any M >= 1; variant 0 needs M a multiple of 16; variant 2 (TMA) needs
    M, N multiples of 128, K a multiple of 128 and at least one 128x128 tile per SM. The default
    steps down to the highest variant that accepts the shape.

    MX mode (MXFP8): `sfa` [M, K // 32] and `sfb` [N, K // 32] are uint8 E8M0 block scales
    (exponent + 127, one per 32 consecutive k of a row, as `reference.quantize_mx` produces
    them), and each product a[m, k] * b_t[n, k] is multiplied by 2**(sfa[m, k // 32] - 127) *
    2**(sfb[n, k // 32] - 127) inside the tensor-core instruction, before the fp32 sum. The
    per-tensor scales may then be omitted (1.0). Needs K a multiple of 256 and 16-byte
    aligned scale storage; the same variants apply.
    """
    return _C.fp8gemm(a, b_t, scale_a, scale_b, variant, sfa, sfb)


def fp8_quantize(x: torch.Tensor, mode: str = "tensor") -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a bfloat16 activation x [rows, K] to e4m3 for `fp8gemm`: (q, scale).

    mode="tensor": one float32 scale, shape [1], the smallest power of two 2^e with
    max|x| / 2^e <= 448 (e clamped to [-126, 127]); two launches (an amax pass, then the
    conversion). mode="row": the same per row, shape [rows, 1]; one launch. mode="mx": the
    OCP MX recipe of `reference.quantize_mx`, a uint8 e8m0 scale per 32 values of a row,
    shape [rows, K // 32], what fp8gemm takes as sfa; one launch. Elements are x / scale
    rounded to nearest even, saturating at 448. A power-of-two scale makes the division
    exact, so the e4m3 rounding is the only one; `reference.quantize_fp8_pow2` is the same
    arithmetic. Needs K a multiple of 8 (32 for "mx").
    """
    return _C.fp8_quantize(x, mode)


def fp4_quantize(
    x: torch.Tensor, fmt: str = "nvfp4", scale: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Quantize the rows of x [rows, K] (bfloat16) to fp4: (q, sf, scale).

    q is [rows, K // 2] uint8, two e2m1 values per byte with the lower-index element in the
    low nibble (the layout of torch.float4_e2m1fn_x2). sf is a flat uint8 tensor with the
    block scales in the blocked layout `fp4gemm` and cuBLASLt take (see
    `reference.to_blocked`; rows padded to 128). For fmt="nvfp4" (K a multiple of 64) there is
    an e4m3 scale per 16 values on top of a per-tensor fp32 decode scale: `scale` if given,
    else max|x| / (6 * 448) computed on the device, returned as a 0-dim float32 tensor. For
    fmt="mxfp4" (K a multiple of 128) a ue8m0 power-of-two scale per 32 values with the OCP
    recipe, and scale is None. Both round to nearest even and saturate at 6; the arithmetic
    is `reference.quantize_nvfp4` / `reference.quantize_mxfp4` operation for operation.
    """
    return _C.fp4_quantize(x, fmt, scale)


def fp4gemm(
    a: torch.Tensor,
    b_t: torch.Tensor,
    sfa: torch.Tensor,
    sfb: torch.Tensor,
    scale_a: torch.Tensor | None = None,
    scale_b: torch.Tensor | None = None,
    fmt: str = "nvfp4",
    variant: int = -1,
) -> torch.Tensor:
    """fp4 block-scaled tensor-core GEMM: scale_a * scale_b * (A @ B_t.T) -> [M, N] bfloat16.

    a [M, K // 2] and b_t [N, K // 2] are packed e2m1 (uint8 or torch.float4_e2m1fn_x2, as
    `fp4_quantize` returns them; b_t is the weight as nn.Linear stores it), sfa and sfb their
    block scales in the blocked layout (uint8, or float8_e4m3fn for NVFP4 / float8_e8m0fnu for
    MXFP4), scale_a and scale_b optional per-tensor float32 CUDA scalars (the NVFP4 second
    level). Every product is scaled by its two block scales inside the tensor-core instruction
    and summed in fp32, rounded once to bf16. Requires N a multiple of 64, K a multiple of 256
    and an sm_120a build. Variant 1 takes any M; variant 0 needs M a multiple of 16; variant 2
    (TMA, the default where it applies) M and N multiples of 128 and at least one 128x128
    tile per SM. The default steps down to the highest variant that takes the shape.
    """
    return _C.fp4gemm(a, b_t, sfa, sfb, scale_a, scale_b, fmt, variant)


W4_GROUP = 128  # k per scale (and zero point) of the int4 weights


def w4_quantize(w: torch.Tensor, asym: bool = False
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Round-to-nearest int4 quantization of a [K, N] bfloat16 weight (the [K, N] layout of
    `hgemm`, x @ w), one scale per 128 consecutive k of a column, computed in fp32.

    Symmetric: s = bf16(max|w| / 7), q = clamp(rint(w / s), -8, 7) + 8, w ~ (q - 8) * s.
    Asymmetric (asym=True): lo = min(min w, 0), hi = max(max w, 0), s = bf16((hi - lo) / 15),
    z = clamp(rint(-lo / s), 0, 15), q = clamp(rint(w / s) + z, 0, 15), w ~ (q - z) * s.

    Returns (qweight [K/8, N] int32 with eight consecutive k per word, lowest k in the lowest
    nibble, as GPTQ packs it; scales [K/128, N] bfloat16; zeros [K/128, N] uint8 or None).
    Needs K a multiple of 128 and N a multiple of 16.
    """
    return _C.w4_quantize(w, asym)


def w4_repack(qweight: torch.Tensor) -> torch.Tensor:
    """qweight [K/8, N] int32 (GPTQ packing) -> the layout `w4gemm` reads, int32 [N/16, 2K]:
    per 16 columns x 64 k, 32 lanes x 16 bytes, each lane's bytes the mma.sync fragments it
    dequantizes (docs/design/w4gemm.md). Done once per weight, at load time."""
    return _C.w4_repack(qweight)


class W4Weight:
    """An int4 weight ready for `w4gemm`: the repacked codes, the scales and the optional zero
    points of a [K, N] (in_features x out_features) matrix."""

    def __init__(self, packed: torch.Tensor, scales: torch.Tensor,
                 zeros: torch.Tensor | None = None):
        self.packed = packed
        self.scales = scales
        self.zeros = zeros

    @staticmethod
    def quantize(w: torch.Tensor, asym: bool = False) -> W4Weight:
        """Quantize (w4_quantize) and repack (w4_repack) a [K, N] bfloat16 weight."""
        qweight, scales, zeros = w4_quantize(w, asym)
        return W4Weight(w4_repack(qweight), scales, zeros)

    @property
    def shape(self) -> tuple[int, int]:
        """(K, N) of the weight it replaces."""
        return self.packed.shape[1] // 2, self.packed.shape[0] * 16

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for t in (self.packed, self.scales, self.zeros) if t is not None)


def w4gemm(a: torch.Tensor, w: W4Weight | torch.Tensor, scales: torch.Tensor | None = None,
           zeros: torch.Tensor | None = None, variant: int = -1) -> torch.Tensor:
    """W4A16 GEMM: a [M, K] bfloat16 @ int4 weight [K, N] -> [M, N] bfloat16, fp32 accumulate.

    `w` is a `W4Weight`, or the packed codes with `scales` (and `zeros`) passed separately.
    The weight each product uses is bf16((q - z) * s), rounded once, bit for bit what
    `reference.w4_dequantize` returns. Needs K a multiple of 128, N a multiple of 16, any
    M >= 1. Variant 2 (the default) is the pipelined kernel; 0 and 1 are the ladder below it.
    """
    if isinstance(w, W4Weight):
        if scales is not None or zeros is not None:
            raise ValueError("pass scales and zeros inside the W4Weight, not separately")
        w, scales, zeros = w.packed, w.scales, w.zeros
    if scales is None:
        raise ValueError("w4gemm needs the scales of the packed weight")
    return _C.w4gemm(a, w, scales, zeros, variant)


def rope_append_(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    pos0: int,
    H_q: int,
    H_kv: int,
) -> torch.Tensor:
    """RoPE on q and k plus the K/V cache append, one launch.

    qkv is [B, S, (H_q + 2 H_kv) * D] bf16, the q | k | v columns of one fused projection.
    cos and sin are [P, D] float32 tables in the rotate-half layout (column d pairs with
    d + D/2 and the tables repeat their first half), P >= pos0 + S. k_cache and v_cache are
    [B, H_kv, cap, D] bf16 with cap >= pos0 + S; token s of the input goes to position
    pos0 + s, in place. Returns the rotated q as [B, H_q, S, D] bf16, the layout `attention`
    takes. D a multiple of 16, fp32 math, every byte read once and written once.
    """
    return _C.rope_append_(qkv, cos, sin, k_cache, v_cache, pos0, H_q, H_kv)


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


def rope_append_paged_(
    qkv: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    H_q: int,
    H_kv: int,
) -> torch.Tensor:
    """RoPE plus the K/V append for packed tokens into a paged cache, one launch.

    qkv is [T, (H_q + 2 H_kv) * D] bf16, one row per token of any sequence. Token t is rotated
    at position positions[t] (int32) and its k and v are written to slot slots[t] (int32,
    page_id * page + row) of k_cache and v_cache, [num_pages, H_kv, page, D] bf16 with page a
    power of two; a negative slot skips the write (a padding token). Returns the rotated q as
    [T, H_q, D], the layout `attention_varlen` and `paged_decode` take.
    """
    return _C.rope_append_paged_(qkv, cos, sin, positions, slots, k_cache, v_cache, H_q, H_kv)


def paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    variant: int = -1,
) -> torch.Tensor:
    """One decode step of a batch against a paged K/V cache: softmax(q k^T / sqrt(D)) v per
    sequence, fp32 math, bf16 out.

    q is [B, H_q, D] (one token per sequence), the caches [num_pages, H_kv, page, D] bf16 (page
    a power of two >= 16), block_table [B, max_pages] int32 (sequence b's key j is row j % page
    of page block_table[b, j // page]) and seq_lens [B] int32, the keys of each sequence
    including the token being decoded (append it first). GQA as `attention`. A sequence of
    length 0 gets zeros, which is how a CUDA graph pads a batch. The default (variant 1)
    divides the keys of the whole batch evenly over the SMs whatever the lengths; the
    lengths are read on the device, so a captured step replays correctly as they grow.
    Returns [B, H_q, D] bfloat16.
    """
    return _C.paged_decode(q, k_cache, v_cache, block_table, seq_lens, variant)


def attention_varlen(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    causal: bool = True,
    variant: int = -1,
) -> torch.Tensor:
    """Attention for a packed batch of prompts against a paged K/V cache, no padding.

    q is [T, H_q, D] bf16 with sequence b's new tokens at rows cu_seqlens_q[b] ..
    cu_seqlens_q[b+1]-1 (cu_seqlens_q [B + 1] int32). Their keys must already be in the cache
    (`rope_append_paged_` first): seq_lens[b] (int32) counts them plus any context before
    them. With `causal` the mask is aligned bottom-right, so query i of sequence b sees keys
    j <= seq_lens[b] - q_len_b + i: the usual causal mask for a fresh prompt, and the right
    one for a chunk of a longer prompt. Returns [T, H_q, D] bfloat16, which is [T, H_q * D]
    for the output projection without a transpose.
    """
    return _C.attention_varlen(q, k_cache, v_cache, cu_seqlens_q, seq_lens, block_table, causal,
                               variant)


E4M3_MAX = 448.0


def quantize_fp8(x: torch.Tensor, per_head: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """e4m3 quantization of an attention operand: (x8, scale) with x ~= scale * x8.

    x is [B, H, S, D] (any float dtype). The scale is max|x| / 448 over the whole tensor (a
    one-element float32 tensor), or with `per_head` over each (b, head) slice (a [B, H] float32
    tensor), so the largest element lands on e4m3's largest finite value; the elements are
    x / scale rounded to nearest even. The result feeds `attention_fp8` as q, k or v with its
    scale as the matching descale factor.
    """
    if x.dim() != 4:
        raise ValueError(f"expected [B, H, S, D], got shape {tuple(x.shape)}")
    xf = x.float()
    amax = xf.abs().amax(dim=(2, 3), keepdim=True) if per_head else xf.abs().amax()
    scale = (amax / E4M3_MAX).clamp(min=torch.finfo(torch.float32).tiny)
    x8 = (xf / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    scale = scale.reshape(x.shape[0], x.shape[1]) if per_head else scale.reshape(1)
    return x8, scale.contiguous()


def attention_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    causal: bool = False,
    variant: int = -1,
) -> torch.Tensor:
    """Fused fp8 attention forward: softmax(sq sk q @ k^T / sqrt(D)) @ (sv v), bf16 out.

    q is [B, H_q, S_q, D], k and v are [B, H_kv, S_kv, D], all float8_e4m3fn and contiguous,
    with the same heads, mask and GQA rules as `attention`. The scales are float32 CUDA
    tensors, the descale factors `quantize_fp8` returns: one element each (per tensor), or
    [B, H_q] for q and [B, H_kv] for k and v (per head). Scores, softmax and accumulation are
    fp32; the default variant (1) runs both products on the fp8 tensor cores with the
    probabilities rounded to e4m3, variant 0 is the fp32 baseline. Returns [B, H_q, S_q, D]
    bfloat16.
    """
    return _C.attention_fp8(q, k, v, q_scale, k_scale, v_scale, causal, variant)


def attention_fwd(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, variant: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """The attention forward for training: returns (out, lse).

    `out` is what `attention` returns. `lse` is float32 [B, H_q, S_q], the natural-log
    log-sum-exp of each row of scaled scores, log(sum_j exp(q_i . k_j / sqrt(D))) over the keys
    row i sees: torch's `logsumexp` convention, and what `attention_bwd` recomputes the softmax
    from. Same inputs and variants as `attention`; a decode shape runs the forward's 64-row
    tile instead of the flash-decoding kernel, which has no lse output.
    """
    return _C.attention_fwd(q, k, v, causal, variant)


def attention_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    d_out: torch.Tensor,
    lse: torch.Tensor,
    causal: bool = False,
    variant: int = -1,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Gradients (dq, dk, dv) of out = attention(q, k, v, causal) for the upstream d_out.

    q, k, v, out, d_out are the forward's bfloat16 tensors, lse its float32 log-sum-exp (from
    `attention_fwd`). dq has q's shape; dk and dv have k's, summed over the H_q / H_kv query
    heads that share each k/v head (GQA). fp32 accumulation, bf16 out; D 64 or 128, any S_q and
    S_kv. Variants (docs/design/attention_bwd.md): 0 scalar reference, 1 FlashAttention-2's
    key-tile loop on mma.sync, 2 the sm_120-sized version with a cp.async Q/dO pipeline, 3
    (default) the same with TMA loads and mbarriers in place of the block barrier. Variants 1
    to 3 add dq with fp32 atomics, so the last bits of dq can differ between runs;
    `deterministic=True` computes dq in a separate pass instead (about 40% more tensor work),
    and then two runs agree bit for bit.
    """
    return _C.attention_bwd(q, k, v, out, d_out, lse, causal, variant, deterministic)


class AttentionFunction(torch.autograd.Function):
    """torch.autograd wrapper: forward by `attention_fwd`, backward by `attention_bwd`.

    Saves q, k, v, out and lse (not the S x S probabilities), as FlashAttention does. Use
    `attention_with_grad` rather than calling `apply` directly.
    """

    @staticmethod
    def forward(ctx, q, k, v, causal, variant, bwd_variant, deterministic):
        out, lse = attention_fwd(q, k, v, causal, variant)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.causal, ctx.bwd_variant, ctx.deterministic = causal, bwd_variant, deterministic
        return out

    @staticmethod
    def backward(ctx, d_out):
        q, k, v, out, lse = ctx.saved_tensors
        dq, dk, dv = attention_bwd(q, k, v, out, d_out.contiguous(), lse, ctx.causal,
                                   ctx.bwd_variant, ctx.deterministic)
        return dq, dk, dv, None, None, None, None


def attention_with_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    variant: int = -1,
    bwd_variant: int = -1,
    deterministic: bool = False,
) -> torch.Tensor:
    """Differentiable `attention`: same forward, and `.backward()` runs `attention_bwd`.

    The drop-in for `F.scaled_dot_product_attention(q, k, v, is_causal=causal,
    enable_gqa=True)` in training code, for bfloat16 [B, H, S, D] tensors with D 64 or 128.
    """
    return AttentionFunction.apply(q, k, v, causal, variant, bwd_variant, deterministic)
