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

KERNELS = ("bandwidth", "rmsnorm", "swiglu", "softmax", "sgemm", "hgemm", "fp8gemm", "attention",
           "paged_decode", "attention_varlen")


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
