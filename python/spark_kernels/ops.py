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
           "w4gemm", "w1gemm", "attention", "attention_bwd", "paged_decode",
           "attention_varlen", "attention_fp8", "sample")


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
    Requires N and K multiples of 64 and 16-byte aligned storage. Variant 1 takes any M >= 1;
    variant 0 needs M a multiple of 16; variant 2 (TMA) needs M, N multiples of 128, K a
    multiple of 128 and at least one 128x128 tile per SM. Variant 3 (the default) takes any
    shape variant 1 does: a TMA tile picked per shape on a Stream-K schedule, the same bits
    every run, and for decode shapes (M <= 64), K % 128 != 0 and MX mode what the default ran
    before it, variant 2 where it takes the shape and variant 1 elsewhere.

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


def w1_quantize(w: torch.Tensor, bits: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """Sign (bits=1) or ternary (bits=2) quantization of a [K, N] bfloat16 weight (the [K, N]
    layout of `hgemm`, x @ w), one scale per 128 consecutive k of a column, computed in fp32.

    s = bf16(mean|w| over the group). bits=1: code = w >= 0, w ~ s * (2 code - 1), so every
    weight becomes +-s. bits=2: code = 0 for w < -s/2, 1 for |w| <= s/2, 2 for w > s/2 (the
    rounded s), w ~ s * (code - 1), so every weight becomes -s, 0 or +s. An all-zero group
    gets s = 0.

    Returns (packed [N, K * bits / 32] int32 with k ascending from the lowest bits of the
    lowest word of row n: bit k % 32 of word k // 32 for bits=1, bits 2 (k % 16) and up of
    word k // 16 for bits=2; scales [K/128, N] bfloat16). Needs K a multiple of 128 and N a
    multiple of 16. `reference.w1_quantize` is the same in plain torch, bit for bit.
    """
    return _C.w1_quantize(w, bits)


class W1Weight:
    """A sign or ternary weight ready for `w1gemm`: the packed codes, the scales and the
    width of the codes (1 or 2 bits) of a [K, N] (in_features x out_features) matrix."""

    def __init__(self, packed: torch.Tensor, scales: torch.Tensor, bits: int = 1):
        self.packed = packed
        self.scales = scales
        self.bits = bits

    @staticmethod
    def quantize(w: torch.Tensor, bits: int = 1) -> W1Weight:
        """Quantize (w1_quantize) a [K, N] bfloat16 weight."""
        packed, scales = w1_quantize(w, bits)
        return W1Weight(packed, scales, bits)

    @property
    def shape(self) -> tuple[int, int]:
        """(K, N) of the weight it replaces."""
        return self.packed.shape[1] * 32 // self.bits, self.packed.shape[0]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.packed, self.scales))


def w1gemm(a: torch.Tensor, w: W1Weight | torch.Tensor, scales: torch.Tensor | None = None,
           bits: int = 1, variant: int = -1) -> torch.Tensor:
    """W1A16 GEMM: a [M, K] bfloat16 @ sign or ternary weight [K, N] -> [M, N] bfloat16, fp32
    accumulate.

    `w` is a `W1Weight` (which carries its own `bits`), or the packed codes with `scales` and
    `bits` passed separately. The weight each product uses is bf16(s * (2 code - 1)) for one
    bit and bf16(s * (code - 1)) for two, what `reference.w1_dequantize` returns. Needs K a
    multiple of 128, N a multiple of 16, any M >= 1. Variant -1 is the top rung.
    """
    if isinstance(w, W1Weight):
        if scales is not None:
            raise ValueError("pass the scales inside the W1Weight, not separately")
        w, scales, bits = w.packed, w.scales, w.bits
    if scales is None:
        raise ValueError("w1gemm needs the scales of the packed weight")
    return _C.w1gemm(a, w, scales, bits, variant)


class W1ExpertWeights:
    """The W1Weights of E experts stacked for `w1gemm_moe`: packed [E, N, K * bits / 32]
    int32, scales [E, K/128, N] bfloat16, one `bits` for all of them."""

    def __init__(self, packed: torch.Tensor, scales: torch.Tensor, bits: int = 1):
        self.packed = packed
        self.scales = scales
        self.bits = bits

    @staticmethod
    def quantize(w: torch.Tensor, bits: int = 1) -> W1ExpertWeights:
        """Quantize an [E, K, N] bfloat16 weight stack expert by expert (w1_quantize) and
        stack the results."""
        if w.dim() != 3:
            raise ValueError(f"expected an [E, K, N] weight stack, got shape {tuple(w.shape)}")
        pairs = [w1_quantize(w[e], bits) for e in range(w.shape[0])]
        packed = torch.stack([p for p, _ in pairs])
        scales = torch.stack([s for _, s in pairs])
        return W1ExpertWeights(packed, scales, bits)

    @property
    def num_experts(self) -> int:
        return self.packed.shape[0]

    @property
    def shape(self) -> tuple[int, int, int]:
        """(E, K, N) of the weight stack it replaces."""
        return (self.packed.shape[0], self.packed.shape[2] * 32 // self.bits,
                self.packed.shape[1])

    def expert(self, e: int) -> W1Weight:
        """Expert e's weight on its own, for `w1gemm`."""
        return W1Weight(self.packed[e], self.scales[e], self.bits)

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.packed, self.scales))


def w1gemm_moe(a: torch.Tensor, w: W1ExpertWeights, offsets: torch.Tensor,
               variant: int = -1) -> torch.Tensor:
    """`w1gemm` grouped over the E experts of `w`: a [T, K] bfloat16 with its rows sorted by
    expert, expert e owning rows offsets[e]..offsets[e+1]-1 -> [T, N] bfloat16, row t the
    product of a[t] with its expert's weight.

    `offsets` is int32 [E + 1] on a's device with offsets[E] == T; an expert with no rows
    (offsets[e] == offsets[e+1]) is skipped. One launch for every expert, the same numbers as
    calling `w1gemm` on each expert's rows with `w.expert(e)`.
    """
    if not isinstance(w, W1ExpertWeights):
        raise TypeError("w1gemm_moe takes a W1ExpertWeights (W1ExpertWeights.quantize)")
    return _C.w1gemm_moe(a, w.packed, w.scales, offsets, w.bits, variant)


def grouped_gemm(a: torch.Tensor, w: torch.Tensor, offsets: torch.Tensor,
                 scale_a: torch.Tensor | None = None,
                 scale_b: torch.Tensor | None = None) -> torch.Tensor:
    """Every expert's GEMM of an MoE layer in one launch: a [T, K] with its rows sorted by
    expert, expert e owning rows offsets[e]..offsets[e+1]-1, times w [E, N, K] (each expert's
    weight as nn.Linear stores it) -> [T, N] bfloat16, row t = a[t] @ w[e].T, fp32 accumulate.

    `offsets` is int32 [E + 1] on a's device with offsets[0] == 0 and offsets[E] == T, empty
    experts allowed; the host never reads it, so the call does not synchronize. a and w are
    both bfloat16, or both float8_e4m3fn with optional float32 scales: scale_a (one element)
    and scale_b (one per expert, or one), row t then being scale_a * scale_b[e] * (a[t] @
    w[e].T). Needs E <= 512, N a multiple of 128, K a multiple of 64 (bf16) or 128 (fp8).
    """
    return _C.grouped_gemm(a, w, offsets, scale_a, scale_b)


def moe_offsets(expert_ids: torch.Tensor, num_experts: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Sort the (token, slot) pairs of a top-k routing by expert for `grouped_gemm`.

    expert_ids is [T, k] integer. Returns (order, offsets): order [T * k] int64 is the
    flattened (token * k + slot) index of each sorted row (stable, so a token's rows keep
    their slot order within an expert), offsets [E + 1] int32. Everything stays on the
    device: a[order // k] is the sorted activation.
    """
    flat = expert_ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=num_experts)
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32, device=flat.device)
    offsets[1:] = torch.cumsum(counts, 0).to(torch.int32)
    return order, offsets


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
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool = False, variant: int = -1,
    window: int = 0, sinks: torch.Tensor | None = None,
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

    `window` > 0 is a sliding window (Mistral, Gemma, gpt-oss): each query sees at most the
    last `window` keys up to its own position, its own included. It is causal and aligned to
    the last key whatever `causal` says: query i sits at key position i + S_kv - S_q (S_q <=
    S_kv), so a prefill (S_q = S_kv) gets the causal mask with a left edge and a decode step
    the last `window` keys of the cache. The kernels skip the k/v tiles outside every row's
    window. `sinks` is a float32 [H_q] tensor of per-head logits added to every softmax
    denominator with no value vector (gpt-oss's attention sinks):
    out_i = sum_j exp(s_ij - m) v_j / (exp(sinks[h] - m) + sum_j exp(s_ij - m)).
    """
    if window or sinks is not None:
        return _C.attention_window(q, k, v, causal, window, sinks, variant)
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
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """RoPE plus the K/V append for packed tokens into a paged cache, one launch.

    qkv is [T, (H_q + 2 H_kv) * D] bf16, one row per token of any sequence. Token t is rotated
    at position positions[t] (int32) and its k and v are written to slot slots[t] (int32,
    page_id * page + row) of k_cache and v_cache, [num_pages, H_kv, page, D] bf16 with page a
    power of two; a negative slot skips the write (a padding token). Returns the rotated q as
    [T, H_q, D], the layout `attention_varlen` and `paged_decode` take.

    With float8_e4m3fn caches, k_scale and v_scale (float32 [H_kv], one per kv head of this
    layer) are required: an element x of kv head h is stored as e4m3(x / scale[h]), rounded to
    nearest even and saturated at +-448, k rounded once from its fp32 rotation. The attention
    ops take the same scales (`quantize_kv_cache` is the same rounding in torch).
    """
    return _C.rope_append_paged_(qkv, cos, sin, positions, slots, k_cache, v_cache, H_q, H_kv,
                                 k_scale, v_scale)


def paged_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    variant: int = -1,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    window: int = 0,
    sinks: torch.Tensor | None = None,
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

    The caches may be float8_e4m3fn (`rope_append_paged_` with scales), read as
    k_scale[h] * k_cache and v_scale[h] * v_cache with k_scale, v_scale float32 [H_kv]: half
    the bytes of a bf16 cache, the same variants. The e4m3 values are converted to bf16
    exactly, so the only rounding added is the cache's own.

    `window` > 0: sequence b's token sees only keys [seq_lens[b] - window, seq_lens[b]), and
    the default variant splits and streams only those. `sinks` (float32 [H_q]) as in
    `attention`.
    """
    if window or sinks is not None:
        return _C.paged_decode_window(q, k_cache, v_cache, block_table, seq_lens, window, sinks,
                                      variant, k_scale, v_scale)
    return _C.paged_decode(q, k_cache, v_cache, block_table, seq_lens, variant, k_scale,
                           v_scale)


def attention_varlen(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table: torch.Tensor,
    causal: bool = True,
    variant: int = -1,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    window: int = 0,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    """Attention for a packed batch of prompts against a paged K/V cache, no padding.

    q is [T, H_q, D] bf16 with sequence b's new tokens at rows cu_seqlens_q[b] ..
    cu_seqlens_q[b+1]-1 (cu_seqlens_q [B + 1] int32). Their keys must already be in the cache
    (`rope_append_paged_` first): seq_lens[b] (int32) counts them plus any context before
    them. With `causal` the mask is aligned bottom-right, so query i of sequence b sees keys
    j <= seq_lens[b] - q_len_b + i: the usual causal mask for a fresh prompt, and the right
    one for a chunk of a longer prompt. Returns [T, H_q, D] bfloat16, which is [T, H_q * D]
    for the output projection without a transpose. float8_e4m3fn caches take k_scale and
    v_scale as `paged_decode` does.

    `window` > 0 adds a left edge to the causal mask (and implies it): query i sees keys
    p - window < j <= p for its position p = seq_lens[b] - q_len_b + i. `sinks` (float32
    [H_q]) as in `attention`.
    """
    if window or sinks is not None:
        return _C.attention_varlen_window(q, k_cache, v_cache, cu_seqlens_q, seq_lens,
                                          block_table, causal, window, sinks, variant, k_scale,
                                          v_scale)
    return _C.attention_varlen(q, k_cache, v_cache, cu_seqlens_q, seq_lens, block_table, causal,
                               variant, k_scale, v_scale)


def _head_scale(scale: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """scale [H_kv] shaped to broadcast over dim 1 of x."""
    if x.dim() not in (3, 4):
        raise ValueError(f"expected [T, H_kv, D] or [num_pages, H_kv, page, D], got {x.shape}")
    return scale.float().view([-1] + [1] * (x.dim() - 2))


def quantize_kv_cache(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """x as e4m3 with one scale per kv head, the rounding `rope_append_paged_` applies to an
    fp8 cache: (x / scale[h]) clamped to +-448, then to float8_e4m3fn (round to nearest even).
    x is [T, H_kv, D] (tokens) or [num_pages, H_kv, page, D] (a cache), any float dtype;
    `scale` is float32 [H_kv]."""
    return (x.float() / _head_scale(scale, x)).clamp(-E4M3_MAX, E4M3_MAX).to(
        torch.float8_e4m3fn)


def dequantize_kv_cache(x8: torch.Tensor, scale: torch.Tensor,
                        dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """The values an fp8 cache stands for: x8 (float8_e4m3fn, kv heads at dim 1 as in
    `quantize_kv_cache`) times its kv head's scale, one float32 rounding, then `dtype`."""
    return (x8.float() * _head_scale(scale, x8)).to(dtype)


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


# ---------------------------------------------------------------------------
# Linear cross-entropy (src/kernels/cross_entropy.cu, docs/design/cross_entropy.md)
# ---------------------------------------------------------------------------
CE_CHUNK = 2048  # rows of logits (or of their gradient) held at once by variants 1 and 2
CE_SLAB = 16384  # vocabulary rows per transposed slab of the gradient for dw += g^T h
CE_STATS_CHUNK = 8192  # rows per STATS GEMM in variant 2's forward, so h stays in L2


def _ce_reference(h, w, targets, ignore_index, reduction):
    """Variant 0, torch's spelling: the bf16 logits, then cross-entropy in fp32."""
    logits = h @ w.t()
    return torch.nn.functional.cross_entropy(logits.float(), targets, ignore_index=ignore_index,
                                             reduction=reduction)


def _ce_counts(targets, ignore_index):
    return (targets != ignore_index).sum()


def _ce_scale(targets, ignore_index, reduction):
    """d loss / d loss_row for a row that counts, float32 [1] on the device: 1 / rows that
    count for "mean" (1 when none does, so the gradient is zero), 1 for "sum"."""
    if reduction == "sum":
        return torch.ones(1, device=targets.device, dtype=torch.float32)
    return (1.0 / _ce_counts(targets, ignore_index).clamp(min=1).float()).reshape(1)


def _ce_reduce(loss_rows, targets, ignore_index, reduction):
    total = loss_rows.sum()
    if reduction == "sum":
        return total
    return total / _ce_counts(targets, ignore_index)  # nan when no row counts, as torch


def _ce_backprop_chunk(g, h_c, w, dh_c, dw, first, slab_buf):
    """dh_c = g @ w and dw (+)= g^T @ h_c for one chunk's logit gradient g [C, V] bf16, both
    by hgemm. hgemm takes its A operand K-contiguous, so g^T goes through `slab_buf` (flat,
    CE_SLAB x the chunk rounded up to 64) one vocabulary slab at a time, its columns padded
    to a multiple of 64 with zeros, and h_c gets zero rows to match. dw accumulates in bf16
    across chunks, as Liger's does."""
    C, V = g.shape
    hgemm(g, w, out=dh_c)
    c_pad = (C + 63) // 64 * 64
    if c_pad != C:
        h_c = torch.nn.functional.pad(h_c, (0, 0, 0, c_pad - C))
    slab = slab_buf.numel() // c_pad
    for v0 in range(0, V, slab):
        v1 = min(V, v0 + slab)
        gt = _C.ce_transpose(g, v0, v1, c_pad, slab_buf[: (v1 - v0) * c_pad].view(v1 - v0, c_pad))
        hgemm(gt, h_c, residual=None if first else dw[v0:v1], out=dw[v0:v1])


def _ce_chunks(T, chunk):
    return [(t0, min(T, t0 + chunk)) for t0 in range(0, T, chunk)]


def _ce_stats(h, w, targets, ignore_index):
    """(lse, loss) float32 [T] by the STATS GEMM, CE_STATS_CHUNK rows at a time: the GEMM
    walks every row of h against each strip of w, and past 8192 rows (64 MB at D = 4096) h
    no longer stays in the 96 MB L2 while w streams through."""
    T = h.shape[0]
    if T <= CE_STATS_CHUNK:
        return _C.ce_stats(h, w, targets, ignore_index)
    parts = [_C.ce_stats(h[t0:t1], w, targets[t0:t1], ignore_index)
             for t0, t1 in _ce_chunks(T, CE_STATS_CHUNK)]
    return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])


def _ce_buffers(h, V, T, chunk, need_grad=True):
    """The one chunk of logits (or gradient) and, for a backward, the transposed slab."""
    rows = min(chunk, T)
    g = torch.empty(rows, V, device=h.device, dtype=h.dtype)
    if not need_grad:
        return g, None
    slab_buf = torch.empty(min(CE_SLAB, V) * ((rows + 63) // 64 * 64), device=h.device,
                           dtype=h.dtype)
    return g, slab_buf


class LinearCrossEntropyChunked(torch.autograd.Function):
    """Variant 1, Liger's recipe: the gradients are computed in the forward, a chunk of rows
    at a time (the logits by `ce_logits`, loss and gradient rows in place by `ce_rows`, dh
    and dw by `hgemm`), and the backward only scales them by the upstream gradient."""

    @staticmethod
    def forward(ctx, h, w, targets, ignore_index, reduction, chunk, need_grad):
        T, V = h.shape[0], w.shape[0]
        scale = _ce_scale(targets, ignore_index, reduction) if need_grad else None
        logits, slab_buf = _ce_buffers(h, V, T, chunk, need_grad)
        losses = []
        dh = dw = None
        if need_grad:
            dh = torch.empty_like(h)
            dw = torch.empty_like(w)
        for t0, t1 in _ce_chunks(T, chunk):
            lg = logits[: t1 - t0]
            _C.ce_logits(h[t0:t1], w, lg)
            loss_c, _ = _C.ce_rows(lg, targets[t0:t1], ignore_index, scale)
            losses.append(loss_c)
            if need_grad:
                _ce_backprop_chunk(lg, h[t0:t1], w, dh[t0:t1], dw, t0 == 0, slab_buf)
        del logits, slab_buf
        loss = _ce_reduce(torch.cat(losses), targets, ignore_index, reduction)
        ctx.save_for_backward(dh, dw)
        return loss

    @staticmethod
    def backward(ctx, grad):
        dh, dw = ctx.saved_tensors
        if grad.item() != 1.0:  # one sync, as Liger's check
            dh = dh * grad.to(dh.dtype)
            dw = dw * grad.to(dw.dtype)
        return dh, dw, None, None, None, None, None


class LinearCrossEntropyFused(torch.autograd.Function):
    """Variant 2: the forward never writes logits (`ce_stats`: per-tile (max, sum) partials
    from the GEMM's epilogue, merged into the log-sum-exp) and saves only the [T] lse. The
    backward recomputes a chunk of logits with the softmax-minus-one-hot folded into the
    GEMM's epilogue (`ce_grad`), then dh and dw as variant 1 does."""

    @staticmethod
    def forward(ctx, h, w, targets, ignore_index, reduction, chunk, need_grad):
        lse, loss_rows = _ce_stats(h, w, targets, ignore_index)
        if need_grad:
            ctx.save_for_backward(h, w, targets, lse)
        ctx.ignore_index, ctx.reduction, ctx.chunk = ignore_index, reduction, chunk
        return _ce_reduce(loss_rows, targets, ignore_index, reduction)

    @staticmethod
    def backward(ctx, grad):
        h, w, targets, lse = ctx.saved_tensors
        T, V = h.shape[0], w.shape[0]
        # The upstream gradient joins the reduction scale on the device: no sync.
        scale = _ce_scale(targets, ctx.ignore_index, ctx.reduction) * grad.float().reshape(1)
        dh = torch.empty_like(h)
        dw = torch.empty_like(w)
        g, slab_buf = _ce_buffers(h, V, T, ctx.chunk)
        for t0, t1 in _ce_chunks(T, ctx.chunk):
            gc = g[: t1 - t0]
            _C.ce_grad(h[t0:t1], w, targets[t0:t1], lse[t0:t1], scale, ctx.ignore_index, gc)
            _ce_backprop_chunk(gc, h[t0:t1], w, dh[t0:t1], dw, t0 == 0, slab_buf)
        return dh, dw, None, None, None, None, None


def linear_cross_entropy(
    h: torch.Tensor,
    w: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    variant: int = -1,
    chunk: int | None = None,
) -> torch.Tensor:
    """`F.cross_entropy(h @ w.T, targets)` for an lm_head without the [T, V] logits.

    h [T, D] and w [V, D] (nn.Linear's layout) bfloat16, targets int64 [T]; rows whose
    target is `ignore_index` do not count. reduction "mean" (over the rows that count) or
    "sum". Differentiable in h and w. Variants (docs/design/cross_entropy.md):

      0  torch's spelling: the bf16 logits, F.cross_entropy on them in fp32
      1  chunked over T, Liger's recipe on these kernels: per chunk of `chunk` rows the
         logits, a fused row kernel for the loss and the gradient in place, dh and dw by
         hgemm, all inside the forward; holds one chunk of logits
      2  the forward with no logits (the GEMM's epilogue keeps a per-row max and sum per
         128-column tile), the backward recomputing a chunk of logits with softmax minus
         one-hot in the GEMM's epilogue; between the two passes it holds only a [T] lse

    -1 picks 1 when a gradient is needed (three GEMMs against variant 2's four) and 2 when
    not. Variants 1 and 2 need V a multiple of 128 and D a multiple of 64; -1 falls back to
    0 otherwise. dw is accumulated in bf16 across chunks (one rounding per chunk).
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")
    V, D = w.shape
    fits = V % 128 == 0 and D % 64 == 0
    need_grad = torch.is_grad_enabled() and (h.requires_grad or w.requires_grad)
    if variant == -1:
        variant = 0 if not fits else 1 if need_grad else 2
    if variant == 0:
        return _ce_reference(h, w, targets, ignore_index, reduction)
    if variant not in (1, 2):
        raise ValueError(f"linear_cross_entropy: variant must be -1, 0, 1 or 2, got {variant}")
    if not fits:
        raise ValueError(f"linear_cross_entropy variant {variant} needs V % 128 == 0 and "
                         f"D % 64 == 0, got V = {V}, D = {D}")
    chunk = CE_CHUNK if chunk is None else int(chunk)
    if chunk < 1:
        raise ValueError(f"chunk must be positive, got {chunk}")
    fn = LinearCrossEntropyChunked if variant == 1 else LinearCrossEntropyFused
    return fn.apply(h, w, targets, ignore_index, reduction, chunk, need_grad)


# ---------------------------------------------------------------------------
# CPU-side ops: the GB10's Arm cores (include/spark/cpu.h, docs/design/grace.md)
# ---------------------------------------------------------------------------
def cpu_w1gemv(a: torch.Tensor, w: W1Weight, threads: int = 1, variant: int = -1) -> torch.Tensor:
    """`w1gemm` on the CPU cores: a [K] or [M, K] bfloat16 @ the sign or ternary weight ->
    [N] or [M, N] bfloat16, fp32 accumulate per group, one GEMV per row of `a`.

    Every tensor must be on the CPU (the Spark's unified memory means a `.cpu()` of a weight
    is a copy today, so quantize on the host or move once). `threads` splits the rows over
    std::threads in a fixed partition. Variant -1 is the top rung the build has: 0 scalar,
    1 NEON fp32 (exact up to summation order), 2 NEON int8 dot products (the activations
    rounded to int8 per group).
    """
    return _C.cpu_w1gemv(a, w.packed, w.scales, w.bits, threads, variant)


def cpu_route(logits: torch.Tensor, top_k: int, bias: torch.Tensor | None = None,
              scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """`spark_kernels.moe.route` on the CPU cores, token by token: fp32 logits [E] or [T, E]
    -> (expert ids int32, gate weights fp32) of shape [k] or [T, k]. The top_k of
    sigmoid(logits) + bias picks, the unbiased sigmoid scores renormalized to sum `scale`
    weight. A partial selection, no sort of the E experts."""
    return _C.cpu_route(logits, top_k, bias, scale)


def cpu_sample(logits: torch.Tensor, temperature: float, top_k: int = 0, top_p: float = 1.0,
               seed: int = 0, offset: int = 0) -> torch.Tensor:
    """One token per bfloat16 logit row ([V] or [B, V]) on the CPU cores, int64. temperature
    <= 0 is argmax (the first index of the max); otherwise top-k by a partial selection, top-p
    over what top-k kept, and a draw from splitmix64(seed, offset + row). Reproducible for a
    (seed, offset), but not the GPU `sample`'s Philox stream: the two do not agree token for
    token."""
    return _C.cpu_sample(logits, temperature, top_k, top_p, seed, offset)


def cpu_bandwidth_probe(bytes: int, threads: int, iters: int = 5) -> float:
    """GB/s the CPU cores stream from a private buffer of `bytes` with 16-byte NEON loads on
    `threads` threads: the median over `iters` sweeps of the slowest thread's wall time. On
    the GB10 this is the share of the 273 GB/s the Arm side gets on its own."""
    return _C.cpu_bandwidth_probe(bytes, threads, iters)
# ---- unified memory experiments (docs/design/unified.md) -----------------------------------
# The GB10 shares one pool of memory between its CPU and GPU; these ops let a kernel read a
# tensor wherever it lives so the bench and the tests can ask what that costs. Unmeasured.

UNIFIED_KERNELS = ("unified_read", "unified_gemv")


def unified_num_variants(name: str) -> int:
    """Number of variants of `unified_read` or `unified_gemv` (their ladders are separate from
    `num_variants`'s list; see docs/design/unified.md)."""
    if name not in UNIFIED_KERNELS:
        raise ValueError(f"unknown unified kernel {name!r}; expected one of {UNIFIED_KERNELS}")
    return _C.unified_num_variants(name)


def unified_device_caps() -> dict[str, bool | int]:
    """What the current CUDA device reports about unified memory, from cudaDeviceGetAttribute.

    Keys: `device`, `sm_count`, and the booleans `unified_addressing`, `integrated`,
    `can_map_host_memory`, `pageable_memory_access` (the GPU can read plain malloc / torch CPU
    memory in place: HMM or ATS), `pageable_memory_access_uses_host_page_tables`,
    `concurrent_managed_access`, `direct_managed_mem_access_from_host`,
    `host_native_atomic_supported`, `can_use_host_pointer_for_registered_mem`.
    """
    return _C.unified_device_caps()


def unified_read_sum(x: torch.Tensor, variant: int = -1) -> torch.Tensor:
    """Stream a bfloat16 tensor through the GPU and return its sum as a float64 0-d tensor.

    `x` may be a CUDA tensor, a pinned CPU tensor (`pin_memory=True`: cudaHostAlloc memory,
    which the GPU reads in place), or a plain CPU tensor when `unified_device_caps()` reports
    `pageable_memory_access`. The kernel writes one fp32 partial per block, summed here in
    fp64. Variants: 0 one element per thread, 1 one 16-byte load per thread, 2 four 16-byte
    loads in flight per thread; -1 is the top rung.
    """
    return _C.unified_read_sum(x, variant).double().sum()


def unified_gemv(a: torch.Tensor, w: torch.Tensor, variant: int = -1) -> torch.Tensor:
    """Decode GEMV y = w @ a for a [K] or [1, K] bfloat16 CUDA tensor and w [N, K] bfloat16
    (K contiguous, a Linear's stored layout) that may live in CUDA, pinned CPU, or (with
    `pageable_memory_access`) plain CPU memory; fp32 accumulation, [N] or [1, N] bfloat16.

    Needs K a multiple of 8 and 16-byte aligned storage. Variants: 0 one warp per row, 1 one
    warp per 4 rows with the next step's loads in flight, 2 variant 1 on a persistent grid
    sized from the SM count; -1 is the top rung.
    """
    return _C.unified_gemv(a, w, variant)
def w1moe_decode_num_variants() -> int:
    """Number of implementation variants of `w1moe_decode` (docs/design/moe_spark.md)."""
    return _C.w1moe_decode_num_variants()


def w1moe_decode(
    x: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_weights: torch.Tensor,
    experts: W1ExpertWeights,
    down: W1ExpertWeights,
    variant: int = -1,
) -> torch.Tensor:
    """The routed-expert half of an MoE decode step in one launch, for up to 8 tokens.

    x is [T, K] bfloat16 (T <= 8), expert_ids int32 [T, topk] the experts each token goes to,
    expert_weights [T, topk] their routing weights (converted to bfloat16 here), `experts` the
    gate|up weights of all E experts as a `W1ExpertWeights` over [E, K, 2I] (gate columns then
    up columns, the layout `MoEBlock1Bit` holds) and `down` the down weights over [E, I, K].
    Returns [T, K] bfloat16: for every token the sum over its experts of
    weight * (silu(x @ gate) * (x @ up)) @ down, fp32 accumulation, with the gate, up and
    SwiGLU values rounded to bfloat16 where the sorted `w1gemm_moe` path rounds them. The
    shared expert is not included.

    Variant 0 runs three plain passes with the intermediate in global memory; variants 1 and
    2 keep it in shared memory, one persistent block per (expert, 128 intermediate columns),
    and add into y with fp32 atomics, so the last bits can differ between runs. Needs
    K % 256 == 0, K <= 16384, I % 128 == 0 and T * topk <= 64.
    """
    if not isinstance(experts, W1ExpertWeights) or not isinstance(down, W1ExpertWeights):
        raise TypeError("w1moe_decode takes two W1ExpertWeights (gate|up and down)")
    if experts.bits != down.bits:
        raise ValueError(f"gate|up is {experts.bits}-bit and down {down.bits}-bit")
    ids = expert_ids.to(device=x.device, dtype=torch.int32).contiguous()
    w = expert_weights.to(device=x.device, dtype=torch.bfloat16).contiguous()
    return _C.w1moe_decode(x, ids, w, experts.packed, experts.scales, down.packed, down.scales,
                           experts.bits, variant)
# ---- launch configuration: what the decode kernels decided about the device --------------


def device_shape() -> dict[str, int | float | bool]:
    """The device as the decode kernels see it (src/kernels/launch_config.cuh).

    Keys: ``sm_count`` (``SPARK_FORCE_SMS`` replaces it, and ``forced`` says so), ``l2_bytes``,
    ``spec_gbps`` (DRAM spec from the memory clock and bus width: 1792 on the RTX 5090, 273 on
    the GB10), ``bytes_in_flight`` (what the device must keep outstanding to stream at that
    bandwidth: the 5090's measured 1.5 MB, bandwidth times an assumed 800 ns on anything else,
    the GB10's loaded latency being unmeasured). Needs a CUDA device.
    """
    return dict(_C.device_shape())


def split_k_for(weight_bytes: int, tiles: int, max_split: int, per_cta_in_flight: int,
                slots: int, device: str = "") -> int:
    """The decode kernels' K split: slices per tile so that ``tiles * split`` CTAs, each with
    ``per_cta_in_flight`` bytes outstanding, carry the device's bytes in flight, then (only
    past one wave of ``slots`` resident CTAs) a split whose last wave is full. ``device`` is
    ``""`` for the current GPU or ``"gb10"`` / ``"rtx5090"`` for the named shapes, which need
    no GPU. On the 5090 shape this is the old ``cdiv(64, strips)`` of hgemm_decode at 24 KB
    per CTA and the old ``cdiv(128, tiles)`` of w4gemm at 12 KB.
    """
    return _C.split_k_for(weight_bytes, tiles, max_split, per_cta_in_flight, slots, device)


def need_ctas(per_cta_in_flight: int, device: str = "") -> int:
    """CTAs the bandwidth rule asks for when each keeps ``per_cta_in_flight`` bytes in flight."""
    return _C.need_ctas(per_cta_in_flight, device)


def fits_l2(bytes_: int, device: str = "") -> bool:
    """Whether a weight of ``bytes_`` can stay L2-resident between two decode launches: at most
    three quarters of the L2 (24 MB on the GB10, 96 on the RTX 5090)."""
    return _C.fits_l2(bytes_, device)


def decode_force_split_k(split: int) -> None:
    """Force the K split of every decode kernel that has one; 0 restores the rule. For sweeps."""
    _C.decode_force_split_k(split)
