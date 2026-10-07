"""A mixture-of-experts block whose routed experts live at 1 (or 2) bits per weight, in the
format `w1gemm_moe` reads, with DeepSeek V4.1 Flash's shapes as the worked example.

DeepSeek V4.1 Flash routes every token to 6 of 384 experts plus one shared expert, each a
SwiGLU MLP from hidden 5120 to 2304 and back. At bf16 one layer's routed experts are 27 GB;
at 1 bit they are 1.7 GB of packed signs plus a bf16 scale per 128 k of a column, so a layer
(or a handful of layers) fits a 32 GB card and a decode step streams only the experts the
batch touched. `MoEConfig.bytes_per_layer` gives the footprint per format so the README can
quote it.

The routing and the sort are plain torch and run on the CPU:

    route(logits, top_k, bias, scale)   DeepSeek-V3-style sigmoid gating: the bias (the
                                        router's e_score_correction_bias) only picks the
                                        experts, the gate weights are the unbiased sigmoid
                                        renormalized to sum 1 per token, times `scale`
    sort_by_expert(expert_ids, E)       the (token, slot) pairs sorted by expert, as the row
                                        permutation, per-expert offsets and the token each
                                        sorted row came from, which is what `w1gemm_moe`
                                        takes (rows grouped by expert, offsets[E] == rows)

`MoEBlock1Bit` holds the routed experts as two `W1ExpertWeights` (gate and up fused into one
[E, hidden, 2 inter], down [E, inter, hidden]), the shared expert as two `W1Weight`, the
router in bf16 with an fp32 bias, and runs a forward of two `w1gemm_moe` launches and two
`w1gemm` launches around the swiglu. `forward_reference` is the same block in plain torch on
the dequantized experts, one expert at a time, for the tests.

The quantizer contract of `W1Weight.quantize` is restated here in `_quantize_experts_ref` so
the block and the tests build the same bytes without the extension: at 1 bit a column's 128
k share one bf16 scale (the mean |w| of the group) and the code is the sign, packed 32 k per
int32 with k ascending from the low bit; at 2 bits the code is 0, 1 or 2 for w below -s/2,
within +-s/2 and above s/2, 16 k per int32, and the weight stands for bf16(s * (code - 1)).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

GROUP = 128  # k per scale, the w1gemm group
FORMATS = ("w1", "bf16", "int4", "mxfp4")


@dataclass(frozen=True)
class MoEConfig:
    """One MoE layer: `n_routed` experts of which `top_k` take each token, `n_shared` always
    on, every expert a SwiGLU MLP hidden -> inter -> hidden, routed experts at `bits` per
    weight. `route_scale` is the model's routed_scaling_factor."""

    hidden: int
    n_routed: int
    n_shared: int
    top_k: int
    inter: int
    bits: int = 1
    route_scale: float = 1.0

    def expert_params(self) -> int:
        """Weights of one expert: gate, up and down."""
        return 3 * self.hidden * self.inter

    def expert_bytes(self, fmt: str | None = None) -> int:
        """Bytes of one routed expert in `fmt`: "w1" (this config's bits, a bf16 scale per 128
        k of a column), "int4" (4 bits, a bf16 scale per 128), "mxfp4" (4 bits, an e8m0 byte
        per 32) or "bf16"."""
        fmt = fmt or "w1"
        n = self.expert_params()
        if fmt == "bf16":
            return 2 * n
        if fmt == "mxfp4":
            return n // 2 + n // 32
        bits = self.bits if fmt == "w1" else 4 if fmt == "int4" else None
        if bits is None:
            raise ValueError(f"unknown format {fmt!r}, one of {FORMATS}")
        return n * bits // 8 + 2 * n // GROUP

    def bytes_per_layer(self, fmt: str | None = None) -> int:
        """The routed experts of one layer in `fmt` (the shared expert and the router are
        bf16 and small: a few MB against the GB here, left out)."""
        return self.n_routed * self.expert_bytes(fmt)


@dataclass(frozen=True)
class ModelConfig:
    """The whole model around the MoE layers, for the total footprint and the bench's
    tokens/s. Every one of `layers` is taken as an MoE layer."""

    moe: MoEConfig
    layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int

    def bytes_total(self, fmt: str | None = None) -> int:
        """The routed experts of every layer in `fmt`."""
        return self.layers * self.moe.bytes_per_layer(fmt)


DEEPSEEK_V41_FLASH = MoEConfig(hidden=5120, n_routed=384, n_shared=1, top_k=6, inter=2304,
                               bits=1)
DEEPSEEK_V41_FLASH_MODEL = ModelConfig(moe=DEEPSEEK_V41_FLASH, layers=40, n_heads=64,
                                       n_kv_heads=1, head_dim=512, vocab=129280)


# ---------------------------------------------------------------------------
# Routing and the sort, plain torch
# ---------------------------------------------------------------------------
def route(logits: torch.Tensor, top_k: int, bias: torch.Tensor | None = None,
          scale: float = 1.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid gating over [T, E] fp32 router logits -> (expert ids [T, top_k] int32, gate
    weights [T, top_k] fp32).

    The experts are the top_k of sigmoid(logits) + bias (DeepSeek's auxiliary-loss-free
    balancing bias, which only picks); their weights are the unbiased sigmoid scores,
    renormalized to sum 1 over the token's experts and multiplied by `scale`."""
    scores = torch.sigmoid(logits.float())
    pick = scores if bias is None else scores + bias.float().reshape(1, -1)
    ids = pick.topk(top_k, dim=-1).indices
    w = scores.gather(-1, ids)
    w = w / w.sum(-1, keepdim=True).clamp(min=torch.finfo(torch.float32).tiny) * scale
    return ids.to(torch.int32), w


def sort_by_expert(expert_ids: torch.Tensor, n_experts: int
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The (token, slot) pairs of [T, top_k] expert ids in expert order.

    Returns `perm` [T*top_k] int64 (the flat (token, slot) index of each sorted row: row r of
    the sorted activations is token perm[r] // top_k, slot perm[r] % top_k), `offsets` [E+1]
    int32 (rows of expert e are offsets[e]..offsets[e+1], offsets[E] == T*top_k; an expert
    nobody picked has an empty range) and `token_of` [T*top_k] int64 (perm // top_k, the row
    to gather from the input and to scatter-add to in the output). The sort is stable, so a
    token's rows keep their slot order inside an expert."""
    flat = expert_ids.reshape(-1).to(torch.int64)
    _, perm = torch.sort(flat, stable=True)
    counts = torch.bincount(flat, minlength=n_experts)
    offsets = torch.zeros(n_experts + 1, dtype=torch.int64, device=flat.device)
    offsets[1:] = counts.cumsum(0)
    return perm, offsets.to(torch.int32), perm // expert_ids.shape[1]


# ---------------------------------------------------------------------------
# The quantizer contract, restated in torch so the block builds without the extension
# ---------------------------------------------------------------------------
def _to_int32(words: torch.Tensor) -> torch.Tensor:
    """int64 words in 0..2^32-1 as their int32 bit patterns."""
    return torch.where(words >= 2**31, words - 2**32, words).to(torch.int32)


def _from_int32(packed: torch.Tensor) -> torch.Tensor:
    """int32 bit patterns as int64 words in 0..2^32-1."""
    w = packed.to(torch.int64)
    return torch.where(w < 0, w + 2**32, w)


def _quantize_ref(w: torch.Tensor, bits: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """[K, N] bf16 -> (packed int32 [N, K*bits/32], scales bf16 [K/128, N]) by the w1gemm
    contract: s = bf16(mean |w| over 128 k of a column); at 1 bit code = (w >= 0), one k per
    bit with k ascending from the low bit; at 2 bits code 0 / 1 / 2 for w < -s/2, |w| <= s/2,
    w > s/2 at bits 2(k % 16)."""
    if bits not in (1, 2):
        raise ValueError(f"bits must be 1 or 2, got {bits}")
    K, N = w.shape
    if K % GROUP:
        raise ValueError(f"K={K} is not a multiple of {GROUP}")
    wf = w.float()
    s = wf.abs().reshape(K // GROUP, GROUP, N).mean(1).to(torch.bfloat16)
    sk = s.float().repeat_interleave(GROUP, dim=0)
    if bits == 1:
        code = (wf >= 0).to(torch.int64)
    else:
        code = (wf >= -sk / 2).to(torch.int64) + (wf > sk / 2).to(torch.int64)
    per_word = 32 // bits
    codes = code.t().reshape(N, K // per_word, per_word)
    shifts = (torch.arange(per_word, device=w.device) * bits).reshape(1, 1, -1)
    return _to_int32((codes << shifts).sum(-1)), s


def _dequantize_ref(packed: torch.Tensor, scales: torch.Tensor, bits: int = 1) -> torch.Tensor:
    """The [K, N] bf16 weight the packed codes and scales stand for: bf16(s * (2 code - 1))
    at 1 bit, bf16(s * (code - 1)) at 2 bits."""
    N = packed.shape[0]
    per_word = 32 // bits
    K = packed.shape[1] * per_word
    shifts = (torch.arange(per_word, device=packed.device) * bits).reshape(1, 1, -1)
    code = (_from_int32(packed).unsqueeze(-1) >> shifts) & ((1 << bits) - 1)
    code = code.reshape(N, K).t().float()
    sk = scales.float().repeat_interleave(GROUP, dim=0)
    v = sk * (2 * code - 1) if bits == 1 else sk * (code - 1)
    return v.to(torch.bfloat16)


def _quantize_experts_ref(w: torch.Tensor, bits: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    """[E, K, N] bf16 -> (packed int32 [E, N, K*bits/32], scales bf16 [E, K/128, N]), each
    expert by `_quantize_ref`."""
    packed, scales = zip(*(_quantize_ref(w[e], bits) for e in range(w.shape[0])), strict=True)
    return torch.stack(packed), torch.stack(scales)


class _Packed:
    """A W1Weight / W1ExpertWeights look-alike for when the extension is not importable:
    the packed codes, the scales and the bit width, with the shape and byte count."""

    def __init__(self, packed: torch.Tensor, scales: torch.Tensor, bits: int):
        self.packed = packed
        self.scales = scales
        self.bits = bits

    @property
    def shape(self) -> tuple[int, ...]:
        per_word = 32 // self.bits
        lead = tuple(self.packed.shape[:-2])
        return (*lead, self.packed.shape[-1] * per_word, self.packed.shape[-2])

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.packed, self.scales))


def _ops():
    """The extension's ops module, or None on a machine without it."""
    try:
        from spark_kernels import ops
    except ImportError:
        return None
    return ops


def quantize_weight(w: torch.Tensor, bits: int = 1):
    """A [K, N] bf16 weight as a `W1Weight` (the extension's quantizer), or as a `_Packed`
    built by the torch quantizer when the extension is not importable."""
    sk = _ops()
    if sk is not None and hasattr(sk, "W1Weight"):
        return sk.W1Weight.quantize(w, bits)
    return _Packed(*_quantize_ref(w, bits), bits)


def quantize_experts(w: torch.Tensor, bits: int = 1):
    """[E, K, N] bf16 experts as a `W1ExpertWeights`, or a `_Packed` without the extension."""
    sk = _ops()
    if sk is not None and hasattr(sk, "W1ExpertWeights"):
        return sk.W1ExpertWeights.quantize(w, bits)
    return _Packed(*_quantize_experts_ref(w, bits), bits)


def dequantize_weight(q) -> torch.Tensor:
    """The bf16 weight a `W1Weight`, `W1ExpertWeights` or `_Packed` stands for, [K, N] or
    [E, K, N]."""
    if q.packed.dim() == 3:
        return torch.stack([_dequantize_ref(q.packed[e], q.scales[e], q.bits)
                            for e in range(q.packed.shape[0])])
    return _dequantize_ref(q.packed, q.scales, q.bits)


def _swiglu_ref(gate_up: torch.Tensor, inter: int) -> torch.Tensor:
    g, u = gate_up[:, :inter].float(), gate_up[:, inter:].float()
    return (F.silu(g) * u).to(torch.bfloat16)


def _gemm_ref(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return (a.float() @ w.float()).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# The block
# ---------------------------------------------------------------------------
class MoEBlock1Bit:
    """One MoE layer's MLP with the routed experts at `cfg.bits` per weight.

    gate_up        W1ExpertWeights over [E, hidden, 2 inter]: an expert's gate columns then
                   its up columns
    down           W1ExpertWeights over [E, inter, hidden]
    shared_gate_up W1Weight over [hidden, 2 inter * n_shared] (None when n_shared == 0)
    shared_down    W1Weight over [inter * n_shared, hidden]
    router_w       bf16 [hidden, E]; router_b fp32 [E] or None

    `forward(x)` takes [T, hidden] bf16 and returns the block's output (not added to the
    residual). The routed path gathers each token's row once per expert it goes to, runs the
    two grouped GEMMs on the expert-sorted rows, weights each row by its gate and scatter-adds
    the rows back in fp32."""

    def __init__(self, cfg: MoEConfig, gate_up, down, shared_gate_up, shared_down,
                 router_w: torch.Tensor, router_b: torch.Tensor | None):
        self.cfg = cfg
        self.gate_up = gate_up
        self.down = down
        self.shared_gate_up = shared_gate_up
        self.shared_down = shared_down
        self.router_w = router_w
        self.router_b = router_b

    @staticmethod
    def from_bf16(w_gate_up: torch.Tensor, w_down: torch.Tensor,
                  shared_gate_up: torch.Tensor | None, shared_down: torch.Tensor | None,
                  router_w: torch.Tensor, router_b: torch.Tensor | None,
                  cfg: MoEConfig) -> MoEBlock1Bit:
        """Quantize bf16 experts ([E, hidden, 2 inter] and [E, inter, hidden]) and the shared
        expert ([hidden, 2 inter], [inter, hidden]) at cfg.bits."""
        want = (cfg.n_routed, cfg.hidden, 2 * cfg.inter)
        if tuple(w_gate_up.shape) != want:
            raise ValueError(f"gate_up is {tuple(w_gate_up.shape)}, config says {want}")
        if tuple(w_down.shape) != (cfg.n_routed, cfg.inter, cfg.hidden):
            raise ValueError(f"down is {tuple(w_down.shape)}")
        if tuple(router_w.shape) != (cfg.hidden, cfg.n_routed):
            raise ValueError(f"router is {tuple(router_w.shape)}, want [hidden, E]")
        sgu = None if shared_gate_up is None else quantize_weight(shared_gate_up, cfg.bits)
        sd = None if shared_down is None else quantize_weight(shared_down, cfg.bits)
        return MoEBlock1Bit(cfg, quantize_experts(w_gate_up, cfg.bits),
                            quantize_experts(w_down, cfg.bits), sgu, sd,
                            router_w.to(torch.bfloat16),
                            None if router_b is None else router_b.float())

    @staticmethod
    def random(cfg: MoEConfig, device="cuda", generator: torch.Generator | None = None,
               std: float = 0.02) -> MoEBlock1Bit:
        """A block of normal weights for benches, quantized expert by expert so the bf16
        source of one expert (3 * hidden * inter * 2 bytes) is all that is live at once."""
        E, H, N_I = cfg.n_routed, cfg.hidden, cfg.inter

        def rand(*shape):
            return torch.randn(*shape, device=device, generator=generator) \
                .mul_(std).to(torch.bfloat16)

        gate_up = _cat_experts([quantize_experts(rand(1, H, 2 * N_I), cfg.bits)
                                for _ in range(E)], cfg.bits)
        down = _cat_experts([quantize_experts(rand(1, N_I, H), cfg.bits)
                             for _ in range(E)], cfg.bits)
        shared_gu = shared_dn = None
        if cfg.n_shared:
            shared_gu = quantize_weight(rand(H, 2 * N_I * cfg.n_shared), cfg.bits)
            shared_dn = quantize_weight(rand(N_I * cfg.n_shared, H), cfg.bits)
        router_w = rand(H, E)
        router_b = torch.randn(E, device=device, generator=generator).mul_(0.1)
        return MoEBlock1Bit(cfg, gate_up, down, shared_gu, shared_dn, router_w, router_b)

    def nbytes(self) -> int:
        """Bytes of the routed experts (packed + scales)."""
        return self.gate_up.nbytes() + self.down.nbytes()

    def shared_nbytes(self) -> int:
        return sum(q.nbytes() for q in (self.shared_gate_up, self.shared_down) if q is not None)

    def routing(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(expert ids [T, top_k], gate weights [T, top_k]) of the rows of x."""
        logits = x.float() @ self.router_w.float()
        return route(logits, self.cfg.top_k, self.router_b, self.cfg.route_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[T, hidden] bf16 -> [T, hidden] bf16 on the kernels: two w1gemm_moe launches over
        the expert-sorted rows, two w1gemm launches for the shared expert."""
        from spark_kernels import ops as sk

        cfg = self.cfg
        T = x.shape[0]
        ids, gate_w = self.routing(x)
        perm, offsets, token_of = sort_by_expert(ids, cfg.n_routed)
        xs = x.index_select(0, token_of)
        gu = sk.w1gemm_moe(xs, self.gate_up, offsets)
        h = sk.swiglu(gu[:, :cfg.inter].contiguous(), gu[:, cfg.inter:].contiguous())
        d = sk.w1gemm_moe(h, self.down, offsets)
        d = d.float() * gate_w.reshape(-1).index_select(0, perm).unsqueeze(1)
        out = torch.zeros(T, cfg.hidden, dtype=torch.float32, device=x.device)
        out.index_add_(0, token_of, d)
        if self.shared_gate_up is not None:
            sgu = sk.w1gemm(x, self.shared_gate_up)
            n = sgu.shape[1] // 2
            sh = sk.swiglu(sgu[:, :n].contiguous(), sgu[:, n:].contiguous())
            out += sk.w1gemm(sh, self.shared_down).float()
        return out.to(torch.bfloat16)

    def forward_reference(self, x: torch.Tensor) -> torch.Tensor:
        """The same block in plain torch on the dequantized weights, one expert at a time:
        fp32 math with a bf16 rounding where the kernels round (each GEMM's output, the
        swiglu), the per-row gate weights and the scatter-add in fp32."""
        cfg = self.cfg
        T = x.shape[0]
        ids, gate_w = self.routing(x)
        out = torch.zeros(T, cfg.hidden, dtype=torch.float32, device=x.device)
        for e in range(cfg.n_routed):
            tok, slot = torch.nonzero(ids == e, as_tuple=True)
            if tok.numel() == 0:
                continue
            w_gu = _dequantize_ref(self.gate_up.packed[e], self.gate_up.scales[e], cfg.bits)
            w_dn = _dequantize_ref(self.down.packed[e], self.down.scales[e], cfg.bits)
            h = _swiglu_ref(_gemm_ref(x[tok], w_gu), cfg.inter)
            d = _gemm_ref(h, w_dn).float() * gate_w[tok, slot].unsqueeze(1)
            out.index_add_(0, tok, d)
        if self.shared_gate_up is not None:
            sgu = _gemm_ref(x, dequantize_weight(self.shared_gate_up))
            sh = _swiglu_ref(sgu, sgu.shape[1] // 2)
            out += _gemm_ref(sh, dequantize_weight(self.shared_down)).float()
        return out.to(torch.bfloat16)


def _cat_experts(parts: list, bits: int):
    """Per-expert quantized weights ([1, ...] each) as one stacked W1ExpertWeights / _Packed."""
    packed = torch.cat([p.packed for p in parts])
    scales = torch.cat([p.scales for p in parts])
    sk = _ops()
    if sk is not None and hasattr(sk, "W1ExpertWeights"):
        try:
            return sk.W1ExpertWeights(packed, scales, bits)
        except TypeError:
            pass
    return _Packed(packed, scales, bits)
