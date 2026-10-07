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
