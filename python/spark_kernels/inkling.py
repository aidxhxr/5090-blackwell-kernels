"""A pure-torch reference of the Inkling 2 Mini text model (Thinking Machines, HF
`thinkingmachines/Inkling-2-Mini`), the thing the Inkling kernels in this repo check against.

It needs only torch (no `_C`, no transformers). `InklingConfig.from_hf` reads the checkpoint's
config.json; the modules further down have the state_dict keys of HF's
`Inkling2LoopForCausalLM`, so `ref.load_state_dict(hf.state_dict(), strict=True)` loads HF's
weights as they are.

The text model as one token sees it (hidden 2304, 24 stored layers, 48 layer applications):

    embed -> RMSNorm
    layers 0-5                       the prelude, once
    layers 6-17, pass 1              the loop
    layers 6-17, passes 2 and 3      restart from the prelude output, with messages read from
                                     the previous pass's block states
    layers 18-23                     once
    block-residual mix -> RMSNorm -> lm_head -> softcap 30

Every layer: a block attention residual (a softmax over the finished block states and the
current partial block, per token, over depth) picks the attention input; attention (sliding
window 1024 or full, 24 q / 8 kv heads of 128, no RoPE, an additive relative position bias,
per-query sinks, a sigmoid output gate) goes through a short causal conv; a second residual
mix picks the MoE input (128 experts, top 8, sigmoid router with a selection bias, softcapped
SwiGLU); the MoE output goes through another short conv. Six layers form one block.

    cfg = InklingConfig.from_hf("~/models/Inkling-2-Mini")
    ref = InklingForCausalLM(cfg).to_model_dtype(torch.bfloat16)
    ref.load_state_dict(hf.state_dict(), strict=True)
    logits = ref(ids)                           # prefill without a cache, [B, T, 200058]
    cache = InklingCache(cfg)
    logits = ref(ids, cache)                    # prefill into the 48-layer cache
    logits = ref.decode_step(next_ids, cache)   # one token per sequence, [B, 200058]

It is a reference: the experts run as a loop over the picked experts, attention builds the
whole [T, T] score matrix, nothing is fused. Padding (an attention mask) is not supported.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

SLIDING = "sliding"
FULL = "full"


@dataclass
class InklingConfig:
    """The text model's shape. The defaults are the released Inkling 2 Mini."""

    hidden_size: int = 2304
    num_layers: int = 24
    vocab_size: int = 201024  # rows of the embedding (padded)
    unpadded_vocab_size: int = 200058  # rows of lm_head, the logits
    num_heads: int = 24
    num_kv_heads: int = 8
    head_dim: int = 128
    swa_num_heads: int = 24
    swa_num_kv_heads: int = 8
    swa_head_dim: int = 128
    sliding_window: int = 1024  # a sliding query sees backward distances 0..sliding_window-1
    rel_extent: int = 1024  # relative bias range in the full layers (sliding layers use the window)
    d_rel: int = 16
    log_scaling_n_floor: int = 8192
    log_scaling_alpha: float = 0.1
    rms_norm_eps: float = 1e-6
    qk_clip: float = 22.627416997969522
    rel_clip: float = 8.0
    conv_kernel: int = 4
    block_layers: int = 6  # layers per attention-residual block
    num_experts: int = 128
    top_k: int = 8
    moe_intermediate: int = 1152
    gate_softcap: float = 4.0
    up_softcap: float = 25.0
    route_scale: float = 8.0
    logit_softcap: float = 30.0
    loop_start: int = 6
    loop_end: int = 18
    loop_passes: int = 3
    gate_rank: int = 64
    layer_types: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.layer_types:
            # HF's default: every sixth layer is full attention
            self.layer_types = [FULL if (i + 1) % 6 == 0 else SLIDING
                                for i in range(self.num_layers)]
        if len(self.layer_types) != self.num_layers or set(self.layer_types) - {SLIDING, FULL}:
            raise ValueError(f"layer_types {self.layer_types} for {self.num_layers} layers")
        b, lo, hi = self.block_layers, self.loop_start, self.loop_end
        if not (0 < lo <= hi <= self.num_layers):
            raise ValueError(f"loop [{lo}, {hi}) in {self.num_layers} layers")
        if lo % b or hi % b or self.loop_passes < 1:
            raise ValueError(f"loop [{lo}, {hi}) is not whole blocks of {b}")

    @classmethod
    def from_hf(cls, cfg: dict | str | Path) -> InklingConfig:
        """From the checkpoint's config.json (a path, a directory holding it, or the parsed
        dict). The text model's fields are in `text_config` (a bare text config works too),
        the loop layout in `text_config.loop_config`. Layer types come from `layer_types` if
        present, else from `local_layer_ids` (the sliding layers), else HF's default."""
        if not isinstance(cfg, dict):
            p = Path(cfg).expanduser()
            cfg = json.loads((p / "config.json" if p.is_dir() else p).read_text())
        t = cfg.get("text_config", cfg)
        loop = t.get("loop_config", {})
        if loop.get("is_varloop"):
            raise ValueError("variable loop counts are not supported")
        n = t.get("num_hidden_layers", 24)
        if t.get("layer_types"):
            types = [SLIDING if s == "hybrid_sliding" else FULL for s in t["layer_types"]]
        elif "local_layer_ids" in t:
            local = set(t["local_layer_ids"])
            types = [SLIDING if i in local else FULL for i in range(n)]
        else:
            types = []
        if t.get("n_shared_experts", 0):
            raise ValueError("shared experts are not part of this model")
        d = cls()
        return cls(
            hidden_size=t.get("hidden_size", d.hidden_size),
            num_layers=n,
            vocab_size=t.get("vocab_size", d.vocab_size),
            unpadded_vocab_size=t.get("unpadded_vocab_size", d.unpadded_vocab_size),
            num_heads=t.get("num_attention_heads", d.num_heads),
            num_kv_heads=t.get("num_key_value_heads", d.num_kv_heads),
            head_dim=t.get("head_dim", d.head_dim),
            swa_num_heads=t.get("swa_num_attention_heads", d.swa_num_heads),
            swa_num_kv_heads=t.get("swa_num_key_value_heads", d.swa_num_kv_heads),
            swa_head_dim=t.get("swa_head_dim", d.swa_head_dim),
            sliding_window=t.get("sliding_window_size", d.sliding_window),
            rel_extent=t.get("rel_extent", d.rel_extent),
            d_rel=t.get("d_rel", d.d_rel),
            log_scaling_n_floor=t.get("log_scaling_n_floor", d.log_scaling_n_floor),
            log_scaling_alpha=t.get("log_scaling_alpha", d.log_scaling_alpha),
            rms_norm_eps=t.get("rms_norm_eps", d.rms_norm_eps),
            qk_clip=t.get("qk_l2_clip_threshold", d.qk_clip),
            rel_clip=t.get("rel_l2_clip_threshold", d.rel_clip),
            conv_kernel=t.get("sconv_kernel_size", t.get("conv_kernel_size", d.conv_kernel)),
            block_layers=t.get("n_layers_per_attn_res_block", d.block_layers),
            num_experts=t.get("n_routed_experts", d.num_experts),
            top_k=t.get("num_experts_per_tok", d.top_k),
            moe_intermediate=t.get("intermediate_size",
                                   t.get("moe_intermediate_size", d.moe_intermediate)),
            gate_softcap=t.get("mlp_gate_softcap", d.gate_softcap),
            up_softcap=t.get("mlp_up_softcap", d.up_softcap),
            route_scale=t.get("route_scale", d.route_scale),
            logit_softcap=t.get("final_logit_softcapping", d.logit_softcap),
            loop_start=loop.get("core_start", t.get("loop_layer_index_start", d.loop_start)),
            loop_end=loop.get("core_end", t.get("loop_layer_index_end", d.loop_end)),
            loop_passes=loop.get("num_passes", t.get("loop_num_passes", d.loop_passes)),
            gate_rank=loop.get("gate_rank", t.get("loop_gate_rank", d.gate_rank)),
            layer_types=types,
        )

    # ---- derived layout ------------------------------------------------------------------

    @property
    def num_loop_layers(self) -> int:
        return self.loop_end - self.loop_start

    @property
    def num_applications(self) -> int:
        """Layer applications per token: 24 + 2 * 12 = 48 for the released model."""
        return self.num_layers + (self.loop_passes - 1) * self.num_loop_layers

    @property
    def num_cache_layers(self) -> int:
        """One KV + conv cache layer per application (48): the stored layers first, then one
        copy of the loop layers for each extra pass, the same order HF appends them in."""
        return self.num_applications

    def is_loop_layer(self, layer: int) -> bool:
        return self.loop_start <= layer < self.loop_end

    def cache_index(self, layer: int, loop_pass: int = 0) -> int:
        """The cache layer of `layer` in pass `loop_pass` (0-based; only loop layers have
        passes past 0). Pass 2 of layer 6 is cache layer 24, pass 3 of layer 17 is 47."""
        if loop_pass == 0:
            return layer
        assert self.is_loop_layer(layer) and loop_pass < self.loop_passes
        return self.num_layers + (loop_pass - 1) * self.num_loop_layers + layer - self.loop_start

    def schedule(self) -> list[tuple[int, int]]:
        """(layer, pass) for the 48 applications, in the order a token runs them."""
        s = [(i, 0) for i in range(self.loop_end)]
        for p in range(1, self.loop_passes):
            s += [(i, p) for i in range(self.loop_start, self.loop_end)]
        return s + [(i, 0) for i in range(self.loop_end, self.num_layers)]

    def cache_layer_types(self) -> list[str]:
        """SLIDING or FULL for each of the 48 cache layers."""
        out = list(self.layer_types)
        for _ in range(1, self.loop_passes):
            out += self.layer_types[self.loop_start:self.loop_end]
        return out

    def attn_shape(self, layer: int) -> tuple[int, int, int]:
        """(q heads, kv heads, head dim) of `layer`."""
        if self.layer_types[layer] == SLIDING:
            return self.swa_num_heads, self.swa_num_kv_heads, self.swa_head_dim
        return self.num_heads, self.num_kv_heads, self.head_dim

    def param_counts(self) -> dict[str, int]:
        """Parameters of the text model, counted from the shapes (no weights needed).

        Included: the embedding (vocab_size rows, padded), every stored layer once (the loop
        reuses layers 6-17, it adds no weights), the loop's message queries and gates, the
        final mix and norm, lm_head (unpadded_vocab_size rows). Not included: the vision
        tower (318,256,128 parameters in HF's module) and the MTP head (317,897,872 bf16
        parameters in a separate mtp.safetensors that HF's model does not load). Buffers (the
        router's selection bias) are not parameters.

        For the released config: total 26,068,156,416 (26.07B). With the vision tower that is
        26,386,412,544, the published 26.4B. Active per token is the total minus the experts
        a token does not use in each stored layer (num_experts - top_k of them): 3,135,799,296
        (3.14B). The published 3.4B is that plus about 0.32B, the size of the vision tower
        and, by coincidence, of the MTP head too (3.45B either way); the numbers do not say
        which. A token runs the loop layers 3 times and can pick different experts in each
        pass: `expert_reads` counts the expert parameters one decode step touches over its 48
        applications (8 experts each), 3,057,647,616.
        """
        h, k = self.hidden_size, self.conv_kernel
        expert = 3 * h * self.moe_intermediate  # gate, up, down of one expert
        counts = dict(embed=self.vocab_size * h + h, lm_head=self.unpadded_vocab_size * h,
                      experts=0, other=2 * h)  # final norm, final mix query
        counts["other"] += self.num_loop_layers * 4 * h  # message queries
        for i, kind in enumerate(self.layer_types):
            nh, nkv, hd = self.attn_shape(i)
            q, kv = nh * hd, nkv * hd
            rel = self.sliding_window if kind == SLIDING else self.rel_extent
            n = q * h + q  # q_proj, its clip scale
            n += 2 * kv * h  # k_proj, v_proj
            n += 2 * (kv * k if kind == SLIDING else k * kv + nkv * k * h)  # static / dynamic taps
            n += 2 * q * h  # output gate, o_proj
            n += nh * h  # sinks
            n += nh * self.d_rel * h + nh * self.d_rel + self.d_rel * rel  # relative bias
            n += (2 if kind == SLIDING else 4) * h  # input norm(s), post-attention norm
            n += 2 * k * h  # post-attention and post-MLP convs
            n += 0 if i == 0 else (h if kind == SLIDING else 3 * h)  # attention input mix
            n += h  # MLP input mix
            n += self.num_experts * h  # router
            if self.is_loop_layer(i):
                n += 4 * (2 * h * self.gate_rank + 2 * self.gate_rank * h)  # q, k, v, mlp gates
            counts["other"] += n
            counts["experts"] += self.num_experts * expert
        counts["total"] = counts["embed"] + counts["lm_head"] + counts["experts"] + counts["other"]
        unused = self.num_layers * (self.num_experts - self.top_k) * expert
        counts["active"] = counts["total"] - unused
        counts["expert_reads"] = self.num_applications * self.top_k * expert
        return counts


# ---- decode cache ----------------------------------------------------------------------------


class InklingCache:
    """Everything a decode step needs from the tokens before it, for the 48 cache layers
    (cfg.cache_index maps a layer and pass to one):

        keys[c], values[c]   [B, Hkv, L, D] in the model dtype, after the conv and the clip or
                             norm; a sliding cache layer keeps the last `sliding_window`
                             positions (1024), a full one keeps all of them
        conv[c][s]           [B, k - 1, C] fp32, the last 3 inputs of conv s of cache layer
                             c: 0 the key conv (C = 1024), 1 the value conv (1024), 2 the
                             attention output conv (2304), 3 the MLP output conv (2304)
        length               tokens seen so far (the next token's position)

    Nothing else carries over between tokens: the block residual mixes and the loop's
    messages mix over depth, within one token, so the block states are rebuilt every step.
    """

    def __init__(self, cfg: InklingConfig):
        self.cfg = cfg
        self.types = cfg.cache_layer_types()
        n = cfg.num_cache_layers
        self.keys: list[torch.Tensor | None] = [None] * n
        self.values: list[torch.Tensor | None] = [None] * n
        self.conv: list[list[torch.Tensor | None]] = [[None] * 4 for _ in range(n)]
        self.length = 0

    def conv_context(self, layer: int, slot: int, x: torch.Tensor, k: int) -> torch.Tensor:
        """x [B, T, C] fp32 -> [B, T + k - 1, C], the stored k - 1 inputs (zeros before the
        first token) in front of x; keeps the last k - 1 of the result."""
        state = self.conv[layer][slot]
        if state is None:
            state = x.new_zeros(x.shape[0], k - 1, x.shape[2])
        ctx = torch.cat([state, x], dim=1)
        self.conv[layer][slot] = ctx[:, ctx.shape[1] - (k - 1):]
        return ctx

    def append_kv(self, layer: int, k: torch.Tensor, v: torch.Tensor):
        """Append this step's k, v [B, Hkv, T, D] (positions length .. length + T - 1).
        Returns the keys and values to attend over (stored + new) and the position of the
        first of them; a sliding layer then keeps only the last `sliding_window`."""
        new = k.shape[2]
        if self.keys[layer] is not None:
            k = torch.cat([self.keys[layer], k], dim=2)
            v = torch.cat([self.values[layer], v], dim=2)
        first = self.length + new - k.shape[2]
        keep = self.cfg.sliding_window if self.types[layer] == SLIDING else k.shape[2]
        self.keys[layer], self.values[layer] = k[:, :, -keep:], v[:, :, -keep:]
        return k, v, first

    def nbytes(self) -> int:
        ts = [t for t in self.keys + self.values if t is not None]
        ts += [t for row in self.conv for t in row if t is not None]
        return sum(t.numel() * t.element_size() for t in ts)


# ---- modules ---------------------------------------------------------------------------------
#
# Attribute names follow HF's Inkling2Loop modules so the state_dict keys are the same. The
# dtype casts follow HF too: the residual stream between layers is in the model dtype, the
# residual mixes, the convs, the router, the message gates and every norm compute in fp32.

# HF keeps these in fp32 when the rest of the model is bf16 (`_keep_in_fp32_modules_strict`)
FP32_KEYS = ("e_score_correction_bias", "mlp.gate.weight", "post_attention_convolution",
             "post_mlp_convolution", "sconv.conv1d", "attention_input", "mlp_input",
             "final_input", "message_input", "message_gate_proj", "forget_gate_proj")


def rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """RMS normalization without a weight, in fp32, back in x's dtype."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def l2_clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Scale each vector (last dim) down to L2 norm `threshold` if it is longer (fp32 math)."""
    xf = x.float()
    norm = torch.linalg.vector_norm(xf, dim=-1, keepdim=True)
    return xf * (threshold / (norm + 1e-6)).clamp(max=1.0)


def log_scale(cfg: InklingConfig, positions: torch.Tensor) -> torch.Tensor:
    """tau = 1 + alpha * log(max((pos + 1) / n_floor, 1)) per position, fp32 [T]."""
    ratio = (positions + 1).float() / cfg.log_scaling_n_floor
    return 1.0 + cfg.log_scaling_alpha * torch.log(ratio.clamp(min=1.0))


def softcap(x: torch.Tensor, cap: float) -> torch.Tensor:
    return torch.tanh(x / cap) * cap


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * xf).to(x.dtype)


class NormedEmbedding(nn.Module):
    def __init__(self, cfg: InklingConfig):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cfg.vocab_size, cfg.hidden_size))
        self.embed_norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed_norm(F.embedding(ids, self.weight))


class ResidualMix(nn.Module):
    """Block attention residual (and the loop's message read): per token, a softmax over
    depth. The keys and values are the RMS-normalized block states plus the current partial
    block; the queries are learned, the same at every position; the scale is 1/sqrt(hidden).
    `query` is [H] (one output), [n, H] (n outputs) or [L, n, H] (one row per loop layer)."""

    def __init__(self, shape: tuple[int, ...]):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(*shape, dtype=torch.float32))

    def forward(self, blocks: list[torch.Tensor], partial: torch.Tensor,
                row: int | None = None) -> list[torch.Tensor]:
        states = torch.stack([*blocks, partial], dim=-2) if blocks else partial[..., None, :]
        states = rms(states.float())  # [B, T, n_states, H]
        q = (self.query if row is None else self.query[row]).float()
        q = q.reshape(-1, states.shape[-1]).expand(*states.shape[:-2], -1, -1)  # [B, T, n_out, H]
        logits = (q @ states.transpose(-1, -2)) * states.shape[-1] ** -0.5
        weights = torch.softmax(logits, dim=-1, dtype=torch.float32)
        return list((weights @ states).unbind(-2))


class MessageGate(nn.Module):
    """out = (1 - tanh(Wf z)) * h + tanh(Wg z) * m with z = Wd [rms(h); rms(m)], rank 64."""

    def __init__(self, cfg: InklingConfig):
        super().__init__()
        h, r = cfg.hidden_size, cfg.gate_rank
        self.down_proj = nn.Linear(2 * h, r, bias=False)
        self.forget_gate_proj = nn.Linear(r, h, bias=False, dtype=torch.float32)
        self.message_gate_proj = nn.Linear(r, h, bias=False, dtype=torch.float32)

    def forward(self, h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        h, m = h.float(), m.float()
        z = torch.cat([rms(h), rms(m)], dim=-1)
        z = self.down_proj(z.to(self.down_proj.weight.dtype)).float()
        forget = torch.tanh(self.forget_gate_proj(z))
        gain = torch.tanh(self.message_gate_proj(z))
        return (1 - forget) * h + gain * m


def _conv_context(x, k, cache, layer, slot):
    """The conv input with its k - 1 tokens of left context: zeros without a cache, the
    cached inputs with one."""
    if cache is None:
        return F.pad(x, (0, 0, k - 1, 0))
    return cache.conv_context(layer, slot, x, k)


class ShortConv(nn.Module):
    """Causal depthwise conv over time with static taps, kernel 4, plus the input:
    out = (conv(x) + x) / sqrt(k + 1), in fp32. conv1d.weight is [C, 1, k], tap k-1 on the
    current token."""

    def __init__(self, channels: int, k: int, slot: int):
        super().__init__()
        self.k, self.slot = k, slot  # slot: which of the cache layer's four conv states
        self.conv1d = nn.Conv1d(channels, channels, k, groups=channels, bias=False)

    def forward(self, x: torch.Tensor, cache: InklingCache | None = None,
                layer: int = 0) -> torch.Tensor:
        xf = x.float()
        ctx = _conv_context(xf, self.k, cache, layer, self.slot)  # [B, T + k - 1, C]
        y = F.conv1d(ctx.transpose(1, 2), self.conv1d.weight.float(), groups=xf.shape[-1])
        return ((y.transpose(1, 2) + xf) / math.sqrt(self.k + 1)).to(x.dtype)


class DynamicShortConv(nn.Module):
    """The full layers' k and v conv: taps = static_weight [k, C] (tap = backward distance)
    plus per-token taps weight_proj(dyn_input), one set of k per kv head.
    out = (conv(x) + x) / sqrt(2k + 1), in fp32."""

    def __init__(self, channels: int, input_size: int, heads: int, k: int, slot: int):
        super().__init__()
        self.k, self.heads, self.slot = k, heads, slot
        self.static_weight = nn.Parameter(torch.zeros(k, channels))
        self.weight_proj = nn.Linear(input_size, heads * k, bias=False)

    def forward(self, x: torch.Tensor, dyn_input: torch.Tensor,
                cache: InklingCache | None = None, layer: int = 0) -> torch.Tensor:
        b, t, c = x.shape
        dyn = self.weight_proj(dyn_input).float().view(b, t, self.heads, 1, self.k)
        taps = self.static_weight.float().T.reshape(self.heads, c // self.heads, self.k) + dyn
        xf = x.float()
        ctx = _conv_context(xf, self.k, cache, layer, self.slot)
        # windows[..., lag] is the token `lag` steps back
        windows = ctx.unfold(1, self.k, 1).flip(-1).reshape(b, t, self.heads, -1, self.k)
        y = (windows * taps).sum(-1).reshape(b, t, c)
        return ((y + xf) / math.sqrt(2 * self.k + 1)).to(x.dtype)


class ScaledClip(nn.Module):
    """x * scale (per head and channel), L2 clip per head, then (full layers) * tau."""

    def __init__(self, cfg: InklingConfig, heads: int, dim: int, threshold: float, log: bool):
        super().__init__()
        self.cfg, self.threshold, self.log = cfg, threshold, log
        self.scale = nn.Parameter(torch.ones(heads, dim))

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        y = l2_clip(x.float() * self.scale.float(), self.threshold).to(x.dtype)
        if self.log:
            y = (y.float() * log_scale(self.cfg, positions)[:, None, None]).to(x.dtype)
        return y


class QueryProjection(nn.Module):
    def __init__(self, cfg: InklingConfig, heads: int, dim: int, log: bool):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(cfg.hidden_size, heads * dim, bias=False)
        self.norm = ScaledClip(cfg, heads, dim, cfg.qk_clip, log)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        q = self.linear(x)
        return self.norm(q.view(*q.shape[:-1], -1, self.dim), positions)  # [B, T, Hq, D]


class KeyValueProjection(nn.Module):
    """linear -> short conv (static, or static + dynamic in full layers) -> per head L2 clip
    (keys) or RMS norm without weight (values)."""

    def __init__(self, cfg: InklingConfig, heads: int, dim: int, is_key: bool, dynamic: bool):
        super().__init__()
        self.cfg, self.dim, self.is_key = cfg, dim, is_key
        self.linear = nn.Linear(cfg.hidden_size, heads * dim, bias=False)
        slot = 0 if is_key else 1
        self.sconv = (DynamicShortConv(heads * dim, cfg.hidden_size, heads, cfg.conv_kernel, slot)
                      if dynamic else ShortConv(heads * dim, cfg.conv_kernel, slot))

    def forward(self, x: torch.Tensor, cache: InklingCache | None = None,
                layer: int = 0) -> torch.Tensor:
        s = self.linear(x)
        if isinstance(self.sconv, DynamicShortConv):
            s = self.sconv(s, x, cache, layer)
        else:
            s = self.sconv(s, cache, layer)
        s = s.view(*s.shape[:-1], -1, self.dim)
        if self.is_key:
            return l2_clip(s, self.cfg.qk_clip).to(s.dtype)
        return rms(s, self.cfg.rms_norm_eps)  # [B, T, Hkv, D]


class GatedOutput(nn.Module):
    def __init__(self, cfg: InklingConfig, heads: int, dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, heads * dim, bias=False)
        self.linear = nn.Linear(heads * dim, cfg.hidden_size, bias=False)

    def forward(self, attn: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.gate_proj(x).float())
        return self.linear((attn.float() * gate).to(attn.dtype))


class RelativeBias(nn.Module):
    """Per query and head: a 16-dim relative state (linear, scaled L2 clip to 8, tau in full
    layers) times the bank proj [16, extent] gives one bias per backward distance 0..extent-1,
    zero beyond."""

    def __init__(self, cfg: InklingConfig, heads: int, extent: int, log: bool):
        super().__init__()
        self.heads, self.extent = heads, extent
        self.linear = nn.Linear(cfg.hidden_size, heads * cfg.d_rel, bias=False)
        self.norm = ScaledClip(cfg, heads, cfg.d_rel, cfg.rel_clip, log)
        self.proj = nn.Parameter(torch.empty(cfg.d_rel, extent))

    def forward(self, x: torch.Tensor, positions: torch.Tensor,
                distance: torch.Tensor) -> torch.Tensor:
        """distance [Tq, Tk] = query position - key position; returns [B, Hq, Tq, Tk]."""
        r = self.linear(x)
        r = self.norm(r.view(*r.shape[:-1], self.heads, -1), positions)
        per_distance = (r @ self.proj).transpose(1, 2)  # [B, Hq, Tq, extent]
        index = distance.clamp(0, self.extent - 1).expand(*per_distance.shape[:2], -1, -1)
        bias = per_distance.gather(-1, index)
        return bias.masked_fill((distance < 0) | (distance >= self.extent), 0.0)


def attend(q, k, v, bias, visible, sinks):
    """Softmax attention with an additive bias and one zero-value sink logit per query.
    q [B, Hq, Tq, D], k and v [B, Hkv, Tk, D], bias [B, Hq, Tq, Tk], visible [Tq, Tk] bool,
    sinks [B, Hq, Tq]. The logits are in the inputs' dtype, the softmax in fp32."""
    groups = q.shape[1] // k.shape[1]
    k, v = k.repeat_interleave(groups, dim=1), v.repeat_interleave(groups, dim=1)
    s = (q @ k.transpose(-1, -2)) * q.shape[-1] ** -0.5 + bias
    s = s.masked_fill(~visible, float("-inf"))
    p = torch.softmax(torch.cat([s, sinks[..., None]], dim=-1).float(), dim=-1)
    return p[..., :-1].to(q.dtype) @ v


class Attention(nn.Module):
    def __init__(self, cfg: InklingConfig, layer: int):
        super().__init__()
        self.full = cfg.layer_types[layer] == FULL
        nh, nkv, hd = cfg.attn_shape(layer)
        self.window = None if self.full else cfg.sliding_window
        extent = cfg.rel_extent if self.full else cfg.sliding_window
        self.q_proj = QueryProjection(cfg, nh, hd, log=self.full)
        self.k_proj = KeyValueProjection(cfg, nkv, hd, is_key=True, dynamic=self.full)
        self.v_proj = KeyValueProjection(cfg, nkv, hd, is_key=False, dynamic=self.full)
        self.o_proj = GatedOutput(cfg, nh, hd)
        self.sink_proj = nn.Linear(cfg.hidden_size, nh, bias=False)
        self.relative_position_bias = RelativeBias(cfg, nh, extent, log=self.full)

    def forward(self, xq, xk, xv, positions, cache=None, layer=0):
        """xq, xk, xv [B, T, H] are the query, key and value inputs (the same tensor in the
        sliding layers outside the loop); the gate, the sinks and the relative bias read xq.
        With a cache, the keys and values of the earlier tokens come from cache layer
        `layer` and this step's are appended to it."""
        q = self.q_proj(xq, positions).transpose(1, 2)
        k = self.k_proj(xk, cache, layer).transpose(1, 2)
        v = self.v_proj(xv, cache, layer).transpose(1, 2)
        key_positions = positions
        if cache is not None:
            k, v, first = cache.append_kv(layer, k, v)
            key_positions = torch.arange(first, first + k.shape[2], device=positions.device)
        distance = positions[:, None] - key_positions[None, :]
        visible = distance >= 0
        if self.window is not None:
            visible &= distance < self.window
        bias = self.relative_position_bias(xq, positions, distance)
        sinks = self.sink_proj(xq).transpose(1, 2)
        o = attend(q, k, v, bias, visible, sinks).transpose(1, 2)
        return self.o_proj(o.reshape(*xq.shape[:-1], -1), xq)


class Router(nn.Module):
    """Sigmoid scores in fp32; the bias only picks the experts; the weights are the picked
    unbiased scores renormalized to sum 1, times route_scale (8)."""

    def __init__(self, cfg: InklingConfig):
        super().__init__()
        self.cfg = cfg
        self.weight = nn.Parameter(torch.empty(cfg.num_experts, cfg.hidden_size,
                                               dtype=torch.float32))
        self.register_buffer("e_score_correction_bias",
                             torch.zeros(cfg.num_experts, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.sigmoid(F.linear(x.float(), self.weight))
        picked = torch.topk(scores + self.e_score_correction_bias, self.cfg.top_k, dim=-1,
                            sorted=False).indices
        w = scores.gather(1, picked)
        return picked, w / (w.sum(-1, keepdim=True) + 1e-20) * self.cfg.route_scale


class Experts(nn.Module):
    def __init__(self, cfg: InklingConfig):
        super().__init__()
        e, h, i = cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate
        self.cfg = cfg
        self.gate_up_proj = nn.Parameter(torch.empty(e, 2 * i, h))  # gate rows, then up rows
        self.down_proj = nn.Parameter(torch.empty(e, h, i))

    def forward(self, x: torch.Tensor, picked: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """x [N, H]; out = sum over the picked experts of w * down(swiglu(gate_up x)),
        summed in fp32 in expert order, softcapped SwiGLU
        softcap(g, 4) * sigmoid(g) * softcap(u, 25)."""
        out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
        for e in torch.unique(picked).tolist():
            token, slot = torch.where(picked == e)
            g, u = F.linear(x[token], self.gate_up_proj[e]).float().chunk(2, dim=-1)
            a = softcap(g, self.cfg.gate_softcap) * torch.sigmoid(g)
            a = a * softcap(u, self.cfg.up_softcap)
            y = F.linear(a.to(x.dtype), self.down_proj[e]) * w[token, slot, None]
            out.index_add_(0, token, y.float())
        return out.to(x.dtype)


class MoE(nn.Module):
    def __init__(self, cfg: InklingConfig):
        super().__init__()
        self.experts = Experts(cfg)
        self.gate = Router(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = x.reshape(-1, x.shape[-1])
        picked, w = self.gate(flat)
        return self.experts(flat, picked, w).view(x.shape)


class DecoderLayer(nn.Module):
    """One of the 24 stored layers. Sliding and full layers differ in the attention input:
    a sliding layer mixes one input for q, k and v and norms it once; a full layer mixes
    three (one query each) and norms each. Loop layers add four message gates (q, k, v, MLP
    inputs) used in passes 2 and 3."""

    def __init__(self, cfg: InklingConfig, layer: int):
        super().__init__()
        h, eps = cfg.hidden_size, cfg.rms_norm_eps
        self.full = cfg.layer_types[layer] == FULL
        self.is_block_boundary = layer % cfg.block_layers == 0
        if layer > 0:
            self.attention_input = ResidualMix((3, h) if self.full else (h,))
        else:
            self.attention_input = None
        self.mlp_input = ResidualMix((h,))
        self.self_attn = Attention(cfg, layer)
        self.mlp = MoE(cfg)
        if self.full:
            self.q_input_layernorm = RMSNorm(h, eps)
            self.k_input_layernorm = RMSNorm(h, eps)
            self.v_input_layernorm = RMSNorm(h, eps)
        else:
            self.input_layernorm = RMSNorm(h, eps)
        self.post_attention_layernorm = RMSNorm(h, eps)
        self.post_attention_convolution = ShortConv(h, cfg.conv_kernel, slot=2)
        self.post_mlp_convolution = ShortConv(h, cfg.conv_kernel, slot=3)
        self.loop = cfg.is_loop_layer(layer)
        if self.loop:
            self.q_message_gate = MessageGate(cfg)
            self.k_message_gate = MessageGate(cfg)
            self.v_message_gate = MessageGate(cfg)
            self.mlp_message_gate = MessageGate(cfg)

    def forward(self, x, blocks, positions, messages=None, cache=None, layer=0):
        """x [B, T, H] the partial block so far (model dtype), blocks the finished block
        states, messages (q, k, v, mlp) from the previous pass or None, `layer` the cache
        layer of this application. Returns the new partial block in the model dtype."""
        dtype = x.dtype
        if self.attention_input is None:
            mixed = [x] * 3
        else:
            mixed = self.attention_input(blocks, x)
            mixed = mixed * 3 if len(mixed) == 1 else mixed
        if messages is not None:
            gates = (self.q_message_gate, self.k_message_gate, self.v_message_gate)
            mixed = [g(m_in, msg) for g, m_in, msg in zip(gates, mixed, messages[:3], strict=True)]
        if self.full:
            norms = (self.q_input_layernorm, self.k_input_layernorm, self.v_input_layernorm)
            xq, xk, xv = (n(m).to(dtype) for n, m in zip(norms, mixed, strict=True))
        else:
            xq, xk, xv = (self.input_layernorm(m).to(dtype) for m in mixed)
        a = self.self_attn(xq, xk, xv, positions, cache, layer)
        a = self.post_attention_convolution(a, cache, layer).float()
        # a block boundary starts a new partial block: the input becomes a finished block
        if self.is_block_boundary:
            blocks, partial = [*blocks, x], a
        else:
            partial = x.float() + a
        m = self.mlp_input(blocks, partial)[0]
        if messages is not None:
            m = self.mlp_message_gate(m, messages[3])
        m = self.mlp(self.post_attention_layernorm(m).to(dtype))
        m = self.post_mlp_convolution(m, cache, layer)
        return (partial + m).to(dtype)


class InklingTextModel(nn.Module):
    def __init__(self, cfg: InklingConfig):
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_size
        self.embed_tokens = NormedEmbedding(cfg)
        self.layers = nn.ModuleList(DecoderLayer(cfg, i) for i in range(cfg.num_layers))
        self.norm = RMSNorm(h, cfg.rms_norm_eps)
        self.final_input = ResidualMix((h,))
        self.message_input = ResidualMix((cfg.num_loop_layers, 4, h))

    def _run(self, first, last, x, blocks, positions, cache, loop_pass=0, msgs_from=None):
        """Run layers first..last-1 in order; a block-boundary layer's input joins the
        finished blocks. msgs_from = (blocks, partial) of the previous loop pass, for passes
        2 and 3."""
        for i in range(first, last):
            layer = self.layers[i]
            messages = None
            if msgs_from is not None:
                messages = self.message_input(*msgs_from, row=i - first)
            c = self.cfg.cache_index(i, loop_pass)
            new = layer(x, blocks, positions, messages, cache, c)
            if layer.is_block_boundary:
                blocks = [*blocks, x]
            x = new
        return x, blocks

    def forward(self, ids: torch.Tensor, cache: InklingCache | None = None) -> torch.Tensor:
        """ids [B, T] -> final hidden states [B, T, H] (model dtype). With a cache, ids are
        the next T tokens after the cached ones, and the cache takes them in."""
        cfg = self.cfg
        start = 0 if cache is None else cache.length
        positions = torch.arange(start, start + ids.shape[1], device=ids.device)
        x = self.embed_tokens(ids)
        lo, hi = cfg.loop_start, cfg.loop_end
        x, blocks = self._run(0, lo, x, [], positions, cache)
        prelude = (x, blocks)
        x, blocks = self._run(lo, hi, x, blocks, positions, cache)
        for p in range(1, cfg.loop_passes):
            # the previous pass's own blocks (past the prelude's and the one it closed at
            # loop_start) and its partial block are what the messages read
            prev = (blocks[len(prelude[1]) + 1:], x)
            x, blocks = self._run(lo, hi, *prelude, positions, cache, p, msgs_from=prev)
        x, blocks = self._run(hi, cfg.num_layers, x, blocks, positions, cache)
        x = self.final_input(blocks, x)[0]
        if cache is not None:
            cache.length += ids.shape[1]
        return self.norm(x).to(self.embed_tokens.weight.dtype)


class InklingForCausalLM(nn.Module):
    """The text model and lm_head, keys as HF's Inkling2LoopForCausalLM."""

    def __init__(self, cfg: InklingConfig):
        super().__init__()
        self.cfg = cfg
        self.model = InklingTextModel(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.unpadded_vocab_size, bias=False)

    def to_model_dtype(self, dtype: torch.dtype) -> InklingForCausalLM:
        """Cast to `dtype` the way HF loads the checkpoint: the FP32_KEYS stay fp32."""
        for name, t in [*self.named_parameters(), *self.named_buffers()]:
            t.data = t.data.to(torch.float32 if any(k in name for k in FP32_KEYS) else dtype)
        return self

    def forward(self, ids: torch.Tensor, cache: InklingCache | None = None) -> torch.Tensor:
        """ids [B, T] -> softcapped logits [B, T, 200058] fp32. Without a cache this is the
        whole-prompt prefill; with one, ids continue the cached tokens (a prefill into the
        cache, or a chunk of one)."""
        h = self.model(ids, cache)
        return softcap(self.lm_head(h).float(), self.cfg.logit_softcap)

    def decode_step(self, token: torch.Tensor, cache: InklingCache) -> torch.Tensor:
        """One token per sequence, token [B] -> logits [B, 200058], through all 48 cache
        layers."""
        return self(token[:, None], cache)[:, -1]
