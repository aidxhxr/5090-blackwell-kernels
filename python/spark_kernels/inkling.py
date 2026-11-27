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
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

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
        tower (318,256,128 parameters in HF's module) and the MTP head (config.json has an
        mtp_config but the released checkpoint has no MTP tensors). Buffers (the router's
        selection bias) are not parameters.

        For the released config: total 26,068,156,416 (26.07B). With the vision tower that is
        26,386,412,544, the published 26.4B. Active per token is the total minus the experts
        a token does not use in each stored layer (num_experts - top_k of them): 3,135,799,296
        (3.14B), or 3,454,055,424 (3.45B) with the vision tower. The published 3.4B is
        closest to the latter. A token runs the loop layers 3 times and can pick different
        experts in each pass: `expert_reads` counts the expert parameters one decode step
        touches over its 48 applications (8 experts each), 3,057,647,616.
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
