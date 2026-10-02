"""Real weights for the engine: a Hugging Face Llama checkpoint (a directory with config.json
and *.safetensors) loaded into `engine.ModelWeights`.

The engine is built for Llama-3-8B's shapes (hidden 4096, 32 query heads over 8 K/V heads of
128, a 14336-wide MLP), so `load` checks the config against them and refuses anything else.
What it converts:

    q_proj, k_proj, v_proj    [out, in] each, fused into w_qkv [4096, 6144] = [q | k | v]
    o_proj, down_proj         transposed to [K, N]
    gate_proj, up_proj        transposed and interleaved (`interleave_gate_up`) into
                              w_gate_up [4096, 28672], the layout hgemm_swiglu reads
    lm_head                   transposed to [4096, V]; the embedding when the config ties them

HF stores q and k already permuted for the rotate-half RoPE, which is the convention of
`layer.RoPE` and the rope kernels, so no weight permutation is needed. Tensors are read
straight to the GPU one at a time; the peak is the model plus one layer's transients.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from . import ops as sk
from .engine import ModelWeights
from .layer import EPS, HEAD_DIM, HIDDEN, INTERMEDIATE, N_HEADS, N_KV_HEADS, LayerWeights, RoPE


@dataclass
class ModelConfig:
    n_layers: int
    vocab: int
    rope_theta: float
    eps: float
    max_pos: int
    tie_embeddings: bool

    @staticmethod
    def from_dir(path: str | Path) -> ModelConfig:
        cfg = json.loads((Path(path) / "config.json").read_text())
        want = {"hidden_size": HIDDEN, "num_attention_heads": N_HEADS,
                "num_key_value_heads": N_KV_HEADS, "intermediate_size": INTERMEDIATE}
        bad = {k: (cfg.get(k), v) for k, v in want.items() if cfg.get(k) != v}
        if cfg.get("head_dim", HEAD_DIM) != HEAD_DIM:
            bad["head_dim"] = (cfg["head_dim"], HEAD_DIM)
        if bad:
            raise ValueError(f"not a Llama-3-8B-shaped model: (got, need) {bad}")
        if cfg.get("rope_scaling"):
            raise ValueError(f"rope_scaling {cfg['rope_scaling']} is not supported")
        if abs(cfg.get("rms_norm_eps", EPS) - EPS) > 1e-12:
            raise ValueError(f"rms_norm_eps {cfg['rms_norm_eps']} != {EPS}")
        if cfg["vocab_size"] % 64:
            raise ValueError(f"vocab {cfg['vocab_size']} is not a multiple of 64 (hgemm's N)")
        return ModelConfig(n_layers=cfg["num_hidden_layers"], vocab=cfg["vocab_size"],
                           rope_theta=float(cfg.get("rope_theta", 10000.0)),
                           eps=cfg.get("rms_norm_eps", EPS),
                           max_pos=cfg.get("max_position_embeddings", 8192),
                           tie_embeddings=bool(cfg.get("tie_word_embeddings", False)))

    def rope(self, max_pos: int | None = None, device="cuda") -> RoPE:
        return RoPE(max_pos or self.max_pos, device=device, theta=self.rope_theta)


class _Tensors:
    """Every tensor of a sharded safetensors checkpoint by name, read on demand."""

    def __init__(self, path: Path, device: str):
        from safetensors import safe_open

        self.files = {}
        self.where = {}
        for f in sorted(path.glob("*.safetensors")):
            h = safe_open(str(f), framework="pt", device=device)
            self.files[f] = h
            for k in h.keys():
                self.where[k] = f

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.files[self.where[name]].get_tensor(name).to(torch.bfloat16)


def load(path: str | Path, device="cuda", n_layers: int | None = None
         ) -> tuple[ModelWeights, ModelConfig]:
    """The checkpoint in `path` as engine weights, and its config. `n_layers` loads only the
    first layers (for quick tests on a truncated model)."""
    path = Path(path)
    cfg = ModelConfig.from_dir(path)
    t = _Tensors(path, device)
    n = cfg.n_layers if n_layers is None else n_layers
    layers = []
    for i in range(n):
        p = f"model.layers.{i}."
        q, k, v = (t[p + f"self_attn.{x}_proj.weight"] for x in "qkv")
        gate, up = t[p + "mlp.gate_proj.weight"], t[p + "mlp.up_proj.weight"]
        layers.append(LayerWeights(
            attn_norm=t[p + "input_layernorm.weight"],
            w_qkv=torch.cat([q, k, v], dim=0).t().contiguous(),
            w_o=t[p + "self_attn.o_proj.weight"].t().contiguous(),
            mlp_norm=t[p + "post_attention_layernorm.weight"],
            w_gate_up=sk.interleave_gate_up(gate.t(), up.t()).contiguous(),
            w_down=t[p + "mlp.down_proj.weight"].t().contiguous()))
        del q, k, v, gate, up
    embed = t["model.embed_tokens.weight"]
    head = embed if cfg.tie_embeddings else t["lm_head.weight"]
    w = ModelWeights(embed=embed, layers=layers, final_norm=t["model.norm.weight"],
                     lm_head=head.t().contiguous())
    torch.cuda.empty_cache()
    return w, cfg
