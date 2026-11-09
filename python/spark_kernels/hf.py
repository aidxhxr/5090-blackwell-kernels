"""Real weights for the engine: a Hugging Face checkpoint in the Llama layout (a directory with
config.json and *.safetensors) loaded into `engine.ModelWeights`.

`HFConfig.from_dir` reads the shape from config.json into the engine's `ModelConfig`
(hidden_size, num_attention_heads, num_key_value_heads, head_dim or hidden_size / heads when
the config has none, intermediate_size, rms_norm_eps), and the weights `load` returns carry
it, so the models, the cache and the engine take theirs from the checkpoint. It refuses what
`ModelConfig.check` refuses, a rope_scaling type other than llama3 and a vocabulary hgemm
cannot take. What it converts:

    q_proj, k_proj, v_proj    [out, in] each, fused into w_qkv [hidden, (H_q + 2 H_kv) D]
                              = [q | k | v]
    o_proj, down_proj         transposed to [K, N]
    gate_proj, up_proj        transposed and interleaved (`interleave_gate_up`) into
                              w_gate_up [hidden, 2 * intermediate], the layout hgemm_swiglu
                              reads
    lm_head                   transposed to [hidden, V]; the embedding when the config ties
                              them

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
from .engine import ModelConfig, ModelWeights
from .layer import LayerWeights, RoPE

# transformers' defaults for a key a config.json leaves out (LlamaConfig's and MistralConfig's)
DEFAULT_EPS = 1e-6
DEFAULT_THETA = 10000.0


def _rope_params(cfg: dict) -> tuple[float, dict | None]:
    """(theta, rope_scaling) from a config: rope_theta and rope_scaling, or the
    rope_parameters dict transformers 5 writes in their place (theta and type in one)."""
    rp = cfg.get("rope_parameters")
    if rp:
        kind = rp.get("rope_type", rp.get("type", "default"))
        return float(rp.get("rope_theta", DEFAULT_THETA)), None if kind == "default" else rp
    return float(cfg.get("rope_theta", DEFAULT_THETA)), cfg.get("rope_scaling") or None


def _refuse(cfg: dict) -> None:
    """Raises ValueError for a config whose model the engine cannot run yet, with the reason.
    The shapes are ModelConfig.check's, after this."""
    _, scaling = _rope_params(cfg)
    rope_kind = scaling.get("rope_type", scaling.get("type")) if scaling else "default"
    if rope_kind not in ("default", "llama3"):
        raise ValueError(f"rope_scaling {scaling} is not supported: plain RoPE and llama3 only")
    if cfg["vocab_size"] % 64:
        raise ValueError(f"vocab {cfg['vocab_size']} is not a multiple of 64 (hgemm's N)")


@dataclass
class HFConfig:
    """What a checkpoint's config.json says: `shape`, the engine's ModelConfig the weights
    carry, and what the engine takes apart from it (the layer count, the vocabulary, the
    RoPE tables' theta, scaling and length)."""

    shape: ModelConfig
    n_layers: int
    vocab: int
    rope_theta: float
    max_pos: int
    tie_embeddings: bool
    rope_scaling: dict | None = None

    @property
    def eps(self) -> float:
        return self.shape.eps

    @staticmethod
    def from_dir(path: str | Path) -> HFConfig:
        path = Path(path)
        cfg = json.loads((path / "config.json").read_text())
        _refuse(cfg)
        hidden, heads = cfg["hidden_size"], cfg["num_attention_heads"]
        head_dim = cfg.get("head_dim") or hidden // heads  # transformers' fallback
        if not cfg.get("head_dim") and hidden % heads:
            raise ValueError(f"no head_dim, and hidden_size {hidden} is not a multiple of "
                             f"{heads} heads")
        shape = ModelConfig(name=path.resolve().name, hidden=hidden, n_heads=heads,
                            n_kv_heads=cfg.get("num_key_value_heads") or heads,
                            head_dim=head_dim, intermediate=cfg["intermediate_size"],
                            eps=float(cfg.get("rms_norm_eps", DEFAULT_EPS)))
        shape.check()
        theta, scaling = _rope_params(cfg)
        return HFConfig(shape=shape, n_layers=cfg["num_hidden_layers"], vocab=cfg["vocab_size"],
                        rope_theta=theta, max_pos=cfg.get("max_position_embeddings", 8192),
                        tie_embeddings=bool(cfg.get("tie_word_embeddings", False)),
                        rope_scaling=scaling)

    def rope(self, max_pos: int | None = None, device="cuda") -> RoPE:
        return RoPE(max_pos or self.max_pos, device=device, theta=self.rope_theta,
                    scaling=self.rope_scaling, head_dim=self.shape.head_dim)


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
         ) -> tuple[ModelWeights, HFConfig]:
    """The checkpoint in `path` as engine weights of its own shape (`ModelWeights.config`),
    and its config. `n_layers` loads only the first layers (for quick tests on a truncated
    model)."""
    path = Path(path)
    cfg = HFConfig.from_dir(path)
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
                     lm_head=head.t().contiguous(), config=cfg.shape)
    torch.cuda.empty_cache()
    return w, cfg
