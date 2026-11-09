"""Real weights for the engine: a Hugging Face checkpoint in the Llama layout (a directory with
config.json and *.safetensors) loaded into `engine.ModelWeights`.

`HFConfig.from_dir` reads the shape from config.json into the engine's `ModelConfig`
(hidden_size, num_attention_heads, num_key_value_heads, head_dim or hidden_size / heads when
the config has none, intermediate_size, rms_norm_eps), and the weights `load` returns carry
it, so the models, the cache and the engine take theirs from the checkpoint. It refuses what
`ModelConfig.check` refuses, and what the engine cannot run yet (`_refuse`): another layout
than Llama's (Mistral's is the same), Qwen3's QK norm (#47), experts, biases, an activation
other than SiLU, a sliding window, a rope_scaling type other than llama3. What it converts:

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
# the model types whose checkpoints have Llama's tensor names and blocks
LAYOUTS = ("llama", "mistral")
# the keys a config of a mixture-of-experts family gives its expert count in
MOE_KEYS = ("num_experts", "num_local_experts", "n_routed_experts")


def _rope_params(cfg: dict) -> tuple[float, dict | None]:
    """(theta, rope_scaling) from a config: rope_theta and rope_scaling, or the
    rope_parameters dict transformers 5 writes in their place (theta and type in one).
    Read the way transformers 5 reads a mix of the two: rope_scaling wins over
    rope_parameters, and the top-level rope_theta fills in a missing theta."""
    rp = cfg.get("rope_scaling") or cfg.get("rope_parameters")
    theta = float(cfg.get("rope_theta", DEFAULT_THETA))
    if not rp:
        return theta, None
    kind = rp.get("rope_type", rp.get("type", "default"))
    return float(rp.get("rope_theta", theta)), None if kind == "default" else rp


def _refuse(cfg: dict) -> None:
    """Raises ValueError for a config whose model the engine cannot run yet, with the reason.
    The shapes are ModelConfig.check's, after this."""
    kind = cfg.get("model_type", "llama")
    if kind.startswith("qwen3"):
        raise ValueError(f"model_type {kind!r}: Qwen3 runs an RMSNorm over each head of q and k "
                         "before RoPE (QK norm), which the engine does not have yet (#47)")
    moe = {k: cfg[k] for k in MOE_KEYS if cfg.get(k)}
    if moe:
        raise ValueError(f"model_type {kind!r} is a mixture of experts ({moe}): the engine runs "
                         "MoE layers, but hf.py does not load experts yet")
    if kind not in LAYOUTS:
        raise ValueError(f"model_type {kind!r}: hf.py reads the Llama layout, model types "
                         f"{LAYOUTS}")
    if cfg.get("hidden_act", "silu") != "silu":
        raise ValueError(f"hidden_act {cfg['hidden_act']!r}: the MLP is SwiGLU (hgemm_swiglu's "
                         "epilogue), SiLU only")
    for k in ("attention_bias", "mlp_bias"):
        if cfg.get(k):
            raise ValueError(f"{k} is set: the engine's projections have no bias")
    if cfg.get("sliding_window") is not None and cfg.get("use_sliding_window", True):
        raise ValueError(f"sliding_window {cfg['sliding_window']}: the engine attends to the "
                         "whole context, which changes the outputs past the window")
    if cfg.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError(f"partial_rotary_factor {cfg['partial_rotary_factor']}: the RoPE "
                         "kernels rotate the whole head")
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

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def __getitem__(self, name: str) -> torch.Tensor:
        return self.files[self.where[name]].get_tensor(name).to(torch.bfloat16)


def _check_layer(t: _Tensors, p: str, c: ModelConfig, tensors: dict[str, torch.Tensor]) -> None:
    """Refuses a layer with tensors the config did not announce (QK norm, biases) or of
    another shape than the config's."""
    if p + "self_attn.q_norm.weight" in t:
        raise ValueError(f"{p}self_attn.q_norm: the checkpoint has QK norm, which the engine "
                         "does not have yet (#47)")
    for n in ("q", "k", "v", "o"):
        if p + f"self_attn.{n}_proj.bias" in t:
            raise ValueError(f"{p}self_attn.{n}_proj.bias: the engine's projections have no bias")
    want = {"self_attn.q_proj": (c.q_width, c.hidden), "self_attn.k_proj": (c.kv_width, c.hidden),
            "self_attn.v_proj": (c.kv_width, c.hidden), "self_attn.o_proj": (c.hidden, c.q_width),
            "mlp.gate_proj": (c.intermediate, c.hidden), "mlp.up_proj": (c.intermediate, c.hidden),
            "mlp.down_proj": (c.hidden, c.intermediate)}
    for n, x in tensors.items():
        if tuple(x.shape) != want[n]:
            raise ValueError(f"{p}{n}.weight is {tuple(x.shape)}, the config says {want[n]}")


def load(path: str | Path, device="cuda", n_layers: int | None = None
         ) -> tuple[ModelWeights, HFConfig]:
    """The checkpoint in `path` as engine weights of its own shape (`ModelWeights.config`),
    and its config. `n_layers` loads only the first layers (for quick tests on a truncated
    model)."""
    path = Path(path)
    cfg = HFConfig.from_dir(path)
    c = cfg.shape
    t = _Tensors(path, device)
    n = cfg.n_layers if n_layers is None else n_layers
    layers = []
    for i in range(n):
        p = f"model.layers.{i}."
        x = {name: t[p + name + ".weight"] for name in (
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
            "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")}
        _check_layer(t, p, c, x)
        layers.append(LayerWeights(
            attn_norm=t[p + "input_layernorm.weight"],
            w_qkv=torch.cat([x[f"self_attn.{s}_proj"] for s in "qkv"], dim=0).t().contiguous(),
            w_o=x["self_attn.o_proj"].t().contiguous(),
            mlp_norm=t[p + "post_attention_layernorm.weight"],
            w_gate_up=sk.interleave_gate_up(x["mlp.gate_proj"].t(),
                                            x["mlp.up_proj"].t()).contiguous(),
            w_down=x["mlp.down_proj"].t().contiguous()))
        del x
    embed = t["model.embed_tokens.weight"]
    if tuple(embed.shape) != (cfg.vocab, c.hidden):
        raise ValueError(f"model.embed_tokens is {tuple(embed.shape)}, the config says "
                         f"{(cfg.vocab, c.hidden)}")
    head = embed if cfg.tie_embeddings else t["lm_head.weight"]
    w = ModelWeights(embed=embed, layers=layers, final_norm=t["model.norm.weight"],
                     lm_head=head.t().contiguous(), config=c)
    torch.cuda.empty_cache()
    return w, cfg
