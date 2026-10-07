"""One layer's MoE experts of a local DeepSeek V4.1 Flash checkpoint (Hugging Face
safetensors) as a `moe.MoEBlock1Bit`, the routed experts requantized to 1 (or 2) bits.

    block = load_moe_layer("~/models/DeepSeek-V4.1-Flash", layer=3, cfg=DEEPSEEK_V41_FLASH,
                           device="cuda")

The checkpoint is hundreds of GB, so every tensor is read on its own through
`safetensors.safe_open` and dropped once quantized: the live set is one expert's three bf16
matrices (3 * 5120 * 2304 * 2 bytes) plus the packed words of the experts done so far.

Key names are an ASSUMPTION about the checkpoint, taken from the DeepSeek-V3 HF layout, and
are all in `KeyMap` so a different checkpoint is a different KeyMap, not a code change:

    model.layers.{layer}.mlp.experts.{e}.gate_proj.weight       routed expert e, [inter, hidden]
    model.layers.{layer}.mlp.experts.{e}.up_proj.weight         [inter, hidden]
    model.layers.{layer}.mlp.experts.{e}.down_proj.weight       [hidden, inter]
    model.layers.{layer}.mlp.shared_experts.gate_proj.weight    the shared expert, same shapes
    model.layers.{layer}.mlp.shared_experts.up_proj.weight
    model.layers.{layer}.mlp.shared_experts.down_proj.weight
    model.layers.{layer}.mlp.gate.weight                        router, [E, hidden]
    model.layers.{layer}.mlp.gate.e_score_correction_bias       router bias, [E] (optional)

Every projection is nn.Linear's [out, in] and is transposed to the [in, out] = [K, N] layout
the quantizer takes. An expert stored in MXFP4 is assumed to be two tensors, `.weight` as
uint8 [out, in / 2] (two e2m1 codes per byte, element 2i in the low nibble) and
`.weight_scale` as uint8 [out, in / 32] e8m0 exponents (scale = 2 ** (byte - 127)) over 32
consecutive inputs; `mxfp4_dequantize` turns them into bf16. A tensor that is already bf16
(or fp16, fp32) is used as it is. None of this has run against the real checkpoint yet.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .moe import MoEBlock1Bit, MoEConfig, _cat_experts, quantize_experts, quantize_weight

# e2m1 magnitudes by the low three bits of a code; bit 3 is the sign
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MX_BLOCK = 32


@dataclass(frozen=True)
class KeyMap:
    """The tensor names of one MoE layer, as format strings over `layer` and `e`."""

    expert_gate: str = "model.layers.{layer}.mlp.experts.{e}.gate_proj"
    expert_up: str = "model.layers.{layer}.mlp.experts.{e}.up_proj"
    expert_down: str = "model.layers.{layer}.mlp.experts.{e}.down_proj"
    shared_gate: str = "model.layers.{layer}.mlp.shared_experts.gate_proj"
    shared_up: str = "model.layers.{layer}.mlp.shared_experts.up_proj"
    shared_down: str = "model.layers.{layer}.mlp.shared_experts.down_proj"
    router: str = "model.layers.{layer}.mlp.gate"
    router_bias: str = "model.layers.{layer}.mlp.gate.e_score_correction_bias"
    weight_suffix: str = ".weight"
    scale_suffix: str = ".weight_scale"


def mxfp4_dequantize(packed_u8: torch.Tensor, scales_u8: torch.Tensor) -> torch.Tensor:
    """MXFP4 [N, K/2] bytes (two e2m1 codes each, element 2i in the low nibble) with [N, K/32]
    e8m0 scales -> bf16 [N, K]: value = e2m1(code) * 2 ** (scale - 127)."""
    q = packed_u8.view(torch.uint8)
    codes = torch.stack((q & 15, q >> 4), dim=-1).reshape(q.shape[0], 2 * q.shape[1])
    mag = torch.tensor(E2M1_VALUES, device=q.device)[(codes & 7).long()]
    v = torch.where((codes & 8) != 0, -mag, mag)
    N, K = v.shape
    scale = torch.exp2(scales_u8.view(torch.uint8).float() - 127.0)
    v = v.reshape(N, K // MX_BLOCK, MX_BLOCK) * scale.reshape(N, K // MX_BLOCK, 1)
    return v.reshape(N, K).to(torch.bfloat16)


class _Checkpoint:
    """The tensors of a sharded safetensors checkpoint by name, each read on demand from its
    shard through `model.safetensors.index.json` (or, without an index, from the one
    `model.safetensors`)."""

    def __init__(self, path: Path, device: str):
        from safetensors import safe_open

        self.path = path
        self.device = device
        self._open = safe_open
        index = path / "model.safetensors.index.json"
        if index.exists():
            with open(index) as f:
                self.where = json.load(f)["weight_map"]
        else:
            single = path / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"no safetensors index or model.safetensors in {path}")
            with safe_open(str(single), framework="pt", device="cpu") as h:
                self.where = {k: single.name for k in h.keys()}
        self.handles: dict[str, object] = {}

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def __getitem__(self, name: str) -> torch.Tensor:
        shard = self.where[name]
        if shard not in self.handles:
            self.handles[shard] = self._open(str(self.path / shard), framework="pt",
                                             device=self.device)
        return self.handles[shard].get_tensor(name)

    def linear(self, prefix: str, keys: KeyMap) -> torch.Tensor:
        """The [out, in] weight under `prefix` as bf16, dequantized if stored as MXFP4."""
        w = self[prefix + keys.weight_suffix]
        if w.dtype == torch.uint8:
            scale_key = prefix + keys.scale_suffix
            if scale_key not in self:
                raise KeyError(f"{prefix} is uint8 but has no {keys.scale_suffix} tensor")
            return mxfp4_dequantize(w, self[scale_key])
        return w.to(torch.bfloat16)


def load_moe_layer(path_or_repo: str | Path, layer: int, cfg: MoEConfig, device: str = "cuda",
                   bits: int | None = None, keys: KeyMap | None = None) -> MoEBlock1Bit:
    """The MoE block of `layer` from the checkpoint directory `path_or_repo`, its routed and
    shared experts quantized to `bits` (default cfg.bits) on `device`. One expert is read,
    transposed to [K, N] and quantized at a time. `keys` defaults to `KeyMap()`."""
    keys = keys or KeyMap()
    if bits is not None and bits != cfg.bits:
        cfg = MoEConfig(**{**cfg.__dict__, "bits": bits})
    path = Path(path_or_repo).expanduser()
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a local checkpoint directory (a hub repo id "
                                "has to be downloaded first)")
    ckpt = _Checkpoint(path, device)
    H, N_I = cfg.hidden, cfg.inter

    def gate_up(gate_prefix: str, up_prefix: str) -> torch.Tensor:
        """[hidden, 2 inter]: the gate columns then the up columns."""
        g = ckpt.linear(gate_prefix, keys)
        u = ckpt.linear(up_prefix, keys)
        if g.shape != (N_I, H) or u.shape != (N_I, H):
            raise ValueError(f"{gate_prefix}: {tuple(g.shape)}, want {(N_I, H)}")
        return torch.cat((g.t(), u.t()), dim=1).contiguous()

    def down(prefix: str) -> torch.Tensor:
        d = ckpt.linear(prefix, keys)
        if d.shape != (H, N_I):
            raise ValueError(f"{prefix}: {tuple(d.shape)}, want {(H, N_I)}")
        return d.t().contiguous()

    gu_parts, dn_parts = [], []
    for e in range(cfg.n_routed):
        f = dict(layer=layer, e=e)
        gu_parts.append(quantize_experts(gate_up(keys.expert_gate.format(**f),
                                                 keys.expert_up.format(**f)).unsqueeze(0),
                                         cfg.bits))
        dn_parts.append(quantize_experts(down(keys.expert_down.format(**f)).unsqueeze(0),
                                         cfg.bits))

    w_gate_up = _cat_experts(gu_parts, cfg.bits)
    w_down = _cat_experts(dn_parts, cfg.bits)

    shared_gu = shared_dn = None
    if cfg.n_shared:
        f = dict(layer=layer)
        shared_gu = quantize_weight(gate_up(keys.shared_gate.format(**f),
                                            keys.shared_up.format(**f)), cfg.bits)
        shared_dn = quantize_weight(down(keys.shared_down.format(**f)), cfg.bits)

    router_w = ckpt.linear(keys.router.format(layer=layer), keys)
    if router_w.shape != (cfg.n_routed, H):
        raise ValueError(f"router is {tuple(router_w.shape)}, want {(cfg.n_routed, H)}")
    router_w = router_w.t().contiguous().to(torch.bfloat16)
    bias_key = keys.router_bias.format(layer=layer)
    router_b = ckpt[bias_key].float() if bias_key in ckpt else None
    return MoEBlock1Bit(cfg, w_gate_up, w_down, shared_gu, shared_dn, router_w, router_b)
