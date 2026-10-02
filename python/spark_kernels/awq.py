"""Activation-aware scales for int4 weights (the scale search of AWQ, Lin et al. 2023), on the
engine's weight layout.

Round to nearest gives every weight in a group of 128 the same step, so the few input
channels that carry large activations lose as much relative precision as the rest while
their errors weigh more in the output. Scaling input channel k of a weight up by s_k before
quantizing, and the activation down by s_k, gives those channels finer steps at no cost in
the bf16 math. The activation side folds into the op before the projection, so nothing is
added at run time:

    qkv      input is rmsnorm(x) * attn_norm       attn_norm /= s
    o        input is softmax(q k^T) v per head    v columns of w_qkv /= s (tied over the four
                                                   query heads that read one K/V head)
    gate_up  input is rmsnorm(x) * mlp_norm        mlp_norm /= s
    down     input is silu(gate) * up              up columns of w_gate_up /= s

For each projection the search tries s = mean|x|^alpha (normalized to a geometric mean of 1)
for alpha on a grid in [0, 1) and keeps the alpha whose int4 product is closest to the bf16
one on calibration tokens: || x @ W - (x / s) @ Q(W * s) ||^2. Layers are processed in order,
each one's calibration inputs coming out of the layers before it with their scales applied.

    scales = awq.search(weights, rope, ids)     # ids [B, S] calibration tokens
    awq.apply(weights, scales)                  # in place, still bf16; then quantize
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import reference as ref
from .layer import EPS, HEAD_DIM, HIDDEN, KV_WIDTH, N_HEADS, N_KV_HEADS, Q_WIDTH

GROUP = N_HEADS // N_KV_HEADS


def fake_quant(w: torch.Tensor, asym: bool) -> torch.Tensor:
    """The bf16 weight int4 round to nearest stands for (what w4gemm multiplies)."""
    return ref.w4_dequantize(*ref.w4_quantize(w, asym))


def _rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [B, S, H, D] rotated by cos, sin [S, D] (rotate-half), fp32 math."""
    xf = x.float()
    rot = torch.cat([-xf[..., HEAD_DIM // 2:], xf[..., :HEAD_DIM // 2]], dim=-1)
    return (xf * cos[None, :, None] + rot * sin[None, :, None]).to(x.dtype)


def _search(x: torch.Tensor, w: torch.Tensor, stat: torch.Tensor, asym: bool,
            grid: int) -> tuple[torch.Tensor, float, float]:
    """The scale s [K] (float32) of the best alpha for x [T, K] @ w [K, N], its error and the
    error of plain round to nearest (alpha = 0), both relative to ||x @ w||^2."""
    want = (x @ w).float()
    norm = want.square().sum().item()
    best = (None, float("inf"), 0.0)
    rtn = 0.0
    stat = stat.float().clamp(min=1e-6)
    for i in range(grid):
        alpha = i / grid
        s = stat.pow(alpha)
        s = s / (s.max() * s.min()).sqrt()
        ws = (w.float() * s[:, None]).to(torch.bfloat16)
        xs = (x.float() / s).to(torch.bfloat16)
        err = ((xs @ fake_quant(ws, asym)).float() - want).square().sum().item() / norm
        if i == 0:
            rtn = err
        if err < best[1]:
            best = (s, err, alpha)
    return best[0], best[1], rtn


@torch.no_grad()
def search(weights, rope, ids: torch.Tensor, asym: bool = False, grid: int = 20,
           sample: int = 4096, log=None) -> list[dict[str, torch.Tensor]]:
    """Per layer, the input scale s [K] (float32) of each projection, searched on the
    calibration tokens ids [B, S] (int64, every row a sequence from position 0). The error
    of each search is measured on `sample` of the B * S tokens (evenly spaced). Applies the
    scales to `weights` as it goes (`apply` with the result on fresh weights does the same)."""
    B, S = ids.shape
    dev = weights.embed.device
    x = F.embedding(ids.to(dev), weights.embed)  # [B, S, 4096]
    cos, sin = rope.cos[:S], rope.sin[:S]
    pick = torch.linspace(0, B * S - 1, min(sample, B * S), device=dev).long()
    out = []
    for li, lw in enumerate(weights.layers):
        sc = {}
        info = []

        def run(name, inp, w, sc=sc, info=info):
            flat = inp.reshape(-1, inp.shape[-1])
            stat = flat.float().abs().mean(dim=0)
            if name == "o":  # one scale per K/V channel, shared by the heads that read it
                stat = stat.view(N_KV_HEADS, GROUP, HEAD_DIM).mean(dim=1)
                stat = stat[:, None].expand(N_KV_HEADS, GROUP, HEAD_DIM).reshape(-1)
            s, err, rtn = _search(flat[pick], w, stat, asym, grid)
            sc[name] = s
            info.append(f"{name} {rtn:.2e}->{err:.2e}")
            return s

        h = F.rms_norm(x, (HIDDEN,), lw.attn_norm, EPS)
        s = run("qkv", h, lw.w_qkv)
        _apply_layer(lw, {"qkv": s})
        h = F.rms_norm(x, (HIDDEN,), lw.attn_norm, EPS)
        qkv = h @ lw.w_qkv
        q = qkv[..., :Q_WIDTH].view(B, S, N_HEADS, HEAD_DIM)
        k = qkv[..., Q_WIDTH:Q_WIDTH + KV_WIDTH].view(B, S, N_KV_HEADS, HEAD_DIM)
        v = qkv[..., Q_WIDTH + KV_WIDTH:].view(B, S, N_KV_HEADS, HEAD_DIM)
        q, k = _rope(q, cos, sin), _rope(k, cos, sin)
        o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                           v.transpose(1, 2), is_causal=True, enable_gqa=True)
        o = o.transpose(1, 2).reshape(B, S, Q_WIDTH)
        s = run("o", o, lw.w_o)
        _apply_layer(lw, {"o": s})
        o = o / s.to(o.dtype)  # what the scaled v columns now produce
        x = x + o @ lw.w_o
        h2 = F.rms_norm(x, (HIDDEN,), lw.mlp_norm, EPS)
        s = run("gate_up", h2, lw.w_gate_up)
        _apply_layer(lw, {"gate_up": s})
        gu = F.rms_norm(x, (HIDDEN,), lw.mlp_norm, EPS) @ lw.w_gate_up
        a = F.silu(gu[..., 0::2]) * gu[..., 1::2]
        s = run("down", a, lw.w_down)
        _apply_layer(lw, {"down": s})
        a = a / s.to(a.dtype)
        x = x + a @ lw.w_down
        out.append(sc)
        if log is not None:
            log(f"layer {li:2d}: " + ", ".join(info))
    return out


def _apply_layer(lw, sc: dict[str, torch.Tensor]) -> None:
    """Folds the scales of one layer into its bf16 weights (in place)."""
    for name, s in sc.items():
        sf = s.float()
        if name == "qkv":
            lw.attn_norm = (lw.attn_norm.float() / sf).to(torch.bfloat16)
            lw.w_qkv = (lw.w_qkv.float() * sf[:, None]).to(torch.bfloat16)
        elif name == "o":
            kv = sf.view(N_KV_HEADS, GROUP, HEAD_DIM)[:, 0].reshape(-1)  # tied over the group
            w = lw.w_qkv.float()
            w[:, Q_WIDTH + KV_WIDTH:] /= kv
            lw.w_qkv = w.to(torch.bfloat16)
            lw.w_o = (lw.w_o.float() * sf[:, None]).to(torch.bfloat16)
        elif name == "gate_up":
            lw.mlp_norm = (lw.mlp_norm.float() / sf).to(torch.bfloat16)
            lw.w_gate_up = (lw.w_gate_up.float() * sf[:, None]).to(torch.bfloat16)
        elif name == "down":
            w = lw.w_gate_up.float()
            w[:, 1::2] /= sf  # up_j sits at column 2j + 1
            lw.w_gate_up = w.to(torch.bfloat16)
            lw.w_down = (lw.w_down.float() * sf[:, None]).to(torch.bfloat16)
        else:
            raise ValueError(f"unknown projection {name!r}")


def apply(weights, scales: list[dict[str, torch.Tensor]]) -> None:
    """Folds the scales `search` found into fresh bf16 weights, in place, in its order."""
    if len(scales) != len(weights.layers):
        raise ValueError(f"{len(scales)} layers of scales for {len(weights.layers)} layers")
    for lw, sc in zip(weights.layers, scales, strict=True):
        for name in ("qkv", "o", "gate_up", "down"):
            _apply_layer(lw, {name: sc[name].to(lw.w_qkv.device)})
