"""One Llama-3-8B decoder layer built from the kernels in this package, and the same layer in
plain PyTorch for parity and timing.

Shapes are Llama-3-8B's: hidden 4096, 32 query heads over 8 K/V heads of 128, a 14336-wide
SwiGLU MLP, RMSNorm with eps 1e-5, RoPE with theta 500000. Everything is bf16 with fp32
math inside every kernel. The weights are plain tensors in the [K, N] layout `hgemm` takes
(`x @ w`), with q, k and v fused into one [4096, 6144] weight and gate and up into one
[4096, 28672] weight, so a decode step streams each of the four weight matrices once: 436 MB
per layer, which is more than the 96 MB L2 on its own.

The residual stream is handled the way a stack of layers runs it: the layer takes the
residual stream `x` and the previous layer's pending MLP output `delta`, fuses `x += delta`
into its first RMSNorm (`add_rmsnorm_`), fuses the attention output's add into the second
norm, and returns the updated `x` and its own MLP output as the next layer's `delta`. Both
adds of a layer are therefore inside a norm kernel, and `finish(x, delta)` is the one plain
add a model does after its last layer. With `delta=None` the first norm is a plain `rmsnorm`
(the first layer of a model).

    prefill(x, cache, delta=None)  x is [B, S, 4096]; causal attention over the sequence;
                                   writes positions 0..S-1 of a fresh cache
    decode(x, cache, delta=None)   x is [B, 1, 4096]; attention over the cache plus this
                                   token; appends it to the cache

`KVCache` holds K and V as [B, 8, capacity, 128] tensors, the layout the attention kernel
reads. The kernel takes a contiguous [B, H_kv, S_kv, D] tensor and has no head stride, so a
step whose cache is not full attends over a contiguous copy of the filled part (an O(L)
copy, `KVCache.kv()` says when it happens); a cache with `capacity == length` after the
append is read in place. RoPE, the q head transpose and the cache append are one kernel
(`rope_append_`, src/kernels/rope.cu); the q/k/v column split is a view of the GEMM output
and the attention output's head transpose is a torch copy (docs/design/layer.md).
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from . import ops as sk

# When True, SparkLayer wraps each stage of its forward in a torch.profiler.record_function
# range so a profile can charge every kernel (ours and torch's) to a stage; scripts/bench_layer.py
# turns it on for its breakdown. Off by default: each range costs host time on a decode step.
PROFILE_STAGES = False


def _stage(name: str):
    if PROFILE_STAGES:
        return torch.profiler.record_function(name)
    return contextlib.nullcontext()

HIDDEN = 4096
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 14336
EPS = 1e-5
ROPE_THETA = 500000.0

Q_WIDTH = N_HEADS * HEAD_DIM  # 4096
KV_WIDTH = N_KV_HEADS * HEAD_DIM  # 1024
QKV_WIDTH = Q_WIDTH + 2 * KV_WIDTH  # 6144


@dataclass
class LayerWeights:
    """bf16 weights in the [K, N] layout of `hgemm` (activations @ weight)."""

    attn_norm: torch.Tensor  # [4096]
    w_qkv: torch.Tensor  # [4096, 6144]: q | k | v columns
    w_o: torch.Tensor  # [4096, 4096]
    mlp_norm: torch.Tensor  # [4096]
    w_gate_up: torch.Tensor  # [4096, 28672]: gate | up columns
    w_down: torch.Tensor  # [14336, 4096]

    @staticmethod
    def random(device="cuda", seed: int = 0) -> LayerWeights:
        """Weights scaled so the activations stay O(1) through the layer: N(0, 1/sqrt(K))
        for the projections, 1 + 0.1 N(0, 1) for the norms."""
        g = torch.Generator(device="cpu").manual_seed(seed)

        def proj(k, n):
            return (torch.randn(k, n, generator=g) / k**0.5).to(device=device, dtype=torch.bfloat16)

        def norm(n):
            return (1.0 + 0.1 * torch.randn(n, generator=g)).to(device=device, dtype=torch.bfloat16)

        return LayerWeights(attn_norm=norm(HIDDEN), w_qkv=proj(HIDDEN, QKV_WIDTH),
                            w_o=proj(Q_WIDTH, HIDDEN), mlp_norm=norm(HIDDEN),
                            w_gate_up=proj(HIDDEN, 2 * INTERMEDIATE),
                            w_down=proj(INTERMEDIATE, HIDDEN))

    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in
                   (self.attn_norm, self.w_qkv, self.w_o, self.mlp_norm, self.w_gate_up,
                    self.w_down))

    def to_float(self) -> LayerWeights:
        return LayerWeights(*(t.float() for t in
                              (self.attn_norm, self.w_qkv, self.w_o, self.mlp_norm,
                               self.w_gate_up, self.w_down)))


class KVCache:
    """K and V for one layer as [B, H_kv, capacity, D] bf16, filled up to `length`."""

    def __init__(self, batch: int, capacity: int, device="cuda", dtype=torch.bfloat16):
        self.k = torch.zeros(batch, N_KV_HEADS, capacity, HEAD_DIM, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self.length = 0

    @property
    def capacity(self) -> int:
        return self.k.shape[2]

    def reserve(self, s: int) -> int:
        """Claims the next s positions; returns the first one."""
        if self.length + s > self.capacity:
            raise ValueError(f"cache of capacity {self.capacity} cannot take {s} more tokens "
                             f"at length {self.length}")
        pos0 = self.length
        self.length += s
        return pos0

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """k, v are [B, S, H_kv, D]; written at positions length..length+S-1 (torch copies)."""
        s = k.shape[1]
        pos0 = self.reserve(s)
        self.k[:, :, pos0:pos0 + s] = k.transpose(1, 2)
        self.v[:, :, pos0:pos0 + s] = v.transpose(1, 2)

    def kv(self) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """(k, v, copied) over the filled positions, contiguous as the attention kernel needs
        them. The slice is a view when the cache is full (or has one head in all); otherwise
        it is a contiguous copy of the filled part, `copied` says so, and that copy is the
        cache-append cost of not having a strided kernel."""
        k = self.k[:, :, :self.length]
        v = self.v[:, :, :self.length]
        if k.is_contiguous():
            return k, v, False
        return k.contiguous(), v.contiguous(), True


class RoPE:
    """cos and sin tables for positions 0..max_pos-1 in the rotate-half convention Llama
    uses (the first half of the head pairs with the second half), fp32."""

    def __init__(self, max_pos: int, device="cuda", theta: float = ROPE_THETA):
        inv_freq = 1.0 / (theta ** (torch.arange(0, HEAD_DIM, 2, device=device).float() / HEAD_DIM))
        pos = torch.arange(max_pos, device=device).float()
        freqs = torch.outer(pos, inv_freq)  # [max_pos, 64]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_pos, 128]
        self.cos = emb.cos()
        self.sin = emb.sin()

    def tables(self, pos0: int, s: int) -> tuple[torch.Tensor, torch.Tensor]:
        if pos0 + s > self.cos.shape[0]:
            raise ValueError(f"RoPE tables cover {self.cos.shape[0]} positions, need {pos0 + s}")
        return self.cos[pos0:pos0 + s], self.sin[pos0:pos0 + s]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x is [B, S, H, D] (any strides), cos and sin [S, D]; fp32 math, x's dtype out."""
    xf = x.float()
    x1, x2 = xf[..., :HEAD_DIM // 2], xf[..., HEAD_DIM // 2:]
    rot = torch.cat([-x2, x1], dim=-1)
    c = cos[None, :, None, :]
    s = sin[None, :, None, :]
    return (xf * c + rot * s).to(x.dtype)


def finish(x: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    """The residual add a model does after its last layer (every other add is fused into the
    next layer's first norm)."""
    return (x.float() + delta.float()).to(x.dtype)


def _split_qkv(qkv: torch.Tensor, b: int, s: int):
    q = qkv[:, :Q_WIDTH].view(b, s, N_HEADS, HEAD_DIM)
    k = qkv[:, Q_WIDTH:Q_WIDTH + KV_WIDTH].view(b, s, N_KV_HEADS, HEAD_DIM)
    v = qkv[:, Q_WIDTH + KV_WIDTH:].view(b, s, N_KV_HEADS, HEAD_DIM)
    return q, k, v


class SparkLayer:
    """The layer on this package's kernels: rmsnorm / add_rmsnorm_, hgemm, rope_append_,
    attention (GQA, the flash-decoding kernel at decode), swiglu. The attention output's head
    transpose is the one torch copy left."""

    def __init__(self, weights: LayerWeights, rope: RoPE):
        self.w = weights
        self.rope = rope

    def _forward(self, x: torch.Tensor, cache: KVCache, delta: torch.Tensor | None,
                 causal: bool):
        w = self.w
        b, s, _ = x.shape
        x2 = x.view(b * s, HIDDEN)  # the residual stream, updated in place below
        with _stage("norm1"):
            if delta is None:
                h = sk.rmsnorm(x2, w.attn_norm, EPS)
            else:
                d2 = delta.reshape(b * s, HIDDEN)
                if not d2.is_contiguous():  # a slice of a longer delta, say
                    d2 = d2.contiguous()
                h = sk.add_rmsnorm_(d2, x2, w.attn_norm, EPS)
        with _stage("qkv_gemm"):
            qkv = sk.hgemm(h, w.w_qkv)  # [B*S, 6144]
        with _stage("rope_append"):
            # RoPE on q and k, q into [B, 32, S, 128], k and v into the cache: one launch
            pos0 = cache.reserve(s)
            self.rope.tables(pos0, s)  # the range check
            q = sk.rope_append_(qkv.view(b, s, QKV_WIDTH), self.rope.cos, self.rope.sin,
                                cache.k, cache.v, pos0, N_HEADS, N_KV_HEADS)
            kc, vc, _ = cache.kv()
        with _stage("attention"):
            o = sk.attention(q, kc, vc, causal=causal)  # [B, 32, S, 128]
        with _stage("o_transpose"):
            o = o.transpose(1, 2).reshape(b * s, Q_WIDTH)
        with _stage("o_gemm"):
            attn = sk.hgemm(o, w.w_o)
        with _stage("norm2"):
            h2 = sk.add_rmsnorm_(attn, x2, w.mlp_norm, EPS)  # x += attn, then the norm
        with _stage("gate_up_gemm"):
            gu = sk.hgemm(h2, w.w_gate_up)  # [B*S, 28672]
        with _stage("swiglu"):
            a = sk.swiglu(gu[:, :INTERMEDIATE], gu[:, INTERMEDIATE:])  # strided halves
        with _stage("down_gemm"):
            down = sk.hgemm(a, w.w_down)
        return x, down.view(b, s, HIDDEN)

    def prefill(self, x: torch.Tensor, cache: KVCache, delta: torch.Tensor | None = None):
        """x is [B, S, 4096] bf16 contiguous, cache empty. Returns (x, delta_out); x is
        updated in place. Causal attention over the S tokens."""
        if cache.length != 0:
            raise ValueError("prefill needs an empty cache (the causal mask is top-left "
                             "aligned, as in torch)")
        return self._forward(x, cache, delta, causal=True)

    def decode(self, x: torch.Tensor, cache: KVCache, delta: torch.Tensor | None = None):
        """x is [B, 1, 4096] bf16 contiguous. Attends over the cache's `length` tokens plus
        this one and appends it. Returns (x, delta_out); x is updated in place."""
        if x.shape[1] != 1:
            raise ValueError("decode takes one token per sequence")
        return self._forward(x, cache, delta, causal=False)


class TorchLayer:
    """The same layer in plain PyTorch on the same weights: F.rms_norm, matmul, RoPE,
    F.scaled_dot_product_attention with enable_gqa, F.silu(gate) * up. Functional in x (it
    does not update the residual stream in place) so torch.compile can take it whole; the
    cache append is the one in-place write."""

    def __init__(self, weights: LayerWeights, rope: RoPE):
        self.w = weights
        self.rope = rope

    def _forward(self, x, cache: KVCache, delta, causal: bool):
        w = self.w
        b, s, _ = x.shape
        if delta is not None:
            x = x + delta
        h = F.rms_norm(x, (HIDDEN,), w.attn_norm, EPS)
        qkv = (h.view(b * s, HIDDEN) @ w.w_qkv)
        q, k, v = _split_qkv(qkv, b, s)
        cos, sin = self.rope.tables(cache.length, s)
        q = apply_rope(q, cos, sin).transpose(1, 2)
        cache.append(apply_rope(k, cos, sin), v)
        kc, vc, _ = cache.kv()
        o = F.scaled_dot_product_attention(q, kc, vc, is_causal=causal, enable_gqa=True)
        o = o.transpose(1, 2).reshape(b * s, Q_WIDTH)
        x = x + (o @ w.w_o).view(b, s, HIDDEN)
        h2 = F.rms_norm(x, (HIDDEN,), w.mlp_norm, EPS)
        gu = h2.view(b * s, HIDDEN) @ w.w_gate_up
        a = F.silu(gu[:, :INTERMEDIATE]) * gu[:, INTERMEDIATE:]
        down = a @ w.w_down
        return x, down.view(b, s, HIDDEN)

    def prefill(self, x, cache: KVCache, delta=None):
        if cache.length != 0:
            raise ValueError("prefill needs an empty cache")
        return self._forward(x, cache, delta, causal=True)

    def decode(self, x, cache: KVCache, delta=None):
        if x.shape[1] != 1:
            raise ValueError("decode takes one token per sequence")
        return self._forward(x, cache, delta, causal=False)
