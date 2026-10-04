"""A small serving engine: the whole Llama-3-8B (32 layers, random weights) on a paged K/V
cache, batched varlen prefill and continuous-batching decode with CUDA graphs, on this
package's kernels; and the same model in plain PyTorch for comparison.

The pieces (docs/design/serving.md):

    PagedKVCache   K and V of every layer as [n_layers, num_pages, 8, page, 128] bf16, a free
                   list of page ids, one reserved scratch page that padding tokens write into
    SparkModel     prefill(batch, cache) and decode(batch, cache) -> logits, on rmsnorm,
                   hgemm (with the residual add and the SwiGLU fused into its epilogue),
                   rope_append_paged_, attention_varlen and paged_decode
    TorchModel     the same model in PyTorch: F.rms_norm, matmul, the RoPE in fp32,
                   F.scaled_dot_product_attention per prompt at prefill and over the gathered
                   pages with a length mask at decode; `compile=True` runs the dense parts
                   under torch.compile
    Engine         the scheduler: requests wait in a queue, are admitted into free batch
                   slots while pages last, prefilled together in one packed batch, then
                   decoded one token per step with the other running sequences until they
                   have their tokens; a finished sequence frees its slot and pages at once.
                   A prompt longer than the prefill token budget is prefilled in chunks of
                   the budget, each attending to the keys the earlier ones cached (the
                   bottom-right causal mask of attention_varlen), so a 100K-token prompt
                   never needs 100K rows of activations at once. Pages are taken as the
                   tokens arrive: a request is admitted with the pages its prompt needs,
                   grows by one page every `page` decode steps, and when the cache runs out
                   the youngest running sequence is preempted (its pages freed, its prompt
                   and the tokens it has so far put back at the head of the queue, to be
                   prefilled again), as vLLM does, so the cache holds as many requests as
                   their current lengths allow instead of as many as their worst case would

A sequence keeps its batch slot from admission to the end, so the decode step's inputs
(token ids, positions, cache slots, lengths, block table rows) live in per-slot device
buffers. The decode step for a batch of up to Bp slots (Bp a power of two) is captured into a
CUDA graph once, with every slot empty, and replayed every step after the host rewrites the
positions, slots and lengths of the slots in use; paged_decode reads the lengths on the device,
so the graph does not depend on them. The next token of every slot is the argmax of its logits
(greedy), written back into the token-id buffer by the graph itself, so a decode loop never
copies a token to the host.

Weights: `ModelWeights.random()` draws every matrix on the GPU. w_gate_up is stored in the
interleaved layout of `interleave_gate_up` (gate_j at column 2j, up_j at 2j + 1), the layout
`hgemm_swiglu` takes; the torch model reads the same tensor as its two strided halves.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from . import ops as sk
from .layer import (
    EPS,
    HEAD_DIM,
    HIDDEN,
    INTERMEDIATE,
    KV_WIDTH,
    N_HEADS,
    N_KV_HEADS,
    Q_WIDTH,
    QKV_WIDTH,
    LayerWeights,
    RoPE,
)

N_LAYERS = 32
VOCAB = 128256  # a multiple of 64, as hgemm needs for N
PAGE = 16
BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
# the largest decode bucket captured into a CUDA graph; larger ones run eagerly (_capture_all)
GRAPH_MAX = 64


def cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass
class ModelWeights:
    """bf16 weights of the whole model in hgemm's [K, N] layout."""

    embed: torch.Tensor  # [V, 4096]
    layers: list[LayerWeights]  # w_gate_up interleaved (see the module docstring)
    final_norm: torch.Tensor  # [4096]
    lm_head: torch.Tensor  # [4096, V]

    @staticmethod
    def random(n_layers: int = N_LAYERS, vocab: int = VOCAB, device="cuda",
               seed: int = 0) -> ModelWeights:
        """N(0, 1/K) projections and 1 + 0.1 N(0, 1) norms, as LayerWeights.random, drawn on
        the GPU (14 GB of layer weights take seconds there and minutes on the CPU)."""
        g = torch.Generator(device=device).manual_seed(seed)

        def proj(k, n):
            return torch.empty(k, n, device=device, dtype=torch.bfloat16).normal_(
                0.0, 1.0 / math.sqrt(k), generator=g)

        def norm(n):
            return torch.empty(n, device=device, dtype=torch.bfloat16).normal_(
                1.0, 0.1, generator=g)

        layers = [LayerWeights(attn_norm=norm(HIDDEN), w_qkv=proj(HIDDEN, QKV_WIDTH),
                               w_o=proj(Q_WIDTH, HIDDEN), mlp_norm=norm(HIDDEN),
                               w_gate_up=proj(HIDDEN, 2 * INTERMEDIATE),
                               w_down=proj(INTERMEDIATE, HIDDEN)) for _ in range(n_layers)]
        embed = torch.empty(vocab, HIDDEN, device=device, dtype=torch.bfloat16).normal_(
            0.0, 1.0, generator=g)
        return ModelWeights(embed=embed, layers=layers, final_norm=norm(HIDDEN),
                            lm_head=proj(HIDDEN, vocab))

    def bytes(self) -> int:
        return (sum(w.bytes() for w in self.layers) + self.embed.numel() * 2 +
                self.lm_head.numel() * 2 + self.final_norm.numel() * 2)


class PagedKVCache:
    """K and V of every layer in pages of `page` tokens: k[l] and v[l] are the
    [num_pages, 8, page, 128] caches the kernels take. The last page is scratch: padding
    tokens of a captured decode step write there and nothing reads it."""

    def __init__(self, n_layers: int, num_pages: int, page: int = PAGE, device="cuda"):
        if num_pages < 2:
            raise ValueError("need at least two pages (one is scratch)")
        shape = (n_layers, num_pages, N_KV_HEADS, page, HEAD_DIM)
        self.k = torch.zeros(shape, device=device, dtype=torch.bfloat16)
        self.v = torch.zeros(shape, device=device, dtype=torch.bfloat16)
        self.page = page
        self.scratch = num_pages - 1
        self.free = list(range(num_pages - 2, -1, -1))

    @staticmethod
    def page_bytes(n_layers: int, page: int = PAGE) -> int:
        return 2 * n_layers * N_KV_HEADS * page * HEAD_DIM * 2

    @property
    def num_pages(self) -> int:
        return self.k.shape[1]

    def alloc(self, n: int) -> list[int]:
        if n > len(self.free):
            raise RuntimeError(f"out of cache pages: need {n}, {len(self.free)} free")
        return [self.free.pop() for _ in range(n)]

    def release(self, pages: list[int]) -> None:
        self.free.extend(reversed(pages))


@dataclass
class Batch:
    """The device inputs of one forward. Prefill packs the prompts' tokens back to back;
    decode has one token per slot."""

    ids: torch.Tensor  # [T] int64
    positions: torch.Tensor  # [T] int32
    slots: torch.Tensor  # [T] int32, page_id * page + row
    seq_lens: torch.Tensor  # [B] int32: keys of each sequence once this step's are appended
    block_table: torch.Tensor  # [B, max_pages] int32
    cu_seqlens: torch.Tensor | None = None  # [B + 1] int32 (prefill)
    last: torch.Tensor | None = None  # [B] int64: each prompt's last row (prefill)
    q_lens: list[int] | None = None  # host copy of the prompt lengths (the torch model's loop)
    max_len: int = 0  # host copy of max(seq_lens) (the torch model's gather)
    n_dec: int = 0  # prefill: the last n_dec sequences are one-token decode rows (mixed step)
    chunked: bool = False  # prefill: some prompt continues a context already in the cache


def _rope_tokens(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x [T, H, D] rotated per token by cos, sin [T, D] (rotate-half), fp32 math."""
    xf = x.float()
    rot = torch.cat([-xf[..., HEAD_DIM // 2:], xf[..., :HEAD_DIM // 2]], dim=-1)
    return (xf * cos[:, None] + rot * sin[:, None]).to(x.dtype)


class SparkModel:
    """The model on this package's kernels. Per layer: rmsnorm, the qkv GEMM, RoPE + the
    paged append, attention (varlen at prefill, paged decode at decode), the o GEMM with the
    residual add in its epilogue, rmsnorm, the gate/up GEMM with SwiGLU in its epilogue, the
    down GEMM with the residual add in its epilogue: eight launches.

    `weights_format` other than "bf16" (see `quant.FORMATS`) quantizes the four projections of
    every layer at construction and frees their bf16 copies: the LayerWeights objects of
    `weights` lose w_qkv, w_o, w_gate_up and w_down (set to None) unless `keep_bf16`. The
    embedding, the norms and lm_head stay bf16. The quantized GEMMs have no fused epilogue,
    so those layers run the residual adds inside the norms instead (`add_rmsnorm_`: the o
    output is added by the MLP norm, the down output by the next layer's attention norm) and
    the SwiGLU as its own launch on the [gate | up] halves (the gate/up weight is
    de-interleaved before it is quantized): nine launches plus the activation quantizers.
    `head_format` does the same for lm_head (freeing its bf16 copy unless `keep_bf16`).

    `prefill_variant` is the hgemm variant of the prefill GEMMs, the bf16 projections and a
    bf16 lm_head (decode steps and the quantized GEMMs are not affected). 4 keeps the prefill
    off the TMA kernels, whose first launch takes 3.4 GiB of device memory outside torch's
    allocator: what a 100K-token cache next to the bf16 weights on a 32 GB card does not
    have (docs/design/llm_models.md)."""

    context_prefill = True  # prefill rows may continue a cached context (attention_varlen)

    def __init__(self, weights: ModelWeights, rope: RoPE, weights_format: str = "bf16",
                 keep_bf16: bool = False, head_format: str = "bf16", prefill_variant: int = -1):
        self.w = weights
        self.rope = rope
        self.prefill_variant = prefill_variant
        # set to a list to collect the residual stream [T, 4096] after every layer (a parity
        # check against another implementation reads it); None, the default, costs nothing
        self.trace: list[torch.Tensor] | None = None
        self.format = weights_format
        self.lin = None
        self.head = None
        if head_format != "bf16":
            from .quant import make_linear

            self.head = make_linear(weights.lm_head, head_format)
            if not keep_bf16:
                weights.lm_head = None
        if weights_format != "bf16":
            from .quant import make_linear

            self.lin = []
            for lw in weights.layers:
                gu = lw.w_gate_up
                gu = torch.cat([gu[:, 0::2], gu[:, 1::2]], dim=1)  # [gate | up]
                self.lin.append({"qkv": make_linear(lw.w_qkv, weights_format),
                                 "o": make_linear(lw.w_o, weights_format),
                                 "gate_up": make_linear(gu, weights_format),
                                 "down": make_linear(lw.w_down, weights_format)})
                del gu
                if not keep_bf16:
                    lw.w_qkv = lw.w_o = lw.w_gate_up = lw.w_down = None
            torch.cuda.empty_cache()

    def weight_bytes(self) -> int:
        """Bytes of every weight the model holds: the projections in their format, plus the
        bf16 embedding, norms and lm_head."""
        w = self.w
        rest = w.embed.numel() * 2 + w.final_norm.numel() * 2
        rest += self.head.nbytes() if self.head is not None else w.lm_head.numel() * 2
        rest += sum((lw.attn_norm.numel() + lw.mlp_norm.numel()) * 2 for lw in w.layers)
        if self.lin is None:
            return rest + sum(getattr(lw, n).numel() * 2 for lw in w.layers
                              for n in ("w_qkv", "w_o", "w_gate_up", "w_down"))
        return rest + sum(m.nbytes() for d in self.lin for m in d.values())

    def dequantized(self) -> ModelWeights:
        """The bf16 weights the quantized projections stand for, in the engine's layout
        (gate/up interleaved again): what a reference model on these weights multiplies."""
        head = self.head.dequantize() if self.head is not None else self.w.lm_head
        if self.lin is None:
            return ModelWeights(embed=self.w.embed, layers=self.w.layers,
                                final_norm=self.w.final_norm, lm_head=head)
        layers = []
        for lw, d in zip(self.w.layers, self.lin, strict=True):
            gu = d["gate_up"].dequantize()
            layers.append(LayerWeights(
                attn_norm=lw.attn_norm, w_qkv=d["qkv"].dequantize(), w_o=d["o"].dequantize(),
                mlp_norm=lw.mlp_norm,
                w_gate_up=sk.interleave_gate_up(gu[:, :INTERMEDIATE], gu[:, INTERMEDIATE:]),
                w_down=d["down"].dequantize()))
        return ModelWeights(embed=self.w.embed, layers=layers, final_norm=self.w.final_norm,
                            lm_head=head)

    def _attention(self, qkv, x, b: Batch, kc, vc, decode: bool):
        q = sk.rope_append_paged_(qkv, self.rope.cos, self.rope.sin, b.positions, b.slots,
                                  kc, vc, N_HEADS, N_KV_HEADS)
        if decode:
            o = sk.paged_decode(q, kc, vc, b.block_table, b.seq_lens)
        elif b.n_dec:
            o = self._mixed_attention(q, kc, vc, b)
        else:
            o = sk.attention_varlen(q, kc, vc, b.cu_seqlens, b.seq_lens, b.block_table)
        return o.view(x.shape[0], Q_WIDTH)

    def _layers(self, x: torch.Tensor, b: Batch, cache: PagedKVCache, decode: bool):
        if self.lin is not None:
            return self._layers_quant(x, b, cache, decode)
        v = -1 if decode else self.prefill_variant
        for layer, w in enumerate(self.w.layers):
            kc, vc = cache.k[layer], cache.v[layer]
            h = sk.rmsnorm(x, w.attn_norm, EPS)
            qkv = sk.hgemm(h, w.w_qkv, v)
            o = self._attention(qkv, x, b, kc, vc, decode)
            sk.hgemm(o, w.w_o, v, residual=x, out=x)
            h2 = sk.rmsnorm(x, w.mlp_norm, EPS)
            a = sk.hgemm_swiglu(h2, w.w_gate_up, v)
            sk.hgemm(a, w.w_down, v, residual=x, out=x)
            if self.trace is not None:
                self.trace.append(x.clone())  # x is updated in place by the next layer
        return x

    @staticmethod
    def _mixed_attention(q, kc, vc, b: Batch) -> torch.Tensor:
        """A mixed step's attention: the prompts' rows through attention_varlen and the
        decode rows (one per sequence, at the end) through paged_decode. attention_varlen
        would give each decode row a whole 128-row query tile, 128x the work it needs."""
        nb = b.seq_lens.shape[0] - b.n_dec
        tp = q.shape[0] - b.n_dec
        o = torch.empty_like(q)
        o[:tp] = sk.attention_varlen(q[:tp], kc, vc, b.cu_seqlens[:nb + 1], b.seq_lens[:nb],
                                     b.block_table[:nb])
        o[tp:] = sk.paged_decode(q[tp:], kc, vc, b.block_table[nb:], b.seq_lens[nb:])
        return o

    def _layers_quant(self, x: torch.Tensor, b: Batch, cache: PagedKVCache, decode: bool):
        delta = None  # the previous layer's down output, not yet added to x
        for layer, (w, lin) in enumerate(zip(self.w.layers, self.lin, strict=True)):
            kc, vc = cache.k[layer], cache.v[layer]
            if delta is None:
                h = sk.rmsnorm(x, w.attn_norm, EPS)
            else:
                h = sk.add_rmsnorm_(delta, x, w.attn_norm, EPS)  # x += delta, then the norm
            o = self._attention(lin["qkv"](h), x, b, kc, vc, decode)
            h2 = sk.add_rmsnorm_(lin["o"](o), x, w.mlp_norm, EPS)
            gu = lin["gate_up"](h2)
            delta = lin["down"](sk.swiglu(gu[:, :INTERMEDIATE], gu[:, INTERMEDIATE:]))
            if self.trace is not None:
                self.trace.append(x + delta)
        if delta is not None:
            x.add_(delta)
        return x

    def _logits(self, x: torch.Tensor, variant: int = -1) -> torch.Tensor:
        h = sk.rmsnorm(x, self.w.final_norm, EPS)
        return self.head(h) if self.head is not None else sk.hgemm(h, self.w.lm_head, variant)

    def warm_workspaces(self, max_m: int = GRAPH_MAX) -> None:
        """Every GEMM shape of the model once at each M from max_m down to 1. hgemm's M <= 64
        kernel keeps its split-K partials in one buffer per process that it frees and
        reallocates when a launch needs more, so a CUDA graph captured before the largest
        launch would keep the address of a freed buffer. After this the buffer has its final
        size for every M a graphed decode step (and any prefill of <= 64 tokens) runs."""
        w = self.w.layers[0]
        for m in range(max_m, 0, -1):
            x = torch.zeros(m, HIDDEN, device=w.w_qkv.device, dtype=torch.bfloat16)
            sk.hgemm(x, w.w_qkv)
            sk.hgemm(x.new_zeros(m, Q_WIDTH), w.w_o, residual=x, out=x.clone())
            sk.hgemm_swiglu(x, w.w_gate_up)
            sk.hgemm(x.new_zeros(m, INTERMEDIATE), w.w_down, residual=x, out=x.clone())
            sk.hgemm(x, self.w.lm_head)

    def prefill(self, b: Batch, cache: PagedKVCache, all_logits: bool = False) -> torch.Tensor:
        """Logits [B, V] of each prompt's last token, or [T, V] of every packed token with
        `all_logits`; the prompts' K/V land in the cache."""
        x = F.embedding(b.ids, self.w.embed)
        x = self._layers(x, b, cache, decode=False)
        return self._logits(x if all_logits else x[b.last], self.prefill_variant)

    def decode(self, b: Batch, cache: PagedKVCache) -> torch.Tensor:
        """Logits [B, V] of one token per sequence, appended to the cache first."""
        x = F.embedding(b.ids, self.w.embed)
        return self._logits(self._layers(x, b, cache, decode=True))


class TorchModel:
    """The same model in PyTorch on the same weights and the same paged cache layout. The
    attention is F.scaled_dot_product_attention: per prompt (causal, enable_gqa) at prefill,
    and at decode over each sequence's pages gathered (one copy) to the longest length in the
    batch with a length mask. At decode the 4 query heads of a K/V head are passed as 4 query
    rows of that head, so SDPA sees plain multi-head attention with a mask (with enable_gqa
    and the mask it was 1.7 to 2.5x slower on batched steps; docs/design/serving.md has both
    measured). `compile=True` puts
    the pre-attention part (norm, qkv GEMM, RoPE) and the post-attention part (o GEMM,
    residual, norm, MLP, residual) of a decode step's layer
    under torch.compile (max-autotune-no-cudagraphs, static shapes: one compilation per batch
    bucket); the attention and the cache writes stay eager, since the gathered length changes
    every step, and prefill stays eager, since its token count changes every batch."""

    def __init__(self, weights: ModelWeights, rope: RoPE, compile: bool = False):
        self.w = weights
        self.rope = rope
        self.trace: list[torch.Tensor] | None = None  # see SparkModel.trace
        self.pre_d, self.post_d = self._pre, self._post
        if compile:
            mode = "max-autotune-no-cudagraphs"
            self.pre_d = torch.compile(self._pre, dynamic=False, mode=mode)
            self.post_d = torch.compile(self._post, dynamic=False, mode=mode)

    @staticmethod
    def _pre(x, attn_norm, w_qkv, cos, sin):
        T = x.shape[0]
        h = F.rms_norm(x, (HIDDEN,), attn_norm, EPS)
        qkv = h @ w_qkv
        q = qkv[:, :Q_WIDTH].view(T, N_HEADS, HEAD_DIM)
        k = qkv[:, Q_WIDTH:Q_WIDTH + KV_WIDTH].view(T, N_KV_HEADS, HEAD_DIM)
        v = qkv[:, Q_WIDTH + KV_WIDTH:].view(T, N_KV_HEADS, HEAD_DIM)
        return _rope_tokens(q, cos, sin), _rope_tokens(k, cos, sin), v

    @staticmethod
    def _post(x, o, w_o, mlp_norm, w_gate_up, w_down):
        x = x + o @ w_o
        gu = F.rms_norm(x, (HIDDEN,), mlp_norm, EPS) @ w_gate_up
        return x + (F.silu(gu[:, 0::2]) * gu[:, 1::2]) @ w_down

    def _forward(self, x, b: Batch, cache: PagedKVCache, decode: bool):
        T = x.shape[0]
        pos = b.positions.long()
        cos, sin = self.rope.cos[pos], self.rope.sin[pos]
        page = cache.page
        slot = b.slots.long()
        pg, row = slot // page, slot % page
        if decode:
            B = b.seq_lens.shape[0]
            npg = cdiv(max(b.max_len, 1), page)
            lens = b.seq_lens.long()
            keep = torch.arange(npg * page, device=x.device)[None] < lens[:, None]
            keep[:, 0] |= lens == 0  # an empty slot attends to one key instead of none (NaN)
            mask = keep[:, None, None, :]
            table = b.block_table[:, :npg].long()
        pre, post = (self.pre_d, self.post_d) if decode else (self._pre, self._post)
        for layer, w in enumerate(self.w.layers):
            kc, vc = cache.k[layer], cache.v[layer]
            q, k, v = pre(x, w.attn_norm, w.w_qkv, cos, sin)
            kc[pg, :, row] = k
            vc[pg, :, row] = v
            if decode:
                # [H_kv, B, npg, page, D] in one gather, viewed as [B, H_kv, S, D] (strided)
                kk = kc.transpose(0, 1)[:, table].view(N_KV_HEADS, B, npg * page, HEAD_DIM)
                vv = vc.transpose(0, 1)[:, table].view(N_KV_HEADS, B, npg * page, HEAD_DIM)
                qg = q.view(B, N_KV_HEADS, N_HEADS // N_KV_HEADS, HEAD_DIM)
                o = F.scaled_dot_product_attention(qg, kk.transpose(0, 1), vv.transpose(0, 1),
                                                   attn_mask=mask)
                o = o.reshape(T, Q_WIDTH)
            else:
                if b.chunked:
                    raise NotImplementedError("TorchModel prefills whole prompts only")
                outs, t0 = [], 0
                for n in b.q_lens:
                    qs = q[t0:t0 + n].transpose(0, 1)[None]
                    ks = k[t0:t0 + n].transpose(0, 1)[None]
                    vs = v[t0:t0 + n].transpose(0, 1)[None]
                    o1 = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True,
                                                        enable_gqa=True)
                    outs.append(o1[0].transpose(0, 1).reshape(n, Q_WIDTH))
                    t0 += n
                o = torch.cat(outs)
            x = post(x, o, w.w_o, w.mlp_norm, w.w_gate_up, w.w_down)
            if self.trace is not None:
                self.trace.append(x)
        return x

    def _logits(self, x):
        return F.rms_norm(x, (HIDDEN,), self.w.final_norm, EPS) @ self.w.lm_head

    def prefill(self, b: Batch, cache: PagedKVCache, all_logits: bool = False) -> torch.Tensor:
        x = self._forward(F.embedding(b.ids, self.w.embed), b, cache, decode=False)
        return self._logits(x if all_logits else x[b.last])

    def decode(self, b: Batch, cache: PagedKVCache) -> torch.Tensor:
        x = F.embedding(b.ids, self.w.embed)
        return self._logits(self._forward(x, b, cache, decode=True))


@dataclass
class Sequence:
    sid: int
    prompt: torch.Tensor  # [P] int64 on the device; after a preemption, plus its tokens so far
    max_new: int
    pages: list[int] = field(default_factory=list)
    slot: int = -1
    length: int = 0  # tokens in the cache
    generated: int = 0
    stopped: int = 0  # with stop ids: how many tokens it had when one of them came out
    resumed_at: int = 0  # `generated` when it was last admitted: the tokens not in `prompt`


@dataclass
class Stats:
    prefill_tokens: int = 0
    prefill_batches: int = 0
    decode_steps: int = 0
    decode_tokens: int = 0  # tokens produced by decode steps (padding slots excluded)
    decode_rows: int = 0  # rows the decode steps computed, the bucket's padding included
    moved: int = 0  # sequences moved to a lower slot by _compact
    mixed_rows: int = 0  # decode tokens produced inside a prefill forward (Engine.mixed)
    stopped: int = 0  # sequences finished by a stop token before max_new
    generated: int = 0  # every generated token, the prefill's first one included
    finished: int = 0
    preempted: int = 0  # sequences taken out of the batch for lack of pages
    recomputed_tokens: int = 0  # prompt tokens prefilled again after a preemption


class Engine:
    """Continuous batching over a paged cache. `submit` queues a request; `step` admits what
    fits (free slots, free pages, the prefill token budget), prefills the admitted prompts as
    one packed batch, then runs one decode step over every running sequence; `run` steps
    until the queue and the batch are empty. With `mixed` (the default for SparkModel) the
    running sequences' decode rows ride in the prefill forward instead, so a step that admits
    is one forward, not two. With `graphs` (SparkModel only) each decode bucket up to
    GRAPH_MAX is captured into a CUDA graph at construction. With `compact` a decode step
    first moves sequences out of high slots so it runs the smallest bucket that fits. With
    `preempt` (the default) a request holds the pages of the tokens it has and takes one more
    every `page` steps; a decode step that finds no free page preempts the youngest running
    sequence, which is prefilled again later with its tokens so far. Without it the pages of
    the prompt and every requested token are reserved at admission and never run out."""

    def __init__(self, model, n_layers: int, max_batch: int = 64, max_seq: int = 4096,
                 num_pages: int | None = None, page: int = PAGE, cache_bytes: int | None = None,
                 prefill_tokens: int = 8192, graphs: bool = True, log_tokens: bool = False,
                 compact: bool = True, mixed: bool | None = None,
                 stop_ids: list[int] | None = None, preempt: bool = True, device="cuda"):
        if max_batch > BUCKETS[-1]:
            raise ValueError(f"max_batch is at most {BUCKETS[-1]}")
        # a decode step runs a whole bucket, so the per-slot buffers are bucket sized
        max_batch = next(x for x in BUCKETS if x >= max_batch)
        if num_pages is None:
            if cache_bytes is None:
                raise ValueError("give num_pages or cache_bytes")
            num_pages = cache_bytes // PagedKVCache.page_bytes(n_layers, page)
        rope_len = model.rope.cos.shape[0]
        if rope_len < max_seq:  # the rope kernel reads the table at any position it is given
            raise ValueError(f"the RoPE tables cover {rope_len} positions, max_seq is {max_seq}")
        self.model = model
        self.cache = PagedKVCache(n_layers, num_pages, page, device)
        self.page = page
        self.max_batch = max_batch
        self.max_seq = max_seq
        self.max_pages = cdiv(max_seq, page)
        self.prefill_tokens = prefill_tokens
        self.compact = compact
        self.preempt = preempt
        # pages kept free at admission so the running sequences can grow a little before
        # anything has to be preempted (vLLM's watermark, 1% of the cache)
        self.watermark = max(1, num_pages // 100) if preempt else 0
        # stop tokens: each forward's new tokens are copied to pinned host memory behind an
        # event, and read once the event has passed, so the loop never waits for the GPU
        self.stop_ids = set(stop_ids or ())
        self.pending: deque[tuple[torch.cuda.Event, list[Sequence], list[int], list[int],
                                  torch.Tensor]] = deque()
        self.stop_counts: dict[int, int] = {}  # sid -> tokens up to and with its stop token
        # decode rows inside a prefill forward need the model's prefill to take a chunk after
        # a cached context (attention_varlen does; TorchModel's per-prompt SDPA does not)
        self.mixed = getattr(model, "context_prefill", False) if mixed is None else mixed
        self.device = device
        i32 = dict(device=device, dtype=torch.int32)
        # per-slot decode inputs; slot i of a Bp-bucket step reads row i
        self.ids = torch.zeros(max_batch, device=device, dtype=torch.long)
        self.positions = torch.zeros(max_batch, **i32)
        self.slots = torch.full((max_batch,), self.cache.scratch * page, **i32)
        self.seq_lens = torch.zeros(max_batch, **i32)
        self.block_table = torch.zeros(max_batch, self.max_pages, **i32)
        # every token each slot has generated since its sequence was admitted, written by the
        # decode step itself (inside the graph) at column gen_n: what a preempted sequence is
        # prefilled again from, so a preemption never waits for the GPU
        self.gen = torch.zeros(max_batch, max_seq, device=device, dtype=torch.long)
        self.gen_n = torch.zeros(max_batch, device=device, dtype=torch.long)
        # pinned staging for the per-step inputs: a ring, so the host can queue a few steps
        # ahead of the GPU without rewriting a buffer whose copy has not run yet
        self.host = [torch.zeros(3, max_batch, dtype=torch.int32).pin_memory()
                     for _ in range(4)]
        self.host_ev: list[torch.cuda.Event | None] = [None] * len(self.host)
        self.host_i = 0
        self.running: list[Sequence | None] = [None] * max_batch
        self.waiting: deque[Sequence] = deque()
        self.stats = Stats()
        self._next_sid = 0
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        # (sequence ids, device tensor of their new tokens) per forward, when log_tokens
        self.log: list[tuple[list[int], torch.Tensor]] | None = [] if log_tokens else None
        if graphs:
            self._capture_all()

    # ---- decode inputs -------------------------------------------------------------------

    def _decode_batch(self, bp: int, max_len: int = 0) -> Batch:
        return Batch(ids=self.ids[:bp], positions=self.positions[:bp], slots=self.slots[:bp],
                     seq_lens=self.seq_lens[:bp], block_table=self.block_table[:bp],
                     max_len=max_len)

    def _decode_once(self, bp: int, max_len: int = 0) -> None:
        logits = self.model.decode(self._decode_batch(bp, max_len), self.cache)
        self.ids[:bp].copy_(torch.argmax(logits, dim=-1))
        self._record(self.ids[:bp], self.gen_n[:bp], slice(0, bp))

    def _record(self, toks: torch.Tensor, col: torch.Tensor, rows) -> None:
        """gen[rows, col] = toks, then col += 1 (saturating, so an empty slot's counter never
        leaves the buffer); device tensors only, so the graphs capture it."""
        self.gen[rows].scatter_(1, col[:, None], toks[:, None])
        col.add_(1).clamp_(max=self.max_seq - 1)

    def _capture_all(self) -> None:
        """One graph per bucket up to GRAPH_MAX, captured with every slot empty (length 0,
        writing the scratch page), all sharing one memory pool since they never run
        concurrently. The kernels' split-K workspaces are per-process buffers that grow (free
        and reallocate) on demand, and a graph keeps the address it was captured with. So the
        model's workspaces are grown to their final size first, and only buckets whose GEMMs
        run the M <= 64 kernel are captured: from M = 128 hgemm runs the Stream-K schedule,
        whose workspace a later prefill can grow, which freed the buffer a captured 128 or 256
        bucket still pointed at (an illegal address on replay). Those buckets run eagerly; the
        graph saves under 1% of a step (docs/design/serving.md)."""
        warm = getattr(self.model, "warm_workspaces", None)
        if warm is not None:
            warm()
        pool = torch.cuda.graph_pool_handle()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        for bp in (x for x in BUCKETS if x <= min(self.max_batch, GRAPH_MAX)):
            with torch.cuda.stream(side):
                for _ in range(2):  # grows every workspace before the capture
                    self._decode_once(bp)
            torch.cuda.current_stream().wait_stream(side)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, pool=pool):
                self._decode_once(bp)
            self.graphs[bp] = g
        torch.cuda.synchronize()
        self.ids.zero_()
        self.gen_n.zero_()

    # ---- requests ------------------------------------------------------------------------

    def submit(self, prompt: torch.Tensor, max_new: int) -> Sequence:
        if prompt.numel() + max_new > self.max_seq:
            raise ValueError(f"prompt + max_new = {prompt.numel() + max_new} > max_seq")
        if max_new < 1:
            raise ValueError("max_new must be >= 1")
        need = cdiv(prompt.numel() + max_new, self.page)
        if need > self.cache.num_pages - 1 - self.watermark:  # it would wait (or thrash) forever
            raise ValueError(f"prompt + max_new needs {need} pages, the cache has "
                             f"{self.cache.num_pages - 1 - self.watermark} to give")
        seq = Sequence(self._next_sid, prompt.to(self.device, torch.long).flatten(), max_new)
        self._next_sid += 1
        self.waiting.append(seq)
        return seq

    def _admit(self, budget: int | None = None) -> list[Sequence]:
        """Admits queued requests into free slots while pages last: the pages of the prompt
        plus one token with `preempt` (the rest come as the tokens do), of the prompt plus
        every requested token without it."""
        free_slots = [i for i, s in enumerate(self.running) if s is None]
        out = []
        budget = self.prefill_tokens if budget is None else budget
        while self.waiting and free_slots:
            seq = self.waiting[0]
            n = seq.prompt.numel()
            remaining = seq.max_new - seq.generated
            need = cdiv(n + (1 if self.preempt else remaining), self.page)
            if (out and n > budget) or need > len(self.cache.free) - self.watermark:
                break
            self.waiting.popleft()
            seq.pages = self.cache.alloc(need)
            seq.slot = free_slots.pop(0)
            self.running[seq.slot] = seq
            row = torch.zeros(self.max_pages, dtype=torch.int32)
            row[:need] = torch.tensor(seq.pages, dtype=torch.int32)
            self.block_table[seq.slot].copy_(row, non_blocking=True)
            out.append(seq)
            budget -= n
        if out:
            self.gen_n[torch.tensor([s.slot for s in out], device=self.device)] = 0
        return out

    def _preempt(self, seq: Sequence) -> None:
        """Takes `seq` out of the batch: its pages go back, its prompt grows by the tokens it
        generated since it was admitted (the slot's gen buffer, read on the device), and it
        goes to the head of the queue to be prefilled again from the start."""
        n = seq.generated - seq.resumed_at
        if n:
            seq.prompt = torch.cat([seq.prompt, self.gen[seq.slot, :n]])
            seq.resumed_at = seq.generated
        self.cache.release(seq.pages)
        seq.pages = []
        self.running[seq.slot] = None
        seq.slot = -1
        self.waiting.appendleft(seq)
        self.stats.preempted += 1
        self.stats.recomputed_tokens += seq.prompt.numel()

    def _grow(self, seqs: list[Sequence]) -> list[Sequence]:
        """Gives every sequence in `seqs` the page its next token (position `length`) needs,
        oldest first; when none is free the youngest running sequence is preempted until one
        is. Returns the sequences of `seqs` still running. New pages land in the block table
        with one indexed copy."""
        if not self.preempt:
            return seqs
        slots, cols, pages = [], [], []
        for seq in sorted(seqs, key=lambda s: s.sid):
            if seq.slot < 0:
                continue  # preempted while an older one grew
            idx = seq.length // self.page
            if idx < len(seq.pages):
                continue
            while not self.cache.free:
                victim = max((s for s in self.running if s is not None), key=lambda s: s.sid)
                self._preempt(victim)
                if victim is seq:
                    break
            if seq.slot < 0:
                continue
            seq.pages.extend(self.cache.alloc(1))
            slots.append(seq.slot)
            cols.append(idx)
            pages.append(seq.pages[-1])
        if slots:
            t = torch.tensor([slots, cols, pages], dtype=torch.int32).pin_memory().to(
                self.device, non_blocking=True)
            self.block_table[t[0].long(), t[1].long()] = t[2]
        return [s for s in seqs if s.slot >= 0]

    def _prefill_batch(self, seqs: list, dec: list[Sequence] = ()) -> Batch:
        """The packed prefill inputs of admitted sequences: their prompts back to back, each
        token at its position with its cache slot from its sequence's pages. An item of
        `seqs` is a Sequence (its whole prompt) or a piece (sequence, start, end): tokens
        start..end-1 of its prompt, after the start keys an earlier forward cached. Each
        running sequence in `dec` adds one row after them: its next token (already on the
        device in its slot's id) at its current length, a one-token chunk after its cached
        context."""
        page = self.page
        parts = [x if isinstance(x, tuple) else (x, 0, x.prompt.numel()) for x in seqs]
        q_lens = [e - a for _, a, e in parts] + [1] * len(dec)
        kv_lens = [e for _, _, e in parts] + [s.length + 1 for s in dec]
        ids = [s.prompt[a:e] for s, a, e in parts]
        if dec:
            ids.append(self.ids[torch.tensor([s.slot for s in dec]).to(self.device)])
        pos, slots = [], []
        for s, a, e in parts:
            p = torch.arange(a, e, dtype=torch.int32)
            pos.append(p)
            slots.append(torch.tensor(s.pages, dtype=torch.int32)[p // page] * page + p % page)
        if dec:
            p = [s.length for s in dec]
            pos.append(torch.tensor(p, dtype=torch.int32))
            slots.append(torch.tensor([s.pages[x // page] * page + x % page
                                       for s, x in zip(dec, p, strict=True)], dtype=torch.int32))
        cu = torch.tensor([0] + q_lens, dtype=torch.int32).cumsum(0, dtype=torch.int32)
        dev = dict(device=self.device, non_blocking=True)
        rows = [s for s, _, _ in parts] + list(dec)
        b = Batch(ids=torch.cat(ids), positions=torch.cat(pos).to(**dev),
                  slots=torch.cat(slots).to(**dev),
                  seq_lens=torch.tensor(kv_lens, dtype=torch.int32).to(**dev),
                  block_table=self.block_table[torch.tensor([s.slot for s in rows])],
                  cu_seqlens=cu.to(**dev), last=(cu[1:] - 1).long().to(**dev), q_lens=q_lens,
                  max_len=max(kv_lens), n_dec=len(dec),
                  chunked=any(a > 0 for _, a, _ in parts))
        return b

    def _pieces(self, seqs: list[Sequence]):
        """The prefill of `seqs` as forwards of at most prefill_tokens prompt tokens: yields
        the (sequence, start, end) pieces of each forward. Prompts are packed in order and a
        prompt longer than what is left of a forward's budget is split, so a 100K-token prompt
        runs as chunks that each attend to the cache the previous ones filled."""
        done = {s.sid: 0 for s in seqs}
        todo = list(seqs)
        while todo:
            parts, budget = [], self.prefill_tokens
            for s in todo:
                if budget == 0:
                    break
                a = done[s.sid]
                e = min(s.prompt.numel(), a + budget)
                parts.append((s, a, e))
                budget -= e - a
                done[s.sid] = e
            todo = [s for s in todo if done[s.sid] < s.prompt.numel()]
            yield parts

    def _prefill(self, seqs: list[Sequence], dec: list[Sequence] = ()) -> None:
        """The admitted prompts in forwards of at most prefill_tokens tokens (`_pieces`) and,
        with `dec`, one decode row per running sequence in the first of them; the next token
        of each prompt that finished and of each decode row lands in its slot's id."""
        for k, parts in enumerate(self._pieces(seqs)):
            rows_dec = list(dec) if k == 0 else []
            logits = self.model.prefill(self._prefill_batch(parts, rows_dec), self.cache)
            self.stats.prefill_batches += 1
            fin = [i for i, (s, _, e) in enumerate(parts) if e == s.prompt.numel()]
            rows = fin + list(range(len(parts), len(parts) + len(rows_dec)))
            if not rows:
                continue
            both = [parts[i][0] for i in fin] + rows_dec
            new = torch.argmax(logits[torch.tensor(rows, device=self.device)], dim=-1)
            slots = torch.tensor([s.slot for s in both], device=self.device)
            self.ids[slots] = new
            self.gen[slots, self.gen_n[slots]] = new
            self.gen_n[slots] += 1
            if self.log is not None:
                self.log.append(([s.sid for s in both], new))
            for s in both[:len(fin)]:
                s.length = s.prompt.numel()
                s.generated += 1  # from 0, or from where a preemption left it
            for s in rows_dec:
                s.length += 1
                s.generated += 1
            self._watch(both, list(range(len(both))), new)
            self.stats.decode_tokens += len(rows_dec)
            self.stats.mixed_rows += len(rows_dec)
            self.stats.generated += len(both)
        self.stats.prefill_tokens += sum(s.prompt.numel() for s in seqs)

    def _watch(self, seqs: list[Sequence], rows: list[int], toks: torch.Tensor) -> None:
        """Queues the copy of a forward's new tokens to the host for the stop check:
        toks[rows[j]] is the token seqs[j] just generated (call after their counts moved)."""
        if not self.stop_ids or not seqs:
            return
        host = torch.empty(toks.shape, dtype=toks.dtype, pin_memory=True)
        host.copy_(toks, non_blocking=True)
        ev = torch.cuda.Event()
        ev.record()
        self.pending.append((ev, seqs, rows, [s.generated for s in seqs], host))

    def _check_stops(self, wait: bool = False) -> None:
        """Marks the sequences whose new token was a stop id, for every forward the GPU has
        finished (every queued one with `wait`). The check trails the GPU by the steps the
        host runs ahead, so a stopped sequence may have generated a few more tokens; it
        retires at the next step and outputs() cuts them off."""
        while self.pending and (wait or self.pending[0][0].query()):
            ev, seqs, rows, counts, host = self.pending.popleft()
            ev.synchronize()
            toks = host.tolist()
            for s, r, n in zip(seqs, rows, counts, strict=True):
                if not s.stopped and toks[r] in self.stop_ids:
                    s.stopped = n
                    self.stop_counts[s.sid] = n

    def _retire(self) -> None:
        if self.stop_ids:
            self._check_stops()
        for s in list(self.waiting):  # a preempted sequence whose stop came out before
            if s.stopped and s.resumed_at:
                self.waiting.remove(s)
                self.stats.finished += 1
                if s.stopped < s.max_new:
                    self.stats.stopped += 1
        for i, s in enumerate(self.running):
            if s is not None and (s.generated >= s.max_new or s.stopped):
                if s.stopped and s.stopped < s.max_new:
                    self.stats.stopped += 1
                self.cache.release(s.pages)
                self.running[i] = None
                self.stats.finished += 1

    def _compact(self) -> None:
        """Moves the sequences in the highest slots into the lowest free ones when the decode
        step would otherwise run a larger bucket than the number of live sequences needs.
        Sequences retire in any order, so without this a few long requests left in high
        slots keep every later step at the bucket of the highest one (a 64-row step for six
        sequences). A move is two device copies, the slot's next token id and its block table
        row; the other per-slot inputs are rewritten every step."""
        live = [i for i, s in enumerate(self.running) if s is not None]
        if not live:
            return
        want = next(x for x in BUCKETS if x >= len(live))
        if live[-1] < want:
            return
        src = [i for i in live if i >= want]
        dst = [i for i in range(want) if self.running[i] is None][:len(src)]
        idx = torch.tensor(src + dst, dtype=torch.long).pin_memory().to(self.device,
                                                                         non_blocking=True)
        s_idx, d_idx = idx[:len(src)], idx[len(src):]
        self.ids[d_idx] = self.ids[s_idx]
        self.block_table[d_idx] = self.block_table[s_idx]
        self.gen[d_idx] = self.gen[s_idx]
        self.gen_n[d_idx] = self.gen_n[s_idx]
        for a, b in zip(src, dst, strict=True):
            seq = self.running[a]
            seq.slot = b
            self.running[b], self.running[a] = seq, None
        self.stats.moved += len(src)

    def _decode(self) -> None:
        if self.compact:
            self._compact()
        self._grow([s for s in self.running if s is not None])
        live = [i for i, s in enumerate(self.running) if s is not None]
        if not live:
            return
        bp = next(x for x in BUCKETS if x > live[-1])
        pos, slot, lens = [0] * bp, [self.cache.scratch * self.page] * bp, [0] * bp
        for i in live:
            s = self.running[i]
            p = s.length  # the position of the token this step appends
            pos[i] = p
            slot[i] = s.pages[p // self.page] * self.page + p % self.page
            lens[i] = p + 1
        k = self.host_i % len(self.host)
        self.host_i += 1
        if self.host_ev[k] is not None:
            self.host_ev[k].synchronize()
        h = self.host[k]
        h[:, :bp] = torch.tensor([pos, slot, lens], dtype=torch.int32)
        self.positions[:bp].copy_(h[0, :bp], non_blocking=True)
        self.slots[:bp].copy_(h[1, :bp], non_blocking=True)
        self.seq_lens[:bp].copy_(h[2, :bp], non_blocking=True)
        self.host_ev[k] = torch.cuda.Event()
        self.host_ev[k].record()
        if bp in self.graphs:
            self.graphs[bp].replay()
        else:
            self._decode_once(bp, max(lens))
        if self.log is not None:
            self.log.append(([self.running[i].sid for i in live],
                             self.ids[torch.tensor(live, device=self.device)]))
        for i in live:
            s = self.running[i]
            s.length += 1
            s.generated += 1
        self._watch([self.running[i] for i in live], live, self.ids[:bp])
        self.stats.decode_steps += 1
        self.stats.decode_tokens += len(live)
        self.stats.decode_rows += bp
        self.stats.generated += len(live)

    def prefill_waiting(self) -> None:
        """Admits and prefills queued requests until the queue is empty or nothing more fits,
        with no decode step in between (how a benchmark separates the two phases)."""
        while self.waiting:
            admitted = self._admit()
            if not admitted:
                break
            self._prefill(admitted)
        self._retire()

    def step(self) -> None:
        """Retire, admit, then one forward: with `mixed`, a step that admits prompts runs
        the running sequences' decode rows inside the prefill forward and has no decode step;
        otherwise the prefill is followed by a decode step over every running sequence."""
        self._retire()
        # the running sequences take the pages their next token needs before anything is
        # admitted, so a preemption never hits a prompt that has not been prefilled yet
        running = self._grow([s for s in self.running if s is not None])
        admitted = self._admit()
        if admitted and self.mixed:
            self._prefill(admitted, running)
            self._retire()
            return
        if admitted:
            self._prefill(admitted)
            self._retire()  # max_new == 1 finishes at the prefill
        self._decode()

    def prompt_logits_chunks(self, prompts: list[torch.Tensor]):
        """Logits at every position of each prompt, prefilled on empty slots (the engine must
        be idle) in forwards of at most prefill_tokens tokens: yields (i, start, logits
        [n, V]) for rows start..start+n-1 of prompt i, in order. A long prompt's logits never
        exist all at once, which is what a perplexity at 32K context needs (32K rows of a
        128K vocabulary are 8 GB). The pages are released at the end."""
        if any(s is not None for s in self.running) or self.waiting:
            raise RuntimeError("prompt_logits needs an idle engine")
        seqs = [self.submit(p, 1) for p in prompts]
        admitted = self._admit(budget=sum(p.numel() for p in prompts))
        if len(admitted) != len(seqs):
            self.waiting.clear()
            for s in admitted:
                self.cache.release(s.pages)
                self.running[s.slot] = None
            raise RuntimeError("prompts do not fit the slots or the cache in one prefill")
        index = {s.sid: i for i, s in enumerate(seqs)}
        try:
            for parts in self._pieces(seqs):
                logits = self.model.prefill(self._prefill_batch(parts), self.cache,
                                            all_logits=True)
                for (s, a, _), lg in zip(parts, logits.split([e - a for _, a, e in parts]),
                                         strict=True):
                    yield index[s.sid], a, lg
        finally:
            for s in seqs:
                self.cache.release(s.pages)
                self.running[s.slot] = None

    def prompt_logits(self, prompts: list[torch.Tensor]) -> list[torch.Tensor]:
        """Logits [P, V] at every position of each prompt (`prompt_logits_chunks` joined):
        what a perplexity or a parity check against another implementation reads."""
        rows: list[list[torch.Tensor]] = [[] for _ in prompts]
        for i, _, lg in self.prompt_logits_chunks(prompts):
            rows[i].append(lg)
        return [torch.cat(r) for r in rows]

    def outputs(self) -> dict[int, list[int]]:
        """Generated tokens per sequence id, in order (needs log_tokens=True); a sequence
        that stopped ends with its stop token."""
        self._check_stops(wait=True)
        out: dict[int, list[int]] = {}
        for sids, toks in self.log or []:
            for sid, t in zip(sids, toks.tolist(), strict=True):
                out.setdefault(sid, []).append(t)
        for sid, n in self.stop_counts.items():
            if sid in out:
                del out[sid][n:]
        return out

    def run(self) -> Stats:
        while self.waiting or any(s is not None for s in self.running):
            self.step()
        self._retire()
        torch.cuda.synchronize()
        return self.stats
