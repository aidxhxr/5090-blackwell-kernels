"""A whole MoE layer on the kernels, bf16: the router, softmax top-k routing with
renormalization, the rows into expert order, the gate/up grouped GEMM with the SwiGLU in its
epilogue, the down grouped GEMM, and the weighted combine back into token order. Mixtral 8x7B
(8 experts, top-2, 4096 -> 14336) and Qwen3-30B-A3B (128 experts, top-8, 2048 -> 768) are the
two configurations the bench runs (docs/design/moe_layer.md). Inkling 2 Mini (128 experts,
top-8, 2304 -> 1152) is the third: a sigmoid router in fp32 with a correction bias for the
picks and route_scale 8 on the renormalized weights, and the softcapped SwiGLU,
softcap(g, 4) * sigmoid(g) * softcap(u, 25), in the same epilogue.

`MoELayer.forward(x)` is six launches, none of which waits on the host:

    logits = x @ router.T                  cuBLAS through F.linear, bf16, as the models do
    moe_dispatch(x, logits, k)             the routing (one launch: top-k, weights, counts,
                                           offsets, each assignment's rank) and the permute
                                           (one launch: the rows into expert order)
    grouped_gemm_swiglu(xs, w13, offs)     gate and up as one [E, 2I, D] weight with the rows
                                           interleaved, silu(g) * u formed in the epilogue
    grouped_gemm(h, w2, offs)              down
    moe_combine(y, pos, weights)           each token's k rows times their weights, fp32, in
                                           slot order

or, where it measured faster (`gather`, the rule below), the permute is replaced by the
routing's inverse permutation (`moe_route_rows`, a small kernel in the permute's place) and
the gate/up GEMM gathers x's rows through it in its loads, so the permuted copy is never
written. Both paths run a fixed grid whose schedule is built on the device from the offsets,
so the whole forward can be captured in a CUDA graph: `MoEGraph` captures it for one token
count and replays it (docs/design/moe_layer.md).

Two torch layers sit next to it, for the tests and the bench, both on the same `MoEWeights`:

    torch_moe          transformers' eager MoE block (the Mixtral / Qwen3-MoE experts loop):
                       softmax, top-k, renormalize, then for every expert that got a token a
                       torch.where for its rows, gate|up, silu * up, down, times the routing
                       weight, index_add_ into the output
    torch_grouped_moe  the same routing, the rows sorted by expert with argsort, the two
                       projections as torch._grouped_mm, and the combine as a gather of each
                       token's k rows and a weighted sum

`MoEWeights` keeps the expert weights the way checkpoints and vLLM do: w13 [E, 2I, D] with
gate rows 0..I-1 and up rows I..2I-1, w2 [E, D, I], router [E, D]; `MoELayer` takes an
interleaved copy of w13 for its epilogue. For a sigmoid config (Inkling) the router is fp32
and run on x.float(), as transformers' Inkling2LoopTopKRouter does, and `bias` is the
e_score_correction_bias [E] fp32; the torch layers then follow Inkling2LoopExperts (the
softcapped activation, the sum in fp32).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class MoELayerConfig:
    """One MoE layer's shape: `experts` SwiGLU experts hidden -> inter -> hidden, each token
    routed to `top_k` of them by a softmax over the router logits, the k weights renormalized
    to sum 1 when `renormalize`. scoring="sigmoid" is the sigmoid router with a pick bias,
    the weights times `route_scale`; `softcap` (cap_gate, cap_up) the softcapped SwiGLU."""

    name: str
    hidden: int
    inter: int
    experts: int
    top_k: int
    renormalize: bool = True
    scoring: str = "softmax"
    route_scale: float = 1.0
    softcap: tuple[float, float] | None = None

    def expert_bytes(self) -> int:
        """bf16 bytes of one expert (gate, up, down)."""
        return 3 * self.hidden * self.inter * 2

    def layer_bytes(self) -> int:
        """bf16 bytes of the layer's experts."""
        return self.experts * self.expert_bytes()


MIXTRAL_8X7B = MoELayerConfig("mixtral", hidden=4096, inter=14336, experts=8, top_k=2)
QWEN3_30B_A3B = MoELayerConfig("qwen3", hidden=2048, inter=768, experts=128, top_k=8)
INKLING_2_MINI = MoELayerConfig("inkling", hidden=2304, inter=1152, experts=128, top_k=8,
                                scoring="sigmoid", route_scale=8.0, softcap=(4.0, 25.0))
CONFIGS = {c.name: c for c in (MIXTRAL_8X7B, QWEN3_30B_A3B, INKLING_2_MINI)}


@dataclass
class MoEWeights:
    """router [E, D], w13 [E, 2I, D] (gate rows then up rows), w2 [E, D, I], all bf16; for a
    sigmoid config the router is fp32 and bias [E] fp32 the correction bias."""

    cfg: MoELayerConfig
    router: torch.Tensor
    w13: torch.Tensor
    w2: torch.Tensor
    bias: torch.Tensor | None = None

    @staticmethod
    def random(cfg: MoELayerConfig, device="cuda", seed: int = 0, std: float = 0.02
               ) -> MoEWeights:
        """Normal weights, std 0.02 for the experts. The router is std 1 / sqrt(D), so the
        logits of unit-variance tokens are about N(0, 1) and the routing spreads over the
        experts the way a trained router's roughly does. A sigmoid config gets an fp32 router
        and a bias of std 0.05, about the gap between neighbouring top scores."""
        g = torch.Generator(device=device).manual_seed(seed)
        E, D, inter = cfg.experts, cfg.hidden, cfg.inter

        def rand(*shape, s):
            out = torch.empty(*shape, device=device, dtype=torch.bfloat16)
            for e in range(shape[0]):  # one expert at a time: no fp32 copy of the whole layer
                out[e] = (torch.randn(*shape[1:], device=device, generator=g) * s)
            return out

        router = rand(E, D, s=D ** -0.5)
        w13 = rand(E, 2 * inter, D, s=std)
        w2 = rand(E, D, inter, s=std)
        if cfg.scoring == "sigmoid":
            bias = torch.randn(E, device=device, generator=g) * 0.05
            return MoEWeights(cfg, router.float(), w13, w2, bias)
        return MoEWeights(cfg, router, w13, w2)


def router_logits(x: torch.Tensor, wts: MoEWeights) -> torch.Tensor:
    """The router GEMM: bf16 for a softmax config, fp32 on x.float() for a sigmoid one."""
    if wts.cfg.scoring == "sigmoid":
        return F.linear(x.float(), wts.router)
    return F.linear(x, wts.router)


def softcap_swiglu_torch(gate: torch.Tensor, up: torch.Tensor, caps: tuple[float, float]
                         ) -> torch.Tensor:
    """transformers' Inkling2LoopSoftcappedSwiGLU: c * tanh(x / c) on gate and up, the
    sigmoid on the raw gate, fp32 math, back to the input dtype."""
    dtype = gate.dtype
    gate, up = gate.float(), up.float()
    cg, cu = caps
    return (torch.tanh(gate / cg) * cg * torch.sigmoid(gate) * (torch.tanh(up / cu) * cu)).to(dtype)


def _act_torch(gate: torch.Tensor, up: torch.Tensor, cfg: MoELayerConfig) -> torch.Tensor:
    if cfg.softcap is not None:
        return softcap_swiglu_torch(gate, up, cfg.softcap)
    return F.silu(gate) * up


def _route_torch(logits: torch.Tensor, cfg: MoELayerConfig, bias: torch.Tensor | None = None
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """transformers' router: softmax in fp32, top-k, renormalized -> (weights fp32, ids).
    For a sigmoid config Inkling2LoopTopKRouter's: sigmoid scores, the picks by score + bias
    (unsorted, as there), the picks' scores over (their sum + 1e-20), times route_scale."""
    if cfg.scoring == "sigmoid":
        scores = torch.sigmoid(logits.float())
        key = scores + bias if bias is not None else scores
        ids = torch.topk(key, cfg.top_k, dim=-1, sorted=False).indices
        w = scores.gather(1, ids)
        if cfg.renormalize:
            w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
        return w * cfg.route_scale, ids
    p = torch.softmax(logits, dim=-1, dtype=torch.float32)
    w, ids = torch.topk(p, cfg.top_k, dim=-1)
    if cfg.renormalize:
        w = w / w.sum(dim=-1, keepdim=True)
    return w, ids


def torch_moe(x: torch.Tensor, wts: MoEWeights, acc_dtype: torch.dtype | None = None
              ) -> torch.Tensor:
    """transformers' eager MoE experts loop over [T, D] bf16. Like the models, the routing
    weights are cast to bf16 and the output accumulates in x's dtype; acc_dtype=float32 keeps
    both in fp32 (the reference the tests compare against). A sigmoid config is
    Inkling2LoopSparseMoeBlock, whose loop keeps both in fp32 already."""
    cfg = wts.cfg
    inter = cfg.inter
    acc = acc_dtype or (torch.float32 if cfg.scoring == "sigmoid" else x.dtype)
    logits = router_logits(x, wts)
    w, ids = _route_torch(logits, cfg, wts.bias)
    w = w.to(acc)
    out = torch.zeros(x.shape[0], cfg.hidden, dtype=acc, device=x.device)
    mask = F.one_hot(ids, num_classes=cfg.experts).permute(2, 1, 0)  # [E, k, T]
    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for e in hit.flatten().tolist():
        slot, tok = torch.where(mask[e])
        gate, up = F.linear(x[tok], wts.w13[e]).split(inter, dim=-1)
        h = _act_torch(gate, up, cfg)
        y = F.linear(h, wts.w2[e]).to(acc) * w[tok, slot, None]
        out.index_add_(0, tok, y)
    return out.to(x.dtype)


def torch_grouped_moe(x: torch.Tensor, wts: MoEWeights) -> torch.Tensor:
    """The routing in torch, the rows sorted by expert, both projections as
    torch._grouped_mm (one launch each, offsets on the device), silu * up, and the combine as
    a gather of each token's k rows times their weights, summed in fp32."""
    cfg = wts.cfg
    T, k, inter = x.shape[0], cfg.top_k, cfg.inter
    logits = router_logits(x, wts)
    w, ids = _route_torch(logits, cfg, wts.bias)
    flat = ids.reshape(-1)
    order = torch.argsort(flat, stable=True)
    offs = torch.cumsum(torch.bincount(flat, minlength=cfg.experts), 0).to(torch.int32)
    xs = x.index_select(0, order // k)
    gu = torch._grouped_mm(xs, wts.w13.transpose(1, 2), offs=offs)
    h = _act_torch(gu[:, :inter], gu[:, inter:], cfg)
    y = torch._grouped_mm(h, wts.w2.transpose(1, 2), offs=offs)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=x.device)
    rows = y.index_select(0, inv).view(T, k, cfg.hidden).float()
    return (rows * w.unsqueeze(-1)).sum(1).to(x.dtype)


# When the gate/up GEMM gathers x through the row table instead of reading a permuted copy.
# The gathered GEMM is 1.5 to 4% slower than the TMA one on the 64- and 128-row tiles, and
# what it saves is the permute, so it wins where the permute is a large share of the GEMM:
# experts of up to 2048 inter columns (2I <= 4096) at every token count, and any width while
# the GEMM runs its 16-row tile, up to 24 rows per expert on average (grouped_gemm.cu,
# SMALL_ROWS; measured on the 5090, docs/design/moe_layer.md).
GATHER_MAX_INTER = 2048
GATHER_SMALL_ROWS = 24


class MoELayer:
    """The layer on the kernels. `fuse_swiglu=False` runs the gate/up GEMM plain on the
    concatenated w13 and the swiglu kernel after it (the ablation of the fused epilogue).
    `gather` picks the gathered gate/up loads over the permuted copy: None applies the rule
    above (GATHER_MAX_INTER, GATHER_SMALL_ROWS), True or False forces one. `softcap` is the
    config's (cap_gate, cap_up), which the epilogue applies; setting it to None runs the
    plain SwiGLU on the same weights (the bench's measure of what the caps cost)."""

    def __init__(self, wts: MoEWeights, fuse_swiglu: bool = True, gather: bool | None = None):
        from spark_kernels import ops as sk

        cfg = wts.cfg
        self.cfg = cfg
        self.router = wts.router
        self.wts = wts
        self.softcap = cfg.softcap
        self.w2 = wts.w2
        self.fuse_swiglu = fuse_swiglu
        self.gather = gather
        if fuse_swiglu:
            inter = cfg.inter
            self.w13 = sk.interleave_experts(wts.w13[:, :inter], wts.w13[:, inter:])
        else:
            self.w13 = wts.w13

    def routing(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                 torch.Tensor]:
        """(ids, weights, offsets, pos) of x's tokens from the routing kernel."""
        from spark_kernels import ops as sk

        logits = router_logits(x, self.wts)
        return sk.moe_route(logits, self.cfg.top_k, self.cfg.renormalize, **self.route_kw())

    def route_kw(self) -> dict:
        """The routing kernel's scoring arguments for the config."""
        cfg = self.cfg
        if cfg.scoring == "sigmoid":
            return dict(scoring="sigmoid", bias=self.wts.bias, scale=cfg.route_scale)
        return {}

    def gathers(self, T: int) -> bool:
        """Whether a forward of T tokens gathers x in the gate/up GEMM's loads."""
        if self.gather is not None:
            return self.gather
        cfg = self.cfg
        rows_per_expert = -(-T * cfg.top_k // cfg.experts)
        return cfg.inter <= GATHER_MAX_INTER or rows_per_expert <= GATHER_SMALL_ROWS

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[T, D] bf16 -> [T, D] bf16, the layer's output (not added to the residual). No
        host sync, no shape that depends on the routing: capturable in a CUDA graph."""
        from spark_kernels import ops as sk

        cfg = self.cfg
        logits = router_logits(x, self.wts)
        kw = self.route_kw()
        if self.gathers(x.shape[0]):
            _, w, offsets, pos, rows = sk.moe_route_rows(logits, cfg.top_k, cfg.renormalize,
                                                         **kw)
            xs = x
        else:
            xs, _, w, offsets, pos = sk.moe_dispatch(x, logits, cfg.top_k, cfg.renormalize,
                                                     **kw)
            rows = None
        if self.fuse_swiglu:
            h = sk.grouped_gemm_swiglu(xs, self.w13, offsets, rows=rows, softcap=self.softcap)
        else:
            gu = sk.grouped_gemm(xs, self.w13, offsets, rows=rows)
            h = sk.swiglu(gu[:, :cfg.inter], gu[:, cfg.inter:], softcap=self.softcap)
        y = sk.grouped_gemm(h, self.w2, offsets)
        return sk.moe_combine(y, pos, w)

    __call__ = forward


class MoEGraph:
    """`layer.forward` captured in a CUDA graph for T tokens: `__call__(x)` copies x into the
    static input, replays, and returns the static output (overwritten by the next call).
    The layer is run twice on a side stream first, so every per-process workspace (the
    routing's, the grouped GEMM's Stream-K slots) exists before the capture, the way the
    engine warms its buckets (engine.py, _capture_all)."""

    def __init__(self, layer: MoELayer, T: int, pool=None):
        x = torch.zeros(T, layer.cfg.hidden, device=layer.router.device, dtype=torch.bfloat16)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                layer(x)
        torch.cuda.current_stream().wait_stream(side)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, pool=pool):
            self.out = layer(x)
        self.x = x
        self.T = T

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        self.x.copy_(x)
        self.graph.replay()
        return self.out
