"""The MoE layer pieces and the layer: sk.moe_route against torch's softmax top-k (the picked
logits, the weights, the offsets and the stable expert order exactly), moe_dispatch's rows and
moe_combine's weighted sum, grouped_gemm_swiglu against the loop over experts, and MoELayer
against the transformers-style torch layer, on scaled-down shapes and on Qwen3-30B-A3B's.

On the CPU only the torch layers run (moe_layer.py loaded from its file, the package's
__init__ needing the extension): the experts loop against a dense computation, and the two
model configurations' sizes."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def _load_moe_layer():
    try:
        from spark_kernels import moe_layer
        return moe_layer
    except ImportError:
        path = Path(__file__).resolve().parent.parent / "python" / "spark_kernels" / "moe_layer.py"
        spec = importlib.util.spec_from_file_location("spark_kernels_moe_layer", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["spark_kernels_moe_layer"] = mod
        spec.loader.exec_module(mod)
        return mod


ml = _load_moe_layer()
SMALL_MIXTRAL = ml.MoELayerConfig("small_mixtral", hidden=512, inter=1024, experts=8, top_k=2)
SMALL_QWEN = ml.MoELayerConfig("small_qwen", hidden=256, inter=128, experts=40, top_k=4)


# ---- CPU ---------------------------------------------------------------------------------
def _dense_reference(x, wts):
    """Every token through its k experts one at a time, fp32."""
    cfg = wts.cfg
    p = torch.softmax(F.linear(x, wts.router).float(), -1)
    w, ids = p.topk(cfg.top_k, -1)
    w = w / w.sum(-1, keepdim=True)
    out = torch.zeros(x.shape[0], cfg.hidden)
    for t in range(x.shape[0]):
        for s in range(cfg.top_k):
            e = ids[t, s]
            gu = x[t].float() @ wts.w13[e].float().t()
            h = F.silu(gu[:cfg.inter]) * gu[cfg.inter:]
            out[t] += w[t, s] * (h @ wts.w2[e].float().t())
    return out


def test_torch_moe_matches_dense_on_cpu():
    cfg = ml.MoELayerConfig("tiny", hidden=64, inter=32, experts=6, top_k=2)
    wts = ml.MoEWeights.random(cfg, device="cpu", seed=3, std=0.1)
    x = torch.randn(9, 64, generator=torch.Generator().manual_seed(4)).to(torch.bfloat16)
    got = ml.torch_moe(x, wts, acc_dtype=torch.float32).float()
    want = _dense_reference(x, wts)
    assert (got - want).abs().max() <= 2e-2 * want.abs().max()


def test_config_sizes():
    assert ml.MIXTRAL_8X7B.layer_bytes() == 8 * 3 * 4096 * 14336 * 2  # 2.8 GB
    assert ml.QWEN3_30B_A3B.layer_bytes() == 128 * 3 * 2048 * 768 * 2  # 1.2 GB
    assert set(ml.CONFIGS) == {"mixtral", "qwen3"}


# ---- GPU: routing --------------------------------------------------------------------------
def _check_route(sk, logits, k, renorm=True):
    T, E = logits.shape
    ids, w, offsets, pos = sk.moe_route(logits, k, renorm)
    assert ids.shape == (T, k) and ids.dtype == torch.int32
    assert w.shape == (T, k) and w.dtype == torch.float32
    assert offsets.shape == (E + 1,) and pos.shape == (T, k)
    lf = logits.float()
    idl = ids.long()
    # The picked logits are torch's top k, in order (ties may pick a different id of the same
    # value, so the values are compared, not the ids), and every id is picked once per token.
    want_vals = lf.topk(k, dim=-1).values
    torch.testing.assert_close(lf.gather(1, idl), want_vals, rtol=0, atol=0)
    assert (idl.sort(-1).values.diff(dim=-1) > 0).all()
    p = torch.softmax(lf, dim=-1).gather(1, idl)
    if renorm:
        p = p / p.sum(-1, keepdim=True)
    torch.testing.assert_close(w, p, rtol=1e-5, atol=1e-7)
    counts = torch.bincount(ids.reshape(-1).long(), minlength=E)
    want_off = torch.zeros(E + 1, dtype=torch.int64, device=logits.device)
    want_off[1:] = counts.cumsum(0)
    assert torch.equal(offsets.long(), want_off)
    # pos is the stable sort of the flat ids: expert, then token, then slot.
    order = torch.argsort(ids.reshape(-1), stable=True)
    want_pos = torch.empty_like(order)
    want_pos[order] = torch.arange(order.numel(), device=order.device)
    assert torch.equal(pos.reshape(-1).long(), want_pos)
    return ids, w, offsets, pos


ROUTE_CASES = [
    (1, 8, 2), (3, 8, 2), (16, 128, 8), (100, 128, 8), (1000, 8, 2), (5000, 128, 8),
    (777, 40, 4), (64, 512, 16), (33, 1, 1), (140000, 16, 8),
]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("case", ROUTE_CASES, ids=lambda c: f"T{c[0]}-E{c[1]}-k{c[2]}")
def test_moe_route(sk, case, dtype):
    T, E, k = case
    g = torch.Generator(device="cuda").manual_seed(T + E)
    bias = torch.randn(E, device="cuda", generator=g)  # uneven load
    logits = (torch.randn(T, E, device="cuda", generator=g) + bias).to(dtype)
    _check_route(sk, logits, k)


def test_moe_route_without_renorm_and_all_on_one_expert(sk):
    g = torch.Generator(device="cuda").manual_seed(5)
    logits = torch.randn(300, 64, device="cuda", generator=g)
    _check_route(sk, logits, 6, renorm=False)
    logits[:, 7] += 100.0  # every token's first pick is expert 7
    ids, _, offsets, _ = _check_route(sk, logits, 3)
    assert (ids[:, 0] == 7).all() and offsets[8] - offsets[7] == 300


def test_moe_route_is_deterministic(sk):
    logits = torch.randn(4096, 128, device="cuda")
    a = sk.moe_route(logits, 8)
    b = sk.moe_route(logits, 8)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


# ---- GPU: permute and combine -------------------------------------------------------------
@pytest.mark.parametrize("T,E,k,D", [(1, 8, 2, 4096), (37, 128, 8, 2048), (1000, 8, 2, 512),
                                     (5, 16, 16, 8192), (300, 40, 4, 136)])
def test_moe_dispatch_and_combine(sk, T, E, k, D):
    g = torch.Generator(device="cuda").manual_seed(T)
    x = torch.randn(T, D, device="cuda", generator=g).to(torch.bfloat16)
    logits = torch.randn(T, E, device="cuda", generator=g).to(torch.bfloat16)
    xs, ids, w, offsets, pos = sk.moe_dispatch(x, logits, k)
    r_ids, r_w, r_off, r_pos = sk.moe_route(logits, k)
    for a, b in ((ids, r_ids), (w, r_w), (offsets, r_off), (pos, r_pos)):
        assert torch.equal(a, b)
    assert xs.shape == (T * k, D)
    tok = torch.arange(T, device="cuda").repeat_interleave(k)
    assert torch.equal(xs[pos.reshape(-1).long()], x[tok])
    y = torch.randn(T * k, D, device="cuda", generator=g).to(torch.bfloat16)
    out = sk.moe_combine(y, pos, w)
    want = (y[pos.long()].double() * w.double().unsqueeze(-1)).sum(1)
    torch.testing.assert_close(out.double(), want, rtol=8e-3, atol=1e-3)
    assert torch.equal(out, sk.moe_combine(y, pos, w))


# ---- GPU: grouped GEMM with the SwiGLU epilogue ------------------------------------------
def _offsets(counts):
    offs = [0]
    for c in counts:
        offs.append(offs[-1] + c)
    return torch.tensor(offs, dtype=torch.int32, device="cuda")


SWIGLU_CASES = [
    ([0, 1, 0, 300, 129, 128, 0, 7], 256, 512),
    ([1000], 384, 256),
    ([0, 0, 0, 5], 128, 4096),
    ([3, 0, 50, 200] * 32, 1536, 2048),  # Qwen3-30B-A3B's gate|up, 128 experts
]


def _swiglu_ref(a, w13, offsets, scale_a=None, scale_b=None):
    offs = offsets.tolist()
    inter = w13.shape[1] // 2
    out = torch.zeros(a.shape[0], inter, device=a.device)
    for e in range(w13.shape[0]):
        lo, hi = offs[e], offs[e + 1]
        if hi == lo:
            continue
        gu = a[lo:hi].float() @ w13[e].float().t()
        if scale_b is not None:
            gu = gu * scale_b[e] * scale_a
        out[lo:hi] = F.silu(gu[:, 0::2]) * gu[:, 1::2]
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("case", SWIGLU_CASES, ids=lambda c: f"E{len(c[0])}-N{c[1]}-K{c[2]}")
def test_grouped_gemm_swiglu_bf16(sk, case):
    counts, N, K = case
    E, T = len(counts), sum(counts)
    torch.manual_seed(0)
    offsets = _offsets(counts)
    a = torch.randn(T, K, device="cuda").to(torch.bfloat16)
    g = (torch.randn(E, N // 2, K, device="cuda") / K**0.5).to(torch.bfloat16)
    u = (torch.randn(E, N // 2, K, device="cuda") / K**0.5).to(torch.bfloat16)
    w13 = sk.interleave_experts(g, u)
    assert torch.equal(w13[:, 0::2], g) and torch.equal(w13[:, 1::2], u)
    got = sk.grouped_gemm_swiglu(a, w13, offsets)
    assert got.shape == (T, N // 2) and got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), _swiglu_ref(a, w13, offsets).float(),
                               atol=3e-2, rtol=3e-2)
    assert torch.equal(got, sk.grouped_gemm_swiglu(a, w13, offsets))


@pytest.mark.parametrize("case", [c for c in SWIGLU_CASES if c[2] % 128 == 0],
                         ids=lambda c: f"E{len(c[0])}-N{c[1]}-K{c[2]}")
def test_grouped_gemm_swiglu_fp8(sk, case):
    counts, N, K = case
    E, T = len(counts), sum(counts)
    torch.manual_seed(1)
    offsets = _offsets(counts)
    a = torch.randn(T, K, device="cuda").to(torch.float8_e4m3fn)
    w13 = torch.randn(E, N, K, device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.tensor([0.5], device="cuda")
    scale_b = (torch.rand(E, device="cuda") + 0.5) / K**0.5
    got = sk.grouped_gemm_swiglu(a, w13, offsets, scale_a, scale_b)
    want = _swiglu_ref(a, w13, offsets, scale_a, scale_b)
    torch.testing.assert_close(got.float(), want.float(), atol=3e-2, rtol=3e-2)


# ---- GPU: the layer --------------------------------------------------------------------------
def _rel_err(got, want):
    return ((got.float() - want.float()).abs().max() / want.float().abs().max()).item()


@pytest.mark.parametrize("cfg", [SMALL_MIXTRAL, SMALL_QWEN, ml.QWEN3_30B_A3B],
                         ids=lambda c: c.name)
@pytest.mark.parametrize("T", [1, 16, 300])
def test_moe_layer(sk, cfg, T):
    wts = ml.MoEWeights.random(cfg, seed=T)
    x = torch.randn(T, cfg.hidden, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(T)).to(torch.bfloat16)
    want = ml.torch_moe(x, wts, acc_dtype=torch.float32)
    layer = ml.MoELayer(wts)
    got = layer(x)
    assert got.shape == x.shape and got.dtype == torch.bfloat16
    assert _rel_err(got, want) < 1e-2
    assert torch.equal(got, layer(x))  # the same bits every run
    assert _rel_err(ml.MoELayer(wts, fuse_swiglu=False)(x), want) < 1e-2
    assert _rel_err(ml.torch_moe(x, wts), want) < 2e-2  # the bf16 torch layer, for scale
    # The routing kernel picks torch's experts on the same logits (ties in the bf16 logits
    # aside, so the picked values are compared).
    ids, _, _, _ = layer.routing(x)
    logits = F.linear(x, wts.router).float()
    _, want_ids = ml._route_torch(logits, cfg)
    assert torch.equal(logits.gather(1, ids.long()).sort(-1).values,
                       logits.gather(1, want_ids).sort(-1).values)


@pytest.mark.parametrize("cfg", [SMALL_QWEN, ml.QWEN3_30B_A3B], ids=lambda c: c.name)
@pytest.mark.parametrize("T", [1, 16, 64, 300])
def test_moe_graph_replays_the_eager_bits(sk, cfg, T):
    """The forward captured in a CUDA graph at T tokens and replayed on inputs that route
    differently gives the eager forward's bits, also after a larger eager call in between
    (nothing the graph holds is reallocated)."""
    wts = ml.MoEWeights.random(cfg, seed=7)
    g = torch.Generator(device="cuda").manual_seed(T)
    layer = ml.MoELayer(wts)
    graph = ml.MoEGraph(layer, T)
    for i in range(6):
        x = torch.randn(T, cfg.hidden, device="cuda", generator=g).to(torch.bfloat16)
        if i == 3:
            layer(torch.randn(4096, cfg.hidden, device="cuda", generator=g).to(torch.bfloat16))
        assert torch.equal(graph(x).clone(), layer(x)), i


def test_torch_grouped_moe(sk):
    cfg = SMALL_QWEN
    wts = ml.MoEWeights.random(cfg, seed=2)
    x = torch.randn(200, cfg.hidden, device="cuda").to(torch.bfloat16)
    try:
        got = ml.torch_grouped_moe(x, wts)
    except (RuntimeError, NotImplementedError) as exc:
        pytest.skip(f"torch._grouped_mm is not available here: {exc}")
    assert _rel_err(got, ml.torch_moe(x, wts, acc_dtype=torch.float32)) < 2e-2
