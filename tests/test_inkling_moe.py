"""Inkling 2 Mini's MoE pieces on the kernels against torch copies of transformers 5.24.1's
modules (modeling_inkling2_loop.py): the sigmoid router with the correction bias
(Inkling2LoopTopKRouter) through moe_route / moe_route_rows / moe_dispatch, and the
softcapped SwiGLU (Inkling2LoopSoftcappedSwiGLU) through swiglu and through the epilogue of
grouped_gemm_swiglu, bf16 and fp8. Every test here needs the extension (the sk fixture)."""

import pytest
import torch

E, K_TOP, ROUTE_SCALE = 128, 8, 8.0  # Inkling 2 Mini: n_routed_experts, top 8, route_scale
CAP_G, CAP_U = 4.0, 25.0  # mlp_gate_softcap, mlp_up_softcap


# ---- the torch modules, copied from modeling_inkling2_loop.py ------------------------------
def hf_router(logits, bias, top_k=K_TOP, scale=ROUTE_SCALE):
    """Inkling2LoopTopKRouter.forward after its fp32 linear: (weights, indices)."""
    scores = torch.sigmoid(logits.float())
    indices = torch.topk(scores + bias, top_k, dim=-1, sorted=False).indices
    weights = scores.gather(1, indices)
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
    return weights * scale, indices


def hf_softcap(x, c):
    return torch.tanh(x / c) * c


def hf_swiglu(gate, up, cap_g=CAP_G, cap_u=CAP_U):
    """Inkling2LoopSoftcappedSwiGLU.forward."""
    dtype = gate.dtype
    gate, up = gate.float(), up.float()
    return (hf_softcap(gate, cap_g) * torch.sigmoid(gate) * hf_softcap(up, cap_u)).to(dtype)


# ---- the router ----------------------------------------------------------------------------
def _by_id(ids, w):
    """ids and weights of each row sorted by expert id, so sets compare as tensors."""
    order = ids.long().sort(-1).indices
    return ids.long().gather(1, order), w.gather(1, order)


def _check_route(sk, logits, bias, k=K_TOP, scale=ROUTE_SCALE):
    T, n_exp = logits.shape
    ids, w, offsets, pos = sk.moe_route(logits, k, scoring="sigmoid", bias=bias, scale=scale)
    want_w, want_ids = hf_router(logits, bias if bias is not None else 0.0, k, scale)
    got_ids, got_w = _by_id(ids, w)
    want_ids, want_w = _by_id(want_ids, want_w)
    assert torch.equal(got_ids, want_ids)
    torch.testing.assert_close(got_w, want_w, rtol=4e-6, atol=1e-7)
    # ids come largest key first, and the expert-order outputs are moe_route's.
    key = torch.sigmoid(logits.float()) + (bias if bias is not None else 0.0)
    kv = key.gather(1, ids.long())
    assert (kv[:, :-1] >= kv[:, 1:]).all()
    counts = torch.bincount(ids.reshape(-1).long(), minlength=n_exp)
    assert torch.equal(offsets[1:].long(), counts.cumsum(0)) and offsets[0] == 0
    order = torch.argsort(ids.reshape(-1), stable=True)
    want_pos = torch.empty_like(order)
    want_pos[order] = torch.arange(order.numel(), device=order.device)
    assert torch.equal(pos.reshape(-1).long(), want_pos)
    return ids, w, offsets, pos


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("T", [1, 8, 64, 333, 4096])
def test_sigmoid_router_matches_hf(sk, T, dtype):
    g = torch.Generator(device="cuda").manual_seed(T)
    bias = torch.randn(E, device="cuda", generator=g) * 0.05
    logits = (torch.randn(T, E, device="cuda", generator=g) * 2.0).to(dtype)
    _check_route(sk, logits, bias)


def test_sigmoid_router_without_bias_and_other_shapes(sk):
    g = torch.Generator(device="cuda").manual_seed(11)
    _check_route(sk, torch.randn(200, E, device="cuda", generator=g), None)
    for T, n_exp, k in ((50, 384, 6), (7, 32, 4), (90, 512, 16)):  # DeepSeek V3-like sizes too
        bias = torch.randn(n_exp, device="cuda", generator=g) * 0.1
        _check_route(sk, torch.randn(T, n_exp, device="cuda", generator=g), bias, k, 2.5)


def test_sigmoid_router_ties_broken_by_the_bias(sk):
    """Scores that tie exactly are told apart by the bias alone: equal logits on a row,
    logits large enough that every sigmoid is 1.0 in fp32, and pairs of equal logits."""
    g = torch.Generator(device="cuda").manual_seed(3)
    bias = torch.randperm(E, device="cuda", generator=g).float() * 1e-3  # distinct
    equal = torch.full((16, E), 0.3, device="cuda")
    saturated = 20.0 + torch.rand(16, E, device="cuda", generator=g) * 10.0
    assert (torch.sigmoid(saturated) == 1.0).all()
    pairs = torch.randn(16, E // 2, device="cuda", generator=g).repeat_interleave(2, dim=1)
    for logits in (equal, saturated, pairs):
        ids, w, _, _ = _check_route(sk, logits, bias)
        if logits is not pairs:  # every score the same: the bias's top 8, equal weights
            want = bias.topk(K_TOP).indices.sort().values
            assert (ids.long().sort(-1).values == want).all()
            torch.testing.assert_close(w, torch.full_like(w, ROUTE_SCALE / K_TOP))


def test_sigmoid_router_rows_and_dispatch(sk):
    g = torch.Generator(device="cuda").manual_seed(5)
    T, D = 300, 2304
    bias = torch.randn(E, device="cuda", generator=g) * 0.05
    logits = torch.randn(T, E, device="cuda", generator=g)
    x = torch.randn(T, D, device="cuda", generator=g).to(torch.bfloat16)
    ref = _check_route(sk, logits, bias)
    kw = dict(scoring="sigmoid", bias=bias, scale=ROUTE_SCALE)
    ids, w, offsets, pos, rows = sk.moe_route_rows(logits, K_TOP, **kw)
    for a, b in zip((ids, w, offsets, pos), ref, strict=True):
        assert torch.equal(a, b)
    tok = torch.arange(T, device="cuda", dtype=torch.int32).repeat_interleave(K_TOP)
    assert torch.equal(rows[pos.reshape(-1).long()], tok)
    xs, ids2, w2, off2, pos2 = sk.moe_dispatch(x, logits, K_TOP, **kw)
    for a, b in zip((ids2, w2, off2, pos2), ref, strict=True):
        assert torch.equal(a, b)
    assert torch.equal(xs[pos2.reshape(-1).long()], x.repeat_interleave(K_TOP, dim=0))
    # Without renormalization the weights are the scores times the scale.
    ids, w, _, _ = sk.moe_route(logits, K_TOP, False, **kw)
    torch.testing.assert_close(w, torch.sigmoid(logits).gather(1, ids.long()) * ROUTE_SCALE)


def test_router_arguments(sk):
    logits = torch.randn(4, E, device="cuda")
    with pytest.raises(ValueError):
        sk.moe_route(logits, K_TOP, bias=torch.zeros(E, device="cuda"))  # softmax + bias
    with pytest.raises(ValueError):
        sk.moe_route(logits, K_TOP, scoring="softplus")
    with pytest.raises(RuntimeError):
        sk.moe_route(logits, K_TOP, scoring="sigmoid", bias=torch.zeros(E - 1, device="cuda"))
    # The softmax default is untouched: the same call as before the scoring option.
    a = sk.moe_route(logits, K_TOP)
    b = sk.moe_route(logits, K_TOP, True, scoring="softmax")
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)


# ---- the softcapped SwiGLU -----------------------------------------------------------------
def _big(shape, g, gs=6.0, us=40.0, dtype=torch.bfloat16):
    """gate and up wide enough that both caps bite: |g| up to ~4 caps, |u| past 25."""
    gate = torch.randn(*shape, device="cuda", generator=g) * gs
    up = torch.randn(*shape, device="cuda", generator=g) * us
    return gate.to(dtype), up.to(dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
@pytest.mark.parametrize("shape", [(1, 1152), (7, 1000), (4096, 1152), (3, 5, 77)],
                         ids=lambda s: "x".join(map(str, s)))
def test_softcap_swiglu_matches_hf(sk, shape, dtype):
    g = torch.Generator(device="cuda").manual_seed(sum(shape))
    gate, up = _big(shape, g, dtype=dtype)
    got = sk.swiglu(gate, up, softcap=(CAP_G, CAP_U))
    want = hf_swiglu(gate, up)
    assert got.shape == gate.shape and got.dtype == dtype
    if dtype == torch.bfloat16:
        # The module's fp32 arithmetic op for op; torch's tanh differs from tanhf in the last
        # bit now and then, which moves the odd bf16 rounding.
        torch.testing.assert_close(got, want, rtol=8e-3, atol=1e-2)
        assert (got != want).float().mean() < 1e-3
    else:
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    # The caps bite: the plain SwiGLU differs.
    assert (sk.swiglu(gate, up).float() - want.float()).abs().max() > 10.0


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_softcap_swiglu_reads_the_halves_in_place(sk, dtype):
    g = torch.Generator(device="cuda").manual_seed(9)
    for rows, inter in ((64, 1152), (5, 100)):  # vector rows, and the scalar fallback
        gate, up = _big((rows, 2 * inter), g, dtype=dtype)
        gu = torch.cat((gate[:, :inter], up[:, inter:]), dim=1)
        got = sk.swiglu(gu[:, :inter], gu[:, inter:], softcap=(CAP_G, CAP_U))
        assert got.is_contiguous()
        assert torch.equal(got, sk.swiglu(gu[:, :inter].contiguous(),
                                          gu[:, inter:].contiguous(), softcap=(CAP_G, CAP_U)))
        torch.testing.assert_close(got.float(), hf_swiglu(gu[:, :inter], gu[:, inter:]).float(),
                                   rtol=8e-3, atol=1e-2)


def test_softcap_swiglu_limits(sk):
    """Very large and very small inputs: the caps hold and nothing turns into NaN."""
    v = torch.tensor([-1e4, -100.0, -30.0, -4.0, -1.0, 0.0, 1.0, 4.0, 30.0, 100.0, 1e4],
                     device="cuda")
    gate = v.repeat_interleave(v.numel())
    up = v.repeat(v.numel())
    got = sk.swiglu(gate, up, softcap=(CAP_G, CAP_U))
    want = hf_swiglu(gate, up)
    assert torch.isfinite(got).all()
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    assert got.abs().max() <= CAP_G * CAP_U


# ---- the grouped GEMM's epilogue -----------------------------------------------------------
def _offsets(counts):
    offs = [0]
    for c in counts:
        offs.append(offs[-1] + c)
    return torch.tensor(offs, dtype=torch.int32, device="cuda")


def _grouped_ref(a, w13, offsets, scale_a=None, scale_b=None):
    """Per expert: a @ w13[e].T in fp32 (scaled for fp8), the HF activation in fp32 on the
    interleaved columns, one rounding to bf16."""
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
        out[lo:hi] = hf_swiglu(gu[:, 0::2], gu[:, 1::2])
    return out.to(torch.bfloat16)


# Rows per expert that put the tile picker on its 16-, 64- and 128-row tiles at Inkling's
# gate|up shape (2I = 2304, H = 2304), and a scaled-down case with empty experts.
GROUPED_CASES = [
    ([1, 0, 2, 0, 0, 3, 1, 1] * 16, 2304, 2304),
    ([40, 0, 90, 7] * 32, 2304, 2304),
    ([700, 0, 1500, 300], 2304, 2304),
    ([0, 1, 0, 300, 129, 128, 0, 7], 256, 512),
]


@pytest.mark.parametrize("case", GROUPED_CASES, ids=lambda c: f"T{sum(c[0])}-E{len(c[0])}")
def test_grouped_gemm_softcap_swiglu_bf16(sk, case):
    counts, N, K = case
    n_exp, T = len(counts), sum(counts)
    g = torch.Generator(device="cuda").manual_seed(T)
    offsets = _offsets(counts)
    # Sums of std ~6 (gate) and ~40 (up) so both caps bite.
    a = torch.randn(T, K, device="cuda", generator=g).to(torch.bfloat16)
    wg = (torch.randn(n_exp, N // 2, K, device="cuda", generator=g) * 6 / K**0.5)
    wu = (torch.randn(n_exp, N // 2, K, device="cuda", generator=g) * 40 / K**0.5)
    w13 = sk.interleave_experts(wg.to(torch.bfloat16), wu.to(torch.bfloat16))
    got = sk.grouped_gemm_swiglu(a, w13, offsets, softcap=(CAP_G, CAP_U))
    assert got.shape == (T, N // 2) and got.dtype == torch.bfloat16
    want = _grouped_ref(a, w13, offsets)
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=5e-2)
    assert torch.equal(got, sk.grouped_gemm_swiglu(a, w13, offsets, softcap=(CAP_G, CAP_U)))
    # The gathered loads give the same bits, and the plain epilogue is still the plain one.
    rows = torch.randperm(T, device="cuda", generator=g).to(torch.int32)
    x = torch.empty_like(a)
    x[rows.long()] = a
    assert torch.equal(sk.grouped_gemm_swiglu(x, w13, offsets, rows=rows, softcap=(CAP_G, CAP_U)),
                       got)
    plain = sk.grouped_gemm_swiglu(a, w13, offsets)
    assert (plain.float() - got.float()).abs().max() > 10.0


@pytest.mark.parametrize("case", GROUPED_CASES, ids=lambda c: f"T{sum(c[0])}-E{len(c[0])}")
def test_grouped_gemm_softcap_swiglu_fp8(sk, case):
    counts, N, K = case
    if K % 128:
        pytest.skip("fp8 needs K % 128 == 0")
    n_exp, T = len(counts), sum(counts)
    g = torch.Generator(device="cuda").manual_seed(T + 1)
    offsets = _offsets(counts)
    a = torch.randn(T, K, device="cuda", generator=g).to(torch.float8_e4m3fn)
    w13 = torch.randn(n_exp, N, K, device="cuda", generator=g).to(torch.float8_e4m3fn)
    scale_a = torch.tensor([0.5], device="cuda")
    scale_b = (torch.rand(n_exp, device="cuda", generator=g) + 0.5) * 20 / K**0.5
    got = sk.grouped_gemm_swiglu(a, w13, offsets, scale_a, scale_b, softcap=(CAP_G, CAP_U))
    want = _grouped_ref(a, w13, offsets, scale_a, scale_b)
    torch.testing.assert_close(got.float(), want.float(), rtol=2e-2, atol=5e-2)


def test_grouped_gemm_softcap_arguments(sk):
    a = torch.randn(4, 256, device="cuda").to(torch.bfloat16)
    w13 = torch.randn(2, 256, 256, device="cuda").to(torch.bfloat16)
    offsets = _offsets([1, 3])
    with pytest.raises(RuntimeError):
        sk.grouped_gemm_swiglu(a, w13, offsets, softcap=(4.0, 0.0))
    with pytest.raises(RuntimeError):
        sk.grouped_gemm_swiglu(a, w13, offsets, softcap=(-4.0, 25.0))


# ---- the layer (moe_layer.py's Inkling config) ---------------------------------------------
def _load_moe_layer():
    import importlib.util
    import sys
    from pathlib import Path

    try:
        from spark_kernels import moe_layer
        return moe_layer
    except ImportError:
        path = Path(__file__).resolve().parent.parent / "python" / "spark_kernels" / "moe_layer.py"
        spec = importlib.util.spec_from_file_location("spark_kernels_moe_layer_inkling", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["spark_kernels_moe_layer_inkling"] = mod
        spec.loader.exec_module(mod)
        return mod


ml = _load_moe_layer()
SMALL_INKLING = ml.MoELayerConfig("small_inkling", hidden=256, inter=128, experts=32, top_k=4,
                                  scoring="sigmoid", route_scale=8.0, softcap=(CAP_G, CAP_U))


def _hf_block(x, wts):
    """Inkling2LoopSparseMoeBlock.forward, copied, on MoEWeights: the fp32 router, then
    Inkling2LoopExperts' loop (gate_up as one [2I, D] weight, gate the first I rows)."""
    cfg = wts.cfg
    weights, idx = hf_router(torch.nn.functional.linear(x.float(), wts.router), wts.bias,
                             cfg.top_k, cfg.route_scale)
    final = torch.zeros_like(x, dtype=torch.float32)
    mask = torch.nn.functional.one_hot(idx, num_classes=cfg.experts + 1).permute(2, 1, 0)
    hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
    for e in hit:
        e = e[0]
        if e == cfg.experts:
            continue
        pos, tok = torch.where(mask[e])
        gate, up = torch.nn.functional.linear(x[tok], wts.w13[e]).chunk(2, dim=-1)
        h = torch.nn.functional.linear(hf_swiglu(gate, up, *cfg.softcap), wts.w2[e])
        final.index_add_(0, tok, (h * weights[tok, pos, None]).to(final.dtype))
    return final.to(x.dtype)


def test_inkling_config():
    cfg = ml.INKLING_2_MINI
    assert (cfg.experts, cfg.top_k, cfg.hidden, cfg.inter) == (128, 8, 2304, 1152)
    assert cfg.scoring == "sigmoid" and cfg.route_scale == 8.0 and cfg.softcap == (4.0, 25.0)
    assert ml.CONFIGS["inkling"] is cfg
    assert cfg.layer_bytes() == 128 * 3 * 2304 * 1152 * 2  # 2.04 GB


def test_torch_moe_is_the_hf_block_on_cpu():
    wts = ml.MoEWeights.random(SMALL_INKLING, device="cpu", seed=1, std=0.3)
    assert wts.router.dtype == torch.float32 and wts.bias.shape == (32,)
    x = torch.randn(40, 256, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    want = _hf_block(x, wts)
    got = ml.torch_moe(x, wts)
    assert torch.equal(got, want)


def _rel_err(got, want):
    return ((got.float() - want.float()).abs().max() / want.float().abs().max()).item()


@pytest.mark.parametrize("cfg", [SMALL_INKLING, ml.INKLING_2_MINI], ids=lambda c: c.name)
@pytest.mark.parametrize("T", [1, 8, 64, 300])
def test_inkling_layer(sk, cfg, T):
    # Expert weights of std 0.3 (D = 256) and 0.1 (D = 2304) give gate and up sums of std ~5,
    # past the gate's cap of 4.
    wts = ml.MoEWeights.random(cfg, seed=T, std=0.3 if cfg.hidden < 1024 else 0.1)
    x = torch.randn(T, cfg.hidden, device="cuda",
                    generator=torch.Generator(device="cuda").manual_seed(T)).to(torch.bfloat16)
    want = _hf_block(x, wts)
    assert _rel_err(ml.torch_moe(x, wts), want) < 1e-5  # the same ops, maybe not the same order
    layer = ml.MoELayer(wts)
    got = layer(x)
    assert got.shape == x.shape and got.dtype == torch.bfloat16
    assert _rel_err(got, want) < 1e-2
    assert torch.equal(got, layer(x))
    assert torch.equal(ml.MoELayer(wts, gather=True)(x), ml.MoELayer(wts, gather=False)(x))
    assert _rel_err(ml.MoELayer(wts, fuse_swiglu=False)(x), want) < 1e-2
    plain = ml.MoELayer(wts)
    plain.softcap = None
    assert _rel_err(plain(x), want) > 5e-2  # the caps matter at this scale
    graph = ml.MoEGraph(layer, T)
    assert torch.equal(graph(x).clone(), got)
