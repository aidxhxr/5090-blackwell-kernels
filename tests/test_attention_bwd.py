"""Parity of the attention backward pass against torch autograd through
F.scaled_dot_product_attention in fp32 (from the same bf16 inputs): every variant, both masks,
both head sizes, lengths that are not a multiple of any tile, S_q != S_kv in both directions,
grouped-query attention (dK and dV summed over the query heads of a group), the deterministic
dQ pass, the forward's log-sum-exp output, and the autograd Function end to end. Then the
sliding window and the sink logits (the window as an explicit mask, the sink as one more
logit per row with no value, autograd through both) with d(sinks), on every variant."""

import pytest
import torch
import torch.nn.functional as F

# (B, H_q, H_kv, S_q, S_kv, D)
SHAPES = [(1, 2, 2, 128, 128, 64), (1, 2, 2, 200, 200, 128), (2, 3, 3, 256, 256, 64),
          (1, 2, 2, 513, 513, 128), (1, 3, 3, 150, 333, 64), (1, 2, 2, 333, 150, 128),
          (1, 8, 2, 300, 300, 128), (2, 8, 1, 200, 200, 64), (1, 4, 1, 64, 1000, 128)]


def _variants():
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants("attention_bwd")))
    except Exception:
        return [0]


def _shape_id(s):
    return "b{}_hq{}_hkv{}_sq{}_skv{}_d{}".format(*s)


def _inputs(B, Hq, Hkv, Sq, Skv, D, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(B, Hq, Sq, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(B, Hkv, Skv, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(B, Hkv, Skv, D, device="cuda", dtype=torch.bfloat16, generator=g)
    do = torch.randn(B, Hq, Sq, D, device="cuda", dtype=torch.bfloat16, generator=g)
    return q, k, v, do


def _reference(q, k, v, do, causal):
    """(out, dq, dk, dv) from torch autograd in fp32 on the bf16 inputs."""
    qf, kf, vf = (t.detach().float().requires_grad_() for t in (q, k, v))
    out = F.scaled_dot_product_attention(qf, kf, vf, is_causal=causal,
                                         enable_gqa=k.shape[1] != q.shape[1])
    out.backward(do.float())
    return out.detach(), qf.grad, kf.grad, vf.grad


def _assert_grad_close(got, ref, name):
    # Per tensor: max |got - ref| <= 2% of max |ref| + 1e-3. The kernels round P and dS to bf16
    # before their products and the outputs to bf16; an elementwise rtol would fail on the
    # entries near zero that every gradient has.
    err = (got.float() - ref).abs().max().item()
    tol = 2e-2 * ref.abs().max().item() + 1e-3
    assert err <= tol, f"{name}: max error {err:.3e} > tolerance {tol:.3e}"


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_attention_bwd_matches_autograd(sk, shape, causal, variant):
    q, k, v, do = _inputs(*shape)
    out, lse = sk.attention_fwd(q, k, v, causal=causal)
    dq, dk, dv = sk.attention_bwd(q, k, v, out, do, lse, causal=causal, variant=variant)
    assert dq.shape == q.shape and dk.shape == k.shape and dv.shape == v.shape
    assert dq.dtype == dk.dtype == dv.dtype == torch.bfloat16
    ref_out, rdq, rdk, rdv = _reference(q, k, v, do, causal)
    torch.testing.assert_close(out.float(), ref_out, atol=2e-2, rtol=2e-2)
    _assert_grad_close(dq, rdq, "dq")
    _assert_grad_close(dk, rdk, "dk")
    _assert_grad_close(dv, rdv, "dv")


@pytest.mark.parametrize("variant", [v for v in _variants() if v >= 1])
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("shape", [(1, 2, 2, 513, 513, 128), (1, 8, 2, 300, 300, 64)],
                         ids=_shape_id)
def test_attention_bwd_deterministic_is_exact_and_correct(sk, shape, causal, variant):
    # the separate dQ pass: no atomics anywhere, so two runs agree bit for bit
    q, k, v, do = _inputs(*shape)
    out, lse = sk.attention_fwd(q, k, v, causal=causal)
    a = sk.attention_bwd(q, k, v, out, do, lse, causal=causal, variant=variant,
                         deterministic=True)
    b = sk.attention_bwd(q, k, v, out, do, lse, causal=causal, variant=variant,
                         deterministic=True)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)
    _, rdq, rdk, rdv = _reference(q, k, v, do, causal)
    for got, ref, name in zip(a, (rdq, rdk, rdv), ("dq", "dk", "dv"), strict=True):
        _assert_grad_close(got, ref, name)


@pytest.mark.parametrize("variant", list(range(6)))
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("shape", [(1, 2, 2, 200, 200, 128), (1, 8, 2, 7, 300, 64),
                                   (1, 32, 8, 1, 512, 128), (1, 3, 3, 333, 150, 64)],
                         ids=_shape_id)
def test_attention_fwd_lse_matches_logsumexp(sk, shape, causal, variant):
    # every forward rung writes L = log sum_j exp(q_i . k_j / sqrt(D)); a decode shape that
    # would run the flash-decoding kernel takes the 64-row tile instead, and its output must
    # still match the plain forward
    B, Hq, Hkv, Sq, Skv, D = shape
    q, k, v, _ = _inputs(*shape)
    out, lse = sk.attention_fwd(q, k, v, causal=causal, variant=variant)
    assert lse.shape == (B, Hq, Sq) and lse.dtype == torch.float32
    kf = k.float().repeat_interleave(Hq // Hkv, dim=1)
    s = q.float() @ kf.transpose(-1, -2) / D ** 0.5
    if causal:
        s = s.masked_fill(torch.ones(Sq, Skv, device="cuda").triu(1).bool(), float("-inf"))
    torch.testing.assert_close(lse, torch.logsumexp(s, dim=-1), atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(out, sk.attention(q, k, v, causal=causal, variant=variant),
                               atol=1e-2, rtol=1e-2)


def test_attention_fwd_lse_leaves_the_default_forward_unchanged(sk):
    # asking for lse must not change what attention() computes on a prefill shape
    q, k, v, _ = _inputs(1, 4, 4, 1000, 1000, 128)
    out, _ = sk.attention_fwd(q, k, v, causal=True)
    assert torch.equal(out, sk.attention(q, k, v, causal=True))


@pytest.mark.parametrize("deterministic", [False, True])
@pytest.mark.parametrize("shape", [(2, 8, 2, 257, 257, 128), (1, 4, 4, 300, 300, 64)],
                         ids=_shape_id)
def test_attention_with_grad_end_to_end(sk, shape, deterministic):
    q, k, v, do = _inputs(*shape)
    q, k, v = (t.clone().requires_grad_() for t in (q, k, v))
    out = sk.attention_with_grad(q, k, v, causal=True, deterministic=deterministic)
    assert torch.equal(out.detach(), sk.attention(q.detach(), k.detach(), v.detach(), causal=True))
    (out.float() * do.float()).sum().backward()
    _, rdq, rdk, rdv = _reference(q, k, v, do, True)
    _assert_grad_close(q.grad, rdq, "dq")
    _assert_grad_close(k.grad, rdk, "dk")
    _assert_grad_close(v.grad, rdv, "dv")


def test_attention_bwd_gqa_equals_summed_repeated_heads(sk):
    # the other spelling of GQA: K/V repeated to the query heads, then dK and dV summed back
    q, k, v, do = _inputs(1, 8, 2, 256, 256, 128)
    out, lse = sk.attention_fwd(q, k, v, causal=True)
    dq, dk, dv = sk.attention_bwd(q, k, v, out, do, lse, causal=True, deterministic=True)
    k4, v4 = k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)
    out4, lse4 = sk.attention_fwd(q, k4, v4, causal=True)
    dq4, dk4, dv4 = sk.attention_bwd(q, k4, v4, out4, do, lse4, causal=True, deterministic=True)
    torch.testing.assert_close(dq, dq4, atol=0, rtol=0)
    for got, rep in ((dk, dk4), (dv, dv4)):
        summed = rep.float().view(1, 2, 4, 256, 128).sum(dim=2)
        _assert_grad_close(got, summed, "dk/dv")


@pytest.mark.parametrize("variant", [v for v in _variants() if v >= 1])
def test_attention_bwd_repeated_calls_and_streams(sk, variant):
    # the workspaces (L and Dv rows, the fp32 dQ buffer, the split partials) are reused by
    # every call; a call on a second stream after the first has finished must agree
    q, k, v, do = _inputs(2, 4, 4, 700, 700, 128)
    out, lse = sk.attention_fwd(q, k, v, causal=True)
    ref = sk.attention_bwd(q, k, v, out, do, lse, causal=True, variant=variant,
                           deterministic=True)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        other = sk.attention_bwd(q, k, v, out, do, lse, causal=True, variant=variant,
                                 deterministic=True)
    torch.cuda.synchronize()
    for _ in range(2):
        again = sk.attention_bwd(q, k, v, out, do, lse, causal=True, variant=variant)
        for x, y in zip(again, ref, strict=True):
            _assert_grad_close(x, y.float(), "repeat")
    for x, y in zip(other, ref, strict=True):
        assert torch.equal(x, y)


def test_attention_bwd_rejects_bad_inputs(sk):
    q, k, v, do = _inputs(1, 2, 2, 64, 64, 128)
    out, lse = sk.attention_fwd(q, k, v)
    with pytest.raises(RuntimeError):
        sk.attention_bwd(q, k, v, out, do, lse.double())
    with pytest.raises(RuntimeError):
        sk.attention_bwd(q, k, v, out, do[:, :, :32].contiguous(), lse)
    with pytest.raises(RuntimeError):
        sk.attention_bwd(q, k, v, out, do, lse, variant=sk.num_variants("attention_bwd"))


# ---- sliding window and sinks ---------------------------------------------------------------

# (B, H_q, H_kv, S_q, S_kv, D, window): a tiny case, a window inside the sequence on the
# 128-row tiles, GQA at a length that is no tile multiple, a chunk against a longer cache (the
# diagonal shifted to the last key, S_q < S_kv), a window longer than the sequence (which is
# plain causal attention, with the sink alone)
WINDOW_SHAPES = [(1, 1, 1, 8, 8, 64, 3), (1, 2, 2, 300, 300, 128, 64), (1, 8, 2, 513, 513, 64, 128),
                 (1, 4, 1, 200, 1000, 128, 256), (2, 4, 4, 256, 256, 128, 1000)]


def _wshape_id(s):
    return "b{}_hq{}_hkv{}_sq{}_skv{}_d{}_w{}".format(*s)


def _sinks(H, seed=11):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.rand(H, generator=g) * 4 - 1).cuda()  # logits in [-1, 3]


def _window_scores(q, k, window, sinks):
    """The fp32 masked scores, the sink column appended when there is one: query i at key
    position i + S_kv - S_q sees keys pos - window < j <= pos."""
    Hq, Sq, D = q.shape[1:]
    Hkv, Skv = k.shape[1], k.shape[2]
    s = q @ k.repeat_interleave(Hq // Hkv, dim=1).transpose(-1, -2) / D ** 0.5
    i = torch.arange(Sq, device=q.device)[:, None]
    j = torch.arange(Skv, device=q.device)[None, :]
    pos = i + Skv - Sq
    s = s.masked_fill(~((j <= pos) & (j > pos - window)), float("-inf"))
    if sinks is None:
        return s
    return torch.cat([s, sinks.view(1, Hq, 1, 1).expand(*s.shape[:-1], 1)], dim=-1)


def _window_reference(q, k, v, do, window, sinks):
    """(out, dq, dk, dv, dsinks) from torch autograd in fp32 on the bf16 inputs."""
    Hq, Hkv = q.shape[1], k.shape[1]
    qf, kf, vf = (t.detach().float().requires_grad_() for t in (q, k, v))
    sf = sinks.detach().float().requires_grad_() if sinks is not None else None
    p = torch.softmax(_window_scores(qf, kf, window, sf), dim=-1)
    if sf is not None:
        p = p[..., :-1]  # the sink has no value
    out = p @ vf.repeat_interleave(Hq // Hkv, dim=1)
    out.backward(do.float())
    return out.detach(), qf.grad, kf.grad, vf.grad, (sf.grad if sf is not None else None)


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("sink", [False, True], ids=["nosink", "sink"])
@pytest.mark.parametrize("shape", WINDOW_SHAPES, ids=_wshape_id)
def test_attention_bwd_window_matches_autograd(sk, shape, sink, variant):
    *dims, window = shape
    q, k, v, do = _inputs(*dims)
    sinks = _sinks(dims[1]) if sink else None
    out, lse = sk.attention_fwd(q, k, v, causal=True, window=window, sinks=sinks)
    grads = sk.attention_bwd(q, k, v, out, do, lse, causal=True, variant=variant, window=window,
                             sinks=sinks)
    assert len(grads) == (4 if sink else 3)
    ref = _window_reference(q, k, v, do, window, sinks)
    torch.testing.assert_close(out.float(), ref[0], atol=2e-2, rtol=2e-2)
    for got, want, name in zip(grads, ref[1:], ("dq", "dk", "dv", "dsinks"), strict=False):
        _assert_grad_close(got, want, name)
    if sink:
        assert grads[3].shape == (dims[1],) and grads[3].dtype == torch.float32


@pytest.mark.parametrize("variant", list(range(6)))
@pytest.mark.parametrize("shape", [(1, 2, 2, 300, 300, 128, 64), (1, 4, 1, 200, 1000, 128, 256),
                                   (1, 8, 2, 7, 500, 64, 128)], ids=_wshape_id)
def test_attention_fwd_lse_includes_the_sink(sk, shape, variant):
    # every forward rung's L under a window and sinks is log(exp(sink) + sum_j exp(s_ij)), what
    # the backward recomputes P from; the decode shape runs the 64-row tile
    *dims, window = shape
    q, k, v, _ = _inputs(*dims)
    sinks = _sinks(dims[1])
    out, lse = sk.attention_fwd(q, k, v, causal=True, variant=variant, window=window, sinks=sinks)
    s = _window_scores(q.float(), k.float(), window, sinks)
    torch.testing.assert_close(lse, torch.logsumexp(s, dim=-1), atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(
        out, sk.attention(q, k, v, causal=True, variant=variant, window=window, sinks=sinks),
        atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("variant", [v for v in _variants() if v >= 1])
def test_attention_bwd_window_deterministic_is_exact_and_correct(sk, variant):
    q, k, v, do = _inputs(1, 8, 2, 513, 513, 128)
    sinks = _sinks(8)
    out, lse = sk.attention_fwd(q, k, v, causal=True, window=200, sinks=sinks)
    kw = dict(causal=True, variant=variant, window=200, sinks=sinks, deterministic=True)
    a = sk.attention_bwd(q, k, v, out, do, lse, **kw)
    b = sk.attention_bwd(q, k, v, out, do, lse, **kw)
    for x, y in zip(a, b, strict=True):
        assert torch.equal(x, y)
    ref = _window_reference(q, k, v, do, 200, sinks)
    for got, want, name in zip(a, ref[1:], ("dq", "dk", "dv", "dsinks"), strict=True):
        _assert_grad_close(got, want, name)


def test_attention_with_grad_window_and_sinks(sk):
    q, k, v, do = _inputs(1, 8, 2, 300, 300, 128)
    sinks = _sinks(8).requires_grad_()
    q, k, v = (t.clone().requires_grad_() for t in (q, k, v))
    out = sk.attention_with_grad(q, k, v, window=100, sinks=sinks)
    assert torch.equal(out.detach(),
                       sk.attention(q.detach(), k.detach(), v.detach(), window=100,
                                    sinks=sinks.detach()))
    (out.float() * do.float()).sum().backward()
    ref = _window_reference(q, k, v, do, 100, sinks)
    for got, want, name in zip((q.grad, k.grad, v.grad, sinks.grad), ref[1:],
                               ("dq", "dk", "dv", "dsinks"), strict=True):
        _assert_grad_close(got, want, name)


def test_attention_bwd_long_window_is_the_causal_backward(sk):
    # a window at least S_kv long with S_q = S_kv is plain causal attention, bit for bit
    q, k, v, do = _inputs(1, 4, 4, 700, 700, 128)
    out, lse = sk.attention_fwd(q, k, v, causal=True)
    want = sk.attention_bwd(q, k, v, out, do, lse, causal=True, deterministic=True)
    got = sk.attention_bwd(q, k, v, out, do, lse, window=700, deterministic=True)
    for x, y in zip(got, want, strict=True):
        assert torch.equal(x, y)


def test_attention_bwd_window_rejects_bad_inputs(sk):
    q, k, v, do = _inputs(1, 4, 4, 128, 64, 128)
    out, lse = sk.attention_fwd(q, k, v)
    with pytest.raises(RuntimeError):  # more queries than keys: no position for the first
        sk.attention_bwd(q, k, v, out, do, lse, window=16)
    with pytest.raises(RuntimeError):  # one sink per query head
        sk.attention_bwd(q, k, v, out, do, lse, sinks=torch.zeros(3, device="cuda"))
