"""Parity of linear_cross_entropy (and its pieces) against torch in fp32: the loss and both
gradients for every variant, rows with ignore_index, T not a multiple of any tile or chunk,
"mean" and "sum", an upstream gradient other than 1, the no-grad forward, and the kernels
on their own (the fused row kernel on bf16 logits, the GEMM's stats and gradient epilogues,
the padded transpose)."""

import pytest
import torch
import torch.nn.functional as F

# (T, D, V)
SHAPES = [(256, 128, 512), (300, 256, 1024), (1000, 512, 4096), (77, 64, 384)]
CHUNKS = [64, 4096]  # several chunks with a ragged last one, and one chunk


def _shape_id(s):
    return "t{}_d{}_v{}".format(*s)


def _inputs(T, D, V, seed=0, ignore_frac=0.15, ignore_index=-100):
    g = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn(T, D, device="cuda", generator=g).to(torch.bfloat16)
    # logits with a standard deviation of about 3, a peaked softmax like a trained model's
    w = (torch.randn(V, D, device="cuda", generator=g) * (3.0 / D**0.5)).to(torch.bfloat16)
    y = torch.randint(0, V, (T,), device="cuda", generator=g)
    drop = torch.rand(T, device="cuda", generator=g) < ignore_frac
    y[drop] = ignore_index
    return h, w, y


def _reference(h, w, y, ignore_index=-100, reduction="mean", upstream=1.0):
    """(loss, dh, dw) from torch autograd in fp32 on the bf16 inputs."""
    hf = h.detach().float().requires_grad_()
    wf = w.detach().float().requires_grad_()
    loss = F.cross_entropy(hf @ wf.t(), y, ignore_index=ignore_index, reduction=reduction)
    (loss * upstream).backward()
    return loss.detach(), hf.grad, wf.grad


def _assert_close(got, ref, name, rel=2e-2, floor=1e-6):
    # Per tensor: max |got - ref| <= rel * max |ref| + floor. The gradients are bf16 and
    # dw is summed in bf16 across chunks; an elementwise rtol fails on the entries near zero.
    err = (got.float() - ref).abs().max().item()
    tol = rel * ref.abs().max().item() + floor
    assert err <= tol, f"{name}: max error {err:.3e} > tolerance {tol:.3e}"


@pytest.mark.parametrize("variant", [0, 1, 2])
@pytest.mark.parametrize("chunk", CHUNKS, ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_linear_cross_entropy_matches_fp32(sk, shape, chunk, variant):
    h, w, y = _inputs(*shape)
    ref_loss, ref_dh, ref_dw = _reference(h, w, y)
    hh = h.clone().requires_grad_()
    ww = w.clone().requires_grad_()
    loss = sk.linear_cross_entropy(hh, ww, y, variant=variant, chunk=chunk)
    loss.backward()
    assert loss.dtype == torch.float32 and loss.dim() == 0
    torch.testing.assert_close(loss, ref_loss, atol=2e-3, rtol=2e-3)
    _assert_close(hh.grad, ref_dh, "dh")
    _assert_close(ww.grad, ref_dw, "dw")
    assert hh.grad.dtype == torch.bfloat16 and ww.grad.dtype == torch.bfloat16


@pytest.mark.parametrize("variant", [1, 2])
def test_sum_reduction_and_upstream_gradient(sk, variant):
    h, w, y = _inputs(500, 256, 1024, seed=3)
    ref_loss, ref_dh, ref_dw = _reference(h, w, y, reduction="sum", upstream=0.37)
    hh = h.clone().requires_grad_()
    ww = w.clone().requires_grad_()
    loss = sk.linear_cross_entropy(hh, ww, y, reduction="sum", variant=variant, chunk=128)
    (loss * 0.37).backward()
    torch.testing.assert_close(loss, ref_loss, atol=1e-2, rtol=2e-3)
    _assert_close(hh.grad, ref_dh, "dh")
    _assert_close(ww.grad, ref_dw, "dw")


@pytest.mark.parametrize("variant", [1, 2])
def test_custom_ignore_index_and_grad_of_one_input(sk, variant):
    h, w, y = _inputs(200, 128, 512, seed=4, ignore_frac=0.5, ignore_index=7)
    ref_loss, ref_dh, _ = _reference(h, w, y, ignore_index=7)
    hh = h.clone().requires_grad_()
    loss = sk.linear_cross_entropy(hh, w, y, ignore_index=7, variant=variant, chunk=64)
    loss.backward()
    torch.testing.assert_close(loss, ref_loss, atol=2e-3, rtol=2e-3)
    _assert_close(hh.grad, ref_dh, "dh")


@pytest.mark.parametrize("variant", [-1, 1, 2])
def test_forward_without_grad(sk, variant):
    h, w, y = _inputs(300, 256, 1024, seed=5)
    ref_loss, _, _ = _reference(h, w, y)
    with torch.no_grad():
        loss = sk.linear_cross_entropy(h, w, y, variant=variant, chunk=128)
    torch.testing.assert_close(loss, ref_loss, atol=2e-3, rtol=2e-3)


def test_every_row_ignored_is_nan_like_torch(sk):
    h, w, y = _inputs(64, 64, 256, seed=6)
    y.fill_(-100)
    for variant in (1, 2):
        loss = sk.linear_cross_entropy(h, w, y, variant=variant)
        assert torch.isnan(loss)


def test_unsupported_shapes(sk):
    h, w, y = _inputs(64, 64, 256)
    w200 = w[:200].contiguous()
    y = torch.where(y >= 200, y - 200, y)
    with pytest.raises(ValueError):
        sk.linear_cross_entropy(h, w200, y, variant=1)
    with pytest.raises(ValueError):
        sk.linear_cross_entropy(h, w, y, reduction="none")
    # the default steps down to torch for a vocabulary that is not a multiple of 128
    ref, _, _ = _reference(h, w200, y)
    torch.testing.assert_close(sk.linear_cross_entropy(h, w200, y), ref, atol=2e-3, rtol=2e-3)


# ---- the pieces --------------------------------------------------------------------------


@pytest.mark.parametrize("rows,V", [(37, 1024), (300, 128256 // 4), (5, 151936)])
def test_ce_rows_matches_torch_on_bf16_logits(sk, rows, V):
    torch.manual_seed(rows)
    logits = (torch.randn(rows, V, device="cuda") * 3).to(torch.bfloat16)
    y = torch.randint(0, V, (rows,), device="cuda")
    y[::4] = -100
    lf = logits.float().requires_grad_()
    ref = F.cross_entropy(lf, y, reduction="none")
    ref.sum().backward()
    scale = torch.full((1,), 0.5, device="cuda")
    loss, lse = sk.ops._C.ce_rows(logits, y, -100, scale)
    torch.testing.assert_close(loss, ref.detach(), atol=1e-4, rtol=1e-4)
    keep = y != -100
    torch.testing.assert_close(lse[keep], torch.logsumexp(lf.detach(), 1)[keep], atol=1e-4,
                               rtol=1e-4)
    # logits now hold the gradient rows, rounded to bf16
    torch.testing.assert_close(logits.float(), lf.grad * 0.5, atol=1e-6, rtol=1e-2)


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_ce_stats_and_grad_epilogues(sk, shape):
    h, w, y = _inputs(*shape, seed=7)
    x = h.float() @ w.float().t()
    lse, loss = sk.ops._C.ce_stats(h, w, y, -100)
    keep = y != -100
    ref_lse = torch.logsumexp(x, 1)
    torch.testing.assert_close(lse, ref_lse, atol=1e-3, rtol=1e-4)
    ref_loss = F.cross_entropy(x, y, reduction="none")
    torch.testing.assert_close(loss, ref_loss, atol=1e-3, rtol=1e-4)
    assert (loss[~keep] == 0).all()
    scale = torch.full((1,), 0.25, device="cuda")
    g = sk.ops._C.ce_grad(h, w, y, lse, scale, -100)
    ref_g = torch.softmax(x, 1)
    ref_g[keep, y[keep]] -= 1
    ref_g[~keep] = 0
    torch.testing.assert_close(g.float(), ref_g * 0.25, atol=1e-5, rtol=1e-2)


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_ce_logits_is_the_bf16_gemm(sk, shape):
    h, w, _ = _inputs(*shape, seed=7)
    x = h.float() @ w.float().t()
    torch.testing.assert_close(sk.ops._C.ce_logits(h, w).float(), x, atol=3e-2, rtol=1e-2)


def test_ce_gemm_is_deterministic(sk):
    # T = 2000 rows on V = 4096: a partial last wave, so the Stream-K fixup runs
    h, w, y = _inputs(2000, 1024, 4096, seed=8)
    assert torch.equal(sk.ops._C.ce_logits(h, w), sk.ops._C.ce_logits(h, w))
    a = sk.ops._C.ce_stats(h, w, y, -100)
    b = sk.ops._C.ce_stats(h, w, y, -100)
    assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


def test_ce_transpose_pads_with_zeros(sk):
    x = torch.randn(100, 640, device="cuda").to(torch.bfloat16)
    out = torch.full((200, 128), 7.0, device="cuda", dtype=torch.bfloat16)
    sk.ops._C.ce_transpose(x, 300, 500, 128, out)
    assert torch.equal(out[:, :100], x[:, 300:500].t())
    assert (out[:, 100:] == 0).all()
