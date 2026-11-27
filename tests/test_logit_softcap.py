"""The final logit softcap (Gemma 2, Inkling 2 Mini: x = c tanh(z / c), c = 30) in
linear_cross_entropy and in sample, and vocabularies that are not a multiple of 128 (Inkling
2's unpadded lm_head has 200058 rows). The loss and both gradients of every variant against
torch autograd in fp32 on HF's spelling (`logits = lm_head(h).float(); logits =
softcap(logits)`, `Inkling2LoopForCausalLM.forward`), with ignored rows, at H = 2304, V =
200058 and at small odd V; the pieces on their own at an odd V (the row kernel on a padded
row, the STATS and GRAD epilogues, the band kernel over two bands with a short last strip);
and the sampler's draws with a fixed seed against the capped softmax."""

import math

import pytest
import torch
import torch.nn.functional as F

INKLING_V, INKLING_H, INKLING_CAP = 200058, 2304, 30.0


def _inputs(T, D, V, std=3.0, seed=0, ignore_frac=0.15, ignore_index=-100):
    g = torch.Generator(device="cuda").manual_seed(seed)
    h = torch.randn(T, D, device="cuda", generator=g).to(torch.bfloat16)
    # raw logits with a standard deviation of about `std`
    w = (torch.randn(V, D, device="cuda", generator=g) * (std / D**0.5)).to(torch.bfloat16)
    y = torch.randint(0, V, (T,), device="cuda", generator=g)
    y[torch.rand(T, device="cuda", generator=g) < ignore_frac] = ignore_index
    return h, w, y


def _softcap(x, c):
    return x if c is None else torch.tanh(x / c) * c


def _reference(h, w, y, softcap, reduction="mean"):
    """(loss, dh, dw) by torch autograd in fp32 on the bf16 inputs, HF's order: the product,
    the cap, the loss."""
    hf = h.detach().float().requires_grad_()
    wf = w.detach().float().requires_grad_()
    loss = F.cross_entropy(_softcap(hf @ wf.t(), softcap), y, reduction=reduction)
    loss.backward()
    return loss.detach(), hf.grad, wf.grad


def _assert_close(got, ref, name, rel=2e-2, floor=1e-6):
    # max |got - ref| <= rel * max |ref| + floor, the bound test_cross_entropy.py uses
    err = (got.float() - ref).abs().max().item()
    tol = rel * ref.abs().max().item() + floor
    assert err <= tol, f"{name}: max error {err:.3e} > tolerance {tol:.3e}"


def _run(sk, h, w, y, variant, chunk, softcap, reduction="mean"):
    hh = h.clone().requires_grad_()
    ww = w.clone().requires_grad_()
    loss = sk.linear_cross_entropy(hh, ww, y, reduction=reduction, variant=variant, chunk=chunk,
                                   softcap=softcap)
    loss.backward()
    assert hh.grad.dtype == torch.bfloat16 and ww.grad.shape == w.shape
    return loss.detach(), hh.grad, ww.grad


# ---- the whole op ------------------------------------------------------------------------


@pytest.mark.parametrize("variant", [-1, 1, 2, 3])
def test_inkling_shape_with_softcap(sk, variant):
    """H = 2304, V = 200058 (1563 tiles of 128, the last with 122 columns), 300 rows in
    chunks of 128 (the last ragged), raw logits with a standard deviation of 12 so the cap
    bends most of them."""
    h, w, y = _inputs(300, INKLING_H, INKLING_V, std=12.0, seed=1)
    ref_loss, ref_dh, ref_dw = _reference(h, w, y, INKLING_CAP)
    loss, dh, dw = _run(sk, h, w, y, variant, 128, INKLING_CAP)
    torch.testing.assert_close(loss, ref_loss, atol=2e-3, rtol=2e-3)
    _assert_close(dh, ref_dh, "dh")
    _assert_close(dw, ref_dw, "dw")


def test_inkling_shape_hf_spelling_and_no_cap(sk):
    """Variant 0 is HF's spelling, bf16 logits then the cap in fp32; variant 1 without a cap
    at the unpadded vocabulary is the plain loss (on logits of standard deviation 3, as in
    test_cross_entropy.py: uncapped logits of 12 make variant 1's bf16 logits, and HF's, too
    coarse for a 2% gradient bound)."""
    h, w, y = _inputs(256, INKLING_H, INKLING_V, std=12.0, seed=2)
    hf = _softcap((h @ w.t()).float(), INKLING_CAP)
    hf_loss = F.cross_entropy(hf, y)
    torch.testing.assert_close(sk.linear_cross_entropy(h, w, y, variant=0, softcap=INKLING_CAP),
                               hf_loss)
    with torch.no_grad():
        got = sk.linear_cross_entropy(h, w, y, softcap=INKLING_CAP)  # -1: variant 2
    torch.testing.assert_close(got, hf_loss, atol=5e-3, rtol=2e-3)
    h, w, y = _inputs(256, INKLING_H, INKLING_V, std=3.0, seed=2)
    ref_loss, ref_dh, ref_dw = _reference(h, w, y, None)
    loss, dh, dw = _run(sk, h, w, y, 1, 2048, None)
    torch.testing.assert_close(loss, ref_loss, atol=2e-3, rtol=2e-3)
    _assert_close(dh, ref_dh, "dh")
    _assert_close(dw, ref_dw, "dw")


# (T, D, V): an odd V whose last tile has a fully masked warp column, a V one past a tile,
# an even V that is not a multiple of 128
SHAPES = [(200, 128, 333), (77, 64, 129), (150, 256, 1000)]


@pytest.mark.parametrize("softcap", [None, 5.0], ids=lambda c: f"cap{c}")
@pytest.mark.parametrize("variant", [1, 2, 3])
@pytest.mark.parametrize("chunk", [64, 4096], ids=lambda c: f"chunk{c}")
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "t{}_d{}_v{}".format(*s))
def test_odd_vocabulary(sk, shape, chunk, variant, softcap):
    h, w, y = _inputs(*shape, seed=3)
    ref_loss, ref_dh, ref_dw = _reference(h, w, y, softcap)
    loss, dh, dw = _run(sk, h, w, y, variant, chunk, softcap)
    torch.testing.assert_close(loss, ref_loss, atol=2e-3, rtol=2e-3)
    _assert_close(dh, ref_dh, "dh")
    _assert_close(dw, ref_dw, "dw")


@pytest.mark.parametrize("variant", [1, 2, 3])
def test_softcap_sum_and_upstream(sk, variant):
    h, w, y = _inputs(500, 256, 1024, seed=4)
    hf = h.float().requires_grad_()
    wf = w.float().requires_grad_()
    ref = F.cross_entropy(_softcap(hf @ wf.t(), 5.0), y, reduction="sum")
    (ref * 0.37).backward()
    hh = h.clone().requires_grad_()
    ww = w.clone().requires_grad_()
    loss = sk.linear_cross_entropy(hh, ww, y, reduction="sum", variant=variant, chunk=128,
                                   softcap=5.0)
    (loss * 0.37).backward()
    torch.testing.assert_close(loss, ref.detach(), atol=1e-2, rtol=2e-3)
    _assert_close(hh.grad, hf.grad, "dh")
    _assert_close(ww.grad, wf.grad, "dw")


def test_softcap_rejects_bad_values(sk):
    h, w, y = _inputs(64, 64, 256)
    for bad in (0.0, -30.0):
        with pytest.raises(ValueError):
            sk.linear_cross_entropy(h, w, y, softcap=bad)


# ---- the pieces at an odd V --------------------------------------------------------------


@pytest.mark.parametrize("softcap", [None, 30.0], ids=lambda c: f"cap{c}")
def test_ce_rows_on_padded_rows(sk, softcap):
    rows, V = 40, 1001
    ld = 1024
    torch.manual_seed(5)
    z = torch.randn(rows, V, device="cuda") * 20
    logits = torch.full((rows, ld), 1e4, device="cuda").to(torch.bfloat16)  # junk past V
    logits[:, :V] = z.to(torch.bfloat16)
    y = torch.randint(0, V, (rows,), device="cuda")
    y[::5] = -100
    lf = logits[:, :V].float().requires_grad_()
    ref = F.cross_entropy(_softcap(lf, softcap), y, reduction="none")
    ref.sum().backward()
    scale = torch.full((1,), 0.5, device="cuda")
    loss, lse = sk.ops._C.ce_rows(logits, y, -100, scale, softcap or 0.0, V)
    torch.testing.assert_close(loss, ref.detach(), atol=1e-4, rtol=1e-4)
    keep = y != -100
    ref_lse = torch.logsumexp(_softcap(lf.detach(), softcap), 1)
    torch.testing.assert_close(lse[keep], ref_lse[keep], atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(logits[:, :V].float(), lf.grad * 0.5, atol=1e-6, rtol=1e-2)
    assert (logits[:, V:] == 0).all()


@pytest.mark.parametrize("softcap", [None, 5.0], ids=lambda c: f"cap{c}")
def test_ce_gemm_epilogues_at_odd_v(sk, softcap):
    T, D, V = 200, 128, 333
    Vp = 384
    h, w, y = _inputs(T, D, V, seed=6)
    C_ = sk.ops._C
    z = h.float() @ w.float().t()
    lg = C_.ce_logits(h, w)
    assert lg.shape == (T, Vp) and (lg[:, V:] == 0).all()
    torch.testing.assert_close(lg[:, :V].float(), z, atol=3e-2, rtol=1e-2)
    x = _softcap(z, softcap)
    cap = softcap or 0.0
    lse, loss = C_.ce_stats(h, w, y, -100, cap)
    keep = y != -100
    torch.testing.assert_close(lse, torch.logsumexp(x, 1), atol=1e-3, rtol=1e-4)
    torch.testing.assert_close(loss, F.cross_entropy(x, y, reduction="none"), atol=1e-3,
                               rtol=1e-4)
    assert (loss[~keep] == 0).all()
    scale = torch.full((1,), 0.25, device="cuda")
    g = C_.ce_grad(h, w, y, lse, scale, -100, None, cap)
    zf = z.clone().requires_grad_()
    F.cross_entropy(_softcap(zf, softcap), y, reduction="sum").backward()
    assert g.shape == (T, Vp) and (g[:, V:] == 0).all()
    torch.testing.assert_close(g[:, :V].float(), zf.grad * 0.25, atol=1e-5, rtol=1e-2)


def test_ce_backward_bands_at_odd_v(sk):
    """The band kernel on its own at V = 4173: 33 vocabulary strips over 16 slots, the last
    with 77 rows, so its fold is short and its slot rows past V only ever get zeros."""
    T, D, V = 300, 256, 4173
    h, w, y = _inputs(T, D, V, seed=7)
    _, ref_dh, ref_dw = _reference(h, w, y, 5.0, reduction="sum")
    C_ = sk.ops._C
    lse, _ = C_.ce_stats(h, w, y, -100, 5.0)
    scale = torch.ones(1, device="cuda")
    f32 = dict(device="cuda", dtype=torch.float32)
    strips = (V + 127) // 128
    dh = torch.empty(T, D, device="cuda", dtype=torch.bfloat16)
    dw = torch.empty_like(w)
    slots = torch.zeros(C_.ce_backward_slots() * 128, D, **f32)
    flags = torch.zeros(2 * strips + 1, device="cuda", dtype=torch.int32)
    for i, (t0, t1) in enumerate(((0, 160), (160, T))):
        dh_f32 = torch.zeros(256, D, **f32)
        flags.zero_()
        C_.ce_backward(h[t0:t1], w, y[t0:t1], lse[t0:t1], scale, -100, dh_f32, slots, flags,
                       dw, i == 0, 5.0)
        dh[t0:t1] = dh_f32[: t1 - t0].to(torch.bfloat16)
    assert (slots == 0).all()
    assert (flags[strips: 2 * strips] == 1).all()
    _assert_close(dh, ref_dh, "dh")
    _assert_close(dw, ref_dw, "dw")


# ---- sampling ----------------------------------------------------------------------------


def _params(B, T=1.0, k=0, p=1.0, seed0=0, offset=0):
    dev = "cuda"
    return (torch.full((B,), T, device=dev, dtype=torch.float32),
            torch.full((B,), k, device=dev, dtype=torch.int32),
            torch.full((B,), p, device=dev, dtype=torch.float32),
            torch.arange(seed0, seed0 + B, device=dev, dtype=torch.long),
            torch.full((B,), offset, device=dev, dtype=torch.long))


def _logits(B, V, scale, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(B, V, device="cuda", generator=g) * scale).to(torch.bfloat16)


def _capped_probs(row, T, k, p, softcap):
    """sample's rule in float64 on the capped logits: softmax(cap(x) / T) on the tokens at or
    above the k-th largest value, then on those whose mass strictly above them is < p of what
    top-k kept."""
    x = _softcap(row.double(), softcap)
    w = torch.exp((x - x.max()) / T)
    keep = torch.ones_like(x, dtype=torch.bool)
    if 0 < k < x.numel():
        keep &= x >= torch.sort(x, descending=True).values[k - 1]
    if p < 1:
        z = w[keep].sum()
        before = torch.stack([w[keep & (x > xi)].sum() for xi in x])
        keep &= before < p * z
    q = torch.where(keep, w, torch.zeros_like(w))
    return q / q.sum()


def _variants(sk):
    return list(range(sk.num_variants("sample")))


def test_sample_softcap_keeps_greedy_and_top_k(sk):
    """The cap is monotonic: temperature 0 and top_k = 1 are the raw argmax, and with top_k
    = 50 every token drawn is among the raw top 50."""
    B = 8
    x = _logits(B, INKLING_V, scale=25.0, seed=8)
    cols = torch.randint(0, INKLING_V, (B,), device="cuda")
    x[torch.arange(B, device="cuda"), cols] = x.max() + 1  # a unique max per row
    for variant in _variants(sk):
        tok = sk.sample(x, *_params(B, T=0.0), variant=variant, softcap=INKLING_CAP)
        assert torch.equal(tok, torch.argmax(x, dim=-1))
        tok = sk.sample(x, *_params(B, T=1.0, k=1), variant=variant, softcap=INKLING_CAP)
        assert torch.equal(tok, torch.argmax(x, dim=-1))
    N = 64
    rows = x.repeat_interleave(N, dim=0)
    tok = sk.sample(rows, *_params(B * N, T=2.0, k=50), softcap=INKLING_CAP).view(B, N)
    kth = torch.sort(x.float(), dim=-1, descending=True).values[:, 49:50]
    assert (x.float().gather(1, tok) >= kth).all()


def test_sample_softcap_variants_agree(sk):
    if sk.num_variants("sample") < 2:
        pytest.skip("one variant")
    B = 16
    x = _logits(B, INKLING_V, scale=20.0, seed=9)
    dev = "cuda"
    T = torch.tensor([0.0, 1.0, 0.7, 3.0] * 4, device=dev)
    k = torch.tensor([0, 0, 50, 1, 0, 10, 0, 2000, 5, 0, 0, 40, 1, 0, 300, 0], device=dev,
                     dtype=torch.int32)
    p = torch.tensor([1.0, 0.9, 1.0, 1.0, 0.5, 0.95, 0.1, 0.8] * 2, device=dev)
    seed = torch.arange(200, 200 + B, device=dev, dtype=torch.long)
    off = torch.zeros(B, device=dev, dtype=torch.long)
    a = sk.sample(x, T, k, p, seed, off.clone(), 0, softcap=INKLING_CAP)
    b = sk.sample(x, T, k, p, seed, off.clone(), 1, softcap=INKLING_CAP)
    assert torch.equal(a, b)


CASES = [(1.0, 0, 1.0), (2.0, 10, 1.0), (1.0, 0, 0.8), (3.0, 20, 0.9), (4.0, 20, 0.7)]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"T{c[0]}-k{c[1]}-p{c[2]}")
def test_sample_softcap_distribution(sk, case):
    """20000 draws of one row of raw logits spread over about +-60 (one seed per draw, fixed):
    nothing outside the capped rule's kept set comes out and the counts pass the same loose
    chi-square test test_sample.py uses against the capped probabilities. The raw softmax
    is far from them, so the test sees the cap."""
    T, k, p = case
    N, V = 20000, 64
    row = _logits(1, V, scale=20.0, seed=10)
    q = _capped_probs(row[0].float().cpu(), T, k, p, INKLING_CAP)
    raw = _capped_probs(row[0].float().cpu(), T, k, p, None)
    assert (q - raw).abs().sum() > 0.2  # the cap moves a lot of mass on this row
    x = row.expand(N, V).contiguous()
    for variant in _variants(sk):
        tok = sk.sample(x, *_params(N, T=T, k=k, p=p, seed0=11), variant=variant,
                        softcap=INKLING_CAP)
        counts = torch.bincount(tok, minlength=V).double().cpu()
        assert counts[q == 0].sum() == 0, "a token outside the capped top-k / top-p set"
        exp = q * N
        big = exp >= 5
        obs = torch.cat([counts[big], counts[~big & (q > 0)].sum()[None]])
        want = torch.cat([exp[big], exp[~big & (q > 0)].sum()[None]])
        if want[-1] == 0:
            obs, want = obs[:-1], want[:-1]
        df = max(int(want.numel()) - 1, 1)
        chi2 = float(((obs - want) ** 2 / want).sum())
        assert chi2 < df + 6 * math.sqrt(2 * df) + 5, (variant, chi2, df)


def test_sample_softcap_rejects_bad_values(sk):
    x = _logits(2, 64, scale=1.0, seed=0)
    with pytest.raises(ValueError):
        sk.sample(x, *_params(2), softcap=0.0)
    with pytest.raises(ValueError):
        sk.sample(x, *_params(2), softcap=-1.0)
