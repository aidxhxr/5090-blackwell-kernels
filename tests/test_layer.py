"""One Llama-3-8B decoder layer on the kernels against the same layer in plain PyTorch: the
two must agree to bf16 tolerance on prefill (S in {1, 7, 128, 1000}) and decode (cache
lengths {1, 500, 4096}), each at B in {1, 4}. Both run the fp32 math of the kernels on
bf16 tensors with different rounding points (our GEMMs against cuBLAS, the fused norms, the
attention's bf16 P), so the check is against an fp32 run of the torch layer on the same
weights: ours has to land as close to it as bf16 torch does."""

import pytest
import torch

# the layer module needs the built extension; without it these tests are skipped, as the
# `sk` fixture's tests are on a machine without a GPU
L = pytest.importorskip("spark_kernels.layer")

PREFILL = [(1, 1), (1, 7), (1, 128), (1, 1000), (4, 1), (4, 7), (4, 128), (4, 1000)]
DECODE = [(1, 1), (1, 500), (1, 4096), (4, 1), (4, 500), (4, 4096)]


@pytest.fixture(scope="module")
def weights(sk):
    return L.LayerWeights.random(seed=0)


@pytest.fixture(scope="module")
def rope(sk):
    return L.RoPE(max_pos=4096 + 8)


def _rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _check(x_ours, d_ours, x_torch, d_torch, x_f32, d_f32):
    """ours and bf16 torch against the fp32 run: same shapes, finite, ours within 1.5x of
    torch's own bf16 error (plus a floor for the tiny cases) and the two bf16 results close
    to each other; the finished residual stream and the pending delta both."""
    for ours, theirs, truth in ((x_ours, x_torch, x_f32), (d_ours, d_torch, d_f32)):
        assert ours.shape == theirs.shape and ours.dtype == torch.bfloat16
        assert torch.isfinite(ours.float()).all()
        e_ours, e_torch = _rel_err(ours, truth), _rel_err(theirs, truth)
        assert e_ours <= 1.5 * e_torch + 2e-3, (e_ours, e_torch)
        assert e_torch < 2e-2  # the torch bf16 layer itself is where it should be
        torch.testing.assert_close(ours.float(), theirs.float(), atol=6e-2, rtol=3e-2)


def _run(fn_name, ours, theirs, truth, x, delta, caches):
    """Run the three layers on copies of x; caches is (ours, torch bf16, torch fp32)."""
    outs = []
    for lyr, cache, dt in ((ours, caches[0], torch.bfloat16), (theirs, caches[1], torch.bfloat16),
                           (truth, caches[2], torch.float32)):
        xi = x.to(dt).clone()
        di = None if delta is None else delta.to(dt).clone()
        outs.append(getattr(lyr, fn_name)(xi, cache, di))
    return outs


def _layers(weights, rope):
    return (L.SparkLayer(weights, rope), L.TorchLayer(weights, rope),
            L.TorchLayer(weights.to_float(), rope))


@pytest.mark.parametrize("delta", [False, True], ids=["first_layer", "with_delta"])
@pytest.mark.parametrize("b,s", PREFILL, ids=[f"b{b}_s{s}" for b, s in PREFILL])
def test_prefill_matches_torch(sk, weights, rope, b, s, delta):
    torch.manual_seed(b * 1000 + s)
    x = torch.randn(b, s, L.HIDDEN, device="cuda", dtype=torch.bfloat16)
    d = torch.randn_like(x) * 0.5 if delta else None
    caches = [L.KVCache(b, s, dtype=dt) for dt in (torch.bfloat16, torch.bfloat16, torch.float32)]
    ours, theirs, truth = _layers(weights, rope)
    (xo, do), (xt, dt), (xf, df) = _run("prefill", ours, theirs, truth, x, d, caches)
    _check(xo, do, xt, dt, xf, df)
    assert caches[0].length == s
    torch.testing.assert_close(caches[0].k.float(), caches[1].k.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(caches[0].v.float(), caches[1].v.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(L.finish(xo, do).float(), L.finish(xt, dt).float(),
                               atol=6e-2, rtol=3e-2)


@pytest.mark.parametrize("full", [True, False], ids=["cache_full", "cache_copied"])
@pytest.mark.parametrize("b,n", DECODE, ids=[f"b{b}_L{n}" for b, n in DECODE])
def test_decode_matches_torch(sk, weights, rope, b, n, full):
    # a cache holding n - 1 random tokens; the step appends the n-th and attends over n.
    # With capacity n the kernel reads the cache in place; with more capacity it reads a
    # contiguous copy of the filled part (KVCache.kv), which must give the same numbers.
    torch.manual_seed(b * 1000 + n)
    cap = n if full else n + 37
    caches = [L.KVCache(b, cap, dtype=dt) for dt in (torch.bfloat16, torch.bfloat16, torch.float32)]
    k0 = torch.randn(b, L.N_KV_HEADS, n - 1, L.HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v0 = torch.randn_like(k0)
    for c in caches:
        c.k[:, :, :n - 1] = k0.to(c.k.dtype)
        c.v[:, :, :n - 1] = v0.to(c.v.dtype)
        c.length = n - 1
    x = torch.randn(b, 1, L.HIDDEN, device="cuda", dtype=torch.bfloat16)
    d = torch.randn_like(x) * 0.5
    ours, theirs, truth = _layers(weights, rope)
    (xo, do), (xt, dt), (xf, df) = _run("decode", ours, theirs, truth, x, d, caches)
    _check(xo, do, xt, dt, xf, df)
    assert caches[0].length == n
    _, _, copied = caches[0].kv()
    assert copied == (not full)
    torch.testing.assert_close(caches[0].k[:, :, n - 1].float(), caches[1].k[:, :, n - 1].float(),
                               atol=2e-2, rtol=2e-2)


def test_decode_follows_prefill(sk, weights, rope):
    # prefill 300 tokens, then decode two more, against torch doing the same
    torch.manual_seed(7)
    b, s = 2, 300
    x = torch.randn(b, s + 2, L.HIDDEN, device="cuda", dtype=torch.bfloat16)
    ours, theirs = L.SparkLayer(weights, rope), L.TorchLayer(weights, rope)
    co, ct = L.KVCache(b, s + 2), L.KVCache(b, s + 2)
    xo, do = ours.prefill(x[:, :s].clone(), co)
    xt, dt = theirs.prefill(x[:, :s].clone(), ct)
    for t in range(s, s + 2):
        xo, do = ours.decode(x[:, t:t + 1].clone(), co, do[:, -1:])
        xt, dt = theirs.decode(x[:, t:t + 1].clone(), ct, dt[:, -1:])
    assert co.length == ct.length == s + 2
    torch.testing.assert_close(L.finish(xo, do).float(), L.finish(xt, dt).float(),
                               atol=6e-2, rtol=3e-2)


def test_layer_rejects_misuse(sk, weights, rope):
    ours = L.SparkLayer(weights, rope)
    cache = L.KVCache(1, 8)
    x = torch.randn(1, 4, L.HIDDEN, device="cuda", dtype=torch.bfloat16)
    ours.prefill(x, cache)
    with pytest.raises(ValueError):
        ours.prefill(x, cache)  # not empty any more
    with pytest.raises(ValueError):
        ours.decode(x, cache)  # more than one token
    for _ in range(4):
        ours.decode(x[:, :1], cache)
    with pytest.raises(ValueError):
        ours.decode(x[:, :1], cache)  # full


def test_weights_are_one_layer_of_llama_3_8b(sk, weights):
    # 2 x (4096 x 6144 + 4096 x 4096 + 4096 x 28672 + 14336 x 4096) bytes plus two norms
    assert weights.bytes() == 2 * (4096 * 6144 + 4096 * 4096 + 4096 * 28672 + 14336 * 4096
                                   + 2 * 4096)
    assert weights.bytes() > 96 << 20  # bigger than the L2 on its own: no rotation needed


@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
@pytest.mark.parametrize("b,n", [(1, 500), (4, 4096)])
def test_int4_decode_matches_torch_on_the_dequantized_weights(sk, weights, rope, b, n, asym):
    # SparkLayer(int4=True) against the torch layer on the bf16 weights the int4 ones stand
    # for (LayerWeights.w4_dequantized): the same check as the bf16 decode, with the int4
    # GEMMs where hgemm was.
    torch.manual_seed(b * 1000 + n)
    deq = weights.w4_dequantized(asym)
    caches = [L.KVCache(b, n, dtype=dt) for dt in (torch.bfloat16, torch.bfloat16, torch.float32)]
    k0 = torch.randn(b, L.N_KV_HEADS, n - 1, L.HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    v0 = torch.randn_like(k0)
    for c in caches:
        c.k[:, :, :n - 1] = k0.to(c.k.dtype)
        c.v[:, :, :n - 1] = v0.to(c.v.dtype)
        c.length = n - 1
    x = torch.randn(b, 1, L.HIDDEN, device="cuda", dtype=torch.bfloat16)
    d = torch.randn_like(x) * 0.5
    ours = L.SparkLayer(weights, rope, int4=True, asym=asym)
    theirs, truth = L.TorchLayer(deq, rope), L.TorchLayer(deq.to_float(), rope)
    (xo, do), (xt, dt), (xf, df) = _run("decode", ours, theirs, truth, x, d, caches)
    _check(xo, do, xt, dt, xf, df)
    # packed int4 + one bf16 scale (and one zero byte) per 128 weights
    per_weight = 0.5 + (3 if asym else 2) / 128
    assert ours.weight_bytes() == int(per_weight * (4096 * 6144 + 4096 * 4096 + 4096 * 28672
                                                    + 14336 * 4096))
