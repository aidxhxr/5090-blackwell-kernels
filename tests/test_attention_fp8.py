"""Parity of the fp8 attention ladder (sk.attention_fp8) against torch's
scaled_dot_product_attention in fp32 on the dequantized e4m3 inputs, on the shapes of
test_attention.py: both head sizes, both masks, lengths that are not a multiple of the tile,
short queries, GQA groups of 4 and 8, the split-KV tail and the persistent queue of variant 1,
and per-head scales.

Tolerance. The reference sees exactly the kernel's inputs (the e4m3 values times their
scales), so what is left is the kernel's own rounding: variant 1 rounds every probability to
e4m3 (up to 2^-4 relative, 2^-5 on average) before the P V product, and both variants round
the output to bf16. On a row whose softmax is spread over many keys those errors average out;
on a row dominated by a few keys (the first rows under the causal mask) they do not, and the
error is a few percent of |v|. So the check is max|O - O_ref| <= 0.04 max|O_ref| + 4e-3, the
bench's bound, plus a mean error under 3% of mean|O| (measured: 1.5 to 2% on these short
sequences), which a systematic error (a wrong scale, a key paired with the wrong value row)
would break by an order of magnitude."""

import pytest
import torch
import torch.nn.functional as F

# (B, H_q, H_kv, S_q, S_kv, D)
SHAPES = [(1, 2, 2, 128, 128, 64), (1, 2, 2, 200, 200, 128), (2, 3, 3, 512, 512, 64),
          (1, 1, 1, 1000, 1000, 128), (1, 4, 4, 1, 512, 128), (1, 3, 3, 64, 1000, 128),
          (1, 2, 2, 7, 300, 64), (1, 8, 2, 300, 300, 128), (2, 8, 1, 200, 200, 64),
          (1, 32, 8, 1, 4096, 128), (1, 16, 2, 2, 1000, 128), (2, 16, 2, 5, 333, 128)]


def _variants():
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants("attention_fp8")))
    except Exception:
        return [0]


def _inputs(sk, B, Hq, Hkv, Sq, Skv, D, per_head=False, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, Hq, Sq, D, device="cuda")
    k = torch.randn(B, Hkv, Skv, D, device="cuda")
    v = torch.randn(B, Hkv, Skv, D, device="cuda")
    return [sk.quantize_fp8(x, per_head=per_head) for x in (q, k, v)]


def _dequant(x8, scale):
    s = scale.reshape(x8.shape[0], x8.shape[1], 1, 1) if scale.numel() > 1 else scale
    return x8.float() * s


def _reference(qs, ks, vs, causal):
    q, k, v = (_dequant(*t) for t in (qs, ks, vs))
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                          enable_gqa=k.shape[1] != q.shape[1])


def _run(sk, qs, ks, vs, causal, variant=-1):
    return sk.attention_fp8(qs[0], ks[0], vs[0], qs[1], ks[1], vs[1], causal=causal,
                            variant=variant)


def _assert_close_fp8(got, ref):
    err = (got.float() - ref).abs()
    bound = 0.04 * ref.abs().max().item() + 4e-3
    assert err.max().item() <= bound, f"max |error| {err.max().item():.3e} > {bound:.3e}"
    assert err.mean().item() <= 0.03 * ref.abs().mean().item() + 1e-3, (
        f"mean |error| {err.mean().item():.3e} against mean |O| {ref.abs().mean().item():.3e}")


def _shape_id(s):
    return "b{}_hq{}_hkv{}_sq{}_skv{}_d{}".format(*s)


def test_quantize_fp8_scales_and_round_trip(sk):
    torch.manual_seed(0)
    x = torch.randn(2, 4, 100, 64, device="cuda") * torch.arange(1, 5, device="cuda").view(
        1, 4, 1, 1)
    x8, s = sk.quantize_fp8(x)
    assert x8.dtype == torch.float8_e4m3fn and x8.shape == x.shape and s.shape == (1,)
    torch.testing.assert_close(s, x.abs().amax().reshape(1) / 448)
    assert x8.float().abs().max().item() == 448.0
    # e4m3 keeps 3 mantissa bits: the round trip is within 2^-4 relative of each normal value
    back = _dequant(x8, s)
    normal = x.abs() >= 2.0**-6 * s
    assert ((back - x).abs() <= 2.0**-4 * x.abs() + 1e-6)[normal].all()
    x8h, sh = sk.quantize_fp8(x, per_head=True)
    assert sh.shape == (2, 4)
    torch.testing.assert_close(sh, x.abs().amax(dim=(2, 3)) / 448)
    # e4m3 is a float format, so a finer scale does not make the normal values more exact;
    # a per-head scale only keeps a small head out of the subnormal range
    normal_h = x.abs() >= 2.0**-6 * sh.view(2, 4, 1, 1)
    assert ((_dequant(x8h, sh) - x).abs() <= 2.0**-4 * x.abs() + 1e-6)[normal_h].all()


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_attention_fp8_matches_sdpa_on_dequantized_inputs(sk, shape, causal, variant):
    qs, ks, vs = _inputs(sk, *shape)
    got = _run(sk, qs, ks, vs, causal, variant)
    assert got.shape == qs[0].shape and got.dtype == torch.bfloat16
    _assert_close_fp8(got, _reference(qs, ks, vs, causal))


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_attention_fp8_per_head_scales(sk, causal, variant):
    # Heads with very different ranges: per-head scales must be read per (b, head) of each
    # tensor, the K/V ones through the GQA head index.
    B, Hq, Hkv, S, D = 2, 8, 2, 300, 128
    torch.manual_seed(1)
    ramp_q = torch.logspace(-2, 1, Hq, device="cuda").view(1, Hq, 1, 1)
    ramp_kv = torch.tensor([0.05, 3.0], device="cuda").view(1, Hkv, 1, 1)
    q = torch.randn(B, Hq, S, D, device="cuda") * ramp_q
    k = torch.randn(B, Hkv, S, D, device="cuda") * ramp_kv
    v = torch.randn(B, Hkv, S, D, device="cuda") * ramp_kv
    qs, ks, vs = (sk.quantize_fp8(x, per_head=True) for x in (q, k, v))
    assert qs[1].shape == (B, Hq) and ks[1].shape == (B, Hkv)
    got = _run(sk, qs, ks, vs, causal, variant)
    ref = _reference(qs, ks, vs, causal)
    for h in range(Hq):  # each head against its own magnitude
        _assert_close_fp8(got[:, h], ref[:, h])


def test_attention_fp8_is_close_to_the_bf16_kernel(sk):
    # The end-to-end question: fp8 inputs against the bf16 kernel on the unquantized values.
    # The input rounding dominates (e4m3 is 2^-4 relative per element), so the bound is looser
    # than the parity one, and it is on the mean, where a real bug would stand out.
    torch.manual_seed(2)
    q, k, v = (torch.randn(1, 8, 1024, 128, device="cuda") for _ in range(3))
    ref = sk.attention(q.bfloat16(), k.bfloat16(), v.bfloat16(), causal=True).float()
    qs, ks, vs = (sk.quantize_fp8(x) for x in (q, k, v))
    got = _run(sk, qs, ks, vs, True).float()
    err = (got - ref).abs()
    assert err.mean().item() <= 0.05 * ref.abs().mean().item()
    assert err.max().item() <= 0.15 * ref.abs().max().item()


def test_attention_fp8_default_is_the_top_rung(sk):
    qs, ks, vs = _inputs(sk, 1, 2, 2, 256, 256, 128)
    top = sk.num_variants("attention_fp8") - 1
    torch.testing.assert_close(_run(sk, qs, ks, vs, True), _run(sk, qs, ks, vs, True, top),
                               atol=0, rtol=0)


TILE_VARIANTS = [v for v in _variants() if v >= 1]


@pytest.mark.skipif(not TILE_VARIANTS, reason="variant 1 not built")
@pytest.mark.parametrize("variant", TILE_VARIANTS)
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_attention_fp8_tma_heads_do_not_bleed(sk, causal, variant):
    # With S_kv = 200 the last 64-row K/V box of every head hangs 56 rows past it; the 3-D
    # tensor map must zero-fill them. Head 0 against a copy whose other heads are hostile.
    qs, ks, vs = _inputs(sk, 1, 3, 3, 200, 200, 128)
    k2, v2 = ks[0].clone(), vs[0].clone()
    k2[:, 1:] = 400.0
    v2[:, 1:] = 400.0
    a = _run(sk, qs, ks, vs, causal, variant)
    b = sk.attention_fp8(qs[0], k2, v2, qs[1], ks[1], vs[1], causal=causal, variant=variant)
    torch.testing.assert_close(a[:, 0], b[:, 0], atol=0, rtol=0)


@pytest.mark.skipif(not TILE_VARIANTS, reason="variant 1 not built")
@pytest.mark.parametrize("variant", TILE_VARIANTS)
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_attention_fp8_queue_split_and_streams(sk, causal, variant):
    # 2 x 8 heads x 16 Q tiles = 256 items on 170 blocks (the non-causal ones with a split
    # tail through the workspace and the combine kernel), the queue counter reset by the last
    # block: repeated launches and one on a second stream must agree bit for bit.
    qs, ks, vs = _inputs(sk, 2, 8, 2, 2048, 2048, 128)
    a = _run(sk, qs, ks, vs, causal, variant)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        c = _run(sk, qs, ks, vs, causal, variant)
    b = _run(sk, qs, ks, vs, causal, variant)
    torch.cuda.synchronize()
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(a, c, atol=0, rtol=0)
    _assert_close_fp8(a, _reference(qs, ks, vs, causal))


def test_attention_fp8_small_grid_split_is_stable(sk):
    # 16 tiles on 170 blocks: every tile is split along the keys, through the workspace
    qs, ks, vs = _inputs(sk, 1, 4, 4, 512, 512, 128)
    a = _run(sk, qs, ks, vs, False)
    torch.testing.assert_close(a, _run(sk, qs, ks, vs, False), atol=0, rtol=0)
    _assert_close_fp8(a, _reference(qs, ks, vs, False))


def test_attention_fp8_rejects_bad_inputs(sk):
    qs, ks, vs = _inputs(sk, 1, 2, 2, 64, 64, 64)
    with pytest.raises(RuntimeError):  # bf16 tensors
        sk.attention_fp8(qs[0].bfloat16(), ks[0].bfloat16(), vs[0].bfloat16(), qs[1], ks[1],
                         vs[1])
    with pytest.raises(RuntimeError):  # per-head q scale next to per-tensor k, v scales
        sk.attention_fp8(qs[0], ks[0], vs[0], qs[1].repeat(2), ks[1], vs[1])
    with pytest.raises(RuntimeError):  # float64 scale
        sk.attention_fp8(qs[0], ks[0], vs[0], qs[1].double(), ks[1], vs[1])
    with pytest.raises(RuntimeError):  # k and v disagree on S_kv
        sk.attention_fp8(qs[0], ks[0][:, :, :32], vs[0], qs[1], ks[1], vs[1])
    q96, k96, v96 = _inputs(sk, 1, 2, 2, 64, 64, 96)
    with pytest.raises(RuntimeError):  # D not in {64, 128}
        sk.attention_fp8(q96[0], k96[0], v96[0], q96[1], k96[1], v96[1])
    with pytest.raises(ValueError):
        sk.quantize_fp8(torch.randn(4, 64, device="cuda"))
