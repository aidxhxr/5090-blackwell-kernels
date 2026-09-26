"""Parity of every attention variant against torch's scaled_dot_product_attention (in fp32,
from the same bf16 inputs), on shapes that cover both head sizes, both masks, a sequence that
is not a multiple of the 128 x 64 tile, a decode step, the split-KV tail of variants 3 and 4,
the TMA boxes of variant 4 that hang off the end of a head, and grouped-query attention
(H_q / H_kv in {1, 4, 8}) in prefill and decode."""

import pytest
import torch
import torch.nn.functional as F

# (B, H_q, H_kv, S_q, S_kv, D)
SHAPES = [(1, 2, 2, 128, 128, 64), (1, 2, 2, 200, 200, 128), (2, 3, 3, 512, 512, 64),
          (1, 1, 1, 1000, 1000, 128), (1, 4, 4, 1, 512, 128), (1, 3, 3, 64, 1000, 128),
          (1, 2, 2, 7, 300, 64),
          # GQA: groups of 4 and 8 in prefill, and decode steps where the whole group's rows
          # fit the 16-row flash-decoding tile (4 heads x 1 token, 8 x 2, 4 x 4) or do not
          # (4 heads x 7 tokens, 8 x 5, which run the 64-row tile with the K/V head stride)
          (1, 8, 2, 300, 300, 128), (2, 8, 1, 200, 200, 64), (1, 32, 8, 1, 4096, 128),
          (1, 16, 2, 2, 1000, 128), (1, 8, 2, 4, 700, 64), (1, 8, 2, 7, 300, 128),
          (2, 16, 2, 5, 333, 128), (3, 8, 2, 1, 96, 128)]
TOL = dict(atol=2e-2, rtol=2e-2)  # one bf16 rounding of an fp32 result, P in bf16 for v2-v4


def _variants():
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants("attention")))
    except Exception:
        return [0]


def _inputs(B, Hq, Hkv, Sq, Skv, D):
    torch.manual_seed(0)
    q = torch.randn(B, Hq, Sq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, Skv, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, Skv, D, device="cuda", dtype=torch.bfloat16)
    return q, k, v


def _reference(q, k, v, causal):
    # enable_gqa (torch >= 2.5) broadcasts the K/V heads over the query heads of each group;
    # with equal head counts it is a no-op.
    return F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=causal,
                                          enable_gqa=k.shape[1] != q.shape[1])


def _shape_id(s):
    return "b{}_hq{}_hkv{}_sq{}_skv{}_d{}".format(*s)


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_attention_matches_sdpa(sk, shape, causal, variant):
    q, k, v = _inputs(*shape)
    got = sk.attention(q, k, v, causal=causal, variant=variant)
    assert got.shape == q.shape and got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), _reference(q, k, v, causal), **TOL)


def test_attention_gqa_matches_repeated_kv_heads(sk):
    # the other spelling of the same thing: K/V heads repeated to the query head count
    q, k, v = _inputs(2, 8, 2, 64, 500, 128)
    k8, v8 = k.repeat_interleave(4, dim=1), v.repeat_interleave(4, dim=1)
    for causal in (False, True):
        torch.testing.assert_close(sk.attention(q, k, v, causal=causal).float(),
                                   sk.attention(q, k8, v8, causal=causal).float(), **TOL)


def test_attention_default_is_the_top_rung(sk):
    q, k, v = _inputs(1, 2, 2, 256, 256, 128)
    top = sk.num_variants("attention") - 1
    torch.testing.assert_close(sk.attention(q, k, v, causal=True),
                               sk.attention(q, k, v, causal=True, variant=top), atol=0, rtol=0)


@pytest.mark.parametrize("variant", [v for v in _variants() if v >= 3])
def test_attention_split_tail_is_stable_across_calls(sk, variant):
    # variants 3 and 4 reuse a workspace for the split tiles; repeated calls must agree
    q, k, v = _inputs(1, 4, 4, 512, 512, 128)
    a = sk.attention(q, k, v, variant=variant)
    b = sk.attention(q, k, v, variant=variant)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(a.float(), _reference(q, k, v, False), **TOL)


def test_attention_decode_split_is_stable_across_calls(sk):
    # the flash-decoding kernel merges its key slices through a workspace and per-head
    # counters that every launch must leave zero; repeated calls and a long cache must agree
    q, k, v = _inputs(1, 32, 8, 1, 20000, 128)
    a = sk.attention(q, k, v, variant=3)
    b = sk.attention(q, k, v, variant=3)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(a.float(), _reference(q, k, v, False), **TOL)


@pytest.mark.skipif(4 not in _variants(), reason="variant 4 not built")
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_attention_v4_heads_do_not_bleed(sk, causal):
    # Variant 4 loads K and V by TMA in 64-row boxes. With S_kv = 200 the last box of every head
    # hangs 56 rows past the head; the 3-D tensor map must zero-fill them, not read the next
    # head's keys. Head 0 is checked alone against a copy of itself with the other heads made
    # hostile (large values), so a bleed would show.
    B, H, S, D = 1, 3, 200, 128
    q, k, v = _inputs(B, H, H, S, S, D)
    k2, v2 = k.clone(), v.clone()
    k2[:, 1:] = 8.0
    v2[:, 1:] = 8.0
    a = sk.attention(q, k, v, causal=causal, variant=4)
    b = sk.attention(q, k2, v2, causal=causal, variant=4)
    torch.testing.assert_close(a[:, 0], b[:, 0], atol=0, rtol=0)
    torch.testing.assert_close(a.float(), _reference(q, k, v, causal), **TOL)


@pytest.mark.skipif(4 not in _variants(), reason="variant 4 not built")
def test_attention_v4_repeated_calls_and_streams(sk):
    # Every launch encodes fresh tensor maps and runs the mbarrier pipeline from scratch; the
    # result must not depend on what ran before, including on another stream.
    q, k, v = _inputs(2, 4, 4, 1000, 1000, 64)
    ref = sk.attention(q, k, v, causal=True, variant=4)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        other = sk.attention(q, k, v, causal=True, variant=4)
    torch.cuda.synchronize()
    for _ in range(3):
        torch.testing.assert_close(sk.attention(q, k, v, causal=True, variant=4), ref,
                                   atol=0, rtol=0)
    torch.testing.assert_close(other, ref, atol=0, rtol=0)
    torch.testing.assert_close(ref.float(), _reference(q, k, v, True), **TOL)


def test_attention_rejects_bad_inputs(sk):
    q, k, v = _inputs(1, 2, 2, 64, 64, 64)
    with pytest.raises(RuntimeError):
        sk.attention(q.float(), k.float(), v.float())  # bf16 only
    with pytest.raises(RuntimeError):
        sk.attention(q, k[:, :, :32], v)  # k and v disagree on S_kv
    q3, k3, v3 = _inputs(1, 3, 3, 64, 64, 64)
    with pytest.raises(RuntimeError):
        sk.attention(q3, k3[:, :2], v3[:, :2])  # H_q not a multiple of H_kv
    with pytest.raises(RuntimeError):
        sk.attention(q, k.repeat(1, 2, 1, 1), v.repeat(1, 2, 1, 1))  # more K/V heads than Q
    q96, k96, v96 = _inputs(1, 2, 2, 64, 64, 96)
    with pytest.raises(RuntimeError):
        sk.attention(q96, k96, v96)  # D not in {64, 128}
    with pytest.raises(RuntimeError):
        sk.attention(q[0], k[0], v[0])  # 3-D
    with pytest.raises(RuntimeError):
        sk.attention(q.transpose(2, 3).contiguous().transpose(2, 3), k, v)  # not contiguous
