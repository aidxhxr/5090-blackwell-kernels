"""Sliding-window attention and attention sinks against an fp32 torch reference that builds
the mask explicitly and puts the sink in the softmax denominator as one more logit with no
value. The dense forward runs every variant on prefill shapes, a chunk against a longer cache,
the short-query tile and decode steps (the flash-decoding kernel), MHA and GQA groups of 4 and
8, both head sizes; the paged decode and the varlen prefill run both variants on ragged
lengths in shuffled pages, prompts appended to a context included. Windows of 1, 64, 128,
1000 and longer than the sequence, each with and without sinks, and sinks with no window."""

import pytest
import torch

TOL = dict(atol=2e-2, rtol=2e-2)  # one bf16 rounding of an fp32 result, P in bf16
WINDOWS = [0, 1, 64, 128, 1000, 100000]  # 0: sinks alone; 100000: longer than any sequence


# The paged ops' variant 2 is the batch-invariant rung (docs/design/batch_invariant.md),
# which takes no window or sinks; the window rungs are the ones below it.
_NO_WINDOW = {"paged_decode": {2}, "attention_varlen": {2}}


def _variants(name):
    try:
        import spark_kernels

        n = spark_kernels.num_variants(name)
    except Exception:
        return [0]
    return [v for v in range(n) if v not in _NO_WINDOW.get(name, set())]


def _sinks(H, seed=11):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.rand(H, generator=g) * 4 - 1).cuda()  # logits in [-1, 3]


def _reference(q, k, v, window, sinks, causal=True):
    """fp32 attention of q [..., H_q, S_q, D] over k, v [..., H_kv, S_kv, D]. A window puts
    query i at key position i + S_kv - S_q and keeps keys pos - window < j <= pos; without one,
    `causal` is torch's top-left mask. `sinks` [H_q] is one more logit per row, no value."""
    Hq, Sq, D = q.shape[-3:]
    Hkv, Skv = k.shape[-3], k.shape[-2]
    kf = k.float().repeat_interleave(Hq // Hkv, dim=-3)
    vf = v.float().repeat_interleave(Hq // Hkv, dim=-3)
    s = q.float() @ kf.transpose(-1, -2) / D**0.5
    i = torch.arange(Sq, device=q.device)[:, None]
    j = torch.arange(Skv, device=q.device)[None, :]
    if window > 0:
        pos = i + Skv - Sq
        s = s.masked_fill(~((j <= pos) & (j > pos - window)), float("-inf"))
    elif causal:
        s = s.masked_fill(j > i, float("-inf"))
    if sinks is None:
        return torch.softmax(s, dim=-1) @ vf
    col = sinks.float().view(Hq, 1, 1).expand(*s.shape[:-1], 1)
    p = torch.softmax(torch.cat([s, col], dim=-1), dim=-1)[..., :-1]
    return p @ vf


def _inputs(B, Hq, Hkv, Sq, Skv, D, seed=0):
    torch.manual_seed(seed)
    q = torch.randn(B, Hq, Sq, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, Skv, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, Skv, D, device="cuda", dtype=torch.bfloat16)
    return q, k, v


# (B, H_q, H_kv, S_q, S_kv, D): prefill (S_q = S_kv) on the 128-row tiles, MHA and GQA, both
# head sizes; a 256-token chunk against 1000 cached keys; a short query on the 64-row tile
# (4 heads x 7 tokens per K/V head); decode steps on the flash-decoding kernel (1 and 2
# tokens); gpt-oss's heads (64 query, 8 K/V, D = 64)
SHAPES = [(1, 4, 4, 300, 300, 128), (1, 8, 2, 1000, 1000, 64), (2, 8, 1, 200, 200, 128),
          (1, 4, 2, 256, 1000, 128), (1, 8, 2, 7, 500, 64), (1, 32, 8, 1, 4096, 128),
          (1, 16, 2, 2, 1000, 128), (1, 64, 8, 600, 600, 64)]


def _shape_id(s):
    return "b{}_hq{}_hkv{}_sq{}_skv{}_d{}".format(*s)


@pytest.mark.parametrize("variant", _variants("attention"))
@pytest.mark.parametrize("sink", [False, True], ids=["nosink", "sink"])
@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: f"w{w}")
@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_attention_window_matches_reference(sk, shape, window, sink, variant):
    if window == 0 and not sink:
        pytest.skip("the plain op, tests/test_attention.py")
    q, k, v = _inputs(*shape)
    sinks = _sinks(shape[1]) if sink else None
    got = sk.attention(q, k, v, causal=True, variant=variant, window=window, sinks=sinks)
    assert got.shape == q.shape and got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), _reference(q, k, v, window, sinks), **TOL)


@pytest.mark.parametrize("variant", _variants("attention"))
@pytest.mark.parametrize("shape", [(1, 4, 4, 300, 300, 128), (1, 32, 8, 1, 4096, 128),
                                   (1, 8, 2, 7, 500, 64)], ids=_shape_id)
def test_attention_sinks_without_a_mask(sk, shape, variant):
    q, k, v = _inputs(*shape)
    sinks = _sinks(shape[1])
    got = sk.attention(q, k, v, causal=False, variant=variant, sinks=sinks)
    torch.testing.assert_close(got.float(), _reference(q, k, v, 0, sinks, causal=False), **TOL)


@pytest.mark.parametrize("variant", _variants("attention"))
def test_attention_window_ignores_the_causal_flag(sk, variant):
    # a window is causal and aligned to the last key whatever `causal` says
    q, k, v = _inputs(1, 8, 2, 300, 300, 128)
    a = sk.attention(q, k, v, causal=True, variant=variant, window=100)
    b = sk.attention(q, k, v, causal=False, variant=variant, window=100)
    torch.testing.assert_close(a, b, atol=0, rtol=0)


def test_attention_window_of_one_returns_v(sk):
    # each query sees only its own key: the output is that key's value, rounded once
    q, k, v = _inputs(1, 8, 2, 333, 333, 128)
    got = sk.attention(q, k, v, causal=True, window=1)
    torch.testing.assert_close(got, v.repeat_interleave(4, dim=1), atol=0, rtol=0)


@pytest.mark.parametrize("shape", [(1, 4, 4, 1000, 1000, 128), (1, 32, 8, 1, 4096, 128)],
                         ids=_shape_id)
def test_attention_long_window_is_the_plain_kernel(sk, shape):
    # a window at least S_kv long with S_q = S_kv and no sinks runs the plain causal kernel,
    # bit for bit; a decode step with such a window sees the whole cache
    q, k, v = _inputs(*shape)
    causal = shape[3] == shape[4]
    want = sk.attention(q, k, v, causal=causal)
    for w in (shape[4], shape[4] + 1, 1 << 30):
        torch.testing.assert_close(sk.attention(q, k, v, causal=True, window=w), want, atol=0,
                                   rtol=0)


def test_attention_sink_limits(sk):
    # a sink far below every score changes nothing; one far above takes all the mass
    q, k, v = _inputs(1, 4, 4, 300, 300, 128)
    plain = sk.attention(q, k, v, causal=True, window=64)
    low = torch.full((4,), -1e4, device="cuda")
    torch.testing.assert_close(sk.attention(q, k, v, causal=True, window=64, sinks=low), plain,
                               atol=0, rtol=0)
    high = torch.full((4,), 1e4, device="cuda")
    assert torch.all(sk.attention(q, k, v, causal=True, window=64, sinks=high) == 0)


@pytest.mark.parametrize("variant", [v for v in _variants("attention") if v >= 3])
def test_attention_window_split_and_queue_are_stable(sk, variant):
    # variants 3 to 5 split the tail tiles under a window and merge them with the sink in the
    # combine kernel; variant 5's queue holds 256 items on 170 blocks. Repeated calls and a
    # second stream must agree bit for bit.
    q, k, v = _inputs(2, 8, 2, 2048, 2048, 128)
    sinks = _sinks(8)
    a = sk.attention(q, k, v, causal=True, variant=variant, window=300, sinks=sinks)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        c = sk.attention(q, k, v, causal=True, variant=variant, window=300, sinks=sinks)
    b = sk.attention(q, k, v, causal=True, variant=variant, window=300, sinks=sinks)
    torch.cuda.synchronize()
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    torch.testing.assert_close(a, c, atol=0, rtol=0)
    torch.testing.assert_close(a.float(), _reference(q, k, v, 300, sinks), **TOL)


@pytest.mark.parametrize("window", [128, 4096])
def test_attention_window_decode_long_cache(sk, window):
    # a decode step at 20000 cached keys reads the last `window`; with the split over blocks
    q, k, v = _inputs(1, 32, 8, 1, 20000, 128)
    sinks = _sinks(32)
    got = sk.attention(q, k, v, window=window, sinks=sinks)
    torch.testing.assert_close(got.float(), _reference(q, k, v, window, sinks), **TOL)
    # and only those: the keys before the window can be anything
    k2, v2 = k.clone(), v.clone()
    k2[:, :, :20000 - window - 16] = float("nan")
    v2[:, :, :20000 - window - 16] = float("nan")
    torch.testing.assert_close(sk.attention(q, k2, v2, window=window, sinks=sinks), got, atol=0,
                               rtol=0)


@pytest.mark.parametrize("variant", _variants("attention"))
def test_attention_window_reads_a_cache_slice_in_place(sk, variant):
    q, k, v = _inputs(1, 16, 2, 2, 1000, 128)
    kc = torch.randn(1, 2, 1100, 128, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    kc[:, :, :1000], vc[:, :, :1000] = k, v
    sinks = _sinks(16)
    want = sk.attention(q, k, v, variant=variant, window=300, sinks=sinks)
    got = sk.attention(q, kc[:, :, :1000], vc[:, :, :1000], variant=variant, window=300,
                       sinks=sinks)
    torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_attention_window_rejects_bad_input(sk):
    q, k, v = _inputs(1, 4, 4, 64, 64, 64)
    with pytest.raises(RuntimeError):
        sk.attention(q, k, v, window=-1)
    q2, _, _ = _inputs(1, 4, 4, 128, 64, 64)
    with pytest.raises(RuntimeError):  # more queries than keys: no position for the first
        sk.attention(q2, k, v, window=16)
    with pytest.raises(RuntimeError):  # one sink per query head
        sk.attention(q, k, v, sinks=torch.zeros(3, device="cuda"))
    with pytest.raises(RuntimeError):
        sk.attention(q, k, v, sinks=torch.zeros(4, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(RuntimeError):
        sk.attention(q, k, v, sinks=torch.zeros(4))  # on the CPU


# ---- the paged ops ---------------------------------------------------------------------------


def _paged_cache(lens, H_kv, D, page, seed=0, fp8=False):
    """Random K/V for sequences of `lens` keys in shuffled pages. Returns (k_cache, v_cache,
    block_table, seq_lens, k_list, v_list, scales) with k_list[b] the [H_kv, len_b, D] keys of
    sequence b as the kernels see them (dequantized for an e4m3 cache) and scales (k, v) or
    None."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    npages = [(n + page - 1) // page for n in lens]
    total = sum(npages) + 2
    perm = torch.randperm(total, generator=g).tolist()
    kc = torch.randn(total, H_kv, page, D, generator=g).to("cuda", torch.bfloat16)
    vc = torch.randn(total, H_kv, page, D, generator=g).to("cuda", torch.bfloat16)
    scales = None
    if fp8:
        from spark_kernels.ops import dequantize_kv_cache, quantize_kv_cache

        scales = (torch.tensor([5.0 * (1 + h / 4) / 448 for h in range(H_kv)], device="cuda"),
                  torch.tensor([3.0 * (1 + h / 4) / 448 for h in range(H_kv)], device="cuda"))
        kc, vc = quantize_kv_cache(kc, scales[0]), quantize_kv_cache(vc, scales[1])
        kd, vd = dequantize_kv_cache(kc, scales[0]), dequantize_kv_cache(vc, scales[1])
    else:
        kd, vd = kc, vc
    bt = torch.zeros(len(lens), max(1, max(npages)), dtype=torch.int32)
    k_list, v_list, i = [], [], 0
    for b, n in enumerate(lens):
        ids = perm[i:i + npages[b]]
        i += npages[b]
        bt[b, :len(ids)] = torch.tensor(ids, dtype=torch.int32)
        k_list.append(kd[ids].permute(1, 0, 2, 3).reshape(H_kv, -1, D)[:, :n])
        v_list.append(vd[ids].permute(1, 0, 2, 3).reshape(H_kv, -1, D)[:, :n])
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    return kc, vc, bt.cuda(), seq_lens, k_list, v_list, scales


# (lens, H_q, H_kv, D, page): a long sequence among short ones (segments that cross warps),
# an empty sequence and a single key, GQA groups of 4 and 8, both head sizes, pages of 16 and 32
DECODE_CASES = [([5000, 20, 300, 1, 0, 64], 32, 8, 128, 16), ([700, 1, 129], 4, 4, 64, 16),
                ([3000] + [100] * 20, 64, 8, 64, 32), ([257, 1000, 40], 16, 2, 128, 16)]


def _decode_id(c):
    return f"n{len(c[0])}_max{max(c[0])}_hq{c[1]}_hkv{c[2]}_d{c[3]}_p{c[4]}"


@pytest.mark.parametrize("variant", _variants("paged_decode"))
@pytest.mark.parametrize("sink", [False, True], ids=["nosink", "sink"])
@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: f"w{w}")
@pytest.mark.parametrize("case", DECODE_CASES, ids=_decode_id)
def test_paged_decode_window_matches_reference(sk, case, window, sink, variant):
    if window == 0 and not sink:
        pytest.skip("the plain op, tests/test_paged.py")
    lens, hq, hkv, d, page = case
    kc, vc, bt, sl, ks, vs, _ = _paged_cache(lens, hkv, d, page)
    torch.manual_seed(1)
    q = torch.randn(len(lens), hq, d, device="cuda", dtype=torch.bfloat16)
    sinks = _sinks(hq) if sink else None
    got = sk.paged_decode(q, kc, vc, bt, sl, variant=variant, window=window, sinks=sinks)
    for b, n in enumerate(lens):
        if n == 0:
            assert torch.all(got[b] == 0)
            continue
        # the paged ops align the mask to the last key; window 0 is an unbounded one
        ref = _reference(q[b][:, None], ks[b], vs[b], window or 1 << 30, sinks)[:, 0]
        torch.testing.assert_close(got[b].float(), ref, **TOL)


def test_paged_window_default_is_not_the_invariant_rung(sk):
    # With a window, variant -1 runs the default rung, never variant 2 (batch invariant),
    # which takes no window or sinks and says so rather than ignoring them.
    lens, hq, hkv, d, page = DECODE_CASES[0]
    kc, vc, bt, sl, _, _, _ = _paged_cache(lens, hkv, d, page)
    torch.manual_seed(1)
    q = torch.randn(len(lens), hq, d, device="cuda", dtype=torch.bfloat16)
    sinks = _sinks(hq)
    want = sk.paged_decode(q, kc, vc, bt, sl, variant=1, window=300, sinks=sinks)
    got = sk.paged_decode(q, kc, vc, bt, sl, window=300, sinks=sinks)
    assert torch.equal(got, want)
    if sk.num_variants("paged_decode") > 2:
        with pytest.raises(ValueError):
            sk.paged_decode(q, kc, vc, bt, sl, variant=2, window=300)
        with pytest.raises(ValueError):
            sk.paged_decode(q, kc, vc, bt, sl, variant=2, sinks=sinks)
    tq = torch.randn(sum(lens[:3]), hq, d, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, lens[0], lens[0] + lens[1], sum(lens[:3])], dtype=torch.int32,
                      device="cuda")
    sl3, bt3 = sl[:3].contiguous(), bt[:3].contiguous()
    want = sk.attention_varlen(tq, kc, vc, cu, sl3, bt3, variant=1, window=300, sinks=sinks)
    got = sk.attention_varlen(tq, kc, vc, cu, sl3, bt3, window=300, sinks=sinks)
    assert torch.equal(got, want)
    if sk.num_variants("attention_varlen") > 2:
        with pytest.raises(ValueError):
            sk.attention_varlen(tq, kc, vc, cu, sl3, bt3, variant=2, window=300)


def test_paged_decode_window_long_context_and_fp8(sk):
    # 128K keys with a 4096-key window next to short sequences, bf16 and e4m3, twice each
    lens = [131072, 5000, 17]
    sinks = _sinks(8)
    torch.manual_seed(5)
    q = torch.randn(len(lens), 8, 128, device="cuda", dtype=torch.bfloat16)
    for fp8 in (False, True):
        kc, vc, bt, sl, ks, vs, sc = _paged_cache(lens, 2, 128, 16, seed=4, fp8=fp8)
        kw = dict(k_scale=sc[0], v_scale=sc[1]) if fp8 else {}
        got = sk.paged_decode(q, kc, vc, bt, sl, window=4096, sinks=sinks, **kw)
        torch.testing.assert_close(sk.paged_decode(q, kc, vc, bt, sl, window=4096, sinks=sinks,
                                                   **kw), got, atol=0, rtol=0)
        for b in range(len(lens)):
            ref = _reference(q[b][:, None], ks[b], vs[b], 4096, sinks)[:, 0]
            torch.testing.assert_close(got[b].float(), ref, **TOL)


# (q_lens, ctx_lens, H_q, H_kv, D, page): fresh prompts, prompts appended to a context (the
# bottom-right mask), an empty prompt, GQA, both head sizes
VARLEN_CASES = [([300, 1, 130, 64], [0, 0, 0, 0], 32, 8, 128, 16),
                ([128, 200, 5], [50, 1000, 3], 32, 8, 128, 16),
                ([1000, 17], [0, 0], 8, 2, 64, 32), ([0, 77, 256], [10, 0, 7], 64, 8, 64, 16)]


def _varlen_id(c):
    return f"q{'-'.join(map(str, c[0]))}_ctx{'-'.join(map(str, c[1]))}_hq{c[2]}_hkv{c[3]}_d{c[4]}"


@pytest.mark.parametrize("variant", _variants("attention_varlen"))
@pytest.mark.parametrize("fp8", [False, True], ids=["bf16", "fp8"])
@pytest.mark.parametrize("sink", [False, True], ids=["nosink", "sink"])
@pytest.mark.parametrize("window", WINDOWS, ids=lambda w: f"w{w}")
@pytest.mark.parametrize("case", VARLEN_CASES, ids=_varlen_id)
def test_attention_varlen_window_matches_reference(sk, case, window, sink, fp8, variant):
    if window == 0 and not sink:
        pytest.skip("the plain op, tests/test_paged.py")
    q_lens, ctx, hq, hkv, d, page = case
    lens = [a + b for a, b in zip(q_lens, ctx, strict=True)]
    kc, vc, bt, sl, ks, vs, sc = _paged_cache(lens, hkv, d, page, seed=2, fp8=fp8)
    kw = dict(k_scale=sc[0], v_scale=sc[1]) if fp8 else {}
    T = sum(q_lens)
    torch.manual_seed(3)
    q = torch.randn(T, hq, d, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32,
                      device="cuda")
    sinks = _sinks(hq) if sink else None
    got = sk.attention_varlen(q, kc, vc, cu, sl, bt, causal=True, variant=variant,
                              window=window, sinks=sinks, **kw)
    t0 = 0
    for b, n in enumerate(q_lens):
        if n == 0:
            continue
        ref = _reference(q[t0:t0 + n].transpose(0, 1), ks[b], vs[b], window or 1 << 30, sinks)
        torch.testing.assert_close(got[t0:t0 + n].float(), ref.transpose(0, 1), **TOL)
        t0 += n


def test_paged_window_rejects_bad_input(sk):
    kc = torch.zeros(4, 8, 16, 128, device="cuda", dtype=torch.bfloat16)
    q = torch.zeros(1, 32, 128, device="cuda", dtype=torch.bfloat16)
    bt = torch.zeros(1, 4, dtype=torch.int32, device="cuda")
    sl = torch.ones(1, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        sk.paged_decode(q, kc, kc, bt, sl, window=-3)
    with pytest.raises(RuntimeError):
        sk.paged_decode(q, kc, kc, bt, sl, sinks=torch.zeros(8, device="cuda"))  # H_kv, not H_q
    cu = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        sk.attention_varlen(q, kc, kc, cu, sl, bt, sinks=torch.zeros(32, device="cuda").double())
