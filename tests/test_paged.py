"""The paged K/V cache ops against torch on the same data gathered into contiguous tensors:
`paged_decode` (every variant) and `attention_varlen` (every variant) against
F.scaled_dot_product_attention in fp32 per sequence, `rope_append_paged_` against the torch
rotation and slot writes. Pages are handed out in a shuffled order, so a sequence's pages are
never adjacent; lengths cover the empty sequence, a single key, partial last pages and slabs,
one long sequence among short ones (segments that cross warps of the length-aware split), GQA
groups of 1, 4 and 8, both head sizes and two page sizes. The varlen cases include prompts
appended to an existing context (the bottom-right causal mask). The same cases run on an e4m3
cache with per-head scales against SDPA on the dequantized bytes."""

import pytest
import torch
import torch.nn.functional as F

TOL = dict(atol=2e-2, rtol=2e-2)  # one bf16 rounding of an fp32 result, P in bf16


def _variants(name):
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants(name)))
    except Exception:
        return [0, 1]


def _paged_cache(lens, H_kv, D, page, seed=0, spare_pages=3):
    """Random K/V for sequences of `lens` keys in a paged cache with shuffled pages. Returns
    (k_cache, v_cache, block_table, seq_lens, k_list, v_list) where k_list[b] is the
    contiguous [H_kv, len_b, D] view of sequence b's keys."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    npages = [(n + page - 1) // page for n in lens]
    total = sum(npages) + spare_pages
    max_pages = max(1, max(npages))
    perm = torch.randperm(total, generator=g).tolist()
    k_cache = torch.randn(total, H_kv, page, D, generator=g).to("cuda", torch.bfloat16)
    v_cache = torch.randn(total, H_kv, page, D, generator=g).to("cuda", torch.bfloat16)
    bt = torch.full((len(lens), max_pages), -1, dtype=torch.int32)
    k_list, v_list, i = [], [], 0
    for b, n in enumerate(lens):
        ids = perm[i:i + npages[b]]
        i += npages[b]
        bt[b, :len(ids)] = torch.tensor(ids, dtype=torch.int32)
        if ids:
            kk = k_cache[ids].permute(1, 0, 2, 3).reshape(H_kv, -1, D)[:, :n]
            vv = v_cache[ids].permute(1, 0, 2, 3).reshape(H_kv, -1, D)[:, :n]
        else:
            kk = torch.empty(H_kv, 0, D, device="cuda", dtype=torch.bfloat16)
            vv = kk
        k_list.append(kk)
        v_list.append(vv)
    # pages past a sequence's length hold garbage that must never be read: poison them
    for b, n in enumerate(lens):
        if n % page:
            last = int(bt[b, npages[b] - 1])
            k_cache[last, :, n % page:] = float("nan")
            v_cache[last, :, n % page:] = float("nan")
    bt[bt < 0] = 0
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    return k_cache, v_cache, bt.cuda(), seq_lens, k_list, v_list


def _sdpa(q, k, v, mask=None):
    """q [H_q, S_q, D], k, v [H_kv, S_kv, D] -> fp32 [H_q, S_q, D]."""
    return F.scaled_dot_product_attention(q.float()[None], k.float()[None], v.float()[None],
                                          attn_mask=mask, enable_gqa=True)[0]


# (lens, H_q, H_kv, D, page)
DECODE_CASES = [
    ([1, 17, 500, 33], 32, 8, 128, 16),
    ([5000, 20, 300, 1, 0, 64], 32, 8, 128, 16),
    ([4096] * 3, 32, 8, 128, 32),
    ([700, 1, 129], 4, 4, 64, 16),
    ([257, 1000], 16, 2, 128, 64),
    ([3000] + [100] * 20, 8, 2, 64, 32),
]


def _case_id(c):
    lens = c[0]
    return f"n{len(lens)}_max{max(lens)}_hq{c[1]}_hkv{c[2]}_d{c[3]}_p{c[4]}"


@pytest.mark.parametrize("variant", _variants("paged_decode"))
@pytest.mark.parametrize("case", DECODE_CASES, ids=_case_id)
def test_paged_decode_matches_sdpa(sk, case, variant):
    lens, hq, hkv, d, page = case
    kc, vc, bt, sl, ks, vs = _paged_cache(lens, hkv, d, page)
    torch.manual_seed(1)
    q = torch.randn(len(lens), hq, d, device="cuda", dtype=torch.bfloat16)
    got = sk.paged_decode(q, kc, vc, bt, sl, variant=variant)
    assert got.shape == q.shape and got.dtype == torch.bfloat16
    for b, n in enumerate(lens):
        if n == 0:
            assert torch.all(got[b] == 0)
            continue
        ref = _sdpa(q[b][:, None], ks[b], vs[b])[:, 0]
        torch.testing.assert_close(got[b].float(), ref, **TOL)


def test_paged_decode_is_deterministic(sk):
    # the combine merges the pieces in a fixed order, so a replayed step gives the same bits
    lens = [5000, 20, 300, 1, 0, 64]
    kc, vc, bt, sl, _, _ = _paged_cache(lens, 8, 128, 16)
    q = torch.randn(len(lens), 32, 128, device="cuda", dtype=torch.bfloat16)
    a = sk.paged_decode(q, kc, vc, bt, sl)
    for _ in range(3):
        torch.testing.assert_close(sk.paged_decode(q, kc, vc, bt, sl), a, atol=0, rtol=0)


def test_paged_decode_matches_contiguous_attention(sk):
    # equal lengths: the paged kernel against the contiguous flash-decoding kernel
    lens = [2000] * 4
    kc, vc, bt, sl, ks, vs = _paged_cache(lens, 8, 128, 16)
    q = torch.randn(4, 32, 128, device="cuda", dtype=torch.bfloat16)
    got = sk.paged_decode(q, kc, vc, bt, sl)
    ref = sk.attention(q[:, :, None].contiguous(), torch.stack(ks).contiguous(),
                       torch.stack(vs).contiguous())[:, :, 0]
    torch.testing.assert_close(got.float(), ref.float(), **TOL)


def test_paged_decode_follows_seq_lens_on_device(sk):
    # the same launch with lengths changed in place (what a CUDA graph replay sees)
    lens = [900, 40, 3]
    kc, vc, bt, sl, ks, vs = _paged_cache(lens, 8, 128, 16)
    q = torch.randn(3, 32, 128, device="cuda", dtype=torch.bfloat16)
    sk.paged_decode(q, kc, vc, bt, sl)
    sl.copy_(torch.tensor([37, 40, 1], dtype=torch.int32))
    got = sk.paged_decode(q, kc, vc, bt, sl)
    for b, n in enumerate([37, 40, 1]):
        ref = _sdpa(q[b][:, None], ks[b][:, :n], vs[b][:, :n])[:, 0]
        torch.testing.assert_close(got[b].float(), ref, **TOL)


# (q_lens, ctx_lens, H_q, H_kv, D, page)
VARLEN_CASES = [
    ([300, 1, 130, 64], [0, 0, 0, 0], 32, 8, 128, 16),
    ([1000, 17], [0, 0], 8, 2, 64, 32),
    ([128, 200, 5], [50, 1000, 3], 32, 8, 128, 16),
    ([513], [0], 4, 4, 128, 64),
    ([0, 77, 256], [10, 0, 7], 16, 2, 128, 16),
]


def _varlen_id(c):
    return f"q{'-'.join(map(str, c[0]))}_ctx{'-'.join(map(str, c[1]))}_hq{c[2]}_hkv{c[3]}_d{c[4]}"


@pytest.mark.parametrize("variant", _variants("attention_varlen"))
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
@pytest.mark.parametrize("case", VARLEN_CASES, ids=_varlen_id)
def test_attention_varlen_matches_sdpa(sk, case, causal, variant):
    q_lens, ctx, hq, hkv, d, page = case
    lens = [a + b for a, b in zip(q_lens, ctx, strict=True)]
    kc, vc, bt, sl, ks, vs = _paged_cache(lens, hkv, d, page, seed=2)
    T = sum(q_lens)
    torch.manual_seed(3)
    q = torch.randn(T, hq, d, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32,
                      device="cuda")
    got = sk.attention_varlen(q, kc, vc, cu, sl, bt, causal=causal, variant=variant)
    assert got.shape == q.shape
    t0 = 0
    for b, (n, c) in enumerate(zip(q_lens, ctx, strict=True)):
        if n == 0:
            continue
        qb = q[t0:t0 + n].transpose(0, 1)  # [H_q, n, D]
        mask = None
        if causal:  # query i at absolute position c + i sees keys j <= c + i
            i = torch.arange(n, device="cuda")[:, None]
            j = torch.arange(n + c, device="cuda")[None, :]
            mask = j <= c + i
        ref = _sdpa(qb, ks[b], vs[b], mask).transpose(0, 1)
        torch.testing.assert_close(got[t0:t0 + n].float(), ref, **TOL)
        t0 += n


@pytest.mark.parametrize("variant", _variants("paged_decode"))
def test_paged_decode_long_context(sk, variant):
    # 128K keys (Llama 3.1's context) next to shorter ones: the 64-bit offsets and the split
    lens = [131072, 70001, 17]
    kc, vc, bt, sl, ks, vs = _paged_cache(lens, 2, 128, 16, seed=4)
    torch.manual_seed(5)
    q = torch.randn(len(lens), 8, 128, device="cuda", dtype=torch.bfloat16)
    got = sk.paged_decode(q, kc, vc, bt, sl, variant=variant)
    for b in range(len(lens)):
        ref = _sdpa(q[b][:, None], ks[b], vs[b])[:, 0]
        torch.testing.assert_close(got[b].float(), ref, **TOL)


@pytest.mark.parametrize("variant", _variants("attention_varlen"))
def test_attention_varlen_long_context(sk, variant):
    # the last chunk of a 128K-token prompt (bottom-right causal over 130816 cached keys)
    # packed with a chunk of a 5K one
    q_lens, ctx = [256, 64], [130816, 5000]
    lens = [a + b for a, b in zip(q_lens, ctx, strict=True)]
    kc, vc, bt, sl, ks, vs = _paged_cache(lens, 2, 128, 16, seed=6)
    torch.manual_seed(7)
    q = torch.randn(sum(q_lens), 8, 128, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, q_lens[0], sum(q_lens)], dtype=torch.int32, device="cuda")
    got = sk.attention_varlen(q, kc, vc, cu, sl, bt, causal=True, variant=variant)
    t0 = 0
    for b, (n, c) in enumerate(zip(q_lens, ctx, strict=True)):
        i = torch.arange(n, device="cuda")[:, None]
        mask = torch.arange(n + c, device="cuda")[None, :] <= c + i
        ref = _sdpa(q[t0:t0 + n].transpose(0, 1), ks[b], vs[b], mask).transpose(0, 1)
        torch.testing.assert_close(got[t0:t0 + n].float(), ref, **TOL)
        t0 += n


def test_rope_append_paged_long_positions(sk):
    # positions past 64K on Llama 3.1's scaled tables
    L = pytest.importorskip("spark_kernels.layer")
    scaling = {"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
               "original_max_position_embeddings": 8192, "rope_type": "llama3"}
    rope = L.RoPE(131072, scaling=scaling)
    hq, hkv, d, page = 32, 8, 128, 16
    positions = [65535, 65536, 100000, 131071]
    slots = [0, 1, 2, 3]
    T = len(positions)
    torch.manual_seed(8)
    qkv = torch.randn(T, (hq + 2 * hkv) * d, device="cuda", dtype=torch.bfloat16)
    kc = torch.zeros(1, hkv, page, d, device="cuda", dtype=torch.bfloat16)
    vc = torch.zeros_like(kc)
    pos_t = torch.tensor(positions, dtype=torch.int32, device="cuda")
    q = sk.rope_append_paged_(qkv, rope.cos, rope.sin, pos_t,
                              torch.tensor(slots, dtype=torch.int32, device="cuda"), kc, vc,
                              hq, hkv)
    cos, sin = rope.cos[pos_t.long()], rope.sin[pos_t.long()]
    x = qkv.view(T, hq + 2 * hkv, d).float()
    r = torch.cat([-x[..., d // 2:], x[..., :d // 2]], dim=-1)
    ref = x * cos[:, None] + r * sin[:, None]
    tol = dict(atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(q.float(), ref[:, :hq], **tol)
    torch.testing.assert_close(kc[0, :, :T].transpose(0, 1).float(), ref[:, hq:hq + hkv], **tol)


@pytest.mark.parametrize("page", [16, 32])
def test_rope_append_paged_matches_torch(sk, page):
    L = pytest.importorskip("spark_kernels.layer")
    hq, hkv, d = 32, 8, 128
    positions = [0, 1, 2, 40, 41, 7, 1000]
    slots = [5, 6, 7, 2 * page + 3, 2 * page + 4, -1, 3 * page + page - 1]
    T = len(positions)
    torch.manual_seed(0)
    qkv = torch.randn(T, (hq + 2 * hkv) * d, device="cuda", dtype=torch.bfloat16)
    rope = L.RoPE(1024)
    kc = torch.randn(4, hkv, page, d, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    k_ref, v_ref = kc.clone(), vc.clone()
    pos_t = torch.tensor(positions, dtype=torch.int32, device="cuda")
    slot_t = torch.tensor(slots, dtype=torch.int32, device="cuda")
    q = sk.rope_append_paged_(qkv, rope.cos, rope.sin, pos_t, slot_t, kc, vc, hq, hkv)

    cos, sin = rope.cos[pos_t.long()], rope.sin[pos_t.long()]  # [T, D]
    x = qkv.view(T, hq + 2 * hkv, d)

    def rot(t):  # [T, H, d]
        tf = t.float()
        r = torch.cat([-tf[..., d // 2:], tf[..., :d // 2]], dim=-1)
        return (tf * cos[:, None] + r * sin[:, None]).to(torch.bfloat16)

    q_ref = rot(x[:, :hq])
    k_new = rot(x[:, hq:hq + hkv])
    for t, s in enumerate(slots):
        if s < 0:
            continue
        k_ref[s // page, :, s % page] = k_new[t]
        v_ref[s // page, :, s % page] = x[t, hq + hkv:]
    assert q.shape == (T, hq, d)
    tol = dict(atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(q.float(), q_ref.float(), **tol)
    torch.testing.assert_close(kc.float(), k_ref.float(), **tol)
    torch.testing.assert_close(vc, v_ref, atol=0, rtol=0)


def test_paged_ops_reject_bad_input(sk):
    kc = torch.zeros(4, 8, 24, 128, device="cuda", dtype=torch.bfloat16)  # page 24
    q = torch.zeros(1, 32, 128, device="cuda", dtype=torch.bfloat16)
    bt = torch.zeros(1, 4, dtype=torch.int32, device="cuda")
    sl = torch.ones(1, dtype=torch.int32, device="cuda")
    with pytest.raises(RuntimeError):
        sk.paged_decode(q, kc, kc, bt, sl)
    kc = torch.zeros(4, 8, 16, 128, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        sk.paged_decode(q, kc, kc, bt.long(), sl)


# ---- the fp8 (e4m3) cache ------------------------------------------------------------------
# The kernels read the e4m3 bytes and the per-head scales; the reference is SDPA in fp32 on the
# same bytes dequantized in torch, so the only error left is the kernel's own (bf16 P, fp32
# accumulation, the same TOL as the bf16 cache).


def _head_scales(H_kv, base=5.0):
    """Distinct per-head scales, kv head h mapping |x| = base * (1 + h / 4) to 448: a
    standard normal cache lands on the whole e4m3 range, subnormals included."""
    return torch.tensor([base * (1 + h / 4) / 448 for h in range(H_kv)], device="cuda")


def _paged_cache_fp8(lens, H_kv, D, page, seed=0):
    """_paged_cache's random cache rounded to e4m3 with _head_scales. Returns (k8, v8,
    k_scale, v_scale, block_table, seq_lens, k_list, v_list) with k_list[b] the fp32 values
    of sequence b's e4m3 keys, [H_kv, len_b, D]. The NaN past each length stays NaN."""
    from spark_kernels.ops import dequantize_kv_cache, quantize_kv_cache

    kc, vc, bt, sl, _, _ = _paged_cache(lens, H_kv, D, page, seed)
    ks, vs = _head_scales(H_kv), _head_scales(H_kv, base=3.0)
    k8, v8 = quantize_kv_cache(kc, ks), quantize_kv_cache(vc, vs)
    kd, vd = dequantize_kv_cache(k8, ks), dequantize_kv_cache(v8, vs)
    k_list, v_list = [], []
    for b, n in enumerate(lens):
        npg = (n + page - 1) // page
        ids = bt[b, :npg].long()
        k_list.append(kd[ids].permute(1, 0, 2, 3).reshape(H_kv, npg * page, D)[:, :n])
        v_list.append(vd[ids].permute(1, 0, 2, 3).reshape(H_kv, npg * page, D)[:, :n])
    return k8, v8, ks, vs, bt, sl, k_list, v_list


@pytest.mark.parametrize("variant", _variants("paged_decode"))
@pytest.mark.parametrize("case", DECODE_CASES, ids=_case_id)
def test_paged_decode_fp8_matches_dequantized(sk, case, variant):
    lens, hq, hkv, d, page = case
    k8, v8, ks, vs, bt, sl, kl, vl = _paged_cache_fp8(lens, hkv, d, page)
    torch.manual_seed(1)
    q = torch.randn(len(lens), hq, d, device="cuda", dtype=torch.bfloat16)
    got = sk.paged_decode(q, k8, v8, bt, sl, variant=variant, k_scale=ks, v_scale=vs)
    assert got.shape == q.shape and got.dtype == torch.bfloat16
    for b, n in enumerate(lens):
        if n == 0:
            assert torch.all(got[b] == 0)
            continue
        ref = _sdpa(q[b][:, None], kl[b], vl[b])[:, 0]
        torch.testing.assert_close(got[b].float(), ref, **TOL)


def test_paged_decode_fp8_is_deterministic_and_long(sk):
    # 128K keys next to short ones through the split and the combine, twice, the same bits
    lens = [131072, 5000, 17, 0]
    k8, v8, ks, vs, bt, sl, kl, vl = _paged_cache_fp8(lens, 2, 128, 16, seed=4)
    torch.manual_seed(5)
    q = torch.randn(len(lens), 8, 128, device="cuda", dtype=torch.bfloat16)
    got = sk.paged_decode(q, k8, v8, bt, sl, k_scale=ks, v_scale=vs)
    torch.testing.assert_close(sk.paged_decode(q, k8, v8, bt, sl, k_scale=ks, v_scale=vs), got,
                               atol=0, rtol=0)
    for b in range(3):
        ref = _sdpa(q[b][:, None], kl[b], vl[b])[:, 0]
        torch.testing.assert_close(got[b].float(), ref, **TOL)


@pytest.mark.parametrize("variant", _variants("attention_varlen"))
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "full"])
@pytest.mark.parametrize("case", VARLEN_CASES, ids=_varlen_id)
def test_attention_varlen_fp8_matches_dequantized(sk, case, causal, variant):
    q_lens, ctx, hq, hkv, d, page = case
    lens = [a + b for a, b in zip(q_lens, ctx, strict=True)]
    k8, v8, ks, vs, bt, sl, kl, vl = _paged_cache_fp8(lens, hkv, d, page, seed=2)
    T = sum(q_lens)
    torch.manual_seed(3)
    q = torch.randn(T, hq, d, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(q_lens).cumsum(0)), dtype=torch.int32,
                      device="cuda")
    got = sk.attention_varlen(q, k8, v8, cu, sl, bt, causal=causal, variant=variant,
                              k_scale=ks, v_scale=vs)
    t0 = 0
    for b, (n, c) in enumerate(zip(q_lens, ctx, strict=True)):
        if n == 0:
            continue
        mask = None
        if causal:
            i = torch.arange(n, device="cuda")[:, None]
            mask = torch.arange(n + c, device="cuda")[None, :] <= c + i
        ref = _sdpa(q[t0:t0 + n].transpose(0, 1), kl[b], vl[b], mask).transpose(0, 1)
        torch.testing.assert_close(got[t0:t0 + n].float(), ref, **TOL)
        t0 += n


def test_paged_fp8_close_to_bf16(sk):
    # the cache's own rounding: e4m3 attention against the bf16 cache it was rounded from
    lens = [3000, 40, 513]
    kc, vc, bt, sl, _, _ = _paged_cache(lens, 8, 128, 16, seed=9)
    k8, v8, ks, vs, _, _, _, _ = _paged_cache_fp8(lens, 8, 128, 16, seed=9)
    q = torch.randn(3, 32, 128, device="cuda", dtype=torch.bfloat16)
    a = sk.paged_decode(q, kc, vc, bt, sl).float()
    b = sk.paged_decode(q, k8, v8, bt, sl, k_scale=ks, v_scale=vs).float()
    # e4m3 keeps 3 mantissa bits: about 2.5% rms per element, 3.7% on these outputs in torch
    assert ((a - b).norm() / a.norm()).item() < 6e-2


@pytest.mark.parametrize("page", [16, 32])
def test_rope_append_paged_fp8(sk, page):
    from spark_kernels.ops import dequantize_kv_cache, quantize_kv_cache

    L = pytest.importorskip("spark_kernels.layer")
    hq, hkv, d = 32, 8, 128
    positions = [0, 1, 2, 40, 41, 7, 1000]
    slots = [5, 6, 7, 2 * page + 3, 2 * page + 4, -1, 3 * page + page - 1]
    T = len(positions)
    torch.manual_seed(0)
    qkv = torch.randn(T, (hq + 2 * hkv) * d, device="cuda", dtype=torch.bfloat16) * 4
    rope = L.RoPE(1024)
    ks, vs = _head_scales(hkv, base=12.0), _head_scales(hkv, base=8.0)
    kc = torch.randint(0, 120, (4, hkv, page, d), device="cuda", dtype=torch.uint8).view(
        torch.float8_e4m3fn)
    vc = torch.randint(0, 120, (4, hkv, page, d), device="cuda", dtype=torch.uint8).view(
        torch.float8_e4m3fn)
    k_ref, v_ref = kc.view(torch.uint8).clone(), vc.view(torch.uint8).clone()
    pos_t = torch.tensor(positions, dtype=torch.int32, device="cuda")
    slot_t = torch.tensor(slots, dtype=torch.int32, device="cuda")
    q = sk.rope_append_paged_(qkv, rope.cos, rope.sin, pos_t, slot_t, kc, vc, hq, hkv,
                              k_scale=ks, v_scale=vs)

    cos, sin = rope.cos[pos_t.long()], rope.sin[pos_t.long()]
    x = qkv.view(T, hq + 2 * hkv, d).float()
    r = torch.cat([-x[..., d // 2:], x[..., :d // 2]], dim=-1)
    rot = x * cos[:, None] + r * sin[:, None]  # fp32, as the kernel rotates
    k_new = quantize_kv_cache(rot[:, hq:hq + hkv], ks).view(torch.uint8)
    v_new = quantize_kv_cache(x[:, hq + hkv:], vs).view(torch.uint8)
    for t, s in enumerate(slots):
        if s < 0:
            continue
        k_ref[s // page, :, s % page] = k_new[t]
        v_ref[s // page, :, s % page] = v_new[t]
    torch.testing.assert_close(q.float(), rot[:, :hq], atol=1e-2, rtol=1e-2)
    # v is a bf16 input divided exactly: the same bytes. k's fp32 rotation may differ from
    # torch's in the last bit (fma contraction), so a value may round to the next e4m3 step.
    assert torch.equal(vc.view(torch.uint8), v_ref)
    kd = dequantize_kv_cache(kc, ks)
    kr = dequantize_kv_cache(k_ref.view(torch.float8_e4m3fn), ks)
    torch.testing.assert_close(kd, kr, atol=float(ks.max()) * 2 ** -9, rtol=2 ** -3)
    assert (kc.view(torch.uint8) != k_ref).float().mean().item() < 1e-3


def test_paged_ops_reject_bad_fp8_input(sk):
    k8 = torch.zeros(4, 8, 16, 128, device="cuda", dtype=torch.uint8).view(torch.float8_e4m3fn)
    kb = torch.zeros(4, 8, 16, 128, device="cuda", dtype=torch.bfloat16)
    q = torch.zeros(1, 32, 128, device="cuda", dtype=torch.bfloat16)
    bt = torch.zeros(1, 4, dtype=torch.int32, device="cuda")
    sl = torch.ones(1, dtype=torch.int32, device="cuda")
    s = torch.ones(8, device="cuda")
    with pytest.raises(RuntimeError):  # no scales
        sk.paged_decode(q, k8, k8, bt, sl)
    with pytest.raises(RuntimeError):  # scales with a bf16 cache
        sk.paged_decode(q, kb, kb, bt, sl, k_scale=s, v_scale=s)
    with pytest.raises(RuntimeError):  # one fp8 and one bf16 cache
        sk.paged_decode(q, k8, kb, bt, sl, k_scale=s, v_scale=s)
    with pytest.raises(RuntimeError):  # a scale per query head instead of per kv head
        sk.paged_decode(q, k8, k8, bt, sl, k_scale=torch.ones(32, device="cuda"), v_scale=s)
    with pytest.raises(RuntimeError):
        sk.paged_decode(q, k8, k8, bt, sl, k_scale=s.double(), v_scale=s)
