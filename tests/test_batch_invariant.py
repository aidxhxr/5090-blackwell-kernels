"""Batch invariance: a request's logits must not depend on what else is in its steps.

The default engine is deterministic (the same schedule gives the same bits) but not batch
invariant: hgemm picks its kernel, tile and K split from M, paged_decode splits the keys of
the whole batch over the SMs, and a decode row computed by attention_varlen (a mixed step, a
resumed sequence, a verify forward) takes a different reduction than the same row in a decode
step. The tests below show that on the default kernels, then check the invariant ones bit for
bit (torch.equal):

  * hgemm variant 7, fp8gemm variant 4 and w4gemm variant 3: row i of A B is the same at
    every M, whichever kernel M picks;
  * paged_decode variant 2 and attention_varlen variant 2, on a bf16 and on an e4m3 cache: a
    query row's output is the same whatever the batch around it, and the same from either op;
  * Engine(batch_invariant=True): every request's logits, step by step, are the same alone,
    in a full batch, with its prompt chunked, preempted and resumed, from a CUDA graph or
    eagerly, in any slot, behind a cached prefix, and under speculative decoding; with int4,
    fp8-tok and mxfp8 weights and with the fp8 cache, alone and in batches of 8 and 32.

Two random layers and a 1,024-token vocabulary, as tests/test_engine.py."""

import math

import pytest
import torch

E = pytest.importorskip("spark_kernels.engine")
L = pytest.importorskip("spark_kernels.layer")

N_LAYERS, VOCAB = 2, 1024
INV_GEMM, INV_ATTN = 7, 2  # the invariant variants of hgemm and of both paged attention ops
INV_FP8, INV_W4 = 4, 3  # of fp8gemm and w4gemm


@pytest.fixture(scope="module")
def weights():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    return E.ModelWeights.random(N_LAYERS, VOCAB, seed=5)


# ---- the GEMM ----------------------------------------------------------------------------

MS = (1, 2, 7, 16, 17, 33, 64, 65, 100, 128, 129, 200, 256, 300, 513)


def _operands(m, n, k, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(m, k, generator=g).to("cuda", torch.bfloat16)
    b = (torch.randn(k, n, generator=g) / math.sqrt(k)).to("cuda", torch.bfloat16)
    return a, b


def test_default_hgemm_rows_depend_on_m(sk):
    """The default variant: the same row of A B in a batch of 1 and a batch of 300 tokens
    comes out of different kernels (the decode kernel, a Stream-K split) and does not have
    the same bits. This is the problem the invariant variant solves."""
    a, b = _operands(300, 4096, 4096, 0)
    full = sk.hgemm(a, b)
    assert any(not torch.equal(sk.hgemm(a[:m].contiguous(), b), full[:m]) for m in (1, 64, 200))


@pytest.mark.parametrize("n,k", [(4096, 4096), (6144, 4096), (1024, 4096), (4096, 1536)])
def test_hgemm_invariant_rows_do_not_depend_on_m(sk, n, k):
    a, b = _operands(MS[-1], n, k, n + k)
    full = sk.hgemm(a, b, INV_GEMM)
    torch.testing.assert_close(full.float(), (a.float() @ b.float()), atol=3e-2, rtol=3e-2)
    for m in MS:
        assert torch.equal(sk.hgemm(a[:m].contiguous(), b, INV_GEMM), full[:m]), f"M = {m}"
        # rows from the middle of the batch, too
        lo = (MS[-1] - m) // 2
        assert torch.equal(sk.hgemm(a[lo:lo + m].contiguous(), b, INV_GEMM), full[lo:lo + m])


def test_hgemm_invariant_epilogues_do_not_depend_on_m(sk):
    """The fused residual add and the SwiGLU product run in the same fp32 epilogue at every
    M, so they keep the property."""
    a, w = _operands(MS[-1], 2 * 2048, 4096, 9)  # an interleaved gate/up weight, N = 2 x 2048
    res = torch.randn(MS[-1], 4096, device="cuda").to(torch.bfloat16)
    wo = _operands(1, 4096, 4096, 10)[1]
    full_sw = sk.hgemm_swiglu(a, w, INV_GEMM)
    full_res = sk.hgemm(a, wo, INV_GEMM, residual=res)
    for m in MS:
        assert torch.equal(sk.hgemm_swiglu(a[:m].contiguous(), w, INV_GEMM), full_sw[:m])
        out = res[:m].clone()
        sk.hgemm(a[:m].contiguous(), wo, INV_GEMM, residual=out, out=out)
        assert torch.equal(out, full_res[:m]), f"M = {m}"


def test_hgemm_invariant_is_not_the_default(sk):
    """Variant 7 is opt-in, like sgemm's TF32 rungs: -1 still runs variant 6."""
    assert sk.num_variants("hgemm") == 8
    a, b = _operands(300, 4096, 4096, 3)
    assert torch.equal(sk.hgemm(a, b), sk.hgemm(a, b, 6))


# ---- the quantized GEMMs -----------------------------------------------------------------


def _fp8_operands(m, n, k, seed):
    """A's rows quantized once per row (fixed bytes, what the GEMM alone sees), the weight
    [N, K] e4m3, unit scales."""
    a, b = _operands(m, n, k, seed)
    a8 = sk_fp8_quantize(a)
    bt = b.t().contiguous().to(torch.float8_e4m3fn)
    return a8, bt, torch.ones(1, device="cuda")


def sk_fp8_quantize(a):
    import spark_kernels as sk

    return sk.fp8_quantize(a, "row")[0]


def test_default_fp8gemm_rows_depend_on_m(sk):
    """Variant 3 (the default) Stream-Ks a 300-row GEMM and runs a 1-row one on the decode
    tile: the same row comes out with different bits."""
    a8, bt, one = _fp8_operands(300, 4096, 4096, 20)
    full = sk.fp8gemm(a8, bt, one, one)
    assert any(not torch.equal(sk.fp8gemm(a8[:m].contiguous(), bt, one, one), full[:m])
               for m in (1, 64, 200))


@pytest.mark.parametrize("n,k", [(4096, 4096), (1024, 4096), (4096, 1536)])
def test_fp8gemm_invariant_rows_do_not_depend_on_m(sk, n, k):
    """Variant 4: the unsplit decode tile at M <= 64, a whole-tile TMA tile above; N = 1024 is
    the narrow N whose decode route split K by strips."""
    a8, bt, one = _fp8_operands(MS[-1], n, k, n + k + 1)
    full = sk.fp8gemm(a8, bt, one, one, INV_FP8)
    torch.testing.assert_close(full.float(), a8.float() @ bt.float().t(), atol=5e-2, rtol=5e-2)
    for m in MS:
        assert torch.equal(sk.fp8gemm(a8[:m].contiguous(), bt, one, one, INV_FP8), full[:m]), m
        lo = (MS[-1] - m) // 2
        assert torch.equal(sk.fp8gemm(a8[lo:lo + m].contiguous(), bt, one, one, INV_FP8),
                           full[lo:lo + m])


def test_fp8gemm_mx_invariant_rows_do_not_depend_on_m(sk):
    """MX mode on variant 4: variant 1's tiles with no split-K tail at any M."""
    from spark_kernels import reference as ref

    n, k = 1024, 4096
    a, b = _operands(MS[-1], n, k, 21)
    a8, sfa = sk.fp8_quantize(a, "mx")
    bt, sfb = ref.quantize_mx(b.t().contiguous())
    try:
        full = sk.fp8gemm(a8, bt, sfa=sfa, sfb=sfb, variant=INV_FP8)
    except (ValueError, RuntimeError) as e:
        pytest.skip(f"MX mode not built: {str(e).splitlines()[0]}")
    for m in MS:
        got = sk.fp8gemm(a8[:m].contiguous(), bt, sfa=sfa[:m].contiguous(), sfb=sfb,
                         variant=INV_FP8)
        assert torch.equal(got, full[:m]), f"M = {m}"


def test_default_w4gemm_rows_depend_on_m(sk):
    """Variant 2 splits K across warps and blocks by M and N: at N = 1024 a one-row GEMM and
    a 300-row one give a row different bits."""
    a, w = _operands(300, 1024, 4096, 22)
    ww = sk.W4Weight.quantize(w)
    full = sk.w4gemm(a, ww)
    assert any(not torch.equal(sk.w4gemm(a[:m].contiguous(), ww), full[:m])
               for m in (1, 8, 200))


@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
@pytest.mark.parametrize("n,k", [(4096, 4096), (1024, 4096), (4096, 1536)])
def test_w4gemm_invariant_rows_do_not_depend_on_m(sk, n, k, asym):
    a, w = _operands(MS[-1], n, k, n + k + 2)
    ww = sk.W4Weight.quantize(w, asym)
    full = sk.w4gemm(a, ww, variant=INV_W4)
    torch.testing.assert_close(full.float(), sk.w4gemm(a, ww, variant=2).float(), atol=3e-2,
                               rtol=3e-2)
    for m in MS:
        assert torch.equal(sk.w4gemm(a[:m].contiguous(), ww, variant=INV_W4), full[:m]), m
        lo = (MS[-1] - m) // 2
        assert torch.equal(sk.w4gemm(a[lo:lo + m].contiguous(), ww, variant=INV_W4),
                           full[lo:lo + m])


def test_quantized_invariant_rungs_are_not_the_default(sk):
    assert sk.num_variants("fp8gemm") == 5 and sk.num_variants("w4gemm") == 4
    a8, bt, one = _fp8_operands(300, 4096, 4096, 23)
    assert torch.equal(sk.fp8gemm(a8, bt, one, one), sk.fp8gemm(a8, bt, one, one, 3))
    a, w = _operands(300, 4096, 4096, 24)
    ww = sk.W4Weight.quantize(w)
    assert torch.equal(sk.w4gemm(a, ww), sk.w4gemm(a, ww, variant=2))


# ---- attention ---------------------------------------------------------------------------

H_Q, H_KV, D, PAGE = 32, 8, 128, 16


def _cache(lens, seed=0, fp8=False):
    """bf16 K/V for sequences of `lens` keys in shuffled pages (or the same rounded to e4m3
    with a scale per kv head); returns the caches, the block table, the lengths and the
    scale keyword arguments of the paged ops."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    npg = [(n + PAGE - 1) // PAGE for n in lens]
    total = sum(npg) + 2
    perm = torch.randperm(total, generator=g).tolist()
    k = torch.randn(total, H_KV, PAGE, D, generator=g).to("cuda", torch.bfloat16)
    v = torch.randn(total, H_KV, PAGE, D, generator=g).to("cuda", torch.bfloat16)
    bt = torch.zeros(len(lens), max(npg), dtype=torch.int32)
    i = 0
    for b, n in enumerate(npg):
        bt[b, :n] = torch.tensor(perm[i:i + n], dtype=torch.int32)
        i += n
    sc = {}
    if fp8:
        from spark_kernels.ops import quantize_kv_cache

        ks = torch.tensor([4.0 * (1 + h / 4) / 448 for h in range(H_KV)], device="cuda")
        vs = torch.tensor([3.0 * (1 + h / 4) / 448 for h in range(H_KV)], device="cuda")
        k, v = quantize_kv_cache(k, ks), quantize_kv_cache(v, vs)
        sc = dict(k_scale=ks, v_scale=vs)
    return k, v, bt.cuda(), torch.tensor(lens, dtype=torch.int32, device="cuda"), sc


LENS = [700, 37, 2000, 1, 300, 4100, 256, 257]
CACHES = ["bf16", "fp8"]


def _decode(sk, q, k, v, bt, lens, idx, variant, sc=None):
    """paged_decode of the sequences `idx` (a batch made of those rows only)."""
    ix = torch.tensor(idx, device="cuda")
    return sk.paged_decode(q[ix].contiguous(), k, v, bt[ix].contiguous(), lens[ix].contiguous(),
                           variant, **(sc or {}))


def test_default_paged_decode_depends_on_the_batch(sk):
    """Variant 1 cuts the keys of the whole batch into equal ranges per warp, so a sequence
    is cut in other places when other sequences come and go."""
    k, v, bt, lens, _ = _cache(LENS)
    q = torch.randn(len(LENS), H_Q, D, device="cuda").to(torch.bfloat16)
    alone = _decode(sk, q, k, v, bt, lens, [5], 1)
    full = _decode(sk, q, k, v, bt, lens, list(range(len(LENS))), 1)
    assert not torch.equal(alone[0], full[5])


@pytest.mark.parametrize("fmt", CACHES)
def test_paged_decode_invariant_to_the_batch(sk, fmt):
    k, v, bt, lens, sc = _cache(LENS, fp8=fmt == "fp8")
    q = torch.randn(len(LENS), H_Q, D, device="cuda").to(torch.bfloat16)
    full = _decode(sk, q, k, v, bt, lens, list(range(len(LENS))), INV_ATTN, sc)
    ref = _decode(sk, q, k, v, bt, lens, list(range(len(LENS))), 0, sc)  # one warp per row
    torch.testing.assert_close(full.float(), ref.float(), atol=2e-2, rtol=2e-2)
    batches = [[b] for b in range(len(LENS))] + [[5, 2], [7, 6, 5, 4, 3, 2, 1, 0],
                                                   [2, 2, 5, 0], list(range(len(LENS))) * 8]
    for idx in batches:
        out = _decode(sk, q, k, v, bt, lens, idx, INV_ATTN, sc)
        for j, b in enumerate(idx):
            assert torch.equal(out[j], full[b]), f"sequence {b} in batch {idx}"


def _varlen(sk, q_rows, k, v, bt, lens, parts, variant, sc=None):
    """attention_varlen over `parts`, a list of (sequence, first position, last position + 1):
    the query rows q_rows[b][first:end] of each, packed, with seq_lens[b] = end keys."""
    q = torch.cat([q_rows[b][a:e] for b, a, e in parts]).contiguous()
    cu = torch.tensor([0] + [e - a for _, a, e in parts], dtype=torch.int32).cumsum(
        0, dtype=torch.int32).cuda()
    ix = torch.tensor([b for b, _, _ in parts], device="cuda")
    kv = torch.tensor([e for _, _, e in parts], dtype=torch.int32, device="cuda")
    out = sk.attention_varlen(q, k, v, cu, kv, bt[ix].contiguous(), True, variant,
                              **(sc or {}))
    return out.split([e - a for _, a, e in parts])


@pytest.mark.parametrize("fmt", CACHES)
def test_varlen_invariant_to_chunking_and_matches_decode(sk, fmt):
    """attention_varlen variant 2 gives a query row the bits it gets from paged_decode
    variant 2, whether the row is a decode row, the last row of a prompt, or anywhere in a
    chunk of any size after any context, packed with any other prompts; on either cache."""
    k, v, bt, lens, sc = _cache(LENS, seed=1, fp8=fmt == "fp8")
    g = torch.Generator(device="cpu").manual_seed(2)
    q_rows = [torch.randn(n, H_Q, D, generator=g).to("cuda", torch.bfloat16) for n in LENS]
    # the decode row of every sequence: its last position
    last = torch.stack([q_rows[b][-1] for b in range(len(LENS))])
    dec = _decode(sk, last, k, v, bt, lens, list(range(len(LENS))), INV_ATTN, sc)
    whole = _varlen(sk, q_rows, k, v, bt, lens, [(b, 0, n) for b, n in enumerate(LENS)],
                    INV_ATTN, sc)
    for b in range(len(LENS)):
        assert torch.equal(whole[b][-1], dec[b]), f"sequence {b}: varlen last row vs decode"
    ref = _varlen(sk, q_rows, k, v, bt, lens, [(b, 0, n) for b, n in enumerate(LENS)], 1, sc)
    for b in range(len(LENS)):
        torch.testing.assert_close(whole[b].float(), ref[b].float(), atol=2e-2, rtol=2e-2)
    # the same prompts cut into chunks (each chunk after the keys of the ones before it),
    # packed in other orders and next to one-row chunks
    for cut in (1, 5, 64, 129, 1000):
        for b, n in enumerate(LENS):
            pieces = [(b, a, min(n, a + cut)) for a in range(0, n, cut)][-3:]
            others = [(o, LENS[o] - 1, LENS[o]) for o in (2, 3) if o != b]
            outs = _varlen(sk, q_rows, k, v, bt, lens, others + pieces, INV_ATTN, sc)
            for (_, a, e), o in zip(pieces, outs[len(others):], strict=True):
                assert torch.equal(o, whole[b][a:e]), f"sequence {b}, chunk {a}:{e} of {cut}"


def test_invariant_attention_fp8_reads_the_scales(sk):
    """The e4m3 path applies K's scale in the softmax scale and V's to the output: other
    scales on the same bytes give other outputs, and unit scales give the bf16 kernel's
    result on the dequantized values to within its rounding."""
    from spark_kernels.ops import dequantize_kv_cache

    k8, v8, bt, lens, sc = _cache([700, 40], fp8=True)
    q = torch.randn(2, H_Q, D, device="cuda").to(torch.bfloat16)
    out = sk.paged_decode(q, k8, v8, bt, lens, INV_ATTN, **sc)
    half = dict(k_scale=sc["k_scale"] * 0.5, v_scale=sc["v_scale"] * 2)
    assert not torch.equal(sk.paged_decode(q, k8, v8, bt, lens, INV_ATTN, **half), out)
    kd = dequantize_kv_cache(k8, sc["k_scale"], torch.bfloat16)
    vd = dequantize_kv_cache(v8, sc["v_scale"], torch.bfloat16)
    ref = sk.paged_decode(q, kd, vd, bt, lens, INV_ATTN)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)


# ---- the engine --------------------------------------------------------------------------


def _requests(n_req=10, seed=11):
    """Prompts of 1 to 300 tokens, a few sharing a 40-token prefix (prefix-cache hits), and
    6 to 47 new tokens each; one request samples (temperature, top-k, its own seed)."""
    g = torch.Generator().manual_seed(seed)
    shared = torch.randint(0, VOCAB, (40,), generator=g)
    prompts, news, params = [], [], []
    for i in range(n_req):
        n = int(torch.randint(1, 300, (), generator=g))
        p = torch.randint(0, VOCAB, (n,), generator=g)
        if i % 4 == 1:
            p = torch.cat([shared, p[:60]])
        prompts.append(p)
        news.append(int(torch.randint(6, 48, (), generator=g)))
        params.append(E.SamplingParams(temperature=0.8, top_k=40, seed=7) if i == 3 else None)
    return prompts, news, params


def _serve(weights, rope, reqs, invariant, fmt="bf16", kv="bf16", **kw):
    """Each request's logits [generated, VOCAB], one row per token in order, and the stats."""
    prompts, news, params = reqs
    opts = dict(max_batch=8, max_seq=1024, num_pages=256, graphs=True)
    opts.update(kw)
    model = E.SparkModel(weights, rope, weights_format=fmt, keep_bf16=True)
    eng = E.Engine(model, N_LAYERS, log_tokens=True, log_logits=True, batch_invariant=invariant,
                   kv_format=kv, **opts)
    seqs = [eng.submit(p, m, pp) for p, m, pp in zip(prompts, news, params, strict=True)]
    st = eng.run()
    out = eng.outputs()
    logits = {s.sid: torch.stack(eng.logit_log[s.sid]) for s in seqs}
    for s, m in zip(seqs, news, strict=True):
        assert logits[s.sid].shape == (m, VOCAB) and len(out[s.sid]) == m
    return logits, out, st


# The schedules a request can see. "alone" is the reference: one slot, no graphs, the whole
# prompt in one forward.
SCHEDULES = {
    "alone": dict(max_batch=1, graphs=False, prefill_tokens=4096, prefix_cache=False),
    "batch": dict(max_batch=8),
    "batch_32": dict(max_batch=32),
    "slots_4_chunked": dict(max_batch=4, prefill_tokens=64),
    "eager_no_mixed": dict(max_batch=16, graphs=False, mixed=False, compact=False),
    "preempting": dict(max_batch=8, num_pages=24, prefill_tokens=128),
}


def _same(a, b):
    return all(torch.equal(a[s], b[s]) for s in a)


def test_default_engine_is_not_batch_invariant(sk, weights):
    """Today's engine: the same requests alone and in a batch get different logits."""
    rope = L.RoPE(1024)
    reqs = _requests()
    ref, _, _ = _serve(weights, rope, reqs, False, **SCHEDULES["alone"])
    got, _, _ = _serve(weights, rope, reqs, False, **SCHEDULES["batch"])
    n = min(min(ref[s].shape[0], got[s].shape[0]) for s in ref)
    assert not all(torch.equal(ref[s][:n], got[s][:n]) for s in ref)


def test_engine_logits_are_batch_invariant(sk, weights):
    rope = L.RoPE(1024)
    reqs = _requests()
    ref, ref_tok, _ = _serve(weights, rope, reqs, True, **SCHEDULES["alone"])
    for name, kw in SCHEDULES.items():
        if name == "alone":
            continue
        got, tok, st = _serve(weights, rope, reqs, True, **kw)
        if name == "preempting":
            assert st.preempted > 0, "the small cache should preempt"
        if name in ("batch", "slots_4_chunked"):
            assert st.mixed_rows > 0 and st.prefix_hit_tokens > 0
        for s in ref:
            assert torch.equal(got[s], ref[s]), f"{name}: request {s}"
        assert tok == ref_tok, name


def test_speculative_logits_are_batch_invariant(sk, weights):
    """The verify forward computes its rows with the same arithmetic as a decode step, so a
    speculative engine's greedy logits are the plain engine's, bit for bit."""
    rope = L.RoPE(1024)
    g = torch.Generator().manual_seed(13)
    prompts = []
    for n in (12, 40, 7, 90, 25):
        seg = torch.randint(0, VOCAB, (n,), generator=g)
        prompts.append(torch.cat([seg, seg, seg[:n // 2]]))
    news = [int(x) for x in torch.randint(5, 40, (len(prompts),), generator=g)]
    reqs = (prompts, news, [None] * len(prompts))
    ref, ref_tok, _ = _serve(weights, rope, reqs, True, **SCHEDULES["alone"])
    got, tok, st = _serve(weights, rope, reqs, True, max_batch=4, speculative=4)
    assert st.spec_steps > 0
    assert tok == ref_tok
    for s in ref:
        assert torch.equal(got[s], ref[s]), f"request {s}"


def test_batch_invariant_from_the_environment(sk, weights, monkeypatch):
    rope = L.RoPE(1024)
    monkeypatch.setenv("SPARK_BATCH_INVARIANT", "1")
    assert E.SparkModel(weights, rope).batch_invariant
    monkeypatch.setenv("SPARK_BATCH_INVARIANT", "0")
    assert not E.SparkModel(weights, rope).batch_invariant
    assert E.SparkModel(weights, rope, batch_invariant=True).batch_invariant


# The quantized engines: (weights_format, kv_format). fp8-tok and mxfp8 scale their
# activations per row or per 32-k block; fp8 scales them over the whole batch and is refused.
QUANT_CASES = [("int4", "bf16"), ("int4-asym", "fp8"), ("fp8-tok", "fp8"), ("mxfp8", "bf16"),
               ("bf16", "fp8")]


@pytest.mark.parametrize("fmt,kv", QUANT_CASES, ids=[f"{f}+{k}" for f, k in QUANT_CASES])
def test_quantized_engine_logits_are_batch_invariant(sk, weights, fmt, kv):
    rope = L.RoPE(1024)
    reqs = _requests()
    ref, ref_tok, _ = _serve(weights, rope, reqs, True, fmt, kv, **SCHEDULES["alone"])
    for name in ("batch", "batch_32", "slots_4_chunked"):
        got, tok, st = _serve(weights, rope, reqs, True, fmt, kv, **SCHEDULES[name])
        if name != "batch_32":
            assert st.mixed_rows > 0 and st.prefix_hit_tokens > 0
        for s in ref:
            assert torch.equal(got[s], ref[s]), f"{fmt} + {kv} cache, {name}: request {s}"
        assert tok == ref_tok, name


def test_default_quantized_engine_is_not_batch_invariant(sk, weights):
    rope = L.RoPE(1024)
    reqs = _requests()
    ref, _, _ = _serve(weights, rope, reqs, False, "int4", "fp8", **SCHEDULES["alone"])
    got, _, _ = _serve(weights, rope, reqs, False, "int4", "fp8", **SCHEDULES["batch_32"])
    n = min(min(ref[s].shape[0], got[s].shape[0]) for s in ref)
    assert not all(torch.equal(ref[s][:n], got[s][:n]) for s in ref)


def test_batch_invariant_refuses_what_it_does_not_cover(sk, weights):
    rope = L.RoPE(1024)
    for fmt in ("fp8", "nvfp4", "mxfp4"):
        with pytest.raises(ValueError):
            E.SparkModel(weights, rope, weights_format=fmt, keep_bf16=True, batch_invariant=True)
    with pytest.raises(ValueError):
        E.SparkModel(weights, rope, head_format="fp8", keep_bf16=True, batch_invariant=True)
    m = E.SparkModel(weights, rope, weights_format="fp8", keep_bf16=True)
    with pytest.raises(ValueError):
        m.set_batch_invariant(True)
    assert not m.batch_invariant
    m = E.SparkModel(weights, rope, weights_format="fp8-tok", keep_bf16=True)
    m.set_batch_invariant(True)
    assert all(lin.variant == INV_FP8 for d in m.lin for lin in d.values())
    m.set_batch_invariant(False)
    assert all(lin.variant == -1 for d in m.lin for lin in d.values())
