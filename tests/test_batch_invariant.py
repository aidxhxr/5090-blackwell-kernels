"""Batch invariance: a request's logits must not depend on what else is in its steps.

The default engine is deterministic (the same schedule gives the same bits) but not batch
invariant: hgemm picks its kernel, tile and K split from M, paged_decode splits the keys of
the whole batch over the SMs, and a decode row computed by attention_varlen (a mixed step, a
resumed sequence, a verify forward) takes a different reduction than the same row in a decode
step. The tests below show that on the default kernels, then check the invariant ones bit for
bit (torch.equal):

  * hgemm variant 7: row i of A B is the same at every M, whichever kernel M picks;
  * paged_decode variant 2 and attention_varlen variant 2: a query row's output is the same
    whatever the batch around it, and the same from either op;
  * Engine(batch_invariant=True): every request's logits, step by step, are the same alone,
    in a full batch, with its prompt chunked, preempted and resumed, from a CUDA graph or
    eagerly, in any slot, behind a cached prefix, and under speculative decoding.

Two random layers and a 1,024-token vocabulary, as tests/test_engine.py."""

import math

import pytest
import torch

E = pytest.importorskip("spark_kernels.engine")
L = pytest.importorskip("spark_kernels.layer")

N_LAYERS, VOCAB = 2, 1024
INV_GEMM, INV_ATTN = 7, 2  # the invariant variants of hgemm and of both paged attention ops


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


# ---- attention ---------------------------------------------------------------------------

H_Q, H_KV, D, PAGE = 32, 8, 128, 16


def _cache(lens, seed=0):
    """bf16 K/V for sequences of `lens` keys in shuffled pages; returns the caches, the block
    table and the lengths."""
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
    return k, v, bt.cuda(), torch.tensor(lens, dtype=torch.int32, device="cuda")


LENS = [700, 37, 2000, 1, 300, 4100, 256, 257]


def _decode(sk, q, k, v, bt, lens, idx, variant):
    """paged_decode of the sequences `idx` (a batch made of those rows only)."""
    ix = torch.tensor(idx, device="cuda")
    return sk.paged_decode(q[ix].contiguous(), k, v, bt[ix].contiguous(), lens[ix].contiguous(),
                           variant)


def test_default_paged_decode_depends_on_the_batch(sk):
    """Variant 1 cuts the keys of the whole batch into equal ranges per warp, so a sequence
    is cut in other places when other sequences come and go."""
    k, v, bt, lens = _cache(LENS)
    q = torch.randn(len(LENS), H_Q, D, device="cuda").to(torch.bfloat16)
    alone = _decode(sk, q, k, v, bt, lens, [5], 1)
    full = _decode(sk, q, k, v, bt, lens, list(range(len(LENS))), 1)
    assert not torch.equal(alone[0], full[5])


def test_paged_decode_invariant_to_the_batch(sk):
    k, v, bt, lens = _cache(LENS)
    q = torch.randn(len(LENS), H_Q, D, device="cuda").to(torch.bfloat16)
    full = _decode(sk, q, k, v, bt, lens, list(range(len(LENS))), INV_ATTN)
    ref = _decode(sk, q, k, v, bt, lens, list(range(len(LENS))), 0)  # one warp per row
    torch.testing.assert_close(full.float(), ref.float(), atol=2e-2, rtol=2e-2)
    batches = [[b] for b in range(len(LENS))] + [[5, 2], [7, 6, 5, 4, 3, 2, 1, 0],
                                                   [2, 2, 5, 0], list(range(len(LENS))) * 8]
    for idx in batches:
        out = _decode(sk, q, k, v, bt, lens, idx, INV_ATTN)
        for j, b in enumerate(idx):
            assert torch.equal(out[j], full[b]), f"sequence {b} in batch {idx}"


def _varlen(sk, q_rows, k, v, bt, lens, parts, variant):
    """attention_varlen over `parts`, a list of (sequence, first position, last position + 1):
    the query rows q_rows[b][first:end] of each, packed, with seq_lens[b] = end keys."""
    q = torch.cat([q_rows[b][a:e] for b, a, e in parts]).contiguous()
    cu = torch.tensor([0] + [e - a for _, a, e in parts], dtype=torch.int32).cumsum(
        0, dtype=torch.int32).cuda()
    ix = torch.tensor([b for b, _, _ in parts], device="cuda")
    kv = torch.tensor([e for _, _, e in parts], dtype=torch.int32, device="cuda")
    out = sk.attention_varlen(q, k, v, cu, kv, bt[ix].contiguous(), True, variant)
    return out.split([e - a for _, a, e in parts])


def test_varlen_invariant_to_chunking_and_matches_decode(sk):
    """attention_varlen variant 2 gives a query row the bits it gets from paged_decode
    variant 2, whether the row is a decode row, the last row of a prompt, or anywhere in a
    chunk of any size after any context, packed with any other prompts."""
    k, v, bt, lens = _cache(LENS, seed=1)
    g = torch.Generator(device="cpu").manual_seed(2)
    q_rows = [torch.randn(n, H_Q, D, generator=g).to("cuda", torch.bfloat16) for n in LENS]
    # the decode row of every sequence: its last position
    last = torch.stack([q_rows[b][-1] for b in range(len(LENS))])
    dec = _decode(sk, last, k, v, bt, lens, list(range(len(LENS))), INV_ATTN)
    whole = _varlen(sk, q_rows, k, v, bt, lens, [(b, 0, n) for b, n in enumerate(LENS)],
                    INV_ATTN)
    for b in range(len(LENS)):
        assert torch.equal(whole[b][-1], dec[b]), f"sequence {b}: varlen last row vs decode"
    ref = _varlen(sk, q_rows, k, v, bt, lens, [(b, 0, n) for b, n in enumerate(LENS)], 1)
    for b in range(len(LENS)):
        torch.testing.assert_close(whole[b].float(), ref[b].float(), atol=2e-2, rtol=2e-2)
    # the same prompts cut into chunks (each chunk after the keys of the ones before it),
    # packed in other orders and next to one-row chunks
    for cut in (1, 5, 64, 129, 1000):
        for b, n in enumerate(LENS):
            pieces = [(b, a, min(n, a + cut)) for a in range(0, n, cut)][-3:]
            others = [(o, LENS[o] - 1, LENS[o]) for o in (2, 3) if o != b]
            outs = _varlen(sk, q_rows, k, v, bt, lens, others + pieces, INV_ATTN)
            for (_, a, e), o in zip(pieces, outs[len(others):], strict=True):
                assert torch.equal(o, whole[b][a:e]), f"sequence {b}, chunk {a}:{e} of {cut}"


def test_invariant_attention_refuses_fp8(sk):
    k, v, bt, lens = _cache([40])
    k8 = sk.quantize_kv_cache(k, torch.ones(H_KV, device="cuda"))
    v8 = sk.quantize_kv_cache(v, torch.ones(H_KV, device="cuda"))
    one = torch.ones(H_KV, device="cuda")
    q = torch.randn(1, H_Q, D, device="cuda").to(torch.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        sk.paged_decode(q, k8, v8, bt, lens, INV_ATTN, k_scale=one, v_scale=one)


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


def _serve(weights, rope, reqs, invariant, **kw):
    """Each request's logits [generated, VOCAB], one row per token in order, and the stats."""
    prompts, news, params = reqs
    opts = dict(max_batch=8, max_seq=1024, num_pages=256, graphs=True)
    opts.update(kw)
    eng = E.Engine(E.SparkModel(weights, rope), N_LAYERS, log_tokens=True, log_logits=True,
                   batch_invariant=invariant, **opts)
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


def test_batch_invariant_refuses_what_it_does_not_cover(sk, weights):
    rope = L.RoPE(1024)
    with pytest.raises(ValueError):
        E.Engine(E.SparkModel(weights, rope), N_LAYERS, max_batch=1, max_seq=512,
                 num_pages=64, graphs=False, kv_format="fp8", batch_invariant=True)
    with pytest.raises(ValueError):
        E.SparkModel(weights, rope, weights_format="fp8", keep_bf16=True, batch_invariant=True)
