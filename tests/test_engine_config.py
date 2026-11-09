"""The serving engine on shapes other than Llama-3-8B's (engine.ModelConfig): two layers of
a dense model with Qwen3-30B-A3B's attention (hidden 2048, 32 query heads over 4 K/V heads,
so the q projection is 2048 -> 4096 and 8 query heads share a K/V head) and of one with
64-wide heads in groups of 8, without Qwen3's QK norm and with it (Qwen3's own config has
it). SparkModel against TorchModel on the same weights (a packed prefill, then teacher-forced
decode steps) on a bf16 and an e4m3 cache, the engine's graphed decode against its eager
decode token for token, and the shapes the kernels refuse. The MoE Qwen3-30B-A3B shape (its
128 experts on this attention) runs in tests/test_engine_moe.py."""

import dataclasses

import pytest
import torch

E = pytest.importorskip("spark_kernels.engine")
L = pytest.importorskip("spark_kernels.layer")

N_LAYERS, VOCAB = 2, 1024

CONFIGS = [E.QWEN3_30B_A3B,
           E.ModelConfig("d64-g8", hidden=1024, n_heads=16, n_kv_heads=2, head_dim=64,
                         intermediate=2816, eps=1e-6),
           E.ModelConfig("d64-g8-qk", hidden=1024, n_heads=16, n_kv_heads=2, head_dim=64,
                         intermediate=2816, eps=1e-6, qk_norm=True)]


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda c: c.name)
def weights(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    return E.ModelWeights.random(N_LAYERS, VOCAB, seed=7, config=request.param)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _rope(w, n):
    return L.RoPE(n, head_dim=w.config.head_dim)


@pytest.mark.parametrize("kv_format", ["bf16", "fp8"])
def test_spark_model_matches_torch_model(sk, weights, kv_format):
    c = weights.config
    rope = _rope(weights, 1024)
    g = torch.Generator().manual_seed(0)
    kw = {}
    if kv_format == "fp8":  # non-unit scales per layer and head, the same in both caches
        kw["kv_scales"] = tuple((0.5 + torch.rand(N_LAYERS, c.n_kv_heads, generator=g)).cuda()
                                for _ in range(2))
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in (37, 200, 1, 64)]
    engines, seqs = [], []
    for m in (E.SparkModel(weights, rope), E.TorchModel(weights, rope)):
        eng = E.Engine(m, N_LAYERS, max_batch=8, max_seq=512, num_pages=128, graphs=False,
                       kv_format=kv_format, **kw)
        assert eng.cache.k.shape[2:] == (c.n_kv_heads, eng.page, c.head_dim)
        for p in prompts:
            eng.submit(p, max_new=8)
        engines.append(eng)
        seqs.append(eng._admit())
    pre = [eng.model.prefill(eng._prefill_batch(s), eng.cache)
           for eng, s in zip(engines, seqs, strict=True)]
    assert pre[0].shape == (4, VOCAB)
    assert _rel(pre[0], pre[1]) < 2e-2
    for ss in seqs:
        for s in ss:
            s.length = s.prompt.numel()
    for step in range(3):
        toks = torch.randint(0, VOCAB, (4,), generator=g).cuda()
        outs = []
        for eng, ss in zip(engines, seqs, strict=True):
            pos = [s.length for s in ss]
            eng.ids[:4] = toks
            eng.positions[:4] = torch.tensor(pos, dtype=torch.int32)
            slots = [s.pages[p // eng.page] * eng.page + p % eng.page
                     for s, p in zip(ss, pos, strict=True)]
            eng.slots[:4] = torch.tensor(slots, dtype=torch.int32)
            eng.seq_lens[:4] = torch.tensor([p + 1 for p in pos], dtype=torch.int32)
            outs.append(eng.model.decode(eng._decode_batch(4, max(pos) + 1), eng.cache))
            for s in ss:
                s.length += 1
        assert _rel(outs[0], outs[1]) < 2e-2, f"step {step}: {_rel(outs[0], outs[1])}"


def test_graph_decode_matches_eager(sk, weights):
    """Ten requests through four slots with mixed steps and chunked prompts, graphs against
    eager, every page back at the end."""
    rope = _rope(weights, 1024)
    g = torch.Generator().manual_seed(3)
    lens = [5, 90, 17, 300, 1, 44, 128, 9, 250, 33]
    news = [6, 3, 9, 4, 12, 1, 5, 7, 2, 8]
    outs = []
    for graphs in (False, True):
        eng = E.Engine(E.SparkModel(weights, rope), N_LAYERS, max_batch=4, max_seq=512,
                       num_pages=160, graphs=graphs, log_tokens=True, prefill_tokens=256)
        g.manual_seed(3)
        for n, m in zip(lens, news, strict=True):
            eng.submit(torch.randint(0, VOCAB, (n,), generator=g), m)
        stats = eng.run()
        assert stats.finished == len(lens) and stats.generated == sum(news)
        assert eng.cache.available() == eng.cache.num_pages - 1
        outs.append(eng.outputs())
    assert outs[0] == outs[1]
    assert [len(outs[1][i]) for i in range(len(lens))] == news


def test_default_is_llama3_8b(sk):
    """The default config is layer.py's shape, and the cache it sizes is the one the engine
    always had: [n_layers, pages, 8, page, 128], 128 KiB a token over 32 layers."""
    c = E.LLAMA3_8B
    assert (c.hidden, c.n_heads, c.n_kv_heads, c.head_dim, c.intermediate, c.eps) == (
        L.HIDDEN, L.N_HEADS, L.N_KV_HEADS, L.HEAD_DIM, L.INTERMEDIATE, L.EPS)
    assert (c.q_width, c.kv_width, c.qkv_width) == (L.Q_WIDTH, L.KV_WIDTH, L.QKV_WIDTH)
    assert E.PagedKVCache.page_bytes(32, 1) == 128 * 1024
    assert E.PagedKVCache(1, 2).k.shape == (1, 2, 8, E.PAGE, 128)
    q = E.QWEN3_30B_A3B
    assert (q.q_width, q.kv_width, q.qkv_width) == (4096, 512, 5120)
    assert E.PagedKVCache.page_bytes(48, 1, config=q) == 96 * 1024


def test_refused_shapes(sk):
    """What a kernel cannot take is refused when the model is built, with the reason."""
    for bad, why in ((dict(head_dim=96), "head_dim"), (dict(n_heads=32, n_kv_heads=1), "16"),
                     (dict(hidden=1000), "multiples of 64"), (dict(n_kv_heads=3), "multiple")):
        cfg = E.ModelConfig("bad", **bad)
        with pytest.raises(ValueError, match=why):
            cfg.check()
    w = E.ModelWeights.random(1, 256, seed=1, config=CONFIGS[1])
    with pytest.raises(ValueError, match="head_dim"):
        E.SparkModel(w, L.RoPE(64))  # 128-wide tables for 64-wide heads
    with pytest.raises(ValueError, match="head_dim"):
        E.TorchModel(w, L.RoPE(64))
    E.SparkModel(w, L.RoPE(64, head_dim=64))
    with pytest.raises(ValueError, match="hidden"):
        E.ModelWeights.random(1, 256, config=CONFIGS[1],
                              moe=E.MoELayerConfig("x", 2048, 256, 8, 2))


def _prefill(model, prompts):
    eng = E.Engine(model, N_LAYERS, max_batch=8, max_seq=512, num_pages=128, graphs=False)
    for p in prompts:
        eng.submit(p, max_new=1)
    return eng.model.prefill(eng._prefill_batch(eng._admit()), eng.cache)


def test_qk_norm(sk):
    """Qwen3's config has the QK norm and its random weights carry it, drawn after the rest,
    which is the same for a seed with it or without. The norm changes the logits, and a model
    whose config wants it refuses weights without it."""
    c = E.QWEN3_30B_A3B
    assert c.qk_norm and not E.LLAMA3_8B.qk_norm
    on = E.ModelWeights.random(N_LAYERS, VOCAB, seed=7, config=c)
    off = E.ModelWeights.random(N_LAYERS, VOCAB, seed=7,
                                config=dataclasses.replace(c, qk_norm=False))
    for a, b in zip(on.layers, off.layers, strict=True):
        assert a.q_norm.shape == a.k_norm.shape == (c.head_dim,)
        assert b.q_norm is None and b.k_norm is None
        assert torch.equal(a.w_qkv, b.w_qkv) and torch.equal(a.w_down, b.w_down)
    assert torch.equal(on.lm_head, off.lm_head) and torch.equal(on.embed, off.embed)
    rope = _rope(on, 1024)
    g = torch.Generator().manual_seed(1)
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in (37, 200)]
    a, b = _prefill(E.SparkModel(on, rope), prompts), _prefill(E.SparkModel(off, rope), prompts)
    assert _rel(a, b) > 2e-2  # the tolerance the models are held to above
    bad = dataclasses.replace(off, config=c)
    with pytest.raises(ValueError, match="q_norm"):
        E.SparkModel(bad, rope)
    with pytest.raises(ValueError, match="q_norm"):
        E.TorchModel(bad, rope)
