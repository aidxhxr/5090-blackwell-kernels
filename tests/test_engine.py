"""The serving engine on a two-layer model with a small vocabulary: SparkModel against
TorchModel on the same weights and inputs (a packed prefill of prompts of different lengths,
then decode steps over the paged cache, teacher-forced so both see the same tokens), and the
engine's CUDA-graph decode against its eager decode token for token, under continuous
batching with more requests than batch slots."""

import pytest
import torch

E = pytest.importorskip("spark_kernels.engine")
L = pytest.importorskip("spark_kernels.layer")

N_LAYERS, VOCAB = 2, 1024


@pytest.fixture(scope="module")
def weights():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    return E.ModelWeights.random(N_LAYERS, VOCAB, seed=1)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def test_spark_model_matches_torch_model(sk, weights):
    rope = L.RoPE(1024)
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in (37, 200, 1, 64)]
    engines, seqs = [], []
    for m in (E.SparkModel(weights, rope), E.TorchModel(weights, rope)):
        eng = E.Engine(m, N_LAYERS, max_batch=8, max_seq=512, num_pages=128, graphs=False)
        for p in prompts:
            eng.submit(p, max_new=8)
        engines.append(eng)
        seqs.append(eng._admit())
    # a packed prefill of four prompts of different lengths into each engine's own cache
    pre = [eng.model.prefill(eng._prefill_batch(s), eng.cache)
           for eng, s in zip(engines, seqs, strict=True)]
    assert pre[0].shape == (4, VOCAB)
    assert _rel(pre[0], pre[1]) < 2e-2
    for ss in seqs:
        for s in ss:
            s.length = s.prompt.numel()
    # teacher-forced decode steps: the same token ids into both, logits compared
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
    rope = L.RoPE(1024)
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
        assert len(eng.cache.free) == eng.cache.num_pages - 1  # every page came back
        outs.append(eng.outputs())
    assert outs[0] == outs[1]
    assert [len(outs[1][i]) for i in range(len(lens))] == news


def test_engine_max_batch_not_a_bucket(sk, weights):
    """Three slots asked for, so decode steps run the bucket of four."""
    eng = E.Engine(E.SparkModel(weights, L.RoPE(512)), N_LAYERS, max_batch=3, max_seq=256,
                   num_pages=64, graphs=False, log_tokens=True)
    for n in (5, 9, 13):
        eng.submit(torch.randint(0, VOCAB, (n,)), max_new=4)
    eng.run()
    assert sorted(len(t) for t in eng.outputs().values()) == [4, 4, 4]


def test_prompt_logits_last_row_is_prefill(sk, weights):
    """prompt_logits' last row of each prompt is what the normal prefill returns."""
    rope = L.RoPE(512)
    eng = E.Engine(E.SparkModel(weights, rope), N_LAYERS, max_batch=4, max_seq=256,
                   num_pages=64, graphs=False)
    prompts = [torch.randint(0, VOCAB, (n,)) for n in (7, 33)]
    free = len(eng.cache.free)
    full = eng.prompt_logits(prompts)
    assert [f.shape for f in full] == [(7, VOCAB), (33, VOCAB)]
    assert len(eng.cache.free) == free and all(s is None for s in eng.running)
    for p in prompts:
        eng.submit(p, 2)
    seqs = eng._admit()
    last = eng.model.prefill(eng._prefill_batch(seqs), eng.cache)
    for f, row in zip(full, last, strict=True):
        assert _rel(f[-1], row) < 1e-2


def test_trace_collects_the_residual_stream(sk, weights):
    """trace holds the residual stream after each layer, the last one is what the logits are
    computed from, and turning it on changes nothing."""
    rope = L.RoPE(1024)
    g = torch.Generator().manual_seed(5)
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in (19, 70)]
    traces = []
    for m in (E.SparkModel(weights, rope), E.TorchModel(weights, rope)):
        eng = E.Engine(m, N_LAYERS, max_batch=2, max_seq=128, num_pages=16, graphs=False)
        plain = torch.cat(eng.prompt_logits(prompts))
        m.trace = []
        traced = torch.cat(eng.prompt_logits(prompts))
        tr, m.trace = m.trace, None
        assert torch.equal(plain, traced)
        assert len(tr) == N_LAYERS and tr[0].shape == (89, L.HIDDEN)
        assert torch.equal(m._logits(tr[-1]), traced)
        traces.append(tr)
    for a, b in zip(*traces, strict=True):
        assert _rel(a, b) < 2e-2


def test_graph_decode_after_workspace_growth(sk, weights):
    """128 slots and prompts of up to 2000 tokens: the prefill GEMMs and the 128-slot step
    run Stream-K schedules that grow the kernels' per-process workspaces after the graphs are
    captured. A graph that kept a freed workspace replays into an illegal address; with the
    workspaces grown first and the Stream-K buckets eager, graphs match eager decode."""
    rope = L.RoPE(4096)
    g = torch.Generator().manual_seed(5)
    lens = [int(x) for x in torch.randint(1, 2000, (150,), generator=g)]
    news = [int(x) for x in torch.randint(1, 24, (150,), generator=g)]
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in lens]
    outs = []
    for graphs in (False, True):
        eng = E.Engine(E.SparkModel(weights, rope), N_LAYERS, max_batch=128, max_seq=4096,
                       num_pages=16384, graphs=graphs, log_tokens=True, prefill_tokens=8192)
        for p, m in zip(prompts, news, strict=True):
            eng.submit(p, m)
        eng.run()
        outs.append(eng.outputs())
    assert outs[0] == outs[1]


def test_compact_keeps_tokens_and_shrinks_steps(sk, weights):
    """Requests of very different output lengths through eight slots: with compaction the
    long ones left in high slots move down, the steps run smaller buckets, and every
    sequence's tokens are the same as without it (graphs on, so a moved slot also has to be
    read correctly by the captured steps)."""
    rope = L.RoPE(1024)
    g = torch.Generator().manual_seed(7)
    lens = [int(x) for x in torch.randint(1, 200, (20,), generator=g)]
    news = [40 if i % 7 == 6 else int(x)
            for i, x in enumerate(torch.randint(1, 8, (20,), generator=g))]
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in lens]
    outs, stats = [], []
    for compact in (False, True):
        eng = E.Engine(E.SparkModel(weights, rope), N_LAYERS, max_batch=8, max_seq=512,
                       num_pages=256, graphs=True, log_tokens=True, compact=compact)
        for p, m in zip(prompts, news, strict=True):
            eng.submit(p, m)
        stats.append(eng.run())
        outs.append(eng.outputs())
        assert len(eng.cache.free) == eng.cache.num_pages - 1
    assert outs[0] == outs[1]
    assert stats[1].moved > 0
    assert stats[1].decode_rows < stats[0].decode_rows
    assert stats[1].decode_tokens == stats[0].decode_tokens
