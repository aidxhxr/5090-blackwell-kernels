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
