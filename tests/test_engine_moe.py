"""The serving engine on MoE models: two layers of the engine's attention with experts in
place of the dense MLP, a small vocabulary. SparkModel (MoELayer) against TorchModel
(moe_layer.torch_moe) on the same weights, a packed prefill then teacher-forced decode steps;
the engine's CUDA-graph decode against its eager decode token for token, with prompts long
enough that a prefill runs a grouped GEMM tile no decode bucket uses after the capture; and
what a SparkModel does to the experts' weights it is built on."""

import pytest
import torch
import torch.nn.functional as F

E = pytest.importorskip("spark_kernels.engine")
L = pytest.importorskip("spark_kernels.layer")
ML = pytest.importorskip("spark_kernels.moe_layer")

N_LAYERS, VOCAB = 2, 1024

# A Mixtral-like top-2 of 8 experts, wide enough (inter > 2048) that a prefill past 24 rows
# per expert writes the permuted copy instead of gathering, and a Qwen-like top-8 of 64
# narrow ones, always gathered.
CONFIGS = [ML.MoELayerConfig("mixtral-like", hidden=4096, inter=2304, experts=8, top_k=2),
           ML.MoELayerConfig("qwen-like", hidden=4096, inter=256, experts=64, top_k=8)]

# Both models compute their MLP inputs to within a bf16 rounding of each other, so a token
# whose k-th and k+1-th router logits are that close can go to a different expert in one of
# them, and its logits then differ by the swapped expert's share (40% of a row's norm, seen
# on a top-2 row). TIE is the logit gap that counts as such a tie: the router logits are
# about 1, where a bf16 ulp is 0.0078.
TIE = 0.05


@pytest.fixture(scope="module", params=CONFIGS, ids=lambda c: c.name)
def moe_weights(request):
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    return E.ModelWeights.random(N_LAYERS, VOCAB, seed=2, moe=request.param)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


class _Inputs:
    """Wraps an MoE call and keeps a copy of every input it gets."""

    def __init__(self, fn, log):
        self.fn, self.log = fn, log

    def __call__(self, h, *args, **kw):
        self.log.append(h.clone())
        return self.fn(h, *args, **kw)


def _rerouted(sk, ours, ref, experts):
    """Per token, whether our routing kernel on our MLP inputs picked other experts than
    torch's routing on the reference's, in any layer. A token rerouted in one layer has a
    different hidden state from then on and can go anywhere in the next; each first
    rerouting must be a near tie in the reference (its k-th and k+1-th logits within TIE)."""
    out = None
    for h, hr, e in zip(ours, ref, experts, strict=True):
        cfg = e.cfg
        ids = sk.moe_route(F.linear(h, e.router), cfg.top_k, cfg.renormalize)[0]
        lr = F.linear(hr, e.router).float()
        ids_ref = ML._route_torch(lr, cfg)[1]
        moved = (ids.long().sort(-1).values != ids_ref.sort(-1).values).any(-1)
        top = lr.topk(cfg.top_k + 1, dim=-1).values
        gap = top[:, cfg.top_k - 1] - top[:, cfg.top_k]
        first = moved if out is None else moved & ~out
        assert (gap[first] <= TIE).all(), f"rerouted at gaps {gap[first].tolist()}"
        out = moved if out is None else out | moved
    return out


def _check(sk, outs, logs, experts, rows):
    """The logits of the rows not rerouted agree within 2%; the rerouted rows are near ties
    (_rerouted) and few."""
    moved = _rerouted(sk, logs[0], logs[1], experts)[rows]
    keep = ~moved
    assert keep.sum() >= 3 * keep.numel() // 4, f"{moved.sum().item()} of {moved.numel()} rerouted"
    err = _rel(outs[0][keep], outs[1][keep])
    assert err < 2e-2, err
    for log in logs:
        log.clear()


def test_moe_model_matches_torch_model(sk, moe_weights, monkeypatch):
    rope = L.RoPE(1024)
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in (37, 200, 1, 64)]
    logs = ([], [])
    spark = E.SparkModel(moe_weights, rope, keep_bf16=True)
    spark.moe = [_Inputs(m, logs[0]) for m in spark.moe]
    monkeypatch.setattr(E, "torch_moe", _Inputs(E.torch_moe, logs[1]))
    engines, seqs = [], []
    for m in (spark, E.TorchModel(moe_weights, rope)):
        eng = E.Engine(m, N_LAYERS, max_batch=8, max_seq=512, num_pages=128, graphs=False)
        for p in prompts:
            eng.submit(p, max_new=8)
        engines.append(eng)
        seqs.append(eng._admit())
    batches = [eng._prefill_batch(s) for eng, s in zip(engines, seqs, strict=True)]
    pre = [eng.model.prefill(b, eng.cache) for eng, b in zip(engines, batches, strict=True)]
    assert pre[0].shape == (4, VOCAB)
    _check(sk, pre, logs, moe_weights.experts, batches[1].last)
    for ss in seqs:
        for s in ss:
            s.length = s.prompt.numel()
    # teacher-forced decode steps: the same token ids into both, logits compared
    for _ in range(3):
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
        _check(sk, outs, logs, moe_weights.experts, torch.arange(4, device="cuda"))


def test_moe_graph_decode_matches_eager(sk, moe_weights):
    """Twelve requests through four slots, graphs against eager, the graphed engine first.
    Two prompts are long enough (past 512 rows per expert) that their prefill runs the
    grouped GEMM's 128-row tile, which no captured step uses, after the graphs are captured:
    nothing a graph holds may move."""
    rope = L.RoPE(8192)
    g = torch.Generator().manual_seed(3)
    lens = [5, 90, 17, 300, 1, 44, 128, 9, 250, 33, 4200, 2500]
    news = [6, 3, 9, 4, 12, 1, 5, 7, 2, 8, 6, 5]
    outs = []
    for graphs in (True, False):
        eng = E.Engine(E.SparkModel(moe_weights, rope, keep_bf16=True), N_LAYERS, max_batch=4,
                       max_seq=8192, num_pages=1024, graphs=graphs, log_tokens=True,
                       prefill_tokens=8192)
        assert graphs == bool(eng.graphs)
        g.manual_seed(3)
        for n, m in zip(lens, news, strict=True):
            eng.submit(torch.randint(0, VOCAB, (n,), generator=g), m)
        stats = eng.run()
        assert stats.finished == len(lens) and stats.generated == sum(news)
        assert eng.cache.available() == eng.cache.num_pages - 1  # every page came back
        outs.append(eng.outputs())
    assert outs[0] == outs[1]
    assert [len(outs[1][i]) for i in range(len(lens))] == news


def test_moe_weights_freed_rebuilt_and_refused(sk):
    """A SparkModel frees the checkpoint-layout w13 it interleaved (unless keep_bf16), counts
    the copy it holds, and gives the w13 back in dequantized(), which a TorchModel takes and
    the freed weights are refused by. Quantized weights and batch invariance are refused."""
    cfg = CONFIGS[1]
    rope = L.RoPE(256)
    w = E.ModelWeights.random(1, VOCAB, seed=4, moe=cfg)
    w13 = [e.w13.clone() for e in w.experts]
    total = w.bytes()
    m = E.SparkModel(w, rope)
    assert all(e.w13 is None for e in w.experts)
    assert m.weight_bytes() == total
    assert w.bytes() == total - sum(t.numel() * 2 for t in w13)
    back = m.dequantized()
    assert all(torch.equal(a, e.w13) for a, e in zip(w13, back.experts, strict=True))
    with pytest.raises(ValueError):
        E.TorchModel(w, rope)
    E.TorchModel(back, rope)
    with pytest.raises(ValueError):
        E.SparkModel(back, rope, batch_invariant=True)
    with pytest.raises(ValueError):
        E.SparkModel(back, rope, weights_format="int4")
    with pytest.raises(ValueError):
        E.Engine(E.SparkModel(back, rope, keep_bf16=True), 1, max_batch=2, max_seq=128,
                 num_pages=16, graphs=False, batch_invariant=True)
    with pytest.raises(ValueError):
        E.ModelWeights.random(1, VOCAB, moe=ML.MoELayerConfig("x", 2048, 256, 8, 2))
