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


def test_mixed_step_same_tokens(sk, weights):
    """Decode rows inside the prefill forward (a one-token chunk after each running
    sequence's context, through attention_varlen) against the separate prefill and decode
    steps: the same tokens for every request, in fewer forwards."""
    rope = L.RoPE(1024)
    g = torch.Generator().manual_seed(11)
    lens = [int(x) for x in torch.randint(1, 300, (12,), generator=g)]
    news = [int(x) for x in torch.randint(1, 10, (12,), generator=g)]
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in lens]
    outs, stats = [], []
    for mixed in (False, True):
        eng = E.Engine(E.SparkModel(weights, rope), N_LAYERS, max_batch=4, max_seq=512,
                       num_pages=160, graphs=True, log_tokens=True, prefill_tokens=256,
                       mixed=mixed)
        for p, m in zip(prompts, news, strict=True):
            eng.submit(p, m)
        stats.append(eng.run())
        outs.append(eng.outputs())
        assert len(eng.cache.free) == eng.cache.num_pages - 1
    assert outs[0] == outs[1]
    assert [len(outs[1][i]) for i in range(len(lens))] == news
    assert stats[1].mixed_rows > 0 and stats[0].mixed_rows == 0
    assert (stats[1].decode_steps + stats[1].prefill_batches <
            stats[0].decode_steps + stats[0].prefill_batches)


def test_stop_ids(sk, weights):
    """With a stop id each sequence ends at its first occurrence of it: the outputs are the
    unstopped run's outputs cut there, the sequences that stopped early are counted, and
    every page comes back. The stop check reads the tokens behind the GPU, so this also
    covers a sequence that ran a few tokens past its stop before it retired."""
    rope = L.RoPE(1024)
    g = torch.Generator().manual_seed(13)
    lens = [int(x) for x in torch.randint(1, 200, (10,), generator=g)]
    news = [int(x) for x in torch.randint(8, 40, (10,), generator=g)]
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in lens]

    def run(stop_ids):
        eng = E.Engine(E.SparkModel(weights, rope), N_LAYERS, max_batch=4, max_seq=512,
                       num_pages=160, graphs=True, log_tokens=True, stop_ids=stop_ids)
        for p, m in zip(prompts, news, strict=True):
            eng.submit(p, m)
        st = eng.run()
        assert len(eng.cache.free) == eng.cache.num_pages - 1
        return eng.outputs(), st

    full, _ = run(None)
    counts = torch.bincount(torch.tensor([t for o in full.values() for t in o[1:]]))
    stop = int(counts.argmax())
    want = {sid: o[:o.index(stop) + 1] if stop in o else o for sid, o in full.items()}
    got, st = run([stop])
    assert got == want
    early = sum(len(want[i]) < news[i] for i in range(len(news)))
    assert early > 0 and st.stopped == early


def test_chunked_prefill_matches_whole(sk, weights):
    """Prompts prefilled in forwards of 96 tokens (split across forwards, packed with the
    next prompt's head) give the logits of one whole forward, and a chunked run gives every
    page back."""
    rope = L.RoPE(2048)
    g = torch.Generator().manual_seed(5)
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in (300, 37, 1000, 5)]
    out = []
    for budget in (4096, 96):
        eng = E.Engine(E.SparkModel(weights, rope), N_LAYERS, max_batch=4, max_seq=1100,
                       num_pages=300, graphs=False, prefill_tokens=budget)
        chunks = list(eng.prompt_logits_chunks(prompts))
        assert all(lg.shape[0] <= budget for _, _, lg in chunks)
        out.append(eng.prompt_logits(prompts))
        assert len(eng.cache.free) == eng.cache.num_pages - 1
    assert len(chunks) > len(prompts)
    for a, b in zip(*out, strict=True):
        assert a.shape == b.shape and _rel(b, a) < 2e-2


def test_chunked_prefill_with_mixed_steps(sk, weights):
    """Long prompts chunked under a 96-token budget while other sequences decode (their
    rows ride in the first chunk's forward of a mixed step): the same tokens as whole-prompt
    prefills with separate decode steps, and every page comes back."""
    rope = L.RoPE(1024)
    g = torch.Generator().manual_seed(17)
    lens = [int(x) for x in torch.randint(1, 400, (10,), generator=g)]
    news = [int(x) for x in torch.randint(1, 10, (10,), generator=g)]
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in lens]
    outs, stats = [], []
    for budget, mixed in ((4096, False), (96, True)):
        eng = E.Engine(E.SparkModel(weights, rope), N_LAYERS, max_batch=4, max_seq=512,
                       num_pages=160, graphs=True, log_tokens=True, prefill_tokens=budget,
                       mixed=mixed)
        for p, m in zip(prompts, news, strict=True):
            eng.submit(p, m)
        stats.append(eng.run())
        outs.append(eng.outputs())
        assert len(eng.cache.free) == eng.cache.num_pages - 1
    assert outs[0] == outs[1]
    assert stats[1].mixed_rows > 0 and stats[1].prefill_batches > stats[0].prefill_batches


def test_engine_refuses_rope_shorter_than_max_seq(sk, weights):
    with pytest.raises(ValueError):
        E.Engine(E.SparkModel(weights, L.RoPE(512)), N_LAYERS, max_batch=1, max_seq=1024,
                 num_pages=80, graphs=False)


def test_preemption_keeps_tokens_in_a_small_cache(sk, weights):
    """More requests than the cache can hold at their full lengths: pages are taken as the
    tokens come, sequences are preempted when they run out and prefilled again from their
    tokens so far, and every request ends with the tokens it gets from a cache that holds
    everything (graphs on, so the token record the preemption reads is the captured step's).
    Every page comes back, and the small cache really did preempt."""
    rope = L.RoPE(1024)
    g = torch.Generator().manual_seed(23)
    lens = [int(x) for x in torch.randint(1, 120, (16,), generator=g)]
    news = [int(x) for x in torch.randint(20, 90, (16,), generator=g)]
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in lens]
    outs, stats = [], []
    for pages in (1024, 48):
        eng = E.Engine(E.SparkModel(weights, rope), N_LAYERS, max_batch=8, max_seq=512,
                       num_pages=pages, graphs=True, log_tokens=True)
        for p, m in zip(prompts, news, strict=True):
            eng.submit(p, m)
        stats.append(eng.run())
        outs.append(eng.outputs())
        assert len(eng.cache.free) == eng.cache.num_pages - 1
    assert outs[0] == outs[1]
    assert [len(outs[1][i]) for i in range(len(lens))] == news
    assert stats[0].preempted == 0 and stats[1].preempted > 0
    assert stats[1].recomputed_tokens > 0
    assert stats[1].finished == stats[0].finished == len(lens)


def test_preemption_with_stop_ids_and_mixed_steps(sk, weights):
    """The same under stop ids (a stopped sequence may be preempted before its stop is
    seen) and with decode rows riding in prefill forwards: the outputs match the big
    cache's, cut at the stop token."""
    rope = L.RoPE(1024)
    g = torch.Generator().manual_seed(29)
    lens = [int(x) for x in torch.randint(1, 100, (12,), generator=g)]
    news = [int(x) for x in torch.randint(30, 80, (12,), generator=g)]
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in lens]

    def run(pages, stop_ids):
        eng = E.Engine(E.SparkModel(weights, rope), N_LAYERS, max_batch=4, max_seq=512,
                       num_pages=pages, graphs=True, log_tokens=True, stop_ids=stop_ids,
                       prefill_tokens=128)
        for p, m in zip(prompts, news, strict=True):
            eng.submit(p, m)
        st = eng.run()
        assert len(eng.cache.free) == eng.cache.num_pages - 1
        return eng.outputs(), st

    full, _ = run(1024, None)
    counts = torch.bincount(torch.tensor([t for o in full.values() for t in o[1:]]))
    stop = int(counts.argmax())
    want = {sid: o[:o.index(stop) + 1] if stop in o else o for sid, o in full.items()}
    got, st = run(32, [stop])
    assert got == want
    assert st.preempted > 0 and st.finished == len(lens)


def test_reserve_mode_never_preempts(sk, weights):
    """preempt=False is the old scheduler: every page of prompt + max_new reserved at
    admission, so a cache that holds two requests at a time runs them two at a time with
    no preemption, and a request the cache cannot hold is refused at submit."""
    rope = L.RoPE(1024)
    eng = E.Engine(E.SparkModel(weights, rope), N_LAYERS, max_batch=8, max_seq=1024,
                   num_pages=33, graphs=False, log_tokens=True, preempt=False)
    for n in (100, 100, 100, 100):
        eng.submit(torch.randint(0, VOCAB, (n,)), 140)  # 15 pages each, 32 to give
    st = eng.run()
    assert st.preempted == 0 and st.finished == 4
    assert st.decode_rows == st.decode_tokens  # two at a time in the 2-bucket
    with pytest.raises(ValueError):
        eng.submit(torch.randint(0, VOCAB, (400,)), 120)  # 33 pages: never fits beside scratch


def _serve(weights, rope, prompts, news, oracle=None, **kw):
    """Runs the requests through an engine (graphs on, every page checked back at the end);
    with `oracle` (sid -> the plain run's tokens) the speculative engine's drafts are the
    true next tokens with some of them changed, in place of prompt lookup."""
    eng = E.Engine(E.SparkModel(weights, rope), N_LAYERS, graphs=True, log_tokens=True, **kw)
    if oracle is not None:
        def draft(seq, k):
            g = seq.generated  # the tokens so far are oracle[sid][:g]
            if k < 1 or g % 5 == 0:
                return []  # some steps have sequences without a draft (paged_decode rows)
            d = list(oracle[seq.sid][g:g + k])
            if g % 3 == 1:  # a wrong token somewhere: partial and zero acceptance
                i = g % len(d)
                d[i] = (d[i] + 1) % VOCAB
            return d

        eng._draft = draft
    for p, m in zip(prompts, news, strict=True):
        eng.submit(p, m)
    st = eng.run()
    assert len(eng.cache.free) == eng.cache.num_pages - 1
    return eng.outputs(), st


def test_speculative_same_tokens(sk, weights):
    """Prompt-lookup speculation gives the plain engine's tokens. The prompts repeat a random
    segment so the drafter finds matches; whether the model agrees with them does not matter
    to the check (random weights rarely copy), only that every kept token is greedy's."""
    rope = L.RoPE(1024)
    g = torch.Generator().manual_seed(31)
    prompts = []
    for n in (12, 40, 7, 90, 25, 3):
        seg = torch.randint(0, VOCAB, (n,), generator=g)
        prompts.append(torch.cat([seg, seg, seg[:n // 2]]))
    news = [int(x) for x in torch.randint(5, 60, (len(prompts),), generator=g)]
    kw = dict(max_batch=4, max_seq=512, num_pages=160, prefill_tokens=256)
    want, plain = _serve(weights, rope, prompts, news, **kw)
    for k in (4, 8):
        got, st = _serve(weights, rope, prompts, news, speculative=k, **kw)
        assert got == want
        assert st.spec_proposed > 0 and st.spec_steps > 0
        assert st.generated == plain.generated == sum(news)
        assert st.forwards == st.prefill_batches + st.decode_steps + st.spec_steps


def test_speculative_accept_and_rollback(sk, weights):
    """Drafts that are the true next tokens, some changed: the accepted prefix, the bonus
    token and the rollback of the rejected rows (their K/V stay past the length and get
    overwritten) give the plain engine's tokens, in fewer forwards."""
    rope = L.RoPE(1024)
    g = torch.Generator().manual_seed(37)
    lens = [int(x) for x in torch.randint(1, 200, (8,), generator=g)]
    news = [int(x) for x in torch.randint(10, 70, (8,), generator=g)]
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in lens]
    kw = dict(max_batch=4, max_seq=512, num_pages=160, prefill_tokens=256)
    want, plain = _serve(weights, rope, prompts, news, **kw)
    got, st = _serve(weights, rope, prompts, news, oracle=want, speculative=6, **kw)
    assert got == want
    assert 0 < st.spec_accepted < st.spec_proposed
    assert st.forwards < plain.forwards
    assert st.generated == sum(news)


def test_speculative_preemption_and_stop_ids(sk, weights):
    """Speculation in a cache small enough to preempt (each step takes pages for its k + 1
    rows first), under stop ids (a stop inside the accepted tokens cuts the rest): the plain
    big-cache run's tokens cut at the stop."""
    rope = L.RoPE(1024)
    g = torch.Generator().manual_seed(41)
    lens = [int(x) for x in torch.randint(1, 100, (12,), generator=g)]
    news = [int(x) for x in torch.randint(30, 80, (12,), generator=g)]
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in lens]
    kw = dict(max_batch=4, max_seq=512, prefill_tokens=128)
    full, _ = _serve(weights, rope, prompts, news, num_pages=1024, **kw)
    counts = torch.bincount(torch.tensor([t for o in full.values() for t in o[1:]]))
    stop = int(counts.argmax())
    want = {sid: o[:o.index(stop) + 1] if stop in o else o for sid, o in full.items()}
    got, st = _serve(weights, rope, prompts, news, oracle=full, num_pages=32,
                     stop_ids=[stop], speculative=8, **kw)
    assert got == want
    assert st.preempted > 0 and st.finished == len(lens) and st.spec_accepted > 0
    got, st = _serve(weights, rope, prompts, news, num_pages=32, stop_ids=[stop],
                     speculative=4, **kw)
    assert got == want


def test_speculative_refuses_torch_model(sk, weights):
    with pytest.raises(ValueError):
        E.Engine(E.TorchModel(weights, L.RoPE(512)), N_LAYERS, max_batch=1, max_seq=256,
                 num_pages=16, graphs=False, speculative=4)
