"""sk.sample against its rule: temperature 0, top_k = 1 and a tiny top_p are the argmax; the
two variants give the same token for the same inputs; over many draws the tokens follow
softmax(logits / T) restricted to the top-k, then top-p set (ties at a threshold kept
together), and nothing outside that set ever comes out; the same seed and offset give the
same token whatever the rest of the batch is; and a captured call draws fresh numbers on every
replay, the ones eager calls at the same offsets draw."""

import math

import pytest
import torch

LLAMA_V = 128256


def _variants():
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants("sample")))
    except Exception:
        return [0, 1]


def _params(B, T=1.0, k=0, p=1.0, seed0=0, offset=0):
    dev = "cuda"
    return (torch.full((B,), T, device=dev, dtype=torch.float32),
            torch.full((B,), k, device=dev, dtype=torch.int32),
            torch.full((B,), p, device=dev, dtype=torch.float32),
            torch.arange(seed0, seed0 + B, device=dev, dtype=torch.long),
            torch.full((B,), offset, device=dev, dtype=torch.long))


def _logits(B, V, scale=4.0, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(B, V, device="cuda", generator=g) * scale).to(torch.bfloat16)


def _reference_probs(row: torch.Tensor, T: float, k: int, p: float) -> torch.Tensor:
    """The rule in float64: softmax(x / T) on the tokens at or above the k-th largest value,
    then on those whose mass strictly above them is < p of what top-k kept."""
    x = row.double()
    w = torch.exp((x - x.max()) / T)
    keep = torch.ones_like(x, dtype=torch.bool)
    if 0 < k < x.numel():
        keep &= x >= torch.sort(x, descending=True).values[k - 1]
    if p < 1:
        z = w[keep].sum()
        before = torch.stack([w[keep & (x > xi)].sum() for xi in x])
        keep &= before < p * z
    q = torch.where(keep, w, torch.zeros_like(w))
    return q / q.sum()


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("V", [LLAMA_V, 1000, 1001])
def test_temperature_zero_is_argmax(sk, variant, V):
    B = 8
    x = _logits(B, V)
    x[3, 17] = x[3, 900] = x[3].max() + 1  # a tie at the max: the first index wins
    T, k, p, seed, off = _params(B, T=0.0, k=5, p=0.5)
    tok = sk.sample(x, T, k, p, seed, off, variant)
    assert tok.dtype == torch.long and tok.shape == (B,)
    assert torch.equal(tok, torch.argmax(x, dim=-1))
    assert tok[3].item() == 17
    assert torch.equal(off, torch.ones_like(off))  # advanced by one, greedy rows too


@pytest.mark.parametrize("variant", _variants())
def test_top_k_one_and_tiny_top_p_are_argmax(sk, variant):
    B = 8
    x = _logits(B, LLAMA_V, seed=1)
    cols = torch.randint(0, LLAMA_V, (B,), device="cuda")
    x[torch.arange(B, device="cuda"), cols] = x.max() + 1  # a unique max per row
    for k, p in ((1, 1.0), (0, 1e-6), (1, 0.5)):
        tok = sk.sample(x, *_params(B, T=0.8, k=k, p=p), variant=variant)
        assert torch.equal(tok, cols), (k, p)


@pytest.mark.parametrize("V", [LLAMA_V, 1001])
def test_variants_give_the_same_token(sk, V):
    """Per-row parameters of every kind in one batch: the radix select of variant 1 picks the
    thresholds the bisection of variant 0 does, and the scan over chunks finds the token the
    serial walk does."""
    if sk.num_variants("sample") < 2:
        pytest.skip("one variant")
    B = 16
    x = _logits(B, V, scale=3.0, seed=2)
    dev = "cuda"
    T = torch.tensor([0.0, 1.0, 0.7, 1.3] * 4, device=dev)
    k = torch.tensor([0, 0, 50, 1, 0, 10, 0, 2000, 5, 0, 0, 40, 1, 0, 300, 0],
                     device=dev, dtype=torch.int32)
    p = torch.tensor([1.0, 0.9, 1.0, 1.0, 0.5, 0.95, 0.1, 0.8] * 2, device=dev)
    seed = torch.arange(100, 100 + B, device=dev, dtype=torch.long)
    for start in (0, 7, 123456789):
        off0 = torch.full((B,), start, device=dev, dtype=torch.long)
        off1 = off0.clone()
        a = sk.sample(x, T, k, p, seed, off0, 0)
        b = sk.sample(x, T, k, p, seed, off1, 1)
        assert torch.equal(a, b), start
        assert torch.equal(off0, off1)


CASES = [(1.0, 0, 1.0), (0.7, 10, 1.0), (1.3, 0, 0.8), (0.9, 20, 0.5), (1.0, 8, 0.9)]


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("V", [64, 37])
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"T{c[0]}-k{c[1]}-p{c[2]}")
def test_distribution_matches_the_rule(sk, variant, V, case):
    """20000 draws of one row (one seed per draw): tokens outside the kept set never appear
    and the counts in it pass a chi-square test at a very loose level (6 standard deviations
    of the statistic), bins expected under 5 times pooled."""
    T, k, p = case
    N = 20000
    row = _logits(1, V, scale=2.0, seed=3)
    x = row.expand(N, V).contiguous()
    tok = sk.sample(x, *_params(N, T=T, k=k, p=p, seed0=7), variant=variant)
    counts = torch.bincount(tok, minlength=V).double().cpu()
    q = _reference_probs(row[0].float().cpu(), T, k, p)
    assert counts[q == 0].sum() == 0, "a token outside the top-k / top-p set was drawn"
    exp = q * N
    big = exp >= 5
    obs = torch.cat([counts[big], counts[~big & (q > 0)].sum()[None]])
    want = torch.cat([exp[big], exp[~big & (q > 0)].sum()[None]])
    if want[-1] == 0:
        obs, want = obs[:-1], want[:-1]
    df = max(int(want.numel()) - 1, 1)
    chi2 = float(((obs - want) ** 2 / want).sum())
    assert chi2 < df + 6 * math.sqrt(2 * df) + 5, (chi2, df)


def test_top_p_set_at_full_vocabulary(sk):
    """Llama-width rows with a peaked distribution: every drawn token is at or above the k-th
    value and has less than top_p of the kept mass strictly above it."""
    B, N = 4, 256
    x = _logits(B, LLAMA_V, scale=6.0, seed=4)
    T, k, p = 0.8, 2000, 0.7
    rows = x.repeat_interleave(N, dim=0)
    tok = sk.sample(rows, *_params(B * N, T=T, k=k, p=p)).view(B, N)
    for b in range(B):
        xb = x[b].double()
        w = torch.exp((xb - xb.max()) / T)
        kth = torch.sort(xb, descending=True).values[k - 1]
        kept = xb >= kth
        z = w[kept].sum()
        for t in tok[b].unique().tolist():
            assert xb[t] >= kth
            assert w[kept & (xb > xb[t])].sum() < p * z * (1 + 1e-4)  # __expf, not exp


@pytest.mark.parametrize("variant", _variants())
def test_same_seed_same_tokens_whatever_the_batch(sk, variant):
    B, V = 32, LLAMA_V if variant > 0 else 4096
    x = _logits(B, V, scale=2.0, seed=5)
    T, k, p, seed, off = _params(B, T=1.0, k=100, p=0.9, seed0=50, offset=3)
    a = sk.sample(x, T, k, p, seed, off.clone(), variant)
    b = sk.sample(x, T, k, p, seed, off.clone(), variant)
    assert torch.equal(a, b)
    c = sk.sample(x, T, k, p, seed + 1000, off.clone(), variant)
    assert not torch.equal(a, c)
    d = sk.sample(x, T, k, p, seed, off + 1, variant)
    assert not torch.equal(a, d)
    # rows 5..12 on their own: the same tokens as inside the batch of 32
    s = slice(5, 13)
    e = sk.sample(x[s].contiguous(), T[s].contiguous(), k[s].contiguous(), p[s].contiguous(),
                  seed[s].contiguous(), off[s].clone(), variant)
    assert torch.equal(a[s], e)


def test_graph_replay_draws_fresh_numbers(sk):
    """A captured call reads the offsets on the device and advances them itself: eight
    replays give eight different draws per row, the draws eager calls at the same offsets
    give."""
    B, V = 4, 1024
    x = torch.zeros(B, V, device="cuda", dtype=torch.bfloat16)  # uniform over 1024 tokens
    T, k, p, seed, off = _params(B, T=1.0, seed0=9)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        sk.sample(x, T, k, p, seed, off)  # warm-up: offset 0 -> 1
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = sk.sample(x, T, k, p, seed, off)
    assert torch.equal(off, torch.ones_like(off))  # a capture does not run the kernel
    replays = []
    for _ in range(8):
        g.replay()
        replays.append(out.clone())
    torch.cuda.synchronize()
    assert torch.equal(off, torch.full_like(off, 9))
    replays = torch.stack(replays)  # [8, B]
    for b in range(B):
        assert replays[:, b].unique().numel() > 1
    off.fill_(1)
    eager = torch.stack([sk.sample(x, T, k, p, seed, off) for _ in range(8)])
    assert torch.equal(replays, eager)


def test_sample_rejects_bad_inputs(sk):
    x = _logits(2, 64)
    T, k, p, seed, off = _params(2)
    with pytest.raises(RuntimeError):
        sk.sample(x.float(), T, k, p, seed, off)
    with pytest.raises(RuntimeError):
        sk.sample(x, T, k.long(), p, seed, off)
    with pytest.raises(RuntimeError):
        sk.sample(x, T[:1], k, p, seed, off)
