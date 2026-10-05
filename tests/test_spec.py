"""CPU tests of the prompt-lookup drafter and the acceptance rule (spark_kernels/spec.py),
loaded from its path so they run without the compiled extension or a GPU. The last test
runs whole speculative decodes against a fake model with a paged-cache-like store, where
rejected rows are written past the sequence's length and must never be read."""

import importlib.util
import random
from pathlib import Path

import pytest
import torch

_path = Path(__file__).resolve().parent.parent / "python" / "spark_kernels" / "spec.py"
_spec = importlib.util.spec_from_file_location("spark_spec", _path)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def test_lookup_most_recent_longest_match():
    ctx = [1, 2, 3, 9, 1, 2, 3, 7, 8, 5, 1, 2, 3]
    # the 3-gram 1 2 3 occurs twice before the end; the latest one is followed by 7 8 5
    assert S.lookup(ctx, 2) == [7, 8]
    assert S.lookup(ctx, 10) == [7, 8, 5, 1, 2, 3]  # runs to the end of the context
    # no 3-gram or 2-gram match, the 1-gram 4 matches
    assert S.lookup([4, 6, 0, 5, 4], 3) == [6, 0, 5]
    # a longer match wins over a more recent shorter one
    assert S.lookup([1, 2, 3, 9, 7, 3, 8, 1, 2, 3], 1) == [9]
    assert S.lookup([1, 2, 3, 9, 7, 3, 8, 1, 2, 3], 1, n_max=1) == [8]
    # a match may overlap the suffix
    assert S.lookup([5, 5, 5, 5], 4) == [5]
    assert S.lookup([1, 2, 3], 4) == []
    assert S.lookup([], 4) == [] and S.lookup([7], 4) == []
    assert S.lookup(ctx, 0) == []
    assert S.lookup([1, 2, 3, 1, 2, 3], 3, n_min=4, n_max=4) == []


@pytest.mark.parametrize("n_max,n_min", [(3, 1), (2, 2), (4, 2), (1, 1)])
def test_drafter_matches_lookup_as_the_context_grows(n_max, n_min):
    rng = random.Random(n_max * 10 + n_min)
    for vocab in (2, 5, 50):
        d = S.NgramDrafter([rng.randrange(vocab) for _ in range(rng.randrange(0, 6))],
                           n_max=n_max, n_min=n_min)
        for _ in range(150):
            for k in (1, 3, 8):
                assert d.draft(k) == S.lookup(d.context, k, n_max, n_min)
            d.extend(rng.randrange(vocab) for _ in range(rng.randrange(0, 5)))


def test_drafter_refuses_bad_n():
    with pytest.raises(ValueError):
        S.NgramDrafter(n_max=1, n_min=2)
    with pytest.raises(ValueError):
        S.NgramDrafter(n_max=3, n_min=0)


def test_accept():
    assert S.accept([], [9]) == [9]  # no draft: the plain decode token
    assert S.accept([4, 5, 6], [4, 5, 6, 7]) == [4, 5, 6, 7]  # all accepted plus the bonus
    assert S.accept([4, 5, 6], [4, 0, 6, 7]) == [4, 0]  # the first disagreement ends it
    assert S.accept([4, 5, 6], [1, 5, 6, 7]) == [1]
    # a later row that agrees does not count after a rejection
    assert S.accept([4, 5, 6, 8], [4, 9, 6, 8, 3]) == [4, 9]
    with pytest.raises(ValueError):
        S.accept([1, 2], [1, 2])


def test_cut_at_stop():
    assert S.cut_at_stop([1, 2, 3], {9}) == ([1, 2, 3], False)
    assert S.cut_at_stop([1, 9, 3, 9], {9}) == ([1, 9], True)
    assert S.cut_at_stop([9], {9, 4}) == ([9], True)
    assert S.cut_at_stop([], {9}) == ([], False)


class FakeModel:
    """A next-token rule on the keys a row can see: the logits of the row at position p are
    a function of the cache entries 0..p, read from a store where entries past the
    sequence's length may hold rejected drafts. One logit is boosted so the argmax is
    unambiguous. `loop` makes the rule periodic in the recent tokens, so prompt lookup gets
    accepted; otherwise it hashes a longer window and drafts mostly miss."""

    def __init__(self, vocab: int, loop: bool):
        self.vocab, self.loop = vocab, loop

    def next_token(self, keys: list[int]) -> int:
        if self.loop:
            return (keys[-1] * 7 + (keys[-2] if len(keys) > 1 else 0)) % 11
        return hash(tuple(keys[-6:])) % self.vocab

    def logits(self, keys: list[int]) -> torch.Tensor:
        g = torch.Generator().manual_seed(len(keys))
        x = torch.rand(self.vocab, generator=g)
        x[self.next_token(keys)] += 2.0
        return x


def greedy(model: FakeModel, prompt: list[int], n: int) -> list[int]:
    ctx = list(prompt)
    out = []
    for _ in range(n):
        t = int(torch.argmax(model.logits(ctx)))
        out.append(t)
        ctx.append(t)
    return out


def speculative(model: FakeModel, prompt: list[int], n: int, k: int):
    """The engine's loop on the host: a store indexed by position (the cache), `length`
    entries valid; the verify forward writes rows at length..length+r-1, each row's logits
    see store[0..its position]; length advances by the accepted count only."""
    store = list(prompt) + [None] * (n + k + 2)
    out = [int(torch.argmax(model.logits(list(prompt))))]  # the prefill's token
    last = out[0]  # the next input row, not cached yet
    length = len(prompt)
    drafter = S.NgramDrafter(prompt + out)
    proposed = accepted = forwards = 0
    while len(out) < n:
        d = drafter.draft(min(k, n - len(out) - 1))
        rows = [last] + d
        for i, t in enumerate(rows):  # the forward appends every row's K/V
            store[length + i] = t
        lg = torch.stack([model.logits(store[:length + i + 1]) for i in range(len(rows))])
        new = S.accept(d, torch.argmax(lg, dim=-1).tolist())
        proposed += len(d)
        accepted += len(new) - 1
        forwards += 1
        length += len(new)  # last and the accepted drafts are now cached, the rest is stale
        out += new
        last = new[-1]
        drafter.extend(new)
        assert store[:length] == prompt + out[:-1]
    return out, proposed, accepted, forwards


@pytest.mark.parametrize("loop", [True, False])
@pytest.mark.parametrize("k", [1, 4, 8])
def test_speculative_loop_gives_greedy_tokens(loop, k):
    model = FakeModel(64, loop)
    rng = random.Random(k)
    for _ in range(4):
        prompt = [rng.randrange(64) for _ in range(rng.randrange(1, 30))]
        n = rng.randrange(1, 60)
        want = greedy(model, prompt, n)
        got, proposed, accepted, forwards = speculative(model, prompt, n, k)
        assert got == want
        assert len(got) == n
        assert forwards == n - 1 - accepted  # each forward gives 1 + its accepted tokens
        if loop and n > 30:
            assert accepted > 0
