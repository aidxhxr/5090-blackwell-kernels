"""Prompt-lookup (n-gram) speculative decoding: the drafter and the acceptance rule, in plain
Python on host token lists, so they run and are tested without a GPU (tests/test_spec.py).

The drafter looks for the last n tokens of a sequence's context (prompt and generated
tokens), n from n_max down to n_min, earlier in the same context, and proposes the tokens
that followed the most recent earlier match. No draft model: on a prompt the output copies
from (a summary quoting its source, an edit of code in the prompt) the proposal is often
what the model would have produced anyway.

The engine verifies the k proposed tokens d1..dk in one forward over the rows
[last, d1, .., dk]: row i's argmax is the model's next token after row i. `accept` keeps
the longest prefix of the draft that agrees with those argmaxes and adds the argmax of the
row after it (the bonus token), so a step yields between 1 and k + 1 tokens and they are
exactly the tokens greedy decoding one at a time would have given.

This module imports nothing from the package (no torch, no extension), so
`importlib` can load it from its path on a machine without CUDA.
"""

from __future__ import annotations

from collections.abc import Iterable

NGRAM_MAX = 3
NGRAM_MIN = 1


def lookup(context: list[int], k: int, n_max: int = NGRAM_MAX,
           n_min: int = NGRAM_MIN) -> list[int]:
    """The reference drafter, a plain scan: for n from n_max down to n_min, the last n tokens
    of `context` are searched for earlier in it (a match ends before the last token, and may
    overlap the suffix), and the up to k tokens after the most recent match are returned.
    The first n that has a match wins. [] when nothing matches or k < 1."""
    L = len(context)
    if k < 1:
        return []
    for n in range(min(n_max, L - 1), n_min - 1, -1):
        suffix = context[L - n:]
        for end in range(L - 2, n - 2, -1):  # end index of a candidate match
            if context[end - n + 1:end + 1] == suffix:
                return context[end + 1:end + 1 + k]
    return []


class NgramDrafter:
    """`lookup` with an index, for a context that only grows: every n-gram (n_min..n_max)
    that ends before the last token maps to the latest position it ends at, so a draft is
    n_max - n_min + 1 dictionary lookups however long the context is, and appending a token
    indexes n_max - n_min + 1 new n-grams. Gives the same drafts as `lookup`."""

    def __init__(self, context: Iterable[int] = (), n_max: int = NGRAM_MAX,
                 n_min: int = NGRAM_MIN):
        if not 1 <= n_min <= n_max:
            raise ValueError("need 1 <= n_min <= n_max")
        self.n_max, self.n_min = n_max, n_min
        self.context: list[int] = []
        self.last: dict[tuple[int, ...], int] = {}  # n-gram -> the latest index it ends at
        self._indexed = 0  # n-grams ending at indices below this are in `last`
        self.extend(context)

    def __len__(self) -> int:
        return len(self.context)

    def extend(self, tokens: Iterable[int]) -> None:
        ctx = self.context
        ctx.extend(int(t) for t in tokens)
        # index every n-gram ending before the (new) last token
        for end in range(self._indexed, len(ctx) - 1):
            for n in range(self.n_min, min(self.n_max, end + 1) + 1):
                self.last[tuple(ctx[end - n + 1:end + 1])] = end
        self._indexed = max(self._indexed, len(ctx) - 1)

    def draft(self, k: int) -> list[int]:
        ctx = self.context
        L = len(ctx)
        if k < 1:
            return []
        for n in range(min(self.n_max, L - 1), self.n_min - 1, -1):
            end = self.last.get(tuple(ctx[L - n:]))
            if end is not None:
                return ctx[end + 1:end + 1 + k]
        return []


def accept(draft: list[int], pred: list[int]) -> list[int]:
    """The tokens a verify step yields. `pred[i]` is the argmax of row i of the forward over
    [last, draft[0], .., draft[k-1]] (so len(pred) == len(draft) + 1): pred[0] is the token
    after `last`, pred[i] the token after draft[i-1]. draft[i] is accepted while it equals
    pred[i] and every earlier one was accepted; the result is the accepted drafts and then
    pred at the first disagreement (or after the last draft), between 1 and k + 1 tokens."""
    if len(pred) != len(draft) + 1:
        raise ValueError(f"{len(draft)} drafts need {len(draft) + 1} predictions, "
                         f"got {len(pred)}")
    a = 0
    while a < len(draft) and draft[a] == pred[a]:
        a += 1
    return list(draft[:a]) + [pred[a]]


def cut_at_stop(tokens: list[int], stop_ids) -> tuple[list[int], bool]:
    """`tokens` up to and with the first stop id, and whether one was found."""
    for i, t in enumerate(tokens):
        if t in stop_ids:
            return tokens[:i + 1], True
    return tokens, False
