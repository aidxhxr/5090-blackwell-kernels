"""The page bookkeeping of the paged K/V cache, in plain Python (no torch, no GPU): which
pages are free, how many sequences hold each page, and which full pages hold a known run of
prompt tokens that a later request with the same prefix can reuse (prefix caching).

A full page's identity is a hash chain over the prompt: page i's hash is
sha256(hash of page i - 1, its 16 token ids), so two prompts get the same hash for page i
exactly when their first 16 (i + 1) tokens are equal. `PageTable` keeps

    free        page ids nobody holds and no hash names, taken first by `alloc`
    refs        page id -> how many sequences hold it (block table rows that point at it)
    cached      hash -> page id, for every registered page, held or not
    evictable   registered pages nobody holds, oldest release first (LRU): still reusable
                by a prompt that matches them, and taken by `alloc` once `free` is empty,
                which drops their hash

Only full pages are registered, and a sequence only appends past the pages it matched, so a
page that two sequences hold is never written again. `writable` is what the engine asserts
before a forward writes a page.
"""

from __future__ import annotations

import hashlib
from array import array
from collections import OrderedDict


def page_hashes(tokens, page: int) -> list[bytes]:
    """The hash chain of the full pages of `tokens` (a list of ints): one digest per full
    page, each over the previous digest and the page's token ids. A partial last page has
    none."""
    out, parent = [], b""
    for i in range(len(tokens) // page):
        h = hashlib.sha256(parent)
        h.update(array("q", tokens[i * page:(i + 1) * page]).tobytes())
        parent = h.digest()
        out.append(parent)
    return out


class PageTable:
    """Free list, reference counts and the prefix hash table of `num_pages` pages (ids
    0..num_pages-1). Without registered pages it is the plain free list it replaces: `alloc`
    takes the lowest ids first and `release` gives them back in the same order."""

    def __init__(self, num_pages: int):
        self.num_pages = num_pages
        self.free = list(range(num_pages - 1, -1, -1))
        self.refs = [0] * num_pages
        self.cached: dict[bytes, int] = {}
        self.page_hash: dict[int, bytes] = {}
        self.evictable: OrderedDict[int, None] = OrderedDict()
        self.evicted = 0  # registered pages taken back by alloc

    def available(self) -> int:
        """Pages `alloc` can give: the free ones and the evictable cached ones."""
        return len(self.free) + len(self.evictable)

    def alloc(self, n: int) -> list[int]:
        """n pages held once each: free pages first, then the least recently released cached
        pages, whose hashes are dropped."""
        if n > self.available():
            raise RuntimeError(f"out of cache pages: need {n}, {self.available()} available")
        out = []
        for _ in range(n):
            if self.free:
                p = self.free.pop()
            else:
                p, _ = self.evictable.popitem(last=False)
                del self.cached[self.page_hash.pop(p)]
                self.evicted += 1
            self.refs[p] = 1
            out.append(p)
        return out

    def release(self, pages: list[int]) -> None:
        """Drops one hold on each page. A page nobody holds goes back to the free list, or,
        if it is registered, to the newest end of the evictable set. The pages are walked
        last to first, so a sequence's deeper pages are evicted before its first ones: a
        later prompt can only reach page i through pages 0..i-1."""
        for p in reversed(pages):
            assert self.refs[p] > 0, f"page {p} released more often than it was taken"
            self.refs[p] -= 1
            if self.refs[p]:
                continue
            if p in self.page_hash:
                self.evictable[p] = None
            else:
                self.free.append(p)

    def lookup(self, hashes: list[bytes], limit: int | None = None) -> list[int]:
        """The pages of the longest registered prefix of `hashes` (at most `limit` of them),
        without taking them."""
        out = []
        for h in hashes[:limit]:
            p = self.cached.get(h)
            if p is None:
                break
            out.append(p)
        return out

    def acquire(self, pages: list[int]) -> None:
        """Takes one hold on each page of a `lookup` result (an evictable one stops being
        evictable)."""
        for p in pages:
            if self.refs[p] == 0:
                del self.evictable[p]
            self.refs[p] += 1

    def register(self, page: int, h: bytes) -> bool:
        """Names a full page whose K/V is written by its hash, so later prompts can reuse it.
        False when another page already holds that hash (two equal prompts prefilled in the
        same forward): this one stays unnamed and goes back to the free list when released."""
        assert self.refs[page] > 0, f"page {page} registered while nobody holds it"
        if h in self.cached or page in self.page_hash:
            return False
        self.cached[h] = page
        self.page_hash[page] = h
        return True

    def writable(self, page: int) -> bool:
        """A forward may write `page` only when one sequence holds it and it is not a
        registered (full, shared or shareable) page."""
        return self.refs[page] == 1 and page not in self.page_hash

    def check(self) -> None:
        """Every page is in exactly one place: held, free, or evictable (tests)."""
        held = {p for p in range(self.num_pages) if self.refs[p] > 0}
        free, ev = set(self.free), set(self.evictable)
        assert len(free) == len(self.free), "a page is twice on the free list"
        assert not (held & free) and not (held & ev) and not (free & ev)
        assert len(held) + len(free) + len(ev) == self.num_pages
        assert set(self.page_hash) == set(self.cached.values())
        assert ev <= set(self.page_hash) and not (free & set(self.page_hash))
