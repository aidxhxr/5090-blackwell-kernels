"""CPU-only tests of the page bookkeeping behind the paged cache and its prefix caching
(python/spark_kernels/pages.py): the hash chain, reference counts, the evictable set and its
LRU order. The module is loaded from its file, since importing the package needs the built
extension."""

import importlib.util
import random
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "python" / "spark_kernels" / "pages.py"
_spec = importlib.util.spec_from_file_location("spark_pages", _PATH)
P = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(P)

PAGE = 16


def test_hash_chain_names_prefixes():
    a = list(range(50))
    b = a[:32] + [999] + a[33:]
    ha, hb = P.page_hashes(a, PAGE), P.page_hashes(b, PAGE)
    assert len(ha) == 3 and len(P.page_hashes(a[:47], PAGE)) == 2  # full pages only
    assert ha[:2] == hb[:2] and ha[2] != hb[2]
    # the same 16 tokens after a different prefix are a different page
    c = [7] * 16 + a[16:32]
    assert P.page_hashes(c, PAGE)[1] != ha[1]
    assert P.page_hashes([], PAGE) == []


def test_plain_free_list_order():
    """Without registered pages the table hands out and takes back pages in the order of the
    list it replaced: lowest ids first, a released run comes back in its order."""
    t = P.PageTable(8)
    assert t.alloc(3) == [0, 1, 2]
    x = t.alloc(2)
    t.release([0, 1, 2])
    assert t.alloc(3) == [0, 1, 2]
    t.release(x)
    t.check()
    assert t.available() == 5


def test_refcounts_and_evictable():
    t = P.PageTable(6)
    toks = list(range(40))
    hs = P.page_hashes(toks, PAGE)
    pages = t.alloc(3)  # two full pages and a partial one
    for p, h in zip(pages, hs, strict=False):
        assert t.writable(p)
        assert t.register(p, h)
        assert not t.writable(p)
    assert t.writable(pages[2])  # the partial page stays writable
    # a second request with the same first 32 tokens finds both pages
    hit = t.lookup(hs)
    assert hit == pages[:2]
    t.acquire(hit)
    assert t.refs[pages[0]] == 2
    t.release(pages)  # the first request finishes: the shared pages stay held
    assert t.refs[pages[0]] == 1 and pages[2] in t.free
    assert not t.evictable
    t.release(hit)  # the second finishes: both cached pages become evictable
    assert list(t.evictable) == [pages[1], pages[0]]  # deeper page first
    assert t.available() == 6
    t.check()
    # a third request finds them again and takes them out of the evictable set
    again = t.lookup(hs)
    t.acquire(again)
    assert again == pages[:2] and not t.evictable
    t.release(again)
    t.check()


def test_lookup_limit_and_miss():
    t = P.PageTable(4)
    hs = P.page_hashes(list(range(64)), PAGE)
    pages = t.alloc(4)
    for p, h in zip(pages, hs, strict=True):
        t.register(p, h)
    assert t.lookup(hs, limit=2) == pages[:2]
    other = P.page_hashes([5] + list(range(1, 64)), PAGE)
    assert t.lookup(other) == []


def test_duplicate_register_keeps_first():
    """Two equal prompts prefilled in the same forward write two pages with one hash: the
    first is registered, the second stays unnamed and returns to the free list."""
    t = P.PageTable(4)
    h = P.page_hashes(list(range(16)), PAGE)[0]
    a, b = t.alloc(2)
    assert t.register(a, h) and not t.register(b, h)
    t.release([a, b])
    assert t.free == [3, 2, b] and list(t.evictable) == [a]
    t.check()


def test_alloc_evicts_lru_and_drops_hash():
    t = P.PageTable(4)
    seqs = []
    for s in range(2):
        toks = [s] * 32
        pages = t.alloc(2)
        for p, h in zip(pages, P.page_hashes(toks, PAGE), strict=True):
            t.register(p, h)
        seqs.append((pages, P.page_hashes(toks, PAGE)))
    t.release(seqs[0][0])
    t.release(seqs[1][0])
    assert t.free == [] and t.available() == 4
    # LRU: the first sequence's pages go first, its deeper page before its first one
    got = t.alloc(1)
    assert got == [seqs[0][0][1]] and t.evicted == 1
    assert t.lookup(seqs[0][1]) == [seqs[0][0][0]]  # the chain now stops after page 0
    assert t.lookup(seqs[1][1]) == seqs[1][0]
    with pytest.raises(RuntimeError):
        t.alloc(4)
    t.release(got)
    t.check()


def test_register_needs_a_holder():
    t = P.PageTable(2)
    with pytest.raises(AssertionError):
        t.register(0, b"x")
    with pytest.raises(AssertionError):
        t.release([1])


def test_random_workload_keeps_invariants():
    """Requests drawn from a few shared prefixes, admitted, registered and released in random
    order against a small table: every page stays in one place, a held page is never
    evicted, and a lookup hit always names pages whose hashes match."""
    rng = random.Random(0)
    t = P.PageTable(24)
    prefixes = [[rng.randrange(100) for _ in range(rng.randrange(16, 80))] for _ in range(4)]
    live = []
    hits = 0
    for _ in range(2000):
        if live and (rng.random() < 0.45 or t.available() < 6):
            pages, _ = live.pop(rng.randrange(len(live)))
            t.release(pages)
        else:
            toks = rng.choice(prefixes) + [rng.randrange(100) for _ in range(rng.randrange(1, 30))]
            hs = P.page_hashes(toks, PAGE)
            need = -(-len(toks) // PAGE)
            hit = t.lookup(hs, limit=(len(toks) - 1) // PAGE)
            if need - len(hit) > t.available() - sum(t.refs[p] == 0 for p in hit):
                continue
            t.acquire(hit)
            pages = hit + t.alloc(need - len(hit))
            for p in pages[len(hit):]:
                assert t.writable(p)
            for i in range(len(hit), len(hs)):
                t.register(pages[i], hs[i])
            for p, h in zip(hit, hs, strict=False):
                assert t.page_hash[p] == h and t.refs[p] >= 1
            hits += len(hit)
            live.append((pages, hs))
        t.check()
    for pages, _ in live:
        t.release(pages)
    t.check()
    assert t.available() == 24 and hits > 0 and t.evicted > 0
