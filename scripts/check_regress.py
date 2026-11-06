#!/usr/bin/env python3
"""Check a fresh bench run against the rows committed in results/ and say what got slower.

    ./build/bench_w1gemm > /tmp/w1gemm.json
    python3 scripts/check_regress.py /tmp/w1gemm.json             # against results/w1gemm.json
    ./build/bench_hgemm --models --ms=128 | python3 scripts/check_regress.py - \\
        --against results/hgemm_models.json
    python3 scripts/check_regress.py fresh/                       # every file of a run directory
    python3 scripts/check_regress.py results/moe_layer.json --rev HEAD   # rerun in place

The fresh side is a file, a directory of them or "-" for stdin: JSON lines as the C++ benches
and most scripts print them (any other line, like a bench's stderr table, is skipped), or one
JSON document (results/llm_*.json), whose rows are the dicts in it that carry a timing. Without
--against a file is checked against the committed file of the same name, or, when there is
none, the one in results/ that shares the most rows with it. --rev reads the committed side
from git, for a results file a rerun has already overwritten.

Rows are matched on their key: every field that is not a measurement and not in NOT_KEY
(kernel, dtype, variant, shape, impl, tokens, ...), so a new bench needs no table here. A
column the committed rows do not have yet still matches when only one row can be meant. The
gated metrics are the times (lower is better: median_ms, ms, v6_ms, ttft_s) and the rates
(higher is better: gbps, tflops, decode_tok_s), minus min_ms, the one-off timings, the
reference columns (ref_ms, torch_*, flex_*, sdpa_*, ...) and a rate that restates a time of
its own row. A rate is turned into the time it implies, so 5% slower means the same for both.

Prints the rows that moved by more than --threshold percent either way and the rows missing
on either side, and exits 1 when anything got slower (or a fresh row failed its check), so it
can gate a run. --update writes the fresh rows into the committed file: matched rows are
replaced, new ones appended, rows the run did not touch are kept.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SUFFIXES = (".json", ".jsonl")

# A measurement is named for its unit. Times: median_ms, ms, v6_ms, decode_ms_per_step, ttft_s,
# seconds. Rates: gbps, tflops, tflops_2tvd, weight_gbps, step_gb_s, decode_tok_s,
# tokens_per_s_32_layers. "gflop" is an amount of work, not a rate.
TIME = re.compile(r"(?:^|_)(ms|us|s|seconds)(?:_|$)")
RATE = re.compile(r"(?:^|_)(?:gbps|gb_s|tflops|gflops|tok_s|tokens_per_s)(?:_|$)")
US_PER = {"ms": 1e3, "us": 1.0, "s": 1e6, "seconds": 1e6}

# Fields that say how a row came out rather than which row it is: pass flags and errors, the
# GPU state strings, library versions, iteration counts, accuracy, counts the run produced and
# sizes that follow from the routing. Kept out of the key, so a rerun with --iters=20 or a new
# torch still lines up with its committed row.
NOT_KEY = frozenset({
    "ok", "same_bits", "fits", "found", "reply", "number", "note", "loss", "nll", "ppl", "kl",
    "kl_tokens", "top1", "iters", "warmup", "reps", "launches", "state_before", "state_after",
    "torch", "vllm", "version", "route_mismatch", "experts_touched", "gflop", "weight_bytes",
    "allocated_bytes", "generated", "mixed_rows", "preempted", "peak_gb", "peak_extra_gb",
})
NOT_KEY_PART = re.compile(r"(?:^|_)err(?:_|$)|_over_|^pct_|^rel_|^margin")

# Compared on request (--metric) but never gated by default: the best iteration (min_ms shows
# the jitter, the median is the number), timings of one call (a first call that compiles, a
# quantization), and the reference columns, which time a library on the same shape.
UNGATED = frozenset({"min_ms", "first_call_s", "quantize_s"})
REFERENCE_PREFIXES = ("ref_", "torch_", "flex_", "sdpa_", "flash_", "vllm_", "hf_", "cublas")
# A rate restates a time of its row when the two share a stem (v6_tflops is v6_ms's FLOPs over
# v6_ms), and in a row timed by median_ms, ms, ours_ms or spark_ms, gbps and tflops are that
# same measurement again. Gating them would report one slowdown three times.
MAIN_STEMS = frozenset({"", "median", "ours", "spark", "weight"})

# What "the same bench" means for committed rows the run did not produce: only rows of a
# kernel the run touched are listed as missing, not the whole file.
FAMILY_FIELDS = ("kernel", "kind", "model", "_path")
# Fields a row's label starts with, in this order, shown bare; the rest are shown as k=v.
LABEL_FIRST = ("_path", "kernel", "kind", "model", "backend", "impl", "method", "mode", "dtype",
               "variant", "shape")


def direction(name: str, value) -> str | None:
    """"time" (lower is better) or "rate" (higher is better) for a number named for its unit,
    None for anything else."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if RATE.search(name):
        return "rate"
    if TIME.search(name):
        return "time"
    return None


def is_key(name: str, value) -> bool:
    if value is not None and not isinstance(value, (str, int, float, bool)):
        return False  # nested dicts and lists (caps, config, breakdown_us) are not identity
    if RATE.search(name) or TIME.search(name):
        return False
    return name not in NOT_KEY and not NOT_KEY_PART.search(name)


def stem(name: str) -> str:
    """The part of a metric's name before its unit: "v6" for v6_ms and v6_tflops."""
    m = RATE.search(name) or TIME.search(name)
    return name[: m.start()] if m else name


def is_reference(name: str) -> bool:
    return name in UNGATED or name.startswith(REFERENCE_PREFIXES)


def gated_metrics(row: dict) -> list[str]:
    """The metrics of a row that decide whether it regressed."""
    times = [k for k, v in row.items() if direction(k, v) == "time" and not is_reference(k)]
    time_stems = {stem(k) for k in times}
    main_time = bool(time_stems & MAIN_STEMS)
    out = list(times)
    for k, v in row.items():
        if direction(k, v) != "rate" or is_reference(k):
            continue
        s = stem(k)
        if s in time_stems or (s in MAIN_STEMS and main_time):
            continue
        out.append(k)
    return out


def show(v) -> str:
    return v if isinstance(v, str) else json.dumps(v)


@dataclass
class Row:
    data: dict
    line: str | None = None  # the JSON line as written, for --update; None inside a document
    key: tuple = field(init=False)

    def __post_init__(self):
        # json.dumps keeps 1, 1.0 and true apart, which Python's == does not
        self.key = tuple(sorted((k, json.dumps(v)) for k, v in self.data.items()
                                if is_key(k, v)))

    @property
    def family(self) -> str:
        return next((show(self.data[k]) for k in FAMILY_FIELDS if k in self.data), "")

    @property
    def label(self) -> str:
        keys = {k for k, _ in self.key} - {"device"}
        parts = []
        for k in LABEL_FIRST:
            if k in keys:
                v = show(self.data[k])
                parts.append(f"v{v}" if k == "variant" else v)
        parts += [f"{k}={show(self.data[k])}" for k in sorted(keys - set(LABEL_FIRST))]
        return " ".join(parts)


def records(node, path: str = "") -> Iterator[dict]:
    """The rows of a JSON document: every dict with a timing in it, under the path of keys
    that leads to it (list positions are not part of it, a row's own fields say which it is)."""
    if isinstance(node, list):
        for x in node:
            yield from records(x, path)
    elif isinstance(node, dict):
        if any(direction(k, v) for k, v in node.items()):
            yield {"_path": path, **node} if path else dict(node)
        for k, v in node.items():
            if isinstance(v, (dict, list)):
                yield from records(v, f"{path}/{k}" if path else k)


def parse(text: str) -> list[Row]:
    """Rows of a bench output: one JSON object per line, or one JSON document."""
    try:
        doc = json.loads(text)
    except json.JSONDecodeError:
        doc = None
    if isinstance(doc, dict) and "\n" not in text.strip() and \
            any(direction(k, v) for k, v in doc.items()):
        return [Row(doc, text.strip())]  # JSON lines with one row in them
    if isinstance(doc, (dict, list)):
        return [Row(r) for r in records(doc)]
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict):
            rows.append(Row(d, line))
    return rows


def index(rows: list[Row]) -> tuple[dict[tuple, Row], int]:
    """Rows by key, the last one kept when a key repeats, and how many repeated."""
    out: dict[tuple, Row] = {}
    for r in rows:
        out[r.key] = r
    return out, len(rows) - len(out)


def loose(a: tuple, b: tuple) -> bool:
    """Same row with a column one side does not have: one key's fields are a strict subset of
    the other's and every shared field agrees."""
    da, db = dict(a), dict(b)
    if da.keys() == db.keys() or not (da.keys() <= db.keys() or db.keys() <= da.keys()):
        return False
    common = da.keys() & db.keys()
    return bool(common) and all(da[k] == db[k] for k in common)


@dataclass
class Match:
    pairs: list[tuple[Row, Row]]  # (fresh, committed)
    new: list[Row]  # fresh rows with nothing committed
    not_rerun: list[Row]  # committed rows of the kernels the run touched, not in it
    dupes: tuple[int, int]  # repeated keys in (fresh, committed)


def match(fresh: list[Row], committed: list[Row]) -> Match:
    new_ix, new_dupes = index(fresh)
    old, old_dupes = index(committed)
    pairs, unmatched = [], []
    for k, r in new_ix.items():
        if k in old:
            pairs.append((r, old.pop(k)))
        else:
            unmatched.append(r)
    # a column added since the commit: take the pair only when neither side has a second
    # option. A strict subset has fewer fields, so keys of the same length are never compared.
    by_len: dict[int, list[Row]] = {}
    for c in old.values():
        by_len.setdefault(len(c.key), []).append(c)
    cands = {id(r): [c for n, cs in by_len.items() if n != len(r.key) for c in cs
                     if loose(r.key, c.key)] for r in unmatched}
    claims: dict[tuple, int] = {}
    for cs in cands.values():
        for c in cs:
            claims[c.key] = claims.get(c.key, 0) + 1
    new = []
    for r in unmatched:
        cs = cands[id(r)]
        if len(cs) == 1 and claims[cs[0].key] == 1:
            pairs.append((r, old.pop(cs[0].key)))
        else:
            new.append(r)
    families = {r.family for r in fresh}
    not_rerun = [c for c in old.values() if c.family in families]
    return Match(pairs, new, not_rerun, (new_dupes, old_dupes))


@dataclass
class Change:
    row: Row  # the fresh row
    metric: str
    old: float
    new: float
    slower: float  # change in time as a fraction, + is slower; a rate as the time it implies
    ref: float | None  # the same for the row's ref_ms (the library or the naive rung)
    tiny: bool  # a time change under --floor-us


def positive(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v > 0


def compare(pairs: list[tuple[Row, Row]], metrics: list[str] | None,
            floor_us: float) -> list[Change]:
    out = []
    for f, c in pairs:
        r_old, r_new = c.data.get("ref_ms"), f.data.get("ref_ms")
        ref = r_new / r_old - 1 if positive(r_old) and positive(r_new) else None
        for m in metrics or gated_metrics(c.data):
            kind = direction(m, c.data.get(m))
            if not kind or not positive(c.data[m]) or not positive(f.data.get(m)):
                continue
            a, b = float(c.data[m]), float(f.data[m])
            slower = b / a - 1 if kind == "time" else a / b - 1
            tiny = kind == "time" and abs(b - a) * US_PER[TIME.search(m).group(1)] < floor_us
            out.append(Change(f, m, a, b, slower, ref, tiny))
    return out


class Tree:
    """The committed side: the working tree, or the files as they are at a git revision."""

    def __init__(self, rev: str | None):
        self.rev = rev
        self.cache: dict[Path, list[Row] | None] = {}

    def _git(self, path: Path, *cmd: str) -> subprocess.CompletedProcess:
        # REV:./name and ls-tree are relative to -C, so any repo the path is in works
        return subprocess.run(["git", "-C", str(path), *cmd], capture_output=True, text=True)

    def read(self, p: Path) -> str | None:
        if self.rev is None:
            return p.read_text() if p.is_file() else None
        if not p.parent.is_dir():
            return None
        r = self._git(p.parent, "show", f"{self.rev}:./{p.name}")
        return r.stdout if r.returncode == 0 else None

    def is_dir(self, p: Path) -> bool:
        if self.rev is None or not p.is_dir():
            return p.is_dir()
        r = self._git(p, "cat-file", "-t", f"{self.rev}:./")
        return r.stdout.strip() == "tree"

    def files(self, d: Path) -> list[Path]:
        if self.rev is None:
            names = [x.name for x in d.iterdir() if x.is_file()]
        else:
            names = self._git(d, "ls-tree", "--name-only", self.rev, "./").stdout.split()
        return [d / n for n in sorted(names) if n.endswith(SUFFIXES)]

    def rows(self, p: Path) -> list[Row] | None:
        if p not in self.cache:
            text = self.read(p)
            self.cache[p] = None if text is None else parse(text)
        return self.cache[p]


def pick(name: str, rows: list[Row], against: list[Path], tree: Tree) -> list[Path] | str:
    """The committed file(s) a fresh input is checked against, or why there is none."""
    files = [a for a in against if not tree.is_dir(a)]
    if files:
        return files
    if name != "-":
        for d in against:
            same = d / Path(name).name
            if tree.read(same) is not None:
                if tree.rev is None and same.resolve() == Path(name).resolve():
                    return f"{name} is the committed file itself; pass --rev HEAD"
                return [same]
    keys = {r.key for r in rows}
    scores = []
    for d in against:
        for f in tree.files(d):
            if tree.rev is None and name != "-" and f.resolve() == Path(name).resolve():
                continue
            scores.append((len(keys & {r.key for r in tree.rows(f) or []}), f))
    best = max((n for n, _ in scores), default=0)
    if best == 0:
        return "no committed file shares a row with it; pass --against"
    top = [f for n, f in scores if n == best]
    if len(top) > 1:
        return (f"{', '.join(rel(f) for f in top)} all share {best} rows with it; "
                "pass --against")
    return top


def fresh_inputs(paths: list[str]) -> Iterator[tuple[str, str]]:
    for p in paths:
        if p == "-":
            yield "-", sys.stdin.read()
        elif Path(p).is_dir():
            for f in sorted(Path(p).iterdir()):
                if f.is_file() and f.name.endswith(SUFFIXES):
                    yield str(f), f.read_text()
        else:
            yield p, Path(p).read_text()


def rel(p: Path | str) -> str:
    """A path as short as it can be said from here."""
    if str(p) == "-":
        return "stdin"
    try:
        return str(Path(p).resolve().relative_to(Path.cwd()))
    except ValueError:
        return str(p)


def n_rows(n: int) -> str:
    return f"{n} row" if n == 1 else f"{n} rows"


def pct(x: float) -> str:
    return f"{100.0 * x:+.1f}%"


def table(changes: list[Change]) -> list[str]:
    head = ("row", "metric", "committed", "fresh", "time", "ref")
    cells = [(c.row.label, c.metric, f"{c.old:.4g}", f"{c.new:.4g}", pct(c.slower),
              "" if c.ref is None else pct(c.ref)) for c in changes]
    w = [max(len(r[i]) for r in [head, *cells]) for i in range(len(head))]
    fmt = f"  {{:<{w[0]}}}  {{:<{w[1]}}}  " + "  ".join(f"{{:>{x}}}" for x in w[2:])
    return [fmt.format(*head).rstrip()] + [fmt.format(*r).rstrip() for r in cells]


def listing(title: str, rows: list[Row], verbose: bool) -> list[str]:
    if not rows:
        return []
    labels = sorted(r.label for r in rows)
    out = [f"{title}: {len(labels)}"] + [f"  {x}" for x in labels[: None if verbose else 10]]
    if not verbose and len(labels) > 10:
        out.append(f"  ... and {len(labels) - 10} more (-v lists them)")
    return out


def update(path: Path, committed: list[Row], m: Match) -> str:
    """Rewrites a JSON-lines file with the fresh rows merged in, file order kept."""
    if any(r.line is None for r in committed) or any(f.line is None for f, _ in m.pairs) or \
            any(r.line is None for r in m.new):
        return f"not updating {path}: a JSON document, rerun the script that writes it"
    fresh_line = {id(c): f.line for f, c in m.pairs}
    lines = [fresh_line.get(id(r), r.line) for r in committed] + [r.line for r in m.new]
    path.write_text("\n".join(lines) + "\n")
    return (f"updated {path}: {n_rows(len(m.pairs))} replaced, {len(m.new)} added, "
            f"{len(committed) - len(m.pairs)} kept")


def check(name: str, rows: list[Row], against: list[Path], tree: Tree,
          args: argparse.Namespace) -> int:
    committed = [r for p in against for r in tree.rows(p) or []]
    where = ", ".join(rel(p) for p in against) + (f" at {tree.rev}" if tree.rev else "")
    m = match(rows, committed)
    changes = compare(m.pairs, args.metric, args.floor_us)
    t = args.threshold / 100.0
    slower = sorted((c for c in changes if c.slower > t and not c.tiny), key=lambda c: -c.slower)
    faster = sorted((c for c in changes if c.slower < -t and not c.tiny), key=lambda c: c.slower)
    moved = {id(c) for c in slower + faster}
    within = [c for c in changes if id(c) not in moved]
    failed = [r for r in rows if r.data.get("ok") is False]

    print(f"{rel(name)} against {where}: {n_rows(len(rows))} run, {len(committed)} committed")
    old_dev = {r.data["device"] for r in committed if "device" in r.data}
    new_dev = {r.data["device"] for r in rows if "device" in r.data}
    if old_dev and new_dev and not old_dev & new_dev:
        print(f"  the run is from {', '.join(sorted(new_dev))}, the committed rows from "
              f"{', '.join(sorted(old_dev))}: the device is part of the key")
    for what, cs in ((f"slower by more than {args.threshold:g}%", slower),
                     (f"faster by more than {args.threshold:g}%", faster),
                     ("within", within if args.verbose else [])):
        if cs:
            print(f"{what}:")
            print("\n".join(table(cs)))
    out = listing("failed the check in this run", failed, True)
    out += listing("committed, not in this run", m.not_rerun, args.verbose)
    out += listing("in this run, not committed", m.new, args.verbose)
    if any(m.dupes):
        out.append(f"repeated keys (the last row kept): {m.dupes[0]} in the run, "
                   f"{m.dupes[1]} committed")
    if out:
        print("\n".join(out))
    tiny = sum(c.tiny and abs(c.slower) > t for c in changes)
    floor = f" or under {args.floor_us:g} us" if args.floor_us else ""
    floor += f" ({tiny} of them past {args.threshold:g}%)" if tiny else ""
    print(f"{n_rows(len(m.pairs))} matched, {len(changes)} metrics compared: "
          f"{len({id(c.row) for c in slower})} slower, {len({id(c.row) for c in faster})} "
          f"faster, the rest within {args.threshold:g}%{floor}")
    if args.update:
        if len(against) != 1 or tree.rev is not None:
            print("not updating: --update needs exactly one committed file in the working tree")
        elif failed:
            print(f"not updating {against[0]}: {n_rows(len(failed))} failed the check")
        else:
            print(update(against[0], committed, m))
            return 0
    missing = args.fail_on_missing and (m.not_rerun or m.new)
    return 1 if slower or failed or missing else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fresh", nargs="+",
                    help="bench output: a file, a directory of them, or - for stdin")
    ap.add_argument("--against", action="append", type=Path, default=None, metavar="PATH",
                    help="committed results file or directory (repeatable; default results/)")
    ap.add_argument("--rev", default=None,
                    help="read the committed side from this git revision (e.g. HEAD)")
    ap.add_argument("--threshold", type=float, default=5.0, metavar="PCT",
                    help="percent slower that counts as a regression (default 5)")
    ap.add_argument("--metric", action="append", default=None, metavar="NAME",
                    help="compare this field instead of the gated ones (repeatable)")
    ap.add_argument("--floor-us", type=float, default=0.0, metavar="US",
                    help="ignore time changes smaller than this many microseconds")
    ap.add_argument("--fail-on-missing", action="store_true",
                    help="exit 1 when a row is missing on either side too")
    ap.add_argument("--update", action="store_true",
                    help="write the fresh rows into the committed file (JSON lines only)")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="also print the rows within the threshold and every missing row")
    args = ap.parse_args(argv)
    for name in args.metric or []:
        if direction(name, 1.0) is None:
            ap.error(f"--metric {name}: not named for a time or a rate, no direction")
    tree = Tree(args.rev)
    against = args.against or [RESULTS]
    status, checked = 0, 0
    for name, text in fresh_inputs(args.fresh):
        rows = parse(text)
        if not rows:
            print(f"{rel(name)}: no rows", file=sys.stderr)
            continue
        files = pick(name, rows, against, tree)
        if isinstance(files, str):
            print(f"{rel(name)}: {files}", file=sys.stderr)
            continue
        missing = [p for p in files if tree.rows(p) is None]
        if missing:
            print(f"{rel(name)}: no committed file {', '.join(map(str, missing))}", file=sys.stderr)
            continue
        if checked:
            print()
        status = max(status, check(name, rows, files, tree, args))
        checked += 1
    if not checked:
        print("nothing was checked", file=sys.stderr)
        return 2
    return status


if __name__ == "__main__":
    sys.exit(main())
