#!/usr/bin/env python3
"""swiglu_fp8, the SwiGLU that writes the down projection's e4m3 rows, against the two ops it
replaces, and the fp8-tok engine with it on and off.

    python scripts/bench_swiglu_fp8.py --part op        # bench_swiglu's widths, rows 1 to 4096
    python scripts/bench_swiglu_fp8.py --part cluster   # the same with each cluster size forced
    python scripts/bench_swiglu_fp8.py --part engine    # fp8-tok decode ms/step, prefill tok/s

The op rows time, at the intermediate widths bench_swiglu runs (2048, 5632, 11008, 14336),
swiglu_fp8 against swiglu followed by fp8_quantize(.., "row"), plus the two pieces alone. gate
and up are the two halves of one [rows, 2 * cols] tensor, as the engine's gate/up projection
writes them, read in place by both. Each is timed two ways: `graph`, --reps calls captured in
one CUDA graph and replayed, the event time over --iters replays divided by --reps (what a
graphed decode step pays per call: the kernels and the gaps between them, no host); and
`eager`, --reps calls queued back to back between two events (what an eager prefill pays,
host launch path included). The inputs are not rotated, as in the engine: the smaller shapes
stay in the 96 MB L2 between calls and the 4096-row ones at 5632 and up (92 to 235 MB of
gate|up) do not. Before timing, the fused op's bytes and scales are checked against the two
ops with torch.equal.

The cluster rows (`kind: cluster_sweep`) are the op rows of swiglu_fp8 and the two ops once per
size in --clusters, with SPARK_SWIGLU_FP8_CLUSTER set to it. The kernel reads that variable
once, so each size runs in a process of its own (this script with --part op); the two ops'
rows are timed again in each, next to the fused op they are compared with.

The engine rows are scripts/bench_serve.py's static batches (prompts log-uniform in 64..2048,
--new tokens each, greedy) on an fp8-tok SparkModel of --layers of Llama-3-8B's layers with
random weights, CUDA graphs on, the norms' fp8 fusion on in both, once with
SparkModel.fuse_swiglu_fp8 and once without, the two modes alternated --repeats times on the
same model and the median of each kept.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "swiglu_fp8.jsonl"


def graph_us(fn, reps: int, iters: int) -> float:
    """Per-call time of fn with `reps` calls captured in one CUDA graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    for _ in range(3):
        g.replay()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps)
    del g
    ts.sort()
    return ts[len(ts) // 2]


def eager_us(fn, reps: int, iters: int) -> float:
    """Per-call time of fn with `reps` calls queued back to back between two events."""
    for _ in range(5):
        fn()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) * 1e3 / reps)
    ts.sort()
    return ts[len(ts) // 2]


def check_same(sk, gate, up) -> None:
    q, s = sk.swiglu_fp8(gate, up)
    q0, s0 = sk.fp8_quantize(sk.swiglu(gate, up), "row")
    assert torch.equal(s, s0) and torch.equal(q.view(torch.uint8), q0.view(torch.uint8))


def bench_shape(sk, m: int, cols: int, args, rows_out) -> None:
    g = torch.Generator(device="cuda").manual_seed(m + cols)
    gu = torch.randn(m, 2 * cols, device="cuda", generator=g).to(torch.bfloat16)
    gate, up = gu[:, :cols], gu[:, cols:]
    check_same(sk, gate, up)
    h = sk.swiglu(gate, up)
    ops = {
        "swiglu_fp8": lambda: sk.swiglu_fp8(gate, up),
        "swiglu + fp8_quantize": lambda: sk.fp8_quantize(sk.swiglu(gate, up), "row"),
        "swiglu": lambda: sk.swiglu(gate, up),
        "fp8_quantize row": lambda: sk.fp8_quantize(h, "row"),
    }
    if args.pair_only:
        ops = {k: ops[k] for k in ("swiglu_fp8", "swiglu + fp8_quantize")}
    # bytes each moves through DRAM or L2 (bf16 = 2, e4m3 = 1 per element, the scales left
    # out): the fused op reads gate and up and writes q
    n = m * cols
    nbytes = {"swiglu_fp8": 5 * n, "swiglu + fp8_quantize": 9 * n, "swiglu": 6 * n,
              "fp8_quantize row": 3 * n}
    res = {}
    for name, fn in ops.items():
        res[name] = (graph_us(fn, args.reps, args.iters), eager_us(fn, args.reps, args.iters))
    for name, (tg, te) in res.items():
        rows_out.append({"kind": "op", "op": name, "rows": m, "cols": cols, "graph_us": tg,
                         "eager_us": te, "bytes": nbytes[name],
                         "graph_gbps": nbytes[name] / tg / 1e3})
    (fg, fe), (tg, te) = res["swiglu_fp8"], res["swiglu + fp8_quantize"]
    print(f"{m:5d} x {cols:5d}  graph {fg:7.2f} us vs {tg:7.2f} ({tg / fg:.2f}x)"
          f"  eager {fe:7.2f} us vs {te:7.2f} ({te / fe:.2f}x)", file=sys.stderr)


def bench_op(args, rows_out):
    import spark_kernels as sk

    for cols in [int(v) for v in args.cols.split(",") if v]:
        for m in [int(v) for v in args.rows.split(",") if v]:
            bench_shape(sk, m, cols, args, rows_out)
            torch.cuda.empty_cache()


def bench_cluster(args, rows_out):
    """The op rows of the fused op and the two ops, once per forced cluster size, each size in
    its own process (the kernel reads SPARK_SWIGLU_FP8_CLUSTER once)."""
    for c in [int(v) for v in args.clusters.split(",") if v]:
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "rows.jsonl"
            cmd = [sys.executable, str(Path(__file__).resolve()), "--part", "op", "--pair-only",
                   "--rows", args.rows, "--cols", args.cols, "--reps", str(args.reps),
                   "--iters", str(args.iters), "--out", str(out)]
            print(f"cluster {c}:", file=sys.stderr)
            subprocess.run(cmd, check=True, env={**os.environ, "SPARK_SWIGLU_FP8_CLUSTER": str(c)})
            for line in out.read_text().splitlines():
                r = json.loads(line)
                r.update(kind="cluster_sweep", cluster=c, iters=args.iters)
                rows_out.append(r)


def bench_engine(args, rows_out):
    from bench_serve import log_uniform, prompts_for, static_batch, warm_up

    from spark_kernels import engine as E
    from spark_kernels.layer import RoPE

    batches = [int(v) for v in args.batches.split(",") if v]
    max_seq = args.max_prompt + max(args.new, 256) + 16
    weights = E.ModelWeights.random(args.layers)
    vocab = weights.embed.shape[0]
    model = E.SparkModel(weights, RoPE(max_seq), weights_format="fp8-tok")
    print(f"model: {args.layers} layers, fp8-tok projections, {args.kv_format} cache, "
          f"{args.cache_gb:g} GB of cache", file=sys.stderr)
    runs: dict[tuple[str, int], list[dict]] = {}
    for rep in range(args.repeats):
        for mode in ("fused", "two ops"):
            model.fuse_swiglu_fp8 = mode == "fused"
            eng = E.Engine(model, args.layers, max_batch=max(batches), max_seq=max_seq,
                           cache_bytes=int(args.cache_gb * (1 << 30)),
                           prefill_tokens=args.prefill_tokens, graphs=True,
                           kv_format=args.kv_format)
            warm_up(eng, [b for b in E.BUCKETS if b <= max(batches)], vocab)
            for b in batches:
                gen = torch.Generator().manual_seed(args.seed + b)
                lens = log_uniform(b, args.min_prompt, args.max_prompt, gen)
                r = static_batch(eng, prompts_for(lens, vocab, gen), args.new)
                r["prompt_tokens"] = sum(lens)
                runs.setdefault((mode, b), []).append(r)
                print(f"rep {rep} {mode:8s} B={b:3d}: decode "
                      f"{r['decode_s'] * 1e3 / max(1, r['steps']):6.3f} ms/step, prefill "
                      f"{sum(lens) / r['prefill_s']:8.0f} tok/s", file=sys.stderr)
            del eng
            torch.cuda.empty_cache()
    for (mode, b), rs in runs.items():
        step = sorted(r["decode_s"] * 1e3 / max(1, r["steps"]) for r in rs)
        pre = sorted(r["prompt_tokens"] / r["prefill_s"] for r in rs)
        rows_out.append({"kind": "engine", "mode": mode, "layers": args.layers, "batch": b,
                         "weights_format": "fp8-tok", "kv_format": args.kv_format,
                         "new": args.new, "repeats": args.repeats,
                         "prompt_tokens": rs[0]["prompt_tokens"],
                         "decode_ms_per_step": step[len(step) // 2],
                         "prefill_tok_s": pre[len(pre) // 2]})
    for b in batches:
        f = next(r for r in rows_out if r.get("mode") == "fused" and r["batch"] == b)
        t = next(r for r in rows_out if r.get("mode") == "two ops" and r["batch"] == b)
        print(f"B={b:3d}: decode {t['decode_ms_per_step']:.3f} -> {f['decode_ms_per_step']:.3f}"
              f" ms/step, prefill {t['prefill_tok_s']:.0f} -> {f['prefill_tok_s']:.0f} tok/s",
              file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", choices=("op", "cluster", "engine"), default="op")
    ap.add_argument("--rows", default=None,
                    help="op: 1,8,32,64,512,4096; cluster: 1,2,4,8,16,24,32,48,64,128,256,512")
    ap.add_argument("--cols", default="2048,5632,11008,14336")
    ap.add_argument("--reps", type=int, default=20, help="calls per graph / per sample")
    ap.add_argument("--iters", type=int, default=None, help="op: 50, cluster: 30")
    ap.add_argument("--clusters", default="1,2,4,8", help="cluster sizes the cluster part forces")
    ap.add_argument("--pair-only", action="store_true",
                    help="op rows of swiglu_fp8 and the two ops only, not the pieces alone")
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--new", type=int, default=64, help="tokens generated per request")
    ap.add_argument("--min-prompt", type=int, default=64)
    ap.add_argument("--max-prompt", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--cache-gb", type=float, default=2.0, help="K/V cache size, all layers")
    ap.add_argument("--kv-format", default="bf16")
    ap.add_argument("--prefill-tokens", type=int, default=16384)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if args.rows is None:
        args.rows = "1,8,32,64,512,4096" if args.part != "cluster" else \
            "1,2,4,8,16,24,32,48,64,128,256,512"
    if args.iters is None:
        args.iters = 30 if args.part == "cluster" else 50
    rows: list[dict] = []
    if args.part == "op":
        bench_op(args, rows)
    elif args.part == "cluster":
        bench_cluster(args, rows)
    else:
        bench_engine(args, rows)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows appended to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
