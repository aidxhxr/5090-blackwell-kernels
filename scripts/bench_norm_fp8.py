#!/usr/bin/env python3
"""The norms that write fp8 (rmsnorm_fp8, add_rmsnorm_fp8_) against the two ops they replace,
and the fp8-tok engine with them on and off.

    python scripts/bench_norm_fp8.py --part op        # hidden 4096, rows 1 to 8192
    python scripts/bench_norm_fp8.py --part engine    # fp8-tok decode ms/step, prefill tok/s

The op rows time, at hidden 4096, add_rmsnorm_fp8_ against add_rmsnorm_ followed by
fp8_quantize(.., "row"), and rmsnorm_fp8 against rmsnorm followed by it, plus the three
pieces alone. Each is timed two ways: `graph`, --reps calls captured in one CUDA graph and
replayed, the event time over --iters replays divided by --reps (what a graphed decode step
pays per call: the kernels and the gaps between them, no host); and `eager`, --reps calls
queued back to back between two events (what an eager prefill pays, host launch path
included). The inputs are not rotated: the residual stream and the norm's output stay in the
96 MB L2 up to 2048 rows, as they do in the engine, and the 8192-row inputs (128 MB) do not.
Before timing, each fused op's bytes, scales and residual are checked against the two ops
with torch.equal.

The engine rows are scripts/bench_serve.py's static batches (prompts log-uniform in 64..2048,
--new tokens each, greedy) on an fp8-tok SparkModel of --layers of Llama-3-8B's layers with
random weights, CUDA graphs on, once with SparkModel.fuse_norm_fp8 and once without, the two
modes alternated --repeats times on the same model and the median of each kept.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "norm_fp8.jsonl"
HIDDEN = 4096


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


def check_same(sk, x, d, w) -> None:
    r1, r2 = x.clone(), x.clone()
    q, s = sk.add_rmsnorm_fp8_(d, r1, w)
    q0, s0 = sk.fp8_quantize(sk.add_rmsnorm_(d, r2, w), "row")
    assert torch.equal(r1, r2) and torch.equal(s, s0)
    assert torch.equal(q.view(torch.uint8), q0.view(torch.uint8))
    q, s = sk.rmsnorm_fp8(x, w)
    q0, s0 = sk.fp8_quantize(sk.rmsnorm(x, w), "row")
    assert torch.equal(s, s0) and torch.equal(q.view(torch.uint8), q0.view(torch.uint8))


def bench_shape(sk, m: int, args, rows_out) -> None:
    g = torch.Generator(device="cuda").manual_seed(m)
    x = torch.randn(m, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    d = torch.randn(m, HIDDEN, device="cuda", generator=g).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(HIDDEN, device="cuda", generator=g)).to(torch.bfloat16)
    check_same(sk, x, d, w)
    h = sk.rmsnorm(x, w)
    r = x.clone()
    ops = {
        "add_rmsnorm_fp8_": lambda: sk.add_rmsnorm_fp8_(d, r, w),
        "add_rmsnorm_ + fp8_quantize": lambda: sk.fp8_quantize(sk.add_rmsnorm_(d, r, w), "row"),
        "rmsnorm_fp8": lambda: sk.rmsnorm_fp8(x, w),
        "rmsnorm + fp8_quantize": lambda: sk.fp8_quantize(sk.rmsnorm(x, w), "row"),
        "add_rmsnorm_": lambda: sk.add_rmsnorm_(d, r, w),
        "rmsnorm": lambda: sk.rmsnorm(x, w),
        "fp8_quantize row": lambda: sk.fp8_quantize(h, "row"),
    }
    # bytes each moves through DRAM or L2 (bf16 = 2, e4m3 = 1 per element; w and the scales
    # left out): the fused add reads x and resid and writes resid and q
    n = m * HIDDEN
    nbytes = {"add_rmsnorm_fp8_": 7 * n, "add_rmsnorm_ + fp8_quantize": 11 * n,
              "rmsnorm_fp8": 3 * n, "rmsnorm + fp8_quantize": 7 * n,
              "add_rmsnorm_": 8 * n, "rmsnorm": 4 * n, "fp8_quantize row": 3 * n}
    res = {}
    for name, fn in ops.items():
        res[name] = (graph_us(fn, args.reps, args.iters), eager_us(fn, args.reps, args.iters))
    for name, (tg, te) in res.items():
        rows_out.append({"kind": "op", "op": name, "rows": m, "cols": HIDDEN, "graph_us": tg,
                         "eager_us": te, "bytes": nbytes[name],
                         "graph_gbps": nbytes[name] / tg / 1e3})
    for fused, two in (("add_rmsnorm_fp8_", "add_rmsnorm_ + fp8_quantize"),
                       ("rmsnorm_fp8", "rmsnorm + fp8_quantize")):
        (fg, fe), (tg, te) = res[fused], res[two]
        print(f"{m:5d} x {HIDDEN}  {fused:17s} graph {fg:7.2f} us vs {tg:7.2f} ({tg / fg:.2f}x)"
              f"  eager {fe:7.2f} us vs {te:7.2f} ({te / fe:.2f}x)", file=sys.stderr)


def bench_op(args, rows_out):
    import spark_kernels as sk

    for m in [int(v) for v in args.rows.split(",") if v]:
        bench_shape(sk, m, args, rows_out)
        torch.cuda.empty_cache()


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
            model.fuse_norm_fp8 = mode == "fused"
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
    ap.add_argument("--part", choices=("op", "engine"), default="op")
    ap.add_argument("--rows", default="1,8,64,512,2048,8192")
    ap.add_argument("--reps", type=int, default=20, help="calls per graph / per sample")
    ap.add_argument("--iters", type=int, default=50)
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
    rows: list[dict] = []
    if args.part == "op":
        bench_op(args, rows)
    else:
        bench_engine(args, rows)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} rows appended to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
