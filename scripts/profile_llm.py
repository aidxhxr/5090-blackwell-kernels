#!/usr/bin/env python3
"""Where the engine's time goes on a real checkpoint: torch.profiler over decode steps and a
prefill (GPU time per kernel family and the GPU idle time between kernels), and the
continuous-batching workload of bench_llm.py with every prefill and decode step timed
(synchronized) and binned by decode bucket.

    python scripts/profile_llm.py MODEL [--workload results/llm_workload.json]
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from spark_kernels import engine as E  # noqa: E402
from spark_kernels import hf  # noqa: E402

FAMILIES = [("paged_decode", "attention"), ("paged_combine", "attention"),
            ("attention", "attention"), ("varlen", "attention"), ("decode_kernel", "gemm"),
            ("hgemm", "gemm"), ("gemm", "gemm"), ("rmsnorm", "rmsnorm"), ("rope", "rope"),
            ("argmax", "argmax"), ("reduce", "argmax"), ("embedding", "embedding"),
            ("index", "embedding"), ("copy", "copy"), ("memcpy", "copy"), ("fill", "copy")]


def family(name: str) -> str:
    low = name.lower()
    return next((f for k, f in FAMILIES if k in low), "other")


def profile(fn, label: str, reps: int) -> dict:
    """GPU time per kernel family over `reps` calls of fn, and the GPU idle time: the wall
    time of the window minus the union of the kernel intervals."""
    fn()
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        t0 = time.perf_counter()
        for _ in range(reps):
            fn()
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    fam: dict[str, float] = collections.defaultdict(float)
    spans = []
    count = 0
    for ev in prof.events():
        if ev.device_type == torch.autograd.DeviceType.CUDA and ev.time_range.elapsed_us() > 0:
            fam[family(ev.name)] += ev.time_range.elapsed_us()
            spans.append((ev.time_range.start, ev.time_range.end))
            count += 1
    spans.sort()
    busy, end = 0.0, -1.0
    for s, e in spans:
        if s >= end:
            busy += e - s
            end = e
        elif e > end:
            busy += e - end
            end = e
    first, last = spans[0][0], max(e for _, e in spans)
    row = {"label": label, "reps": reps, "wall_ms": wall * 1e3 / reps,
           "gpu_window_ms": (last - first) / 1e3 / reps, "kernel_ms": busy / 1e3 / reps,
           "launches": count / reps,
           "families_ms": {k: v / 1e3 / reps for k, v in sorted(fam.items(), key=lambda x: -x[1])}}
    row["idle_ms"] = row["gpu_window_ms"] - row["kernel_ms"]
    print(json.dumps(row), file=sys.stderr, flush=True)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--workload", default=str(ROOT / "results" / "llm_workload.json"))
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--cache-gb", type=float, default=8.5)
    ap.add_argument("--prefill-tokens", type=int, default=8192)
    ap.add_argument("--graphs", type=int, default=1)
    ap.add_argument("--mixed", type=int, default=1, help="decode rows inside the prefill")
    ap.add_argument("--compact", type=int, default=1)
    ap.add_argument("--skip-kernels", action="store_true")
    ap.add_argument("--skip-continuous", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    w, cfg = hf.load(args.model)
    eng = E.Engine(E.SparkModel(w, cfg.rope(8192)), cfg.n_layers, max_batch=args.max_batch,
                   max_seq=8192, cache_bytes=int(args.cache_gb * (1 << 30)),
                   prefill_tokens=args.prefill_tokens, graphs=bool(args.graphs),
                   mixed=bool(args.mixed), compact=bool(args.compact))
    wl = json.loads(Path(args.workload).read_text())
    rows = []

    if not args.skip_kernels:
        for b in (1, 8, 64):
            # B sequences of 512 + 32 tokens in the cache (the middle of profile_vllm.py's
            # 64-token window), then decode steps on the graph
            s = next(x for x in wl["static"] if x["batch"] == b)
            for p in s["prompts"]:
                eng.submit(torch.tensor(p), 200)
            eng.prefill_waiting()
            for _ in range(32):
                eng._decode()
            rows.append(profile(eng._decode, f"decode B={b}", 20))
            for i, q in enumerate(eng.running):  # drop the batch
                if q is not None:
                    eng.cache.release(q.pages)
                    eng.running[i] = None
        for n in (512, 8000):
            ids = torch.tensor(wl["prefill"][1]["prompts"][0][:n])

            def pre(ids=ids):
                eng.submit(ids, 1)
                eng.prefill_waiting()
            rows.append(profile(pre, f"prefill {n}", 3))

    if args.skip_continuous:
        if args.out:
            with open(args.out, "a") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
        return 0

    # the continuous workload, every forward synchronized and timed
    c = wl["continuous"]
    t = collections.defaultdict(float)
    n = collections.Counter()
    live_sum = collections.Counter()
    orig_prefill, orig_decode = eng._prefill, eng._decode

    def timed_prefill(seqs, dec=()):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        orig_prefill(seqs, dec)
        torch.cuda.synchronize()
        t["prefill"] += time.perf_counter() - t0
        n["prefill"] += 1
        n["prefill_tokens"] += sum(s.prompt.numel() for s in seqs)
        n["mixed_rows"] += len(dec)

    def timed_decode():
        live = [i for i, s in enumerate(eng.running) if s is not None]
        if not live:
            return
        bp = next(x for x in E.BUCKETS if x > live[-1])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        orig_decode()
        torch.cuda.synchronize()
        t[f"decode {bp}"] += time.perf_counter() - t0
        n[f"decode {bp}"] += 1
        live_sum[bp] += len(live)

    eng._prefill, eng._decode = timed_prefill, timed_decode
    for r in c:
        eng.submit(torch.tensor(r["prompt"]), r["out"])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    eng.run()
    wall = time.perf_counter() - t0
    eng._prefill, eng._decode = orig_prefill, orig_decode
    cont = {"label": f"continuous, synchronized, mixed={int(eng.mixed)} compact={int(eng.compact)}",
            "wall_s": wall,
            "prefill_s": t["prefill"], "prefill_batches": n["prefill"],
            "prefill_tokens": n["prefill_tokens"], "mixed_rows": n["mixed_rows"],
            "decode": {k: {"steps": n[k], "s": v, "ms_per_step": v * 1e3 / n[k],
                           "mean_live": live_sum[int(k.split()[1])] / n[k]}
                       for k, v in t.items() if k.startswith("decode")},
            "other_s": wall - sum(t.values())}
    print(json.dumps(cont, indent=1), file=sys.stderr)
    rows.append(cont)
    if args.out:
        with open(args.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
