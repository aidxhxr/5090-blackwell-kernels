#!/usr/bin/env python3
"""Time sk.sample against the PyTorch sampling path (softmax, topk or a full sort, cumsum,
multinomial) on Llama 3 logits: B in 1, 8, 64 rows of V = 128256 bf16, for greedy, plain
temperature, top-k, top-p and both. Writes results/sample.jsonl (one JSON object per row) and
prints a table.

    python scripts/bench_sample.py
    python scripts/bench_sample.py --batch 1 64 --iters 200 --variant 0   # the reference rung
    python scripts/bench_sample.py --vocab 200058 --softcap 30 --out results/sample_softcap.jsonl

With --softcap C both paths apply Inkling 2's final logit softcap C * tanh(x / C) (torch's as
one more elementwise op on the fp32 logits), and each row also times our kernel without the
cap on the same logits, so the cap's own cost is ours_ms - ours_nocap_ms.

Times are CUDA-event medians of one call each (no graph); `torch` is the eager path a
serving loop without a fused sampler runs. The PyTorch path with top-p and no top-k sorts the
whole row, which is what the kernel's radix select avoids. Not measured yet.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

OUT = ROOT / "results" / "sample.jsonl"
V = 128256
# name, temperature, top_k, top_p
CASES = [("greedy", 0.0, 0, 1.0), ("temperature", 1.0, 0, 1.0), ("top_k=50", 1.0, 50, 1.0),
         ("top_p=0.9", 1.0, 0, 0.9), ("top_k=50,top_p=0.9", 1.0, 50, 0.9)]


def torch_sample(logits: torch.Tensor, T: float, k: int, p: float,
                 gen: torch.Generator, softcap: float | None = None) -> torch.Tensor:
    """The usual eager path: top-k by torch.topk, top-p by a sort and a cumsum, then
    torch.multinomial over what is left."""
    if T <= 0:
        return torch.argmax(logits, dim=-1)
    x = logits.float()
    if softcap is not None:
        x = torch.tanh(x / softcap) * softcap
    x = x / T
    idx = None
    if k > 0:
        x, idx = torch.topk(x, k, dim=-1)  # sorted, descending
    elif p < 1:
        x, idx = torch.sort(x, dim=-1, descending=True)
    if p < 1:
        probs = torch.softmax(x, dim=-1)
        before = probs.cumsum(dim=-1) - probs
        x = x.masked_fill(before >= p, float("-inf"))
    pick = torch.multinomial(torch.softmax(x, dim=-1), 1, generator=gen)
    return (idx.gather(-1, pick) if idx is not None else pick).squeeze(-1)


def time_ms(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return statistics.median(times)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, nargs="+", default=[1, 8, 64])
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--variant", type=int, default=-1)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--vocab", type=int, default=V)
    ap.add_argument("--softcap", type=float, default=None)
    args = ap.parse_args()
    vocab, cap = args.vocab, args.softcap

    import spark_kernels as sk

    dev = "cuda"
    device = torch.cuda.get_device_name()
    gen = torch.Generator(device=dev).manual_seed(0)
    rows = []
    print(f"{'case':<20} {'B':>3} {'ours ms':>9} {'torch ms':>9} {'speedup':>8}  {device}")
    for B in args.batch:
        # random normal logits: flatter than a model's, so more tokens land in the refined bins
        logits = (torch.randn(B, vocab, device=dev, generator=gen) * 3).to(torch.bfloat16)
        for name, T, k, p in CASES:
            temp = torch.full((B,), T, device=dev)
            top_k = torch.full((B,), k, device=dev, dtype=torch.int32)
            top_p = torch.full((B,), p, device=dev)
            seed = torch.arange(B, device=dev, dtype=torch.long)
            offset = torch.zeros(B, device=dev, dtype=torch.long)
            ours = time_ms(lambda: sk.sample(logits, temp, top_k, top_p, seed, offset,  # noqa: B023
                                             args.variant, softcap=cap), args.warmup, args.iters)
            ref = time_ms(lambda: torch_sample(logits, T, k, p, gen, cap),  # noqa: B023
                          args.warmup, args.iters)
            row = {"device": device, "kernel": "sample", "case": name, "B": B, "V": vocab,
                   "temperature": T, "top_k": k, "top_p": p, "variant": args.variant,
                   "ours_ms": ours, "torch_ms": ref, "speedup": ref / ours,
                   # the one read of the logits the kernel must do, against its time
                   "gbps": B * vocab * 2 / (ours * 1e-3) / 1e9}
            if cap is not None:
                row["softcap"] = cap
                row["ours_nocap_ms"] = time_ms(
                    lambda: sk.sample(logits, temp, top_k, top_p, seed, offset,  # noqa: B023
                                      args.variant), args.warmup, args.iters)
            rows.append(row)
            print(f"{name:<20} {B:>3} {ours:9.4f} {ref:9.4f} {ref / ours:7.2f}x")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
