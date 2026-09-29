#!/usr/bin/env python3
"""Serve a batch of requests on the whole Llama-3-8B (32 layers, random bf16 weights) with
spark_kernels.engine, and the same batches on the same model in PyTorch, eager and with the
decode step under torch.compile. Prints the tables and writes one JSON object per row.

    python scripts/bench_serve.py                      # batches 1, 8, 32, 64, all backends
    python scripts/bench_serve.py --batches 8 --new 64 --no-compile

A static-batch row is B requests submitted at once, prompt lengths drawn log-uniformly
between --min-prompt and --max-prompt (seeded, the same for every backend), --new tokens
generated per request, greedy. It reports the prefill (every prompt admitted and prefilled
in packed batches of up to --prefill-tokens tokens, no decode in between), the decode phase
(the --new - 1 steps after it, B tokens per step, the context growing by one each step) and
the two together. The continuous row submits --requests requests with mixed prompt and output
lengths to an engine of --max-batch slots and runs it to the end: admissions, prefills,
decode steps and retirements interleave, and the throughput is every generated token over
the wall time. Times are wall clock around each phase with a device synchronize at both ends,
after a warm-up run of the same batch shape (the clock ramp, the workspaces, the CUDA graphs,
and for the compiled torch model the compilation of every decode bucket).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

OUT = ROOT / "results" / "serve.json"


def log_uniform(n: int, lo: int, hi: int, gen: torch.Generator) -> list[int]:
    u = torch.rand(n, generator=gen)
    return [int(round(math.exp(math.log(lo) + float(x) * (math.log(hi) - math.log(lo)))))
            for x in u]


def prompts_for(lens: list[int], vocab: int, gen: torch.Generator) -> list[torch.Tensor]:
    return [torch.randint(0, vocab, (n,), generator=gen) for n in lens]


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


def static_batch(eng, prompts, new: int) -> dict:
    """Prefill every prompt, then decode to the end; returns the phase times."""
    t0 = sync_time()
    for p in prompts:
        eng.submit(p, new)
    eng.prefill_waiting()
    t1 = sync_time()
    steps = 0
    while any(s is not None for s in eng.running):
        eng.step()
        steps += 1
    t2 = sync_time()
    return {"prefill_s": t1 - t0, "decode_s": t2 - t1, "steps": steps}


def warm_up(eng, buckets, vocab: int, new: int = 4) -> None:
    """A short batch of every bucket size, twice: each decode bucket has run (and, for the
    compiled torch model, compiled) before anything is timed."""
    gen = torch.Generator().manual_seed(123)
    for _ in range(2):
        for b in buckets:
            for p in prompts_for([16] * b, vocab, gen):
                eng.submit(p, new)
            eng.run()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--new", type=int, default=128, help="tokens generated per request")
    ap.add_argument("--min-prompt", type=int, default=64)
    ap.add_argument("--max-prompt", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--cache-gb", type=float, default=7.0, help="K/V cache size, all layers")
    ap.add_argument("--prefill-tokens", type=int, default=16384)
    ap.add_argument("--requests", type=int, default=256, help="continuous run; 0 skips it")
    ap.add_argument("--max-batch", type=int, default=64, help="slots of the continuous run")
    ap.add_argument("--eager", action="store_true", help="also our engine without CUDA graphs")
    ap.add_argument("--no-torch", action="store_true")
    ap.add_argument("--no-compile", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from spark_kernels import engine as E
    from spark_kernels.layer import RoPE

    batches = [int(x) for x in args.batches.split(",") if x]
    max_seq = args.max_prompt + max(args.new, 256) + 16
    weights = E.ModelWeights.random(args.layers)
    vocab = weights.embed.shape[0]
    rope = RoPE(max_seq)
    cache_bytes = int(args.cache_gb * (1 << 30))
    print(f"model: {args.layers} layers, {weights.bytes() / 1e9:.2f} GB of bf16 weights; cache "
          f"{args.cache_gb:g} GB = {cache_bytes // E.PagedKVCache.page_bytes(args.layers)} pages "
          f"of {E.PAGE} tokens", file=sys.stderr)

    backends = [("spark", lambda: E.SparkModel(weights, rope), True)]
    if args.eager:
        backends.append(("spark_eager", lambda: E.SparkModel(weights, rope), False))
    if not args.no_torch:
        backends.append(("torch", lambda: E.TorchModel(weights, rope), False))
        if not args.no_compile:
            backends.append(("torch_compiled", lambda: E.TorchModel(weights, rope, compile=True),
                             False))

    rows = []
    for name, make, graphs in backends:
        max_b = max(batches + ([args.max_batch] if args.requests else []))
        eng = E.Engine(make(), args.layers, max_batch=max_b, max_seq=max_seq,
                       cache_bytes=cache_bytes, prefill_tokens=args.prefill_tokens,
                       graphs=graphs)
        warm_up(eng, [x for x in E.BUCKETS if x <= max_b], vocab)
        for b in batches:
            gen = torch.Generator().manual_seed(args.seed + b)
            lens = log_uniform(b, args.min_prompt, args.max_prompt, gen)
            prompts = prompts_for(lens, vocab, gen)
            r = static_batch(eng, prompts, args.new)
            ptoks, dtoks = sum(lens), b * (args.new - 1)
            row = {"backend": name, "kind": "static", "batch": b, "prompt_tokens": ptoks,
                   "prompt_max": max(lens), "new": args.new, "prefill_ms": r["prefill_s"] * 1e3,
                   "prefill_tok_s": ptoks / r["prefill_s"],
                   "decode_ms_per_step": r["decode_s"] * 1e3 / max(1, r["steps"]),
                   "decode_tok_s": dtoks / r["decode_s"],
                   "total_tok_s": b * args.new / (r["prefill_s"] + r["decode_s"])}
            rows.append(row)
            print(f"{name:15s} B={b:3d} prompts {ptoks:6d} tok (max {max(lens):4d}): prefill "
                  f"{row['prefill_ms']:8.1f} ms {row['prefill_tok_s']:9.0f} tok/s | decode "
                  f"{row['decode_ms_per_step']:6.2f} ms/step {row['decode_tok_s']:7.0f} tok/s | "
                  f"total {row['total_tok_s']:7.0f} tok/s", file=sys.stderr)
        if args.requests:
            gen = torch.Generator().manual_seed(args.seed + 1000)
            lens = log_uniform(args.requests, args.min_prompt, args.max_prompt, gen)
            news = [int(x) for x in torch.randint(16, 2 * args.new, (args.requests,),
                                                  generator=gen)]
            prompts = prompts_for(lens, vocab, gen)
            eng.stats = E.Stats()
            t0 = sync_time()
            for p, n in zip(prompts, news, strict=True):
                eng.submit(p, n)
            st = eng.run()
            dt = sync_time() - t0
            row = {"backend": name, "kind": "continuous", "requests": args.requests,
                   "max_batch": args.max_batch, "prompt_tokens": sum(lens),
                   "generated": st.generated, "wall_s": dt, "gen_tok_s": st.generated / dt,
                   "total_tok_s": (st.generated + sum(lens)) / dt,
                   "decode_steps": st.decode_steps, "prefill_batches": st.prefill_batches,
                   "mean_batch": st.decode_tokens / max(1, st.decode_steps)}
            rows.append(row)
            print(f"{name:15s} continuous: {args.requests} requests, {sum(lens)} prompt tokens, "
                  f"{st.generated} generated in {dt:.2f} s: {row['gen_tok_s']:.0f} generated "
                  f"tok/s, {st.decode_steps} decode steps (mean batch {row['mean_batch']:.1f}), "
                  f"{st.prefill_batches} prefill batches", file=sys.stderr)
        del eng
        torch.cuda.empty_cache()

    # the ratios against the torch backends, per batch
    by = {(r["backend"], r.get("batch")): r for r in rows if r["kind"] == "static"}
    for b in batches:
        s = by.get(("spark", b))
        for ref in ("torch", "torch_compiled"):
            t = by.get((ref, b))
            if s and t:
                print(f"B={b:3d} vs {ref:15s}: prefill "
                      f"{s['prefill_tok_s'] / t['prefill_tok_s']:.2f}x, decode "
                      f"{s['decode_tok_s'] / t['decode_tok_s']:.2f}x, total "
                      f"{s['total_tok_s'] / t['total_tok_s']:.2f}x", file=sys.stderr)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
