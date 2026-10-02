#!/usr/bin/env python3
"""Speed and memory of the engine's SparkModel on a real Llama-3-8B checkpoint with its
projections in one weight format (spark_kernels.quant), one format per run so only one model
is ever on the card.

    python scripts/bench_quant.py ~/models/llama3-8b-instruct --format int4
    python scripts/bench_quant.py MODEL --format nvfp4 --profile

What it measures, after a warm-up of every shape:

    weights     the bytes the model holds (projections in the format, the rest bf16) and
                the seconds the quantization at construction took
    prefill     2048-token prompts, one alone and --prefill-batch packed into one prefill,
                tokens per second (median of --reps)
    decode      batches of 1, 8 and 32 (--batches) with --context-token prompts, --new decode
                steps each on the CUDA-graph path, ms per step and tokens per second

With --profile it also runs one eager decode step at batch 1 and one prefill of a 2048-token
prompt under torch.profiler and prints the GPU time per kernel family, which says what the
format's step spends on GEMMs, on quantizing activations and on the rest.

Each run updates the format's entry of --out (results/llm_quant.json), a dict keyed by
format (plus "+head-<format>" with --head-format), and copies that format's perplexity rows
from --ppl (results/llm_ppl.jsonl) into it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from bench_serve import static_batch, sync_time  # noqa: E402


def family(name: str) -> str:
    """A kernel name folded into the family a breakdown reports."""
    n = name.lower()
    for key, fam in (("w4_", "gemm (w4gemm)"), ("fp4gemm", "gemm (fp4gemm)"),
                     ("fp8gemm", "gemm (fp8gemm)"), ("hgemm", "gemm (hgemm)"),
                     ("quantize_", "activation quantize"), ("amax_kernel", "activation quantize"),
                     ("memset", "memset"), ("paged_decode", "attention"),
                     ("varlen", "attention"), ("attention", "attention"),
                     ("rope", "rope + append"), ("rmsnorm", "norms"), ("swiglu", "swiglu")):
        if key in n:
            return fam
    if "gemm" in n or "cutlass" in n or "nvjet" in n:
        return "gemm (torch)"
    return "torch elementwise / reductions"


def profile(fn) -> dict[str, float]:
    """GPU microseconds per kernel family of one call of fn."""
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as tprofile

    fn()
    torch.cuda.synchronize()
    with tprofile(activities=[ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    out: dict[str, float] = defaultdict(float)
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            out[family(e.name)] += e.device_time
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--format", default="bf16")
    ap.add_argument("--head-format", default="bf16", help="lm_head format")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--context", type=int, default=512, help="prompt tokens of a decode row")
    ap.add_argument("--new", type=int, default=129, help="tokens per request (decode steps + 1)")
    ap.add_argument("--prefill-len", type=int, default=2048)
    ap.add_argument("--prefill-batch", type=int, default=4)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cache-gb", type=float, default=4.0)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "results" / "llm_quant.json"))
    ap.add_argument("--ppl", default=str(ROOT / "results" / "llm_ppl.jsonl"))
    args = ap.parse_args()

    from spark_kernels import engine as E
    from spark_kernels import hf

    batches = [int(x) for x in args.batches.split(",") if x]
    w, cfg = hf.load(args.model)
    max_seq = max(args.prefill_len, args.context + args.new) + 16
    rope = cfg.rope(max_seq)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model = E.SparkModel(w, rope, weights_format=args.format, head_format=args.head_format)
    torch.cuda.synchronize()
    quant_s = time.perf_counter() - t0
    del w
    torch.cuda.empty_cache()
    key = args.format + ("" if args.head_format == "bf16" else f"+head-{args.head_format}")
    row = {"format": args.format, "head_format": args.head_format,
           "weight_bytes": model.weight_bytes(),
           "allocated_bytes": torch.cuda.memory_allocated(), "quantize_s": quant_s}
    print(f"{args.format}: weights {row['weight_bytes'] / 2**30:.2f} GiB, allocated "
          f"{row['allocated_bytes'] / 2**30:.2f} GiB, quantized in {quant_s:.1f} s",
          file=sys.stderr)

    max_b = max(batches + [args.prefill_batch])
    eng = E.Engine(model, cfg.n_layers, max_batch=max_b, max_seq=max_seq,
                   cache_bytes=int(args.cache_gb * 2**30),
                   prefill_tokens=args.prefill_batch * args.prefill_len, graphs=True)
    gen = torch.Generator().manual_seed(0)

    def prompts(n, length):
        return [torch.randint(0, cfg.vocab, (length,), generator=gen) for _ in range(n)]

    # prefill: k prompts of prefill_len in one packed batch, max_new = 1 (retired at once)
    for k in sorted({1, args.prefill_batch}):
        times = []
        for rep in range(args.reps + 1):
            ps = prompts(k, args.prefill_len)
            t0 = sync_time()
            for p in ps:
                eng.submit(p, 1)
            eng.prefill_waiting()
            dt = sync_time() - t0
            if rep:  # the first is the warm-up
                times.append(dt)
        dt = statistics.median(times)
        row[f"prefill_{k}x{args.prefill_len}_ms"] = dt * 1e3
        row[f"prefill_{k}x{args.prefill_len}_tok_s"] = k * args.prefill_len / dt
        print(f"prefill {k} x {args.prefill_len}: {dt * 1e3:8.1f} ms "
              f"{k * args.prefill_len / dt:9.0f} tok/s", file=sys.stderr)

    # decode: B prompts of `context` tokens, then new - 1 graphed decode steps
    for b in batches:
        static_batch(eng, prompts(b, args.context), 8)  # warm-up of this bucket
        r = static_batch(eng, prompts(b, args.context), args.new)
        ms = r["decode_s"] * 1e3 / r["steps"]
        row[f"decode_b{b}_ms_per_step"] = ms
        row[f"decode_b{b}_tok_s"] = b * 1e3 / ms
        print(f"decode B={b:3d}: {ms:6.2f} ms/step {b * 1e3 / ms:8.0f} tok/s", file=sys.stderr)

    if args.profile:
        eng.ids[:1] = 1
        eng.positions[:1] = 0
        eng.seq_lens[:1] = 1
        prof_d = profile(lambda: eng._decode_once(1, 1))
        p = prompts(1, args.prefill_len)[0]

        def pre():
            eng.submit(p, 1)
            eng.prefill_waiting()

        prof_p = profile(pre)
        row["profile_decode_b1_us"] = prof_d
        row[f"profile_prefill_{args.prefill_len}_us"] = prof_p
        for name, prof in (("decode B=1 (eager)", prof_d),
                           (f"prefill {args.prefill_len}", prof_p)):
            tot = sum(prof.values())
            print(f"{name}: {tot / 1e3:.2f} ms of kernels", file=sys.stderr)
            for fam, us in prof.items():
                print(f"  {fam:32s} {us / 1e3:8.3f} ms {100 * us / tot:5.1f}%", file=sys.stderr)

    ppl_path = Path(args.ppl)
    if ppl_path.exists():  # every perplexity row of this format and head (with --awq or not)
        row["ppl"] = []
        for line in ppl_path.read_text().splitlines():
            r = json.loads(line)
            if (r.get("backend") == "spark" and r.get("format", "bf16") == args.format
                    and r.get("head_format", "bf16") == args.head_format):
                row["ppl"].append({k: r[k] for k in ("label", "ppl", "windows", "ctx", "kl",
                                                     "top1", "awq") if k in r})
    out = Path(args.out)
    data = json.loads(out.read_text()) if out.exists() else {}
    data[key] = row
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
