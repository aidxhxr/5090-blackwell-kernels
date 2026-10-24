#!/usr/bin/env python3
"""What batch invariance costs: the engine and its kernels, default against invariant.

    python scripts/bench_invariant.py --part kernels   # hgemm 6 vs 7, paged ops 1 vs 2
    python scripts/bench_invariant.py --part engine    # decode ms/step and prefill tok/s

The kernel rows time each op on its own: hgemm variant 6 (the default) against variant 7 on
the Llama-3-8B projections at decode and prefill M, paged_decode and attention_varlen variant
1 against variant 2 on the batches of docs/design/serving.md. Each row is the median of
--iters samples of ten launches queued back to back between CUDA events, after a warm-up,
the weights (or the K/V pages) rotated over copies past the 96 MB L2 when one copy would fit
in it, and the two variants' outputs checked against each other first.

The engine rows are scripts/bench_serve.py's static batches (prompts log-uniform in 64..2048,
--new tokens each, greedy) on the same random weights, once with the default kernels and once
with Engine(batch_invariant=True). --layers defaults to 8 of Llama-3-8B's 32, so the weights,
the 1 GB lm_head and the cache stay under 10 GB; a step's layer part scales with the layer
count, the lm_head part does not.
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
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "batch_invariant.jsonl"

# Llama-3-8B: (name, K, N) of every GEMM a layer runs, and the lm_head.
GEMMS = [("qkv", 4096, 6144), ("o", 4096, 4096), ("gate_up", 4096, 28672),
         ("down", 14336, 4096), ("lm_head", 4096, 128256)]
L2 = 96 << 20


def time_ms(fn, iters: int, warm: int = 5, reps: int = 10) -> float:
    """Per-launch time of fn: the median over `iters` samples of `reps` launches queued back
    to back between two CUDA events, so the host's launch path is hidden as in a model."""
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            fn()
        b.record()
        b.synchronize()
        ts.append(a.elapsed_time(b) / reps)
    ts.sort()
    return ts[len(ts) // 2]


def rotating(fn):
    """fn(i) as a no-argument callable that passes 0, 1, 2, ... on successive calls (the copy
    to read next)."""
    n = [0]

    def call():
        fn(n[0])
        n[0] += 1

    return call


def copies_for(nbytes: int) -> int:
    """Copies whose total is past three times the L2 (one copy if it is already)."""
    return 1 if nbytes > 3 * L2 else max(2, math.ceil(3 * L2 / nbytes))


def bench_gemm(sk, ms, iters, rows):
    if not ms:
        return
    for name, k, n in GEMMS:
        nb = k * n * 2
        ncopy = copies_for(nb) if max(ms) <= 64 else 1
        g = torch.Generator(device="cuda").manual_seed(0)
        bs = [torch.empty(k, n, device="cuda", dtype=torch.bfloat16).normal_(
            0, 1 / math.sqrt(k), generator=g) for _ in range(ncopy)]
        for m in ms:
            if name == "lm_head" and m > 2048:
                continue  # a prefill takes the logits of each prompt's last token only
            a = torch.randn(m, k, device="cuda").to(torch.bfloat16)
            c6, c7 = sk.hgemm(a, bs[0], 6), sk.hgemm(a, bs[0], 7)
            assert torch.allclose(c6[:64].float(), c7[:64].float(), atol=3e-2, rtol=3e-2)
            same = bool(torch.equal(c6, c7))  # variant 6 split nothing on this shape
            del c6, c7
            t = {v: time_ms(rotating(lambda i, a=a, bs=bs, v=v: sk.hgemm(a, bs[i % len(bs)], v)),
                            iters) for v in (6, 7)}
            flops = 2.0 * m * n * k
            row = {"kind": "hgemm", "gemm": name, "M": m, "N": n, "K": k, "copies": ncopy,
                   "same_bits": same,
                   "v6_ms": t[6], "v7_ms": t[7], "v7_over_v6": t[7] / t[6],
                   "v6_tflops": flops / t[6] / 1e9, "v7_tflops": flops / t[7] / 1e9}
            rows.append(row)
            print(f"hgemm {name:8s} {m:5d} x {n:6d} x {k:5d}: v6 {t[6] * 1e3:9.1f} us  v7 "
                  f"{t[7] * 1e3:9.1f} us  v7/v6 {t[7] / t[6]:.3f}{'  same bits' if same else ''}",
                  file=sys.stderr)
        del bs
        torch.cuda.empty_cache()


H_Q, H_KV, D, PAGE = 32, 8, 128, 16


def paged_inputs(lens, copies, seed=0):
    """A page pool holding `copies` replicas of the batch's pages; block tables per copy."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    npg = [(n + PAGE - 1) // PAGE for n in lens]
    total = sum(npg)
    k = torch.empty(copies * total + 1, H_KV, PAGE, D, device="cuda",
                    dtype=torch.bfloat16).normal_(generator=g)
    v = torch.empty_like(k).normal_(generator=g)
    base = torch.zeros(len(lens), max(npg), dtype=torch.int32)
    i = 0
    for b, n in enumerate(npg):
        base[b, :n] = torch.arange(i, i + n, dtype=torch.int32)
        i += n
    bts = [(base + c * total).cuda() for c in range(copies)]
    return k, v, bts, torch.tensor(lens, dtype=torch.int32, device="cuda")


DECODE_BATCHES = {"1x4096": [4096], "1x32768": [32768], "8x4096": [4096] * 8,
                  "32x2048": [2048] * 32, "64x1024": [1024] * 64, "64x512": [512] * 64,
                  "1x32768+50x500": [32768] + [500] * 50}
PREFILL_BATCHES = {"1x4096": [4096], "4x2048": [2048] * 4,
                   "16xU(64,1024)": [64, 1024, 300, 700, 128, 900, 500, 250, 1000, 80, 600, 420,
                                     760, 200, 350, 900]}


def bench_paged(sk, iters, rows):
    for name, lens in DECODE_BATCHES.items():
        kv = sum(lens) * H_KV * D * 2 * 2
        ncopy = copies_for(kv)
        k, v, bts, sl = paged_inputs(lens, ncopy)
        q = torch.randn(len(lens), H_Q, D, device="cuda").to(torch.bfloat16)
        a, b = sk.paged_decode(q, k, v, bts[0], sl, 1), sk.paged_decode(q, k, v, bts[0], sl, 2)
        assert torch.allclose(a.float(), b.float(), atol=2e-2, rtol=2e-2)
        t = {var: time_ms(rotating(lambda i, q=q, k=k, v=v, bts=bts, sl=sl, var=var:
                                   sk.paged_decode(q, k, v, bts[i % len(bts)], sl, var)), iters)
             for var in (1, 2)}
        row = {"kind": "paged_decode", "batch": name, "keys": sum(lens), "kv_mb": kv / 1e6,
               "copies": ncopy, "v1_ms": t[1], "v2_ms": t[2], "v2_over_v1": t[2] / t[1],
               "v1_gbps": kv / t[1] / 1e6, "v2_gbps": kv / t[2] / 1e6}
        rows.append(row)
        print(f"paged_decode {name:16s} {kv / 1e6:7.1f} MB: v1 {t[1] * 1e3:8.1f} us  v2 "
              f"{t[2] * 1e3:8.1f} us  v2/v1 {t[2] / t[1]:.3f}", file=sys.stderr)
        del k, v
        torch.cuda.empty_cache()
    for name, lens in PREFILL_BATCHES.items():
        k, v, bts, sl = paged_inputs(lens, 1)
        T = sum(lens)
        q = torch.randn(T, H_Q, D, device="cuda").to(torch.bfloat16)
        cu = torch.tensor([0] + lens, dtype=torch.int32).cumsum(0, dtype=torch.int32).cuda()
        a = sk.attention_varlen(q, k, v, cu, sl, bts[0], True, 1)
        b = sk.attention_varlen(q, k, v, cu, sl, bts[0], True, 2)
        assert torch.allclose(a.float(), b.float(), atol=2e-2, rtol=2e-2)
        t = {var: time_ms(rotating(lambda i, q=q, k=k, v=v, cu=cu, sl=sl, bt=bts[0], var=var:
                                   sk.attention_varlen(q, k, v, cu, sl, bt, True, var)), iters)
             for var in (1, 2)}
        flops = sum(4.0 * H_Q * D * n * n / 2 for n in lens)  # causal: half the products
        row = {"kind": "attention_varlen", "batch": name, "tokens": T, "v1_ms": t[1],
               "v2_ms": t[2], "v2_over_v1": t[2] / t[1], "v1_tflops": flops / t[1] / 1e9,
               "v2_tflops": flops / t[2] / 1e9}
        rows.append(row)
        print(f"attention_varlen {name:14s}: v1 {t[1] * 1e3:8.1f} us ({row['v1_tflops']:.0f} "
              f"TFLOPS)  v2 {t[2] * 1e3:8.1f} us ({row['v2_tflops']:.0f})  v2/v1 "
              f"{t[2] / t[1]:.3f}", file=sys.stderr)
        del k, v
        torch.cuda.empty_cache()


def bench_engine(args, rows):
    from bench_serve import log_uniform, prompts_for, static_batch, warm_up

    from spark_kernels import engine as E
    from spark_kernels.layer import RoPE

    batches = [int(x) for x in args.batches.split(",") if x]
    max_seq = args.max_prompt + max(args.new, 256) + 16
    weights = E.ModelWeights.random(args.layers)
    vocab = weights.embed.shape[0]
    rope = RoPE(max_seq)
    cache_bytes = int(args.cache_gb * (1 << 30))
    print(f"model: {args.layers} layers, {weights.bytes() / 1e9:.2f} GB of bf16 weights, "
          f"{args.cache_gb:g} GB of cache", file=sys.stderr)
    modes = [m for m in args.modes.split(",") if m]
    for mode in modes:
        inv = mode == "invariant"
        eng = E.Engine(E.SparkModel(weights, rope, batch_invariant=inv), args.layers,
                       max_batch=max(batches), max_seq=max_seq, cache_bytes=cache_bytes,
                       prefill_tokens=args.prefill_tokens, graphs=True)
        warm_up(eng, [x for x in E.BUCKETS if x <= max(batches)], vocab)
        for b in batches:
            gen = torch.Generator().manual_seed(args.seed + b)
            lens = log_uniform(b, args.min_prompt, args.max_prompt, gen)
            r = static_batch(eng, prompts_for(lens, vocab, gen), args.new)
            ptoks = sum(lens)
            row = {"kind": "engine", "mode": mode, "layers": args.layers, "batch": b,
                   "prompt_tokens": ptoks, "new": args.new,
                   "prefill_tok_s": ptoks / r["prefill_s"],
                   "decode_ms_per_step": r["decode_s"] * 1e3 / max(1, r["steps"]),
                   "decode_tok_s": b * (args.new - 1) / r["decode_s"]}
            rows.append(row)
            print(f"{mode:10s} B={b:3d} prompts {ptoks:6d}: prefill {row['prefill_tok_s']:8.0f} "
                  f"tok/s | decode {row['decode_ms_per_step']:6.3f} ms/step", file=sys.stderr)
        if args.requests:
            # bench_serve's continuous run: admissions, mixed steps, retirements interleaved
            gen = torch.Generator().manual_seed(args.seed + 1000)
            lens = log_uniform(args.requests, args.min_prompt, args.max_prompt, gen)
            news = [int(x) for x in torch.randint(16, 2 * args.new, (args.requests,),
                                                  generator=gen)]
            eng.stats = E.Stats()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for p, n in zip(prompts_for(lens, vocab, gen), news, strict=True):
                eng.submit(p, n)
            st = eng.run()
            dt = time.perf_counter() - t0
            row = {"kind": "engine_continuous", "mode": mode, "layers": args.layers,
                   "requests": args.requests, "max_batch": max(batches),
                   "prompt_tokens": sum(lens), "generated": st.generated, "wall_s": dt,
                   "gen_tok_s": st.generated / dt, "mixed_rows": st.mixed_rows,
                   "preempted": st.preempted}
            rows.append(row)
            print(f"{mode:10s} continuous: {args.requests} requests in {dt:.2f} s, "
                  f"{row['gen_tok_s']:.0f} generated tok/s, {st.mixed_rows} mixed rows, "
                  f"{st.preempted} preempted", file=sys.stderr)
        del eng
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--part", choices=("kernels", "gemm", "attention", "engine"),
                    default="kernels")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--ms", default="1,8,32,64,128,512,2048,4096", help="hgemm M values")
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--modes", default="default,invariant")
    ap.add_argument("--new", type=int, default=32, help="tokens generated per request")
    ap.add_argument("--min-prompt", type=int, default=64)
    ap.add_argument("--max-prompt", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--cache-gb", type=float, default=2.0, help="K/V cache size, all layers")
    ap.add_argument("--prefill-tokens", type=int, default=16384)
    ap.add_argument("--requests", type=int, default=0, help="a continuous run of this many")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    import spark_kernels as sk

    rows: list[dict] = []
    if args.part in ("kernels", "gemm"):
        ms = [int(x) for x in args.ms.split(",") if x]
        bench_gemm(sk, [m for m in ms if m <= 64], args.iters, rows)
        bench_gemm(sk, [m for m in ms if m > 64], max(10, args.iters // 5), rows)
    if args.part in ("kernels", "attention"):
        bench_paged(sk, args.iters, rows)
    if args.part == "engine":
        bench_engine(args, rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"appended {len(rows)} rows to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
