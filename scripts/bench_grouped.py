#!/usr/bin/env python3
"""Time sk.grouped_gemm against the torch loop over experts most MoE code runs, on the
expert projections of three models: Mixtral 8x7B (8 experts, top-2, 4096 -> 14336),
Qwen3-30B-A3B (128, top-8, 2048 -> 768) and a DeepSeek V3-like layer (256, top-8,
7168 -> 2048), at 512, 2048 and 8192 tokens. Routing is reference.moe_route (seeded, with a
per-expert bias so the load is uneven). Writes results/grouped_gemm_torch.jsonl and prints a
table.

    python scripts/bench_grouped.py
    python scripts/bench_grouped.py --model qwen3 --tokens 2048 --iters 50

Rows per shape (CUDA-event medians):
  torch_loop       for e: out[lo:hi] = a[lo:hi] @ w[e].T, the counts already on the host
  torch_loop_sync  the same plus the offsets.tolist() a caller does to get them (a sync)
  grouped          sk.grouped_gemm, offsets on the device only
  torch_fp8_loop   torch._scaled_mm per expert (rows padded to 16, which it needs), counts
                   on the host
  grouped_fp8      sk.grouped_gemm on e4m3 a and w with per-expert scales
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

OUT = ROOT / "results" / "grouped_gemm_torch.jsonl"
MODELS = {
    "mixtral": dict(E=8, k=2, K=4096, N=14336),
    "qwen3": dict(E=128, k=8, K=2048, N=768),
    "deepseek": dict(E=256, k=8, K=7168, N=2048),
}


def time_ms(fn, iters: int, warmup: int = 3) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        out.append(s.elapsed_time(e))
    return statistics.median(out)


def bench_shape(sk, ref, args, name, m, w, T):
    """Check and time one model at one token count; returns the result rows."""
    E, k, K, N = m["E"], m["k"], m["K"], m["N"]
    rows = []
    ids = ref.moe_route(T, E, k, seed=T)
    order, offsets = sk.moe_offsets(ids, E)
    rows_n = T * k
    a = torch.empty(rows_n, K, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
    offs = offsets.tolist()
    flops = 2.0 * rows_n * N * K
    shape = f"{name} t={T}"

    def loop(offs=offs):
        out = torch.empty(rows_n, N, device="cuda", dtype=torch.bfloat16)
        for e in range(E):
            lo, hi = offs[e], offs[e + 1]
            if hi > lo:
                torch.matmul(a[lo:hi], w[e].t(), out=out[lo:hi])
        return out

    def loop_sync():
        return loop(offsets.tolist())

    got = sk.grouped_gemm(a, w, offsets)
    want = loop()
    err = (got.float() - want.float()).abs().max().item()
    scale = want.float().abs().max().item()
    assert err <= 1e-2 * scale, f"{shape}: grouped bf16 off by {err} (max {scale})"

    t_loop = time_ms(loop, args.iters)
    res = {"torch_loop": t_loop,
           "torch_loop_sync": time_ms(loop_sync, args.iters),
           "grouped": time_ms(lambda: sk.grouped_gemm(a, w, offsets), args.iters)}

    if not args.no_fp8:
        a8 = a.to(torch.float8_e4m3fn)
        w8 = w.to(torch.float8_e4m3fn)
        sa = torch.tensor([1.0], device="cuda")
        sb = torch.ones(E, device="cuda")
        one = torch.tensor(1.0, device="cuda")
        # _scaled_mm wants M a multiple of 16: each expert's rows padded on a copy.
        pad = [((offs[e + 1] - offs[e] + 15) // 16) * 16 for e in range(E)]
        poffs = [0]
        for p_ in pad:
            poffs.append(poffs[-1] + p_)
        a8p = torch.zeros(poffs[-1], K, device="cuda", dtype=torch.float8_e4m3fn)
        for e in range(E):
            n_ = offs[e + 1] - offs[e]
            a8p[poffs[e]:poffs[e] + n_] = a8[offs[e]:offs[e + 1]]

        def loop8():
            out = torch.empty(poffs[-1], N, device="cuda", dtype=torch.bfloat16)
            for e in range(E):
                lo, hi = poffs[e], poffs[e + 1]
                if hi > lo:
                    out[lo:hi] = torch._scaled_mm(a8p[lo:hi], w8[e].t(), scale_a=one,
                                                  scale_b=one, out_dtype=torch.bfloat16)
            return out

        got8 = sk.grouped_gemm(a8, w8, offsets, sa, sb)
        want8 = ref.grouped_gemm(a8, w8, offsets, sa, sb)
        err8 = (got8.float() - want8.float()).abs().max().item()
        scale8 = want8.float().abs().max().item()
        assert err8 <= 1e-2 * scale8, f"{shape}: grouped fp8 off by {err8}"
        try:
            res["torch_fp8_loop"] = time_ms(loop8, args.iters)
        except RuntimeError as exc:  # no fp8 _scaled_mm on this build
            print(f"# torch._scaled_mm failed: {exc}")
        res["grouped_fp8"] = time_ms(lambda: sk.grouped_gemm(a8, w8, offsets, sa, sb),
                                     args.iters)

    for impl, ms in res.items():
        tf = flops / (ms * 1e-3) / 1e12
        print(f"{shape:<20} {impl:<16} {ms:9.4f} {tf:8.1f} {t_loop / ms:7.2f}x")
        rows.append({"device": torch.cuda.get_device_name(), "kernel": "grouped_gemm",
                     "impl": impl, "shape": shape, "E": E, "top_k": k, "K": K, "N": N,
                     "tokens": T, "rows": rows_n, "median_ms": ms, "tflops": tf})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="*", default=list(MODELS))
    ap.add_argument("--tokens", nargs="*", type=int, default=[512, 2048, 8192])
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-fp8", action="store_true")
    args = ap.parse_args()

    import spark_kernels as sk
    from spark_kernels import reference as ref

    rows = []
    print(f"{'shape':<20} {'impl':<16} {'ms':>9} {'TFLOPS':>8} {'vs loop':>8}")
    for name in args.model:
        m = MODELS[name]
        E, K, N = m["E"], m["K"], m["N"]
        w = torch.empty(E, N, K, device="cuda", dtype=torch.bfloat16).uniform_(-1, 1)
        for T in args.tokens:
            rows += bench_shape(sk, ref, args, name, m, w, T)
        del w
        torch.cuda.empty_cache()

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
