#!/usr/bin/env python3
"""Inkling 2 Mini's short convolutions and head norms against the torch reference, on the GPU box.

The torch side is HF's own code (the copies in tests/test_short_conv.py): F.conv1d over the
fp32 [B, C, 4] cache state for prefill and causal_conv1d_update for decode, the dynamic conv's
unfold/flip/sum, and the norm modules. Shapes are the model's (bf16 activations):

    attn/mlp out   static conv, C = 2304                 (post_attention / post_mlp conv)
    k sliding      static conv, C = 1024, L2 clip epilogue
    v full         dynamic conv, C = 1024, 8 kv heads of 4 taps (bf16, weight_proj's output),
                   RMS epilogue
    q full         head_l2_clip over [B, T, 24, 128] with the learned scale and log scaling

at decode B = 1, 8, 64 (T = 1, the state updated in place) and an 8192-token prefill (B = 1).
Each conv shape also runs as conv-then-separate-op (sk.short_conv, then sk.head_l2_clip or
sk.head_rms_norm) to show what the fused epilogue saves. Decode rows are timed as CUDA graphs
of 50 calls (the per-call number is GPU time, without the Python launch, everything L2-warm);
prefill rows as plain launches, once L2-warm and once with the L2 overwritten before every
call. GB/s counts the bytes the op must move once (x, out, state both ways, weights, taps)
and is set against a 256 MB device-to-device copy measured in the same run; the prefill rows
also time a copy of x of the same size, the floor at that size.

    PYTHONPATH=python flock ~/sk26/gpu.lock python scripts/bench_short_conv.py
        [--out=results/short_conv.jsonl]

Appends one JSON object per row to --out and prints the rows.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tests"))
import test_short_conv as hf  # noqa: E402

import spark_kernels as sk  # noqa: E402

WARMUP, ITERS, GRAPH_CALLS = 5, 30, 50
QK = hf.QK_CLIP


def time_ms(fn, warmup=WARMUP, iters=ITERS, flush=None) -> float:
    """Median ms of fn; with `flush`, it runs before every timed call, outside the events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        if flush is not None:
            flush()
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def graph_ms(fn) -> float:
    """Per-call GPU time of fn from a CUDA graph of GRAPH_CALLS calls."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(GRAPH_CALLS):
            fn()
    return time_ms(g.replay) / GRAPH_CALLS


def copy_gbps() -> float:
    n = 256 << 20
    a = torch.empty(n, dtype=torch.uint8, device="cuda")
    b = torch.empty_like(a)
    return 2 * n / (time_ms(lambda: b.copy_(a)) * 1e6)


def conv_case(kind: str, B: int, T: int):
    """(C, {label: (fn, bytes it must move)}) for one model shape."""
    C, H, D = (2304, None, 128) if kind == "attn/mlp out" else (1024, 8, 128)
    epi = {"attn/mlp out": None, "k sliding": "l2clip", "v full": "rms"}[kind]
    dt = torch.bfloat16
    x = torch.randn(B, T, C, device="cuda", dtype=dt)
    w = 0.5 * torch.randn(C, 4, device="cuda")
    sw = 0.5 * torch.randn(4, C, device="cuda")
    taps = (0.3 * torch.randn(B, T, H, 4, device="cuda")).to(dt) if kind == "v full" else None
    state = sk.short_conv_state(B, C, dt)
    decode = T == 1

    def ours(epilogue):
        if taps is None:
            return lambda: sk.short_conv(x, w, state=state, epilogue=epilogue)
        return lambda: sk.dynamic_short_conv(x, sw, taps, state=state, epilogue=epilogue)

    sep_norm = {"l2clip": sk.head_l2_clip, "rms": sk.head_rms_norm}.get(epi)
    hf_norm = {"l2clip": lambda y: hf.hf_l2_clip(y, QK), "rms": hf.hf_unweighted_rmsnorm}.get(epi)

    def torch_ref():
        cache = hf.ConvCache()
        if decode:  # a cache left by a prefill: the in-place update path
            cache.conv_state = torch.zeros(B, C, 4, device="cuda")

        def run():
            # a prefill starts from an empty cache, as HF's first call does
            c = cache if decode else hf.ConvCache()
            if taps is None:
                y = hf.hf_short_conv(x, w, cache=c)
            else:
                y = hf.hf_dynamic_short_conv(x, sw, taps.view(B, T, -1), H, cache=c)
            return y if hf_norm is None else hf_norm(y.view(B, T, C // D, D))
        return run

    nbytes = 2 * B * T * C * 2 + 2 * B * 3 * C * 2 + 4 * C * 4
    if taps is not None:
        nbytes += B * T * H * 4 * 2
    fns = {"sk fused" if epi else "sk": (ours(epi), nbytes),
           "torch (HF code)": (torch_ref(), nbytes)}
    if epi:
        plain = ours(None)
        fns["sk conv + separate op"] = (lambda: sep_norm(plain().view(B, T, C // D, D)), nbytes)
    if not decode:  # the floor at this size: a copy of x into a tensor like it
        y = torch.empty_like(x)
        fns["copy of x (same size)"] = (lambda: y.copy_(x), 2 * x.numel() * 2)
    return C, fns


def q_case(B: int, T: int):
    H, D = 24, 128
    q = torch.randn(B, T, H, D, device="cuda", dtype=torch.bfloat16) * 3
    scale = 1.0 + 0.1 * torch.randn(H, D, device="cuda")
    pos = torch.randint(0, 32768, (B, T), device="cuda")
    nbytes = 2 * q.numel() * 2 + H * D * 4 + pos.numel() * 8
    fns = {
        "sk": (lambda: sk.head_l2_clip(q, QK, scale=scale, positions=pos), nbytes),
        "torch (HF code)": (lambda: hf.hf_log_scaled_l2_clip(q, scale, QK, pos), nbytes),
    }
    if T > 1:
        y = torch.empty_like(q)
        fns["copy of x (same size)"] = (lambda: y.copy_(q), 2 * q.numel() * 2)
    return H * D, fns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results" / "short_conv.jsonl"))
    args = ap.parse_args()
    torch.manual_seed(0)
    peak = copy_gbps()
    gpu = torch.cuda.get_device_name()
    print(f"{gpu}: device-to-device copy {peak:.0f} GB/s")
    # Prefill is timed twice: with the L2 as the last call left it (x and out, 38 MB each at
    # C = 2304, fit the 128 MB L2), and with it overwritten before every call (DRAM-bound).
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    scratch = torch.empty(2 * l2, dtype=torch.uint8, device="cuda")
    rows = []
    shapes = [(1, 1), (8, 1), (64, 1), (1, 8192)]
    for kind in ["attn/mlp out", "k sliding", "v full", "q full"]:
        for B, T in shapes:
            C, fns = q_case(B, T) if kind == "q full" else conv_case(kind, B, T)
            for label, (fn, nbytes) in fns.items():
                if T == 1:
                    timings = [("cuda graph, L2 warm", graph_ms(fn))]
                else:
                    timings = [("launch, L2 warm", time_ms(fn)),
                               ("launch, L2 flushed", time_ms(fn, flush=scratch.zero_))]
                for timing, ms in timings:
                    gbps = nbytes / (ms * 1e6)
                    row = dict(kernel="short_conv", case=kind, impl=label, B=B, T=T, C=C,
                               us=round(ms * 1e3, 2), mbytes=round(nbytes / 1e6, 3),
                               gbps=round(gbps, 1), pct_copy=round(100 * gbps / peak, 1),
                               copy_gbps=round(peak, 1), gpu=gpu, timing=timing)
                    rows.append(row)
                    print(f"{kind:13s} B={B:<3d} T={T:<5d} {label:23s} {row['us']:9.2f} us "
                          f"{row['gbps']:8.1f} GB/s {row['pct_copy']:6.1f}% of copy  {timing}")
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
