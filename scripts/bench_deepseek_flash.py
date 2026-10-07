#!/usr/bin/env python3
"""The decode step of one DeepSeek V4.1 Flash MoE layer with its routed experts at 1 or 2
bits (spark_kernels.moe.MoEBlock1Bit over w1gemm_moe), on random weights.

    python scripts/bench_deepseek_flash.py                # 1 bit, T in 1, 4, 16, 64
    python scripts/bench_deepseek_flash.py --bits 2 --check --layers 40

Prints the expert footprint per layer and for --layers layers at 1 bit (or --bits), int4,
MXFP4 and bf16, and how many layers of routed experts fit in the free memory of the card at
that width; then for each decode batch T the median over rounds of 100 CUDA-event timed
forwards (after warm-up, the same timing as scripts/bench_layer.py), with the GB/s of expert
bytes the step actually touched (the distinct experts the batch routed to, 6 per token, plus
the shared expert) and the tokens/s a model of --layers such layers would do with nothing
else in it. --check runs forward against forward_reference on T=8 and prints the max abs
error. Writes nothing.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))

try:
    from spark_kernels import moe  # noqa: E402
except ImportError:  # no extension on this machine: the footprint table still prints
    import importlib.util

    _path = Path(__file__).resolve().parent.parent / "python" / "spark_kernels" / "moe.py"
    _spec = importlib.util.spec_from_file_location("spark_kernels_moe", _path)
    moe = importlib.util.module_from_spec(_spec)
    sys.modules["spark_kernels_moe"] = moe
    _spec.loader.exec_module(moe)

BATCHES = (1, 4, 16, 64)
FORMAT_NAMES = {"w1": "{bits}-bit", "int4": "int4", "mxfp4": "mxfp4", "bf16": "bf16"}


def time_ms(fn, warmup: int, iters: int, rounds: int = 5) -> float:
    """Median over `rounds` of the mean time of `iters` back-to-back calls of fn."""
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e3 < 300:  # the clocks up before the first sample
        fn()
        torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(rounds):
        start.record()
        for _ in range(iters):
            fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop) / iters)
    return statistics.median(samples)


def footprint_table(cfg: moe.MoEConfig, layers: int, free_bytes: int | None) -> None:
    print(f"routed experts of one layer ({cfg.n_routed} x 3 x {cfg.hidden} x {cfg.inter}) "
          f"and of {layers} layers:")
    print(f"  {'format':10} {'per layer':>12} {f'{layers} layers':>12}")
    for fmt in ("w1", "int4", "mxfp4", "bf16"):
        per = cfg.bytes_per_layer(fmt)
        name = FORMAT_NAMES[fmt].format(bits=cfg.bits)
        print(f"  {name:10} {per / 1e9:9.2f} GB {layers * per / 1e9:9.1f} GB")
    if free_bytes is not None:
        per = cfg.bytes_per_layer("w1")
        print(f"  free on the card: {free_bytes / 1e9:.1f} GB, {free_bytes // per} layers of "
              f"{cfg.bits}-bit routed experts fit")


def touched_bytes(block: moe.MoEBlock1Bit, x: torch.Tensor) -> int:
    """Expert weight bytes a step over x reads: the distinct experts it routes to, each
    packed + scales, plus the shared expert."""
    ids, _ = block.routing(x)
    n = int(ids.unique().numel())
    per_expert = block.nbytes() // block.cfg.n_routed
    return n * per_expert + block.shared_nbytes()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bits", type=int, default=1, choices=(1, 2))
    ap.add_argument("--layers", type=int, default=40, help="layers for the totals and tokens/s")
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--check", action="store_true", help="forward vs forward_reference, T=8")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = moe.MoEConfig(**{**moe.DEEPSEEK_V41_FLASH.__dict__, "bits": args.bits})
    free = torch.cuda.mem_get_info()[0] if torch.cuda.is_available() else None
    footprint_table(cfg, args.layers, free)
    if not torch.cuda.is_available():
        print("no CUDA device: footprint only")
        return

    g = torch.Generator(device="cuda").manual_seed(args.seed)
    t0 = time.perf_counter()
    block = moe.MoEBlock1Bit.random(cfg, "cuda", g)
    torch.cuda.synchronize()
    print(f"random block at {cfg.bits} bit: {block.nbytes() / 1e9:.2f} GB routed, "
          f"{block.shared_nbytes() / 1e6:.0f} MB shared, built in {time.perf_counter() - t0:.1f} s")

    if args.check:
        x = torch.randn(8, cfg.hidden, device="cuda", generator=g).to(torch.bfloat16)
        y = block.forward(x)
        y_ref = block.forward_reference(x)
        err = (y.float() - y_ref.float()).abs().max().item()
        scale = y_ref.float().abs().max().item()
        print(f"check T=8: max abs error {err:.3e} (max |ref| {scale:.3e})")

    print(f"\n  {'T':>4} {'ms':>9} {'experts':>8} {'GB':>7} {'GB/s':>8} {'tok/s':>9}")
    for T in (int(b) for b in args.batches.split(",")):
        x = torch.randn(T, cfg.hidden, device="cuda", generator=g).to(torch.bfloat16)
        ms = time_ms(lambda x=x: block.forward(x), args.warmup, args.iters)
        nbytes = touched_bytes(block, x)
        n_experts = int(block.routing(x)[0].unique().numel())
        tok_s = T * 1e3 / (args.layers * ms)
        print(f"  {T:4d} {ms:9.3f} {n_experts:8d} {nbytes / 1e9:7.2f} {nbytes / ms / 1e6:8.0f} "
              f"{tok_s:9.0f}")


if __name__ == "__main__":
    main()
