#!/usr/bin/env python3
"""A whole decode step of DeepSeek V4.1 Flash's 40 MoE layers at 1 or 2 bits on one device,
for the DGX Spark's 128 GB: every layer's routed experts resident at once where the memory
allows, one layer re-used 40 times where it does not (the 5090), and the step time through
MoEBlock1Bit.decode (the fused w1moe_decode kernel plus the shared expert) layer by layer.

    python scripts/bench_deepseek_spark.py                 # 1 bit, T in 1, 4, 8
    python scripts/bench_deepseek_spark.py --bits 2 --layers 40 --variant 1
    python scripts/bench_deepseek_spark.py --footprint     # the tables only, no device needed

Prints the whole-model footprint table (routed experts at the chosen width, the shared
experts, the attention weights and the embeddings at bf16 and int4, the K/V cache at 32K and
128K context in bf16 and e4m3) against the device's memory, then builds the layers from
random packed codes (MoEBlock1Bit.random_bits: no bf16 source, a layer is written at the
speed of its 1.9 GB) and times, for each T, the step over all layers: the median over rounds
of CUDA-event timed loops of --iters steps, with the expert bytes the step touched, the GB/s
and the tokens/s the MoE part alone bounds. The attention, the norms, the embedding and the
head are not run: this is the expert floor of a step, not a model. Unmeasured so far, written
without a GPU in reach. Writes nothing.
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

BATCHES = (1, 4, 8)
GB10_BYTES_PER_S = 273e9
RTX5090_BYTES_PER_S = 1532e9
# The attention parameters per layer are not in MoEConfig: the model's 552B total minus the
# experts, the shared experts, the embedding and the head leaves about 5.5B over 40 layers,
# 140M per layer, for the attention projections, the norms and the router. A derived number,
# not one from a config file (docs/design/moe_spark.md).
ATTN_PARAMS_PER_LAYER = 140_000_000
KV_PER_TOKEN_PER_LAYER = 2 * 512  # one K/V head of dim 512: K and V values per token
CONTEXTS = (32 * 1024, 128 * 1024)


def footprint_rows(cfg: moe.MoEConfig, model: moe.ModelConfig) -> list[tuple[str, float]]:
    """(name, bytes) of every part of the model, the experts at cfg.bits."""
    L = model.layers
    shared = cfg.n_shared * cfg.expert_params()
    emb = model.vocab * cfg.hidden
    rows = [
        (f"routed experts, {cfg.bits}-bit g128", L * cfg.bytes_per_layer("w1")),
        (f"shared experts, {cfg.bits}-bit g128", L * (shared * cfg.bits // 8 + 2 * shared // 128)),
        ("shared experts, bf16", L * shared * 2),
        ("attention + norms + router, bf16 (derived)", L * ATTN_PARAMS_PER_LAYER * 2),
        ("attention + norms + router, int4 g128", L * (ATTN_PARAMS_PER_LAYER // 2 +
                                                       2 * ATTN_PARAMS_PER_LAYER // 128)),
        ("embedding + head, bf16", 2 * emb * 2),
        ("embedding + head, int4 g128", 2 * (emb // 2 + 2 * emb // 128)),
    ]
    for ctx in CONTEXTS:
        kv = L * ctx * KV_PER_TOKEN_PER_LAYER
        rows.append((f"K/V cache, {ctx // 1024}K context, bf16", kv * 2))
        rows.append((f"K/V cache, {ctx // 1024}K context, e4m3", kv))
    return rows


def footprint_table(cfg: moe.MoEConfig, model: moe.ModelConfig, total_bytes: int | None) -> None:
    print(f"DeepSeek V4.1 Flash, {model.layers} layers, routed experts at {cfg.bits} bit:")
    for name, b in footprint_rows(cfg, model):
        print(f"  {name:44} {b / 1e9:8.2f} GB")
    lean = (footprint_rows(cfg, model)[0][1] + footprint_rows(cfg, model)[1][1] +
            footprint_rows(cfg, model)[4][1] + footprint_rows(cfg, model)[5][1])
    print(f"  {'weights, lean (low-bit experts, int4 attention, bf16 embeddings)':44} "
          f"{lean / 1e9:8.2f} GB")
    for ctx in CONTEXTS:
        kv = model.layers * ctx * KV_PER_TOKEN_PER_LAYER
        print(f"  {'  + K/V at ' + str(ctx // 1024) + 'K bf16 / e4m3':44} "
              f"{(lean + 2 * kv) / 1e9:8.2f} GB / {(lean + kv) / 1e9:.2f} GB")
    if total_bytes is not None:
        print(f"  device memory: {total_bytes / 1e9:.1f} GB")


def step_bytes(blocks: list[moe.MoEBlock1Bit], xs: list[torch.Tensor]) -> int:
    """Expert bytes a step over `xs` (one activation per layer) reads: per layer the distinct
    experts routed to plus the shared expert."""
    total = 0
    for block, x in zip(blocks, xs, strict=True):
        ids, _ = block.routing(x)
        per_expert = block.nbytes() // block.cfg.n_routed
        total += int(ids.unique().numel()) * per_expert + block.shared_nbytes()
    return total


def time_ms(fn, warmup: int, iters: int, rounds: int = 5) -> float:
    """Median over `rounds` of the mean time of `iters` back-to-back calls of fn."""
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e3 < 300:
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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--bits", type=int, default=1, choices=(1, 2))
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument("--batches", default=",".join(map(str, BATCHES)))
    ap.add_argument("--iters", type=int, default=10, help="steps per timed loop")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--variant", type=int, default=-1, help="w1moe_decode variant, -1 the top")
    ap.add_argument("--reserve-gb", type=float, default=12.0,
                    help="memory left free for the rest of the model and the OS")
    ap.add_argument("--footprint", action="store_true", help="print the tables and stop")
    ap.add_argument("--check", action="store_true",
                    help="decode vs forward_reference on the first layer at T=4")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cfg = moe.MoEConfig(**{**moe.DEEPSEEK_V41_FLASH.__dict__, "bits": args.bits})
    model = moe.ModelConfig(**{**moe.DEEPSEEK_V41_FLASH_MODEL.__dict__, "moe": cfg,
                               "layers": args.layers})
    has_cuda = torch.cuda.is_available()
    free, total = torch.cuda.mem_get_info() if has_cuda else (None, None)
    footprint_table(cfg, model, total)
    if args.footprint or not has_cuda:
        if not has_cuda:
            print("no CUDA device: footprint only")
        return

    per_layer = cfg.bytes_per_layer("w1") + cfg.n_shared * cfg.expert_bytes("w1") + \
        2 * cfg.hidden * cfg.n_routed
    budget = free - int(args.reserve_gb * 1e9)
    resident = max(1, min(args.layers, budget // per_layer))
    if resident == args.layers:
        print(f"\n{free / 1e9:.1f} GB free: all {args.layers} layers resident "
              f"({args.layers * per_layer / 1e9:.1f} GB)")
    else:
        print(f"\n{free / 1e9:.1f} GB free holds {resident} of {args.layers} layers "
              f"({per_layer / 1e9:.2f} GB each, {args.reserve_gb:.0f} GB reserved): the step "
              f"re-uses {resident} resident layer{'s' if resident > 1 else ''} "
              f"{args.layers} times, so from the second pass on their weights may sit in L2 "
              f"only if the active set fits it, and the DRAM traffic is the same")

    g = torch.Generator(device="cuda").manual_seed(args.seed)
    t0 = time.perf_counter()
    blocks = [moe.MoEBlock1Bit.random_bits(cfg, "cuda", g) for _ in range(resident)]
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e9
    print(f"built {resident} random layer{'s' if resident > 1 else ''} at {cfg.bits} bit in "
          f"{time.perf_counter() - t0:.1f} s, {allocated:.1f} GB allocated")
    layers = [blocks[i % resident] for i in range(args.layers)]

    if args.check:
        x = torch.randn(4, cfg.hidden, device="cuda", generator=g).to(torch.bfloat16)
        y = blocks[0].decode(x, args.variant)
        y_ref = blocks[0].forward_reference(x)
        err = (y.float() - y_ref.float()).abs().max().item()
        scale = y_ref.float().abs().max().item()
        print(f"check T=4: max abs error {err:.3e} (max |ref| {scale:.3e})")

    print(f"\n  {'T':>4} {'step ms':>9} {'GB/step':>8} {'GB/s':>8} {'tok/s':>8} "
          f"{'floor ms @273':>14} {'floor ms @1532':>15}")
    for T in (int(b) for b in args.batches.split(",")):
        # One activation per layer: a step's routing differs layer to layer.
        xs = [torch.randn(T, cfg.hidden, device="cuda", generator=g).to(torch.bfloat16)
              for _ in range(args.layers)]

        def step(xs=xs):
            for block, x in zip(layers, xs, strict=True):
                block.decode(x, args.variant)

        ms = time_ms(step, args.warmup, args.iters)
        nbytes = step_bytes(layers, xs)
        print(f"  {T:4d} {ms:9.3f} {nbytes / 1e9:8.2f} {nbytes / ms / 1e6:8.0f} "
              f"{T * 1e3 / ms:8.1f} {nbytes / GB10_BYTES_PER_S * 1e3:14.2f} "
              f"{nbytes / RTX5090_BYTES_PER_S * 1e3:15.2f}")
    print("\nthe MoE experts of a step only: no attention, norms, embedding or head in these "
          "times; the floors are the touched bytes at the GB10's 273 GB/s and the 5090's "
          "1,532 GB/s copy roof")


if __name__ == "__main__":
    main()
