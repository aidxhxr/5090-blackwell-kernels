#!/usr/bin/env python3
"""The serving engine on MoE models: decode ms/step with the decode steps captured in CUDA
graphs and without, at batch 1, 8, 32 and 64, next to the dense model of the same layers.
Appends JSON lines to results/engine_moe.json and prints a table.

    python scripts/bench_engine_moe.py                     # mixtral, qwen-like and dense
    python scripts/bench_engine_moe.py --models qwen-like --batches 1,8

Models, each --layers layers of the engine's attention (Llama-3-8B's, which is also
Mixtral's) and a --vocab-token embedding and lm_head, random bf16 weights:

    mixtral     Mixtral 8x7B's MoE layer: 8 experts of 4096 -> 14336, top-2, so a token runs
                twice the dense MLP's columns
    qwen-like   64 experts of 4096 -> 1792, top-8 (Qwen3-235B-A22B's routing at half its
                experts): the 8 experts a token runs are 14336 columns, the dense MLP's, so
                the active parameters are the dense model's
    dense       Llama-3-8B's layer, the 14336-wide MLP

A static row is scripts/bench_serve.py's: B prompts log-uniform in --min-prompt..--max-prompt
(seeded, the same for every model), --new tokens each, greedy; decode ms/step is the wall
time of the --new - 1 steps after the prefill over the steps, with a device sync at both
ends. Each batch runs --repeats times and the row keeps the median. One SparkModel per model;
the engine is built around it once with graphs (every bucket up to GRAPH_MAX captured) and
once without, each warmed on every bucket first. Nothing here times the prefill, and its
GEMMs run hgemm variant 4 (SparkModel's prefill_variant) to keep the dense model off the TMA
kernels' memory (docs/design/llm_models.md); the grouped GEMM is a TMA kernel at every size,
so the MoE models pay that once anyway.

A step row is one bucket's decode step on its own, every slot empty (no K/V to read; the
weights stream as in a real step, the padding rows being routed like tokens): the GPU time of
the captured step replayed back to back between CUDA events, and the host time to issue the
same step eagerly (model.decode with no sync, medians of --iters). Where the host takes longer
than the GPU, an eager step waits on the host and the graph gives that time back.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "results" / "engine_moe.json"
MODELS = ("mixtral", "qwen-like", "dense")


def moe_config(name: str):
    from spark_kernels.moe_layer import MIXTRAL_8X7B, MoELayerConfig

    if name == "mixtral":
        return MIXTRAL_8X7B
    if name == "qwen-like":
        return MoELayerConfig("qwen-like", hidden=4096, inter=1792, experts=64, top_k=8)
    return None


def step_bytes_one_token(model, cfg) -> int:
    """Weight bytes a one-token step reads: the attention projections, the MLP a token runs
    (its top_k experts and the router, or the dense MLP), the lm_head."""
    from spark_kernels import engine as E

    w = model.w
    attn = sum((lw.w_qkv.numel() + lw.w_o.numel()) * 2 for lw in w.layers)
    if cfg is None:
        mlp = sum((lw.w_gate_up.numel() + lw.w_down.numel()) * 2 for lw in w.layers)
    else:
        mlp = len(w.layers) * (cfg.top_k * cfg.expert_bytes() + cfg.experts * E.HIDDEN * 2)
    return attn + mlp + w.lm_head.numel() * 2


def step_times(eng, buckets, iters: int) -> dict[int, tuple[float, float]]:
    """bucket -> (GPU ms of the captured step replayed back to back, host ms to issue it
    eagerly), every slot empty."""
    eng.positions.zero_()  # the state the graphs were captured in
    eng.slots.fill_(eng.cache.scratch * eng.page)
    eng.seq_lens.zero_()
    out = {}
    for bp in buckets:
        g = eng.graphs[bp]
        for _ in range(3):
            g.replay()
        gpu = []
        for _ in range(iters):
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(5):
                g.replay()
            b.record()
            b.synchronize()
            gpu.append(a.elapsed_time(b) / 5)
        host = []
        batch = eng._decode_batch(bp)
        for _ in range(iters):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            eng.model.decode(batch, eng.cache)
            host.append((time.perf_counter() - t0) * 1e3)
        torch.cuda.synchronize()
        out[bp] = (statistics.median(gpu), statistics.median(host))
    eng.ids.zero_()  # as _capture_all leaves them
    eng.gen_n.zero_()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--batches", default="1,8,32,64")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=32000, help="Mixtral's; a multiple of 64")
    ap.add_argument("--new", type=int, default=64, help="tokens generated per request")
    ap.add_argument("--min-prompt", type=int, default=64)
    ap.add_argument("--max-prompt", type=int, default=2048)
    ap.add_argument("--cache-gb", type=float, default=0.5, help="K/V cache size, all layers")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--iters", type=int, default=30, help="samples per step row")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from bench_serve import log_uniform, prompts_for, static_batch, warm_up

    from spark_kernels import engine as E
    from spark_kernels.layer import RoPE

    dev = torch.cuda.get_device_name()
    batches = [int(x) for x in args.batches.split(",") if x]
    max_seq = args.max_prompt + max(args.new, 256) + 16
    rope = RoPE(max_seq)
    cache_bytes = int(args.cache_gb * (1 << 30))
    buckets = [x for x in E.BUCKETS if x <= max(batches)]
    rows = []
    for name in [m for m in args.models.split(",") if m]:
        cfg = moe_config(name)
        weights = E.ModelWeights.random(args.layers, args.vocab, moe=cfg)
        model = E.SparkModel(weights, rope, prefill_variant=4)
        base = {"device": dev, "kernel": "engine_moe", "model": name, "layers": args.layers,
                "vocab": args.vocab, "weight_bytes": model.weight_bytes(),
                "step_bytes_1tok": step_bytes_one_token(model, cfg)}
        if cfg is not None:
            base.update(experts=cfg.experts, top_k=cfg.top_k, inter=cfg.inter)
        print(f"{name}: {args.layers} layers, {base['weight_bytes'] / 1e9:.2f} GB of weights, "
              f"{base['step_bytes_1tok'] / 1e9:.3f} GB read by a one-token step",
              file=sys.stderr)
        for graphs in (True, False):
            eng = E.Engine(model, args.layers, max_batch=max(batches), max_seq=max_seq,
                           cache_bytes=cache_bytes, graphs=graphs)
            warm_up(eng, buckets, args.vocab)
            for b in batches:
                gen = torch.Generator().manual_seed(args.seed + b)
                lens = log_uniform(b, args.min_prompt, args.max_prompt, gen)
                ms = []
                for _ in range(args.repeats):
                    r = static_batch(eng, prompts_for(lens, args.vocab, gen), args.new)
                    ms.append(r["decode_s"] * 1e3 / max(1, r["steps"]))
                row = dict(base, kind="static", graphs=graphs, batch=b, prompt_tokens=sum(lens),
                           new=args.new, repeats=args.repeats,
                           decode_ms_per_step=statistics.median(ms), decode_ms_all=ms,
                           decode_tok_s=b * 1e3 / statistics.median(ms))
                rows.append(row)
                print(f"{name:9s} {'graph' if graphs else 'eager'} B={b:3d}: decode "
                      f"{row['decode_ms_per_step']:7.3f} ms/step "
                      f"({', '.join(f'{x:.3f}' for x in ms)})", file=sys.stderr)
            if graphs:
                for bp, (gpu, host) in step_times(eng, buckets, args.iters).items():
                    rows.append(dict(base, kind="step", bucket=bp, gpu_ms=gpu, host_ms=host))
                    print(f"{name:9s} step bucket {bp:3d}: GPU {gpu:7.3f} ms (graph replay), "
                          f"host {host:7.3f} ms to issue eagerly", file=sys.stderr)
            del eng
            torch.cuda.empty_cache()
        del model, weights
        torch.cuda.empty_cache()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"appended {len(rows)} rows to {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
