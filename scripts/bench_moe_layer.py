#!/usr/bin/env python3
"""Time a whole MoE layer, bf16: ours (spark_kernels.moe_layer.MoELayer) against the torch
layer transformers runs (the experts loop), the same routing with torch._grouped_mm for the
two projections, and vLLM's fused_moe, on Mixtral 8x7B (8 experts, top-2, 4096 -> 14336) and
Qwen3-30B-A3B (128, top-8, 2048 -> 768). Writes results/moe_layer.json (JSON lines, one per
config, token count and implementation; a rerun replaces the rows it measures) and prints a
table plus, for ours, the kernel time per stage from the torch profiler.

    python scripts/bench_moe_layer.py         # ours, ours_graph, ours_gather, ours_permute,
                                              # ours_unfused, torch, torch_grouped
    ~/vllm-env/bin/python scripts/bench_moe_layer.py --impls vllm
    python scripts/bench_moe_layer.py --config qwen3 --tokens 1 16 --iters 100

vLLM runs from its own environment (its torch is not the one the extension is built for), so
that invocation loads moe_layer.py from its file and never imports spark_kernels.

Every implementation is checked first against torch_moe with fp32 accumulation on the same
input (`rel_err`, max |diff| over max |ref|); the routing picks are compared on the same
logits for ours (`route_mismatch`: tokens whose picked logits differ from torch.topk's). A row
is the median of `iters` calls, each between two CUDA events with no sync between calls, after
a 300 ms clock ramp. The input cycles through a pool of different token batches (8 below 256
tokens, 2 above), so consecutive calls route to different experts and the weights come from
DRAM: one Qwen3 expert is 9.4 MB and 8 of them would otherwise sit in the 96 MB L2. Every
implementation includes the router GEMM (F.linear on the same router weight).

impls:
  ours           router GEMM, the routing, grouped_gemm_swiglu, grouped_gemm, moe_combine;
                 the gate/up GEMM gathers x through the routing's row table (moe_route_rows)
                 where moe_layer's rule says so, else moe_dispatch writes the permuted copy
  ours_graph     ours captured in a CUDA graph for the token count (MoEGraph) and replayed,
                 up to 1024 tokens; the copy of the input into the graph's static buffer is
                 inside the timing
  ours_gather    ours with the gather forced at every token count
  ours_permute   ours with the permuted copy forced at every token count
  ours_unfused   the same with the plain grouped GEMM on gate|up and the swiglu kernel after it
  torch          moe_layer.torch_moe: transformers' loop, a torch.where per expert hit
  torch_grouped  moe_layer.torch_grouped_moe: argsort, torch._grouped_mm twice, a gather and
                 a weighted sum
  vllm           F.linear, vllm fused_topk (renormalize), vllm fused_experts (Triton)
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "moe_layer.json"
TOKENS = [1, 16, 128, 1024, 4096, 16384]
DEFAULT_IMPLS = ["ours", "ours_graph", "ours_gather", "ours_permute", "ours_unfused", "torch",
                 "torch_grouped"]
OURS = ("ours", "ours_gather", "ours_permute", "ours_unfused")
STAGES = ["router", "route", "permute", "gate_up", "down", "combine"]
MEM_GB = 10.0  # device memory this process may take
GRAPH_MAX_TOKENS = 1024  # ours_graph rows stop here: a graph is for decode-sized steps


def load_moe_layer(use_package: bool):
    if use_package:
        sys.path.insert(0, str(ROOT / "python"))
        from spark_kernels import moe_layer
        return moe_layer
    path = ROOT / "python" / "spark_kernels" / "moe_layer.py"
    spec = importlib.util.spec_from_file_location("moe_layer_file", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["moe_layer_file"] = mod
    spec.loader.exec_module(mod)
    return mod


def time_calls(fn, pool, iters: int, warmup: int = 3) -> tuple[float, float]:
    """(median, min) ms of `iters` calls of fn over the input pool, events per call."""
    for i in range(warmup):
        fn(pool[i % len(pool)])
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
          for _ in range(iters)]
    for i, (s, e) in enumerate(ev):
        x = pool[i % len(pool)]
        s.record()
        fn(x)
        e.record()
    torch.cuda.synchronize()
    ts = [s.elapsed_time(e) for s, e in ev]
    return statistics.median(ts), min(ts)


def ramp(fn, x, ms: float = 300.0) -> None:
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e3 < ms:
        fn(x)
        torch.cuda.synchronize()


def stage_of(name: str) -> str:
    """Our kernels by name; anything else in our layer is the router GEMM."""
    if "route_kernel" in name:
        return "route"
    if "permute_kernel" in name or "positions_kernel" in name:
        return "permute"
    if "combine_kernel" in name:
        return "combine"
    if "grouped_kernel<" in name:  # <FP8, BM, BKB, STAGES, SWIGLU, GATHER>
        args = name.split("grouped_kernel<", 1)[1].split(">", 1)[0].replace(" ", "").split(",")
        return "gate_up" if args[4] == "true" else "down"
    return "router"


def breakdown(fn, pool, reps: int = 10) -> dict[str, float]:
    """Kernel microseconds per stage of one call of our layer, from the torch profiler."""
    from torch.profiler import ProfilerActivity, profile

    fn(pool[0])
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for i in range(reps):
            fn(pool[i % len(pool)])
        torch.cuda.synchronize()
    out = dict.fromkeys(STAGES, 0.0)
    for evt in prof.events():
        if evt.device_type != torch.autograd.DeviceType.CUDA:
            continue
        out[stage_of(evt.name)] += evt.device_time / reps
    out["total"] = sum(out[s] for s in STAGES)
    return out


def vllm_fn(cfg, wts):
    from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import fused_topk

    def run(x):
        logits = F.linear(x, wts.router)
        tw, ti, _ = fused_topk(x, logits, cfg.top_k, renormalize=cfg.renormalize)
        return fused_experts(x, wts.w13, wts.w2, tw, ti)

    return run


def impl_fns(ml, cfg, wts, impls):
    fns = {}
    ours = None  # the layer the ours_* variants share their interleaved w13 with (1.75 GB
    for name in impls:  # on Mixtral; one copy fits the memory cap, four did not)
        if name in ("ours", "ours_graph", "ours_gather", "ours_permute") and ours is None:
            ours = ml.MoELayer(wts)
        if name == "ours":
            fns[name] = ours
        elif name == "ours_graph":
            fns[name] = None  # captured per token count in run_config, from `ours`
        elif name in ("ours_gather", "ours_permute"):
            fns[name] = copy.copy(ours)
            fns[name].gather = name == "ours_gather"
        elif name == "ours_unfused":
            fns[name] = ml.MoELayer(wts, fuse_swiglu=False)
        elif name == "torch":
            fns[name] = lambda x: ml.torch_moe(x, wts)
        elif name == "torch_grouped":
            fns[name] = lambda x: ml.torch_grouped_moe(x, wts)
        elif name == "vllm":
            fns[name] = vllm_fn(cfg, wts)
        else:
            raise SystemExit(f"unknown impl {name}")
    return fns


def route_mismatch(ml, layer, wts, x) -> int:
    ids = layer.routing(x)[0].long()
    logits = F.linear(x, wts.router).float()
    _, want = ml._route_torch(logits, wts.cfg)
    got_v = logits.gather(1, ids).sort(-1).values
    want_v = logits.gather(1, want).sort(-1).values
    return int((got_v != want_v).any(-1).sum().item())


def run_config(ml, cfg, args) -> list[dict]:
    rows = []
    wts = ml.MoEWeights.random(cfg, seed=0)
    fns = impl_fns(ml, cfg, wts, args.impls)
    dev = torch.cuda.get_device_name()
    for T in args.tokens:
        g = torch.Generator(device="cuda").manual_seed(1000 + T)
        n_pool = 8 if T < 256 else 2
        pool = [torch.randn(T, cfg.hidden, device="cuda", generator=g).to(torch.bfloat16)
                for _ in range(n_pool)]
        ref = ml.torch_moe(pool[0], wts, acc_dtype=torch.float32).float()
        scale = ref.abs().max().item()
        # What the call has to move and compute: the experts the first input touches (their
        # bf16 weights read once is the floor at small T) and the FLOPs of its T * k rows.
        _, ids0 = ml._route_torch(F.linear(pool[0], wts.router), cfg)
        touched = int(torch.unique(ids0).numel())
        moved = touched * cfg.expert_bytes()
        gflop = 2.0 * T * cfg.top_k * 3 * cfg.hidden * cfg.inter / 1e9
        for name, fn in fns.items():
            torch.cuda.empty_cache()  # one implementation's cached blocks at a time
            torch.cuda.reset_peak_memory_stats()
            if name == "ours_graph":
                if T > GRAPH_MAX_TOKENS:
                    continue
                fn = ml.MoEGraph(fns.get("ours") or copy.copy(fns["ours_gather"]), T)
            out = fn(pool[0])
            err = (out.float() - ref).abs().max().item() / scale
            if err > 2e-2:
                raise SystemExit(f"{cfg.name} T={T} {name}: rel err {err:.3g}")
            ramp(fn, pool[0])
            med, mn = time_calls(fn, pool, args.iters)
            row = {"device": dev, "kernel": "moe_layer", "config": cfg.name, "tokens": T,
                   "experts": cfg.experts, "top_k": cfg.top_k, "hidden": cfg.hidden,
                   "inter": cfg.inter, "impl": name, "median_ms": med, "min_ms": mn,
                   "rel_err": err, "iters": args.iters, "experts_touched": touched,
                   "weight_bytes": moved, "gflop": gflop,
                   "weight_gbps": moved / (med * 1e-3) / 1e9,
                   "tflops": gflop / (med * 1e-3) / 1e3}
            if name in OURS:
                row["route_mismatch"] = route_mismatch(ml, fn, wts, pool[0])
                row["gather"] = fn.gathers(T)
            if name in ("ours", "ours_gather", "ours_permute") and not args.no_profile:
                row["breakdown_us"] = breakdown(fn, pool)
            if name == "vllm":
                import vllm
                row["vllm"] = vllm.__version__
            row["torch"] = torch.__version__
            row["peak_gb"] = torch.cuda.max_memory_reserved() / 1e9
            rows.append(row)
            print(f"{cfg.name:<8} T={T:<6} {name:<14} {med:9.4f} ms  (min {mn:.4f})  "
                  f"rel_err {err:.2e}  {row['weight_gbps']:6.0f} GB/s  "
                  f"{row['tflops']:6.1f} TFLOPS", flush=True)
            if "breakdown_us" in row:
                b = row["breakdown_us"]
                print("    " + "  ".join(f"{s} {b[s]:.1f}" for s in STAGES)
                      + f"  total {b['total']:.1f} us", flush=True)
        del pool, ref
        torch.cuda.empty_cache()
    return rows


def merge(rows: list[dict]) -> None:
    key = lambda r: (r["config"], r["tokens"], r["impl"])  # noqa: E731
    new = {key(r) for r in rows}
    old = []
    if OUT.exists():
        old = [json.loads(line) for line in OUT.read_text().splitlines() if line.strip()]
    keep = [r for r in old if key(r) not in new]
    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for r in sorted(keep + rows, key=lambda r: (r["config"], r["tokens"], r["impl"])):
            f.write(json.dumps(r) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=["mixtral", "qwen3"])
    ap.add_argument("--tokens", nargs="*", type=int, default=TOKENS)
    ap.add_argument("--impls", nargs="*", default=DEFAULT_IMPLS)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    # The card is shared: hold this process to MEM_GB, so a shape that would need more fails
    # instead of crowding the other tenants.
    torch.cuda.set_per_process_memory_fraction(
        min(1.0, MEM_GB * 1e9 / torch.cuda.get_device_properties(0).total_memory))
    needs_package = any(i.startswith("ours") for i in args.impls)
    if needs_package and "vllm" in args.impls:
        raise SystemExit("run vllm on its own: its environment cannot load the extension")
    ml = load_moe_layer(needs_package)
    rows = []
    for name in args.config:
        rows += run_config(ml, ml.CONFIGS[name], args)
        torch.cuda.empty_cache()
    if not args.no_write:
        merge(rows)


if __name__ == "__main__":
    main()
