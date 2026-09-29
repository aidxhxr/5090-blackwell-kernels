#!/usr/bin/env python3
"""Time one Llama-3-8B decoder layer built from spark_kernels against the same layer in
PyTorch: eager, torch.compile (max-autotune-no-cudagraphs for prefill, reduce-overhead for
decode, which runs it under CUDA graphs), and for decode our forward replayed from a
torch.cuda.CUDAGraph. Writes results/layer.json (one JSON object per shape) and prints the
tables plus a per-stage breakdown of our decode step from the torch profiler.

Run on the GPU box after the extension is built:

    python scripts/bench_layer.py                 # everything
    python scripts/bench_layer.py --only decode --no-compile --iters 20

A prefill row is one layer over B sequences of S tokens from an empty cache. A decode row is
one layer over B tokens against a cache holding L - 1 tokens, appended in place (the cache's
capacity is L, so the attention kernel reads it where it is; `spark_copy_ms` is the same
step against a cache with spare capacity, where the filled part is copied out contiguous
first). With the int4 path on (the default; --no-int4 skips it), every decode row also
times the same step with the four projections as W4A16 GEMMs (SparkLayer(int4=True):
`spark_int4_ms` eager, `spark_int4_graph_ms` from a CUDA graph) and prints its per-stage
breakdown. One layer's int4 weights (107 MiB) would mostly stay in the 96 MB L2 from one
step to the next, which a 32-layer model never sees, so the int4 steps rotate over
INT4_COPIES separately quantized copies (428 MiB) and the graph holds one step per copy.
Every step is the steady-state form: the previous layer's MLP output comes in as
`delta` and both residual adds run inside the norm kernels. Timings are the median over
rounds of `iters` back-to-back steps between two CUDA events, so a host-bound eager step is
reported as what it costs a loop that queues layers as fast as it can.
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
OUT = ROOT / "results" / "layer.json"

# (B, S)
PREFILL_SHAPES = [(1, 4096), (1, 8192), (4, 2048)]
# (B, L): the cache holds L - 1 tokens and the step appends the L-th
DECODE_SHAPES = [(1, 4096), (1, 16384), (1, 131072), (8, 4096)]
N_LAYERS = 32  # Llama-3-8B, for the tokens/s a whole model implies
INT4_COPIES = 4  # int4 layers the int4 decode rotates over: 4 x 107 MiB, past the L2
PREFILL_COMPILE = "max-autotune-no-cudagraphs"
DECODE_COMPILE = "reduce-overhead"
STAGES = ["norm1", "qkv_gemm", "rope_append", "attention", "o_transpose", "o_gemm", "norm2",
          "gate_up_gemm", "swiglu", "down_gemm"]

RAMP_MS = 300
_ramped = False


def ramp_clocks(fn) -> None:
    """Spin `fn` for RAMP_MS once per process so the card is at its boost clock before the
    first timing, as the C++ benches and bench_torch.py do."""
    global _ramped
    if _ramped:
        return
    _ramped = True
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e3 < RAMP_MS:
        fn()
        torch.cuda.synchronize()


def time_ms(fn, warmup: int, iters: int, rounds: int = 5) -> float:
    """Median over `rounds` of the mean time of `iters` back-to-back calls of fn."""
    ramp_clocks(fn)
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


def shape_name(mode: str, b: int, n: int) -> str:
    return f"prefill_b{b}_s{n}" if mode == "prefill" else f"decode_b{b}_L{n}"


def tokens_per_s(tokens_per_step: int, ms_per_layer: float) -> float:
    """Tokens per second a model of N_LAYERS such layers would do, with nothing else in it
    (no embedding, no final norm and lm_head, no sampling)."""
    return tokens_per_step * 1e3 / (N_LAYERS * ms_per_layer) if ms_per_layer > 0 else 0.0


def fill_cache(cache, n: int, seed: int) -> None:
    """n random tokens in the cache, so the step reads real bytes."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    cache.k[:, :, :n].normal_(generator=g)
    cache.v[:, :, :n].normal_(generator=g)
    cache.length = n


def compile_or_none(fn, mode: str, enabled: bool):
    if not enabled:
        return None
    return torch.compile(fn, mode=mode, dynamic=False)


def mark_static(*tensors) -> None:
    """Tell torch.compile these tensors keep their addresses between calls. The weights and
    RoPE tables are plain tensors here, not parameters, and under reduce-overhead cudagraph
    trees would otherwise copy every input into its static buffers before each replay:
    570 MB per step, three times the step itself (measured)."""
    for t in tensors:
        torch._dynamo.mark_static_address(t)


def static_inputs(weights, rope, cache, *tensors):
    mark_static(weights.attn_norm, weights.w_qkv, weights.w_o, weights.mlp_norm,
                weights.w_gate_up, weights.w_down, rope.cos, rope.sin, cache.k, cache.v, *tensors)


# ---------------------------------------------------------------------------
# The breakdown: which kernel takes what share of our decode step
# ---------------------------------------------------------------------------
def stage_breakdown(step, iters: int = 3) -> dict:
    """Run `step` under the torch profiler with the layer's stage ranges on and charge every
    CUDA kernel of the last step to the stage whose range its launch fell in. Returns
    {stage: {"us": kernel time, "kernels": {name: us}}} plus "total_us" (the sum of kernel
    times, the GPU-busy time of one step, which is less than the step's wall time when the
    host cannot launch fast enough)."""
    import tempfile

    from torch.profiler import ProfilerActivity, profile

    from spark_kernels import layer as L

    L.PROFILE_STAGES = True
    try:
        step()
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
            for _ in range(iters):
                with torch.profiler.record_function("layer_step"):
                    step()
            torch.cuda.synchronize()
    finally:
        L.PROFILE_STAGES = False
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        p.export_chrome_trace(f.name)
        trace = json.load(open(f.name))["traceEvents"]
    steps = sorted((e for e in trace if e.get("cat") == "user_annotation"
                    and e.get("name") == "layer_step"), key=lambda e: e["ts"])
    if not steps:
        return {}
    last = steps[-1]
    t0, t1 = last["ts"], last["ts"] + last["dur"]
    ranges = [e for e in trace if e.get("cat") == "user_annotation" and e["name"] in STAGES
              and t0 <= e["ts"] <= t1]
    launches = {e["args"]["correlation"]: e for e in trace
                if e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {})
                and t0 <= e["ts"] <= t1}
    out: dict = {s: {"us": 0.0, "kernels": {}} for s in STAGES}
    out["other"] = {"us": 0.0, "kernels": {}}
    total = 0.0
    for e in trace:
        if e.get("cat") != "kernel":
            continue
        corr = e.get("args", {}).get("correlation")
        launch = launches.get(corr)
        if launch is None:
            continue
        stage = "other"
        for r in ranges:
            if r["ts"] <= launch["ts"] <= r["ts"] + r["dur"]:
                stage = r["name"]
                break
        name = kernel_label(e["name"])
        out[stage]["us"] += e["dur"]
        out[stage]["kernels"][name] = out[stage]["kernels"].get(name, 0.0) + e["dur"]
        total += e["dur"]
    out["total_us"] = total
    return out


def kernel_label(name: str) -> str:
    """A kernel's name without its template arguments and namespaces:
    "void spark::hgemm_decode::(anonymous namespace)::decode_kernel<16, 64, ...>(...)" is
    "decode_kernel"; torch's "void at::native::vectorized_elementwise_kernel<4, ...>" is
    "vectorized_elementwise_kernel"."""
    name = name.replace("(anonymous namespace)::", "")
    if name.startswith("void "):
        name = name[5:]
    for cut in ("<", "("):
        if cut in name:
            name = name[:name.index(cut)]
    return name.rsplit("::", 1)[-1]


def print_breakdown(shape: str, bd: dict, wall_ms: float) -> None:
    if not bd:
        return
    total = bd["total_us"]
    print(f"\n{shape}: kernel time per stage of our step "
          f"(GPU busy {total / 1e3:.3f} ms of {wall_ms:.3f} ms wall)", file=sys.stderr)
    for stage in STAGES + ["other"]:
        us = bd[stage]["us"]
        if us == 0:
            continue
        kernels = ", ".join(f"{k} {v:.1f}" for k, v in
                            sorted(bd[stage]["kernels"].items(), key=lambda kv: -kv[1]))
        print(f"  {stage:14s} {us:9.1f} us  {100 * us / total:5.1f}%   {kernels}",
              file=sys.stderr)


# ---------------------------------------------------------------------------
# Prefill
# ---------------------------------------------------------------------------
def bench_prefill(sk, L, weights, rope, b: int, s: int, args) -> dict:
    torch.manual_seed(1)
    x0 = torch.randn(b, s, L.HIDDEN, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn_like(x0) * 0.5
    x = x0.clone()
    ours = L.SparkLayer(weights, rope)
    theirs = L.TorchLayer(weights, rope)
    cache = L.KVCache(b, s)

    def ours_step():
        cache.length = 0
        x.copy_(x0)
        return ours.prefill(x, cache, delta)

    def torch_step():
        cache.length = 0
        return theirs.prefill(x0, cache, delta)

    ours_ms = time_ms(ours_step, args.warmup, args.iters)
    breakdown = stage_breakdown(ours_step)
    torch_ms = time_ms(torch_step, args.warmup, args.iters)
    compiled = compile_or_none(theirs.prefill, PREFILL_COMPILE, args.compile)
    compiled_ms = 0.0
    if compiled is not None:
        static_inputs(weights, rope, cache, x0, delta)

        def compiled_step():
            cache.length = 0
            return compiled(x0, cache, delta)

        t0 = time.perf_counter()
        compiled_step()
        torch.cuda.synchronize()
        print(f"  compiled prefill b{b} s{s} in {time.perf_counter() - t0:.0f} s", file=sys.stderr)
        compiled_ms = time_ms(compiled_step, args.warmup, args.iters)
    row = {"kernel": "layer", "dtype": "bf16", "mode": "prefill", "B": b, "S": s,
           "shape": shape_name("prefill", b, s), "spark_ms": ours_ms, "torch_ms": torch_ms,
           "torch_compiled_ms": compiled_ms, "torch_compile_mode": PREFILL_COMPILE,
           "tokens_per_s_32_layers": tokens_per_s(b * s, ours_ms),
           "gpu_busy_ms": breakdown.get("total_us", 0.0) / 1e3,
           "breakdown_us": {k: breakdown[k]["us"] for k in STAGES + ["other"] if k in breakdown}}
    print(f"prefill b{b} s{s:6d}: ours {ours_ms:8.3f} ms  torch {torch_ms:8.3f} ms  "
          f"compiled {compiled_ms:8.3f} ms  ({tokens_per_s(b * s, ours_ms):,.0f} tok/s over "
          f"{N_LAYERS} layers)", file=sys.stderr)
    print_breakdown(row["shape"], breakdown, ours_ms)
    return row


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------
def capture_decode(layer, x, x0, cache, delta, length: int):
    """Warm the kernels' workspaces up on a side stream, then capture one step into a
    CUDA graph (one step per layer when `layer` is a list, run one after the other). The
    capture runs in torch's default "global" error mode, so a cudaMalloc or any other
    capture-unsafe call inside a launch path would raise here instead of being silently
    recorded."""
    layers = layer if isinstance(layer, list) else [layer]

    def step():
        for lyr in layers:
            cache.length = length
            x.copy_(x0)
            out = lyr.decode(x, cache, delta)
        return out

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = step()
    return g, out


def bench_decode(sk, L, weights, rope, b: int, n: int, args, ours4=()) -> dict:
    torch.manual_seed(2)
    x0 = torch.randn(b, 1, L.HIDDEN, device="cuda", dtype=torch.bfloat16)
    delta = torch.randn_like(x0) * 0.5
    x = x0.clone()
    ours = L.SparkLayer(weights, rope)
    theirs = L.TorchLayer(weights, rope)
    cache = L.KVCache(b, n)  # capacity n: the step fills the last slot, read in place
    fill_cache(cache, n - 1, seed=n)

    def ours_step():
        cache.length = n - 1
        x.copy_(x0)
        return ours.decode(x, cache, delta)

    def torch_step():
        cache.length = n - 1
        return theirs.decode(x0, cache, delta)

    ours_ms = time_ms(ours_step, args.warmup, args.iters)
    ref_x, ref_d = ours_step()
    ref_x, ref_d = ref_x.clone(), ref_d.clone()
    breakdown = stage_breakdown(ours_step)

    # our forward as a CUDA graph: replays must reproduce the eager step bit for bit
    graph, (gx, gd) = capture_decode(ours, x, x0, cache, delta, n - 1)
    graph_ok = True
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        graph_ok &= torch.equal(gx, ref_x) and torch.equal(gd, ref_d)
    graph_ms = time_ms(graph.replay, args.warmup, args.iters)

    # the int4 layer: the same step with the projections as W4A16 GEMMs, rotating over
    # INT4_COPIES copies of the weights so that they come from DRAM as in a full model
    int4_ms = int4_graph_ms = 0.0
    int4_breakdown: dict = {}
    if ours4:
        turn = [0]

        def int4_step():
            cache.length = n - 1
            x.copy_(x0)
            lyr = ours4[turn[0] % len(ours4)]
            turn[0] += 1
            return lyr.decode(x, cache, delta)

        int4_ms = time_ms(int4_step, args.warmup, args.iters)
        int4_breakdown = stage_breakdown(int4_step)
        graph4, _ = capture_decode(ours4, x, x0, cache, delta, n - 1)
        int4_graph_ms = time_ms(graph4.replay, args.warmup, args.iters) / len(ours4)
        del graph4

    torch_ms = time_ms(torch_step, args.warmup, args.iters)
    compiled_ms = 0.0
    compiled = compile_or_none(theirs.decode, DECODE_COMPILE, args.compile)
    if compiled is not None:
        static_inputs(weights, rope, cache, x0, delta)

        def compiled_step():
            # tells cudagraph trees the previous step's outputs are dead, so it replays
            # instead of guarding them (torch.compile docs, "CUDA graph trees")
            torch.compiler.cudagraph_mark_step_begin()
            cache.length = n - 1
            return compiled(x0, cache, delta)

        t0 = time.perf_counter()
        for _ in range(3):  # reduce-overhead records its graph on the warmup calls
            compiled_step()
        torch.cuda.synchronize()
        print(f"  compiled decode b{b} L{n} in {time.perf_counter() - t0:.0f} s", file=sys.stderr)
        compiled_ms = time_ms(compiled_step, args.warmup, args.iters)

    # the same step against a cache with spare capacity: the filled part is copied out
    # contiguous before the attention kernel reads it (KVCache.kv), an O(L) cost per step
    copy_ms = 0.0
    if args.copy:
        del graph
        big = L.KVCache(b, n + 256)
        fill_cache(big, n - 1, seed=n)

        def copy_step():
            big.length = n - 1
            x.copy_(x0)
            return ours.decode(x, big, delta)

        copy_ms = time_ms(copy_step, args.warmup, args.iters)
        del big

    row = {"kernel": "layer", "dtype": "bf16", "mode": "decode", "B": b, "L": n,
           "shape": shape_name("decode", b, n), "spark_ms": ours_ms, "spark_graph_ms": graph_ms,
           "spark_graph_ok": graph_ok, "spark_copy_ms": copy_ms, "torch_ms": torch_ms,
           "torch_compiled_ms": compiled_ms, "torch_compile_mode": DECODE_COMPILE,
           "tokens_per_s_32_layers": tokens_per_s(b, ours_ms),
           "tokens_per_s_32_layers_graph": tokens_per_s(b, graph_ms),
           "spark_int4_ms": int4_ms, "spark_int4_graph_ms": int4_graph_ms,
           "gpu_busy_ms": breakdown.get("total_us", 0.0) / 1e3,
           "breakdown_us": {s: breakdown[s]["us"] for s in STAGES + ["other"] if s in breakdown}}
    print(f"decode  b{b} L{n:6d}: ours {ours_ms:8.3f} ms  graph {graph_ms:8.3f} ms"
          f"{'' if graph_ok else ' (REPLAY MISMATCH)'}  torch {torch_ms:8.3f} ms  compiled "
          f"{compiled_ms:8.3f} ms  copy {copy_ms:8.3f} ms  ({tokens_per_s(b, graph_ms):,.0f} "
          f"tok/s over {N_LAYERS} layers from the graph)", file=sys.stderr)
    print_breakdown(row["shape"], breakdown, ours_ms)
    if ours4:
        print(f"decode  b{b} L{n:6d} int4: ours {int4_ms:8.3f} ms  graph {int4_graph_ms:8.3f} ms"
              f"  ({graph_ms / int4_graph_ms:.2f}x the bf16 graph, "
              f"{tokens_per_s(b, int4_graph_ms):,.0f} tok/s over {N_LAYERS} layers)",
              file=sys.stderr)
        print_breakdown(row["shape"] + " int4", int4_breakdown, int4_ms)
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["prefill", "decode"])
    ap.add_argument("--iters", type=int, default=20, help="steps per timed round (5 rounds)")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--no-compile", dest="compile", action="store_false",
                    help="skip the torch.compile columns (compiling takes minutes)")
    ap.add_argument("--no-copy", dest="copy", action="store_false",
                    help="skip the spare-capacity (copied cache) decode timing")
    ap.add_argument("--no-int4", dest="int4", action="store_false",
                    help="skip the int4 (W4A16 projections) decode timings")
    ap.add_argument("--shapes",
                    help="comma-separated subset, e.g. prefill_b1_s4096,decode_b1_L4096")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1
    import spark_kernels as sk
    from spark_kernels import layer as L

    device = torch.cuda.get_device_name()
    print(f"device: {device} | torch {torch.__version__}", file=sys.stderr)
    weights = L.LayerWeights.random(seed=0)
    print(f"one layer's weights: {weights.bytes() / 2**20:.0f} MiB", file=sys.stderr)
    max_pos = max(max(s for _, s in PREFILL_SHAPES), max(n for _, n in DECODE_SHAPES)) + 1
    rope = L.RoPE(max_pos)
    ours4 = ([L.SparkLayer(weights, rope, int4=True) for _ in range(INT4_COPIES)]
             if args.int4 else [])
    if ours4:
        print(f"int4 projections: {ours4[0].weight_bytes() / 2**20:.0f} MiB per layer, "
              f"{INT4_COPIES} copies", file=sys.stderr)
    wanted = set(args.shapes.split(",")) if args.shapes else None
    rows = []
    if args.only != "decode":
        for b, s in PREFILL_SHAPES:
            if wanted and shape_name("prefill", b, s) not in wanted:
                continue
            rows.append({"device": device, **bench_prefill(sk, L, weights, rope, b, s, args)})
            torch.cuda.empty_cache()
    if args.only != "prefill":
        for b, n in DECODE_SHAPES:
            if wanted and shape_name("decode", b, n) not in wanted:
                continue
            rows.append({"device": device,
                         **bench_decode(sk, L, weights, rope, b, n, args, ours4)})
            torch.cuda.empty_cache()
    if wanted or args.only or not args.compile:
        print("partial run: results/layer.json not written", file=sys.stderr)
        print("\n".join(json.dumps(r) for r in rows))
        return 0
    args.out.parent.mkdir(exist_ok=True)
    with args.out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
