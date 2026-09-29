#!/usr/bin/env python3
"""Turn results/*.json (from run_all_benches.sh) and results/torch_comparison.json (from
bench_torch.py) into docs/RESULTS.md plus a short results/headline.md for the README."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shape_utils import (  # noqa: E402
    LAYER_FILE,
    TORCH_COMPARISON,
    arithmetic_intensity,
    compute_peak,
    device_key,
    is_compute_bound_kernel,
    is_reference,
    load_bench_rows,
    load_jsonl,
    measured_peaks,
    parse_shape,
    peaks_for_rows,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT_MD = ROOT / "docs" / "RESULTS.md"
OUT_HEADLINE = RESULTS / "headline.md"

SHEETS = {"RTX 5090": "RTX5090.md", "GB10": "GB10.md"}
KERNEL_ORDER = ["bandwidth", "rmsnorm", "add_rmsnorm", "swiglu", "softmax", "sgemm", "hgemm",
                "fp8gemm", "attention", "rope"]
# The GEMM benches time a library reference next to every variant (ref_ms).
REFERENCE_NAME = {"sgemm": "cuBLAS", "hgemm": "cuBLAS", "fp8gemm": "cuBLASLt",
                  "attention_varlen": "dense attention per sequence"}


def on_device(r: dict, device: str) -> bool:
    try:
        return device_key(r.get("device")) == device
    except ValueError:  # a device without peaks is certainly not this one
        return False


def load_torch_rows(device: str) -> dict[tuple, dict]:
    """Torch comparison rows keyed for the join, only those measured on `device`: a
    torch_comparison.json left over from the other machine must not end up next to these rows."""
    p = RESULTS / TORCH_COMPARISON
    if not p.exists():
        return {}
    rows = load_jsonl(p)
    kept = [r for r in rows if on_device(r, device)]
    if len(kept) < len(rows):
        print(f"ignoring {len(rows) - len(kept)} rows of {p.name} that were not measured on the "
              f"{device}; rerun scripts/bench_torch.py on this machine", file=sys.stderr)
    return {(r["kernel"], r["dtype"], r["shape"]): r for r in kept}


def load_layer_rows(device: str) -> list[dict]:
    """Rows of results/layer.json (scripts/bench_layer.py) measured on `device`."""
    p = RESULTS / LAYER_FILE
    if not p.exists():
        return []
    rows = load_jsonl(p)
    kept = [r for r in rows if on_device(r, device)]
    if len(kept) < len(rows):
        print(f"ignoring {len(rows) - len(kept)} rows of {p.name} that were not measured on the "
              f"{device}; rerun scripts/bench_layer.py on this machine", file=sys.stderr)
    return kept


def speedup(ref_ms: float, ms: float) -> str:
    return f"{ref_ms / ms:.2f}×" if ref_ms > 0 and ms > 0 else "—"


LAYER_STAGES = [("norm1", "norm 1"), ("qkv_gemm", "qkv GEMM"),
                ("rope_append", "RoPE + cache append"), ("attention", "attention"),
                ("o_transpose", "o transpose"),
                ("o_gemm", "o GEMM"), ("norm2", "norm 2"), ("gate_up_gemm", "gate/up GEMM"),
                ("swiglu", "swiglu"), ("down_gemm", "down GEMM"), ("other", "other")]


def layer_section(rows: list[dict]) -> list[str]:
    """The "layer" tables: prefill and decode of one Llama-3-8B decoder layer, ours against
    PyTorch eager and compiled (and our forward as a CUDA graph for decode), the tokens/s a
    32-layer model would get from each decode row, and the per-stage kernel breakdown of our
    decode step where the bench recorded one."""
    out = ["## layer", "",
           "One Llama-3-8B decoder layer (hidden 4096, 32 query heads over 8 K/V heads of 128, "
           "MLP 14336) built from the kernels, against the same layer in PyTorch: eager, and "
           "`torch.compile` (`max-autotune-no-cudagraphs` for prefill, `reduce-overhead` for "
           "decode). Times are ms per layer; a decode row is one token per sequence against a "
           "cache of L - 1 tokens, appended in place. \"graph\" is our forward replayed from "
           "a `torch.cuda.CUDAGraph`. tokens/s is what 32 such layers would do, nothing else "
           "counted.", ""]
    prefill = sorted((r for r in rows if r.get("mode") == "prefill"),
                     key=lambda r: (r.get("B", 0), r.get("S", 0)))
    decode = sorted((r for r in rows if r.get("mode") == "decode"),
                    key=lambda r: (r.get("B", 0), r.get("L", 0)))
    if prefill:
        out += ["| shape | ours ms | torch eager ms | torch compiled ms | vs eager | "
                "vs compiled | tokens/s (32 layers) |", "|---|---|---|---|---|---|---|"]
        for r in prefill:
            out.append(f"| {r['shape']} | {fmt(r['spark_ms'])} | {fmt(r['torch_ms'])} | "
                       f"{fmt(r.get('torch_compiled_ms', 0))} | "
                       f"{speedup(r['torch_ms'], r['spark_ms'])} | "
                       f"{speedup(r.get('torch_compiled_ms', 0), r['spark_ms'])} | "
                       f"{r.get('tokens_per_s_32_layers', 0):,.0f} |")
        out.append("")
    if decode:
        out += ["| shape | ours ms | ours, graph ms | torch eager ms | torch compiled ms | "
                "graph vs eager (ours) | graph vs torch compiled | tokens/s (32 layers, graph) |",
                "|---|---|---|---|---|---|---|---|"]
        for r in decode:
            g = r.get("spark_graph_ms", 0)
            out.append(f"| {r['shape']} | {fmt(r['spark_ms'])} | {fmt(g)} | "
                       f"{fmt(r['torch_ms'])} | {fmt(r.get('torch_compiled_ms', 0))} | "
                       f"{speedup(r['spark_ms'], g)} | "
                       f"{speedup(r.get('torch_compiled_ms', 0), g)} | "
                       f"{r.get('tokens_per_s_32_layers_graph', 0):,.0f} |")
        out.append("")
        with_bd = [r for r in decode if r.get("breakdown_us")]
        if with_bd:
            out += ["Kernel time per stage of our decode step (us, from the torch profiler; "
                    "the sum is the GPU-busy time of one step):", "",
                    "| stage | " + " | ".join(r["shape"] for r in with_bd) + " |",
                    "|---|" + "---|" * len(with_bd)]
            for key, label in LAYER_STAGES:
                vals = [r["breakdown_us"].get(key, 0.0) for r in with_bd]
                if not any(vals):
                    continue
                out.append(f"| {label} | " + " | ".join(f"{v:.1f}" for v in vals) + " |")
            out.append("| total | " + " | ".join(
                f"{sum(r['breakdown_us'].values()):.1f}" for r in with_bd) + " |")
            out.append("")
    return out


def shape_size(kernel: str, shape: str) -> float:
    d = parse_shape(kernel, shape)
    if "M" in d:
        return d["M"] * d["N"] * d["K"]
    if "S_q" in d:
        return d["B"] * d["H"] * d["S_q"] * d["S_kv"] * d["D"]
    if "pos0" in d:  # rope: the qkv rows
        return d["B"] * d["S"] * (d["H"] + 2 * d["H_kv"]) * d["D"]
    if "rows" in d:
        return d["rows"] * d["cols"]
    return d.get("n", 0)


def pct(x: float) -> str:
    return f"{100.0 * x:.1f}%" if x > 0 else "—"


def fmt(x: float, nd=3) -> str:
    return f"{x:.{nd}f}" if x else "—"


def vs_ref(kernel: str, r: dict) -> str:
    """ref_ms is cuBLAS for the GEMMs (shown as % of cuBLAS throughput) and the naive variant
    or cudaMemcpy for everything else (shown as a speedup). Attention has no library
    reference in the C++ bench (ref_ms = 0); its comparison is the torch column."""
    ref_ms, ms = r.get("ref_ms", 0), r["median_ms"]
    if ref_ms <= 0 or ms <= 0:
        return "—"
    return pct(ref_ms / ms) if is_compute_bound_kernel(kernel) else f"{ref_ms / ms:.2f}×"


def peak_for(kernel: str, peaks: dict, dtype: str = "") -> tuple[str, float | None]:
    """(unit, ceiling) for a kernel and dtype; the ceiling is None when it is not known for
    the device. sgemm's tf32 rows are judged against the tf32 tensor-core peak (a third of
    it for 3xTF32), its f32 rows against the fp32 FMA peak."""
    if is_compute_bound_kernel(kernel):
        return "TFLOPS", compute_peak(kernel, dtype, peaks)
    return "GB/s", peaks["bw_gbps"]


def peak_label(kernel: str, peaks: dict, dtypes: list[str]) -> str:
    """Header text for the "% of peak" column: one figure when every dtype in the table shares
    a ceiling, else one per dtype ("f32 123.4, tf32 129.4, 3xtf32 43.1 TFLOPS")."""
    unit = peak_for(kernel, peaks)[0]
    per_dtype = {d: peak_for(kernel, peaks, d)[1] for d in dtypes}
    known = {d: p for d, p in per_dtype.items() if p}
    if not known:
        return "peak TBD"
    if len(set(round(p, 3) for p in known.values())) == 1:
        return f"{next(iter(known.values())):g} {unit}"
    return ", ".join(f"{d} {p:.1f}" for d, p in known.items()) + f" {unit}"


def ref_label(kernel: str) -> str:
    """Header of the reference column: % of the library for the GEMMs, a speedup otherwise."""
    if is_compute_bound_kernel(kernel):
        return f"% of {REFERENCE_NAME.get(kernel, 'cuBLAS')}"
    return "speedup vs ref"


def table_for(kernel: str, rows: list[dict], torch_rows: dict, peaks: dict) -> str:
    unit = peak_for(kernel, peaks)[0]
    dtypes = sorted({r["dtype"] for r in rows})
    has_ref = any(r.get("ref_ms", 0) > 0 for r in rows)
    has_torch = any((kernel, r["dtype"], r["shape"]) in torch_rows for r in rows)

    hdr = ["dtype", "shape", "variant", "median ms", "min ms", unit,
           f"% of peak ({peak_label(kernel, peaks, dtypes)})"]
    if has_ref:
        hdr.append(ref_label(kernel))
    if has_torch:
        hdr.append("speedup vs torch")
    hdr.append("ok")
    lines = ["| " + " | ".join(hdr) + " |", "|" + "|".join(["---"] * len(hdr)) + "|"]

    rows = sorted(rows, key=lambda r: (r["dtype"], shape_size(kernel, r["shape"]), r["variant"]))
    for r in rows:
        peak = peak_for(kernel, peaks, r["dtype"])[1]
        val = r.get("tflops", 0) if unit == "TFLOPS" else r.get("gbps", 0)
        cells = [
            r["dtype"],
            r["shape"],
            r["reference"] if is_reference(r) else str(r["variant"]),
            fmt(r["median_ms"], 4),
            fmt(r.get("min_ms", 0), 4),
            fmt(val, 2),
            pct(val / peak if peak else 0),
        ]
        if has_ref:
            cells.append(vs_ref(kernel, r))
        if has_torch:
            t = torch_rows.get((kernel, r["dtype"], r["shape"]))
            # torch comparison is for the fastest variant only
            best_variant = max(
                x["variant"] for x in rows if x["dtype"] == r["dtype"] and x["shape"] == r["shape"]
            )
            cells.append(f"{t['speedup']:.2f}×" if t and r["variant"] == best_variant else "—")
        cells.append("✅" if r.get("ok", True) else "❌")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def headline(by_kernel: dict[str, list[dict]], torch_rows: dict, peaks: dict) -> str:
    out = [
        "| kernel | dtype | shape | best variant | median ms | achieved | % of peak "
        "| vs cuBLAS / naive | vs torch |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for kernel in KERNEL_ORDER:
        rows = by_kernel.get(kernel)
        if not rows:
            continue
        for dtype in sorted({r["dtype"] for r in rows}):
            unit, peak = peak_for(kernel, peaks, dtype)
            drows = [r for r in rows
                     if r["dtype"] == dtype and r.get("ok", True) and not is_reference(r)]
            if not drows:
                continue
            biggest = max(shape_size(kernel, r["shape"]) for r in drows)
            cands = [r for r in drows if shape_size(kernel, r["shape"]) == biggest]
            best = min(cands, key=lambda r: r["median_ms"])
            val = best.get("tflops", 0) if unit == "TFLOPS" else best.get("gbps", 0)
            t = torch_rows.get((kernel, dtype, best["shape"]))
            out.append(
                "| {k} | {d} | {s} | v{v} | {ms:.4f} | {val:.1f} {u} | {p} | {c} | {t} |".format(
                    k=kernel, d=dtype, s=best["shape"], v=best["variant"], ms=best["median_ms"],
                    val=val, u=unit, p=pct(val / peak if peak else 0),
                    c=vs_ref(kernel, best),
                    t=f"{t['speedup']:.2f}×" if t else "—",
                )
            )
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bf16-peak", type=float, default=None, metavar="TFLOPS",
                    help="measured dense bf16 tensor-core peak, for devices where it is not "
                         "in shape_utils.DEVICE_PEAKS yet (RTX 5090)")
    args = ap.parse_args()
    rows = load_bench_rows(RESULTS)
    if not rows:
        print(f"no results in {RESULTS}/ — run ./scripts/run_all_benches.sh first", file=sys.stderr)
        return 1
    try:
        device, peaks = peaks_for_rows(rows, args.bf16_peak, measured_peaks(RESULTS))
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    torch_rows = load_torch_rows(device)
    by_kernel: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_kernel[r["kernel"]].append(r)

    bf16 = (f"{peaks['bf16_tflops']:.1f} TFLOPS bf16 tensor" if peaks["bf16_tflops"]
            else "bf16 tensor peak TBD (pass --bf16-peak)")
    if peaks.get("fp8_tflops"):
        bf16 += f", {peaks['fp8_tflops']:.1f} TFLOPS fp8 tensor"
    if peaks.get("tf32_tflops"):
        bf16 += f", {peaks['tf32_tflops']:.1f} TFLOPS tf32 tensor"
    if peaks.get("measured"):
        bf16 += (f"; the compute peaks were measured by bench_peak at "
                 f"{peaks.get('sm_mhz', 0):.0f} MHz (results/peak.json)")
    md = ["# Benchmark results", "",
          "Generated by `scripts/make_results_table.py` from `results/*.json` "
          "(C++ benches, CUDA-event median; 100 iterations by default, 50 for hgemm and "
          "bandwidth, unless `--iters` was passed) and `results/torch_comparison.json` "
          "(PyTorch eager on the same shapes).", "",
          f"Device: **{device}**. Peaks used: {peaks['bw_gbps']:g} GB/s DRAM, "
          f"{peaks['fp32_tflops']:.1f} TFLOPS fp32, {bf16} "
          f"(see docs/{SHEETS.get(device, 'RTX5090.md')}).", "",
          "## Headline", "", headline(by_kernel, torch_rows, peaks), ""]
    for kernel in KERNEL_ORDER + sorted(k for k in by_kernel if k not in KERNEL_ORDER):
        if kernel not in by_kernel:
            continue
        krows = by_kernel[kernel]
        md += [f"## {kernel}", ""]
        ai_examples = sorted({(r["dtype"], r["shape"]) for r in krows}, key=lambda x: x[1])[:1]
        if ai_examples and not is_compute_bound_kernel(kernel):
            d, s = ai_examples[0]
            ai = arithmetic_intensity(kernel, d, s)
            if ai > 0:
                md.append(f"Arithmetic intensity ≈ {ai:.2f} FLOP/byte "
                          f"({d}, {s}) — memory-bound; the ceiling is DRAM bandwidth.")
            else:  # the bandwidth copy
                md.append("No arithmetic: a pure copy, so GB/s here is the realistic DRAM roof "
                          "for the row-wise kernels below.")
            md.append("")
        md += [table_for(kernel, krows, torch_rows, peaks), ""]
    layer_rows = load_layer_rows(device)
    if layer_rows:
        md += layer_section(layer_rows)
    OUT_MD.parent.mkdir(exist_ok=True)
    OUT_MD.write_text("\n".join(md))
    OUT_HEADLINE.write_text(headline(by_kernel, torch_rows, peaks) + "\n")
    print(f"wrote {OUT_MD} and {OUT_HEADLINE}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
