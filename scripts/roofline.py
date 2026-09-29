#!/usr/bin/env python3
"""Roofline plot with every kernel's best measured point.

Reads results/*.json (from run_all_benches.sh) and writes results/roofline.png. The ceilings
come from shape_utils.DEVICE_PEAKS for the device named in the rows (RTX 5090: 1792 GB/s DRAM,
104.8 TFLOPS fp32; GB10: 273 GB/s, 31 TFLOPS fp32, 213 TFLOPS bf16), overridden by the measured
ones in results/peak.json. The bf16, fp8, fp4 and tf32 tensor-core roofs are only drawn when they
are known: run bench_peak, or pass --bf16-peak=<TFLOPS> for the bf16 one. sgemm's tf32 and
3xtf32 rows (variants 6 and 7) are plotted as their own series against the tf32 roof.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shape_utils import (  # noqa: E402
    TF32_DTYPES,
    arithmetic_intensity,
    flops,
    is_reference,
    load_bench_rows,
    measured_peaks,
    parse_shape,
    peaks_for_rows,
    traffic_bytes,
)

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
OUT = RESULTS / "roofline.png"

MARKERS = {"bandwidth": "P", "rmsnorm": "o", "add_rmsnorm": "D", "swiglu": "s", "softmax": "^",
           "sgemm": "v", "sgemm (tf32)": "<", "sgemm (3xtf32)": ">", "hgemm": "*", "fp8gemm": "h",
           "fp4gemm": "H", "fp4quant": "d", "attention": "X"}


def series_of(r: dict) -> str:
    """Legend entry of a row: the kernel, except that sgemm's tensor-core rows get their own."""
    if r["dtype"] in TF32_DTYPES:
        return f"{r['kernel']} ({r['dtype']})"
    return r["kernel"]


def achieved_tflops(r: dict) -> float:
    """Recompute from time so memory-bound kernels get a FLOP/s too."""
    dims = parse_shape(r["kernel"], r["shape"])
    fl = flops(r["kernel"], dims)
    if fl <= 0 and r["kernel"] == "bandwidth":
        # copy has no FLOPs; plot it at its byte-throughput as if 1 FLOP/byte for visibility
        fl = traffic_bytes(r["kernel"], r["dtype"], dims)
    return fl / (r["median_ms"] * 1e-3) / 1e12 if r["median_ms"] > 0 else 0.0


def plot_intensity(r: dict) -> float:
    """x position on the roofline. The copy has no FLOPs, so it is pinned at 1 FLOP/byte to
    match achieved_tflops(); there its height reads directly as bytes/s against the DRAM roof."""
    if r["kernel"] == "bandwidth":
        return 1.0
    return arithmetic_intensity(r["kernel"], r["dtype"], r["shape"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bf16-peak", type=float, default=None, metavar="TFLOPS",
                    help="measured dense bf16 tensor-core peak for this device")
    args = ap.parse_args()
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: pip install matplotlib", file=sys.stderr)
        return 1

    # our variants only: the cuBLAS / cudaMemcpy reference rows are not points of the ladder
    rows = [r for r in load_bench_rows(RESULTS) if r.get("ok", True) and not is_reference(r)]
    if not rows:
        print(f"no results in {RESULTS}/ — run ./scripts/run_all_benches.sh first", file=sys.stderr)
        return 1

    # best (fastest) row per (kernel, dtype, shape), then keep every shape so the plot shows
    # the trend along arithmetic intensity
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["kernel"], r["dtype"], r["shape"])
        if key not in best or r["median_ms"] < best[key]["median_ms"]:
            best[key] = r

    try:
        device, peaks = peaks_for_rows(rows, args.bf16_peak, measured_peaks(RESULTS))
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    bw = peaks["bw_gbps"] * 1e9
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)
    ai = [2.0**k for k in range(-4, 13)]
    roofs = [("fp32 CUDA cores", peaks["fp32_tflops"], "#444"),
             ("tf32 tensor cores", peaks.get("tf32_tflops"), "#d62728"),
             ("bf16 tensor cores", peaks["bf16_tflops"], "#76b900"),
             ("fp8 tensor cores", peaks.get("fp8_tflops"), "#1f77b4"),
             ("fp4 tensor cores", peaks.get("fp4_tflops"), "#9467bd")]
    for label, tflops, color in roofs:
        if not tflops:  # unknown for this device (the tensor peaks until bench_peak has run)
            continue
        ax.plot(ai, [min(bw * x, tflops * 1e12) / 1e12 for x in ai], color=color, lw=1.5,
                label=f"{label}: {tflops:.0f} TFLOPS")
        ax.axvline(tflops * 1e12 / bw, color=color, ls=":", lw=0.8)

    by_kernel: dict[str, list[dict]] = defaultdict(list)
    for r in best.values():
        by_kernel[series_of(r)].append(r)
    for kernel, krows in sorted(by_kernel.items()):
        xs = [plot_intensity(r) for r in krows]
        ys = [achieved_tflops(r) for r in krows]
        pts = [(x, y) for x, y in zip(xs, ys, strict=True) if x > 0 and y > 0]
        if not pts:
            continue
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], marker=MARKERS.get(kernel, "o"), s=55,
                   label=kernel, zorder=3, edgecolors="black", linewidths=0.4)

    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=10)
    ax.set_xlabel("arithmetic intensity (FLOP / DRAM byte)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title(f"{device} roofline — DRAM {peaks['bw_gbps']:g} GB/s")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
