"""Linear cross-entropy at an lm_head's shapes: torch eager, torch.compile of the same
function, and spark_kernels.linear_cross_entropy variants 1 to 3 (docs/design/cross_entropy.md).

    python scripts/bench_cross_entropy.py --vocab 128256 --tokens 2048 4096 8192
    python scripts/bench_cross_entropy.py --vocab 151936 --tokens 8192 --methods v1 v2

For each (V, T) and method it times the forward alone (under no_grad: the loss) and the
forward plus the backward (torch.autograd.grad of the loss for h and w, so nothing is
accumulated into .grad), and records the peak memory above what was allocated before the
call (torch.cuda.max_memory_allocated; the returned gradients count, dw is V x D bf16).
The process caps itself at --budget-gb with set_per_process_memory_fraction, so a method
that would need more fails with an out-of-memory error and is recorded as not fitting
instead of pushing the card's other tenants out. Before timing, each method's first call
is checked against the first method that fits (torch's, or variant 1's where torch does
not fit): the loss, and dh and dw. Times are medians of --iters launches between CUDA
events after a warmup and a 300 ms clock ramp. Rows are appended to
results/cross_entropy.json as they are measured, one JSON object per (V, T, method, mode).
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cross_entropy.json"
D_MODEL = 4096
RAMP_MS = 300
_ramped = False


def gpu_state() -> str:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,clocks.sm,power.draw,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def ramp_clocks(fn) -> None:
    global _ramped
    if _ramped:
        return
    _ramped = True
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e3 < RAMP_MS:
        fn()
        torch.cuda.synchronize()


def time_ms(fn, warmup: int, iters: int) -> float:
    """Median of `iters` launches of fn, each between its own pair of CUDA events."""
    ramp_clocks(fn)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    stops = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for a, b in zip(starts, stops, strict=True):
        a.record()
        fn()
        b.record()
    torch.cuda.synchronize()
    return statistics.median(a.elapsed_time(b) for a, b in zip(starts, stops, strict=True))


def peak_gb(fn) -> float:
    """Peak allocated memory above the current allocation while fn runs, in GB."""
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    del out
    return peak / 1e9


def reference(h, w, y):
    return F.cross_entropy((h @ w.t()).float(), y)


def make_methods(sk, chunk):
    compiled = torch.compile(reference, dynamic=False)
    return {
        "torch": reference,
        "torch_compile": compiled,
        "v1": lambda h, w, y: sk.linear_cross_entropy(h, w, y, variant=1, chunk=chunk),
        "v2": lambda h, w, y: sk.linear_cross_entropy(h, w, y, variant=2, chunk=chunk),
        "v3": lambda h, w, y: sk.linear_cross_entropy(h, w, y, variant=3, chunk=chunk),
    }


def write_row(row: dict, args) -> None:
    if not args.no_write:
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")


def max_err(a: torch.Tensor, b: torch.Tensor, rows: int = 8192) -> tuple[float, float]:
    """(max |a - b|, max |b|) in fp32, a slab of rows at a time so dw needs no fp32 copy."""
    err = ref = 0.0
    for i in range(0, a.shape[0], rows):
        x, y = a[i:i + rows].float(), b[i:i + rows].float()
        err = max(err, (x - y).abs().max().item())
        ref = max(ref, y.abs().max().item())
    return err, ref


def run_shape(sk, V: int, T: int, methods: dict, names: list[str], args) -> list[dict]:
    g = torch.Generator(device="cuda").manual_seed(V + T)
    h = torch.randn(T, D_MODEL, device="cuda", generator=g).to(torch.bfloat16)
    w = (torch.randn(V, D_MODEL, device="cuda", generator=g) * (3.0 / D_MODEL**0.5)).to(
        torch.bfloat16)
    y = torch.randint(0, V, (T,), device="cuda", generator=g)
    y[torch.rand(T, device="cuda", generator=g) < 0.05] = -100
    h.requires_grad_()
    w.requires_grad_()
    rows = []
    # The first method that fits is the reference for the others' loss and gradients
    # (torch's, or variant 1's where torch and torch.compile do not fit), checked on the
    # first call, before anything is timed.
    ref_loss, ref_grads = None, None
    for name in names:
        fn = methods[name]

        def fwd(fn=fn):
            with torch.no_grad():
                return fn(h, w, y)

        def fwd_bwd(fn=fn):
            return torch.autograd.grad(fn(h, w, y), (h, w))

        for mode, step in (("fwd", fwd), ("fwd_bwd", fwd_bwd)):
            row = {"kernel": "cross_entropy", "V": V, "D": D_MODEL, "T": T, "method": name,
                   "mode": mode, "chunk": args.chunk if name in ("v1", "v2", "v3") else None,
                   "budget_gb": args.budget_gb}
            t0 = time.perf_counter()
            try:
                out = step()  # also compiles torch_compile
                torch.cuda.synchronize()
                row["first_call_s"] = round(time.perf_counter() - t0, 2)
                if mode == "fwd":
                    loss = out.item()
                    if ref_loss is None:
                        ref_loss, row["reference"] = loss, True
                    row["loss"] = loss
                    assert abs(loss - ref_loss) <= 2e-3 * abs(ref_loss) + 1e-3, (
                        f"{name} loss {loss} vs {ref_loss} at V={V} T={T}")
                elif ref_grads is None:
                    ref_grads, row["reference"] = tuple(t.detach() for t in out), True
                else:
                    for a, b, what in zip(out, ref_grads, ("dh", "dw"), strict=True):
                        err, top = max_err(a, b)
                        row[f"{what}_err_rel"] = round(err / top, 5)
                        assert err <= 3e-2 * top + 1e-6, (
                            f"{name} {what} max error {err:.3e} against {top:.3e} at V={V} T={T}")
                del out
                row["peak_extra_gb"] = round(peak_gb(step), 3)
                row["state_before"] = gpu_state()
                row["ms"] = round(time_ms(step, args.warmup, args.iters), 3)
                row["state_after"] = gpu_state()
                row["fits"] = True
                flops = (2 if mode == "fwd" else 6) * T * V * D_MODEL
                row["tflops_6tvd" if mode == "fwd_bwd" else "tflops_2tvd"] = round(
                    flops / row["ms"] / 1e9, 1)
            except torch.OutOfMemoryError:
                row = {k: v for k, v in row.items() if k in ("kernel", "V", "D", "T", "method",
                                                             "mode", "chunk", "budget_gb")}
                row["fits"] = False
                row["note"] = f"does not fit in {args.budget_gb:g} GB"
            gc.collect()
            torch.cuda.empty_cache()
            print(json.dumps(row), flush=True)
            write_row(row, args)
            rows.append(row)
    return rows


def run_pieces(sk, V: int, T: int, chunk: int, args) -> list[dict]:
    """The kernels one at a time at one chunk (and the forward GEMMs over all T rows): the
    three ce_gemm epilogues against torch's h @ w.T (cuBLAS), the row kernel, the transpose,
    the two hgemm products of a chunk's backward, and variant 3's band kernel (the recompute
    and both products in one launch, counted as three GEMMs)."""
    C_ = sk.ops._C
    g = torch.Generator(device="cuda").manual_seed(1)
    h = torch.randn(T, D_MODEL, device="cuda", generator=g).to(torch.bfloat16)
    w = (torch.randn(V, D_MODEL, device="cuda", generator=g) * (3.0 / D_MODEL**0.5)).to(
        torch.bfloat16)
    y = torch.randint(0, V, (T,), device="cuda", generator=g)
    hc, yc = h[:chunk], y[:chunk]
    logits = torch.empty(chunk, V, device="cuda", dtype=torch.bfloat16)
    lse, _ = C_.ce_stats(hc, w, yc, -100)
    scale = torch.full((1,), 1.0 / T, device="cuda")
    c_pad = (chunk + 63) // 64 * 64
    slab = min(sk.ops.CE_SLAB, V)
    slab_buf = torch.empty(slab, c_pad, device="cuda", dtype=torch.bfloat16)
    dh = torch.empty(chunk, D_MODEL, device="cuda", dtype=torch.bfloat16)
    dw = torch.empty(V, D_MODEL, device="cuda", dtype=torch.bfloat16)
    C_.ce_logits(hc, w, logits)
    dh_f32 = torch.zeros((chunk + 127) // 128 * 128, D_MODEL, device="cuda")
    slots = torch.zeros(C_.ce_backward_slots() * 128, D_MODEL, device="cuda")
    flags = torch.zeros(2 * (V // 128) + 1, device="cuda", dtype=torch.int32)

    def bwd_band():
        flags.zero_()
        dh_f32.zero_()
        C_.ce_backward(hc, w, yc, lse, scale, -100, dh_f32, slots, flags, dw, True)

    gemm = 2 * chunk * V * D_MODEL
    move = chunk * V * 2
    pieces = {
        "torch h @ w.T": (lambda: hc @ w.t(), gemm, None),
        "ce_gemm logits": (lambda: C_.ce_logits(hc, w, logits), gemm, None),
        "ce_gemm stats + combine": (lambda: C_.ce_stats(hc, w, yc, -100), gemm, None),
        "ce_gemm grad": (lambda: C_.ce_grad(hc, w, yc, lse, scale, -100, logits), gemm, None),
        "ce_rows loss only": (lambda: C_.ce_rows(logits, yc, -100, None), None, move),
        "ce_rows loss + grad": (lambda: C_.ce_rows(logits, yc, -100, scale), None, 2 * move),
        "transpose one slab": (lambda: C_.ce_transpose(logits, 0, slab, c_pad, slab_buf), None,
                               2 * chunk * slab * 2),
        "hgemm dh = g @ w": (lambda: sk.hgemm(logits, w, out=dh), gemm, None),
        "hgemm dw slab += g^T h": (
            lambda: sk.hgemm(slab_buf, hc, residual=dw[:slab], out=dw[:slab]),
            2 * slab * c_pad * D_MODEL, None),
        "ce_backward band (v3)": (bwd_band, 3 * gemm, None),
    }
    rows = []
    for name, (fn, flops, nbytes) in pieces.items():
        ms = time_ms(fn, args.warmup, args.iters)
        row = {"kernel": "cross_entropy_piece", "V": V, "D": D_MODEL, "T": T, "chunk": chunk,
               "piece": name, "ms": round(ms, 4), "state_after": gpu_state()}
        if flops:
            row["tflops"] = round(flops / ms / 1e9, 1)
        if nbytes:
            row["gbps"] = round(nbytes / ms / 1e6, 0)
        print(json.dumps(row), flush=True)
        rows.append(row)
        write_row(row, args)
    for name, fn in (("ce_gemm stats + combine, all T", lambda: C_.ce_stats(h, w, y, -100)),
                     ("torch h @ w.T, all T", lambda: h @ w.t())):
        ms = time_ms(fn, args.warmup, max(10, args.iters // 5))
        row = {"kernel": "cross_entropy_piece", "V": V, "D": D_MODEL, "T": T, "chunk": T,
               "piece": name, "ms": round(ms, 4),
               "tflops": round(2 * T * V * D_MODEL / ms / 1e9, 1), "state_after": gpu_state()}
        print(json.dumps(row), flush=True)
        rows.append(row)
        write_row(row, args)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--vocab", type=int, nargs="+", default=[128256, 151936])
    ap.add_argument("--tokens", type=int, nargs="+", default=[2048, 4096, 8192])
    ap.add_argument("--methods", nargs="+",
                    default=["torch", "torch_compile", "v1", "v2", "v3"])
    ap.add_argument("--chunk", type=int, default=None, help="rows per chunk (default CE_CHUNK)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--budget-gb", type=float, default=10.0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--pieces", action="store_true",
                    help="time the kernels one at a time instead of the whole op")
    args = ap.parse_args()

    import spark_kernels as sk

    if args.chunk is None:
        args.chunk = sk.ops.CE_CHUNK
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.budget_gb * 1e9 / total))
    rows = []
    if args.pieces:
        for V in args.vocab:
            for T in args.tokens:
                rows += run_pieces(sk, V, T, min(args.chunk, T), args)
    else:
        methods = make_methods(sk, args.chunk)
        for V in args.vocab:
            for T in args.tokens:
                rows += run_shape(sk, V, T, methods, args.methods, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
