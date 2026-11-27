"""The final logit softcap at Inkling 2 Mini's lm_head (V = 200058 unpadded, D = 2304, c = 30):
linear_cross_entropy variants 1 to 3 with and without the cap, against torch eager and
torch.compile with it (HF's spelling: the bf16 logits, `.float()`, c * tanh(x / c), the loss).

    python scripts/bench_logit_softcap.py --tokens 4096 16384

Per (T, method): the forward and the backward (torch.autograd.grad of the loss for h and w),
its peak memory above what was allocated before the call (the returned dh and dw count, dw is
V x D bf16, 0.92 GB), and its time. The timing interleaves the methods: every round times one
call of each, in an order that rotates by one per round, and the row is the median over the
rounds. On a card other jobs share, a slow stretch then lands on every method instead of on
whichever ran during it, so the cap's cost (the same variant with and without it, timed in
the same rounds) holds up where back-to-back medians did not. Before timing, each method's
loss and gradients are checked against torch's (capped methods) or variant 1's (uncapped).
Rows are appended to results/cross_entropy.json with "bench": "logit_softcap".
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_cross_entropy import gpu_state, max_err, peak_gb  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "cross_entropy.json"
V, D, CAP = 200058, 2304, 30.0


def torch_loss(cap):
    def fn(h, w, y):
        logits = (h @ w.t()).float()
        if cap is not None:
            logits = torch.tanh(logits / cap) * cap
        return F.cross_entropy(logits, y)
    return fn


def methods(sk, chunk):
    def ours(variant, cap):
        return lambda h, w, y: sk.linear_cross_entropy(h, w, y, variant=variant, chunk=chunk,
                                                       softcap=cap)
    m = {"torch": torch_loss(CAP), "torch_compile": torch.compile(torch_loss(CAP), dynamic=False)}
    for v in (1, 2, 3):
        m[f"v{v}"] = ours(v, CAP)
        m[f"v{v}_nocap"] = ours(v, None)
    return m


def run(sk, T, names, args):
    g = torch.Generator(device="cuda").manual_seed(T)
    h = torch.randn(T, D, device="cuda", generator=g).to(torch.bfloat16).requires_grad_()
    # raw logits of standard deviation 3, as bench_cross_entropy.py's: the time does not
    # depend on the values, and uncapped logits much wider than that make variant 1's bf16
    # logits too coarse for the 3% gradient check against it
    w = (torch.randn(V, D, device="cuda", generator=g) * (3.0 / D**0.5)).to(torch.bfloat16)
    w.requires_grad_()
    y = torch.randint(0, V, (T,), device="cuda", generator=g)
    y[torch.rand(T, device="cuda", generator=g) < 0.05] = -100
    fns = methods(sk, args.chunk)
    steps, rows, refs = {}, {}, {}
    for name in names:
        fn = fns[name]

        def step(fn=fn):
            return torch.autograd.grad(fn(h, w, y), (h, w))

        row = {"kernel": "cross_entropy", "bench": "logit_softcap", "V": V, "D": D, "T": T,
               "method": name, "mode": "fwd_bwd", "softcap": None if "nocap" in name else CAP,
               "chunk": args.chunk if name.startswith("v") else None,
               "budget_gb": args.budget_gb, "device": torch.cuda.get_device_name()}
        try:
            with torch.no_grad():
                loss = fn(h, w, y).item()
            grads = tuple(t.detach() for t in step())
            ref_key = "nocap" if "nocap" in name else "cap"
            if ref_key not in refs:
                refs[ref_key] = (loss, grads)
                row["reference"] = True
            ref_loss, ref_grads = refs[ref_key]
            row["loss"] = loss
            assert abs(loss - ref_loss) <= 2e-3 * abs(ref_loss) + 1e-3, (name, loss, ref_loss)
            for a, b, what in zip(grads, ref_grads, ("dh", "dw"), strict=True):
                err, top = max_err(a, b)
                row[f"{what}_err_rel"] = round(err / top, 5)
                assert err <= 3e-2 * top + 1e-6, (name, what, err, top)
            del grads
            row["peak_extra_gb"] = round(peak_gb(step), 3)
            row["fits"] = True
            steps[name] = step
        except torch.OutOfMemoryError:
            row["fits"] = False
            row["note"] = f"does not fit in {args.budget_gb:g} GB"
        rows[name] = row
        gc.collect()
        torch.cuda.empty_cache()
    live = list(steps)
    for name in live:  # warmup
        for _ in range(args.warmup):
            steps[name]()
    torch.cuda.synchronize()
    times = {name: [] for name in live}
    state_before = gpu_state()
    for r in range(args.rounds):
        order = live[r % len(live):] + live[: r % len(live)]
        for name in order:
            a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            a.record()
            steps[name]()
            b.record()
            b.synchronize()
            times[name].append(a.elapsed_time(b))
    state_after = gpu_state()
    out = []
    for name in names:
        row = rows[name]
        if name in times:
            ts = sorted(times[name])
            row["ms"] = round(statistics.median(ts), 3)
            row["ms_min"] = round(ts[0], 3)
            row["rounds"] = args.rounds
            row["tflops_6tvd"] = round(6 * T * V * D / row["ms"] / 1e9, 1)
            row["state_before"], row["state_after"] = state_before, state_after
        print(json.dumps(row), flush=True)
        if not args.no_write:
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokens", type=int, nargs="+", default=[4096, 16384])
    ap.add_argument("--methods", nargs="+",
                    default=["torch", "torch_compile", "v1", "v1_nocap", "v2", "v2_nocap", "v3",
                             "v3_nocap"])
    ap.add_argument("--chunk", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--budget-gb", type=float, default=32.0)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args()

    import spark_kernels as sk

    if args.chunk is None:
        args.chunk = sk.ops.CE_CHUNK
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1.0, args.budget_gb * 1e9 / total))
    for T in args.tokens:
        run(sk, T, args.methods, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
