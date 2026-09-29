#!/usr/bin/env python3
"""Error of the fp8 attention (sk.attention_fp8) against an fp64 reference and against the bf16
kernel (sk.attention), for the input distributions that tell the quantization choices apart.

    python scripts/attention_fp8_accuracy.py [--b 1 --h 8 --s 4096 --d 128] [--seed 0]

For each input (q, k, v in fp32) the reference is F.scaled_dot_product_attention in fp64 on
the unquantized values. The rows are: the bf16 kernel on the bf16-rounded inputs; the fp8
baseline (variant 0: e4m3 inputs, P kept in fp32), which isolates the error of quantizing q,
k and v; the fp8 kernel (variant 1: P rounded to e4m3 as well) with per-tensor and with
per-head scales; and variant 1 with the probabilities rounded without the 2^8 shift, run in a
child process with SPARK_ATTN_FP8_PSHIFT=0 because the kernel reads its knobs once. Each cell
is max |O - O_ref| / mean |O - O_ref| over the whole output, both absolute; the last columns
are max |O_ref| and mean |O_ref| for scale. The numbers are in docs/design/attention.md.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
import spark_kernels as sk  # noqa: E402

KINDS = ["gaussian", "gaussian, q x 4 (peaked softmax)", "sparse outliers, 0.1% x 20",
         "key channel outliers, 4 of D x 20", "value channel outliers, 4 of D x 20"]


def inputs(kind: str, B: int, H: int, S: int, D: int, gen: torch.Generator):
    q, k, v = (torch.randn(B, H, S, D, generator=gen, device="cuda") for _ in range(3))
    if kind == "gaussian, q x 4 (peaked softmax)":
        q *= 4.0
    elif kind == "sparse outliers, 0.1% x 20":
        for x in (q, k, v):
            x[torch.rand(x.shape, generator=gen, device="cuda") < 1e-3] *= 20.0
    elif kind.startswith("key channel outliers"):
        k[..., torch.randperm(D, generator=gen, device="cuda")[:4]] *= 20.0
    elif kind.startswith("value channel outliers"):
        v[..., torch.randperm(D, generator=gen, device="cuda")[:4]] *= 20.0
    elif kind != "gaussian":
        raise ValueError(kind)
    return q, k, v


def err(got: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    e = (got.double() - ref).abs()
    return e.max().item(), e.mean().item()


def fp8_run(q, k, v, causal, variant, per_head):
    qs, ks, vs = (sk.quantize_fp8(x, per_head=per_head) for x in (q, k, v))
    return sk.attention_fp8(qs[0], ks[0], vs[0], qs[1], ks[1], vs[1], causal=causal,
                            variant=variant)


def measure(args, only_shift0: bool = False) -> dict:
    out = {}
    for kind in KINDS:
        gen = torch.Generator(device="cuda").manual_seed(args.seed)
        q, k, v = inputs(kind, args.b, args.h, args.s, args.d, gen)
        for causal in (False, True):
            key = f"{kind} | {'causal' if causal else 'full'}"
            if only_shift0:
                ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(),
                                                     is_causal=causal)
                out[key] = {"v1 per-tensor, no P shift": err(
                    fp8_run(q, k, v, causal, 1, False), ref)}
                continue
            ref = F.scaled_dot_product_attention(q.double(), k.double(), v.double(),
                                                 is_causal=causal)
            bf = sk.attention(q.bfloat16(), k.bfloat16(), v.bfloat16(), causal=causal)
            row = {"bf16 kernel": err(bf, ref),
                   "v0 per-tensor (P in fp32)": err(fp8_run(q, k, v, causal, 0, False), ref),
                   "v1 per-tensor": err(fp8_run(q, k, v, causal, 1, False), ref),
                   "v1 per-head": err(fp8_run(q, k, v, causal, 1, True), ref),
                   "v1 per-tensor vs the bf16 kernel": err(
                       fp8_run(q, k, v, causal, 1, False), bf.double()),
                   "ref": (ref.abs().max().item(), ref.abs().mean().item())}
            out[key] = row
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--b", type=int, default=1)
    ap.add_argument("--h", type=int, default=8)
    ap.add_argument("--s", type=int, default=4096)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shift0-json", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.shift0_json:  # the child: the knob is read once per process
        print(json.dumps(measure(args, only_shift0=True)))
        return 0
    rows = measure(args)
    child = subprocess.run(
        [sys.executable, __file__, "--b", str(args.b), "--h", str(args.h), "--s", str(args.s),
         "--d", str(args.d), "--seed", str(args.seed), "--shift0-json"],
        env={**os.environ, "SPARK_ATTN_FP8_PSHIFT": "0"}, capture_output=True, text=True,
        check=True)
    for key, cell in json.loads(child.stdout).items():
        rows[key].update(cell)

    cols = ["bf16 kernel", "v0 per-tensor (P in fp32)", "v1 per-tensor", "v1 per-head",
            "v1 per-tensor, no P shift", "v1 per-tensor vs the bf16 kernel"]
    print(f"b{args.b} h{args.h} s{args.s} d{args.d}: max |error| / mean |error| (absolute)\n")
    print("| input | mask | " + " | ".join(cols) + " | max / mean |O_ref| |")
    print("|---|---|" + "---|" * (len(cols) + 1))
    for key, row in rows.items():
        kind, mask = key.split(" | ")
        cells = [f"{row[c][0]:.2e} / {row[c][1]:.2e}" for c in cols]
        ref = row["ref"]
        print(f"| {kind} | {mask} | " + " | ".join(cells) + f" | {ref[0]:.2f} / {ref[1]:.3f} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
