#!/usr/bin/env python3
"""Quantization error of the fp4 GEMM (NVFP4, MXFP4) on the GEMM output, next to the fp8 ones,
for Gaussian inputs and the outlier patterns that tell block scales apart.

    python scripts/fp4_accuracy.py [--m 4096 --n 4096 --k 4096] [--seed 0]

For each input pair (bf16 A[M,K], B[N,K]) the fp32 product of the bf16 values is the
reference. Each scheme quantizes A and B and multiplies: NVFP4 (e2m1, an e4m3 scale per 16 and
a per-tensor fp32 scale) and MXFP4 (e2m1, a power-of-two scale per 32) through sk.fp4_quantize
and sk.fp4gemm, per-tensor e4m3 and MXFP8 through the fp8 reference quantizers and
sk.fp8gemm; without the extension the reference functions do the same arithmetic. The table
gives the largest error relative to the largest |C| and the Frobenius norm of the error
relative to that of C. The inputs are those of scripts/mx_accuracy.py: Gaussian; 0.1% of the
entries x 100; 8 of the K channels of A x 50 and x 2000. The numbers are in
docs/design/fp4gemm.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mx_accuracy import errors, gemm_mx, gemm_per_tensor, inputs  # noqa: E402

from spark_kernels import reference  # noqa: E402

try:
    import spark_kernels as sk

    sk.num_variants("fp4gemm")
except Exception:  # not built: the reference does the same fp32 arithmetic
    sk = None


def gemm_fp4(a: torch.Tensor, b_t: torch.Tensor, fmt: str) -> torch.Tensor:
    if sk is not None:
        qa, sfa, sa = sk.fp4_quantize(a, fmt)
        qb, sfb, sb = sk.fp4_quantize(b_t, fmt)
        return sk.fp4gemm(qa, qb, sfa, sfb, sa, sb, fmt=fmt).float()
    if fmt == "nvfp4":
        qa, sfa, sa = reference.quantize_nvfp4(a)
        qb, sfb, sb = reference.quantize_nvfp4(b_t)
    else:
        (qa, sfa), (qb, sfb) = reference.quantize_mxfp4(a), reference.quantize_mxfp4(b_t)
        sa = sb = None
    return reference.fp4gemm(qa, sfa, qb, sfb, fmt, sa, sb).float()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    print(f"# {args.m}x{args.n}x{args.k}, errors relative to the fp32 product of the bf16 "
          f"inputs ({'sk.fp4gemm / sk.fp8gemm' if sk else 'reference'}): max |err| / max |C|, "
          "||err|| / ||C||")
    print("| input | NVFP4 | MXFP4 | e4m3 per-tensor | MXFP8 |")
    print("|---|---|---|---|---|")
    for kind in ("gaussian", "sparse outliers, 0.1% x 100", "channel outliers, 8 of K x 50",
                 "channel outliers, 8 of K x 2000"):
        a, b = inputs(kind, args.m, args.n, args.k, gen)
        ref = a.float() @ b.float().t()
        cells = [errors(gemm_fp4(a, b, "nvfp4"), ref), errors(gemm_fp4(a, b, "mxfp4"), ref),
                 errors(gemm_per_tensor(a, b), ref), errors(gemm_mx(a, b, "floor"), ref)]
        print(f"| {kind} | " + " | ".join(f"{m:.2e} / {f:.2e}" for m, f in cells) + " |")


if __name__ == "__main__":
    main()
