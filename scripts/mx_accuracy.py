#!/usr/bin/env python3
"""Quantization error of per-tensor e4m3 against MXFP8 (a ue8m0 scale per 32 elements) on the
GEMM output, for the input distributions that tell them apart.

    python scripts/mx_accuracy.py [--m 4096 --n 4096 --k 4096] [--seed 0]

For each input pair (bf16 A[M,K], B[N,K]) the fp32 product of the original values is the
reference; both quantizations are applied to A and B, the GEMM of the dequantized values is
computed in fp32 (through sk.fp8gemm when the extension is importable, else through the
reference matmul, the same arithmetic), and the table reports the largest error relative to
the largest |C| and the Frobenius norm of the error relative to that of C. The inputs:
Gaussian; Gaussian with 0.1% of the entries scaled by 100 (isolated outliers); Gaussian with
8 of the K channels of A scaled by 50 and by 2000 (the activation-outlier pattern of
transformer residual streams). A per-tensor scale puts the tensor's largest element at 448
and every element more than 2^14.8 below it into e4m3's subnormals, where the absolute step
is 2^-9 of the scale; a per-32 scale only does that inside the block that holds the
outlier. The numbers are in docs/design/fp8gemm.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from spark_kernels import reference  # noqa: E402

try:
    import spark_kernels as sk

    sk.num_variants("fp8gemm")
except Exception:  # not built: the reference does the same fp32 arithmetic
    sk = None


def inputs(
    kind: str, M: int, N: int, K: int, gen: torch.Generator
) -> tuple[torch.Tensor, torch.Tensor]:
    a = torch.randn(M, K, generator=gen, device="cuda")
    b = torch.randn(N, K, generator=gen, device="cuda")
    if kind == "sparse outliers, 0.1% x 100":
        for x in (a, b):
            mask = torch.rand(x.shape, generator=gen, device="cuda") < 1e-3
            x[mask] *= 100.0
    elif kind.startswith("channel outliers, 8 of K x "):
        ch = torch.randperm(K, generator=gen, device="cuda")[:8]
        a[:, ch] *= float(kind.rsplit("x ", 1)[1])
    elif kind != "gaussian":
        raise ValueError(kind)
    return a.to(torch.bfloat16), b.to(torch.bfloat16)


def gemm_per_tensor(a: torch.Tensor, b_t: torch.Tensor) -> torch.Tensor:
    qa, sa = reference.quantize_per_tensor(a)
    qb, sb = reference.quantize_per_tensor(b_t)
    if sk is not None:
        return sk.fp8gemm(qa, qb, sa, sb).float()
    return ((qa.float() @ qb.float().t()) * (sa * sb)).to(torch.bfloat16).float()


def gemm_mx(a: torch.Tensor, b_t: torch.Tensor, mode: str) -> torch.Tensor:
    qa, sfa = reference.quantize_mx(a, mode=mode)
    qb, sfb = reference.quantize_mx(b_t, mode=mode)
    if sk is not None:
        return sk.fp8gemm(qa, qb, sfa=sfa, sfb=sfb).float()
    return reference.fp8gemm_mx(qa, sfa, qb, sfb).float()


def errors(c: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """(max |error| / max |ref|, ||error|| / ||ref||)."""
    d = c - ref
    return (d.abs().max() / ref.abs().max()).item(), (d.norm() / ref.norm()).item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--k", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    print(f"# {args.m}x{args.n}x{args.k}, errors relative to max|C| of the fp32 product of the "
          f"bf16 inputs ({'sk.fp8gemm' if sk else 'reference'})")
    print("| input | per-tensor max | per-tensor Frobenius | MX (floor) max | MX (floor) "
          "Frobenius | MX (ceil) max | MX (ceil) Frobenius |")
    print("|---|---|---|---|---|---|---|")
    for kind in ("gaussian", "sparse outliers, 0.1% x 100", "channel outliers, 8 of K x 50",
                 "channel outliers, 8 of K x 2000"):
        a, b = inputs(kind, args.m, args.n, args.k, gen)
        ref = a.float() @ b.float().t()
        pt = errors(gemm_per_tensor(a, b), ref)
        fl = errors(gemm_mx(a, b, "floor"), ref)
        ce = errors(gemm_mx(a, b, "ceil"), ref)
        print(f"| {kind} | {pt[0]:.2e} | {pt[1]:.2e} | {fl[0]:.2e} | {fl[1]:.2e} | {ce[0]:.2e} "
              f"| {ce[1]:.2e} |")


if __name__ == "__main__":
    main()
