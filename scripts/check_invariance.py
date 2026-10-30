#!/usr/bin/env python3
"""Which ops are batch invariant: each op run on the same rows alone and in batches of 1, 8,
32 (and --prefill rows for the GEMMs), the common rows compared bit for bit.

    python scripts/check_invariance.py              # every op, every variant, one table
    python scripts/check_invariance.py --ops fp8gemm,w4gemm --ms 1,8,32,256

A row of the table is one (op, variant, shape). The op runs on the first M rows of one fixed
input for every M in --ms (the GEMMs and the row ops), or on the first B sequences of one
fixed set for every B in --batches (the paged attention ops, with the sequences' pages
shuffled so a batch's keys are never contiguous), and the output rows two runs have in common
are compared with torch.equal. "same" means every pair of batch sizes agreed on their common
rows; otherwise the first disagreeing pair is printed. A variant that refuses a shape or a
format is printed as refused. Nothing here is timed.

The quantizers are included because a scale taken over the whole activation
(fp8_quantize "tensor", fp4_quantize's per-tensor scale) depends on every row in the batch,
so the quantized bytes of a row, and everything after them, move with the batch even when
the GEMM does not. The last row of the GEMM input is an outlier (64x the others, as the
residual stream of a Llama layer has), so the largest batch carries a scale the small ones
do not.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

H_Q, H_KV, D, PAGE = 32, 8, 128, 16
# (name, K, N): a square projection, a narrow one (fp8gemm's decode route splits K when a
# narrow N leaves few strips, w4gemm's when the weight is small), and the long-K down GEMM.
GEMM_SHAPES = [("o", 4096, 4096), ("narrow", 4096, 1024), ("down", 14336, 4096)]


def _check(rows, name, variant, shape, fn, sizes, *args):
    """Runs fn(n, *args) for every n in sizes and compares every pair on their common rows."""
    try:
        outs = {n: fn(n, *args) for n in sizes}
    except (ValueError, RuntimeError) as e:
        print(f"{name:18s} v{variant!s:<6} {shape:16s} refused: {str(e).splitlines()[0]}",
              file=sys.stderr)
        rows.append((name, variant, shape, "refused"))
        return
    ns = sorted(outs)
    bad = None
    for i, a in enumerate(ns):
        for b in ns[i + 1:]:
            n = min(a, b)
            if not torch.equal(outs[a][:n], outs[b][:n]):
                bad = (a, b)
                break
        if bad:
            break
    rows.append((name, variant, shape, bad))
    verdict = "same" if bad is None else f"DIFF {bad[0]} vs {bad[1]}"
    print(f"{name:18s} v{variant!s:<6} {shape:16s} {verdict}", file=sys.stderr)


def _gemm_shape(sk, rows, ms, which, name, k, n):
    from spark_kernels import reference as ref

    M = max(ms)
    g = torch.Generator(device="cuda").manual_seed(k + n)
    a = torch.randn(M, k, device="cuda", generator=g).to(torch.bfloat16)
    a[M - 1] *= 64  # one outlier row, as a residual stream has: a per-tensor scale sees it
    shape = f"{name} {k}x{n}"
    if "hgemm" in which:
        b = (torch.randn(k, n, device="cuda", generator=g) / k ** 0.5).to(torch.bfloat16)
        for v in range(3, sk.num_variants("hgemm")):  # the rungs that take any M
            _check(rows, "hgemm", v, shape, lambda m, v: sk.hgemm(a[:m], b, v), ms, v)
        del b
    if "fp8gemm" in which:
        bt = (torch.randn(n, k, device="cuda", generator=g) / k ** 0.5).to(torch.float8_e4m3fn)
        one = torch.ones(1, device="cuda")
        # the activation quantizers: per tensor, per row, MX
        for mode in ("tensor", "row", "mx"):
            _check(rows, "fp8_quantize", mode, shape,
                   lambda m, mode: sk.fp8_quantize(a[:m], mode)[0], ms, mode)
        a8 = sk.fp8_quantize(a, "row")[0]  # fixed bytes: what the GEMM alone does
        for v in range(1, sk.num_variants("fp8gemm")):
            _check(rows, "fp8gemm", v, shape, lambda m, v: sk.fp8gemm(a8[:m], bt, one, one, v),
                   ms, v)
        if k % 256 == 0:  # MX mode (refused on a build without the block-scaled mma)
            am, sfa = sk.fp8_quantize(a, "mx")
            btm, sfb = ref.quantize_mx(bt.to(torch.bfloat16))  # [N, K] and [N, K / 32]
            for v in range(1, sk.num_variants("fp8gemm")):
                _check(rows, "fp8gemm mx", v, shape,
                       lambda m, v: sk.fp8gemm(am[:m], btm, sfa=sfa[:m], sfb=sfb, variant=v),
                       ms, v)
        del bt
    if "w4gemm" in which:
        w = (torch.randn(k, n, device="cuda", generator=g) / k ** 0.5).to(torch.bfloat16)
        for asym in (False, True):
            ww = sk.W4Weight.quantize(w, asym)
            for v in range(2, sk.num_variants("w4gemm")):
                _check(rows, "w4gemm" + (" asym" if asym else ""), v, shape,
                       lambda m, v, ww: sk.w4gemm(a[:m], ww, variant=v), ms, v, ww)
        del w
    if "fp4gemm" in which and k % 256 == 0:
        for fmt in ("nvfp4", "mxfp4"):
            try:
                wt, sfb, sw = sk.fp4_quantize(a.new_empty(n, k).normal_(), fmt)
            except (ValueError, RuntimeError) as e:  # a build without the instruction
                print(f"fp4 skipped: {str(e).splitlines()[0]}", file=sys.stderr)
                break
            _check(rows, "fp4_quantize", fmt, shape,
                   lambda m, fmt: sk.fp4_quantize(a[:m], fmt)[0], ms, fmt)
            q, sfa, s = sk.fp4_quantize(a, fmt)
            for v in range(1, sk.num_variants("fp4gemm")):
                _check(rows, f"fp4gemm {fmt}", v, shape,
                       lambda m, v, *t: sk.fp4gemm(t[0][:m], t[1], t[2][:m], t[3], t[4], t[5],
                                                   fmt=t[6], variant=v),
                       ms, v, q, wt, sfa, sfb, s, sw, fmt)
    torch.cuda.empty_cache()


def check_gemms(sk, rows, ms, which):
    for name, k, n in GEMM_SHAPES:
        _gemm_shape(sk, rows, ms, which, name, k, n)


def check_row_ops(sk, rows, ms):
    M = max(ms)
    g = torch.Generator(device="cuda").manual_seed(7)
    x = torch.randn(M, 4096, device="cuda", generator=g).to(torch.bfloat16)
    w = torch.randn(4096, device="cuda", generator=g).to(torch.bfloat16)
    d = torch.randn(M, 4096, device="cuda", generator=g).to(torch.bfloat16)
    _check(rows, "rmsnorm", -1, "4096", lambda m: sk.rmsnorm(x[:m], w), ms)

    def add_norm(m):
        xm = x[:m].clone()
        return sk.add_rmsnorm_(d[:m], xm, w)

    _check(rows, "add_rmsnorm_", -1, "4096", add_norm, ms)
    _check(rows, "swiglu", -1, "2x4096", lambda m: sk.swiglu(x[:m], d[:m]), ms)


def _cache(lens, fp8, seed=0):
    from spark_kernels.ops import quantize_kv_cache

    g = torch.Generator(device="cpu").manual_seed(seed)
    npg = [(n + PAGE - 1) // PAGE for n in lens]
    total = sum(npg) + 2
    perm = torch.randperm(total, generator=g).tolist()
    k = torch.randn(total, H_KV, PAGE, D, generator=g).to("cuda", torch.bfloat16)
    v = torch.randn(total, H_KV, PAGE, D, generator=g).to("cuda", torch.bfloat16)
    bt = torch.zeros(len(lens), max(npg), dtype=torch.int32)
    i = 0
    for b, n in enumerate(npg):
        bt[b, :n] = torch.tensor(perm[i:i + n], dtype=torch.int32)
        i += n
    sc = {}
    if fp8:
        ks = torch.tensor([4.0 * (1 + h / 4) / 448 for h in range(H_KV)], device="cuda")
        vs = torch.tensor([3.0 * (1 + h / 4) / 448 for h in range(H_KV)], device="cuda")
        k, v = quantize_kv_cache(k, ks), quantize_kv_cache(v, vs)
        sc = dict(k_scale=ks, v_scale=vs)
    return k, v, bt.cuda(), torch.tensor(lens, dtype=torch.int32, device="cuda"), sc


def check_attention(sk, rows, batches):
    g = torch.Generator(device="cpu").manual_seed(3)
    B = max(batches)
    lens = [int(x) for x in torch.randint(1, 4100, (B,), generator=g)]
    lens[0] = 4100  # the row under test spans several warps of the length-aware split
    q = torch.randn(B, H_Q, D, generator=g).to("cuda", torch.bfloat16)
    for fp8 in (False, True):
        k, v, bt, sl, sc = _cache(lens, fp8)
        fmt = "fp8 cache" if fp8 else "bf16 cache"
        def dec(b, var, k, v, bt, sl, sc):
            return sk.paged_decode(q[:b].contiguous(), k, v, bt[:b].contiguous(),
                                   sl[:b].contiguous(), var, **sc)

        for var in range(1, sk.num_variants("paged_decode")):
            _check(rows, "paged_decode", var, fmt, dec, batches, var, k, v, bt, sl, sc)
        # a 300-token prompt after a 1000-key context, alone and packed before other prompts
        qlens = [300] + [int(x) for x in torch.randint(1, 500, (B - 1,), generator=g)]
        ctx = [1000] + [int(x) for x in torch.randint(0, 2000, (B - 1,), generator=g)]
        tot = [a + c for a, c in zip(qlens, ctx, strict=True)]
        k, v, bt, sl, sc = _cache(tot, fp8, seed=1)
        qp = torch.randn(sum(qlens), H_Q, D, generator=g).to("cuda", torch.bfloat16)
        def pre(b, var, k, v, bt, sl, sc, qp, qlens):
            T = sum(qlens[:b])
            cu = torch.tensor([0] + qlens[:b], dtype=torch.int32).cumsum(
                0, dtype=torch.int32).cuda()
            o = sk.attention_varlen(qp[:T].contiguous(), k, v, cu, sl[:b].contiguous(),
                                    bt[:b].contiguous(), True, var, **sc)
            return o[:qlens[0]]  # the first prompt's rows

        for var in range(1, sk.num_variants("attention_varlen")):
            _check(rows, "attention_varlen", var, fmt, pre, batches, var, k, v, bt, sl, sc, qp,
                   qlens)
        del k, v
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ops", default="hgemm,fp8gemm,w4gemm,fp4gemm,rows,attention")
    ap.add_argument("--ms", default="1,8,32,256", help="GEMM row counts")
    ap.add_argument("--batches", default="1,8,32", help="attention batch sizes")
    args = ap.parse_args()
    import spark_kernels as sk

    which = set(args.ops.split(","))
    ms = [int(x) for x in args.ms.split(",")]
    batches = [int(x) for x in args.batches.split(",")]
    rows: list[tuple] = []
    check_gemms(sk, rows, ms, which)
    if "rows" in which:
        check_row_ops(sk, rows, ms)
    if "attention" in which:
        check_attention(sk, rows, batches)
    print(f"\n| op | variant | shape | rows M in {ms} / sequences B in {batches} |")
    print("|---|---|---|---|")
    for name, variant, shape, bad in rows:
        if bad is None:
            verdict = "same bits"
        elif bad == "refused":
            verdict = "refused"
        else:
            verdict = f"differ ({bad[0]} vs {bad[1]})"
        print(f"| {name} | {variant} | {shape} | {verdict} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
