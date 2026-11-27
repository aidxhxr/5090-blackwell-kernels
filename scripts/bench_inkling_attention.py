#!/usr/bin/env python3
"""Inkling 2 Mini's attention core against the same attention without the bias and against
torch, on the GPU box.

Inkling's heads (24 query / 8 K/V, D = 128, d_rel 16, extent 1024), sliding layers (window
1024) and full layers. Prefill at 4K and 16K tokens, B = 1; decode steps (T = 1) at 1K and
32K cached keys, B = 1 and 8. Each row times `sk.inkling_attention` (bias from r and the
bank, per-query sinks), `sk.attention` with the same window and no bias or sinks (the cost
of the bias; also its variant 2, the cp.async tile this kernel extends, and ours with an
8-column bank, which skips the bias work on all but the diagonal tiles), and
`F.scaled_dot_product_attention` with the bias, the mask and the sink as
one explicit float mask (the sink is a zero key and value with its logit in the mask, the
columns padded to a multiple of 8 with -inf; K and V repeated to the query heads), where
that mask fits in free memory. The mask is built once and is not part of the timed SDPA
call. Decode K and V rotate through copies so every launch reads from DRAM, as
bench_attention_window.py does.

    PYTHONPATH=python python scripts/bench_inkling_attention.py [--only=prefill|decode]
        [--out=results/inkling_attention_torch.jsonl]

Appends one JSON object per row to --out and prints the rows. Run it under the GPU lock.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
HQ, HKV, D, DREL, EXTENT = 24, 8, 128, 16, 1024
WINDOWS = (1024, 0)  # sliding, full


def time_ms(fn, warmup=5, iters=30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def pairs(T: int, S: int, W: int) -> int:
    """(query, key) pairs the causal mask (and a window of W keys) leaves; queries last."""
    off = S - T
    return sum(min(i + off + 1, W) if W else i + off + 1 for i in range(T))


def inputs(B, T, S, cap=None, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev, bf = "cuda", torch.bfloat16
    q = torch.randn(B, HQ, T, D, device=dev, dtype=bf, generator=g)
    k = torch.randn(B, HKV, cap or S, D, device=dev, dtype=bf, generator=g)
    v = torch.randn(B, HKV, cap or S, D, device=dev, dtype=bf, generator=g)
    r = torch.randn(B, HQ, T, DREL, device=dev, generator=g)
    r = (r / r.norm(dim=-1, keepdim=True) * 4.0).to(bf)
    proj = (torch.randn(DREL, EXTENT, device=dev, generator=g) * 0.25).to(bf)
    sinks = torch.rand(B, HQ, T, device=dev, generator=g) * 4 - 1
    return q, k, v, r, proj, sinks


def sdpa_mask(r, proj, sinks, T, S, W):
    """The float mask SDPA takes for the whole op: bias + causal/window -inf, then the sink
    column (a zero key), padded with -inf to a multiple of 8 columns. None if it does not fit."""
    B = r.shape[0]
    cols = (S + 1 + 7) // 8 * 8
    need = B * HQ * T * cols * 2 * 3  # the mask plus the temporaries that build it
    if need > torch.cuda.mem_get_info()[0] * 0.6:
        return None
    dist = (torch.arange(T, device="cuda") + S - T)[:, None] - torch.arange(S, device="cuda")
    allowed = (dist >= 0) & ((dist < W) if W else True)
    mask = torch.full((B, HQ, T, cols), float("-inf"), device="cuda", dtype=torch.bfloat16)
    rel = (r.float() @ proj.float()).to(torch.bfloat16)  # [B, HQ, T, extent]
    for i0 in range(0, T, 1024):  # row blocks keep the gather's temporaries small
        i1 = min(T, i0 + 1024)
        d = dist[i0:i1]
        bias = rel[:, :, i0:i1].gather(-1, d.clamp(0, EXTENT - 1).expand(B, HQ, -1, -1))
        bias = bias.masked_fill((d < 0) | (d >= EXTENT), 0.0)
        mask[:, :, i0:i1, :S] = bias.masked_fill(~allowed[i0:i1], float("-inf"))
    mask[:, :, :, S] = sinks.to(torch.bfloat16)
    return mask


def run(sk, add, B, T, S, W, kind):
    # Decode: `copies` views of one cache shifted by the keys a step reads, so the rows each
    # launch reads are its own and come from DRAM (about 512 MB between two reads of a row).
    keys = min(S, W) if W and kind == "decode" else S
    copies = 1
    if kind == "decode":
        copies = max(1, min(64, (512 << 20) // (2 * B * HKV * keys * D * 2) + 1))
    q, k, v, r, proj, sinks = inputs(B, T, S, cap=S + (copies - 1) * keys)
    views = [(k[:, :, c * keys:c * keys + S], v[:, :, c * keys:c * keys + S])
             for c in range(copies)]
    turn = [0]

    def nxt():
        kv = views[turn[0] % copies]
        turn[0] += 1
        return kv

    def ours():
        kk, vv = nxt()
        return sk.inkling_attention(q, kk, vv, r, proj, sinks, window=W)

    # A decode step sees every cached key: causal=True would be the top-left mask of one key.
    causal = kind == "prefill"

    def plain():
        kk, vv = nxt()
        return sk.attention(q, kk, vv, causal=causal, window=W)

    p8 = proj[:, :8].contiguous()

    def ours_extent8():  # the same kernel with almost no bias work: only distances 0..7
        kk, vv = nxt()
        return sk.inkling_attention(q, kk, vv, r, p8, sinks, window=W)

    def plain_v2():  # attention.cu variant 2, the cp.async + mma.sync tile this kernel extends
        kk, vv = nxt()
        return sk.attention(q, kk, vv, causal=causal, window=W, variant=2)

    it = (5, 30) if kind == "prefill" else (10, 100)
    ours_ms = time_ms(ours, *it)
    plain_ms = time_ms(plain, *it)
    extent8_ms = time_ms(ours_extent8, *it)
    v2_ms = time_ms(plain_v2, *it)
    sdpa_ms = err = None
    k0, v0 = views[0]
    mask = sdpa_mask(r, proj, sinks, T, S, W)
    if mask is not None:
        pad = mask.shape[-1] - S
        z = torch.zeros(B, HQ, pad, D, device="cuda", dtype=torch.bfloat16)
        kr = torch.cat([k0.repeat_interleave(HQ // HKV, 1), z], 2)
        vr = torch.cat([v0.repeat_interleave(HQ // HKV, 1), z], 2)

        def sdpa():
            return F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask)

        want = sdpa()
        got = sk.inkling_attention(q, k0, v0, r, proj, sinks, window=W)
        err = (got.float() - want.float()).abs().max().item()
        sdpa_ms = time_ms(sdpa, 2, 10) if kind == "prefill" else time_ms(sdpa, 10, 100)
        del mask, kr, vr, z, want
    torch.cuda.empty_cache()
    row = dict(kind=kind, copies=copies, layer="sliding" if W else "full", B=B, T=T, S=S, window=W,
               ours_ms=ours_ms, no_bias_ms=plain_ms, no_bias_v2_ms=v2_ms,
               ours_extent8_ms=extent8_ms, sdpa_bias_ms=sdpa_ms,
               bias_cost=ours_ms / plain_ms - 1.0, max_err_vs_sdpa=err)
    if kind == "prefill":
        flops = 4.0 * B * HQ * pairs(T, S, W) * D
        row.update(tflops=flops / ours_ms / 1e9, no_bias_tflops=flops / plain_ms / 1e9)
    else:
        gb = (2 * B * HKV * keys * D * 2 + B * HQ * T * (2 * D + DREL) * 2) / 1e9
        row.update(gbps=gb / (ours_ms * 1e-3), no_bias_gbps=gb / (plain_ms * 1e-3))
    add(row)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=("prefill", "decode"), default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "inkling_attention_torch.jsonl"))
    args = ap.parse_args()
    import spark_kernels as sk

    rows = []

    def add(r):
        r = {"kernel": "inkling_attention", "device": torch.cuda.get_device_name(), **r}
        rows.append(r)
        cols = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in r.items() if k not in ("kernel", "device"))
        print(cols, file=sys.stderr)

    if args.only in (None, "prefill"):
        for T in (4096, 16384):
            for W in WINDOWS:
                run(sk, add, 1, T, T, W, "prefill")
    if args.only in (None, "decode"):
        for S in (1024, 32768):
            for B in (1, 8):
                for W in WINDOWS:
                    run(sk, add, B, 1, S, W, "decode")
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
