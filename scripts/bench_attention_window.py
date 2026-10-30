#!/usr/bin/env python3
"""Sliding-window attention against torch, on the GPU box.

Prefill (32 query / 8 K/V heads, D = 128, 4K / 16K / 32K tokens, windows 128 / 1024 / 4096):
`sk.attention(window=W)` against our own full causal kernel, torch's `flex_attention` with a
sliding-window mask_mod (compiled, with the block mask that lets it skip the tiles outside the
window; the bar people use for this) and `F.scaled_dot_product_attention` with the window as
an explicit boolean mask where the mask fits (K and V repeated to the query heads, since the
masked kernel has no GQA). gpt-oss's heads (64 / 8, D = 64, window 128) with its sink logits,
which flex takes as a correction from its log-sum-exp. Decode at 128K cached keys with a
4096-key window: ours against full decode, flex, and SDPA on the window's slice of the cache.
K and V of a decode step rotate through shifted copies so every launch reads its window from
DRAM, as bench_attention does. Paged decode (`sk.paged_decode(window=W)`) against the same
batch without a window: 16 sequences of 32K tokens and the ragged mix of
docs/design/serving.md (one 32K sequence among fifty of 500). The backward (`--only=bwd`,
"attention_bwd_window" rows): `sk.attention_bwd(window=W)` at 8K and 32K tokens with windows
of 128, 512 and 4096 against our own full causal backward, torch's full causal backward
(flash, through SDPA) and `torch.autograd.grad` through the compiled flex_attention with the
window's block mask; gpt-oss's heads with sinks against ours without them (flex's sink
correction is on the output and has no backward of its own).

    PYTHONPATH=python python scripts/bench_attention_window.py
        [--only=prefill|decode|sinks|paged|bwd] [--s=4096,16384,32768]
        [--out=results/attention_window_torch.jsonl]

Appends one JSON object per row to --out and prints the rows. The file is .jsonl, not .json:
scripts/make_results_table.py reads results/*.json as C++ bench rows, and these are not.
Each part fits one hold of the GPU lock (under five minutes) once the compiled flex kernels
are in the inductor cache.
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
WARMUP, ITERS = 5, 30


def time_ms(fn, warmup=WARMUP, iters=ITERS) -> float:
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


def pairs(S_q: int, S_kv: int, W: int) -> int:
    """(query, key) pairs a causal window of W keys aligned to the last key leaves."""
    off = S_kv - S_q
    return sum(min(i + off + 1, W) if W else i + off + 1 for i in range(S_q))


# One mask_mod for every window: the window lives in a tensor, so a new W reuses the compiled
# kernel and only the block mask is rebuilt. The query sits at key position q_idx + OFF.
WIN = None
OFF = None


def sliding(b, h, q_idx, kv_idx):
    pos = q_idx + OFF
    return (kv_idx <= pos) & (pos - kv_idx < WIN)


def flex_setup():
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    def block_mask(**kw):
        # compiled, so the mask is built tile by tile instead of as a dense S x S tensor (eager
        # vmap needs gigabytes at 32K tokens)
        try:
            return create_block_mask(sliding, None, None, _compile=True, device="cuda", **kw)
        except TypeError:
            return create_block_mask(sliding, None, None, device="cuda", **kw)

    return torch.compile(flex_attention, dynamic=False), block_mask


def flex_call(flex, q, k, v, bm, sinks=None):
    """flex_attention, with the sinks as a correction from its log-sum-exp: a sink s scales the
    output by exp(lse) / (exp(lse) + exp(s)) = sigmoid(lse - s)."""
    if sinks is None:
        return flex(q, k, v, block_mask=bm, enable_gqa=k.shape[1] != q.shape[1])
    out, lse = flex(q, k, v, block_mask=bm, enable_gqa=k.shape[1] != q.shape[1], return_lse=True)
    return (out.float() * torch.sigmoid(lse - sinks.view(1, -1, 1))[..., None]).to(out.dtype)


def check(name, got, want, tol=3e-2):
    err = (got.float() - want.float()).abs().max().item()
    if not err <= tol * max(1.0, want.float().abs().max().item()):
        raise SystemExit(f"{name}: max error {err:.3e} against the reference")
    return err


def prefill(sk, add, S, windows, Hq=32, Hkv=8, D=128, sink=False):
    global WIN, OFF
    torch.manual_seed(0)
    q = torch.randn(1, Hq, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    sinks = (torch.rand(Hq, device="cuda") * 3) if sink else None
    flex, block_mask = flex_setup()
    WIN = torch.tensor(S, device="cuda", dtype=torch.int32)
    OFF = torch.tensor(0, device="cuda", dtype=torch.int32)
    full_ms = time_ms(lambda: sk.attention(q, k, v, causal=True, sinks=sinks))
    sdpa_full_ms = time_ms(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True,
                                                                  enable_gqa=True))
    for W in windows:
        WIN.fill_(W)
        bm = block_mask(Q_LEN=S, KV_LEN=S)
        ours = sk.attention(q, k, v, causal=True, window=W, sinks=sinks)
        theirs = flex_call(flex, q, k, v, bm, sinks)
        err = check(f"flex S={S} W={W}", ours, theirs)
        ours_ms = time_ms(lambda w=W: sk.attention(q, k, v, causal=True, window=w, sinks=sinks))
        flex_ms = time_ms(lambda m=bm: flex_call(flex, q, k, v, m, sinks))
        sdpa_ms = None
        if not sink and S <= 32768:
            try:  # the boolean mask is S^2 bytes; K, V repeated to the query heads
                mask = torch.ones(S, S, device="cuda", dtype=torch.bool).tril_()
                mask.triu_(-(W - 1))
                kr, vr = k.repeat_interleave(Hq // Hkv, 1), v.repeat_interleave(Hq // Hkv, 1)

                def sdpa(kr=kr, vr=vr, mask=mask):
                    return F.scaled_dot_product_attention(q, kr, vr, attn_mask=mask)

                check(f"sdpa S={S} W={W}", sdpa(), ours)
                sdpa_ms = time_ms(sdpa, warmup=2, iters=10)
                del mask, kr, vr, sdpa
            except torch.OutOfMemoryError:
                sdpa_ms = None
            torch.cuda.empty_cache()
        flops = 4.0 * Hq * pairs(S, S, W) * D
        add(dict(shape=f"b1_hq{Hq}_hkv{Hkv}_s{S}_d{D}_causal", window=W, sinks=sink,
                 ours_ms=ours_ms, ours_full_ms=full_ms, flex_ms=flex_ms, sdpa_mask_ms=sdpa_ms,
                 sdpa_full_causal_ms=sdpa_full_ms, tflops=flops / ours_ms / 1e9,
                 flex_tflops=flops / flex_ms / 1e9, max_err_vs_flex=err))
        del bm, ours, theirs
        torch.cuda.empty_cache()


def decode(sk, add, S_kv=131072, W=4096, Hq=32, Hkv=8, D=128, sink=False):
    """A decode step against S_kv cached keys. The window's rows of K and V rotate through
    shifted copies of one strided cache, cache[:, :, c W : c W + S_kv] (the layout attention
    reads in place), so every launch reads its own window from DRAM; full decode reads
    537 MB a step, past L2 on its own."""
    global WIN, OFF
    torch.manual_seed(1)
    q = torch.randn(1, Hq, 1, D, device="cuda", dtype=torch.bfloat16)
    win_bytes = 2 * Hkv * W * D * 2
    copies = (256 << 20) // win_bytes + 1
    cap = S_kv + (copies - 1) * W
    kc = torch.randn(1, Hkv, cap, D, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn(1, Hkv, cap, D, device="cuda", dtype=torch.bfloat16)
    views = [(kc[:, :, c * W:c * W + S_kv], vc[:, :, c * W:c * W + S_kv]) for c in range(copies)]
    sinks = (torch.rand(Hq, device="cuda") * 3) if sink else None
    turn = [0]

    def nxt():
        kv = views[turn[0] % copies]
        turn[0] += 1
        return kv

    def ours():
        k, v = nxt()
        return sk.attention(q, k, v, window=W, sinks=sinks)

    def ours_full():
        k, v = views[0]
        return sk.attention(q, k, v, sinks=sinks)

    def sdpa_slice():  # what a torch decode step with a window does: the last W keys
        k, v = nxt()
        return F.scaled_dot_product_attention(q, k[:, :, -W:], v[:, :, -W:], enable_gqa=True)

    flex, block_mask = flex_setup()
    WIN = torch.tensor(W, device="cuda", dtype=torch.int32)
    OFF = torch.tensor(S_kv - 1, device="cuda", dtype=torch.int32)
    bm = block_mask(Q_LEN=1, KV_LEN=S_kv)

    def flex_dec():
        k, v = nxt()
        return flex_call(flex, q, k, v, bm, sinks)

    k0, v0 = views[0]
    err = check("flex decode", sk.attention(q, k0, v0, window=W, sinks=sinks),
                flex_call(flex, q, k0, v0, bm, sinks))
    ours_ms, full_ms, flex_ms = time_ms(ours, 10, 100), time_ms(ours_full, 10, 50), time_ms(
        flex_dec, 10, 100)
    sdpa_ms = time_ms(sdpa_slice, 10, 100) if not sink else None
    gb = (2 * Hq * D * 2 + win_bytes) / 1e9
    add(dict(shape=f"b1_hq{Hq}_hkv{Hkv}_sq1_skv{S_kv}_d{D}", window=W, sinks=sink,
             ours_ms=ours_ms, ours_full_ms=full_ms, flex_ms=flex_ms, sdpa_slice_ms=sdpa_ms,
             gbps=gb / (ours_ms * 1e-3), max_err_vs_flex=err))


def paged(sk, add, lens, W, Hq=32, Hkv=8, D=128, page=16, sink=False):
    """One paged decode step with and without a window. The windows of the batch (16 x 16.8
    MB at 4096 keys) are past L2 without a rotation."""
    torch.manual_seed(2)
    npages = [(n + page - 1) // page for n in lens]
    kc = torch.randn(sum(npages), Hkv, page, D, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    perm = torch.randperm(sum(npages), device="cuda").to(torch.int32)
    bt = torch.zeros(len(lens), max(npages), dtype=torch.int32, device="cuda")
    i = 0
    for b, n in enumerate(npages):
        bt[b, :n] = perm[i:i + n]
        i += n
    sl = torch.tensor(lens, dtype=torch.int32, device="cuda")
    q = torch.randn(len(lens), Hq, D, device="cuda", dtype=torch.bfloat16)
    sinks = (torch.rand(Hq, device="cuda") * 3) if sink else None
    ours_ms = time_ms(lambda: sk.paged_decode(q, kc, vc, bt, sl, window=W, sinks=sinks), 10, 100)
    full_ms = time_ms(lambda: sk.paged_decode(q, kc, vc, bt, sl, sinks=sinks), 10, 100)
    keys = sum(min(n, W) for n in lens)
    gb = (2 * len(lens) * Hq * D * 2 + 2 * Hkv * keys * D * 2) / 1e9
    tag = f"n{len(lens)}_max{max(lens)}_sum{sum(lens)}_hq{Hq}_hkv{Hkv}_d{D}_p{page}"
    add(dict(shape=f"paged_{tag}", window=W, sinks=sink, ours_ms=ours_ms, ours_full_ms=full_ms,
             window_keys=keys, gbps=gb / (ours_ms * 1e-3)))


def bwd(sk, add, S, windows, Hq=32, Hkv=8, D=128, sink=False):
    """The backward alone, from the saved (out, lse): ours with the window against our own full
    causal backward, torch's full causal backward (autograd.grad of the SDPA output, flash on
    this card) and autograd.grad through flex_attention compiled with the window's block mask,
    what loss.backward() runs with flex. The full rows at 32K tokens take 5 iterations (10 s
    each). With sinks, ours only, against ours without them on the same window."""
    global WIN, OFF
    import torch._functorch.config

    # the compiled backward is run repeatedly with retain_graph, which the donated buffers of
    # the default compile refuse
    torch._functorch.config.donated_buffer = False
    torch.manual_seed(0)
    q = torch.randn(1, Hq, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(1, Hq, S, D, device="cuda", dtype=torch.bfloat16)
    sinks = (torch.rand(Hq, device="cuda") * 3) if sink else None
    flex, block_mask = flex_setup()
    WIN = torch.tensor(S, device="cuda", dtype=torch.int32)
    OFF = torch.tensor(0, device="cuda", dtype=torch.int32)
    full_iters, full_warmup = (5, 2) if S > 16384 else (ITERS, WARMUP)
    out_f, lse_f = sk.attention_fwd(q, k, v, causal=True, sinks=sinks)
    full_ms = time_ms(
        lambda: sk.attention_bwd(q, k, v, out_f, do, lse_f, causal=True, sinks=sinks),
        full_warmup, full_iters)
    qr, kr, vr = (t.clone().requires_grad_() for t in (q, k, v))
    flash_full_ms = None
    if not sink:
        o = F.scaled_dot_product_attention(qr, kr, vr, is_causal=True, enable_gqa=True)
        flash_full_ms = time_ms(lambda: torch.autograd.grad(o, (qr, kr, vr), do, retain_graph=True),
                                full_warmup, full_iters)
        del o
    for W in windows:
        WIN.fill_(W)
        out_w, lse_w = sk.attention_fwd(q, k, v, causal=True, window=W, sinks=sinks)

        def ours(W=W, out=out_w, lse=lse_w):
            return sk.attention_bwd(q, k, v, out, do, lse, causal=True, window=W, sinks=sinks)

        flex_ms = err = None
        nosink_ms = None
        if sink:
            out0, lse0 = sk.attention_fwd(q, k, v, causal=True, window=W)
            nosink_ms = time_ms(lambda w=W, o=out0, ls=lse0: sk.attention_bwd(
                q, k, v, o, do, ls, causal=True, window=w))
            del out0, lse0
        else:
            bm = block_mask(Q_LEN=S, KV_LEN=S)
            o = flex(qr, kr, vr, block_mask=bm, enable_gqa=True)

            def theirs(o=o):
                return torch.autograd.grad(o, (qr, kr, vr), do, retain_graph=True)

            err = max(check(f"flex bwd S={S} W={W} {n}", g, r) for n, g, r in zip(
                ("dq", "dk", "dv"), ours(), theirs(), strict=True))
            flex_ms = time_ms(theirs)
            del o, bm
        ours_ms = time_ms(ours)
        flops = 10.0 * Hq * pairs(S, S, W) * D
        add(dict(kernel="attention_bwd_window", shape=f"b1_hq{Hq}_hkv{Hkv}_s{S}_d{D}_causal",
                 window=W, sinks=sink, ours_ms=ours_ms, ours_full_ms=full_ms, flex_ms=flex_ms,
                 flash_full_causal_ms=flash_full_ms, ours_nosink_ms=nosink_ms,
                 tflops=flops / ours_ms / 1e9,
                 flex_tflops=flops / flex_ms / 1e9 if flex_ms else None, max_err_vs_flex=err))
        del out_w, lse_w
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--only", choices=("prefill", "decode", "sinks", "paged", "bwd"), default=None)
    ap.add_argument("--s", default="4096,16384,32768", help="prefill lengths")
    ap.add_argument("--out", default=str(ROOT / "results" / "attention_window_torch.jsonl"))
    args = ap.parse_args()
    import spark_kernels as sk

    rows = []

    def add(r):
        r = {"kernel": "attention_window", "device": torch.cuda.get_device_name(), **r}
        rows.append(r)
        cols = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                        for k, v in r.items() if k not in ("kernel", "device"))
        print(cols, file=sys.stderr)

    if args.only in (None, "prefill"):
        for S in (int(x) for x in args.s.split(",")):
            prefill(sk, add, S, (128, 1024, 4096))
    if args.only in (None, "sinks"):
        for S in (4096, 16384):
            prefill(sk, add, S, (128,), Hq=64, Hkv=8, D=64, sink=True)
    if args.only in (None, "decode"):
        decode(sk, add)
        decode(sk, add, W=128, Hq=64, Hkv=8, D=64, sink=True)
    if args.only in (None, "paged"):
        for W in (4096, 128):
            paged(sk, add, [32768] * 16, W)
        paged(sk, add, [32768] + [500] * 50, 4096)
        paged(sk, add, [32768] * 16, 128, Hq=64, Hkv=8, D=64, sink=True)
    if args.only in (None, "bwd"):
        for S in (8192, 32768):
            bwd(sk, add, S, (128, 512, 4096))
        bwd(sk, add, 8192, (128,), Hq=64, Hkv=8, D=64, sink=True)
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
