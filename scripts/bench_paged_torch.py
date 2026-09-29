#!/usr/bin/env python3
"""The paged attention ops against PyTorch on the same paged cache (GQA 32/8, D = 128,
16-token pages handed out shuffled), timed as a loop of back-to-back calls between CUDA
events after a clock ramp, median of five rounds:

    decode   ours: paged_decode
             torch, gathered: each sequence's pages gathered (one copy) to the batch's longest
                 length, then F.scaled_dot_product_attention with a length mask, the four
                 query heads of a K/V head passed as four query rows (memory-efficient kernel)
             torch, gathered, enable_gqa: the same gather, SDPA with enable_gqa and the mask
             torch, contiguous: SDPA(enable_gqa) over caches already contiguous, no gather,
                 no mask; only for equal lengths (what a server with a contiguous cache per
                 sequence would run)
    prefill  ours: attention_varlen over the packed prompts
             torch: SDPA(is_causal, enable_gqa) once per prompt over contiguous K/V

    python scripts/bench_paged_torch.py [--iters 20]

GB/s counts K and V once per K/V head, as bench_paged does; TFLOPS the causal-halved count.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time

import torch
import torch.nn.functional as F

HQ, HKV, D, PAGE = 32, 8, 128, 16


def time_ms(fn, iters: int, rounds: int = 5) -> float:
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 0.3:  # clock ramp
        fn()
        torch.cuda.synchronize()
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    out = []
    for _ in range(rounds):
        a.record()
        for _ in range(iters):
            fn()
        b.record()
        b.synchronize()
        out.append(a.elapsed_time(b) / iters)
    return statistics.median(out)


def paged(lens, seed=0):
    g = torch.Generator().manual_seed(seed)
    npg = [math.ceil(n / PAGE) for n in lens]
    total = sum(npg) + 1
    perm = torch.randperm(total, generator=g)
    kc = torch.randn(total, HKV, PAGE, D, device="cuda", dtype=torch.bfloat16)
    vc = torch.randn_like(kc)
    bt = torch.zeros(len(lens), max(npg), dtype=torch.int32)
    i = 0
    for b, n in enumerate(npg):
        bt[b, :n] = perm[i:i + n]
        i += n
    return kc, vc, bt.cuda(), torch.tensor(lens, dtype=torch.int32, device="cuda")


def parse(s: str) -> list[int]:
    out = []
    for tok in s.split(","):
        n, _, r = tok.partition("x")
        out += [int(n)] * int(r or 1)
    return out


def uniform(n, lo, hi, seed):
    g = torch.Generator().manual_seed(seed)
    return [int(x) for x in torch.randint(lo, hi + 1, (n,), generator=g)]


def decode_row(sk, lens, iters):
    B = len(lens)
    kc, vc, bt, sl = paged(lens)
    q = torch.randn(B, HQ, D, device="cuda", dtype=torch.bfloat16)
    npg = math.ceil(max(lens) / PAGE)
    table = bt[:, :npg].long()
    keep = (torch.arange(npg * PAGE, device="cuda")[None] < sl[:, None].long())[:, None, None]

    def gathered():
        kk = kc.transpose(0, 1)[:, table].view(HKV, B, npg * PAGE, D).transpose(0, 1)
        vv = vc.transpose(0, 1)[:, table].view(HKV, B, npg * PAGE, D).transpose(0, 1)
        return F.scaled_dot_product_attention(q.view(B, HKV, HQ // HKV, D), kk, vv,
                                              attn_mask=keep)

    def gathered_gqa():
        kk = kc[table].permute(0, 2, 1, 3, 4).reshape(B, HKV, npg * PAGE, D)
        vv = vc[table].permute(0, 2, 1, 3, 4).reshape(B, HKV, npg * PAGE, D)
        return F.scaled_dot_product_attention(q[:, :, None], kk, vv, attn_mask=keep,
                                              enable_gqa=True)

    ours = time_ms(lambda: sk.paged_decode(q, kc, vc, bt, sl), iters)
    t_g = time_ms(gathered, iters)
    t_gqa = time_ms(gathered_gqa, max(1, iters // 4))
    t_c = None
    if min(lens) == max(lens):
        kk = kc[table].permute(0, 2, 1, 3, 4).reshape(B, HKV, npg * PAGE, D)[:, :, :lens[0]]
        vv = vc[table].permute(0, 2, 1, 3, 4).reshape(B, HKV, npg * PAGE, D)[:, :, :lens[0]]
        kk, vv = kk.contiguous(), vv.contiguous()
        t_c = time_ms(lambda: F.scaled_dot_product_attention(q[:, :, None], kk, vv,
                                                             enable_gqa=True), iters)
    gb = 2 * sum(lens) * HKV * D * 2 / 1e9
    name = f"{len(lens)} x {lens[0]}" if min(lens) == max(lens) else \
        f"{len(lens)} seqs, max {max(lens)}, sum {sum(lens)}"
    c = f"{t_c * 1e3:8.1f}" if t_c else "       -"
    print(f"| {name} | {gb * 1e3:.0f} | {ours * 1e3:.1f} | {gb / ours * 1e3:.0f} | "
          f"{t_g * 1e3:.1f} | {t_gqa * 1e3:.1f} | {c.strip()} | {t_g / ours:.2f}x |")


def prefill_row(sk, lens, iters):
    kc, vc, bt, sl = paged(lens, seed=1)
    T = sum(lens)
    q = torch.randn(T, HQ, D, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0] + list(torch.tensor(lens).cumsum(0)), dtype=torch.int32, device="cuda")
    ks = [torch.randn(1, HKV, n, D, device="cuda", dtype=torch.bfloat16) for n in lens]
    qs = [torch.randn(1, HQ, n, D, device="cuda", dtype=torch.bfloat16) for n in lens]

    def torch_loop():
        for qq, kk in zip(qs, ks, strict=True):
            F.scaled_dot_product_attention(qq, kk, kk, is_causal=True, enable_gqa=True)

    ours = time_ms(lambda: sk.attention_varlen(q, kc, vc, cu, sl, bt), iters)
    t = time_ms(torch_loop, iters)
    fl = sum(4 * HQ * n * n * D / 2 for n in lens)
    name = f"{len(lens)} x {lens[0]}" if min(lens) == max(lens) else \
        f"{len(lens)} prompts, max {max(lens)}, sum {T}"
    print(f"| {name} | {ours * 1e3:.1f} | {fl / ours / 1e9:.0f} | {t * 1e3:.1f} | "
          f"{fl / t / 1e9:.0f} | {t / ours:.2f}x |")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    import spark_kernels as sk

    print("| decode batch | K+V MB | ours us | GB/s | torch gathered us | gathered + enable_gqa "
          "us | torch contiguous us | ours vs gathered |")
    print("|---|---|---|---|---|---|---|---|")
    for lens in (parse("4096"), parse("32768"), parse("4096x8"), parse("2048x32"),
                 parse("1024x64"), parse("32768,500x50"), uniform(64, 1, 8192, 3),
                 uniform(32, 100, 2000, 4)):
        decode_row(sk, lens, args.iters)
    print()
    print("| prefill batch | ours us | TFLOPS | torch per prompt us | TFLOPS | ours vs torch |")
    print("|---|---|---|---|---|---|")
    for lens in (parse("4096"), parse("2048x4"), [3000, 1500, 700, 300, 200, 100, 50, 20],
                 uniform(16, 64, 1024, 5)):
        prefill_row(sk, lens, args.iters)
    return 0


if __name__ == "__main__":
    sys.exit(main())
