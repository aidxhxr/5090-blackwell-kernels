#!/usr/bin/env python3
"""Time Inkling 2 Mini's block attention residual (sk.attn_res) and loop message gates
(sk.message_gates) against the HF modeling code's torch ops, at the model's shapes: H = 2304,
nb = 1..4 block states, nq = 1, 3, 4 (and 48, a loop pass's twelve message inputs in one
call), rank-64 gates one at a time (the MLP gate) and three at a time (q, k, v).

    flock ~/sk26/gpu.lock python scripts/bench_attn_res.py
    python scripts/bench_attn_res.py --tokens 4096 --append   # a second run into the same file

Each op runs `calls` times inside one CUDA graph, cycling through enough copies of its inputs
and weights that each call reads cold memory (about three L2s' worth in all, capped), and the
replay is timed with CUDA events; the time is per call. `torch` is the HF forward's own torch
ops replayed the same way, so it carries no Python overhead: a lower bound on eager. GB/s is
the op's unique bytes (inputs, outputs, weights once) over its time, next to a measured
device-to-device copy. Then the per-token total: what one decoded token spends in these ops
over the 48 layer applications, from the T = 1 rows. Writes results/attn_res.json.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

OUT = ROOT / "results" / "attn_res.json"
H = 2304
RANK = 64
EPS = 1e-6


# ---- the HF forward's math (transformers 5.24.1 modeling_inkling2_loop.py) ----------------
def hf_norm(x):
    dt = x.dtype
    x = x.float()
    return (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)).to(dt)


def hf_attn_res(block_states, partial, query):
    stacked = hf_norm(torch.cat([block_states, partial.unsqueeze(-2)], dim=-2).float())
    q = query.float().view(-1, H).expand(*stacked.shape[:-2], -1, -1)
    w = F.softmax(torch.matmul(q, stacked.transpose(-1, -2)) * H**-0.5, dim=-1,
                  dtype=torch.float32)
    return torch.matmul(w, stacked).unbind(-2)


def hf_gate(h, m, wd, wf, wg):
    h, m = h.float(), m.float()
    z = F.linear(torch.cat([hf_norm(h), hf_norm(m)], dim=-1).to(wd.dtype), wd).float()
    return (1 - torch.tanh(F.linear(z, wf))) * h + torch.tanh(F.linear(z, wg)) * m


# ---- timing ---------------------------------------------------------------------------------
def graph_time_us(fns, calls: int, reps: int) -> float:
    """Median per-call microseconds of `calls` calls cycling through `fns`, from a graph."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for f in fns[:3]:
            f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(calls):
            fns[i % len(fns)]()
    g.replay()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / calls)
    del g
    torch.cuda.empty_cache()
    return statistics.median(times)


def copy_gbps() -> float:
    n = 1 << 30
    src = torch.empty(n, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(3):
        dst.copy_(src)
    torch.cuda.synchronize()
    times = []
    for _ in range(10):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        dst.copy_(src)
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b))
    return 2 * n / (statistics.median(times) * 1e-3) / 1e9


def copies_for(bytes_per_call: int, cap: int) -> int:
    l2 = torch.cuda.get_device_properties(0).L2_cache_size
    return max(1, min(cap, math.ceil(3 * l2 / max(bytes_per_call, 1))))


# ---- the ops --------------------------------------------------------------------------------
def bench_attn_res(sk, T, nb, nq, pdt, calls, reps):
    psize = torch.tensor([], dtype=pdt).element_size()
    nbytes = T * (nb * H * 2 + H * psize + nq * H * 4) + nq * H * 4
    n = copies_for(nbytes, 64)
    sets = []
    for _ in range(n):
        blocks = torch.randn(T, nb, H, device="cuda").to(torch.bfloat16)
        partial = torch.randn(T, H, device="cuda").to(pdt)
        query = torch.randn(nq, H, device="cuda")
        sets.append((blocks, partial, query))
    ours = graph_time_us([lambda s=s: sk.attn_res(*s) for s in sets], calls, reps)
    ref = graph_time_us([lambda s=s: hf_attn_res(*s) for s in sets], calls, reps)
    return ours, ref, nbytes


def bench_gates(sk, T, G, calls, reps, path=None, up_dtype=torch.bfloat16):
    """G gates at once; ours reads wf, wg in `up_dtype` (bf16 as the checkpoint stores them,
    or fp32 as HF holds them), torch in fp32 as HF does."""
    shared_h = G == 3  # a sliding loop layer: q, k, v gates read one h
    usize = 2 if up_dtype == torch.bfloat16 else 4
    weight_bytes = G * (RANK * 2 * H * 2 + 2 * H * RANK * usize)
    act_bytes = T * H * 4 * ((1 if shared_h else G) + 2 * G)
    nbytes = weight_bytes + act_bytes
    n = copies_for(nbytes, 160)
    sets = []
    for _ in range(n):
        h = [torch.randn(T, H, device="cuda")] * G
        m = [torch.randn(T, H, device="cuda") for _ in range(G)]
        wd = [(torch.randn(RANK, 2 * H, device="cuda") / 68).to(torch.bfloat16) for _ in range(G)]
        wf = [torch.randn(H, RANK, device="cuda") / 8 for _ in range(G)]
        wg = [torch.randn(H, RANK, device="cuda") / 8 for _ in range(G)]
        wdt = [w.t().contiguous() for w in wd]
        sets.append((h, m, wd, wf, wg, wdt, [w.to(up_dtype) for w in wf],
                     [w.to(up_dtype) for w in wg]))
    ours = graph_time_us(
        [lambda s=s: sk.message_gates(s[0], s[1], s[2], s[6], s[7], path=path, wd_t=s[5])
         for s in sets], calls, reps)
    ref = graph_time_us(
        [lambda s=s: [hf_gate(s[0][g], s[1][g], s[2][g], s[3][g], s[4][g]) for g in range(G)]
         for s in sets], calls, reps)
    return ours, ref, nbytes


# ---- what one decoded token runs --------------------------------------------------------------
def decode_schedule():
    """(op, nb, nq, partial dtype) for every attention residual of one token's forward, and
    the gate launches, from the HF forward: blocks of 6 layers, layers 0-5 once, 6-17 three
    times, 18-23 once; the block states grow at layers 0, 6, 12 and 18."""
    full = {5, 11, 17, 23}
    order = list(range(6)) + list(range(6, 18)) * 3 + list(range(18, 24))
    res, gates = [], []
    nb = 0
    loop_pass = 0
    for layer in order:
        if layer == 6:
            loop_pass += 1
            nb = 1  # every pass restarts from the prelude's block states
        if layer > 0:  # layer 0 has no attention_input
            res.append(("attn_in", nb, 3 if layer in full else 1, torch.bfloat16))
        if layer % 6 == 0:
            nb += 1  # the layer's input joins the block states before mlp_input
        res.append(("mlp_in", nb, 1, torch.float32))
        if 6 <= layer < 18 and loop_pass > 1:
            res.append(("message_in", 1, 4, torch.bfloat16))
            gates += [3, 1]  # the q, k, v gates in one launch, then the MLP gate
    res.append(("final_in", nb, 1, torch.bfloat16))
    return res, gates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 64, 4096])
    ap.add_argument("--calls", type=int, default=64)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--append", action="store_true", help="add to the rows already in --out")
    args = ap.parse_args()

    import spark_kernels as sk

    device = torch.cuda.get_device_name()
    peak = copy_gbps()
    print(f"{device}: device-to-device copy {peak:.0f} GB/s")
    res, gates = decode_schedule()
    configs = sorted({(nb, nq, dt) for _, nb, nq, dt in res}, key=lambda c: (c[1], c[0], str(c[2])))
    configs.append((1, 48, torch.bfloat16))  # one call for a loop pass's 12 message inputs
    rows = []
    hdr = f"{'op':<11} {'nb':>2} {'nq':>2} {'partial':>7} {'T':>5} {'ours us':>9} {'torch us':>9}"
    print(hdr + f" {'speedup':>7} {'GB/s':>6} {'% copy':>6}")
    for T in args.tokens:
        for nb, nq, dt in configs:
            if nq == 48 and T > 64:
                continue  # a decode trick: at 4096 tokens the 48 outputs alone are 1.8 GB
            calls = args.calls if T <= 64 else 8
            ours, ref, nbytes = bench_attn_res(sk, T, nb, nq, dt, calls, args.reps)
            gbps = nbytes / (ours * 1e-6) / 1e9
            name = "bf16" if dt == torch.bfloat16 else "f32"
            rows.append({"op": "attn_res", "nb": nb, "nq": nq, "partial": name, "T": T,
                         "ours_us": ours, "torch_us": ref, "bytes": nbytes, "gbps": gbps})
            print(f"{'attn_res':<11} {nb:>2} {nq:>2} {name:>7} {T:>5} {ours:9.2f} {ref:9.2f}"
                  f" {ref / ours:6.2f}x {gbps:6.0f} {100 * gbps / peak:5.0f}%")
        for G in (1, 3):
            paths = ["decode", "prefill"] if T <= 64 else ["prefill"]
            for path in paths:
                for udt, uname in ((torch.bfloat16, "bf16"), (torch.float32, "f32")):
                    calls = args.calls if T <= 64 else 8
                    ours, ref, nbytes = bench_gates(sk, T, G, calls, args.reps, path, udt)
                    gbps = nbytes / (ours * 1e-6) / 1e9
                    rows.append({"op": f"gate x{G}", "path": path, "up": uname, "T": T,
                                 "ours_us": ours, "torch_us": ref, "bytes": nbytes,
                                 "gbps": gbps})
                    print(f"{'gate x' + str(G):<11} {path:>8} {uname:>4} {T:>5} {ours:9.2f}"
                          f" {ref:9.2f} {ref / ours:6.2f}x {gbps:6.0f} {100 * gbps / peak:5.0f}%")

    # Per decoded token, from the measured rows (one token, or the batch's time / batch).
    summary = {}
    for T in [t for t in args.tokens if t <= 64]:
        def look(op, T=T, **k):
            for r in rows:
                if r["op"] == op and r["T"] == T and all(r.get(a) == b for a, b in k.items()):
                    return r
            raise KeyError((op, T, k))
        tot = {"ours": 0.0, "torch": 0.0}
        n_res = n_msg = 0
        for kind, nb, nq, dt in res:
            r = look("attn_res", nb=nb, nq=nq, partial="bf16" if dt == torch.bfloat16 else "f32")
            tot["ours"] += r["ours_us"]
            tot["torch"] += r["torch_us"]
            n_msg += kind == "message_in"
            n_res += kind != "message_in"
        fused_msg = look("attn_res", nb=1, nq=48, partial="bf16")
        msg = look("attn_res", nb=1, nq=4, partial="bf16")
        ours_fused = tot["ours"] - n_msg * msg["ours_us"] + 2 * fused_msg["ours_us"]
        path = "decode" if T <= 16 else "prefill"  # message_gates' default
        g_ours = sum(look(f"gate x{G}", path=path, up="bf16")["ours_us"] for G in gates)
        # torch runs every gate on its own; the x3 row already holds three of them
        g_torch = sum(look(f"gate x{G}", path=path, up="bf16")["torch_us"] for G in gates)
        s = {"tokens": T, "attn_res_ops": n_res, "message_input_ops": n_msg,
             "gate_ops": sum(gates), "gate_launches": len(gates),
             "attn_res_us": tot["ours"], "attn_res_torch_us": tot["torch"],
             "attn_res_fused_messages_us": ours_fused, "gates_us": g_ours,
             "gates_torch_us": g_torch,
             "per_token_us": (ours_fused + g_ours) / T,
             "per_token_torch_us": (tot["torch"] + g_torch) / T}
        summary[T] = s
        print(f"\nper decode step at batch {T}: {n_res} attention residuals + {n_msg} message "
              f"inputs + {sum(gates)} gates in {len(gates)} launches")
        print(f"  attention residuals {tot['ours']:.1f} us (torch {tot['torch']:.1f}); with each "
              f"pass's 12 message inputs as one nq = 48 call {ours_fused:.1f} us")
        print(f"  gates ({path} path, bf16 wf/wg) {g_ours:.1f} us (torch {g_torch:.1f})")
        print(f"  total {ours_fused + g_ours:.1f} us a step, {s['per_token_us']:.2f} us a token "
              f"(torch {s['per_token_torch_us']:.2f})")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.append and args.out.exists():  # one file from runs split to stay short
        old = json.loads(args.out.read_text())
        rows = old["rows"] + rows
        summary = {**old["per_token"], **summary}
    args.out.write_text(json.dumps({"device": device, "copy_gbps": peak, "rows": rows,
                                    "per_token": summary}, indent=1) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
