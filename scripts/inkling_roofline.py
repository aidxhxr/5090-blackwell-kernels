#!/usr/bin/env python3
"""Per-step decode cost of Inkling 2 Mini: the bytes one decode step reads, and the tokens/s
ceiling that gives on each card at its measured bandwidth.

    python scripts/inkling_roofline.py                          # the table, embedded config
    python scripts/inkling_roofline.py --batch 1,8,64 --context 4096,131072 --card pro6000
    python scripts/inkling_roofline.py --config config.json --format ckpt --pin --json
    python scripts/inkling_roofline.py --blog                   # the 614 GB/s check only

One step decodes one token for each of B sequences at context L (the new token included). The
step reads, from DRAM:

- dense weights: every non-expert weight of each layer application. Layers 6-17 run three times
  per token on shared weights (48 applications from 24 stored layers), and each pass reads them
  again unless they stay in L2 (--pin, below). The message gates are read in passes 2 and 3
  only. Then the final norm and lm_head, and B rows of the embedding (a gather, not the table).
- experts: per layer application, the experts the B tokens pick. With uniform routing a token
  misses a given expert with probability 1 - k/E, so B tokens touch E (1 - (1 - k/E)^B) distinct
  experts in expectation (8 at B = 1, 51.6 at B = 8, 125.9 at B = 64 for k = 8, E = 128). Each
  pass routes on its own hidden state, so each pass reads its own set.
- KV cache: 48 cache layers (24 stored + 2 x 12 for passes 2 and 3), k and v of 8 heads x 128
  per position; sliding layers read min(L, 1024) positions, full ones (5, 11, 17, 23 and the two
  extra copies of 11 and 17: 8 cache layers) read L. Plus the new position's k and v, written.
- conv states: k and v (1024 channels each) and the post-attention and post-MLP convolutions
  (2304 each) keep k - 1 = 3 past inputs per layer application, read and written each step, in
  fp32 as the reference keeps the post convolutions (it stores k = 4 positions; 3 are needed).
- block-state reads: every attention and MLP input is a softmax mix over the block states plus
  the partial state, fp32 vectors of 2304, read per token (1 to 5 of them, the message input of
  passes 2 and 3 reads 2). Activations, not weights, so they scale with B.

Formats: bf16 (every weight 2 bytes), ckpt (the released checkpoint: NVFP4 experts in layers
0-19, MXFP8 experts in layers 20-23, everything else bf16) and fp4 (every GEMM weight NVFP4,
lm_head included; norms, conv taps and the relative bias bank stay bf16). NVFP4 is 0.5 byte per
weight plus an e4m3 scale per 16 (0.5625), MXFP8 1 byte plus a ue8m0 scale per 32 x 32 block.
The attention-residual queries are fp32 in every format, as in the checkpoint.

L2: the passes of one loop layer are 11 layer applications apart, and one application streams
far more than any L2 here (182 MB of weights at B = 1 in bf16), so nothing survives between
passes by itself. --pin assumes an L2 persistence window (cudaAccessPolicyWindow) holding the
dense weights of as many loop layers as fit in --l2-frac of L2: their passes 2 and 3 read them
from L2. Experts are not pinned: a layer's 128 experts are 2.0 GB in bf16 and 573 MB in fp4, and
at B = 1 the three passes share 1.5 of their 24 picks in expectation, so there is little to keep.

Bandwidths are what this repo measured with bench_bandwidth (cudaMemcpy D2D of 1 GiB, the
roof the row kernels reach): RTX 5090 1,532 GB/s (docs/RTX5090.md), RTX PRO 6000 Blackwell
Server Edition 1,456 GB/s (2026-11-26, ECC on, docs/design/inkling_decode.md). GB10 has no
measured number in the repo; its row uses the 273 GB/s spec. L2 is what the bench banner
prints (cudaDeviceProp::l2CacheSize): 96, 128 and 24 MiB. A ceiling is bytes over bandwidth:
no compute, no launch overhead, no attention math.
"""

from __future__ import annotations

import argparse
import json
import sys

# config.json's text_config of thinkingmachines/Inkling-2-Mini (the fields this model uses).
INKLING2_MINI = {
    "hidden_size": 2304,
    "num_hidden_layers": 24,
    "vocab_size": 201024,
    "unpadded_vocab_size": 200058,
    "num_attention_heads": 24,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "d_rel": 16,
    "rel_extent": 1024,
    "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
    "sconv_kernel_size": 4,
    "sliding_window_size": 1024,
    "n_layers_per_attn_res_block": 6,
    "n_routed_experts": 128,
    "num_experts_per_tok": 8,
    "intermediate_size": 1152,
    "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64},
}

# The checkpoint's expert formats by stored layer (hf_quant_config.json): NVFP4 below 20.
CKPT_MXFP8_FROM = 20

NVFP4_BYTES = 0.5 + 1 / 16  # e2m1 + an e4m3 scale per 16
MXFP8_BYTES = 1 + 1 / 1024  # e4m3 + a ue8m0 scale per 32 x 32 weight block
FORMATS = ("bf16", "ckpt", "fp4")

MB = 1 << 20
CARDS = {
    "rtx5090": {"name": "RTX 5090", "gbps": 1532.0, "measured": True, "l2_mb": 96, "mem_gb": 32},
    "pro6000": {"name": "RTX PRO 6000", "gbps": 1456.0, "measured": True, "l2_mb": 128,
                "mem_gb": 96},
    "gb10": {"name": "GB10 (DGX Spark)", "gbps": 273.0, "measured": False, "l2_mb": 24,
             "mem_gb": 128},
}
BLOG_GBPS = 614.0


def load_config(path: str | None) -> dict:
    """The embedded config, or a config.json (its text_config if it has one)."""
    if path is None:
        return dict(INKLING2_MINI)
    with open(path) as f:
        cfg = json.load(f)
    cfg = cfg.get("text_config", cfg)
    out = dict(INKLING2_MINI)
    out.update({k: cfg[k] for k in INKLING2_MINI if k in cfg})
    return out


def loop_range(cfg: dict) -> tuple[int, int, int]:
    lc = cfg["loop_config"]
    return lc["core_start"], lc["core_end"], lc["num_passes"]


def is_full(cfg: dict, layer: int) -> bool:
    return layer not in cfg["local_layer_ids"]


def schedule(cfg: dict) -> list[tuple[int, int]]:
    """The layer applications of one token in order, as (stored layer, pass)."""
    start, end, passes = loop_range(cfg)
    out = [(i, 0) for i in range(start)]
    for p in range(passes):
        out += [(i, p) for i in range(start, end)]
    out += [(i, 0) for i in range(end, cfg["num_hidden_layers"])]
    return out


# ---- weights -------------------------------------------------------------------------------
#
# A tensor is (name, elements, kind). Kinds: "gemm" (a projection weight, which fp4 quantizes),
# "small" (norm scales, conv taps, the relative bias bank: bf16 in every format) and "fp32"
# (the attention-residual queries).


def layer_tensors(cfg: dict, layer: int, with_gates: bool) -> list[tuple[str, int, str]]:
    """The non-expert tensors one application of `layer` reads; `with_gates` adds the four
    message gates (passes 2 and 3 of a loop layer)."""
    h = cfg["hidden_size"]
    q = cfg["num_attention_heads"] * cfg["head_dim"]
    kv = cfg["num_key_value_heads"] * cfg["head_dim"]
    heads, d_rel, taps = cfg["num_attention_heads"], cfg["d_rel"], cfg["sconv_kernel_size"]
    full = is_full(cfg, layer)
    extent = cfg["rel_extent"] if full else cfg["sliding_window_size"]
    t = [
        ("q_proj", q * h, "gemm"),
        ("k_proj", kv * h, "gemm"),
        ("v_proj", kv * h, "gemm"),
        ("gate_proj", q * h, "gemm"),
        ("o_proj", h * q, "gemm"),
        ("sink_proj", heads * h, "gemm"),
        ("rel state", heads * d_rel * h, "gemm"),
        ("rel bank", d_rel * extent, "small"),
        ("q/rel clip scales", heads * cfg["head_dim"] + heads * d_rel, "small"),
        ("router", cfg["n_routed_experts"] * h, "gemm"),
        ("post convs", 2 * h * taps, "small"),
        ("post_attention_layernorm", h, "small"),
        ("mlp_input query", h, "fp32"),
    ]
    if full:
        t += [
            ("k/v static taps", 2 * taps * kv, "small"),
            ("k/v conv taps", 2 * cfg["num_key_value_heads"] * taps * h, "gemm"),
            ("q/k/v input norms", 3 * h, "small"),
            ("attention_input queries", 3 * h, "fp32"),
        ]
    else:
        t += [("k/v conv taps", 2 * kv * taps, "small"), ("input_layernorm", h, "small")]
        if layer > 0:
            t.append(("attention_input query", h, "fp32"))
    if with_gates:
        r = cfg["loop_config"]["gate_rank"]
        t.append(("message gates", 4 * (r * 2 * h + 2 * h * r), "gemm"))
    return t


def expert_elems(cfg: dict) -> int:
    """One expert: gate_up [2 I, h] and down [h, I]."""
    h, inter = cfg["hidden_size"], cfg["intermediate_size"]
    return 2 * inter * h + h * inter


def tensor_bytes(elems: int, kind: str, fmt: str) -> float:
    if kind == "fp32":
        return 4.0 * elems
    if kind == "gemm" and fmt == "fp4":
        return NVFP4_BYTES * elems
    return 2.0 * elems


def expert_bytes(cfg: dict, layer: int, fmt: str) -> float:
    e = expert_elems(cfg)
    if fmt == "bf16":
        return 2.0 * e
    if fmt == "fp4" or layer < CKPT_MXFP8_FROM:
        return NVFP4_BYTES * e
    return MXFP8_BYTES * e


def dense_bytes(cfg: dict, layer: int, fmt: str, with_gates: bool) -> float:
    return sum(tensor_bytes(n, k, fmt) for _, n, k in layer_tensors(cfg, layer, with_gates))


def head_tensors(cfg: dict) -> list[tuple[str, int, str]]:
    """Read once per step outside the layers: final norm, final_input query, lm_head."""
    h = cfg["hidden_size"]
    return [
        ("norm", h, "small"),
        ("final_input query", h, "fp32"),
        ("lm_head", cfg["unpadded_vocab_size"] * h, "gemm"),
    ]


def weight_footprint(cfg: dict, fmt: str) -> float:
    """Bytes of every text-decoder weight stored once (what has to fit in memory)."""
    h = cfg["hidden_size"]
    start, end, _ = loop_range(cfg)
    total = 2.0 * cfg["vocab_size"] * h + 2.0 * h  # embedding (bf16, gathered) and its norm
    total += sum(tensor_bytes(n, k, fmt) for _, n, k in head_tensors(cfg))
    total += 4.0 * (end - start) * 4 * h  # message_input queries
    for i in range(cfg["num_hidden_layers"]):
        total += dense_bytes(cfg, i, fmt, start <= i < end)
        total += cfg["n_routed_experts"] * expert_bytes(cfg, i, fmt)
    return total


def param_counts(cfg: dict) -> dict:
    """Text-decoder parameter counts: total, active (8 experts per layer, every other weight
    once, the embedding table whole, the convention for active parameters) and effective
    (active plus passes 2 and 3 of the loop layers). Effective counts the message gates in all
    three passes, which gives the model card's 5.4B; they only run in passes 2 and 3, so
    step_bytes reads 28M fewer weights than this."""
    h = cfg["hidden_size"]
    start, end, passes = loop_range(cfg)
    e, k = cfg["n_routed_experts"], cfg["num_experts_per_tok"]
    glob = cfg["vocab_size"] * h + h + h + h + (end - start) * 4 * h  # embed, norms, queries
    glob += cfg["unpadded_vocab_size"] * h
    per_layer = [sum(n for _, n, _ in layer_tensors(cfg, i, start <= i < end))
                 for i in range(cfg["num_hidden_layers"])]
    ex = expert_elems(cfg)
    total = glob + sum(per_layer) + cfg["num_hidden_layers"] * e * ex
    active = glob + sum(per_layer) + cfg["num_hidden_layers"] * k * ex
    extra = sum(per_layer[i] + k * ex for i in range(start, end)) * (passes - 1)
    return {"total": total, "active": active, "effective": active + extra}


# ---- one step ------------------------------------------------------------------------------


def distinct_experts(cfg: dict, tokens: int) -> float:
    """Expected distinct experts `tokens` uniform top-k picks touch."""
    e, k = cfg["n_routed_experts"], cfg["num_experts_per_tok"]
    return e * (1.0 - (1.0 - k / e) ** tokens)


def cache_layers(cfg: dict) -> list[int]:
    """The stored layer behind each of the cache layers (48 for Inkling 2 Mini)."""
    return [i for i, _ in schedule(cfg)]


def kv_read_bytes(cfg: dict, context: int, kv_bytes: float = 2.0) -> float:
    """KV bytes one sequence's step reads at `context` positions, plus the new k, v written."""
    per_pos = 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * kv_bytes
    w = cfg["sliding_window_size"]
    total = 0.0
    for i in cache_layers(cfg):
        total += (context if is_full(cfg, i) else min(context, w)) * per_pos + per_pos
    return total


def kv_footprint(cfg: dict, context: int, kv_bytes: float = 2.0) -> float:
    """KV bytes one sequence holds at `context` positions (sliding layers keep the window)."""
    per_pos = 2 * cfg["num_key_value_heads"] * cfg["head_dim"] * kv_bytes
    w = cfg["sliding_window_size"]
    return sum((context if is_full(cfg, i) else min(context, w)) * per_pos
               for i in cache_layers(cfg))


def conv_state_bytes(cfg: dict) -> float:
    """Conv-state bytes one sequence's step reads and writes (fp32, k - 1 positions)."""
    ch = 2 * cfg["num_key_value_heads"] * cfg["head_dim"] + 2 * cfg["hidden_size"]
    per_app = ch * (cfg["sconv_kernel_size"] - 1) * 4.0
    return 2 * per_app * len(schedule(cfg))


def block_state_reads(cfg: dict) -> int:
    """Vectors of `hidden` one token's block-attention-residual mixes read in one step: the
    attention and MLP inputs of every layer application, the message input of each loop layer
    in passes 2 and 3, and the final input. Follows the reference model's bookkeeping."""
    start, end, passes = loop_range(cfg)
    blk = cfg["n_layers_per_attn_res_block"]
    reads = 0

    def layer(i: int, nb: int) -> tuple[int, int]:
        boundary = i % blk == 0
        r = (nb + 1 if i > 0 else 0) + (nb + 2 if boundary else nb + 1)
        return r, nb + (1 if boundary else 0)

    nb = 0
    for i in range(start):
        r, nb = layer(i, nb)
        reads += r
    pre = nb
    for p in range(passes):
        if p > 0:
            prev = nb - (pre + 1)  # the previous pass's blocks past its first
            nb = pre
        for i in range(start, end):
            if p > 0:
                reads += prev + 1
            r, nb = layer(i, nb)
            reads += r
    for i in range(end, cfg["num_hidden_layers"]):
        r, nb = layer(i, nb)
        reads += r
    return reads + nb + 1


def step_bytes(cfg: dict, batch: int, context: int, fmt: str, pinned_layers: int = 0,
               kv_bytes: float = 2.0) -> dict:
    """Bytes one decode step reads, by part. `pinned_layers` loop layers keep their dense
    weights in L2 across passes, so their passes 2 and 3 do not read them again."""
    start, end, _ = loop_range(cfg)
    h = cfg["hidden_size"]
    dense = rereads = experts = 0.0
    for i, p in schedule(cfg):
        base = dense_bytes(cfg, i, fmt, with_gates=False)
        gates = dense_bytes(cfg, i, fmt, with_gates=True) - base
        pinned = start <= i < start + pinned_layers
        if p == 0:
            dense += base
        elif p == 1:
            dense += gates  # the message gates' first read
            rereads += 0.0 if pinned else base
        else:
            rereads += 0.0 if pinned else base + gates
        experts += distinct_experts(cfg, batch) * expert_bytes(cfg, i, fmt)
    dense += sum(tensor_bytes(n, k, fmt) for _, n, k in head_tensors(cfg))
    dense += 4.0 * (end - start) * 4 * h  # message_input queries
    dense += batch * h * 2.0 + 2.0 * h  # embedding rows and their norm
    parts = {
        "dense": dense,
        "loop rereads": rereads,
        "experts": experts,
        "kv": batch * kv_read_bytes(cfg, context, kv_bytes),
        "conv states": batch * conv_state_bytes(cfg),
        "block states": batch * block_state_reads(cfg) * h * 4.0,
    }
    parts["total"] = sum(parts.values())
    return parts


def pinnable_layers(cfg: dict, fmt: str, l2_mb: float, frac: float) -> int:
    """Loop layers whose dense weights (message gates included) fit together in `frac` of
    the L2."""
    start, end, _ = loop_range(cfg)
    per = dense_bytes(cfg, start, fmt, with_gates=True)
    return min(end - start, int(frac * l2_mb * MB // per))


def ceiling(parts: dict, batch: int, gbps: float) -> tuple[float, float]:
    """(ms per step, tokens/s) at `gbps` GB/s."""
    ms = parts["total"] / (gbps * 1e9) * 1e3
    return ms, batch / (ms * 1e-3)


def blog_check(cfg: dict) -> dict:
    """The blog's ceiling: 614 GB/s over the effective parameters in bf16 at batch 1, and this
    model's byte count at the same point (B = 1, L = 1, embedding gathered, not streamed)."""
    pc = param_counts(cfg)
    parts = step_bytes(cfg, 1, 1, "bf16")
    return {
        "effective_params": pc["effective"],
        "blog_tok_s_5p4": BLOG_GBPS * 1e9 / (2 * 5.4e9),
        "params_tok_s": BLOG_GBPS * 1e9 / (2 * pc["effective"]),
        "bytes_tok_s": ceiling(parts, 1, BLOG_GBPS)[1],
        "bytes": parts["total"],
    }


# ---- output --------------------------------------------------------------------------------


def parse_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", help="config.json (default: the embedded Inkling 2 Mini one)")
    ap.add_argument("--batch", default="1,8,64", help="comma-separated B")
    ap.add_argument("--context", default="4096,32768,131072", help="comma-separated L")
    ap.add_argument("--card", default=",".join(CARDS), help="comma-separated: " + ",".join(CARDS))
    ap.add_argument("--format", default=",".join(FORMATS), help="comma-separated: bf16,ckpt,fp4")
    ap.add_argument("--gbps", type=float, help="override every card's bandwidth")
    ap.add_argument("--pin", action="store_true", help="pin loop layers' dense weights in L2")
    ap.add_argument("--l2-frac", type=float, default=0.75, help="L2 share --pin may use")
    ap.add_argument("--kv-bytes", type=float, default=2.0, help="bytes per KV element")
    ap.add_argument("--json", action="store_true", help="one JSON object per row")
    ap.add_argument("--blog", action="store_true", help="only the 614 GB/s check")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)

    pc = param_counts(cfg)
    chk = blog_check(cfg)
    print(f"text decoder: {pc['total'] / 1e9:.3f}B total, {pc['active'] / 1e9:.3f}B active, "
          f"{pc['effective'] / 1e9:.3f}B effective with the loop")
    print(f"614 GB/s, bf16, B = 1: {chk['blog_tok_s_5p4']:.1f} tok/s from 5.4B x 2 bytes, "
          f"{chk['params_tok_s']:.1f} from {chk['effective_params'] / 1e9:.3f}B, "
          f"{chk['bytes_tok_s']:.1f} from the {chk['bytes'] / 1e9:.2f} GB this model reads")
    if args.blog:
        return 0
    print("expected distinct experts: " + ", ".join(
        f"B={b}: {distinct_experts(cfg, b):.1f}" for b in parse_list(args.batch)))

    rows = []
    for card in args.card.split(","):
        c = CARDS[card]
        gbps = args.gbps or c["gbps"]
        for fmt in args.format.split(","):
            pin = pinnable_layers(cfg, fmt, c["l2_mb"], args.l2_frac) if args.pin else 0
            for b in parse_list(args.batch):
                for ctx in parse_list(args.context):
                    parts = step_bytes(cfg, b, ctx, fmt, pin, args.kv_bytes)
                    ms, tok_s = ceiling(parts, b, gbps)
                    need = weight_footprint(cfg, fmt) + b * kv_footprint(cfg, ctx, args.kv_bytes)
                    rows.append({"card": c["name"], "gbps": gbps, "format": fmt, "batch": b,
                                 "context": ctx, "pinned_layers": pin,
                                 "fits": need <= c["mem_gb"] * 1e9, "memory_gb": need / 1e9,
                                 "ms": ms, "tok_s": tok_s,
                                 **{k: v / 1e6 for k, v in parts.items()}})
    if args.json:
        for r in rows:
            print(json.dumps(r))
        return 0
    cols = ["dense", "loop rereads", "experts", "kv", "conv states", "block states", "total"]
    print("| card | GB/s | format | B | L | " + " | ".join(f"{c} MB" for c in cols)
          + " | ms | tok/s | memory GB |")
    print("|" + "---|" * (len(cols) + 8))
    for r in rows:
        mem = f"{r['memory_gb']:.1f}" + ("" if r["fits"] else " (does not fit)")
        print(f"| {r['card']} | {r['gbps']:.0f} | {r['format']} | {r['batch']} | {r['context']} | "
              + " | ".join(f"{r[c]:.0f}" for c in cols)
              + f" | {r['ms']:.2f} | {r['tok_s']:.0f} | {mem} |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
