#!/usr/bin/env python3
"""Chat with a real Llama-3-8B checkpoint on the engine: the prompts go through the
tokenizer's chat template, are prefilled together and decoded greedily, and the replies are
printed with the decode rate.

    python scripts/generate.py ~/models/llama3-8b-instruct "why is the sky blue?" "hi"
    python scripts/generate.py MODEL --new 256 --torch     # the same on engine.TorchModel

A request stops at the tokenizer's end-of-text or end-of-turn token, or after --new tokens.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from spark_kernels import engine as E  # noqa: E402
from spark_kernels import hf  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("prompts", nargs="*", default=["Explain what an L2 cache is in two sentences."])
    ap.add_argument("--new", type=int, default=128)
    ap.add_argument("--torch", action="store_true", help="run engine.TorchModel instead")
    ap.add_argument("--raw", action="store_true", help="no chat template")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.perf_counter()
    w, cfg = hf.load(args.model)
    print(f"loaded {cfg.n_layers} layers, {w.bytes() / 1e9:.1f} GB in "
          f"{time.perf_counter() - t0:.1f} s")
    rope = cfg.rope(8192)
    model = E.TorchModel(w, rope) if args.torch else E.SparkModel(w, rope)
    stop = {tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")}
    eng = E.Engine(model, cfg.n_layers, max_batch=max(1, len(args.prompts)), max_seq=8192,
                   cache_bytes=4 << 30, graphs=not args.torch, log_tokens=True,
                   stop_ids=sorted(stop))
    ids = []
    for p in args.prompts:
        if args.raw:
            x = tok(p, return_tensors="pt").input_ids[0]
        else:
            x = tok.apply_chat_template([{"role": "user", "content": p}],
                                        add_generation_prompt=True, return_tensors="pt")
            x = x["input_ids"][0] if hasattr(x, "keys") else x[0]
        ids.append(x)
        eng.submit(x, args.new)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    stats = eng.run()
    dt = time.perf_counter() - t0
    outs = eng.outputs()
    for sid, out in sorted(outs.items()):
        print(f"\n### {args.prompts[sid]}\n{tok.decode(out[:-1] if out[-1] in stop else out)}")
    n = sum(map(len, outs.values()))
    print(f"\n{n} tokens in {dt:.2f} s, {n / dt:.1f} tok/s ({len(args.prompts)} requests, "
          f"{stats.stopped} stopped before --new)")


if __name__ == "__main__":
    main()
