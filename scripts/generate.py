#!/usr/bin/env python3
"""Chat with a real checkpoint on the engine (Llama-3-8B, Llama-3.1-8B, Mistral-7B-v0.3 or
anything else of their shape): the prompts go through the tokenizer's chat template, are
prefilled together and decoded greedily, and the replies are printed with the decode rate.

    python scripts/generate.py ~/models/llama3-8b-instruct "why is the sky blue?" "hi"
    python scripts/generate.py MODEL --new 256 --torch     # the same on engine.TorchModel
    python scripts/generate.py MODEL --spec 4 "summarize: ..."  # prompt-lookup speculation

A request stops at a stop token (the tokenizer's eos, the generation config's eos ids and
Llama 3's <|eot_id|> when the vocabulary has it), or after --new tokens.
"""

from __future__ import annotations

import argparse
import json
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
    ap.add_argument("--spec", type=int, default=0,
                    help="prompt-lookup speculative decoding with up to this many draft tokens")
    args = ap.parse_args()
    if args.spec and args.torch:
        ap.error("--spec needs the engine's own model (no --torch)")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    t0 = time.perf_counter()
    w, cfg = hf.load(args.model)
    print(f"loaded {cfg.n_layers} layers, {w.bytes() / 1e9:.1f} GB in "
          f"{time.perf_counter() - t0:.1f} s")
    rope = cfg.rope(8192)
    model = E.TorchModel(w, rope) if args.torch else E.SparkModel(w, rope)
    stop = {tok.eos_token_id}
    gen = Path(args.model) / "generation_config.json"
    if gen.exists():
        eos = json.loads(gen.read_text()).get("eos_token_id", [])
        stop.update(eos if isinstance(eos, list) else [eos])
    if "<|eot_id|>" in tok.get_vocab():
        stop.add(tok.convert_tokens_to_ids("<|eot_id|>"))
    eng = E.Engine(model, cfg.n_layers, max_batch=max(1, len(args.prompts)), max_seq=8192,
                   cache_bytes=4 << 30, graphs=not args.torch, log_tokens=True,
                   stop_ids=sorted(stop), speculative=args.spec)
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
    if args.spec:
        steps = stats.decode_steps + stats.spec_steps
        print(f"speculative, up to {args.spec} drafts: {stats.spec_accepted} of "
              f"{stats.spec_proposed} draft tokens accepted "
              f"({100 * stats.spec_accepted / max(1, stats.spec_proposed):.0f}%), "
              f"{stats.decode_tokens / max(1, steps):.2f} tokens per decode forward "
              f"({stats.spec_steps} verify forwards, {stats.decode_steps} plain decode steps)")


if __name__ == "__main__":
    main()
