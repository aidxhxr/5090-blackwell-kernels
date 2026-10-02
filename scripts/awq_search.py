#!/usr/bin/env python3
"""Search activation-aware int4 scales (spark_kernels.awq) for a checkpoint on calibration
text and save them, so eval_ppl.py --awq can fold them in before quantizing.

    python scripts/awq_search.py ~/models/llama3-8b-instruct --text wikitext2_train.txt \\
        --out awq_int4.pt [--asym]

The calibration set is --samples windows of --seq tokens taken at even strides through the
tokenized text (with the tokenizer's BOS at the start of the text only). Use text the model
is not scored on: the wikitext-2 train split, not the test split eval_ppl.py reads.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model")
    ap.add_argument("--text", required=True)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--asym", action="store_true")
    ap.add_argument("--grid", type=int, default=20)
    ap.add_argument("--sample-tokens", type=int, default=4096)
    ap.add_argument("--layers", type=int, default=None, help="only the first N (a quick test)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from spark_kernels import awq, hf

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(Path(args.text).read_text(), return_tensors="pt").input_ids[0]
    stride = (ids.numel() - args.seq) // args.samples
    calib = torch.stack([ids[i * stride:i * stride + args.seq] for i in range(args.samples)])
    w, cfg = hf.load(args.model, n_layers=args.layers)
    rope = cfg.rope(args.seq)
    t0 = time.perf_counter()
    scales = awq.search(w, rope, calib, asym=args.asym, grid=args.grid,
                        sample=args.sample_tokens, log=lambda s: print(s, file=sys.stderr))
    print(f"searched {len(scales)} layers in {time.perf_counter() - t0:.0f} s", file=sys.stderr)
    torch.save({"asym": args.asym, "scales": [{k: v.cpu() for k, v in s.items()}
                                              for s in scales]}, args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
