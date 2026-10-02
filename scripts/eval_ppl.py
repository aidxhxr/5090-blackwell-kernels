#!/usr/bin/env python3
"""Perplexity of a real checkpoint on a text file (wikitext-2 test by default), on the engine
or on Hugging Face transformers, so a kernel or a weight format can be judged by the number
people quote.

    python scripts/eval_ppl.py ~/models/llama3-8b-instruct                  # engine, bf16
    python scripts/eval_ppl.py MODEL --backend hf                           # transformers
    python scripts/eval_ppl.py MODEL --backend torch --ctx 4096 --windows 20

The text is tokenized once (with the BOS token the tokenizer adds) and cut into
non-overlapping windows of --ctx tokens; every token of a window except its first is scored
given the ones before it in that window. Windows are prefilled --batch at a time in one packed
batch (the engine) or as one padded-free batch of equal lengths (transformers). Writes one
JSON line per run to --out.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

DEFAULT_TEXT = Path.home() / "models" / "wikitext2_test.txt"


def windows(ids: torch.Tensor, ctx: int, n: int | None) -> list[torch.Tensor]:
    out = [ids[i:i + ctx] for i in range(0, ids.numel() - ctx + 1, ctx)]
    return out[:n] if n else out


def nll_sum(logits: torch.Tensor, ids: torch.Tensor) -> float:
    """Summed negative log likelihood of ids[1:] under logits[:-1], fp32, in chunks."""
    total = 0.0
    for s in range(0, ids.numel() - 1, 1024):
        e = min(s + 1024, ids.numel() - 1)
        total += F.cross_entropy(logits[s:e].float(), ids[s + 1:e + 1].to(logits.device),
                                 reduction="sum").item()
    return total


def engine_scorer(args):
    from spark_kernels import engine as E
    from spark_kernels import hf

    w, cfg = hf.load(args.model)
    rope = cfg.rope(max(args.ctx, 8192))
    model = E.TorchModel(w, rope) if args.backend == "torch" else E.SparkModel(w, rope)
    eng = E.Engine(model, cfg.n_layers, max_batch=args.batch, max_seq=args.ctx + 1,
                   num_pages=args.batch * E.cdiv(args.ctx + 1, E.PAGE) + 1,
                   prefill_tokens=args.batch * args.ctx, graphs=False)

    def score(batch):
        with torch.no_grad():
            return eng.prompt_logits(batch)

    return score


def hf_scorer(args):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                             attn_implementation="sdpa").cuda().eval()

    def score(batch):
        with torch.no_grad():
            x = torch.stack(batch).cuda()
            return list(m(x).logits)

    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--backend", choices=("spark", "torch", "hf"), default="spark")
    ap.add_argument("--text", default=str(DEFAULT_TEXT))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=None, help="score only the first N")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "llm_ppl.jsonl"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(Path(args.text).read_text(), return_tensors="pt").input_ids[0]
    wins = windows(ids, args.ctx, args.windows)
    score = hf_scorer(args) if args.backend == "hf" else engine_scorer(args)
    total, count = 0.0, 0
    t0 = time.perf_counter()
    for i in range(0, len(wins), args.batch):
        batch = wins[i:i + args.batch]
        for w, lg in zip(batch, score([b.cuda() for b in batch]), strict=True):
            total += nll_sum(lg, w)
            count += w.numel() - 1
    dt = time.perf_counter() - t0
    ppl = math.exp(total / count)
    row = dict(model=Path(args.model).name, backend=args.backend, label=args.label or args.backend,
               ctx=args.ctx, windows=len(wins), tokens=count, nll=total / count, ppl=ppl,
               seconds=dt)
    print(json.dumps(row))
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
