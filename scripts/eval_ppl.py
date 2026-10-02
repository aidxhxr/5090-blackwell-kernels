#!/usr/bin/env python3
"""Perplexity of a real checkpoint on a text file (wikitext-2 test by default), on the engine
or on Hugging Face transformers, so a kernel or a weight format can be judged by the number
people quote.

    python scripts/eval_ppl.py ~/models/llama3-8b-instruct                  # engine, bf16
    python scripts/eval_ppl.py MODEL --backend hf                           # transformers
    python scripts/eval_ppl.py MODEL --backend torch --ctx 4096 --windows 20
    python scripts/eval_ppl.py MODEL --save-ref ref.pt                      # bf16 logits
    python scripts/eval_ppl.py MODEL --format int4 --ref ref.pt             # + KL vs bf16

The text is tokenized once (with the BOS token the tokenizer adds) and cut into
non-overlapping windows of --ctx tokens; every token of a window except its first is scored
given the ones before it in that window. Windows are prefilled --batch at a time in one packed
batch (the engine) or as one padded-free batch of equal lengths (transformers). Writes one
JSON line per run to --out.

--format runs the engine's SparkModel with its projections in a low-precision format
(spark_kernels.quant); --awq folds activation-aware scales from scripts/awq_search.py into
the bf16 weights before they are quantized. --save-ref stores the logits of the first
--kl-windows windows (bf16, on the CPU, then to a file); a later run with --ref compares its
own logits on those windows with them: the mean KL divergence KL(ref || this) per token in
nats and the fraction of tokens whose argmax agrees, the two numbers that say how far a
quantized model moved from the reference beyond the one perplexity.
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
    if args.awq:
        from spark_kernels import awq

        awq.apply(w, torch.load(args.awq)["scales"])
    if args.backend == "torch":
        model = E.TorchModel(w, rope)
    else:
        model = E.SparkModel(w, rope, weights_format=args.format)
        del w
        print(f"weights: {model.weight_bytes() / 2**30:.2f} GiB ({args.format})", file=sys.stderr)
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


def kl_top1(logits: torch.Tensor, ref: torch.Tensor) -> tuple[float, int]:
    """(summed KL(ref || logits) in nats, tokens whose argmax agrees), fp32, in chunks."""
    kl, agree = 0.0, 0
    for s in range(0, logits.shape[0], 512):
        p = F.log_softmax(ref[s:s + 512].to(logits.device).float(), dim=-1)
        q = F.log_softmax(logits[s:s + 512].float(), dim=-1)
        kl += F.kl_div(q, p, log_target=True, reduction="sum").item()
        agree += (p.argmax(-1) == q.argmax(-1)).sum().item()
    return kl, agree


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--backend", choices=("spark", "torch", "hf"), default="spark")
    ap.add_argument("--text", default=str(DEFAULT_TEXT))
    ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=None, help="score only the first N")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--format", default="bf16", help="weights format of the spark backend")
    ap.add_argument("--awq", default=None, help="fold these scales in first (awq_search.py)")
    ap.add_argument("--save-ref", default=None, help="store the first windows' logits here")
    ap.add_argument("--ref", default=None, help="KL and top-1 agreement against these logits")
    ap.add_argument("--kl-windows", type=int, default=4)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "llm_ppl.jsonl"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    ids = tok(Path(args.text).read_text(), return_tensors="pt").input_ids[0]
    wins = windows(ids, args.ctx, args.windows)
    score = hf_scorer(args) if args.backend == "hf" else engine_scorer(args)
    refs = torch.load(args.ref) if args.ref else None
    saved = []
    total, count = 0.0, 0
    kl, agree, kl_count = 0.0, 0, 0
    t0 = time.perf_counter()
    for i in range(0, len(wins), args.batch):
        batch = wins[i:i + args.batch]
        for j, (w, lg) in enumerate(zip(batch, score([b.cuda() for b in batch]), strict=True)):
            total += nll_sum(lg, w)
            count += w.numel() - 1
            if i + j < args.kl_windows:
                if args.save_ref:
                    saved.append(lg.to(torch.bfloat16).cpu())
                if refs is not None:
                    k, a = kl_top1(lg, refs[i + j])
                    kl, agree, kl_count = kl + k, agree + a, kl_count + lg.shape[0]
    dt = time.perf_counter() - t0
    ppl = math.exp(total / count)
    label = args.label or (args.format + ("-awq" if args.awq else "")
                           if args.backend == "spark" else args.backend)
    row = dict(model=Path(args.model).name, backend=args.backend, format=args.format,
               awq=bool(args.awq), label=label, ctx=args.ctx, windows=len(wins), tokens=count,
               nll=total / count, ppl=ppl, seconds=dt)
    if refs is not None:
        row.update(kl=kl / kl_count, top1=agree / kl_count, kl_tokens=kl_count)
    if args.save_ref:
        torch.save(saved, args.save_ref)
    print(json.dumps(row))
    with open(args.out, "a") as f:
        f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
