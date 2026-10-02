#!/usr/bin/env python3
"""Needle in a haystack on a long-context checkpoint, on the engine or on Hugging Face
transformers: a sentence with a random seven-digit number is hidden at a given depth of a long
text, the chat template asks for the number, and the greedy reply is checked for it. Each run
also gives the time to first token (the whole prefill) and the decode rate at that context.

    python scripts/needle.py ~/models/llama3.1-8b-instruct --lengths 8192 32768
    python scripts/needle.py MODEL --backend hf --lengths 32768 --depths 0.5
    python scripts/needle.py MODEL --lengths 32768 --depths 0.5 --chunk 32768   # one forward

--lengths are prompt lengths in tokens (haystack, needle, question and template together, to
within a few tokens). The engine is sized for the longest one: its cache holds that prompt
plus the reply and nothing else, so on a 32 GB card next to 16 GB of weights about 100K tokens
fit. The engine prefills in forwards of at most --chunk tokens; transformers prefills in one
forward. TTFT is timed on a forward of the whole prompt with the logits of the last position
only: the engine's prefill, and transformers' forward with use_cache=False and
logits_to_keep=1 (its cached generate needs another full-length cache). Decode is --new greedy
tokens after the prompt, with no stop token, timed apart from the prefill (for transformers,
generate's time minus a cached prefill's). Rows are appended to the "needle" list of --out.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

DEFAULT_TEXT = Path.home() / "models" / "war_and_peace.txt"
QUESTION = ("What is the special magic number mentioned in the text above? "
            "Answer with the number only.")


def chat_ids(tok, content: str) -> torch.Tensor:
    x = tok.apply_chat_template([{"role": "user", "content": content}],
                                add_generation_prompt=True, return_tensors="pt")
    return (x["input_ids"] if hasattr(x, "keys") else x)[0]


def build_prompt(tok, hay: list[int], length: int, depth: float, number: str) -> torch.Tensor:
    """A chat prompt of about `length` tokens: haystack text with the needle sentence put at
    the first sentence boundary after `depth` of it, then the question."""
    needle = f" The special magic number is {number}. "
    overhead = chat_ids(tok, needle + "\n\n" + QUESTION).numel()
    text = tok.decode(hay[:max(0, length - overhead)])
    at = text.find(". ", int(depth * len(text)))
    at = len(text) if at < 0 else at + 1
    return chat_ids(tok, text[:at] + needle + text[at:] + "\n\n" + QUESTION)


def sync_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


class Spark:
    def __init__(self, args, max_len: int):
        from spark_kernels import engine as E
        from spark_kernels import hf

        w, cfg = hf.load(args.model)
        max_seq = max_len + args.new + 64
        model = E.SparkModel(w, cfg.rope(max_seq), prefill_variant=args.prefill_variant)
        self.eng = E.Engine(model, cfg.n_layers, max_batch=1, max_seq=max_seq,
                            num_pages=E.cdiv(max_seq, E.PAGE) + 1, prefill_tokens=args.chunk,
                            graphs=True, log_tokens=True)
        self.new = args.new

    def run(self, ids: torch.Tensor) -> dict:
        eng = self.eng
        seq = eng.submit(ids, self.new)
        t0 = sync_time()
        eng.prefill_waiting()
        t1 = sync_time()
        eng.run()
        t2 = sync_time()
        out = eng.outputs()[seq.sid]
        return dict(ttft_s=t1 - t0, decode_tok_s=(len(out) - 1) / (t2 - t1), out=out)


class HF:
    def __init__(self, args, max_len: int):
        from transformers import AutoModelForCausalLM

        self.m = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
        # no stop token, as on the engine (min_new_tokens would mask eos and change the
        # greedy tokens instead)
        self.m.generation_config.eos_token_id = None
        self.new = args.new

    @torch.no_grad()
    def run(self, ids: torch.Tensor) -> dict:
        x = ids[None].cuda()
        try:
            t0 = sync_time()
            self.m(x, use_cache=False, logits_to_keep=1)
            row = dict(ttft_s=sync_time() - t0)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            return dict(ttft_s=None, decode_tok_s=None, out=None, note="out of memory")
        torch.cuda.empty_cache()
        try:
            t0 = sync_time()
            self.m(x, use_cache=True, logits_to_keep=1)
            cached = sync_time() - t0
            torch.cuda.empty_cache()
            t0 = sync_time()
            g = self.m.generate(x, max_new_tokens=self.new, do_sample=False)
            dt = sync_time() - t0
            out = g[0, ids.numel():].tolist()
            row.update(decode_tok_s=(len(out) - 1) / max(dt - cached, 1e-9), out=out)
        except torch.OutOfMemoryError:
            row.update(decode_tok_s=None, out=None, note="generate out of memory")
        torch.cuda.empty_cache()
        return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--backend", choices=("spark", "hf"), default="spark")
    ap.add_argument("--lengths", type=int, nargs="+", default=[8192, 32768, 65536, 100000])
    ap.add_argument("--depths", type=float, nargs="+", default=[0.1, 0.5, 0.9])
    ap.add_argument("--text", default=str(DEFAULT_TEXT))
    ap.add_argument("--chunk", type=int, default=8192, help="engine prefill tokens per forward")
    ap.add_argument("--new", type=int, default=64)
    ap.add_argument("--prefill-variant", type=int, default=-1,
                    help="hgemm variant of the engine's prefill GEMMs (4 for ~100K tokens)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "llm_models.json"))
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    hay = tok(Path(args.text).read_text(), add_special_tokens=False).input_ids
    rng = random.Random(args.seed)
    cases = []
    for n in args.lengths:
        for d in args.depths:
            number = str(rng.randrange(1_000_000, 10_000_000))
            cases.append((n, d, number, build_prompt(tok, hay, n, d, number)))
    max_len = max(c[3].numel() for c in cases)
    runner = (HF if args.backend == "hf" else Spark)(args, max_len)
    warm = min(args.chunk + 500, max_len - 200)  # more than one chunk when the cache allows
    runner.run(chat_ids(tok, tok.decode(hay[:warm]) + "\n\n" + QUESTION))

    stop = {tok.eos_token_id}
    gen = Path(args.model) / "generation_config.json"
    if gen.exists():
        eos = json.loads(gen.read_text()).get("eos_token_id", [])
        stop.update(eos if isinstance(eos, list) else [eos])

    rows = []
    for n, d, number, ids in cases:
        r = runner.run(ids)
        out = r.pop("out")
        reply = None
        if out is not None:
            cut = next((i for i, t in enumerate(out) if t in stop), len(out))
            reply = tok.decode(out[:cut], skip_special_tokens=True)
        row = dict(model=Path(args.model).name, backend=args.backend,
                   label=args.label or args.backend, target=n, tokens=ids.numel(), depth=d,
                   chunk=args.chunk if args.backend == "spark" else None,
                   prefill_variant=args.prefill_variant if args.backend == "spark" else None,
                   number=number,
                   found=None if reply is None else number in reply,
                   reply=None if reply is None else reply.strip()[:80],
                   prefill_tok_s=r["ttft_s"] and ids.numel() / r["ttft_s"], **r)
        print(json.dumps(row), flush=True)
        rows.append(row)

    path = Path(args.out)
    data = json.loads(path.read_text()) if path.exists() else {}
    data.setdefault("needle", []).extend(rows)
    path.write_text(json.dumps(data, indent=1) + "\n")


if __name__ == "__main__":
    main()
