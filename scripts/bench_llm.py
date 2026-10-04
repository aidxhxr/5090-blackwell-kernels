#!/usr/bin/env python3
"""Serve a real Llama-3-8B checkpoint on this package's engine, on vLLM and on Hugging Face
transformers, with the same prompt token ids, greedy, every request generating exactly its
output length (no stop on EOS). Each backend runs in its own process (vLLM lives in its own
venv) and appends JSON rows; `report` merges them into one table and one JSON file.

    python scripts/bench_llm.py workload MODEL                  # results/llm_workload.json
    python scripts/bench_llm.py run spark MODEL --out rows.jsonl
    ~/vllm-env/bin/python scripts/bench_llm.py run vllm MODEL --out rows.jsonl
    python scripts/bench_llm.py run hf MODEL --out rows.jsonl   # hf_static, hf_cb as well
    python scripts/bench_llm.py report rows.jsonl               # results/llm_serve.json

Workloads (`workload` writes them once, so every backend reads the same token ids):

    static     B in 1, 8, 32, 64 prompts of 512 tokens of wikitext-2 text, 256 tokens out
    decode     one prompt of 128 tokens, 1024 out: single-stream tok/s
    prefill    one prompt of 2048 and one of 8000 tokens, 1 token out: time to first token
    continuous 256 chat requests ("summarize this passage" over a wikitext excerpt), prompt
               and output lengths drawn log-uniformly (seeded), all submitted at once
    parity     six chat prompts, 128 tokens each, the tokens kept for a divergence check

How a row is timed, the same on every backend: a warm-up of the same shape with 4 output
tokens (the full output length for hf_static, whose compiled step is specialized to the cache
length, prompt + out), then the batch with 1 output token (its wall time is the time to first
token, TTFT), then the batch with the full output length. ITL (the per-token decode latency)
is (full - TTFT) / (out - 1), output tok/s is B * out / full. Times are wall clock around a
blocking generate call, so the host side of each backend (scheduling, sampling; detokenizing
is off) counts.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

WORKLOAD = ROOT / "results" / "llm_workload.json"
OUT = ROOT / "results" / "llm_serve.json"
WIKITEXT = Path.home() / "models" / "wikitext2_test.txt"
STATIC_B = (1, 8, 32, 64)
MAX_SEQ = 8192
BOS = 128000
# what one decode step reads at B = 1: every layer's weights and the lm_head (the embedding
# is one row); filled from the checkpoint by the spark backend, this is Llama-3-8B's value
STEP_BYTES = 32 * 218_103_808 * 2 + 4096 * 128256 * 2
COPY_ROOF_GBS = 1532.0  # measured device-to-device copy on this RTX 5090 (bandwidth.md)

PARITY_PROMPTS = [
    "Explain what an L2 cache is in two sentences.",
    "Write a haiku about a graphics card.",
    "What is the capital of Australia, and why is it not Sydney?",
    "List three differences between TCP and UDP.",
    "Translate 'the weather is nice today' into French and German.",
    "Why does ice float on water?",
]


# ---- workloads ---------------------------------------------------------------------------

def log_uniform(rng: random.Random, lo: int, hi: int) -> int:
    return int(round(math.exp(rng.uniform(math.log(lo), math.log(hi)))))


def chat_ids(tok, text: str) -> list[int]:
    x = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True,
                                tokenize=True)
    if hasattr(x, "keys"):
        x = x["input_ids"]
    return [int(t) for t in x]


def make_workload(model: str, requests: int, seed: int) -> dict:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    stream = tok(WIKITEXT.read_text(), add_special_tokens=False).input_ids
    rng = random.Random(seed)

    def window(n: int) -> list[int]:
        off = rng.randrange(0, len(stream) - n)
        return stream[off:off + n]

    w: dict = {"model": Path(model).name, "seed": seed, "static": [], "prefill": []}
    for b in STATIC_B:
        w["static"].append({"batch": b, "out": 256,
                            "prompts": [[BOS] + window(511) for _ in range(b)]})
    w["decode"] = {"batch": 1, "out": 1024, "prompts": [[BOS] + window(127)]}
    for n in (2048, 8000):
        w["prefill"].append({"batch": 1, "out": 1, "prompts": [[BOS] + window(n - 1)]})
    cont = []
    for _ in range(requests):
        text = tok.decode(window(log_uniform(rng, 32, 1536)))
        ids = chat_ids(tok, "Summarize this passage in a few sentences.\n\n" + text)
        cont.append({"prompt": ids, "out": log_uniform(rng, 16, 768)})
    w["continuous"] = cont
    w["parity"] = {"out": 128, "prompts": [chat_ids(tok, p) for p in PARITY_PROMPTS]}
    return w


# ---- backends: each is generate(prompts, outs) -> list of token lists ----------------------

class Spark:
    name = "spark"
    static = False

    def __init__(self, model: str, max_batch: int, cache_gb: float, prefill_tokens: int):
        import torch

        from spark_kernels import engine as E
        from spark_kernels import hf

        self.torch, self.E = torch, E
        w, cfg = hf.load(model)
        global STEP_BYTES
        STEP_BYTES = sum(x.bytes() for x in w.layers) + w.lm_head.numel() * 2
        self.eng = E.Engine(E.SparkModel(w, cfg.rope(MAX_SEQ)), cfg.n_layers,
                            max_batch=max_batch, max_seq=MAX_SEQ,
                            cache_bytes=int(cache_gb * (1 << 30)),
                            prefill_tokens=prefill_tokens, graphs=True)
        self.config = {"max_batch": self.eng.max_batch, "cache_gb": cache_gb,
                       "cache_tokens": self.eng.cache.num_pages * self.eng.page,
                       "prefill_tokens": prefill_tokens, "graphs": True}
        print(f"spark: {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated after the "
              f"graphs, {torch.cuda.mem_get_info()[0] / 2**30:.2f} GiB free", file=sys.stderr)

    def generate(self, prompts, outs, keep=False):
        torch, eng = self.torch, self.eng
        eng.log = [] if keep else None
        seqs = [eng.submit(torch.tensor(p), o) for p, o in zip(prompts, outs, strict=True)]
        eng.run()
        if not keep:
            return None
        got = eng.outputs()
        eng.log = None
        return [got[s.sid] for s in seqs]

    def generate_latency(self, prompts, outs):
        """generate, with a CUDA event recorded after every forward and the sequences that
        got their first or last token in it, so each request's time to first and last token
        comes out without a synchronize in the loop."""
        torch, eng = self.torch, self.eng
        marks = []
        orig_p, orig_d = eng._prefill, eng._decode

        def mark(first, both):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            marks.append((ev, [s.sid for s in first],
                          [s.sid for s in both if s.generated >= s.max_new]))

        def prefill(seqs, dec=()):
            orig_p(seqs, dec)
            mark(seqs, [*seqs, *dec])

        def decode():
            live = [s for s in eng.running if s is not None]
            orig_d()
            mark([], live)

        eng._prefill, eng._decode = prefill, decode
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            seqs = [eng.submit(torch.tensor(p), o) for p, o in zip(prompts, outs, strict=True)]
            eng.run()
        finally:
            eng._prefill, eng._decode = orig_p, orig_d
        first, last = {}, {}
        for ev, f, d in marks:
            t = start.elapsed_time(ev) / 1e3
            for sid in f:
                first.setdefault(sid, t)  # a preempted request is prefilled again later
            last.update((sid, t) for sid in d)
        return [(first[s.sid], last[s.sid]) for s in seqs]


class VLLM:
    name = "vllm"
    static = False

    def __init__(self, model: str, max_batch: int, mem: float):
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.inputs import TokensPrompt

        self.SP, self.TP = SamplingParams, TokensPrompt
        self.llm = LLM(model=model, dtype="bfloat16", max_model_len=MAX_SEQ,
                       gpu_memory_utilization=mem, enable_prefix_caching=False, seed=0,
                       max_num_seqs=max_batch, disable_log_stats=False)
        sc = self.llm.llm_engine.vllm_config.scheduler_config
        self.config = {"version": vllm.__version__, "max_num_seqs": sc.max_num_seqs,
                       "max_num_batched_tokens": sc.max_num_batched_tokens,
                       "chunked_prefill": sc.enable_chunked_prefill,
                       "gpu_memory_utilization": mem, "prefix_caching": False}

    def generate(self, prompts, outs, keep=False):
        sps = [self.SP(temperature=0.0, max_tokens=o, ignore_eos=True, detokenize=False)
               for o in outs]
        res = self.llm.generate([self.TP(prompt_token_ids=p) for p in prompts], sps,
                                use_tqdm=False)
        return [list(r.outputs[0].token_ids) for r in res] if keep else None

    def generate_latency(self, prompts, outs):
        """Time to first and last token per request from vLLM's own request stats: the
        frontend's first-token latency, plus the engine core's first to last token span."""
        sps = [self.SP(temperature=0.0, max_tokens=o, ignore_eos=True, detokenize=False)
               for o in outs]
        res = self.llm.generate([self.TP(prompt_token_ids=p) for p in prompts], sps,
                                use_tqdm=False)
        out = []
        for r in res:
            m = r.metrics
            out.append((m.first_token_latency,
                        m.first_token_latency + m.last_token_ts - m.first_token_ts))
        self.preemptions = sum(r.metrics.num_preemptions for r in res)
        return out


class HF:
    """transformers generate on one batch of equal prompt lengths (no padding), bf16, sdpa;
    `static` uses the static cache, whose decode step transformers compiles on its own."""

    def __init__(self, model: str, static: bool):
        import torch
        from transformers import AutoModelForCausalLM

        self.torch = torch
        self.name = "hf_static" if static else "hf"
        self.m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16,
                                                      attn_implementation="sdpa").cuda().eval()
        gc = self.m.generation_config
        gc.eos_token_id, gc.do_sample, gc.temperature, gc.top_p = None, False, None, None
        gc.pad_token_id = 128001
        self.static = static
        import transformers
        self.config = {"version": transformers.__version__, "attn": "sdpa",
                       "cache": "static + compiled decode" if static else "dynamic"}

    def generate(self, prompts, outs, keep=False):
        torch = self.torch
        if len(set(map(len, prompts))) > 1 or len(set(outs)) > 1:
            # no padding: prompts of different lengths (the parity set) run one at a time
            res = [self.generate([p], [o], keep) for p, o in zip(prompts, outs, strict=True)]
            return [r[0] for r in res] if keep else None
        ids = torch.tensor(prompts, device="cuda")
        kw = {"cache_implementation": "static"} if self.static else {}
        with torch.inference_mode():
            y = self.m.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=outs[0],
                                min_new_tokens=outs[0], **kw)
        torch.cuda.synchronize()
        return y[:, ids.shape[1]:].tolist() if keep else None


class HFContinuous(HF):
    """transformers' own continuous batching (generate_batch's manager, paged sdpa)."""

    def __init__(self, model: str):
        super().__init__(model, static=False)
        self.name = "hf_cb"
        self.config["cache"] = "continuous batching manager, paged"

    def generate(self, prompts, outs, keep=False):
        from transformers import GenerationConfig

        gc = GenerationConfig(max_new_tokens=max(outs), do_sample=False, eos_token_id=None,
                              pad_token_id=128001)
        mgr = self.m.init_continuous_batching(generation_config=gc)
        mgr.start()
        rids = [mgr.add_request(p, max_new_tokens=o) for p, o in
                zip(prompts, outs, strict=True)]
        res = {}
        while len(res) < len(rids):
            r = mgr.get_result(timeout=600)
            if r is not None:
                res[r.request_id] = r
        mgr.stop(block=True)
        self.torch.cuda.synchronize()
        return [list(res[r].generated_tokens) for r in rids] if keep else None


# ---- timing ------------------------------------------------------------------------------

def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def latency_summary(lat: list[tuple[float, float]], outs: list[int]) -> dict:
    """(time to first token, time to last token) per request, all submitted at t = 0."""
    ttft = [a * 1e3 for a, _ in lat]
    e2e = [b for _, b in lat]
    itl = [(b - a) * 1e3 / (o - 1) for (a, b), o in zip(lat, outs, strict=True) if o > 1]
    return {"ttft_ms_p50": pct(ttft, 0.5), "ttft_ms_p90": pct(ttft, 0.9),
            "ttft_ms_mean": sum(ttft) / len(ttft), "e2e_s_p50": pct(e2e, 0.5),
            "e2e_s_p90": pct(e2e, 0.9), "itl_ms_mean": sum(itl) / len(itl),
            "itl_ms_p90": pct(itl, 0.9)}


def timed(fn) -> float:
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def batch_row(be, kind: str, prompts, out: int, warm: bool = True) -> dict:
    b = len(prompts)
    if warm:  # the static cache is sized prompt + out, so its compiled step needs the real out
        be.generate(prompts, [out if getattr(be, "static", False) else min(out, 4)] * b)
    ttft = timed(lambda: be.generate(prompts, [1] * b))
    full = ttft if out == 1 else timed(lambda: be.generate(prompts, [out] * b))
    row = {"backend": be.name, "kind": kind, "batch": b, "prompt": len(prompts[0]),
           "out": out, "ttft_ms": ttft * 1e3, "total_s": full,
           "prefill_tok_s": b * len(prompts[0]) / ttft}
    if out > 1:
        itl = (full - ttft) / (out - 1)
        row.update(itl_ms=itl * 1e3, out_tok_s=b * out / full, decode_tok_s=b / itl)
        if b == 1:
            row["step_gb_s"] = STEP_BYTES / itl / 1e9
            row["pct_copy_roof"] = 100 * row["step_gb_s"] / COPY_ROOF_GBS
    print(json.dumps(row), file=sys.stderr, flush=True)
    return row


def run(args) -> int:
    w = json.loads(Path(args.workload).read_text())
    if args.backend == "spark":
        be = Spark(args.model, args.max_batch, args.cache_gb, args.prefill_tokens)
    elif args.backend == "vllm":
        be = VLLM(args.model, args.max_batch, args.vllm_mem)
    elif args.backend in ("hf", "hf_static"):
        be = HF(args.model, static=args.backend == "hf_static")
    elif args.backend == "hf_cb":
        be = HFContinuous(args.model)
    else:
        raise SystemExit(f"unknown backend {args.backend}")
    only = set(args.only.split(","))
    rows = []
    if "static" in only:
        for s in w["static"]:
            if s["batch"] in [int(x) for x in args.batches.split(",")]:
                rows.append(batch_row(be, "static", s["prompts"], s["out"]))
    if "decode" in only:
        d = w["decode"]
        rows.append(batch_row(be, "decode", d["prompts"], d["out"]))
    if "prefill" in only:
        for p in w["prefill"]:
            rows.append(batch_row(be, "prefill", p["prompts"], 1))
    if "continuous" in only:
        c = w["continuous"][:args.requests]
        prompts, outs = [r["prompt"] for r in c], [r["out"] for r in c]
        be.generate(prompts[:16], [4] * 16)  # warm-up
        lat = None
        if hasattr(be, "generate_latency"):
            t0 = time.perf_counter()
            lat = be.generate_latency(prompts, outs)
            dt = time.perf_counter() - t0
        else:
            dt = timed(lambda: be.generate(prompts, outs))
        row = {"backend": be.name, "kind": "continuous", "requests": len(c),
               "prompt_tokens": sum(map(len, prompts)), "generated": sum(outs), "total_s": dt,
               "out_tok_s": sum(outs) / dt,
               "total_tok_s": (sum(outs) + sum(map(len, prompts))) / dt}
        if lat:
            row.update(latency_summary(lat, outs))
        if hasattr(be, "preemptions"):
            row["preemptions"] = be.preemptions
        if hasattr(be, "eng"):
            row["preemptions"] = be.eng.stats.preempted
            row["recomputed_tokens"] = be.eng.stats.recomputed_tokens
        print(json.dumps(row), file=sys.stderr, flush=True)
        rows.append(row)
    if "parity" in only:
        p = w["parity"]
        toks = be.generate(p["prompts"], [p["out"]] * len(p["prompts"]), keep=True)
        rows.append({"backend": be.name, "kind": "parity", "tokens": toks})
    for r in rows:
        r["config"] = be.config
    with open(args.out, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}", file=sys.stderr)
    return 0


# ---- report ------------------------------------------------------------------------------

def first_divergence(a: list[int], b: list[int]) -> int:
    pairs = enumerate(zip(a, b, strict=False))
    return next((i for i, (x, y) in pairs if x != y), min(len(a), len(b)))


def report(args) -> int:
    rows = []
    for f in args.rows:
        rows += [json.loads(x) for x in Path(f).read_text().splitlines() if x.strip()]
    backends = list(dict.fromkeys(r["backend"] for r in rows))
    out = {"backends": {}, "rows": [r for r in rows if r["kind"] != "parity"], "parity": {}}
    for r in rows:
        out["backends"].setdefault(r["backend"], r.get("config"))

    def fmt(r, k, f="{:,.1f}"):
        return f.format(r[k]) if r and k in r else ""

    by = {(r["backend"], r["kind"], r.get("batch"), r.get("prompt")): r for r in rows}
    print("| workload | " + " | ".join(backends) + " |")
    print("|---|" + "---|" * len(backends))
    keys = sorted({(r["kind"], r.get("batch"), r.get("prompt"), r.get("out")) for r in rows
                   if r["kind"] in ("static", "decode", "prefill")},
                  key=lambda k: (["decode", "static", "prefill"].index(k[0]), k[1], k[2]))
    for kind, b, p, o in keys:
        cells = []
        for be in backends:
            r = by.get((be, kind, b, p))
            if r is None:
                cells.append("")
            elif kind == "prefill":
                cells.append(f"TTFT {fmt(r, 'ttft_ms')} ms")
            else:
                cells.append(f"{fmt(r, 'out_tok_s', '{:,.0f}')} tok/s, "
                             f"ITL {fmt(r, 'itl_ms', '{:.2f}')} ms, "
                             f"TTFT {fmt(r, 'ttft_ms', '{:,.0f}')} ms")
        print(f"| {kind} B={b} {p}/{o} | " + " | ".join(cells) + " |")
    for be in backends:
        r = by.get((be, "continuous", None, None))
        if r:
            print(f"continuous {be}: {r['requests']} requests in {r['total_s']:.2f} s, "
                  f"{r['out_tok_s']:,.0f} generated tok/s")
    par = {r["backend"]: r["tokens"] for r in rows if r["kind"] == "parity"}
    if "vllm" in par:
        for be, toks in par.items():
            if be == "vllm":
                continue
            div = [first_divergence(a, b) for a, b in zip(toks, par["vllm"], strict=True)]
            out["parity"][f"{be}_vs_vllm"] = div
            print(f"parity {be} vs vllm, first differing token per prompt: {div}")
    out["parity_tokens"] = par
    for f in args.profile or []:
        out.setdefault("profile", []).extend(
            json.loads(x) for x in Path(f).read_text().splitlines() if x.strip())
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"wrote {args.out}", file=sys.stderr)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("workload")
    a.add_argument("model")
    a.add_argument("--requests", type=int, default=256)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default=str(WORKLOAD))
    a = sub.add_parser("run")
    a.add_argument("backend", choices=["spark", "vllm", "hf", "hf_static", "hf_cb"])
    a.add_argument("model")
    a.add_argument("--workload", default=str(WORKLOAD))
    a.add_argument("--out", required=True)
    a.add_argument("--only", default="static,decode,prefill,continuous,parity")
    a.add_argument("--batches", default=",".join(map(str, STATIC_B)))
    a.add_argument("--requests", type=int, default=256)
    a.add_argument("--max-batch", type=int, default=256, help="slots (ours), max_num_seqs")
    a.add_argument("--cache-gb", type=float, default=8.5, help="our K/V cache")
    a.add_argument("--prefill-tokens", type=int, default=8192, help="our prefill budget")
    a.add_argument("--vllm-mem", type=float, default=0.85)
    a = sub.add_parser("report")
    a.add_argument("rows", nargs="+")
    a.add_argument("--profile", nargs="*", help="profile_llm.py output to keep alongside")
    a.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    if args.cmd == "workload":
        w = make_workload(args.model, args.requests, args.seed)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(w))
        print(f"wrote {args.out}", file=sys.stderr)
        return 0
    return run(args) if args.cmd == "run" else report(args)


if __name__ == "__main__":
    sys.exit(main())
