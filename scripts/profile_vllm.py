#!/usr/bin/env python3
"""vLLM's decode step, kernel by kernel, to set beside profile_llm.py's numbers for our
engine: the static batches of bench_llm.py (B = 1, 8, 64, 64 tokens out) under nsys, each in
an NVTX range, then the GPU time per kernel family per decode step from the trace.

    # in vLLM's venv, the engine core in this process so nsys sees its kernels
    VLLM_ENABLE_V1_MULTIPROCESSING=0 nsys profile -o vllm_prof --trace=cuda,nvtx \\
        --cuda-graph-trace=node --sample=none ~/vllm-env/bin/python \\
        scripts/profile_vllm.py run MODEL
    nsys export --type sqlite vllm_prof.nsys-rep     # or nsys stats, which writes it too
    python scripts/profile_vllm.py steps vllm_prof.sqlite

A decode step ends with one `_gumbel_sample_kernel` launch (vLLM's sampler, greedy here), so
the last 64 of them in a range delimit the 63 decode steps; a chunked prefill adds marks
before them.
"""

from __future__ import annotations

import argparse
import collections
import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKLOAD = ROOT / "results" / "llm_workload.json"
OUT_TOKENS = 64


def run(args) -> int:
    import torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    w = json.loads(Path(args.workload).read_text())
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=args.vllm_mem, enable_prefix_caching=False, seed=0)
    sp = SamplingParams(temperature=0.0, max_tokens=OUT_TOKENS, ignore_eos=True,
                        detokenize=False)
    for b in (1, 8, 64):
        s = next(x for x in w["static"] if x["batch"] == b)
        ps = [TokensPrompt(prompt_token_ids=p) for p in s["prompts"]]
        llm.generate(ps, sp, use_tqdm=False)
        torch.cuda.nvtx.range_push(f"B{b}")
        t0 = time.perf_counter()
        llm.generate(ps, sp, use_tqdm=False)
        print(f"B={b}: {time.perf_counter() - t0:.3f} s", file=sys.stderr)
        torch.cuda.nvtx.range_pop()
    return 0


def family(name: str) -> str:
    n = name.lower()
    for keys, fam in ((("flash", "attn"), "attention"), (("gemm", "gemv", "splitkreduce"), "gemm"),
                      (("rms_norm",), "rmsnorm"), (("silu",), "silu"),
                      (("rotary", "rope"), "rope"), (("cache",), "kv write"),
                      (("sample", "argmax"), "sampling")):
        if any(k in n for k in keys):
            return fam
    return "other"


def steps(args) -> int:
    db = sqlite3.connect(args.sqlite)
    names = dict(db.execute("select id, value from StringIds"))
    ranges = []
    for s, e, t, tid in db.execute("select start, end, text, textId from NVTX_EVENTS"):
        t = t or names.get(tid) or ""
        if t.startswith("B") and e:
            ranges.append((t, s, e))
    kernels = db.execute("select start, end, coalesce(demangledName, shortName) "
                         "from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
    for label, s, e in ranges:
        ks = [k for k in kernels if k[0] >= s and k[1] <= e]
        marks = [k for k in ks if "_gumbel_sample_kernel" in names.get(k[2], "")]
        marks = marks[-OUT_TOKENS:]
        t0, t1, n = marks[0][1], marks[-1][1], len(marks) - 1
        fam: dict[str, float] = collections.defaultdict(float)
        win = [k for k in ks if k[0] >= t0 and k[1] <= t1]
        for a, b, name in win:
            fam[family(names.get(name, ""))] += b - a
        row = {"label": f"vllm decode {label}", "steps": n, "wall_ms": (t1 - t0) / n / 1e6,
               "kernel_ms": sum(fam.values()) / n / 1e6, "launches": len(win) / n,
               "families_ms": {k: v / n / 1e6 for k, v in
                               sorted(fam.items(), key=lambda x: -x[1])}}
        print(json.dumps(row))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("run")
    a.add_argument("model")
    a.add_argument("--workload", default=str(WORKLOAD))
    a.add_argument("--vllm-mem", type=float, default=0.85)
    a = sub.add_parser("steps")
    a.add_argument("sqlite")
    args = ap.parse_args()
    return run(args) if args.cmd == "run" else steps(args)


if __name__ == "__main__":
    sys.exit(main())
