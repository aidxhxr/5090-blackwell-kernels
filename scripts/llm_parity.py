#!/usr/bin/env python3
"""Numerical parity of the engine against Hugging Face transformers on a real checkpoint, with
an fp32 reference to say which of the two bf16 stacks is closer to the truth.

Four backends see the same token sequences:

    spark   engine.SparkModel, this package's kernels, bf16
    torch   engine.TorchModel, the same model in plain PyTorch, bf16
    hf      transformers' LlamaForCausalLM, bf16, sdpa attention
    ref     the same math in fp32 on the bf16 weights (TF32 off, SDPA's math backend),
            streamed one layer at a time so it fits next to nothing else on the GPU

Only one model is on the GPU at a time, so each backend is its own stage and writes its
logits, its residual stream after every layer and its generations to --work; `report` reads
them back on the CPU (no GPU) and writes the tables and results/llm_parity.json.

    python scripts/llm_parity.py MODEL hf       # greedy generations, then logits + hidden
    python scripts/llm_parity.py MODEL spark    # the same on the engine, and its margins
    python scripts/llm_parity.py MODEL torch    # logits + hidden only
    python scripts/llm_parity.py MODEL ref      # logits + hidden + margins, fp32
    python scripts/llm_parity.py MODEL report

hf runs first: the logit set is the first --chats chat prompts followed by hf's greedy reply
(so the scored positions are a model predicting its own reply), plus two wikitext-2 windows of
2048 tokens. The generation check runs every prompt for --new tokens greedily on the engine (one
packed prefill, then CUDA-graph decode, all prompts in one batch and again one at a time) and
with transformers' generate (one prompt at a time), and finds the first position where the
two disagree; spark and ref then score that common prefix to say how close the tie was.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

TEXT = Path.home() / "models" / "wikitext2_test.txt"
WIKI_CTX = 2048
TOPK = 5

PROMPTS = [
    "Explain how a CPU cache hierarchy works, from L1 to main memory.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "What were the main causes of the French Revolution? Answer in detail.",
    "Write a Python function that parses a CSV line without the csv module, and explain it.",
    "Compare TCP and UDP and give examples of when to use each.",
    "Describe the process of photosynthesis step by step.",
    "Give me a seven-day meal plan for a vegetarian runner.",
    "Explain the difference between a mutex and a semaphore, with code examples in C.",
    "Summarize the plot of Hamlet.",
    "How does public key cryptography work? Explain RSA with a small numeric example.",
    "Write a cover letter for a junior data analyst position.",
    "What is the derivative of x^x? Show every step.",
    "Explain gradient descent to a high school student.",
    "List ten interesting facts about octopuses.",
    "Translate into French and explain the grammar: 'I would have gone if I had known.'",
    "Write four haiku, one per season, then explain the imagery of each.",
]


# ---- inputs ----------------------------------------------------------------------------------

def tokenizer(model):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def chat_ids(tok, prompt: str) -> torch.Tensor:
    x = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                add_generation_prompt=True, return_tensors="pt")
    x = x["input_ids"] if hasattr(x, "keys") else x
    return x[0].long()


def wiki_windows(tok) -> list[torch.Tensor]:
    """Window 0 (which starts with BOS) and the middle window of the ppl split."""
    ids = tok(TEXT.read_text(), return_tensors="pt").input_ids[0]
    n = ids.numel() // WIKI_CTX
    return [ids[i * WIKI_CTX:(i + 1) * WIKI_CTX] for i in (0, n // 2)]


def gen_prompts(tok) -> list[torch.Tensor]:
    return [chat_ids(tok, p) for p in PROMPTS]


def logit_set(work: Path) -> tuple[list[torch.Tensor], list[dict]]:
    meta = json.loads((work / "meta.json").read_text())
    return [torch.tensor(s["ids"]) for s in meta], meta


# ---- storage: .npy files read back with mmap -----------------------------------------------------

def npy_out(work: Path, name: str, shape, dtype: torch.dtype):
    np_dtype = {torch.bfloat16: np.int16, torch.float32: np.float32}[dtype]
    suffix = {torch.bfloat16: "bf16", torch.float32: "f32"}[dtype]
    return np.lib.format.open_memmap(work / f"{name}.{suffix}.npy", mode="w+", dtype=np_dtype,
                                     shape=tuple(shape))


def put(arr, index, t: torch.Tensor) -> None:
    t = t.detach().cpu()
    arr[index] = (t.view(torch.int16) if t.dtype == torch.bfloat16 else t).numpy()


def npy_in(work: Path, name: str):
    """(memmapped array, function from a slice of it to an fp32 tensor)."""
    for suffix in ("bf16", "f32"):
        p = work / f"{name}.{suffix}.npy"
        if p.exists():
            arr = np.load(p, mmap_mode="r")
            if suffix == "bf16":
                return arr, lambda a: torch.from_numpy(np.array(a)).view(torch.bfloat16).float()
            return arr, lambda a: torch.from_numpy(np.array(a))
    raise FileNotFoundError(work / name)


def save_json(work: Path, name: str, obj) -> None:
    (work / f"{name}.json").write_text(json.dumps(obj))


def load_json(work: Path, name: str):
    return json.loads((work / f"{name}.json").read_text())


def topk_row(row: torch.Tensor, extra: list[int]) -> dict:
    v, i = row.float().topk(TOPK)
    return dict(top_ids=i.tolist(), top_vals=v.tolist(),
                extra={str(t): row[t].float().item() for t in extra})


def divergences(work: Path) -> list[dict]:
    """Per generation prompt: hf's tokens, the engine's (batched), and the first position
    where they differ within hf's length (None if they agree all the way)."""
    hfg, spg = load_json(work, "gen_hf"), load_json(work, "gen_spark")
    out = []
    for i, (h, s) in enumerate(zip(hfg["tokens"], spg["batched"], strict=True)):
        d = next((j for j in range(len(h)) if h[j] != s[j]), None)
        out.append(dict(prompt=i, d=d, hf=h, spark=s))
    return out


def prefixes(work: Path, prompts: list[torch.Tensor]) -> list[tuple[int, torch.Tensor, int, int]]:
    """(prompt index, prompt + common tokens, hf's next token, the engine's next token) for
    every prompt that diverges."""
    out = []
    for dv in divergences(work):
        d = dv["d"]
        if d is None:
            continue
        p = torch.cat([prompts[dv["prompt"]], torch.tensor(dv["hf"][:d], dtype=torch.long)])
        out.append((dv["prompt"], p, dv["hf"][d], dv["spark"][d]))
    return out


# ---- hf ----------------------------------------------------------------------------------------

def stage_hf(args, work: Path) -> None:
    from transformers import AutoModelForCausalLM

    tok = tokenizer(args.model)
    m = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                             attn_implementation="sdpa").cuda().eval()
    gen = dict(tokens=[], top_ids=[], top_vals=[], seconds=0.0)
    t0 = time.perf_counter()
    for x in gen_prompts(tok):
        with torch.no_grad():
            out = m.generate(x[None].cuda(), max_new_tokens=args.new, do_sample=False,
                             temperature=None, top_p=None, output_logits=True,
                             return_dict_in_generate=True, pad_token_id=tok.eos_token_id)
        lg = torch.stack(out.logits)[:, 0].float()
        v, i = lg.topk(TOPK, dim=-1)
        gen["tokens"].append(out.sequences[0, x.numel():].tolist())
        gen["top_ids"].append(i.tolist())
        gen["top_vals"].append(v.tolist())
    gen["seconds"] = time.perf_counter() - t0
    save_json(work, "gen_hf", gen)
    print(f"hf generate: {sum(map(len, gen['tokens']))} tokens in {gen['seconds']:.1f} s")

    meta = []
    for k, (p, x) in enumerate(zip(PROMPTS[:args.chats], gen_prompts(tok), strict=False)):
        ids = torch.cat([x, torch.tensor(gen["tokens"][k])])
        meta.append(dict(kind="chat", name=p, ids=ids.tolist(), start=x.numel() - 1))
    for k, w in enumerate(wiki_windows(tok)):
        meta.append(dict(kind="wiki", name=f"wikitext window {k}", ids=w.tolist(), start=0))
    save_json(work, "meta", meta)

    seqs, _ = logit_set(work)
    T = sum(s.numel() for s in seqs)
    L = m.config.num_hidden_layers
    logits = npy_out(work, "logits_hf", (T, m.config.vocab_size), torch.bfloat16)
    hidden = npy_out(work, "hidden_hf", (L, T, m.config.hidden_size), torch.bfloat16)
    caught: list[torch.Tensor] = []
    hooks = [layer.register_forward_hook(
        lambda mod, inp, out: caught.append(out[0] if isinstance(out, tuple) else out))
        for layer in m.model.layers]
    t0 = 0
    for s in seqs:
        caught.clear()
        with torch.no_grad():
            lg = m(s[None].cuda()).logits[0]
        n = s.numel()
        put(logits, slice(t0, t0 + n), lg.to(torch.bfloat16))
        for layer, h in enumerate(caught):
            put(hidden, (layer, slice(t0, t0 + n)), h.reshape(n, -1).to(torch.bfloat16))
        t0 += n
    for h in hooks:
        h.remove()
    logits.flush()
    hidden.flush()
    print(f"hf logits: {len(seqs)} sequences, {T} positions")


# ---- the engine --------------------------------------------------------------------------------

def scoring_engine(E, model, n_layers, sets):
    """An idle engine whose one packed prefill fits the largest of `sets` of sequences."""
    pages = max(sum(E.cdiv(s.numel() + 1, E.PAGE) for s in ss) for ss in sets)
    toks = max(sum(s.numel() for s in ss) for ss in sets)
    longest = max(s.numel() for ss in sets for s in ss)
    return E.Engine(model, n_layers, max_batch=max(len(ss) for ss in sets),
                    max_seq=longest + 1, num_pages=pages + 1, prefill_tokens=toks, graphs=False)


def stage_engine(args, work: Path) -> None:
    from spark_kernels import engine as E
    from spark_kernels import hf

    tok = tokenizer(args.model)
    w, cfg = hf.load(args.model)
    rope = cfg.rope(8192)
    spark = args.stage == "spark"
    model = E.SparkModel(w, rope) if spark else E.TorchModel(w, rope)
    prompts = gen_prompts(tok)

    if spark:
        # greedy generation: every prompt in one batch, then each prompt alone
        eng = E.Engine(model, cfg.n_layers, max_batch=len(prompts), max_seq=4096,
                       cache_bytes=2 << 30, graphs=True, log_tokens=True)
        gen = {}
        for mode in ("batched", "single"):
            outs = []
            t0 = time.perf_counter()
            if mode == "batched":
                sids = [eng.submit(p, args.new).sid for p in prompts]
                eng.run()
                res = eng.outputs()
                outs = [res[s] for s in sids]
            else:
                for p in prompts:
                    sid = eng.submit(p, args.new).sid
                    eng.run()
                    outs.append(eng.outputs()[sid])
            gen[mode] = outs
            gen[mode + "_seconds"] = time.perf_counter() - t0
        save_json(work, "gen_spark", gen)
        print(f"spark generate: batched {gen['batched_seconds']:.1f} s, "
              f"single {gen['single_seconds']:.1f} s")
        del eng
        torch.cuda.empty_cache()

    seqs, _ = logit_set(work)
    pre = prefixes(work, prompts) if spark else []
    eng = scoring_engine(E, model, cfg.n_layers, [seqs] + ([[p for _, p, _, _ in pre]] if pre
                                                          else []))
    T = sum(s.numel() for s in seqs)
    model.trace = []
    with torch.no_grad():
        lg = torch.cat(eng.prompt_logits([s.cuda() for s in seqs]))
    trace, model.trace = model.trace, None
    name = args.stage
    logits = npy_out(work, f"logits_{name}", (T, cfg.vocab), torch.bfloat16)
    for r in range(0, T, 1024):
        put(logits, slice(r, r + 1024), lg[r:r + 1024])
    hidden = npy_out(work, f"hidden_{name}", (len(trace), T, 4096), torch.bfloat16)
    for layer, h in enumerate(trace):
        put(hidden, layer, h)
    logits.flush()
    hidden.flush()
    del lg, trace
    print(f"{name} logits: {len(seqs)} sequences, {T} positions")

    if pre:
        with torch.no_grad():
            lgs = eng.prompt_logits([p.cuda() for _, p, _, _ in pre])
        save_json(work, "margins_spark", [dict(prompt=i, **topk_row(x[-1], [h, s]))
                                          for (i, _, h, s), x in zip(pre, lgs, strict=True)])


# ---- the fp32 reference -------------------------------------------------------------------------

def ref_forward(model_path: str, seqs: list[torch.Tensor], on_layer=None) -> torch.Tensor:
    """fp32 forward of the checkpoint over each sequence (positions from 0), one layer's
    weights on the GPU at a time. `on_layer(i, x)` sees the residual stream after layer i.
    Returns the final normed hidden states [T, 4096] in fp32."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from spark_kernels import hf

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    cfg = hf.ModelConfig.from_dir(model_path)
    t = hf._Tensors(Path(model_path), "cuda")
    rope = cfg.rope(8192)
    ids = torch.cat(seqs).cuda()
    lens = [s.numel() for s in seqs]
    pos = torch.cat([torch.arange(n) for n in lens]).cuda()
    cos, sin = rope.cos[pos][:, None], rope.sin[pos][:, None]

    def rms(x, w):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + cfg.eps) * w

    def rot(x):  # [T, H, 128], rotate-half
        return x * cos + torch.cat([-x[..., 64:], x[..., :64]], dim=-1) * sin

    x = F.embedding(ids, t["model.embed_tokens.weight"]).float()
    T = x.shape[0]
    for i in range(cfg.n_layers):
        p = f"model.layers.{i}."

        def g(name, p=p):
            return t[p + name].float()

        h = rms(x, g("input_layernorm.weight"))
        q = rot(F.linear(h, g("self_attn.q_proj.weight")).view(T, 32, 128))
        k = rot(F.linear(h, g("self_attn.k_proj.weight")).view(T, 8, 128))
        v = F.linear(h, g("self_attn.v_proj.weight")).view(T, 8, 128)
        outs, t0 = [], 0
        with sdpa_kernel(SDPBackend.MATH):
            for n in lens:
                qs, ks, vs = (z[t0:t0 + n].transpose(0, 1)[None] for z in (q, k, v))
                o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, enable_gqa=True)
                outs.append(o[0].transpose(0, 1).reshape(n, 4096))
                t0 += n
        x = x + F.linear(torch.cat(outs), g("self_attn.o_proj.weight"))
        h = rms(x, g("post_attention_layernorm.weight"))
        a = F.silu(F.linear(h, g("mlp.gate_proj.weight"))) * F.linear(h, g("mlp.up_proj.weight"))
        x = x + F.linear(a, g("mlp.down_proj.weight"))
        del q, k, v, outs, h, a
        if on_layer is not None:
            on_layer(i, x)
    return rms(x, t["model.norm.weight"].float())


def ref_logits(model_path: str, xn: torch.Tensor, rows=None):
    """lm_head in fp32 over the rows of xn, a chunk at a time: yields (start, logits)."""
    from spark_kernels import hf

    head = hf._Tensors(Path(model_path), "cuda")["lm_head.weight"].float()
    rows = range(xn.shape[0]) if rows is None else rows
    rows = list(rows)
    for r in range(0, len(rows), 512):
        idx = torch.tensor(rows[r:r + 512], device=xn.device)
        yield r, F.linear(xn[idx], head)


def stage_ref(args, work: Path) -> None:
    from spark_kernels import hf

    cfg = hf.ModelConfig.from_dir(args.model)
    seqs, _ = logit_set(work)
    T = sum(s.numel() for s in seqs)
    hidden = npy_out(work, "hidden_ref", (cfg.n_layers, T, 4096), torch.float32)
    t0 = time.perf_counter()
    with torch.no_grad():
        xn = ref_forward(args.model, seqs, on_layer=lambda i, x: put(hidden, i, x))
        hidden.flush()
        logits = npy_out(work, "logits_ref", (T, cfg.vocab), torch.float32)
        for r, lg in ref_logits(args.model, xn):
            put(logits, slice(r, r + lg.shape[0]), lg)
        logits.flush()
    print(f"ref logits: {T} positions in {time.perf_counter() - t0:.1f} s")
    del xn
    torch.cuda.empty_cache()

    pre = prefixes(work, gen_prompts(tokenizer(args.model)))
    if pre:
        with torch.no_grad():
            xn = ref_forward(args.model, [p for _, p, _, _ in pre])
            ends = np.cumsum([p.numel() for _, p, _, _ in pre]) - 1
            (_, lg), = ref_logits(args.model, xn, ends.tolist())
        save_json(work, "margins_ref", [dict(prompt=i, **topk_row(x, [h, s]))
                                        for (i, _, h, s), x in zip(pre, lg, strict=True)])


# ---- report ------------------------------------------------------------------------------------

def q(x: torch.Tensor, p: float) -> float:
    return torch.quantile(x.double(), p).item() if x.numel() else float("nan")


def logit_metrics(work: Path, meta: list[dict], a: str, b: str) -> dict:
    """Per-position comparison of backend a against backend b (the reference side): logit
    relative error, top-1 agreement, KL(b || a), and the NLL of the next token under each."""
    A, fa = npy_in(work, f"logits_{a}")
    B, fb = npy_in(work, f"logits_{b}")
    ids = torch.cat([torch.tensor(s["ids"]) for s in meta])
    kind, keep, nxt = [], [], []
    for s in meta:
        n = len(s["ids"])
        kind += [s["kind"]] * n
        keep += [j >= s["start"] for j in range(n)]
        nxt += [j + 1 < n for j in range(n)]
    cols = {k: [] for k in ("rel", "agree", "kl", "margin", "tie_a", "nll_a", "nll_b")}
    for r in range(0, ids.numel(), 256):
        x, y = fa(A[r:r + 256]), fb(B[r:r + 256])
        cols["rel"].append((x - y).norm(dim=-1) / y.norm(dim=-1))
        cols["agree"].append((x.argmax(-1) == y.argmax(-1)).float())
        lx, ly = x.log_softmax(-1), y.log_softmax(-1)
        cols["kl"].append((ly.exp() * (ly - lx)).sum(-1))
        top2 = y.topk(2, dim=-1).values
        cols["margin"].append(top2[:, 0] - top2[:, 1])
        top2a = x.topk(2, dim=-1).values  # a's bf16 logits: an exact tie is a coin flip
        cols["tie_a"].append((top2a[:, 0] == top2a[:, 1]).float())
        tgt = torch.cat([ids[r + 1:r + 257], torch.zeros(1, dtype=torch.long)])[:x.shape[0]]
        cols["nll_a"].append(-lx.gather(1, tgt[:, None])[:, 0])
        cols["nll_b"].append(-ly.gather(1, tgt[:, None])[:, 0])
    c = {k: torch.cat(v) for k, v in cols.items()}
    kind_t = np.array(kind)
    keep_t, nxt_t = torch.tensor(keep), torch.tensor(nxt)
    out = {}
    for sub in ("chat", "wiki", "all"):
        m = keep_t & (torch.ones_like(keep_t) if sub == "all"
                      else torch.from_numpy(kind_t == sub))
        mn = m & nxt_t
        dis = m & (c["agree"] == 0)
        out[sub] = dict(
            positions=int(m.sum()),
            rel_median=q(c["rel"][m], 0.5), rel_p99=q(c["rel"][m], 0.99),
            rel_max=c["rel"][m].max().item(),
            top1_agree=c["agree"][m].mean().item(), disagreements=int(dis.sum()),
            margin_at_disagree_median=q(c["margin"][dis], 0.5),
            margin_at_disagree_max=c["margin"][dis].max().item() if dis.any() else None,
            margin_median=q(c["margin"][m], 0.5), top2_tie_a=c["tie_a"][m].mean().item(),
            kl_mean=c["kl"][m].mean().item(), kl_median=q(c["kl"][m], 0.5),
            kl_p99=q(c["kl"][m], 0.99), kl_max=c["kl"][m].max().item(),
            nll_a=c["nll_a"][mn].mean().item(), nll_b=c["nll_b"][mn].mean().item(),
            ppl_a=math.exp(c["nll_a"][mn].mean().item()),
            ppl_b=math.exp(c["nll_b"][mn].mean().item()))
    return out


def layer_metrics(work: Path, meta: list[dict], a: str, b: str) -> list[dict]:
    """Relative error of the residual stream after every layer, per token (first token of
    every sequence left out: its attention-sink activations are orders of magnitude larger and
    would dominate a pooled norm), plus the pooled error over all tokens."""
    A, fa = npy_in(work, f"hidden_{a}")
    B, fb = npy_in(work, f"hidden_{b}")
    first = np.zeros(A.shape[1], dtype=bool)
    t0 = 0
    for s in meta:
        first[t0] = True
        t0 += len(s["ids"])
    rest = torch.from_numpy(~first)
    out = []
    for layer in range(A.shape[0]):
        x, y = fa(A[layer]), fb(B[layer])
        err, ref = (x - y).norm(dim=-1), y.norm(dim=-1)
        rel = err / ref
        out.append(dict(layer=layer, rel_median=q(rel[rest], 0.5), rel_p90=q(rel[rest], 0.9),
                        rel_first_token=q(rel[~rest], 0.5), norm_median=q(ref[rest], 0.5),
                        norm_first_token=q(ref[~rest], 0.5),
                        rel_pooled=(err.square().sum() / ref.square().sum()).sqrt().item()))
    return out


def gen_report(work: Path, tok) -> dict:
    spg, hfg = load_json(work, "gen_spark"), load_json(work, "gen_hf")
    msp = {m["prompt"]: m for m in load_json(work, "margins_spark")} \
        if (work / "margins_spark.json").exists() else {}
    mref = {m["prompt"]: m for m in load_json(work, "margins_ref")} \
        if (work / "margins_ref.json").exists() else {}
    rows = []
    for dv in divergences(work):
        i, d, h, s = dv["prompt"], dv["d"], dv["hf"], dv["spark"]
        single = spg["single"][i]
        ds = next((j for j in range(len(s)) if s[j] != single[j]), None)
        row = dict(prompt=PROMPTS[i], hf_len=len(h), hf_stopped=len(h) < len(s), d=d,
                   batched_vs_single=ds)
        if d is not None:
            hi, ho = h[d], s[d]
            ids, vals = hfg["top_ids"][i][d], hfg["top_vals"][i][d]
            row["hf_token"], row["spark_token"] = tok.decode([hi]), tok.decode([ho])
            row["hf_gap"] = (vals[0] - vals[ids.index(ho)]) if ho in ids else None
            row["hf_margin"] = vals[0] - vals[1]
            if i in msp:
                e = msp[i]["extra"]
                row["spark_gap"] = e[str(hi)] - e[str(ho)]
            if i in mref:
                e = mref[i]["extra"]
                row["ref_gap"] = e[str(hi)] - e[str(ho)]
                top = mref[i]["top_ids"][0]
                row["ref_pick"] = "hf" if top == hi else "spark" if top == ho else "other"
            lo = max(0, d - 30)
            row["sample"] = dict(common=tok.decode(h[lo:d]), hf=tok.decode(h[d:d + 30]),
                                 spark=tok.decode(s[d:d + 30]))
        rows.append(row)
    div = [r["d"] for r in rows if r["d"] is not None]
    return dict(new=len(spg["batched"][0]), prompts=len(rows), diverged=len(div),
                d_median=float(np.median(div)) if div else None,
                batched_vs_single_differ=sum(r["batched_vs_single"] is not None for r in rows),
                hf_seconds=hfg["seconds"], spark_batched_seconds=spg["batched_seconds"],
                spark_single_seconds=spg["single_seconds"], rows=rows)


def fmt(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}g}" if abs(x) < 1e-3 or abs(x) >= 1e4 else f"{x:.{nd}f}".rstrip("0")
    return str(x)


def stage_report(args, work: Path) -> None:
    tok = tokenizer(args.model)
    _, meta = logit_set(work)
    have = [b for b in ("spark", "torch", "hf") if (work / f"logits_{b}.bf16.npy").exists()]
    pairs = [(b, "ref") for b in have] + ([("spark", "hf")] if {"spark", "hf"} <= set(have)
                                          else [])
    res = dict(model=Path(args.model).name, sequences=[dict(kind=s["kind"], name=s["name"],
               tokens=len(s["ids"]), scored=len(s["ids"]) - s["start"]) for s in meta],
               logits={}, layers={}, generation=None)
    for a, b in pairs:
        res["logits"][f"{a}_vs_{b}"] = logit_metrics(work, meta, a, b)
        res["layers"][f"{a}_vs_{b}"] = layer_metrics(work, meta, a, b)
    if (work / "gen_spark.json").exists() and (work / "gen_hf.json").exists():
        res["generation"] = gen_report(work, tok)
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")

    print("| pair | subset | positions | rel err median | p99 | top-1 agree | KL mean | "
          "KL p99 | KL max | ppl a | ppl b |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for k, v in res["logits"].items():
        for sub, m in v.items():
            print(f"| {k} | {sub} | {m['positions']} | {fmt(m['rel_median'])} | "
                  f"{fmt(m['rel_p99'])} | {100 * m['top1_agree']:.2f}% ({m['disagreements']}) | "
                  f"{fmt(m['kl_mean'])} | {fmt(m['kl_p99'])} | {fmt(m['kl_max'])} | "
                  f"{m['ppl_a']:.4f} | {m['ppl_b']:.4f} |")
    print()
    keys = list(res["layers"])
    print("| layer | " + " | ".join(keys) + " |")
    print("|---" * (len(keys) + 1) + "|")
    for layer in range(len(res["layers"][keys[0]])):
        print(f"| {layer} | " + " | ".join(
            f"{fmt(res['layers'][k][layer]['rel_median'])}" for k in keys) + " |")
    g = res["generation"]
    if g:
        print(f"\n{g['diverged']}/{g['prompts']} diverge, median at {g['d_median']}; "
              f"batched vs single differ on {g['batched_vs_single_differ']}")
        for r in g["rows"]:
            print(f"| {r['prompt'][:40]} | {r['hf_len']} | {fmt(r['d'])} | "
                  f"{fmt(r.get('hf_gap'))} | {fmt(r.get('spark_gap'))} | "
                  f"{fmt(r.get('ref_gap'))} | {r.get('ref_pick', '-')} | "
                  f"{fmt(r['batched_vs_single'])} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("stage", choices=("hf", "spark", "torch", "ref", "report"))
    ap.add_argument("--work", default=str(ROOT / "parity_work"))
    ap.add_argument("--chats", type=int, default=6, help="chat replies in the logit set")
    ap.add_argument("--new", type=int, default=256, help="greedy tokens per prompt")
    ap.add_argument("--out", default=str(ROOT / "results" / "llm_parity.json"))
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    {"hf": stage_hf, "spark": stage_engine, "torch": stage_engine, "ref": stage_ref,
     "report": stage_report}[args.stage](args, work)


if __name__ == "__main__":
    main()
