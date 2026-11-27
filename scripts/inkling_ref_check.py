#!/usr/bin/env python3
"""The pure-torch Inkling reference (python/spark_kernels/inkling.py) against HF's
Inkling2LoopForCausalLM on the real checkpoint, both in bf16 on one GPU.

    python scripts/inkling_ref_check.py [--prompt "..."] [--decode 16] [--layers]
        [--lock ~/sk26/gpu.lock] [--device cuda]

HF loads the checkpoint (it dequantizes the NVFP4 and MXFP8 experts to bf16 on load); the
reference is built on the meta device and takes HF's tensors with load_state_dict(strict=True,
assign=True), so both run on the same storage and the GPU holds one copy (48.6 GiB). It
compares the prompt's logits (max and mean |difference|, relative error, top-1 agreement,
KL) against HF with its default grouped_mm experts and with its per-expert loop, and HF's
two paths against each other; it times one prefill of each (after a warm-up, under the GPU
lock if one is given). --layers prints the difference after each of the 48 applications.
--decode N decodes N greedy tokens through the reference's cache and HF's, and compares each
against the prefill of the whole sequence and against each other. The numbers are in
docs/design/inkling.md.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import importlib.util
import sys
import time
from pathlib import Path

import torch

_PATH = Path(__file__).resolve().parent.parent / "python" / "spark_kernels" / "inkling.py"
_spec = importlib.util.spec_from_file_location("spark_inkling", _PATH)
ink = importlib.util.module_from_spec(_spec)
sys.modules["spark_inkling"] = ink
_spec.loader.exec_module(ink)

PROMPT = ("The short causal convolution in front of each attention layer mixes the last four "
          "tokens of the key and value projections. Explain in two sentences why a model would")


@contextlib.contextmanager
def gpu_lock(path):
    if not path:
        yield
        return
    with open(Path(path).expanduser(), "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def timed(fn, reps=1):
    sync = torch.cuda.synchronize if torch.cuda.is_available() else (lambda: None)
    sync()
    t = time.perf_counter()
    for _ in range(reps):
        out = fn()
    sync()
    return out, (time.perf_counter() - t) / reps


def report(name, a, b):
    """a against b (the side taken as the reference): max and mean |logit difference|, the
    median of the per-position relative error ||a - b|| / ||b||, top-1 agreement and the
    mean KL(b || a) of the next-token distributions, all in fp32."""
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    rel = ((a - b).norm(dim=-1) / b.norm(dim=-1)).flatten()
    top1 = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    lb, la = torch.log_softmax(b, -1), torch.log_softmax(a, -1)
    kl = (lb.exp() * (lb - la)).sum(-1)
    print(f"{name}: max |logit diff| {diff.max().item():.4f}, mean {diff.mean().item():.5f}, "
          f"rel err median {rel.median().item() * 100:.2f}%, top-1 agreement "
          f"{top1 * 100:.1f}% over {a.shape[1]} positions, KL mean {kl.mean().item():.2e} "
          f"max {kl.max().item():.2e} (max |logit| {b.abs().max().item():.2f})")


def layer_diffs(hf, ref, ids):
    """Max |difference| of each of the 48 layer outputs (and the routers' picks)."""
    outs = {"hf": [], "ref": []}

    def hook(store):
        return lambda m, a, o: store.append(o[0] if isinstance(o, tuple) else o)

    hs = []
    for i in range(len(hf.model.layers)):
        hs.append(hf.model.layers[i].register_forward_hook(hook(outs["hf"])))
        hs.append(ref.model.layers[i].register_forward_hook(hook(outs["ref"])))
    with torch.no_grad():
        hf(input_ids=ids, use_cache=False)
        ref(ids)
    for h in hs:
        h.remove()
    # the hooks fire in call order, which is the 48-application schedule for both
    sched = ref.cfg.schedule()
    for j, (a, b) in enumerate(zip(outs["hf"], outs["ref"], strict=True)):
        d = (a.float() - b.float()).abs()
        print(f"  application {j:2d} (layer {sched[j][0]:2d}, pass {sched[j][1] + 1}): "
              f"max {d.max().item():.4f} mean {d.mean().item():.5f} "
              f"|out| mean {a.float().abs().mean().item():.3f}")


def route_flips(hf, ref, ids):
    """For each of the 48 applications: how many tokens pick a different set of 8 experts
    under HF's grouped_mm experts than under its per-expert loop."""
    picks = {}
    for mode in ("grouped_mm", "eager"):
        hf.set_experts_implementation(mode)
        store = []
        hs = [layer.mlp.gate.register_forward_hook(
            lambda m, a, o, store=store: store.append(o[2].sort(-1).values))
            for layer in hf.model.layers]
        with torch.no_grad():
            hf(input_ids=ids, use_cache=False)
        for h in hs:
            h.remove()
        picks[mode] = store
    sched = ref.cfg.schedule()
    flips = [(a != b).any(-1).sum().item()
             for a, b in zip(picks["grouped_mm"], picks["eager"], strict=True)]
    first = next((j for j, f in enumerate(flips) if f), None)
    print(f"expert sets that differ, grouped_mm vs eager: {sum(flips)} of "
          f"{len(flips) * ids.shape[1]} (token, application) pairs; first at application "
          f"{first} (layer {sched[first][0] if first is not None else '-'})")
    print("  per application:", flips)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--decode", type=int, default=0)
    ap.add_argument("--lock", default="")
    ap.add_argument("--device", default="cuda", help="cuda, or cpu when the GPU is full")
    ap.add_argument("--layers", action="store_true", help="per-application differences")
    ap.add_argument("--routes", action="store_true", help="expert picks, HF's two paths")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, Inkling2LoopForCausalLM

    try:
        path = snapshot_download(args.repo, local_files_only=True)
    except Exception:  # a snapshot missing a file the model does not need (an eval yaml)
        from huggingface_hub.constants import HF_HUB_CACHE

        name = "models--" + args.repo.replace("/", "--")
        path = str(sorted(Path(HF_HUB_CACHE, name, "snapshots").iterdir())[-1])
    t = time.perf_counter()
    # loaded on the CPU and moved: device_map needs accelerate, which the venv may not have
    hf = Inkling2LoopForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    hf = hf.to(args.device).eval()
    gib = sum(t.numel() * t.element_size() for t in hf.state_dict().values()) / 2**30
    print(f"HF load {time.perf_counter() - t:.0f} s, {gib:.1f} GiB of weights on {args.device}")
    experts = hf.model.layers[0].mlp.experts.gate_up_proj
    print(f"experts as loaded: {experts.dtype} {tuple(experts.shape)} "
          f"(layer 0 is NVFP4 in the checkpoint), layer 23: "
          f"{hf.model.layers[23].mlp.experts.down_proj.dtype}")

    cfg = ink.InklingConfig.from_hf(path)
    with torch.device("meta"):
        ref = ink.InklingForCausalLM(cfg)
    ref.load_state_dict(hf.state_dict(), strict=True, assign=True)
    ref.eval()
    shared = all(a.data_ptr() == b.data_ptr()
                 for a, b in zip(ref.state_dict().values(), hf.state_dict().values(), strict=True))
    print(f"reference loaded strict, sharing HF's storage: {shared}")

    tok = AutoTokenizer.from_pretrained(path)
    ids = tok(args.prompt, return_tensors="pt").input_ids.to(args.device)
    print(f"prompt: {ids.shape[1]} tokens")

    with torch.no_grad():
        hf(input_ids=ids, use_cache=False)  # warm-up
        ref(ids)
        with gpu_lock(args.lock):
            want, t_hf = timed(lambda: hf(input_ids=ids, use_cache=False).logits)
            got, t_ref = timed(lambda: ref(ids))
    report("reference vs HF (grouped_mm experts, HF's default)", got, want)
    print(f"prefill wall time: HF {t_hf * 1e3:.0f} ms, reference {t_ref * 1e3:.0f} ms")
    print("greedy next token:", repr(tok.decode(want[0, -1].argmax().item())),
          repr(tok.decode(got[0, -1].argmax().item())))

    # HF's per-expert loop runs the same GEMMs as the reference; HF against itself with its
    # two expert paths is the size of the bf16 noise alone
    hf.set_experts_implementation("eager")
    with torch.no_grad():
        eager = hf(input_ids=ids, use_cache=False).logits
    report("HF eager experts vs HF grouped_mm", eager, want)
    report("reference vs HF eager experts", got, eager)
    if args.layers:
        layer_diffs(hf, ref, ids)
    if args.routes:
        route_flips(hf, ref, ids)
        hf.set_experts_implementation("eager")

    if args.decode:
        decode_check(hf, ref, cfg, tok, ids, got, args)


def decode_check(hf, ref, cfg, tok, ids, logits, args):
    """N greedy tokens (picked by the reference's prefill and decode), then the logits of
    positions L..L+N-1 four ways: the reference and HF each decoding through their own cache
    (after a prefill of the prompt into it), and each prefilling the whole sequence."""
    n, L = args.decode, ids.shape[1]
    with torch.no_grad():
        seq = ids
        cache = ink.InklingCache(cfg)
        nxt = ref(ids, cache)[:, -1].argmax(-1, keepdim=True)
        for _ in range(n):  # the tokens, from the reference's own decode
            seq = torch.cat([seq, nxt], dim=1)
            nxt = ref.decode_step(nxt[:, 0], cache).argmax(-1, keepdim=True)
        with gpu_lock(args.lock):
            cache = ink.InklingCache(cfg)
            ref(ids, cache)
            ref_steps, t_ref = timed(lambda: torch.stack(
                [ref.decode_step(seq[:, L + i], cache) for i in range(n)], dim=1))
            out = hf(input_ids=ids, use_cache=True)
            past = out.past_key_values

            def hf_decode():
                nonlocal past
                steps = []
                for i in range(n):
                    o = hf(input_ids=seq[:, L + i:L + i + 1], past_key_values=past, use_cache=True)
                    past = o.past_key_values
                    steps.append(o.logits[:, -1])
                return torch.stack(steps, dim=1)

            hf_steps, t_hf = timed(hf_decode)
        ref_full = ref(seq)[:, L:L + n]
        hf_full = hf(input_ids=seq, use_cache=False).logits[:, L:L + n]
    print(f"decode {n} tokens: {repr(tok.decode(seq[0, L:].tolist()))}")
    report("reference decode vs reference prefill", ref_steps, ref_full)
    report("HF decode vs HF prefill", hf_steps, hf_full)
    report("reference decode vs HF decode", ref_steps, hf_steps)
    print(f"decode step wall time (prompt {L}, batch 1): HF with eager experts "
          f"{t_hf / n * 1e3:.0f} ms, reference {t_ref / n * 1e3:.0f} ms; "
          f"reference cache {cache.nbytes() / 2**20:.1f} MiB")

if __name__ == "__main__":
    main()
