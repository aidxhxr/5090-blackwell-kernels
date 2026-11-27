#!/usr/bin/env python3
"""The Inkling 2 Mini expert repack on the real file: lossless against transformers, and the
error of one NVFP4 and one MXFP8 expert GEMM through fp4gemm / fp8gemm.

    python scripts/inkling_ckpt_check.py [PATH] [--layers 0 20] [--m 512] [--device cuda]

PATH is the safetensors file or its snapshot directory (default: the file in the HF cache,
through huggingface_hub.hf_hub_download(local_files_only=True)). For each layer and projection
(w13, w2), every expert in chunks of 16:

  * the reader's bytes against safetensors.safe_open (when installed);
  * the repacked operands dequantized (`dequantize_nvfp4` / `dequantize_mxfp8`, bf16, w13 in
    HF's [gate; up] order) against what transformers' loader computes for gate_up_proj /
    down_proj (its Inkling2LoopDequantizeExperts, then Interleave(dim=1) for w13), counting
    the bf16 elements whose bits differ.

For an MXFP8 layer, whether its experts would fit grouped_gemm's fp8 operand (one scale per
expert) without loss.

Then expert 0 of the first NVFP4 and the first MXFP8 layer: x [M, K] bf16 (Gaussian), the
activation quantized the way the repo's kernels do it (fp4_quantize with a per-tensor scale,
fp8_quantize "mx"), and the GEMM's output compared with (a) the reference GEMM of the same
quantized operands (the kernel's own error), (b) the quantized activation in fp32 times W.T,
W what transformers loads (the weight path alone: W's bf16 rounding and the kernel), and
(c) x @ W.T in bf16 (what a user of the repacked weights sees, the activation quantization
included).
Relative errors are ||out - ref|| / ||ref|| (Frobenius) and max |out - ref| / max |ref|.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from spark_kernels import inkling_ckpt as IC  # noqa: E402
from spark_kernels import reference as R  # noqa: E402

try:
    import spark_kernels as sk

    sk.num_variants("fp4gemm")
except Exception:
    sk = None


def find_file(path: str | None) -> str:
    if path:
        return path
    from huggingface_hub import hf_hub_download

    return hf_hub_download("thinkingmachines/Inkling-2-Mini", "model-00001-of-00001.safetensors",
                           local_files_only=True)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def hf_loaded(ck: IC.Checkpoint, layer: int, proj: str, rows: slice, device: str):
    """transformers' loader on these experts: its own ops on the raw tensors."""
    from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave

    p = f"model.llm.layers.{layer}.mlp.experts.{proj}_weight"
    src = [p, p + "_scale", p + "_amax", p + "_outer_scale_denominator"]
    inputs = {}
    for n in src:
        if n in ck.header:
            t = ck.tensor(n, rows if n in (p, p + "_scale") else None).to(device)
            inputs[n] = t[rows] if n == p + "_amax" else t
    w = Inkling2LoopDequantizeExperts().convert(inputs, src, ["t"])["t"]
    return Interleave(dim=1).convert({"t": w}, None, ["t"])["t"] if proj == "w13" else w


def check_layer(ck: IC.Checkpoint, layer: int, device: str, chunk: int = 16) -> None:
    try:
        import safetensors
    except ImportError:
        safetensors = None
    for proj in ("w13", "w2"):
        p = f"model.llm.layers.{layer}.mlp.experts.{proj}_weight"
        E = ck.shape(p)[0]
        diff = total = reader_diff = 0
        for start in range(0, E, chunk):
            rows = slice(start, min(start + chunk, E))
            ex = ck.experts(layer, proj, rows).to(device)
            deq = IC.dequantize_nvfp4(ex) if ex.fmt == "nvfp4" else IC.dequantize_mxfp8(ex)
            deq = deq.to(torch.bfloat16)
            if proj == "w13":
                deq = IC.hf_gate_up(deq)
            hf = hf_loaded(ck, layer, proj, rows, device)
            diff += (bits(deq) != bits(hf)).sum().item()
            total += hf.numel()
            if safetensors is not None and start == 0:
                with safetensors.safe_open(str(ck.path), framework="pt") as f:
                    for n in (p, p + "_scale"):
                        a = f.get_slice(n)[rows].contiguous().view(torch.uint8)
                        reader_diff += (a != ck.tensor(n, rows).view(torch.uint8)).sum().item()
        fmt = "nvfp4" if p + "_amax" in ck.header else "mxfp8"
        print(f"layer {layer:2d} {proj:3s} {fmt}: {E} experts {tuple(ck.shape(p)[1:])}, "
              f"bf16 elements differing from transformers: {diff} of {total}; reader vs "
              f"safetensors (first chunk): "
              f"{'n/a' if safetensors is None else f'{reader_diff} bytes differ'}")
        if fmt == "nvfp4":
            codes = R.unpack_e2m1(ck.tensor(p, slice(0, 1))[0])
            peak = R.e2m1_values(codes).abs().view(codes.shape[0], -1, 16).amax(-1)
            vals, counts = torch.unique(peak, return_counts=True)
            hist = dict(zip(vals.tolist(), counts.tolist(), strict=True))
            den = ck.tensor(p + "_outer_scale_denominator").item()
            print(f"    expert 0 block maxima (e2m1): {hist},"
                  f" outer denominator {den}")
        else:
            s = ck.tensor(p + "_scale", slice(0, 1))[0]
            uniq = torch.unique(s, dim=0).shape[0]
            print(f"    expert 0 scale rows: {uniq} distinct of {s.shape[0]}, "
                  f"bytes {s.min().item()}..{s.max().item()}")


def per_expert_fp8_fit(ck: IC.Checkpoint, layer: int, device: str, chunk: int = 16) -> None:
    """Could an MXFP8 expert be one e4m3 tensor with one power-of-two scale (what grouped_gemm
    takes)? Every block rescaled to the expert's largest block scale, 2 ** (s - s_max) times
    its e4m3 values: count the elements that do not survive the trip through e4m3 (they fall
    into the subnormals or below)."""
    for proj in ("w13", "w2"):
        p = f"model.llm.layers.{layer}.mlp.experts.{proj}_weight"
        E = ck.shape(p)[0]
        lost = nonzero = 0
        spread = []
        for start in range(0, E, chunk):
            rows = slice(start, min(start + chunk, E))
            q = ck.tensor(p, rows).to(device).float()
            s = ck.tensor(p + "_scale", rows).to(device).int()
            smax = s.flatten(1).amax(1)
            spread += (smax - s.flatten(1).amin(1)).tolist()
            shift = (s - smax[:, None, None]).float().repeat_interleave(32, dim=-1)
            v = q * torch.exp2(shift)
            back = v.to(torch.float8_e4m3fn).float()
            lost += ((back != v) & (q != 0)).sum().item()
            nonzero += (q != 0).sum().item()
        print(f"layer {layer:2d} {proj:3s} as one fp8 scale per expert: {lost} of {nonzero} "
              f"nonzero elements not exact; scale byte spread per expert "
              f"{min(spread)}..{max(spread)}")


def rel(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    out, ref = out.float(), ref.float()
    return ((out - ref).norm() / ref.norm()).item(), \
        ((out - ref).abs().max() / ref.abs().max()).item()


def gemm_check(ck: IC.Checkpoint, layer: int, M: int, seed: int = 0) -> None:
    g = torch.Generator(device="cuda").manual_seed(seed)
    for proj in ("w13", "w2"):
        ex = ck.experts(layer, proj, slice(0, 1)).to("cuda")
        _, N, K = ex.shape
        kp = ex.q.shape[-1] * (2 if ex.fmt == "nvfp4" else 1)
        x = torch.zeros(M, kp, dtype=torch.bfloat16, device="cuda")
        x[:, :K] = torch.randn(M, K, generator=g, device="cuda").to(torch.bfloat16)
        w_hf = hf_loaded(ck, layer, proj, slice(0, 1), "cuda")[0]
        if proj == "w13":  # back to the file's interleaved rows, the GEMM's output order
            gate, up = w_hf.chunk(2, dim=0)
            w_hf = torch.stack((gate, up), dim=1).reshape(N, K)
        dense = x[:, :K] @ w_hf.t()
        if ex.fmt == "nvfp4":
            qa, sfa, sa = sk.fp4_quantize(x)
            q, sf, s = ex.expert(0)
            out = sk.fp4gemm(qa, q, sfa, sf, sa, s)
            ref = R.fp4gemm(qa, sfa, q, sf, scale_a=sa, scale_b=s)
            xq = R.dequantize_fp4(qa, sfa, "nvfp4", sa)[:, :K]
            what = "fp4gemm (NVFP4 x NVFP4, per-tensor activation scale)"
        else:
            qa, sfa = sk.fp8_quantize(x, "mx")
            q, sf = ex.expert(0)
            out = sk.fp8gemm(qa, q, sfa=sfa, sfb=sf)
            ref = R.fp8gemm_mx(qa, sfa, q, sf)
            xq = R.dequantize_mx(qa, sfa)[:, :K]
            what = "fp8gemm MX (MXFP8 x MXFP8)"
        k_note = f"K {K} padded to {kp}" if kp != K else f"K {K}"
        e_ref, m_ref = rel(out, ref)
        e_bf, m_bf = rel(out, dense)
        e_w, m_w = rel(out, xq @ w_hf.float().t())
        print(f"layer {layer:2d} {proj:3s} expert 0, {what}, M {M}, N {N}, {k_note}:\n"
              f"    vs reference GEMM of the same operands: rel {e_ref:.2e}, max {m_ref:.2e}\n"
              f"    vs quantized x (fp32) @ W_hf.T:         rel {e_w:.2e}, max {m_w:.2e}\n"
              f"    vs bf16 x @ W_hf.T:                     rel {e_bf:.2e}, max {m_bf:.2e}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("path", nargs="?")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 20])
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    ck = IC.Checkpoint(find_file(args.path))
    llm = {n for n in ck.names() if n.startswith("model.llm.")}
    print(f"{ck.path}: {len(ck.names())} tensors, model.llm.* equal to raw_names(): "
          f"{llm == set(IC.raw_names())}")
    for layer in args.layers:
        check_layer(ck, layer, args.device)
        if layer in IC.MXFP8_LAYERS:
            per_expert_fp8_fit(ck, layer, args.device)
    if sk is None or not torch.cuda.is_available():
        print("no built extension or no GPU: GEMM check skipped")
        return 0
    nv = next(L for L in args.layers if L in IC.NVFP4_LAYERS)
    mx = next(L for L in args.layers if L in IC.MXFP8_LAYERS)
    gemm_check(ck, nv, args.m)
    gemm_check(ck, mx, args.m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
