"""The Inkling 2 Mini checkpoint reader (python/spark_kernels/inkling_ckpt.py) and header
inspector (scripts/inkling_inspect.py): the name map against transformers' own renaming, the
NVFP4 and MXFP8 expert repack (dequantizing the repacked operands gives the file's values bit
for bit), and on a GPU the two GEMMs on a repacked expert.

The expert tensors are synthetic, with the file's layouts and the real shapes scaled down
(128 experts x 2304 rows x K = 2304 or 1152 becomes 3 x 256 x 512 or 384, the K = 1152 case
kept as a K that needs padding to 256). The modules are loaded from their files, since importing
the package needs the built extension. With INKLING_CKPT set to the real file (or its
snapshot directory), test_real_file also checks layers 0 and 20 of it.
"""

import importlib.util
import json
import os
import struct
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parent.parent


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


IC = _load("spark_inkling_ckpt", _ROOT / "python" / "spark_kernels" / "inkling_ckpt.py")
INSPECT = _load("inkling_inspect", _ROOT / "scripts" / "inkling_inspect.py")
R = IC.R  # reference.py, loaded by inkling_ckpt


def hf_core():
    """transformers.core_model_loading with the Inkling2Loop ops (5.24 on), or skip."""
    core = pytest.importorskip("transformers.core_model_loading")
    if not hasattr(core, "Inkling2LoopDequantizeExperts"):
        pytest.skip("this transformers has no Inkling2Loop conversion ops")
    return core


def bits(t):
    """The bit pattern, so -0.0 != 0.0 and the comparison is exact."""
    t = t.contiguous()
    return t.view({torch.float32: torch.int32, torch.bfloat16: torch.int16}[t.dtype])


def nvfp4_tensors(E, N, K, seed=0):
    """The file's four NVFP4 tensors for E experts of [N, K]: random codes, positive e4m3 block
    scales (some zero, as an all-zero block has), amax in the file's range, denominator 2688."""
    g = torch.Generator().manual_seed(seed)
    weight = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, generator=g)
    scale = (torch.rand(E, N, K // 16, generator=g) * 448).to(torch.float8_e4m3fn)
    scale.view(torch.uint8)[0, 0, :3] = 0
    amax = 0.1 + 0.25 * torch.rand(E, generator=g)
    return weight, scale, amax, torch.tensor([6.0 * 448.0])


def mxfp8_tensors(E, N, K, seed=0):
    g = torch.Generator().manual_seed(seed)
    weight = (torch.randn(E, N, K, generator=g) * 100).clamp(-448, 448).to(torch.float8_e4m3fn)
    scale = torch.randint(100, 130, (E, N, K // 32), dtype=torch.uint8, generator=g)
    return weight, scale


# ---- names --------------------------------------------------------------------------------

def test_raw_names_and_map():
    names = IC.raw_names()
    assert len(names) == len(set(names)) == 822
    assert all(IC.hf_name(n) is not None for n in names)
    assert IC.hf_name("model.llm.layers.0.attn.wq_du.weight") == \
        "model.layers.0.self_attn.q_proj.linear.weight"
    assert IC.hf_name("model.llm.attn_res.attn_queries.4") == \
        "model.layers.5.attention_input.query"
    assert IC.hf_name("model.llm.layers.7.controllers.attn.gain_weight") == \
        "model.layers.7.q_message_gate.message_gate_proj.weight"
    assert IC.hf_name("model.llm.layers.3.mlp.experts.w13_weight_amax") == \
        "model.layers.3.mlp.experts.gate_up_proj"
    assert IC.hf_name("model.visual.backbone.layers.0.ln_1.weight") is None
    # full layers have the dynamic taps and no attn_norm, core layers the message gates
    assert "model.llm.layers.5.attn.k_sconv.weight_proj" in names
    assert "model.llm.layers.5.attn_norm.weight" not in names
    assert "model.llm.layers.4.attn.k_sconv.weight" in names
    assert "model.llm.layers.18.controllers.k.down_weight" not in names
    assert "model.llm.layers.20.mlp.experts.w2_weight_amax" not in names


def test_names_match_transformers():
    hf_core()
    modeling = pytest.importorskip("transformers.models.inkling2_loop.modeling_inkling2_loop")
    from transformers.conversion_mapping import get_model_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key
    from transformers.models.inkling2_loop.configuration_inkling2_loop import (
        Inkling2LoopTextConfig,
    )

    with torch.device("meta"):
        model = modeling.Inkling2LoopForCausalLM(Inkling2LoopTextConfig())
    conv = get_model_conversion_mapping(model)
    renamings = [c for c in conv if isinstance(c, WeightRenaming)
                 and not isinstance(c, WeightConverter)]
    converters = [c for c in conv if isinstance(c, WeightConverter)]
    sd = model.state_dict()
    for raw in IC.raw_names():
        hf, _ = rename_source_key(raw, renamings, converters, model.base_model_prefix, sd)
        assert hf == IC.hf_name(raw), raw
    assert {IC.hf_name(n) for n in IC.raw_names()} == set(sd)


def test_message_queries_and_lm_head():
    C, D = 12, 8
    q = torch.randn(2 * C, D)
    kv = torch.randn(2 * C, D)
    m = IC.merge_message_queries(q, kv)
    assert m.shape == (C, 4, D)
    for c in range(C):
        assert torch.equal(m[c], torch.stack((q[2 * c], kv[2 * c], kv[2 * c + 1], q[2 * c + 1])))
    assert IC.unpad_lm_head(torch.zeros(IC.VOCAB_PADDED, 4)).shape == (IC.VOCAB, 4)
    core = hf_core()
    op = core.Inkling2LoopMergeMessageQueries()
    hf = op.convert({"a": q, "b": kv}, ["a", "b"], ["t"])["t"]
    assert torch.equal(hf, m)


def test_gate_up_order():
    E, i, k = 2, 6, 4
    gate, up = torch.randn(E, i, k), torch.randn(E, i, k)
    w13 = torch.stack((gate, up), dim=2).reshape(E, 2 * i, k)  # ops.interleave_experts
    g, u = IC.gate_up_split(w13)
    assert torch.equal(g, gate) and torch.equal(u, up)
    assert torch.equal(IC.hf_gate_up(w13), torch.cat((gate, up), dim=1))
    core = hf_core()
    hf = core.Interleave(dim=1).convert({"t": w13}, None, ["t"])["t"]
    assert torch.equal(hf, IC.hf_gate_up(w13))


# ---- expert repack ------------------------------------------------------------------------

@pytest.mark.parametrize("K", [512, 384])
def test_nvfp4_repack_lossless(K):
    E, N = 3, 256
    weight, scale, amax, den = nvfp4_tensors(E, N, K)
    ex = IC.repack_nvfp4(weight, scale, amax, den)
    kp = -(-K // 256) * 256
    assert ex.q.shape == (E, N, kp // 2) and ex.sf.shape == (E, N * kp // 16)
    assert ex.shape == (E, N, K)
    # the same codes and e4m3 bytes, padding zero
    assert torch.equal(ex.q[..., :K // 2], weight) and not ex.q[..., K // 2:].any()
    for e in range(E):
        rows = R.from_blocked(ex.sf[e], N, kp // 16)
        assert torch.equal(rows[:, :K // 16], scale[e].view(torch.uint8))
        assert not rows[:, K // 16:].any()
    assert torch.equal(bits(ex.scale), bits(amax / den))
    ref = IC.nvfp4_dequantize_hf(weight, scale, amax, den)
    out = IC.dequantize_nvfp4(ex)
    assert torch.equal(bits(out), bits(ref))
    assert torch.equal(bits(out.to(torch.bfloat16)), bits(ref.to(torch.bfloat16)))


@pytest.mark.parametrize("K", [512, 384])
def test_mxfp8_repack_lossless(K):
    E, N = 3, 256
    weight, scale = mxfp8_tensors(E, N, K)
    ex = IC.repack_mxfp8(weight, scale)
    kp = -(-K // 256) * 256
    assert ex.q.shape == (E, N, kp) and ex.sf.shape == (E, N, kp // 32)
    assert torch.equal(ex.q[..., :K].view(torch.uint8), weight.view(torch.uint8))
    assert not ex.q[..., K:].view(torch.uint8).any()
    assert torch.equal(ex.sf[..., :K // 32], scale) and (ex.sf[..., K // 32:] == 127).all()
    ref = IC.mxfp8_dequantize_hf(weight, scale)
    assert torch.equal(bits(IC.dequantize_mxfp8(ex)), bits(ref))
    # padding dequantizes to zero, not 0 * inf
    full = R.dequantize_mx(ex.q, ex.sf)
    assert torch.equal(full[..., K:], torch.zeros_like(full[..., K:]))


def test_dequant_matches_transformers():
    core = hf_core()
    op = core.Inkling2LoopDequantizeExperts()
    weight, scale, amax, den = nvfp4_tensors(9, 128, 256, seed=1)  # more than its 8-expert chunk
    hf = op.convert({"w": weight, "s": scale, "a": amax, "d": den}, ["w", "s", "a", "d"],
                    ["t"])["t"]
    ex = IC.repack_nvfp4(weight, scale, amax, den)
    assert hf.dtype == torch.bfloat16
    assert torch.equal(bits(IC.dequantize_nvfp4(ex).to(torch.bfloat16)), bits(hf))
    weight, scale = mxfp8_tensors(9, 128, 256, seed=1)
    hf = op.convert({"w": weight, "s": scale}, ["w", "s", "a", "d"], ["t"])["t"]
    ex = IC.repack_mxfp8(weight, scale)
    assert torch.equal(bits(IC.dequantize_mxfp8(ex).to(torch.bfloat16)), bits(hf))


# ---- the file -----------------------------------------------------------------------------

def write_safetensors(path, tensors):
    """A safetensors file by hand (the format is an 8-byte length, JSON, then the bytes)."""
    header, blobs, off = {"__metadata__": {"format": "pt"}}, [], 0
    names = {torch.uint8: "U8", torch.float32: "F32", torch.bfloat16: "BF16",
             torch.float8_e4m3fn: "F8_E4M3"}
    for name, t in tensors.items():
        b = t.contiguous().view(torch.uint8).reshape(-1).numpy().tobytes()
        header[name] = {"dtype": names[t.dtype], "shape": list(t.shape),
                        "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    js = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(js)) + js + b"".join(blobs))


def small_checkpoint(path):
    p0, p20 = "model.llm.layers.0.mlp.experts.", "model.llm.layers.20.mlp.experts."
    w, s, a, d = nvfp4_tensors(4, 128, 256)
    fw, fs = mxfp8_tensors(4, 128, 256)
    t = {p0 + "w13_weight": w, p0 + "w13_weight_scale": s, p0 + "w13_weight_amax": a,
         p0 + "w13_weight_outer_scale_denominator": d,
         p20 + "w13_weight": fw, p20 + "w13_weight_scale": fs,
         "model.llm.norm.weight": torch.randn(64).to(torch.bfloat16),
         "model.llm.layers.0.mlp.gate.bias": torch.randn(4)}
    write_safetensors(path, t)
    return t


def test_reader_and_inspector(tmp_path):
    path = tmp_path / "model.safetensors"
    t = small_checkpoint(path)
    ck = IC.Checkpoint(tmp_path)
    assert ck.metadata == {"format": "pt"} and set(ck.names()) == set(t)
    for name, ref in t.items():
        got = ck.tensor(name)
        assert got.dtype == ref.dtype and torch.equal(got.view(torch.uint8), ref.view(torch.uint8))
    p = "model.llm.layers.0.mlp.experts.w13_weight"
    assert torch.equal(ck.tensor(p, slice(1, 3)), t[p][1:3])
    ex = ck.experts(0, "w13", slice(1, 3))
    den = t[p + "_outer_scale_denominator"]
    assert torch.equal(bits(ex.scale), bits(t[p + "_amax"][1:3] / den))
    assert isinstance(ck.experts(20, "w13"), IC.MXFP8Experts)
    hf = IC.hf_expert_weight(ck, 0, "w13")
    ref = IC.nvfp4_dequantize_hf(t[p], t[p + "_scale"], t[p + "_amax"],
                                 t[p + "_outer_scale_denominator"]).to(torch.bfloat16)
    assert torch.equal(bits(hf), bits(IC.hf_gate_up(ref)))

    header, start = INSPECT.read_header(str(path))
    header.pop("__metadata__")
    assert start == ck.data_start and not INSPECT.check_sizes(header)
    fmts = INSPECT.expert_formats(header)
    assert fmts == {0: "nvfp4", 20: "mxfp8"}
    groups = {n: INSPECT.classify(n, i, fmts) for n, i in header.items()}
    assert groups[p + "_amax"] == "NVFP4 expert"
    assert groups["model.llm.layers.20.mlp.experts.w13_weight_scale"] == "MXFP8 expert"
    assert groups["model.llm.norm.weight"] == "bf16"
    assert groups["model.llm.layers.0.mlp.gate.bias"] == "fp32"


def test_reader_matches_safetensors(tmp_path):
    st = pytest.importorskip("safetensors")
    path = tmp_path / "model.safetensors"
    t = small_checkpoint(path)
    ck = IC.Checkpoint(path)
    with st.safe_open(str(path), framework="pt") as f:
        for name in t:
            assert torch.equal(f.get_tensor(name).view(torch.uint8),
                               ck.tensor(name).view(torch.uint8))


def _real_file():
    p = os.environ.get("INKLING_CKPT")
    if not p:
        pytest.skip("INKLING_CKPT is not set")
    return IC.Checkpoint(p)


def test_real_file():
    ck = _real_file()
    llm = {n for n in ck.names() if n.startswith("model.llm.")}
    assert llm == set(IC.raw_names())
    core = hf_core()
    op, il = core.Inkling2LoopDequantizeExperts(), core.Interleave(dim=1)
    some = slice(0, 9)
    for layer in (0, 20):
        for proj in ("w13", "w2"):
            p = f"model.llm.layers.{layer}.mlp.experts.{proj}_weight"
            src = [p, p + "_scale", p + "_amax", p + "_outer_scale_denominator"]
            inputs = {n: ck.tensor(n, some if n in (p, p + "_scale") else None)
                      for n in src if n in ck.header}
            if p + "_amax" in inputs:
                inputs[p + "_amax"] = inputs[p + "_amax"][some]
            hf = op.convert(inputs, src, ["t"])["t"]
            if proj == "w13":
                hf = il.convert({"t": hf}, None, ["t"])["t"]
            ex = ck.experts(layer, proj, some)
            deq = IC.dequantize_nvfp4(ex) if ex.fmt == "nvfp4" else IC.dequantize_mxfp8(ex)
            deq = deq.to(torch.bfloat16)
            if proj == "w13":
                deq = IC.hf_gate_up(deq)
            assert torch.equal(bits(deq), bits(hf)), (layer, proj)


# ---- the GEMMs on repacked experts (GPU) ----------------------------------------------------

def _err(out, ref):
    out, ref = out.float(), ref.float()
    return ((out - ref).norm() / ref.norm()).item()


@pytest.mark.parametrize("K", [512, 384])
def test_fp4gemm_on_repacked_expert(sk, K):
    E, N, M = 2, 256, 64
    weight, scale, amax, den = nvfp4_tensors(E, N, K, seed=2)
    ex = IC.repack_nvfp4(weight, scale, amax, den).to("cuda")
    kp = ex.q.shape[-1] * 2
    x = torch.zeros(M, kp, dtype=torch.bfloat16, device="cuda")
    x[:, :K] = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    qa, sfa, sa = sk.fp4_quantize(x)
    for e in range(E):
        q, sf, s = ex.expert(e)
        out = sk.fp4gemm(qa, q, sfa, sf, sa, s)
        ref = R.fp4gemm(qa, sfa, q, sf, scale_a=sa, scale_b=s)
        assert _err(out, ref) < 1e-2
        # against the bf16 matmul of transformers' weight: only the activation's fp4 error
        w = IC.nvfp4_dequantize_hf(weight[e:e + 1], scale[e:e + 1], amax[e:e + 1], den)
        dense = x[:, :K] @ w[0].to(torch.bfloat16).cuda().t()
        assert _err(out, dense) < 0.2


@pytest.mark.parametrize("K", [512, 384])
def test_fp8gemm_mx_on_repacked_expert(sk, K):
    E, N, M = 2, 256, 64
    weight, scale = mxfp8_tensors(E, N, K, seed=2)
    ex = IC.repack_mxfp8(weight, scale).to("cuda")
    kp = ex.q.shape[-1]
    x = torch.zeros(M, kp, dtype=torch.bfloat16, device="cuda")
    x[:, :K] = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    qa, sfa = sk.fp8_quantize(x, "mx")
    for e in range(E):
        q, sf = ex.expert(e)
        out = sk.fp8gemm(qa, q, sfa=sfa, sfb=sf)
        ref = R.fp8gemm_mx(qa, sfa, q, sf)
        assert _err(out, ref) < 1e-2
        w = IC.mxfp8_dequantize_hf(weight[e:e + 1], scale[e:e + 1])[0]
        dense = x[:, :K] @ w.to(torch.bfloat16).cuda().t()
        assert _err(out, dense) < 0.1
