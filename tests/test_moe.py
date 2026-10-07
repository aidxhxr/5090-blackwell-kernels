"""spark_kernels.moe on the CPU: the sigmoid routing, the expert sort, the 1-bit block's
reference forward against a dense computation on the same dequantized experts, and the
DeepSeek V4.1 Flash footprint against hand numbers; on a GPU, forward against
forward_reference.

The package's __init__ imports the extension, so on a machine without it moe.py is loaded
from its file; the GPU test goes through the normal import."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F


def _load_moe():
    try:
        from spark_kernels import moe
        return moe
    except ImportError:
        path = Path(__file__).resolve().parent.parent / "python" / "spark_kernels" / "moe.py"
        spec = importlib.util.spec_from_file_location("spark_kernels_moe", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["spark_kernels_moe"] = mod
        spec.loader.exec_module(mod)
        return mod


moe = _load_moe()
TINY = moe.MoEConfig(hidden=256, n_routed=8, n_shared=1, top_k=2, inter=128, bits=1)


def test_route_picks_biased_topk_weights_from_unbiased_sigmoid():
    g = torch.Generator().manual_seed(0)
    T, E, k = 16, 8, 3
    logits = torch.randn(T, E, generator=g)
    bias = torch.randn(E, generator=g) * 3  # large enough to change the pick
    ids, w = moe.route(logits, k, bias, scale=2.5)
    assert ids.dtype == torch.int32 and w.dtype == torch.float32
    assert ids.shape == (T, k) and w.shape == (T, k)
    s = torch.sigmoid(logits)
    want = (s + bias).topk(k, dim=-1).indices
    assert torch.equal(ids.long().sort(-1).values, want.sort(-1).values)
    # the pick differs from the unbiased one somewhere, or the bias did nothing
    assert not torch.equal(ids.long().sort(-1).values, s.topk(k, -1).indices.sort(-1).values)
    picked = s.gather(-1, ids.long())
    torch.testing.assert_close(w, picked / picked.sum(-1, keepdim=True) * 2.5)
    torch.testing.assert_close(w.sum(-1), torch.full((T,), 2.5))


def test_route_without_bias():
    logits = torch.randn(4, 6, generator=torch.Generator().manual_seed(1))
    ids, w = moe.route(logits, 2)
    assert torch.equal(ids.long().sort(-1).values, logits.topk(2, -1).indices.sort(-1).values)
    torch.testing.assert_close(w.sum(-1), torch.ones(4))


def test_sort_by_expert():
    g = torch.Generator().manual_seed(2)
    T, E, k = 32, 8, 3
    ids = torch.stack([torch.randperm(E, generator=g)[:k] for _ in range(T)]).to(torch.int32)
    perm, offsets, token_of = moe.sort_by_expert(ids, E)
    assert perm.dtype == torch.int64 and offsets.dtype == torch.int32
    assert offsets.shape == (E + 1,) and offsets[0] == 0 and offsets[-1] == T * k
    assert torch.equal(perm.sort().values, torch.arange(T * k))
    assert torch.equal(token_of, perm // k)
    flat = ids.reshape(-1).long()
    sorted_ids = flat[perm]
    for e in range(E):
        lo, hi = offsets[e].item(), offsets[e + 1].item()
        assert (sorted_ids[lo:hi] == e).all()
        assert hi - lo == (flat == e).sum()
        assert torch.equal(perm[lo:hi], perm[lo:hi].sort().values)  # stable
    # an expert nobody picked has an empty range
    ids2 = torch.zeros(4, 2, dtype=torch.int32)
    _, offsets2, _ = moe.sort_by_expert(ids2, E)
    assert offsets2.tolist() == [0, 8, 8, 8, 8, 8, 8, 8, 8]


@pytest.mark.parametrize("bits", [1, 2])
def test_quantizer_contract(bits):
    g = torch.Generator().manual_seed(3)
    K, N = 256, 32
    w = torch.randn(K, N, generator=g).to(torch.bfloat16)
    packed, scales = moe._quantize_ref(w, bits)
    assert packed.dtype == torch.int32 and packed.shape == (N, K * bits // 32)
    assert scales.dtype == torch.bfloat16 and scales.shape == (K // 128, N)
    s = w.float().abs().reshape(K // 128, 128, N).mean(1).to(torch.bfloat16)
    assert torch.equal(scales, s)
    sk = s.float().repeat_interleave(128, 0)
    if bits == 1:
        code = (w.float() >= 0).long()
        want = (sk * (2 * code - 1)).to(torch.bfloat16)
    else:
        code = (w.float() >= -sk / 2).long() + (w.float() > sk / 2).long()
        want = (sk * (code - 1)).to(torch.bfloat16)
    assert torch.equal(moe._dequantize_ref(packed, scales, bits), want)
    # k ascending from the low bit of the word: k = 0 of column 0 is the low bits of word 0
    assert (packed[0, 0].item() & ((1 << bits) - 1)) == code[0, 0].item()
    assert ((packed[0, 0].item() >> bits) & ((1 << bits) - 1)) == code[1, 0].item()


def _dense_forward(block, x):
    """The block on its dequantized experts, every token against every expert it picked,
    with the same roundings forward_reference uses."""
    cfg = block.cfg
    ids, w = block.routing(x)
    gu = moe.dequantize_weight(block.gate_up)
    dn = moe.dequantize_weight(block.down)
    out = torch.zeros(x.shape[0], cfg.hidden)
    for t in range(x.shape[0]):
        for slot in range(cfg.top_k):
            e = ids[t, slot].item()
            h = (x[t:t + 1].float() @ gu[e].float()).to(torch.bfloat16).float()
            h = (F.silu(h[:, :cfg.inter]) * h[:, cfg.inter:]).to(torch.bfloat16)
            d = (h.float() @ dn[e].float()).to(torch.bfloat16).float()
            out[t] += d[0] * w[t, slot]
    sgu = (x.float() @ moe.dequantize_weight(block.shared_gate_up).float()).to(torch.bfloat16)
    n = sgu.shape[1] // 2
    sh = (F.silu(sgu[:, :n].float()) * sgu[:, n:].float()).to(torch.bfloat16)
    out += (sh.float() @ moe.dequantize_weight(block.shared_down).float()).to(torch.bfloat16)
    return out.to(torch.bfloat16)


@pytest.mark.parametrize("bits", [1, 2])
def test_forward_reference_matches_dense(bits):
    cfg = moe.MoEConfig(**{**TINY.__dict__, "bits": bits})
    g = torch.Generator().manual_seed(4)
    E, H, N_I = cfg.n_routed, cfg.hidden, cfg.inter
    w_gu = torch.randn(E, H, 2 * N_I, generator=g).mul_(0.05).to(torch.bfloat16)
    w_dn = torch.randn(E, N_I, H, generator=g).mul_(0.05).to(torch.bfloat16)
    s_gu = torch.randn(H, 2 * N_I, generator=g).mul_(0.05).to(torch.bfloat16)
    s_dn = torch.randn(N_I, H, generator=g).mul_(0.05).to(torch.bfloat16)
    rw = torch.randn(H, E, generator=g).mul_(0.05).to(torch.bfloat16)
    rb = torch.randn(E, generator=g).mul_(0.1)
    block = moe.MoEBlock1Bit.from_bf16(w_gu, w_dn, s_gu, s_dn, rw, rb, cfg)
    assert block.gate_up.shape == (E, H, 2 * N_I) and block.down.shape == (E, N_I, H)
    scale_bytes = 2 * (2 * N_I * H // 128 + H * N_I // 128)
    assert block.nbytes() == E * (3 * H * N_I * bits // 8 + scale_bytes)
    x = torch.randn(8, H, generator=g).to(torch.bfloat16)
    y = block.forward_reference(x)
    assert y.dtype == torch.bfloat16 and y.shape == (8, H)
    torch.testing.assert_close(y.float(), _dense_forward(block, x).float(), atol=2e-2, rtol=2e-2)


def test_random_block_builds_on_cpu():
    block = moe.MoEBlock1Bit.random(TINY, "cpu", torch.Generator().manual_seed(5))
    assert block.gate_up.shape == (8, 256, 256)
    y = block.forward_reference(torch.randn(3, 256).to(torch.bfloat16))
    assert y.shape == (3, 256) and torch.isfinite(y.float()).all()


def test_deepseek_v41_flash_bytes():
    cfg = moe.DEEPSEEK_V41_FLASH
    per_expert = 3 * 5120 * 2304
    packed = 384 * per_expert // 8
    scales = 384 * 2 * (per_expert // 128)
    assert cfg.bytes_per_layer() == packed + scales
    assert packed == 1_698_693_120
    assert cfg.bytes_per_layer("bf16") == 384 * per_expert * 2
    assert cfg.bytes_per_layer("int4") == 384 * (per_expert // 2 + 2 * per_expert // 128)
    assert cfg.bytes_per_layer("mxfp4") == 384 * (per_expert // 2 + per_expert // 32)
    model = moe.DEEPSEEK_V41_FLASH_MODEL
    assert model.layers == 40 and model.bytes_total() == 40 * cfg.bytes_per_layer()
    assert model.bytes_total("bf16") > 1e12 and model.bytes_total() < 80e9
    two = moe.MoEConfig(**{**cfg.__dict__, "bits": 2})
    assert two.bytes_per_layer() == 2 * packed + scales


def test_forward_matches_reference_on_gpu(sk):
    if not hasattr(sk, "w1gemm_moe"):
        pytest.skip("w1gemm_moe is not in this build")
    cfg = moe.MoEConfig(hidden=512, n_routed=16, n_shared=1, top_k=4, inter=256, bits=1)
    g = torch.Generator(device="cuda").manual_seed(6)
    block = moe.MoEBlock1Bit.random(cfg, "cuda", g)
    x = torch.randn(32, cfg.hidden, device="cuda", generator=g).to(torch.bfloat16)
    y = block.forward(x)
    y_ref = block.forward_reference(x)
    torch.testing.assert_close(y.float(), y_ref.float(), atol=2e-2, rtol=2e-2)
