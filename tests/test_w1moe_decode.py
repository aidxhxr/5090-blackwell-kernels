"""The fused decode step over 1-bit and ternary experts (w1moe_decode) against the sorted
w1gemm_moe path of MoEBlock1Bit and against a dense torch reference on the dequantized
experts, every variant, T in 1, 3, 8, a small expert count at the real hidden and
intermediate sizes of DeepSeek V4.1 Flash.

Everything here needs the extension and a GPU (the `sk` fixture skips otherwise). Nothing
has run yet: the kernel was written on a machine without nvcc."""

import pytest
import torch

from spark_kernels import moe
from spark_kernels import reference as ref

H, N_I = 5120, 2304  # DeepSeek V4.1 Flash's hidden and intermediate
E, TOPK = 8, 6
TS = (1, 3, 8)
BITS = (1, 2)
# fp32 sums in a different order, a bf16 routing weight and no bf16 rounding of the per-slot
# down result on the fused side: a few bf16 ulps of an output of magnitude about 1.
TOL = dict(atol=2e-2, rtol=2e-2)


def _variants(sk):
    return list(range(sk.ops.w1moe_decode_num_variants()))


@pytest.fixture(scope="module")
def blocks(sk):
    """One random block per bit width, quantized from bf16 so the reference dequantizer
    agrees with the bits the kernel reads."""
    out = {}
    for bits in BITS:
        cfg = moe.MoEConfig(hidden=H, n_routed=E, n_shared=1, top_k=TOPK, inter=N_I, bits=bits)
        g = torch.Generator(device="cuda").manual_seed(100 + bits)
        out[bits] = moe.MoEBlock1Bit.random(cfg, "cuda", g)
    return out


def _x(T, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(T, H, device="cuda", generator=g).to(torch.bfloat16)


def _dense_routed(block, x):
    """The routed experts only, every (token, slot) against the dequantized expert, with the
    roundings the sorted path makes: gate|up to bf16, the SwiGLU to bf16, the down product
    to bf16, the routing weights in fp32."""
    cfg = block.cfg
    ids, w = block.routing(x)
    out = torch.zeros(x.shape[0], cfg.hidden, device=x.device)
    for t in range(x.shape[0]):
        for slot in range(cfg.top_k):
            e = int(ids[t, slot])
            w_gu = ref.w1_dequantize(block.gate_up.packed[e], block.gate_up.scales[e], cfg.bits)
            w_dn = ref.w1_dequantize(block.down.packed[e], block.down.scales[e], cfg.bits)
            gu = (x[t:t + 1].float() @ w_gu.float()).to(torch.bfloat16).float()
            h = (torch.nn.functional.silu(gu[:, :cfg.inter]) * gu[:, cfg.inter:]).to(torch.bfloat16)
            d = (h.float() @ w_dn.float()).to(torch.bfloat16).float()
            out[t] += d[0] * w[t, slot]
    return out


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("T", TS)
def test_routed_kernel_matches_dense_reference(sk, blocks, T, bits):
    block = blocks[bits]
    x = _x(T, 7 * T + bits)
    ids, w = block.routing(x)
    want = _dense_routed(block, x)
    for variant in _variants(sk):
        got = sk.ops.w1moe_decode(x, ids, w, block.gate_up, block.down, variant)
        assert got.shape == (T, H) and got.dtype == torch.bfloat16
        assert torch.isfinite(got.float()).all(), f"variant {variant}"
        torch.testing.assert_close(got.float(), want, **TOL, msg=f"variant {variant}")


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("T", TS)
def test_decode_matches_sorted_forward(sk, blocks, T, bits):
    """decode (fused routed experts + shared expert) against forward (the sorted w1gemm_moe
    path + the same shared expert), every variant, and against forward_reference."""
    block = blocks[bits]
    x = _x(T, 31 * T + bits)
    want = block.forward(x).float()
    want_ref = block.forward_reference(x).float()
    torch.testing.assert_close(want, want_ref, **TOL)
    for variant in _variants(sk):
        got = block.decode(x, variant).float()
        torch.testing.assert_close(got, want, **TOL, msg=f"variant {variant} vs forward")
        torch.testing.assert_close(got, want_ref, **TOL, msg=f"variant {variant} vs reference")


def test_shared_expert_shows_up(sk, blocks):
    """decode without the shared expert is the routed kernel alone."""
    block = blocks[1]
    x = _x(2, 5)
    ids, w = block.routing(x)
    routed = sk.ops.w1moe_decode(x, ids, w, block.gate_up, block.down).float()
    bare = moe.MoEBlock1Bit(block.cfg, block.gate_up, block.down, None, None, block.router_w,
                            block.router_b)
    torch.testing.assert_close(bare.decode(x).float(), routed, **TOL)
    assert not torch.allclose(block.decode(x).float(), routed, atol=1e-3)


def test_two_tokens_on_one_expert(sk, blocks):
    """Two identical tokens pick the same experts; the fused kernel streams each expert once
    for both and must give both the same answer as a single token."""
    block = blocks[1]
    x1 = _x(1, 11)
    x2 = torch.cat([x1, x1])
    ids1, w1 = block.routing(x1)
    ids2, w2 = block.routing(x2)
    assert torch.equal(ids2[0], ids2[1]) and torch.equal(ids2[0], ids1[0])
    for variant in _variants(sk):
        y1 = sk.ops.w1moe_decode(x1, ids1, w1, block.gate_up, block.down, variant).float()
        y2 = sk.ops.w1moe_decode(x2, ids2, w2, block.gate_up, block.down, variant).float()
        torch.testing.assert_close(y2[0], y1[0], **TOL)
        torch.testing.assert_close(y2[1], y1[0], **TOL)


def test_large_batch_falls_back_to_forward(sk, blocks):
    block = blocks[1]
    x = _x(moe.DECODE_MAX_TOKENS + 1, 13)
    torch.testing.assert_close(block.decode(x).float(), block.forward(x).float(), **TOL)


def test_random_bits_block_runs(sk):
    """The packed-format random block: a valid code set (no ternary code 3), and the kernel
    runs on it."""
    cfg = moe.MoEConfig(hidden=512, n_routed=4, n_shared=1, top_k=2, inter=256, bits=2)
    block = moe.MoEBlock1Bit.random_bits(cfg, "cuda", torch.Generator(device="cuda").manual_seed(1))
    codes = ref.w1_unpack(block.gate_up.packed[0], 2)
    assert int(codes.max()) <= 2
    assert block.gate_up.shape == (4, 512, 512)
    y = block.decode(torch.randn(3, 512, device="cuda").to(torch.bfloat16))
    assert y.shape == (3, 512) and torch.isfinite(y.float()).all()


def test_rejects_bad_input(sk, blocks):
    block = blocks[1]
    x = _x(2, 17)
    ids, w = block.routing(x)
    with pytest.raises(RuntimeError):
        sk.ops.w1moe_decode(_x(9, 1), *block.routing(_x(9, 1)), block.gate_up, block.down)
    with pytest.raises(RuntimeError):  # down and gate|up swapped: the shapes do not match
        sk.ops.w1moe_decode(x, ids, w, block.down, block.gate_up)
    with pytest.raises(RuntimeError):  # one token's ids for two tokens
        sk.ops.w1moe_decode(x, ids[:1], w[:1], block.gate_up, block.down)
