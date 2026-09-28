"""rope_append_ (RoPE on q and k plus the K/V cache append from a fused qkv projection)
against the torch spelling in spark_kernels.layer: apply_rope, the head transpose and the
strided cache write. Shapes cover Llama-3-8B's heads, MHA, a 64-wide head, a prefill from
position 0, a decode step at a later position and a batch."""

import pytest
import torch

# the layer module needs the built extension; without it these tests are skipped, as the
# `sk` fixture's tests are on a machine without a GPU
L = pytest.importorskip("spark_kernels.layer")

# (B, S, H_q, H_kv, D, pos0, cap)
SHAPES = [(1, 300, 32, 8, 128, 0, 300), (1, 1, 32, 8, 128, 4095, 4096),
          (8, 1, 32, 8, 128, 100, 256), (2, 17, 8, 8, 128, 5, 40), (1, 33, 4, 1, 64, 0, 33),
          (3, 5, 16, 2, 64, 7, 12)]


def _shape_id(s):
    return "b{}_s{}_hq{}_hkv{}_d{}_pos{}_cap{}".format(*s)


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_rope_append_matches_torch(sk, shape):
    b, s, hq, hkv, d, pos0, cap = shape
    torch.manual_seed(0)
    width = (hq + 2 * hkv) * d
    qkv = torch.randn(b, s, width, device="cuda", dtype=torch.bfloat16)
    # tables in the rotate-half layout, as layer.RoPE builds them
    inv_freq = 1.0 / (L.ROPE_THETA ** (torch.arange(0, d, 2, device="cuda").float() / d))
    freqs = torch.outer(torch.arange(cap, device="cuda").float(), inv_freq)
    emb = torch.cat([freqs, freqs], dim=-1)
    cos, sin = emb.cos(), emb.sin()

    k_cache = torch.randn(b, hkv, cap, d, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    k_ref, v_ref = k_cache.clone(), v_cache.clone()
    q = sk.rope_append_(qkv, cos, sin, k_cache, v_cache, pos0, hq, hkv)

    # the torch spelling: rotate in fp32, transpose q, write k and v at pos0
    def rot(x):  # x [B, S, H, d]
        xf = x.float()
        x1, x2 = xf[..., :d // 2], xf[..., d // 2:]
        c, sn = cos[pos0:pos0 + s][None, :, None, :], sin[pos0:pos0 + s][None, :, None, :]
        return (xf * c + torch.cat([-x2, x1], dim=-1) * sn).to(torch.bfloat16)

    q_ref = rot(qkv[:, :, :hq * d].view(b, s, hq, d)).transpose(1, 2)
    k_new = rot(qkv[:, :, hq * d:(hq + hkv) * d].view(b, s, hkv, d))
    k_ref[:, :, pos0:pos0 + s] = k_new.transpose(1, 2)
    v_ref[:, :, pos0:pos0 + s] = qkv[:, :, (hq + hkv) * d:].view(b, s, hkv, d).transpose(1, 2)
    assert q.shape == (b, hq, s, d) and q.is_contiguous()
    tol = dict(atol=1e-2, rtol=1e-2)  # one bf16 rounding of an fp32 result on each side
    torch.testing.assert_close(q.float(), q_ref.float(), **tol)
    torch.testing.assert_close(k_cache.float(), k_ref.float(), **tol)
    torch.testing.assert_close(v_cache, v_ref, atol=0, rtol=0)  # a plain copy
    # positions outside pos0..pos0+S-1 are untouched (k_ref started as a copy of k_cache)
    untouched = torch.ones(cap, dtype=torch.bool, device="cuda")
    untouched[pos0:pos0 + s] = False
    torch.testing.assert_close(k_cache[:, :, untouched], k_ref[:, :, untouched], atol=0, rtol=0)


def test_rope_append_rejects_bad_inputs(sk):
    b, s, hq, hkv, d, cap = 1, 4, 8, 2, 64, 8
    qkv = torch.randn(b, s, (hq + 2 * hkv) * d, device="cuda", dtype=torch.bfloat16)
    cos = torch.randn(cap, d, device="cuda")
    k_cache = torch.zeros(b, hkv, cap, d, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.zeros_like(k_cache)
    sk.rope_append_(qkv, cos, cos, k_cache, v_cache, 4, hq, hkv)  # fills the cache exactly
    with pytest.raises(RuntimeError):
        sk.rope_append_(qkv, cos, cos, k_cache, v_cache, 5, hq, hkv)  # past the capacity
    with pytest.raises(RuntimeError):
        sk.rope_append_(qkv, cos[:6], cos[:6], k_cache, v_cache, 4, hq, hkv)  # tables too short
    with pytest.raises(RuntimeError):
        sk.rope_append_(qkv, cos, cos, k_cache, v_cache, 0, hq + 1, hkv)  # width mismatch
    with pytest.raises(RuntimeError):
        sk.rope_append_(qkv.float(), cos, cos, k_cache, v_cache, 0, hq, hkv)  # bf16 only
    with pytest.raises(RuntimeError):
        sk.rope_append_(qkv, cos.to(torch.bfloat16), cos, k_cache, v_cache, 0, hq, hkv)
