"""Inkling 2 Mini's attention core (spark_kernels.inkling_attention) against an fp32 torch
reference that copies the math of HF's Inkling2LoopRelativePositionBias and
eager_attention_forward (transformers 5.24.1, modeling_inkling2_loop.py): rel_logits = r @ proj,
gathered at the distance q_pos - k_pos and zeroed outside 0 <= d < extent, added to
q k^T / sqrt(128), the causal (and sliding window, kv_idx > q_idx - W) mask, and the sink as one
more softmax column with no value. Both layer kinds (window 1024 and full), the Inkling shape
(24 query heads, 8 kv heads, B = 2), query counts 1 to 2048 against caches longer than the
extent so the zero-bias distances are read, both kernels on the decode shapes and the decode
rows against the prefill rows."""

import pytest
import torch

TOL = dict(atol=2e-2, rtol=2e-2)  # one bf16 rounding of an fp32 result, P in bf16
H_Q, H_KV, D, D_REL, EXTENT = 24, 8, 128, 16, 1024
WINDOWS = [1024, 0]  # a sliding layer, a full layer


def _inputs(B, T, S, seed=0, cap=None):
    """q, k, v, r, proj, sinks at the model's scales: unit-variance q and k (logits of std 1),
    relative states of norm up to 8 (the L2 clip), a bank whose biases have std ~2, sinks in
    [-2, 8] (large enough to move the output at 4K keys). k and v are the leading S rows of a
    cache of `cap` rows when cap is given."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(B, H_Q, T, D, generator=g)
    cap = cap or S
    k = torch.randn(B, H_KV, cap, D, generator=g)
    v = torch.randn(B, H_KV, cap, D, generator=g)
    r = torch.randn(B, H_Q, T, D_REL, generator=g)
    r = r / r.norm(dim=-1, keepdim=True) * 8.0 * torch.rand(B, H_Q, T, 1, generator=g)
    proj = torch.randn(D_REL, EXTENT, generator=g) * 0.25
    sinks = torch.rand(B, H_Q, T, generator=g) * 10 - 2
    out = [x.to(torch.bfloat16).cuda() for x in (q, k, v, r, proj)]
    out[1], out[2] = out[1][:, :, :S], out[2][:, :, :S]
    return (*out, sinks.cuda())


def _reference(q, k, v, r, proj, sinks, window):
    """HF's math in fp32, one batch row at a time. Queries are the last T of S positions."""
    B, Hq, T, Dh = q.shape
    Hkv, S = k.shape[1], k.shape[2]
    extent = proj.shape[1]
    dev = q.device
    q_pos = torch.arange(T, device=dev) + (S - T)
    dist = q_pos[:, None] - torch.arange(S, device=dev)[None, :]  # get_relative_distance
    allowed = dist >= 0
    if window > 0:
        allowed &= dist < window  # sliding_window_overlay: kv_idx > q_idx - window
    outs = []
    for b in range(B):
        kf = k[b].float().repeat_interleave(Hq // Hkv, dim=0)
        vf = v[b].float().repeat_interleave(Hq // Hkv, dim=0)
        s = q[b].float() @ kf.transpose(-1, -2) * Dh**-0.5
        rel_logits = r[b].float() @ proj.float()  # [Hq, T, extent]
        idx = dist.clamp(0, extent - 1).expand(Hq, T, S)
        bias = rel_logits.gather(-1, idx)
        bias = bias.masked_fill((dist < 0) | (dist >= extent), 0.0)
        s = (s + bias).masked_fill(~allowed, float("-inf"))
        if sinks is None:
            p = torch.softmax(s, dim=-1)
        else:
            col = sinks[b].float()[..., None]
            p = torch.softmax(torch.cat([s, col], dim=-1), dim=-1)[..., :-1]
        outs.append(p @ vf)
        del s, bias, rel_logits, p
    return torch.stack(outs)


def _check(sk, T, S, window, with_sinks=True, variant=-1, cap=None, split=0, seed=0):
    q, k, v, r, proj, sinks = _inputs(2, T, S, seed=seed, cap=cap)
    sinks = sinks if with_sinks else None
    out = sk.inkling_attention(q, k, v, r, proj, sinks, window=window, variant=variant,
                               split=split)
    torch.cuda.synchronize()
    ref = _reference(q, k, v, r, proj, sinks, window)
    torch.testing.assert_close(out.float(), ref, **TOL)
    return out


# Prefill and chunked prefill: the 64-row tile. S > EXTENT everywhere but T = 2048's own
# prefill, which reaches distances up to 2047 anyway.
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("T,S", [(17, 1300), (64, 1200), (64, 3000), (1000, 2100), (2048, 2048)])
def test_prefill(sk, T, S, window):
    _check(sk, T, S, window, variant=0)


# Decode and speculative decode (T <= 5 fills the 16-row tile at 3:1), split and whole.
@pytest.mark.parametrize("window", WINDOWS)
@pytest.mark.parametrize("T,S", [(1, 37), (1, 1025), (1, 3000), (3, 1500), (5, 4100)])
def test_decode(sk, T, S, window):
    assert sk._C.inkling_attention_decode_fits(H_Q, H_KV, T)
    _check(sk, T, S, window, variant=1)
    _check(sk, T, S, window, variant=1, split=1)


# The default picks decode for T = 1 and the prefill tile for T = 17 (51 rows per K/V head).
@pytest.mark.parametrize("T", [1, 17])
def test_default_and_no_sinks(sk, T):
    _check(sk, T, 2000, 1024, with_sinks=False)
    _check(sk, T, 2000, 0, with_sinks=False)


# k and v read in place as the leading rows of a cache with spare capacity.
@pytest.mark.parametrize("variant", [0, 1])
def test_cache_capacity(sk, variant):
    _check(sk, 1, 1500, 1024, variant=variant, cap=2048)
    _check(sk, 1, 1500, 0, variant=variant, cap=2048)


# The rows a decode step computes are the rows the prefill computed for the same positions:
# the last query of a 2000-token prefill against its own cache, by both kernels, and HF's
# bf16 sink dtype taken as given.
@pytest.mark.parametrize("window", WINDOWS)
def test_decode_matches_prefill_rows(sk, window):
    S = 2000
    q, k, v, r, proj, sinks = _inputs(2, S, S, seed=3)
    full = sk.inkling_attention(q, k, v, r, proj, sinks, window=window, variant=0)
    for n in (1, 4):
        tail = (q[:, :, -n:].contiguous(), r[:, :, -n:].contiguous(), sinks[:, :, -n:].contiguous())
        for variant in (0, 1):
            step = sk.inkling_attention(k=k, v=v, proj=proj, window=window, variant=variant,
                                        q=tail[0], r=tail[1], sinks=tail[2].to(torch.bfloat16))
            ref = sk.inkling_attention(k=k, v=v, proj=proj, window=window, variant=variant,
                                       q=tail[0], r=tail[1], sinks=tail[2].to(torch.bfloat16)
                                       .float())
            torch.testing.assert_close(step, ref, atol=0, rtol=0)
            torch.testing.assert_close(step.float(), full[:, :, -n:].float(), atol=1e-2,
                                       rtol=1e-2)


def test_rejects(sk):
    q, k, v, r, proj, sinks = _inputs(1, 17, 100)
    with pytest.raises(RuntimeError):  # 51 rows do not fit the decode tile
        sk.inkling_attention(q, k, v, r, proj, sinks, variant=1)
    with pytest.raises(RuntimeError):  # proj must be [16, extent]
        sk.inkling_attention(q, k, v, r, proj[:8].contiguous(), sinks)
    with pytest.raises(RuntimeError):  # r must be [B, H_q, S_q, 16]
        sk.inkling_attention(q, k, v, r[:, :, :16].contiguous(), proj, sinks)
