"""Inkling 2 Mini's block attention residual and loop message gates vs the HF modeling code.

The references below are copied from transformers 5.24.1 `modeling_inkling2_loop.py`
(`Inkling2LoopUnweightedRMSNorm`, `Inkling2LoopBlockAttentionResidual` with the eager attention
it calls, `Inkling2LoopMessageGate`), the modules' state passed in as arguments. Shapes are the
model's: H = 2304, rank 64, nb = 1..4 block states (4 is the most the forward reaches, at
layers 18-23 and final_input), nq = 1, 3 (full layers' q/k/v) and 4 (message_input).
"""

import pytest
import torch
import torch.nn.functional as F

H = 2304
RANK = 64
EPS = 1e-6


def hf_unweighted_rms_norm(hidden_states, eps=EPS):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    hidden_states = hidden_states * torch.rsqrt(hidden_states.square().mean(-1, keepdim=True) + eps)
    return hidden_states.to(input_dtype)


def hf_block_attention_residual(block_states, partial_block_states, query):
    """[..., nb, H], [..., H], [nq, H] or [H] -> [..., nq, H] (HF unbinds the nq axis)."""
    stacked_states = torch.cat([block_states, partial_block_states.unsqueeze(-2)], dim=-2).float()
    stacked_states = hf_unweighted_rms_norm(stacked_states)
    hidden_size = stacked_states.shape[-1]
    query_states = query.float().view(-1, hidden_size).expand(*stacked_states.shape[:-2], -1, -1)
    # eager_attention_forward with one "head" per token, keys = values = the stacked states
    attn_weights = torch.matmul(query_states, stacked_states.transpose(-1, -2)) * hidden_size**-0.5
    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    return torch.matmul(attn_weights, stacked_states)


def hf_message_gate(hidden_states, message, down_proj, forget_gate_proj, message_gate_proj):
    hidden_states, message = hidden_states.float(), message.float()
    gate_hidden_states = torch.cat(
        [hf_unweighted_rms_norm(hidden_states), hf_unweighted_rms_norm(message)], dim=-1)
    gate_hidden_states = F.linear(gate_hidden_states.to(down_proj.dtype), down_proj).float()
    forget_gate = torch.tanh(F.linear(gate_hidden_states, forget_gate_proj))
    message_gate = torch.tanh(F.linear(gate_hidden_states, message_gate_proj))
    return (1 - forget_gate) * hidden_states + message_gate * message


def _sources(T, nb, partial_dtype, gen):
    # each source at its own scale, so the normalization matters
    scale = torch.rand(T, nb + 1, 1, generator=gen) * 4 + 0.25
    x = torch.randn(T, nb + 1, H, generator=gen) * scale
    blocks = x[:, :nb].to(torch.bfloat16).cuda()
    partial = x[:, nb].to(partial_dtype).cuda()
    return blocks, partial


def _queries(nq, gen, gain=1.0):
    # learned queries; gain 1 puts the logits at O(1), gain 8 makes the softmax nearly one-hot
    return (torch.randn(nq, H, generator=gen) * gain).cuda()


@pytest.mark.parametrize("T", [1, 7, 256])
@pytest.mark.parametrize("nq", [1, 3, 4])
@pytest.mark.parametrize("nb", [1, 2, 3, 4])
@pytest.mark.parametrize("partial_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"])
def test_attn_res_matches_hf(sk, T, nq, nb, partial_dtype):
    gen = torch.Generator().manual_seed(1000 * nb + 10 * nq + T)
    blocks, partial = _sources(T, nb, partial_dtype, gen)
    query = _queries(nq, gen)
    out = sk.attn_res(blocks, partial, query)
    assert out.shape == (nq, T, H) and out.dtype == torch.float32
    ref = hf_block_attention_residual(blocks, partial, query)  # [T, nq, H]
    torch.testing.assert_close(out, ref.transpose(0, 1), atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("gain", [0.0, 8.0])
def test_attn_res_extreme_queries(sk, gain):
    """Zero queries (HF's init: a plain mean of the normalized sources) and sharp ones."""
    gen = torch.Generator().manual_seed(7)
    blocks, partial = _sources(33, 4, torch.float32, gen)
    query = _queries(3, gen, gain)
    out = sk.attn_res(blocks, partial, query)
    ref = hf_block_attention_residual(blocks, partial, query)
    torch.testing.assert_close(out, ref.transpose(0, 1), atol=1e-4, rtol=1e-4)


def test_attn_res_layouts(sk):
    """[B, S, nb, H] like HF, a slice of a preallocated [T, 8, H] buffer, an [nb, T, H] view,
    no block states at all, and the 48 queries of a whole loop pass's message inputs."""
    gen = torch.Generator().manual_seed(3)
    blocks, partial = _sources(2 * 5, 3, torch.bfloat16, gen)
    q1 = _queries(1, gen)
    out = sk.attn_res(blocks.view(2, 5, 3, H), partial.view(2, 5, H), q1[0])
    ref = hf_block_attention_residual(blocks, partial, q1)
    torch.testing.assert_close(out.view(10, H), ref[:, 0], atol=1e-4, rtol=1e-4)

    buf = torch.zeros(10, 8, H, dtype=torch.bfloat16, device="cuda")
    buf[:, :3] = blocks
    torch.testing.assert_close(sk.attn_res(buf[:, :3], partial, q1)[0], ref[:, 0], atol=1e-4,
                               rtol=1e-4)
    by_block = blocks.transpose(0, 1).contiguous()  # [nb, T, H], one tensor per block state
    torch.testing.assert_close(sk.attn_res(by_block.transpose(0, 1), partial, q1)[0], ref[:, 0],
                               atol=1e-4, rtol=1e-4)

    none = sk.attn_res(None, partial, q1)[0]
    torch.testing.assert_close(none, hf_unweighted_rms_norm(partial.float()), atol=1e-5,
                               rtol=1e-5)

    q48 = _queries(48, gen)
    torch.testing.assert_close(sk.attn_res(blocks[:, :1], partial, q48),
                               hf_block_attention_residual(blocks[:, :1], partial, q48)
                               .transpose(0, 1), atol=1e-4, rtol=1e-4)


def _gate_weights(gen):
    # the checkpoint stores wf and wg in bf16 and HF upcasts them: fp32 holding bf16 values
    wd = (torch.randn(RANK, 2 * H, generator=gen) / (2 * H) ** 0.5).to(torch.bfloat16).cuda()
    wf = (torch.randn(H, RANK, generator=gen) / RANK**0.5).to(torch.bfloat16).float().cuda()
    wg = (torch.randn(H, RANK, generator=gen) / RANK**0.5).to(torch.bfloat16).float().cuda()
    return wd, wf, wg


def _gate_err(out, args):
    """max |out - exact| next to HF's own max error against the same exact (fp64) result."""
    h, m, wd, wf, wg = (a.double() for a in args)
    n = lambda x: x * torch.rsqrt(x.square().mean(-1, keepdim=True) + EPS)  # noqa: E731
    z = F.linear(torch.cat([n(h), n(m)], dim=-1), wd)
    exact = (1 - torch.tanh(F.linear(z, wf))) * h + torch.tanh(F.linear(z, wg)) * m
    h, m, wd, wf, wg = args
    hf = hf_message_gate(h, m, wd, wf, wg)
    return (out.double() - exact).abs().max().item(), (hf.double() - exact).abs().max().item()


@pytest.mark.parametrize("T", [1, 7, 64, 256])
@pytest.mark.parametrize("path", ["decode", "prefill"])
@pytest.mark.parametrize("up_dtype", [torch.float32, torch.bfloat16], ids=["f32", "bf16"])
def test_message_gate_matches_hf(sk, T, path, up_dtype):
    if path == "decode" and T > 64:
        pytest.skip("the decode kernel takes at most 64 tokens")
    gen = torch.Generator().manual_seed(T)
    h = (torch.randn(T, H, generator=gen) * 1.5).cuda()
    m = torch.randn(T, H, generator=gen).cuda()
    wd, wf, wg = _gate_weights(gen)
    out = sk.message_gate(h, m, wd, wf.to(up_dtype), wg.to(up_dtype), path=path)
    ref = hf_message_gate(h, m, wd, wf, wg)
    # HF rounds z (and on the prefill path so do we) to bf16: a flipped rounding moves a gate
    # by ~1/256 of its argument. Compare the error against exact math with HF's.
    ours, theirs = _gate_err(out, (h, m, wd, wf, wg))
    assert ours <= 2 * theirs + 1e-4, (ours, theirs)
    torch.testing.assert_close(out, ref, atol=5e-2, rtol=0)


@pytest.mark.parametrize("T", [1, 7, 256])
@pytest.mark.parametrize("shared_h", [True, False], ids=["sliding", "full"])
def test_message_gates_qkv_one_launch(sk, T, shared_h):
    """A loop layer's q, k, v gates together: sliding layers feed all three the same h,
    full layers one h per query."""
    gen = torch.Generator().manual_seed(100 + T)
    hs = [(torch.randn(T, H, generator=gen)).cuda() for _ in range(1 if shared_h else 3)]
    hs = hs * 3 if shared_h else hs
    ms = [torch.randn(T, H, generator=gen).cuda() for _ in range(3)]
    ws = [_gate_weights(gen) for _ in range(3)]
    outs = sk.message_gates(hs, ms, [w[0] for w in ws], [w[1] for w in ws], [w[2] for w in ws])
    assert len(outs) == 3
    for g in range(3):
        ours, theirs = _gate_err(outs[g], (hs[g], ms[g], *ws[g]))
        assert ours <= 2 * theirs + 1e-4, (g, ours, theirs)
        # and the same as the gate run on its own
        alone = sk.message_gate(hs[g], ms[g], *ws[g])
        torch.testing.assert_close(outs[g], alone, atol=2e-2, rtol=0)


def test_message_gate_leading_dims(sk):
    gen = torch.Generator().manual_seed(5)
    h = torch.randn(2, 3, H, generator=gen).cuda()
    m = torch.randn(2, 3, H, generator=gen).cuda()
    wd, wf, wg = _gate_weights(gen)
    out = sk.message_gate(h, m, wd, wf, wg)
    assert out.shape == (2, 3, H)
    torch.testing.assert_close(out, hf_message_gate(h, m, wd, wf, wg), atol=5e-2, rtol=0)
