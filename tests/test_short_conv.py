"""Inkling 2 Mini's short convolutions and head norms (src/kernels/short_conv.cu) against
torch references copied from HF transformers 5.24.1 (inkling2_loop/modeling_inkling2_loop.py),
cache semantics included: HF's LinearAttentionLayer.update_conv_state keeps [B, C, 4] fp32 with
column 3 the newest; ours is [B, 3, C] in x's dtype, hf[:, :, 1:].transpose(1, 2).

The references run on the CPU in fp32 (cuDNN's conv could take TF32 on the GPU)."""

import math

import pytest
import torch
import torch.nn.functional as F

from conftest import TOL, dtype_id

K = 4
QK_CLIP = 22.627416997969522
REL_CLIP = 8.0
DTYPES = [torch.float32, torch.bfloat16]


# ---- copied from HF (decorators, the hub kernels and the Cache class stripped) -------------


def apply_mask_to_padding_states(hidden_states, attention_mask):
    if attention_mask is not None:
        dtype = hidden_states.dtype
        hidden_states = (hidden_states * attention_mask[:, :, None]).to(dtype)
    return hidden_states


def causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(hidden_states_new, weight.unsqueeze(1), bias, padding=0, groups=hidden_size)
    out = out[:, :, -seq_len:]
    return out.to(hidden_states.dtype)


def causal_conv1d_fn(hidden_states, weight, bias=None, activation=None, **kwargs):
    _, hidden_size, seq_len = hidden_states.shape
    padding = weight.shape[-1] - 1
    out = F.conv1d(hidden_states.to(weight.dtype), weight=weight.unsqueeze(1), bias=bias,
                   padding=padding, groups=hidden_size)[:, :, :seq_len]
    return out.to(hidden_states.dtype)


class ConvCache:
    """One conv state of HF's LinearAttentionLayer (record_past False)."""

    def __init__(self):
        self.conv_state = None

    def update_conv_state(self, conv_states, conv_kernel_size):
        if self.conv_state is None:  # prefill: pad to the kernel size, keep the last K
            full = conv_states
            if full.shape[-1] < conv_kernel_size:
                full = F.pad(full, (conv_kernel_size - full.shape[-1], 0), value=0)
            self.conv_state = torch.zeros((*conv_states.shape[:-1], conv_kernel_size),
                                          dtype=conv_states.dtype, device=conv_states.device)
        else:
            full = torch.cat([self.conv_state, conv_states], dim=-1)
        self.conv_state.copy_(full[..., -conv_kernel_size:])
        return full


def causal_conv1d_forward(weight, hidden_states, cache):
    seq_len = hidden_states.shape[-1]
    conv_state = cache.conv_state if cache is not None else None
    if conv_state is not None and seq_len == 1:
        return causal_conv1d_update(hidden_states, conv_state, weight)
    if cache is not None:
        hidden_states = cache.update_conv_state(hidden_states, conv_kernel_size=K)
    hidden_states = causal_conv1d_fn(hidden_states, weight)
    if cache is not None:
        hidden_states = hidden_states[:, :, -seq_len:]
    return hidden_states


def hf_short_conv(hidden_states, weight, cache=None, conv_mask=None):
    """Inkling2LoopShortConvolution.forward; weight = conv1d.weight.squeeze(1), [C, 4] fp32."""
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    residual = hidden_states
    hidden_states = apply_mask_to_padding_states(hidden_states, conv_mask)
    hidden_states = hidden_states.transpose(1, 2)
    hidden_states = causal_conv1d_forward(weight, hidden_states, cache)
    hidden_states = hidden_states.transpose(1, 2)
    hidden_states = (hidden_states + residual) / math.sqrt(K + 1)
    return hidden_states.to(dtype=input_dtype)


def hf_dynamic_short_conv(hidden_states, static_weight, dynamic_weights, num_heads, cache=None,
                          conv_mask=None):
    """Inkling2LoopDynamicShortConvolution.forward with weight_proj's output passed in
    (dynamic_weights [B, T, num_heads * 4])."""
    input_dtype = hidden_states.dtype
    batch_size, seq_len, hidden_size = hidden_states.shape
    dynamic_weights = dynamic_weights.float()
    dynamic_weights = dynamic_weights.view(batch_size, seq_len, num_heads, 1, K)
    weights = static_weight.float().T.view(num_heads, -1, K) + dynamic_weights
    hidden_states = hidden_states.float()
    residual = hidden_states
    context = apply_mask_to_padding_states(hidden_states, conv_mask).transpose(1, 2)
    if cache is not None:
        context = cache.update_conv_state(context, conv_kernel_size=K)
    context = F.pad(context, (K - 1, 0))[..., -(seq_len + K - 1):]
    windows = context.unfold(-1, K, 1).flip(-1).transpose(1, 2)
    windows = windows.reshape(batch_size, seq_len, num_heads, -1, K)
    hidden_states = (windows * weights).sum(-1).reshape(batch_size, seq_len, hidden_size)
    hidden_states = (hidden_states + residual) / math.sqrt(2 * K + 1)
    return hidden_states.to(input_dtype)


def hf_l2_clip(hidden_states, threshold):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    norm = torch.linalg.vector_norm(hidden_states, dim=-1, keepdim=True)
    return (hidden_states * (threshold / (norm + 1e-6)).clamp(max=1.0)).to(input_dtype)


def hf_scaled_l2_clip(hidden_states, scale, threshold):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.float() * scale.float()
    norm = torch.linalg.vector_norm(hidden_states, dim=-1, keepdim=True)
    return (hidden_states * (threshold / (norm + 1e-6)).clamp(max=1.0)).to(input_dtype)


def hf_log_scale(hidden_states, position_ids, alpha=0.1, n_floor=8192):
    input_dtype = hidden_states.dtype
    scale = 1.0 + alpha * torch.log(((position_ids + 1).float() / n_floor).clamp(min=1.0))
    return (hidden_states.float() * scale[..., None, None]).to(input_dtype)


def hf_log_scaled_l2_clip(hidden_states, scale, threshold, position_ids):
    input_dtype = hidden_states.dtype
    hidden_states = hf_l2_clip(hidden_states.float() * scale.float(), threshold).to(input_dtype)
    return hf_log_scale(hidden_states, position_ids)


def hf_unweighted_rmsnorm(hidden_states, eps=1e-6):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.float()
    hidden_states = hidden_states * torch.rsqrt(hidden_states.square().mean(-1, keepdim=True) + eps)
    return hidden_states.to(input_dtype)


# ---- helpers ------------------------------------------------------------------------------


def ours_state_from_hf(hf_state, dtype):
    return hf_state[:, :, 1:].transpose(1, 2).to(dtype).contiguous()


def make_conv(B, T, C, dtype, seed=0, heads=None):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, C, generator=g).to(dtype)
    w = 0.5 * torch.randn(C, K, generator=g)  # conv1d.weight.squeeze(1)
    sw = 0.5 * torch.randn(K, C, generator=g)  # static_weight
    taps = None if heads is None else 0.3 * torch.randn(B, T, heads, K, generator=g)
    return x, w, sw, taps


def run_ours(sk, x, w, state=None, mask=None, taps=None, epilogue=None, head_dim=128):
    xc = x.cuda()
    if taps is None:
        return sk.short_conv(xc, w.cuda(), state=state, mask=mask, epilogue=epilogue,
                             head_dim=head_dim)
    return sk.dynamic_short_conv(xc, w.cuda(), taps.cuda(), state=state, mask=mask,
                                 epilogue=epilogue, head_dim=head_dim)


def run_hf(x, w, cache=None, mask=None, taps=None, heads=None):
    if taps is None:
        return hf_short_conv(x, w, cache=cache, conv_mask=mask)
    return hf_dynamic_short_conv(x, w, taps.reshape(*taps.shape[:2], -1), heads, cache=cache,
                                 conv_mask=mask)


def close(got, ref, dtype):
    torch.testing.assert_close(got.cpu(), ref, **TOL[dtype])


# ---- static and dynamic conv against HF ----------------------------------------------------

SHAPES = [(1, 1, 1024), (2, 2, 2304), (3, 5, 1024), (1, 16, 2304), (2, 17, 1024),
          (1, 100, 2304), (2, 1000, 1024)]


@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("B,T,C", SHAPES)
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_prefill_matches_hf(sk, dtype, B, T, C, dynamic):
    heads = 8 if dynamic else None
    x, w, sw, taps = make_conv(B, T, C, dtype, heads=heads)
    weight = sw if dynamic else w
    cache = ConvCache()
    ref = run_hf(x, weight, cache=cache, taps=taps, heads=heads)
    state = sk.short_conv_state(B, C, dtype)
    got = run_ours(sk, x, weight, state=state, taps=taps)
    assert got.shape == x.shape and got.dtype == dtype
    close(got, ref, dtype)
    # the state is exactly the cache's last three inputs
    assert torch.equal(state.cpu(), ours_state_from_hf(cache.conv_state, dtype))
    # without a state: zeros before t = 0, same output
    assert torch.equal(run_ours(sk, x, weight, taps=taps), got)


@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("split", [1, 2, 3, 4, 5, 15, 16, 17, 40, 63])
def test_two_chunks_equal_one_prefill(sk, dtype, dynamic, split):
    B, T, C = 2, 64, 1024
    heads = 8 if dynamic else None
    x, w, sw, taps = make_conv(B, T, C, dtype, seed=1, heads=heads)
    weight = sw if dynamic else w
    s1 = sk.short_conv_state(B, C, dtype)
    s1.copy_(torch.randn(B, 3, C).to(dtype))  # a state left by an earlier call
    s2 = s1.clone()
    full = run_ours(sk, x, weight, state=s1, taps=taps)
    t_a = None if taps is None else taps[:, :split]
    t_b = None if taps is None else taps[:, split:]
    a = run_ours(sk, x[:, :split].contiguous(), weight, state=s2, taps=t_a)
    b = run_ours(sk, x[:, split:].contiguous(), weight, state=s2, taps=t_b)
    assert torch.equal(torch.cat([a, b], dim=1), full)
    assert torch.equal(s2, s1)


@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("B,C", [(1, 2304), (8, 1024), (64, 2304)])
def test_decode_steps_equal_prefill(sk, dtype, dynamic, B, C):
    T0, steps = 5, 6
    heads = 8 if dynamic else None
    if dynamic and C % (8 * heads):
        pytest.skip("heads must split C into multiples of 8")
    x, w, sw, taps = make_conv(B, T0 + steps, C, dtype, seed=2, heads=heads)
    weight = sw if dynamic else w
    full = run_ours(sk, x, weight, taps=taps)
    state = sk.short_conv_state(B, C, dtype)
    outs = [run_ours(sk, x[:, :T0].contiguous(), weight, state=state,
                     taps=None if taps is None else taps[:, :T0])]
    cache = ConvCache()
    refs = [run_hf(x[:, :T0], weight, cache=cache,
                   taps=None if taps is None else taps[:, :T0], heads=heads)]
    for t in range(T0, T0 + steps):
        tt = None if taps is None else taps[:, t:t + 1]
        outs.append(run_ours(sk, x[:, t:t + 1].contiguous(), weight, state=state, taps=tt))
        refs.append(run_hf(x[:, t:t + 1], weight, cache=cache, taps=tt, heads=heads))
        assert torch.equal(state.cpu(), ours_state_from_hf(cache.conv_state, dtype))
    assert torch.equal(torch.cat(outs, dim=1), full)
    close(torch.cat(outs, dim=1), torch.cat(refs, dim=1), dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
def test_padding_mask_matches_hf(sk, dtype, dynamic):
    B, T, C = 3, 40, 1024
    heads = 8 if dynamic else None
    x, w, sw, taps = make_conv(B, T, C, dtype, seed=3, heads=heads)
    weight = sw if dynamic else w
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[0, :7] = False  # left padding
    mask[1, 30:] = False  # right padding
    mask[2, ::5] = False  # scattered
    cache = ConvCache()
    ref = run_hf(x, weight, cache=cache, mask=mask, taps=taps, heads=heads)
    state = sk.short_conv_state(B, C, dtype)
    got = run_ours(sk, x, weight, state=state, mask=mask.cuda(), taps=taps)
    close(got, ref, dtype)
    assert torch.equal(state.cpu(), ours_state_from_hf(cache.conv_state, dtype))
    # an integer mask is taken too
    assert torch.equal(run_ours(sk, x, weight, mask=mask.cuda().int(), taps=taps), got)


@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
def test_bf16_taps(sk, dtype):
    B, T, C, heads = 2, 33, 1024, 8
    x, _, sw, taps = make_conv(B, T, C, dtype, seed=4, heads=heads)
    taps16 = taps.bfloat16()
    ref = run_hf(x, sw, taps=taps16, heads=heads)
    close(run_ours(sk, x, sw, taps=taps16), ref, dtype)


# ---- the fused epilogues -------------------------------------------------------------------


def spread_heads(x, heads, seed):
    """Scale every (token, head) of x by a log-uniform factor so that the conv output norms
    straddle the clip threshold (a random 128-vector has norm ~11)."""
    g = torch.Generator().manual_seed(seed)
    B, T, C = x.shape
    f = torch.exp(torch.empty(B, T, heads, 1).uniform_(math.log(0.5), math.log(6.0), generator=g))
    return (x.float().view(B, T, heads, -1) * f).view(B, T, C).to(x.dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize("epilogue", ["l2clip", "rms"])
def test_fused_epilogue_matches_hf(sk, dtype, dynamic, epilogue):
    B, T, C, D = 2, 50, 1024, 128
    heads = 8 if dynamic else None
    x, w, sw, taps = make_conv(B, T, C, dtype, seed=5, heads=heads)
    x = spread_heads(x, C // D, seed=6)
    weight = sw if dynamic else w
    conv = run_hf(x, weight, taps=taps, heads=heads).view(B, T, C // D, D)
    if epilogue == "l2clip":
        norms = conv.float().norm(dim=-1)
        assert (norms < QK_CLIP).any() and (norms > QK_CLIP).any()
        ref = hf_l2_clip(conv, QK_CLIP)
    else:
        ref = hf_unweighted_rmsnorm(conv)
    state = sk.short_conv_state(B, C, dtype)
    got = run_ours(sk, x, weight, state=state, taps=taps, epilogue=epilogue)
    close(got.view(B, T, C // D, D), ref, dtype)
    # fused equals conv then the standalone op
    plain = run_ours(sk, x, weight, taps=taps).view(B, T, C // D, D)
    sep = sk.head_l2_clip(plain) if epilogue == "l2clip" else sk.head_rms_norm(plain)
    torch.testing.assert_close(got.view(B, T, C // D, D), sep, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
def test_l2_clip_around_threshold(sk, dtype):
    """Vectors of norm thr * (1 +- eps) for eps from 1e-1 down to 1e-4, and exactly thr."""
    D = 128
    g = torch.Generator().manual_seed(7)
    u = torch.randn(64, 2, D, generator=g)
    u = u / u.norm(dim=-1, keepdim=True)
    rel = torch.tensor([1e-1, 1e-2, 1e-3, 1e-4, 0.0, -1e-4, -1e-3, -1e-2, -1e-1, 1.0, -0.5, 3.0])
    rel = rel.repeat(64 * 2 // len(rel) + 1)[: 64 * 2].view(64, 2, 1)
    x = (u * QK_CLIP * (1 + rel)).to(dtype)
    got = sk.head_l2_clip(x.cuda(), QK_CLIP)
    ref = hf_l2_clip(x, QK_CLIP)
    torch.testing.assert_close(got.cpu(), ref, **TOL[dtype])
    n = got.float().norm(dim=-1).cpu()
    assert (n <= QK_CLIP * (1 + 1e-2 if dtype == torch.bfloat16 else 1 + 1e-5)).all()
    short = x.float().norm(dim=-1) < QK_CLIP / (1 + 1e-6)
    assert torch.equal(got.cpu()[short], x[short])  # shorter vectors come back unchanged


# ---- the standalone q op -------------------------------------------------------------------


@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("D,thr", [(128, QK_CLIP), (16, REL_CLIP)])
@pytest.mark.parametrize("log_scaling", [False, True], ids=["plain", "log"])
@pytest.mark.parametrize("pos_dtype", [torch.int64, torch.int32], ids=["i64", "i32"])
def test_q_scaled_l2_clip_matches_hf(sk, dtype, D, thr, log_scaling, pos_dtype):
    B, T, H = 2, 37, 24
    g = torch.Generator().manual_seed(8)
    x = (3.0 * torch.randn(B, T, H, D, generator=g)).to(dtype)
    scale = (1.0 + 0.5 * torch.randn(H, D, generator=g)).to(dtype)  # loaded as a bf16 param
    pos = torch.randint(0, 100000, (B, T), generator=g)
    pos[0, :5] = torch.tensor([0, 8190, 8191, 8192, 16383])
    if log_scaling:
        ref = hf_log_scaled_l2_clip(x, scale, thr, pos)
        got = sk.head_l2_clip(x.cuda(), thr, scale=scale.cuda(), positions=pos.to(pos_dtype).cuda())
    else:
        if pos_dtype == torch.int32:
            pytest.skip("positions only matter with log scaling")
        ref = hf_scaled_l2_clip(x, scale, thr)
        got = sk.head_l2_clip(x.cuda(), thr, scale=scale.cuda())
    close(got, ref, dtype)


@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("D", [16, 128])
def test_head_rms_norm_and_inplace(sk, dtype, D):
    x = (2.0 * torch.randn(3, 11, 8, D)).to(dtype).cuda()
    ref = hf_unweighted_rmsnorm(x.cpu())
    got = sk.head_rms_norm(x)
    close(got, ref, dtype)
    y = x.clone()
    out = sk.head_l2_clip(y, QK_CLIP, inplace=True)
    assert out.data_ptr() == y.data_ptr()
    torch.testing.assert_close(y.cpu(), hf_l2_clip(x.cpu(), QK_CLIP), **TOL[dtype])


def test_bad_shapes_raise(sk):
    x = torch.randn(1, 4, 1024, device="cuda")
    with pytest.raises(RuntimeError):
        sk.short_conv(x, torch.randn(4, 1024, device="cuda"))  # dynamic's layout
    with pytest.raises(RuntimeError):
        sk.short_conv(x, torch.randn(1024, 4, device="cuda"),
                      state=torch.zeros(1, 4, 1024, device="cuda"))
    with pytest.raises(RuntimeError):
        sk.short_conv(x, torch.randn(1024, 4, device="cuda"),
                      state=torch.zeros(1, 3, 1024, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(RuntimeError):
        sk.dynamic_short_conv(x, torch.randn(4, 1024, device="cuda"),
                              torch.randn(1, 4, 3, 4, device="cuda"))  # 3 heads do not split C
    with pytest.raises(ValueError):
        sk.short_conv(x, torch.randn(1024, 4, device="cuda"), epilogue="l2clip", head_dim=96)
