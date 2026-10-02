"""The activation-aware int4 scales of spark_kernels.awq on a two-layer random model: folding
them into the bf16 weights leaves the model's function alone (up to bf16 rounding), every
searched scale is no worse than plain round to nearest on its projection, and the scaled
weights quantize to a model closer to the bf16 one."""

import pytest
import torch

E = pytest.importorskip("spark_kernels.engine")
L = pytest.importorskip("spark_kernels.layer")
A = pytest.importorskip("spark_kernels.awq")

N_LAYERS, VOCAB = 2, 1024


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _logits(weights, rope, prompts, fmt=None):
    model = (E.TorchModel(weights, rope) if fmt is None else
             E.SparkModel(weights, rope, weights_format=fmt, keep_bf16=True))
    eng = E.Engine(model, N_LAYERS, max_batch=4, max_seq=256, num_pages=64, graphs=False)
    return torch.cat(eng.prompt_logits(prompts))


def _outliers(w):
    """Random weights with a few large norm channels, the activation outliers AWQ is for."""
    for lw in w.layers:
        lw.attn_norm[:8] *= 20
        lw.mlp_norm[8:16] *= 20
    return w


def test_awq_scales_keep_the_function_and_help_int4(sk):
    rope = L.RoPE(256)
    g = torch.Generator().manual_seed(0)
    calib = torch.randint(0, VOCAB, (4, 128), generator=g)
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in (50, 120)]
    base = _outliers(E.ModelWeights.random(N_LAYERS, VOCAB, seed=5))
    scaled = _outliers(E.ModelWeights.random(N_LAYERS, VOCAB, seed=5))
    logs = []
    scales = A.search(scaled, rope, calib, grid=10, log=logs.append)
    assert len(scales) == N_LAYERS and len(logs) == N_LAYERS
    for sc in scales:
        assert set(sc) == {"qkv", "o", "gate_up", "down"}
    # the same function in bf16
    ref = _logits(base, rope, prompts)
    assert _rel(_logits(scaled, rope, prompts), ref) < 3e-2
    # apply() on fresh weights reproduces what search() left behind
    again = _outliers(E.ModelWeights.random(N_LAYERS, VOCAB, seed=5))
    A.apply(again, scales)
    for a, b in zip(again.layers, scaled.layers, strict=True):
        assert torch.equal(a.w_gate_up, b.w_gate_up) and torch.equal(a.attn_norm, b.attn_norm)
    # and int4 of the scaled weights is no further from bf16 than int4 of the plain ones
    rtn = _rel(_logits(base, rope, prompts, "int4"), ref)
    awq = _rel(_logits(scaled, rope, prompts, "int4"), ref)
    assert awq <= rtn * 1.02, f"awq {awq:.4f} against round to nearest {rtn:.4f}"
