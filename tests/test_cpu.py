"""The CPU-side ops (include/spark/cpu.h, the GB10's Arm cores) against the plain-PyTorch
references: the 1-bit and ternary GEMV against reference.w1_dequantize and a torch matmul,
the routing against spark_kernels.moe.route, sampling at temperature 0 against argmax, and
the bandwidth probe for a sane number.

Everything here runs on the CPU, but it needs the built extension (the ops live in
spark_kernels._C), so the whole file skips when the import fails."""

from __future__ import annotations

import pytest
import torch

sk = pytest.importorskip("spark_kernels")
from spark_kernels import moe  # noqa: E402
from spark_kernels import reference as ref  # noqa: E402

BITS = [1, 2]
# (N, K): a one-group column, a few groups, and the DeepSeek V4.1 Flash expert projection
# in both orientations.
SHAPES = [(16, 128), (48, 384), (2304, 5120), (5120, 2304)]
# Rung 1 multiplies the exact +-s weights and differs from the reference by summation order
# only, the w1gemm test's tolerance. Rung 2 rounds the activations to int8 per group.
TOL_EXACT = dict(atol=3e-2, rtol=3e-2)


def _weight(K: int, N: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(K, N, generator=g) / K**0.5).to(torch.bfloat16)


def _row(K: int, M: int | None = None, seed: int = 1) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    shape = (K,) if M is None else (M, K)
    return torch.randn(*shape, generator=g).to(torch.bfloat16)


def _variants() -> list[int]:
    return [v for v in range(sk._C.cpu_w1gemv_num_variants()) if sk._C.cpu_w1gemv_supports(v)]


# ---- the GEMV -------------------------------------------------------------------------------


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: f"N{s[0]}_K{s[1]}")
@pytest.mark.parametrize("variant", _variants())
def test_cpu_w1gemv_matches_dequantized_matmul(shape, bits, variant):
    N, K = shape
    w = _weight(K, N)
    packed, scales = ref.w1_quantize(w, bits)
    a = _row(K)
    y = sk.cpu_w1gemv(a, sk.W1Weight(packed, scales, bits), threads=4, variant=variant)
    assert y.shape == (N,) and y.dtype == torch.bfloat16
    want = (a.float() @ ref.w1_dequantize(packed, scales, bits).float()).to(torch.bfloat16)
    if variant == 2:
        # int8 activations: the error is a few parts in a thousand of the row's scale.
        err = (y.float() - want.float()).abs().max()
        assert err <= 0.05 * want.float().abs().max() + 1e-3
    else:
        torch.testing.assert_close(y, want, **TOL_EXACT)


@pytest.mark.parametrize("bits", BITS)
def test_cpu_w1gemv_rows_and_thread_counts_agree(bits):
    N, K = 80, 640
    packed, scales = ref.w1_quantize(_weight(K, N, seed=3), bits)
    w1 = sk.W1Weight(packed, scales, bits)
    a = _row(K, M=5, seed=4)
    y = sk.cpu_w1gemv(a, w1, threads=1)
    assert y.shape == (5, N)
    for threads in (2, 3, 7, 20):
        torch.testing.assert_close(sk.cpu_w1gemv(a, w1, threads=threads), y, atol=0, rtol=0)
    for m in range(5):
        torch.testing.assert_close(sk.cpu_w1gemv(a[m], w1, threads=2), y[m], atol=0, rtol=0)


def test_cpu_w1gemv_rejects_cuda_and_bad_shapes():
    packed, scales = ref.w1_quantize(_weight(128, 16), 1)
    w1 = sk.W1Weight(packed, scales, 1)
    with pytest.raises(RuntimeError):
        sk.cpu_w1gemv(_row(256), w1)  # K does not match the weight
    with pytest.raises(RuntimeError):
        sk.cpu_w1gemv(_row(128).float(), w1)  # not bf16
    if torch.cuda.is_available():
        with pytest.raises(RuntimeError):
            sk.cpu_w1gemv(_row(128).cuda(), w1)


# ---- routing ------------------------------------------------------------------------------------


@pytest.mark.parametrize("with_bias", [False, True], ids=["nobias", "bias"])
@pytest.mark.parametrize("E,k", [(8, 3), (384, 6)])
def test_cpu_route_matches_moe_route(E, k, with_bias):
    g = torch.Generator().manual_seed(0)
    T = 16
    logits = torch.randn(T, E, generator=g)
    bias = torch.randn(E, generator=g) * 3 if with_bias else None
    ids, w = sk.cpu_route(logits, k, bias, scale=2.5)
    want_ids, want_w = moe.route(logits, k, bias, scale=2.5)
    assert ids.dtype == torch.int32 and w.dtype == torch.float32
    assert ids.shape == (T, k) and w.shape == (T, k)
    # The same set of experts per token (the order within the k is not part of the contract),
    # and the same weight for each expert.
    assert torch.equal(ids.long().sort(-1).values, want_ids.long().sort(-1).values)
    got = torch.zeros(T, E).scatter_(1, ids.long(), w)
    want = torch.zeros(T, E).scatter_(1, want_ids.long(), want_w)
    torch.testing.assert_close(got, want, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(w.sum(-1), torch.full((T,), 2.5))


def test_cpu_route_one_token_in_descending_pick_order():
    logits = torch.tensor([0.1, 2.0, -1.0, 0.5, 1.5])
    ids, w = sk.cpu_route(logits, 3)
    assert ids.tolist() == [1, 4, 3]
    s = torch.sigmoid(logits)[[1, 4, 3]]
    torch.testing.assert_close(w, s / s.sum())


# ---- sampling -----------------------------------------------------------------------------------


def test_cpu_sample_temperature_zero_is_argmax():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(4, 129280, generator=g).to(torch.bfloat16)
    out = sk.cpu_sample(logits, temperature=0.0)
    assert out.dtype == torch.int64 and out.shape == (4,)
    assert torch.equal(out, logits.float().argmax(-1))
    # A tie goes to the first index, as torch.argmax reports it.
    tied = torch.zeros(10, dtype=torch.bfloat16)
    tied[3] = tied[7] = 1.0
    assert int(sk.cpu_sample(tied, 0.0)) == 3


def test_cpu_sample_top_k_one_is_argmax_and_draws_stay_in_the_kept_set():
    g = torch.Generator().manual_seed(1)
    logits = torch.randn(2, 4096, generator=g).to(torch.bfloat16)
    assert torch.equal(sk.cpu_sample(logits, 0.8, top_k=1, seed=5), logits.float().argmax(-1))
    top = logits.float().topk(20, dim=-1).indices
    for offset in range(8):
        tok = sk.cpu_sample(logits, 1.0, top_k=20, top_p=0.95, seed=11, offset=offset)
        assert all(int(tok[b]) in top[b].tolist() for b in range(2))
    # The same (seed, offset) draws the same token; another offset is allowed to differ.
    a = sk.cpu_sample(logits, 1.0, top_k=0, top_p=0.9, seed=3, offset=2)
    b = sk.cpu_sample(logits, 1.0, top_k=0, top_p=0.9, seed=3, offset=2)
    assert torch.equal(a, b)


# ---- the probe -----------------------------------------------------------------------------------


def test_cpu_bandwidth_probe_returns_a_rate():
    gbps = sk.cpu_bandwidth_probe(8 << 20, threads=2, iters=2)
    assert isinstance(gbps, float) and gbps > 0
