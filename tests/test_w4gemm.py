"""Parity tests for the W4A16 GEMM (int4 weights, bf16 activations), its quantizer and its
repack, against the plain-PyTorch versions in spark_kernels.reference."""

import pytest
import torch

from spark_kernels import reference as ref

# (K, N) weight shapes: the smallest legal one, one with a single 16-column strip, the
# Llama-3-8B o projection, and shapes whose K is not a multiple of the four-group stage of
# the decode blocks (K = 384, 1152: 3 and 9 groups) so a block ends on a partial stage.
WEIGHTS = [(128, 16), (256, 48), (384, 4096), (1152, 256), (4096, 4096)]
# Token counts: decode (1, 3), each n8 / m-tile boundary of the block shapes (8, 9, 16, 17,
# 33, 64), and chunks of 64 with a ragged last one (100, 256).
MS = [1, 3, 8, 9, 16, 17, 33, 64, 100, 256]
# Every block shape of variant 2 against a weight wide enough for all of them.
W4_TOL = dict(atol=3e-2, rtol=3e-2)


def _variants():
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants("w4gemm")))
    except Exception:
        return [0]


def _weight(K, N, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(K, N, generator=g) / K**0.5).to(device="cuda", dtype=torch.bfloat16)


@pytest.mark.parametrize("asym", [False, True], ids=["sym", "asym"])
@pytest.mark.parametrize("shape", WEIGHTS, ids=lambda s: f"K{s[0]}_N{s[1]}")
def test_w4_quantize_matches_reference(sk, shape, asym):
    K, N = shape
    w = _weight(K, N)
    w[:128, :16] = 0  # an all-zero group: scale 0, every code the zero point
    qweight, scales, zeros = sk.w4_quantize(w, asym=asym)
    q_ref, s_ref, z_ref = ref.w4_quantize(w, asym=asym)
    assert qweight.shape == (K // 8, N) and qweight.dtype == torch.int32
    assert scales.shape == (K // 128, N) and scales.dtype == torch.bfloat16
    # Bit for bit: the same fp32 operations on both sides.
    assert torch.equal(ref.w4_unpack_gptq(qweight), q_ref)
    assert torch.equal(scales.view(torch.int16), s_ref.view(torch.int16))
    if asym:
        assert torch.equal(zeros, z_ref)
    else:
        assert zeros is None
    # Round to nearest: every weight within half a step of its reconstruction, plus the
    # rounding of (q - z) s to bf16 (|q - z| <= 15, relative 2^-9: under 0.03 of a step).
    # Asymmetric codes can also clip at the top: z is rounded and s rounded down to bf16
    # stretches (hi - lo) / s to up to 15 (1 + 2^-8), a further 0.06 of a step at most.
    deq = ref.w4_dequantize(q_ref, s_ref, z_ref).float()
    step = s_ref.float().repeat_interleave(128, dim=0)
    bound = 0.6 if asym else 0.53
    assert ((deq - w.float()).abs() <= bound * step + 1e-6).all()


@pytest.mark.parametrize("shape", WEIGHTS, ids=lambda s: f"K{s[0]}_N{s[1]}")
def test_w4_repack_matches_reference(sk, shape):
    K, N = shape
    q = torch.randint(0, 16, (K, N), dtype=torch.uint8, device="cuda")
    qweight = ref.w4_pack_gptq(q)
    packed = sk.w4_repack(qweight)
    assert packed.shape == (N // 16, 2 * K) and packed.dtype == torch.int32
    assert torch.equal(packed, ref.w4_repack(qweight))
    assert torch.equal(ref.w4_unpack(packed), q)


def _case(sk, M, K, N, variant, asym=False, reps=1):
    w = _weight(K, N, seed=K + N)
    torch.manual_seed(M)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    ww = sk.W4Weight.quantize(w, asym=asym)
    q, s, z = ref.w4_quantize(w, asym=asym)
    expected = ref.w4gemm(a, q, s, z)
    for _ in range(reps):
        got = sk.w4gemm(a, ww, variant=variant)
        assert got.shape == (M, N) and got.dtype == torch.bfloat16
        torch.testing.assert_close(got.float(), expected.float(), **W4_TOL)


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("shape", WEIGHTS, ids=lambda s: f"K{s[0]}_N{s[1]}")
def test_w4gemm_matches_dequantized_matmul(sk, shape, M, variant):
    K, N = shape
    if variant == 0 and M * N * K > 2**28:
        pytest.skip("the naive rung is only checked on the small shapes")
    # Three calls: the K-split workspace and its counters are put back to zero by the kernel.
    _case(sk, M, K, N, variant, reps=3 if variant == 2 else 1)


@pytest.mark.parametrize("M", [1, 16, 64, 100])
@pytest.mark.parametrize("variant", _variants())
def test_w4gemm_asymmetric(sk, M, variant):
    _case(sk, M, 1152, 256, variant, asym=True)
    if variant > 0:
        _case(sk, M, 4096, 4096, variant, asym=True)


# Llama-3-8B projections (fused q|k|v, o, fused gate|up, down) at decode and serving sizes.
LLAMA = [(4096, 6144), (4096, 4096), (4096, 28672), (14336, 4096)]


@pytest.mark.parametrize("M", [1, 16, 256])
@pytest.mark.parametrize("shape", LLAMA, ids=lambda s: f"K{s[0]}_N{s[1]}")
def test_w4gemm_llama_shapes(sk, shape, M):
    K, N = shape
    _case(sk, M, K, N, -1, reps=2)


def test_w4gemm_weight_bits_are_the_reference(sk):
    """The weights the kernel multiplies are exactly reference.w4_dequantize: with a one-hot
    activation row the output is the dequantized weight row itself, whatever the variant."""
    K, N = 256, 64
    w = _weight(K, N)
    ww = sk.W4Weight.quantize(w, asym=True)
    q, s, z = ref.w4_quantize(w, asym=True)
    deq = ref.w4_dequantize(q, s, z)
    a = torch.zeros(K, K, device="cuda", dtype=torch.bfloat16)
    a[torch.arange(K), torch.arange(K)] = 1
    for v in _variants():
        assert torch.equal(sk.w4gemm(a, ww, variant=v), deq)


def test_w4gemm_rejects_bad_input(sk):
    w = _weight(256, 64)
    ww = sk.W4Weight.quantize(w)
    a = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        sk.w4gemm(a[:, :128].contiguous(), ww)  # K mismatch
    with pytest.raises(RuntimeError):
        sk.w4gemm(a.float(), ww)  # not bf16
    with pytest.raises(RuntimeError):
        sk.w4_quantize(torch.randn(200, 64, device="cuda", dtype=torch.bfloat16))  # K % 128
    with pytest.raises(RuntimeError):
        sk.w4gemm(a, ww, variant=sk.num_variants("w4gemm"))
    assert ww.shape == (256, 64)
