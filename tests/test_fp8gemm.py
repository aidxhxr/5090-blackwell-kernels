"""Parity of sk.fp8gemm with torch._scaled_mm (cuBLASLt) on the same e4m3 bytes, scales and
TN layout: a is [M, K], b_t is [N, K], both K-contiguous, per-tensor fp32 scales, bf16 out."""

import pytest
import torch

# Every tile the ladder picks (128x128, 64x128, 64x64), the split-K tail (a K long enough to
# split, few tiles), and ragged M for the zero-filled rows.
FP8_SHAPES_V1 = [(256, 256, 256), (272, 128, 4096), (208, 1088, 192), (1024, 1024, 1024),
                 (100, 1024, 1024), (2048, 2048, 2048)]
# variant 0: M % 16 == 0
FP8_SHAPES_V0 = [(16, 64, 64), (256, 256, 256), (272, 1024, 4096)]
# The decode path (M <= 64): every row tile (16, 32, 64) against N from {256, 4096, 11008}, the
# shapes with few strips (N = 256, 1024) that split K through the workspace, and M = 1.
FP8_SHAPES_DECODE = [(1, 4096, 4096), (16, 4096, 4096), (16, 11008, 4096), (16, 256, 4096),
                     (16, 4096, 256), (5, 4096, 256), (32, 4096, 4096), (32, 1024, 11008),
                     (33, 256, 4096), (64, 4096, 4096), (64, 4096, 11008), (63, 11008, 256),
                     (64, 1024, 4096)]
# variant 2 (TMA): M, N % 128 == 0, K % 128 == 0 and at least one 128x128 tile per SM (170 on
# the RTX 5090): one wave with a single K-tile, a two-K-tile grid with a split-K tail, a deep
# K, and both pipeline configurations (operands within L2 and beyond it).
FP8_SHAPES_V2 = [(128, 21760, 128), (1792, 1792, 256), (256, 11008, 4096), (2048, 2048, 2048),
                 (2560, 2560, 4096), (8192, 8192, 1024)]
# Shapes variant 2 refuses; the default steps down to variant 1.
FP8_SHAPES_NOT_V2 = [(1024, 1024, 1024), (16, 4096, 4096), (2048, 2112, 2048), (2048, 2048, 2112)]
# Both sides accumulate exact e4m3 products in fp32 and round once to bf16; the tolerance
# admits one bf16 ulp on each side plus summation-order noise.
TOL = dict(atol=3e-2, rtol=3e-2)


def _variants(name):
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants(name)))
    except Exception:
        return [0]


def _inputs(M, N, K, seed=0):
    torch.manual_seed(seed)
    a = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    b_t = torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.tensor(0.75, device="cuda")
    scale_b = torch.tensor(1.5, device="cuda")
    return a, b_t, scale_a, scale_b


def _reference(a, b_t, scale_a, scale_b):
    # cuBLASLt through torch: the second operand must be column-major, which b_t.t() is.
    return torch._scaled_mm(a, b_t.t(), scale_a=scale_a, scale_b=scale_b,
                            out_dtype=torch.bfloat16)


def _case(sk, shape, variant):
    M, N, K = shape
    a, b_t, sa, sb = _inputs(M, N, K)
    got = sk.fp8gemm(a, b_t, sa, sb, variant)
    assert got.shape == (M, N) and got.dtype == torch.bfloat16
    ref = _reference(a, b_t, sa, sb)
    torch.testing.assert_close(got.float(), ref.float(), **TOL)
    # ... and the plain fp32 product of the same values, which the scale must multiply.
    exact = (a.float() @ b_t.float().t()) * (sa * sb)
    torch.testing.assert_close(got.float(), exact, **TOL)


@pytest.mark.parametrize("shape", FP8_SHAPES_V0, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_fp8gemm_variant0(sk, shape):
    _case(sk, shape, 0)


@pytest.mark.parametrize("shape", FP8_SHAPES_V1, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_fp8gemm_variant1(sk, shape):
    _case(sk, shape, 1)
    _case(sk, shape, 1)  # the split-K tail reuses a workspace between calls


@pytest.mark.parametrize("shape", FP8_SHAPES_DECODE, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_fp8gemm_decode_shapes(sk, shape):
    for _ in range(3):
        _case(sk, shape, 1)


@pytest.mark.parametrize("shape", FP8_SHAPES_V2, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_fp8gemm_variant2_full_waves(sk, shape):
    if 2 not in _variants("fp8gemm"):
        pytest.skip("variant 2 not built")
    _case(sk, shape, 2)
    _case(sk, shape, 2)


@pytest.mark.parametrize("shape", FP8_SHAPES_NOT_V2, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_fp8gemm_variant2_steps_down(sk, shape):
    if 2 not in _variants("fp8gemm"):
        pytest.skip("variant 2 not built")
    M, N, K = shape
    a, b_t, sa, sb = _inputs(M, N, K)
    with pytest.raises(ValueError):  # an explicit variant is never substituted
        sk.fp8gemm(a, b_t, sa, sb, 2)
    got = sk.fp8gemm(a, b_t, sa, sb)  # the default steps down to variant 1
    torch.testing.assert_close(got.float(), _reference(a, b_t, sa, sb).float(), **TOL)


def test_fp8gemm_matches_scaled_mm_bitwise_on_exact_sums(sk):
    # With K = 64 every fp32 partial sum of e4m3 products is exact whatever the order, so both
    # sides must produce the same bf16 bits, scale included.
    a, b_t, sa, sb = _inputs(256, 256, 64, seed=3)
    got = sk.fp8gemm(a, b_t, sa, sb)
    assert torch.equal(got, _reference(a, b_t, sa, sb))


def test_fp8gemm_rejects_bad_inputs(sk):
    a, b_t, sa, sb = _inputs(64, 64, 64)
    with pytest.raises(RuntimeError):  # inner dimensions
        sk.fp8gemm(a, torch.empty(64, 128, device="cuda", dtype=torch.float8_e4m3fn), sa, sb)
    with pytest.raises(RuntimeError):  # dtype
        sk.fp8gemm(a.to(torch.bfloat16), b_t, sa, sb)
    with pytest.raises(RuntimeError):  # scale on the host
        sk.fp8gemm(a, b_t, torch.tensor(1.0), sb)
    with pytest.raises((ValueError, RuntimeError)):  # N % 64 != 0
        sk.fp8gemm(a, torch.empty(48, 64, device="cuda", dtype=torch.float8_e4m3fn), sa, sb)
    with pytest.raises(ValueError):  # variant 0 needs M % 16 == 0
        sk.fp8gemm(a[:5], b_t, sa, sb, 0)


def test_fp8gemm_num_variants(sk):
    assert sk.num_variants("fp8gemm") == 3
