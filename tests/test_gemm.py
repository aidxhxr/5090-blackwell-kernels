import pytest
import torch

from conftest import TOL

SGEMM_SHAPES = [(64, 64, 64), (129, 257, 65), (1024, 1024, 1024), (1, 4096, 4096)]
HGEMM_SHAPES_16 = [(256, 256, 256), (272, 144, 48), (16, 16, 16)]
HGEMM_SHAPES_128 = [(256, 384, 256), (128, 128, 32), (512, 512, 4096)]
# variant 3: any M % 16 == 0 (rows past M are zero-filled), N % 64 == 0, K % 64 == 0. The
# shapes cover every tile it picks (128x128, 64x128, 64x64), the decode case (M <= 64) and
# the split-K tail (a K long enough to split, few tiles).
HGEMM_SHAPES_V3 = [(16, 64, 64), (16, 4096, 1024), (64, 256, 512), (272, 128, 4096),
                   (208, 1088, 192), (1024, 1024, 1024)]
# The decode path inside variant 3 (M <= 64, src/kernels/hgemm_decode.cu): every row tile
# (16, 32, 64) against N and K from {256, 4096, 11008}, and the shapes with few column strips
# (N = 256, 1024) that split K and reduce through the self-resetting workspace.
HGEMM_SHAPES_DECODE = [(16, 4096, 4096), (16, 11008, 4096), (16, 4096, 11008), (16, 256, 4096),
                       (16, 4096, 256), (32, 4096, 4096), (32, 1024, 11008), (32, 256, 256),
                       (64, 4096, 4096), (64, 4096, 11008), (64, 11008, 256), (64, 1024, 4096)]
# Rows past M are zero-filled, so variant 3 takes any M >= 1: single-token decode (M = 1),
# odd counts inside the 16, 32 and 64 row tiles, and one above 64 for the general tiles.
HGEMM_SHAPES_ANY_M = [(1, 4096, 4096), (5, 4096, 256), (17, 1024, 1024), (33, 256, 4096),
                      (63, 4096, 1024), (100, 1024, 1024)]
HGEMM_TOL = dict(atol=3e-2, rtol=3e-2)


def _variants(name):
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants(name)))
    except Exception:
        return [0]


@pytest.mark.parametrize("variant", _variants("sgemm"))
@pytest.mark.parametrize("shape", SGEMM_SHAPES, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_sgemm_matches_torch(sk, shape, variant):
    M, N, K = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    got = sk.sgemm(a, b, variant)
    ref = a @ b
    assert got.shape == (M, N)
    # fp32 accumulation order differs from cuBLAS; scale tolerance with K.
    torch.testing.assert_close(got, ref, atol=1e-3 * (K**0.5), rtol=1e-4)


def _hgemm_case(sk, shape, variant):
    M, N, K = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    got = sk.hgemm(a, b, variant)
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    assert got.shape == (M, N) and got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), ref.float(), **HGEMM_TOL)


@pytest.mark.parametrize("variant", [v for v in _variants("hgemm") if v < 2])
@pytest.mark.parametrize("shape", HGEMM_SHAPES_16, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_multiples_of_16(sk, shape, variant):
    _hgemm_case(sk, shape, variant)


@pytest.mark.parametrize("variant", _variants("hgemm"))
@pytest.mark.parametrize("shape", HGEMM_SHAPES_128, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_multiples_of_128(sk, shape, variant):
    if variant == 3 and shape[2] % 64 != 0:
        pytest.skip("variant 3 needs K % 64 == 0")
    _hgemm_case(sk, shape, variant)


@pytest.mark.parametrize("shape", HGEMM_SHAPES_V3, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_variant3_any_m(sk, shape):
    if 3 not in _variants("hgemm"):
        pytest.skip("variant 3 not built")
    _hgemm_case(sk, shape, 3)
    # Repeated calls must agree with the reference too: the split-K tail reuses a workspace.
    _hgemm_case(sk, shape, 3)


@pytest.mark.parametrize("shape", HGEMM_SHAPES_DECODE, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_decode_shapes(sk, shape):
    if 3 not in _variants("hgemm"):
        pytest.skip("variant 3 not built")
    # Three calls in a row: the split-K workspace and its arrival counters are put back to
    # zero by the kernel itself, so a second and third launch must agree with the reference.
    for _ in range(3):
        _hgemm_case(sk, shape, 3)


@pytest.mark.parametrize("shape", HGEMM_SHAPES_ANY_M, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_variant3_any_row_count(sk, shape):
    if 3 not in _variants("hgemm"):
        pytest.skip("variant 3 not built")
    _hgemm_case(sk, shape, 3)
    M, N, K = shape
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    got = sk.hgemm(a, b)  # the default variant is 3 on these shapes, no step-down needed
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    torch.testing.assert_close(got.float(), ref.float(), **HGEMM_TOL)


def test_hgemm_rejects_unaligned(sk):
    a = torch.randn(17, 32, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(32, 32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises((ValueError, RuntimeError)):
        sk.hgemm(a, b)


def test_sgemm_rejects_inner_mismatch(sk):
    a = torch.randn(8, 16, device="cuda")
    b = torch.randn(8, 16, device="cuda")
    with pytest.raises(RuntimeError):
        sk.sgemm(a, b)


def test_tolerance_dict_present():
    assert torch.float32 in TOL


def test_hgemm_default_variant_takes_any_multiple_of_16(sk):
    M, N, K = 272, 144, 48  # multiples of 16 but not of the 128x128x32 tile of variant 2
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    got = sk.hgemm(a, b)  # N % 64 != 0: steps down past variants 3 and 2 to variant 1
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    torch.testing.assert_close(got.float(), ref.float(), **HGEMM_TOL)
    with pytest.raises(ValueError):  # an explicit variant is never substituted
        sk.hgemm(a, b, 2)
