import pytest
import torch

from conftest import TOL, dtype_id

DTYPES = [torch.float32, torch.bfloat16]
ROWS = [1, 7, 4096]
COLS = [256, 1024, 4096, 11008]


def _variants():
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants("rmsnorm")))
    except Exception:  # extension not built / no GPU: tests get skipped anyway
        return [0]


def _vector_width(dtype):
    return 4 if dtype == torch.float32 else 8


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("cols", COLS)
def test_rmsnorm_matches_reference(sk, dtype, rows, cols, variant):
    if cols % _vector_width(dtype) != 0 and variant >= 2:
        pytest.skip("vectorized variant needs aligned cols")
    torch.manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    w = (1.0 + 0.1 * torch.randn(cols, device="cuda")).to(dtype)
    got = sk.rmsnorm(x, w, 1e-6, variant)
    ref = sk.reference.rmsnorm(x, w, 1e-6)
    assert got.shape == x.shape and got.dtype == dtype
    torch.testing.assert_close(got, ref, **TOL[dtype])


@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
def test_rmsnorm_flattens_leading_dims(sk, dtype):
    x = torch.randn(2, 3, 5, 1024, device="cuda", dtype=dtype)
    w = torch.ones(1024, device="cuda", dtype=dtype)
    got = sk.rmsnorm(x, w)
    torch.testing.assert_close(got, sk.reference.rmsnorm(x, w), **TOL[dtype])


@pytest.mark.parametrize("rows", ROWS)
@pytest.mark.parametrize("cols", [256, 4096, 11008])
def test_add_rmsnorm_inplace(sk, rows, cols):
    torch.manual_seed(1)
    dtype = torch.bfloat16
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    resid = torch.randn(rows, cols, device="cuda", dtype=dtype)
    w = (1.0 + 0.1 * torch.randn(cols, device="cuda")).to(dtype)
    ref_resid, ref_out = sk.reference.add_rmsnorm(x, resid.clone(), w)
    out = sk.add_rmsnorm_(x, resid, w)
    torch.testing.assert_close(resid, ref_resid, **TOL[dtype])
    torch.testing.assert_close(out, ref_out, **TOL[dtype])


def test_rmsnorm_rejects_bad_weight(sk):
    x = torch.randn(4, 256, device="cuda")
    w = torch.ones(128, device="cuda")
    with pytest.raises(RuntimeError):
        sk.rmsnorm(x, w)


def test_rmsnorm_vectorized_variant_rejects_unaligned_storage(sk):
    dtype = torch.bfloat16
    # 4 bf16 = 8 bytes past a 16-byte boundary: contiguous, but no 128-bit loads possible
    x = torch.randn(4 * 256 + 4, device="cuda", dtype=dtype)[4:].view(4, 256)
    assert x.is_contiguous() and x.data_ptr() % 16 != 0
    w = torch.ones(256, device="cuda", dtype=dtype)
    with pytest.raises(ValueError):
        sk.rmsnorm(x, w, 1e-6, 2)
    with pytest.raises(ValueError):
        sk.add_rmsnorm_(x, x.clone(), w)
    # the default variant (block per row, scalar loads) takes it
    torch.testing.assert_close(sk.rmsnorm(x, w), sk.reference.rmsnorm(x, w), **TOL[dtype])


# The norms that write fp8 (rmsnorm_fp8, add_rmsnorm_fp8_) against the two ops they replace,
# the bf16 norm and fp8_quantize's row mode: the same e4m3 bytes, scales and residual, bit for
# bit. The widths cover every group size of the single-pass kernel (32 to 1024 threads a row,
# one to four vectors a thread), a ragged row (2056 = 257 vectors over 128 threads) and the
# widest row it takes; the weight scales push the row scales to both ends of their range.
FP8_COLS = [64, 256, 1024, 2056, 4096, 14336, 32768]
FP8_WSCALE = [1.0, 2.0**-100, 2.0**60]


def _norm_inputs(rows, cols, wscale, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(rows, cols, device="cuda", generator=g).to(torch.bfloat16)
    d = torch.randn(rows, cols, device="cuda", generator=g).to(torch.bfloat16)
    x[0] = 0  # an all-zero row (its scale is the smallest one, 2^-126)
    d[0] = 0
    x[-1, cols // 3] = 3000.0  # an outlier: this row's scale is far from the others'
    w = (1.0 + 0.3 * torch.randn(cols, device="cuda", generator=g)) * wscale
    return x, d, w.to(torch.bfloat16)


def _same_fp8(got, want):
    (q, s), (q0, s0) = got, want
    assert q.dtype == torch.float8_e4m3fn and q.shape == q0.shape
    assert s.dtype == torch.float32 and s.shape == s0.shape
    assert torch.equal(q.view(torch.uint8), q0.view(torch.uint8))
    assert torch.equal(s, s0)


@pytest.mark.parametrize("wscale", FP8_WSCALE)
@pytest.mark.parametrize("rows", [1, 3, 64, 257])
@pytest.mark.parametrize("cols", FP8_COLS)
def test_rmsnorm_fp8_matches_rmsnorm_then_quantize(sk, rows, cols, wscale):
    x, _, w = _norm_inputs(rows, cols, wscale, 3)
    _same_fp8(sk.rmsnorm_fp8(x, w, 1e-6), sk.fp8_quantize(sk.rmsnorm(x, w, 1e-6), "row"))


@pytest.mark.parametrize("wscale", FP8_WSCALE)
@pytest.mark.parametrize("rows", [1, 3, 64, 257])
@pytest.mark.parametrize("cols", FP8_COLS)
def test_add_rmsnorm_fp8_matches_add_rmsnorm_then_quantize(sk, rows, cols, wscale):
    x, d, w = _norm_inputs(rows, cols, wscale, 4)
    r_fused, r_two = x.clone(), x.clone()
    got = sk.add_rmsnorm_fp8_(d, r_fused, w, 1e-6)
    want = sk.fp8_quantize(sk.add_rmsnorm_(d, r_two, w, 1e-6), "row")
    assert torch.equal(r_fused, r_two)  # the residual is updated in place, the same bits
    _same_fp8(got, want)


def test_norm_fp8_flattens_leading_dims(sk):
    x, d, w = _norm_inputs(6, 1024, 1.0, 5)
    q, s = sk.rmsnorm_fp8(x.view(2, 3, 1024), w)
    assert q.shape == (2, 3, 1024) and s.shape == (2, 3, 1)
    _same_fp8((q.view(6, 1024), s.view(6, 1)), sk.rmsnorm_fp8(x, w))
    q, s = sk.add_rmsnorm_fp8_(d.view(2, 3, 1024), x.clone().view(2, 3, 1024), w)
    _same_fp8((q.view(6, 1024), s.view(6, 1)), sk.add_rmsnorm_fp8_(d, x.clone(), w))


def test_norm_fp8_rows_do_not_depend_on_the_batch(sk):
    """A row's bytes and scale are the same alone and in any batch (one group a row, its size
    picked from cols), so the batch-invariant engine can take the fused norm as it is."""
    x, d, w = _norm_inputs(300, 4096, 1.0, 6)
    full = sk.add_rmsnorm_fp8_(d, x.clone(), w)
    full_plain = sk.rmsnorm_fp8(x, w)
    for m in (1, 2, 7, 64, 299):
        q, s = sk.add_rmsnorm_fp8_(d[:m], x[:m].clone(), w)
        _same_fp8((q, s), (full[0][:m], full[1][:m]))
        q, s = sk.rmsnorm_fp8(x[:m], w)
        _same_fp8((q, s), (full_plain[0][:m], full_plain[1][:m]))


def test_norm_fp8_rejects_what_it_cannot_hold(sk):
    w = torch.ones(260, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(4, 260, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):  # cols not a multiple of 8
        sk.rmsnorm_fp8(x, w)
    wide = torch.randn(2, 32776, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):  # wider than the registers of 1024 threads hold
        sk.add_rmsnorm_fp8_(wide, wide.clone(), torch.ones(32776, device="cuda",
                                                           dtype=torch.bfloat16))
    flat = torch.randn(4 * 256 + 4, device="cuda", dtype=torch.bfloat16)
    x = flat[4:].view(4, 256)  # contiguous, 8 bytes past a 16-byte boundary
    with pytest.raises(ValueError):
        sk.rmsnorm_fp8(x, torch.ones(256, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(RuntimeError):  # bf16 only
        sk.rmsnorm_fp8(torch.randn(4, 256, device="cuda"), torch.ones(256, device="cuda"))
