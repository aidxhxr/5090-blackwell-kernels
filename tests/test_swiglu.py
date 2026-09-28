import pytest
import torch

from conftest import TOL, dtype_id

DTYPES = [torch.float32, torch.bfloat16]
SHAPES = [(1, 256), (7, 1024), (4096, 4096), (4096, 11008), (3, 5, 1000)]


def _variants():
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants("swiglu")))
    except Exception:
        return [0]


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("shape", SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_swiglu_matches_reference(sk, dtype, shape, variant):
    torch.manual_seed(0)
    gate = torch.randn(*shape, device="cuda", dtype=dtype)
    up = torch.randn(*shape, device="cuda", dtype=dtype)
    got = sk.swiglu(gate, up, variant)
    ref = sk.reference.swiglu(gate, up)
    assert got.shape == gate.shape and got.dtype == dtype
    torch.testing.assert_close(got, ref, **TOL[dtype])


def test_swiglu_shape_mismatch(sk):
    g = torch.randn(4, 256, device="cuda")
    u = torch.randn(4, 128, device="cuda")
    with pytest.raises(RuntimeError):
        sk.swiglu(g, u)


def _unaligned(n, dtype):
    """A contiguous 1-D tensor whose storage starts 1 element past a 16-byte boundary."""
    t = torch.randn(n + 1, device="cuda", dtype=dtype)[1:]
    assert t.is_contiguous() and t.data_ptr() % 16 != 0
    return t


@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
def test_default_variant_takes_unaligned_storage(sk, dtype):
    torch.manual_seed(0)
    gate, up = _unaligned(1024, dtype), _unaligned(1024, dtype)
    got = sk.swiglu(gate, up)  # falls back to the scalar variant
    torch.testing.assert_close(got, sk.reference.swiglu(gate, up), **TOL[dtype])
    with pytest.raises(ValueError):  # an explicit variant is never substituted
        sk.swiglu(gate, up, 1)


@pytest.mark.parametrize("dtype", DTYPES, ids=dtype_id)
@pytest.mark.parametrize("shape", [(1, 14336), (7, 1000), (64, 4096), (3, 5)],
                         ids=lambda s: "x".join(map(str, s)))
def test_swiglu_reads_the_halves_of_a_fused_projection_in_place(sk, dtype, shape):
    # gu = [gate | up] from one GEMM: gu[:, :I] and gu[:, I:] have contiguous rows 2I apart;
    # the default variant takes them without a copy (odd widths run the scalar path)
    torch.manual_seed(2)
    rows, cols = shape
    gu = torch.randn(rows, 2 * cols, device="cuda", dtype=dtype)
    gate, up = gu[:, :cols], gu[:, cols:]
    got = sk.swiglu(gate, up)
    assert got.shape == (rows, cols) and got.is_contiguous()
    torch.testing.assert_close(got, sk.reference.swiglu(gate.contiguous(), up.contiguous()),
                               **TOL[dtype])
    # gate and up from different tensors with different row strides
    up2 = torch.randn(rows, 3 * cols, device="cuda", dtype=dtype)[:, cols:2 * cols]
    torch.testing.assert_close(sk.swiglu(gate, up2),
                               sk.reference.swiglu(gate.contiguous(), up2.contiguous()),
                               **TOL[dtype])
    if rows > 1:  # a one-row slice is contiguous; the others are not
        with pytest.raises(RuntimeError):
            sk.swiglu(gate, up, variant=1)  # an explicit variant still needs contiguous inputs
