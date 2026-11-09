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


# swiglu_fp8 against the two ops it replaces, swiglu and fp8_quantize's row mode: the same e4m3
# bytes and scales, bit for bit. The widths cover every block size (128 to 1024 threads a row,
# one, two or four vectors a thread, in a cluster or not), ragged rows (2056 = 257 vectors over
# 128 threads, 11008 = 1376 over 512) and the widest row it takes; the scales on up push the
# row scales to both ends of their range.
FP8_COLS = [8, 1024, 2048, 2056, 5632, 11008, 14336, 32768]
FP8_SCALE = [1.0, 2.0**-100, 2.0**100]


def _gate_up(rows, cols, scale, seed):
    """gu = [gate | up] as the gate/up projection writes it, gate over the whole silu curve."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    gate = 3 * torch.randn(rows, cols, device="cuda", generator=g)
    up = scale * torch.randn(rows, cols, device="cuda", generator=g)
    gu = torch.cat([gate, up], dim=1).to(torch.bfloat16)
    gu[0] = 0  # an all-zero row (its scale is the smallest one, 2^-126)
    gu[-1, cols + cols // 3] = 3000.0 * scale  # an outlier: this row's scale is far from the rest
    return gu


def _same_fp8(got, want):
    (q, s), (q0, s0) = got, want
    assert q.dtype == torch.float8_e4m3fn and q.shape == q0.shape and q.is_contiguous()
    assert s.dtype == torch.float32 and s.shape == s0.shape
    assert torch.equal(q.view(torch.uint8), q0.view(torch.uint8))
    assert torch.equal(s, s0)


@pytest.mark.parametrize("scale", FP8_SCALE)
@pytest.mark.parametrize("rows", [1, 3, 64, 257])
@pytest.mark.parametrize("cols", FP8_COLS)
def test_swiglu_fp8_matches_swiglu_then_quantize(sk, rows, cols, scale):
    gu = _gate_up(rows, cols, scale, 3)
    gate, up = gu[:, :cols], gu[:, cols:]
    _same_fp8(sk.swiglu_fp8(gate, up), sk.fp8_quantize(sk.swiglu(gate, up), "row"))


def test_swiglu_fp8_takes_contiguous_and_unequal_strides(sk):
    gu = _gate_up(37, 4096, 1.0, 4)
    gate, up = gu[:, :4096].contiguous(), gu[:, 4096:].contiguous()
    want = sk.fp8_quantize(sk.swiglu(gate, up), "row")
    _same_fp8(sk.swiglu_fp8(gate, up), want)
    _same_fp8(sk.swiglu_fp8(gu[:, :4096], gu[:, 4096:]), want)
    wide = torch.zeros(37, 3 * 4096 + 8, device="cuda", dtype=torch.bfloat16)
    wide[:, 4096:8192] = up  # rows 12296 elements apart where gate's are 4096
    _same_fp8(sk.swiglu_fp8(gate, wide[:, 4096:8192]), want)
    q, s = sk.swiglu_fp8(gate[:0], up[:0])
    assert q.shape == (0, 4096) and s.shape == (0, 1)


def test_swiglu_fp8_rows_do_not_depend_on_the_batch(sk):
    """A row's bytes and scale are the same alone and in any batch, whichever geometry the row
    count picks (a cluster of 8 to 41 rows, of 4 to 64, one block a row past that, the block
    sized for one vector a thread to 128 rows), so the batch-invariant engine can take it as
    it is."""
    gu = _gate_up(300, 14336, 1.0, 5)
    q, s = sk.swiglu_fp8(gu[:, :14336], gu[:, 14336:])
    for m in (1, 2, 7, 41, 42, 64, 65, 128, 129, 299):
        _same_fp8(sk.swiglu_fp8(gu[:m, :14336], gu[:m, 14336:]), (q[:m], s[:m]))


def test_swiglu_fp8_rejects_what_it_cannot_hold(sk):
    x = torch.randn(4, 260, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):  # cols not a multiple of 8
        sk.swiglu_fp8(x, x)
    wide = torch.randn(2, 32776, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):  # wider than the registers of 1024 threads hold
        sk.swiglu_fp8(wide, wide)
    gu = torch.randn(4, 2 * 256 + 4, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):  # rows 8 bytes past a 16-byte boundary
        sk.swiglu_fp8(gu[:, 4:260], gu[:, 260:516])
    with pytest.raises(RuntimeError):  # bf16 only
        sk.swiglu_fp8(torch.randn(4, 256, device="cuda"), torch.randn(4, 256, device="cuda"))
    with pytest.raises(RuntimeError):  # 2-D only
        sk.swiglu_fp8(x[:, :256].reshape(2, 2, 256), x[:, :256].reshape(2, 2, 256))


# Every geometry, forced for a whole process (the kernel reads the variables once) whatever the
# dispatch would pick at these shapes: each cluster size, blocks sized for one, two or four
# vectors a thread, and 2 and 4 rows to a cluster (swiglu_fp8_rows_kernel, which only the
# variable picks; its last cluster is short at 33 and 127 rows). Each must still be the two ops'
# bits.
_FORCED = """
import sys
import torch
sys.path.insert(0, sys.argv[1])
import spark_kernels as sk
from test_swiglu import FP8_COLS, FP8_SCALE, _gate_up, _same_fp8
for cols in FP8_COLS:
    for rows in (1, 3, 33, 64, 127):
        for scale in FP8_SCALE:
            gu = _gate_up(rows, cols, scale, rows + cols)
            gate, up = gu[:, :cols], gu[:, cols:]
            _same_fp8(sk.swiglu_fp8(gate, up), sk.fp8_quantize(sk.swiglu(gate, up), "row"))
"""


@pytest.mark.parametrize("cluster,rows_per,vpt", [
    (1, 1, 1), (1, 1, 2), (1, 1, 4), (2, 1, 4), (4, 1, 1), (4, 1, 4), (8, 1, 1), (8, 1, 4),
    (2, 2, 4), (4, 4, 4), (8, 2, 1), (8, 2, 4), (8, 4, 1)])
def test_swiglu_fp8_forced_geometries_match(sk, cluster, rows_per, vpt):
    import os
    import subprocess
    import sys
    from pathlib import Path

    env = {k: v for k, v in os.environ.items() if not k.startswith("SPARK_SWIGLU_FP8_")}
    env.update(SPARK_SWIGLU_FP8_CLUSTER=str(cluster), SPARK_SWIGLU_FP8_ROWS=str(rows_per),
               SPARK_SWIGLU_FP8_VPT=str(vpt))
    subprocess.run([sys.executable, "-c", _FORCED, str(Path(__file__).resolve().parent)],
                   env=env, check=True)
