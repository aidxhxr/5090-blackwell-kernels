"""Parity of sk.fp4gemm and sk.fp4_quantize with spark_kernels.reference: NVFP4 (e2m1 with an
e4m3 scale per 16 and a per-tensor fp32 scale) and MXFP4 (a ue8m0 scale per 32), a [M, K/2]
and b_t [N, K/2] packed two values per byte, block scales in the blocked layout, bf16 out.

The GEMM tests quantize Gaussian inputs with the kernel quantizer and compare every rung against
the fp32 product of the dequantized values. Those products are exact in fp32 and both sides
round once to bf16, so only the summation order differs. Where torch has an NVFP4 scaled_mm
(cuBLASLt), the output must match it bit for bit, as the C++ bench measures."""

import pytest
import torch
import torch.nn.functional as F

FORMATS = ["nvfp4", "mxfp4"]
# variant 0: M % 16 == 0
FP4_SHAPES_V0 = [(16, 64, 256), (256, 256, 256), (272, 1024, 4096)]
# variant 1: every tile (128x128, 64x128, 64x64), the split-K tail (few tiles, long K), ragged
# M (zero-filled rows, scale rows from the 128-row padding), K = 14336 (Llama-3-8B down_proj).
FP4_SHAPES_V1 = [(256, 256, 256), (272, 128, 4096), (208, 1088, 256), (1024, 1024, 1024),
                 (100, 1024, 1024), (2048, 2048, 2048), (200, 256, 768), (256, 512, 14336)]
# The decode path (M <= 64): every row tile (16, 32, 64), strips that split K, M = 1.
FP4_SHAPES_DECODE = [(1, 4096, 4096), (16, 4096, 4096), (16, 256, 4096), (5, 4096, 256),
                     (32, 4096, 4096), (32, 1024, 4096), (33, 256, 4096), (64, 4096, 4096),
                     (63, 1024, 256), (64, 1024, 14336)]
# variant 2 (TMA): M, N % 128 == 0 and at least one 128x128 tile per SM (170 on the RTX 5090):
# one wave with a single k-tile, the two-block configuration, a split-K tail, a deep K.
FP4_SHAPES_V2 = [(128, 21760, 256), (1792, 1792, 256), (256, 11008, 4096), (2048, 2048, 2048),
                 (2560, 2560, 4096), (4096, 4096, 1024)]
FP4_SHAPES_NOT_V2 = [(1024, 1024, 1024), (16, 4096, 4096), (2048, 2112, 2048)]
# Outputs are sums of K exact products of size O(1) (O(sqrt(K)) overall); bf16 rounding and the
# fp32 summation order are the only differences.
TOL = dict(atol=1e-2, rtol=3e-2)


def _ids(s):
    return f"M{s[0]}_N{s[1]}_K{s[2]}"


def _variants(sk):
    return list(range(sk.num_variants("fp4gemm")))


def _available(sk):
    x = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    q, sf, s = sk.fp4_quantize(x)
    try:
        sk.fp4gemm(q, q, sf, sf, s, s)
    except ValueError as e:  # a plain sm_120 build has no fp4 mma
        pytest.skip(str(e))


def _inputs(sk, M, N, K, fmt, seed=0):
    torch.manual_seed(seed)
    a, sfa, sa = sk.fp4_quantize(torch.randn(M, K, device="cuda", dtype=torch.bfloat16), fmt)
    b_t, sfb, sb = sk.fp4_quantize(torch.randn(N, K, device="cuda", dtype=torch.bfloat16), fmt)
    return a, sfa, sa, b_t, sfb, sb


def _case(sk, shape, variant, fmt):
    M, N, K = shape
    a, sfa, sa, b_t, sfb, sb = _inputs(sk, M, N, K, fmt)
    got = sk.fp4gemm(a, b_t, sfa, sfb, sa, sb, fmt=fmt, variant=variant)
    assert got.shape == (M, N) and got.dtype == torch.bfloat16
    ref = sk.reference.fp4gemm(a, sfa, b_t, sfb, fmt, sa, sb)
    torch.testing.assert_close(got.float(), ref.float(), **TOL)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("shape", FP4_SHAPES_V0, ids=_ids)
def test_fp4gemm_variant0(sk, shape, fmt):
    _available(sk)
    _case(sk, shape, 0, fmt)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("shape", FP4_SHAPES_V1, ids=_ids)
def test_fp4gemm_variant1(sk, shape, fmt):
    _available(sk)
    _case(sk, shape, 1, fmt)
    _case(sk, shape, 1, fmt)  # the split-K workspace must be left clean for the next call


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("shape", FP4_SHAPES_DECODE, ids=_ids)
def test_fp4gemm_decode_shapes(sk, shape, fmt):
    _available(sk)
    for _ in range(2):
        _case(sk, shape, 1, fmt)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("shape", FP4_SHAPES_V2, ids=_ids)
def test_fp4gemm_variant2(sk, shape, fmt):
    _available(sk)
    _case(sk, shape, 2, fmt)
    _case(sk, shape, 2, fmt)


@pytest.mark.parametrize("shape", FP4_SHAPES_NOT_V2, ids=_ids)
def test_fp4gemm_variant2_steps_down(sk, shape):
    _available(sk)
    M, N, K = shape
    a, sfa, sa, b_t, sfb, sb = _inputs(sk, M, N, K, "nvfp4")
    with pytest.raises(ValueError):  # an explicit variant is never substituted
        sk.fp4gemm(a, b_t, sfa, sfb, sa, sb, variant=2)
    got = sk.fp4gemm(a, b_t, sfa, sfb, sa, sb)
    ref = sk.reference.fp4gemm(a, sfa, b_t, sfb, "nvfp4", sa, sb)
    torch.testing.assert_close(got.float(), ref.float(), **TOL)


def test_fp4gemm_variants_agree_bitwise_on_exact_sums(sk):
    # Codes in {0, +-0.5, +-1} and one block scale per operand make every partial sum a small
    # multiple of 2^-2 times the scales, exact in fp32 in any order: every variant, tile and
    # split-K schedule must produce the same bits, and the scale words must reach the right
    # rows, columns and k-blocks for that to hold with scales that differ per block.
    _available(sk)
    for fmt in FORMATS:
        for M, N, K in [(256, 256, 256), (272, 1088, 4096), (33, 4096, 768), (2048, 2048, 2048)]:
            torch.manual_seed(M + N + K)
            a = sk.reference.pack_e2m1(torch.randint(-2, 3, (M, K), device="cuda").abs()
                                       .to(torch.uint8) | (torch.randint(0, 2, (M, K),
                                                                         device="cuda")
                                                           .to(torch.uint8) << 3))
            b_t = sk.reference.pack_e2m1(torch.randint(0, 3, (N, K), device="cuda")
                                         .to(torch.uint8))
            block = 16 if fmt == "nvfp4" else 32
            if fmt == "nvfp4":  # e4m3 powers of two 2^-3 .. 2^3
                pick = torch.tensor([0x20, 0x28, 0x30, 0x38, 0x40, 0x48, 0x50], dtype=torch.uint8,
                                    device="cuda")
            else:
                pick = torch.arange(124, 131, dtype=torch.uint8, device="cuda")
            sfa = sk.reference.to_blocked(pick[torch.randint(0, 7, (M, K // block),
                                                             device="cuda")])
            sfb = sk.reference.to_blocked(pick[torch.randint(0, 7, (N, K // block),
                                                             device="cuda")])
            outs = []
            for v in _variants(sk):
                if v == 0 and M % 16 != 0:
                    continue
                try:
                    outs.append(sk.fp4gemm(a, b_t, sfa, sfb, fmt=fmt, variant=v))
                except ValueError:  # variant 2 refuses the shape
                    continue
            ref = sk.reference.fp4gemm(a, sfa, b_t, sfb, fmt)
            for got in outs:
                assert torch.equal(got, ref), (fmt, M, N, K)


def test_fp4gemm_per_tensor_scales(sk):
    _available(sk)
    a, sfa, _, b_t, sfb, _ = _inputs(sk, 256, 512, 1024, "nvfp4")
    got = sk.fp4gemm(a, b_t, sfa, sfb, torch.tensor(0.5, device="cuda"),
                     torch.tensor(1.5, device="cuda"))
    ref = sk.reference.fp4gemm(a, sfa, b_t, sfb).float() * 0.75
    torch.testing.assert_close(got.float(), ref, **TOL)


def test_fp4gemm_accepts_torch_fp4_and_fp8_dtypes(sk):
    _available(sk)
    a, sfa, sa, b_t, sfb, sb = _inputs(sk, 256, 256, 512, "nvfp4")
    want = sk.fp4gemm(a, b_t, sfa, sfb, sa, sb)
    got = sk.fp4gemm(a.view(torch.float4_e2m1fn_x2), b_t.view(torch.float4_e2m1fn_x2),
                     sfa.view(torch.float8_e4m3fn), sfb.view(torch.float8_e4m3fn), sa, sb)
    assert torch.equal(got, want)


def _nvfp4_scaled_mm(a, b_t, sfa, sfb, sa, sb):
    """torch's NVFP4 scaled_mm (cuBLASLt) with both scale levels, the same with the block
    scales only (then sa and sb are left out on our side too), or None if this torch has no
    NVFP4 scaled_mm for the GPU. Returns (output, per-tensor scales applied)."""
    fa, fb = a.view(torch.float4_e2m1fn_x2), b_t.view(torch.float4_e2m1fn_x2)
    ea, eb = sfa.view(torch.float8_e4m3fn), sfb.view(torch.float8_e4m3fn)
    errors = (AttributeError, RuntimeError, TypeError, ValueError, NotImplementedError)
    try:
        return F.scaled_mm(fa, fb.t(), [ea, sa.reshape(1)],
                           [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
                           [eb, sb.reshape(1)],
                           [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
                           swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
                           swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
                           output_dtype=torch.bfloat16), True
    except errors:
        pass
    try:  # the private entry point: block scales only
        return torch._scaled_mm(fa, fb.t(), scale_a=ea, scale_b=eb, out_dtype=torch.bfloat16), False
    except errors:
        return None, False


@pytest.mark.parametrize("shape", [(256, 256, 4096), (512, 1024, 14336), (2048, 2048, 2048)],
                         ids=_ids)
def test_fp4gemm_matches_torch_scaled_mm_bitwise(sk, shape):
    # cuBLASLt's NVFP4 kernel and ours sum the same exact products in fp32; on these inputs the
    # partial sums stay exact in any order, and the C++ bench finds every output identical.
    _available(sk)
    M, N, K = shape
    a, sfa, sa, b_t, sfb, sb = _inputs(sk, M, N, K, "nvfp4")
    ref, two_level = _nvfp4_scaled_mm(a, b_t, sfa, sfb, sa, sb)
    if ref is None:
        pytest.skip("torch has no NVFP4 scaled_mm here")
    for v in _variants(sk):
        if not sk_supports(sk, M, N, K, v):
            continue
        got = (sk.fp4gemm(a, b_t, sfa, sfb, sa, sb, variant=v) if two_level
               else sk.fp4gemm(a, b_t, sfa, sfb, variant=v))
        assert torch.equal(got, ref), (shape, v)


def sk_supports(sk, M, N, K, v):
    if v == 0:
        return M % 16 == 0
    if v == 2:
        return M % 128 == 0 and N % 128 == 0 and (M // 128) * (N // 128) >= \
            torch.cuda.get_device_properties(0).multi_processor_count
    return True


def test_fp4gemm_rejects_bad_inputs(sk):
    _available(sk)
    a, sfa, sa, b_t, sfb, sb = _inputs(sk, 64, 128, 256, "nvfp4")
    with pytest.raises(RuntimeError):  # inner dimensions
        sk.fp4gemm(a, b_t[:, :64].contiguous(), sfa, sfb, sa, sb)
    with pytest.raises(RuntimeError):  # dtype
        sk.fp4gemm(a.to(torch.int16), b_t, sfa, sfb, sa, sb)
    with pytest.raises(RuntimeError):  # scale tensor of the wrong size
        sk.fp4gemm(a, b_t, sfa[:256], sfb, sa, sb)
    with pytest.raises(RuntimeError):  # only one per-tensor scale
        sk.fp4gemm(a, b_t, sfa, sfb, sa, None)
    with pytest.raises(RuntimeError):  # unknown format
        sk.fp4gemm(a, b_t, sfa, sfb, sa, sb, fmt="fp4")
    with pytest.raises((RuntimeError, ValueError)):  # K % 256 != 0
        q, sf, s = sk.fp4_quantize(torch.randn(64, 128, device="cuda", dtype=torch.bfloat16))
        sk.fp4gemm(q, q, sf, sf, s, s)
    with pytest.raises(ValueError):  # variant 0 needs M % 16 == 0
        sk.fp4gemm(a[:5].contiguous(), b_t, sk.fp4_quantize(
            torch.randn(5, 256, device="cuda", dtype=torch.bfloat16))[1], sfb, sa, sb, variant=0)


def test_fp4gemm_num_variants(sk):
    assert sk.num_variants("fp4gemm") == 3


# ---- the quantizer ------------------------------------------------------------------------

@pytest.mark.parametrize("rows,K", [(1, 256), (100, 1024), (128, 4096), (300, 14336)])
def test_fp4_quantize_matches_reference_bytes(sk, rows, K):
    torch.manual_seed(rows + K)
    x = (torch.randn(rows, K, device="cuda") * 3).to(torch.bfloat16)
    x[0, :16] = 0  # an all-zero block
    x[min(3, rows - 1), 40] = 500.0  # an outlier
    q, sf, s = sk.fp4_quantize(x, "nvfp4")
    rq, rsf, rs = sk.reference.quantize_nvfp4(x)
    assert torch.equal(s, rs) and torch.equal(q, rq) and torch.equal(sf, rsf)
    q2, sf2, s2 = sk.fp4_quantize(x, "nvfp4", torch.tensor(0.01, device="cuda"))
    rq2, rsf2, _ = sk.reference.quantize_nvfp4(x, torch.tensor(0.01, device="cuda"))
    assert torch.equal(q2, rq2) and torch.equal(sf2, rsf2)  # a given scale, some saturation
    if K % 128 == 0:
        q, sf, s = sk.fp4_quantize(x, "mxfp4")
        rq, rsf = sk.reference.quantize_mxfp4(x)
        assert s is None and torch.equal(q, rq) and torch.equal(sf, rsf)


def test_e2m1_rounding_reference():
    ref = __import__("spark_kernels").reference
    v = torch.tensor([0.0, 0.25, 0.26, 0.74, 0.75, 1.25, 1.26, 1.75, 2.5, 2.51, 3.49, 3.5, 5.0,
                      5.01, 100.0, -0.75, -6.0])
    want = [0.0, 0.0, 0.5, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 6.0, 6.0, -1.0, -6.0]
    assert ref.e2m1_values(ref.e2m1_codes(v)).tolist() == want
    codes = torch.arange(16, dtype=torch.uint8)
    assert torch.equal(ref.unpack_e2m1(ref.pack_e2m1(codes)), codes)
    sf = torch.randint(0, 255, (200, 12), dtype=torch.uint8)
    assert torch.equal(ref.from_blocked(ref.to_blocked(sf), 200, 12), sf)
    # the blocked layout: row r, column c of a 128 x 4 tile at (r % 32) * 16 + (r // 32) * 4 + c
    flat = ref.to_blocked(sf)
    assert flat[(37 % 32) * 16 + (37 // 32) * 4 + 2] == sf[37, 2]
    assert flat[512 + 5 * 16 + 1] == sf[5, 5]  # the second tile along the columns


def test_fp4_accuracy_against_bf16():
    # NVFP4's two-level scale keeps the GEMM error near what e2m1's one mantissa bit allows on
    # Gaussian data, and its per-16 e4m3 scales follow outlier channels that a coarser scale
    # cannot; MXFP4's power-of-two per-32 scale costs more (scripts/fp4_accuracy.py has the
    # table, docs/design/fp4gemm.md the numbers).
    ref = __import__("spark_kernels").reference
    torch.manual_seed(2)
    b = torch.randn(512, 2048, device="cuda")

    def rel_err(a, fmt):
        exact = a @ b.t()
        if fmt == "nvfp4":
            qa, sfa, sa = ref.quantize_nvfp4(a.to(torch.bfloat16))
            qb, sfb, sb = ref.quantize_nvfp4(b.to(torch.bfloat16))
        else:
            (qa, sfa), (qb, sfb) = ref.quantize_mxfp4(a.to(torch.bfloat16)), ref.quantize_mxfp4(
                b.to(torch.bfloat16))
            sa = sb = None
        got = ref.dequantize_fp4(qa, sfa, fmt, sa) @ ref.dequantize_fp4(qb, sfb, fmt, sb).t()
        return ((got - exact).norm() / exact.norm()).item()

    a = torch.randn(512, 2048, device="cuda")
    nv, mx = rel_err(a, "nvfp4"), rel_err(a, "mxfp4")
    assert 0.12 < nv < 0.15 and nv * 1.1 < mx < 0.19, (nv, mx)  # measured 0.134, 0.162
    a[:, ::256] *= 100.0  # 8 outlier channels: their blocks get their own scales
    nv, mx = rel_err(a, "nvfp4"), rel_err(a, "mxfp4")
    assert nv < 0.12 and mx > 1.5 * nv, (nv, mx)  # measured 0.105, 0.177
