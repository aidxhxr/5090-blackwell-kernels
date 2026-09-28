import contextlib

import pytest
import torch
import torch.nn.functional as F

from conftest import TOL

SGEMM_SHAPES = [(64, 64, 64), (129, 257, 65), (1024, 1024, 1024), (1, 4096, 4096)]
# The tensor-core rungs (6: TF32, 7: 3xTF32) on top of those: the 128x128 tile with a
# split-K tail (2048^3 is 256 tiles on 340 resident blocks, every tile split), the 64x128
# tile, a K that is not a multiple of the 16-wide k-slab, and the unaligned register-fill
# path with a ragged N.
SGEMM_SHAPES_TF32 = SGEMM_SHAPES + [(2048, 2048, 2048), (384, 1024, 4104), (200, 300, 100)]
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
# variant 5 (TMA + mbarrier pipeline): M % 128 == 0, N % 128 == 0, K % 64 == 0 and at least
# one 128x128 tile per SM (170 on the RTX 5090). The shapes cover exactly one wave with a
# single K-tile, a two-K-tile grid with a split-K tail, a deep K, and both pipeline configs
# (operands within L2 and beyond it, see hgemm_tma.cu).
HGEMM_SHAPES_V5 = [(128, 21760, 64), (1792, 1792, 128), (256, 11008, 4096), (2048, 2048, 2048),
                   (2560, 2560, 4096), (1024, 16384, 3072)]
# Shapes variant 5 refuses: a grid under one wave, a decode shape, and misaligned N and K.
# The default then steps down to variant 6, which takes them all (through variant 4's tiles
# or the decode kernel where the 128x128 TMA tile does not fit).
HGEMM_SHAPES_NOT_V5 = [(1024, 1024, 1024), (16, 4096, 4096), (2048, 2112, 2048),
                       (2048, 2048, 2080)]
# variant 6 (the TMA mainloop on the Stream-K schedule): variant 4's shape rules. On the RTX
# 5090 (170 resident blocks of the shipped configuration) the shapes cover static ranges with
# two or three pieces per tile (2048^3: 256 tiles, 1.5 waves), static ranges with a ragged M
# that is not a multiple of 16 (1000x4096x4096: 256 tiles, the last tile row 104 rows tall,
# zero-filled by the copy engine), the queue with two K-passes per Stream-K tile on operands
# that fit in L2 (2064x11008x2048: 1462 tiles, 8.6 waves) and with three passes on operands
# that do not (2100x11008x4096), and the shapes the TMA tile hands to variant 4's tiles or
# the decode kernel: under a 32 k-step share per block (1024^3, 272x128x4096, 100x1024x1024),
# N % 128 != 0 (128x64x16384), M <= 64 (16x4096x1024, 1x4096x4096).
HGEMM_SHAPES_V6 = [(2048, 2048, 2048), (1000, 4096, 4096), (2064, 11008, 2048),
                   (2100, 11008, 4096), (1024, 1024, 1024), (272, 128, 4096), (100, 1024, 1024),
                   (128, 64, 16384), (16, 4096, 1024), (1, 4096, 4096)]
HGEMM_TOL = dict(atol=3e-2, rtol=3e-2)
# The fused epilogue (variants 4 and 6): one shape per route the default variant takes.
# 1000x4096x4096 is the TMA tile on static ranges with a ragged M; 2064x11008x2048 the queue
# with two K-passes per Stream-K tile (the finishing piece applies the epilogue after the
# fixup); 1024^3 and 272x128x4096 variant 4's 64-row tiles; 128x64x16384 its 64-column tile
# with 8 pieces per tile; 16x4096x1024 and 1x4096x4096 the decode kernel unsplit; 33x256x4096
# and 5x4096x256 the decode kernel with K split four ways, where the last slice to arrive
# applies the epilogue from the fp32 workspace.
HGEMM_SHAPES_EPILOGUE = [(1000, 4096, 4096), (2064, 11008, 2048), (1024, 1024, 1024),
                         (272, 128, 4096), (128, 64, 16384), (16, 4096, 1024), (1, 4096, 4096),
                         (33, 256, 4096), (5, 4096, 256)]
# (bias, act, residual mode): every option alone, then the decoder-block combinations.
HGEMM_EPILOGUES = [(True, None, None), (False, "silu", None), (False, "gelu", None),
                   (False, "relu", None), (False, None, "residual"), (False, None, "accumulate"),
                   (True, "gelu", "residual"), (True, "silu", "accumulate")]
# variant 4 (Stream-K): the same shape rules as variant 3. The shapes cover a tile split into
# many K-pieces (128x64x16384: two 64x64 tiles, 8 pieces each), static ranges with two or
# three pieces per tile, a problem past four waves with a tail and a ragged M (2064x11008x1024:
# 1462 tiles of 128x128 on 340 blocks, queue mode, two K-passes per tile in the last wave), a
# decode shape, and a one-wave shape with no partials at all (1024^3).
# Decode shapes (M <= 64) run the same weight-streaming kernel as variant 3, and any M >= 1
# works: (1, 4096, 4096) is the single-token decode, (100, 1024, 1024) a ragged M on the tile.
HGEMM_SHAPES_V4 = [(16, 64, 64), (16, 4096, 1024), (64, 256, 512), (272, 128, 4096),
                   (208, 1088, 192), (1024, 1024, 1024), (1152, 1152, 4096), (128, 22016, 64),
                   (2048, 2048, 2048), (128, 64, 16384), (2064, 11008, 1024), (1, 4096, 4096),
                   (100, 1024, 1024)]


def _variants(name):
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants(name)))
    except Exception:
        return [0]


@contextlib.contextmanager
def matmul_tf32(enabled: bool):
    """torch's fp32 matmul on the tensor cores in TF32 (the contract of sgemm variant 6) or
    on the CUDA cores. torch >= 2.9 spells it fp32_precision, older releases allow_tf32."""
    m = torch.backends.cuda.matmul
    attr, on, off = (("fp32_precision", "tf32", "ieee") if hasattr(m, "fp32_precision")
                     else ("allow_tf32", True, False))
    prev = getattr(m, attr)
    setattr(m, attr, on if enabled else off)
    try:
        yield
    finally:
        setattr(m, attr, prev)


def _sgemm_tol(variant: int, K: int) -> dict:
    """fp32 rungs and 3xTF32: the summation order differs from cuBLAS, so the tolerance
    scales with sqrt(K). TF32 (variant 6): each operand carries a rounding error of up to
    2^-11, so a product is off by up to 2^-10 |a b| and a sum of K products by about
    sqrt(K) 2^-11 rms(a b) in the typical case; for N(0, 1) inputs that is 5e-4 sqrt(K), and
    the max over M x N outputs a few times that. 1.5e-2 sqrt(K) is fifteen times the fp32
    bound and well under the K 2^-10 worst case."""
    if variant == 6:
        return dict(atol=1.5e-2 * (K**0.5), rtol=1e-2)
    return dict(atol=1e-3 * (K**0.5), rtol=1e-4)


@pytest.mark.parametrize("variant", _variants("sgemm"))
@pytest.mark.parametrize("shape", SGEMM_SHAPES, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_sgemm_matches_torch(sk, shape, variant):
    M, N, K = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    got = sk.sgemm(a, b, variant)
    with matmul_tf32(False):
        ref = a @ b
    assert got.shape == (M, N)
    torch.testing.assert_close(got, ref, **_sgemm_tol(variant, K))


@pytest.mark.parametrize("variant", [v for v in _variants("sgemm") if v >= 6])
@pytest.mark.parametrize("shape", SGEMM_SHAPES_TF32, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_sgemm_tensor_core_variants(sk, shape, variant):
    M, N, K = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    with matmul_tf32(False):
        ref32 = a @ b
    with matmul_tf32(True):
        ref_tf32 = a @ b
    # Twice: the split-K tail zeroes its tiles of C on every call.
    for _ in range(2):
        got = sk.sgemm(a, b, variant)
        torch.testing.assert_close(got, ref32, **_sgemm_tol(variant, K))
    if torch.equal(ref_tf32, ref32):
        return  # torch ran this shape as a GEMV (M = 1), which never rounds to TF32
    if variant == 6:
        # The same precision contract as torch in TF32 mode: the two agree to fp32
        # summation noise, not just to the TF32 tolerance.
        torch.testing.assert_close(got, ref_tf32, **_sgemm_tol(5, K))
    else:
        # 3xTF32 is closer to fp32 than TF32 is, by an order of magnitude at least.
        assert (got - ref32).abs().max() * 10 < (ref_tf32 - ref32).abs().max()


def test_sgemm_default_variant_is_the_fp32_rung(sk):
    # -1 never picks the tensor-core variants: they round the operands to TF32 and are
    # opt-in, as torch's allow_tf32 is.
    M, N, K = 512, 512, 4096
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    with matmul_tf32(False):
        ref = a @ b
    assert sk.num_variants("sgemm") == 8
    err = (sk.sgemm(a, b) - ref).abs().max()
    err_tf32 = (sk.sgemm(a, b, 6) - ref).abs().max()
    # An fp32 rung's error is summation-order noise, an order of magnitude under TF32's
    # rounding error (the two rungs are not compared bit for bit: variant 5's split-K
    # tail adds its slices with atomics, in whichever order they finish).
    assert err * 10 < err_tf32
    torch.testing.assert_close(sk.sgemm(a, b), ref, **_sgemm_tol(5, K))


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


# Variant 5 needs a full wave of 128x128 tiles and has its own shapes below.
@pytest.mark.parametrize("variant", [v for v in _variants("hgemm") if v != 5])
@pytest.mark.parametrize("shape", HGEMM_SHAPES_128, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_multiples_of_128(sk, shape, variant):
    if variant >= 3 and shape[2] % 64 != 0:
        pytest.skip("variants 3, 4 and 6 need K % 64 == 0")
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


@pytest.mark.parametrize("shape", HGEMM_SHAPES_V5, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_variant5_full_waves(sk, shape):
    if 5 not in _variants("hgemm"):
        pytest.skip("variant 5 not built")
    _hgemm_case(sk, shape, 5)
    # Again: the tensor maps are rebuilt per call and the split-K tail reuses a workspace.
    _hgemm_case(sk, shape, 5)


@pytest.mark.parametrize("shape", HGEMM_SHAPES_NOT_V5, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_variant5_steps_down(sk, shape):
    if 5 not in _variants("hgemm"):
        pytest.skip("variant 5 not built")
    M, N, K = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):  # an explicit variant is never substituted
        sk.hgemm(a, b, 5)
    got = sk.hgemm(a, b)  # the default is variant 6, which takes them all
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    torch.testing.assert_close(got.float(), ref.float(), **HGEMM_TOL)


@pytest.mark.parametrize("shape", HGEMM_SHAPES_V4, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_variant4_any_m(sk, shape):
    if 4 not in _variants("hgemm"):
        pytest.skip("variant 4 not built")
    _hgemm_case(sk, shape, 4)
    # Repeated calls must agree too: the Stream-K flags, slots and queue are reused across
    # launches and are never cleared by the host.
    for _ in range(3):
        _hgemm_case(sk, shape, 4)


@pytest.mark.parametrize("shape", [(272, 128, 4096), (128, 64, 16384), (2064, 11008, 1024),
                                   (16, 4096, 1024)],
                         ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_variant4_is_deterministic(sk, shape):
    # The fixup sums the K-pieces of a tile in K order whichever block computed them, so two
    # calls give the same bits (variant 3's atomics do not promise that). In queue mode
    # (2064x11008x1024) the block-to-piece assignment changes from call to call.
    if 4 not in _variants("hgemm"):
        pytest.skip("variant 4 not built")
    M, N, K = shape
    torch.manual_seed(1)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    first = sk.hgemm(a, b, 4)
    for _ in range(5):
        again = sk.hgemm(a, b, 4)
        assert torch.equal(first, again)


@pytest.mark.parametrize("shape", HGEMM_SHAPES_V6, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_variant6_any_m(sk, shape):
    if 6 not in _variants("hgemm"):
        pytest.skip("variant 6 not built")
    # Repeated calls must agree too: the tensor maps are rebuilt per call, and the flags,
    # slots and queue are reused across launches and never cleared by the host.
    for _ in range(3):
        _hgemm_case(sk, shape, 6)


@pytest.mark.parametrize("shape", [(2048, 2048, 2048), (1000, 4096, 4096), (2064, 11008, 2048),
                                   (2100, 11008, 4096)],
                         ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_variant6_is_deterministic(sk, shape):
    # Same fixup as variant 4: the K-pieces of a tile are summed in K order whichever block
    # computed them, so two calls give the same bits. In queue mode (the 11008-wide shapes)
    # the block-to-piece assignment changes from call to call.
    if 6 not in _variants("hgemm"):
        pytest.skip("variant 6 not built")
    M, N, K = shape
    torch.manual_seed(1)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    first = sk.hgemm(a, b, 6)
    for _ in range(5):
        again = sk.hgemm(a, b, 6)
        assert torch.equal(first, again)


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
    got = sk.hgemm(a, b)  # N % 64 != 0: steps down past variants 6 to 2 to variant 1
    ref = (a.float() @ b.float()).to(torch.bfloat16)
    torch.testing.assert_close(got.float(), ref.float(), **HGEMM_TOL)
    with pytest.raises(ValueError):  # an explicit variant is never substituted
        sk.hgemm(a, b, 2)


def _epilogue_ref(a, b, bias, act, residual):
    x = a.float() @ b.float()
    if bias is not None:
        x = x + bias.float()
    if act == "silu":
        x = F.silu(x)
    elif act == "gelu":
        x = F.gelu(x, approximate="tanh")
    elif act == "relu":
        x = F.relu(x)
    if residual is not None:
        x = x + residual.float()
    return x.to(torch.bfloat16)


def _epilogue_id(e):
    bias, act, res = e
    return "+".join(t for t in (("bias" if bias else ""), act or "", res or "") if t) or "plain"


@pytest.mark.parametrize("epilogue", HGEMM_EPILOGUES, ids=_epilogue_id)
@pytest.mark.parametrize("shape", HGEMM_SHAPES_EPILOGUE, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_fused_epilogue(sk, shape, epilogue):
    if 6 not in _variants("hgemm"):
        pytest.skip("variant 6 not built")
    M, N, K = shape
    has_bias, act, res_mode = epilogue
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16) * 4 if has_bias else None
    residual = (torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 4
                if res_mode else None)
    ref = _epilogue_ref(a, b, bias, act, residual)
    for variant in (-1, 6, 4):
        if res_mode == "accumulate":
            out = residual.clone()
            got = sk.hgemm(a, b, variant, bias=bias, act=act, residual=out, out=out)
            assert got.data_ptr() == out.data_ptr()
        else:
            got = sk.hgemm(a, b, variant, bias=bias, act=act, residual=residual)
        assert got.shape == (M, N) and got.dtype == torch.bfloat16
        torch.testing.assert_close(got.float(), ref.float(), **HGEMM_TOL)


@pytest.mark.parametrize("shape", HGEMM_SHAPES_EPILOGUE, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_hgemm_swiglu(sk, shape):
    # N here is the width of each of gate and up; the interleaved weight is [K, 2N].
    if 6 not in _variants("hgemm"):
        pytest.skip("variant 6 not built")
    M, N, K = shape
    torch.manual_seed(0)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    # Weights scaled to K^-1/2 so gate and up are O(1): the product silu(g) * u multiplies the
    # fp32 summation-order noise of u by |g|, and with |g| ~ 64 (unscaled randn at K = 4096)
    # a near-cancelling u turns 2e-4 of noise into a 20% miss on one element in a million.
    w_gate = (torch.randn(K, N, device="cuda") * K**-0.5).to(torch.bfloat16)
    w_up = (torch.randn(K, N, device="cuda") * K**-0.5).to(torch.bfloat16)
    w = sk.interleave_gate_up(w_gate, w_up)
    assert w.shape == (K, 2 * N) and torch.equal(w[:, 0::2], w_gate) and torch.equal(w[:, 1::2],
                                                                                    w_up)
    ref = (F.silu(a.float() @ w_gate.float()) * (a.float() @ w_up.float())).to(torch.bfloat16)
    for variant in (-1, 6, 4):
        got = sk.hgemm_swiglu(a, w, variant)
        assert got.shape == (M, N) and got.dtype == torch.bfloat16
        torch.testing.assert_close(got.float(), ref.float(), **HGEMM_TOL)
    # With an interleaved bias and a residual on the product.
    b_gate = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    b_up = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(M, N, device="cuda", dtype=torch.bfloat16) * 4
    ref = (F.silu(a.float() @ w_gate.float() + b_gate.float())
           * (a.float() @ w_up.float() + b_up.float()) + residual.float()).to(torch.bfloat16)
    got = sk.hgemm_swiglu(a, w, bias=sk.interleave_gate_up(b_gate, b_up), residual=residual)
    torch.testing.assert_close(got.float(), ref.float(), **HGEMM_TOL)


def test_hgemm_swiglu_matches_unfused(sk):
    # The fused product against our own unfused route: two GEMMs then swiglu. Both round the
    # inputs of the product differently (the fused form never rounds gate and up), so this is
    # a tolerance check, not bitwise.
    M, N, K = 64, 2048, 1024
    torch.manual_seed(2)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w_gate = (torch.randn(K, N, device="cuda") * K**-0.5).to(torch.bfloat16)
    w_up = (torch.randn(K, N, device="cuda") * K**-0.5).to(torch.bfloat16)
    fused = sk.hgemm_swiglu(a, sk.interleave_gate_up(w_gate, w_up))
    unfused = sk.swiglu(sk.hgemm(a, w_gate), sk.hgemm(a, w_up))
    torch.testing.assert_close(fused.float(), unfused.float(), **HGEMM_TOL)


def test_hgemm_epilogue_is_deterministic(sk):
    # The Stream-K fixup still sums the pieces in K order; the epilogue runs once on the
    # finished sum, so the fused output is the same bits every call too (queue mode here).
    M, N, K = 2064, 11008, 2048
    torch.manual_seed(1)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    first = sk.hgemm(a, b, 6, bias=bias, act="gelu", residual=residual)
    for _ in range(3):
        assert torch.equal(first, sk.hgemm(a, b, 6, bias=bias, act="gelu", residual=residual))


def test_hgemm_epilogue_rejected_on_other_variants(sk):
    M, N, K = 256, 256, 256
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
    for variant in (0, 1, 2, 3, 5):
        if variant not in _variants("hgemm"):
            continue
        with pytest.raises(ValueError):
            sk.hgemm(a, b, variant, bias=bias)
        with pytest.raises(ValueError):
            sk.hgemm(a, b, variant, act="gelu")
        with pytest.raises(ValueError):
            sk.hgemm(a, b, variant, residual=residual)
        with pytest.raises(ValueError):
            sk.hgemm_swiglu(a, b, variant)
    # A shape variants 4 and 6 refuse (N % 64 != 0) steps the default down to a variant that
    # has no epilogue, which is a ValueError, not a silent plain GEMM.
    b48 = torch.randn(K, 48, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        sk.hgemm(a, b48, bias=torch.randn(48, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(ValueError):
        sk.hgemm(a, b, act="tanh")  # not an activation


def test_hgemm_epilogue_rejects_bad_shapes(sk):
    M, N, K = 128, 256, 256
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):  # bias must be [N]
        sk.hgemm(a, b, bias=torch.randn(M, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(RuntimeError):  # residual must be [M, N]
        sk.hgemm(a, b, residual=torch.randn(M, 2 * N, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(RuntimeError):  # swiglu residual is [M, N / 2]
        sk.hgemm_swiglu(a, b, residual=torch.randn(M, N, device="cuda", dtype=torch.bfloat16))
    with pytest.raises(RuntimeError):  # out must be [M, N] bf16
        sk.hgemm(a, b, out=torch.empty(M, N, device="cuda", dtype=torch.float32))
    with pytest.raises(ValueError):
        sk.interleave_gate_up(torch.zeros(4, 8), torch.zeros(4, 6))
