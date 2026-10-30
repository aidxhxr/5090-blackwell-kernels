"""Parity of sk.fp8gemm with torch._scaled_mm (cuBLASLt) on the same e4m3 bytes, scales and
TN layout: a is [M, K], b_t is [N, K], both K-contiguous, per-tensor fp32 scales, bf16 out.
The MX tests at the end run the same rungs with a ue8m0 scale per 32 k on both operands
against the fp32 product of the dequantized values and, where torch has it, cuBLASLt's
MXFP8 kernel through torch.nn.functional.scaled_mm."""

import pytest
import torch
import torch.nn.functional as F

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
    got = sk.fp8gemm(a, b_t, sa, sb)  # the default (variant 3) takes any shape
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
    assert sk.num_variants("fp8gemm") == 4


# ---- variant 3: the TMA tiles on the Stream-K schedule, routed per shape ------------------
# Shapes the default rule sends to each tile (128x64 at M = 128 and ragged M under 128, 64x128
# for ragged M with fewer zero-filled rows, 128x128, 128x256), a K too short for the share
# rule, and the shapes it leaves to the old route (decode, K % 128 != 0).
FP8_SHAPES_V3 = [(128, 4096, 4096), (100, 1024, 768), (320, 1024, 1024), (256, 4096, 4096),
                 (1000, 2048, 1024), (512, 8192, 2048), (256, 5120, 2304), (16, 4096, 4096),
                 (256, 1024, 192)]
V3_ROUTES = {(128, 4096, 4096): "tma 128x64 bk128 s4", (100, 1024, 768): "tma 128x64 bk128 s4",
             (320, 1024, 1024): "tma 64x128 bk128 s4", (256, 4096, 4096): "tma 128x128 bk128 s3",
             (1000, 2048, 1024): "tma 128x128 bk128 s3", (512, 8192, 2048): "tma 128x256 bk128 s2",
             (256, 5120, 2304): "tma 128x64 bk128 s4", (16, 4096, 4096): "v1 decode",
             (256, 1024, 192): "v1"}
# Every tile forced on shapes that exercise its schedule: one piece per tile, Stream-K
# splits of a tile over several blocks (static ranges), rows past M, and the queue with
# K-passes (A and Bt beyond L2, more than two waves).
FP8_SHAPES_V3_FORCED = [(128, 2048, 512), (200, 1024, 4096), (777, 2560, 2048),
                        (256, 81920, 1024)]


@pytest.fixture
def v3_route(sk):
    """Sets fp8gemm variant 3's route for one test and puts the old one back."""
    if 3 not in _variants("fp8gemm"):
        pytest.skip("variant 3 not built")
    prev = sk._C.fp8gemm_v3_set_route(-1)
    yield sk._C.fp8gemm_v3_set_route
    sk._C.fp8gemm_v3_set_route(prev)


@pytest.mark.parametrize("shape", FP8_SHAPES_V3, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_fp8gemm_variant3_routes(sk, v3_route, shape):
    assert sk._C.fp8gemm_v3_route_name(*shape) == V3_ROUTES[shape]
    _case(sk, shape, 3)
    _case(sk, shape, 3)  # the Stream-K workspace and the queue are reused between calls


@pytest.mark.parametrize("shape", FP8_SHAPES_V3_FORCED, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_fp8gemm_variant3_forced_tiles(sk, v3_route, shape):
    for tile in range(sk._C.fp8gemm_v3_num_tiles()):
        v3_route(tile)
        assert sk._C.fp8gemm_v3_route_name(*shape).startswith("tma")
        _case(sk, shape, 3)


def test_fp8gemm_variant3_same_bits_every_run(sk, v3_route):
    # Gaussian e4m3 inputs at K = 8192: the fp32 partial sums are not exact, so the bits
    # depend on the order the pieces of a tile are added in. The fixup adds them in K order
    # whichever block computed them, so every run gives the same bits, on every tile.
    a, b_t, sa, sb = _inputs(384, 2048, 8192, seed=5)
    for tile in [-1] + list(range(sk._C.fp8gemm_v3_num_tiles())):
        v3_route(tile)
        first = sk.fp8gemm(a, b_t, sa, sb, 3)
        for _ in range(3):
            assert torch.equal(sk.fp8gemm(a, b_t, sa, sb, 3), first), tile


def test_fp8gemm_variant3_legacy_route_is_the_old_default(sk, v3_route):
    # Route -2 runs what the default ran before variant 3: variant 2 where it takes the
    # shape, variant 1 elsewhere. Neither has a split-K tail on these two shapes, so the bits
    # are comparable on Gaussian inputs.
    v3_route(-2)
    for shape, old, name in [((2048, 2048, 2048), 2, "v2"), ((16, 4096, 4096), 1, "v1 decode")]:
        a, b_t, sa, sb = _inputs(*shape, seed=7)
        assert sk._C.fp8gemm_v3_route_name(*shape) == name
        assert torch.equal(sk.fp8gemm(a, b_t, sa, sb, 3), sk.fp8gemm(a, b_t, sa, sb, old))
    assert sk._C.fp8gemm_v3_route_name(256, 4096, 4096) == "v1"


def test_fp8gemm_variant3_rejects_bad_routes(sk, v3_route):
    with pytest.raises(RuntimeError):
        v3_route(sk._C.fp8gemm_v3_num_tiles())
    with pytest.raises(RuntimeError):
        v3_route(-3)


# ---- MX mode: sfa [M, K/32] and sfb [N, K/32] ue8m0 block scales -------------------------
# The same rungs and tile paths as above, on K a multiple of 256 (the MX requirement): both
# scale chunk widths (K % 512 == 0 takes 16-byte chunks, K = 11008 or 256 + 512n the 8-byte
# ones), the split-K tail on chunk-group boundaries, ragged M, every decode row tile.
MX_SHAPES_V0 = [(16, 64, 256), (256, 256, 256), (272, 1024, 4096), (48, 128, 11008)]
MX_SHAPES_V1 = [(256, 256, 256), (272, 128, 4096), (208, 1088, 256), (1024, 1024, 1024),
                (100, 1024, 1024), (2048, 2048, 2048), (272, 128, 11008), (200, 256, 768)]
MX_SHAPES_DECODE = [(1, 4096, 4096), (16, 4096, 4096), (16, 11008, 4096), (16, 256, 4096),
                    (16, 4096, 256), (5, 4096, 256), (32, 4096, 4096), (32, 1024, 11008),
                    (33, 256, 4096), (64, 4096, 4096), (64, 4096, 11008), (63, 11008, 256),
                    (64, 1024, 4096), (64, 4096, 768)]
MX_SHAPES_V2 = [(128, 21760, 256), (1792, 1792, 256), (256, 11008, 4096), (2048, 2048, 2048),
                (2560, 2560, 4096), (8192, 8192, 1024), (256, 22016, 768)]
# The dequantized products are exact in fp32 and both sides round once to bf16; the sums of
# blocks with different exponents differ by fp32 order only. Outputs are O(sqrt(K)), so the
# relative term dominates.
MX_TOL = dict(atol=1e-2, rtol=3e-2)


def _mx_inputs(M, N, K, seed=0):
    torch.manual_seed(seed)
    ref = pytest.importorskip("spark_kernels").reference
    a, sfa = ref.quantize_mx(torch.randn(M, K, device="cuda", dtype=torch.bfloat16))
    b_t, sfb = ref.quantize_mx(torch.randn(N, K, device="cuda", dtype=torch.bfloat16))
    return a, sfa, b_t, sfb


def _mx_case(sk, shape, variant):
    M, N, K = shape
    a, sfa, b_t, sfb = _mx_inputs(M, N, K)
    got = sk.fp8gemm(a, b_t, variant=variant, sfa=sfa, sfb=sfb)
    assert got.shape == (M, N) and got.dtype == torch.bfloat16
    ref = sk.reference.fp8gemm_mx(a, sfa, b_t, sfb)
    torch.testing.assert_close(got.float(), ref.float(), **MX_TOL)


def _mx_supported(sk):
    a, sfa, b_t, sfb = _mx_inputs(16, 64, 256)
    try:
        sk.fp8gemm(a, b_t, sfa=sfa, sfb=sfb)
    except ValueError as e:  # a plain sm_120 build has no block-scaled mma
        pytest.skip(str(e))


@pytest.mark.parametrize("shape", MX_SHAPES_V0, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_fp8gemm_mx_variant0(sk, shape):
    _mx_supported(sk)
    _mx_case(sk, shape, 0)


@pytest.mark.parametrize("shape", MX_SHAPES_V1, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_fp8gemm_mx_variant1(sk, shape):
    _mx_supported(sk)
    _mx_case(sk, shape, 1)
    _mx_case(sk, shape, 1)


@pytest.mark.parametrize("shape", MX_SHAPES_DECODE, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_fp8gemm_mx_decode_shapes(sk, shape):
    _mx_supported(sk)
    for _ in range(2):
        _mx_case(sk, shape, 1)


@pytest.mark.parametrize("shape", MX_SHAPES_V2, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_fp8gemm_mx_variant2(sk, shape):
    if 2 not in _variants("fp8gemm"):
        pytest.skip("variant 2 not built")
    _mx_supported(sk)
    _mx_case(sk, shape, 2)
    _mx_case(sk, shape, 2)


@pytest.mark.parametrize("shape", [(256, 4096, 4096), (2048, 2048, 2048), (33, 1024, 768)],
                         ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_fp8gemm_mx_variant3(sk, shape):
    # MX mode on variant 3 runs variant 2 where it takes the shape and variant 1 elsewhere.
    if 3 not in _variants("fp8gemm"):
        pytest.skip("variant 3 not built")
    _mx_supported(sk)
    _mx_case(sk, shape, 3)


# Variant 3's tiles forced in MX mode: the Stream-K pieces cut on scale-chunk groups, both
# chunk widths (K % 512 == 0 fetches 16-byte chunks, four stages a group; K = 768, 2304 and
# 11008 the 8-byte ones, two a group), group counts that do not divide into the ring or the
# grid (9 groups at K = 2304, 43 at 11008), and rows past M.
# Every tile forced: one piece per tile, static ranges that split a tile over several
# blocks on group boundaries, rows past M, the queue with K-passes, and the 8-byte chunks.
MX_SHAPES_V3_FORCED = [(128, 2048, 512), (200, 1024, 4096), (777, 2560, 2304), (256, 81920, 1024),
                       (300, 1024, 768)]


@pytest.mark.parametrize("shape", MX_SHAPES_V3_FORCED, ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_fp8gemm_mx_variant3_forced_tiles(sk, v3_route, shape):
    _mx_supported(sk)
    for tile in range(sk._C.fp8gemm_v3_num_tiles()):
        v3_route(tile)
        assert sk._C.fp8gemm_v3_route_name(*shape, mx=True).startswith("tma")
        _mx_case(sk, shape, 3)


def test_fp8gemm_mx_variant3_pieces_carry_the_scales(sk, v3_route):
    # The pieces of a tile each fetch their own scale chunks and are summed in K order
    # through the slot. Inputs that are multiples of 1/16 in [-1, 1] with block scales in
    # 2^-1 .. 2^1 at K = 2304: every product is a multiple of 2^-12, every partial sum is at
    # most 2^14 in magnitude, so the fp32 sums are exact whatever the split, and every tile
    # must give variant 0's bits (which are cuBLASLt's). 128x768 is 12 tiles of 128x64 or 6
    # of 128x128 on a grid that cuts each into two or more pieces of 9 groups of two stages;
    # 300 rows add a ragged tile.
    _mx_supported(sk)
    for M, N, K in [(128, 768, 2304), (300, 768, 2304), (128, 4096, 11008)]:
        torch.manual_seed(M + K)
        a = (torch.randint(-16, 17, (M, K), device="cuda") / 16.0).to(torch.float8_e4m3fn)
        b_t = (torch.randint(-16, 17, (N, K), device="cuda") / 16.0).to(torch.float8_e4m3fn)
        sfa = torch.randint(126, 129, (M, K // 32), device="cuda").to(torch.uint8)
        sfb = torch.randint(126, 129, (N, K // 32), device="cuda").to(torch.uint8)
        if K > 4096:  # 2^-12 granularity against 2^14 of magnitude at K = 2304; scale 1 above
            sfa.fill_(127)
            sfb.fill_(127)
        want = sk.fp8gemm(a[: M // 16 * 16], b_t, variant=0, sfa=sfa[: M // 16 * 16], sfb=sfb)
        for tile in [-1] + list(range(sk._C.fp8gemm_v3_num_tiles())):
            v3_route(tile)
            got = sk.fp8gemm(a, b_t, variant=3, sfa=sfa, sfb=sfb)
            assert torch.equal(got[: M // 16 * 16], want), (M, N, K, tile)


def test_fp8gemm_mx_variant3_same_bits_every_run(sk, v3_route):
    # Gaussian MX inputs at K = 8192: the partial sums are not exact, and the K-ordered fixup
    # makes the bits independent of which block computed which piece, on every tile.
    _mx_supported(sk)
    a, sfa, b_t, sfb = _mx_inputs(384, 2048, 8192, seed=5)
    for tile in range(sk._C.fp8gemm_v3_num_tiles()):
        v3_route(tile)
        first = sk.fp8gemm(a, b_t, variant=3, sfa=sfa, sfb=sfb)
        for _ in range(3):
            assert torch.equal(sk.fp8gemm(a, b_t, variant=3, sfa=sfa, sfb=sfb), first), tile


def test_fp8gemm_mx_variant3_legacy_route(sk, v3_route):
    _mx_supported(sk)
    v3_route(-2)
    a, sfa, b_t, sfb = _mx_inputs(2048, 2048, 2048, seed=7)
    assert sk._C.fp8gemm_v3_route_name(2048, 2048, 2048, mx=True) == "v2"
    assert torch.equal(sk.fp8gemm(a, b_t, variant=3, sfa=sfa, sfb=sfb),
                       sk.fp8gemm(a, b_t, variant=2, sfa=sfa, sfb=sfb))


def test_fp8gemm_mx_default_steps_down(sk):
    _mx_supported(sk)
    a, sfa, b_t, sfb = _mx_inputs(1024, 1024, 1024)  # too few tiles for variant 2
    with pytest.raises(ValueError):
        sk.fp8gemm(a, b_t, variant=2, sfa=sfa, sfb=sfb)
    got = sk.fp8gemm(a, b_t, sfa=sfa, sfb=sfb)
    torch.testing.assert_close(got.float(), sk.reference.fp8gemm_mx(a, sfa, b_t, sfb).float(),
                               **MX_TOL)


def test_fp8gemm_mx_unit_scales_match_per_tensor_bitwise(sk):
    # With every block scale 2^0 the instruction computes the per-tensor kernel's bits, and
    # with a constant 2^ea on A and 2^eb on B it computes the per-tensor kernel's bits with
    # scale 2^(ea+eb): the scale words reach the right rows, columns and k-blocks or the
    # outputs differ, at every variant and tile. The inputs are multiples of 1/16 in [-1, 1],
    # so every product is a multiple of 2^-8 and every partial sum (at most 2^20 in
    # magnitude) is exact in fp32 whatever the split-K order, which MX mode changes: its
    # slices start on chunk-group boundaries.
    _mx_supported(sk)
    for M, N, K in [(256, 256, 256), (272, 1088, 4096), (33, 4096, 768), (2048, 2048, 2048),
                    (128, 768, 2304), (200, 1024, 11008)]:
        torch.manual_seed(M + N + K)
        a = (torch.randint(-16, 17, (M, K), device="cuda") / 16.0).to(torch.float8_e4m3fn)
        b_t = (torch.randint(-16, 17, (N, K), device="cuda") / 16.0).to(torch.float8_e4m3fn)
        one = torch.ones((), device="cuda")
        for ea, eb in [(0, 0), (3, -2)]:
            sfa = torch.full((M, K // 32), 127 + ea, dtype=torch.uint8, device="cuda")
            sfb = torch.full((N, K // 32), 127 + eb, dtype=torch.uint8, device="cuda")
            scale = torch.tensor(2.0 ** (ea + eb), device="cuda")
            plain = sk.fp8gemm(a, b_t, scale, one)
            for v in _variants("fp8gemm"):
                if not (M % 16 == 0 or v != 0):
                    continue
                try:
                    got = sk.fp8gemm(a, b_t, variant=v, sfa=sfa, sfb=sfb)
                except ValueError:  # variant 2 refuses the shape
                    continue
                assert torch.equal(got, plain), (M, N, K, ea, eb, v)


def test_fp8gemm_mx_per_tensor_scale_on_top(sk):
    _mx_supported(sk)
    a, sfa, b_t, sfb = _mx_inputs(256, 512, 1024)
    got = sk.fp8gemm(a, b_t, torch.tensor(0.5, device="cuda"), torch.tensor(1.5, device="cuda"),
                     sfa=sfa, sfb=sfb)
    ref = (sk.reference.dequantize_mx(a, sfa) @ sk.reference.dequantize_mx(b_t, sfb).t()) * 0.75
    torch.testing.assert_close(got.float(), ref, **MX_TOL)


def _to_blocked(sf):
    # cuBLASLt's 32x4x4 tiled layout of a [rows, K/32] scale tensor (torch's SWIZZLE_32_4_4):
    # 128-row by 4-block tiles of 512 bytes, byte (r % 32) * 16 + (r // 32) * 4 + kb % 4.
    rows, cols = sf.shape
    assert rows % 128 == 0 and cols % 4 == 0
    t = sf.view(rows // 128, 128, cols // 4, 4).permute(0, 2, 1, 3)
    return t.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1).contiguous()


@pytest.mark.parametrize("shape", [(256, 256, 4096), (512, 1024, 11008), (2048, 2048, 2048)],
                         ids=lambda s: f"M{s[0]}_N{s[1]}_K{s[2]}")
def test_fp8gemm_mx_matches_torch_scaled_mm(sk, shape):
    _mx_supported(sk)
    M, N, K = shape
    a, sfa, b_t, sfb = _mx_inputs(M, N, K)
    try:
        ref = F.scaled_mm(a, b_t.t(), _to_blocked(sfa).view(torch.float8_e8m0fnu),
                          F.ScalingType.BlockWise1x32, _to_blocked(sfb).view(torch.float8_e8m0fnu),
                          F.ScalingType.BlockWise1x32, swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
                          swizzle_b=F.SwizzleType.SWIZZLE_32_4_4, output_dtype=torch.bfloat16)
    except (AttributeError, RuntimeError, ValueError, NotImplementedError) as e:
        pytest.skip(f"torch has no MXFP8 scaled_mm here: {e}")
    got = sk.fp8gemm(a, b_t, sfa=sfa, sfb=sfb)
    torch.testing.assert_close(got.float(), ref.float(), **MX_TOL)


def test_fp8gemm_mx_rejects_bad_inputs(sk):
    _mx_supported(sk)
    a, sfa, b_t, sfb = _mx_inputs(64, 128, 256)
    with pytest.raises(RuntimeError):  # only one scale tensor
        sk.fp8gemm(a, b_t, sfa=sfa)
    with pytest.raises(RuntimeError):  # wrong shape
        sk.fp8gemm(a, b_t, sfa=sfa[:, :4], sfb=sfb)
    with pytest.raises(RuntimeError):  # wrong dtype
        sk.fp8gemm(a, b_t, sfa=sfa.to(torch.int8), sfb=sfb)
    with pytest.raises(RuntimeError):  # per-tensor mode needs its scales
        sk.fp8gemm(a, b_t)
    a2, sfa2, b2, sfb2 = _mx_inputs(64, 128, 128)  # K % 256 != 0
    with pytest.raises((RuntimeError, ValueError)):
        sk.fp8gemm(a2, b2, sfa=sfa2, sfb=sfb2)


def test_quantize_mx_reference():
    ref = pytest.importorskip("spark_kernels").reference
    torch.manual_seed(1)
    x = torch.randn(64, 1024, device="cuda", dtype=torch.bfloat16) * 8
    x[3, 40] = 3000.0  # an outlier: its block scales up, the others do not
    x[5, 64:96] = 0.0  # an all-zero block
    nonzero = x.float().abs().reshape(64, 32, 32).amax(dim=-1) > 0
    for mode, lo, hi in (("floor", 256, 448), ("ceil", 224, 448)):
        q, sf = ref.quantize_mx(x, mode=mode)
        assert q.dtype == torch.float8_e4m3fn and sf.dtype == torch.uint8
        assert q.shape == x.shape and sf.shape == (64, 32)
        assert int(sf.max()) <= 254
        assert int(sf[5, 2]) == 0 and torch.all(q[5, 64:96].float() == 0)
        # the largest element of every block lands in e4m3's top binade: [256, 448] with the
        # OCP floor recipe (a maximum above 448 saturates), (224, 448] with the ceil one
        qb = q.float().abs().reshape(64, 32, 32).amax(dim=-1)
        assert torch.all(qb[nonzero] >= lo) and torch.all(qb[nonzero] <= hi)
        # e4m3 keeps 3 mantissa bits: every element in its normal range (at least 2^-6 after
        # scaling, at most 448) is within 1/16 of itself after the round trip; a block's
        # smallest elements, more than 2^14 below its largest, land in the subnormals and
        # round to an absolute step of 2^-9 of the scale; above 448 (floor mode) they saturate
        back = ref.dequantize_mx(q, sf)
        scale = torch.exp2(sf.float() - 127).repeat_interleave(32, dim=-1)
        scaled = x.float().abs() / scale
        err = (back - x.float()).abs()
        rel = err / x.float().abs().clamp(min=1e-30)
        normal = (scaled >= 2 ** -6) & (scaled <= 448)
        assert torch.all(rel[normal] <= 2 ** -4 + 1e-6)
        assert torch.all(err[scaled < 2 ** -6] <= 2 ** -10 * scale[scaled < 2 ** -6] + 1e-30)
        assert torch.all(back.abs()[scaled > 448] == 448 * scale[scaled > 448])
        assert mode == "floor" or not torch.any(scaled > 448)
        # the outlier's block is the only one whose scale moved
        assert int(sf[3, 1]) > int(sf[3, 0]) and int(sf[3, 1]) >= 127 + 3


def test_mx_against_per_tensor_on_outliers():
    # What the two quantizations cost on the GEMM output (scripts/mx_accuracy.py has the
    # table): e4m3's 3 mantissa bits put every scheme near 3.7% relative Frobenius error;
    # the ceil block scale matches per-tensor scaling within a percent of that, on Gaussian
    # data and with 8 channels of A at 2000x the rest alike (a per-tensor scale keeps such
    # outliers inside e4m3's 2^14.8 of dynamic range, and the outlier channels dominate the
    # outputs in both schemes); the OCP floor recipe costs more, since a block maximum in
    # the top eighth of its binade saturates at 448.
    ref = pytest.importorskip("spark_kernels").reference
    torch.manual_seed(2)
    b = torch.randn(512, 2048, device="cuda")

    def errors(a):
        exact = a @ b.t()
        qa, sa = ref.quantize_per_tensor(a)
        qb, sb = ref.quantize_per_tensor(b)
        per_tensor = (qa.float() @ qb.float().t()) * (sa * sb)
        out = [((per_tensor - exact).norm() / exact.norm()).item()]
        for mode in ("floor", "ceil"):
            ma, sfa = ref.quantize_mx(a, mode=mode)
            mb, sfb = ref.quantize_mx(b, mode=mode)
            mx = ref.dequantize_mx(ma, sfa) @ ref.dequantize_mx(mb, sfb).t()
            out.append(((mx - exact).norm() / exact.norm()).item())
        return out

    for outlier in (1.0, 2000.0):
        a = torch.randn(512, 2048, device="cuda")
        a[:, ::256] *= outlier
        err_pt, err_floor, err_ceil = errors(a)
        assert err_ceil <= err_pt * 1.03, (outlier, err_pt, err_floor, err_ceil)
        assert err_pt < err_floor <= err_pt * 1.6, (outlier, err_pt, err_floor, err_ceil)


@pytest.mark.parametrize("shape", [(1, 4096), (3, 72), (64, 14336), (300, 1024), (2048, 4096)],
                         ids=lambda s: f"{s[0]}x{s[1]}")
@pytest.mark.parametrize("mode", ["tensor", "row", "mx"])
def test_fp8_quantize_matches_reference(sk, shape, mode):
    """The activation quantizer against reference.quantize_fp8_pow2 / quantize_mx, byte for
    byte: an outlier, an all-zero row, tiny and large values."""
    rows, K = shape
    if mode == "mx" and K % 32:
        pytest.skip("mx needs K % 32 == 0")
    g = torch.Generator(device="cpu").manual_seed(rows * K)
    x = (torch.randn(rows, K, generator=g) * 3).to("cuda", torch.bfloat16)
    x[0, K // 2] = 900.0
    if rows > 2:
        x[1] = 0
        x[2] *= 1e-30
    q, s = sk.fp8_quantize(x, mode)
    assert q.dtype == torch.float8_e4m3fn and q.shape == x.shape
    if mode == "mx":
        q_ref, s_ref = sk.reference.quantize_mx(x)
        assert torch.equal(s, s_ref)
    else:
        q_ref, s_ref = sk.reference.quantize_fp8_pow2(x, per_row=mode == "row")
        assert s.shape == s_ref.shape and torch.equal(s, s_ref)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))


def test_fp8_quantize_feeds_fp8gemm(sk):
    """Per-tensor and MX quantized activations through fp8gemm against the dequantized GEMM."""
    a = torch.randn(48, 512, device="cuda").to(torch.bfloat16)
    w = torch.randn(256, 512, device="cuda").to(torch.bfloat16)
    wq, ws = sk.reference.quantize_per_tensor(w)
    q, s = sk.fp8_quantize(a, "tensor")
    out = sk.fp8gemm(q, wq, s, ws.reshape(1))
    want = (q.float() * s) @ (wq.float() * ws).t()
    torch.testing.assert_close(out.float(), want, atol=2e-2, rtol=2e-2)
    qm, sfa = sk.fp8_quantize(a, "mx")
    wm, sfb = sk.reference.quantize_mx(w)
    out = sk.fp8gemm(qm, wm, sfa=sfa, sfb=sfb)
    want = sk.reference.dequantize_mx(qm, sfa) @ sk.reference.dequantize_mx(wm, sfb).t()
    torch.testing.assert_close(out.float(), want, atol=2e-2, rtol=2e-2)
