"""The decode kernels' launch rules (src/kernels/launch_config.cuh) through the bindings.

The split-K rule and the L2 question are pure host arithmetic on the two named device shapes
(the GB10 and the RTX 5090), so they run wherever the extension is built, GPU or not; the
`sk` fixture tests read the real device through `spark_kernels.ops` and are skipped without
CUDA (tests/conftest.py).
Nothing here has run on a GB10."""

from __future__ import annotations

import pytest
import torch

try:
    from spark_kernels import _C
except Exception:  # the extension is not built on this machine
    _C = None

needs_ext = pytest.mark.skipif(_C is None, reason="spark_kernels extension not built")

# The two named shapes, as launch_config.cuh defines them.
GB10 = dict(sm_count=48, l2_bytes=24 << 20, spec_gbps=273.0, bytes_in_flight=273 * 800)
RTX5090 = dict(sm_count=170, l2_bytes=96 << 20, spec_gbps=1792.0, bytes_in_flight=64 * 24 * 1024)

HGEMM_CTA = 24576  # 3 stages x 64 x 64 bf16 of B in flight, the 16x64x64 decode tile
W4_BLOCK = 12288   # the figure that reproduces w4gemm's swept 128 blocks on the 5090
W1_BLOCK = 8192    # 4 warps x 2 groups x 2 rows x 512 B


def cdiv(a: int, b: int) -> int:
    return -(-a // b)


# ---- the bandwidth rule, pure arithmetic ----------------------------------------------------


@needs_ext
@pytest.mark.parametrize("per_cta, want", [(HGEMM_CTA, 64), (W4_BLOCK, 128), (W1_BLOCK, 192)])
def test_need_ctas_rtx5090_reproduces_the_old_constants(per_cta: int, want: int) -> None:
    assert _C.need_ctas(per_cta, "rtx5090") == want


@needs_ext
@pytest.mark.parametrize("per_cta, want", [(HGEMM_CTA, 9), (W4_BLOCK, 18), (W1_BLOCK, 27)])
def test_need_ctas_gb10_at_800ns(per_cta: int, want: int) -> None:
    assert _C.need_ctas(per_cta, "gb10") == want
    assert want == cdiv(GB10["bytes_in_flight"], per_cta)


@needs_ext
@pytest.mark.parametrize("strips", [1, 4, 16, 36, 48, 64, 96, 448])
@pytest.mark.parametrize("kt", [8, 64, 224])
def test_split_k_rtx5090_is_the_old_hgemm_decode_line(strips: int, kt: int) -> None:
    # Old: p.split = min(K / BK, max(1, cdiv(kMinCtas, strips))) with kMinCtas = 64, for
    # every tile configuration. slots = 340 (2 CTAs per SM) as for the 16x64x64 tile.
    old = min(kt, max(1, cdiv(64, strips)))
    got = _C.split_k_for(2 * 4096 * strips * 64, strips, kt, HGEMM_CTA, 340, "rtx5090")
    assert got == old


@needs_ext
@pytest.mark.parametrize("tiles", [16, 64, 128, 144, 256, 384, 1792])
@pytest.mark.parametrize("groups", [4, 32, 112])
def test_split_k_rtx5090_is_the_old_w4gemm_line(tiles: int, groups: int) -> None:
    # Old: split = cdiv(kMinBlocks, tiles) with kMinBlocks = 128, then clamped to [1, G].
    old = max(1, min(cdiv(128, tiles), groups))
    got = _C.split_k_for(tiles * 16 * 4096 // 2, tiles, groups, W4_BLOCK, 2720, "rtx5090")
    assert got == old


@needs_ext
@pytest.mark.parametrize("strips, want",
                         [(1, 9), (4, 3), (8, 2), (9, 1), (16, 1), (36, 1), (64, 1)])
def test_split_k_gb10_hgemm(strips: int, want: int) -> None:
    assert _C.split_k_for(2 * 4096 * strips * 64, strips, 64, HGEMM_CTA, 96, "gb10") == want


@needs_ext
def test_split_k_gb10_stops_splitting_the_narrow_llama_shapes() -> None:
    # k or v alone: 16 strips of 64 columns split 4 ways on the 5090, not on the GB10.
    assert _C.split_k_for(2 * 4096 * 1024, 16, 64, HGEMM_CTA, 340, "rtx5090") == 4
    assert _C.split_k_for(2 * 4096 * 1024, 16, 64, HGEMM_CTA, 96, "gb10") == 1
    # The same shape through w4gemm's 64 16-column strips: 2 ways against 1.
    assert _C.split_k_for(4096 * 1024 // 2, 64, 32, W4_BLOCK, 2720, "rtx5090") == 2
    assert _C.split_k_for(4096 * 1024 // 2, 64, 32, W4_BLOCK, 768, "gb10") == 1


@needs_ext
def test_split_k_clamps_to_max_split() -> None:
    assert _C.split_k_for(0, 1, 8, HGEMM_CTA, 340, "rtx5090") == 8
    assert _C.split_k_for(0, 1, 1, HGEMM_CTA, 96, "gb10") == 1
    assert _C.split_k_for(0, 1, 64, HGEMM_CTA, 96, "gb10") == 9


@needs_ext
def test_wave_clause_only_past_one_wave_of_the_bandwidth_rule() -> None:
    # 1 KB per CTA on the GB10 asks for 214 CTAs, more than 48 slots: 50 tiles would run
    # 5 slices (250 CTAs, last wave 250 / 288 = 0.87 full), and the clause moves to 6
    # (300 CTAs, 300 / 336 = 0.89). With 1,000 slots the rule is under one wave and stays.
    assert _C.need_ctas(1024, "gb10") == 214
    assert _C.split_k_for(0, 50, 64, 1024, 48, "gb10") == 6
    assert _C.split_k_for(0, 50, 64, 1024, 1000, "gb10") == 5
    # On the 5090 the bandwidth rule never exceeds one wave for the two kernels that call it
    # (hgemm_decode and w4gemm). w1gemm's 8 KB block would ask for 192, but its grid is N / 16
    # blocks and never goes through split_k_for (src/kernels/w1gemm.cu).
    for per_cta in (HGEMM_CTA, W4_BLOCK):
        assert _C.need_ctas(per_cta, "rtx5090") <= 170


@needs_ext
def test_forced_split_replaces_the_rule() -> None:
    try:
        _C.decode_force_split_k(3)
        assert _C.split_k_for(0, 64, 64, HGEMM_CTA, 340, "rtx5090") == 3
        assert _C.split_k_for(0, 64, 2, HGEMM_CTA, 340, "rtx5090") == 2  # clamped
    finally:
        _C.decode_force_split_k(0)
    assert _C.split_k_for(0, 64, 64, HGEMM_CTA, 340, "rtx5090") == 1


# ---- the L2 question, pure arithmetic ----------------------------------------------------------

MB = 1 << 20


@needs_ext
@pytest.mark.parametrize(
    "nbytes, fits",
    [
        (4096 * 4096 * 2, False),  # bf16 4096 x 4096: 32 MB against 18 MB
        (4096 * 4096 // 2 + 32 * 4096 * 2, True),  # int4 with scales
        (4096 * 4096 // 8 + 32 * 4096 * 2, True),  # 1 bit with scales
        (4096 * 28672 // 8 + 32 * 28672 * 2, True),  # gate|up at 1 bit, 15.75 MB
        (4096 * 28672 // 2 + 32 * 28672 * 2, False),  # gate|up at int4, 57.75 MB
        (18 * MB, True),  # exactly three quarters of 24 MB
        (18 * MB + 1, False),
        (0, False),
    ],
)
def test_fits_l2_gb10(nbytes: int, fits: bool) -> None:
    assert _C.fits_l2(nbytes, "gb10") is fits


@needs_ext
def test_fits_l2_rtx5090() -> None:
    assert _C.fits_l2(4096 * 4096 * 2, "rtx5090") is True
    assert _C.fits_l2(72 * MB, "rtx5090") is True
    assert _C.fits_l2(72 * MB + 1, "rtx5090") is False


@needs_ext
def test_unknown_device_name_is_an_error() -> None:
    with pytest.raises((ValueError, RuntimeError)):
        _C.fits_l2(1, "h100")


# ---- the real device ----------------------------------------------------------------------------


def test_device_shape_dict(sk) -> None:
    d = sk.ops.device_shape()
    assert set(d) >= {"sm_count", "l2_bytes", "spec_gbps", "bytes_in_flight", "forced"}
    assert d["sm_count"] > 0 and d["l2_bytes"] > 0 and d["spec_gbps"] > 0
    assert d["bytes_in_flight"] > 0
    props = torch.cuda.get_device_properties(0)
    if not d["forced"]:
        assert d["sm_count"] == props.multi_processor_count
    if d["sm_count"] == 170:
        assert d["bytes_in_flight"] == RTX5090["bytes_in_flight"]
        assert d["l2_bytes"] == RTX5090["l2_bytes"]
    if d["sm_count"] == 48 and not d["forced"]:
        assert d["bytes_in_flight"] == GB10["bytes_in_flight"]
        assert d["l2_bytes"] == GB10["l2_bytes"]
        assert abs(d["spec_gbps"] - GB10["spec_gbps"]) < 10  # the attribute path, to verify


def test_current_device_rules_match_its_named_shape(sk) -> None:
    d = sk.ops.device_shape()
    name = {170: "rtx5090", 48: "gb10"}.get(d["sm_count"])
    if name is None or d["forced"]:
        pytest.skip("not one of the two named shapes")
    for strips in (4, 16, 64, 448):
        assert sk.ops.split_k_for(0, strips, 64, HGEMM_CTA, 340, "") == sk.ops.split_k_for(
            0, strips, 64, HGEMM_CTA, 340, name)
    assert sk.ops.fits_l2(4096 * 4096 * 2, "") == sk.ops.fits_l2(4096 * 4096 * 2, name)


@pytest.mark.parametrize("split", [1, 4, 8])
def test_hgemm_decode_under_a_forced_split(sk, split: int) -> None:
    # The narrow shape whose split the rule changes: 16 strips of 64. Every forced split
    # must give cuBLAS's answer, including the atomics path.
    torch.manual_seed(0)
    a = torch.randn(16, 4096, device="cuda", dtype=torch.bfloat16)
    b = (torch.randn(4096, 1024, device="cuda") / 64).to(torch.bfloat16)
    ref = a.float() @ b.float()
    try:
        sk.ops.decode_force_split_k(split)
        out = sk.hgemm(a, b, variant=6)
    finally:
        sk.ops.decode_force_split_k(0)
    torch.testing.assert_close(out.float(), ref, atol=3e-2, rtol=3e-2)
