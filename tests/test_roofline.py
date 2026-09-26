"""CPU-only tests for the point placement in scripts/roofline.py (matplotlib is not needed)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import roofline  # noqa: E402


def test_bandwidth_copy_is_plotted_on_the_dram_roof_scale():
    # 1 GiB of f32 copied in 10 ms: 2 passes (read + write) -> 2 GiB of traffic
    r = {"kernel": "bandwidth", "dtype": "f32", "shape": "n=268435456", "median_ms": 10.0}
    assert roofline.plot_intensity(r) == 1.0  # not 0, which the plot would drop
    # at 1 FLOP/byte the height is bytes/s, so it compares directly against the bandwidth roof
    assert roofline.achieved_tflops(r) * 1e12 == pytest.approx(2 * 2**30 / 10e-3)
    # same point from the abbreviated shape the bench writes
    assert roofline.achieved_tflops({**r, "shape": "n=256M"}) == roofline.achieved_tflops(r)


def test_compute_kernels_use_their_arithmetic_intensity():
    r = {"kernel": "sgemm", "dtype": "f32", "shape": "M1024_N1024_K1024", "median_ms": 5.0}
    assert roofline.plot_intensity(r) == pytest.approx(2 * 1024 / 12)
    assert roofline.achieved_tflops(r) == pytest.approx(2 * 1024**3 / 5e-3 / 1e12)


def test_fp8gemm_intensity_counts_one_byte_operands():
    # 16 x 4096 x 4096: 2 M N K FLOP over M K + K N bytes of e4m3 plus 2 M N bytes of bf16 out
    r = {"kernel": "fp8gemm", "dtype": "e4m3", "shape": "16x4096x4096", "median_ms": 0.0133}
    fl = 2 * 16 * 4096 * 4096
    by = 16 * 4096 + 4096 * 4096 + 2 * 16 * 4096
    assert roofline.plot_intensity(r) == pytest.approx(fl / by)
    assert roofline.achieved_tflops(r) == pytest.approx(fl / 0.0133e-3 / 1e12)
    # the same shape in bf16 has half the intensity: twice the bytes for the same FLOPs
    h = {**r, "kernel": "hgemm", "dtype": "bf16"}
    assert roofline.plot_intensity(h) == pytest.approx(fl / (2 * (16 * 4096 + 4096 * 4096 +
                                                                 16 * 4096)))
    assert roofline.MARKERS["fp8gemm"] != roofline.MARKERS["hgemm"]


def test_zero_time_rows_do_not_divide_by_zero():
    r = {"kernel": "rmsnorm", "dtype": "bf16", "shape": "4096x8192", "median_ms": 0.0}
    assert roofline.achieved_tflops(r) == 0.0
