"""CPU-only tests for scripts/shape_utils.py (the math behind docs/RESULTS.md and the roofline)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import shape_utils as su  # noqa: E402


@pytest.mark.parametrize(
    "kernel,shape,expected",
    [
        ("sgemm", "M4096_N4096_K11008", {"M": 4096, "N": 4096, "K": 11008}),
        ("hgemm", "4096", {"M": 4096, "N": 4096, "K": 4096}),
        ("fp8gemm", "16x4096x4096", {"M": 16, "N": 4096, "K": 4096}),
        ("rmsnorm", "4096x8192", {"rows": 4096, "cols": 8192}),
        ("softmax", "rows16_cols4096", {"rows": 16, "cols": 4096}),
        ("swiglu", "14336", {"rows": 1, "cols": 14336}),
        ("bandwidth", "n=268435456", {"n": 268435456}),
        ("bandwidth", "n=256M", {"n": 256 << 20}),  # what bench_bandwidth.cu writes
        ("bandwidth", "n=64M", {"n": 64 << 20}),
        ("bandwidth", "", {"n": 0}),
        ("attention", "b1_h32_s4096_d128_causal",
         {"B": 1, "H": 32, "H_kv": 32, "S_q": 4096, "S_kv": 4096, "D": 128, "causal": True}),
        ("attention", "b1_h32_sq1_skv4096_d128",
         {"B": 1, "H": 32, "H_kv": 32, "S_q": 1, "S_kv": 4096, "D": 128, "causal": False}),
        ("attention", "b1_hq32_hkv8_sq1_skv131072_d128",
         {"B": 1, "H": 32, "H_kv": 8, "S_q": 1, "S_kv": 131072, "D": 128, "causal": False}),
        ("attention", "b1_hq32_hkv8_s4096_d128_causal",
         {"B": 1, "H": 32, "H_kv": 8, "S_q": 4096, "S_kv": 4096, "D": 128, "causal": True}),
    ],
)
def test_parse_shape(kernel, shape, expected):
    assert su.parse_shape(kernel, shape) == expected


def test_attention_shape_round_trips_and_counts_flops_and_bytes():
    # bench_attention.cu: b{B}_h{H}_s{S}_d{D}[_causal], sq/skv when the lengths differ
    assert su.attention_shape(1, 32, 4096, 4096, 128, True) == "b1_h32_s4096_d128_causal"
    assert su.attention_shape(1, 32, 1, 4096, 128, False) == "b1_h32_sq1_skv4096_d128"
    dims = su.parse_shape("attention", su.attention_shape(4, 32, 2048, 2048, 128, True))
    assert dims == {"B": 4, "H": 32, "H_kv": 32, "S_q": 2048, "S_kv": 2048, "D": 128,
                    "causal": True}
    # 4 B H S_q S_kv D, halved under the causal mask; Q, K, V, O each once in bf16
    assert su.flops("attention", dims) == 2 * 4 * 32 * 2048 * 2048 * 128
    assert su.traffic_bytes("attention", "bf16", dims) == 4 * 32 * (2048 + 2048) * 128 * 2 * 2
    assert su.is_compute_bound_kernel("attention") and su.uses_tensor_cores("attention")
    with pytest.raises(ValueError):
        su.parse_shape("attention", "4096x4096")


def test_attention_gqa_shape_keeps_mha_keys_and_scales_kv_bytes_with_kv_heads():
    # an equal K/V head count keeps the old spelling, so existing result rows keep their keys
    assert su.attention_shape(1, 32, 1, 4096, 128, False, H_kv=32) == "b1_h32_sq1_skv4096_d128"
    s = su.attention_shape(1, 32, 1, 131072, 128, False, H_kv=8)
    assert s == "b1_hq32_hkv8_sq1_skv131072_d128"
    dims = su.parse_shape("attention", s)
    assert dims["H"] == 32 and dims["H_kv"] == 8
    # FLOPs follow the query heads; K and V bytes follow the K/V heads (the traffic floor
    # when the kernel reads each K/V head once for the whole group)
    assert su.flops("attention", dims) == 4 * 32 * 1 * 131072 * 128
    q_bytes = 2 * (1 * 32 * 1 * 128) * 2
    kv_bytes = 2 * (1 * 8 * 131072 * 128) * 2
    assert su.traffic_bytes("attention", "bf16", dims) == q_bytes + kv_bytes
    with pytest.raises(ValueError):
        su.parse_shape("attention", "b1_hq32_s4096_d128")  # hq without hkv


def test_shape_strings_match_the_cpp_benches_and_round_trip():
    # bench_sgemm.cu / bench_hgemm.cu: M x N x K; the row-wise benches: rows x cols
    assert su.gemm_shape(4096, 11008, 4096) == "4096x11008x4096"
    assert su.parse_shape("hgemm", su.gemm_shape(4096, 11008, 2048)) == {
        "M": 4096, "N": 11008, "K": 2048}
    assert su.row_shape(4096, 14336) == "4096x14336"
    assert su.parse_shape("swiglu", su.row_shape(4096, 14336)) == {"rows": 4096, "cols": 14336}


def test_traffic_bytes_counts_every_pass():
    dims = {"rows": 4096, "cols": 8192}
    elems = 4096 * 8192
    assert su.traffic_bytes("rmsnorm", "bf16", dims) == 2 * elems * 2
    assert su.traffic_bytes("rmsnorm", "f32", dims) == 2 * elems * 4
    assert su.traffic_bytes("swiglu", "bf16", dims) == 3 * elems * 2
    # read x, read + write resid, write out
    assert su.traffic_bytes("add_rmsnorm", "bf16", dims) == 4 * elems * 2
    assert su.traffic_bytes("bandwidth", "f32", {"n": 1024}) == 2 * 1024 * 4


def test_gemm_flops_and_intensity():
    dims = {"M": 1024, "N": 1024, "K": 1024}
    assert su.flops("sgemm", dims) == 2 * 1024**3
    # 2*n^3 FLOP over 3*n^2 elements of 4 bytes
    assert su.arithmetic_intensity("sgemm", "f32", "1024") == pytest.approx(2 * 1024 / 12)


def test_fp8gemm_counts_bytes_per_operand_and_uses_the_fp8_peak():
    dims = {"M": 16, "N": 4096, "K": 4096}
    # e4m3 operands are one byte each, the bf16 output two: half the bytes of the bf16 GEMM
    assert su.traffic_bytes("fp8gemm", "e4m3", dims) == 16 * 4096 + 4096 * 4096 + 2 * 16 * 4096
    assert su.traffic_bytes("hgemm", "bf16", dims) == 2 * (16 * 4096 + 4096 * 4096 + 16 * 4096)
    assert su.flops("fp8gemm", dims) == 2 * 16 * 4096 * 4096
    assert su.is_compute_bound_kernel("fp8gemm") and su.uses_tensor_cores("fp8gemm")
    assert su.compute_peak_key("fp8gemm") == "fp8_tflops"
    assert su.compute_peak_key("hgemm") == "bf16_tflops"
    assert su.compute_peak_key("attention") == "bf16_tflops"
    assert su.compute_peak_key("sgemm") == "fp32_tflops"
    assert su.DEVICE_PEAKS["RTX 5090"]["fp8_tflops"] is None  # measured, never guessed


def test_measured_peaks_reads_the_fp8_roof_and_not_the_documentation_rows(tmp_path):
    rows = [{"device": "NVIDIA GeForce RTX 5090", "kernel": "peak_bf16_mma", "tflops": 258.7},
            {"device": "NVIDIA GeForce RTX 5090", "kernel": "sm_clock", "ref_ms": 2976.0},
            {"device": "NVIDIA GeForce RTX 5090", "kernel": "peak_fp8_mma", "tflops": 1013.9},
            {"device": "NVIDIA GeForce RTX 5090", "kernel": "peak_fp8_mma_plain",
             "tflops": 517.4},
            {"device": "NVIDIA GeForce RTX 5090", "kernel": "peak_fp8_mma_f16acc",
             "tflops": 1026.8},
            {"device": "NVIDIA GeForce RTX 5090", "kernel": "peak_fp32_fma", "tflops": 123.4}]
    import json
    (tmp_path / su.PEAK_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    measured = su.measured_peaks(tmp_path)
    assert measured["fp8_tflops"] == 1013.9 and measured["bf16_tflops"] == 258.7
    key, peaks = su.peaks_for_rows([{"device": "NVIDIA GeForce RTX 5090"}], measured=measured)
    assert peaks["fp8_tflops"] == 1013.9 and peaks["bf16_tflops"] == 258.7
    assert peaks["sm_mhz"] == 2976.0 and peaks["measured"]


def test_ridge_points_match_the_hardware_sheets():
    gb10 = su.DEVICE_PEAKS["GB10"]
    assert gb10["bf16_tflops"] * 1e3 / gb10["bw_gbps"] == pytest.approx(780, rel=0.01)
    assert gb10["fp32_tflops"] * 1e3 / gb10["bw_gbps"] == pytest.approx(114, rel=0.01)
    rtx = su.DEVICE_PEAKS["RTX 5090"]
    assert rtx["fp32_tflops"] * 1e3 / rtx["bw_gbps"] == pytest.approx(58.5, rel=0.01)
    assert rtx["bf16_tflops"] is None  # not measured yet; never guess it


def test_elementwise_kernels_sit_left_of_the_ridge():
    rtx = su.DEVICE_PEAKS["RTX 5090"]
    ridge = rtx["fp32_tflops"] * 1e3 / rtx["bw_gbps"]  # the lower (fp32) ridge, FLOP/byte
    for kernel in ("rmsnorm", "add_rmsnorm", "softmax", "swiglu"):
        ai = su.arithmetic_intensity(kernel, "bf16", "4096x8192")
        assert 0 < ai < ridge
        assert not su.is_compute_bound_kernel(kernel)
    assert su.arithmetic_intensity("bandwidth", "f32", "1024") == 0.0


def test_load_bench_rows_skips_noise_and_torch_comparison(tmp_path):
    (tmp_path / "rmsnorm.json").write_text(
        'device: GB10\n{"kernel":"rmsnorm","variant":0}\n\n{truncated\n'
        '{"kernel":"rmsnorm","variant":1}\n'
    )
    (tmp_path / su.TORCH_COMPARISON).write_text('{"kernel":"rmsnorm","speedup":2.0}\n')
    rows = su.load_bench_rows(tmp_path)
    assert [r["variant"] for r in rows] == [0, 1]


def test_bench_entry_point_names_map_to_kernel_families(tmp_path):
    # the names bench_hgemm.cu and bench_bandwidth.cu actually write
    (tmp_path / "hgemm.json").write_text('{"kernel":"hgemm_bf16","variant":2}\n')
    (tmp_path / "bandwidth.json").write_text('{"kernel":"bandwidth_copy","variant":0}\n')
    (tmp_path / "rmsnorm.json").write_text('{"kernel":"rmsnorm","variant":1}\n')
    rows = su.load_bench_rows(tmp_path)
    assert sorted(r["kernel"] for r in rows) == ["bandwidth", "hgemm", "rmsnorm"]
    for r in rows:
        if r["kernel"] == "hgemm":
            assert su.is_compute_bound_kernel(r["kernel"])


def test_device_key_and_peaks_selection():
    assert su.device_key("NVIDIA GeForce RTX 5090") == "RTX 5090"
    assert su.device_key("NVIDIA GB10") == "GB10"
    assert su.device_key(None) == su.DEFAULT_DEVICE  # rows from before the device field
    with pytest.raises(ValueError, match="no peaks known"):
        su.device_key("NVIDIA H100")

    key, peaks = su.peaks_for_rows([{"device": "NVIDIA GeForce RTX 5090"}], bf16_peak=700.0)
    assert key == "RTX 5090" and peaks["bf16_tflops"] == 700.0
    assert su.DEVICE_PEAKS["RTX 5090"]["bf16_tflops"] is None  # override does not leak
    with pytest.raises(ValueError, match="mixes devices"):
        su.peaks_for_rows([{"device": "NVIDIA GeForce RTX 5090"}, {"device": "NVIDIA GB10"}])
