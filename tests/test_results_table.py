"""CPU-only end-to-end test of scripts/make_results_table.py on rows shaped exactly like the
ones the C++ benches print (kernel names, reference rows, shape strings)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import make_results_table as mrt  # noqa: E402

DEVICE = "NVIDIA GeForce RTX 5090"


def bench_row(kernel, dtype, variant, shape, median_ms, **kw):
    return {"device": DEVICE, "kernel": kernel, "dtype": dtype, "variant": variant,
            "shape": shape, "median_ms": median_ms, "min_ms": median_ms * 0.98, "gbps": 0.0,
            "tflops": 0.0, "ref_ms": 0.0, "max_abs_err": 0.0, "max_rel_err": 0.0, "ok": True, **kw}


@pytest.fixture
def results(tmp_path, monkeypatch):
    monkeypatch.setattr(mrt, "RESULTS", tmp_path)
    monkeypatch.setattr(mrt, "OUT_MD", tmp_path / "RESULTS.md")
    monkeypatch.setattr(mrt, "OUT_HEADLINE", tmp_path / "headline.md")
    monkeypatch.setattr(sys, "argv", ["make_results_table.py"])

    def write(name, rows):
        (tmp_path / name).write_text("".join(json.dumps(r) + "\n" for r in rows))

    return tmp_path, write


def test_cublas_row_is_a_labelled_reference_and_never_the_best_variant(results):
    out, write = results
    write("sgemm.json", [
        bench_row("sgemm_cublas", "f32", -1, "4096x4096x4096", 2.0, tflops=68.7, ref_ms=2.0),
        bench_row("sgemm", "f32", 2, "4096x4096x4096", 4.0, tflops=34.4, ref_ms=2.0),
        bench_row("sgemm", "f32", 3, "4096x4096x4096", 3.0, tflops=45.8, ref_ms=2.0),
    ])
    assert mrt.main() == 0
    md = (out / "RESULTS.md").read_text()
    assert "## sgemm_cublas" not in md  # filed under sgemm, not a kernel of its own
    assert "| f32 | 4096x4096x4096 | cuBLAS | 2.0000 |" in md
    headline = (out / "headline.md").read_text()
    assert "| sgemm | f32 | 4096x4096x4096 | v3 |" in headline  # cuBLAS is faster, still not "best"
    assert "66.7%" in headline  # 2.0 ms cuBLAS / 3.0 ms ours


def test_sgemm_tf32_rows_are_judged_against_the_tf32_peak(results):
    # bench_sgemm writes variant 6 as dtype "tf32" (reference: cuBLAS in TF32 math mode) and
    # variant 7 as "3xtf32" (reference: cuBLAS fp32). They share the sgemm table with the
    # f32 rows, so the peak column names a ceiling per dtype, the 3xTF32 one a third of the
    # tf32 peak; the headline gets one row per dtype.
    out, write = results
    write("sgemm.json", [
        bench_row("sgemm_cublas", "f32", -1, "4096x4096x4096", 2.0, tflops=68.7, ref_ms=2.0),
        bench_row("sgemm_cublas_tf32", "tf32", -1, "4096x4096x4096", 1.3, tflops=105.7,
                  ref_ms=1.3),
        bench_row("sgemm", "f32", 5, "4096x4096x4096", 2.2, tflops=62.5, ref_ms=2.0),
        bench_row("sgemm", "tf32", 6, "4096x4096x4096", 1.3, tflops=105.7, ref_ms=1.3),
        bench_row("sgemm", "3xtf32", 7, "4096x4096x4096", 3.5, tflops=39.3, ref_ms=2.0),
    ])
    write("peak.json", [
        {"device": DEVICE, "kernel": "peak_tf32_mma", "tflops": 129.4},
        {"device": DEVICE, "kernel": "peak_fp32_fma", "tflops": 123.4},
    ])
    assert mrt.main() == 0
    md = (out / "RESULTS.md").read_text()
    sgemm = md.split("## sgemm")[1]
    assert "% of peak (3xtf32 43.1, f32 123.4, tf32 129.4 TFLOPS)" in sgemm
    assert "| tf32 | 4096x4096x4096 | cuBLAS TF32 | 1.3000 |" in sgemm
    assert "| tf32 | 4096x4096x4096 | 6 | 1.3000 | 1.2740 | 105.70 | 81.7% | 100.0% |" in sgemm
    assert "| 3xtf32 | 4096x4096x4096 | 7 | 3.5000 | 3.4300 | 39.30 | 91.1% | 57.1% |" in sgemm
    assert "| f32 | 4096x4096x4096 | 5 | 2.2000 | 2.1560 | 62.50 | 50.6% | 90.9% |" in sgemm
    headline = (out / "headline.md").read_text()
    assert "| sgemm | 3xtf32 | 4096x4096x4096 | v7 | 3.5000 | 39.3 TFLOPS | 91.1% |" in headline
    assert "| sgemm | f32 | 4096x4096x4096 | v5 |" in headline
    assert "| sgemm | tf32 | 4096x4096x4096 | v6 | 1.3000 | 105.7 TFLOPS | 81.7% |" in headline
    assert "129.4 TFLOPS tf32 tensor" in md


def test_hgemm_rows_are_reported_in_tflops(results):
    out, write = results
    write("hgemm.json", [
        bench_row("hgemm_bf16", "bf16", 2, "4096x4096x4096", 0.5, tflops=274.9, ref_ms=0.4),
    ])
    assert mrt.main() == 0
    md = (out / "RESULTS.md").read_text()
    assert "## hgemm\n" in md and "TFLOPS" in md and "% of cuBLAS" in md
    assert "| hgemm | bf16 | 4096x4096x4096 | v2 | 0.5000 | 274.9 TFLOPS | — | 80.0% |" in (
        out / "headline.md").read_text()


def test_fp8gemm_rows_use_the_fp8_peak_and_the_cublaslt_label(results):
    out, write = results
    write("fp8gemm.json", [
        bench_row("fp8gemm", "e4m3", 1, "4096x4096x4096", 0.25, tflops=549.8, ref_ms=0.2),
        bench_row("fp8gemm", "e4m3", 2, "4096x4096x4096", 0.2, tflops=687.2, ref_ms=0.2),
    ])
    write("hgemm.json", [
        bench_row("hgemm_bf16", "bf16", 5, "4096x4096x4096", 0.56, tflops=245.4, ref_ms=0.6),
    ])
    write("peak.json", [
        {"device": DEVICE, "kernel": "peak_bf16_mma", "tflops": 258.7},
        {"device": DEVICE, "kernel": "peak_fp8_mma", "tflops": 1013.9},
        {"device": DEVICE, "kernel": "peak_fp8_mma_plain", "tflops": 517.4},
    ])
    assert mrt.main() == 0
    md = (out / "RESULTS.md").read_text()
    fp8 = md.split("## fp8gemm")[1]
    assert "% of peak (1013.9 TFLOPS)" in fp8 and "% of cuBLASLt" in fp8
    assert "| e4m3 | 4096x4096x4096 | 2 | 0.2000 | 0.1960 | 687.20 | 67.8% | 100.0% |" in fp8
    hgemm = md.split("## hgemm")[1].split("## fp8gemm")[0]
    assert "% of peak (258.7 TFLOPS)" in hgemm and "% of cuBLAS |" in hgemm
    assert "1013.9 TFLOPS fp8 tensor" in md
    headline = (out / "headline.md").read_text()
    assert "| fp8gemm | e4m3 | 4096x4096x4096 | v2 | 0.2000 | 687.2 TFLOPS | 67.8% | 100.0% |" in (
        headline)
    assert headline.index("| hgemm |") < headline.index("| fp8gemm |")  # KERNEL_ORDER


def test_fp4gemm_rows_use_the_fp4_peak_and_mxfp4_has_no_reference(results):
    out, write = results
    write("fp4gemm.json", [
        bench_row("fp4gemm", "nvfp4", 2, "8192x8192x8192", 0.8, tflops=1374.4, ref_ms=0.76),
        bench_row("fp4gemm", "mxfp4", 2, "8192x8192x8192", 0.78, tflops=1409.6),
        bench_row("fp4quant", "nvfp4", 0, "4096x14336", 0.077, gbps=1955.2),
    ])
    write("peak.json", [
        {"device": DEVICE, "kernel": "peak_bf16_mma", "tflops": 258.7},
        {"device": DEVICE, "kernel": "peak_fp4_mma", "tflops": 2028.7},
    ])
    assert mrt.main() == 0
    md = (out / "RESULTS.md").read_text()
    fp4 = md.split("## fp4gemm")[1].split("## fp4quant")[0]
    assert "% of peak (2028.7 TFLOPS)" in fp4 and "% of cuBLASLt" in fp4
    assert "| nvfp4 | 8192x8192x8192 | 2 | 0.8000 | 0.7840 | 1374.40 | 67.7% | 95.0% |" in fp4
    assert "| mxfp4 | 8192x8192x8192 | 2 | 0.7800 | 0.7644 | 1409.60 | 69.5% | — |" in fp4
    assert "2028.7 TFLOPS fp4 tensor" in md
    quant = md.split("## fp4quant")[1]
    assert "GB/s" in quant and "| nvfp4 | 4096x14336 | 0 |" in quant


def test_torch_rows_from_another_machine_are_ignored(results, capsys):
    out, write = results
    shape = "4096x8192"
    write("rmsnorm.json", [bench_row("rmsnorm", "bf16", 3, shape, 0.1, gbps=1300.0)])
    torch_row = {"kernel": "rmsnorm", "dtype": "bf16", "shape": shape, "speedup": 3.21}

    write("torch_comparison.json", [{**torch_row, "device": "NVIDIA GB10"}])
    assert mrt.main() == 0
    assert "3.21×" not in (out / "RESULTS.md").read_text()
    assert "not measured on the RTX 5090" in capsys.readouterr().err

    write("torch_comparison.json", [{**torch_row, "device": DEVICE}])
    assert mrt.main() == 0
    assert "3.21×" in (out / "RESULTS.md").read_text()
    assert "| 3.21× |" in (out / "headline.md").read_text()


def test_the_copy_is_not_given_an_arithmetic_intensity(results):
    out, write = results
    write("bandwidth.json", [
        bench_row("cudaMemcpy_d2d", "f32", -1, "n=256M", 1.5, gbps=1431.0, ref_ms=1.5),
        bench_row("bandwidth_copy", "f32", 2, "n=256M", 1.6, gbps=1342.0, ref_ms=1.5),
    ])
    assert mrt.main() == 0
    md = (out / "RESULTS.md").read_text()
    assert "0.00 FLOP/byte" not in md and "a pure copy" in md
    assert "| f32 | n=256M | cudaMemcpy | 1.5000 |" in md
    assert "| bandwidth | f32 | n=256M | v2 |" in (out / "headline.md").read_text()


def test_attention_rows_use_the_tensor_peak_and_the_torch_backend_column(results):
    out, write = results
    shape = "b1_h32_s4096_d128_causal"
    write("attention.json", [
        bench_row("attention", "bf16", 2, shape, 0.65, tflops=211.7),
        bench_row("attention", "bf16", 3, shape, 0.64, tflops=214.5),
        bench_row("attention", "bf16", 3, "b1_h32_sq1_skv4096_d128", 0.05, gbps=1367.0,
                  tflops=1.4),
    ])
    write("peak.json", [{"device": DEVICE, "kernel": "peak_bf16_mma", "tflops": 258.7}])
    write("torch_comparison.json", [{"device": DEVICE, "kernel": "attention", "dtype": "bf16",
                                     "shape": shape, "speedup": 1.22, "torch_backend": "flash"}])
    assert mrt.main() == 0
    md = (out / "RESULTS.md").read_text()
    assert "## attention\n" in md and "% of peak (258.7 TFLOPS)" in md
    assert "% of cuBLAS" not in md.split("## attention")[1]  # no library reference in the bench
    row = "| bf16 | b1_h32_s4096_d128_causal | 3 | 0.6400 | 0.6272 | 214.50 | 82.9% | 1.22× |"
    assert row in md
    headline = (out / "headline.md").read_text()
    assert "| attention | bf16 | b1_h32_s4096_d128_causal | v3 |" in headline  # the largest shape


def test_attention_fp8_rows_use_the_fp8_peak_and_the_bf16_kernel_as_reference(results):
    # bench_attention_fp8 times attention variant 5 (bf16) on the same shape as ref_ms
    out, write = results
    shape = "b1_h32_s4096_d128_causal"
    write("attention_fp8.json", [
        bench_row("attention_fp8", "e4m3", 0, shape, 12.2, tflops=11.3, ref_ms=0.58),
        bench_row("attention_fp8", "e4m3", 1, shape, 0.232, tflops=592.4, ref_ms=0.58),
    ])
    write("peak.json", [{"device": DEVICE, "kernel": "peak_bf16_mma", "tflops": 258.7},
                        {"device": DEVICE, "kernel": "peak_fp8_mma", "tflops": 1014.0}])
    assert mrt.main() == 0
    md = (out / "RESULTS.md").read_text()
    section = md.split("## attention_fp8\n")[1]
    assert "% of peak (1014 TFLOPS)" in section and "% of bf16 attention v5" in section
    assert "| e4m3 | b1_h32_s4096_d128_causal | 1 | 0.2320 | 0.2274 | 592.40 | 58.4% | 250.0% |" \
        in section
    headline = (out / "headline.md").read_text()
    assert "| attention_fp8 | e4m3 | b1_h32_s4096_d128_causal | v1 |" in headline


def test_layer_rows_get_their_own_section_and_stay_out_of_the_kernel_tables(results):
    out, write = results
    write("hgemm.json", [
        bench_row("hgemm_bf16", "bf16", 6, "4096x4096x4096", 0.55, tflops=250.0, ref_ms=0.6),
    ])
    write("layer.json", [
        {"device": DEVICE, "kernel": "layer", "dtype": "bf16", "mode": "prefill", "B": 1,
         "S": 4096, "shape": "prefill_b1_s4096", "spark_ms": 4.0, "torch_ms": 5.0,
         "torch_compiled_ms": 4.5, "tokens_per_s_32_layers": 32000.0},
        {"device": DEVICE, "kernel": "layer", "dtype": "bf16", "mode": "decode", "B": 1,
         "L": 4096, "shape": "decode_b1_L4096", "spark_ms": 0.5, "spark_graph_ms": 0.25,
         "torch_ms": 0.6, "torch_compiled_ms": 0.3, "tokens_per_s_32_layers": 62.5,
         "tokens_per_s_32_layers_graph": 125.0,
         "breakdown_us": {"qkv_gemm": 40.0, "attention": 17.0, "rope": 5.0, "other": 1.0}},
        {"device": "NVIDIA GB10", "kernel": "layer", "dtype": "bf16", "mode": "decode", "B": 1,
         "L": 16384, "shape": "decode_b1_L16384", "spark_ms": 9.0, "torch_ms": 9.0},
    ])
    assert mrt.main() == 0
    md = (out / "RESULTS.md").read_text()
    assert "## layer" in md and md.index("## hgemm") < md.index("## layer")
    assert "decode_b1_L16384" not in md  # measured on the other machine
    assert "| prefill_b1_s4096 | 4.000 | 5.000 | 4.500 | 1.25× | 1.12× | 32,000 |" in md
    assert "| decode_b1_L4096 | 0.500 | 0.250 | 0.600 | 0.300 | 2.00× | 1.20× | 125 |" in md
    assert "| qkv GEMM | 40.0 |" in md and "| attention | 17.0 |" in md
    assert "| total | 63.0 |" in md
    assert "| norm 1 |" not in md  # stages without kernels are left out
    hgemm = md.split("## hgemm")[1].split("## layer")[0]
    assert "prefill_b1_s4096" not in hgemm  # layer rows are not bench rows
    assert "layer" not in (out / "headline.md").read_text()


def test_results_without_a_layer_file_have_no_layer_section(results):
    out, write = results
    write("rmsnorm.json", [bench_row("rmsnorm", "bf16", 4, "4096x8192", 0.1, gbps=1300.0)])
    assert mrt.main() == 0
    assert "## layer" not in (out / "RESULTS.md").read_text()


def test_rope_rows_are_memory_bound_and_the_largest_shape_is_the_headline(results):
    out, write = results
    write("rope.json", [
        bench_row("rope", "bf16", 0, "b1_s1_hq32_hkv8_d128_pos4095", 0.003, gbps=8.0),
        bench_row("rope", "bf16", 0, "b1_s8192_hq32_hkv8_d128_pos0", 0.138, gbps=1458.0),
    ])
    assert mrt.main() == 0
    md = (out / "RESULTS.md").read_text()
    assert "## rope" in md and "GB/s" in md.split("## rope")[1]
    assert "| bf16 | b1_s8192_hq32_hkv8_d128_pos0 | 0 | 0.1380 |" in md
    headline = (out / "headline.md").read_text()
    assert "| rope | bf16 | b1_s8192_hq32_hkv8_d128_pos0 | v0 |" in headline


def test_w4gemm_rows_are_memory_bound_with_the_bf16_speedup_and_a_decode_headline(results):
    out, write = results
    write("w4gemm.json", [
        bench_row("w4gemm", "w4a16", 2, "1x28672x4096", 0.04, gbps=1500.0, ref_ms=0.15),
        bench_row("w4gemm", "w4a16", 2, "256x28672x4096", 0.3, gbps=240.0, ref_ms=0.3),
        bench_row("w4gemm", "w4a16", 2, "1x4096x4096", 0.009, gbps=950.0, ref_ms=0.0235),
    ])
    assert mrt.main() == 0
    md = (out / "RESULTS.md").read_text()
    w4 = md.split("## w4gemm")[1]
    assert "GB/s" in w4 and "speedup vs bf16 hgemm" in w4
    assert "| w4a16 | 1x28672x4096 | 2 | 0.0400 |" in w4 and "3.75×" in w4
    headline = (out / "headline.md").read_text()
    assert "| w4gemm | w4a16 | 1x28672x4096 | v2 |" in headline
