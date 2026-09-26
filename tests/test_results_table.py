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
