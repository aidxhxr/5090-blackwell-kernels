"""CPU-only checks on scripts/bench_torch.py: it must time the shapes the C++ benches time,
because the results table joins the two on (kernel, dtype, shape)."""

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import bench_torch  # noqa: E402


def ints_after(bench: str, pattern: str) -> list[int]:
    """Integers of the first brace list that follows `pattern` in src/bench/<bench>.cu."""
    src = (ROOT / "src" / "bench" / f"{bench}.cu").read_text()
    m = re.search(pattern + r"\s*(\{[^;]*?\})\s*[;)]", src)
    assert m, f"default shape list not found in {bench}.cu; update this test and bench_torch.py"
    return [int(t) for t in re.findall(r"\d+", m.group(1))]


def test_row_kernel_widths_match_the_cpp_defaults():
    flat = ints_after("bench_rmsnorm", r"shapes =")
    assert [tuple(flat[i:i + 2]) for i in range(0, len(flat), 2)] == bench_torch.RMSNORM_SHAPES
    assert ints_after("bench_softmax", r"cols_list =") == bench_torch.SOFTMAX_COLS
    assert ints_after("bench_swiglu", r"int cols :") == bench_torch.SWIGLU_COLS


def test_gemm_shapes_match_the_cpp_defaults():
    for bench, shapes in (("bench_sgemm", bench_torch.SGEMM_SHAPES),
                          ("bench_hgemm", bench_torch.HGEMM_SHAPES),
                          ("bench_fp8gemm", bench_torch.FP8GEMM_SHAPES)):
        flat = ints_after(bench, r"shapes =")
        assert [tuple(flat[i:i + 3]) for i in range(0, len(flat), 3)] == shapes


def test_clock_ramp_spins_once_for_the_whole_process(monkeypatch):
    monkeypatch.setattr(bench_torch.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(bench_torch, "RAMP_MS", 20)
    monkeypatch.setattr(bench_torch, "_ramped", False)
    calls = []
    t0 = time.perf_counter()
    bench_torch.ramp_clocks(lambda: calls.append(1))
    assert (time.perf_counter() - t0) * 1e3 >= 20 and calls
    n = len(calls)
    bench_torch.ramp_clocks(lambda: calls.append(1))  # later ops are not ramped again
    assert len(calls) == n


def test_attention_shapes_match_the_cpp_defaults():
    # (B, H_q, H_kv, S_q, S_kv, D, causal) in both files
    flat = ints_after("bench_attention", r"shapes =")
    assert len(flat) % 7 == 0
    cpp = [tuple(flat[i:i + 7]) for i in range(0, len(flat), 7)]
    # bench_torch times the large shapes only; every one of them must be in the C++ list
    for shape in bench_torch.ATTENTION_SHAPES:
        assert shape in cpp, f"{shape} is timed by bench_torch.py but not by bench_attention.cu"
