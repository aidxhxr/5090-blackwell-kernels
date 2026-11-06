"""CPU-only tests of scripts/check_regress.py on rows shaped like the ones the C++ benches and
the bench scripts write: matching, the direction of each metric, the threshold, rows missing
on either side, JSON documents, --update and --rev."""

import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_regress as cr  # noqa: E402

DEVICE = "NVIDIA GeForce RTX 5090"


def bench_row(kernel, dtype, variant, shape, median_ms, ref_ms=0.0, **kw):
    """A row as bench_common.hpp's print_row writes it."""
    return {"device": DEVICE, "kernel": kernel, "dtype": dtype, "variant": variant,
            "shape": shape, "median_ms": median_ms, "min_ms": median_ms * 0.98,
            "gbps": 1.0 / median_ms, "tflops": 2.0 / median_ms, "ref_ms": ref_ms,
            "max_abs_err": 1e-3, "max_rel_err": 1e-3, "ok": True, **kw}


def rms(variant, ms, **kw):
    return bench_row("rmsnorm", "bf16", variant, "4096x4096", ms, **kw)


COMMITTED = [rms(0, 0.2), rms(3, 0.05), rms(4, 0.04)]


def write(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    return path


@pytest.fixture
def check(tmp_path, capsys):
    """check(fresh_rows, committed_rows, *flags) -> (exit code, stdout + stderr): the fresh
    rows in tmp/run/fresh.json, the committed ones in tmp/results/rmsnorm.json."""

    def go(fresh, committed, *flags, name="fresh.json"):
        against = write(tmp_path / "results" / "rmsnorm.json", committed)
        run = write(tmp_path / "run" / name, fresh)
        code = cr.main([str(run), "--against", str(against), *flags])
        out = capsys.readouterr()
        return code, out.out + out.err

    return go


def test_an_identical_run_passes(check):
    code, out = check(COMMITTED, COMMITTED)
    assert code == 0
    assert "3 rows matched, 3 metrics compared: 0 slower, 0 faster" in out


def test_a_slower_median_fails_the_run(check):
    code, out = check([rms(0, 0.2), rms(3, 0.05), rms(4, 0.044)], COMMITTED)
    assert code == 1
    slower = out.split("slower by more than 5%:")[1]
    assert "rmsnorm bf16 v4 4096x4096" in slower and "median_ms" in slower
    assert "+10.0%" in slower
    assert "1 slower, 0 faster" in out


def test_a_faster_row_is_listed_and_passes(check):
    code, out = check([rms(0, 0.2), rms(3, 0.05), rms(4, 0.036)], COMMITTED)
    assert code == 0
    assert "-10.0%" in out.split("faster by more than 5%:")[1]


def test_the_threshold(check):
    fresh = [rms(0, 0.2), rms(3, 0.05), rms(4, 0.0416)]  # 4% slower
    assert check(fresh, COMMITTED)[0] == 0
    assert check(fresh, COMMITTED, "--threshold", "3")[0] == 1
    fresh = [rms(0, 0.2), rms(3, 0.05), rms(4, 0.0424)]  # 6% slower
    assert check(fresh, COMMITTED)[0] == 1
    assert check(fresh, COMMITTED, "--threshold", "10")[0] == 0


def test_a_rate_is_higher_is_better_and_judged_as_the_time_it_implies(check):
    # an engine row of scripts/bench_invariant.py: prefill is only a rate
    row = {"kind": "engine", "mode": "default", "batch": 8, "decode_ms_per_step": 10.0,
           "decode_tok_s": 800.0, "prefill_tok_s": 1000.0}
    code, out = check([dict(row, prefill_tok_s=900.0)], [row])
    assert code == 1
    assert "prefill_tok_s" in out and "+11.1%" in out  # 1000 / 900: 11.1% more time
    code, out = check([dict(row, prefill_tok_s=1100.0)], [row])
    assert code == 0
    assert "-9.1%" in out.split("faster by more than 5%:")[1]
    # a time is lower is better
    code, out = check([dict(row, decode_ms_per_step=11.0, decode_tok_s=727.3)], [row])
    assert code == 1
    assert "decode_ms_per_step" in out and "+10.0%" in out


def test_which_metrics_are_gated():
    # the C++ benches: gbps and tflops restate median_ms, min_ms and ref_ms are not gated
    assert cr.gated_metrics(rms(4, 0.04)) == ["median_ms"]
    # scripts/bench_invariant.py: two of our rungs per row, each with its own rate
    hgemm = {"kind": "hgemm", "gemm": "qkv", "M": 1, "N": 6144, "K": 4096, "v6_ms": 0.0316,
             "v7_ms": 0.0316, "v7_over_v6": 1.0, "v6_tflops": 1.6, "v7_tflops": 1.6}
    assert cr.gated_metrics(hgemm) == ["v6_ms", "v7_ms"]
    # scripts/bench_attention_window.py: flex_attention and SDPA are references
    window = {"kernel": "attention_window", "shape": "b1_hq32_hkv8_s4096_d128_causal",
              "window": 128, "sinks": False, "ours_ms": 0.067, "ours_full_ms": 0.58,
              "flex_ms": 0.137, "sdpa_mask_ms": 3.7, "tflops": 125.7, "flex_tflops": 61.6}
    assert cr.gated_metrics(window) == ["ours_ms", "ours_full_ms"]
    # scripts/bench_cross_entropy.py: the first call compiles
    ce = {"kernel": "cross_entropy", "method": "v2", "mode": "fwd", "V": 128256, "T": 4096,
          "ms": 30.0, "first_call_s": 2.0, "tflops_2tvd": 140.0, "peak_extra_gb": 1.2}
    assert cr.gated_metrics(ce) == ["ms"]
    moe = {"kernel": "moe_layer", "impl": "ours", "tokens": 1, "median_ms": 0.44,
           "weight_gbps": 1600.0, "tflops": 1.6}
    assert cr.gated_metrics(moe) == ["median_ms"]


def test_a_tflops_change_alone_does_not_flag(check):
    fresh = [dict(r, tflops=r["tflops"] * 0.5) for r in COMMITTED]
    assert check(fresh, COMMITTED)[0] == 0


def test_measurement_fields_are_not_part_of_the_key(check):
    # a rerun with --iters=20, other errors, another torch and another GPU state
    fresh = [dict(r, iters=20, max_abs_err=0.5, min_ms=1.0, torch="2.15.0",
                  state_after="100 %, 2805 MHz") for r in COMMITTED]
    code, out = check(fresh, COMMITTED)
    assert code == 0 and "3 rows matched" in out
    # a different variant is a different row
    code, out = check([rms(5, 0.04)], COMMITTED)
    assert "0 rows matched" in out and "rmsnorm bf16 v5 4096x4096" in out


def test_missing_rows_on_either_side(check):
    softmax = bench_row("softmax", "bf16", 3, "4096x4096", 0.05)
    fresh = [rms(3, 0.05), rms(4, 0.04), rms(5, 0.03)]
    code, out = check(fresh, [*COMMITTED, softmax])
    assert code == 0  # partial runs are normal
    not_rerun = out.split("committed, not in this run: 1")[1].split("in this run, not committed")[0]
    assert "rmsnorm bf16 v0 4096x4096" in not_rerun
    assert "softmax" not in out  # a kernel the run did not touch is not missing
    assert "in this run, not committed: 1\n  rmsnorm bf16 v5 4096x4096" in out
    assert check(fresh, [*COMMITTED, softmax], "--fail-on-missing")[0] == 1


def test_a_column_added_since_the_commit_still_matches(check):
    old = {"kernel": "moe_layer", "impl": "ours", "tokens": 1, "median_ms": 0.44}
    code, out = check([dict(old, gather=True, median_ms=0.5)], [old])
    assert code == 1 and "1 row matched" in out
    # two fresh rows could be the committed one: neither is taken
    code, out = check([dict(old, gather=True), dict(old, gather=False)], [old])
    assert code == 0 and "0 rows matched" in out
    assert "in this run, not committed: 2" in out and "committed, not in this run: 1" in out


def test_the_ref_column_says_whether_the_library_moved_too(check):
    committed = [bench_row("hgemm_models", "bf16", 6, "128x4096x4096@v6", 0.030, ref_ms=0.033)]
    fresh = [bench_row("hgemm_models", "bf16", 6, "128x4096x4096@v6", 0.033, ref_ms=0.0363)]
    code, out = check(fresh, committed)
    assert code == 1
    line = next(x for x in out.splitlines() if "128x4096x4096@v6" in x)
    assert line.split()[-2:] == ["+10.0%", "+10.0%"]


def test_a_row_that_failed_its_check_fails_the_run(check):
    code, out = check([*COMMITTED[:2], dict(COMMITTED[2], ok=False)], COMMITTED)
    assert code == 1
    assert "failed the check in this run: 1\n  rmsnorm bf16 v4 4096x4096" in out


def test_the_stderr_table_in_a_captured_run_is_skipped(tmp_path, capsys):
    against = write(tmp_path / "results" / "rmsnorm.json", COMMITTED)
    run = tmp_path / "run.txt"
    run.write_text("device: NVIDIA GeForce RTX 5090 | sm_120 | 170 SMs\n"
                   "kernel  dtype var shape  median_ms\n"
                   + "".join(json.dumps(r) + "\n" for r in COMMITTED) + "{not json\n")
    assert cr.main([str(run), "--against", str(against)]) == 0
    assert "3 rows matched" in capsys.readouterr().out


def test_floor_us_ignores_changes_under_the_timer_noise(check):
    committed = [rms(4, 0.005)]  # 5 us
    fresh = [rms(4, 0.0055)]  # 0.5 us, 10%
    assert check(fresh, committed)[0] == 1
    code, out = check(fresh, committed, "--floor-us", "1")
    assert code == 0 and "within 5% or under 1 us (1 of them past 5%)" in out


def test_metric_picks_the_field_and_needs_a_direction(check):
    fresh = [dict(r, min_ms=r["min_ms"] * 1.2) for r in COMMITTED]
    assert check(fresh, COMMITTED)[0] == 0  # min_ms is not gated
    code, out = check(fresh, COMMITTED, "--metric", "min_ms")
    assert code == 1 and "min_ms" in out
    with pytest.raises(SystemExit):
        check(fresh, COMMITTED, "--metric", "v7_over_v6")


def test_another_device_is_said(check):
    code, out = check([dict(r, device="NVIDIA GB10") for r in COMMITTED], COMMITTED)
    assert "the run is from NVIDIA GB10, the committed rows from NVIDIA GeForce RTX 5090" in out
    assert "0 rows matched" in out


QUANT = {
    "bf16": {"format": "bf16", "head_format": "bf16", "quantize_s": 1e-5,
             "decode_b1_ms_per_step": 9.8, "decode_b1_tok_s": 102.0,
             "profile_decode_b1_us": {"gemm (hgemm)": 9300.0, "norms": 217.0},
             "ppl": [{"label": "bf16", "ppl": 8.29, "ctx": 2048}]},
    "int4": {"format": "int4", "head_format": "bf16", "quantize_s": 0.06,
             "decode_b1_ms_per_step": 4.1, "decode_b1_tok_s": 244.0,
             "ppl": [{"label": "int4", "ppl": 8.6, "ctx": 2048}]},
}


def test_a_json_document_is_read_as_its_timed_records(tmp_path, capsys):
    rows = cr.parse(json.dumps(QUANT, indent=1))
    assert sorted(r.data["_path"] for r in rows) == ["bf16", "int4"]  # the ppl lists are untimed
    assert cr.gated_metrics(rows[0].data) == ["decode_b1_ms_per_step"]
    serve = cr.parse(json.dumps({"backends": {"spark": {"max_batch": 256}}, "rows": [
        {"backend": "spark", "kind": "static", "batch": 1, "ttft_ms": 33.0, "config": {}}]}))
    assert [r.label for r in serve] == ["rows static spark batch=1"]

    fresh = json.loads(json.dumps(QUANT))
    fresh["int4"]["decode_b1_ms_per_step"] = 4.6
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "llm_quant.json").write_text(json.dumps(QUANT, indent=1))
    (tmp_path / "llm_quant.json").write_text(json.dumps(fresh, indent=1))
    code = cr.main([str(tmp_path / "llm_quant.json"), "--against", str(tmp_path / "results")])
    out = capsys.readouterr().out
    assert code == 1
    assert "int4 format=int4 head_format=bf16  decode_b1_ms_per_step" in out


def test_the_committed_file_is_found_by_name_then_by_rows(tmp_path, capsys):
    results = tmp_path / "results"
    write(results / "rmsnorm.json", COMMITTED)
    write(results / "softmax.json", [bench_row("softmax", "bf16", 3, "4096x4096", 0.05)])
    write(tmp_path / "run" / "rmsnorm.json", COMMITTED)
    assert cr.main([str(tmp_path / "run" / "rmsnorm.json"), "--against", str(results)]) == 0
    assert "against " + str(results / "rmsnorm.json") in capsys.readouterr().out
    write(tmp_path / "run" / "today.json", COMMITTED[1:])
    assert cr.main([str(tmp_path / "run" / "today.json"), "--against", str(results)]) == 0
    assert "against " + str(results / "rmsnorm.json") in capsys.readouterr().out
    # decode_spark.json and decode_spark_force48.json hold the same keys: say so, check nothing
    write(results / "rmsnorm_force48.json", COMMITTED)
    assert cr.main([str(tmp_path / "run" / "today.json"), "--against", str(results)]) == 2
    assert "pass --against" in capsys.readouterr().err


def test_stdin(tmp_path, capsys, monkeypatch):
    against = write(tmp_path / "rmsnorm.json", COMMITTED)
    fresh = [rms(0, 0.2), rms(3, 0.06), rms(4, 0.04)]
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(json.dumps(r) + "\n" for r in fresh)))
    assert cr.main(["-", "--against", str(against)]) == 1
    assert capsys.readouterr().out.startswith("stdin against ")


def test_update_merges_the_run_into_the_committed_file(tmp_path, capsys):
    softmax = bench_row("softmax", "bf16", 3, "4096x4096", 0.05)
    against = write(tmp_path / "results" / "rmsnorm.json", [*COMMITTED, softmax])
    old = against.read_text().splitlines()
    fresh = [rms(4, 0.044), rms(5, 0.03)]
    run = write(tmp_path / "run.json", fresh)
    new_lines = run.read_text().splitlines()
    assert cr.main([str(run), "--against", str(against), "--update"]) == 0
    assert "1 row replaced, 1 added, 3 kept" in capsys.readouterr().out
    # in place, in file order: v0, v3, the new v4, softmax, then the new v5
    assert against.read_text().splitlines() == [old[0], old[1], new_lines[0], old[3],
                                                new_lines[1]]


def test_update_refuses_a_document_and_failed_rows(tmp_path, capsys):
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "llm_quant.json").write_text(json.dumps(QUANT, indent=1))
    (tmp_path / "llm_quant.json").write_text(json.dumps(QUANT, indent=1))
    cr.main([str(tmp_path / "llm_quant.json"), "--against", str(tmp_path / "results"),
             "--update"])
    assert "not updating" in capsys.readouterr().out
    against = write(tmp_path / "rmsnorm.json", COMMITTED)
    before = against.read_text()
    run = write(tmp_path / "run.json", [dict(COMMITTED[0], ok=False)])
    assert cr.main([str(run), "--against", str(against), "--update"]) == 1
    assert "1 row failed the check" in capsys.readouterr().out
    assert against.read_text() == before


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
def test_rev_checks_a_rewritten_results_file_against_the_commit(tmp_path, capsys):
    results = tmp_path / "results"
    write(results / "rmsnorm.json", COMMITTED)
    git = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "."], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "rows"], check=True)
    # the bench rerun into the checkout, as run_all_benches.sh does
    write(results / "rmsnorm.json", [rms(0, 0.2), rms(3, 0.05), rms(4, 0.05)])
    fresh = str(results / "rmsnorm.json")
    assert cr.main([fresh, "--against", str(results)]) == 2  # the file against itself
    assert "pass --rev HEAD" in capsys.readouterr().err
    assert cr.main([fresh, "--against", str(results), "--rev", "HEAD"]) == 1
    out = capsys.readouterr().out
    assert "at HEAD" in out and "+25.0%" in out
    # a directory at a revision, and a file it does not have
    write(results / "softmax.json", [bench_row("softmax", "bf16", 3, "4096x4096", 0.05)])
    assert cr.main([str(results), "--against", str(results), "--rev", "HEAD"]) == 1
    assert "+25.0%" in capsys.readouterr().out
