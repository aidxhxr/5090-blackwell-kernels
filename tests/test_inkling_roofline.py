"""CPU-only tests of scripts/inkling_roofline.py, the Inkling 2 Mini decode cost model: byte
counts small enough to check by hand, the parameter count against the reference model, and the
blog's 614 GB/s ceiling. The script is loaded from its file."""

import importlib.util
import json
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "scripts" / "inkling_roofline.py"
_spec = importlib.util.spec_from_file_location("inkling_roofline", _PATH)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

CFG = R.load_config(None)
KV_POS = 2 * 8 * 128 * 2  # k and v, 8 heads x 128, bf16: bytes per position per cache layer


def test_schedule_runs_the_loop_three_times():
    s = R.schedule(CFG)
    assert len(s) == 48 and len(R.cache_layers(CFG)) == 48
    layers = [i for i, _ in s]
    assert all(layers.count(i) == 3 for i in range(6, 18))
    assert all(layers.count(i) == 1 for i in [*range(6), *range(18, 24)])
    assert layers[:7] == [0, 1, 2, 3, 4, 5, 6] and layers[-7:] == [17, 18, 19, 20, 21, 22, 23]
    # full attention: 5 and 23 once, 11 and 17 three times each
    assert sum(R.is_full(CFG, i) for i in layers) == 8


def test_parameter_count_matches_the_reference_model():
    # Inkling2LoopForCausalLM built on the meta device from the HF config (transformers 5.24.1):
    # 26,068,156,416 parameters in the text decoder; the vision encoder adds 318,256,128, for the
    # 26.4B the model card gives.
    pc = R.param_counts(CFG)
    assert pc["total"] == 26_068_156_416
    assert (pc["total"] + 318_256_128) / 1e9 == pytest.approx(26.4, abs=0.05)
    # 5.4B weight activations per token with the loop
    assert pc["effective"] / 1e9 == pytest.approx(5.4, abs=0.05)


def test_distinct_experts_uniform_routing():
    assert R.distinct_experts(CFG, 1) == pytest.approx(8.0)
    assert R.distinct_experts(CFG, 2) == pytest.approx(128 * (1 - (15 / 16) ** 2))  # 15.5
    assert R.distinct_experts(CFG, 2) == pytest.approx(15.5)
    assert R.distinct_experts(CFG, 10_000) == pytest.approx(128.0)


def test_expert_bytes_by_format():
    e = 2 * 1152 * 2304 + 2304 * 1152  # gate_up [2304, 2304] and down [2304, 1152]
    assert R.expert_elems(CFG) == e == 7_962_624
    assert R.expert_bytes(CFG, 0, "bf16") == 2 * e
    assert R.expert_bytes(CFG, 19, "ckpt") == pytest.approx(e * (0.5 + 1 / 16))  # NVFP4
    assert R.expert_bytes(CFG, 20, "ckpt") == pytest.approx(e * (1 + 1 / 1024))  # MXFP8
    assert R.expert_bytes(CFG, 23, "fp4") == R.expert_bytes(CFG, 0, "ckpt")


def test_dense_bytes_of_layer_0():
    proj = 3072 * 2304 * 2 + 1024 * 2304 * 2 + 2304 * 3072  # q, gate; k, v; o
    proj += 24 * 2304 + 384 * 2304 + 128 * 2304  # sinks, relative state, router
    small = 16 * 1024 + 24 * 128 + 24 * 16  # relative bias bank, clip scales
    small += 2 * 2304 * 4 + 2 * 1024 * 4 + 2 * 2304  # post convs, k/v taps, two norms
    query = 2304  # mlp_input only: layer 0 has no attention_input
    assert R.dense_bytes(CFG, 0, "bf16", False) == 2 * (proj + small) + 4 * query
    assert R.dense_bytes(CFG, 0, "fp4", False) == pytest.approx(
        0.5625 * proj + 2 * small + 4 * query)
    gates = 4 * (64 * 4608 + 2 * 2304 * 64)
    assert R.dense_bytes(CFG, 6, "bf16", True) - R.dense_bytes(CFG, 6, "bf16", False) == 2 * gates


def test_kv_bytes_by_hand():
    # L = 1: every cache layer reads one position and writes the new one
    assert R.kv_read_bytes(CFG, 1) == 48 * 2 * KV_POS
    # L = 4096: 8 full cache layers read 4096 positions, 40 sliding ones 1024
    assert R.kv_read_bytes(CFG, 4096) == (8 * 4096 + 40 * 1024 + 48) * KV_POS
    assert R.kv_footprint(CFG, 4096) == (8 * 4096 + 40 * 1024) * KV_POS
    assert R.kv_footprint(CFG, 512) == 48 * 512 * KV_POS
    assert R.kv_read_bytes(CFG, 4096, kv_bytes=1.0) == R.kv_read_bytes(CFG, 4096) / 2


def test_conv_and_block_state_bytes():
    # (1024 + 1024 + 2304 + 2304) channels x 3 positions x fp32, read and written, 48 times
    assert R.conv_state_bytes(CFG) == 2 * 6656 * 3 * 4 * 48
    # prelude 22, pass 1 82, passes 2 and 3 106 each (24 message-input reads), postlude 59,
    # final input 5
    assert R.block_state_reads(CFG) == 22 + 82 + 2 * 106 + 59 + 5


def test_step_bytes_parts_and_scaling():
    one = R.step_bytes(CFG, 1, 1, "bf16")
    assert one["total"] == pytest.approx(sum(v for k, v in one.items() if k != "total"))
    assert one["experts"] == pytest.approx(48 * 8 * 2 * 7_962_624)
    # passes 2 and 3 re-read the loop's dense weights; pass 2 is the gates' first read
    loop = sum(R.dense_bytes(CFG, i, "bf16", False) for i in range(6, 18))
    gates = sum(R.dense_bytes(CFG, i, "bf16", True) for i in range(6, 18)) - loop
    assert one["loop rereads"] == pytest.approx(2 * loop + gates)
    # B sequences: the dense part is shared (but for B embedding rows), the rest is per token
    b8 = R.step_bytes(CFG, 8, 1, "bf16")
    assert b8["dense"] - one["dense"] == 7 * 2304 * 2
    assert b8["kv"] == 8 * one["kv"] and b8["conv states"] == 8 * one["conv states"]
    assert b8["experts"] == pytest.approx(one["experts"] * R.distinct_experts(CFG, 8) / 8)


def test_pinning_removes_the_rereads_of_pinned_layers():
    full = R.step_bytes(CFG, 1, 1, "fp4", pinned_layers=0)
    two = R.step_bytes(CFG, 1, 1, "fp4", pinned_layers=2)
    saved = sum(2 * R.dense_bytes(CFG, i, "fp4", False) + R.dense_bytes(CFG, i, "fp4", True)
                - R.dense_bytes(CFG, i, "fp4", False) for i in (6, 7))
    assert full["total"] - two["total"] == pytest.approx(saved)
    assert R.step_bytes(CFG, 1, 1, "fp4", pinned_layers=12)["loop rereads"] == 0
    # one bf16 loop layer is 56.5 MiB of dense weights: one fits in 3/4 of 96 or 128 MiB, none
    # in 3/4 of GB10's 24 MiB; in fp4 (16 MiB a layer) six fit in 3/4 of 128 MiB
    assert R.dense_bytes(CFG, 6, "bf16", True) / R.MB == pytest.approx(56.5, abs=0.1)
    assert R.pinnable_layers(CFG, "bf16", 96, 0.75) == 1
    assert R.pinnable_layers(CFG, "bf16", 128, 0.75) == 1
    assert R.pinnable_layers(CFG, "bf16", 24, 0.75) == 0
    assert R.pinnable_layers(CFG, "fp4", 128, 0.75) == 6


def test_ceiling_is_bytes_over_bandwidth():
    parts = {"total": 2e9}
    ms, tok_s = R.ceiling(parts, 4, 1000.0)
    assert ms == pytest.approx(2.0) and tok_s == pytest.approx(2000.0)


def test_blog_614_gbps_ceiling():
    # "a theoretical ceiling at 614 GB/s with BF16 weights and batch size one, including
    # repeated reads across loops": 614e9 / (5.4e9 x 2 bytes) = 56.9 tokens/s
    chk = R.blog_check(CFG)
    assert chk["blog_tok_s_5p4"] == pytest.approx(56.85, abs=0.01)
    assert chk["params_tok_s"] == pytest.approx(chk["blog_tok_s_5p4"], rel=0.01)
    # the embedding is a gather, not a stream, and the message gates only run in passes 2 and
    # 3 (the 5.4B counts them in all three): the bytes read are the effective parameters minus
    # the table and one pass of gates, plus the small activation reads, so the byte ceiling is
    # about 10% higher
    table = 201024 * 2304 * 2
    gates = 12 * 4 * (64 * 4608 + 2 * 2304 * 64) * 2
    assert chk["bytes"] == pytest.approx(2 * chk["effective_params"] - table - gates, rel=0.002)
    assert chk["bytes_tok_s"] == pytest.approx(62.8, abs=0.1)


def test_config_file_option(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"text_config": {**R.INKLING2_MINI, "num_experts_per_tok": 4}}))
    cfg = R.load_config(str(p))
    assert cfg["num_experts_per_tok"] == 4 and cfg["hidden_size"] == 2304
    assert R.distinct_experts(cfg, 1) == pytest.approx(4.0)
    assert R.main(["--config", str(p), "--card", "gb10", "--format", "fp4", "--batch", "1",
                   "--context", "1024", "--json"]) == 0
