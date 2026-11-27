"""CPU tests of the pure-torch Inkling 2 Mini reference (python/spark_kernels/inkling.py):
the config and parameter counts of the released model, parity of the prefill against HF's
Inkling2LoopForCausalLM on a tiny random model in fp32, and token-by-token decode through the
48-layer cache against the prefill. The module is loaded from its file, since importing the
package needs the built extension."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_PATH = Path(__file__).resolve().parent.parent / "python" / "spark_kernels" / "inkling.py"
_spec = importlib.util.spec_from_file_location("spark_inkling", _PATH)
ink = importlib.util.module_from_spec(_spec)
sys.modules["spark_inkling"] = ink  # dataclasses look the module up by name
_spec.loader.exec_module(ink)

# text_config of the released config.json (thinkingmachines/Inkling-2-Mini), as published
REAL_TEXT_CONFIG = {
    "model_type": "inkling2_loop_text", "hidden_size": 2304, "num_hidden_layers": 24,
    "vocab_size": 201024, "num_attention_heads": 24, "num_key_value_heads": 8, "head_dim": 128,
    "d_rel": 16, "rel_extent": 1024, "q_bias": False, "o_bias": False,
    "log_scaling_n_floor": 8192, "log_scaling_alpha": 0.1, "rms_norm_eps": 1e-06,
    "use_value_norm_no_gamma": True, "qk_l2_clip_threshold": 22.627416997969522,
    "rel_l2_clip_threshold": 8.0,
    "local_layer_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22],
    "sconv_kernel_size": 4, "unpadded_vocab_size": 200058, "final_logit_softcapping": 30.0,
    "swa_head_dim": 128, "swa_num_attention_heads": 24, "swa_num_key_value_heads": 8,
    "sliding_window_size": 1024, "n_layers_per_attn_res_block": 6,
    "attn_res_normalize_values": True, "n_routed_experts": 128, "num_experts_per_tok": 8,
    "n_shared_experts": 0, "intermediate_size": 1152, "mlp_gate_softcap": 4.0,
    "mlp_up_softcap": 25.0, "route_scale": 8.0, "use_gate_bias": True,
    "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3, "gate_rank": 64,
                    "is_varloop": False},
}

# Small enough for CPU, still with every feature: 12 layers in blocks of 3, the loop over
# layers 3-8 three times (24 applications), full layers in the prelude (2), the loop (5, 8)
# and after it (11), a window of 8 and a full-layer bias range of 6 shorter than the test
# sequences, log scaling from position 3, clip thresholds low enough to clip.
TINY_TEXT_CONFIG = {
    "hidden_size": 64, "num_hidden_layers": 12, "vocab_size": 128, "unpadded_vocab_size": 120,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
    "swa_num_attention_heads": 4, "swa_num_key_value_heads": 2, "swa_head_dim": 16,
    "sliding_window_size": 8, "rel_extent": 6, "d_rel": 4, "log_scaling_n_floor": 4,
    "qk_l2_clip_threshold": 2.0, "rel_l2_clip_threshold": 1.0,
    "local_layer_ids": [0, 1, 3, 4, 6, 7, 9, 10], "n_layers_per_attn_res_block": 3,
    "n_routed_experts": 8, "num_experts_per_tok": 2, "intermediate_size": 32,
    "loop_config": {"core_start": 3, "core_end": 9, "num_passes": 3, "gate_rank": 8},
}


def test_real_config_layout():
    cfg = ink.InklingConfig.from_hf({"text_config": REAL_TEXT_CONFIG})
    assert cfg == ink.InklingConfig()  # the defaults are the released model
    full = [i for i, t in enumerate(cfg.layer_types) if t == ink.FULL]
    assert full == [5, 11, 17, 23]  # and HF's (i + 1) % 6 rule agrees with local_layer_ids
    assert (cfg.loop_start, cfg.loop_end, cfg.loop_passes, cfg.gate_rank) == (6, 18, 3, 64)
    assert cfg.num_applications == cfg.num_cache_layers == 48
    sched = cfg.schedule()
    assert len(sched) == 48 and sched[:7] == [(i, 0) for i in range(7)]
    assert sched[18] == (6, 1) and sched[41] == (17, 2) and sched[42] == (18, 0)
    caches = [cfg.cache_index(layer, p) for layer, p in sched]
    assert sorted(caches) == list(range(48))
    assert cfg.cache_index(6, 1) == 24 and cfg.cache_index(17, 2) == 47
    types = cfg.cache_layer_types()
    assert types.count(ink.FULL) == 8 and types.count(ink.SLIDING) == 40
    assert all(types[cfg.cache_index(layer, p)] == cfg.layer_types[layer] for layer, p in sched)


def test_real_param_counts():
    c = ink.InklingConfig().param_counts()
    assert c["total"] == 26_068_156_416  # HF's Inkling2LoopForCausalLM on the meta device
    assert c["experts"] == 24 * 128 * 3 * 2304 * 1152
    assert c["lm_head"] == 200058 * 2304 and c["embed"] == 201024 * 2304 + 2304
    vision = 318_256_128  # HF's vision tower for the released vision_config
    assert round((c["total"] + vision) / 1e9, 1) == 26.4  # the published total
    assert c["active"] == 3_135_799_296
    assert round((c["active"] + vision) / 1e9, 2) == 3.45  # published: 3.4B
    assert c["expert_reads"] == 48 * 8 * 3 * 2304 * 1152


def test_bad_loop_layout():
    with pytest.raises(ValueError):
        ink.InklingConfig(loop_start=4)  # not a multiple of the 6-layer block
    with pytest.raises(ValueError):
        ink.InklingConfig(loop_start=0, loop_end=6)


def _randomize(model, seed=0):
    """Random weights at a scale that keeps the activations O(1), queries and taps included
    (HF initializes the residual queries and the dynamic taps to zero)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.state_dict().items():
            r = torch.randn(p.shape, generator=g)
            if name.endswith(("scale", "norm.weight", "layernorm.weight")):
                p.copy_(1 + 0.2 * r)
            elif name.endswith("e_score_correction_bias"):
                p.copy_(0.1 * r)
            elif name.endswith("query"):
                p.copy_(r)
            elif p.dim() >= 2:
                p.copy_(r * p.shape[-1] ** -0.5)
            else:
                p.copy_(0.5 * r)


def _tiny():
    cfg = ink.InklingConfig.from_hf({"text_config": TINY_TEXT_CONFIG})
    torch.manual_seed(0)
    model = ink.InklingForCausalLM(cfg).eval()
    _randomize(model)
    return cfg, model


def test_tiny_param_count_matches_module():
    cfg, model = _tiny()
    assert sum(p.numel() for p in model.parameters()) == cfg.param_counts()["total"]
    real = ink.InklingConfig()
    with torch.device("meta"):
        big = ink.InklingForCausalLM(real)
    assert sum(p.numel() for p in big.parameters()) == real.param_counts()["total"]


def _hf_model(cfg_dict):
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Inkling2LoopForCausalLM"):
        pytest.skip("this transformers has no Inkling2LoopForCausalLM")
    from transformers.models.inkling2_loop.configuration_inkling2_loop import (
        Inkling2LoopTextConfig,
    )

    d = dict(cfg_dict)
    local = set(d.pop("local_layer_ids"))
    d["layer_types"] = ["hybrid_sliding" if i in local else "hybrid"
                        for i in range(d["num_hidden_layers"])]
    hf_cfg = Inkling2LoopTextConfig(**d)
    hf_cfg._attn_implementation = "eager"
    hf_cfg._experts_implementation = "eager"  # the per-expert loop, the same GEMMs as ours
    return transformers.Inkling2LoopForCausalLM(hf_cfg).eval()


def test_prefill_matches_hf():
    hf = _hf_model(TINY_TEXT_CONFIG)
    cfg, ref = _tiny()
    hf.load_state_dict(ref.state_dict(), strict=True)
    ref.load_state_dict(hf.state_dict(), strict=True)  # and back: the keys are the same
    ids = torch.randint(0, cfg.vocab_size, (2, 21), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        want = hf(input_ids=ids, use_cache=False).logits
        got = ref(ids)
    assert got.shape == want.shape == (2, 21, cfg.unpadded_vocab_size)
    assert got.abs().max() > 1.0 and got.abs().max() <= cfg.logit_softcap
    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("chunk", [0, 5])
def test_decode_matches_prefill(chunk):
    """Token by token through the cache (after an optional prefill of `chunk` tokens into
    it) equals the prefill of the whole sequence at every position. 21 tokens is past the
    sliding window (8) and the full layers' bias range (6), so the window eviction, the
    bias cut-off and the log scaling (from position 3) are all in play."""
    cfg, ref = _tiny()
    ids = torch.randint(0, cfg.vocab_size, (2, 21), generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        want = ref(ids)
        cache = ink.InklingCache(cfg)
        steps = [ref(ids[:, :chunk], cache)] if chunk else []
        steps += [ref.decode_step(ids[:, t], cache)[:, None] for t in range(chunk, ids.shape[1])]
    got = torch.cat(steps, dim=1)
    torch.testing.assert_close(got, want, rtol=1e-4, atol=2e-4)  # fp32 rounding, |logit| ~ 10
    assert cache.length == 21
    for c, kind in enumerate(cfg.cache_layer_types()):
        kept = cache.keys[c].shape[2]
        assert kept == (cfg.sliding_window if kind == ink.SLIDING else 21)
        assert [s.shape[1:] for s in cache.conv[c]] == [(3, 32), (3, 32), (3, 64), (3, 64)]
    assert len(cache.keys) == cfg.num_cache_layers == 24


def test_decode_matches_hf_cache():
    """The same tokens decoded by HF through its own DynamicCache: the 24 (here) cache layers
    line up with HF's, pass copies included."""
    hf = _hf_model(TINY_TEXT_CONFIG)
    cfg, ref = _tiny()
    hf.load_state_dict(ref.state_dict(), strict=True)
    ids = torch.randint(0, cfg.vocab_size, (1, 14), generator=torch.Generator().manual_seed(3))
    with torch.no_grad():
        out = hf(input_ids=ids[:, :4], use_cache=True)
        hf_steps, past = [out.logits], out.past_key_values
        for t in range(4, ids.shape[1]):
            out = hf(input_ids=ids[:, t:t + 1], past_key_values=past, use_cache=True)
            hf_steps.append(out.logits)
            past = out.past_key_values
        cache = ink.InklingCache(cfg)
        ours = [ref(ids[:, :4], cache)] + [ref.decode_step(ids[:, t], cache)[:, None]
                                            for t in range(4, ids.shape[1])]
    torch.testing.assert_close(torch.cat(ours, 1), torch.cat(hf_steps, 1), rtol=1e-4, atol=1e-4)
