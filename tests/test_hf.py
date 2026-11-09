"""hf.load on a one-layer checkpoint written in the Hugging Face Llama layout: every weight
comes back in the engine's layout (fused qkv, interleaved gate/up, [K, N] projections). The
engine's ModelConfig built from config.json (the head_dim and K/V head fallbacks, eps, the
rope keys of transformers 4 and 5), every config and checkpoint refused with its reason, and
two small models saved by transformers itself (a Llama with heads of 64, a Mistral with heads
wider than hidden / heads) through hf.load and SparkModel against transformers' forward."""

import json
import math

import pytest
import torch

hf = pytest.importorskip("spark_kernels.hf")
pytest.importorskip("safetensors")
L = pytest.importorskip("spark_kernels.layer")
E = pytest.importorskip("spark_kernels.engine")
sk = pytest.importorskip("spark_kernels.ops")

VOCAB = 1024


def _config(**kw):
    cfg = dict(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
               intermediate_size=14336, num_hidden_layers=1, vocab_size=VOCAB,
               rope_theta=500000.0, rms_norm_eps=1e-5, max_position_embeddings=8192,
               tie_word_embeddings=False)
    cfg.update(kw)
    return cfg


def test_load_matches_hf_layout(tmp_path):
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(0)

    def r(*shape):
        return torch.randn(*shape, generator=g).to(torch.bfloat16)

    p = "model.layers.0."
    t = {p + "self_attn.q_proj.weight": r(4096, 4096), p + "self_attn.k_proj.weight": r(1024, 4096),
         p + "self_attn.v_proj.weight": r(1024, 4096), p + "self_attn.o_proj.weight": r(4096, 4096),
         p + "mlp.gate_proj.weight": r(14336, 4096), p + "mlp.up_proj.weight": r(14336, 4096),
         p + "mlp.down_proj.weight": r(4096, 14336), p + "input_layernorm.weight": r(4096),
         p + "post_attention_layernorm.weight": r(4096), "model.norm.weight": r(4096),
         "model.embed_tokens.weight": r(VOCAB, 4096), "lm_head.weight": r(VOCAB, 4096)}
    save_file(t, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps(_config()))
    w, cfg = hf.load(tmp_path)
    assert cfg.n_layers == 1 and cfg.vocab == VOCAB and cfg.rope_theta == 500000.0
    assert w.config is cfg.shape
    lw = w.layers[0]
    x = torch.randn(3, 4096, generator=g).to("cuda", torch.bfloat16)

    def lin(name, inp):
        return inp.float() @ t[name].cuda().float().t()

    qkv = torch.cat([lin(p + f"self_attn.{n}_proj.weight", x) for n in "qkv"], dim=1)
    torch.testing.assert_close(x.float() @ lw.w_qkv.float(), qkv, rtol=1e-3, atol=1e-2)
    gate, up = lin(p + "mlp.gate_proj.weight", x), lin(p + "mlp.up_proj.weight", x)
    gu = x.float() @ lw.w_gate_up.float()
    torch.testing.assert_close(gu[:, 0::2], gate, rtol=1e-3, atol=1e-2)
    torch.testing.assert_close(gu[:, 1::2], up, rtol=1e-3, atol=1e-2)
    assert torch.equal(lw.w_o.cpu(), t[p + "self_attn.o_proj.weight"].t())
    assert torch.equal(lw.w_down.cpu(), t[p + "mlp.down_proj.weight"].t())
    assert torch.equal(w.lm_head.cpu(), t["lm_head.weight"].t())
    assert torch.equal(w.embed.cpu(), t["model.embed_tokens.weight"])


LLAMA31_SCALING = {"factor": 8.0, "high_freq_factor": 4.0, "low_freq_factor": 1.0,
                   "original_max_position_embeddings": 8192, "rope_type": "llama3"}


def test_config_reads_llama3_rope_scaling(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(_config(
        rope_scaling=LLAMA31_SCALING, max_position_embeddings=131072)))
    cfg = hf.HFConfig.from_dir(tmp_path)
    assert cfg.rope_scaling == LLAMA31_SCALING and cfg.max_pos == 131072


def test_llama3_rope_matches_transformers():
    """inv_freq of the llama3 rope type against transformers' own init, and the tables built
    from it: high frequencies untouched, low ones divided by the factor."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    rope_utils = pytest.importorskip("transformers.modeling_rope_utils")
    from transformers import LlamaConfig

    hcfg = LlamaConfig(hidden_size=4096, num_attention_heads=32, num_key_value_heads=8,
                       rope_theta=500000.0, max_position_embeddings=131072,
                       rope_scaling=LLAMA31_SCALING)
    want, _ = rope_utils.ROPE_INIT_FUNCTIONS["llama3"](hcfg, "cpu")
    rope = L.RoPE(4096, theta=500000.0, scaling=LLAMA31_SCALING)
    plain = L.RoPE(4096, theta=500000.0)
    torch.testing.assert_close(rope.inv_freq.cpu(), want.float(), rtol=1e-6, atol=0)
    assert torch.equal(rope.inv_freq[:8], plain.inv_freq[:8])
    torch.testing.assert_close(rope.inv_freq[-8:], plain.inv_freq[-8:] / 8)
    pos = torch.arange(4096, device="cuda").float()
    # fp32 angles up to ~4000 rad: a one-ulp inv_freq difference moves cos by ~1e-4
    torch.testing.assert_close(rope.cos[:, :64], torch.outer(pos, want.cuda().float()).cos(),
                               rtol=0, atol=1e-3)


def _from(tmp_path, cfg):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    return hf.HFConfig.from_dir(tmp_path)


def _fields(c):
    return (c.hidden, c.n_heads, c.n_kv_heads, c.head_dim, c.intermediate, c.eps)


def test_config_llama3_8b_is_the_default_shape(tmp_path):
    """Llama-3-8B's config (no head_dim key, as the checkpoint ships it) gives the engine's
    default shape, so a Llama checkpoint runs the model it always ran."""
    cfg = _from(tmp_path, _config())
    assert _fields(cfg.shape) == _fields(E.LLAMA3_8B)
    assert cfg.shape.name == tmp_path.name and cfg.eps == 1e-5


@pytest.mark.parametrize("kw, want", [
    # head_dim from hidden / heads when the config has none (transformers' fallback)
    (dict(hidden_size=1024, num_attention_heads=16, num_key_value_heads=2,
          intermediate_size=2816), (1024, 16, 2, 64, 2816)),
    # head_dim given and wider than hidden / heads: q is 2048 -> 4096, as Qwen3's
    (dict(hidden_size=2048, num_attention_heads=32, num_key_value_heads=4, head_dim=128,
          intermediate_size=6144), (2048, 32, 4, 128, 6144)),
    # head_dim null is the fallback too
    (dict(hidden_size=2048, num_attention_heads=16, head_dim=None, num_key_value_heads=8),
     (2048, 16, 8, 128, 14336)),
    # no num_key_value_heads: one K/V head per query head
    (dict(hidden_size=1024, num_attention_heads=8, num_key_value_heads=None),
     (1024, 8, 8, 128, 14336)),
])
def test_config_builds_the_shape(tmp_path, kw, want):
    cfg = _from(tmp_path, _config(**kw))
    assert _fields(cfg.shape)[:5] == want
    assert cfg.shape.q_width == want[1] * want[3]


def test_config_eps_and_rope_keys(tmp_path):
    """rms_norm_eps from the config, 1e-6 (transformers' default) without one; rope_theta and
    rope_scaling, or the rope_parameters transformers 5 writes in their place."""
    cfg = _from(tmp_path, _config(rms_norm_eps=1e-6))
    assert cfg.eps == 1e-6 and cfg.shape.eps == 1e-6
    c = _config()
    del c["rms_norm_eps"], c["rope_theta"]
    cfg = _from(tmp_path, c)
    assert cfg.eps == 1e-6 and cfg.rope_theta == 10000.0 and cfg.rope_scaling is None
    cfg = _from(tmp_path, dict(c, rope_parameters={"rope_type": "default", "rope_theta": 1e6}))
    assert cfg.rope_theta == 1e6 and cfg.rope_scaling is None
    rp = dict(LLAMA31_SCALING, rope_theta=500000.0)
    cfg = _from(tmp_path, dict(c, rope_parameters=rp))
    assert cfg.rope_theta == 500000.0 and cfg.rope_scaling == rp
    old = _from(tmp_path, _config(rope_scaling=LLAMA31_SCALING))
    assert torch.equal(cfg.rope(256, device="cpu").cos, old.rope(256, device="cpu").cos)


def test_config_rope_has_the_head_dim(tmp_path):
    cfg = _from(tmp_path, _config(hidden_size=1024, num_attention_heads=16,
                                  num_key_value_heads=2, intermediate_size=2816))
    rope = cfg.rope(64, device="cpu")
    assert rope.cos.shape == (64, 64)
    assert torch.equal(rope.cos, L.RoPE(64, "cpu", theta=500000.0, head_dim=64).cos)


@pytest.mark.parametrize("bad, why", [
    # what ModelConfig.check refuses
    (dict(hidden_size=3584), "head_dim 64 or 128"),  # 3584 / 32 = 112
    (dict(head_dim=96), "head_dim 64 or 128"),
    (dict(num_key_value_heads=5), "not a multiple of 5 K/V heads"),
    (dict(num_key_value_heads=1), "at most 16 query heads per K/V head"),
    (dict(intermediate_size=14000), "multiples of 64"),
    (dict(hidden_size=4100), "not a multiple of 32 heads"),
    # what the engine cannot run yet
    (dict(model_type="qwen3", hidden_size=4096, head_dim=128), "#47"),
    (dict(model_type="qwen3_moe", num_experts=128, moe_intermediate_size=768), "#47"),
    (dict(model_type="mixtral", num_local_experts=8, num_experts_per_tok=2), "experts"),
    (dict(model_type="deepseek_v3", n_routed_experts=256), "experts"),
    (dict(model_type="gemma"), "Llama layout"),
    (dict(model_type="qwen2"), "Llama layout"),
    (dict(hidden_act="gelu_pytorch_tanh"), "SiLU"),
    (dict(attention_bias=True), "no bias"),
    (dict(mlp_bias=True), "no bias"),
    (dict(model_type="mistral", sliding_window=4096), "sliding_window 4096"),
    (dict(partial_rotary_factor=0.5), "rotate the whole head"),
    (dict(rope_scaling={"rope_type": "yarn", "factor": 4.0}), "not supported"),
    (dict(rope_scaling={"type": "dynamic", "factor": 2.0}), "not supported"),
    (dict(rope_parameters={"rope_type": "yarn", "factor": 4.0, "rope_theta": 1e6}),
     "not supported"),
    (dict(vocab_size=32001), "multiple of 64"),
])
def test_config_refuses(tmp_path, bad, why):
    with pytest.raises(ValueError, match=why):
        _from(tmp_path, _config(**bad))


def test_config_takes_what_runs(tmp_path):
    """The keys of the configs that do run: Mistral's null sliding window, a window that is
    off, false biases, tied embeddings."""
    for kw in (dict(model_type="mistral", sliding_window=None, head_dim=128),
               dict(use_sliding_window=False, sliding_window=32768),
               dict(attention_bias=False, mlp_bias=False, hidden_act="silu"),
               dict(tie_word_embeddings=True)):
        cfg = _from(tmp_path, _config(**kw))
        assert _fields(cfg.shape) == _fields(E.LLAMA3_8B)
    assert cfg.tie_embeddings


SMALL = dict(hidden_size=256, num_attention_heads=4, num_key_value_heads=2,
             intermediate_size=512, num_hidden_layers=1, vocab_size=128, rms_norm_eps=1e-6)


def _small_checkpoint(path, extra=None):
    """A one-layer checkpoint of SMALL's shape (heads of 64); `extra` adds tensors of layer 0
    or gives one another shape (name -> shape)."""
    from safetensors.torch import save_file

    g = torch.Generator().manual_seed(1)
    p = "model.layers.0."
    shapes = {p + "self_attn.q_proj.weight": (256, 256), p + "self_attn.k_proj.weight": (128, 256),
              p + "self_attn.v_proj.weight": (128, 256), p + "self_attn.o_proj.weight": (256, 256),
              p + "mlp.gate_proj.weight": (512, 256), p + "mlp.up_proj.weight": (512, 256),
              p + "mlp.down_proj.weight": (256, 512), p + "input_layernorm.weight": (256,),
              p + "post_attention_layernorm.weight": (256,), "model.norm.weight": (256,),
              "model.embed_tokens.weight": (128, 256), "lm_head.weight": (128, 256)}
    shapes.update({p + n: s for n, s in (extra or {}).items()})
    save_file({k: torch.randn(*s, generator=g).to(torch.bfloat16) for k, s in shapes.items()},
              str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(_config(**SMALL)))


def test_load_small_shape(tmp_path):
    """hf.load on a shape other than Llama-3-8B's: the weights carry it, and the engine's
    layout comes out in its widths (on the CPU: the conversion is plain torch)."""
    _small_checkpoint(tmp_path)
    w, cfg = hf.load(tmp_path, device="cpu")
    assert w.config is cfg.shape and _fields(cfg.shape)[:5] == (256, 4, 2, 64, 512)
    lw = w.layers[0]
    assert lw.w_qkv.shape == (256, 256 + 2 * 128) and lw.w_o.shape == (256, 256)
    assert lw.w_gate_up.shape == (256, 1024) and lw.w_down.shape == (512, 256)
    assert w.lm_head.shape == (256, 128)


@pytest.mark.parametrize("extra, why", [
    ({"self_attn.q_norm.weight": (64,), "self_attn.k_norm.weight": (64,)}, "#47"),
    ({"self_attn.q_proj.bias": (256,)}, "no bias"),
    ({"self_attn.k_proj.weight": (256, 256)}, r"k_proj.weight is \(256, 256\)"),
    ({"mlp.down_proj.weight": (256, 1024)}, "the config says"),
])
def test_load_refuses_checkpoints(tmp_path, extra, why):
    """Tensors the config did not announce (Qwen3's q_norm and k_norm, a bias), and tensors of
    another shape than the config's, are refused before anything runs."""
    _small_checkpoint(tmp_path, extra)
    with pytest.raises(ValueError, match=why):
        hf.load(tmp_path, device="cpu")


TINY = [("llama", dict(hidden_size=1024, num_attention_heads=16, num_key_value_heads=2,
                       intermediate_size=2816)),
        ("mistral", dict(hidden_size=1024, num_attention_heads=16, num_key_value_heads=4,
                         head_dim=128, intermediate_size=2048, sliding_window=None))]


@pytest.mark.parametrize("kind, kw", TINY, ids=[k for k, _ in TINY])
def test_transformers_checkpoint_matches(sk, tmp_path, kind, kw):
    """A two-layer model of a small shape built and saved by transformers (its own config.json
    and tensor names), through hf.load and SparkModel on the engine, against transformers'
    forward of the same bf16 weights in fp32: the logits at every prompt position."""
    transformers = pytest.importorskip("transformers")
    cls = {"llama": (transformers.LlamaConfig, transformers.LlamaForCausalLM),
           "mistral": (transformers.MistralConfig, transformers.MistralForCausalLM)}[kind]
    hcfg = cls[0](num_hidden_layers=2, vocab_size=1024, rms_norm_eps=1e-6, rope_theta=10000.0,
                  max_position_embeddings=1024, tie_word_embeddings=False, **kw)
    torch.manual_seed(0)
    model = cls[1](hcfg).to(torch.bfloat16)
    with torch.no_grad():  # the engine's scale: N(0, 1/K) projections, N(0, 1) embedding
        for name, prm in model.named_parameters():
            if prm.dim() == 2:
                prm.normal_(0.0, 1.0 if "embed" in name else 1.0 / math.sqrt(prm.shape[1]))
    model.save_pretrained(tmp_path)
    w, cfg = hf.load(tmp_path)
    assert cfg.shape.head_dim == hcfg.head_dim and cfg.shape.n_kv_heads == kw["num_key_value_heads"]
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(0, 1024, (n,), generator=g) for n in (37, 100)]
    eng = E.Engine(E.SparkModel(w, cfg.rope(512)), cfg.n_layers, max_batch=2, max_seq=512,
                   num_pages=64, graphs=False)
    got = eng.prompt_logits(prompts)
    ref = model.float().cuda()
    for p, lg in zip(prompts, got, strict=True):
        with torch.no_grad():
            want = ref(p[None].cuda()).logits[0]
        rel = ((lg.float() - want).norm() / want.norm()).item()
        assert rel < 2e-2, rel
