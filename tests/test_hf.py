"""hf.load on a one-layer checkpoint written in the Hugging Face Llama layout: every weight
comes back in the engine's layout (fused qkv, interleaved gate/up, [K, N] projections), and
configs of other shapes are refused."""

import json

import pytest
import torch

hf = pytest.importorskip("spark_kernels.hf")
pytest.importorskip("safetensors")
L = pytest.importorskip("spark_kernels.layer")
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


@pytest.mark.parametrize("bad", [dict(hidden_size=3584), dict(num_key_value_heads=4),
                                 dict(rope_scaling={"rope_type": "llama3"}),
                                 dict(vocab_size=32001)])
def test_config_refuses_other_shapes(tmp_path, bad):
    (tmp_path / "config.json").write_text(json.dumps(_config(**bad)))
    with pytest.raises(ValueError):
        hf.ModelConfig.from_dir(tmp_path)
