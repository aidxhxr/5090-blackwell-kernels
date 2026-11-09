"""scripts/llm_parity.py's fp32 reference on a model that is not Llama-3-8B's shape: a
two-layer Mistral saved by transformers with heads wider than hidden / heads (hidden 1024, 16
heads of 128 over 4 K/V heads, so q is 1024 -> 2048) and a Llama with heads of 64, the
reference's logits against transformers' own forward in fp32."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

pytest.importorskip("spark_kernels.hf")

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "llm_parity.py"

TINY = [("llama", dict(hidden_size=1024, num_attention_heads=16, num_key_value_heads=2,
                       intermediate_size=2816)),
        ("mistral", dict(hidden_size=1024, num_attention_heads=16, num_key_value_heads=4,
                         head_dim=128, intermediate_size=2048, sliding_window=None))]


def _script():
    spec = importlib.util.spec_from_file_location("llm_parity", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("kind, kw", TINY, ids=[k for k, _ in TINY])
def test_ref_forward_matches_transformers(sk, tmp_path, kind, kw):
    transformers = pytest.importorskip("transformers")
    cls = {"llama": (transformers.LlamaConfig, transformers.LlamaForCausalLM),
           "mistral": (transformers.MistralConfig, transformers.MistralForCausalLM)}[kind]
    hcfg = cls[0](num_hidden_layers=2, vocab_size=1024, rms_norm_eps=1e-6, rope_theta=10000.0,
                  max_position_embeddings=1024, tie_word_embeddings=False, **kw)
    torch.manual_seed(0)
    model = cls[1](hcfg)
    # .to(bfloat16) rounds the rotary inv_freq buffer too (0.37% off), and .float() later
    # doesn't bring it back, so keep the fp32 one for transformers' side of the comparison
    inv_freq = model.model.rotary_emb.inv_freq.clone()
    model = model.to(torch.bfloat16)
    with torch.no_grad():  # the engine's scale: N(0, 1/K) projections, N(0, 1) embedding
        for name, prm in model.named_parameters():
            if prm.dim() == 2:
                prm.normal_(0.0, 1.0 if "embed" in name else 1.0 / math.sqrt(prm.shape[1]))
    model.save_pretrained(tmp_path)
    parity = _script()
    g = torch.Generator().manual_seed(0)
    seqs = [torch.randint(0, 1024, (n,), generator=g) for n in (37, 100)]
    with torch.no_grad():
        xn = parity.ref_forward(str(tmp_path), seqs)
        got = torch.cat([lg for _, lg in parity.ref_logits(str(tmp_path), xn)])
        ref = model.float().cuda()
        ref.model.rotary_emb.inv_freq.copy_(inv_freq)
        want = torch.cat([ref(s[None].cuda()).logits[0] for s in seqs])
    assert got.shape == want.shape == (137, 1024)
    rel = ((got - want).norm() / want.norm()).item()
    assert rel < 1e-4, rel
