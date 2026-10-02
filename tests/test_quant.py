"""The quantized projections of spark_kernels.quant and the engine's quantized SparkModel on a
two-layer model with a small vocabulary.

Each format's projection is checked against the GEMM on its own dequantized weight (and, for
the formats that quantize activations, on the activation quantized by the reference
quantizer); the whole model against TorchModel on the dequantized weights, at prefill and at
decode, which leaves only the activation quantization (W8A8, W4A4) between the two; and the
CUDA-graph decode of a quantized model against its eager decode, token by token."""

import pytest
import torch

E = pytest.importorskip("spark_kernels.engine")
L = pytest.importorskip("spark_kernels.layer")
Q = pytest.importorskip("spark_kernels.quant")
from spark_kernels import reference as ref  # noqa: E402

N_LAYERS, VOCAB = 2, 1024
QUANT = [f for f in Q.FORMATS if f != "bf16"]
# relative error of the model's logits against TorchModel on the dequantized weights: int4 is
# weight only, so the gap is rounding; the others also quantize every activation
MODEL_TOL = {"int4": 2e-2, "int4-asym": 2e-2, "fp8": 0.08, "fp8-tok": 0.08, "mxfp8": 0.08,
             "nvfp4": 0.25, "mxfp4": 0.3}


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.fixture(scope="module")
def weights():
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    return E.ModelWeights.random(N_LAYERS, VOCAB, seed=1)


def _act_ref(a, fmt):
    """The activation as the format's GEMM sees it, dequantized to fp32."""
    if fmt in ("int4", "int4-asym"):
        return a.float()
    if fmt in ("fp8", "fp8-tok"):
        q, s = ref.quantize_fp8_pow2(a, per_row=fmt == "fp8-tok")
        return q.float() * s
    if fmt == "mxfp8":
        return ref.dequantize_mx(*ref.quantize_mx(a))
    if fmt == "nvfp4":
        q, sf, s = ref.quantize_nvfp4(a)
        return ref.dequantize_fp4(q, sf, "nvfp4", s)
    q, sf = ref.quantize_mxfp4(a)
    return ref.dequantize_fp4(q, sf, "mxfp4")


@pytest.mark.parametrize("fmt", QUANT)
@pytest.mark.parametrize("m", [1, 7, 64, 300])
def test_linear_matches_reference(sk, fmt, m):
    K, N = 512, 768
    g = torch.Generator(device="cpu").manual_seed(m)
    w = (torch.randn(K, N, generator=g) / K**0.5).to("cuda", torch.bfloat16)
    a = torch.randn(m, K, generator=g).to("cuda", torch.bfloat16)
    a[0, 3] = 40.0  # an outlier channel, as real activations have
    lin = Q.make_linear(w, fmt)
    out = lin(a)
    assert out.shape == (m, N) and out.dtype == torch.bfloat16
    want = _act_ref(a, fmt) @ lin.dequantize().float()
    assert _rel(out, want) < 1e-2
    # and the quantized weight is close to the bf16 one it replaces
    wtol = {"fp8": 0.06, "fp8-tok": 0.06, "mxfp8": 0.06}.get(fmt, 0.2)
    assert _rel(lin.dequantize(), w) < wtol


def _prefill_and_decode(model, prompts, toks):
    """Packed prefill logits, then three teacher-forced decode steps' logits."""
    eng = E.Engine(model, N_LAYERS, max_batch=4, max_seq=512, num_pages=128, graphs=False)
    for p in prompts:
        eng.submit(p, max_new=8)
    seqs = eng._admit()
    outs = [model.prefill(eng._prefill_batch(seqs), eng.cache)]
    for s in seqs:
        s.length = s.prompt.numel()
    for t in toks:
        pos = [s.length for s in seqs]
        eng.ids[:4] = t
        eng.positions[:4] = torch.tensor(pos, dtype=torch.int32)
        slots = [s.pages[p // eng.page] * eng.page + p % eng.page
                 for s, p in zip(seqs, pos, strict=True)]
        eng.slots[:4] = torch.tensor(slots, dtype=torch.int32)
        eng.seq_lens[:4] = torch.tensor([p + 1 for p in pos], dtype=torch.int32)
        outs.append(model.decode(eng._decode_batch(4, max(pos) + 1), eng.cache))
        for s in seqs:
            s.length += 1
    return outs


@pytest.mark.parametrize("fmt", QUANT)
def test_quant_model_matches_torch_on_dequantized(sk, weights, fmt):
    rope = L.RoPE(1024)
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(0, VOCAB, (n,), generator=g) for n in (37, 200, 1, 64)]
    toks = [torch.randint(0, VOCAB, (4,), generator=g).cuda() for _ in range(3)]
    spark = E.SparkModel(weights, rope, weights_format=fmt, keep_bf16=True)
    torch_ = E.TorchModel(spark.dequantized(), rope)
    a = _prefill_and_decode(spark, prompts, toks)
    b = _prefill_and_decode(torch_, prompts, toks)
    for i, (x, y) in enumerate(zip(a, b, strict=True)):
        assert _rel(x, y) < MODEL_TOL[fmt], f"{fmt} step {i}: {_rel(x, y)}"


def test_quant_model_frees_bf16(sk):
    w = E.ModelWeights.random(1, VOCAB, seed=2)
    bf16 = E.SparkModel(w, L.RoPE(64)).weight_bytes()
    m = E.SparkModel(w, L.RoPE(64), weights_format="int4")
    lw = w.layers[0]
    assert lw.w_qkv is None and lw.w_o is None and lw.w_gate_up is None and lw.w_down is None
    # 4.25 bits per projection weight (int4 plus a bf16 scale per 128) against 16
    proj = bf16 - m.weight_bytes()
    assert proj > 0.7 * (bf16 - (w.embed.numel() + w.lm_head.numel()) * 2)


@pytest.mark.parametrize("fmt", ["int4", "fp8", "fp8-tok", "mxfp8", "nvfp4"])
def test_quant_graph_decode_matches_eager(sk, weights, fmt):
    rope = L.RoPE(1024)
    g = torch.Generator().manual_seed(3)
    lens = [5, 90, 17, 300, 1, 44]
    news = [6, 3, 9, 4, 12, 5]
    outs = []
    for graphs in (False, True):
        model = E.SparkModel(weights, rope, weights_format=fmt, keep_bf16=True)
        eng = E.Engine(model, N_LAYERS, max_batch=4, max_seq=512, num_pages=160,
                       graphs=graphs, log_tokens=True, prefill_tokens=256)
        g.manual_seed(3)
        for n, m in zip(lens, news, strict=True):
            eng.submit(torch.randint(0, VOCAB, (n,), generator=g), m)
        eng.run()
        assert eng.graphs or not graphs
        outs.append(eng.outputs())
    # Greedy tokens of a random model sit near ties, and the split-K GEMMs add their fp32
    # partial sums in any order, so one flipped argmax can send a sequence down another path
    # (more often with the formats that quantize activations). Count each sequence's tokens
    # up to its first difference: a broken graph agrees on almost none.
    assert [len(outs[1][i]) for i in range(len(lens))] == news
    same = 0
    for i in range(len(lens)):
        for x, y in zip(outs[0][i], outs[1][i], strict=True):
            if x != y:
                break
            same += 1
    assert same >= 0.6 * sum(news), f"{same} of {sum(news)} tokens agree"
