"""The quantized projections of spark_kernels.quant: each format's projection against the GEMM
on its own dequantized weight (and, for the formats that quantize activations, on the
activation quantized by the reference quantizer)."""

import pytest
import torch

Q = pytest.importorskip("spark_kernels.quant")
from spark_kernels import reference as ref  # noqa: E402

QUANT = [f for f in Q.FORMATS if f != "bf16"]


def _act_ref(a, fmt):
    """The activation as the format's GEMM sees it, dequantized to fp32."""
    if fmt in ("int4", "int4-asym"):
        return a.float()
    if fmt in ("fp8", "fp8-tok"):
        amax = a.float().abs().amax(dim=1, keepdim=True)
        if fmt == "fp8":
            amax = amax.amax()
        s = 2.0 ** torch.ceil(torch.log2(amax / 448.0))
        return (a.float() / s).clamp(-448, 448).to(torch.float8_e4m3fn).float() * s
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


def test_mx_bf16_quantizer_matches_reference(sk):
    a = torch.randn(33, 512, device="cuda").to(torch.bfloat16) * 3
    a[:, :32] = 0  # an all-zero block
    q, sf = Q.quantize_mx_bf16(a)
    q_ref, sf_ref = ref.quantize_mx(a)
    assert torch.equal(sf, sf_ref)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
