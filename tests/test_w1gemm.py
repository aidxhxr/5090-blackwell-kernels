"""Parity tests for the W1A16 GEMM (sign and ternary weights, bf16 activations), its quantizer
and its grouped MoE form, against the plain-PyTorch versions in spark_kernels.reference.

The reference quantizer is tested on the CPU (no `sk` fixture); the extension's quantizer and
GEMMs are checked against it on the GPU."""

import importlib.util
from pathlib import Path

import pytest
import torch

try:
    from spark_kernels import reference as ref
except ImportError:
    # A checkout with no built extension (a CPU-only machine): the reference module alone,
    # which needs only torch, so the CPU half of this file still runs.
    _spec = importlib.util.spec_from_file_location(
        "spark_reference",
        Path(__file__).resolve().parent.parent / "python" / "spark_kernels" / "reference.py")
    ref = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(ref)

# (K, N) weight shapes for the GEMM: the Llama-3-8B-sized ones in both orientations.
WEIGHTS = [(5120, 2304), (2304, 5120)]
MS = [1, 7, 16, 64]
BITS = [1, 2]
# Half-step rounding of the fp32 sum to bf16 on both sides, as the w4 test tolerates.
W1_TOL = dict(atol=3e-2, rtol=3e-2)


def _variants():
    try:
        import spark_kernels

        return list(range(spark_kernels.num_variants("w1gemm")))
    except Exception:
        return [0]


def _weight(K, N, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return (torch.randn(K, N, generator=g) / K**0.5).to(device=device, dtype=torch.bfloat16)


# ---- the reference quantizer, CPU --------------------------------------------------------------


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("shape", [(128, 16), (256, 48), (1152, 256)],
                         ids=lambda s: f"K{s[0]}_N{s[1]}")
def test_ref_w1_quantize_shapes(shape, bits):
    K, N = shape
    packed, scales = ref.w1_quantize(_weight(K, N), bits)
    assert packed.shape == (N, K * bits // 32) and packed.dtype == torch.int32
    assert scales.shape == (K // 128, N) and scales.dtype == torch.bfloat16
    codes = ref.w1_unpack(packed, bits)
    assert codes.shape == (K, N) and int(codes.max()) <= (1 << bits) - 1


def test_ref_w1_sign_bits_are_w_ge_0():
    w = _weight(384, 64, seed=1)
    w[0, 0] = 0.0  # zero is coded as positive
    packed, _ = ref.w1_quantize(w, bits=1)
    assert torch.equal(ref.w1_unpack(packed, 1).bool(), w.float() >= 0)
    # k ascends from the lowest bit of the lowest word of its column's row.
    first_word = packed[0, 0].to(torch.int64) & 0xFFFFFFFF
    for k in range(32):
        assert bool((first_word >> k) & 1) == bool(w[k, 0] >= 0)


def test_ref_w1_ternary_codes_match_thresholds():
    w = _weight(384, 64, seed=2)
    packed, scales = ref.w1_quantize(w, bits=2)
    codes = ref.w1_unpack(packed, 2).float()
    half = scales.float().repeat_interleave(128, dim=0) * 0.5
    wf = w.float()
    assert torch.equal(codes == 0, wf < -half)
    assert torch.equal(codes == 2, wf > half)
    assert torch.equal(codes == 1, wf.abs() <= half)
    # Two codes per word, lowest k lowest, in the first word of the first column.
    first_word = packed[0, 0].to(torch.int64) & 0xFFFFFFFF
    for k in range(16):
        assert float((first_word >> (2 * k)) & 3) == float(codes[k, 0])


@pytest.mark.parametrize("bits", BITS)
def test_ref_w1_scale_is_mean_abs(bits):
    w = _weight(512, 32, seed=3)
    w[:128, :16] = 0  # an all-zero group: scale 0, every code the zero point
    packed, scales = ref.w1_quantize(w, bits)
    expected = w.float().reshape(4, 128, 32).abs().mean(dim=1).to(torch.bfloat16)
    assert torch.equal(scales.view(torch.int16), expected.view(torch.int16))
    assert (scales[0, :16] == 0).all()
    deq = ref.w1_dequantize(packed, scales, bits)
    assert deq.shape == (512, 32) and deq.dtype == torch.bfloat16
    s = scales.float().repeat_interleave(128, dim=0)
    if bits == 1:
        # Every weight becomes +-s, so the mean magnitude of the group is s exactly.
        assert torch.equal(deq.float().abs(), s)
    else:
        # Every weight is in {-s, 0, +s}.
        assert ((deq.float().abs() == s) | (deq.float() == 0)).all()
    assert (deq[:128, :16] == 0).all()


def test_ref_w1_round_trip_small_matrix():
    # One group, one strip of 16 columns; column 0 holds a known pattern.
    w = torch.zeros(128, 16, dtype=torch.bfloat16)
    col = torch.tensor([1.0, -1.0, 0.25, -0.25, 2.0, -2.0, 0.0, 0.5] * 16)
    w[:, 0] = col
    w[:, 1] = -col
    packed1, s1 = ref.w1_quantize(w, bits=1)
    assert float(s1[0, 0]) == pytest.approx(col.abs().mean().item(), rel=2**-8)
    d1 = ref.w1_dequantize(packed1, s1, 1)
    assert torch.equal(d1[:, 0], torch.where(col >= 0, s1[0, 0], -s1[0, 0]))
    assert torch.equal(d1[:, 1], torch.where(-col >= 0, s1[0, 1], -s1[0, 1]))
    packed2, s2 = ref.w1_quantize(w, bits=2)
    d2 = ref.w1_dequantize(packed2, s2, 2)
    half = s2[0, 0].float() * 0.5
    want = torch.where(col > half, s2[0, 0], torch.where(col < -half, -s2[0, 0], 0.0))
    assert torch.equal(d2[:, 0], want.to(torch.bfloat16))
    # Untouched columns are all-zero groups: scale 0, dequantized to 0.
    assert (s1[0, 2:] == 0).all() and (d1[:, 2:] == 0).all() and (d2[:, 2:] == 0).all()
    # The GEMM against it is the plain product with the dequantized weight.
    a = torch.randn(3, 128).to(torch.bfloat16)
    got = ref.w1gemm_ref(a, packed2, s2, 2)
    torch.testing.assert_close(got.float(), a.float() @ d2.float(), **W1_TOL)


# ---- the extension against it, GPU ------------------------------------------------------------


@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("shape", [(128, 16), (256, 48), (1152, 256), (5120, 2304)],
                         ids=lambda s: f"K{s[0]}_N{s[1]}")
def test_w1_quantize_matches_reference(sk, shape, bits):
    K, N = shape
    w = _weight(K, N, device="cuda")
    w[:128, :16] = 0
    packed, scales = sk.w1_quantize(w, bits)
    p_ref, s_ref = ref.w1_quantize(w, bits)
    assert packed.shape == (N, K * bits // 32) and packed.dtype == torch.int32
    assert scales.shape == (K // 128, N) and scales.dtype == torch.bfloat16
    # Bit for bit: the same fp32 operations on both sides.
    assert torch.equal(scales.view(torch.int16), s_ref.view(torch.int16))
    assert torch.equal(packed, p_ref)


def _case(sk, M, K, N, bits, variant):
    w = _weight(K, N, seed=K + N, device="cuda")
    torch.manual_seed(M)
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    ww = sk.W1Weight.quantize(w, bits)
    assert ww.shape == (K, N) and ww.bits == bits
    packed, scales = ref.w1_quantize(w, bits)
    expected = ref.w1gemm_ref(a, packed, scales, bits)
    got = sk.w1gemm(a, ww, variant=variant)
    assert got.shape == (M, N) and got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), expected.float(), **W1_TOL)
    # The packed codes and scales passed on their own.
    got2 = sk.w1gemm(a, ww.packed, ww.scales, bits=bits, variant=variant)
    assert torch.equal(got, got2)


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("bits", BITS)
@pytest.mark.parametrize("M", MS)
@pytest.mark.parametrize("shape", WEIGHTS, ids=lambda s: f"K{s[0]}_N{s[1]}")
def test_w1gemm_matches_dequantized_matmul(sk, shape, M, bits, variant):
    K, N = shape
    _case(sk, M, K, N, bits, variant)


@pytest.mark.parametrize("bits", BITS)
def test_w1gemm_weight_bits_are_the_reference(sk, bits):
    # A one-hot row picks out one column of the weight: the GEMM returns the dequantized
    # weight itself, bit for bit, for every variant.
    K, N = 256, 32
    w = _weight(K, N, seed=7, device="cuda")
    ww = sk.W1Weight.quantize(w, bits)
    deq = ref.w1_dequantize(ww.packed, ww.scales, bits)
    a = torch.eye(K, device="cuda", dtype=torch.bfloat16)
    for variant in _variants():
        got = sk.w1gemm(a, ww, variant=variant)
        assert torch.equal(got, deq), f"variant {variant}"


def test_w1gemm_rejects_bad_input(sk):
    w = _weight(256, 32, device="cuda")
    ww = sk.W1Weight.quantize(w, 1)
    a = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        sk.w1_quantize(w, 3)
    with pytest.raises(RuntimeError):
        sk.w1_quantize(w[:100], 1)  # K not a multiple of 128
    with pytest.raises(RuntimeError):
        sk.w1gemm(a[:, :128], ww)  # K mismatch
    with pytest.raises(RuntimeError):
        sk.w1gemm(a, ww.packed, ww.scales, bits=2)  # packed sized for one bit
    with pytest.raises(RuntimeError):
        sk.w1gemm(a.float(), ww)
    with pytest.raises(RuntimeError):
        sk.w1gemm(a, ww.packed, ww.scales[:1], bits=1)
    with pytest.raises(ValueError):
        sk.w1gemm(a, ww, scales=ww.scales)


@pytest.mark.parametrize("variant", _variants())
@pytest.mark.parametrize("bits", BITS)
def test_w1gemm_moe_matches_per_expert(sk, bits, variant):
    E, K, N, T = 8, 2304, 1024, 200
    g = torch.Generator(device="cpu").manual_seed(E + K + N)
    w = (torch.randn(E, K, N, generator=g) / K**0.5).to(device="cuda", dtype=torch.bfloat16)
    ew = sk.W1ExpertWeights.quantize(w, bits)
    assert ew.packed.shape == (E, N, K * bits // 32) and ew.scales.shape == (E, K // 128, N)
    assert ew.shape == (E, K, N) and ew.num_experts == E
    # Random routing with experts 2 and 5 left empty.
    choices = torch.tensor([e for e in range(E) if e not in (2, 5)])
    route = choices[torch.randint(0, len(choices), (T,), generator=g)]
    counts = torch.bincount(route, minlength=E)
    offsets = torch.zeros(E + 1, dtype=torch.int32)
    offsets[1:] = counts.cumsum(0).to(torch.int32)
    assert int(offsets[-1]) == T and int(counts[2]) == 0 and int(counts[5]) == 0
    a = torch.randn(T, K, generator=g).to(device="cuda", dtype=torch.bfloat16)
    got = sk.w1gemm_moe(a, ew, offsets.cuda(), variant=variant)
    assert got.shape == (T, N) and got.dtype == torch.bfloat16
    for e in range(E):
        lo, hi = int(offsets[e]), int(offsets[e + 1])
        if lo == hi:
            continue
        want = sk.w1gemm(a[lo:hi], ew.expert(e), variant=variant)
        assert torch.equal(got[lo:hi], want), f"expert {e}"
        expected = ref.w1gemm_ref(a[lo:hi], ew.packed[e], ew.scales[e], bits)
        torch.testing.assert_close(got[lo:hi].float(), expected.float(), **W1_TOL)


def test_w1gemm_moe_rejects_bad_input(sk):
    E, K, N = 2, 256, 32
    w = torch.randn(E, K, N, device="cuda", dtype=torch.bfloat16)
    ew = sk.W1ExpertWeights.quantize(w, 1)
    a = torch.randn(6, K, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor([0, 4, 6], dtype=torch.int32, device="cuda")
    assert sk.w1gemm_moe(a, ew, offsets).shape == (6, N)
    with pytest.raises(RuntimeError):
        sk.w1gemm_moe(a, ew, offsets.to(torch.int64))
    with pytest.raises(RuntimeError):
        sk.w1gemm_moe(a, ew, offsets[:2])  # E + 1 entries needed
    with pytest.raises(RuntimeError):
        sk.w1gemm_moe(a, ew, offsets.cpu())
    with pytest.raises(TypeError):
        sk.w1gemm_moe(a, ew.expert(0), offsets)
