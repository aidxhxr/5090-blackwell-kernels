"""The unified-memory ops through the Python bindings: the decode GEMV against torch on every
variant, the streaming read's sum against torch.sum in fp64, over the kinds of memory a torch
tensor can be in.

Kinds: "device" is a CUDA tensor and always runs; "host_mapped" is a pinned CPU tensor
(cudaHostAlloc memory, which the GPU reads in place under unified addressing) and runs
wherever the device can map host memory; "system" is a plain CPU tensor and runs only where
`unified_device_caps()` reports pageable_memory_access (HMM on the 5090 with the open kernel
modules, ATS on the GB10). Everything skips without CUDA or the extension (conftest)."""

from __future__ import annotations

import pytest
import torch

KINDS = ["device", "host_mapped", "system"]
# (N, K): the DeepSeek V4.1 Flash expert, a Llama-3-8B projection, and odd row counts for
# the tails of the 4-row and persistent variants.
GEMV_SHAPES = [(2304, 5120), (4096, 4096), (37, 256), (1, 8), (4097, 1024)]
READ_SIZES = [8, 1000, 1 << 20, (1 << 20) + 5]
GEMV_TOL = dict(atol=2e-2, rtol=2e-2)


def _variants(name):
    try:
        import spark_kernels

        return list(range(spark_kernels.unified_num_variants(name)))
    except Exception:  # extension not built / no GPU: the tests skip anyway
        return [0]


def _caps(sk):
    return sk.unified_device_caps()


def _place(sk, t: torch.Tensor, kind: str) -> torch.Tensor:
    """A copy of the CPU tensor `t` in the memory `kind`, or a skip where the device cannot
    read that kind."""
    caps = _caps(sk)
    if kind == "device":
        return t.cuda()
    if kind == "host_mapped":
        if not caps["can_map_host_memory"] or not caps["unified_addressing"]:
            pytest.skip("device cannot map pinned host memory")
        return t.pin_memory()
    if kind == "system":
        if not caps["pageable_memory_access"]:
            pytest.skip("no pageableMemoryAccess: plain CPU memory is not GPU-readable here")
        return t.clone()
    raise ValueError(kind)


def test_device_caps_shape(sk):
    caps = _caps(sk)
    for key in ("device", "sm_count", "unified_addressing", "integrated", "can_map_host_memory",
                "pageable_memory_access", "pageable_memory_access_uses_host_page_tables",
                "concurrent_managed_access", "direct_managed_mem_access_from_host",
                "host_native_atomic_supported", "can_use_host_pointer_for_registered_mem"):
        assert key in caps, key
    assert caps["sm_count"] == torch.cuda.get_device_properties(0).multi_processor_count
    assert caps["unified_addressing"], "every 64-bit CUDA machine has unified addressing"


def test_num_variants(sk):
    assert sk.unified_num_variants("unified_read") >= 3
    assert sk.unified_num_variants("unified_gemv") >= 3
    with pytest.raises(ValueError):
        sk.unified_num_variants("unified_nothing")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("variant", _variants("unified_read"))
@pytest.mark.parametrize("n", READ_SIZES)
def test_read_sum_matches_fp64(sk, n, variant, kind):
    torch.manual_seed(n)
    x_cpu = torch.randn(n).to(torch.bfloat16)
    x = _place(sk, x_cpu, kind)
    got = sk.unified_read_sum(x, variant)
    assert got.dtype == torch.float64 and got.dim() == 0
    want = x_cpu.double().sum()
    # fp32 partials over a few thousand elements per thread at most, then fp64.
    tol = 1e-5 * float(x_cpu.double().abs().sum()) + 1e-3
    assert abs(float(got) - float(want)) <= tol, (float(got), float(want), tol)


def test_read_sum_partials_cover_every_element(sk):
    # A one-hot array: the sum is the single element wherever it sits, so a stride or tail
    # bug in any variant drops it to zero.
    n = (1 << 16) + 3
    for pos in (0, 7, n - 1, n - 8, n // 2):
        x = torch.zeros(n, dtype=torch.bfloat16, device="cuda")
        x[pos] = 3.0
        for v in _variants("unified_read"):
            assert float(sk.unified_read_sum(x, v)) == 3.0, (pos, v)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("variant", _variants("unified_gemv"))
@pytest.mark.parametrize("shape", GEMV_SHAPES, ids=lambda s: f"N{s[0]}_K{s[1]}")
def test_gemv_matches_torch(sk, shape, variant, kind):
    N, K = shape
    torch.manual_seed(N + K)
    w_cpu = (torch.randn(N, K) / K**0.5).to(torch.bfloat16)
    a = torch.randn(K, device="cuda", dtype=torch.bfloat16)
    w = _place(sk, w_cpu, kind)
    got = sk.unified_gemv(a, w, variant)
    assert got.shape == (N,) and got.dtype == torch.bfloat16 and got.is_cuda
    want = (w_cpu.cuda().float() @ a.float()).to(torch.bfloat16)
    torch.testing.assert_close(got.float(), want.float(), **GEMV_TOL)
    # The [1, K] form returns [1, N].
    got2 = sk.unified_gemv(a[None, :], w, variant)
    assert got2.shape == (1, N)
    assert torch.equal(got2[0], got)


def test_gemv_one_hot_reads_the_right_column(sk):
    # a = e_k picks column k of W: the output is W[:, k] exactly, for every variant.
    N, K = 96, 256
    w = (torch.randn(N, K, device="cuda") / 4).to(torch.bfloat16)
    for k in (0, 7, 8, K - 1):
        a = torch.zeros(K, device="cuda", dtype=torch.bfloat16)
        a[k] = 1.0
        for v in _variants("unified_gemv"):
            assert torch.equal(sk.unified_gemv(a, w, v), w[:, k].contiguous()), (k, v)


def test_gemv_rejects_bad_input(sk):
    w = torch.randn(16, 24, device="cuda", dtype=torch.bfloat16)
    a = torch.randn(24, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError):
        sk.unified_gemv(a, torch.randn(16, 12, device="cuda", dtype=torch.bfloat16))  # K mismatch
    with pytest.raises(RuntimeError):
        sk.unified_gemv(a.float(), w)  # dtype
    with pytest.raises(RuntimeError):
        sk.unified_gemv(a[:20], w[:, :20])  # K % 8 != 0 (and the slice is not contiguous)
    with pytest.raises(RuntimeError):
        sk.unified_gemv(a.cpu(), w)  # a must be on the device
    if not _caps(sk)["pageable_memory_access"]:
        with pytest.raises(RuntimeError):
            sk.unified_gemv(a, w.cpu())  # plain CPU memory the GPU cannot read here
    with pytest.raises((RuntimeError, ValueError)):
        sk.unified_gemv(a, w, variant=99)
