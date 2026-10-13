"""Parity of sk.grouped_gemm with the per-expert loop (reference.grouped_gemm): a sorted by
expert with int32 offsets, w [E, N, K], bf16 and e4m3. The routings cover empty experts, an
expert with one row, experts of more than one 128-row tile, all rows on one expert, more
experts than one warp scans at a time, and shapes whose tile count is below, at and above
one wave so that both the whole tiles and the Stream-K chains run."""

import pytest
import torch

from spark_kernels import reference as ref

TOL = dict(atol=3e-2, rtol=3e-2)


def _offsets(counts, device="cuda"):
    offs = [0]
    for c in counts:
        offs.append(offs[-1] + c)
    return torch.tensor(offs, dtype=torch.int32, device=device)


def _routed_counts(T, E, k, seed):
    ids = ref.moe_route(T, E, k, seed=seed, device="cpu")
    return torch.bincount(ids.reshape(-1), minlength=E).tolist()


# (counts per expert, N, K)
CASES = [
    ([0, 1, 0, 300, 129, 128, 0, 7], 256, 512),    # empty, 1 row, ragged and exact tiles
    ([1000], 384, 256),                             # one expert
    ([0, 0, 0, 5], 128, 4096),                      # few tiles, long K: Stream-K chains
    (_routed_counts(256, 40, 4, 1), 256, 256),      # 40 experts: two scan rounds
    (_routed_counts(512, 128, 8, 2), 768, 2048),    # Qwen3-30B-A3B experts, small batch
    (_routed_counts(1024, 8, 2, 3), 1792, 1024),    # Mixtral-like routing, several waves
]


def _case_id(c):
    counts, N, K = c
    return f"E{len(counts)}-T{sum(counts)}-N{N}-K{K}"


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_grouped_gemm_bf16(sk, case):
    counts, N, K = case
    E = len(counts)
    torch.manual_seed(0)
    offsets = _offsets(counts)
    T = sum(counts)
    a = torch.randn(T, K, device="cuda").to(torch.bfloat16)
    w = (torch.randn(E, N, K, device="cuda") / K**0.5).to(torch.bfloat16)
    got = sk.grouped_gemm(a, w, offsets)
    want = ref.grouped_gemm(a, w, offsets)
    assert got.shape == (T, N) and got.dtype == torch.bfloat16
    torch.testing.assert_close(got.float(), want.float(), **TOL)


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_grouped_gemm_fp8(sk, case):
    counts, N, K = case
    if K % 128:
        pytest.skip("fp8 needs K % 128 == 0")
    E = len(counts)
    torch.manual_seed(1)
    offsets = _offsets(counts)
    T = sum(counts)
    a = torch.randn(T, K, device="cuda").to(torch.float8_e4m3fn)
    w = torch.randn(E, N, K, device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.tensor([0.5], device="cuda")
    scale_b = (torch.rand(E, device="cuda") + 0.5) / K**0.5
    got = sk.grouped_gemm(a, w, offsets, scale_a, scale_b)
    want = ref.grouped_gemm(a, w, offsets, scale_a, scale_b)
    torch.testing.assert_close(got.float(), want.float(), **TOL)
    # No scales: 1.0. Sums of K products of N(0, 1) values, so the absolute slack scales
    # with sqrt(K).
    got1 = sk.grouped_gemm(a, w, offsets)
    want1 = ref.grouped_gemm(a, w, offsets)
    torch.testing.assert_close(got1.float(), want1.float(), atol=0.02 * K**0.5, rtol=3e-2)


def test_grouped_gemm_deterministic(sk):
    counts = _routed_counts(2048, 8, 2, 4)
    offsets = _offsets(counts)
    torch.manual_seed(2)
    a = torch.randn(sum(counts), 1024, device="cuda").to(torch.bfloat16)
    w = (torch.randn(8, 1408, 1024, device="cuda") / 32).to(torch.bfloat16)
    first = sk.grouped_gemm(a, w, offsets)
    for _ in range(3):
        assert torch.equal(sk.grouped_gemm(a, w, offsets), first)


def test_moe_offsets_matches_sort(sk):
    ids = ref.moe_route(300, 16, 4, seed=5)
    order, offsets = sk.moe_offsets(ids, 16)
    flat = ids.reshape(-1)
    assert torch.equal(flat[order], torch.sort(flat, stable=True).values)
    counts = torch.bincount(flat, minlength=16)
    assert offsets[0].item() == 0
    assert torch.equal(offsets[1:].long(), torch.cumsum(counts, 0))


def test_grouped_gemm_rejects(sk):
    a = torch.randn(64, 256, device="cuda").to(torch.bfloat16)
    w = torch.randn(4, 200, 256, device="cuda").to(torch.bfloat16)  # N % 128 != 0
    with pytest.raises(RuntimeError):
        sk.grouped_gemm(a, w, _offsets([16, 16, 16, 16]))
    w = torch.randn(4, 256, 256, device="cuda").to(torch.bfloat16)
    with pytest.raises(RuntimeError):
        sk.grouped_gemm(a, w, _offsets([16, 16, 32]))  # offsets not [E + 1]
