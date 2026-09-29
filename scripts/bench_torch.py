#!/usr/bin/env python3
"""Time spark_kernels ops against PyTorch eager on the same shapes as the C++ benches.

Writes results/torch_comparison.json (one JSON object per row) and prints a table.
Run on the GPU box (RTX 5090 or DGX Spark) after `pip install -e . --no-build-isolation`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shape_utils import attention_shape, gemm_shape, row_shape  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "torch_comparison.json"

# Same default shapes as the C++ benches (src/bench/bench_*.cu): the results table joins the two
# on (kernel, dtype, shape), so a shape that is only timed here never shows up.
ROWS = 4096
# also add_rmsnorm
RMSNORM_SHAPES = [(4096, 1024), (4096, 2048), (4096, 4096), (4096, 8192), (16384, 8192)]
SOFTMAX_COLS = [128, 1024, 4096, 16384]
SWIGLU_COLS = [2048, 5632, 11008, 14336]
# (M, N, K): C = A(MxK) @ B(KxN)
SGEMM_SHAPES = [(512, 512, 512), (1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096),
                (8192, 8192, 8192), (4096, 4096, 11008), (4096, 11008, 4096)]
HGEMM_SHAPES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192),
                (4096, 4096, 11008), (4096, 11008, 4096),
                (1, 4096, 4096), (16, 4096, 4096), (32, 4096, 4096), (64, 4096, 4096),
                (16, 11008, 4096), (64, 4096, 11008)]
FP8GEMM_SHAPES = HGEMM_SHAPES  # bench_fp8gemm.cu times bench_hgemm's list
# bench_fp4gemm.cu: squares, the Llama-3-8B projections at 4096 tokens (qkv, gate|up, down),
# decode shapes
FP4GEMM_SHAPES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192),
                  (4096, 6144, 4096), (4096, 28672, 4096), (4096, 4096, 14336), (1, 4096, 4096),
                  (16, 4096, 4096), (64, 4096, 4096), (16, 28672, 4096), (16, 4096, 14336)]
# The fused-epilogue rows of bench_hgemm.cu (`fused_rows`): (M, N, K, tag), N the width of
# b, so the swiglu rows have N = 2 x 11008 and an [M, 11008] output. The torch side is the
# eager sequence a model runs without the fusion: F.linear + F.gelu, torch.addmm, and two
# GEMMs + F.silu * up for the gate/up pair.
HGEMM_FUSED = [(4096, 4096, 4096, "bias+gelu"), (4096, 4096, 4096, "residual"),
               (4096, 11008, 4096, "bias+gelu"), (4096, 11008, 4096, "residual"),
               (4096, 22016, 4096, "swiglu"), (16, 4096, 4096, "bias+gelu"),
               (16, 4096, 4096, "residual"), (16, 11008, 4096, "bias+gelu"),
               (16, 11008, 4096, "residual"), (16, 22016, 4096, "swiglu")]
# (B, H_q, H_kv, S_q, S_kv, D, causal): the timed shapes of bench_attention.cu (its small
# validation shapes are timed there too but are not worth a torch line). H_kv < H_q is
# grouped-query attention (Llama-3-8B is 32/8), which torch's SDPA takes with enable_gqa=True.
ATTENTION_SHAPES = [(1, 32, 32, 4096, 4096, 128, 0), (1, 32, 32, 4096, 4096, 128, 1),
                    (1, 32, 32, 8192, 8192, 128, 1), (4, 32, 32, 2048, 2048, 128, 1),
                    (1, 32, 32, 4096, 4096, 64, 1), (1, 32, 32, 1, 4096, 128, 0),
                    (1, 32, 8, 4096, 4096, 128, 1), (1, 32, 8, 1, 4096, 128, 0),
                    (1, 32, 32, 1, 131072, 128, 0), (1, 32, 8, 1, 131072, 128, 0),
                    (8, 32, 8, 1, 4096, 128, 0)]
# (B, H_q, H_kv, S, D, causal): the timed prefill shapes of bench_attention_fp8.cu. The torch
# side is its bf16 SDPA on the unquantized inputs: torch has no fp8 attention to compare with,
# so the row says what the fp8 kernel buys over the bf16 kernel a model would otherwise call.
ATTENTION_FP8_SHAPES = [(1, 32, 32, s, 128, c) for s in (1024, 2048, 4096, 8192, 16384)
                        for c in (0, 1)] + [(1, 32, 8, 4096, 128, 0), (1, 32, 8, 4096, 128, 1),
                                            (4, 32, 32, 2048, 128, 1), (1, 32, 32, 4096, 64, 1)]
# (B, H_q, H_kv, S, D, causal): the timed backward shapes of bench_attention_bwd.cu (S_q = S_kv),
# 1K to 8K tokens with and without the mask, a batch of four, D = 64 and Llama-3-8B's 32 / 8.
ATTENTION_BWD_SHAPES = [(1, 32, 32, 1024, 128, 0), (1, 32, 32, 1024, 128, 1),
                        (1, 32, 32, 2048, 128, 0), (1, 32, 32, 2048, 128, 1),
                        (1, 32, 32, 4096, 128, 0), (1, 32, 32, 4096, 128, 1),
                        (1, 32, 32, 8192, 128, 0), (1, 32, 32, 8192, 128, 1),
                        (4, 32, 32, 2048, 128, 1), (1, 32, 32, 4096, 64, 1),
                        (1, 32, 8, 4096, 128, 0), (1, 32, 8, 4096, 128, 1),
                        (1, 32, 8, 8192, 128, 1)]
WARMUP, ITERS = 10, 100
SDPA_BACKEND = None  # --sdpa-backend: force torch's SDPA kernel instead of letting it choose


RAMP_MS = 300  # kRampMs in src/bench/bench_common.hpp
_ramped = False


def ramp_clocks(fn) -> None:
    """Spin the first op of the process for RAMP_MS, once, like the C++ harness: the RTX 5090
    idles at low clocks, and the first row would otherwise be timed while it is still ramping."""
    global _ramped
    if _ramped:
        return
    _ramped = True
    t0 = time.perf_counter()
    while (time.perf_counter() - t0) * 1e3 < RAMP_MS:
        fn()
        torch.cuda.synchronize()


def time_ms(fn, warmup=None, iters=None) -> float:
    # read the globals at call time so --warmup/--iters apply to every call site
    warmup = WARMUP if warmup is None else warmup
    iters = ITERS if iters is None else iters
    if warmup > 0:  # --warmup=0 also skips the ramp, as in the C++ benches
        ramp_clocks(fn)
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(iters):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def dtype_name(dtype) -> str:
    return "f32" if dtype == torch.float32 else "bf16"


def row_gbps(rows, cols, dtype, passes, ms) -> float:
    """Achieved GB/s for `passes` full reads/writes of a rows x cols tensor in `ms`."""
    isz = torch.tensor([], dtype=dtype).element_size()
    return rows * cols * isz * passes / ms / 1e6


def bench_rmsnorm(sk, add, dtype, rows, cols):
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    w = torch.ones(cols, device="cuda", dtype=dtype)
    shape = row_shape(rows, cols)

    ours = time_ms(lambda: sk.rmsnorm(x, w))
    ref = time_ms(lambda: F.rms_norm(x, (cols,), w, 1e-6))
    add("rmsnorm", dtype_name(dtype), shape, ours, ref, gbps=row_gbps(rows, cols, dtype, 2, ours))

    if dtype == torch.bfloat16:
        resid = torch.randn(rows, cols, device="cuda", dtype=dtype)
        ours = time_ms(lambda: sk.add_rmsnorm_(x, resid, w))

        def unfused():
            resid.add_(x)
            return F.rms_norm(resid, (cols,), w, 1e-6)

        ref = time_ms(unfused)
        # traffic: read x, read+write resid, write out = 4 passes
        add("add_rmsnorm", "bf16", shape, ours, ref, gbps=row_gbps(rows, cols, dtype, 4, ours))


def bench_softmax(sk, add, dtype, rows, cols):
    x = torch.randn(rows, cols, device="cuda", dtype=dtype)
    ours = time_ms(lambda: sk.softmax(x))
    ref = time_ms(lambda: torch.softmax(x, dim=-1))
    add("softmax", dtype_name(dtype), row_shape(rows, cols), ours, ref,
        gbps=row_gbps(rows, cols, dtype, 2, ours))


def bench_swiglu(sk, add, dtype, rows, cols):
    gate = torch.randn(rows, cols, device="cuda", dtype=dtype)
    up = torch.randn(rows, cols, device="cuda", dtype=dtype)
    ours = time_ms(lambda: sk.swiglu(gate, up))
    ref = time_ms(lambda: F.silu(gate) * up)
    # traffic: read gate, read up, write out = 3 passes
    add("swiglu", dtype_name(dtype), row_shape(rows, cols), ours, ref,
        gbps=row_gbps(rows, cols, dtype, 3, ours))


@contextlib.contextmanager
def matmul_tf32(enabled: bool):
    """torch's fp32 matmul precision for the block: "tf32" lets cuBLAS use the tensor cores in
    TF32 (the contract of sgemm variant 6), "ieee" keeps it on the CUDA cores. torch >= 2.9
    spells this torch.backends.cuda.matmul.fp32_precision; older releases have allow_tf32."""
    m = torch.backends.cuda.matmul
    if hasattr(m, "fp32_precision"):
        prev = m.fp32_precision
        m.fp32_precision = "tf32" if enabled else "ieee"
        try:
            yield
        finally:
            m.fp32_precision = prev
    else:
        prev = m.allow_tf32
        m.allow_tf32 = enabled
        try:
            yield
        finally:
            m.allow_tf32 = prev


def bench_sgemm(sk, add, M, N, K):
    a = torch.randn(M, K, device="cuda")
    b = torch.randn(K, N, device="cuda")
    shape = gemm_shape(M, N, K)
    flops = 2.0 * M * N * K
    with matmul_tf32(False):
        ours = time_ms(lambda: sk.sgemm(a, b))
        ref = time_ms(lambda: a @ b)
        add("sgemm", "f32", shape, ours, ref, tflops=flops / ours / 1e9)
        # 3xTF32 (variant 7) claims fp32 accuracy, so its reference is torch's fp32 matmul
        ours = time_ms(lambda: sk.sgemm(a, b, 7))
        add("sgemm", "3xtf32", shape, ours, ref, tflops=flops / ours / 1e9)
    # TF32 (variant 6) against torch with the tensor cores allowed: the same rounding contract
    with matmul_tf32(True):
        ours = time_ms(lambda: sk.sgemm(a, b, 6))
        ref = time_ms(lambda: a @ b)
        add("sgemm", "tf32", shape, ours, ref, tflops=flops / ours / 1e9)


def bench_hgemm(sk, add, M, N, K):
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    # Decode shapes (M <= 64) are bound by streaming B, which alone fits the RTX 5090's 96 MB
    # L2, so the loop rotates through enough copies of B to exceed L2, as bench_hgemm does.
    copies = (256 << 20) // (K * N * 2) + 1 if M <= 64 else 1
    bs = [torch.randn(K, N, device="cuda", dtype=torch.bfloat16) for _ in range(copies)]
    turn = [0]

    def next_b():
        b = bs[turn[0] % copies]
        turn[0] += 1
        return b

    ours = time_ms(lambda: sk.hgemm(a, next_b()))
    ref = time_ms(lambda: a @ next_b())
    add("hgemm", "bf16", gemm_shape(M, N, K), ours, ref, tflops=2.0 * M * N * K / ours / 1e9)


def bench_hgemm_fused(sk, add, M, N, K, tag):
    """One fused-epilogue row: ours in one kernel against eager torch's sequence, with our own
    unfused sequence (sk.hgemm then torch's add / our swiglu) as a third number."""
    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    copies = (256 << 20) // (K * N * 2) + 1 if M <= 64 else 1
    bs = [torch.randn(K, N, device="cuda", dtype=torch.bfloat16) for _ in range(copies)]
    turn = [0]

    def next_b():
        b = bs[turn[0] % copies]
        turn[0] += 1
        return b

    if tag == "swiglu":
        # The interleaved weight is the same bytes as gate and up side by side; the eager
        # form runs the two projections separately, as an HF Llama MLP does.
        gates = [b[:, 0::2].contiguous() for b in bs]
        ups = [b[:, 1::2].contiguous() for b in bs]

        def next_gu():
            i = turn[0] % copies
            turn[0] += 1
            return gates[i], ups[i]

        ours = time_ms(lambda: sk.hgemm_swiglu(a, next_b()))

        def unfused():
            g, u = next_gu()
            return sk.swiglu(sk.hgemm(a, g), sk.hgemm(a, u))

        def eager():
            g, u = next_gu()
            return F.silu(a @ g) * (a @ u)

    elif tag == "bias+gelu":
        bias = torch.randn(N, device="cuda", dtype=torch.bfloat16)
        ours = time_ms(lambda: sk.hgemm(a, next_b(), bias=bias, act="gelu"))

        def unfused():
            return F.gelu(sk.hgemm(a, next_b()) + bias, approximate="tanh")

        def eager():  # F.linear takes the weight as [N, K]; b.t() is that without a copy
            return F.gelu(F.linear(a, next_b().t(), bias), approximate="tanh")

    elif tag == "residual":
        res = torch.randn(M, N, device="cuda", dtype=torch.bfloat16)
        ours = time_ms(lambda: sk.hgemm(a, next_b(), residual=res))

        def unfused():
            return sk.hgemm(a, next_b()) + res

        def eager():
            return torch.addmm(res, a, next_b())

    else:
        raise ValueError(f"unknown fused row tag {tag!r}")
    unfused_ms = time_ms(unfused)
    ref = time_ms(eager)
    add("hgemm", "bf16", gemm_shape(M, N, K) + "+" + tag, ours, ref,
        tflops=2.0 * M * N * K / ours / 1e9, unfused_ms=unfused_ms)


def bench_fp8gemm(sk, add, M, N, K):
    # a [M, K] and b_t [N, K] e4m3, per-tensor fp32 scales, bf16 out. torch._scaled_mm is
    # cuBLASLt and takes b_t.t() (column-major [K, N]), the same bytes in the same layout.
    a = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    scale_a = torch.tensor(0.75, device="cuda")
    scale_b = torch.tensor(1.5, device="cuda")
    # Decode shapes (M <= 64) stream b_t, which alone fits the RTX 5090's 96 MB L2 (16 MB for
    # 4096^2 in e4m3), so the loop rotates through enough copies to exceed L2.
    copies = (256 << 20) // (K * N) + 1 if M <= 64 else 1
    bts = [torch.randn(N, K, device="cuda").to(torch.float8_e4m3fn) for _ in range(copies)]
    turn = [0]

    def next_bt():
        b_t = bts[turn[0] % copies]
        turn[0] += 1
        return b_t

    ours = time_ms(lambda: sk.fp8gemm(a, next_bt(), scale_a, scale_b))
    ref = time_ms(lambda: torch._scaled_mm(a, next_bt().t(), scale_a=scale_a, scale_b=scale_b,
                                           out_dtype=torch.bfloat16))
    add("fp8gemm", "e4m3", gemm_shape(M, N, K), ours, ref, tflops=2.0 * M * N * K / ours / 1e9)


def bench_fp4gemm(sk, add, M, N, K):
    # NVFP4: a [M, K/2] and b_t [N, K/2] packed e2m1 from sk.fp4_quantize, e4m3 block scales
    # in the blocked layout, per-tensor fp32 scales, bf16 out. torch's side is
    # F.scaled_mm with the two scale levels (cuBLASLt's NVFP4 kernel, the same bytes and the
    # same output bits). Skipped where torch has no NVFP4 scaled_mm.
    a, sfa, sa = sk.fp4_quantize(torch.randn(M, K, device="cuda", dtype=torch.bfloat16))
    copies = (256 << 20) // (K * N // 2) + 1 if M <= 64 else 1
    bs = [sk.fp4_quantize(torch.randn(N, K, device="cuda", dtype=torch.bfloat16))
          for _ in range(copies)]
    turn = [0]

    def next_b():
        b = bs[turn[0] % copies]
        turn[0] += 1
        return b

    def torch_mm():
        b_t, sfb, sb = next_b()
        return F.scaled_mm(a.view(torch.float4_e2m1fn_x2),
                           b_t.view(torch.float4_e2m1fn_x2).t(),
                           [sfa.view(torch.float8_e4m3fn), sa.reshape(1)],
                           [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
                           [sfb.view(torch.float8_e4m3fn), sb.reshape(1)],
                           [F.ScalingType.BlockWise1x16, F.ScalingType.TensorWise],
                           swizzle_a=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
                           swizzle_b=[F.SwizzleType.SWIZZLE_32_4_4, F.SwizzleType.NO_SWIZZLE],
                           output_dtype=torch.bfloat16)

    try:
        torch_mm()
    except (AttributeError, RuntimeError, TypeError, ValueError, NotImplementedError):
        print(f"fp4gemm {M}x{N}x{K}: torch has no NVFP4 scaled_mm here, skipped", file=sys.stderr)
        return

    def ours():
        b_t, sfb, sb = next_b()
        return sk.fp4gemm(a, b_t, sfa, sfb, sa, sb)

    ours_ms = time_ms(ours)
    ref = time_ms(torch_mm)
    add("fp4gemm", "nvfp4", gemm_shape(M, N, K), ours_ms, ref,
        tflops=2.0 * M * N * K / ours_ms / 1e9)


def sdpa_backend(fn) -> str:
    """Which kernel torch's SDPA dispatcher picked for `fn`, from the profiler's kernel names:
    "flash" (FlashAttention-2), "cudnn", "efficient" (the CUTLASS memory-efficient kernel) or
    "math". Empty if the profiler is unavailable."""
    try:
        from torch.profiler import ProfilerActivity, profile

        with profile(activities=[ProfilerActivity.CUDA]) as p:
            fn()
            torch.cuda.synchronize()
        names = " ".join(e.name for e in p.events() if e.device_type.name == "CUDA")
    except Exception:
        return ""
    for tag, keys in (("flash", ("flash_fwd", "flash_bwd")), ("cudnn", ("cudnn",)),
                      ("efficient", ("fmha_cutlass",))):
        if any(key in names for key in keys):
            return tag
    return "math" if names else ""


def bench_attention(sk, add, B, Hq, Hkv, Sq, Skv, D, causal):
    q = torch.randn(B, Hq, Sq, D, device="cuda", dtype=torch.bfloat16)
    # A decode step (S_q <= 64) streams K and V, and one layer's cache fits the RTX 5090's
    # 96 MB L2, so the loop rotates through copies that exceed it, as bench_attention does.
    kv_bytes = 2 * B * Hkv * Skv * D * 2
    copies = (256 << 20) // kv_bytes + 1 if Sq <= 64 else 1
    kvs = [(torch.randn(B, Hkv, Skv, D, device="cuda", dtype=torch.bfloat16),
            torch.randn(B, Hkv, Skv, D, device="cuda", dtype=torch.bfloat16))
           for _ in range(copies)]
    turn = [0]
    gqa = Hkv != Hq  # torch >= 2.5 broadcasts the K/V heads itself with enable_gqa=True

    def next_kv():
        kv = kvs[turn[0] % copies]
        turn[0] += 1
        return kv

    def ours():
        k, v = next_kv()
        return sk.attention(q, k, v, causal=bool(causal))

    def theirs():
        k, v = next_kv()
        if SDPA_BACKEND is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=bool(causal), enable_gqa=gqa)
        from torch.nn.attention import sdpa_kernel

        with sdpa_kernel(SDPA_BACKEND):
            return F.scaled_dot_product_attention(q, k, v, is_causal=bool(causal), enable_gqa=gqa)

    ours_ms = time_ms(ours)
    ref = time_ms(theirs)
    flops = 4.0 * B * Hq * Sq * Skv * D * (0.5 if causal else 1.0)
    # GB/s of the traffic floor: Q and O once, K and V once per K/V head (not per query head)
    gbps = (2 * B * Hq * Sq * D * 2 + kv_bytes) / ours_ms / 1e6
    add("attention", "bf16", attention_shape(B, Hq, Sq, Skv, D, bool(causal), H_kv=Hkv), ours_ms,
        ref, gbps=gbps, tflops=flops / ours_ms / 1e9, backend=sdpa_backend(theirs))


def bench_attention_fp8(sk, add, B, Hq, Hkv, S, D, causal):
    q = torch.randn(B, Hq, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    (q8, sq), (k8, sk_), (v8, sv) = (sk.quantize_fp8(x) for x in (q, k, v))
    gqa = Hkv != Hq

    def theirs():
        if SDPA_BACKEND is None:
            return F.scaled_dot_product_attention(q, k, v, is_causal=bool(causal), enable_gqa=gqa)
        from torch.nn.attention import sdpa_kernel

        with sdpa_kernel(SDPA_BACKEND):
            return F.scaled_dot_product_attention(q, k, v, is_causal=bool(causal), enable_gqa=gqa)

    ours_ms = time_ms(lambda: sk.attention_fp8(q8, k8, v8, sq, sk_, sv, causal=bool(causal)))
    ref = time_ms(theirs)
    flops = 4.0 * B * Hq * S * S * D * (0.5 if causal else 1.0)
    gbps = (3 * B * Hq * S * D + 2 * B * Hkv * S * D) / ours_ms / 1e6  # e4m3 in, bf16 out
    add("attention_fp8", "e4m3", attention_shape(B, Hq, S, S, D, bool(causal), H_kv=Hkv),
        ours_ms, ref, gbps=gbps, tflops=flops / ours_ms / 1e9, backend=sdpa_backend(theirs))


def bench_attention_bwd(sk, add, B, Hq, Hkv, S, D, causal):
    """The backward pass alone: ours from the saved (out, lse), torch's as autograd.grad of
    its SDPA output (the same kernels loss.backward() runs, preprocess and dq conversion
    included on both sides)."""
    q = torch.randn(B, Hq, S, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, S, D, device="cuda", dtype=torch.bfloat16)
    do = torch.randn(B, Hq, S, D, device="cuda", dtype=torch.bfloat16)
    gqa = Hkv != Hq
    out, lse = sk.attention_fwd(q, k, v, causal=bool(causal))

    def ours():
        return sk.attention_bwd(q, k, v, out, do, lse, causal=bool(causal))

    qr, kr, vr = (t.clone().requires_grad_() for t in (q, k, v))
    backend = SDPA_BACKEND
    ctx = contextlib.nullcontext()
    if backend is not None:
        from torch.nn.attention import sdpa_kernel

        ctx = sdpa_kernel(backend)
    with ctx:
        o = F.scaled_dot_product_attention(qr, kr, vr, is_causal=bool(causal), enable_gqa=gqa)

    def theirs():
        return torch.autograd.grad(o, (qr, kr, vr), do, retain_graph=True)

    ours_ms = time_ms(ours)
    ref = time_ms(theirs)
    # the five products of the backward: 2.5x the forward's 4 B H S S D, halved under the mask
    flops = 10.0 * B * Hq * S * S * D * (0.5 if causal else 1.0)
    # Q, O, dO read and dQ written per query head; K, V read and dK, dV written per K/V head
    gbps = 2 * 4 * B * (Hq + Hkv) * S * D / ours_ms / 1e6
    add("attention_bwd", "bf16", attention_shape(B, Hq, S, S, D, bool(causal), H_kv=Hkv),
        ours_ms, ref, gbps=gbps, tflops=flops / ours_ms / 1e9, backend=sdpa_backend(theirs))


def positive_int(text: str) -> int:
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")  # median of no samples is undefined
    return n


def non_negative_int(text: str) -> int:
    n = int(text)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def main() -> int:
    global WARMUP, ITERS, SDPA_BACKEND
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sdpa-backend", choices=["flash", "cudnn", "efficient", "math"],
                    help="force this SDPA kernel for the attention rows instead of torch's own "
                         "choice (the default, which the JSON records as torch_backend)")
    ap.add_argument("--only", choices=["attention", "attention_fp8", "attention_bwd",
                                       "hgemm_fused", "sgemm", "fp4gemm"],
                    help="time only this kernel's rows (with --sdpa-backend: the attention "
                         "rows against that kernel; hgemm_fused: the fused-epilogue rows; "
                         "attention_fp8: the fp8 kernel against torch's bf16 SDPA; "
                         "attention_bwd: the backward rows)")
    ap.add_argument("--iters", type=positive_int, default=ITERS,
                    help="timed iterations per op (default: %(default)s, same as the C++ benches)")
    ap.add_argument("--warmup", type=non_negative_int, default=WARMUP,
                    help="untimed iterations before each measurement (default: %(default)s); "
                         "0 also skips the one-time clock ramp")
    args = ap.parse_args()
    WARMUP, ITERS = args.warmup, args.iters
    if args.sdpa_backend:
        from torch.nn.attention import SDPBackend

        SDPA_BACKEND = {"flash": SDPBackend.FLASH_ATTENTION, "cudnn": SDPBackend.CUDNN_ATTENTION,
                        "efficient": SDPBackend.EFFICIENT_ATTENTION,
                        "math": SDPBackend.MATH}[args.sdpa_backend]
    if not torch.cuda.is_available():
        print("CUDA not available", file=sys.stderr)
        return 1
    import spark_kernels as sk

    rows_out: list[dict] = []
    device = torch.cuda.get_device_name()
    print(f"device: {device} | cc {torch.cuda.get_device_capability()}", file=sys.stderr)

    def add(kernel, dtype, shape, ours_ms, torch_ms, gbps=None, tflops=None, backend=None,
            unfused_ms=None):
        r = {
            "device": device,
            "kernel": kernel,
            "dtype": dtype,
            "shape": shape,
            "spark_ms": ours_ms,
            "torch_ms": torch_ms,
            "speedup": torch_ms / ours_ms if ours_ms > 0 else 0.0,
        }
        if gbps is not None:
            r["gbps"] = gbps
        if tflops is not None:
            r["tflops"] = tflops
        if backend:
            r["torch_backend"] = backend  # which SDPA kernel torch picked for this shape
        if unfused_ms is not None:
            r["spark_unfused_ms"] = unfused_ms  # our GEMM followed by the separate op(s)
        rows_out.append(r)
        print(
            f"{kernel:12s} {dtype:5s} {shape:26s} spark {ours_ms:9.4f} ms  torch {torch_ms:9.4f} ms"
            f"  x{r['speedup']:.2f}" + (f"  ({backend})" if backend else "")
            + (f"  (unfused ours {unfused_ms:.4f} ms)" if unfused_ms is not None else ""),
            file=sys.stderr,
        )

    if args.only == "hgemm_fused":
        for M, N, K, tag in HGEMM_FUSED:
            bench_hgemm_fused(sk, add, M, N, K, tag)
    if args.only == "sgemm":
        for M, N, K in SGEMM_SHAPES:
            bench_sgemm(sk, add, M, N, K)
        return 0
    if args.only == "fp4gemm":
        for M, N, K in FP4GEMM_SHAPES:
            bench_fp4gemm(sk, add, M, N, K)
        return 0
    if args.only not in ("attention", "attention_fp8", "attention_bwd"):
        for dtype in (torch.float32, torch.bfloat16):
            for rows, cols in RMSNORM_SHAPES:
                bench_rmsnorm(sk, add, dtype, rows, cols)
            for cols in SOFTMAX_COLS:
                bench_softmax(sk, add, dtype, ROWS, cols)
            for cols in SWIGLU_COLS:
                bench_swiglu(sk, add, dtype, ROWS, cols)
        for M, N, K in SGEMM_SHAPES:
            bench_sgemm(sk, add, M, N, K)
        for M, N, K in HGEMM_SHAPES:
            bench_hgemm(sk, add, M, N, K)
        for M, N, K, tag in HGEMM_FUSED:
            bench_hgemm_fused(sk, add, M, N, K, tag)
        for M, N, K in FP8GEMM_SHAPES:
            bench_fp8gemm(sk, add, M, N, K)
        for M, N, K in FP4GEMM_SHAPES:
            bench_fp4gemm(sk, add, M, N, K)
    if args.only not in ("attention_fp8", "attention_bwd"):
        for B, Hq, Hkv, Sq, Skv, D, causal in ATTENTION_SHAPES:
            bench_attention(sk, add, B, Hq, Hkv, Sq, Skv, D, causal)
    if args.only in (None, "attention_fp8"):
        for B, Hq, Hkv, S, D, causal in ATTENTION_FP8_SHAPES:
            bench_attention_fp8(sk, add, B, Hq, Hkv, S, D, causal)
    if args.only in (None, "attention_bwd"):
        for B, Hq, Hkv, S, D, causal in ATTENTION_BWD_SHAPES:
            bench_attention_bwd(sk, add, B, Hq, Hkv, S, D, causal)
    if args.only or args.sdpa_backend:
        return 0  # a partial or forced-backend run is not the results table's input

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w") as f:
        for r in rows_out:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {OUT}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
