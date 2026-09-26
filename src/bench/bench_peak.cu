// Measured compute peaks for the roofline: what this GPU actually sustains, at the clocks it
// actually runs at under load, rather than a spec-sheet number at a spec-sheet boost clock.
//
//   peak_bf16_mma   dense bf16 tensor-core throughput, fp32 accumulate (mma.sync.m16n8k16),
//                   register-resident operands, no memory traffic: the hgemm roof
//   peak_fp8_mma    the same for e4m3 operands, fp32 accumulate, on the instruction the fp8gemm
//                   kernels use (the block-scaled mma.sync.m16n8k32.kind::mxf8f6f4 with unit
//                   scales on an sm_120a build): the fp8gemm roof
//   peak_fp8_mma_plain    the plain mma.sync.m16n8k32...e4m3 with fp32 accumulate, which the
//                   RTX 5090 runs at half that rate, as the GeForce Ada parts did
//   peak_fp8_mma_f16acc   the plain instruction with fp16 accumulators, the full rate
//   peak_fp32_fma   fp32 FMA throughput on the CUDA cores: the sgemm roof
//   sm_clock        SM clock observed *during* the mma loop (clock64 / globaltimer), so the
//                   peaks can be related to the spec clock
//
// NVIDIA publishes no dense bf16 figure for the RTX 5090, and the card boosts well above its
// 2.41 GHz spec clock until it hits the power limit, so both peaks in the results tables come
// from here (scripts/shape_utils.py reads results/peak.json).
//
//   ./bench_peak [--iters=5] [--ms=200]     --ms: target runtime of each timed launch

#include "bench_common.hpp"
#include "spark/common.cuh"

using namespace spark;
using namespace spark::bench;

namespace {

constexpr int kThreads = 256;
constexpr int kChains = 8;  // independent accumulators per thread / warp: hides mma/FMA latency

// One m16n8k16 bf16 mma per chain per iteration, operands never change (registers only).
__global__ void __launch_bounds__(kThreads)
    mma_peak_kernel(int iters, float* __restrict__ sink, long long* __restrict__ clk) {
    unsigned a0, a1, a2, a3, b0, b1;
    {
        const __nv_bfloat162 one = __floats2bfloat162_rn(1.0f, 1.0f);
        const unsigned u = *reinterpret_cast<const unsigned*>(&one);
        a0 = a1 = a2 = a3 = b0 = b1 = u ^ (threadIdx.x & 1);  // keep the compiler honest
    }
    float acc[kChains][4];
#pragma unroll
    for (int c = 0; c < kChains; ++c) acc[c][0] = acc[c][1] = acc[c][2] = acc[c][3] = 0.f;

    long long c0 = 0, t0 = 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        c0 = clock64();
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    }
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int c = 0; c < kChains; ++c) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(acc[c][0]), "+f"(acc[c][1]), "+f"(acc[c][2]), "+f"(acc[c][3])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        long long c1 = clock64(), t1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        clk[0] = c1 - c0;
        clk[1] = t1 - t0;
    }
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < kChains; ++c) s += acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3];
    if (s == 12345.f) sink[threadIdx.x] = s;  // never true; defeats dead-code elimination
}

// One m16n8k32 e4m3 mma per chain per iteration, fp32 accumulators. 0x38 is e4m3 1.0.
// PLAIN selects the plain instruction over the block-scaled one the kernels use.
template <bool PLAIN>
__global__ void __launch_bounds__(kThreads)
    mma_fp8_peak_kernel(int iters, float* __restrict__ sink, long long* __restrict__ clk) {
    const unsigned u = 0x38383838u ^ (threadIdx.x & 1);
    const unsigned a[4] = {u, u, u, u};
    const unsigned b[2] = {u, u};
    float acc[kChains][4];
#pragma unroll
    for (int c = 0; c < kChains; ++c) acc[c][0] = acc[c][1] = acc[c][2] = acc[c][3] = 0.f;

    long long c0 = 0, t0 = 0;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        c0 = clock64();
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    }
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int c = 0; c < kChains; ++c) {
            if (PLAIN)
                mma_e4m3_16832_plain(acc[c], a, b);
            else
                mma_e4m3_16832(acc[c], a, b);
        }
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        long long c1 = clock64(), t1;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
        clk[0] = c1 - c0;
        clk[1] = t1 - t0;
    }
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < kChains; ++c) s += acc[c][0] + acc[c][1] + acc[c][2] + acc[c][3];
    if (s == 12345.f) sink[threadIdx.x] = s;
}

// The same with fp16 accumulators (two f16x2 registers per chain).
__global__ void __launch_bounds__(kThreads)
    mma_fp8_f16acc_peak_kernel(int iters, float* __restrict__ sink) {
    unsigned a0, a1, a2, a3, b0, b1;
    a0 = a1 = a2 = a3 = b0 = b1 = 0x38383838u ^ (threadIdx.x & 1);
    unsigned acc[kChains][2];
#pragma unroll
    for (int c = 0; c < kChains; ++c) acc[c][0] = acc[c][1] = 0u;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int c = 0; c < kChains; ++c) {
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 "
                "{%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                : "+r"(acc[c][0]), "+r"(acc[c][1])
                : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        }
    }
    unsigned s = 0u;
#pragma unroll
    for (int c = 0; c < kChains; ++c) s += acc[c][0] + acc[c][1];
    if (s == 12345u) sink[threadIdx.x] = static_cast<float>(s);
}

// kChains independent FMA chains per thread.
__global__ void __launch_bounds__(kThreads) fma_peak_kernel(int iters, float* __restrict__ sink) {
    float acc[kChains];
#pragma unroll
    for (int c = 0; c < kChains; ++c) acc[c] = 0.f;
    const float a = 1.0f + 1e-7f * threadIdx.x;
    const float b = 1e-7f;
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int c = 0; c < kChains; ++c) acc[c] = fmaf(acc[c], a, b);
    }
    float s = 0.f;
#pragma unroll
    for (int c = 0; c < kChains; ++c) s += acc[c];
    if (s == 12345.f) sink[threadIdx.x] = s;
}

}  // namespace

int main(int argc, char** argv) {
    Args args(argc, argv);
    const int iters = args.geti("iters", 5);  // timed launches; the best one is reported
    const double target_ms = args.geti("ms", 200);
    print_device_banner();
    print_header();

    cudaStream_t stream;
    SPARK_CUDA_CHECK(cudaStreamCreate(&stream));
    float* sink = nullptr;
    long long* clk = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&sink, kThreads * sizeof(float)));
    SPARK_CUDA_CHECK(cudaMalloc(&clk, 2 * sizeof(long long)));
    // Enough resident warps to keep every tensor core / FMA pipe busy: 4 blocks x 8 warps per SM.
    const int blocks = num_sms() * 4;

    auto time_best = [&](const std::function<void()>& fn) {
        Timing best;
        best.median_ms = 1e30f;
        for (int i = 0; i < iters; ++i) {
            const Timing t = time_kernel(fn, stream, /*warmup=*/1, /*iters=*/1);
            if (t.median_ms < best.median_ms) best = t;
        }
        return best;
    };

    // ---- bf16 mma.sync peak ---------------------------------------------------------------
    {
        // Calibrate the iteration count to ~target_ms per launch.
        int n = 1000;
        Timing t = time_kernel(
            [&] { mma_peak_kernel<<<blocks, kThreads, 0, stream>>>(n, sink, clk); }, stream, 1, 1);
        n = static_cast<int>(n * target_ms / std::max(0.01, static_cast<double>(t.median_ms)));
        n = std::max(n, 100);
        const Timing best =
            time_best([&] { mma_peak_kernel<<<blocks, kThreads, 0, stream>>>(n, sink, clk); });
        const double warps = static_cast<double>(blocks) * (kThreads / kWarpSize);
        const double flops = warps * n * kChains * (16.0 * 8.0 * 16.0 * 2.0);
        long long h[2] = {0, 0};
        SPARK_CUDA_CHECK(cudaMemcpy(h, clk, sizeof(h), cudaMemcpyDeviceToHost));
        const double mhz = h[1] > 0 ? static_cast<double>(h[0]) / h[1] * 1e3 : 0.0;

        Row r;
        r.kernel = "peak_bf16_mma";
        r.dtype = "bf16";
        r.variant = 0;
        r.shape = "m16n8k16";
        r.median_ms = best.median_ms;
        r.min_ms = best.min_ms;
        r.tflops = flops / (best.median_ms * 1e-3) / 1e12;
        print_row(r);

        Row c;
        c.kernel = "sm_clock";
        c.dtype = "-";
        c.variant = 0;
        c.shape = "during_mma";
        c.median_ms = best.median_ms;
        c.min_ms = best.min_ms;
        c.tflops = 0;
        c.gbps = 0;
        c.ref_ms = mhz;  // MHz, in the one free numeric column; the results script knows
        print_row(c);
        std::fprintf(stderr, "    SM clock during the mma loop: %.0f MHz\n", mhz);
    }

    // ---- fp8 (e4m3) mma.sync peaks -----------------------------------------------------------
    const double fp8_warps = static_cast<double>(blocks) * (kThreads / kWarpSize);
    for (int plain = 0; plain < 2; ++plain) {
        auto run = [&](int n) {
            if (plain)
                mma_fp8_peak_kernel<true><<<blocks, kThreads, 0, stream>>>(n, sink, clk);
            else
                mma_fp8_peak_kernel<false><<<blocks, kThreads, 0, stream>>>(n, sink, clk);
        };
        int n = 1000;
        Timing t = time_kernel([&] { run(n); }, stream, 1, 1);
        n = static_cast<int>(n * target_ms / std::max(0.01, static_cast<double>(t.median_ms)));
        n = std::max(n, 100);
        const Timing best = time_best([&] { run(n); });
        const double flops = fp8_warps * n * kChains * (16.0 * 8.0 * 32.0 * 2.0);
        long long h[2] = {0, 0};
        SPARK_CUDA_CHECK(cudaMemcpy(h, clk, sizeof(h), cudaMemcpyDeviceToHost));
        const double mhz = h[1] > 0 ? static_cast<double>(h[0]) / h[1] * 1e3 : 0.0;

        Row r;
        r.kernel = plain ? "peak_fp8_mma_plain" : "peak_fp8_mma";
        r.dtype = "e4m3";
        r.variant = 0;
        r.shape = plain ? "m16n8k32" : (SPARK_HAS_MX_MMA ? "m16n8k32_mxf8f6f4" : "m16n8k32");
        r.median_ms = best.median_ms;
        r.min_ms = best.min_ms;
        r.tflops = flops / (best.median_ms * 1e-3) / 1e12;
        r.ref_ms = mhz;  // the clock this loop ran at, MHz, as for sm_clock above
        print_row(r);
        std::fprintf(stderr, "    SM clock during the %s fp8 mma loop: %.0f MHz\n",
                     plain ? "plain" : "block-scaled", mhz);
    }
    {
        int n16 = 1000;
        Timing t16 = time_kernel(
            [&] { mma_fp8_f16acc_peak_kernel<<<blocks, kThreads, 0, stream>>>(n16, sink); }, stream,
            1, 1);
        n16 =
            static_cast<int>(n16 * target_ms / std::max(0.01, static_cast<double>(t16.median_ms)));
        n16 = std::max(n16, 100);
        const Timing best16 = time_best(
            [&] { mma_fp8_f16acc_peak_kernel<<<blocks, kThreads, 0, stream>>>(n16, sink); });
        Row r16;
        r16.kernel = "peak_fp8_mma_f16acc";
        r16.dtype = "e4m3";
        r16.variant = 0;
        r16.shape = "m16n8k32_f16acc";
        r16.median_ms = best16.median_ms;
        r16.min_ms = best16.min_ms;
        r16.tflops = fp8_warps * n16 * kChains * (16.0 * 8.0 * 32.0 * 2.0) /
                     (best16.median_ms * 1e-3) / 1e12;
        print_row(r16);
    }

    // ---- fp32 FMA peak -----------------------------------------------------------------------
    {
        int n = 4000;
        Timing t = time_kernel([&] { fma_peak_kernel<<<blocks, kThreads, 0, stream>>>(n, sink); },
                               stream, 1, 1);
        n = static_cast<int>(n * target_ms / std::max(0.01, static_cast<double>(t.median_ms)));
        n = std::max(n, 100);
        const Timing best =
            time_best([&] { fma_peak_kernel<<<blocks, kThreads, 0, stream>>>(n, sink); });
        const double threads = static_cast<double>(blocks) * kThreads;
        const double flops = threads * n * kChains * 2.0;
        Row r;
        r.kernel = "peak_fp32_fma";
        r.dtype = "f32";
        r.variant = 0;
        r.shape = "fma";
        r.median_ms = best.median_ms;
        r.min_ms = best.min_ms;
        r.tflops = flops / (best.median_ms * 1e-3) / 1e12;
        print_row(r);
    }

    SPARK_CUDA_CHECK(cudaFree(sink));
    SPARK_CUDA_CHECK(cudaFree(clk));
    SPARK_CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}
