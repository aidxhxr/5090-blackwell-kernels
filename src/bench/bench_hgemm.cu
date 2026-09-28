// Benchmark + correctness check for the bf16 tensor-core GEMM ladder against cuBLAS.
//
//   ./bench_hgemm                       # default LLM-ish shape sweep, all variants
//   ./bench_hgemm --m=4096 --n=4096 --k=4096 --variant=2 --iters=50
//   ./bench_hgemm --m=4096 --n=4096 --k=4096 --bias --act=gelu --residual   # fused epilogue
//   ./bench_hgemm --m=16 --n=22016 --k=4096 --swiglu   # interleaved gate/up, C is [16][11008]
//
// Fused rows (--bias, --act=silu|gelu|relu, --residual, --accumulate for C += ..., --swiglu;
// variant 6 unless --variant says otherwise) are validated against the fused math applied on
// the CPU to cuBLAS's fp32 result, and timed against the unfused sequence: our GEMM (or
// cuBLAS's) followed by an elementwise kernel, or two GEMMs and the swiglu kernel. The
// default sweep ends with the fused rows of `fused_rows` below for variant 6.
//
// stdout: one JSON object per row (collected by scripts/run_all_benches.sh)
// stderr: human-readable table incl. "% of cuBLAS"

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "spark/kernels.h"

#define CUBLAS_CHECK(expr)                                                                      \
    do {                                                                                        \
        cublasStatus_t _st = (expr);                                                            \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(_st), __FILE__, \
                         __LINE__);                                                             \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

namespace {

struct Shape {
    int M, N, K;
};

// Row-major C = A*B via column-major cuBLAS: C^T = B^T * A^T, i.e. GemmEx(N, M, K, B, A).
void cublas_gemm(cublasHandle_t handle, const __nv_bfloat16* A, const __nv_bfloat16* B,
                 __nv_bfloat16* C, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, N,
                              A, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
}

std::vector<float> to_host_f32(const __nv_bfloat16* d, size_t n) {
    std::vector<__nv_bfloat16> h(n);
    SPARK_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = __bfloat162float(h[i]);
    return out;
}

__nv_bfloat16* upload_bf16(const std::vector<float>& h, int copies = 1) {
    std::vector<__nv_bfloat16> hb(h.size());
    for (size_t i = 0; i < h.size(); ++i) hb[i] = __float2bfloat16(h[i]);
    const size_t bytes = h.size() * sizeof(__nv_bfloat16);
    __nv_bfloat16* d = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&d, copies * bytes));
    for (int c = 0; c < copies; ++c)
        SPARK_CUDA_CHECK(cudaMemcpy(d + c * h.size(), hb.data(), bytes, cudaMemcpyHostToDevice));
    return d;
}

// ---- fused epilogue rows -------------------------------------------------------------------

// The epilogue options of one row and their tag in the shape string ("+bias+gelu+residual").
struct Fused {
    bool bias = false;
    int act = spark::HGEMM_ACT_NONE;
    bool residual = false;    // C = ... + residual, a separate [M][N] tensor
    bool accumulate = false;  // C += ..., the residual is C itself
    bool swiglu = false;      // B is [K][N] interleaved gate/up, C is [M][N/2]
    bool any() const {
        return bias || act != spark::HGEMM_ACT_NONE || residual || accumulate || swiglu;
    }
    std::string tag() const {
        std::string t;
        if (bias) t += "+bias";
        if (act == spark::HGEMM_ACT_SILU) t += "+silu";
        if (act == spark::HGEMM_ACT_GELU) t += "+gelu";
        if (act == spark::HGEMM_ACT_RELU) t += "+relu";
        if (residual) t += "+residual";
        if (accumulate) t += "+accumulate";
        if (swiglu) t += "+swiglu";
        return t;
    }
    // Parses "bias+gelu+residual"; throws on an unknown token.
    static Fused from_tag(const std::string& tag) {
        Fused f;
        size_t i = 0;
        while (i < tag.size()) {
            size_t j = tag.find('+', i);
            if (j == std::string::npos) j = tag.size();
            const std::string tok = tag.substr(i, j - i);
            if (tok == "bias")
                f.bias = true;
            else if (tok == "silu")
                f.act = spark::HGEMM_ACT_SILU;
            else if (tok == "gelu")
                f.act = spark::HGEMM_ACT_GELU;
            else if (tok == "relu")
                f.act = spark::HGEMM_ACT_RELU;
            else if (tok == "residual")
                f.residual = true;
            else if (tok == "accumulate")
                f.accumulate = true;
            else if (tok == "swiglu")
                f.swiglu = true;
            else if (!tok.empty())
                SPARK_REQUIRE(false, "unknown epilogue option: " + tok);
            i = j + 1;
        }
        SPARK_REQUIRE(!(f.residual && f.accumulate), "--residual and --accumulate are exclusive");
        return f;
    }
};

float act_host(float x, int act) {
    switch (act) {
        case spark::HGEMM_ACT_SILU:
            return x / (1.0f + std::exp(-x));
        case spark::HGEMM_ACT_GELU: {
            const float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
            return 0.5f * x * (1.0f + std::tanh(u));
        }
        case spark::HGEMM_ACT_RELU:
            return x > 0.0f ? x : 0.0f;
        default:
            return x;
    }
}

__device__ __forceinline__ float act_device(float x, int act) {
    switch (act) {
        case spark::HGEMM_ACT_SILU:
            return x / (1.0f + __expf(-x));
        case spark::HGEMM_ACT_GELU: {
            const float u = 0.7978845608028654f * (x + 0.044715f * x * x * x);
            return 0.5f * x * (1.0f + tanhf(u));
        }
        case spark::HGEMM_ACT_RELU:
            return fmaxf(x, 0.0f);
        default:
            return x;
    }
}

// The unfused epilogue as a separate pass: out = act(c + bias[col]) + res, 8 bf16 per thread.
// What a model without the fusion runs after its GEMM (torch: an add and an activation).
__global__ void epilogue_pass_kernel(const __nv_bfloat16* __restrict__ c,
                                     const __nv_bfloat16* __restrict__ bias,
                                     const __nv_bfloat16* __restrict__ res,
                                     __nv_bfloat16* __restrict__ out, size_t n, int N, int act) {
    const size_t i = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
    if (i >= n) return;
    const int col = static_cast<int>(i % N);
    const spark::bf16x8 v = *reinterpret_cast<const spark::bf16x8*>(c + i);
    spark::bf16x8 o;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        float2 x = __bfloat1622float2(v.h[k]);
        if (bias) {
            const float2 b =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(bias + col + 2 * k));
            x.x += b.x;
            x.y += b.y;
        }
        x.x = act_device(x.x, act);
        x.y = act_device(x.y, act);
        if (res) {
            const float2 r =
                __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(res + i + 2 * k));
            x.x += r.x;
            x.y += r.y;
        }
        o.h[k] = __float22bfloat162_rn(x);
    }
    *reinterpret_cast<spark::bf16x8*>(out + i) = o;
}

// Runs one shape for one variant. Returns false on a correctness failure.
bool run_one(cublasHandle_t handle, cudaStream_t stream, const Shape& s, int variant, int iters) {
    const int M = s.M, N = s.N, K = s.K;
    if (!spark::hgemm_supports(M, N, K, variant)) {
        std::fprintf(stderr, "  skip variant %d shape %dx%dx%d: shape not supported\n", variant, M,
                     N, K);
        return true;  // an unsupported shape is not a failure
    }
    const size_t nA = static_cast<size_t>(M) * K, nB = static_cast<size_t>(K) * N,
                 nC = static_cast<size_t>(M) * N;

    std::vector<float> hA(nA), hB(nB);
    spark::bench::fill_uniform(hA, -1.0f, 1.0f, 1234);
    spark::bench::fill_uniform(hB, -1.0f, 1.0f, 5678);
    std::vector<__nv_bfloat16> hAb(nA), hBb(nB);
    for (size_t i = 0; i < nA; ++i) hAb[i] = __float2bfloat16(hA[i]);
    for (size_t i = 0; i < nB; ++i) hBb[i] = __float2bfloat16(hB[i]);

    // Decode shapes (M <= 64) are bound by streaming B, and B alone (32 MB for 4096x4096) fits
    // in the RTX 5090's 96 MB L2, so timing one B back to back would measure L2, not DRAM. A
    // real decode step touches every layer's weights once per token, so the timing loop
    // rotates through enough copies of B to exceed L2 (256 MB+), for cuBLAS and for us alike.
    const size_t b_bytes = nB * sizeof(__nv_bfloat16);
    const int copies = M <= 64 ? static_cast<int>((size_t{256} << 20) / b_bytes) + 1 : 1;

    __nv_bfloat16 *dA = nullptr, *dB = nullptr, *dC = nullptr, *dRef = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&dA, nA * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dB, copies * b_bytes));
    SPARK_CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dRef, nC * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(
        cudaMemcpy(dA, hAb.data(), nA * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(
        cudaMemcpy(dB, hBb.data(), nB * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    for (int c = 1; c < copies; ++c)
        SPARK_CUDA_CHECK(cudaMemcpy(dB + c * nB, dB, b_bytes, cudaMemcpyDeviceToDevice));
    SPARK_CUDA_CHECK(cudaMemset(dC, 0, nC * sizeof(__nv_bfloat16)));

    // Reference.
    cublas_gemm(handle, dA, dB, dRef, M, N, K);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto ref = to_host_f32(dRef, nC);

    // Correctness. Tolerance: both outputs are fp32 accumulations rounded once to bf16
    // (relative step 2^-8 = 0.39%), so we allow 2% of max|ref| to cover a 1-ulp difference
    // on each side plus fp32 summation-order noise.
    spark::hgemm_bf16(dA, dB, dC, M, N, K, variant, stream);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto got = to_host_f32(dC, nC);
    const auto err = spark::bench::compare(got.data(), ref.data(), nC);
    double max_ref = 0.0;
    for (size_t i = 0; i < nC; ++i) max_ref = std::max(max_ref, std::fabs((double)ref[i]));
    const double tol = 2e-2 * max_ref + 1e-3;
    const bool ok = err.max_abs <= tol;
    if (!ok) {
        std::fprintf(stderr, "  FAIL variant %d shape %dx%dx%d: max_abs=%.4e tol=%.4e\n", variant,
                     M, N, K, err.max_abs, tol);
    }

    // Timing. Each launch takes the next copy of B (see `copies` above).
    int turn = 0;
    auto next_b = [&] { return dB + static_cast<size_t>(turn++ % copies) * nB; };
    const auto t_ref = spark::bench::time_kernel(
        [&] { cublas_gemm(handle, dA, next_b(), dRef, M, N, K); }, stream, 5, iters);
    const auto t_us = spark::bench::time_kernel(
        [&] { spark::hgemm_bf16(dA, next_b(), dC, M, N, K, variant, stream); }, stream, 5, iters);

    const double flops = 2.0 * M * N * static_cast<double>(K);
    spark::bench::Row row;
    row.kernel = "hgemm_bf16";
    row.dtype = "bf16";
    row.variant = variant;
    row.shape = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    row.median_ms = t_us.median_ms;
    row.min_ms = t_us.min_ms;
    row.tflops = flops / (t_us.median_ms * 1e-3) / 1e12;
    // A + B + C once each: the traffic floor, which is what a decode shape is bound by.
    row.gbps = 2.0 * (nA + nB + nC) / (t_us.median_ms * 1e-3) / 1e9;
    row.ref_ms = t_ref.median_ms;
    row.max_abs_err = err.max_abs;
    row.max_rel_err = err.max_rel;
    row.ok = ok;
    spark::bench::print_row(row);
    std::fprintf(stderr, "    cuBLAS: %.2f TFLOPS | this kernel = %.1f%% of cuBLAS%s\n",
                 flops / (t_ref.median_ms * 1e-3) / 1e12, 100.0 * t_ref.median_ms / t_us.median_ms,
                 copies > 1 ? " | B rotated over copies that exceed L2 (DRAM-bound)" : "");

    SPARK_CUDA_CHECK(cudaFree(dA));
    SPARK_CUDA_CHECK(cudaFree(dB));
    SPARK_CUDA_CHECK(cudaFree(dC));
    SPARK_CUDA_CHECK(cudaFree(dRef));
    return ok;
}

void epilogue_pass(const __nv_bfloat16* c, const __nv_bfloat16* bias, const __nv_bfloat16* res,
                   __nv_bfloat16* out, int M, int N, int act, cudaStream_t stream) {
    const size_t n = static_cast<size_t>(M) * N;
    const unsigned grid = static_cast<unsigned>((n / 8 + 255) / 256);
    epilogue_pass_kernel<<<grid, 256, 0, stream>>>(c, bias, res, out, n, N, act);
    SPARK_CUDA_CHECK(cudaGetLastError());
}

// One fused row: C = act(A B + bias) + residual, or C[M][N/2] = silu(g) * u from an
// interleaved gate/up B. Validated against the same math on the CPU from cuBLAS's fp32
// result, timed fused and unfused (our GEMM + a pass, and cuBLAS + the same pass; two GEMMs
// and swiglu_bf16 for the SwiGLU form). Returns false on a correctness failure.
bool run_fused(cublasHandle_t handle, cudaStream_t stream, const Shape& s, const Fused& f,
               int variant, int iters) {
    const int M = s.M, N = s.N, K = s.K;
    const int Nout = f.swiglu ? N / 2 : N;
    if (!spark::hgemm_supports(M, N, K, variant) || (variant != 4 && variant != 6)) {
        std::fprintf(stderr, "  skip variant %d shape %dx%dx%d%s: not supported\n", variant, M, N,
                     K, f.tag().c_str());
        return true;
    }
    const size_t nA = static_cast<size_t>(M) * K, nB = static_cast<size_t>(K) * N,
                 nX = static_cast<size_t>(M) * N, nC = static_cast<size_t>(M) * Nout;

    std::vector<float> hA(nA), hB(nB), hBias(N), hRes(nC);
    spark::bench::fill_uniform(hA, -1.0f, 1.0f, 1234);
    spark::bench::fill_uniform(hB, -1.0f, 1.0f, 5678);
    spark::bench::fill_uniform(hBias, -4.0f, 4.0f, 91);
    spark::bench::fill_uniform(hRes, -4.0f, 4.0f, 92);
    // The bf16 values the GPU sees, for the CPU reference.
    for (auto* v : {&hBias, &hRes})
        for (auto& x : *v) x = __bfloat162float(__float2bfloat16(x));

    // Decode shapes rotate B past L2 as run_one does; the de-interleaved halves for the
    // unfused SwiGLU rotate the same way.
    const size_t b_bytes = nB * sizeof(__nv_bfloat16);
    const int copies = M <= 64 ? static_cast<int>((size_t{256} << 20) / b_bytes) + 1 : 1;
    __nv_bfloat16* dA = upload_bf16(hA);
    __nv_bfloat16* dB = upload_bf16(hB, copies);
    __nv_bfloat16* dBias = f.bias ? upload_bf16(hBias) : nullptr;
    __nv_bfloat16* dRes = (f.residual || f.accumulate) ? upload_bf16(hRes) : nullptr;
    __nv_bfloat16 *dC = nullptr, *dX = nullptr, *dBg = nullptr, *dBu = nullptr, *dCg = nullptr,
                  *dCu = nullptr;
    float* dRef32 = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dX, nX * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dRef32, nX * sizeof(float)));
    if (f.swiglu) {
        std::vector<float> hBg(nB / 2), hBu(nB / 2);
        for (int k = 0; k < K; ++k)
            for (int j = 0; j < Nout; ++j) {
                hBg[static_cast<size_t>(k) * Nout + j] = hB[static_cast<size_t>(k) * N + 2 * j];
                hBu[static_cast<size_t>(k) * Nout + j] = hB[static_cast<size_t>(k) * N + 2 * j + 1];
            }
        dBg = upload_bf16(hBg, copies);
        dBu = upload_bf16(hBu, copies);
        SPARK_CUDA_CHECK(cudaMalloc(&dCg, nC * sizeof(__nv_bfloat16)));
        SPARK_CUDA_CHECK(cudaMalloc(&dCu, nC * sizeof(__nv_bfloat16)));
    }

    spark::HgemmEpilogue ep;
    ep.bias = dBias;
    ep.act = f.act;
    ep.residual = f.accumulate ? dC : dRes;
    ep.swiglu = f.swiglu;

    // Reference: cuBLAS in fp32 out (bf16 in, CUBLAS_COMPUTE_32F), the fused math on the host
    // in fp32, one rounding to bf16: the same rounding the kernel does.
    {
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, dB,
                                  CUDA_R_16BF, N, dA, CUDA_R_16BF, K, &beta, dRef32, CUDA_R_32F, N,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<float> x(nX);
    SPARK_CUDA_CHECK(cudaMemcpy(x.data(), dRef32, nX * sizeof(float), cudaMemcpyDeviceToHost));
    std::vector<float> ref(nC);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < Nout; ++j) {
            float y;
            if (f.swiglu) {
                const float g =
                    x[static_cast<size_t>(i) * N + 2 * j] + (f.bias ? hBias[2 * j] : 0.f);
                const float u =
                    x[static_cast<size_t>(i) * N + 2 * j + 1] + (f.bias ? hBias[2 * j + 1] : 0.f);
                y = act_host(g, spark::HGEMM_ACT_SILU) * u;
            } else {
                y = act_host(x[static_cast<size_t>(i) * N + j] + (f.bias ? hBias[j] : 0.f), f.act);
            }
            if (f.residual || f.accumulate) y += hRes[static_cast<size_t>(i) * Nout + j];
            ref[static_cast<size_t>(i) * Nout + j] = __bfloat162float(__float2bfloat16(y));
        }
    }

    // Correctness, from a fresh C (the accumulate form reads it).
    if (f.accumulate)
        SPARK_CUDA_CHECK(
            cudaMemcpy(dC, dRes, nC * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
    else
        SPARK_CUDA_CHECK(cudaMemset(dC, 0, nC * sizeof(__nv_bfloat16)));
    spark::hgemm_bf16(dA, dB, dC, M, N, K, variant, stream, ep);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto got = to_host_f32(dC, nC);
    const auto err = spark::bench::compare(got.data(), ref.data(), nC);
    double max_ref = 0.0;
    for (size_t i = 0; i < nC; ++i) max_ref = std::max(max_ref, std::fabs((double)ref[i]));
    const double tol = 2e-2 * max_ref + 1e-3;
    const bool ok = err.max_abs <= tol;
    if (!ok) {
        std::fprintf(stderr, "  FAIL variant %d shape %dx%dx%d%s: max_abs=%.4e tol=%.4e\n", variant,
                     M, N, K, f.tag().c_str(), err.max_abs, tol);
    }

    // Timing. The accumulate form keeps adding into C, which is fine for a timing loop.
    int turn = 0;
    auto next = [&](__nv_bfloat16* base, size_t n) {
        return base + static_cast<size_t>(turn % copies) * n;
    };
    auto fused = [&] {
        spark::hgemm_bf16(dA, next(dB, nB), dC, M, N, K, variant, stream, ep);
        ++turn;
    };
    // Unfused: our GEMM(s) then the extra pass; the residual of the accumulate form is C.
    const __nv_bfloat16* res_in = f.accumulate ? dC : dRes;
    auto unfused_ours = [&] {
        if (f.swiglu) {
            spark::hgemm_bf16(dA, next(dBg, nB / 2), dCg, M, Nout, K, variant, stream);
            spark::hgemm_bf16(dA, next(dBu, nB / 2), dCu, M, Nout, K, variant, stream);
            spark::swiglu_bf16(dCg, dCu, dC, static_cast<int64_t>(nC), 1, stream);
            if (res_in) epilogue_pass(dC, nullptr, res_in, dC, M, Nout, 0, stream);
        } else {
            spark::hgemm_bf16(dA, next(dB, nB), dX, M, N, K, variant, stream);
            epilogue_pass(dX, dBias, res_in, dC, M, N, f.act, stream);
        }
        ++turn;
    };
    auto unfused_cublas = [&] {
        if (f.swiglu) {
            cublas_gemm(handle, dA, next(dBg, nB / 2), dCg, M, Nout, K);
            cublas_gemm(handle, dA, next(dBu, nB / 2), dCu, M, Nout, K);
            spark::swiglu_bf16(dCg, dCu, dC, static_cast<int64_t>(nC), 1, stream);
            if (res_in) epilogue_pass(dC, nullptr, res_in, dC, M, Nout, 0, stream);
        } else {
            cublas_gemm(handle, dA, next(dB, nB), dX, M, N, K);
            epilogue_pass(dX, dBias, res_in, dC, M, N, f.act, stream);
        }
        ++turn;
    };
    auto plain_ours = [&] {
        spark::hgemm_bf16(dA, next(dB, nB), dX, M, N, K, variant, stream);
        ++turn;
    };
    const auto t_fused = spark::bench::time_kernel(fused, stream, 5, iters);
    const auto t_plain = spark::bench::time_kernel(plain_ours, stream, 5, iters);
    const auto t_ours = spark::bench::time_kernel(unfused_ours, stream, 5, iters);
    const auto t_ref = spark::bench::time_kernel(unfused_cublas, stream, 5, iters);

    const double flops = 2.0 * M * N * static_cast<double>(K);
    spark::bench::Row row;
    row.kernel = "hgemm_bf16";
    row.dtype = "bf16";
    row.variant = variant;
    row.shape = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K) + f.tag();
    row.median_ms = t_fused.median_ms;
    row.min_ms = t_fused.min_ms;
    row.tflops = flops / (t_fused.median_ms * 1e-3) / 1e12;
    // The traffic floor of the fused call: A, B and C once, plus the bias and the residual.
    const double bytes = 2.0 * (nA + nB + nC + (f.bias ? N : 0) + (res_in ? nC : 0));
    row.gbps = bytes / (t_fused.median_ms * 1e-3) / 1e9;
    row.ref_ms = t_ref.median_ms;  // the unfused sequence with cuBLAS's GEMM(s)
    row.max_abs_err = err.max_abs;
    row.max_rel_err = err.max_rel;
    row.ok = ok;
    spark::bench::print_row(row);
    // Bytes the fusion takes off the bus: the plain GEMM's C write and the pass's read of it
    // (the SwiGLU form: both halves, written by two GEMMs and read by swiglu).
    const double removed_mb = 4.0 * nX / 1e6;
    std::fprintf(stderr,
                 "    fused %.1f us | plain GEMM %.1f us | unfused ours %.1f us | unfused cuBLAS "
                 "%.1f us | saves %.1f us (%.0f%% of the unfused ours) | %.1f MB of DRAM traffic "
                 "removed%s%s\n",
                 t_fused.median_ms * 1e3, t_plain.median_ms * 1e3, t_ours.median_ms * 1e3,
                 t_ref.median_ms * 1e3, (t_ours.median_ms - t_fused.median_ms) * 1e3,
                 100.0 * (t_ours.median_ms - t_fused.median_ms) / t_ours.median_ms, removed_mb,
                 f.swiglu ? " (+ A read once, not twice)" : "",
                 copies > 1 ? " | B rotated past L2" : "");

    for (auto* d : {dA, dB, dBias, dRes, dC, dX, dBg, dBu, dCg, dCu})
        if (d) SPARK_CUDA_CHECK(cudaFree(d));
    SPARK_CUDA_CHECK(cudaFree(dRef32));
    return ok;
}

// Fused rows of the default sweep, variant 6: the bias + activation and the residual-stream
// forms on the prefill and decode projection shapes, and SwiGLU with the interleaved
// [K][2 x 11008] gate/up weight. scripts/bench_torch.py times the same list against eager
// PyTorch (tests/test_bench_torch.py keeps the two in step).
struct FusedRow {
    Shape shape;
    const char* tag;
};
const FusedRow fused_rows[] = {
    {{4096, 4096, 4096}, "bias+gelu"},  {{4096, 4096, 4096}, "residual"},
    {{4096, 11008, 4096}, "bias+gelu"}, {{4096, 11008, 4096}, "residual"},
    {{4096, 22016, 4096}, "swiglu"},    {{16, 4096, 4096}, "bias+gelu"},
    {{16, 4096, 4096}, "residual"},     {{16, 11008, 4096}, "bias+gelu"},
    {{16, 11008, 4096}, "residual"},    {{16, 22016, 4096}, "swiglu"},
};

}  // namespace

int run(int argc, char** argv) {
    spark::bench::Args args(argc, argv);
    spark::bench::print_device_banner();

    std::vector<Shape> shapes;
    if (args.has("m") || args.has("n") || args.has("k")) {
        const int m = args.geti("m", 4096);
        shapes.push_back({m, args.geti("n", m), args.geti("k", m)});
    } else {
        // Square and Llama-7B projection shapes at 4096 tokens, then decode shapes: 1 (a
        // single token), 16, 32 and 64 tokens against the same weights, where the GEMM is
        // bound by streaming B.
        shapes = {{1024, 1024, 1024},  {2048, 2048, 2048},  {4096, 4096, 4096}, {8192, 8192, 8192},
                  {4096, 4096, 11008}, {4096, 11008, 4096}, {1, 4096, 4096},    {16, 4096, 4096},
                  {32, 4096, 4096},    {64, 4096, 4096},    {16, 11008, 4096},  {64, 4096, 11008}};
    }
    const int iters = args.geti("iters", 50);
    std::vector<int> variants;
    if (args.has("variant")) {
        variants.push_back(args.geti("variant", 0));
    } else {
        for (int v = 0; v < spark::hgemm_num_variants(); ++v) variants.push_back(v);
    }

    // Epilogue flags: one fused row per shape instead of the plain one.
    Fused fused;
    fused.bias = args.has("bias");
    fused.residual = args.has("residual");
    fused.accumulate = args.has("accumulate");
    fused.swiglu = args.has("swiglu");
    if (args.has("act")) fused.act = Fused::from_tag(args.get("act", "")).act;
    fused = Fused::from_tag(fused.tag());  // re-parse: validates the combination

    cudaStream_t stream = nullptr;  // legacy default stream: matches cuBLAS default
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    spark::bench::print_header();
    bool all_ok = true;
    if (fused.any()) {
        const int v = args.has("variant") ? args.geti("variant", 6) : 6;
        for (const auto& s : shapes)
            all_ok = run_fused(handle, stream, s, fused, v, iters) && all_ok;
    } else {
        for (const auto& s : shapes) {
            for (int v : variants) all_ok = run_one(handle, stream, s, v, iters) && all_ok;
        }
        // The fused rows of the default sweep, for the default variant only.
        const bool sweep = !(args.has("m") || args.has("n") || args.has("k"));
        if (sweep && std::find(variants.begin(), variants.end(), 6) != variants.end()) {
            for (const auto& r : fused_rows)
                all_ok =
                    run_fused(handle, stream, r.shape, Fused::from_tag(r.tag), 6, iters) && all_ok;
        }
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    if (!all_ok) {
        std::fprintf(stderr, "CORRECTNESS FAILURE\n");
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_hgemm: %s\n", e.what());
        return 2;
    }
}
