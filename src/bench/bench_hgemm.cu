// Benchmark + correctness check for the bf16 tensor-core GEMM ladder against cuBLAS.
//
//   ./bench_hgemm                       # default LLM-ish shape sweep, all variants
//   ./bench_hgemm --m=4096 --n=4096 --k=4096 --variant=2 --iters=50
//
// stdout: one JSON object per row (collected by scripts/run_all_benches.sh)
// stderr: human-readable table incl. "% of cuBLAS"

#include <cublas_v2.h>
#include <cuda_bf16.h>

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

    cudaStream_t stream = nullptr;  // legacy default stream: matches cuBLAS default
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    spark::bench::print_header();
    bool all_ok = true;
    for (const auto& s : shapes) {
        for (int v : variants) all_ok = run_one(handle, stream, s, v, iters) && all_ok;
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
