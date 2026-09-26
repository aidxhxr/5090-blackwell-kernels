// Benchmark + correctness check for the fp8 (e4m3) tensor-core GEMM ladder against cuBLASLt.
//
//   ./bench_fp8gemm                       # default shape sweep (bench_hgemm's), all variants
//   ./bench_fp8gemm --m=4096 --n=4096 --k=4096 --variant=1 --iters=50
//
// C = scale_a * scale_b * A[M,K] * Bt[N,K]^T, e4m3 in, bf16 out. cuBLASLt's fp8 path only takes
// this layout (A row-major, B column-major, both K-contiguous), so the reference is timed on
// exactly the same bytes. Correctness is checked two ways: the whole output against cuBLASLt,
// and a sample of outputs against a CPU fp32 dot product of the same e4m3-rounded inputs.
//
// stdout: one JSON object per row (collected by scripts/run_all_benches.sh)
// stderr: human-readable table incl. "% of cuBLASLt"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <cstdio>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "spark/kernels.h"

#define CUBLAS_CHECK(expr)                                                                        \
    do {                                                                                          \
        cublasStatus_t _st = (expr);                                                              \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                                       \
            std::fprintf(stderr, "cuBLASLt error %d at %s:%d\n", static_cast<int>(_st), __FILE__, \
                         __LINE__);                                                               \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

namespace {

struct Shape {
    int M, N, K;
};

constexpr float kScaleA = 0.75f;  // exact in every format, so the check sees no scale rounding
constexpr float kScaleB = 1.5f;

// cuBLASLt fp8 matmul in the one layout it supports: op(A') = A'^T with A' K x N column-major
// (that is our Bt, K contiguous), B' = K x M column-major (our A), D = N x M column-major with
// ld N, which is our row-major C. So cuBLASLt computes C^T = Bt * A^T and the scale pointers
// are swapped to match.
struct LtGemm {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t heur{};
    void* ws = nullptr;
    size_t ws_bytes = size_t{32} << 20;

    LtGemm(int M, int N, int K, const float* d_scale_a, const float* d_scale_b) {
        CUBLAS_CHECK(cublasLtCreate(&handle));
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t t = CUBLAS_OP_T, n = CUBLAS_OP_N;
        CUBLAS_CHECK(
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t)));
        CUBLAS_CHECK(
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &n, sizeof(n)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                                    &d_scale_b, sizeof(d_scale_b)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                                    &d_scale_a, sizeof(d_scale_a)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, K, N, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, K, M, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N));
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
        int returned = 0;
        CUBLAS_CHECK(
            cublasLtMatmulAlgoGetHeuristic(handle, op, la, lb, lc, lc, pref, 1, &heur, &returned));
        if (returned == 0) {
            std::fprintf(stderr, "cuBLASLt: no fp8 algorithm for %dx%dx%d\n", M, N, K);
            std::exit(1);
        }
        SPARK_CUDA_CHECK(cudaMalloc(&ws, ws_bytes));
    }
    void run(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* Bt, __nv_bfloat16* C,
             cudaStream_t stream) {
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasLtMatmul(handle, op, &alpha, Bt, la, A, lb, &beta, C, lc, C, lc,
                                    &heur.algo, ws, ws_bytes, stream));
    }
    ~LtGemm() {
        cudaFree(ws);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(lc);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatmulDescDestroy(op);
        cublasLtDestroy(handle);
    }
};

std::vector<float> to_host_f32(const __nv_bfloat16* d, size_t n) {
    std::vector<__nv_bfloat16> h(n);
    SPARK_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = __bfloat162float(h[i]);
    return out;
}

// Uniform [-1, 1] rounded to e4m3 (round to nearest even, saturating), plus the fp32 value of
// each rounded element for the CPU reference.
void fill_e4m3(std::vector<__nv_fp8_e4m3>& q, std::vector<float>& f, uint32_t seed) {
    std::vector<float> raw(q.size());
    spark::bench::fill_uniform(raw, -1.0f, 1.0f, seed);
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = __nv_fp8_e4m3(raw[i]);
        f[i] = static_cast<float>(q[i]);
    }
}

// Runs one shape for one variant. Returns false on a correctness failure.
bool run_one(cudaStream_t stream, const Shape& s, int variant, int iters) {
    const int M = s.M, N = s.N, K = s.K;
    if (!spark::fp8gemm_supports(M, N, K, variant)) {
        std::fprintf(stderr, "  skip variant %d shape %dx%dx%d: shape not supported\n", variant, M,
                     N, K);
        return true;  // an unsupported shape is not a failure
    }
    const size_t nA = static_cast<size_t>(M) * K, nB = static_cast<size_t>(N) * K,
                 nC = static_cast<size_t>(M) * N;

    std::vector<__nv_fp8_e4m3> hA(nA), hB(nB);
    std::vector<float> fA(nA), fB(nB);
    fill_e4m3(hA, fA, 1234);
    fill_e4m3(hB, fB, 5678);

    // Decode shapes (M <= 64) are bound by streaming Bt, and Bt alone (16 MB for 4096x4096)
    // fits in the RTX 5090's 96 MB L2, so the timing loop rotates through enough copies to
    // exceed L2 (256 MB+), for cuBLASLt and for us alike. fp8 weights are half the bytes of
    // bf16, so twice the copies.
    const int copies = M <= 64 ? static_cast<int>((size_t{256} << 20) / nB) + 1 : 1;

    __nv_fp8_e4m3 *dA = nullptr, *dB = nullptr;
    __nv_bfloat16 *dC = nullptr, *dRef = nullptr;
    float* dScale = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&dA, nA));
    SPARK_CUDA_CHECK(cudaMalloc(&dB, copies * nB));
    SPARK_CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dRef, nC * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dScale, 2 * sizeof(float)));
    const float scales[2] = {kScaleA, kScaleB};
    SPARK_CUDA_CHECK(cudaMemcpy(dScale, scales, sizeof(scales), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA, cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB, cudaMemcpyHostToDevice));
    for (int c = 1; c < copies; ++c)
        SPARK_CUDA_CHECK(cudaMemcpy(dB + c * nB, dB, nB, cudaMemcpyDeviceToDevice));
    SPARK_CUDA_CHECK(cudaMemset(dC, 0, nC * sizeof(__nv_bfloat16)));
    const float* d_scale_a = dScale;
    const float* d_scale_b = dScale + 1;

    // Reference.
    LtGemm lt(M, N, K, d_scale_a, d_scale_b);
    lt.run(dA, dB, dRef, stream);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto ref = to_host_f32(dRef, nC);

    // Correctness against cuBLASLt. Both outputs are fp32 accumulations of exact e4m3 products,
    // scaled and rounded once to bf16 (relative step 2^-8 = 0.39%), so 2% of max|ref| covers
    // a 1-ulp difference on each side plus fp32 summation-order noise.
    spark::fp8gemm(dA, dB, dC, M, N, K, d_scale_a, d_scale_b, variant, stream);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto got = to_host_f32(dC, nC);
    const auto err = spark::bench::compare(got.data(), ref.data(), nC);
    double max_ref = 0.0;
    for (size_t i = 0; i < nC; ++i) max_ref = std::max(max_ref, std::fabs((double)ref[i]));
    const double tol = 2e-2 * max_ref + 1e-3;
    bool ok = err.max_abs <= tol;
    if (!ok) {
        std::fprintf(stderr,
                     "  FAIL variant %d shape %dx%dx%d vs cuBLASLt: max_abs=%.4e tol=%.4e\n",
                     variant, M, N, K, err.max_abs, tol);
    }
    // ... and against the CPU on a sample of outputs (the whole matrix would be a TFLOP at
    // 8192^3): the same tolerance against the fp32 sum of the same e4m3-rounded products.
    {
        std::mt19937 rng(99);
        const int samples = 4096;
        double cpu_max_abs = 0.0;
        for (int i = 0; i < samples; ++i) {
            const int r = static_cast<int>(rng() % static_cast<unsigned>(M));
            const int c = static_cast<int>(rng() % static_cast<unsigned>(N));
            float acc = 0.f;
            const float* a = fA.data() + static_cast<size_t>(r) * K;
            const float* b = fB.data() + static_cast<size_t>(c) * K;
            for (int k = 0; k < K; ++k) acc = std::fmaf(a[k], b[k], acc);
            const double want = static_cast<double>(acc) * kScaleA * kScaleB;
            cpu_max_abs =
                std::max(cpu_max_abs, std::fabs(want - got[static_cast<size_t>(r) * N + c]));
        }
        if (cpu_max_abs > tol) {
            ok = false;
            std::fprintf(stderr, "  FAIL variant %d shape %dx%dx%d vs CPU: max_abs=%.4e tol=%.4e\n",
                         variant, M, N, K, cpu_max_abs, tol);
        }
    }

    // Timing. Each launch takes the next copy of Bt (see `copies` above).
    int turn = 0;
    auto next_b = [&] { return dB + static_cast<size_t>(turn++ % copies) * nB; };
    const auto t_ref =
        spark::bench::time_kernel([&] { lt.run(dA, next_b(), dRef, stream); }, stream, 5, iters);
    const auto t_us = spark::bench::time_kernel(
        [&] { spark::fp8gemm(dA, next_b(), dC, M, N, K, d_scale_a, d_scale_b, variant, stream); },
        stream, 5, iters);

    const double flops = 2.0 * M * N * static_cast<double>(K);
    spark::bench::Row row;
    row.kernel = "fp8gemm";
    row.dtype = "e4m3";
    row.variant = variant;
    row.shape = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    row.median_ms = t_us.median_ms;
    row.min_ms = t_us.min_ms;
    row.tflops = flops / (t_us.median_ms * 1e-3) / 1e12;
    // A and Bt once as bytes, C once as bf16: the traffic floor a decode shape is bound by.
    row.gbps = (static_cast<double>(nA) + nB + 2.0 * nC) / (t_us.median_ms * 1e-3) / 1e9;
    row.ref_ms = t_ref.median_ms;
    row.max_abs_err = err.max_abs;
    row.max_rel_err = err.max_rel;
    row.ok = ok;
    spark::bench::print_row(row);
    std::fprintf(stderr, "    cuBLASLt: %.2f TFLOPS | this kernel = %.1f%% of cuBLASLt%s\n",
                 flops / (t_ref.median_ms * 1e-3) / 1e12, 100.0 * t_ref.median_ms / t_us.median_ms,
                 copies > 1 ? " | Bt rotated over copies that exceed L2 (DRAM-bound)" : "");

    SPARK_CUDA_CHECK(cudaFree(dA));
    SPARK_CUDA_CHECK(cudaFree(dB));
    SPARK_CUDA_CHECK(cudaFree(dC));
    SPARK_CUDA_CHECK(cudaFree(dRef));
    SPARK_CUDA_CHECK(cudaFree(dScale));
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
        // bench_hgemm's list: square and Llama-7B projection shapes at 4096 tokens, then the
        // decode shapes (1, 16, 32, 64 tokens) where the GEMM is bound by streaming Bt.
        shapes = {{1024, 1024, 1024},  {2048, 2048, 2048},  {4096, 4096, 4096}, {8192, 8192, 8192},
                  {4096, 4096, 11008}, {4096, 11008, 4096}, {1, 4096, 4096},    {16, 4096, 4096},
                  {32, 4096, 4096},    {64, 4096, 4096},    {16, 11008, 4096},  {64, 4096, 11008}};
    }
    const int iters = args.geti("iters", 50);
    std::vector<int> variants;
    if (args.has("variant")) {
        variants.push_back(args.geti("variant", 0));
    } else {
        for (int v = 0; v < spark::fp8gemm_num_variants(); ++v) variants.push_back(v);
    }

    cudaStream_t stream = nullptr;  // legacy default stream, as bench_hgemm
    spark::bench::print_header();
    bool all_ok = true;
    for (const auto& s : shapes) {
        for (int v : variants) all_ok = run_one(stream, s, v, iters) && all_ok;
    }
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
        std::fprintf(stderr, "bench_fp8gemm: %s\n", e.what());
        return 2;
    }
}
