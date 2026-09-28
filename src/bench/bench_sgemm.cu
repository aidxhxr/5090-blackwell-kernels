// Benchmark + correctness check for the fp32 GEMM ladder against cuBLAS SGEMM.
//
//   ./bench_sgemm                      # default shape sweep, all variants
//   ./bench_sgemm --m=4096 --n=4096 --k=4096 --variant=3 --iters=50
//   ./bench_sgemm --all                # also run the naive variant on the largest shapes
//
// cuBLAS is timed twice per shape: with CUBLAS_DEFAULT_MATH (fp32 FMAs on the CUDA cores, the
// reference of variants 0 to 5 and 7) and with CUBLAS_TF32_TENSOR_OP_MATH (the tensor cores
// in TF32, the reference of variant 6). Every variant is checked against the fp32 result;
// the tolerance depends on the variant's precision contract (see `tolerance_for`).
//
// stdout: one JSON object per row (collected by scripts/run_all_benches.sh); the rows of
//         variant 6 carry dtype "tf32" and those of variant 7 "3xtf32", so the results
//         scripts judge them against the tf32 tensor-core peak
// stderr: human-readable table
// exit code 1 if any variant disagrees with cuBLAS.

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "spark/kernels.h"

namespace {

#define CUBLAS_CHECK(expr)                                                                      \
    do {                                                                                        \
        cublasStatus_t _st = (expr);                                                            \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(_st), __FILE__, \
                         __LINE__);                                                             \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

struct Shape {
    int M, N, K;
};

// Row-major C = A * B via column-major cuBLAS: treat the row-major matrices as their
// column-major transposes, so C^T[N,M] = B^T[N,K] * A^T[K,M].
void cublas_sgemm_rowmajor(cublasHandle_t h, const float* A, const float* B, float* C, int M, int N,
                           int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(
        cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N));
}

double tflops_of(const Shape& s, double ms) {
    return 2.0 * s.M * s.N * s.K / (ms * 1e-3) / 1e12;
}

// The precision contract of each rung, as the results scripts see it: variant 6 computes in
// TF32, variant 7 in 3xTF32, everything else in fp32.
const char* dtype_of(int variant) {
    return variant == 6 ? "tf32" : variant == 7 ? "3xtf32" : "f32";
}

// Max absolute error allowed against the fp32 cuBLAS result.
//   fp32 rungs and 3xTF32: only the summation order differs (3xTF32 adds a 2^-22 relative
//     error per product, below fp32 accumulation noise), so the bound scales with K.
//   TF32: each operand is rounded to 11 significant bits, a relative error up to 2^-11, so a
//     product is off by up to 2^-10 |a||b| and a dot product of K such terms by up to
//     K 2^-10 max|A| max|B| in the worst case (4 at K = 4096 for inputs in [-1, 1]). The
//     rounding errors are independent, so the typical error is sqrt(K) 2^-11 times the rms
//     product, about 1e-2 at K = 4096, against a max |C| of about 110. The check is 1% of
//     max |C|, ten times the errors actually measured and well inside the worst case; it
//     is loose enough that a broken kernel (a wrong k, a dropped chunk) still fails it by
//     orders of magnitude.
double tolerance_for(int variant, int K, double max_abs_ref) {
    if (variant == 6) return 1e-2 * max_abs_ref;
    return 2e-5 * K + 1e-3;
}

}  // namespace

int run(int argc, char** argv) {
    using namespace spark::bench;
    Args args(argc, argv);
    print_device_banner();

    const int iters = args.geti("iters", 100);
    const int warmup = args.geti("warmup", 10);
    const int only_variant = args.geti("variant", -1);
    const bool run_all = args.has("all");

    std::vector<Shape> shapes;
    if (args.has("m") || args.has("n") || args.has("k")) {
        const int m = args.geti("m", 1024);
        shapes.push_back({m, args.geti("n", m), args.geti("k", m)});
    } else {
        shapes = {{512, 512, 512},    {1024, 1024, 1024},  {2048, 2048, 2048}, {4096, 4096, 4096},
                  {8192, 8192, 8192}, {4096, 4096, 11008}, {4096, 11008, 4096}};
    }

    cudaStream_t stream;
    SPARK_CUDA_CHECK(cudaStreamCreate(&stream));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    // cuBLAS starts on the plain fp32 path (no TF32), the like-for-like reference of the
    // fp32 rungs; the TF32 reference below switches the math mode and switches it back.
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    print_header();
    bool all_ok = true;

    for (const Shape& s : shapes) {
        const size_t nA = static_cast<size_t>(s.M) * s.K;
        const size_t nB = static_cast<size_t>(s.K) * s.N;
        const size_t nC = static_cast<size_t>(s.M) * s.N;

        std::vector<float> hA(nA), hB(nB), hRef(nC), hOut(nC);
        fill_uniform(hA, -1.0f, 1.0f, 1);
        fill_uniform(hB, -1.0f, 1.0f, 2);

        float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dRef = nullptr;
        SPARK_CUDA_CHECK(cudaMalloc(&dA, nA * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMalloc(&dB, nB * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMalloc(&dRef, nC * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA * sizeof(float), cudaMemcpyHostToDevice));
        SPARK_CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB * sizeof(float), cudaMemcpyHostToDevice));

        // ---- cuBLAS fp32 reference (result + timing)
        cublas_sgemm_rowmajor(handle, dA, dB, dRef, s.M, s.N, s.K);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        SPARK_CUDA_CHECK(cudaMemcpy(hRef.data(), dRef, nC * sizeof(float), cudaMemcpyDeviceToHost));
        const Timing tref =
            time_kernel([&] { cublas_sgemm_rowmajor(handle, dA, dB, dRef, s.M, s.N, s.K); }, stream,
                        warmup, iters);
        double max_abs_ref = 0.0;
        for (size_t i = 0; i < nC; ++i)
            max_abs_ref = std::max(max_abs_ref, std::fabs(static_cast<double>(hRef[i])));

        const std::string shape_str =
            std::to_string(s.M) + "x" + std::to_string(s.N) + "x" + std::to_string(s.K);

        {
            Row r;
            r.kernel = "sgemm_cublas";
            r.dtype = "f32";
            r.variant = -1;
            r.shape = shape_str;
            r.median_ms = tref.median_ms;
            r.min_ms = tref.min_ms;
            r.tflops = tflops_of(s, tref.median_ms);
            r.ref_ms = tref.median_ms;
            print_row(r);
        }

        // ---- cuBLAS TF32 reference: the same call with the tensor cores allowed. Its error
        // against the fp32 result is recorded too, so the TF32 rung's error can be read
        // next to the library's.
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
        cublas_sgemm_rowmajor(handle, dA, dB, dC, s.M, s.N, s.K);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        SPARK_CUDA_CHECK(cudaMemcpy(hOut.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
        const ErrorStats err_tf32 = compare(hOut.data(), hRef.data(), nC);
        const Timing tref_tf32 =
            time_kernel([&] { cublas_sgemm_rowmajor(handle, dA, dB, dC, s.M, s.N, s.K); }, stream,
                        warmup, iters);
        CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
        {
            Row r;
            r.kernel = "sgemm_cublas_tf32";
            r.dtype = "tf32";
            r.variant = -1;
            r.shape = shape_str;
            r.median_ms = tref_tf32.median_ms;
            r.min_ms = tref_tf32.min_ms;
            r.tflops = tflops_of(s, tref_tf32.median_ms);
            r.ref_ms = tref_tf32.median_ms;
            r.max_abs_err = err_tf32.max_abs;
            r.max_rel_err = err_tf32.max_rel;
            r.ok = err_tf32.max_abs <= tolerance_for(6, s.K, max_abs_ref);
            print_row(r);
        }

        for (int v = 0; v < spark::sgemm_num_variants(); ++v) {
            if (only_variant >= 0 && v != only_variant) continue;
            // The naive kernel is ~10-20x slower than cuBLAS; skip the huge shapes by default.
            if (v == 0 && !run_all && static_cast<double>(s.M) * s.N * s.K > 2048.0 * 2048 * 2048)
                continue;

            SPARK_CUDA_CHECK(cudaMemsetAsync(dC, 0, nC * sizeof(float), stream));
            spark::sgemm(dA, dB, dC, s.M, s.N, s.K, v, stream);
            SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
            SPARK_CUDA_CHECK(
                cudaMemcpy(hOut.data(), dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
            const ErrorStats err = compare(hOut.data(), hRef.data(), nC);
            const double tol = tolerance_for(v, s.K, max_abs_ref);
            const bool ok = err.max_abs <= tol && std::isfinite(err.max_abs);

            const Timing t = time_kernel(
                [&] { spark::sgemm(dA, dB, dC, s.M, s.N, s.K, v, stream); }, stream, warmup, iters);

            // ref_ms is the reference with the same precision contract: cuBLAS TF32 for
            // variant 6, cuBLAS fp32 for everything else (3xTF32 claims fp32 accuracy).
            Row r;
            r.kernel = "sgemm";
            r.dtype = dtype_of(v);
            r.variant = v;
            r.shape = shape_str;
            r.median_ms = t.median_ms;
            r.min_ms = t.min_ms;
            r.tflops = tflops_of(s, t.median_ms);
            r.ref_ms = v == 6 ? tref_tf32.median_ms : tref.median_ms;
            r.max_abs_err = err.max_abs;
            r.max_rel_err = err.max_rel;
            r.ok = ok;
            print_row(r);
            if (v >= 6) {
                std::fprintf(stderr,
                             "    variant %d: %.1f%% of cuBLAS fp32, %.1f%% of cuBLAS TF32\n", v,
                             100.0 * tref.median_ms / t.median_ms,
                             100.0 * tref_tf32.median_ms / t.median_ms);
            }
            if (!ok) {
                std::fprintf(stderr, "  MISMATCH: variant %d shape %s max_abs=%.3e tol=%.3e\n", v,
                             shape_str.c_str(), err.max_abs, tol);
                all_ok = false;
            }
        }

        SPARK_CUDA_CHECK(cudaFree(dA));
        SPARK_CUDA_CHECK(cudaFree(dB));
        SPARK_CUDA_CHECK(cudaFree(dC));
        SPARK_CUDA_CHECK(cudaFree(dRef));
    }

    CUBLAS_CHECK(cublasDestroy(handle));
    SPARK_CUDA_CHECK(cudaStreamDestroy(stream));
    if (!all_ok) {
        std::fprintf(stderr, "FAILED: at least one variant disagrees with cuBLAS\n");
        return 1;
    }
    std::fprintf(stderr, "all variants match cuBLAS\n");
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_sgemm: %s\n", e.what());
        return 2;
    }
}
