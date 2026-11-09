// Benchmark + correctness check for RMSNorm and fused add+RMSNorm.
//
//   ./bench_rmsnorm [--rows=4096] [--cols=4096] [--iters=100]   (rows/cols: one shape)
//   ./bench_rmsnorm --chain=1                                    # rows of single launches
//
// Every variant is validated against a double-precision CPU reference; the process
// exits 1 if any check fails. JSON rows go to stdout, a human table to stderr.
//
// A row's time is per launch, --chain=N launches (20 by default) between one event pair,
// median and min over 20 pairs (spark::bench::time_chained). As single launches the rows of
// 7 to 11 us moved by up to 12.5% between two runs (docs/design/rmsnorm.md, "Timing"). The
// single-launch median is still printed on stderr.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "spark/kernels.h"

using namespace spark;
using namespace spark::bench;

namespace {

constexpr float kEps = 1e-6f;

// --chain=N, see the top of the file.
int g_chain = kChain;

// Times one row and prints the single-launch median on stderr when the row is a chain.
Timing time_row(const std::function<void()>& fn, int iters, cudaStream_t stream) {
    const ChainedTiming t = time_chained(fn, stream, g_chain, 10, iters, iters);
    if (g_chain > 1) std::fprintf(stderr, "    single launch: %.1f us\n", t.single_ms * 1e3);
    return t.row;
}

// ---- host-side dtype helpers ----------------------------------------------------------------
template <typename T>
struct HostDtype;

template <>
struct HostDtype<float> {
    static const char* name() { return "f32"; }
    static float from_f32(float v) { return v; }
    static float to_f32(float v) { return v; }
    static double tol() { return 1e-4; }
    static void run(const float* x, const float* w, float* out, int rows, int cols, int variant,
                    cudaStream_t s) {
        rmsnorm_f32(x, w, out, rows, cols, kEps, variant, s);
    }
};

template <>
struct HostDtype<__nv_bfloat16> {
    static const char* name() { return "bf16"; }
    static __nv_bfloat16 from_f32(float v) { return __float2bfloat16(v); }
    static float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }
    static double tol() { return 2e-2; }
    static void run(const __nv_bfloat16* x, const __nv_bfloat16* w, __nv_bfloat16* out, int rows,
                    int cols, int variant, cudaStream_t s) {
        rmsnorm_bf16(x, w, out, rows, cols, kEps, variant, s);
    }
};

// CPU reference in double. x/w are the (already dtype-rounded) values as floats.
void rmsnorm_reference(const std::vector<float>& x, const std::vector<float>& w, int rows, int cols,
                       std::vector<float>& ref) {
    ref.resize(static_cast<size_t>(rows) * cols);
    for (int r = 0; r < rows; ++r) {
        const float* xr = x.data() + static_cast<size_t>(r) * cols;
        double sumsq = 0.0;
        for (int c = 0; c < cols; ++c) sumsq += static_cast<double>(xr[c]) * xr[c];
        const double inv = 1.0 / std::sqrt(sumsq / cols + static_cast<double>(kEps));
        for (int c = 0; c < cols; ++c) {
            ref[static_cast<size_t>(r) * cols + c] = static_cast<float>(xr[c] * inv * w[c]);
        }
    }
}

template <typename T>
bool bench_rmsnorm_dtype(int rows, int cols, int iters, cudaStream_t stream) {
    using HD = HostDtype<T>;
    const size_t n = static_cast<size_t>(rows) * cols;

    // Host data (rounded through the storage dtype so the reference sees identical inputs).
    std::vector<float> hx_f(n), hw_f(cols);
    fill_uniform(hx_f, -1.0f, 1.0f, 1);
    fill_uniform(hw_f, 0.5f, 1.5f, 2);
    std::vector<T> hx(n), hw(cols), hout(n);
    for (size_t i = 0; i < n; ++i) {
        hx[i] = HD::from_f32(hx_f[i]);
        hx_f[i] = HD::to_f32(hx[i]);
    }
    for (int c = 0; c < cols; ++c) {
        hw[c] = HD::from_f32(hw_f[c]);
        hw_f[c] = HD::to_f32(hw[c]);
    }
    std::vector<float> ref;
    rmsnorm_reference(hx_f, hw_f, rows, cols, ref);

    T *dx = nullptr, *dw = nullptr, *dout = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&dx, n * sizeof(T)));
    SPARK_CUDA_CHECK(cudaMalloc(&dw, static_cast<size_t>(cols) * sizeof(T)));
    SPARK_CUDA_CHECK(cudaMalloc(&dout, n * sizeof(T)));
    SPARK_CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(
        cudaMemcpy(dw, hw.data(), static_cast<size_t>(cols) * sizeof(T), cudaMemcpyHostToDevice));

    // Traffic model: read x once, read w once, write out once.
    const double bytes =
        static_cast<double>(n) * sizeof(T) * 2.0 + static_cast<double>(cols) * sizeof(T);

    bool all_ok = true;
    double naive_ms = 0.0;
    std::vector<float> got(n);
    for (int variant = 0; variant < rmsnorm_num_variants(); ++variant) {
        SPARK_CUDA_CHECK(cudaMemsetAsync(dout, 0, n * sizeof(T), stream));
        HD::run(dx, dw, dout, rows, cols, variant, stream);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        SPARK_CUDA_CHECK(cudaMemcpy(hout.data(), dout, n * sizeof(T), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) got[i] = HD::to_f32(hout[i]);
        const ErrorStats err = compare(got.data(), ref.data(), n);

        const Timing t =
            time_row([&]() { HD::run(dx, dw, dout, rows, cols, variant, stream); }, iters, stream);
        if (variant == 0) naive_ms = t.median_ms;

        Row row;
        row.kernel = "rmsnorm";
        row.dtype = HD::name();
        row.variant = variant;
        row.shape = std::to_string(rows) + "x" + std::to_string(cols);
        row.median_ms = t.median_ms;
        row.min_ms = t.min_ms;
        row.gbps = bytes / (t.median_ms * 1e-3) / 1e9;
        row.ref_ms = naive_ms;
        row.max_abs_err = err.max_abs;
        row.max_rel_err = err.max_rel;
        row.ok = err.max_abs <= HD::tol();
        all_ok = all_ok && row.ok;
        print_row(row);
    }

    SPARK_CUDA_CHECK(cudaFree(dx));
    SPARK_CUDA_CHECK(cudaFree(dw));
    SPARK_CUDA_CHECK(cudaFree(dout));
    return all_ok;
}

bool bench_add_rmsnorm(int rows, int cols, int iters, cudaStream_t stream) {
    using T = __nv_bfloat16;
    using HD = HostDtype<T>;
    const size_t n = static_cast<size_t>(rows) * cols;

    std::vector<float> hx_f(n), hr_f(n), hw_f(cols);
    fill_uniform(hx_f, -1.0f, 1.0f, 3);
    fill_uniform(hr_f, -1.0f, 1.0f, 4);
    fill_uniform(hw_f, 0.5f, 1.5f, 5);
    std::vector<T> hx(n), hr(n), hw(cols), hout(n), hr_out(n);
    for (size_t i = 0; i < n; ++i) {
        hx[i] = HD::from_f32(hx_f[i]);
        hx_f[i] = HD::to_f32(hx[i]);
        hr[i] = HD::from_f32(hr_f[i]);
        hr_f[i] = HD::to_f32(hr[i]);
    }
    for (int c = 0; c < cols; ++c) {
        hw[c] = HD::from_f32(hw_f[c]);
        hw_f[c] = HD::to_f32(hw[c]);
    }
    // Reference: sum rounded to bf16 (as the kernel stores it), then normalized in double.
    std::vector<float> hsum_f(n), ref, ref_resid(n);
    for (size_t i = 0; i < n; ++i) {
        hsum_f[i] = HD::to_f32(HD::from_f32(hx_f[i] + hr_f[i]));
        ref_resid[i] = hsum_f[i];
    }
    rmsnorm_reference(hsum_f, hw_f, rows, cols, ref);

    T *dx = nullptr, *dr = nullptr, *dw = nullptr, *dout = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&dx, n * sizeof(T)));
    SPARK_CUDA_CHECK(cudaMalloc(&dr, n * sizeof(T)));
    SPARK_CUDA_CHECK(cudaMalloc(&dw, static_cast<size_t>(cols) * sizeof(T)));
    SPARK_CUDA_CHECK(cudaMalloc(&dout, n * sizeof(T)));
    SPARK_CUDA_CHECK(cudaMemcpy(dx, hx.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(dr, hr.data(), n * sizeof(T), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(
        cudaMemcpy(dw, hw.data(), static_cast<size_t>(cols) * sizeof(T), cudaMemcpyHostToDevice));

    // Correctness on fresh data.
    add_rmsnorm_bf16(dx, dr, dw, dout, rows, cols, kEps, stream);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    SPARK_CUDA_CHECK(cudaMemcpy(hout.data(), dout, n * sizeof(T), cudaMemcpyDeviceToHost));
    SPARK_CUDA_CHECK(cudaMemcpy(hr_out.data(), dr, n * sizeof(T), cudaMemcpyDeviceToHost));
    std::vector<float> got(n), got_resid(n);
    for (size_t i = 0; i < n; ++i) {
        got[i] = HD::to_f32(hout[i]);
        got_resid[i] = HD::to_f32(hr_out[i]);
    }
    const ErrorStats err_out = compare(got.data(), ref.data(), n);
    const ErrorStats err_res = compare(got_resid.data(), ref_resid.data(), n);

    // Timing: resid accumulates in place across iterations, which is fine for timing
    // (magnitude stays far below bf16 range for the ~550 launches of a chained row, |x| <= 1).
    const Timing t = time_row(
        [&]() { add_rmsnorm_bf16(dx, dr, dw, dout, rows, cols, kEps, stream); }, iters, stream);

    // Traffic model: read x, read resid, write resid, write out, read w.
    const double bytes =
        static_cast<double>(n) * sizeof(T) * 4.0 + static_cast<double>(cols) * sizeof(T);

    Row row;
    row.kernel = "add_rmsnorm";
    row.dtype = HD::name();
    row.variant = 0;
    row.shape = std::to_string(rows) + "x" + std::to_string(cols);
    row.median_ms = t.median_ms;
    row.min_ms = t.min_ms;
    row.gbps = bytes / (t.median_ms * 1e-3) / 1e9;
    row.ref_ms = 0.0;
    row.max_abs_err = std::max(err_out.max_abs, err_res.max_abs);
    row.max_rel_err = std::max(err_out.max_rel, err_res.max_rel);
    row.ok = err_out.max_abs <= HD::tol() && err_res.max_abs == 0.0;
    print_row(row);

    SPARK_CUDA_CHECK(cudaFree(dx));
    SPARK_CUDA_CHECK(cudaFree(dr));
    SPARK_CUDA_CHECK(cudaFree(dw));
    SPARK_CUDA_CHECK(cudaFree(dout));
    return row.ok;
}

}  // namespace

int run(int argc, char** argv) {
    Args args(argc, argv);
    const int iters = args.geti("iters", 100);
    g_chain = chain_arg(args);
    // Llama/Qwen hidden sizes at 4096 tokens, plus one shape whose input alone (256 MB bf16)
    // is bigger than the RTX 5090's 96 MB L2, so the headline row is DRAM-bound for certain.
    std::vector<std::pair<int, int>> shapes = {
        {4096, 1024}, {4096, 2048}, {4096, 4096}, {4096, 8192}, {16384, 8192}};
    if (args.has("rows") || args.has("cols"))
        shapes = {{args.geti("rows", 4096), args.geti("cols", 4096)}};

    print_device_banner();
    print_header();

    cudaStream_t stream;
    SPARK_CUDA_CHECK(cudaStreamCreate(&stream));

    bool ok = true;
    for (auto [rows, cols] : shapes)
        ok = bench_rmsnorm_dtype<float>(rows, cols, iters, stream) && ok;
    for (auto [rows, cols] : shapes)
        ok = bench_rmsnorm_dtype<__nv_bfloat16>(rows, cols, iters, stream) && ok;
    for (auto [rows, cols] : shapes) ok = bench_add_rmsnorm(rows, cols, iters, stream) && ok;

    SPARK_CUDA_CHECK(cudaStreamDestroy(stream));
    if (!ok) {
        std::fprintf(stderr, "bench_rmsnorm: VALIDATION FAILED\n");
        return 1;
    }
    std::fprintf(stderr, "bench_rmsnorm: all variants validated\n");
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_rmsnorm: %s\n", e.what());
        return 2;
    }
}
