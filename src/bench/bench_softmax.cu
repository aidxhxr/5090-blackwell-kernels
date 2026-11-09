// Benchmark + correctness check for the softmax kernels.
//
//   ./bench_softmax                       # rows=4096, cols in {128, 1024, 4096, 16384}, f32 + bf16
//   ./bench_softmax --rows=4096 --cols=8192 --iters=200
//   ./bench_softmax --chain=1             # rows of single launches
//
// Every variant is validated against a double-precision CPU reference; the
// process exits 1 if any variant fails. ref_ms in the output is the naive
// variant-0 time for the same shape, so speedups can be read off directly.
//
// A row's time is per launch, --chain=N launches (20 by default) between one event pair,
// median and min over 20 pairs (spark::bench::time_chained). As single launches a row of
// 5 us moved by 18% between two runs (docs/design/softmax.md, "Timing"). The single-launch
// median is still printed on stderr.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "spark/kernels.h"

using namespace spark;
using namespace spark::bench;

namespace {

// --chain=N, see the top of the file.
int g_chain = kChain;

// ---- dtype helpers (host side) --------------------------------------------------------
template <typename T>
struct Dtype;

template <>
struct Dtype<float> {
    static const char* name() { return "f32"; }
    static float to_host(float v) { return v; }
    static float from_host(float v) { return v; }
    static constexpr double tol = 1e-5;
};

template <>
struct Dtype<__nv_bfloat16> {
    static const char* name() { return "bf16"; }
    static float to_host(__nv_bfloat16 v) { return __bfloat162float(v); }
    static __nv_bfloat16 from_host(float v) { return __float2bfloat16(v); }
    static constexpr double tol = 1e-2;
};

// ---- double-precision reference ---------------------------------------------------
void cpu_softmax(const std::vector<float>& x, std::vector<float>& ref, int rows, int cols) {
    ref.resize(x.size());
    for (int r = 0; r < rows; ++r) {
        const float* xr = x.data() + static_cast<size_t>(r) * cols;
        float* orow = ref.data() + static_cast<size_t>(r) * cols;
        double m = -INFINITY;
        for (int c = 0; c < cols; ++c) m = std::max(m, static_cast<double>(xr[c]));
        double s = 0.0;
        for (int c = 0; c < cols; ++c) s += std::exp(static_cast<double>(xr[c]) - m);
        for (int c = 0; c < cols; ++c) {
            orow[c] = static_cast<float>(std::exp(static_cast<double>(xr[c]) - m) / s);
        }
    }
}

template <typename T>
void launch(const T* x, T* out, int rows, int cols, int variant, cudaStream_t stream);

template <>
void launch<float>(const float* x, float* out, int rows, int cols, int variant,
                   cudaStream_t stream) {
    softmax_f32(x, out, rows, cols, variant, stream);
}

template <>
void launch<__nv_bfloat16>(const __nv_bfloat16* x, __nv_bfloat16* out, int rows, int cols,
                           int variant, cudaStream_t stream) {
    softmax_bf16(x, out, rows, cols, variant, stream);
}

// Runs all variants for one (dtype, shape). Returns false if any variant is wrong.
template <typename T>
bool run_shape(int rows, int cols, int iters, cudaStream_t stream) {
    const size_t n = static_cast<size_t>(rows) * cols;

    // Input: uniform in [-4, 4], rounded through T so the reference sees the same values.
    std::vector<float> hx(n);
    fill_uniform(hx, -4.0f, 4.0f);
    std::vector<T> hx_t(n);
    for (size_t i = 0; i < n; ++i) {
        hx_t[i] = Dtype<T>::from_host(hx[i]);
        hx[i] = Dtype<T>::to_host(hx_t[i]);
    }
    std::vector<float> ref;
    cpu_softmax(hx, ref, rows, cols);

    T* dx = nullptr;
    T* dout = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&dx, n * sizeof(T)));
    SPARK_CUDA_CHECK(cudaMalloc(&dout, n * sizeof(T)));
    SPARK_CUDA_CHECK(cudaMemcpy(dx, hx_t.data(), n * sizeof(T), cudaMemcpyHostToDevice));

    const std::string shape = std::to_string(rows) + "x" + std::to_string(cols);
    const double bytes = 2.0 * static_cast<double>(n) * sizeof(T);  // one read + one write
    std::vector<T> hout(n);
    std::vector<float> hout_f(n);

    bool all_ok = true;
    double ref_ms = 0.0;
    for (int variant = 0; variant < softmax_num_variants(); ++variant) {
        // Correctness.
        SPARK_CUDA_CHECK(cudaMemsetAsync(dout, 0, n * sizeof(T), stream));
        launch<T>(dx, dout, rows, cols, variant, stream);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        SPARK_CUDA_CHECK(cudaMemcpy(hout.data(), dout, n * sizeof(T), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; ++i) hout_f[i] = Dtype<T>::to_host(hout[i]);
        const ErrorStats err = compare(hout_f.data(), ref.data(), n);
        const bool ok = err.max_abs <= Dtype<T>::tol && std::isfinite(err.max_abs);

        // Timing. The naive variant is very slow on long rows: fewer iterations.
        const int it = (variant == 0) ? std::max(10, iters / 10) : iters;
        const ChainedTiming tt =
            time_chained([&]() { launch<T>(dx, dout, rows, cols, variant, stream); }, stream,
                         g_chain, 5, iters, it);
        const Timing& t = tt.row;
        if (variant == 0) ref_ms = t.median_ms;

        Row r;
        r.kernel = "softmax";
        r.dtype = Dtype<T>::name();
        r.variant = variant;
        r.shape = shape;
        r.median_ms = t.median_ms;
        r.min_ms = t.min_ms;
        r.gbps = bytes / (t.median_ms * 1e-3) / 1e9;
        r.tflops = 0.0;
        r.ref_ms = ref_ms;
        r.max_abs_err = err.max_abs;
        r.max_rel_err = err.max_rel;
        r.ok = ok;
        print_row(r);
        if (g_chain > 1) std::fprintf(stderr, "    single launch: %.1f us\n", tt.single_ms * 1e3);
        all_ok = all_ok && ok;
    }

    SPARK_CUDA_CHECK(cudaFree(dx));
    SPARK_CUDA_CHECK(cudaFree(dout));
    return all_ok;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args(argc, argv);
        print_device_banner();
        print_header();

        const int rows = args.geti("rows", 4096);
        const int iters = args.geti("iters", 100);
        g_chain = chain_arg(args);
        std::vector<int> cols_list;
        if (args.has("cols")) {
            cols_list.push_back(args.geti("cols", 1024));
        } else {
            cols_list = {128, 1024, 4096, 16384};
        }

        cudaStream_t stream;
        SPARK_CUDA_CHECK(cudaStreamCreate(&stream));

        bool all_ok = true;
        for (int cols : cols_list) {
            all_ok = run_shape<float>(rows, cols, iters, stream) && all_ok;
            all_ok = run_shape<__nv_bfloat16>(rows, cols, iters, stream) && all_ok;
        }
        SPARK_CUDA_CHECK(cudaStreamDestroy(stream));

        if (!all_ok) {
            std::fprintf(stderr, "bench_softmax: CORRECTNESS FAILURE\n");
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_softmax: %s\n", e.what());
        return 1;
    }
}
