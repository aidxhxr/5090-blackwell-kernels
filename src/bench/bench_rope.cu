// Benchmark: RoPE on q and k plus the K/V cache append from a fused q|k|v projection.
//   ./bench_rope [--b=<B> --s=<S> --hq=<H_q> --hkv=<H_kv> --d=<D> --pos0=<P> --cap=<C>]
//   [--iters=<N>]
// Default shapes are Llama-3-8B's heads (32 query, 8 K/V, 128 wide): a 4096-token prefill,
// an 8192-token prefill, and decode steps of 1 and 8 sequences against a 4096-token cache.
// Validates against a CPU double-precision reference; exits 1 on mismatch.
#include <cmath>
#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "spark/kernels.h"

using namespace spark::bench;

namespace {

struct Shape {
    int B, S, H_q, H_kv, D, pos0, cap;
};

std::string shape_name(const Shape& s) {
    return "b" + std::to_string(s.B) + "_s" + std::to_string(s.S) + "_hq" + std::to_string(s.H_q) +
           "_hkv" + std::to_string(s.H_kv) + "_d" + std::to_string(s.D) + "_pos" +
           std::to_string(s.pos0);
}

bool run_shape(const Shape& sh, int iters, cudaStream_t stream) {
    const int width = (sh.H_q + 2 * sh.H_kv) * sh.D;
    const size_t tokens = static_cast<size_t>(sh.B) * sh.S;
    const size_t n_qkv = tokens * width;
    const size_t n_q = tokens * sh.H_q * sh.D;
    const size_t n_cache = static_cast<size_t>(sh.B) * sh.H_kv * sh.cap * sh.D;
    const size_t n_tab = static_cast<size_t>(sh.pos0 + sh.S) * sh.D;

    std::vector<float> h_qkv(n_qkv);
    fill_uniform(h_qkv, -2.0f, 2.0f, 1);
    std::vector<__nv_bfloat16> t_qkv(n_qkv);
    for (size_t i = 0; i < n_qkv; ++i) {
        t_qkv[i] = __float2bfloat16(h_qkv[i]);
        h_qkv[i] = __bfloat162float(t_qkv[i]);
    }
    // rotate-half tables: theta 500000, column d and d + D/2 share a frequency
    std::vector<float> h_cos(n_tab), h_sin(n_tab);
    for (int p = 0; p < sh.pos0 + sh.S; ++p) {
        for (int d = 0; d < sh.D; ++d) {
            const int f = d % (sh.D / 2);
            const double inv = std::pow(500000.0, -2.0 * f / sh.D);
            h_cos[static_cast<size_t>(p) * sh.D + d] = static_cast<float>(std::cos(p * inv));
            h_sin[static_cast<size_t>(p) * sh.D + d] = static_cast<float>(std::sin(p * inv));
        }
    }

    // CPU reference in double: q [B, H_q, S, D], k and v caches [B, H_kv, cap, D]
    std::vector<float> ref_q(n_q), ref_k(n_cache, 0.0f), ref_v(n_cache, 0.0f);
    for (int b = 0; b < sh.B; ++b) {
        for (int s = 0; s < sh.S; ++s) {
            const float* row = h_qkv.data() + (static_cast<size_t>(b) * sh.S + s) * width;
            const int pos = sh.pos0 + s;
            const float* cs = h_cos.data() + static_cast<size_t>(pos) * sh.D;
            const float* sn = h_sin.data() + static_cast<size_t>(pos) * sh.D;
            for (int h = 0; h < sh.H_q + sh.H_kv; ++h) {
                const float* x = row + h * sh.D;
                float* out =
                    h < sh.H_q
                        ? ref_q.data() + ((static_cast<size_t>(b) * sh.H_q + h) * sh.S + s) * sh.D
                        : ref_k.data() +
                              ((static_cast<size_t>(b) * sh.H_kv + h - sh.H_q) * sh.cap + pos) *
                                  sh.D;
                for (int d = 0; d < sh.D / 2; ++d) {
                    const double x1 = x[d], x2 = x[d + sh.D / 2];
                    out[d] = static_cast<float>(x1 * cs[d] - x2 * sn[d]);
                    out[d + sh.D / 2] = static_cast<float>(x2 * cs[d] + x1 * sn[d]);
                }
            }
            for (int vh = 0; vh < sh.H_kv; ++vh) {
                const float* x = row + (sh.H_q + sh.H_kv + vh) * sh.D;
                float* out =
                    ref_v.data() + ((static_cast<size_t>(b) * sh.H_kv + vh) * sh.cap + pos) * sh.D;
                for (int d = 0; d < sh.D; ++d) out[d] = x[d];
            }
        }
    }

    __nv_bfloat16 *d_qkv = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_cos = nullptr, *d_sin = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&d_qkv, n_qkv * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_q, n_q * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_k, n_cache * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_v, n_cache * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_cos, n_tab * sizeof(float)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_sin, n_tab * sizeof(float)));
    SPARK_CUDA_CHECK(
        cudaMemcpy(d_qkv, t_qkv.data(), n_qkv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(
        cudaMemcpy(d_cos, h_cos.data(), n_tab * sizeof(float), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(
        cudaMemcpy(d_sin, h_sin.data(), n_tab * sizeof(float), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemset(d_k, 0, n_cache * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMemset(d_v, 0, n_cache * sizeof(__nv_bfloat16)));

    auto launch = [&] {
        spark::rope_append_bf16(d_qkv, d_cos, d_sin, d_q, d_k, d_v, sh.B, sh.S, sh.H_q, sh.H_kv,
                                sh.D, sh.pos0, sh.cap, stream);
    };
    launch();
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto fetch = [&](const __nv_bfloat16* d, size_t n) {
        std::vector<__nv_bfloat16> t(n);
        SPARK_CUDA_CHECK(
            cudaMemcpy(t.data(), d, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        std::vector<float> f(n);
        for (size_t i = 0; i < n; ++i) f[i] = __bfloat162float(t[i]);
        return f;
    };
    const std::vector<float> got_q = fetch(d_q, n_q), got_k = fetch(d_k, n_cache),
                             got_v = fetch(d_v, n_cache);
    ErrorStats err = compare(got_q.data(), ref_q.data(), n_q);
    const ErrorStats ek = compare(got_k.data(), ref_k.data(), n_cache);
    const ErrorStats ev = compare(got_v.data(), ref_v.data(), n_cache);
    err.max_abs = std::max(err.max_abs, std::max(ek.max_abs, ev.max_abs));
    err.max_rel = std::max(err.max_rel, std::max(ek.max_rel, ev.max_rel));
    // one bf16 rounding of an fp32 product sum; |x| <= 2 so 2e-2 absolute covers it
    const bool ok = err.max_abs <= 2e-2;

    const Timing t = time_kernel(launch, stream, 10, iters);
    // traffic: the qkv rows read once, q and the k / v slots written once
    const double moved_gb = 2.0 * static_cast<double>(n_qkv) * sizeof(__nv_bfloat16) / 1e9;
    Row r;
    r.kernel = "rope";
    r.dtype = "bf16";
    r.variant = 0;
    r.shape = shape_name(sh);
    r.median_ms = t.median_ms;
    r.min_ms = t.min_ms;
    r.gbps = moved_gb / (t.median_ms * 1e-3);
    r.max_abs_err = err.max_abs;
    r.max_rel_err = err.max_rel;
    r.ok = ok;
    print_row(r);

    SPARK_CUDA_CHECK(cudaFree(d_qkv));
    SPARK_CUDA_CHECK(cudaFree(d_q));
    SPARK_CUDA_CHECK(cudaFree(d_k));
    SPARK_CUDA_CHECK(cudaFree(d_v));
    SPARK_CUDA_CHECK(cudaFree(d_cos));
    SPARK_CUDA_CHECK(cudaFree(d_sin));
    return ok;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args(argc, argv);
        print_device_banner();
        const int iters = args.geti("iters", 100);
        std::vector<Shape> shapes;
        if (args.has("s")) {
            const int S = args.geti("s", 1);
            const int pos0 = args.geti("pos0", 0);
            shapes.push_back({args.geti("b", 1), S, args.geti("hq", 32), args.geti("hkv", 8),
                              args.geti("d", 128), pos0, args.geti("cap", pos0 + S)});
        } else {
            shapes = {{1, 4096, 32, 8, 128, 0, 4096},
                      {1, 8192, 32, 8, 128, 0, 8192},
                      {1, 1, 32, 8, 128, 4095, 4096},
                      {8, 1, 32, 8, 128, 4095, 4096}};
        }
        cudaStream_t stream;
        SPARK_CUDA_CHECK(cudaStreamCreate(&stream));
        print_header();
        bool ok = true;
        for (const Shape& s : shapes) ok = run_shape(s, iters, stream) && ok;
        SPARK_CUDA_CHECK(cudaStreamDestroy(stream));
        if (!ok) {
            std::fprintf(stderr, "VALIDATION FAILED\n");
            return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 2;
    }
}
