// Benchmark + correctness check for the grouped MoE GEMM (src/kernels/grouped_gemm.cu).
//
//   ./bench_grouped_gemm                                  # three models x 512, 2048, 8192 tokens
//   ./bench_grouped_gemm --model=qwen3 --tokens=2048 --iters=50
//   ./bench_grouped_gemm --dtype=bf16                     # skip the fp8 rows
//
// Models (one expert projection each): Mixtral 8x7B (8 experts, top-2, K 4096 -> N 14336),
// Qwen3-30B-A3B (128 experts, top-8, 2048 -> 768), DeepSeek V3-like (256 experts, top-8,
// 7168 -> 2048). The routing is a seeded random top-k router with uneven load: per (token,
// expert) logits N(0, 1) plus a per-expert bias N(0, --skew), the k largest per token. A is
// [tokens * k, K] sorted by expert, W [E, N, K], both uniform in [-1, 1].
//
// Rows per shape (stderr shows the routing's busiest and idlest expert):
//   grouped bf16    one launch, offsets on the device only
//   cublas_grouped  cublasGemmGroupedBatchedEx, one group per non-empty expert, the counts
//                   already on the host (not timed: the best case for it)
//   cublas_gsync    the same with what a caller has to do to get there: offsets copied to
//                   the host, a stream sync, the pointer arrays built and copied back
//   cublas_loop     one cublasGemmEx per non-empty expert, counts on the host: the GPU side
//                   of the torch loop most MoE code runs
//   grouped fp8     e4m3 A and W, scale_a and one scale per expert
// The bf16 kernel is checked against cuBLAS grouped; the fp8 kernel against cuBLAS grouped
// in bf16 on the dequantized e4m3 values (exact in bf16) with alpha = scale_a * scale_b[e].
// Both must match to 1% of max|C|. ref_ms of the grouped rows is cuBLAS grouped.
//
// stdout: one JSON object per row; stderr: a table with the speedups.

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <numeric>
#include <random>
#include <sstream>
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

struct Model {
    const char* name;
    int E, top_k, K, N;
};
constexpr Model kModels[] = {
    {"mixtral", 8, 2, 4096, 14336},
    {"qwen3", 128, 8, 2048, 768},
    {"deepseek", 256, 8, 7168, 2048},
};

template <typename T>
T* dmalloc(size_t n) {
    T* p = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

__device__ __forceinline__ float hash_uniform(size_t i, uint32_t seed) {
    uint64_t x = (static_cast<uint64_t>(i) + 0x9E3779B97F4A7C15ull) * (seed | 1u);
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    x ^= x >> 33;
    return static_cast<float>(x >> 40) * (2.f / 16777216.f) - 1.f;  // [-1, 1)
}
__global__ void fill_kernel(__nv_bfloat16* p, size_t n, uint32_t seed) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x)
        p[i] = __float2bfloat16(hash_uniform(i, seed));
}
// bf16 -> e4m3 (into q), and the bf16 buffer overwritten with the e4m3 value, exactly.
__global__ void to_fp8_kernel(__nv_bfloat16* p, __nv_fp8_e4m3* q, size_t n) {
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const __nv_fp8_e4m3 v(__bfloat162float(p[i]));
        q[i] = v;
        p[i] = __float2bfloat16(static_cast<float>(v));
    }
}
// max |got - ref| and max |ref| over n values, as float bits (both non-negative).
__global__ void diff_kernel(const __nv_bfloat16* got, const __nv_bfloat16* ref, size_t n,
                            unsigned* out) {
    float d = 0.f, m = 0.f;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float r = __bfloat162float(ref[i]);
        d = fmaxf(d, fabsf(__bfloat162float(got[i]) - r));
        m = fmaxf(m, fabsf(r));
    }
    atomicMax(out, __float_as_uint(d));
    atomicMax(out + 1, __float_as_uint(m));
}

void fill(__nv_bfloat16* p, size_t n, uint32_t seed) {
    fill_kernel<<<1024, 256>>>(p, n, seed);
    SPARK_CHECK_LAUNCH();
}

struct Err {
    double max_abs, max_ref;
};
Err compare(const __nv_bfloat16* got, const __nv_bfloat16* ref, size_t n) {
    unsigned* d = dmalloc<unsigned>(2);
    SPARK_CUDA_CHECK(cudaMemset(d, 0, 2 * sizeof(unsigned)));
    diff_kernel<<<1024, 256>>>(got, ref, n, d);
    SPARK_CHECK_LAUNCH();
    unsigned h[2];
    SPARK_CUDA_CHECK(cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost));
    SPARK_CUDA_CHECK(cudaFree(d));
    float a, b;
    std::memcpy(&a, &h[0], 4);
    std::memcpy(&b, &h[1], 4);
    return {a, b};
}

// Expert row counts of a seeded top-k router with a per-expert bias.
std::vector<int> route(int tokens, int E, int k, double skew, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> bias(E), logit(E);
    for (auto& b : bias) b = static_cast<float>(nd(rng) * skew);
    std::vector<int> counts(E, 0), idx(E);
    for (int t = 0; t < tokens; ++t) {
        for (int e = 0; e < E; ++e) logit[e] = nd(rng) + bias[e];
        std::iota(idx.begin(), idx.end(), 0);
        std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                          [&](int x, int y) { return logit[x] > logit[y]; });
        for (int j = 0; j < k; ++j) ++counts[idx[j]];
    }
    return counts;
}

// cuBLAS grouped GEMM over the non-empty experts. Row-major C_e [m_e, N] = A_e [m_e, K] W_e^T
// is column-major C_e^T [N, m_e] = op_T(W_e) [N, K] * A_e^T [K, m_e].
struct CublasGroups {
    std::vector<cublasOperation_t> ta, tb;
    std::vector<int> m, n, k, lda, ldb, ldc, size;
    std::vector<float> alpha, beta;
    std::vector<const void*> hA, hB;
    std::vector<void*> hC;
    const void** dA = nullptr;
    const void** dB = nullptr;
    void** dC = nullptr;
    int count = 0;

    void build(const std::vector<int>& offs, int E, int N, int K, const __nv_bfloat16* W,
               const __nv_bfloat16* A, __nv_bfloat16* C, const std::vector<float>& alphas) {
        ta.clear();
        tb.clear();
        m.clear();
        n.clear();
        k.clear();
        lda.clear();
        ldb.clear();
        ldc.clear();
        size.clear();
        alpha.clear();
        beta.clear();
        hA.clear();
        hB.clear();
        hC.clear();
        for (int e = 0; e < E; ++e) {
            const int rows = offs[e + 1] - offs[e];
            if (rows == 0) continue;
            ta.push_back(CUBLAS_OP_T);
            tb.push_back(CUBLAS_OP_N);
            m.push_back(N);
            n.push_back(rows);
            k.push_back(K);
            lda.push_back(K);
            ldb.push_back(K);
            ldc.push_back(N);
            size.push_back(1);
            alpha.push_back(alphas[e]);
            beta.push_back(0.f);
            hA.push_back(W + static_cast<size_t>(e) * N * K);
            hB.push_back(A + static_cast<size_t>(offs[e]) * K);
            hC.push_back(C + static_cast<size_t>(offs[e]) * N);
        }
        count = static_cast<int>(m.size());
        if (!dA) {
            dA = dmalloc<const void*>(E);
            dB = dmalloc<const void*>(E);
            dC = dmalloc<void*>(E);
        }
    }
    void upload(cudaStream_t s) {
        SPARK_CUDA_CHECK(
            cudaMemcpyAsync(dA, hA.data(), count * sizeof(void*), cudaMemcpyHostToDevice, s));
        SPARK_CUDA_CHECK(
            cudaMemcpyAsync(dB, hB.data(), count * sizeof(void*), cudaMemcpyHostToDevice, s));
        SPARK_CUDA_CHECK(
            cudaMemcpyAsync(dC, hC.data(), count * sizeof(void*), cudaMemcpyHostToDevice, s));
    }
    void run(cublasHandle_t h) {
        CUBLAS_CHECK(cublasGemmGroupedBatchedEx(
            h, ta.data(), tb.data(), m.data(), n.data(), k.data(), alpha.data(), dA, CUDA_R_16BF,
            lda.data(), dB, CUDA_R_16BF, ldb.data(), beta.data(), dC, CUDA_R_16BF, ldc.data(),
            count, size.data(), CUBLAS_COMPUTE_32F));
    }
    void free_all() {
        if (dA) SPARK_CUDA_CHECK(cudaFree(dA));
        if (dB) SPARK_CUDA_CHECK(cudaFree(dB));
        if (dC) SPARK_CUDA_CHECK(cudaFree(dC));
        dA = nullptr;
        dB = nullptr;
        dC = nullptr;
    }
};

void cublas_loop(cublasHandle_t h, const std::vector<int>& offs, int E, int N, int K,
                 const __nv_bfloat16* W, const __nv_bfloat16* A, __nv_bfloat16* C) {
    const float alpha = 1.f, beta = 0.f;
    for (int e = 0; e < E; ++e) {
        const int rows = offs[e + 1] - offs[e];
        if (rows == 0) continue;
        CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, rows, K, &alpha,
                                  W + static_cast<size_t>(e) * N * K, CUDA_R_16BF, K,
                                  A + static_cast<size_t>(offs[e]) * K, CUDA_R_16BF, K, &beta,
                                  C + static_cast<size_t>(offs[e]) * N, CUDA_R_16BF, N,
                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    }
}

spark::bench::Row make_row(const char* kernel, const char* dtype, const std::string& shape,
                           const spark::bench::Timing& t, double flops, double ref_ms) {
    spark::bench::Row r;
    r.kernel = kernel;
    r.dtype = dtype;
    r.variant = 0;
    r.shape = shape;
    r.median_ms = t.median_ms;
    r.min_ms = t.min_ms;
    r.tflops = flops / (t.median_ms * 1e-3) / 1e12;
    r.ref_ms = ref_ms;
    return r;
}

bool run_one(cublasHandle_t handle, cudaStream_t stream, const Model& md, int tokens, bool do_bf16,
             bool do_fp8, double skew, int iters) {
    const int E = md.E, N = md.N, K = md.K;
    const std::vector<int> counts = route(tokens, E, md.top_k, skew, 1234u + tokens);
    std::vector<int> offs(E + 1, 0);
    for (int e = 0; e < E; ++e) offs[e + 1] = offs[e] + counts[e];
    const int T = offs[E];
    const int busiest = *std::max_element(counts.begin(), counts.end());
    const int idlest = *std::min_element(counts.begin(), counts.end());
    const int empty = static_cast<int>(std::count(counts.begin(), counts.end(), 0));
    std::ostringstream shape;
    shape << md.name << " t=" << tokens;
    std::fprintf(stderr,
                 "# %s: E=%d top-%d K=%d N=%d, %d rows, mean %.1f per expert, max %d, "
                 "min %d, %d empty\n",
                 shape.str().c_str(), E, md.top_k, K, N, T, static_cast<double>(T) / E, busiest,
                 idlest, empty);
    const double flops = 2.0 * T * N * K;

    const size_t nA = static_cast<size_t>(T) * K, nW = static_cast<size_t>(E) * N * K,
                 nC = static_cast<size_t>(T) * N;
    __nv_bfloat16* dA = dmalloc<__nv_bfloat16>(nA);
    __nv_bfloat16* dW = dmalloc<__nv_bfloat16>(nW);
    __nv_bfloat16* dC = dmalloc<__nv_bfloat16>(nC);
    __nv_bfloat16* dRef = dmalloc<__nv_bfloat16>(nC);
    int32_t* dOff = dmalloc<int32_t>(E + 1);
    SPARK_CUDA_CHECK(
        cudaMemcpy(dOff, offs.data(), (E + 1) * sizeof(int32_t), cudaMemcpyHostToDevice));
    fill(dA, nA, 11u);
    fill(dW, nW, 23u);
    SPARK_CUDA_CHECK(cudaDeviceSynchronize());

    bool ok = true;
    CublasGroups cg;
    std::vector<float> ones(E, 1.f);
    cg.build(offs, E, N, K, dW, dA, dRef, ones);
    cg.upload(stream);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (do_bf16) {
        cg.run(handle);
        spark::grouped_gemm_bf16(dA, dW, dOff, dC, T, E, N, K, stream);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        const Err err = compare(dC, dRef, nC);
        const bool good = err.max_abs <= 1e-2 * err.max_ref;
        ok = ok && good;

        const auto t_cg = spark::bench::time_kernel([&] { cg.run(handle); }, stream, 5, iters);
        int32_t* hOff = nullptr;
        SPARK_CUDA_CHECK(cudaMallocHost(&hOff, (E + 1) * sizeof(int32_t)));
        std::vector<int> offs_sync(E + 1);
        const auto t_gsync = spark::bench::time_kernel(
            [&] {
                SPARK_CUDA_CHECK(cudaMemcpyAsync(hOff, dOff, (E + 1) * sizeof(int32_t),
                                                 cudaMemcpyDeviceToHost, stream));
                SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
                offs_sync.assign(hOff, hOff + E + 1);
                cg.build(offs_sync, E, N, K, dW, dA, dRef, ones);
                cg.upload(stream);
                cg.run(handle);
            },
            stream, 5, iters);
        SPARK_CUDA_CHECK(cudaFreeHost(hOff));
        const auto t_loop = spark::bench::time_kernel(
            [&] { cublas_loop(handle, offs, E, N, K, dW, dA, dRef); }, stream, 5, iters);
        const auto t_sk = spark::bench::time_kernel(
            [&] { spark::grouped_gemm_bf16(dA, dW, dOff, dC, T, E, N, K, stream); }, stream, 5,
            iters);

        auto r = make_row("grouped", "bf16", shape.str(), t_sk, flops, t_cg.median_ms);
        r.max_abs_err = err.max_abs;
        r.max_rel_err = err.max_abs / std::max(1e-30, err.max_ref);
        r.ok = good;
        spark::bench::print_row(r);
        spark::bench::print_row(make_row("cublas_grouped", "bf16", shape.str(), t_cg, flops, 0));
        spark::bench::print_row(make_row("cublas_gsync", "bf16", shape.str(), t_gsync, flops, 0));
        spark::bench::print_row(make_row("cublas_loop", "bf16", shape.str(), t_loop, flops, 0));
        std::fprintf(stderr,
                     "#   bf16 speedup: %.2fx over cuBLAS grouped, %.2fx over it with "
                     "the sync, %.2fx over the per-expert loop\n",
                     t_cg.median_ms / t_sk.median_ms, t_gsync.median_ms / t_sk.median_ms,
                     t_loop.median_ms / t_sk.median_ms);
    }

    if (do_fp8) {
        // e4m3 copies; the bf16 buffers become their exact dequantized values for the reference.
        __nv_fp8_e4m3* qA = dmalloc<__nv_fp8_e4m3>(nA);
        __nv_fp8_e4m3* qW = dmalloc<__nv_fp8_e4m3>(nW);
        to_fp8_kernel<<<1024, 256>>>(dA, qA, nA);
        to_fp8_kernel<<<1024, 256>>>(dW, qW, nW);
        SPARK_CHECK_LAUNCH();
        std::vector<float> sb(E);
        const float sa = 0.5f;
        for (int e = 0; e < E; ++e) sb[e] = 0.25f + 0.0625f * (e % 8);
        float* dSa = dmalloc<float>(1);
        float* dSb = dmalloc<float>(E);
        SPARK_CUDA_CHECK(cudaMemcpy(dSa, &sa, sizeof(float), cudaMemcpyHostToDevice));
        SPARK_CUDA_CHECK(cudaMemcpy(dSb, sb.data(), E * sizeof(float), cudaMemcpyHostToDevice));
        std::vector<float> alphas(E);
        for (int e = 0; e < E; ++e) alphas[e] = sa * sb[e];
        cg.build(offs, E, N, K, dW, dA, dRef, alphas);
        cg.upload(stream);
        cg.run(handle);
        spark::grouped_gemm_fp8(qA, qW, dOff, dC, T, E, N, K, dSa, dSb, stream);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        const Err err = compare(dC, dRef, nC);
        const bool good = err.max_abs <= 1e-2 * err.max_ref;
        ok = ok && good;
        const auto t_cg = spark::bench::time_kernel([&] { cg.run(handle); }, stream, 5, iters);
        const auto t_sk = spark::bench::time_kernel(
            [&] { spark::grouped_gemm_fp8(qA, qW, dOff, dC, T, E, N, K, dSa, dSb, stream); },
            stream, 5, iters);
        auto r = make_row("grouped", "fp8", shape.str(), t_sk, flops, t_cg.median_ms);
        r.max_abs_err = err.max_abs;
        r.max_rel_err = err.max_abs / std::max(1e-30, err.max_ref);
        r.ok = good;
        spark::bench::print_row(r);
        std::fprintf(stderr, "#   fp8 speedup: %.2fx over cuBLAS grouped in bf16\n",
                     t_cg.median_ms / t_sk.median_ms);
        SPARK_CUDA_CHECK(cudaFree(qA));
        SPARK_CUDA_CHECK(cudaFree(qW));
        SPARK_CUDA_CHECK(cudaFree(dSa));
        SPARK_CUDA_CHECK(cudaFree(dSb));
    }

    cg.free_all();
    SPARK_CUDA_CHECK(cudaFree(dA));
    SPARK_CUDA_CHECK(cudaFree(dW));
    SPARK_CUDA_CHECK(cudaFree(dC));
    SPARK_CUDA_CHECK(cudaFree(dRef));
    SPARK_CUDA_CHECK(cudaFree(dOff));
    return ok;
}

std::vector<int> parse_list(const std::string& s) {
    std::vector<int> v;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) v.push_back(std::stoi(item));
    return v;
}

int run(int argc, char** argv) {
    spark::bench::Args args(argc, argv);
    const int iters = args.geti("iters", 20);
    const std::string model = args.get("model", "all");
    const std::string dtype = args.get("dtype", "both");
    const std::vector<int> tokens = parse_list(args.get("tokens", "512,2048,8192"));
    const double skew = std::stod(args.get("skew", "1.0"));
    spark::bench::print_device_banner();

    cudaStream_t stream = nullptr;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    spark::bench::print_header();
    bool all_ok = true;
    for (const Model& md : kModels) {
        if (model != "all" && model != md.name) continue;
        for (int t : tokens)
            all_ok = run_one(handle, stream, md, t, dtype != "fp8", dtype != "bf16", skew, iters) &&
                     all_ok;
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    if (!all_ok) {
        std::fprintf(stderr, "CORRECTNESS FAILURE\n");
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_grouped_gemm: %s\n", e.what());
        return 2;
    }
}
