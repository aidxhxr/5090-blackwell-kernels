// Benchmark + correctness check for the W4A16 GEMM ladder (int4 weights, bf16 activations).
//
//   ./bench_w4gemm                               # Llama-3-8B projections x M = 1, 4, 16, 64, 256
//   ./bench_w4gemm --m=1 --n=4096 --k=4096 --variant=2 --iters=50
//   ./bench_w4gemm --m=16 --n=28672 --k=4096 --asym   # asymmetric weights (zero points)
//   ./bench_w4gemm --m=1 --n=4096 --k=4096 --chain=20 # also time 20 launches per event pair
//
// For every weight shape the bench quantizes a random bf16 W [K, N] on the GPU
// (w4_quantize_bf16) and checks the codes, scales and zero points bit for bit against the same
// round-to-nearest on the CPU, repacks it (w4_repack), and dequantizes it on the CPU into the
// bf16 matrix the kernels must be multiplying. The reference is cuBLAS on that matrix; every
// variant must match it to 2% of max|C| (the same weights, only the fp32 summation order and
// the final rounding differ).
//
// Timing: the packed weights (with their scales) are rotated over copies that exceed the
// 96 MB L2, as a decode step reads every layer's weights once. ref_ms is the bf16 path this
// op replaces, hgemm_bf16 variant 6 (its decode kernel for M <= 64) on the dequantized
// weights, rotated the same way; cuBLAS on the same matrix is printed on stderr.
// gbps = the traffic floor: A, the packed weights, the scales (and zeros), C, once each.
//
// stdout: one JSON object per row (collected by scripts/run_all_benches.sh)
// stderr: human-readable table incl. the speedup over the bf16 path

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <functional>
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

constexpr int kG = spark::kW4GroupSize;

// Row-major C = A*B via column-major cuBLAS: C^T = B^T * A^T, i.e. GemmEx(N, M, K, B, A).
void cublas_gemm(cublasHandle_t handle, const __nv_bfloat16* A, const __nv_bfloat16* B,
                 __nv_bfloat16* C, int M, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, CUDA_R_16BF, N,
                              A, CUDA_R_16BF, K, &beta, C, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
}

template <typename T>
T* dmalloc(size_t n) {
    T* p = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

// The quantizer of kernels.h on the CPU, in the same fp32 operations, so the GPU's codes,
// scales and zero points can be compared bit for bit.
void quantize_host(const std::vector<__nv_bfloat16>& W, int K, int N, bool asym,
                   std::vector<uint8_t>& q, std::vector<__nv_bfloat16>& s,
                   std::vector<uint8_t>& z) {
    const int G = K / kG;
    q.assign(static_cast<size_t>(K) * N, 0);
    s.assign(static_cast<size_t>(G) * N, __float2bfloat16(0.f));
    z.assign(static_cast<size_t>(G) * N, 8);
    std::vector<float> lo(N), hi(N), amax(N), sc(N);
    std::vector<int> zz(N);
    for (int grp = 0; grp < G; ++grp) {
        std::fill(lo.begin(), lo.end(), 0.f);
        std::fill(hi.begin(), hi.end(), 0.f);
        std::fill(amax.begin(), amax.end(), 0.f);
        for (int k = grp * kG; k < (grp + 1) * kG; ++k)  // row by row: W is [K][N]
            for (int n = 0; n < N; ++n) {
                const float w = __bfloat162float(W[static_cast<size_t>(k) * N + n]);
                lo[n] = std::fmin(lo[n], w);
                hi[n] = std::fmax(hi[n], w);
                amax[n] = std::fmax(amax[n], std::fabs(w));
            }
        for (int n = 0; n < N; ++n) {
            const __nv_bfloat16 sb =
                __float2bfloat16(asym ? (hi[n] - lo[n]) / 15.0f : amax[n] / 7.0f);
            sc[n] = __bfloat162float(sb);
            zz[n] = 8;
            if (asym)
                zz[n] = sc[n] > 0.f ? static_cast<int>(std::fmin(
                                          std::fmax(std::rint(-lo[n] / sc[n]), 0.f), 15.f))
                                    : 0;
            s[static_cast<size_t>(grp) * N + n] = sb;
            z[static_cast<size_t>(grp) * N + n] = static_cast<uint8_t>(zz[n]);
        }
        for (int k = grp * kG; k < (grp + 1) * kG; ++k)
            for (int n = 0; n < N; ++n) {
                const float w = __bfloat162float(W[static_cast<size_t>(k) * N + n]);
                int qq = asym ? zz[n] : 8;
                if (sc[n] > 0.f) {
                    const float r = std::rint(w / sc[n]) + (asym ? static_cast<float>(zz[n]) : 0.f);
                    qq = asym ? static_cast<int>(std::fmin(std::fmax(r, 0.f), 15.f))
                              : static_cast<int>(std::fmin(std::fmax(r, -8.f), 7.f)) + 8;
                }
                q[static_cast<size_t>(k) * N + n] = static_cast<uint8_t>(qq);
            }
    }
}

// One quantized weight on the GPU, rotated over `copies` copies of (packed, scales, zeros).
struct Weight {
    int N = 0, K = 0;
    bool asym = false;
    int copies = 1;
    size_t packed_words = 0, groups_n = 0;
    int32_t* packed = nullptr;        // copies x packed_words
    __nv_bfloat16* scales = nullptr;  // copies x groups_n
    uint8_t* zeros = nullptr;         // copies x groups_n, asym only
    int bf16_copies = 1;
    __nv_bfloat16* deq = nullptr;  // bf16_copies x K x N, the dequantized weights
    bool quant_ok = true;
    double quant_ms = 0, repack_ms = 0;

    size_t bytes_per_copy() const {
        return packed_words * 4 + groups_n * 2 + (asym ? groups_n : 0);
    }
    const int32_t* p(int c) const {
        return packed + static_cast<size_t>(c % copies) * packed_words;
    }
    const __nv_bfloat16* s(int c) const {
        return scales + static_cast<size_t>(c % copies) * groups_n;
    }
    const uint8_t* z(int c) const {
        return asym ? zeros + static_cast<size_t>(c % copies) * groups_n : nullptr;
    }
    const __nv_bfloat16* d(int c) const {
        return deq + static_cast<size_t>(c % bf16_copies) * K * N;
    }
    void free_all() {
        for (void* x : {static_cast<void*>(packed), static_cast<void*>(scales),
                        static_cast<void*>(zeros), static_cast<void*>(deq)})
            if (x) SPARK_CUDA_CHECK(cudaFree(x));
    }
};

Weight make_weight(int N, int K, bool asym, cudaStream_t stream) {
    Weight w;
    w.N = N;
    w.K = K;
    w.asym = asym;
    const size_t nW = static_cast<size_t>(K) * N;
    std::vector<float> hWf(nW);
    spark::bench::fill_uniform(hWf, -1.0f, 1.0f, 5678);
    std::vector<__nv_bfloat16> hW(nW);
    for (size_t i = 0; i < nW; ++i) hW[i] = __float2bfloat16(hWf[i]);
    hWf.clear();
    hWf.shrink_to_fit();

    w.packed_words = nW / 8;
    w.groups_n = static_cast<size_t>(K / kG) * N;
    constexpr size_t kRotate = size_t{256} << 20;  // past the 96 MB L2
    w.copies = static_cast<int>(kRotate / w.bytes_per_copy()) + 1;
    w.bf16_copies = static_cast<int>(kRotate / (nW * 2)) + 1;

    __nv_bfloat16* dW = dmalloc<__nv_bfloat16>(nW);
    SPARK_CUDA_CHECK(cudaMemcpy(dW, hW.data(), nW * 2, cudaMemcpyHostToDevice));
    int32_t* dq = dmalloc<int32_t>(w.packed_words);
    w.packed = dmalloc<int32_t>(w.packed_words * w.copies);
    w.scales = dmalloc<__nv_bfloat16>(w.groups_n * w.copies);
    if (asym) w.zeros = dmalloc<uint8_t>(w.groups_n * w.copies);

    // Quantize and repack (timed once each, cold: they run once per weight at load time).
    cudaEvent_t e0, e1, e2;
    SPARK_CUDA_CHECK(cudaEventCreate(&e0));
    SPARK_CUDA_CHECK(cudaEventCreate(&e1));
    SPARK_CUDA_CHECK(cudaEventCreate(&e2));
    SPARK_CUDA_CHECK(cudaEventRecord(e0, stream));
    spark::w4_quantize_bf16(dW, dq, w.scales, w.zeros, K, N, stream);
    SPARK_CUDA_CHECK(cudaEventRecord(e1, stream));
    spark::w4_repack(dq, w.packed, K, N, stream);
    SPARK_CUDA_CHECK(cudaEventRecord(e2, stream));
    SPARK_CUDA_CHECK(cudaEventSynchronize(e2));
    float ms = 0.f;
    SPARK_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    w.quant_ms = ms;
    SPARK_CUDA_CHECK(cudaEventElapsedTime(&ms, e1, e2));
    w.repack_ms = ms;
    for (cudaEvent_t e : {e0, e1, e2}) SPARK_CUDA_CHECK(cudaEventDestroy(e));

    // The GPU quantizer against the CPU one, bit for bit.
    std::vector<int32_t> hq(w.packed_words);
    std::vector<__nv_bfloat16> hs(w.groups_n);
    std::vector<uint8_t> hz(w.groups_n, 8);
    SPARK_CUDA_CHECK(cudaMemcpy(hq.data(), dq, w.packed_words * 4, cudaMemcpyDeviceToHost));
    SPARK_CUDA_CHECK(cudaMemcpy(hs.data(), w.scales, w.groups_n * 2, cudaMemcpyDeviceToHost));
    if (asym) SPARK_CUDA_CHECK(cudaMemcpy(hz.data(), w.zeros, w.groups_n, cudaMemcpyDeviceToHost));
    std::vector<uint8_t> rq, rz;
    std::vector<__nv_bfloat16> rs;
    quantize_host(hW, K, N, asym, rq, rs, rz);
    size_t bad = 0;
    for (size_t i = 0; i < w.groups_n; ++i) {
        const auto a = reinterpret_cast<const uint16_t&>(hs[i]);
        const auto b = reinterpret_cast<const uint16_t&>(rs[i]);
        bad += (a != b) + (asym && hz[i] != rz[i]);
    }
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) {
            const uint32_t word = static_cast<uint32_t>(hq[static_cast<size_t>(k / 8) * N + n]);
            bad += ((word >> (4 * (k % 8))) & 0xF) != rq[static_cast<size_t>(k) * N + n];
        }
    if (bad) {
        std::fprintf(stderr, "  FAIL w4_quantize %dx%d%s: %zu codes/scales differ from the CPU\n",
                     K, N, asym ? " asym" : "", bad);
        w.quant_ok = false;
    }

    // The bf16 weights the kernels multiply, for the reference and for the bf16 timing.
    std::vector<__nv_bfloat16> hd(nW);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) {
            const size_t gi = static_cast<size_t>(k / kG) * N + n;
            const int zz = asym ? rz[gi] : 8;
            const float v = static_cast<float>(rq[static_cast<size_t>(k) * N + n] - zz) *
                            __bfloat162float(rs[gi]);
            hd[static_cast<size_t>(k) * N + n] = __float2bfloat16(v);
        }
    w.deq = dmalloc<__nv_bfloat16>(nW * w.bf16_copies);
    for (int c = 0; c < w.bf16_copies; ++c)
        SPARK_CUDA_CHECK(cudaMemcpy(w.deq + c * nW, hd.data(), nW * 2, cudaMemcpyHostToDevice));
    for (int c = 1; c < w.copies; ++c) {
        SPARK_CUDA_CHECK(cudaMemcpy(w.packed + c * w.packed_words, w.packed, w.packed_words * 4,
                                    cudaMemcpyDeviceToDevice));
        SPARK_CUDA_CHECK(cudaMemcpy(w.scales + c * w.groups_n, w.scales, w.groups_n * 2,
                                    cudaMemcpyDeviceToDevice));
        if (asym)
            SPARK_CUDA_CHECK(cudaMemcpy(w.zeros + c * w.groups_n, w.zeros, w.groups_n,
                                        cudaMemcpyDeviceToDevice));
    }
    SPARK_CUDA_CHECK(cudaFree(dW));
    SPARK_CUDA_CHECK(cudaFree(dq));
    std::fprintf(stderr,
                 "  weight %dx%d%s: quantize %.3f ms, repack %.3f ms (cold, once per weight), "
                 "%.1f MB packed + scales, rotated over %d copies\n",
                 K, N, asym ? " asym" : "", w.quant_ms, w.repack_ms, w.bytes_per_copy() / 1e6,
                 w.copies);
    return w;
}

// Per-launch time of `chain` launches between one event pair (median over `iters` pairs).
double time_chain(const std::function<void()>& fn, cudaStream_t stream, int chain, int iters) {
    const auto t = spark::bench::time_kernel(
        [&] {
            for (int i = 0; i < chain; ++i) fn();
        },
        stream, 3, iters);
    return t.median_ms / chain;
}

std::vector<float> to_host_f32(const __nv_bfloat16* d, size_t n) {
    std::vector<__nv_bfloat16> h(n);
    SPARK_CUDA_CHECK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = __bfloat162float(h[i]);
    return out;
}

// Runs every requested variant on one M against one weight. Returns false on a mismatch.
bool run_m(cublasHandle_t handle, cudaStream_t stream, const Weight& w, int M,
           const std::vector<int>& variants, int iters, int chain, bool skip_slow_v0) {
    const int N = w.N, K = w.K;
    const size_t nA = static_cast<size_t>(M) * K, nC = static_cast<size_t>(M) * N;
    std::vector<float> hA(nA);
    spark::bench::fill_uniform(hA, -1.0f, 1.0f, 1234 + M);
    std::vector<__nv_bfloat16> hAb(nA);
    for (size_t i = 0; i < nA; ++i) hAb[i] = __float2bfloat16(hA[i]);
    __nv_bfloat16* dA = dmalloc<__nv_bfloat16>(nA);
    __nv_bfloat16* dC = dmalloc<__nv_bfloat16>(nC);
    __nv_bfloat16* dRef = dmalloc<__nv_bfloat16>(nC);
    SPARK_CUDA_CHECK(cudaMemcpy(dA, hAb.data(), nA * 2, cudaMemcpyHostToDevice));

    cublas_gemm(handle, dA, w.d(0), dRef, M, N, K);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto ref = to_host_f32(dRef, nC);
    double max_ref = 0.0;
    for (size_t i = 0; i < nC; ++i) max_ref = std::max(max_ref, std::fabs((double)ref[i]));
    const double tol = 2e-2 * max_ref + 1e-3;

    // The bf16 path this op replaces (hgemm's default variant, or cuBLAS on a shape hgemm
    // refuses), and cuBLAS, on the dequantized weights.
    int turn = 0;
    const bool hgemm_ok = spark::hgemm_supports(M, N, K, 6);
    auto bf16_gemm = [&] {
        if (hgemm_ok)
            spark::hgemm_bf16(dA, w.d(turn++), dC, M, N, K, 6, stream);
        else
            cublas_gemm(handle, dA, w.d(turn++), dC, M, N, K);
    };
    const auto t_hgemm = spark::bench::time_kernel(bf16_gemm, stream, 5, iters);
    const auto t_cublas = spark::bench::time_kernel(
        [&] { cublas_gemm(handle, dA, w.d(turn++), dRef, M, N, K); }, stream, 5, iters);
    double chain_hgemm = 0;
    if (chain > 1) chain_hgemm = time_chain(bf16_gemm, stream, chain, iters);

    const std::string shape = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    const double bytes = 2.0 * nA + static_cast<double>(w.bytes_per_copy()) + 2.0 * nC;
    const double flops = 2.0 * M * N * static_cast<double>(K);
    bool all_ok = true;
    for (int v : variants) {
        if (v == 0 && skip_slow_v0 && M > 16) {
            std::fprintf(stderr, "  skip variant 0 at %s: the naive rung is only timed to M = 16\n",
                         shape.c_str());
            continue;
        }
        SPARK_CUDA_CHECK(cudaMemset(dC, 0, nC * 2));
        spark::w4gemm_bf16(dA, w.p(0), w.s(0), w.z(0), dC, M, N, K, v, stream);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto got = to_host_f32(dC, nC);
        const auto err = spark::bench::compare(got.data(), ref.data(), nC);
        const bool ok = err.max_abs <= tol && w.quant_ok;
        if (!ok)
            std::fprintf(stderr, "  FAIL variant %d shape %s: max_abs=%.4e tol=%.4e\n", v,
                         shape.c_str(), err.max_abs, tol);
        const auto t = spark::bench::time_kernel(
            [&] {
                spark::w4gemm_bf16(dA, w.p(turn), w.s(turn), w.z(turn), dC, M, N, K, v, stream);
                ++turn;
            },
            stream, 5, v == 0 ? std::min(iters, 10) : iters);

        spark::bench::Row row;
        row.kernel = "w4gemm";
        row.dtype = w.asym ? "w4a16_asym" : "w4a16";
        row.variant = v;
        row.shape = shape;
        row.median_ms = t.median_ms;
        row.min_ms = t.min_ms;
        row.gbps = bytes / (t.median_ms * 1e-3) / 1e9;
        row.tflops = flops / (t.median_ms * 1e-3) / 1e12;
        row.ref_ms = t_hgemm.median_ms;
        row.max_abs_err = err.max_abs;
        row.max_rel_err = err.max_rel;
        row.ok = ok;
        spark::bench::print_row(row);
        std::fprintf(stderr,
                     "    %.2fx the bf16 hgemm (%.1f us) | cuBLAS bf16 %.1f us | weights %.0f "
                     "GB/s int4, %.0f GB/s as bf16\n",
                     t_hgemm.median_ms / t.median_ms, t_hgemm.median_ms * 1e3,
                     t_cublas.median_ms * 1e3, w.bytes_per_copy() / (t.median_ms * 1e-3) / 1e9,
                     2.0 * K * N / (t.median_ms * 1e-3) / 1e9);
        if (chain > 1 && v > 0) {
            const double c = time_chain(
                [&] {
                    spark::w4gemm_bf16(dA, w.p(turn), w.s(turn), w.z(turn), dC, M, N, K, v, stream);
                    ++turn;
                },
                stream, chain, iters);
            std::fprintf(stderr,
                         "    %d launches per event pair: %.2f us per launch (%.0f GB/s), bf16 "
                         "hgemm %.2f us, %.2fx\n",
                         chain, c * 1e3, bytes / (c * 1e-3) / 1e9, chain_hgemm * 1e3,
                         chain_hgemm / c);
        }
        all_ok = all_ok && ok;
    }
    for (void* p : {static_cast<void*>(dA), static_cast<void*>(dC), static_cast<void*>(dRef)})
        SPARK_CUDA_CHECK(cudaFree(p));
    return all_ok;
}

}  // namespace

int run(int argc, char** argv) {
    spark::bench::Args args(argc, argv);
    spark::bench::print_device_banner();

    struct WShape {
        int N, K;
    };
    std::vector<WShape> weights;
    std::vector<int> ms;
    const bool custom = args.has("m") || args.has("n") || args.has("k");
    if (custom) {
        weights.push_back({args.geti("n", 4096), args.geti("k", 4096)});
        ms.push_back(args.geti("m", 1));
    } else {
        // Llama-3-8B's four projections (scripts/shape_utils.py LLAMA3_8B_PROJECTIONS): the
        // fused q|k|v, o, the fused gate|up, down; decode (1, 4, 16 tokens) to small-batch
        // serving (64, 256).
        weights = {{6144, 4096}, {4096, 4096}, {28672, 4096}, {4096, 14336}};
        ms = {1, 4, 16, 64, 256};
    }
    const int iters = args.geti("iters", 50);
    const int chain = args.geti("chain", 1);
    std::vector<int> variants;
    if (args.has("variant"))
        variants.push_back(args.geti("variant", 2));
    else
        for (int v = 0; v < spark::w4gemm_num_variants(); ++v) variants.push_back(v);
    // --asym: zero points instead of the symmetric code; the default sweep adds asymmetric
    // rows for the top variant at M = 1 and 16.
    const bool asym_only = args.has("asym");

    cudaStream_t stream = nullptr;
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    spark::bench::print_header();
    bool all_ok = true;
    for (const auto& ws : weights) {
        {
            Weight w = make_weight(ws.N, ws.K, asym_only, stream);
            for (int m : ms)
                all_ok = run_m(handle, stream, w, m, variants, iters, chain, !custom) && all_ok;
            all_ok = all_ok && w.quant_ok;
            w.free_all();
        }
        if (!custom && !asym_only) {
            Weight w = make_weight(ws.N, ws.K, true, stream);
            const std::vector<int> top = {spark::w4gemm_num_variants() - 1};
            for (int m : {1, 16})
                all_ok = run_m(handle, stream, w, m, top, iters, chain, true) && all_ok;
            all_ok = all_ok && w.quant_ok;
            w.free_all();
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
        std::fprintf(stderr, "bench_w4gemm: %s\n", e.what());
        return 2;
    }
}
