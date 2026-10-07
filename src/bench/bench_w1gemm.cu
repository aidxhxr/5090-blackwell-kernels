// Benchmark + correctness check for the W1A16 GEMM ladder (1-bit sign or 2-bit ternary
// weights, bf16 activations).
//
//   ./bench_w1gemm                               # DeepSeek V4.1 Flash experts + a Llama row
//   ./bench_w1gemm --m=1 --n=2304 --k=5120 --bits=1 --variant=2 --iters=50
//   ./bench_w1gemm --bits=2                      # ternary rows only
//
// For every weight shape the bench quantizes a random bf16 W [K, N] on the GPU
// (w1_quantize_bf16) at 1 and at 2 bits, checks the codes and scales bit for bit against the
// same fp32 recipe on the CPU, and dequantizes it on the CPU into the bf16 matrix the kernels
// must be multiplying. The reference is a CPU double-precision product of A with that matrix;
// every variant must match it to 2% of max|C| (the same weights, only the fp32 summation
// order and the final rounding differ).
//
// Timing: the packed weights (with their scales) are rotated over copies that exceed the
// 96 MB L2, as a decode step reads every layer's weights once. ref_ms is the bf16 path this
// op replaces, hgemm_bf16 variant 6 on the dequantized weights, rotated the same way; the
// int4 path (w4gemm variant 2 on the same W, symmetric) is printed on stderr next to it.
// gbps = the weight bytes streamed (packed + scales) per second.
//
// stdout: one JSON object per row (collected by scripts/run_all_benches.sh)
// stderr: human-readable table incl. the speedups over the bf16 and the int4 paths

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

namespace {

constexpr int kG = spark::kW1GroupSize;
constexpr size_t kRotate = size_t{256} << 20;  // past the 96 MB L2

template <typename T>
T* dmalloc(size_t n) {
    T* p = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

// The quantizer of kernels.h on the CPU, in the same fp32 operations, so the GPU's codes and
// scales can be compared bit for bit. codes is [K][N] (0/1 or 0/1/2).
void quantize_host(const std::vector<__nv_bfloat16>& W, int K, int N, int bits,
                   std::vector<uint8_t>& codes, std::vector<__nv_bfloat16>& s) {
    const int G = K / kG;
    codes.assign(static_cast<size_t>(K) * N, 0);
    s.assign(static_cast<size_t>(G) * N, __float2bfloat16(0.f));
    std::vector<float> sum(N), sc(N);
    for (int grp = 0; grp < G; ++grp) {
        std::fill(sum.begin(), sum.end(), 0.f);
        for (int k = grp * kG; k < (grp + 1) * kG; ++k)  // k ascending, as the kernel sums
            for (int n = 0; n < N; ++n)
                sum[n] += std::fabs(__bfloat162float(W[static_cast<size_t>(k) * N + n]));
        for (int n = 0; n < N; ++n) {
            const __nv_bfloat16 sb = __float2bfloat16(sum[n] / static_cast<float>(kG));
            sc[n] = __bfloat162float(sb);
            s[static_cast<size_t>(grp) * N + n] = sb;
        }
        for (int k = grp * kG; k < (grp + 1) * kG; ++k)
            for (int n = 0; n < N; ++n) {
                const float w = __bfloat162float(W[static_cast<size_t>(k) * N + n]);
                uint8_t q;
                if (bits == 1) {
                    q = w >= 0.f ? 1 : 0;
                } else {
                    const float h = 0.5f * sc[n];
                    q = w < -h ? 0 : (w > h ? 2 : 1);
                }
                codes[static_cast<size_t>(k) * N + n] = q;
            }
    }
}

// Code of (n, k) in the packed layout of kernels.h.
int code_of(const std::vector<uint32_t>& packed, int n, int k, int K, int bits) {
    const size_t wpr = static_cast<size_t>(K) * bits / 32;
    if (bits == 1) return (packed[n * wpr + k / 32] >> (k % 32)) & 1;
    return (packed[n * wpr + k / 16] >> (2 * (k % 16))) & 3;
}

// One quantized weight on the GPU, rotated over `copies` copies of (packed, scales), plus the
// dequantized bf16 matrix for the reference and the bf16 timing.
struct Weight {
    int N = 0, K = 0, bits = 1;
    int copies = 1;
    size_t words = 0, groups_n = 0;
    uint32_t* packed = nullptr;       // copies x words
    __nv_bfloat16* scales = nullptr;  // copies x groups_n
    int bf16_copies = 1;
    __nv_bfloat16* deq = nullptr;  // bf16_copies x K x N
    std::vector<float> hd;         // the dequantized weights, [K][N]
    bool quant_ok = true;
    double quant_ms = 0;

    size_t bytes_per_copy() const { return words * 4 + groups_n * 2; }
    const uint32_t* p(int c) const { return packed + static_cast<size_t>(c % copies) * words; }
    const __nv_bfloat16* s(int c) const {
        return scales + static_cast<size_t>(c % copies) * groups_n;
    }
    const __nv_bfloat16* d(int c) const {
        return deq + static_cast<size_t>(c % bf16_copies) * K * N;
    }
    void free_all() {
        for (void* x : {static_cast<void*>(packed), static_cast<void*>(scales),
                        static_cast<void*>(deq)})
            if (x) SPARK_CUDA_CHECK(cudaFree(x));
    }
};

// The same W quantized to int4 (symmetric) for w4gemm, rotated the same way.
struct W4Weight {
    int copies = 1;
    size_t words = 0, groups_n = 0;
    int32_t* packed = nullptr;
    __nv_bfloat16* scales = nullptr;
    const int32_t* p(int c) const { return packed + static_cast<size_t>(c % copies) * words; }
    const __nv_bfloat16* s(int c) const {
        return scales + static_cast<size_t>(c % copies) * groups_n;
    }
    void free_all() {
        if (packed) SPARK_CUDA_CHECK(cudaFree(packed));
        if (scales) SPARK_CUDA_CHECK(cudaFree(scales));
    }
};

std::vector<__nv_bfloat16> random_weight(int N, int K) {
    const size_t nW = static_cast<size_t>(K) * N;
    std::vector<float> hWf(nW);
    spark::bench::fill_uniform(hWf, -1.0f, 1.0f, 5678);
    std::vector<__nv_bfloat16> hW(nW);
    for (size_t i = 0; i < nW; ++i) hW[i] = __float2bfloat16(hWf[i]);
    return hW;
}

Weight make_weight(const std::vector<__nv_bfloat16>& hW, int N, int K, int bits,
                   cudaStream_t stream) {
    Weight w;
    w.N = N;
    w.K = K;
    w.bits = bits;
    const size_t nW = static_cast<size_t>(K) * N;
    w.words = nW * bits / 32;
    w.groups_n = static_cast<size_t>(K / kG) * N;
    w.copies = static_cast<int>(kRotate / w.bytes_per_copy()) + 1;
    w.bf16_copies = static_cast<int>(kRotate / (nW * 2)) + 1;

    __nv_bfloat16* dW = dmalloc<__nv_bfloat16>(nW);
    SPARK_CUDA_CHECK(cudaMemcpy(dW, hW.data(), nW * 2, cudaMemcpyHostToDevice));
    w.packed = dmalloc<uint32_t>(w.words * w.copies);
    w.scales = dmalloc<__nv_bfloat16>(w.groups_n * w.copies);

    cudaEvent_t e0, e1;
    SPARK_CUDA_CHECK(cudaEventCreate(&e0));
    SPARK_CUDA_CHECK(cudaEventCreate(&e1));
    SPARK_CUDA_CHECK(cudaEventRecord(e0, stream));
    spark::w1_quantize_bf16(dW, w.packed, w.scales, K, N, bits, stream);
    SPARK_CUDA_CHECK(cudaEventRecord(e1, stream));
    SPARK_CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0.f;
    SPARK_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    w.quant_ms = ms;
    SPARK_CUDA_CHECK(cudaEventDestroy(e0));
    SPARK_CUDA_CHECK(cudaEventDestroy(e1));

    // The GPU quantizer against the CPU one, bit for bit.
    std::vector<uint32_t> hq(w.words);
    std::vector<__nv_bfloat16> hs(w.groups_n);
    SPARK_CUDA_CHECK(cudaMemcpy(hq.data(), w.packed, w.words * 4, cudaMemcpyDeviceToHost));
    SPARK_CUDA_CHECK(cudaMemcpy(hs.data(), w.scales, w.groups_n * 2, cudaMemcpyDeviceToHost));
    std::vector<uint8_t> rq;
    std::vector<__nv_bfloat16> rs;
    quantize_host(hW, K, N, bits, rq, rs);
    size_t bad = 0;
    for (size_t i = 0; i < w.groups_n; ++i) {
        const auto a = reinterpret_cast<const uint16_t&>(hs[i]);
        const auto b = reinterpret_cast<const uint16_t&>(rs[i]);
        bad += a != b;
    }
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n)
            bad += code_of(hq, n, k, K, bits) != rq[static_cast<size_t>(k) * N + n];
    if (bad) {
        std::fprintf(stderr, "  FAIL w1_quantize %dx%d bits=%d: %zu codes/scales differ from the CPU\n",
                     K, N, bits, bad);
        w.quant_ok = false;
    }

    // The bf16 weights the kernels multiply, for the reference and for the bf16 timing.
    w.hd.resize(nW);
    std::vector<__nv_bfloat16> hdb(nW);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) {
            const size_t gi = static_cast<size_t>(k / kG) * N + n;
            const float s = __bfloat162float(rs[gi]);
            const int q = rq[static_cast<size_t>(k) * N + n];
            const float v = bits == 1 ? (q ? s : -s) : s * static_cast<float>(q - 1);
            const __nv_bfloat16 vb = __float2bfloat16(v);
            hdb[static_cast<size_t>(k) * N + n] = vb;
            w.hd[static_cast<size_t>(k) * N + n] = __bfloat162float(vb);
        }
    w.deq = dmalloc<__nv_bfloat16>(nW * w.bf16_copies);
    for (int c = 0; c < w.bf16_copies; ++c)
        SPARK_CUDA_CHECK(cudaMemcpy(w.deq + c * nW, hdb.data(), nW * 2, cudaMemcpyHostToDevice));
    for (int c = 1; c < w.copies; ++c) {
        SPARK_CUDA_CHECK(cudaMemcpy(w.packed + c * w.words, w.packed, w.words * 4,
                                    cudaMemcpyDeviceToDevice));
        SPARK_CUDA_CHECK(cudaMemcpy(w.scales + c * w.groups_n, w.scales, w.groups_n * 2,
                                    cudaMemcpyDeviceToDevice));
    }
    SPARK_CUDA_CHECK(cudaFree(dW));
    std::fprintf(stderr,
                 "  weight %dx%d bits=%d: quantize %.3f ms (cold, once per weight), %.2f MB "
                 "packed + scales, rotated over %d copies\n",
                 K, N, bits, w.quant_ms, w.bytes_per_copy() / 1e6, w.copies);
    return w;
}

W4Weight make_w4(const std::vector<__nv_bfloat16>& hW, int N, int K, cudaStream_t stream) {
    W4Weight w;
    const size_t nW = static_cast<size_t>(K) * N;
    w.words = nW / 8;
    w.groups_n = static_cast<size_t>(K / spark::kW4GroupSize) * N;
    w.copies = static_cast<int>(kRotate / (w.words * 4 + w.groups_n * 2)) + 1;
    __nv_bfloat16* dW = dmalloc<__nv_bfloat16>(nW);
    SPARK_CUDA_CHECK(cudaMemcpy(dW, hW.data(), nW * 2, cudaMemcpyHostToDevice));
    int32_t* dq = dmalloc<int32_t>(w.words);
    w.packed = dmalloc<int32_t>(w.words * w.copies);
    w.scales = dmalloc<__nv_bfloat16>(w.groups_n * w.copies);
    spark::w4_quantize_bf16(dW, dq, w.scales, nullptr, K, N, stream);
    spark::w4_repack(dq, w.packed, K, N, stream);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int c = 1; c < w.copies; ++c) {
        SPARK_CUDA_CHECK(cudaMemcpy(w.packed + c * w.words, w.packed, w.words * 4,
                                    cudaMemcpyDeviceToDevice));
        SPARK_CUDA_CHECK(cudaMemcpy(w.scales + c * w.groups_n, w.scales, w.groups_n * 2,
                                    cudaMemcpyDeviceToDevice));
    }
    SPARK_CUDA_CHECK(cudaFree(dW));
    SPARK_CUDA_CHECK(cudaFree(dq));
    return w;
}

std::vector<float> to_host_f32(const __nv_bfloat16* d, size_t n) {
    std::vector<__nv_bfloat16> h(n);
    SPARK_CUDA_CHECK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = __bfloat162float(h[i]);
    return out;
}

// Runs every requested variant on one M against one weight. Returns false on a mismatch.
bool run_m(cudaStream_t stream, const Weight& w, const W4Weight* w4, int M,
           const std::vector<int>& variants, int iters, bool skip_slow_v0) {
    const int N = w.N, K = w.K;
    const size_t nA = static_cast<size_t>(M) * K, nC = static_cast<size_t>(M) * N;
    std::vector<float> hA(nA);
    spark::bench::fill_uniform(hA, -1.0f, 1.0f, 1234 + M);
    std::vector<__nv_bfloat16> hAb(nA);
    for (size_t i = 0; i < nA; ++i) {
        hAb[i] = __float2bfloat16(hA[i]);
        hA[i] = __bfloat162float(hAb[i]);
    }
    __nv_bfloat16* dA = dmalloc<__nv_bfloat16>(nA);
    __nv_bfloat16* dC = dmalloc<__nv_bfloat16>(nC);
    SPARK_CUDA_CHECK(cudaMemcpy(dA, hAb.data(), nA * 2, cudaMemcpyHostToDevice));

    // CPU double reference on the dequantized weights.
    std::vector<double> refd(nC, 0.0);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            const double a = hA[static_cast<size_t>(m) * K + k];
            const float* wrow = w.hd.data() + static_cast<size_t>(k) * N;
            double* crow = refd.data() + static_cast<size_t>(m) * N;
            for (int n = 0; n < N; ++n) crow[n] += a * wrow[n];
        }
    std::vector<float> ref(nC);
    double max_ref = 0.0;
    for (size_t i = 0; i < nC; ++i) {
        ref[i] = static_cast<float>(refd[i]);
        max_ref = std::max(max_ref, std::fabs(refd[i]));
    }
    const double tol = 2e-2 * max_ref + 1e-3;

    // The bf16 path this op replaces, and the int4 one, on the same shape.
    int turn = 0;
    const bool hgemm_ok = spark::hgemm_supports(M, N, K, 6);
    double t_bf16 = 0.0;
    if (hgemm_ok)
        t_bf16 = spark::bench::time_kernel(
                     [&] { spark::hgemm_bf16(dA, w.d(turn++), dC, M, N, K, 6, stream); },
                     stream, 5, iters)
                     .median_ms;
    double t_w4 = 0.0;
    if (w4 && spark::w4gemm_supports(M, N, K, 2))
        t_w4 = spark::bench::time_kernel(
                   [&] {
                       spark::w4gemm_bf16(dA, w4->p(turn), w4->s(turn), nullptr, dC, M, N, K, 2,
                                          stream);
                       ++turn;
                   },
                   stream, 5, iters)
                   .median_ms;

    const std::string shape = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    const double wbytes = static_cast<double>(w.bytes_per_copy());
    const double flops = 2.0 * M * N * static_cast<double>(K);
    bool all_ok = true;
    for (int v : variants) {
        if (v == 0 && skip_slow_v0 && M > 16) {
            std::fprintf(stderr, "  skip variant 0 at %s: the naive rung is only timed to M = 16\n",
                         shape.c_str());
            continue;
        }
        SPARK_CUDA_CHECK(cudaMemset(dC, 0, nC * 2));
        spark::w1gemm_bf16(dA, w.p(0), w.s(0), dC, M, N, K, w.bits, v, stream);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto got = to_host_f32(dC, nC);
        const auto err = spark::bench::compare(got.data(), ref.data(), nC);
        const bool ok = err.max_abs <= tol && w.quant_ok;
        if (!ok)
            std::fprintf(stderr, "  FAIL variant %d bits=%d shape %s: max_abs=%.4e tol=%.4e\n", v,
                         w.bits, shape.c_str(), err.max_abs, tol);
        const auto t = spark::bench::time_kernel(
            [&] {
                spark::w1gemm_bf16(dA, w.p(turn), w.s(turn), dC, M, N, K, w.bits, v, stream);
                ++turn;
            },
            stream, 5, v == 0 ? std::min(iters, 10) : iters);

        spark::bench::Row row;
        row.kernel = "w1gemm";
        row.dtype = w.bits == 1 ? "w1a16" : "w2a16";
        row.variant = v;
        row.shape = shape;
        row.median_ms = t.median_ms;
        row.min_ms = t.min_ms;
        row.gbps = wbytes / (t.median_ms * 1e-3) / 1e9;
        row.tflops = flops / (t.median_ms * 1e-3) / 1e12;
        row.ref_ms = t_bf16;
        row.max_abs_err = err.max_abs;
        row.max_rel_err = err.max_rel;
        row.ok = ok;
        spark::bench::print_row(row);
        std::fprintf(stderr,
                     "    %.2fx the bf16 hgemm (%.1f us) | %.2fx the int4 w4gemm (%.1f us) | "
                     "weights %.0f GB/s packed, %.0f GB/s as bf16\n",
                     t_bf16 > 0 ? t_bf16 / t.median_ms : 0.0, t_bf16 * 1e3,
                     t_w4 > 0 ? t_w4 / t.median_ms : 0.0, t_w4 * 1e3, row.gbps,
                     2.0 * K * N / (t.median_ms * 1e-3) / 1e9);
        all_ok = all_ok && ok;
    }
    SPARK_CUDA_CHECK(cudaFree(dA));
    SPARK_CUDA_CHECK(cudaFree(dC));
    return all_ok;
}

}  // namespace

int run(int argc, char** argv) {
    spark::bench::Args args(argc, argv);
    spark::bench::print_device_banner();

    struct WShape {
        int N, K;
        std::vector<int> ms;
    };
    std::vector<WShape> weights;
    const bool custom = args.has("m") || args.has("n") || args.has("k");
    if (custom) {
        weights.push_back({args.geti("n", 2304), args.geti("k", 5120), {args.geti("m", 1)}});
    } else {
        // DeepSeek V4.1 Flash routed experts (hidden 5120, intermediate 2304): gate / up are
        // K = 5120, N = 2304 and down is K = 2304, N = 5120; decode (1, 8 tokens, the 6 active
        // experts of one token or a small batch) to a 64-token chunk of a prefill. Then one
        // dense Llama-3-8B row, the fused gate|up at M = 1, next to bench_w4gemm's.
        weights = {{2304, 5120, {1, 8, 16, 64}}, {5120, 2304, {1, 8, 16, 64}}, {28672, 4096, {1}}};
    }
    const int iters = args.geti("iters", 50);
    std::vector<int> variants;
    if (args.has("variant"))
        variants.push_back(args.geti("variant", 2));
    else
        for (int v = 0; v < spark::w1gemm_num_variants(); ++v) variants.push_back(v);
    std::vector<int> bits_list;
    if (args.has("bits"))
        bits_list.push_back(args.geti("bits", 1));
    else
        bits_list = {1, 2};

    cudaStream_t stream = nullptr;
    spark::bench::print_header();
    bool all_ok = true;
    for (const auto& ws : weights) {
        const auto hW = random_weight(ws.N, ws.K);
        W4Weight w4 = make_w4(hW, ws.N, ws.K, stream);
        for (int bits : bits_list) {
            Weight w = make_weight(hW, ws.N, ws.K, bits, stream);
            for (int m : ws.ms)
                all_ok = run_m(stream, w, &w4, m, variants, iters, !custom) && all_ok;
            all_ok = all_ok && w.quant_ok;
            w.free_all();
        }
        w4.free_all();
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
        std::fprintf(stderr, "bench_w1gemm: %s\n", e.what());
        return 2;
    }
}
