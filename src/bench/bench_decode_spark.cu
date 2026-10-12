// The decode shapes on the device's own numbers: bf16 (hgemm's decode path), int4 (w4gemm)
// and 1-bit (w1gemm) weight streams at M = 1 and M = 16 for the four projections of
// Llama-3-8B and one DeepSeek V4.1 Flash expert, each checked against cuBLAS on the
// dequantized weight and reported with its bytes, GB/s and the floor at the device's spec
// bandwidth (273 GB/s on a 48-SM device, 1,792 on 170) and at a measured copy roof.
//
//   ./bench_decode_spark                      # the shapes below, every format, rotated past L2
//   ./bench_decode_spark --sweep-split        # every legal split-K per shape (bf16 and int4)
//   ./bench_decode_spark --l2-resident        # the same weight twice back to back, 2nd call apart
//   ./bench_decode_spark --roof=240           # the copy roof in GB/s instead of measuring one
//   ./bench_decode_spark --m=16 --n=4096 --k=4096 --iters=50
//
// Written for the GB10 (DGX Spark) before one was in reach: this bench has not run. The
// questions it is built to answer are in docs/design/decode_spark.md: how many CTAs a weight
// stream needs on 48 SMs at 273 GB/s (Little's law with an unmeasured latency; the split-K
// rule of src/kernels/launch_config.cuh assumes 800 ns and --sweep-split says whether it
// chose right), and whether a weight that fits the 24 MB L2 (int4 and 1-bit at 4096 x 4096,
// not bf16) is still there when the next step comes round (--l2-resident).
//
// Default timing rotates the weight over copies that exceed four times the L2, as a decode
// step reads every layer's weights once. ref_ms is cuBLAS bf16 on the dequantized weight,
// rotated the same way. gbps counts the kernel's own weight bytes (packed plus scales).
// stdout: one JSON object per row; stderr: the table and the plan the split-K rule gives.

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
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

constexpr int kG = 128;  // the group of both quantized formats (kW4GroupSize == kW1GroupSize)
static_assert(spark::kW4GroupSize == kG && spark::kW1GroupSize == kG, "");

template <typename T>
T* dmalloc(size_t n) {
    T* p = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

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
    SPARK_CUDA_CHECK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = __bfloat162float(h[i]);
    return out;
}

enum Format { BF16 = 0, INT4 = 1, W1 = 2 };
const char* format_name(Format f) {
    return f == BF16 ? "bf16" : f == INT4 ? "w4a16" : "w1a16";
}

// One weight in one format, rotated over `copies` copies of its streamed bytes, plus the
// dequantized bf16 matrix the kernel is multiplying (for cuBLAS), itself rotated.
struct Weight {
    Format fmt = BF16;
    int K = 0, N = 0;
    int copies = 1;
    int64_t bytes = 0;  // streamed per call: the weight as the kernel reads it, scales included
    // bf16: the matrix itself; int4 / 1-bit: packed words and scales.
    __nv_bfloat16* w = nullptr;    // copies x K x N (bf16 only)
    int32_t* packed4 = nullptr;    // copies x K N / 8 (int4)
    uint32_t* packed1 = nullptr;   // copies x K N / 32 (1-bit)
    __nv_bfloat16* scales = nullptr;  // copies x (K / 128) x N
    size_t words = 0, groups_n = 0;
    int deq_copies = 1;
    __nv_bfloat16* deq = nullptr;  // deq_copies x K x N, what the kernel multiplies

    const __nv_bfloat16* wb(int c) const {
        return w + static_cast<size_t>(c % copies) * K * N;
    }
    const int32_t* p4(int c) const { return packed4 + static_cast<size_t>(c % copies) * words; }
    const uint32_t* p1(int c) const { return packed1 + static_cast<size_t>(c % copies) * words; }
    const __nv_bfloat16* s(int c) const {
        return scales + static_cast<size_t>(c % copies) * groups_n;
    }
    const __nv_bfloat16* d(int c) const {
        return deq + static_cast<size_t>(c % deq_copies) * K * N;
    }
    void free_all() {
        for (void* x : {static_cast<void*>(w), static_cast<void*>(packed4),
                        static_cast<void*>(packed1), static_cast<void*>(scales),
                        static_cast<void*>(deq)})
            if (x) SPARK_CUDA_CHECK(cudaFree(x));
    }
};

// Copies that take the rotation past the L2: four times the device's L2, at least two.
int copies_for(int64_t bytes_per_copy) {
    const int64_t rotate =
        std::max<int64_t>(4 * spark::device_shape_l2_bytes(), 2 * bytes_per_copy);
    return static_cast<int>(rotate / std::max<int64_t>(1, bytes_per_copy)) + 1;
}

template <typename T>
void replicate(T* base, size_t n, int copies) {
    for (int c = 1; c < copies; ++c)
        SPARK_CUDA_CHECK(cudaMemcpy(base + static_cast<size_t>(c) * n, base, n * sizeof(T),
                                    cudaMemcpyDeviceToDevice));
}

// Quantizes W [K, N] (device, bf16) into `fmt` on the GPU, reads the codes back and builds
// the dequantized bf16 matrix on the host from the layouts kernels.h documents, so cuBLAS
// multiplies bit for bit the weights the kernel does.
Weight make_weight(const __nv_bfloat16* dW, const std::vector<__nv_bfloat16>& hW, int K, int N,
                   Format fmt, cudaStream_t stream) {
    Weight wt;
    wt.fmt = fmt;
    wt.K = K;
    wt.N = N;
    const size_t nW = static_cast<size_t>(K) * N;
    const int G = K / kG;
    wt.groups_n = static_cast<size_t>(G) * N;
    std::vector<__nv_bfloat16> hdeq;

    if (fmt == BF16) {
        wt.bytes = static_cast<int64_t>(nW) * 2;
        wt.copies = copies_for(wt.bytes);
        wt.w = dmalloc<__nv_bfloat16>(nW * wt.copies);
        SPARK_CUDA_CHECK(cudaMemcpy(wt.w, dW, nW * 2, cudaMemcpyDeviceToDevice));
        replicate(wt.w, nW, wt.copies);
        hdeq = hW;
    } else if (fmt == INT4) {
        wt.words = nW / 8;
        wt.bytes = static_cast<int64_t>(wt.words) * 4 + static_cast<int64_t>(wt.groups_n) * 2;
        wt.copies = copies_for(wt.bytes);
        int32_t* dq = dmalloc<int32_t>(wt.words);  // GPTQ qweight [K/8, N], before the repack
        wt.packed4 = dmalloc<int32_t>(wt.words * wt.copies);
        wt.scales = dmalloc<__nv_bfloat16>(wt.groups_n * wt.copies);
        spark::w4_quantize_bf16(dW, dq, wt.scales, nullptr, K, N, stream);
        spark::w4_repack(dq, wt.packed4, K, N, stream);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<int32_t> hq(wt.words);
        std::vector<__nv_bfloat16> hs(wt.groups_n);
        SPARK_CUDA_CHECK(cudaMemcpy(hq.data(), dq, wt.words * 4, cudaMemcpyDeviceToHost));
        SPARK_CUDA_CHECK(cudaMemcpy(hs.data(), wt.scales, wt.groups_n * 2, cudaMemcpyDeviceToHost));
        SPARK_CUDA_CHECK(cudaFree(dq));
        // qweight: eight consecutive k per int32 word of row k / 8, lowest k in the lowest
        // nibble; w = bf16((q - 8) * s[k / 128][n]) for the symmetric weights used here.
        hdeq.resize(nW);
        for (int k = 0; k < K; ++k)
            for (int n = 0; n < N; ++n) {
                const uint32_t word = static_cast<uint32_t>(hq[static_cast<size_t>(k / 8) * N + n]);
                const int q = static_cast<int>((word >> (4 * (k % 8))) & 0xFu);
                const float s = __bfloat162float(hs[static_cast<size_t>(k / kG) * N + n]);
                hdeq[static_cast<size_t>(k) * N + n] =
                    __float2bfloat16(static_cast<float>(q - 8) * s);
            }
        replicate(wt.packed4, wt.words, wt.copies);
        replicate(wt.scales, wt.groups_n, wt.copies);
    } else {
        wt.words = nW / 32;
        wt.bytes = static_cast<int64_t>(wt.words) * 4 + static_cast<int64_t>(wt.groups_n) * 2;
        wt.copies = copies_for(wt.bytes);
        wt.packed1 = dmalloc<uint32_t>(wt.words * wt.copies);
        wt.scales = dmalloc<__nv_bfloat16>(wt.groups_n * wt.copies);
        spark::w1_quantize_bf16(dW, wt.packed1, wt.scales, K, N, 1, stream);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<uint32_t> hq(wt.words);
        std::vector<__nv_bfloat16> hs(wt.groups_n);
        SPARK_CUDA_CHECK(cudaMemcpy(hq.data(), wt.packed1, wt.words * 4, cudaMemcpyDeviceToHost));
        SPARK_CUDA_CHECK(cudaMemcpy(hs.data(), wt.scales, wt.groups_n * 2, cudaMemcpyDeviceToHost));
        // packed [N, K / 32], bit k % 32 of word k / 32 of row n: 1 = +s, 0 = -s.
        const size_t wpr = static_cast<size_t>(K) / 32;
        hdeq.resize(nW);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k) {
                const int bit = (hq[n * wpr + k / 32] >> (k % 32)) & 1u;
                const float s = __bfloat162float(hs[static_cast<size_t>(k / kG) * N + n]);
                hdeq[static_cast<size_t>(k) * N + n] = __float2bfloat16(bit ? s : -s);
            }
        replicate(wt.packed1, wt.words, wt.copies);
        replicate(wt.scales, wt.groups_n, wt.copies);
    }

    wt.deq_copies = copies_for(static_cast<int64_t>(nW) * 2);
    wt.deq = dmalloc<__nv_bfloat16>(nW * wt.deq_copies);
    SPARK_CUDA_CHECK(cudaMemcpy(wt.deq, hdeq.data(), nW * 2, cudaMemcpyHostToDevice));
    replicate(wt.deq, nW, wt.deq_copies);
    std::fprintf(stderr, "  weight %dx%d %s: %.2f MB streamed per call, %d copies, %s the L2\n", K,
                 N, format_name(fmt), wt.bytes / 1e6, wt.copies,
                 spark::decode_fits_l2(wt.bytes) ? "fits 3/4 of" : "does not fit 3/4 of");
    return wt;
}

// The kernel under test on copy `c`.
void run_kernel(const Weight& w, const __nv_bfloat16* A, __nv_bfloat16* C, int M, int c,
                cudaStream_t stream) {
    if (w.fmt == BF16)
        spark::hgemm_bf16(A, w.wb(c), C, M, w.N, w.K, 6, stream);
    else if (w.fmt == INT4)
        spark::w4gemm_bf16(A, w.p4(c), w.s(c), nullptr, C, M, w.N, w.K, 2, stream);
    else
        spark::w1gemm_bf16(A, w.p1(c), w.s(c), C, M, w.N, w.K, 1, 2, stream);
}

int variant_of(Format f) {
    return f == BF16 ? 6 : 2;
}

// Device-wide numbers the rows are judged against.
struct Device {
    int sms = 0;
    int64_t l2 = 0;
    double spec_gbps = 0;   // DRAM spec
    double roof_gbps = 0;   // --roof= or a measured D2D copy, 2 x bytes / time as bench_bandwidth
    int64_t in_flight = 0;  // what launch_config thinks the device needs outstanding
    bool forced = false;
};

double measure_roof(cudaStream_t stream, int iters) {
    const size_t bytes = size_t{256} << 20;
    char* src = dmalloc<char>(bytes);
    char* dst = dmalloc<char>(bytes);
    SPARK_CUDA_CHECK(cudaMemset(src, 1, bytes));
    const auto t = spark::bench::time_kernel(
        [&] {
            SPARK_CUDA_CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, stream));
        },
        stream, 3, std::max(3, std::min(iters, 20)));
    SPARK_CUDA_CHECK(cudaFree(src));
    SPARK_CUDA_CHECK(cudaFree(dst));
    return 2.0 * static_cast<double>(bytes) / (t.median_ms * 1e-3) / 1e9;
}

// The split-K rule's answer for the default tiles of each kernel at this shape, printed so
// the Spark run shows what the launcher decided next to what the sweep finds. hgemm's M <= 16
// tile is 16x64x64 at 24 KB per CTA and 2 CTAs per SM; w4gemm's small-M blocks hold 12 KB
// (the figure that reproduces the 5090's 128) over 16-column strips and 16-token chunks.
void print_plan(const Weight& w, int M) {
    const int sms = spark::device_shape_sm_count();
    if (w.fmt == BF16) {
        const int strips = w.N / 64;
        const int split = spark::decode_split_k(w.bytes, strips, w.K / 64, 24576, 2 * sms);
        std::fprintf(stderr,
                     "    plan hgemm_decode: %d strips of 64, rule wants %d CTAs at 24 KB -> split "
                     "%d, "
                     "%d CTAs = %.2f waves of %d\n",
                     strips, spark::decode_need_ctas(24576), split, strips * split,
                     static_cast<double>(strips * split) / (2 * sms), 2 * sms);
    } else if (w.fmt == INT4) {
        const int tiles = (w.N / 16) * ((M + 15) / 16);
        const int split = spark::decode_split_k(w.bytes, tiles, w.K / kG, 12288, 8 * sms);
        std::fprintf(stderr,
                     "    plan w4gemm: %d tiles of 16 cols, rule wants %d blocks at 12 KB -> split "
                     "%d, %d blocks = %.2f waves of %d (if the 4-warp entry is picked)\n",
                     tiles, spark::decode_need_ctas(12288), split, tiles * split,
                     static_cast<double>(tiles * split) / (8 * sms), 8 * sms);
    } else {
        const int blocks = w.N / 16;
        std::fprintf(stderr,
                     "    plan w1gemm: %d blocks of 4 warps, no cross-block split, 8 KB each in "
                     "flight = %.2f MB against the device's %.2f MB, %.2f waves of %d\n",
                     blocks, blocks * 8192 / 1e6, spark::device_shape_bytes_in_flight() / 1e6,
                     static_cast<double>(blocks) / sms, sms);
    }
}

struct Options {
    int iters = 100;
    bool sweep = false;
    bool l2 = false;
};

// Legal cross-block splits to sweep for a format: powers of two up to the k-steps of the
// kernel's tile (hgemm: K / 64; w4gemm: the 128-k groups), plus 0 for the rule's own choice.
std::vector<int> splits_for(const Weight& w) {
    std::vector<int> out{0};
    if (w.fmt == W1) return out;  // no cross-block split exists
    const int max_split = w.fmt == BF16 ? w.K / 64 : w.K / kG;
    for (int s = 1; s <= max_split; s *= 2) out.push_back(s);
    if (max_split > 1 && (max_split & (max_split - 1)) != 0) out.push_back(max_split);
    return out;
}

// Validates and times one (weight, M). Returns false on a mismatch.
bool run_m(cublasHandle_t handle, cudaStream_t stream, const Weight& w, int M, const Device& dev,
           const Options& opt) {
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

    // cuBLAS on the dequantized weight: the reference and the bf16 path this format replaces.
    cublas_gemm(handle, dA, w.d(0), dRef, M, N, K);
    SPARK_CUDA_CHECK(cudaMemset(dC, 0, nC * 2));
    run_kernel(w, dA, dC, M, 0, stream);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto got = to_host_f32(dC, nC);
    const auto ref = to_host_f32(dRef, nC);
    double max_ref = 0.0;
    for (float r : ref) max_ref = std::max(max_ref, static_cast<double>(std::fabs(r)));
    const auto err = spark::bench::compare(got.data(), ref.data(), nC);
    const double tol = 2e-2 * max_ref + 1e-3;  // the same weights; only the fp32 order differs
    const bool ok = err.max_abs <= tol;
    const std::string shape = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    if (!ok)
        std::fprintf(stderr, "  FAIL %s %s: max_abs=%.4e tol=%.4e\n", format_name(w.fmt),
                     shape.c_str(), err.max_abs, tol);

    int turn = 0;
    const auto t_ref = spark::bench::time_kernel(
        [&] { cublas_gemm(handle, dA, w.d(turn++), dRef, M, N, K); }, stream, 5, opt.iters);
    const auto t = spark::bench::time_kernel([&] { run_kernel(w, dA, dC, M, turn++, stream); },
                                             stream, 5, opt.iters);

    const double wbytes = static_cast<double>(w.bytes);
    const double floor_spec_us = wbytes / (dev.spec_gbps * 1e3);
    const double floor_roof_us = dev.roof_gbps > 0 ? wbytes / (dev.roof_gbps * 1e3) : 0.0;
    spark::bench::Row row;
    row.kernel = "decode_spark";
    row.dtype = format_name(w.fmt);
    row.variant = variant_of(w.fmt);
    row.shape = shape;
    row.median_ms = t.median_ms;
    row.min_ms = t.min_ms;
    row.gbps = wbytes / (t.median_ms * 1e-3) / 1e9;
    row.tflops = 2.0 * M * N * static_cast<double>(K) / (t.median_ms * 1e-3) / 1e12;
    row.ref_ms = t_ref.median_ms;
    row.max_abs_err = err.max_abs;
    row.max_rel_err = err.max_rel;
    row.ok = ok;
    spark::bench::print_row(row);
    std::fprintf(stderr,
                 "    %.2f MB | %.1f us = %.0f GB/s, %.0f%% of spec (%.0f GB/s, floor %.1f us), "
                 "%.0f%% of the copy roof (%.0f GB/s, floor %.1f us) | cuBLAS bf16 %.1f us\n",
                 wbytes / 1e6, t.median_ms * 1e3, row.gbps, 100.0 * row.gbps / dev.spec_gbps,
                 dev.spec_gbps, floor_spec_us,
                 dev.roof_gbps > 0 ? 100.0 * row.gbps / dev.roof_gbps : 0.0, dev.roof_gbps,
                 floor_roof_us, t_ref.median_ms * 1e3);
    print_plan(w, M);

    if (opt.sweep) {
        for (int s : splits_for(w)) {
            spark::decode_force_split_k(s);
            try {
                const auto ts = spark::bench::time_kernel(
                    [&] { run_kernel(w, dA, dC, M, turn++, stream); }, stream, 3, opt.iters);
                SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
                const auto got_s = to_host_f32(dC, nC);
                const auto err_s = spark::bench::compare(got_s.data(), ref.data(), nC);
                spark::bench::Row r = row;
                r.kernel = "decode_spark_split";
                r.shape = shape + (s == 0 ? "/auto" : "/s" + std::to_string(s));
                r.median_ms = ts.median_ms;
                r.min_ms = ts.min_ms;
                r.gbps = wbytes / (ts.median_ms * 1e-3) / 1e9;
                r.tflops = 2.0 * M * N * static_cast<double>(K) / (ts.median_ms * 1e-3) / 1e12;
                r.max_abs_err = err_s.max_abs;
                r.max_rel_err = err_s.max_rel;
                r.ok = err_s.max_abs <= tol;
                spark::bench::print_row(r);
            } catch (const std::exception& e) {
                std::fprintf(stderr, "    split %d: %s\n", s, e.what());
            }
            spark::decode_force_split_k(0);
        }
    }

    if (opt.l2) {
        // A cold call on a fresh copy, then the same copy again with the events around the
        // second call only: the time of a weight the previous step may have left in L2.
        cudaEvent_t e0, e1;
        SPARK_CUDA_CHECK(cudaEventCreate(&e0));
        SPARK_CUDA_CHECK(cudaEventCreate(&e1));
        std::vector<float> second;
        second.reserve(opt.iters);
        for (int i = 0; i < opt.iters + 3; ++i) {
            run_kernel(w, dA, dC, M, turn, stream);
            SPARK_CUDA_CHECK(cudaEventRecord(e0, stream));
            run_kernel(w, dA, dC, M, turn, stream);
            SPARK_CUDA_CHECK(cudaEventRecord(e1, stream));
            SPARK_CUDA_CHECK(cudaEventSynchronize(e1));
            ++turn;
            if (i < 3) continue;  // warmup pairs
            float ms = 0.f;
            SPARK_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
            second.push_back(ms);
        }
        SPARK_CUDA_CHECK(cudaEventDestroy(e0));
        SPARK_CUDA_CHECK(cudaEventDestroy(e1));
        std::sort(second.begin(), second.end());
        spark::bench::Row r = row;
        r.kernel = "decode_spark_l2";
        r.median_ms = second[second.size() / 2];
        r.min_ms = second.front();
        r.gbps = wbytes / (r.median_ms * 1e-3) / 1e9;
        r.tflops = 2.0 * M * N * static_cast<double>(K) / (r.median_ms * 1e-3) / 1e12;
        r.ref_ms = t.median_ms;  // the cold, rotated call of the same kernel
        spark::bench::print_row(r);
        std::fprintf(stderr,
                     "    second call on the same weight: %.1f us against %.1f us cold (%.2fx); "
                     "launch_config says the %.2f MB %s the L2 (%.0f MB)\n",
                     r.median_ms * 1e3, t.median_ms * 1e3,
                     r.median_ms > 0 ? t.median_ms / r.median_ms : 0.0, wbytes / 1e6,
                     spark::decode_fits_l2(w.bytes) ? "fits 3/4 of" : "does not fit 3/4 of",
                     dev.l2 / 1e6);
    }

    SPARK_CUDA_CHECK(cudaFree(dA));
    SPARK_CUDA_CHECK(cudaFree(dC));
    SPARK_CUDA_CHECK(cudaFree(dRef));
    return ok;
}

}  // namespace

int run(int argc, char** argv) {
    spark::bench::Args args(argc, argv);
    spark::bench::print_device_banner();
    Options opt;
    opt.iters = args.geti("iters", 100);
    opt.sweep = args.has("sweep-split");
    opt.l2 = args.has("l2-resident");
    cudaStream_t stream = nullptr;

    Device dev;
    dev.sms = spark::device_shape_sm_count();
    dev.l2 = spark::device_shape_l2_bytes();
    dev.spec_gbps = spark::device_shape_spec_gbps();
    dev.in_flight = spark::device_shape_bytes_in_flight();
    dev.forced = spark::device_shape_forced();
    dev.roof_gbps = args.has("roof") ? static_cast<double>(args.geti("roof", 0))
                                     : measure_roof(stream, opt.iters);
    std::fprintf(stderr,
                 "launch_config: %d SMs%s, L2 %.0f MB, spec %.0f GB/s, bytes in flight %.2f MB "
                 "(%.1f KB per SM), copy roof %.0f GB/s (%s)\n",
                 dev.sms, dev.forced ? " (SPARK_FORCE_SMS)" : "", dev.l2 / 1e6, dev.spec_gbps,
                 dev.in_flight / 1e6, dev.in_flight / 1024.0 / dev.sms, dev.roof_gbps,
                 args.has("roof") ? "--roof" : "cudaMemcpy D2D, 256 MB, 2 x bytes / time");
    if (dev.forced)
        std::fprintf(stderr,
                     "note: the SM count is forced; grids are the forced device's, times are this "
                     "one's, so only the ok column means anything\n");

    struct WShape {
        const char* name;
        int K, N;
        std::vector<int> ms;
    };
    std::vector<WShape> shapes;
    if (args.has("m") || args.has("n") || args.has("k")) {
        shapes.push_back(
            {"custom", args.geti("k", 4096), args.geti("n", 4096), {args.geti("m", 1)}});
    } else {
        // Llama-3-8B: the fused q|k|v projection (4096 -> 4096 + 2 x 1024), o, the fused
        // gate|up and down; then one DeepSeek V4.1 Flash routed expert, up (5120 -> 2304) and
        // down (2304 -> 5120). Batch 1 and batch 16.
        shapes = {{"llama_qkv", 4096, 6144, {1, 16}},   {"llama_o", 4096, 4096, {1, 16}},
                  {"llama_gate_up", 4096, 28672, {1, 16}}, {"llama_down", 14336, 4096, {1, 16}},
                  {"dsflash_expert_up", 5120, 2304, {1, 16}},
                  {"dsflash_expert_down", 2304, 5120, {1, 16}}};
    }
    std::vector<Format> formats;
    const std::string fmt = args.get("format", "all");
    if (fmt == "all" || fmt == "bf16") formats.push_back(BF16);
    if (fmt == "all" || fmt == "int4") formats.push_back(INT4);
    if (fmt == "all" || fmt == "w1") formats.push_back(W1);
    SPARK_REQUIRE(!formats.empty(), "--format must be all, bf16, int4 or w1");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));
    spark::bench::print_header();
    bool all_ok = true;
    for (const auto& sh : shapes) {
        SPARK_REQUIRE(sh.K % kG == 0 && sh.N % 64 == 0, "shapes need K % 128 == 0 and N % 64 == 0");
        std::fprintf(stderr, "%s: K = %d, N = %d\n", sh.name, sh.K, sh.N);
        const size_t nW = static_cast<size_t>(sh.K) * sh.N;
        std::vector<float> hWf(nW);
        spark::bench::fill_uniform(hWf, -1.0f, 1.0f, 5678);
        std::vector<__nv_bfloat16> hW(nW);
        for (size_t i = 0; i < nW; ++i) hW[i] = __float2bfloat16(hWf[i]);
        hWf.clear();
        hWf.shrink_to_fit();
        __nv_bfloat16* dW = dmalloc<__nv_bfloat16>(nW);
        SPARK_CUDA_CHECK(cudaMemcpy(dW, hW.data(), nW * 2, cudaMemcpyHostToDevice));
        for (Format f : formats) {
            Weight w = make_weight(dW, hW, sh.K, sh.N, f, stream);
            for (int m : sh.ms) all_ok = run_m(handle, stream, w, m, dev, opt) && all_ok;
            w.free_all();
        }
        SPARK_CUDA_CHECK(cudaFree(dW));
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
        std::fprintf(stderr, "bench_decode_spark: %s\n", e.what());
        return 2;
    }
}
