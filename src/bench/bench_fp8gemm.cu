// Benchmark + correctness check for the fp8 (e4m3) tensor-core GEMM ladder against cuBLASLt.
//
//   ./bench_fp8gemm                       # default shape sweep (bench_hgemm's), all variants,
//                                         # per-tensor scales (dtype e4m3) then MX (dtype mxfp8)
//   ./bench_fp8gemm --m=4096 --n=4096 --k=4096 --variant=1 --iters=50 --mode=mx
//   ./bench_fp8gemm --models[=llama3-8b] [--ms=128,512] [--tiles] [--mode=mx]   # model shapes
//
// C = scale_a * scale_b * A[M,K] * Bt[N,K]^T, e4m3 in, bf16 out. cuBLASLt's fp8 path only takes
// this layout (A row-major, B column-major, both K-contiguous), so the reference is timed on
// exactly the same bytes. Correctness is checked two ways: the whole output against cuBLASLt,
// and a sample of outputs against a CPU fp32 dot product of the same e4m3-rounded inputs.
//
// --mode=mx (rows with dtype "mxfp8") runs the MX mode: Gaussian inputs quantized on the CPU
// with the OCP MX recipe (a ue8m0 scale per 32 k, docs/design/fp8gemm.md), sfa[M][K/32] and
// sfb[N][K/32] handed to fp8gemm_mx, against cuBLASLt's MXFP8 path (scale mode VEC32_UE8M0,
// which wants the scales in its 32x4x4 tiled layout) if the installed library has it for
// this GPU, and a CPU fp32 dot product of the dequantized values either way.
//
// stdout: one JSON object per row (collected by scripts/run_all_benches.sh)
// stderr: human-readable table incl. "% of cuBLASLt"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "model_shapes.hpp"
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
// are swapped to match: its A scale is our sfb / scale_b, its B scale our sfa / scale_a. In
// MX mode the scale pointers are the tiled ue8m0 tensors (to_blocked below) and the scale
// mode is VEC32_UE8M0 on both; `available` is false if the heuristic has no such kernel.
struct LtGemm {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t heur{};
    void* ws = nullptr;
    size_t ws_bytes = size_t{32} << 20;
    bool available = false;

    LtGemm(int M, int N, int K, const void* d_scale_a, const void* d_scale_b, bool mx) {
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
        if (mx) {
            const cublasLtMatmulMatrixScale_t mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                                                        &mode, sizeof(mode)));
            CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                                                        &mode, sizeof(mode)));
        }
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_8F_E4M3, K, N, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8F_E4M3, K, M, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N));
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
        int returned = 0;
        const cublasStatus_t st =
            cublasLtMatmulAlgoGetHeuristic(handle, op, la, lb, lc, lc, pref, 1, &heur, &returned);
        available = st == CUBLAS_STATUS_SUCCESS && returned > 0;
        if (!available && !mx) {
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
    // Points cuBLASLt's A scale (our Bt's) at another copy, for the rotated decode timing.
    void set_a_scale(const void* p) {
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &p,
                                                    sizeof(p)));
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

// Gaussian rows quantized with the OCP MX recipe, 32 elements per block along K: the block's
// shared exponent is floor(log2(amax)) - 8 (8 = the largest e4m3 exponent, so the largest
// element lands in the top binade), clamped to the E8M0 range and stored as exponent + 127;
// the elements are x / 2^e rounded to e4m3 (nearest even, saturating at 448). `f` gets the
// dequantized fp32 value of each element, exact since the scale is a power of two.
void fill_mx(std::vector<__nv_fp8_e4m3>& q, std::vector<unsigned char>& sf, std::vector<float>& f,
             int rows, int K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<float> raw(q.size());
    for (auto& x : raw) x = dist(rng);
    const int KB = K / 32;
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < KB; ++b) {
            const float* x = raw.data() + static_cast<size_t>(r) * K + b * 32;
            float amax = 0.f;
            for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(x[i]));
            int e = amax > 0.f ? static_cast<int>(std::floor(std::log2(amax))) - 8 : -127;
            e = std::max(-127, std::min(127, e));
            sf[static_cast<size_t>(r) * KB + b] = static_cast<unsigned char>(e + 127);
            for (int i = 0; i < 32; ++i) {
                const size_t idx = static_cast<size_t>(r) * K + b * 32 + i;
                q[idx] = __nv_fp8_e4m3(std::ldexp(x[i], -e));
                f[idx] = std::ldexp(static_cast<float>(q[idx]), e);
            }
        }
    }
}

// cuBLASLt's (and torch's SWIZZLE_32_4_4) layout of a [rows][K/32] ue8m0 scale tensor: tiles
// of 128 rows by 4 k-blocks, 512 bytes each, row-major over tiles, and inside a tile byte
// (r % 32) * 16 + (r / 32) * 4 + kb % 4 for r = row % 128. Rows and k-blocks are padded to
// whole tiles.
std::vector<unsigned char> to_blocked(const std::vector<unsigned char>& sf, int rows, int KB) {
    const int rt = (rows + 127) / 128, ct = (KB + 3) / 4;
    std::vector<unsigned char> out(static_cast<size_t>(rt) * ct * 512, 0);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < KB; ++c) {
            const size_t tile = static_cast<size_t>(r / 128) * ct + c / 4;
            const int rr = r % 128;
            out[tile * 512 + (rr % 32) * 16 + (rr / 32) * 4 + (c % 4)] =
                sf[static_cast<size_t>(r) * KB + c];
        }
    }
    return out;
}

// Runs one shape for one variant, per-tensor or MX. Returns false on a correctness failure.
bool run_one(cudaStream_t stream, const Shape& s, int variant, int iters, bool mx) {
    const int M = s.M, N = s.N, K = s.K;
    const bool supported = mx ? spark::fp8gemm_mx_supports(M, N, K, variant)
                              : spark::fp8gemm_supports(M, N, K, variant);
    if (!supported) {
        std::fprintf(stderr, "  skip variant %d shape %dx%dx%d%s: shape not supported\n", variant,
                     M, N, K, mx ? " (mx)" : "");
        return true;  // an unsupported shape is not a failure
    }
    const size_t nA = static_cast<size_t>(M) * K, nB = static_cast<size_t>(N) * K,
                 nC = static_cast<size_t>(M) * N;
    const int KB = K / 32;

    std::vector<__nv_fp8_e4m3> hA(nA), hB(nB);
    std::vector<float> fA(nA), fB(nB);
    std::vector<unsigned char> hSfa, hSfb;
    if (mx) {
        hSfa.resize(static_cast<size_t>(M) * KB);
        hSfb.resize(static_cast<size_t>(N) * KB);
        fill_mx(hA, hSfa, fA, M, K, 1234);
        fill_mx(hB, hSfb, fB, N, K, 5678);
    } else {
        fill_e4m3(hA, fA, 1234);
        fill_e4m3(hB, fB, 5678);
    }

    // Decode shapes (M <= 64) are bound by streaming Bt, and Bt alone (16 MB for 4096x4096)
    // fits in the RTX 5090's 96 MB L2, so the timing loop rotates through enough copies to
    // exceed L2 (256 MB+), for cuBLASLt and for us alike. fp8 weights are half the bytes of
    // bf16, so twice the copies.
    const int copies = M <= 64 ? static_cast<int>((size_t{256} << 20) / nB) + 1 : 1;

    __nv_fp8_e4m3 *dA = nullptr, *dB = nullptr;
    __nv_bfloat16 *dC = nullptr, *dRef = nullptr;
    float* dScale = nullptr;
    unsigned char *dSfa = nullptr, *dSfb = nullptr, *dSfaBlk = nullptr, *dSfbBlk = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&dA, nA));
    SPARK_CUDA_CHECK(cudaMalloc(&dB, copies * nB));
    SPARK_CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dRef, nC * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dScale, 2 * sizeof(float)));
    // MX mode has no per-tensor scale (the kernel takes 1.0 for a null pair).
    const float scales[2] = {mx ? 1.0f : kScaleA, mx ? 1.0f : kScaleB};
    SPARK_CUDA_CHECK(cudaMemcpy(dScale, scales, sizeof(scales), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(dA, hA.data(), nA, cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(dB, hB.data(), nB, cudaMemcpyHostToDevice));
    for (int c = 1; c < copies; ++c)
        SPARK_CUDA_CHECK(cudaMemcpy(dB + c * nB, dB, nB, cudaMemcpyDeviceToDevice));
    SPARK_CUDA_CHECK(cudaMemset(dC, 0, nC * sizeof(__nv_bfloat16)));
    const float* d_scale_a = dScale;
    const float* d_scale_b = dScale + 1;
    if (mx) {
        // The scale tensors rotate with Bt on the decode shapes (they are 1/32 of its bytes).
        const auto blkA = to_blocked(hSfa, M, KB);
        const auto blkB = to_blocked(hSfb, N, KB);
        SPARK_CUDA_CHECK(cudaMalloc(&dSfa, hSfa.size()));
        SPARK_CUDA_CHECK(cudaMalloc(&dSfb, copies * hSfb.size()));
        SPARK_CUDA_CHECK(cudaMalloc(&dSfaBlk, blkA.size()));
        SPARK_CUDA_CHECK(cudaMalloc(&dSfbBlk, copies * blkB.size()));
        SPARK_CUDA_CHECK(cudaMemcpy(dSfa, hSfa.data(), hSfa.size(), cudaMemcpyHostToDevice));
        SPARK_CUDA_CHECK(cudaMemcpy(dSfaBlk, blkA.data(), blkA.size(), cudaMemcpyHostToDevice));
        for (int c = 0; c < copies; ++c) {
            SPARK_CUDA_CHECK(cudaMemcpy(dSfb + c * hSfb.size(), hSfb.data(), hSfb.size(),
                                        cudaMemcpyHostToDevice));
            SPARK_CUDA_CHECK(cudaMemcpy(dSfbBlk + c * blkB.size(), blkB.data(), blkB.size(),
                                        cudaMemcpyHostToDevice));
        }
    }
    const size_t sfb_bytes = hSfb.size();
    const size_t sfb_blk_bytes = static_cast<size_t>((N + 127) / 128) * ((KB + 3) / 4) * 512;

    // Reference. In MX mode cuBLASLt's MXFP8 kernel if the library has one for this GPU,
    // otherwise its per-tensor kernel is timed (and said so) and the CPU sample is the only
    // check.
    LtGemm lt(M, N, K, mx ? static_cast<const void*>(dSfaBlk) : d_scale_a,
              mx ? static_cast<const void*>(dSfbBlk) : d_scale_b, mx);
    LtGemm* lt_time = &lt;
    std::unique_ptr<LtGemm> lt_plain;
    if (!lt.available) {
        lt_plain = std::make_unique<LtGemm>(M, N, K, d_scale_a, d_scale_b, false);
        lt_time = lt_plain.get();
    }
    std::vector<float> ref;
    if (lt.available) {
        lt.run(dA, dB, dRef, stream);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        ref = to_host_f32(dRef, nC);
    }

    auto run_ours = [&](const __nv_fp8_e4m3* b, const unsigned char* sfb) {
        if (mx)
            spark::fp8gemm_mx(dA, b, dC, M, N, K, nullptr, nullptr, dSfa, sfb, variant, stream);
        else
            spark::fp8gemm(dA, b, dC, M, N, K, d_scale_a, d_scale_b, variant, stream);
    };

    // Correctness against cuBLASLt. Both outputs are fp32 accumulations of exact e4m3 products,
    // scaled and rounded once to bf16 (relative step 2^-8 = 0.39%), so 2% of max|ref| covers
    // a 1-ulp difference on each side plus fp32 summation-order noise. The MX products are
    // exact too (an e4m3 pair times two powers of two), only the sum order differs.
    run_ours(dB, dSfb);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto got = to_host_f32(dC, nC);
    spark::bench::ErrorStats err;
    double max_ref = 0.0;
    if (lt.available) {
        err = spark::bench::compare(got.data(), ref.data(), nC);
        for (size_t i = 0; i < nC; ++i) max_ref = std::max(max_ref, std::fabs((double)ref[i]));
    } else {
        for (size_t i = 0; i < nC; ++i) max_ref = std::max(max_ref, std::fabs((double)got[i]));
    }
    const double tol = 2e-2 * max_ref + 1e-3;
    bool ok = err.max_abs <= tol;
    if (!ok) {
        std::fprintf(stderr,
                     "  FAIL variant %d shape %dx%dx%d vs cuBLASLt: max_abs=%.4e tol=%.4e\n",
                     variant, M, N, K, err.max_abs, tol);
    }
    // ... and against the CPU on a sample of outputs (the whole matrix would be a TFLOP at
    // 8192^3): the same tolerance against the fp32 sum of the same e4m3-rounded (in MX mode,
    // dequantized) products.
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
            const double want = static_cast<double>(acc) * scales[0] * scales[1];
            cpu_max_abs =
                std::max(cpu_max_abs, std::fabs(want - got[static_cast<size_t>(r) * N + c]));
        }
        if (cpu_max_abs > tol) {
            ok = false;
            std::fprintf(stderr, "  FAIL variant %d shape %dx%dx%d vs CPU: max_abs=%.4e tol=%.4e\n",
                         variant, M, N, K, cpu_max_abs, tol);
        }
        if (!lt.available) {
            err.max_abs = cpu_max_abs;
            err.max_rel = cpu_max_abs / std::max(1e-6, max_ref);
        }
    }

    // Timing. Each launch takes the next copy of Bt (see `copies` above) and of its scales.
    int turn = 0;
    const auto t_ref = spark::bench::time_kernel(
        [&] {
            const int c = turn++ % copies;
            if (mx && lt.available) lt.set_a_scale(dSfbBlk + c * sfb_blk_bytes);
            lt_time->run(dA, dB + static_cast<size_t>(c) * nB, dRef, stream);
        },
        stream, 5, iters);
    const auto t_us = spark::bench::time_kernel(
        [&] {
            const int c = turn++ % copies;
            run_ours(dB + static_cast<size_t>(c) * nB, dSfb + c * sfb_bytes);
        },
        stream, 5, iters);

    const double flops = 2.0 * M * N * static_cast<double>(K);
    spark::bench::Row row;
    row.kernel = "fp8gemm";
    row.dtype = mx ? "mxfp8" : "e4m3";
    row.variant = variant;
    row.shape = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    row.median_ms = t_us.median_ms;
    row.min_ms = t_us.min_ms;
    row.tflops = flops / (t_us.median_ms * 1e-3) / 1e12;
    // A and Bt once as bytes (plus their scales in MX mode), C once as bf16: the traffic floor
    // a decode shape is bound by.
    row.gbps = (static_cast<double>(nA) + nB + 2.0 * nC + (mx ? (nA + nB) / 32.0 : 0.0)) /
               (t_us.median_ms * 1e-3) / 1e9;
    row.ref_ms = t_ref.median_ms;
    row.max_abs_err = err.max_abs;
    row.max_rel_err = err.max_rel;
    row.ok = ok;
    spark::bench::print_row(row);
    std::fprintf(stderr, "    cuBLASLt%s: %.2f TFLOPS | this kernel = %.1f%% of cuBLASLt%s\n",
                 mx ? (lt.available ? " (MXFP8)" : " (per-tensor: no MXFP8 kernel here)") : "",
                 flops / (t_ref.median_ms * 1e-3) / 1e12, 100.0 * t_ref.median_ms / t_us.median_ms,
                 copies > 1 ? " | Bt rotated over copies that exceed L2 (DRAM-bound)" : "");

    SPARK_CUDA_CHECK(cudaFree(dA));
    SPARK_CUDA_CHECK(cudaFree(dB));
    SPARK_CUDA_CHECK(cudaFree(dC));
    SPARK_CUDA_CHECK(cudaFree(dRef));
    SPARK_CUDA_CHECK(cudaFree(dScale));
    if (mx) {
        SPARK_CUDA_CHECK(cudaFree(dSfa));
        SPARK_CUDA_CHECK(cudaFree(dSfb));
        SPARK_CUDA_CHECK(cudaFree(dSfaBlk));
        SPARK_CUDA_CHECK(cudaFree(dSfbBlk));
    }
    return ok;
}

// ---- model shapes (--models) ----------------------------------------------------------------
//
// bench_hgemm --models' twenty weights (model_shapes.hpp) at M = 128 to 4096 tokens, Bt as
// nn.Linear stores it. Per weight the operands are generated on the device once (a
// 128256-row lm_head is 525 MB of e4m3), every row runs cuBLASLt and each applicable
// variant on the same buffers, the output is checked on the device against cuBLASLt with
// run_one's tolerance, and Bt is rotated over copies past L2 whenever one copy would fit. In
// MX mode the scales are random powers of two in [2^-2, 2^2] (cuBLASLt gets them in its
// tiled layout) and the reference is cuBLASLt's MXFP8 kernel. Rows are "MxNxK@label".

// Uniform [-1, 1) rounded to e4m3, from a hash of the index.
__global__ void fill_e4m3_kernel(__nv_fp8_e4m3* p, size_t n, unsigned seed) {
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        unsigned x = static_cast<unsigned>(i) * 0x9E3779B1u ^ static_cast<unsigned>(i >> 32) ^ seed;
        x ^= x >> 16;
        x *= 0x7FEB352Du;
        x ^= x >> 15;
        x *= 0x846CA68Bu;
        x ^= x >> 16;
        p[i] = __nv_fp8_e4m3(static_cast<float>(x >> 8) * (2.0f / 16777216.0f) - 1.0f);
    }
}

void fill_e4m3_dev(__nv_fp8_e4m3* p, size_t n, unsigned seed) {
    fill_e4m3_kernel<<<1024, 256>>>(p, n, seed);
    SPARK_CUDA_CHECK(cudaGetLastError());
}

std::vector<unsigned char> random_ue8m0(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<unsigned char> v(n);
    for (auto& x : v) x = static_cast<unsigned char>(125 + rng() % 5);
    return v;
}

// A candidate row: label, variant and variant 3's route. Labels: v1, v2, v3old (variant 3
// on the route before it, which is what the default ran before variant 3), v3 (the
// default) and, with --tiles, v3t<n> (TMA tile n forced).
struct ModelCand {
    std::string label;
    int variant;
    int route;
};

// One weight at every M of `ms`. Returns false on a correctness failure.
bool run_models_weight(cudaStream_t stream, const spark::bench::ModelWeight& w,
                       const std::vector<int>& ms, bool mx, bool tiles, int iters) {
    const int N = w.N, K = w.K, KB = K / 32;
    const int max_m = *std::max_element(ms.begin(), ms.end());
    const size_t nB = static_cast<size_t>(N) * K;  // bytes
    const int copies =
        nB >= (size_t{256} << 20) ? 1 : static_cast<int>((size_t{256} << 20) / nB) + 1;
    const size_t nA = static_cast<size_t>(max_m) * K, nC = static_cast<size_t>(max_m) * N;
    __nv_fp8_e4m3 *dA = nullptr, *dB = nullptr;
    __nv_bfloat16 *dC = nullptr, *dRef = nullptr;
    float* dScale = nullptr;
    unsigned* scratch = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&dA, nA));
    SPARK_CUDA_CHECK(cudaMalloc(&dB, copies * nB));
    SPARK_CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dRef, nC * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dScale, 2 * sizeof(float)));
    SPARK_CUDA_CHECK(cudaMalloc(&scratch, 2 * sizeof(unsigned)));
    const float scales[2] = {mx ? 1.0f : kScaleA, mx ? 1.0f : kScaleB};
    SPARK_CUDA_CHECK(cudaMemcpy(dScale, scales, sizeof(scales), cudaMemcpyHostToDevice));
    fill_e4m3_dev(dA, nA, 1234u);
    fill_e4m3_dev(dB, nB, 5678u);
    for (int c = 1; c < copies; ++c)
        SPARK_CUDA_CHECK(cudaMemcpy(dB + c * nB, dB, nB, cudaMemcpyDeviceToDevice));
    // MX scales, row-major for us and tiled for cuBLASLt. The first M rows of the max_m-row
    // tensors are the M-row tensors in both layouts (the tiled one is row tiles outermost).
    unsigned char *dSfa = nullptr, *dSfb = nullptr, *dSfaBlk = nullptr, *dSfbBlk = nullptr;
    size_t sfb_bytes = 0, sfb_blk_bytes = 0;
    if (mx) {
        const auto sfa = random_ue8m0(static_cast<size_t>(max_m) * KB, 11);
        const auto sfb = random_ue8m0(static_cast<size_t>(N) * KB, 12);
        const auto blkA = to_blocked(sfa, max_m, KB);
        const auto blkB = to_blocked(sfb, N, KB);
        sfb_bytes = sfb.size();
        sfb_blk_bytes = blkB.size();
        SPARK_CUDA_CHECK(cudaMalloc(&dSfa, sfa.size()));
        SPARK_CUDA_CHECK(cudaMalloc(&dSfaBlk, blkA.size()));
        SPARK_CUDA_CHECK(cudaMalloc(&dSfb, copies * sfb_bytes));
        SPARK_CUDA_CHECK(cudaMalloc(&dSfbBlk, copies * sfb_blk_bytes));
        SPARK_CUDA_CHECK(cudaMemcpy(dSfa, sfa.data(), sfa.size(), cudaMemcpyHostToDevice));
        SPARK_CUDA_CHECK(cudaMemcpy(dSfaBlk, blkA.data(), blkA.size(), cudaMemcpyHostToDevice));
        for (int c = 0; c < copies; ++c) {
            SPARK_CUDA_CHECK(
                cudaMemcpy(dSfb + c * sfb_bytes, sfb.data(), sfb_bytes, cudaMemcpyHostToDevice));
            SPARK_CUDA_CHECK(cudaMemcpy(dSfbBlk + c * sfb_blk_bytes, blkB.data(), sfb_blk_bytes,
                                        cudaMemcpyHostToDevice));
        }
    }
    SPARK_CUDA_CHECK(cudaDeviceSynchronize());

    const int dflt = spark::FP8GEMM_V3_ROUTE_DEFAULT;
    std::vector<ModelCand> cands = {{"v1", 1, dflt},
                                    {"v2", 2, dflt},
                                    {"v3old", 3, spark::FP8GEMM_V3_ROUTE_LEGACY},
                                    {"v3", 3, dflt}};
    if (tiles)
        for (int t = 0; t < spark::fp8gemm_v3_num_tiles(); ++t)
            cands.push_back({"v3t" + std::to_string(t), 3, t});
    const int saved_route = spark::fp8gemm_v3_route();

    bool all_ok = true;
    for (int M : ms) {
        const size_t nc = static_cast<size_t>(M) * N;
        const std::string shape =
            std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
        std::fprintf(stderr, "  %s %s, M = %d (Bt %.1f MB%s)%s\n", w.model, w.name, M, nB / 1e6,
                     copies > 1 ? ", rotated over copies past L2" : "", mx ? ", MX" : "");
        if (mx && K % 256 != 0) continue;
        LtGemm lt(M, N, K, mx ? static_cast<const void*>(dSfaBlk) : dScale,
                  mx ? static_cast<const void*>(dSfbBlk) : dScale + 1, mx);
        if (!lt.available) {
            std::fprintf(stderr, "    cuBLASLt has no MXFP8 kernel for this shape: skipped\n");
            continue;
        }
        lt.run(dA, dB, dRef, stream);
        int turn = 0;
        const auto t_ref = spark::bench::time_kernel(
            [&] {
                const int c = turn++ % copies;
                if (mx) lt.set_a_scale(dSfbBlk + c * sfb_blk_bytes);
                lt.run(dA, dB + static_cast<size_t>(c) * nB, dC, stream);
            },
            stream, 5, iters);
        if (mx) lt.set_a_scale(dSfbBlk);
        const double flops = 2.0 * M * N * static_cast<double>(K);
        std::fprintf(stderr, "    cuBLASLt%s %.4f ms, %.1f TFLOPS\n", mx ? " MXFP8" : "",
                     t_ref.median_ms, flops / (t_ref.median_ms * 1e-3) / 1e12);
        for (const auto& c : cands) {
            const bool sup = mx ? spark::fp8gemm_mx_supports(M, N, K, c.variant)
                                : spark::fp8gemm_supports(M, N, K, c.variant);
            if (!sup) continue;
            if (c.route >= 0) {
                // A forced tile that does not fit falls back to the old route, and a forced
                // tile the rule picks anyway is the v3 row: skip both duplicates.
                spark::fp8gemm_v3_set_route(dflt);
                const std::string by_rule = spark::fp8gemm_v3_route_name(M, N, K, mx);
                spark::fp8gemm_v3_set_route(c.route);
                const std::string forced = spark::fp8gemm_v3_route_name(M, N, K, mx);
                if (by_rule == forced || forced == "v1" || forced == "v2") continue;
            }
            spark::fp8gemm_v3_set_route(c.route);
            auto run_ours = [&](int copy) {
                const __nv_fp8_e4m3* b = dB + static_cast<size_t>(copy) * nB;
                if (mx)
                    spark::fp8gemm_mx(dA, b, dC, M, N, K, nullptr, nullptr, dSfa,
                                      dSfb + copy * sfb_bytes, c.variant, stream);
                else
                    spark::fp8gemm(dA, b, dC, M, N, K, dScale, dScale + 1, c.variant, stream);
            };
            SPARK_CUDA_CHECK(cudaMemsetAsync(dC, 0, nc * sizeof(__nv_bfloat16), stream));
            run_ours(0);
            const auto err = spark::bench::max_err_dev(dC, dRef, nc, scratch, stream);
            const double tol = 2e-2 * err.second + 1e-3;
            const bool ok = err.first <= tol;
            if (!ok)
                std::fprintf(stderr, "  FAIL %s %s: max_abs=%.4e tol=%.4e\n", shape.c_str(),
                             c.label.c_str(), err.first, tol);
            all_ok = all_ok && ok;
            turn = 0;
            const auto t =
                spark::bench::time_kernel([&] { run_ours(turn++ % copies); }, stream, 5, iters);
            spark::bench::Row row;
            row.kernel = "fp8gemm_models";
            row.dtype = mx ? "mxfp8" : "e4m3";
            row.variant = c.variant;
            row.shape = shape + "@" + c.label;
            row.median_ms = t.median_ms;
            row.min_ms = t.min_ms;
            row.tflops = flops / (t.median_ms * 1e-3) / 1e12;
            row.gbps = (static_cast<double>(M) * K + nB + 2.0 * nc) / (t.median_ms * 1e-3) / 1e9;
            row.ref_ms = t_ref.median_ms;
            row.max_abs_err = err.first;
            row.max_rel_err = err.second > 0 ? err.first / err.second : 0.0;
            row.ok = ok;
            spark::bench::print_row(row);
            std::fprintf(stderr, "    %-8s %-34s %.1f%% of cuBLASLt\n", c.label.c_str(),
                         c.variant == 3 ? spark::fp8gemm_v3_route_name(M, N, K, mx) : "",
                         100.0 * t_ref.median_ms / t.median_ms);
        }
    }
    spark::fp8gemm_v3_set_route(saved_route);
    for (void* d : {static_cast<void*>(dA), static_cast<void*>(dB), static_cast<void*>(dC),
                    static_cast<void*>(dRef), static_cast<void*>(dScale),
                    static_cast<void*>(scratch), static_cast<void*>(dSfa), static_cast<void*>(dSfb),
                    static_cast<void*>(dSfaBlk), static_cast<void*>(dSfbBlk)})
        if (d) SPARK_CUDA_CHECK(cudaFree(d));
    return all_ok;
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
    // --mode=tensor (per-tensor scales), mx (per-32-k ue8m0 scales) or both (the default;
    // tensor with --models).
    const std::string mode = args.get("mode", args.has("models") ? "tensor" : "both");
    SPARK_REQUIRE(mode == "tensor" || mode == "mx" || mode == "both",
                  "--mode must be tensor, mx or both");
    std::vector<bool> modes;
    if (mode != "mx") modes.push_back(false);
    if (mode != "tensor") {
        if (spark::fp8gemm_mx_available())
            modes.push_back(true);
        else
            std::fprintf(stderr, "no block-scaled mma in this build: skipping MX rows\n");
    }

    cudaStream_t stream = nullptr;  // legacy default stream, as bench_hgemm
    spark::bench::print_header();
    bool all_ok = true;
    if (args.has("models")) {
        // --models[=<model substring>] [--ms=128,256,...] [--mode=tensor|mx|both] [--tiles]
        const std::string filter = args.get("models", "1");
        const std::vector<int> ms =
            spark::bench::parse_int_list(args.get("ms", "128,256,512,1024,2048,4096"));
        SPARK_REQUIRE(!ms.empty(), "--ms: expected a comma-separated list of M");
        for (bool mx : modes) {
            for (const auto& w : spark::bench::model_weights()) {
                if (filter != "1" && std::string(w.model).find(filter) == std::string::npos)
                    continue;
                all_ok = run_models_weight(stream, w, ms, mx, args.has("tiles"), iters) && all_ok;
            }
        }
    } else {
        for (bool mx : modes) {
            for (const auto& s : shapes) {
                for (int v : variants) all_ok = run_one(stream, s, v, iters, mx) && all_ok;
            }
        }
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
