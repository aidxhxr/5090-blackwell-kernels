// Benchmark + correctness check for the fp4 (e2m1) block-scaled GEMM ladder, NVFP4 and MXFP4,
// against cuBLASLt, and of the bf16 -> fp4 quantizer.
//
//   ./bench_fp4gemm                         # default shapes, all variants, NVFP4 then MXFP4,
//                                           # then the quantizer rows
//   ./bench_fp4gemm --m=4096 --n=4096 --k=4096 --variant=2 --iters=50 --format=nvfp4
//   ./bench_fp4gemm --quant=0               # GEMM rows only
//   ./bench_fp4gemm --ours-first=1          # time our kernel before cuBLASLt
//
// C = scale_a * scale_b * sum_k (sfa A)(sfb Bt), A [M, K] and Bt [N, K] e2m1 packed two per
// byte with K contiguous, block scales in the blocked layout (spark/kernels.h), bf16 out. The
// inputs are Gaussian rows quantized on the CPU with the recipe of spark::fp4_quantize
// (NVFP4: per-tensor s = max|x| / 2688, an e4m3 scale per 16; MXFP4: a ue8m0 scale per 32),
// and the GPU quantizer's bytes are checked against the CPU's. Every GEMM row is checked
// against cuBLASLt's block-scaled fp4 matmul (CUDA_R_4F_E2M1 with VEC16_UE4M3 or
// VEC32_UE8M0 scales, which take exactly this layout) when the library has one for this GPU,
// and against a CPU fp32 dot product of the dequantized values on 4,096 sampled outputs.
//
// stdout: one JSON object per row (kernel "fp4gemm", dtype "nvfp4" / "mxfp4"; kernel
// "fp4quant" for the quantizer). stderr: human-readable table incl. "% of cuBLASLt".

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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

// ---- CPU quantizer: the same operations as src/kernels/fp4quant.cu --------------------------

unsigned e2m1_code(float v) {
    const float a = std::fabs(v);
    const unsigned c = a <= 0.25f   ? 0u
                       : a < 0.75f  ? 1u
                       : a <= 1.25f ? 2u
                       : a < 1.75f  ? 3u
                       : a <= 2.5f  ? 4u
                       : a < 3.5f   ? 5u
                       : a <= 5.0f  ? 6u
                                    : 7u;
    return c | (v < 0.f ? 8u : 0u);
}

float e2m1_value(unsigned c) {
    static const float mag[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    return (c & 8u) ? -mag[c & 7u] : mag[c & 7u];
}

size_t sf_offset(int r, int j, int cols) {
    const size_t tile = static_cast<size_t>(r / 128) * (cols / 4) + j / 4;
    return tile * 512 + (r % 32) * 16 + ((r / 32) % 4) * 4 + (j % 4);
}

// A quantized operand: packed codes, blocked scales, the per-tensor decode scale, and the
// dequantized fp32 value of every element (sf * code, without the per-tensor scale) for the
// CPU reference.
struct Quantized {
    std::vector<unsigned char> q, sf;
    std::vector<float> deq;
    float s = 1.f;
};

Quantized quantize(const std::vector<__nv_bfloat16>& x, int rows, int K, bool mx) {
    Quantized o;
    const int V = mx ? 32 : 16;
    const int cols = K / V;
    const int padded = (rows + 127) / 128 * 128;
    o.q.assign(static_cast<size_t>(rows) * K / 2, 0);
    o.sf.assign(static_cast<size_t>(padded) * cols, 0);
    o.deq.assign(static_cast<size_t>(rows) * K, 0.f);
    if (!mx) {
        float amax = 0.f;
        for (const auto& v : x) amax = std::max(amax, std::fabs(__bfloat162float(v)));
        o.s = amax / (6.0f * 448.0f);
    }
    std::vector<float> f(V);
    for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < cols; ++j) {
            float amax = 0.f;
            for (int i = 0; i < V; ++i) {
                f[i] = __bfloat162float(x[static_cast<size_t>(r) * K + j * V + i]);
                amax = std::max(amax, std::fabs(f[i]));
            }
            unsigned char sfb;
            float blk;
            unsigned c[32];
            if (!mx) {
                const __nv_fp8_e4m3 sfe(std::min(amax / 6.0f / o.s, 448.0f));
                sfb = sfe.__x;
                blk = static_cast<float>(sfe);
                const float denom = blk * o.s;
                for (int i = 0; i < V; ++i) c[i] = denom > 0.f ? e2m1_code(f[i] / denom) : 0u;
            } else {
                int p = 0;
                std::frexp(amax, &p);
                int e = amax > 0.f ? p - 1 - 2 : -127;
                e = std::max(-127, std::min(127, e));
                sfb = static_cast<unsigned char>(e + 127);
                blk = std::ldexp(1.0f, e);
                for (int i = 0; i < V; ++i) c[i] = e2m1_code(std::ldexp(f[i], -e));
            }
            o.sf[sf_offset(r, j, cols)] = sfb;
            for (int i = 0; i < V; ++i) {
                const size_t k = static_cast<size_t>(r) * K + j * V + i;
                o.q[k / 2] |= static_cast<unsigned char>(c[i] << (4 * (k % 2)));
                o.deq[k] = e2m1_value(c[i]) * blk;
            }
        }
    }
    return o;
}

std::vector<__nv_bfloat16> gaussian(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);
    std::vector<__nv_bfloat16> x(n);
    for (auto& v : x) v = __float2bfloat16(dist(rng));
    return x;
}

// ---- cuBLASLt reference ---------------------------------------------------------------------

// cuBLASLt block-scaled fp4 matmul in its TN layout: op(A') = A'^T with A' K x N column-major
// (our Bt), B' K x M column-major (our A), D N x M column-major (our row-major C). So its A
// scale tensor is our sfb and its B scale our sfa, and the per-tensor scales go into alpha.
struct LtGemm {
    cublasLtHandle_t handle = nullptr;
    cublasLtMatmulDesc_t op = nullptr;
    cublasLtMatrixLayout_t la = nullptr, lb = nullptr, lc = nullptr;
    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulHeuristicResult_t heur{};
    void* ws = nullptr;
    size_t ws_bytes = size_t{32} << 20;
    bool available = false;
    float alpha = 1.f;

    LtGemm(int M, int N, int K, const void* sfa, const void* sfb, bool mx, float alpha_)
        : alpha(alpha_) {
        CUBLAS_CHECK(cublasLtCreate(&handle));
        CUBLAS_CHECK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t t = CUBLAS_OP_T, n = CUBLAS_OP_N;
        CUBLAS_CHECK(
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof(t)));
        CUBLAS_CHECK(
            cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &n, sizeof(n)));
        const cublasLtMatmulMatrixScale_t mode = mx ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
                                                    : CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode,
                                                    sizeof(mode)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode,
                                                    sizeof(mode)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sfb,
                                                    sizeof(sfb)));
        CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sfa,
                                                    sizeof(sfa)));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, K, N, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, K, M, K));
        CUBLAS_CHECK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, N, M, N));
        CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
        CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
            pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
        int returned = 0;
        const cublasStatus_t st =
            cublasLtMatmulAlgoGetHeuristic(handle, op, la, lb, lc, lc, pref, 1, &heur, &returned);
        available = st == CUBLAS_STATUS_SUCCESS && returned > 0;
        SPARK_CUDA_CHECK(cudaMalloc(&ws, ws_bytes));
    }
    void run(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* C,
             cudaStream_t stream) {
        const float beta = 0.0f;
        CUBLAS_CHECK(cublasLtMatmul(handle, op, &alpha, Bt, la, A, lb, &beta, C, lc, C, lc,
                                    &heur.algo, ws, ws_bytes, stream));
    }
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

// The operands of one shape, quantized once and shared by every variant of that shape.
struct Operands {
    Shape s{};
    bool mx = false;
    Quantized a, b;
};

bool run_one(cudaStream_t stream, const Operands& op, int variant, int iters, bool ours_first) {
    const int M = op.s.M, N = op.s.N, K = op.s.K;
    const bool mx = op.mx;
    const int format = mx ? spark::FP4_MXFP4 : spark::FP4_NVFP4;
    if (!spark::fp4gemm_supports(M, N, K, variant)) {
        std::fprintf(stderr, "  skip variant %d shape %dx%dx%d: shape not supported\n", variant, M,
                     N, K);
        return true;
    }
    const size_t nA = static_cast<size_t>(M) * K / 2, nB = static_cast<size_t>(N) * K / 2,
                 nC = static_cast<size_t>(M) * N;
    const size_t sfa_bytes = op.a.sf.size(), sfb_bytes = op.b.sf.size();
    // Decode shapes stream Bt from DRAM: rotate through copies that exceed the 96 MB L2.
    const int copies = M <= 64 ? static_cast<int>((size_t{256} << 20) / (nB + sfb_bytes)) + 1 : 1;

    unsigned char *dA = nullptr, *dB = nullptr, *dSfa = nullptr, *dSfb = nullptr;
    __nv_bfloat16 *dC = nullptr, *dRef = nullptr;
    float* dScale = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&dA, nA));
    SPARK_CUDA_CHECK(cudaMalloc(&dB, copies * nB));
    SPARK_CUDA_CHECK(cudaMalloc(&dSfa, sfa_bytes));
    SPARK_CUDA_CHECK(cudaMalloc(&dSfb, copies * sfb_bytes));
    SPARK_CUDA_CHECK(cudaMalloc(&dC, nC * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dRef, nC * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dScale, 2 * sizeof(float)));
    const float scales[2] = {op.a.s, op.b.s};
    SPARK_CUDA_CHECK(cudaMemcpy(dScale, scales, sizeof(scales), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(dA, op.a.q.data(), nA, cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(dSfa, op.a.sf.data(), sfa_bytes, cudaMemcpyHostToDevice));
    for (int c = 0; c < copies; ++c) {
        SPARK_CUDA_CHECK(cudaMemcpy(dB + c * nB, op.b.q.data(), nB, cudaMemcpyHostToDevice));
        SPARK_CUDA_CHECK(
            cudaMemcpy(dSfb + c * sfb_bytes, op.b.sf.data(), sfb_bytes, cudaMemcpyHostToDevice));
    }
    SPARK_CUDA_CHECK(cudaMemset(dC, 0, nC * sizeof(__nv_bfloat16)));
    const float* d_sa = mx ? nullptr : dScale;
    const float* d_sb = mx ? nullptr : dScale + 1;
    const double alpha = mx ? 1.0 : static_cast<double>(op.a.s) * op.b.s;

    LtGemm lt(M, N, K, dSfa, dSfb, mx, static_cast<float>(alpha));
    std::vector<float> ref;
    if (lt.available) {
        lt.run(dA, dB, dRef, stream);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        ref = to_host_f32(dRef, nC);
    }
    auto run_ours = [&](int c) {
        spark::fp4gemm(dA, dB + static_cast<size_t>(c) * nB, dC, M, N, K, dSfa,
                       dSfb + static_cast<size_t>(c) * sfb_bytes, d_sa, d_sb, format, variant,
                       stream);
    };

    // Correctness. The products of two dequantized e2m1 values are exact in fp32, and both
    // sides sum them in fp32 and round once to bf16 (a 2^-8 relative step), so 2% of max|ref|
    // covers an ulp on each side plus summation-order noise.
    run_ours(0);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto got = to_host_f32(dC, nC);
    spark::bench::ErrorStats err;
    double max_ref = 0.0;
    size_t identical = 0;
    if (lt.available) {
        err = spark::bench::compare(got.data(), ref.data(), nC);
        for (size_t i = 0; i < nC; ++i) {
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref[i])));
            identical += got[i] == ref[i];
        }
    } else {
        for (size_t i = 0; i < nC; ++i)
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(got[i])));
    }
    const double tol = 2e-2 * max_ref + 1e-3;
    bool ok = err.max_abs <= tol;
    if (!ok)
        std::fprintf(stderr,
                     "  FAIL variant %d shape %dx%dx%d vs cuBLASLt: max_abs=%.4e tol=%.4e\n",
                     variant, M, N, K, err.max_abs, tol);
    {
        std::mt19937 rng(99);
        double cpu_max_abs = 0.0;
        for (int i = 0; i < 4096; ++i) {
            const int r = static_cast<int>(rng() % static_cast<unsigned>(M));
            const int c = static_cast<int>(rng() % static_cast<unsigned>(N));
            float acc = 0.f;
            const float* a = op.a.deq.data() + static_cast<size_t>(r) * K;
            const float* b = op.b.deq.data() + static_cast<size_t>(c) * K;
            for (int k = 0; k < K; ++k) acc = std::fmaf(a[k], b[k], acc);
            const double want = static_cast<double>(acc) * alpha;
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

    // Timing, cuBLASLt first unless --ours-first: at the 600 W limit the card runs a few
    // percent slower once it is hot, so the order is a knob for checking that the ratio is
    // not an artifact of who runs second.
    int turn = 0;
    spark::bench::Timing t_ref, t_us;
    auto time_ref = [&] {
        if (!lt.available) return;
        t_ref = spark::bench::time_kernel(
            [&] {
                const int c = turn++ % copies;
                lt.set_a_scale(dSfb + static_cast<size_t>(c) * sfb_bytes);
                lt.run(dA, dB + static_cast<size_t>(c) * nB, dRef, stream);
            },
            stream, 5, iters);
    };
    auto time_us = [&] {
        t_us = spark::bench::time_kernel([&] { run_ours(turn++ % copies); }, stream, 5, iters);
    };
    if (ours_first) {
        time_us();
        time_ref();
    } else {
        time_ref();
        time_us();
    }

    const double flops = 2.0 * M * N * static_cast<double>(K);
    spark::bench::Row row;
    row.kernel = "fp4gemm";
    row.dtype = mx ? "mxfp4" : "nvfp4";
    row.variant = variant;
    row.shape = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
    row.median_ms = t_us.median_ms;
    row.min_ms = t_us.min_ms;
    row.tflops = flops / (t_us.median_ms * 1e-3) / 1e12;
    // A and Bt at half a byte per value plus a scale byte per block, C once as bf16.
    const double V = mx ? 32.0 : 16.0;
    row.gbps = ((static_cast<double>(M) + N) * K * (0.5 + 1.0 / V) + 2.0 * nC) /
               (t_us.median_ms * 1e-3) / 1e9;
    row.ref_ms = lt.available ? t_ref.median_ms : 0.0;
    row.max_abs_err = err.max_abs;
    row.max_rel_err = err.max_rel;
    row.ok = ok;
    spark::bench::print_row(row);
    if (lt.available)
        std::fprintf(stderr,
                     "    cuBLASLt %s: %.2f TFLOPS | this kernel = %.1f%% of cuBLASLt | %.4f%% of "
                     "outputs bit-identical%s\n",
                     mx ? "MXFP4" : "NVFP4", flops / (t_ref.median_ms * 1e-3) / 1e12,
                     100.0 * t_ref.median_ms / t_us.median_ms, 100.0 * identical / nC,
                     copies > 1 ? " | Bt rotated over copies that exceed L2" : "");
    else
        std::fprintf(stderr, "    cuBLASLt has no %s kernel for this GPU: CPU check only\n",
                     mx ? "MXFP4" : "NVFP4");

    SPARK_CUDA_CHECK(cudaFree(dA));
    SPARK_CUDA_CHECK(cudaFree(dB));
    SPARK_CUDA_CHECK(cudaFree(dSfa));
    SPARK_CUDA_CHECK(cudaFree(dSfb));
    SPARK_CUDA_CHECK(cudaFree(dC));
    SPARK_CUDA_CHECK(cudaFree(dRef));
    SPARK_CUDA_CHECK(cudaFree(dScale));
    return ok;
}

// The quantizer: its bytes against the CPU recipe's, and its time as GB/s of the traffic floor
// (2 bytes in, half a byte plus the scale bytes out per value).
bool run_quant(cudaStream_t stream, int rows, int K, bool mx, int iters) {
    const int format = mx ? spark::FP4_MXFP4 : spark::FP4_NVFP4;
    const auto x = gaussian(static_cast<size_t>(rows) * K, 4321);
    const Quantized want = quantize(x, rows, K, mx);
    __nv_bfloat16* dX = nullptr;
    unsigned char *dQ = nullptr, *dSf = nullptr;
    float* dS = nullptr;
    const size_t nq = want.q.size(), nsf = want.sf.size();
    SPARK_CUDA_CHECK(cudaMalloc(&dX, x.size() * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dQ, nq));
    SPARK_CUDA_CHECK(cudaMalloc(&dSf, nsf));
    SPARK_CUDA_CHECK(cudaMalloc(&dS, sizeof(float)));
    SPARK_CUDA_CHECK(
        cudaMemcpy(dX, x.data(), x.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(dS, &want.s, sizeof(float), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemset(dSf, 0xFF, nsf));  // the kernel must write every byte
    auto run = [&] { spark::fp4_quantize(dX, dQ, dSf, rows, K, dS, format, stream); };
    run();
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::vector<unsigned char> q(nq), sf(nsf);
    SPARK_CUDA_CHECK(cudaMemcpy(q.data(), dQ, nq, cudaMemcpyDeviceToHost));
    SPARK_CUDA_CHECK(cudaMemcpy(sf.data(), dSf, nsf, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    for (size_t i = 0; i < nq; ++i) bad += q[i] != want.q[i];
    for (size_t i = 0; i < nsf; ++i) bad += sf[i] != want.sf[i];
    const bool ok = bad == 0;
    if (!ok) std::fprintf(stderr, "  FAIL quantizer %dx%d: %zu bytes differ\n", rows, K, bad);
    const auto t = spark::bench::time_kernel(run, stream, 5, iters);
    spark::bench::Row row;
    row.kernel = "fp4quant";
    row.dtype = mx ? "mxfp4" : "nvfp4";
    row.variant = 0;
    row.shape = std::to_string(rows) + "x" + std::to_string(K);
    row.median_ms = t.median_ms;
    row.min_ms = t.min_ms;
    row.gbps = (2.0 * x.size() + nq + nsf) / (t.median_ms * 1e-3) / 1e9;
    row.max_abs_err = static_cast<double>(bad);
    row.ok = ok;
    spark::bench::print_row(row);
    SPARK_CUDA_CHECK(cudaFree(dX));
    SPARK_CUDA_CHECK(cudaFree(dQ));
    SPARK_CUDA_CHECK(cudaFree(dSf));
    SPARK_CUDA_CHECK(cudaFree(dS));
    return ok;
}

}  // namespace

int run(int argc, char** argv) {
    spark::bench::Args args(argc, argv);
    spark::bench::print_device_banner();
    if (!spark::fp4gemm_available()) {
        std::fprintf(stderr,
                     "no fp4 mma in this build (needs sm_120a / sm_121a): nothing to run\n");
        return 0;
    }

    std::vector<Shape> shapes;
    if (args.has("m") || args.has("n") || args.has("k")) {
        const int m = args.geti("m", 4096);
        shapes.push_back({m, args.geti("n", m), args.geti("k", m)});
    } else {
        // Squares, the Llama-3-8B projections at 4096 tokens (qkv 4096 -> 6144, gate|up
        // 4096 -> 2 x 14336, down 14336 -> 4096; o is the 4096 square), then decode shapes
        // where the GEMM streams the fp4 weights.
        shapes = {{1024, 1024, 1024}, {2048, 2048, 2048},  {4096, 4096, 4096},  {8192, 8192, 8192},
                  {4096, 6144, 4096}, {4096, 28672, 4096}, {4096, 4096, 14336}, {1, 4096, 4096},
                  {16, 4096, 4096},   {64, 4096, 4096},    {16, 28672, 4096},   {16, 4096, 14336}};
    }
    const int iters = args.geti("iters", 50);
    const bool ours_first = args.geti("ours-first", 0) != 0;
    std::vector<int> variants;
    if (args.has("variant")) {
        variants.push_back(args.geti("variant", 0));
    } else {
        for (int v = 0; v < spark::fp4gemm_num_variants(); ++v) variants.push_back(v);
    }
    const std::string format = args.get("format", "both");
    SPARK_REQUIRE(format == "nvfp4" || format == "mxfp4" || format == "both",
                  "--format must be nvfp4, mxfp4 or both");
    std::vector<bool> modes;
    if (format != "mxfp4") modes.push_back(false);
    if (format != "nvfp4") modes.push_back(true);

    cudaStream_t stream = nullptr;
    spark::bench::print_header();
    bool all_ok = true;
    for (bool mx : modes) {
        for (const auto& s : shapes) {
            Operands op;
            op.s = s;
            op.mx = mx;
            op.a = quantize(gaussian(static_cast<size_t>(s.M) * s.K, 1234), s.M, s.K, mx);
            op.b = quantize(gaussian(static_cast<size_t>(s.N) * s.K, 5678), s.N, s.K, mx);
            for (int v : variants) all_ok = run_one(stream, op, v, iters, ours_first) && all_ok;
        }
    }
    if (args.geti("quant", 1)) {
        for (bool mx : modes) {
            for (const auto& rk : {std::pair<int, int>{4096, 4096}, {4096, 14336}}) {
                all_ok = run_quant(stream, rk.first, rk.second, mx, iters) && all_ok;
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
        std::fprintf(stderr, "bench_fp4gemm: %s\n", e.what());
        return 2;
    }
}
