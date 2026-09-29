// Benchmark + correctness check for the fp8 (e4m3) attention ladder.
//
//   ./bench_attention_fp8                                   # default shape sweep, all variants
//   ./bench_attention_fp8 --b=1 --h=32 --s=4096 --d=128 --causal=1 --variant=1 --iters=20
//   ./bench_attention_fp8 --hq=32 --hkv=8 --s=4096 ...      # grouped-query attention
//   ./bench_attention_fp8 ... --dist=outlier                # inputs with 0.1% of entries x 20
//   ./bench_attention_fp8 ... --per_head=1                  # one descale factor per head
//
// Inputs: Q and K uniform in [-2, 2], V in [-1, 1] (bench_attention's), or with
// --dist=gauss standard normal, or --dist=outlier normal with 0.1% of the entries times 20.
// Each tensor is quantized to e4m3 with a per-tensor (or per-head) scale max|x| / 448 on the
// host; the kernel gets the e4m3 bytes and the scales.
//
// Correctness: a CPU double-precision reference computed from the dequantized e4m3 values (so
// the only difference left is the kernel's own rounding: P to e4m3 in variant 1, the bf16
// output), on every row of the small shapes and 64 sampled (b, h, row) triples of the large
// ones. Tolerance max|O - O_ref| <= 0.04 max|O_ref| + 4e-3: an e4m3 probability carries a
// relative rounding error of up to 2^-4, and on a row whose softmax is dominated by a few keys
// that error does not average out.
//
// The reference time (ref_ms) is attention variant 5, the bf16 kernel, on the same shape
// with the unquantized inputs rounded to bf16: "%" in the table is how much faster the fp8
// kernel is than the fastest bf16 rung.
//
// stdout: one JSON object per row (collected by scripts/run_all_benches.sh)
// stderr: human-readable table, and per row the error of both kernels against the fp64
// attention of the unquantized inputs

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "spark/kernels.h"

namespace {

struct Shape {
    int B, Hq, Hkv, Sq, Skv, D;
    int causal;
};

// bench_attention's shape string (parsed back by scripts/shape_utils.py).
std::string shape_str(const Shape& s) {
    std::string r = "b" + std::to_string(s.B);
    if (s.Hq == s.Hkv)
        r += "_h" + std::to_string(s.Hq);
    else
        r += "_hq" + std::to_string(s.Hq) + "_hkv" + std::to_string(s.Hkv);
    if (s.Sq == s.Skv)
        r += "_s" + std::to_string(s.Sq);
    else
        r += "_sq" + std::to_string(s.Sq) + "_skv" + std::to_string(s.Skv);
    r += "_d" + std::to_string(s.D);
    if (s.causal) r += "_causal";
    return r;
}

std::vector<float> to_host_f32(const __nv_bfloat16* d, size_t n) {
    std::vector<__nv_bfloat16> h(n);
    SPARK_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = __bfloat162float(h[i]);
    return out;
}

void fill(std::vector<float>& v, const std::string& dist, float lo, float hi, uint32_t seed) {
    if (dist == "uniform") {
        spark::bench::fill_uniform(v, lo, hi, seed);
        return;
    }
    std::mt19937 rng(seed);
    std::normal_distribution<float> n01(0.f, 1.f);
    std::uniform_real_distribution<float> u01(0.f, 1.f);
    for (auto& x : v) {
        x = n01(rng);
        if (dist == "outlier" && u01(rng) < 1e-3f) x *= 20.f;
    }
}

// Quantizes `x` in `groups` equal contiguous groups (1 = per tensor, B * H = per head) with
// scale max|x| / 448 each: e4m3 bytes (round to nearest even, saturating), the scales, and
// x replaced by the dequantized values the kernel will see.
void quantize(std::vector<float>& x, int groups, std::vector<__nv_fp8_e4m3>& q,
              std::vector<float>& scales) {
    const size_t n = x.size(), per = n / groups;
    q.resize(n);
    scales.resize(groups);
    for (int gi = 0; gi < groups; ++gi) {
        float amax = 0.f;
        for (size_t i = gi * per; i < (gi + 1) * per; ++i) amax = std::max(amax, std::fabs(x[i]));
        const float s = std::max(amax / 448.f, 1e-30f);
        scales[gi] = s;
        for (size_t i = gi * per; i < (gi + 1) * per; ++i) {
            q[i] = __nv_fp8_e4m3(x[i] / s);
            x[i] = static_cast<float>(q[i]) * s;
        }
    }
}

// One output row in double precision (softmax as exp(s - max) / sum). Query head h of batch b
// reads K/V head h / (H_q / H_kv).
void reference_row(const std::vector<float>& hq, const std::vector<float>& hk,
                   const std::vector<float>& hv, const Shape& s, int bh, int i,
                   std::vector<double>& out) {
    const int D = s.D;
    const int b = bh / s.Hq, h = bh % s.Hq;
    const int bkv = b * s.Hkv + h / (s.Hq / s.Hkv);
    const float* q = hq.data() + (static_cast<size_t>(bh) * s.Sq + i) * D;
    const float* k = hk.data() + static_cast<size_t>(bkv) * s.Skv * D;
    const float* v = hv.data() + static_cast<size_t>(bkv) * s.Skv * D;
    const int n = s.causal ? std::min(s.Skv, i + 1) : s.Skv;
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    std::vector<double> sc(n);
    double mx = -INFINITY;
    for (int j = 0; j < n; ++j) {
        double acc = 0.0;
        for (int d = 0; d < D; ++d)
            acc += static_cast<double>(q[d]) * k[static_cast<size_t>(j) * D + d];
        sc[j] = acc * scale;
        mx = std::max(mx, sc[j]);
    }
    out.assign(D, 0.0);
    double sum = 0.0;
    for (int j = 0; j < n; ++j) {
        const double p = std::exp(sc[j] - mx);
        sum += p;
        for (int d = 0; d < D; ++d) out[d] += p * v[static_cast<size_t>(j) * D + d];
    }
    for (int d = 0; d < D; ++d) out[d] /= sum;
}

struct Err {
    double max_abs = 0, mean_abs = 0, max_rel = 0, max_ref = 0;
};

Err compare_rows(const std::vector<float>& got, const std::vector<std::vector<double>>& refs,
                 const std::vector<std::pair<int, int>>& rows, const Shape& s) {
    Err e;
    size_t n = 0;
    for (size_t r = 0; r < rows.size(); ++r) {
        const auto& [bh, i] = rows[r];
        const float* o = got.data() + (static_cast<size_t>(bh) * s.Sq + i) * s.D;
        for (int d = 0; d < s.D; ++d) {
            const double diff = std::fabs(static_cast<double>(o[d]) - refs[r][d]);
            e.max_abs = std::max(e.max_abs, diff);
            e.mean_abs += diff;
            e.max_rel = std::max(e.max_rel, diff / std::max(1e-6, std::fabs(refs[r][d])));
            e.max_ref = std::max(e.max_ref, std::fabs(refs[r][d]));
            ++n;
        }
    }
    e.mean_abs /= static_cast<double>(std::max<size_t>(n, 1));
    return e;
}

struct Opts {
    std::string dist = "uniform";
    int per_head = 0;
    int iters = 50;
    int compare_bf16 = 1;  // time attention variant 5 as the reference
};

// Runs one shape for the given variants. Returns false on a correctness failure.
bool run_shape(cudaStream_t stream, const Shape& s, const std::vector<int>& variants,
               const Opts& opt) {
    const size_t BH = static_cast<size_t>(s.B) * s.Hq;
    const size_t BHkv = static_cast<size_t>(s.B) * s.Hkv;
    const size_t nQ = BH * s.Sq * s.D, nK = BHkv * s.Skv * s.D;

    std::vector<float> fq(nQ), fk(nK), fv(nK);
    fill(fq, opt.dist, -2.f, 2.f, 1234);
    fill(fk, opt.dist, -2.f, 2.f, 5678);
    fill(fv, opt.dist, -1.f, 1.f, 9012);
    // The bf16 kernel's inputs, and the fp64 attention of those (the "true" answer the fp8
    // error is quoted against on stderr).
    std::vector<__nv_bfloat16> bq(nQ), bk(nK), bv(nK);
    std::vector<float> rq(nQ), rk(nK), rv(nK);
    for (size_t i = 0; i < nQ; ++i) {
        bq[i] = __float2bfloat16(fq[i]);
        rq[i] = __bfloat162float(bq[i]);
    }
    for (size_t i = 0; i < nK; ++i) {
        bk[i] = __float2bfloat16(fk[i]);
        rk[i] = __bfloat162float(bk[i]);
        bv[i] = __float2bfloat16(fv[i]);
        rv[i] = __bfloat162float(bv[i]);
    }
    std::vector<__nv_fp8_e4m3> q8, k8, v8;
    std::vector<float> sq, sk, sv;
    quantize(fq, opt.per_head ? static_cast<int>(BH) : 1, q8, sq);
    quantize(fk, opt.per_head ? static_cast<int>(BHkv) : 1, k8, sk);
    quantize(fv, opt.per_head ? static_cast<int>(BHkv) : 1, v8, sv);

    __nv_fp8_e4m3 *dQ = nullptr, *dK = nullptr, *dV = nullptr;
    __nv_bfloat16 *dO = nullptr, *dQb = nullptr, *dKb = nullptr, *dVb = nullptr;
    float* dS = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&dQ, nQ));
    SPARK_CUDA_CHECK(cudaMalloc(&dK, nK));
    SPARK_CUDA_CHECK(cudaMalloc(&dV, nK));
    SPARK_CUDA_CHECK(cudaMalloc(&dO, nQ * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dS, (sq.size() + sk.size() + sv.size()) * sizeof(float)));
    SPARK_CUDA_CHECK(cudaMemcpy(dQ, q8.data(), nQ, cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(dK, k8.data(), nK, cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(dV, v8.data(), nK, cudaMemcpyHostToDevice));
    float* dsq = dS;
    float* dsk = dsq + sq.size();
    float* dsv = dsk + sk.size();
    SPARK_CUDA_CHECK(cudaMemcpy(dsq, sq.data(), sq.size() * 4, cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(dsk, sk.data(), sk.size() * 4, cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(dsv, sv.data(), sv.size() * 4, cudaMemcpyHostToDevice));

    // Which rows to check: all of them while the reference is cheap, else 64 sampled rows
    // (always the first and last query row of the first head).
    const double ref_work = static_cast<double>(BH) * s.Sq * s.Skv * s.D;
    std::vector<std::pair<int, int>> rows;
    if (ref_work <= 2.0e9) {
        for (size_t bh = 0; bh < BH; ++bh)
            for (int i = 0; i < s.Sq; ++i) rows.emplace_back(static_cast<int>(bh), i);
    } else {
        std::mt19937 rng(7);
        rows.emplace_back(0, 0);
        rows.emplace_back(0, s.Sq - 1);
        for (int n = 0; n < 62; ++n)
            rows.emplace_back(static_cast<int>(rng() % BH), static_cast<int>(rng() % s.Sq));
    }
    std::vector<std::vector<double>> ref_q(rows.size()), ref_true(rows.size());
    for (size_t r = 0; r < rows.size(); ++r) {
        reference_row(fq, fk, fv, s, rows[r].first, rows[r].second, ref_q[r]);
        reference_row(rq, rk, rv, s, rows[r].first, rows[r].second, ref_true[r]);
    }

    // The bf16 reference: attention variant 5 on the bf16-rounded inputs.
    double bf16_ms = 0.0;
    Err bf16_err;
    if (opt.compare_bf16) {
        const int v5 = spark::attention_num_variants() - 1;
        SPARK_CUDA_CHECK(cudaMalloc(&dQb, nQ * sizeof(__nv_bfloat16)));
        SPARK_CUDA_CHECK(cudaMalloc(&dKb, nK * sizeof(__nv_bfloat16)));
        SPARK_CUDA_CHECK(cudaMalloc(&dVb, nK * sizeof(__nv_bfloat16)));
        SPARK_CUDA_CHECK(cudaMemcpy(dQb, bq.data(), nQ * 2, cudaMemcpyHostToDevice));
        SPARK_CUDA_CHECK(cudaMemcpy(dKb, bk.data(), nK * 2, cudaMemcpyHostToDevice));
        SPARK_CUDA_CHECK(cudaMemcpy(dVb, bv.data(), nK * 2, cudaMemcpyHostToDevice));
        auto run_bf16 = [&] {
            spark::attention_bf16(dQb, dKb, dVb, dO, s.B, s.Hq, s.Hkv, s.Sq, s.Skv, s.D,
                                  s.causal != 0, v5, stream);
        };
        run_bf16();
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        bf16_err = compare_rows(to_host_f32(dO, nQ), ref_true, rows, s);
        bf16_ms = spark::bench::time_kernel(run_bf16, stream, 5, opt.iters).median_ms;
    }

    double flops = 4.0 * BH * s.Sq * static_cast<double>(s.Skv) * s.D;
    if (s.causal) flops *= 0.5;
    // Q, K, V once as e4m3 (K and V once per K/V head), O once as bf16.
    const double bytes = static_cast<double>(nQ) + 2.0 * nK + 2.0 * nQ;

    bool all_ok = true;
    for (int variant : variants) {
        if (!spark::attention_fp8_supports(s.Sq, s.Skv, s.D, variant)) continue;
        auto run = [&] {
            spark::attention_fp8(dQ, dK, dV, dO, s.B, s.Hq, s.Hkv, s.Sq, s.Skv, s.D, dsq, dsk, dsv,
                                 opt.per_head != 0, s.causal != 0, variant, stream);
        };
        SPARK_CUDA_CHECK(cudaMemset(dO, 0, nQ * sizeof(__nv_bfloat16)));
        run();
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto got = to_host_f32(dO, nQ);
        const Err e = compare_rows(got, ref_q, rows, s);
        const Err et = compare_rows(got, ref_true, rows, s);
        const double tol = 4e-2 * e.max_ref + 4e-3;
        const bool ok = e.max_abs <= tol && std::isfinite(e.max_abs);
        if (!ok)
            std::fprintf(stderr,
                         "  FAIL variant %d shape %s: max_abs=%.4e tol=%.4e (%zu rows checked)\n",
                         variant, shape_str(s).c_str(), e.max_abs, tol, rows.size());

        const auto t = spark::bench::time_kernel(run, stream, 5, opt.iters);
        spark::bench::Row row;
        row.kernel = "attention_fp8";
        row.dtype = "e4m3";
        row.variant = variant;
        row.shape = shape_str(s);
        row.median_ms = t.median_ms;
        row.min_ms = t.min_ms;
        row.tflops = flops / (t.median_ms * 1e-3) / 1e12;
        row.gbps = bytes / (t.median_ms * 1e-3) / 1e9;
        row.ref_ms = bf16_ms;
        row.max_abs_err = e.max_abs;
        row.max_rel_err = e.max_rel;
        row.ok = ok;
        spark::bench::print_row(row);
        std::fprintf(stderr,
                     "    vs fp64 of the unquantized inputs: fp8 v%d max %.3e mean %.3e | bf16 "
                     "v5 max %.3e mean %.3e (max|O| %.3f, %s, %s scales)\n",
                     variant, et.max_abs, et.mean_abs, bf16_err.max_abs, bf16_err.mean_abs,
                     et.max_ref, opt.dist.c_str(), opt.per_head ? "per-head" : "per-tensor");
        all_ok = all_ok && ok;
    }

    SPARK_CUDA_CHECK(cudaFree(dQ));
    SPARK_CUDA_CHECK(cudaFree(dK));
    SPARK_CUDA_CHECK(cudaFree(dV));
    SPARK_CUDA_CHECK(cudaFree(dO));
    SPARK_CUDA_CHECK(cudaFree(dS));
    if (dQb) SPARK_CUDA_CHECK(cudaFree(dQb));
    if (dKb) SPARK_CUDA_CHECK(cudaFree(dKb));
    if (dVb) SPARK_CUDA_CHECK(cudaFree(dVb));
    return all_ok;
}

}  // namespace

int run(int argc, char** argv) {
    spark::bench::Args args(argc, argv);
    spark::bench::print_device_banner();

    Opts opt;
    opt.dist = args.get("dist", "uniform");
    SPARK_REQUIRE(opt.dist == "uniform" || opt.dist == "gauss" || opt.dist == "outlier",
                  "--dist must be uniform, gauss or outlier");
    opt.per_head = args.geti("per_head", 0);
    opt.iters = args.geti("iters", 50);
    opt.compare_bf16 = args.geti("bf16", 1);

    std::vector<Shape> shapes;
    if (args.has("b") || args.has("h") || args.has("hq") || args.has("hkv") || args.has("s") ||
        args.has("sq") || args.has("skv") || args.has("d") || args.has("causal")) {
        const int s = args.geti("s", 4096);
        const int h = args.geti("h", 32);
        const int hq = args.geti("hq", h);
        shapes.push_back({args.geti("b", 1), hq, args.geti("hkv", hq), args.geti("sq", s),
                          args.geti("skv", s), args.geti("d", 128), args.geti("causal", 0)});
    } else {
        // Small shapes checked on every row (both head sizes and masks, a length that is not a
        // multiple of the tile, a GQA group, a short query), then the prefill sweep at
        // D = 128: 32 heads from 1K to 16K tokens with and without the mask, Llama-3-8B's
        // 32 / 8 GQA heads at 4K, a batch of four 2K sequences, and a D = 64 head.
        shapes = {{1, 4, 4, 512, 512, 128, 0},       {1, 4, 4, 512, 512, 128, 1},
                  {1, 4, 4, 512, 512, 64, 0},        {1, 4, 4, 512, 512, 64, 1},
                  {1, 4, 4, 200, 200, 128, 1},       {1, 8, 2, 512, 512, 128, 1},
                  {1, 4, 4, 1, 700, 128, 0},         {1, 32, 32, 1024, 1024, 128, 0},
                  {1, 32, 32, 1024, 1024, 128, 1},   {1, 32, 32, 2048, 2048, 128, 0},
                  {1, 32, 32, 2048, 2048, 128, 1},   {1, 32, 32, 4096, 4096, 128, 0},
                  {1, 32, 32, 4096, 4096, 128, 1},   {1, 32, 32, 8192, 8192, 128, 0},
                  {1, 32, 32, 8192, 8192, 128, 1},   {1, 32, 32, 16384, 16384, 128, 0},
                  {1, 32, 32, 16384, 16384, 128, 1}, {1, 32, 8, 4096, 4096, 128, 0},
                  {1, 32, 8, 4096, 4096, 128, 1},    {4, 32, 32, 2048, 2048, 128, 1},
                  {1, 32, 32, 4096, 4096, 64, 1}};
    }
    std::vector<int> variants;
    if (args.has("variant")) {
        variants.push_back(args.geti("variant", 0));
    } else {
        for (int v = 0; v < spark::attention_fp8_num_variants(); ++v) variants.push_back(v);
    }

    cudaStream_t stream = nullptr;
    spark::bench::print_header();
    bool all_ok = true;
    for (const auto& s : shapes) all_ok = run_shape(stream, s, variants, opt) && all_ok;
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
        std::fprintf(stderr, "bench_attention_fp8: %s\n", e.what());
        return 2;
    }
}
