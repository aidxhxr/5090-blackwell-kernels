// Benchmark + correctness check for the attention backward ladder.
//
//   ./bench_attention_bwd                                  # default shape sweep, all variants
//   ./bench_attention_bwd --b=1 --h=32 --s=4096 --d=128 --causal=1 --variant=2 --iters=20
//   ./bench_attention_bwd --hq=32 --hkv=8 ...              # grouped-query attention
//   ./bench_attention_bwd --deterministic=1 ...            # time the deterministic dQ pass
//
// Setup per shape: Q, K uniform in [-2, 2], V and dO in [-1, 1] (bf16), then the forward
// (variant 5, the default rung) with its log-sum-exp output gives O and L. The rows time the
// backward alone: attention_bwd_bf16 from (Q, K, V, O, dO, L) to (dQ, dK, dV), preprocess and
// dQ conversion included.
//
// Correctness, against a CPU double-precision reference from the same bf16 inputs, the GPU's
// O (bf16) and the GPU's L:
//   * L itself, on the checked query rows, against the reference log-sum-exp of the row;
//   * dQ rows and dK / dV rows: every row for the small shapes, 32 sampled rows of each for
//     the large ones (always including the first and last rows of the first head, the two
//     ends of the causal mask).
// Tolerance: max|x - x_ref| <= 0.02 * max|x_ref| + 1e-3 per tensor over the rows checked (the
// kernels round P and dS to bf16 before their products, and the outputs to bf16). Variants 1
// and 2 are also run in deterministic mode once per shape: checked the same way, and two
// launches must agree bit for bit.
//
// FLOPs: the five products of S_q x S_kv x D, 2.5x the forward's count,
// 10 * B * H_q * S_q * S_kv * D, halved under the causal mask (FlashAttention's convention).
// GB/s: Q, O, dO read and dQ written (H_q heads), K, V read and dK, dV written (H_kv heads).
//
// stdout: one JSON object per row (collected by scripts/run_all_benches.sh)
// stderr: human-readable table

#include <cuda_bf16.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "spark/kernels.h"

namespace {

using bf16 = __nv_bfloat16;

struct Shape {
    int B, Hq, Hkv, Sq, Skv, D;
    int causal;
};

// Same spelling as bench_attention (scripts/shape_utils.py parses both).
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

std::vector<float> to_host(const bf16* d, size_t n) {
    std::vector<bf16> h(n);
    SPARK_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(bf16), cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = __bfloat162float(h[i]);
    return out;
}

std::vector<bf16> upload_rounded(std::vector<float>& h, bf16** dptr) {
    std::vector<bf16> b(h.size());
    for (size_t i = 0; i < h.size(); ++i) {
        b[i] = __float2bfloat16(h[i]);
        h[i] = __bfloat162float(b[i]);  // the reference sees exactly what the kernel sees
    }
    SPARK_CUDA_CHECK(cudaMalloc(dptr, b.size() * sizeof(bf16)));
    SPARK_CUDA_CHECK(cudaMemcpy(*dptr, b.data(), b.size() * sizeof(bf16), cudaMemcpyHostToDevice));
    return b;
}

// Everything the reference needs, on the host.
struct Host {
    Shape s;
    std::vector<float> q, k, v, o, dout, lse;
    double scale = 0;
    int group = 1;
    size_t qrow(int bh, int i) const { return (static_cast<size_t>(bh) * s.Sq + i) * s.D; }
    size_t krow(int bkv, int j) const { return (static_cast<size_t>(bkv) * s.Skv + j) * s.D; }
    int bkv_of(int bh) const { return (bh / s.Hq) * s.Hkv + (bh % s.Hq) / group; }
    double dot(const float* a, const float* b) const {
        double acc = 0.0;
        for (int d = 0; d < s.D; ++d) acc += static_cast<double>(a[d]) * b[d];
        return acc;
    }
    bool visible(int i, int j) const { return !s.causal || j <= i; }
    // Dv_i = dO_i . O_i from the GPU's bf16 O, as the kernels compute it.
    double dvec(int bh, int i) const {
        return dot(dout.data() + qrow(bh, i), o.data() + qrow(bh, i));
    }
};

// Reference log-sum-exp of one query row.
double ref_lse(const Host& h, int bh, int i) {
    const int bkv = h.bkv_of(bh);
    double mx = -INFINITY;
    std::vector<double> sc;
    for (int j = 0; j < h.s.Skv; ++j) {
        if (!h.visible(i, j)) break;
        sc.push_back(h.scale * h.dot(h.q.data() + h.qrow(bh, i), h.k.data() + h.krow(bkv, j)));
        mx = std::max(mx, sc.back());
    }
    double sum = 0.0;
    for (double x : sc) sum += std::exp(x - mx);
    return mx + std::log(sum);
}

// dQ row i of head bh.
void ref_dq(const Host& h, int bh, int i, std::vector<double>& out) {
    const int D = h.s.D, bkv = h.bkv_of(bh);
    out.assign(D, 0.0);
    const float* q = h.q.data() + h.qrow(bh, i);
    const float* d = h.dout.data() + h.qrow(bh, i);
    const double L = h.lse[static_cast<size_t>(bh) * h.s.Sq + i], Dv = h.dvec(bh, i);
    for (int j = 0; j < h.s.Skv && h.visible(i, j); ++j) {
        const float* k = h.k.data() + h.krow(bkv, j);
        const double p = std::exp(h.scale * h.dot(q, k) - L);
        const double ds = p * (h.dot(d, h.v.data() + h.krow(bkv, j)) - Dv);
        for (int e = 0; e < D; ++e) out[e] += ds * k[e];
    }
    for (int e = 0; e < D; ++e) out[e] *= h.scale;
}

// dK and dV row j of K/V head bkv, over every query head of its group.
void ref_dkv(const Host& h, int bkv, int j, const std::vector<double>& dv_all,
             std::vector<double>& dk, std::vector<double>& dvv) {
    const int D = h.s.D;
    dk.assign(D, 0.0);
    dvv.assign(D, 0.0);
    const float* k = h.k.data() + h.krow(bkv, j);
    const float* v = h.v.data() + h.krow(bkv, j);
    const int b = bkv / h.s.Hkv, kvh = bkv % h.s.Hkv;
    for (int hg = 0; hg < h.group; ++hg) {
        const int bh = b * h.s.Hq + kvh * h.group + hg;
        for (int i = h.s.causal ? j : 0; i < h.s.Sq; ++i) {
            const float* q = h.q.data() + h.qrow(bh, i);
            const float* d = h.dout.data() + h.qrow(bh, i);
            const size_t r = static_cast<size_t>(bh) * h.s.Sq + i;
            const double p = std::exp(h.scale * h.dot(q, k) - h.lse[r]);
            const double ds = p * (h.dot(d, v) - dv_all[r]);
            for (int e = 0; e < D; ++e) {
                dvv[e] += p * d[e];
                dk[e] += ds * q[e];
            }
        }
    }
    for (int e = 0; e < D; ++e) dk[e] *= h.scale;
}

struct Check {
    double max_err = 0, max_ref = 0;
    void add(double got, double ref) {
        max_err = std::max(max_err, std::fabs(got - ref));
        max_ref = std::max(max_ref, std::fabs(ref));
    }
    double tol() const { return 2e-2 * max_ref + 1e-3; }
    bool ok() const { return std::isfinite(max_err) && max_err <= tol(); }
    double ratio() const { return max_err / tol(); }
};

struct Result {
    bool ok = true;
    double worst_err = 0, worst_rel = 0;
};

// Compares the GPU gradients with the reference on the chosen rows.
Result validate(const Host& h, const std::vector<float>& gq, const std::vector<float>& gk,
                const std::vector<float>& gv, const std::vector<std::pair<int, int>>& qrows,
                const std::vector<std::pair<int, int>>& krows, const std::vector<double>& dv_all,
                const char* what, int variant) {
    Check cq, ck, cv;
    std::vector<double> r1, r2;
    for (const auto& [bh, i] : qrows) {
        ref_dq(h, bh, i, r1);
        const float* g = gq.data() + h.qrow(bh, i);
        for (int e = 0; e < h.s.D; ++e) cq.add(g[e], r1[e]);
    }
    for (const auto& [bkv, j] : krows) {
        ref_dkv(h, bkv, j, dv_all, r1, r2);
        const float* a = gk.data() + h.krow(bkv, j);
        const float* b = gv.data() + h.krow(bkv, j);
        for (int e = 0; e < h.s.D; ++e) {
            ck.add(a[e], r1[e]);
            cv.add(b[e], r2[e]);
        }
    }
    Result r;
    r.ok = cq.ok() && ck.ok() && cv.ok();
    for (const Check* c : {&cq, &ck, &cv}) {
        if (c->ratio() > r.worst_rel || !std::isfinite(c->max_err)) {
            r.worst_rel = c->ratio();
            r.worst_err = c->max_err;
        }
    }
    if (!r.ok) {
        std::fprintf(stderr,
                     "  FAIL %s variant %d shape %s: dQ %.3e (tol %.3e), dK %.3e (tol %.3e), "
                     "dV %.3e (tol %.3e)\n",
                     what, variant, shape_str(h.s).c_str(), cq.max_err, cq.tol(), ck.max_err,
                     ck.tol(), cv.max_err, cv.tol());
    }
    return r;
}

bool run_one(cudaStream_t stream, const Shape& s, int variant, int iters, bool deterministic) {
    if (!spark::attention_bwd_supports(s.Sq, s.Skv, s.D, variant)) {
        std::fprintf(stderr, "  skip variant %d shape %s: not supported\n", variant,
                     shape_str(s).c_str());
        return true;
    }
    Host h;
    h.s = s;
    h.group = s.Hq / s.Hkv;
    h.scale = 1.0 / std::sqrt(static_cast<double>(s.D));
    const size_t BH = static_cast<size_t>(s.B) * s.Hq;
    const size_t BHkv = static_cast<size_t>(s.B) * s.Hkv;
    const size_t nQ = BH * s.Sq * s.D, nK = BHkv * s.Skv * s.D;

    h.q.resize(nQ);
    h.k.resize(nK);
    h.v.resize(nK);
    h.dout.resize(nQ);
    spark::bench::fill_uniform(h.q, -2.0f, 2.0f, 1234);
    spark::bench::fill_uniform(h.k, -2.0f, 2.0f, 5678);
    spark::bench::fill_uniform(h.v, -1.0f, 1.0f, 9012);
    spark::bench::fill_uniform(h.dout, -1.0f, 1.0f, 3456);
    bf16 *dq_in, *dk_in, *dv_in, *ddo;
    upload_rounded(h.q, &dq_in);
    upload_rounded(h.k, &dk_in);
    upload_rounded(h.v, &dv_in);
    upload_rounded(h.dout, &ddo);
    bf16 *d_o, *gQ, *gK, *gV;
    float* d_lse;
    SPARK_CUDA_CHECK(cudaMalloc(&d_o, nQ * sizeof(bf16)));
    SPARK_CUDA_CHECK(cudaMalloc(&gQ, nQ * sizeof(bf16)));
    SPARK_CUDA_CHECK(cudaMalloc(&gK, nK * sizeof(bf16)));
    SPARK_CUDA_CHECK(cudaMalloc(&gV, nK * sizeof(bf16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_lse, BH * s.Sq * sizeof(float)));

    spark::attention_bf16(dq_in, dk_in, dv_in, d_o, s.B, s.Hq, s.Hkv, s.Sq, s.Skv, s.D,
                          s.causal != 0, spark::attention_num_variants() - 1, stream, d_lse);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    h.o = to_host(d_o, nQ);
    h.lse.resize(BH * s.Sq);
    SPARK_CUDA_CHECK(
        cudaMemcpy(h.lse.data(), d_lse, h.lse.size() * sizeof(float), cudaMemcpyDeviceToHost));

    auto bwd = [&](int v, bool det) {
        spark::attention_bwd_bf16(dq_in, dk_in, dv_in, d_o, ddo, d_lse, gQ, gK, gV, s.B, s.Hq,
                                  s.Hkv, s.Sq, s.Skv, s.D, s.causal != 0, det, v, stream);
    };
    bwd(variant, deterministic);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));

    // Rows to check.
    const double ref_work = static_cast<double>(BH) * s.Sq * s.Skv * s.D;
    std::vector<std::pair<int, int>> qrows, krows;
    if (ref_work <= 2.0e9) {
        for (size_t bh = 0; bh < BH; ++bh)
            for (int i = 0; i < s.Sq; ++i) qrows.emplace_back(static_cast<int>(bh), i);
        for (size_t bkv = 0; bkv < BHkv; ++bkv)
            for (int j = 0; j < s.Skv; ++j) krows.emplace_back(static_cast<int>(bkv), j);
    } else {
        std::mt19937 rng(7);
        qrows = {{0, 0}, {0, s.Sq - 1}};
        krows = {{0, 0}, {0, s.Skv - 1}};
        for (int n = 0; n < 30; ++n) {
            qrows.emplace_back(static_cast<int>(rng() % BH), static_cast<int>(rng() % s.Sq));
            krows.emplace_back(static_cast<int>(rng() % BHkv), static_cast<int>(rng() % s.Skv));
        }
    }
    // Dv for every row a checked dK / dV row reads, and the forward's L on the dQ rows.
    std::vector<double> dv_all(BH * s.Sq, 0.0);
    {
        std::vector<char> need(BH, 0);
        for (const auto& kr : krows) {
            const int b = kr.first / s.Hkv, kvh = kr.first % s.Hkv;
            for (int hg = 0; hg < h.group; ++hg) need[b * s.Hq + kvh * h.group + hg] = 1;
        }
        for (size_t bh = 0; bh < BH; ++bh)
            if (need[bh])
                for (int i = 0; i < s.Sq; ++i)
                    dv_all[bh * s.Sq + i] = h.dvec(static_cast<int>(bh), i);
    }
    bool ok = true;
    {
        double lse_err = 0;
        for (const auto& [bh, i] : qrows)
            lse_err = std::max(
                lse_err, std::fabs(h.lse[static_cast<size_t>(bh) * s.Sq + i] - ref_lse(h, bh, i)));
        if (!(lse_err <= 1e-3)) {
            std::fprintf(stderr, "  FAIL forward log-sum-exp, shape %s: max error %.3e\n",
                         shape_str(s).c_str(), lse_err);
            ok = false;
        }
    }
    const Result res = validate(h, to_host(gQ, nQ), to_host(gK, nK), to_host(gV, nK), qrows, krows,
                                dv_all, deterministic ? "deterministic" : "atomic", variant);
    ok = ok && res.ok;

    // The other dQ mode once, for its correctness and (deterministic) reproducibility.
    if (variant >= 1 && !deterministic) {
        bwd(variant, true);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto a_q = to_host(gQ, nQ), a_k = to_host(gK, nK), a_v = to_host(gV, nK);
        const Result rd =
            validate(h, a_q, a_k, a_v, qrows, krows, dv_all, "deterministic", variant);
        bwd(variant, true);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto b_q = to_host(gQ, nQ), b_k = to_host(gK, nK), b_v = to_host(gV, nK);
        const bool same = std::memcmp(a_q.data(), b_q.data(), nQ * sizeof(float)) == 0 &&
                          std::memcmp(a_k.data(), b_k.data(), nK * sizeof(float)) == 0 &&
                          std::memcmp(a_v.data(), b_v.data(), nK * sizeof(float)) == 0;
        if (!same)
            std::fprintf(stderr, "  FAIL deterministic variant %d shape %s: two runs differ\n",
                         variant, shape_str(s).c_str());
        ok = ok && rd.ok && same;
    }

    const auto t =
        spark::bench::time_kernel([&] { bwd(variant, deterministic); }, stream, 5, iters);

    double flops = 10.0 * BH * s.Sq * static_cast<double>(s.Skv) * s.D;
    if (s.causal) flops *= 0.5;
    const double bytes = 2.0 * (4.0 * nQ + 4.0 * nK);
    spark::bench::Row row;
    row.kernel = "attention_bwd";
    row.dtype = "bf16";
    row.variant = variant;
    row.shape = shape_str(s);
    row.median_ms = t.median_ms;
    row.min_ms = t.min_ms;
    row.tflops = flops / (t.median_ms * 1e-3) / 1e12;
    row.gbps = bytes / (t.median_ms * 1e-3) / 1e9;
    row.ref_ms = 0.0;
    row.max_abs_err = res.worst_err;
    row.max_rel_err = res.worst_rel;
    row.ok = ok;
    spark::bench::print_row(row);

    for (void* p : {static_cast<void*>(dq_in), static_cast<void*>(dk_in), static_cast<void*>(dv_in),
                    static_cast<void*>(ddo), static_cast<void*>(d_o), static_cast<void*>(gQ),
                    static_cast<void*>(gK), static_cast<void*>(gV), static_cast<void*>(d_lse)})
        SPARK_CUDA_CHECK(cudaFree(p));
    return ok;
}

}  // namespace

int run(int argc, char** argv) {
    spark::bench::Args args(argc, argv);
    spark::bench::print_device_banner();

    std::vector<Shape> shapes;
    if (args.has("b") || args.has("h") || args.has("hq") || args.has("hkv") || args.has("s") ||
        args.has("sq") || args.has("skv") || args.has("d") || args.has("causal")) {
        const int s = args.geti("s", 4096);
        const int h = args.geti("h", 32);
        const int hq = args.geti("hq", h);
        shapes.push_back({args.geti("b", 1), hq, args.geti("hkv", hq), args.geti("sq", s),
                          args.geti("skv", s), args.geti("d", 128), args.geti("causal", 0)});
    } else {
        // Small shapes, checked on every row: both head sizes, both masks, a length that is not
        // a multiple of any tile, S_q != S_kv, and GQA groups of 4. Then the forward bench's
        // prefill shapes (H = 32, D = 128) from 1K to 8K tokens with and without the mask, a
        // batch of four 2048-token sequences, D = 64, and Llama-3-8B's 32 / 8 heads.
        shapes = {{1, 4, 4, 512, 512, 128, 0},     {1, 4, 4, 512, 512, 128, 1},
                  {1, 4, 4, 512, 512, 64, 0},      {1, 4, 4, 512, 512, 64, 1},
                  {1, 4, 4, 200, 200, 128, 1},     {1, 3, 3, 150, 333, 64, 0},
                  {2, 8, 2, 300, 300, 128, 1},     {1, 8, 2, 512, 512, 64, 0},
                  {1, 32, 32, 1024, 1024, 128, 0}, {1, 32, 32, 1024, 1024, 128, 1},
                  {1, 32, 32, 2048, 2048, 128, 0}, {1, 32, 32, 2048, 2048, 128, 1},
                  {1, 32, 32, 4096, 4096, 128, 0}, {1, 32, 32, 4096, 4096, 128, 1},
                  {1, 32, 32, 8192, 8192, 128, 0}, {1, 32, 32, 8192, 8192, 128, 1},
                  {4, 32, 32, 2048, 2048, 128, 1}, {1, 32, 32, 4096, 4096, 64, 1},
                  {1, 32, 8, 4096, 4096, 128, 0},  {1, 32, 8, 4096, 4096, 128, 1},
                  {1, 32, 8, 8192, 8192, 128, 1}};
    }
    const int iters = args.geti("iters", 50);
    const bool deterministic = args.geti("deterministic", 0) != 0;
    std::vector<int> variants;
    if (args.has("variant")) {
        variants.push_back(args.geti("variant", 0));
    } else {
        for (int v = 0; v < spark::attention_bwd_num_variants(); ++v) variants.push_back(v);
    }

    cudaStream_t stream = nullptr;
    spark::bench::print_header();
    bool all_ok = true;
    for (const auto& s : shapes) {
        for (int v : variants) all_ok = run_one(stream, s, v, iters, deterministic) && all_ok;
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
        std::fprintf(stderr, "bench_attention_bwd: %s\n", e.what());
        return 2;
    }
}
