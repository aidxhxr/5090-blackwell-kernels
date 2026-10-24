// Benchmark + correctness check for the fused attention ladder.
//
//   ./bench_attention                                   # default shape sweep, all variants
//   ./bench_attention --b=1 --h=32 --s=4096 --d=128 --causal=1 --variant=2 --iters=20
//   ./bench_attention --sq=1 --skv=4096 ...             # a decode step (S_q != S_kv)
//   ./bench_attention --hq=32 --hkv=8 --sq=1 --skv=131072   # grouped-query attention
//   ./bench_attention --sq=1 --skv=4096 --kvcap=4352       # K/V as the first 4096 rows of
//                                                          # heads allocated 4352 (a cache)
//   ./bench_attention --hq=32 --hkv=8 --s=16384 --causal=1 --window=1024 --sinks=1
//                                                       # a sliding window and sink logits
//   ./bench_attention --sweep=window                    # the window sweep (attention_window
//                                                       # rows, results/attention_window.json)
//
// Correctness: a CPU double-precision reference on every output row for the small shapes,
// and on 64 sampled (b, h, row) triples for the large ones (the full reference at S = 4096
// takes minutes). There is no cuBLAS attention, so ref_ms stays 0; the library comparison is
// PyTorch's scaled_dot_product_attention in scripts/bench_torch.py.
//
// GB/s is the traffic floor: Q and O once (H_q heads), K and V once (H_kv heads). Under GQA
// the K/V bytes scale with H_kv, so the number says how close the kernel gets to reading
// each K/V head once for the whole group of query heads.
//
// stdout: one JSON object per row (collected by scripts/run_all_benches.sh)
// stderr: human-readable table

#include <cuda_bf16.h>

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
    int kvcap = 0;   // rows each K/V head is allocated (0: Skv, packed); the kernel's kv_cap
    int window = 0;  // keys each query sees, its own included (0: no window); causal, the
                     // query at key position i + Skv - Sq (attention_window_bf16)
    int sinks = 0;   // a sink logit per query head
};

// "b1_h32_s4096_d128_causal", or "b1_h32_sq1_skv4096_d128" when S_q != S_kv, and
// "b1_hq32_hkv8_..." when the K/V head count differs from the query head count (GQA). Parsed
// back by scripts/shape_utils.py, so the two must agree.
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
    if (s.kvcap) r += "_cap" + std::to_string(s.kvcap);
    if (s.window) r += "_w" + std::to_string(s.window);
    if (s.sinks) r += "_sink";
    return r;
}

// Keys [lo, hi) query row i sees: the causal bound (top-left, or at key position
// i + Skv - Sq under a window) and the window's left edge.
void key_range(const Shape& s, int i, int& lo, int& hi) {
    const int pos = s.window ? i + s.Skv - s.Sq : i;
    hi = s.causal || s.window ? std::min(s.Skv, pos + 1) : s.Skv;
    lo = s.window ? std::max(0, pos - s.window + 1) : 0;
}

std::vector<float> to_host_f32(const __nv_bfloat16* d, size_t n) {
    std::vector<__nv_bfloat16> h(n);
    SPARK_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = __bfloat162float(h[i]);
    return out;
}

// One output row in double precision from the bf16-rounded inputs (hq/hk/hv are those values
// as floats). Softmax as exp(s - max) / sum, all D columns of the row. Query head h of batch
// b reads K/V head h / (H_q / H_kv). `hs`, when not empty, holds a sink logit per query head,
// one more term of the denominator.
void reference_row(const std::vector<float>& hq, const std::vector<float>& hk,
                   const std::vector<float>& hv, const std::vector<float>& hs, const Shape& s,
                   int bh, int i, std::vector<double>& out) {
    const int D = s.D;
    const int b = bh / s.Hq, h = bh % s.Hq;
    const int bkv = b * s.Hkv + h / (s.Hq / s.Hkv);
    const float* q = hq.data() + (static_cast<size_t>(bh) * s.Sq + i) * D;
    const float* k = hk.data() + static_cast<size_t>(bkv) * s.Skv * D;
    const float* v = hv.data() + static_cast<size_t>(bkv) * s.Skv * D;
    int lo, n;
    key_range(s, i, lo, n);
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    std::vector<double> sc(n);
    double mx = hs.empty() ? -INFINITY : static_cast<double>(hs[h]);
    for (int j = lo; j < n; ++j) {
        double acc = 0.0;
        for (int d = 0; d < D; ++d)
            acc += static_cast<double>(q[d]) * k[static_cast<size_t>(j) * D + d];
        sc[j] = acc * scale;
        mx = std::max(mx, sc[j]);
    }
    out.assign(D, 0.0);
    double sum = hs.empty() ? 0.0 : std::exp(static_cast<double>(hs[h]) - mx);
    for (int j = lo; j < n; ++j) {
        const double p = std::exp(sc[j] - mx);
        sum += p;
        for (int d = 0; d < D; ++d) out[d] += p * v[static_cast<size_t>(j) * D + d];
    }
    for (int d = 0; d < D; ++d) out[d] /= sum;
}

// Runs one shape for one variant. Returns false on a correctness failure. The median time goes
// to `median_ms` when given; `ref_ms` is printed as the row's reference (the window sweep
// passes the full causal time of the same shape), and `kernel` names the row when given (the
// sweep's full causal rows are "attention_window" rows too, apart from the default sweep's).
bool run_one(cudaStream_t stream, const Shape& s, int variant, int iters,
             double* median_ms = nullptr, double ref_ms = 0.0, const char* kernel = nullptr) {
    if (!spark::attention_supports(s.Sq, s.Skv, s.D, variant)) {
        std::fprintf(stderr, "  skip variant %d shape %s: not supported\n", variant,
                     shape_str(s).c_str());
        return true;
    }
    const size_t BH = static_cast<size_t>(s.B) * s.Hq;
    const size_t BHkv = static_cast<size_t>(s.B) * s.Hkv;
    const size_t nQ = BH * s.Sq * s.D, nK = BHkv * s.Skv * s.D;
    // With --kvcap the device K and V hold `cap` rows per head and the kernel reads the
    // first Skv of each; the host copies stay packed (the reference does not care).
    const int cap = s.kvcap > 0 ? s.kvcap : s.Skv;
    const size_t nKd = BHkv * cap * s.D;

    // Q and K wider than V so the scores have a spread of a few units after the 1/sqrt(D)
    // scale: a near-uniform softmax would not exercise the running max and the rescale.
    std::vector<float> hq(nQ), hk(nK), hv(nK);
    spark::bench::fill_uniform(hq, -2.0f, 2.0f, 1234);
    spark::bench::fill_uniform(hk, -2.0f, 2.0f, 5678);
    spark::bench::fill_uniform(hv, -1.0f, 1.0f, 9012);
    std::vector<__nv_bfloat16> bq(nQ), bk(nK), bv(nK);
    for (size_t i = 0; i < nQ; ++i) {
        bq[i] = __float2bfloat16(hq[i]);
        hq[i] = __bfloat162float(bq[i]);  // the reference sees exactly what the kernel sees
    }
    for (size_t i = 0; i < nK; ++i) {
        bk[i] = __float2bfloat16(hk[i]);
        hk[i] = __bfloat162float(bk[i]);
        bv[i] = __float2bfloat16(hv[i]);
        hv[i] = __bfloat162float(bv[i]);
    }

    // A decode step (S_q <= 64) is bound by streaming K and V, and one layer's cache (64 MB
    // at H = 32, S_kv = 4096, D = 128) fits the RTX 5090's 96 MB L2, so timing one copy back
    // to back would measure L2. A real step reads every layer's cache once per token, so the
    // timing loop rotates through enough copies of K and V to exceed L2 (256 MB+), as
    // bench_hgemm does with the weights of a decode GEMM. A 128K-token cache is past L2 on
    // its own (512 MB at 8 heads, 2 GB at 32) and gets one copy.
    //
    // A windowed decode step reads only the last rows of each head (4096 of a 128K cache is
    // 16.8 MB at 8 heads), which L2 would hold between launches, and whole copies of the
    // cache to rotate through would not fit the card. So the copies are shifts along each
    // head instead: one [B * H_kv, cap + (copies - 1) * rows, D] allocation, copy c is the
    // tensor starting c * rows rows in, with the head stride of the allocation as its kv_cap,
    // and the rows of its window hold the window's data. Each launch reads its own rows.
    int lo_first, hi_first;
    key_range(s, 0, lo_first, hi_first);
    const int rows_read = s.window ? s.Skv - lo_first : s.Skv;  // per K/V head
    const bool shifted = s.window > 0 && s.Sq <= 64;
    int copies;
    int cap_alloc = cap;  // rows per head of the allocation
    size_t copy_stride;   // elements between copies
    size_t alloc_elems;
    if (shifted) {
        const size_t win_bytes = BHkv * rows_read * s.D * sizeof(__nv_bfloat16);
        copies = static_cast<int>((size_t{256} << 20) / (2 * win_bytes)) + 1;
        cap_alloc = cap + (copies - 1) * rows_read;
        copy_stride = static_cast<size_t>(rows_read) * s.D;
        alloc_elems = BHkv * cap_alloc * s.D;
    } else {
        const size_t kv_bytes = nKd * sizeof(__nv_bfloat16);
        copies = s.Sq <= 64 ? static_cast<int>((size_t{256} << 20) / (2 * kv_bytes)) + 1 : 1;
        copy_stride = nKd;
        alloc_elems = copies * nKd;
    }
    const int kv_cap_arg = shifted ? cap_alloc : s.kvcap;
    const size_t kv_bytes = BHkv * cap_alloc * s.D * sizeof(__nv_bfloat16);  // one copy's span

    __nv_bfloat16 *dQ = nullptr, *dK = nullptr, *dV = nullptr, *dO = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&dQ, nQ * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dK, alloc_elems * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dV, alloc_elems * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&dO, nQ * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMemcpy(dQ, bq.data(), nQ * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    // the rows past Skv of every head hold garbage the kernel must not read
    SPARK_CUDA_CHECK(cudaMemset(dK, 0xff, kv_bytes));
    SPARK_CUDA_CHECK(cudaMemset(dV, 0xff, kv_bytes));
    for (size_t h = 0; h < BHkv; ++h) {
        const size_t rows = static_cast<size_t>(s.Skv) * s.D;
        SPARK_CUDA_CHECK(cudaMemcpy(dK + h * cap_alloc * s.D, bk.data() + h * rows,
                                    rows * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        SPARK_CUDA_CHECK(cudaMemcpy(dV + h * cap_alloc * s.D, bv.data() + h * rows,
                                    rows * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    }
    for (int c = 1; c < copies; ++c) {
        if (shifted) {  // the window's rows of every head, c * rows_read rows further on
            const size_t win = static_cast<size_t>(rows_read) * s.D;
            for (size_t h = 0; h < BHkv; ++h) {
                const size_t src = h * cap_alloc * s.D + static_cast<size_t>(lo_first) * s.D;
                SPARK_CUDA_CHECK(cudaMemcpy(dK + src + c * copy_stride, dK + src,
                                            win * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
                SPARK_CUDA_CHECK(cudaMemcpy(dV + src + c * copy_stride, dV + src,
                                            win * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
            }
            continue;
        }
        SPARK_CUDA_CHECK(cudaMemcpy(dK + c * nKd, dK, kv_bytes, cudaMemcpyDeviceToDevice));
        SPARK_CUDA_CHECK(cudaMemcpy(dV + c * nKd, dV, kv_bytes, cudaMemcpyDeviceToDevice));
    }
    SPARK_CUDA_CHECK(cudaMemset(dO, 0, nQ * sizeof(__nv_bfloat16)));

    // Sink logits in [0, 3]: the scores spread over a few units, so a sink takes a visible
    // share of the denominator without swamping it.
    std::vector<float> hs;
    float* dS = nullptr;
    if (s.sinks) {
        hs.resize(s.Hq);
        spark::bench::fill_uniform(hs, 0.0f, 3.0f, 3456);
        SPARK_CUDA_CHECK(cudaMalloc(&dS, s.Hq * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMemcpy(dS, hs.data(), s.Hq * sizeof(float), cudaMemcpyHostToDevice));
    }
    const bool local = s.window > 0 || s.sinks;
    auto launch = [&](const __nv_bfloat16* k, const __nv_bfloat16* v) {
        if (local)
            spark::attention_window_bf16(dQ, k, v, dO, s.B, s.Hq, s.Hkv, s.Sq, s.Skv, s.D,
                                         s.causal != 0, s.window, dS, variant, stream, nullptr,
                                         kv_cap_arg);
        else
            spark::attention_bf16(dQ, k, v, dO, s.B, s.Hq, s.Hkv, s.Sq, s.Skv, s.D, s.causal != 0,
                                  variant, stream, nullptr, kv_cap_arg);
    };
    launch(dK, dV);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto got = to_host_f32(dO, nQ);

    // Which rows to check: all of them while the reference is cheap, else 64 sampled rows
    // (always including the first and last query row of the first head, where the causal
    // mask is at its two extremes).
    const double ref_work = static_cast<double>(BH) * s.Sq * s.Skv * s.D;
    std::vector<std::pair<int, int>> rows;  // (bh, i)
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
    // Tolerance: the kernel's output is one bf16 rounding of an fp32 result (relative step
    // 2^-8), and variant 2 also rounds P to bf16 before the P V product, so the check is
    // max|O - O_ref| <= 0.02 * max|O_ref| + 1e-3 over the rows checked, like bench_hgemm.
    double max_abs = 0.0, max_rel = 0.0, max_ref = 0.0;
    std::vector<double> ref;
    for (const auto& [bh, i] : rows) {
        reference_row(hq, hk, hv, hs, s, bh, i, ref);
        const float* o = got.data() + (static_cast<size_t>(bh) * s.Sq + i) * s.D;
        for (int d = 0; d < s.D; ++d) {
            const double diff = std::fabs(static_cast<double>(o[d]) - ref[d]);
            max_abs = std::max(max_abs, diff);
            max_rel = std::max(max_rel, diff / std::max(1e-6, std::fabs(ref[d])));
            max_ref = std::max(max_ref, std::fabs(ref[d]));
        }
    }
    const double tol = 2e-2 * max_ref + 1e-3;
    const bool ok = max_abs <= tol && std::isfinite(max_abs);
    if (!ok) {
        std::fprintf(stderr,
                     "  FAIL variant %d shape %s: max_abs=%.4e tol=%.4e (%zu rows checked)\n",
                     variant, shape_str(s).c_str(), max_abs, tol, rows.size());
    }

    // Timing. Each launch takes the next copy of K and V (see `copies` above).
    int turn = 0;
    const auto t = spark::bench::time_kernel(
        [&] {
            const size_t off = static_cast<size_t>(turn++ % copies) * copy_stride;
            launch(dK + off, dV + off);
        },
        stream, 5, iters);

    // 4 * B * H_q * S_q * S_kv * D, halved under the causal mask (the FlashAttention
    // convention: only the tiles at or below the diagonal are computed). Under a window,
    // 4 * D per (query, visible key) pair, counted exactly.
    double flops = 4.0 * BH * s.Sq * static_cast<double>(s.Skv) * s.D;
    if (s.causal) flops *= 0.5;
    // Q and O once (H_q heads), K and V once (H_kv heads): the GQA traffic floor. Under a
    // window only the keys some query's window holds.
    double bytes = 2.0 * (2.0 * nQ + 2.0 * nK);
    if (s.window) {
        double pairs = 0.0;
        int lo0, hi0, lo1, hi1;
        for (int i = 0; i < s.Sq; ++i) {
            key_range(s, i, lo0, hi0);
            pairs += hi0 - lo0;
        }
        flops = 4.0 * BH * pairs * s.D;
        key_range(s, 0, lo0, hi0);
        key_range(s, s.Sq - 1, lo1, hi1);
        bytes = 2.0 * (2.0 * nQ + 2.0 * BHkv * static_cast<double>(hi1 - lo0) * s.D);
    }
    spark::bench::Row row;
    row.kernel = kernel ? kernel : local ? "attention_window" : "attention";
    row.dtype = "bf16";
    row.variant = variant;
    row.shape = shape_str(s);
    row.median_ms = t.median_ms;
    row.min_ms = t.min_ms;
    row.tflops = flops / (t.median_ms * 1e-3) / 1e12;
    row.gbps = bytes / (t.median_ms * 1e-3) / 1e9;
    row.ref_ms = ref_ms;
    row.max_abs_err = max_abs;
    row.max_rel_err = max_rel;
    row.ok = ok;
    spark::bench::print_row(row);
    if (copies > 1)
        std::fprintf(stderr, "    K and V rotated over %d copies that exceed L2 (DRAM-bound)\n",
                     copies);

    SPARK_CUDA_CHECK(cudaFree(dQ));
    SPARK_CUDA_CHECK(cudaFree(dK));
    SPARK_CUDA_CHECK(cudaFree(dV));
    SPARK_CUDA_CHECK(cudaFree(dO));
    if (dS) SPARK_CUDA_CHECK(cudaFree(dS));
    if (median_ms) *median_ms = t.median_ms;
    return ok;
}

// --sweep=window: sliding windows against the full causal kernel on the same shapes, every
// windowed row with the full row's time as its ref_ms. Llama-3-8B heads (32 query, 8 K/V,
// D = 128) at 4K, 16K and 32K tokens with windows of 128 (gpt-oss), 1024 and 4096 (Mistral,
// Gemma); the sink logits on top at 4K and 16K; gpt-oss's own heads (64 query, 8 K/V, D = 64,
// window 128, sinks); and decode steps against 128K cached keys with a 4096-key window and
// gpt-oss's 128.
bool window_sweep(cudaStream_t stream, int variant, int iters) {
    bool ok = true;
    auto pair = [&](Shape full, const std::vector<std::pair<int, int>>& wins) {
        double full_ms = 0.0;
        ok = run_one(stream, full, variant, iters, &full_ms, 0.0, "attention_window") && ok;
        for (const auto& [w, sk] : wins) {
            Shape s = full;
            s.window = w;
            s.sinks = sk;
            ok = run_one(stream, s, variant, iters, nullptr, full_ms, "attention_window") && ok;
        }
    };
    for (int S : {4096, 16384, 32768}) {
        std::vector<std::pair<int, int>> wins = {{128, 0}, {1024, 0}, {4096, 0}};
        if (S <= 16384) wins.insert(wins.end(), {{128, 1}, {1024, 1}});
        pair({1, 32, 8, S, S, 128, 1}, wins);
    }
    for (int S : {4096, 16384}) pair({1, 64, 8, S, S, 64, 1}, {{128, 1}});
    pair({1, 32, 8, 1, 131072, 128, 0}, {{4096, 0}, {128, 0}});
    pair({1, 64, 8, 1, 131072, 64, 0}, {{128, 1}});
    return ok;
}

}  // namespace

int run(int argc, char** argv) {
    spark::bench::Args args(argc, argv);
    spark::bench::print_device_banner();

    if (args.get("sweep", "") == "window") {
        spark::bench::print_header();
        const int variant = args.geti("variant", spark::attention_num_variants() - 1);
        if (!window_sweep(nullptr, variant, args.geti("iters", 50))) {
            std::fprintf(stderr, "CORRECTNESS FAILURE\n");
            return 1;
        }
        return 0;
    }
    std::vector<Shape> shapes;
    if (args.has("b") || args.has("h") || args.has("hq") || args.has("hkv") || args.has("s") ||
        args.has("sq") || args.has("skv") || args.has("d") || args.has("causal") ||
        args.has("kvcap") || args.has("window") || args.has("sinks")) {
        // --h sets both head counts; --hq / --hkv set them apart (GQA).
        const int s = args.geti("s", 4096);
        const int h = args.geti("h", 32);
        const int hq = args.geti("hq", h);
        shapes.push_back({args.geti("b", 1), hq, args.geti("hkv", hq), args.geti("sq", s),
                          args.geti("skv", s), args.geti("d", 128), args.geti("causal", 0),
                          args.geti("kvcap", 0), args.geti("window", 0), args.geti("sinks", 0)});
    } else {
        // Small shapes, checked on every row: both head sizes, both masks, a sequence that is
        // not a multiple of the 128 x 64 tile, and a GQA group of 4. Then Llama-7B-sized
        // prefill (H = 32, D = 128) at 4096 and 8192 tokens, a batch of four 2048-token
        // sequences, a D = 64 head, and a decode step (one query against a 4096-token cache).
        // Then the GQA shapes: Llama-3-8B heads (32 query, 8 K/V) in prefill and decode, the
        // 128K-token decode caches for both head layouts, and a batch of eight decode steps.
        // Last, caches with spare capacity (K/V heads 256 rows apart from where they end),
        // read in place through the capacity stride: the layer's growing cache.
        shapes = {{1, 4, 4, 512, 512, 128, 0, 0},     {1, 4, 4, 512, 512, 128, 1, 0},
                  {1, 4, 4, 512, 512, 64, 0, 0},      {1, 4, 4, 512, 512, 64, 1, 0},
                  {1, 4, 4, 200, 200, 128, 0, 0},     {1, 4, 4, 200, 200, 128, 1, 0},
                  {1, 8, 2, 512, 512, 128, 1, 0},     {1, 32, 32, 4096, 4096, 128, 0, 0},
                  {1, 32, 32, 4096, 4096, 128, 1, 0}, {1, 32, 32, 8192, 8192, 128, 1, 0},
                  {4, 32, 32, 2048, 2048, 128, 1, 0}, {1, 32, 32, 4096, 4096, 64, 1, 0},
                  {1, 32, 32, 1, 4096, 128, 0, 0},    {1, 32, 8, 4096, 4096, 128, 1, 0},
                  {1, 32, 8, 1, 4096, 128, 0, 0},     {1, 32, 32, 1, 131072, 128, 0, 0},
                  {1, 32, 8, 1, 131072, 128, 0, 0},   {8, 32, 8, 1, 4096, 128, 0, 0},
                  {1, 32, 8, 1, 4096, 128, 0, 4352},  {1, 32, 8, 1, 131072, 128, 0, 131328},
                  {8, 32, 8, 1, 4096, 128, 0, 4352},  {1, 32, 8, 4096, 4096, 128, 1, 4352}};
    }
    const int iters = args.geti("iters", 50);
    std::vector<int> variants;
    if (args.has("variant")) {
        variants.push_back(args.geti("variant", 0));
    } else {
        for (int v = 0; v < spark::attention_num_variants(); ++v) variants.push_back(v);
    }

    cudaStream_t stream = nullptr;
    spark::bench::print_header();
    bool all_ok = true;
    for (const auto& s : shapes) {
        for (int v : variants) all_ok = run_one(stream, s, v, iters) && all_ok;
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
        std::fprintf(stderr, "bench_attention: %s\n", e.what());
        return 2;
    }
}
