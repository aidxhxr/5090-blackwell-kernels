// Benchmark + correctness check for attention over a paged K/V cache: the paged decode ladder
// and the varlen prefill ladder (src/kernels/attention_paged.cu).
//
//   ./bench_paged                                  # default sweep, every variant
//   ./bench_paged --op=decode --lens=32768,500x50  # one decode batch: 1 x 32768 + 50 x 500
//   ./bench_paged --op=prefill --lens=2048x4 --variant=1 --iters=20
//   ./bench_paged --op=decode --kv=fp8              # only the e4m3 cache rows
//   flags: --hq=32 --hkv=8 --d=128 --page=16 --kv=bf16,fp8
//
// The cache is a pool of pages handed out in a shuffled order, so no sequence's pages are
// adjacent. Correctness: a CPU double-precision reference on every row of up to eight
// sequences per decode batch (the longest among them) and on 256 sampled (token, head) rows
// per prefill batch.
//
// Decode: GB/s counts K and V once per K/V head (Q and O are 0.1% of it), the floor a
// decode step has to stream. The timing loop rotates through copies of the page pool that
// together exceed L2 (256 MB+). ref_ms is the contiguous flash-decoding kernel (attention
// variant 3) on the same keys in [B, H_kv, L, D] tensors, for the batches of equal lengths.
// Prefill: TFLOPS by FlashAttention's count, halved for the causal mask, per sequence;
// ref_ms is the dense attention ladder's top rung run once per sequence (one launch each,
// what a server without a varlen kernel does, short of padding).
//
// fp8 rows (dtype "fp8"): the same pool rounded to e4m3 with a scale per kv head, through
// paged_decode_fp8 / attention_varlen_fp8, checked against the CPU reference on the e4m3
// values times their scales (so the tolerance is the bf16 rows'). Decode GB/s counts the e4m3
// bytes, half the bf16 rows'. ref_ms of an fp8 row is the bf16 cache's time for the same
// variant and batch, so ref_ms / median_ms is the speedup the format buys.
//
// stdout: one JSON object per row; stderr: the human table.

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "spark/kernels.h"

using namespace spark::bench;

namespace {

struct Cfg {
    int Hq = 32, Hkv = 8, D = 128, page = 16;
    bool bf16 = true, fp8 = true;  // which cache formats get rows (--kv)
};

// "32768,500x50" -> {32768, 500, 500, ... (50 times)}
std::vector<int> parse_lens(const std::string& s) {
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        const auto x = tok.find('x');
        const int len = std::stoi(tok.substr(0, x));
        const int rep = x == std::string::npos ? 1 : std::stoi(tok.substr(x + 1));
        for (int i = 0; i < rep; ++i) out.push_back(len);
    }
    return out;
}

// "mix_n51_max32768_sum57768"-style name, or "n8_l4096" for equal lengths.
std::string lens_name(const std::vector<int>& lens) {
    const int mx = *std::max_element(lens.begin(), lens.end());
    const int mn = *std::min_element(lens.begin(), lens.end());
    const long long sum = std::accumulate(lens.begin(), lens.end(), 0LL);
    if (mx == mn) return "n" + std::to_string(lens.size()) + "_l" + std::to_string(mx);
    return "n" + std::to_string(lens.size()) + "_max" + std::to_string(mx) + "_sum" +
           std::to_string(sum);
}

std::vector<float> fetch(const __nv_bfloat16* d, size_t n) {
    std::vector<__nv_bfloat16> h(n);
    SPARK_CUDA_CHECK(cudaMemcpy(h.data(), d, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
    std::vector<float> f(n);
    for (size_t i = 0; i < n; ++i) f[i] = __bfloat162float(h[i]);
    return f;
}

// Random bf16 values in [lo, hi): the device copy and the rounded host floats.
void random_bf16(std::vector<float>& h, std::vector<__nv_bfloat16>& t, size_t n, uint32_t seed) {
    h.resize(n);
    t.resize(n);
    fill_uniform(h, -1.f, 1.f, seed);
    for (size_t i = 0; i < n; ++i) {
        t[i] = __float2bfloat16(h[i]);
        h[i] = __bfloat162float(t[i]);
    }
}

// The paged cache for sequences of `lens` keys: `copies` identical pools (the timing loop
// rotates through them), a block table and the lengths, all on the device, plus the host
// pool for the reference.
struct Paged {
    int B = 0, max_pages = 0, pages = 0, copies = 1;
    size_t pool_elems = 0;
    std::vector<int> bt;  // [B, max_pages]
    std::vector<float> hk, hv;
    __nv_bfloat16 *dk = nullptr, *dv = nullptr;
    int *dbt = nullptr, *dlens = nullptr;

    Paged(const std::vector<int>& lens, const Cfg& c, uint32_t seed) {
        B = static_cast<int>(lens.size());
        std::vector<int> np(B);
        for (int b = 0; b < B; ++b) np[b] = (lens[b] + c.page - 1) / c.page;
        pages = std::accumulate(np.begin(), np.end(), 0) + 1;
        max_pages = std::max(1, *std::max_element(np.begin(), np.end()));
        std::vector<int> perm(pages);
        std::iota(perm.begin(), perm.end(), 0);
        std::mt19937 rng(seed);
        std::shuffle(perm.begin(), perm.end(), rng);
        bt.assign(static_cast<size_t>(B) * max_pages, 0);
        for (int b = 0, i = 0; b < B; ++b)
            for (int j = 0; j < np[b]; ++j) bt[static_cast<size_t>(b) * max_pages + j] = perm[i++];
        pool_elems = static_cast<size_t>(pages) * c.Hkv * c.page * c.D;
        std::vector<__nv_bfloat16> tk, tv;
        random_bf16(hk, tk, pool_elems, seed + 1);
        random_bf16(hv, tv, pool_elems, seed + 2);
        const size_t bytes = pool_elems * sizeof(__nv_bfloat16);
        copies = static_cast<int>((size_t{256} << 20) / (2 * bytes)) + 1;
        SPARK_CUDA_CHECK(cudaMalloc(&dk, copies * bytes));
        SPARK_CUDA_CHECK(cudaMalloc(&dv, copies * bytes));
        for (int k = 0; k < copies; ++k) {
            SPARK_CUDA_CHECK(
                cudaMemcpy(dk + k * pool_elems, tk.data(), bytes, cudaMemcpyHostToDevice));
            SPARK_CUDA_CHECK(
                cudaMemcpy(dv + k * pool_elems, tv.data(), bytes, cudaMemcpyHostToDevice));
        }
        SPARK_CUDA_CHECK(cudaMalloc(&dbt, bt.size() * sizeof(int)));
        SPARK_CUDA_CHECK(
            cudaMemcpy(dbt, bt.data(), bt.size() * sizeof(int), cudaMemcpyHostToDevice));
        SPARK_CUDA_CHECK(cudaMalloc(&dlens, B * sizeof(int)));
        SPARK_CUDA_CHECK(cudaMemcpy(dlens, lens.data(), B * sizeof(int), cudaMemcpyHostToDevice));
    }
    ~Paged() {
        cudaFree(dk);
        cudaFree(dv);
        cudaFree(dbt);
        cudaFree(dlens);
        cudaFree(dk8);
        cudaFree(dv8);
        cudaFree(dks);
        cudaFree(dvs);
    }
    // Host offset of key j of sequence b, kv head h.
    size_t at(const Cfg& c, int b, int h, int j) const {
        const int page = bt[static_cast<size_t>(b) * max_pages + j / c.page];
        return ((static_cast<size_t>(page) * c.Hkv + h) * c.page + j % c.page) * c.D;
    }

    // The e4m3 form of the pool, made on first use: element x of kv head h is
    // e4m3(x / scale[h]), scale[h] = (1 + h / 4) / 448 for K and 0.75 times that for V, so the
    // values (in [-1, 1)) reach most of e4m3's range, subnormals included. lut decodes a byte.
    std::vector<unsigned char> hk8, hv8;
    std::vector<float> ks, vs;
    float lut[256] = {};
    unsigned char *dk8 = nullptr, *dv8 = nullptr;
    float *dks = nullptr, *dvs = nullptr;
    int copies8 = 1;
    void make_fp8(const Cfg& c) {
        if (dk8) return;
        for (int i = 0; i < 256; ++i) {
            __nv_fp8_e4m3 e;
            e.__x = static_cast<__nv_fp8_storage_t>(i);
            lut[i] = static_cast<float>(e);
        }
        ks.resize(c.Hkv);
        vs.resize(c.Hkv);
        for (int h = 0; h < c.Hkv; ++h) {
            ks[h] = (1.f + h / 4.f) / 448.f;
            vs[h] = 0.75f * ks[h];
        }
        hk8.resize(pool_elems);
        hv8.resize(pool_elems);
        for (size_t i = 0; i < pool_elems; ++i) {
            const int h = static_cast<int>(i / (static_cast<size_t>(c.page) * c.D) % c.Hkv);
            hk8[i] = __nv_fp8_e4m3(hk[i] / ks[h]).__x;
            hv8[i] = __nv_fp8_e4m3(hv[i] / vs[h]).__x;
        }
        copies8 = static_cast<int>((size_t{256} << 20) / (2 * pool_elems)) + 1;
        SPARK_CUDA_CHECK(cudaMalloc(&dk8, copies8 * pool_elems));
        SPARK_CUDA_CHECK(cudaMalloc(&dv8, copies8 * pool_elems));
        for (int k = 0; k < copies8; ++k) {
            SPARK_CUDA_CHECK(
                cudaMemcpy(dk8 + k * pool_elems, hk8.data(), pool_elems, cudaMemcpyHostToDevice));
            SPARK_CUDA_CHECK(
                cudaMemcpy(dv8 + k * pool_elems, hv8.data(), pool_elems, cudaMemcpyHostToDevice));
        }
        SPARK_CUDA_CHECK(cudaMalloc(&dks, c.Hkv * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMalloc(&dvs, c.Hkv * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMemcpy(dks, ks.data(), c.Hkv * sizeof(float), cudaMemcpyHostToDevice));
        SPARK_CUDA_CHECK(cudaMemcpy(dvs, vs.data(), c.Hkv * sizeof(float), cudaMemcpyHostToDevice));
    }
    const __nv_fp8_e4m3* k8(size_t off = 0) const {
        return reinterpret_cast<const __nv_fp8_e4m3*>(dk8 + off);
    }
    const __nv_fp8_e4m3* v8(size_t off = 0) const {
        return reinterpret_cast<const __nv_fp8_e4m3*>(dv8 + off);
    }
    // Element i of kv head h as the kernels read it: the bf16 value, or the e4m3 value times
    // the head's scale.
    double kval(bool fp8, size_t i, int h) const {
        return fp8 ? static_cast<double>(lut[hk8[i]]) * ks[h] : hk[i];
    }
    double vval(bool fp8, size_t i, int h) const {
        return fp8 ? static_cast<double>(lut[hv8[i]]) * vs[h] : hv[i];
    }
};

// One query row q (D floats) against keys [0, n) of sequence b, kv head h, in double, on the
// bf16 pool or (fp8) the e4m3 one.
std::vector<double> ref_row(const Cfg& c, const Paged& pc, const float* q, int b, int h, int n,
                            bool fp8 = false) {
    std::vector<double> s(n), o(c.D, 0.0);
    const double scale = 1.0 / std::sqrt(static_cast<double>(c.D));
    double mx = -1e300;
    for (int j = 0; j < n; ++j) {
        const size_t k = pc.at(c, b, h, j);
        double d = 0;
        for (int e = 0; e < c.D; ++e) d += static_cast<double>(q[e]) * pc.kval(fp8, k + e, h);
        s[j] = d * scale;
        mx = std::max(mx, s[j]);
    }
    double l = 0;
    for (int j = 0; j < n; ++j) {
        const double p = std::exp(s[j] - mx);
        l += p;
        const size_t v = pc.at(c, b, h, j);
        for (int e = 0; e < c.D; ++e) o[e] += p * pc.vval(fp8, v + e, h);
    }
    for (auto& x : o) x /= l;
    return o;
}

double kv_bytes(const Cfg& c, const std::vector<int>& lens) {
    const double keys = std::accumulate(lens.begin(), lens.end(), 0.0);
    return 2.0 * keys * c.Hkv * c.D * sizeof(__nv_bfloat16);
}

// ---- decode ------------------------------------------------------------------------------

// The contiguous flash-decoding kernel on the same keys, for batches of equal lengths.
double contiguous_ms(const Cfg& c, const Paged& pc, const std::vector<int>& lens,
                     const __nv_bfloat16* dq, int iters, cudaStream_t stream) {
    const int B = pc.B, L = lens[0];
    const size_t n = static_cast<size_t>(B) * c.Hkv * L * c.D;
    std::vector<__nv_bfloat16> tk(n), tv(n);
    for (int b = 0; b < B; ++b)
        for (int h = 0; h < c.Hkv; ++h)
            for (int j = 0; j < L; ++j) {
                const size_t src = pc.at(c, b, h, j);
                const size_t dst = ((static_cast<size_t>(b) * c.Hkv + h) * L + j) * c.D;
                for (int e = 0; e < c.D; ++e) {
                    tk[dst + e] = __float2bfloat16(pc.hk[src + e]);
                    tv[dst + e] = __float2bfloat16(pc.hv[src + e]);
                }
            }
    const size_t bytes = n * sizeof(__nv_bfloat16);
    const int copies = static_cast<int>((size_t{256} << 20) / (2 * bytes)) + 1;
    __nv_bfloat16 *dk, *dv, *dout;
    SPARK_CUDA_CHECK(cudaMalloc(&dk, copies * bytes));
    SPARK_CUDA_CHECK(cudaMalloc(&dv, copies * bytes));
    SPARK_CUDA_CHECK(cudaMalloc(&dout, static_cast<size_t>(B) * c.Hq * c.D * 2));
    for (int k = 0; k < copies; ++k) {
        SPARK_CUDA_CHECK(cudaMemcpy(dk + k * n, tk.data(), bytes, cudaMemcpyHostToDevice));
        SPARK_CUDA_CHECK(cudaMemcpy(dv + k * n, tv.data(), bytes, cudaMemcpyHostToDevice));
    }
    int turn = 0;
    const Timing t = time_kernel(
        [&] {
            const size_t off = static_cast<size_t>(turn++ % copies) * n;
            spark::attention_bf16(dq, dk + off, dv + off, dout, B, c.Hq, c.Hkv, 1, L, c.D, false, 3,
                                  stream);
        },
        stream, 10, iters);
    SPARK_CUDA_CHECK(cudaFree(dk));
    SPARK_CUDA_CHECK(cudaFree(dv));
    SPARK_CUDA_CHECK(cudaFree(dout));
    return t.median_ms;
}

bool run_decode(const Cfg& c, const std::vector<int>& lens, const std::vector<int>& variants,
                int iters, cudaStream_t stream) {
    Paged pc(lens, c, 7);
    const int B = pc.B;
    const size_t nq = static_cast<size_t>(B) * c.Hq * c.D;
    std::vector<float> hq;
    std::vector<__nv_bfloat16> tq;
    random_bf16(hq, tq, nq, 11);
    __nv_bfloat16 *dq, *dout;
    SPARK_CUDA_CHECK(cudaMalloc(&dq, nq * 2));
    SPARK_CUDA_CHECK(cudaMalloc(&dout, nq * 2));
    SPARK_CUDA_CHECK(cudaMemcpy(dq, tq.data(), nq * 2, cudaMemcpyHostToDevice));

    // Reference rows: the longest sequence and up to seven others spread over the batch.
    std::vector<int> check;
    check.push_back(static_cast<int>(std::max_element(lens.begin(), lens.end()) - lens.begin()));
    for (int i = 0; i < 7 && i < B; ++i) check.push_back(i * B / 7 % B);
    std::sort(check.begin(), check.end());
    check.erase(std::unique(check.begin(), check.end()), check.end());
    const int group = c.Hq / c.Hkv;
    auto refs = [&](bool fp8) {
        std::vector<std::vector<double>> ref;
        for (int b : check)
            for (int h = 0; h < c.Hq; ++h)
                ref.push_back(lens[b] > 0
                                  ? ref_row(c, pc,
                                            hq.data() + (static_cast<size_t>(b) * c.Hq + h) * c.D,
                                            b, h / group, lens[b], fp8)
                                  : std::vector<double>(c.D, 0.0));
        return ref;
    };

    const bool equal = std::all_of(lens.begin(), lens.end(), [&](int l) { return l == lens[0]; });
    const double contig_ms = equal && c.bf16 ? contiguous_ms(c, pc, lens, dq, iters, stream) : 0.0;
    const std::string shape = "paged_" + lens_name(lens) + "_hq" + std::to_string(c.Hq) + "_hkv" +
                              std::to_string(c.Hkv) + "_d" + std::to_string(c.D) + "_p" +
                              std::to_string(c.page);
    bool all_ok = true;
    std::vector<double> bf16_ms(variants.size(), 0.0);  // the fp8 rows' ref_ms
    for (const bool fp8 : {false, true}) {
        if (fp8 ? !c.fp8 : !c.bf16) continue;
        if (fp8) pc.make_fp8(c);
        const std::vector<std::vector<double>> ref = refs(fp8);
        const double gb = kv_bytes(c, lens) / (fp8 ? 2 : 1) / 1e9;
        // Variant v on copy `turn` of the pool (the timing loop rotates them).
        auto run = [&](int v, int turn) {
            if (fp8) {
                const size_t off = static_cast<size_t>(turn % pc.copies8) * pc.pool_elems;
                spark::paged_decode_fp8(dq, pc.k8(off), pc.v8(off), pc.dks, pc.dvs, pc.dbt,
                                        pc.dlens, dout, B, c.Hq, c.Hkv, c.D, c.page, pc.max_pages,
                                        v, stream);
            } else {
                const size_t off = static_cast<size_t>(turn % pc.copies) * pc.pool_elems;
                spark::paged_decode_bf16(dq, pc.dk + off, pc.dv + off, pc.dbt, pc.dlens, dout, B,
                                         c.Hq, c.Hkv, c.D, c.page, pc.max_pages, v, stream);
            }
        };
        for (size_t vi = 0; vi < variants.size(); ++vi) {
            const int v = variants[vi];
            run(v, 0);
            SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
            const std::vector<float> got = fetch(dout, nq);
            double max_abs = 0, max_ref = 0;
            size_t r = 0;
            for (int b : check)
                for (int h = 0; h < c.Hq; ++h, ++r)
                    for (int e = 0; e < c.D; ++e) {
                        const double x = got[(static_cast<size_t>(b) * c.Hq + h) * c.D + e];
                        max_abs = std::max(max_abs, std::fabs(x - ref[r][e]));
                        max_ref = std::max(max_ref, std::fabs(ref[r][e]));
                    }
            const double tol = 2e-2 * max_ref + 1e-3;
            const bool ok = max_abs <= tol && std::isfinite(max_abs);
            int turn = 0;
            const Timing t = time_kernel([&] { run(v, turn++); }, stream, 10,
                                         v == 0 ? std::min(iters, 10) : iters);
            Row row;
            row.kernel = "paged_decode";
            row.dtype = fp8 ? "fp8" : "bf16";
            row.variant = v;
            row.shape = shape;
            row.median_ms = t.median_ms;
            row.min_ms = t.min_ms;
            row.gbps = gb / (t.median_ms * 1e-3);
            row.ref_ms = fp8 ? bf16_ms[vi] : contig_ms;
            row.max_abs_err = max_abs;
            row.max_rel_err = max_ref > 0 ? max_abs / max_ref : 0;
            row.ok = ok;
            print_row(row);
            if (!ok) std::fprintf(stderr, "  FAIL: max_abs %.3e > tol %.3e\n", max_abs, tol);
            if (!fp8) bf16_ms[vi] = t.median_ms;
            if (fp8 && bf16_ms[vi] > 0)
                std::fprintf(stderr, "    e4m3 cache, variant %d: %.2fx the bf16 cache's speed\n",
                             v, bf16_ms[vi] / t.median_ms);
            all_ok = all_ok && ok;
        }
    }
    if (contig_ms > 0)
        std::fprintf(
            stderr,
            "    contiguous flash-decoding (attention v3) on the same keys: %.4f ms, %.0f GB/s\n",
            contig_ms, kv_bytes(c, lens) / 1e9 / (contig_ms * 1e-3));
    SPARK_CUDA_CHECK(cudaFree(dq));
    SPARK_CUDA_CHECK(cudaFree(dout));
    return all_ok;
}

// ---- varlen prefill ----------------------------------------------------------------------

bool run_prefill(const Cfg& c, const std::vector<int>& lens, const std::vector<int>& variants,
                 int iters, cudaStream_t stream) {
    Paged pc(lens, c, 9);
    const int B = pc.B;
    std::vector<int> cu(B + 1, 0);
    for (int b = 0; b < B; ++b) cu[b + 1] = cu[b] + lens[b];
    const int T = cu[B];
    const size_t nq = static_cast<size_t>(T) * c.Hq * c.D;
    std::vector<float> hq;
    std::vector<__nv_bfloat16> tq;
    random_bf16(hq, tq, nq, 13);
    __nv_bfloat16 *dq, *dout;
    int* dcu;
    SPARK_CUDA_CHECK(cudaMalloc(&dq, nq * 2));
    SPARK_CUDA_CHECK(cudaMalloc(&dout, nq * 2));
    SPARK_CUDA_CHECK(cudaMalloc(&dcu, (B + 1) * sizeof(int)));
    SPARK_CUDA_CHECK(cudaMemcpy(dq, tq.data(), nq * 2, cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(dcu, cu.data(), (B + 1) * sizeof(int), cudaMemcpyHostToDevice));

    // 256 sampled (token, head) rows, the last token of the longest sequence among them.
    std::mt19937 rng(5);
    std::vector<std::pair<int, int>> rows;
    const int longest = static_cast<int>(std::max_element(lens.begin(), lens.end()) - lens.begin());
    rows.push_back({cu[longest + 1] - 1, c.Hq - 1});
    for (int i = 0; i < 255; ++i)
        rows.push_back({static_cast<int>(rng() % T), static_cast<int>(rng() % c.Hq)});
    const int group = c.Hq / c.Hkv;
    auto refs = [&](bool fp8) {
        std::vector<std::vector<double>> ref;
        for (auto [t, h] : rows) {
            const int b =
                static_cast<int>(std::upper_bound(cu.begin(), cu.end(), t) - cu.begin()) - 1;
            ref.push_back(ref_row(c, pc, hq.data() + (static_cast<size_t>(t) * c.Hq + h) * c.D, b,
                                  h / group, t - cu[b] + 1, fp8));
        }
        return ref;
    };

    // Reference timing: the dense ladder's top rung once per sequence on contiguous copies.
    std::vector<__nv_bfloat16*> ck(B), cv(B), cq(B), co(B);
    for (int b = 0; b < B; ++b) {
        const size_t n = static_cast<size_t>(c.Hkv) * lens[b] * c.D;
        const size_t nqb = static_cast<size_t>(c.Hq) * lens[b] * c.D;
        std::vector<__nv_bfloat16> tk(n), tv(n);
        for (int h = 0; h < c.Hkv; ++h)
            for (int j = 0; j < lens[b]; ++j)
                for (int e = 0; e < c.D; ++e) {
                    const size_t src = pc.at(c, b, h, j) + e;
                    tk[(static_cast<size_t>(h) * lens[b] + j) * c.D + e] =
                        __float2bfloat16(pc.hk[src]);
                    tv[(static_cast<size_t>(h) * lens[b] + j) * c.D + e] =
                        __float2bfloat16(pc.hv[src]);
                }
        SPARK_CUDA_CHECK(cudaMalloc(&ck[b], n * 2));
        SPARK_CUDA_CHECK(cudaMalloc(&cv[b], n * 2));
        SPARK_CUDA_CHECK(cudaMalloc(&cq[b], nqb * 2));
        SPARK_CUDA_CHECK(cudaMalloc(&co[b], nqb * 2));
        SPARK_CUDA_CHECK(cudaMemcpy(ck[b], tk.data(), n * 2, cudaMemcpyHostToDevice));
        SPARK_CUDA_CHECK(cudaMemcpy(cv[b], tv.data(), n * 2, cudaMemcpyHostToDevice));
        SPARK_CUDA_CHECK(cudaMemset(cq[b], 0, nqb * 2));
    }
    const int top = spark::attention_num_variants() - 1;
    const Timing tr = time_kernel(
        [&] {
            for (int b = 0; b < B; ++b)
                spark::attention_bf16(cq[b], ck[b], cv[b], co[b], 1, c.Hq, c.Hkv, lens[b], lens[b],
                                      c.D, true, top, stream);
        },
        stream, 5, iters);
    for (int b = 0; b < B; ++b) {
        SPARK_CUDA_CHECK(cudaFree(ck[b]));
        SPARK_CUDA_CHECK(cudaFree(cv[b]));
        SPARK_CUDA_CHECK(cudaFree(cq[b]));
        SPARK_CUDA_CHECK(cudaFree(co[b]));
    }

    double flop = 0;
    for (int l : lens) flop += 4.0 * c.Hq * static_cast<double>(l) * l * c.D / 2;
    const std::string shape = "varlen_" + lens_name(lens) + "_hq" + std::to_string(c.Hq) + "_hkv" +
                              std::to_string(c.Hkv) + "_d" + std::to_string(c.D) + "_p" +
                              std::to_string(c.page) + "_causal";
    bool all_ok = true;
    std::vector<double> bf16_ms(variants.size(), 0.0);  // the fp8 rows' ref_ms
    for (const bool fp8 : {false, true}) {
        if (fp8 ? !c.fp8 : !c.bf16) continue;
        if (fp8) pc.make_fp8(c);
        const std::vector<std::vector<double>> ref = refs(fp8);
        auto run = [&](int v) {
            if (fp8)
                spark::attention_varlen_fp8(dq, pc.k8(), pc.v8(), pc.dks, pc.dvs, dcu, pc.dlens,
                                            pc.dbt, dout, B, T, c.Hq, c.Hkv, c.D, c.page,
                                            pc.max_pages, true, v, stream);
            else
                spark::attention_varlen_bf16(dq, pc.dk, pc.dv, dcu, pc.dlens, pc.dbt, dout, B, T,
                                             c.Hq, c.Hkv, c.D, c.page, pc.max_pages, true, v,
                                             stream);
        };
        for (size_t vi = 0; vi < variants.size(); ++vi) {
            const int v = variants[vi];
            run(v);
            SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
            const std::vector<float> got = fetch(dout, nq);
            double max_abs = 0, max_ref = 0;
            for (size_t r = 0; r < rows.size(); ++r)
                for (int e = 0; e < c.D; ++e) {
                    const double x =
                        got[(static_cast<size_t>(rows[r].first) * c.Hq + rows[r].second) * c.D + e];
                    max_abs = std::max(max_abs, std::fabs(x - ref[r][e]));
                    max_ref = std::max(max_ref, std::fabs(ref[r][e]));
                }
            const double tol = 2e-2 * max_ref + 1e-3;
            const bool ok = max_abs <= tol && std::isfinite(max_abs);
            const Timing t =
                time_kernel([&] { run(v); }, stream, 5, v == 0 ? std::min(iters, 5) : iters);
            Row row;
            row.kernel = "attention_varlen";
            row.dtype = fp8 ? "fp8" : "bf16";
            row.variant = v;
            row.shape = shape;
            row.median_ms = t.median_ms;
            row.min_ms = t.min_ms;
            row.tflops = flop / (t.median_ms * 1e-3) / 1e12;
            row.ref_ms = fp8 ? bf16_ms[vi] : tr.median_ms;
            row.max_abs_err = max_abs;
            row.max_rel_err = max_ref > 0 ? max_abs / max_ref : 0;
            row.ok = ok;
            print_row(row);
            if (!ok) std::fprintf(stderr, "  FAIL: max_abs %.3e > tol %.3e\n", max_abs, tol);
            if (!fp8) bf16_ms[vi] = t.median_ms;
            all_ok = all_ok && ok;
        }
    }
    std::fprintf(stderr, "    dense attention v%d once per sequence: %.4f ms, %.1f TFLOPS\n", top,
                 tr.median_ms, flop / (tr.median_ms * 1e-3) / 1e12);
    SPARK_CUDA_CHECK(cudaFree(dq));
    SPARK_CUDA_CHECK(cudaFree(dout));
    SPARK_CUDA_CHECK(cudaFree(dcu));
    return all_ok;
}

// A batch of n lengths drawn uniformly from [lo, hi], seeded.
std::string uniform_lens(int n, int lo, int hi, uint32_t seed) {
    std::mt19937 rng(seed);
    std::string s;
    for (int i = 0; i < n; ++i) {
        if (i) s += ",";
        s += std::to_string(lo + static_cast<int>(rng() % (hi - lo + 1)));
    }
    return s;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args(argc, argv);
        print_device_banner();
        const int iters = args.geti("iters", 50);
        Cfg c;
        c.Hq = args.geti("hq", 32);
        c.Hkv = args.geti("hkv", 8);
        c.D = args.geti("d", 128);
        c.page = args.geti("page", 16);
        const std::string kv = args.get("kv", "bf16,fp8");
        c.bf16 = kv.find("bf16") != std::string::npos;
        c.fp8 = kv.find("fp8") != std::string::npos;
        if (!c.bf16 && !c.fp8) throw std::invalid_argument("--kv takes bf16, fp8 or both");
        const std::string op = args.get("op", "all");
        std::vector<int> variants;
        if (args.has("variant")) {
            variants.push_back(args.geti("variant", 1));
        } else {
            for (int v = 0; v < spark::paged_decode_num_variants(); ++v) variants.push_back(v);
        }
        std::vector<std::string> decode, prefill;
        if (args.has("lens")) {
            (op == "prefill" ? prefill : decode).push_back(args.get("lens", ""));
        } else {
            decode = {"4096",
                      "32768",
                      "4096x8",
                      "2048x32",
                      "1024x64",
                      "32768,500x50",
                      uniform_lens(64, 1, 8192, 3),
                      uniform_lens(32, 100, 2000, 4)};
            prefill = {"4096", "2048x4", "3000,1500,700,300,200,100,50,20",
                       uniform_lens(16, 64, 1024, 5)};
            if (op == "decode") prefill.clear();
            if (op == "prefill") decode.clear();
        }
        cudaStream_t stream;
        SPARK_CUDA_CHECK(cudaStreamCreate(&stream));
        print_header();
        bool ok = true;
        for (const auto& s : decode)
            ok = run_decode(c, parse_lens(s), variants, iters, stream) && ok;
        for (const auto& s : prefill)
            ok = run_prefill(c, parse_lens(s), variants, iters, stream) && ok;
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
