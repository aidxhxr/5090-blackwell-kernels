// Benchmark + correctness check for the fused MoE decode step over 1-bit and ternary experts
// (w1moe_decode_bf16), DeepSeek V4.1 Flash's layer: 384 experts of 5120 -> 2304 -> 5120,
// 6 per token.
//
//   ./bench_w1moe_decode                           # T = 1, 4, 8; bits 1 and 2; every variant
//   ./bench_w1moe_decode --t=1 --bits=1 --variant=2 --iters=50
//   ./bench_w1moe_decode --repeat-routing          # the same six experts twice back to back
//   ./bench_w1moe_decode --chain=20                # also 20 calls per event pair
//
// One layer's routed experts are allocated at the requested width (1.91 GB at 1 bit, 3.61 GB
// at ternary, packed plus scales) and filled with random codes and scales on the device: the
// kernel reads a byte stream and does not care where the bits came from, and quantizing 27 GB
// of bf16 is not the point. For every T the bench draws --routings random routings (topk
// distinct experts per token, routing weights summing to one), computes a CPU double
// reference for the first one on the chosen experts (gate, up and SwiGLU rounded to bf16 as
// the kernels round them), checks every variant against it to 2% of max|y|, then times each
// variant. By default consecutive launches take consecutive routings, so a launch finds its
// experts cold in L2 the way a decode step does (a layer's whole expert set is 1.9 GB, the L2
// 24 or 96 MB). With --repeat-routing every timed call is the same routing launched twice,
// and the per-launch time is half of it: the second launch meets whatever the first left in
// L2, which is how much of a 31 MB active set a 24 MB L2 keeps.
//
// gbps = the active bytes (the distinct experts' packed codes and scales, x and y) over the
// time. stderr also prints the floor of those bytes at 273 GB/s (the GB10's spec) and at
// 1,532 GB/s (the 5090's measured copy roof), and the tokens/s that 40 such layers bound at
// this time, so one JSON file reads the same on both machines.
//
// stdout: one JSON object per row (collected by scripts/run_all_benches.sh)
// stderr: human-readable table and the floors

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "spark/kernels.h"

namespace {

constexpr int kG = spark::kW1GroupSize;
constexpr int kHidden = 5120, kInter = 2304, kExperts = 384, kTopK = 6;
constexpr double kGB10Roof = 273e9, k5090Roof = 1532e9;

template <typename T>
T* dmalloc(size_t n) {
    T* p = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

// A hash, so the fill is reproducible and needs no state.
__host__ __device__ __forceinline__ uint32_t hash32(uint64_t v) {
    v ^= v >> 33;
    v *= 0xff51afd7ed558ccdull;
    v ^= v >> 33;
    v *= 0xc4ceb9fe1a85ec53ull;
    v ^= v >> 33;
    return static_cast<uint32_t>(v);
}

// Random packed words: every bit pattern at 1 bit; at 2 bits a code's high bit is cleared
// where its low bit is set, so the codes are 0, 1, 2 and never 3.
__global__ void fill_words_kernel(uint32_t* w, size_t n, int bits, uint64_t seed) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t v = hash32(seed * 0x9E3779B97F4A7C15ull + i);
    if (bits == 2) v &= ~((v & 0x55555555u) << 1);
    w[i] = v;
}

// Scales in [0.5 s, 1.5 s] as bf16, s the mean |w| of N(0, 0.02) weights.
__global__ void fill_scales_kernel(__nv_bfloat16* s, size_t n, uint64_t seed) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float u = (hash32(seed * 0x9E3779B97F4A7C15ull + i) >> 8) * (1.0f / 16777216.0f);
    s[i] = __float2bfloat16(0.016f * (0.5f + u));
}

void fill_words(uint32_t* w, size_t n, int bits, uint64_t seed, cudaStream_t stream) {
    const size_t grid = (n + 255) / 256;
    fill_words_kernel<<<static_cast<unsigned>(grid), 256, 0, stream>>>(w, n, bits, seed);
    SPARK_CHECK_LAUNCH();
}

void fill_scales(__nv_bfloat16* s, size_t n, uint64_t seed, cudaStream_t stream) {
    const size_t grid = (n + 255) / 256;
    fill_scales_kernel<<<static_cast<unsigned>(grid), 256, 0, stream>>>(s, n, seed);
    SPARK_CHECK_LAUNCH();
}

// One layer's experts on the device.
struct Layer {
    int E, K, I, bits;
    size_t gu_words_per_expert, gu_scales_per_expert, dn_words_per_expert, dn_scales_per_expert;
    uint32_t* gu_packed = nullptr;
    __nv_bfloat16* gu_scales = nullptr;
    uint32_t* dn_packed = nullptr;
    __nv_bfloat16* dn_scales = nullptr;

    size_t bytes_per_expert() const {
        return (gu_words_per_expert + dn_words_per_expert) * 4 +
               (gu_scales_per_expert + dn_scales_per_expert) * 2;
    }
    size_t bytes_total() const { return bytes_per_expert() * E; }
    void free_all() {
        for (void* p : {static_cast<void*>(gu_packed), static_cast<void*>(gu_scales),
                        static_cast<void*>(dn_packed), static_cast<void*>(dn_scales)})
            if (p) SPARK_CUDA_CHECK(cudaFree(p));
    }
};

Layer make_layer(int E, int K, int I, int bits, cudaStream_t stream) {
    Layer L;
    L.E = E;
    L.K = K;
    L.I = I;
    L.bits = bits;
    L.gu_words_per_expert = static_cast<size_t>(2 * I) * (K * bits / 32);
    L.gu_scales_per_expert = static_cast<size_t>(K / kG) * 2 * I;
    L.dn_words_per_expert = static_cast<size_t>(K) * (I * bits / 32);
    L.dn_scales_per_expert = static_cast<size_t>(I / kG) * K;
    L.gu_packed = dmalloc<uint32_t>(L.gu_words_per_expert * E);
    L.gu_scales = dmalloc<__nv_bfloat16>(L.gu_scales_per_expert * E);
    L.dn_packed = dmalloc<uint32_t>(L.dn_words_per_expert * E);
    L.dn_scales = dmalloc<__nv_bfloat16>(L.dn_scales_per_expert * E);
    fill_words(L.gu_packed, L.gu_words_per_expert * E, bits, 1, stream);
    fill_scales(L.gu_scales, L.gu_scales_per_expert * E, 2, stream);
    fill_words(L.dn_packed, L.dn_words_per_expert * E, bits, 3, stream);
    fill_scales(L.dn_scales, L.dn_scales_per_expert * E, 4, stream);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    std::fprintf(
        stderr,
        "  layer: %d experts x (%d -> %d -> %d) at %d bit%s: %.2f GB (%.2f MB per expert)\n", E, K,
        I, K, bits, bits == 1 ? "" : "s", L.bytes_total() / 1e9, L.bytes_per_expert() / 1e6);
    return L;
}

// A routing: topk distinct experts per token, weights summing to one.
struct Routing {
    std::vector<int32_t> ids;        // [T][topk]
    std::vector<__nv_bfloat16> wts;  // [T][topk]
    std::vector<float> wts_f;        // the bf16 values, as the kernel sees them
    int distinct = 0;
    int32_t* d_ids = nullptr;
    __nv_bfloat16* d_wts = nullptr;
};

Routing make_routing(int T, int topk, int E, uint32_t seed) {
    Routing r;
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, E - 1);
    std::uniform_real_distribution<float> uni(0.1f, 1.0f);
    std::set<int> all;
    for (int t = 0; t < T; ++t) {
        std::set<int> chosen;
        while (static_cast<int>(chosen.size()) < topk) chosen.insert(pick(rng));
        std::vector<float> w(topk);
        float sum = 0.f;
        for (auto& v : w) {
            v = uni(rng);
            sum += v;
        }
        int slot = 0;
        for (int e : chosen) {
            r.ids.push_back(e);
            const __nv_bfloat16 b = __float2bfloat16(w[slot] / sum);
            r.wts.push_back(b);
            r.wts_f.push_back(__bfloat162float(b));
            all.insert(e);
            ++slot;
        }
    }
    r.distinct = static_cast<int>(all.size());
    r.d_ids = dmalloc<int32_t>(r.ids.size());
    r.d_wts = dmalloc<__nv_bfloat16>(r.wts.size());
    SPARK_CUDA_CHECK(cudaMemcpy(r.d_ids, r.ids.data(), r.ids.size() * 4, cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(r.d_wts, r.wts.data(), r.wts.size() * 2, cudaMemcpyHostToDevice));
    return r;
}

// Code (n, k) of a packed [N][K bits / 32] row-major matrix on the host.
inline int code_of(const uint32_t* packed, int n, int k, int K, int bits) {
    const size_t wpr = static_cast<size_t>(K) * bits / 32;
    if (bits == 1) return (packed[n * wpr + k / 32] >> (k % 32)) & 1;
    return (packed[n * wpr + k / 16] >> (2 * (k % 16))) & 3;
}

inline float bf16r(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

inline double dequant(int code, float s, int bits) {
    const float v = bits == 1 ? (code ? s : -s) : s * static_cast<float>(code - 1);
    return bf16r(v);
}

// y = x · dequant(W) for one token on the host, W packed [N][K bits/32] with scales [K/128][N].
void gemv_host(const std::vector<float>& x, const uint32_t* packed, const __nv_bfloat16* scales,
               int K, int N, int bits, std::vector<double>& y) {
    y.assign(N, 0.0);
    for (int n = 0; n < N; ++n) {
        double acc = 0.0;
        for (int grp = 0; grp < K / kG; ++grp) {
            const float s = __bfloat162float(scales[static_cast<size_t>(grp) * N + n]);
            double gs = 0.0;
            for (int k = grp * kG; k < (grp + 1) * kG; ++k)
                gs += x[k] * dequant(code_of(packed, n, k, K, bits), s, bits);
            acc += gs;
        }
        y[n] = acc;
    }
}

// The reference for one routing: every (token, slot) on the chosen expert's bytes copied to
// the host, gate / up / SwiGLU rounded to bf16, the down product in double, weighted sum.
std::vector<float> reference(const Layer& L, const Routing& r, const std::vector<float>& x, int T,
                             int topk) {
    const int K = L.K, I = L.I;
    std::vector<float> y(static_cast<size_t>(T) * K, 0.f);
    std::vector<uint32_t> gu(L.gu_words_per_expert), dn(L.dn_words_per_expert);
    std::vector<__nv_bfloat16> gus(L.gu_scales_per_expert), dns(L.dn_scales_per_expert);
    std::vector<double> gu_out, d_out;
    std::vector<float> xt(K), h(I);
    int last_e = -1;
    for (int t = 0; t < T; ++t) {
        std::copy(x.begin() + static_cast<size_t>(t) * K,
                  x.begin() + static_cast<size_t>(t + 1) * K, xt.begin());
        for (int slot = 0; slot < topk; ++slot) {
            const int e = r.ids[t * topk + slot];
            if (e != last_e) {
                SPARK_CUDA_CHECK(cudaMemcpy(gu.data(), L.gu_packed + e * L.gu_words_per_expert,
                                            gu.size() * 4, cudaMemcpyDeviceToHost));
                SPARK_CUDA_CHECK(cudaMemcpy(gus.data(), L.gu_scales + e * L.gu_scales_per_expert,
                                            gus.size() * 2, cudaMemcpyDeviceToHost));
                SPARK_CUDA_CHECK(cudaMemcpy(dn.data(), L.dn_packed + e * L.dn_words_per_expert,
                                            dn.size() * 4, cudaMemcpyDeviceToHost));
                SPARK_CUDA_CHECK(cudaMemcpy(dns.data(), L.dn_scales + e * L.dn_scales_per_expert,
                                            dns.size() * 2, cudaMemcpyDeviceToHost));
                last_e = e;
            }
            gemv_host(xt, gu.data(), gus.data(), K, 2 * I, L.bits, gu_out);
            for (int i = 0; i < I; ++i) {
                const float g = bf16r(static_cast<float>(gu_out[i]));
                const float u = bf16r(static_cast<float>(gu_out[I + i]));
                h[i] = bf16r(g / (1.f + std::exp(-g)) * u);
            }
            gemv_host(h, dn.data(), dns.data(), I, K, L.bits, d_out);
            const float w = r.wts_f[t * topk + slot];
            for (int n = 0; n < K; ++n)
                y[static_cast<size_t>(t) * K + n] += w * static_cast<float>(d_out[n]);
        }
    }
    return y;
}

std::vector<float> to_host_f32(const __nv_bfloat16* d, size_t n) {
    std::vector<__nv_bfloat16> h(n);
    SPARK_CUDA_CHECK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = __bfloat162float(h[i]);
    return out;
}

// --chain=N: also time N calls between one event pair, the calls queued back to back as a
// decode step queues its layers, so the launcher's host work (the scratch, the memset, three
// launches) overlaps the previous call's kernels instead of sitting inside the event pair.
int g_chain = 1;

bool run_t(cudaStream_t stream, const Layer& L, int T, const std::vector<int>& variants, int iters,
           int n_routings, bool repeat_routing) {
    const int K = L.K, I = L.I, topk = kTopK;
    const size_t nX = static_cast<size_t>(T) * K;
    std::vector<float> hx(nX);
    spark::bench::fill_uniform(hx, -1.0f, 1.0f, 1000 + T);
    std::vector<__nv_bfloat16> hxb(nX);
    for (size_t i = 0; i < nX; ++i) {
        hxb[i] = __float2bfloat16(hx[i]);
        hx[i] = __bfloat162float(hxb[i]);
    }
    __nv_bfloat16* dX = dmalloc<__nv_bfloat16>(nX);
    __nv_bfloat16* dY = dmalloc<__nv_bfloat16>(nX);
    SPARK_CUDA_CHECK(cudaMemcpy(dX, hxb.data(), nX * 2, cudaMemcpyHostToDevice));

    std::vector<Routing> routings;
    double mean_distinct = 0.0;
    for (int i = 0; i < n_routings; ++i) {
        routings.push_back(make_routing(T, topk, L.E, 77 * T + i));
        mean_distinct += routings.back().distinct;
    }
    mean_distinct /= n_routings;

    std::fprintf(stderr, "  T=%d bits=%d: reference on %d experts of the first routing...\n", T,
                 L.bits, routings[0].distinct);
    const auto ref = reference(L, routings[0], hx, T, topk);
    double max_ref = 0.0;
    for (float v : ref) max_ref = std::max(max_ref, std::fabs(static_cast<double>(v)));
    const double tol = 2e-2 * max_ref + 1e-3;

    // The active bytes of a step: the distinct experts, x in and y out.
    const double active = (repeat_routing ? routings[0].distinct : mean_distinct) *
                              static_cast<double>(L.bytes_per_expert()) +
                          2.0 * nX * 2;
    const double floor_gb10_us = active / kGB10Roof * 1e6, floor_5090_us = active / k5090Roof * 1e6;
    std::fprintf(stderr,
                 "  active set %.1f MB (%.1f distinct experts of %d): floor %.1f us at 273 GB/s, "
                 "%.1f us at 1,532 GB/s\n",
                 active / 1e6, repeat_routing ? routings[0].distinct : mean_distinct, L.E,
                 floor_gb10_us, floor_5090_us);

    const std::string shape = "T" + std::to_string(T) + "_k" + std::to_string(topk) + "of" +
                              std::to_string(L.E) + (repeat_routing ? "_repeat" : "");
    bool all_ok = true;
    for (int v : variants) {
        SPARK_CUDA_CHECK(cudaMemset(dY, 0, nX * 2));
        spark::w1moe_decode_bf16(dX, routings[0].d_ids, routings[0].d_wts, L.gu_packed, L.gu_scales,
                                 L.dn_packed, L.dn_scales, dY, T, topk, L.E, K, I, L.bits, v,
                                 stream);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto got = to_host_f32(dY, nX);
        const auto err = spark::bench::compare(got.data(), ref.data(), nX);
        const bool ok = err.max_abs <= tol;
        if (!ok)
            std::fprintf(stderr, "  FAIL variant %d bits=%d T=%d: max_abs=%.4e tol=%.4e\n", v,
                         L.bits, T, err.max_abs, tol);

        int turn = 0;
        const int it = v == 0 ? std::min(iters, 3) : iters;
        auto launch = [&](const Routing& r) {
            spark::w1moe_decode_bf16(dX, r.d_ids, r.d_wts, L.gu_packed, L.gu_scales, L.dn_packed,
                                     L.dn_scales, dY, T, topk, L.E, K, I, L.bits, v, stream);
        };
        auto t = spark::bench::time_kernel(
            [&] {
                if (repeat_routing) {
                    launch(routings[0]);
                    launch(routings[0]);
                } else {
                    launch(routings[turn++ % n_routings]);
                }
            },
            stream, 3, it);
        if (repeat_routing) {
            t.median_ms *= 0.5f;
            t.min_ms *= 0.5f;
        }

        spark::bench::Row row;
        row.kernel = "w1moe_decode";
        row.dtype = L.bits == 1 ? "w1a16" : "w2a16";
        row.variant = v;
        row.shape = shape;
        row.median_ms = t.median_ms;
        row.min_ms = t.min_ms;
        row.gbps = active / (t.median_ms * 1e-3) / 1e9;
        row.tflops = 2.0 * T * topk * 3.0 * K * I / (t.median_ms * 1e-3) / 1e12;
        row.max_abs_err = err.max_abs;
        row.max_rel_err = err.max_rel;
        row.ok = ok;
        spark::bench::print_row(row);
        std::fprintf(stderr,
                     "    %.1fx the 273 GB/s floor, %.1fx the 1,532 GB/s floor | 40 layers of this "
                     "bound %.0f tok/s at T=%d\n",
                     t.median_ms * 1e3 / floor_gb10_us, t.median_ms * 1e3 / floor_5090_us,
                     T * 1e3 / (40.0 * t.median_ms), T);
        if (g_chain > 1 && v > 0 && !repeat_routing) {
            const auto c = spark::bench::time_kernel(
                [&] {
                    for (int i = 0; i < g_chain; ++i) launch(routings[turn++ % n_routings]);
                },
                stream, 2, std::min(it, 20));
            const double us = c.median_ms * 1e3 / g_chain;
            std::fprintf(stderr,
                         "    chain of %d: %.1f us per call, %.0f GB/s, %.2fx the 1,532 floor\n",
                         g_chain, us, active / (us * 1e-6) / 1e9, us / floor_5090_us);
        }
        all_ok = all_ok && ok;
    }
    for (auto& r : routings) {
        SPARK_CUDA_CHECK(cudaFree(r.d_ids));
        SPARK_CUDA_CHECK(cudaFree(r.d_wts));
    }
    SPARK_CUDA_CHECK(cudaFree(dX));
    SPARK_CUDA_CHECK(cudaFree(dY));
    return all_ok;
}

}  // namespace

int run(int argc, char** argv) {
    spark::bench::Args args(argc, argv);
    spark::bench::print_device_banner();

    const int iters = args.geti("iters", 20);
    g_chain = std::max(1, args.geti("chain", 1));
    const int n_routings = std::max(1, args.geti("routings", 16));
    const bool repeat = args.has("repeat-routing");
    const int E = args.geti("experts", kExperts);
    std::vector<int> ts;
    if (args.has("t"))
        ts.push_back(args.geti("t", 1));
    else
        ts = {1, 4, 8};
    std::vector<int> bits_list;
    if (args.has("bits"))
        bits_list.push_back(args.geti("bits", 1));
    else
        bits_list = {1, 2};
    std::vector<int> variants;
    if (args.has("variant"))
        variants.push_back(args.geti("variant", 2));
    else
        for (int v = 0; v < spark::w1moe_decode_num_variants(); ++v) variants.push_back(v);

    cudaStream_t stream = nullptr;
    spark::bench::print_header();
    bool all_ok = true;
    for (int bits : bits_list) {
        Layer L = make_layer(E, kHidden, kInter, bits, stream);
        for (int T : ts)
            all_ok = run_t(stream, L, T, variants, iters, n_routings, repeat) && all_ok;
        L.free_all();
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
        std::fprintf(stderr, "bench_w1moe_decode: %s\n", e.what());
        return 2;
    }
}
