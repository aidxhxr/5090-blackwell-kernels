// bench_cpu: the CPU side of a decode step on its own, no CUDA anywhere. Validates the NEON
// GEMV rungs against the scalar one and exits 1 on a mismatch, then times, over 1, 2, 4, 8,
// 10 and 20 threads:
//   * the 1-bit and ternary GEMV at the DeepSeek V4.1 Flash expert shapes (hidden 5120 to
//     inter 2304 for gate and up, 2304 back to 5120 for down) and the three matrices of one
//     expert back to back, GB/s of the packed bytes plus scales over the time,
//   * the sigmoid top-6-of-384 routing of one token,
//   * sampling over a 129,280-wide bf16 row: greedy, top-k 50 with top-p 0.9, and top-p alone,
//   * the bandwidth probe (16-byte NEON loads over a private buffer, default 1 GiB).
// A stderr table and one JSON object per row on stdout, in bench_common.hpp's row format
// (device, kernel, dtype, variant, shape, median_ms, min_ms, gbps, tflops, ref_ms, errors,
// ok) so scripts/make_results_table.py reads it like the GPU files; the helpers are a small
// CPU-only copy of that header's, since it includes the CUDA runtime.
//
// Flags: --iters=N (default 20), --threads=1,2,4,8,10,20, --probe-mb=1024, --device=NAME
// (the JSON device field; default "cpu" plus the hardware thread count), --quick (small
// shapes and a 64 MB probe, to shake out the build on any Arm machine). Not run on a GB10:
// every number this prints is from whatever host built it.
#include "spark/cpu.h"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

using spark::cpu::bf16;

// ---- a CPU-only copy of bench_common.hpp's row and argument helpers ------------------------

struct Row {
    std::string kernel;
    std::string dtype;
    int variant = -1;
    std::string shape;
    double median_ms = 0;
    double min_ms = 0;
    double gbps = 0;
    double tflops = 0;
    double ref_ms = 0;
    double max_abs_err = 0;
    double max_rel_err = 0;
    bool ok = true;
};

std::string g_device = "cpu";

void print_header() {
    std::fprintf(stderr, "%-14s %-5s %-3s %-24s %10s %10s %9s %9s %10s %10s %s\n", "kernel",
                 "dtype", "var", "shape", "median_ms", "min_ms", "GB/s", "TFLOPS", "ref_ms",
                 "max_abs", "ok");
}

void print_row(const Row& r) {
    std::fprintf(stderr, "%-14s %-5s %-3d %-24s %10.4f %10.4f %9.1f %9.2f %10.4f %10.2e %s\n",
                 r.kernel.c_str(), r.dtype.c_str(), r.variant, r.shape.c_str(), r.median_ms,
                 r.min_ms, r.gbps, r.tflops, r.ref_ms, r.max_abs_err, r.ok ? "yes" : "NO");
    std::printf(
        "{\"device\":\"%s\",\"kernel\":\"%s\",\"dtype\":\"%s\",\"variant\":%d,\"shape\":\"%s\","
        "\"median_ms\":%.6f,\"min_ms\":%.6f,\"gbps\":%.3f,\"tflops\":%.4f,\"ref_ms\":%.6f,\"max_"
        "abs_err\":%.6e,\"max_rel_err\":%.6e,"
        "\"ok\":%s}\n",
        g_device.c_str(), r.kernel.c_str(), r.dtype.c_str(), r.variant, r.shape.c_str(),
        r.median_ms, r.min_ms, r.gbps, r.tflops, r.ref_ms, r.max_abs_err, r.max_rel_err,
        r.ok ? "true" : "false");
    std::fflush(stdout);
}

struct Args {
    std::vector<std::pair<std::string, std::string>> kv;
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string s = argv[i];
            if (s.rfind("--", 0) != 0) continue;
            s = s.substr(2);
            const auto eq = s.find('=');
            if (eq == std::string::npos) {
                kv.emplace_back(s, "1");
            } else {
                kv.emplace_back(s.substr(0, eq), s.substr(eq + 1));
            }
        }
    }
    bool has(const std::string& k) const {
        for (auto& p : kv)
            if (p.first == k) return true;
        return false;
    }
    std::string get(const std::string& k, const std::string& def) const {
        for (auto& p : kv)
            if (p.first == k) return p.second;
        return def;
    }
    int geti(const std::string& k, int def) const {
        if (!has(k)) return def;
        const std::string v = get(k, "");
        char* end = nullptr;
        errno = 0;
        const long parsed = std::strtol(v.c_str(), &end, 10);
        if (v.empty() || end == nullptr || *end != '\0' || errno != 0) {
            std::fprintf(stderr, "--%s=%s: expected an integer\n", k.c_str(), v.c_str());
            std::exit(2);
        }
        return static_cast<int>(parsed);
    }
    std::vector<int> geti_list(const std::string& k, const std::vector<int>& def) const {
        if (!has(k)) return def;
        std::vector<int> out;
        std::string v = get(k, "");
        size_t start = 0;
        while (start <= v.size()) {
            const size_t comma = v.find(',', start);
            const std::string item = v.substr(start, comma == std::string::npos ? std::string::npos
                                                                                : comma - start);
            if (!item.empty()) out.push_back(std::atoi(item.c_str()));
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        return out;
    }
};

struct Timing {
    double median_ms = 0, min_ms = 0;
};

// Wall time of fn, warmup then iters samples.
Timing time_fn(const std::function<void()>& fn, int warmup, int iters) {
    for (int i = 0; i < warmup; ++i) fn();
    std::vector<double> ms;
    ms.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        fn();
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                         .count());
    }
    std::sort(ms.begin(), ms.end());
    return {ms[ms.size() / 2], ms.front()};
}

// ---- bf16 helpers and random data -------------------------------------------------------------

float bf16_to_f32(bf16 h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
bf16 f32_to_bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return static_cast<bf16>(u >> 16);
}

struct W1 {
    int N = 0, K = 0, bits = 1;
    std::vector<uint32_t> packed;
    std::vector<bf16> scales;
};

W1 random_weight(int N, int K, int bits, uint32_t seed) {
    W1 w;
    w.N = N;
    w.K = K;
    w.bits = bits;
    std::mt19937 rng(seed);
    w.packed.resize(static_cast<size_t>(N) * (K * bits / 32));
    for (auto& x : w.packed) x = rng();
    std::normal_distribution<float> nd(0.f, 1.f);
    w.scales.resize(static_cast<size_t>(K / 128) * N);
    // Scales the size a 1/sqrt(K) initialization's mean |w| would be.
    const float base = 0.8f / std::sqrt(static_cast<float>(K));
    for (auto& s : w.scales) s = f32_to_bf16(base * (1.f + 0.1f * nd(rng)));
    return w;
}

std::vector<bf16> random_row(int K, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<bf16> a(static_cast<size_t>(K));
    for (auto& x : a) x = f32_to_bf16(nd(rng));
    return a;
}

std::string shape_str(int N, int K) { return "N" + std::to_string(N) + "xK" + std::to_string(K); }

// ---- validation --------------------------------------------------------------------------------

bool validate(const W1& w, int variant, double tol_rel, Row& row) {
    const auto a = random_row(w.K, 7);
    std::vector<bf16> y_ref(w.N), y(w.N);
    spark::cpu::w1gemv_bf16(a.data(), w.packed.data(), w.scales.data(), y_ref.data(), w.N, w.K,
                            w.bits, 1, 0);
    spark::cpu::w1gemv_bf16(a.data(), w.packed.data(), w.scales.data(), y.data(), w.N, w.K, w.bits,
                            4, variant);
    double max_abs = 0, max_ref = 0;
    for (int n = 0; n < w.N; ++n) {
        const double r = bf16_to_f32(y_ref[n]);
        max_abs = std::max(max_abs, std::fabs(bf16_to_f32(y[n]) - r));
        max_ref = std::max(max_ref, std::fabs(r));
    }
    row.max_abs_err = max_abs;
    row.max_rel_err = max_abs / std::max(max_ref, 1e-6);
    row.ok = row.max_rel_err <= tol_rel;
    return row.ok;
}

}  // namespace

int main(int argc, char** argv) {
    Args args(argc, argv);
    const int iters = std::max(1, args.geti("iters", 20));
    const bool quick = args.has("quick");
    const std::vector<int> thread_counts = args.geti_list("threads", {1, 2, 4, 8, 10, 20});
    const int probe_mb = args.geti("probe-mb", quick ? 64 : 1024);
    const unsigned hw = std::thread::hardware_concurrency();
    g_device = args.get("device", "cpu-" + std::to_string(hw) + "t");
    std::fprintf(stderr, "device: %s | %u hardware threads | NEON %s | dotprod %s\n",
                 g_device.c_str(), hw,
#if defined(__ARM_NEON)
                 "yes",
#else
                 "no",
#endif
#if defined(__ARM_FEATURE_DOTPROD)
                 "yes"
#else
                 "no"
#endif
    );
    print_header();

    // DeepSeek V4.1 Flash: hidden 5120, expert intermediate 2304. gate and up are [K 5120,
    // N 2304], down is [K 2304, N 5120]. --quick shrinks both so the whole thing runs in
    // seconds on a laptop.
    const int hidden = quick ? 1280 : 5120;
    const int inter = quick ? 512 : 2304;
    const int nvar = spark::cpu::w1gemv_num_variants();

    // 1. Validate every built rung against the scalar one on the gate shape at both widths.
    bool all_ok = true;
    for (int bits = 1; bits <= 2; ++bits) {
        const W1 w = random_weight(inter, hidden, bits, 100 + bits);
        for (int v = 1; v < nvar; ++v) {
            if (!spark::cpu::w1gemv_supports(v)) continue;
            Row r;
            r.kernel = "cpu_w1gemv";
            r.dtype = bits == 1 ? "w1" : "w2";
            r.variant = v;
            r.shape = shape_str(inter, hidden) + " check";
            // Rung 1 differs from rung 0 by summation order only; rung 2 rounds the
            // activations to int8 per group.
            const bool ok = validate(w, v, v == 2 ? 5e-2 : 2e-2, r);
            print_row(r);
            all_ok = all_ok && ok;
        }
    }
    if (!all_ok) {
        std::fprintf(stderr, "cpu_w1gemv: validation FAILED\n");
        return 1;
    }

    // 2. The GEMV over thread counts, each rung, both widths, gate/up and down shapes.
    for (int bits = 1; bits <= 2; ++bits) {
        const W1 gate = random_weight(inter, hidden, bits, 200 + bits);
        const W1 down = random_weight(hidden, inter, bits, 300 + bits);
        const auto a_h = random_row(hidden, 11);
        const auto a_i = random_row(inter, 12);
        std::vector<bf16> y_i(inter), y_h(hidden);
        for (int v = 1; v < nvar; ++v) {
            if (!spark::cpu::w1gemv_supports(v)) continue;
            for (const int threads : thread_counts) {
                for (const W1* w : {&gate, &down}) {
                    const bf16* a = w->K == hidden ? a_h.data() : a_i.data();
                    bf16* y = w->N == inter ? y_i.data() : y_h.data();
                    const Timing t = time_fn(
                        [&] {
                            spark::cpu::w1gemv_bf16(a, w->packed.data(), w->scales.data(), y, w->N,
                                                    w->K, w->bits, threads, v);
                        },
                        3, iters);
                    Row r;
                    r.kernel = "cpu_w1gemv";
                    r.dtype = bits == 1 ? "w1" : "w2";
                    r.variant = v;
                    r.shape = shape_str(w->N, w->K) + " t" + std::to_string(threads);
                    r.median_ms = t.median_ms;
                    r.min_ms = t.min_ms;
                    r.gbps = static_cast<double>(spark::cpu::w1gemv_bytes(w->N, w->K, w->bits)) /
                             (t.median_ms * 1e-3) / 1e9;
                    print_row(r);
                }
                // One expert: gate, up (the gate weights stand in for up) and down.
                const Timing t = time_fn(
                    [&] {
                        spark::cpu::w1gemv_bf16(a_h.data(), gate.packed.data(), gate.scales.data(),
                                                y_i.data(), inter, hidden, bits, threads, v);
                        spark::cpu::w1gemv_bf16(a_h.data(), gate.packed.data(), gate.scales.data(),
                                                y_i.data(), inter, hidden, bits, threads, v);
                        spark::cpu::w1gemv_bf16(y_i.data(), down.packed.data(), down.scales.data(),
                                                y_h.data(), hidden, inter, bits, threads, v);
                    },
                    3, iters);
                Row r;
                r.kernel = "cpu_expert";
                r.dtype = bits == 1 ? "w1" : "w2";
                r.variant = v;
                r.shape = "3x" + shape_str(inter, hidden) + " t" + std::to_string(threads);
                r.median_ms = t.median_ms;
                r.min_ms = t.min_ms;
                r.gbps = static_cast<double>(2 * spark::cpu::w1gemv_bytes(inter, hidden, bits) +
                                             spark::cpu::w1gemv_bytes(hidden, inter, bits)) /
                         (t.median_ms * 1e-3) / 1e9;
                print_row(r);
            }
        }
    }

    // 3. Routing: one token, 384 experts, top 6, with a bias.
    {
        const int E = 384, k = 6;
        std::mt19937 rng(5);
        std::normal_distribution<float> nd(0.f, 1.f);
        std::vector<float> logits(E), bias(E), wts(k);
        std::vector<int32_t> ids(k);
        for (auto& x : logits) x = nd(rng);
        for (auto& x : bias) x = 0.1f * nd(rng);
        const int rep = quick ? 100 : 1000;
        const Timing t = time_fn(
            [&] {
                for (int i = 0; i < rep; ++i) {
                    logits[i % E] += 1e-3f;  // keep the compiler from hoisting the call
                    spark::cpu::route_sigmoid_topk(logits.data(), bias.data(), E, k, 2.5f,
                                                   ids.data(), wts.data());
                }
            },
            2, iters);
        Row r;
        r.kernel = "cpu_route";
        r.dtype = "f32";
        r.variant = 0;
        r.shape = "E384 k6";
        r.median_ms = t.median_ms / rep;
        r.min_ms = t.min_ms / rep;
        print_row(r);
    }

    // 4. Sampling over the vocabulary.
    {
        const int V = 129280;
        std::mt19937 rng(9);
        std::normal_distribution<float> nd(0.f, 2.f);
        std::vector<bf16> logits(V);
        for (auto& x : logits) x = f32_to_bf16(nd(rng));
        struct Case {
            const char* name;
            float temperature, top_p;
            int top_k;
        };
        const Case cases[] = {{"greedy", 0.f, 1.f, 0}, {"k50 p0.9", 0.7f, 0.9f, 50},
                              {"p0.95", 0.7f, 0.95f, 0}};
        uint64_t offset = 0;
        int64_t sink = 0;
        for (const Case& c : cases) {
            const Timing t = time_fn(
                [&] {
                    sink ^= spark::cpu::sample_bf16(logits.data(), V, c.temperature, c.top_k, c.top_p,
                                                    1234, offset++);
                },
                2, iters);
            Row r;
            r.kernel = "cpu_sample";
            r.dtype = "bf16";
            r.variant = 0;
            r.shape = std::string("V129280 ") + c.name;
            r.median_ms = t.median_ms;
            r.min_ms = t.min_ms;
            r.gbps = static_cast<double>(V) * 2 / (t.median_ms * 1e-3) / 1e9;
            print_row(r);
        }
        if (sink == -1) std::fprintf(stderr, "\n");
    }

    // 5. The bandwidth probe over the thread counts.
    {
        const size_t bytes = static_cast<size_t>(probe_mb) << 20;
        for (const int threads : thread_counts) {
            const double gbps = spark::cpu::bandwidth_probe(bytes, threads, quick ? 3 : 7);
            Row r;
            r.kernel = "cpu_bandwidth";
            r.dtype = "u8";
            r.variant = 0;
            r.shape = std::to_string(probe_mb) + "MB t" + std::to_string(threads);
            r.median_ms = static_cast<double>(bytes) / (gbps * 1e9) * 1e3;
            r.min_ms = r.median_ms;
            r.gbps = gbps;
            print_row(r);
        }
    }
    return 0;
}
