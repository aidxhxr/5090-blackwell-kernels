// The small serial pieces of a decode step on the CPU, and the probe that says what the
// Arm side of the GB10 can stream.
//
// Routing: DeepSeek V4.1 Flash picks 6 of 384 experts per token from a sigmoid over the
// router logits plus a per-expert bias (the bias only picks; the gate weights are the
// unbiased scores renormalized). On the GPU this is a handful of launches over 384 numbers,
// each a few microseconds of latency for no bandwidth; on a core it is 384 sigmoids and a
// partial selection of 6, which is what route_sigmoid_topk does: one pass to keep a sorted
// list of the top k by insertion (k is 6, a sort of 384 would be the wrong tool).
//
// Sampling: the GPU sampler (src/kernels/sample.cu) does an exact integer-weight radix
// select over a 129,280-wide row. The CPU version here is the plain floating-point form:
// temperature 0 is argmax, top-k is a partial selection (std::nth_element over an index
// array, a quickselect, not a sort), top-p sorts only what top-k kept and cuts by the mass
// before each tied value. The random draw is splitmix64 of (seed, offset) so a pair is
// reproducible here; it is a different stream from the GPU's Philox and the two do not
// agree token for token. The row is read once to widen bf16 to fp32 and find the max (NEON,
// 8 logits per step).
//
// The bandwidth probe: `threads` threads each stream their slice of a private buffer with
// 16-byte NEON loads, four independent XOR accumulators per thread so the loads are not
// elided and do not serialize on one register. A spin barrier brackets every sweep, so the
// time is the wall time of the slowest thread, as it would be for a GEMV split across the
// same cores. Pages are first-touched by the thread that will read them. The result over
// 1, 2, 4, 8, 10, 20 threads is the first number to read on the Spark: it is the fraction of
// the 273 GB/s the cores can pull without the GPU, and run concurrently with bench_w1gemm
// it says whether the two sides add or split (docs/design/grace.md).
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>

#include "spark/cpu.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace spark::cpu {
namespace {

inline float bf16_to_f32(bf16 h) {
    const uint32_t u = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

// bf16[n] -> fp32[n], returning the max and its first index.
void widen_row(const bf16* x, int n, float* out, float& max_value, int& max_index) {
    float m = -INFINITY;
    int mi = 0;
    int i = 0;
#if defined(__ARM_NEON)
    float32x4_t vm = vdupq_n_f32(-INFINITY);
    for (; i + 8 <= n; i += 8) {
        const uint16x8_t h = vld1q_u16(x + i);
        const float32x4_t lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(h), 16));
        const float32x4_t hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(h), 16));
        vst1q_f32(out + i, lo);
        vst1q_f32(out + i + 4, hi);
        vm = vmaxq_f32(vm, vmaxq_f32(lo, hi));
    }
    m = vmaxvq_f32(vm);
#endif
    for (; i < n; ++i) {
        out[i] = bf16_to_f32(x[i]);
        m = std::max(m, out[i]);
    }
    // The first index of the max, the way torch.argmax reports it.
    for (mi = 0; mi < n; ++mi) {
        if (out[mi] == m) break;
    }
    max_value = m;
    max_index = mi;
}

inline uint64_t splitmix64(uint64_t z) {
    z += 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// A reusable spin barrier for `n` participants.
struct SpinBarrier {
    explicit SpinBarrier(int n) : n_(n) {}
    void wait() {
        const int g = gen_.load(std::memory_order_acquire);
        if (count_.fetch_add(1, std::memory_order_acq_rel) + 1 == n_) {
            count_.store(0, std::memory_order_relaxed);
            gen_.store(g + 1, std::memory_order_release);
        } else {
            // Spin, yielding so an oversubscribed host (more threads than cores) still
            // makes progress; on the GB10 at 20 threads nobody yields.
            while (gen_.load(std::memory_order_acquire) == g) std::this_thread::yield();
        }
    }
    int n_;
    std::atomic<int> count_{0};
    std::atomic<int> gen_{0};
};

// One sweep over [p, p + bytes) (bytes a multiple of 64), folded to 8 bytes so the loads
// cannot be dropped.
uint64_t sweep(const uint8_t* p, size_t bytes) {
#if defined(__ARM_NEON)
    uint8x16_t a0 = vdupq_n_u8(0), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i < bytes; i += 64) {
        a0 = veorq_u8(a0, vld1q_u8(p + i));
        a1 = veorq_u8(a1, vld1q_u8(p + i + 16));
        a2 = veorq_u8(a2, vld1q_u8(p + i + 32));
        a3 = veorq_u8(a3, vld1q_u8(p + i + 48));
    }
    const uint8x16_t a = veorq_u8(veorq_u8(a0, a1), veorq_u8(a2, a3));
    return vgetq_lane_u64(vreinterpretq_u64_u8(a), 0) ^ vgetq_lane_u64(vreinterpretq_u64_u8(a), 1);
#else
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    const uint64_t* q = reinterpret_cast<const uint64_t*>(p);
    for (size_t i = 0; i < bytes / 8; i += 8) {
        a0 ^= q[i] ^ q[i + 1];
        a1 ^= q[i + 2] ^ q[i + 3];
        a2 ^= q[i + 4] ^ q[i + 5];
        a3 ^= q[i + 6] ^ q[i + 7];
    }
    return a0 ^ a1 ^ a2 ^ a3;
#endif
}

}  // namespace

// ---- routing ---------------------------------------------------------------------------------

void route_sigmoid_topk(const float* logits, const float* bias, int E, int top_k, float scale,
                        int32_t* ids, float* weights) {
    if (E <= 0 || top_k <= 0 || top_k > E) {
        throw std::invalid_argument("route_sigmoid_topk: need 1 <= top_k <= E");
    }
    // The pick values of the current top k, descending, and their ids. Insertion keeps the
    // list sorted; for k = 6 of 384 that is at most 6 moves per candidate that enters.
    float best[64];
    int32_t best_id[64];
    if (top_k > 64) throw std::invalid_argument("route_sigmoid_topk: top_k <= 64");
    int filled = 0;
    for (int e = 0; e < E; ++e) {
        const float s = 1.f / (1.f + std::exp(-logits[e]));
        const float v = bias ? s + bias[e] : s;
        if (filled == top_k && !(v > best[filled - 1])) continue;  // ties go to the lower index
        int pos = filled < top_k ? filled : filled - 1;
        while (pos > 0 && v > best[pos - 1]) {
            best[pos] = best[pos - 1];
            best_id[pos] = best_id[pos - 1];
            --pos;
        }
        best[pos] = v;
        best_id[pos] = e;
        if (filled < top_k) ++filled;
    }
    float sum = 0.f;
    for (int i = 0; i < top_k; ++i) {
        const float s = 1.f / (1.f + std::exp(-logits[best_id[i]]));
        weights[i] = s;
        ids[i] = best_id[i];
        sum += s;
    }
    const float denom = std::max(sum, 1.1754944e-38f);
    for (int i = 0; i < top_k; ++i) weights[i] = weights[i] / denom * scale;
}

// ---- sampling --------------------------------------------------------------------------------

int64_t sample_bf16(const bf16* logits, int V, float temperature, int top_k, float top_p,
                    uint64_t seed, uint64_t offset) {
    if (V <= 0) throw std::invalid_argument("sample_bf16: V must be >= 1");
    std::vector<float> x(static_cast<size_t>(V));
    float m;
    int argmax;
    widen_row(logits, V, x.data(), m, argmax);
    if (!(temperature > 0.f)) return argmax;

    // The kept set as indices: top_k largest by a partial selection, or everything.
    std::vector<int32_t> idx(static_cast<size_t>(V));
    std::iota(idx.begin(), idx.end(), 0);
    const auto greater = [&](int32_t i, int32_t j) {
        return x[i] > x[j] || (x[i] == x[j] && i < j);
    };
    int kept = V;
    if (top_k > 0 && top_k < V) {
        std::nth_element(idx.begin(), idx.begin() + (top_k - 1), idx.end(), greater);
        // Ties at the k-th value stay in, as on the GPU.
        const float kth = x[idx[top_k - 1]];
        kept = top_k;
        for (int i = top_k; i < V; ++i) {
            if (x[idx[i]] == kth) std::swap(idx[kept++], idx[i]);
        }
    }
    // Probabilities over the kept set (the max of the row is the max of the kept set).
    std::vector<float> p(static_cast<size_t>(kept));
    double Z = 0.0;
    for (int i = 0; i < kept; ++i) {
        p[i] = std::exp((x[idx[i]] - m) / temperature);
        Z += p[i];
    }
    if (top_p < 1.f) {
        // Descending by value; only what top-k kept is sorted.
        std::vector<int32_t> order(static_cast<size_t>(kept));
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(),
                  [&](int32_t i, int32_t j) { return greater(idx[i], idx[j]); });
        const double limit = std::max(top_p, 0.f) * Z;
        double before = 0.0;
        int n = 0;
        while (n < kept) {
            // A run of tied values is kept or dropped together.
            int r = n;
            double run = 0.0;
            while (r < kept && x[idx[order[r]]] == x[idx[order[n]]]) run += p[order[r++]];
            if (n > 0 && !(before < limit)) break;
            before += run;
            n = r;
        }
        std::vector<int32_t> idx2(static_cast<size_t>(n));
        std::vector<float> p2(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            idx2[i] = idx[order[i]];
            p2[i] = p[order[i]];
        }
        idx.swap(idx2);
        p.swap(p2);
        kept = n;
        Z = before;
    }
    const uint64_t r = splitmix64(seed ^ splitmix64(offset));
    const double u = static_cast<double>(r >> 11) * (1.0 / 9007199254740992.0);  // [0, 1)
    const double target = u * Z;
    double acc = 0.0;
    for (int i = 0; i < kept; ++i) {
        acc += p[i];
        if (acc > target) return idx[i];
    }
    return idx[kept - 1];
}

// ---- bandwidth probe -------------------------------------------------------------------------

double bandwidth_probe(size_t bytes, int threads, int iters) {
    threads = std::max(1, threads);
    iters = std::max(1, iters);
    bytes = (bytes / (static_cast<size_t>(threads) * 64)) * (static_cast<size_t>(threads) * 64);
    if (bytes == 0) throw std::invalid_argument("bandwidth_probe: buffer too small");
    std::unique_ptr<uint8_t[]> buf(new uint8_t[bytes]);
    const size_t slice = bytes / threads;

    SpinBarrier barrier(threads);
    std::vector<double> ms(static_cast<size_t>(iters), 0.0);
    std::atomic<uint64_t> sink{0};
    auto worker = [&](int t) {
        uint8_t* p = buf.get() + slice * t;
        std::memset(p, static_cast<int>(0x11 * (t + 1)), slice);  // first touch
        barrier.wait();
        uint64_t acc = sweep(p, slice);  // warm
        for (int it = 0; it < iters; ++it) {
            barrier.wait();
            const auto t0 = std::chrono::steady_clock::now();
            acc ^= sweep(p, slice);
            barrier.wait();
            if (t == 0) {
                ms[it] =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0)
                        .count();
            }
        }
        sink.fetch_xor(acc, std::memory_order_relaxed);
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < threads; ++t) pool.emplace_back(worker, t);
    worker(0);
    for (auto& th : pool) th.join();
    std::sort(ms.begin(), ms.end());
    const double median_ms = ms[ms.size() / 2];
    return static_cast<double>(bytes) / (median_ms * 1e-3) / 1e9;
}

}  // namespace spark::cpu
