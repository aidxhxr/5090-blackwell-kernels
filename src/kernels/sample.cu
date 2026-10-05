// Token sampling from a batch of logit rows: temperature, top-k, top-p and one Philox draw per
// row, on the device, so a CUDA-graphed decode step can sample without the host. See
// docs/design/sample.md.
//
// Semantics (both variants, bit for bit; spark::sample_bf16 in kernels.h states them too):
//   * x = the row's bf16 logits as fp32, m = max(x), T = temperature[row]. T <= 0 (or a row
//     whose max is not finite) is greedy: the index of the first maximum, as torch.argmax.
//   * Weights w_i = floor(exp((x_i - m) / T) * 2^40), unsigned 64-bit fixed point. Every sum
//     below is an integer sum, so it does not depend on the order it is taken in: the token
//     is a function of (logits row, T, k, p, seed, offset) only. A token whose probability is
//     under 2^-40 of the most likely one's has weight 0 and is never drawn.
//   * Values are ordered by a 16-bit key of the bf16 bits (larger key = larger value).
//   * top-k (0 < k): keep the tokens whose key is >= the k-th largest key among the tokens
//     of nonzero weight (ties at the threshold are all kept). k = 0, or k at least the
//     number of nonzero weights, keeps everything.
//   * top-p (p < 1), on what top-k kept, Z its total weight: keep the tokens whose key is >= the
//     largest key v with W(key >= v) >= ceil(p Z), W the kept weight at or above a key. That
//     is the usual rule (sorted descending, keep a token while the mass strictly before it
//     is < p), with tied values kept or dropped together. p <= 0 keeps the top value only.
//   * Draw: r = floor(u Z_f / 2^64), u a 64-bit uniform from Philox4_32_10(seed[row]) at
//     block offset[row] (the first two 32-bit outputs), Z_f the kept total; the token is the
//     smallest index i whose running sum of kept weights (in index order) exceeds r.
//   * offset[row] += 1 on every call, greedy rows included, so graph replays draw fresh
//     numbers without the host touching the offsets.
//
// variant 0: one thread per row, bisection over the 16-bit key for each threshold (a full
//            pass over the row per probe). The reference.
// variant 1: one 1024-thread block per row: the max from DRAM in one pass, each threshold by a
//            two-pass radix select (8 bits per pass) over 256-bin count and weight histograms
//            whose passes read the row again from L2, then a scan of per-thread sums over
//            contiguous chunks to find the token. No sort.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <climits>
#include <cmath>
#include <cstdint>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {
namespace {

using u64 = unsigned long long;  // the type the 64-bit atomics and shuffles take

constexpr float kMassScale = 1099511627776.0f;  // 2^40
constexpr float kMinTemperature = 1e-6f;        // 1 / T stays finite
constexpr uint32_t kNone = 0xFFFFFFFFu;
constexpr int kMaxVocab = 1 << 23;  // V * 2^40 < 2^64

__device__ __forceinline__ float bits_to_f32(uint32_t b) {
    return __uint_as_float(b << 16);
}

// Order-preserving map of the bf16 bit pattern: unsigned compare of keys = float compare.
__device__ __forceinline__ uint32_t key16(uint32_t b) {
    return (b & 0x8000u) ? (~b & 0xFFFFu) : (b | 0x8000u);
}

__device__ __forceinline__ u64 weight(float f, float m, float inv_t) {
    return __float2ull_rz(__expf((f - m) * inv_t) * kMassScale);
}

// Element e (0..7) of a 16-byte vector of bf16, as raw bits.
__device__ __forceinline__ uint32_t lane_bits(const uint4& q, int e) {
    const uint32_t w = (e >> 1) == 0 ? q.x : (e >> 1) == 1 ? q.y : (e >> 1) == 2 ? q.z : q.w;
    return (w >> ((e & 1) * 16)) & 0xFFFFu;
}

// 64 random bits: the first two words of Philox4_32_10 block `offset` of stream `seed`.
__device__ __forceinline__ u64 philox_u64(int64_t seed, int64_t offset) {
    curandStatePhilox4_32_10_t st;
    curand_init(static_cast<u64>(seed), 0ull, static_cast<u64>(offset) * 4ull, &st);
    const uint4 r = curand4(&st);
    return (static_cast<u64>(r.x) << 32) | r.y;
}

// The threshold of the top-p rule from the total Z of what top-k kept.
__device__ __forceinline__ u64 top_p_target(float p, u64 z) {
    const double pp = static_cast<double>(fminf(fmaxf(p, 0.0f), 1.0f));
    u64 t = static_cast<u64>(ceil(pp * static_cast<double>(z)));
    if (t < 1) t = 1;
    if (t > z) t = z;
    return t;
}

// ---------------------------------------------------------------------------
// Variant 0: one thread per row.
// ---------------------------------------------------------------------------

// Count and weight of the nonzero-weight tokens whose key is >= v.
__device__ void naive_at_least(const unsigned short* xs, int V, float m, float inv_t, uint32_t v,
                               u64& cnt, u64& mass) {
    cnt = 0;
    mass = 0;
    for (int c = 0; c < V; ++c) {
        const uint32_t b = xs[c];
        if (key16(b) < v) continue;
        const u64 w = weight(bits_to_f32(b), m, inv_t);
        if (w == 0) continue;
        ++cnt;
        mass += w;
    }
}

// The largest key v with F(key >= v) >= target, F the count (by_mass false) or the weight of
// the nonzero-weight tokens. Needs F(key >= 0) >= target >= 1.
__device__ uint32_t naive_bisect(const unsigned short* xs, int V, float m, float inv_t,
                                 bool by_mass, u64 target) {
    uint32_t lo = 0, hi = 65536;  // F(>= lo) >= target, F(>= hi) < target
    while (hi - lo > 1) {
        const uint32_t mid = (lo + hi) / 2;
        u64 cnt, mass;
        naive_at_least(xs, V, m, inv_t, mid, cnt, mass);
        if ((by_mass ? mass : cnt) >= target) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

__global__ void sample_naive_kernel(const __nv_bfloat16* __restrict__ logits, int B, int V,
                                    const float* __restrict__ temperature,
                                    const int* __restrict__ top_k, const float* __restrict__ top_p,
                                    const int64_t* __restrict__ seed, int64_t* __restrict__ offset,
                                    int64_t* __restrict__ out) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B) return;
    const unsigned short* xs =
        reinterpret_cast<const unsigned short*>(logits) + static_cast<size_t>(row) * V;
    const u64 rnd = philox_u64(seed[row], offset[row]);
    offset[row] += 1;

    float m = -INFINITY;
    int amax = INT_MAX;
    for (int c = 0; c < V; ++c) {
        const float f = bits_to_f32(xs[c]);
        if (f > m || (f == m && c < amax)) {
            m = f;
            amax = c;
        }
    }
    const float T = temperature[row];
    if (!(T > 0.0f) || !(fabsf(m) < INFINITY)) {
        out[row] = amax;
        return;
    }
    const float inv_t = 1.0f / fmaxf(T, kMinTemperature);
    const int k = top_k[row];
    const float p = top_p[row];

    u64 npos, z;
    naive_at_least(xs, V, m, inv_t, 0, npos, z);
    uint32_t thr = 0;
    u64 zk = z;
    if (k > 0 && static_cast<u64>(k) < npos) {
        thr = naive_bisect(xs, V, m, inv_t, false, static_cast<u64>(k));
        u64 cnt;
        naive_at_least(xs, V, m, inv_t, thr, cnt, zk);
    }
    if (p < 1.0f) {
        const uint32_t tp = naive_bisect(xs, V, m, inv_t, true, top_p_target(p, zk));
        thr = tp > thr ? tp : thr;
    }
    u64 cnt, zf;
    naive_at_least(xs, V, m, inv_t, thr, cnt, zf);
    const u64 r = __umul64hi(rnd, zf);
    u64 acc = 0;
    for (int c = 0; c < V; ++c) {
        const uint32_t b = xs[c];
        if (key16(b) < thr) continue;
        acc += weight(bits_to_f32(b), m, inv_t);
        if (r < acc) {
            out[row] = c;
            return;
        }
    }
    out[row] = amax;  // unreachable: r < zf
}

// ---------------------------------------------------------------------------
// Variant 1: one block per row.
// ---------------------------------------------------------------------------
constexpr int kThreads = 1024;
constexpr int kWarps = kThreads / kWarpSize;
constexpr int kBins = 256;
constexpr int kUnroll = 4;  // 16-byte loads in flight per thread in the strided passes

struct Hist {
    uint32_t cnt[kBins];
    u64 mass[kBins];
};

__device__ __forceinline__ void argmax_merge(float& f, int& i, float f2, int i2) {
    if (f2 > f || (f2 == f && i2 < i)) {
        f = f2;
        i = i2;
    }
}

// The row's max and the index of its first occurrence, in every thread. One read of the row,
// coalesced 16-byte loads (kVec) or one element per thread per step.
template <bool kVec>
__device__ void block_argmax(const __nv_bfloat16* __restrict__ xr, int V, float* red_f, int* red_i,
                             float& m, int& amax) {
    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    const int warp = tid >> 5;
    float bf = -INFINITY;
    int bi = INT_MAX;
    if constexpr (kVec) {
        const int nvec = V / 8;
        const uint4* xv = reinterpret_cast<const uint4*>(xr);
        for (int v0 = tid; v0 < nvec; v0 += kUnroll * kThreads) {
            uint4 q[kUnroll];
#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                const int v = v0 + j * kThreads;
                q[j] = v < nvec ? xv[v] : make_uint4(0u, 0u, 0u, 0u);
            }
#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                const int v = v0 + j * kThreads;
                if (v < nvec) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        argmax_merge(bf, bi, bits_to_f32(lane_bits(q[j], e)), v * 8 + e);
                    }
                }
            }
        }
    } else {
        const unsigned short* xs = reinterpret_cast<const unsigned short*>(xr);
        for (int c = tid; c < V; c += kThreads) argmax_merge(bf, bi, bits_to_f32(xs[c]), c);
    }
#pragma unroll
    for (int o = kWarpSize / 2; o > 0; o >>= 1) {
        const float f2 = __shfl_xor_sync(kFullMask, bf, o);
        const int i2 = __shfl_xor_sync(kFullMask, bi, o);
        argmax_merge(bf, bi, f2, i2);
    }
    if (lane == 0) {
        red_f[warp] = bf;
        red_i[warp] = bi;
    }
    __syncthreads();
    if (warp == 0) {
        bf = red_f[lane];
        bi = red_i[lane];
#pragma unroll
        for (int o = kWarpSize / 2; o > 0; o >>= 1) {
            const float f2 = __shfl_xor_sync(kFullMask, bf, o);
            const int i2 = __shfl_xor_sync(kFullMask, bi, o);
            argmax_merge(bf, bi, f2, i2);
        }
        if (lane == 0) {
            red_f[0] = bf;
            red_i[0] = bi;
        }
    }
    __syncthreads();
    m = red_f[0];
    amax = red_i[0];
    __syncthreads();  // red_f / red_i may be reused
}

// Histogram of one radix digit over the nonzero-weight tokens: with hi == kNone the high byte
// of the key over every token, otherwise the low byte over the tokens whose high byte is hi.
// Counts and weights per bin. The lanes of a warp that share a digit are grouped with
// __match_any_sync and their weights summed with redux.sync, so a bin takes one shared atomic
// per group instead of one per lane (logits crowd into a few exponents, so the high-byte bins
// are hot). Integer atomics: the result does not depend on their order.
template <bool kVec>
__device__ void build_hist(const __nv_bfloat16* __restrict__ xr, int V, float m, float inv_t,
                           uint32_t hi, Hist& h) {
    const int tid = threadIdx.x;
    const int lane = tid & (kWarpSize - 1);
    for (int i = tid; i < kBins; i += kThreads) {
        h.cnt[i] = 0;
        h.mass[i] = 0;
    }
    __syncthreads();
    // Every lane of the warp calls this together (the loops below are warp-uniform).
    auto add = [&](bool ok, uint32_t bits) {
        const uint32_t key = key16(bits);
        const bool match = ok && (hi == kNone || (key >> 8) == hi);
        const u64 w = match ? weight(bits_to_f32(bits), m, inv_t) : 0ull;
        const bool take = w != 0;
        const uint32_t digit = take ? (hi == kNone ? key >> 8 : key & 0xFFu) : kNone;
        const unsigned peers = __match_any_sync(kFullMask, digit);
        // w <= 2^40: 21 + 20 bits, so 32 lanes of either half fit 32 bits
        const unsigned lo_sum = __reduce_add_sync(peers, static_cast<unsigned>(w & 0x1FFFFFull));
        const unsigned hi_sum = __reduce_add_sync(peers, static_cast<unsigned>(w >> 21));
        if (take && lane == __ffs(peers) - 1) {
            atomicAdd(&h.cnt[digit], static_cast<uint32_t>(__popc(peers)));
            atomicAdd(&h.mass[digit], (static_cast<u64>(hi_sum) << 21) + lo_sum);
        }
    };
    const int base = tid - lane;  // the warp's first thread: the loop bounds are warp-uniform
    if constexpr (kVec) {
        const int nvec = V / 8;
        const uint4* xv = reinterpret_cast<const uint4*>(xr);
        for (int b0 = base; b0 < nvec; b0 += kUnroll * kThreads) {
            uint4 q[kUnroll];
#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                const int v = b0 + j * kThreads + lane;
                q[j] = v < nvec ? xv[v] : make_uint4(0u, 0u, 0u, 0u);
            }
#pragma unroll
            for (int j = 0; j < kUnroll; ++j) {
                const bool ok = b0 + j * kThreads + lane < nvec;
#pragma unroll
                for (int e = 0; e < 8; ++e) add(ok, lane_bits(q[j], e));
            }
        }
    } else {
        const unsigned short* xs = reinterpret_cast<const unsigned short*>(xr);
        for (int b0 = base; b0 < V; b0 += kThreads) {
            const int c = b0 + lane;
            const bool ok = c < V;
            add(ok, ok ? static_cast<uint32_t>(xs[c]) : 0u);
        }
    }
    __syncthreads();
}

// One thread: walks the bins from the top and returns the largest d with
// acc + F[d] + F[d+1] + ... >= target, F the counts or the weights; acc_c and acc_m gain the
// bins above d. Needs acc + F[0..255] >= target.
__device__ int select_bin(const Hist& h, bool by_mass, u64 target, u64& acc_c, u64& acc_m) {
    for (int d = kBins - 1; d > 0; --d) {
        const u64 f = by_mass ? h.mass[d] : static_cast<u64>(h.cnt[d]);
        if ((by_mass ? acc_m : acc_c) + f >= target) return d;
        acc_c += h.cnt[d];
        acc_m += h.mass[d];
    }
    return 0;
}

// Visits this thread's contiguous chunk of the row in index order: f(index, bits) returns
// true to stop. Thread t's chunk comes before thread t + 1's, so a scan over the threads is a
// scan in index order.
template <bool kVec, typename F>
__device__ __forceinline__ void for_chunk(const __nv_bfloat16* __restrict__ xr, int V, F f) {
    const int tid = threadIdx.x;
    if constexpr (kVec) {
        const int nvec = V / 8;
        const int cv = cdiv(nvec, kThreads);
        const int end = min((tid + 1) * cv, nvec);
        const uint4* xv = reinterpret_cast<const uint4*>(xr);
        for (int v = tid * cv; v < end; ++v) {
            const uint4 q = xv[v];
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                if (f(v * 8 + e, lane_bits(q, e))) return;
            }
        }
    } else {
        const int cs = cdiv(V, kThreads);
        const int end = min((tid + 1) * cs, V);
        const unsigned short* xs = reinterpret_cast<const unsigned short*>(xr);
        for (int c = tid * cs; c < end; ++c) {
            if (f(c, static_cast<uint32_t>(xs[c]))) return;
        }
    }
}

// Exclusive scan of one u64 per thread in thread order; `total` gets the sum.
__device__ u64 block_exclusive_scan(u64 x, u64* ws, u64& total) {
    const int lane = threadIdx.x & (kWarpSize - 1);
    const int warp = threadIdx.x >> 5;
    u64 incl = x;
#pragma unroll
    for (int o = 1; o < kWarpSize; o <<= 1) {
        const u64 y = __shfl_up_sync(kFullMask, incl, o);
        if (lane >= o) incl += y;
    }
    if (lane == kWarpSize - 1) ws[warp] = incl;
    __syncthreads();
    if (warp == 0) {
        u64 t = ws[lane];  // kWarps == 32
#pragma unroll
        for (int o = 1; o < kWarpSize; o <<= 1) {
            const u64 y = __shfl_up_sync(kFullMask, t, o);
            if (lane >= o) t += y;
        }
        ws[lane] = t;
    }
    __syncthreads();
    total = ws[kWarps - 1];
    return (warp > 0 ? ws[warp - 1] : 0ull) + incl - x;
}

template <bool kVec>
__global__ void __launch_bounds__(kThreads)
    sample_block_kernel(const __nv_bfloat16* __restrict__ logits, int V,
                        const float* __restrict__ temperature, const int* __restrict__ top_k,
                        const float* __restrict__ top_p, const int64_t* __restrict__ seed,
                        int64_t* __restrict__ offset, int64_t* __restrict__ out) {
    static_assert(kWarps == kWarpSize, "the scan's second level is one warp");
    __shared__ Hist h1, h2;
    __shared__ float red_f[kWarps];
    __shared__ int red_i[kWarps];
    __shared__ u64 scan_ws[kWarps];
    __shared__ u64 s_rand, s_zk;
    __shared__ int s_d1, s_e1;
    __shared__ uint32_t s_thr;

    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* xr = logits + static_cast<size_t>(row) * V;
    if (tid == 0) {
        s_rand = philox_u64(seed[row], offset[row]);
        offset[row] += 1;
    }

    float m;
    int amax;
    block_argmax<kVec>(xr, V, red_f, red_i, m, amax);  // its syncs publish s_rand
    const float T = temperature[row];
    if (!(T > 0.0f) || !(fabsf(m) < INFINITY)) {  // the same in every thread
        if (tid == 0) out[row] = amax;
        return;
    }
    const float inv_t = 1.0f / fmaxf(T, kMinTemperature);
    const int k = top_k[row];
    const float p = top_p[row];
    const bool use_k = k > 0 && k < V;
    const bool use_p = p < 1.0f;

    uint32_t thr = 0;
    if (use_k || use_p) {
        build_hist<kVec>(xr, V, m, inv_t, kNone, h1);
        u64 acc_c = 0, acc_m = 0;  // thread 0's running sums above the chosen bins
        if (tid == 0) {
            u64 npos = 0, z = 0;
            for (int d = 0; d < kBins; ++d) {
                npos += h1.cnt[d];
                z += h1.mass[d];
            }
            s_zk = z;
            s_d1 = -1;
            s_thr = 0;
            if (use_k && static_cast<u64>(k) < npos) {
                s_d1 = select_bin(h1, false, static_cast<u64>(k), acc_c, acc_m);
            }
        }
        __syncthreads();
        const int d1 = s_d1;
        if (d1 >= 0) {  // top-k: the low byte among the tokens in bin d1
            build_hist<kVec>(xr, V, m, inv_t, static_cast<uint32_t>(d1), h2);
            if (tid == 0) {
                const int d2 = select_bin(h2, false, static_cast<u64>(k), acc_c, acc_m);
                s_thr = (static_cast<uint32_t>(d1) << 8) | static_cast<uint32_t>(d2);
                s_zk = acc_m + h2.mass[d2];
            }
            __syncthreads();
        }
        if (use_p) {
            u64 target = 0;
            if (tid == 0) {
                target = top_p_target(p, s_zk);
                acc_c = 0;
                acc_m = 0;
                s_e1 = select_bin(h1, true, target, acc_c, acc_m);
            }
            __syncthreads();
            const int e1 = s_e1;
            if (e1 != d1) {  // otherwise h2 already holds bin e1's low bytes
                build_hist<kVec>(xr, V, m, inv_t, static_cast<uint32_t>(e1), h2);
            }
            if (tid == 0) {
                const int e2 = select_bin(h2, true, target, acc_c, acc_m);
                const uint32_t tp = (static_cast<uint32_t>(e1) << 8) | static_cast<uint32_t>(e2);
                if (tp > s_thr) s_thr = tp;
            }
        }
        __syncthreads();
        thr = s_thr;
    }

    // Kept weight of this thread's chunk, a scan over the threads, and the one thread whose
    // range holds r walks its chunk again to find the token.
    u64 mine = 0;
    for_chunk<kVec>(xr, V, [&](int, uint32_t b) {
        if (key16(b) >= thr) mine += weight(bits_to_f32(b), m, inv_t);
        return false;
    });
    u64 total;
    const u64 before = block_exclusive_scan(mine, scan_ws, total);
    const u64 r = __umul64hi(s_rand, total);
    if (before <= r && r < before + mine) {
        u64 acc = before;
        for_chunk<kVec>(xr, V, [&](int c, uint32_t b) {
            if (key16(b) < thr) return false;
            acc += weight(bits_to_f32(b), m, inv_t);
            if (r < acc) {
                out[row] = c;
                return true;
            }
            return false;
        });
    }
}

constexpr int kNumVariants = 2;

}  // namespace

void sample_bf16(const __nv_bfloat16* logits, int B, int V, const float* temperature,
                 const int* top_k, const float* top_p, const int64_t* seed, int64_t* offset,
                 int64_t* out, int variant, cudaStream_t stream) {
    SPARK_REQUIRE(logits != nullptr && temperature != nullptr && top_k != nullptr &&
                      top_p != nullptr && seed != nullptr && offset != nullptr && out != nullptr,
                  "sample: null pointer");
    SPARK_REQUIRE(B >= 0 && V >= 1, "sample: B must be >= 0 and V >= 1");
    SPARK_REQUIRE(V <= kMaxVocab, "sample: V must be <= 2^23");
    SPARK_REQUIRE(variant >= 0 && variant < kNumVariants, "sample: variant out of range");
    if (B == 0) return;
    if (variant == 0) {
        const int threads = 128;
        sample_naive_kernel<<<cdiv(B, threads), threads, 0, stream>>>(
            logits, B, V, temperature, top_k, top_p, seed, offset, out);
    } else if (V % 8 == 0 && is_aligned16(logits)) {
        sample_block_kernel<true>
            <<<B, kThreads, 0, stream>>>(logits, V, temperature, top_k, top_p, seed, offset, out);
    } else {
        sample_block_kernel<false>
            <<<B, kThreads, 0, stream>>>(logits, V, temperature, top_k, top_p, seed, offset, out);
    }
    SPARK_CHECK_LAUNCH();
}

int sample_num_variants() {
    return kNumVariants;
}

}  // namespace spark
