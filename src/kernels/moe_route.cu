// MoE routing on the device: router logits -> each token's top-k experts, their softmax
// weights renormalized over the k, the per-expert row counts and offsets, and the row of
// expert order every (token, slot) goes to. Softmax top-k with renormalization is the routing
// of Mixtral (top-2 of 8) and of Qwen3-MoE (top-8 of 128, norm_topk_prob); without the
// renormalization it is the plain softmax top-k.
//
// One launch, `route_kernel`, does all of it but the last addition:
//   * a warp per token: the token's E logits (E <= 512) in registers, the fp32 softmax over all
//     E (max, sum of exp), then k rounds of a warp argmax over the logits, ties to the lower
//     expert id, so the experts come out in torch.topk's order (largest first). A round is two
//     redux.sync instructions: the max of the lanes' best logits as order-preserving u32 keys,
//     then the min expert id among the lanes holding it. The k weights are exp(l - max) / sum
//     over all E, divided by their own sum when renormalizing;
//   * the block's assignments in (token, slot) order are ranked within their expert by one
//     warp with __match_any_sync, 32 at a time, which gives every assignment its index among
//     the block's assignments to the same expert, and the block's per-expert counts;
//   * the last block to finish (one atomic counter, reset by that block) turns the counts of
//     every block into offsets[E + 1] and into base[b][e], the first row of expert order that
//     block b's assignments to expert e take. It stages the [nb][E] counts through shared
//     memory with coalesced loads, all in flight at once, and scans each expert's column
//     there. A routing of one block (up to 16 tokens) skips the counter.
// An assignment's row is then base[block][e] + rank: rows are sorted by expert, and within
// an expert by token, then slot, which is the stable sort of the flattened expert ids. That
// addition is done by whoever reads the routing next: `moe_positions` below, or the permute
// kernel (moe_permute.cu), which moves each token's row to its k places in the same launch.
//
// Every output is the same bits every run: no float atomics, and the integer counter only
// picks which block does the scan.

#include <algorithm>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

namespace mr {

constexpr int THREADS = 512;  // 16 warps: a routing of up to 16 tokens is one token per warp
constexpr int WARPS = THREADS / 32;
constexpr int MAX_E = 512;
constexpr int MAX_K = 16;
constexpr int MAX_BLOCKS = 128;       // route blocks, at most; and nb * E <= MAX_ASSIGN if it can
constexpr int MAX_ASSIGN = 8192;      // assignments per block (their experts sit in smem)
constexpr int TOKENS_PER_BLOCK = 16;  // the least, before the block count is capped

struct Params {
    int T, E, k, tb, nb;
    bool renorm;
    int32_t* ids;      // [T, k]
    float* weights;    // [T, k]
    int32_t* rank;     // [T, k]: index among the block's assignments to the same expert
    int32_t* base;     // [nb, E]: counts, then (last block) each block's first row per expert
    int32_t* offsets;  // [E + 1]
    unsigned* counter;
};

__device__ __forceinline__ float logit_f32(const float* p) {
    return __ldg(p);
}
__device__ __forceinline__ float logit_f32(const __nv_bfloat16* p) {
    return __bfloat162float(*p);
}

// A float as a u32 whose unsigned order is the float order (NaN aside); 0 is below -inf.
__device__ __forceinline__ unsigned order_key(float f) {
    const unsigned u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ __forceinline__ float key_float(unsigned k) {
    return __uint_as_float((k & 0x80000000u) ? (k & 0x7fffffffu) : ~k);
}

// The exclusive scan of the per-expert totals in s_cnt (one warp, 32 experts at a time) into
// offsets[0 .. E] and back into s_cnt.
__device__ __forceinline__ void offsets_scan(const Params& p, int32_t* s_cnt, int warp, int lane) {
    if (warp != 0) return;
    const int E = p.E;
    int run = 0;
    for (int base = 0; base < E; base += 32) {
        const int e = base + lane;
        const int c = e < E ? s_cnt[e] : 0;
        int x = c;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = __shfl_up_sync(kFullMask, x, o);
            if (lane >= o) x += y;
        }
        if (e < E) {
            p.offsets[e] = run + x - c;
            s_cnt[e] = run + x - c;
        }
        run += __shfl_sync(kFullMask, x, 31);
    }
    if (lane == 0) p.offsets[E] = run;
}

// J = logits per lane (E <= 32 J).
template <typename LT, int J>
__global__ void __launch_bounds__(THREADS) route_kernel(const LT* __restrict__ logits, Params p) {
    __shared__ int32_t s_ids[MAX_ASSIGN];
    __shared__ int32_t s_cnt[MAX_E + 1];
    __shared__ bool s_last;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int E = p.E, k = p.k;
    const int t0 = blockIdx.x * p.tb;
    const int t1 = min(p.T, t0 + p.tb);

    for (int e = tid; e < E; e += THREADS) s_cnt[e] = 0;

    // A warp's tokens one after another, the next token's logits loaded before the current
    // one's picks run.
    auto load = [&](int t, float(&dst)[J]) {
        const LT* row = logits + static_cast<size_t>(t) * E;
#pragma unroll
        for (int j = 0; j < J; ++j) {
            const int e = lane + 32 * j;
            dst[j] = e < E ? logit_f32(row + e) : -INFINITY;
        }
    };
    float vn[J];
    if (t0 + warp < t1) load(t0 + warp, vn);
    for (int t = t0 + warp; t < t1; t += WARPS) {
        float v[J];
#pragma unroll
        for (int j = 0; j < J; ++j) v[j] = vn[j];
        if (t + WARPS < t1) load(t + WARPS, vn);
        float m = v[0];
#pragma unroll
        for (int j = 1; j < J; ++j) m = fmaxf(m, v[j]);
        m = key_float(__reduce_max_sync(kFullMask, order_key(m)));
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < J; ++j)
            if (lane + 32 * j < E) s += expf(v[j] - m);
        s = warp_reduce_sum(s);

        unsigned taken = 0;  // bit j: expert lane + 32 j already picked
        int my_e = 0;
        float my_p = 0.f;
        for (int r = 0; r < k; ++r) {
            // The lane's best untaken logit (the lowest id among equals), as a key; 0 = none.
            unsigned bk = 0;
            int bi = 0x7fffffff;
#pragma unroll
            for (int j = 0; j < J; ++j) {
                const int e = lane + 32 * j;
                const unsigned key = order_key(v[j]);
                if (e < E && !((taken >> j) & 1u) && key > bk) {
                    bk = key;
                    bi = e;
                }
            }
            const unsigned top = __reduce_max_sync(kFullMask, bk);
            bi = static_cast<int>(
                __reduce_min_sync(kFullMask, bk == top ? static_cast<unsigned>(bi) : 0xffffffffu));
            const float bv = key_float(top);
            if ((bi & 31) == lane) taken |= 1u << (bi >> 5);
            if (lane == r) {
                my_e = bi;
                my_p = expf(bv - m) / s;
            }
        }
        float w = my_p;
        if (p.renorm) {
            const float sum = warp_reduce_sum(lane < k ? my_p : 0.f);
            w = my_p / sum;
        }
        if (lane < k) {
            const size_t i = static_cast<size_t>(t) * k + lane;
            p.ids[i] = my_e;
            p.weights[i] = w;
            s_ids[(t - t0) * k + lane] = my_e;
        }
    }
    __syncthreads();

    // Rank the block's assignments within their experts, in (token, slot) order.
    if (warp == 0) {
        const int n = (t1 - t0) * k;
        int32_t* rank = p.rank + static_cast<size_t>(t0) * k;
        for (int i0 = 0; i0 < n; i0 += 32) {
            const int i = i0 + lane;
            const int e = i < n ? s_ids[i] : E;  // E: a bin of its own for the idle lanes
            const unsigned peers = __match_any_sync(kFullMask, e);
            const int before = __popc(peers & ((1u << lane) - 1u));
            const int c = s_cnt[e];
            __syncwarp();
            if (i < n) rank[i] = c + before;
            if (lane == 31 - __clz(peers)) s_cnt[e] = c + __popc(peers);
            __syncwarp();
        }
    }
    __syncthreads();
    int32_t* cnt = p.base + static_cast<size_t>(blockIdx.x) * E;
    for (int e = tid; e < E; e += THREADS) cnt[e] = s_cnt[e];

    const int nb = p.nb;
    if (nb > 1) {
        // The last block scans the counts of all of them.
        __threadfence();
        __syncthreads();
        if (tid == 0) s_last = atomicAdd(p.counter, 1u) == static_cast<unsigned>(nb - 1);
        __syncthreads();
        if (!s_last) return;
        __threadfence();
    }

    if (nb * E <= MAX_ASSIGN) {
        // The counts through shared memory (s_ids is free now): coalesced loads all in
        // flight, a thread per expert down its column, the totals into s_cnt.
        int32_t* sc = s_ids;
        if (nb > 1) {
            for (int i = tid; i < nb * E; i += THREADS) sc[i] = __ldcg(p.base + i);
            __syncthreads();
            for (int e = tid; e < E; e += THREADS) {
                int run = 0;
                for (int b = 0; b < nb; ++b) {
                    const int c = sc[b * E + e];
                    sc[b * E + e] = run;
                    run += c;
                }
                s_cnt[e] = run;
            }
        } else {
            for (int e = tid; e < E; e += THREADS) sc[e] = 0;
        }
        __syncthreads();
        offsets_scan(p, s_cnt, warp, lane);
        __syncthreads();
        for (int i = tid; i < nb * E; i += THREADS) p.base[i] = sc[i] + s_cnt[i % E];
        if (tid == 0 && nb > 1) *p.counter = 0u;  // ready for the next launch
        return;
    }

    // Per expert: the exclusive prefix over blocks in place, the total into s_cnt.
    for (int e = tid; e < E; e += THREADS) {
        int run = 0;
        int b = 0;
        for (; b + 8 <= nb; b += 8) {
            int c[8];
#pragma unroll
            for (int u = 0; u < 8; ++u) c[u] = __ldcg(p.base + static_cast<size_t>(b + u) * E + e);
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                p.base[static_cast<size_t>(b + u) * E + e] = run;
                run += c[u];
            }
        }
        for (; b < nb; ++b) {
            const int c = __ldcg(p.base + static_cast<size_t>(b) * E + e);
            p.base[static_cast<size_t>(b) * E + e] = run;
            run += c;
        }
        s_cnt[e] = run;
    }
    __syncthreads();
    offsets_scan(p, s_cnt, warp, lane);
    __syncthreads();
    for (int i = tid; i < nb * E; i += THREADS) p.base[i] += s_cnt[i % E];
    if (tid == 0) *p.counter = 0u;  // ready for the next launch
}

// pos[i] = base[block of token i / k][ids[i]] + rank[i], in place over rank.
__global__ void positions_kernel(const int32_t* __restrict__ ids, int32_t* __restrict__ pos,
                                 const int32_t* __restrict__ base, int n, int k, int E, int tb) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const int b = (i / k) / tb;
        pos[i] += base[static_cast<size_t>(b) * E + ids[i]];
    }
}

// One workspace per process: base[nb][E] and the counter, zeroed once at allocation (the
// last block of every launch puts the counter back to 0).
struct Workspace {
    int32_t* base = nullptr;
    unsigned* counter = nullptr;
    size_t cap = 0;
};

template <typename LT>
void launch_route(const LT* logits, const Params& p, cudaStream_t stream) {
    if (p.E <= 32)
        route_kernel<LT, 1><<<p.nb, THREADS, 0, stream>>>(logits, p);
    else if (p.E <= 128)
        route_kernel<LT, 4><<<p.nb, THREADS, 0, stream>>>(logits, p);
    else
        route_kernel<LT, 16><<<p.nb, THREADS, 0, stream>>>(logits, p);
    SPARK_CHECK_LAUNCH();
}

}  // namespace mr

}  // namespace

bool moe_route_supports(int E, int k) {
    return E >= 1 && E <= mr::MAX_E && k >= 1 && k <= mr::MAX_K && k <= E;
}

MoeRoutePlan moe_route(const void* logits, bool logits_fp32, int T, int E, int k, bool renorm,
                       int32_t* ids, float* weights, int32_t* rank, int32_t* offsets,
                       cudaStream_t stream) {
    SPARK_REQUIRE(moe_route_supports(E, k), "moe_route: requires 1 <= k <= min(E, 16), E <= 512");
    SPARK_REQUIRE(logits && ids && weights && rank && offsets, "moe_route: null pointer");
    SPARK_REQUIRE(T >= 1, "moe_route: needs at least one token");
    // Blocks of at least 16 tokens, at most 128 blocks and at most 8192 / E of them (so the
    // last block's scan fits shared memory) if the 8192 assignments per block allow it.
    const int cap = std::max(1, std::min(mr::MAX_BLOCKS, mr::MAX_ASSIGN / E));
    int nb = std::min(cdiv(T, mr::TOKENS_PER_BLOCK), cap);
    int tb = cdiv(T, nb);
    if (tb * k > mr::MAX_ASSIGN) tb = mr::MAX_ASSIGN / k;
    nb = cdiv(T, tb);

    static mr::Workspace w;
    const size_t need = static_cast<size_t>(nb) * E;
    if (w.cap < need) {
        if (w.base) SPARK_CUDA_CHECK(cudaFree(w.base));
        SPARK_CUDA_CHECK(cudaMalloc(&w.base, need * sizeof(int32_t)));
        w.cap = need;
    }
    if (!w.counter) {
        SPARK_CUDA_CHECK(cudaMalloc(&w.counter, sizeof(unsigned)));
        SPARK_CUDA_CHECK(cudaMemset(w.counter, 0, sizeof(unsigned)));  // once
    }

    mr::Params p;
    p.T = T;
    p.E = E;
    p.k = k;
    p.tb = tb;
    p.nb = nb;
    p.renorm = renorm;
    p.ids = ids;
    p.weights = weights;
    p.rank = rank;
    p.base = w.base;
    p.offsets = offsets;
    p.counter = w.counter;
    if (logits_fp32)
        mr::launch_route(static_cast<const float*>(logits), p, stream);
    else
        mr::launch_route(static_cast<const __nv_bfloat16*>(logits), p, stream);

    MoeRoutePlan plan;
    plan.base = w.base;
    plan.tb = tb;
    plan.nb = nb;
    return plan;
}

void moe_positions(const int32_t* ids, int32_t* pos, const MoeRoutePlan& plan, int T, int E, int k,
                   cudaStream_t stream) {
    const int n = T * k;
    if (n <= 0) return;
    const int threads = 256;
    const int grid = std::min(cdiv(n, threads), 4 * num_sms());
    mr::positions_kernel<<<grid, threads, 0, stream>>>(ids, pos, plan.base, n, k, E, plan.tb);
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
