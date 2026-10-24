// Fused batch-1 (and up to 8 token) MoE expert step over 1-bit or ternary experts:
//   for every token t and every chosen expert e = expert_ids[t, slot]
//       h = silu(x[t] @ Wgate[e]) * (x[t] @ Wup[e])          [I]
//       y[t] += expert_weights[t, slot] * (h @ Wdown[e])      [K]
// with bf16 x and y, the weights in the w1gemm format (one sign or ternary code per weight,
// a bf16 scale per 128 k of a column), fp32 accumulation. The shared expert and the router
// are not here: the caller routes (python/spark_kernels/moe.py) and adds the shared expert.
//
// Variants:
//   0  three plain passes: gate|up GEMV per (token, slot) into a global [T topk, 2I] buffer,
//      a SwiGLU pass into [T topk, I], a down GEMV that sums the slots with their routing
//      weights. Scalar dequant. The baseline the others are checked against.
//   1  one persistent kernel. A work item is (expert, slice of 128 intermediate columns):
//      the block runs the gate and up GEMVs of its slice for every token routed to that
//      expert (the weight bytes are read once per expert, not once per token), the SwiGLU
//      in shared memory, then the down projection of its 128 rows of Wdown over all K
//      output columns, added into an fp32 y with atomics. The intermediate never leaves the
//      SM. The routing weight is applied to h in shared memory.
//   2  variant 1 with the routing weight folded into the down epilogue (h stays the bf16
//      value the sorted path uses) and the first groups of the next item's gate|up rows
//      loaded into registers before the SwiGLU and the down pass, so the weight stream
//      does not drain at the item boundary.
//   3  the same item on the tensor cores with 16 warps a block: a warp per 16-column gate or
//      up strip over the whole K on w1gemm variant 3's lane-word body, the SwiGLU through
//      shared memory, the down pass as one-group strips with h as the B tile (see below).
//
// Why CUDA cores and not mma.sync: at T <= 8 this is a byte stream with almost no reuse.
// Per weight bit the inner loop spends one bit test and T predicated fp32 adds (the sign is
// applied by adding or not: sum(+-x) = 2 sum_{bit set} x - sum x, with sum x per group
// computed once per token), so about 1 + 1.25 T lane instructions per weight at 1 bit
// including the activation reads from shared memory. The DRAM floor gives each SM
// 273 GB/s / 48 = 5.7 GB/s = 45 G weights/s at 1 bit on the GB10 and the SM issues about
// 4 x 32 x 2.4 G = 307 G lane instructions/s, so there are 6.7 instructions per weight to
// spend: T = 1 uses a third of them, T = 4 all of them, and T = 8 is issue-bound by about
// 1.6x on paper. On the 5090 (1,532 GB/s over 170 SMs, 4.3 instructions per weight) the
// same holds up to T = 2. Within that range the tensor-core strip of w1gemm would only add
// fragment plumbing for weights that are each used once; past it, w1gemm's strip_chunk is
// the obvious third pass. Nothing here has run on a GPU; these are instruction counts.
//
// Work decomposition (variants 1 and 2): items are (pair, slice) with pair = t * topk + slot
// in row-major order and slice = i0 / 128; the block that gets a pair owns it only if it is
// the first pair with that expert, so an expert two tokens share is done once with both
// tokens (the other pair's items are skipped, which costs a scan of the <= 64 ids). The
// block's token count ne for the expert picks a template width TP in {1, 2, 4, 8} so a
// one-token expert does not pay for eight accumulators. The activations of the block's
// tokens are staged in shared memory as fp32 in chunks of 8192 / TP k, laid out [k][TP] so
// one 16-byte read gives the k-th activation of four tokens, warp-uniform (a broadcast).
// 256 threads: in the gate|up pass thread i < 128 owns gate column i0 + i and thread
// 128 + i owns up column I + i0 + i, each streaming its own 16-byte-per-group row of packed
// codes (K bits / 8 bytes long, contiguous) with the next kPF groups in flight; in the down
// pass thread j owns output columns j, j + 256, ..., reading the 16 bytes (32 at 2 bits) of
// its slice from each row, two rows in flight. The down bytes of one row come as half a
// 32-byte sector per block, the other half belonging to the neighbouring slice, so the
// stream relies on L2 to merge the two blocks' reads; the slices of one expert are
// consecutive items and run together.
//
// Layout of the gate|up weight: MoEBlock1Bit's, gate columns 0..I-1 then up columns
// I..2I-1 of a [K, 2I] matrix per expert (the pairing (i, I + i); interleave_gate_up's
// (2i, 2i + 1) would be a change to two row offsets). packed [E, 2I, K bits / 32],
// scales [E, K / 128, 2I]; down packed [E, K, I bits / 32], scales [E, I / 128, K].
#include <algorithm>
#include <cstdint>
#include <mutex>

#include "spark/common.cuh"
#include "spark/kernels.h"
#include "w1gemm_internal.cuh"

namespace spark {

namespace {

constexpr int kG = kW1GroupSize;  // 128 k per scale, and the slice of I per item
constexpr int kThreads = 256;     // 128 gate columns + 128 up columns of the slice
constexpr int kMaxPairs = 64;     // T * topk
constexpr int kMaxTok = 8;        // T
constexpr int kMaxGroups = 128;   // K / 128 (K <= 16384)
constexpr int kXFloats = 8192;    // the fp32 activation stage: 32 KB
constexpr int kPF = 4;            // groups of a column in flight ahead of the compute

// The dequantized weight of one code, as the kernels and the reference see it.
__device__ __forceinline__ float dequant_scalar(int code, float s, int bits) {
    const float v = bits == 1 ? (code ? s : -s) : s * static_cast<float>(code - 1);
    return __bfloat162float(__float2bfloat16(v));
}

__device__ __forceinline__ float bf16_round(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

__device__ __forceinline__ float silu(float g) {
    return g / (1.f + __expf(-g));
}

// ---- variant 0: three plain passes -----------------------------------------------------

// gu[pair, n] = x[t] . dequant(Wgu[e])[:, n], one thread per output column, scalar.
__global__ void v0_gate_up_kernel(const __nv_bfloat16* __restrict__ x,
                                  const int32_t* __restrict__ ids,
                                  const uint32_t* __restrict__ packed,
                                  const __nv_bfloat16* __restrict__ scales,
                                  __nv_bfloat16* __restrict__ gu, int topk, int K, int I,
                                  int bits) {
    const int pair = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int N = 2 * I;
    if (n >= N) return;
    const int e = ids[pair];
    const int t = pair / topk;
    const uint32_t* pe = packed + static_cast<size_t>(e) * N * w1::words_per_row(K, bits);
    const __nv_bfloat16* se = scales + static_cast<size_t>(e) * (K / kG) * N;
    const __nv_bfloat16* xt = x + static_cast<size_t>(t) * K;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        const float s = __bfloat162float(se[static_cast<size_t>(k / kG) * N + n]);
        const float w = dequant_scalar(w1::code_at(pe, n, k, K, bits), s, bits);
        acc = fmaf(__bfloat162float(xt[k]), w, acc);
    }
    gu[static_cast<size_t>(pair) * N + n] = __float2bfloat16(acc);
}

// h[r, i] = bf16(silu(gate) * up) over rows of [2I] gate-then-up values.
__global__ void v0_swiglu_kernel(const __nv_bfloat16* __restrict__ gu,
                                 __nv_bfloat16* __restrict__ h, int rows, int I) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= rows * I) return;
    const int r = idx / I, i = idx % I;
    const float g = __bfloat162float(gu[static_cast<size_t>(r) * 2 * I + i]);
    const float u = __bfloat162float(gu[static_cast<size_t>(r) * 2 * I + I + i]);
    h[idx] = __float2bfloat16(silu(g) * u);
}

// y[t, n] = sum over slots of w[t, slot] * bf16(h[t slot] . dequant(Wdown[e])[:, n]).
__global__ void v0_down_kernel(const __nv_bfloat16* __restrict__ h, const int32_t* __restrict__ ids,
                               const __nv_bfloat16* __restrict__ wts,
                               const uint32_t* __restrict__ packed,
                               const __nv_bfloat16* __restrict__ scales,
                               __nv_bfloat16* __restrict__ y, int topk, int K, int I, int bits) {
    const int t = blockIdx.y;
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= K) return;
    float out = 0.f;
    for (int slot = 0; slot < topk; ++slot) {
        const int pair = t * topk + slot;
        const int e = ids[pair];
        const uint32_t* pe = packed + static_cast<size_t>(e) * K * w1::words_per_row(I, bits);
        const __nv_bfloat16* se = scales + static_cast<size_t>(e) * (I / kG) * K;
        const __nv_bfloat16* hr = h + static_cast<size_t>(pair) * I;
        float acc = 0.f;
        for (int i = 0; i < I; ++i) {
            const float s = __bfloat162float(se[static_cast<size_t>(i / kG) * K + n]);
            const float w = dequant_scalar(w1::code_at(pe, n, i, I, bits), s, bits);
            acc = fmaf(__bfloat162float(hr[i]), w, acc);
        }
        out = fmaf(__bfloat162float(wts[pair]), bf16_round(acc), out);
    }
    y[static_cast<size_t>(t) * K + n] = __float2bfloat16(out);
}

// ---- variants 1 and 2: the fused persistent kernel ------------------------------------------

// One work item's expert and its tokens, resolved by the block into shared memory.
struct Item {
    int id;              // item index, >= n_items when there is nothing left
    int e;               // expert
    int ne;              // tokens routed to it (1..T)
    int slice;           // i0 / 128
    int toks[kMaxTok];   // their rows of x and y, ascending
    float wts[kMaxTok];  // their routing weights
};

struct Smem {
    alignas(16) float x_s[kXFloats];      // activations of the chunk, [k][TP] fp32
    alignas(16) float h_s[kG * kMaxTok];  // the slice of the intermediate, [i][TP]
    alignas(16) float u_s[kG * kMaxTok];  // the up values handed to the gate threads
    float x_all[kMaxGroups * kMaxTok];    // sum of x over each 128-k group, [grp][8]
    float h_part[4 * kMaxTok];            // per-warp partial sums of h, [warp][t]
    float h_all[kMaxTok];                 // sum of h over the slice, per token
    Item item[2];
    int owned;
};

template <int TP>
__device__ __forceinline__ void load_tp(const float* p, float (&v)[TP]) {
    if constexpr (TP == 1) {
        v[0] = p[0];
    } else if constexpr (TP == 2) {
        const float2 f = *reinterpret_cast<const float2*>(p);
        v[0] = f.x;
        v[1] = f.y;
    } else {
#pragma unroll
        for (int q = 0; q < TP / 4; ++q) {
            const float4 f = reinterpret_cast<const float4*>(p)[q];
            v[4 * q] = f.x;
            v[4 * q + 1] = f.y;
            v[4 * q + 2] = f.z;
            v[4 * q + 3] = f.w;
        }
    }
}

// The 16 BITS bytes of one 128-k group of one packed row (an output column).
template <int BITS>
struct Group {
    uint4 v[BITS];
    __device__ __forceinline__ void load(const uint32_t* row, int grp) {
        const uint4* p = reinterpret_cast<const uint4*>(row) + grp * BITS;
#pragma unroll
        for (int b = 0; b < BITS; ++b) v[b] = __ldg(p + b);
    }
    __device__ __forceinline__ unsigned word(int w) const {
        const uint4& u = v[w >> 2];
        const int j = w & 3;
        return j == 0 ? u.x : j == 1 ? u.y : j == 2 ? u.z : u.w;
    }
};

// acc[t] += s * sum_k code(k) a[k][t] over the 128 k of one group whose activations are at
// xs[k * TP + t] and whose plain sums are all[t]. 1 bit: sum(+-a) = 2 sum_{bit} a - sum a.
// Ternary: code - 1 = 2 b1 + b0 - 1, so the sum is 2 sum_{b1} a + sum_{b0} a - sum a. One
// bit test and TP predicated adds per code; nothing is multiplied until the scale.
template <int BITS, int TP>
__device__ __forceinline__ void accumulate_group(const Group<BITS>& g, const float* xs,
                                                 const float* all, float s, float (&acc)[TP]) {
    float hi[TP], lo[TP];
#pragma unroll
    for (int t = 0; t < TP; ++t) hi[t] = lo[t] = 0.f;
    constexpr int per_word = 32 / BITS;
#pragma unroll
    for (int w = 0; w < 4 * BITS; ++w) {
        const unsigned q = g.word(w);
#pragma unroll
        for (int b = 0; b < per_word; ++b) {
            float a[TP];
            load_tp<TP>(xs + (w * per_word + b) * TP, a);
            if constexpr (BITS == 1) {
                if ((q >> b) & 1u) {
#pragma unroll
                    for (int t = 0; t < TP; ++t) hi[t] += a[t];
                }
            } else {
                const unsigned c = (q >> (2 * b)) & 3u;
                if (c & 2u) {
#pragma unroll
                    for (int t = 0; t < TP; ++t) hi[t] += a[t];
                }
                if (c & 1u) {
#pragma unroll
                    for (int t = 0; t < TP; ++t) lo[t] += a[t];
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < TP; ++t) {
        const float sum = BITS == 1 ? 2.f * hi[t] - all[t] : 2.f * hi[t] + lo[t] - all[t];
        acc[t] = fmaf(s, sum, acc[t]);
    }
}

// Stage x[toks[t], k0 .. k0 + nk) of the item's ne tokens as fp32 [k][TP] (t >= ne zero),
// then the sum over each 128-k group into x_all[grp][t].
template <int TP>
__device__ __forceinline__ void stage_x(const __nv_bfloat16* __restrict__ x, const Item& it, int K,
                                        int k0, int nk, float* x_s, float* x_all) {
    const int vecs = nk / 8;  // 16-byte loads per token
    for (int idx = threadIdx.x; idx < TP * vecs; idx += kThreads) {
        const int t = idx / vecs, v = idx % vecs;
        float a[8];
        if (t < it.ne) {
            const uint4 u = __ldg(reinterpret_cast<const uint4*>(
                x + static_cast<size_t>(it.toks[t]) * K + k0 + 8 * v));
            const unsigned w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                a[2 * j] = __uint_as_float(w[j] << 16);
                a[2 * j + 1] = __uint_as_float(w[j] & 0xFFFF0000u);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) a[j] = 0.f;
        }
#pragma unroll
        for (int j = 0; j < 8; ++j) x_s[(8 * v + j) * TP + t] = a[j];
    }
    __syncthreads();
    const int ng = nk / kG;
    for (int idx = threadIdx.x; idx < ng * TP; idx += kThreads) {
        const int g = idx / TP, t = idx % TP;
        float sum = 0.f;
        const float* p = x_s + static_cast<size_t>(g) * kG * TP + t;
#pragma unroll 8
        for (int k = 0; k < kG; ++k) sum += p[k * TP];
        x_all[(k0 / kG + g) * kMaxTok + t] = sum;
    }
    __syncthreads();
}

// Resolve item `id` for the block: its expert, whether this block owns it (the pair is the
// first with that expert) and the expert's tokens. Advances over unowned items. Block-wide.
__device__ void resolve_item(Smem& sm, Item& it, int id, int n_items, int n_slices, int T, int topk,
                             const int32_t* __restrict__ ids,
                             const __nv_bfloat16* __restrict__ wts) {
    const int n_pairs = T * topk;
    for (;;) {
        if (id >= n_items) {
            if (threadIdx.x == 0) it.id = id;
            __syncthreads();
            return;
        }
        const int pair = id / n_slices;
        if (threadIdx.x < 32) {
            const int e = ids[pair];
            int first = n_pairs;
            for (int j = threadIdx.x; j < n_pairs; j += 32)
                if (ids[j] == e) first = min(first, j);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1)
                first = min(first, __shfl_xor_sync(kFullMask, first, o));
            if (threadIdx.x == 0) {
                sm.owned = first == pair;
                if (sm.owned) {
                    it.id = id;
                    it.e = e;
                    it.slice = id % n_slices;
                    int ne = 0;
                    for (int j = 0; j < n_pairs; ++j)
                        if (ids[j] == e) {
                            it.toks[ne] = j / topk;
                            it.wts[ne] = __bfloat162float(wts[j]);
                            ++ne;
                        }
                    it.ne = ne;
                }
            }
        }
        __syncthreads();
        const bool owned = sm.owned;
        __syncthreads();
        if (owned) return;
        id += gridDim.x;
    }
}

// The first kPF groups of this thread's gate|up row of item `it` into the ring.
template <int BITS>
__device__ __forceinline__ void prefetch_gate_up(const uint32_t* __restrict__ packed_gu,
                                                 const Item& it, int K, int I,
                                                 Group<BITS> (&ring)[kPF]) {
    const int G = K / kG;
    const int wpr = w1::words_per_row(K, BITS);
    const int col =
        threadIdx.x < kG ? it.slice * kG + threadIdx.x : I + it.slice * kG + threadIdx.x - kG;
    const uint32_t* row = packed_gu + (static_cast<size_t>(it.e) * 2 * I + col) * wpr;
#pragma unroll
    for (int d = 0; d < kPF; ++d)
        if (d < G) ring[d].load(row, d);
}

// One item: the gate|up GEMVs of the slice, the SwiGLU, the down projection of the slice.
// FOLD: the routing weight goes into the down epilogue (variant 2) instead of into h.
template <int BITS, int TP, bool FOLD>
__device__ __forceinline__ void run_item(
    Smem& sm, const Item& it, const __nv_bfloat16* __restrict__ x,
    const uint32_t* __restrict__ packed_gu, const __nv_bfloat16* __restrict__ scales_gu,
    const uint32_t* __restrict__ packed_dn, const __nv_bfloat16* __restrict__ scales_dn,
    float* __restrict__ yacc, int K, int I, Group<BITS> (&ring)[kPF], const Item* next,
    int next_valid) {
    const int tid = threadIdx.x;
    const int G = K / kG;
    const int N = 2 * I;
    const int wpr = w1::words_per_row(K, BITS);
    const bool is_gate = tid < kG;
    const int i_local = is_gate ? tid : tid - kG;
    const int col = is_gate ? it.slice * kG + tid : I + it.slice * kG + i_local;
    const uint32_t* row = packed_gu + (static_cast<size_t>(it.e) * N + col) * wpr;
    const __nv_bfloat16* scol = scales_gu + static_cast<size_t>(it.e) * G * N + col;

    // ---- gate|up: this thread's column over the whole K, the ring kPF groups ahead ----
    float acc[TP];
#pragma unroll
    for (int t = 0; t < TP; ++t) acc[t] = 0.f;
    constexpr int chunk_groups = kXFloats / TP / kG;
    for (int g0 = 0; g0 < G; g0 += chunk_groups) {
        const int ng = min(chunk_groups, G - g0);
        __syncthreads();  // the previous chunk's activations are consumed
        stage_x<TP>(x, it, K, g0 * kG, ng * kG, sm.x_s, sm.x_all);
        for (int gi = 0; gi < ng; gi += kPF) {
#pragma unroll
            for (int d = 0; d < kPF; ++d) {
                const int grp = g0 + gi + d;
                if (gi + d < ng) {
                    const Group<BITS> cur = ring[d];
                    if (grp + kPF < G) ring[d].load(row, grp + kPF);
                    const float s = __bfloat162float(__ldg(scol + static_cast<size_t>(grp) * N));
                    accumulate_group<BITS, TP>(cur, sm.x_s + (gi + d) * kG * TP,
                                               sm.x_all + grp * kMaxTok, s, acc);
                }
            }
        }
    }

    // ---- SwiGLU in shared memory: up threads hand their values to the gate threads ----
    if (!is_gate) {
#pragma unroll
        for (int t = 0; t < TP; ++t) sm.u_s[i_local * TP + t] = acc[t];
    }
    __syncthreads();
    if (is_gate) {
        float part[TP];
#pragma unroll
        for (int t = 0; t < TP; ++t) {
            const float g = bf16_round(acc[t]);
            const float u = bf16_round(sm.u_s[i_local * TP + t]);
            float h = bf16_round(silu(g) * u);  // the sorted path's bf16 intermediate
            if (!FOLD) h *= (t < it.ne ? it.wts[t] : 0.f);
            sm.h_s[i_local * TP + t] = h;
            part[t] = h;
        }
#pragma unroll
        for (int t = 0; t < TP; ++t) {
            const float v = warp_reduce_sum(part[t]);
            if ((tid & 31) == 0) sm.h_part[(tid >> 5) * kMaxTok + t] = v;
        }
    }
    __syncthreads();
    if (tid < TP)
        sm.h_all[tid] = sm.h_part[tid] + sm.h_part[kMaxTok + tid] + sm.h_part[2 * kMaxTok + tid] +
                        sm.h_part[3 * kMaxTok + tid];

    // ---- variant 2: the next item's first groups go out before the down pass ----
    if (FOLD && next_valid) prefetch_gate_up<BITS>(packed_gu, *next, K, I, ring);
    __syncthreads();

    // ---- down: rows n of Wdown[e], the slice's 128 codes of each, into y ----
    const int wpr_dn = w1::words_per_row(I, BITS);
    const uint32_t* dn = packed_dn + static_cast<size_t>(it.e) * K * wpr_dn;
    const __nv_bfloat16* sdn = scales_dn + (static_cast<size_t>(it.e) * (I / kG) + it.slice) * K;
    float wfold[TP];
#pragma unroll
    for (int t = 0; t < TP; ++t) wfold[t] = FOLD ? (t < it.ne ? it.wts[t] : 0.f) : 1.f;
    auto load_row = [&](int n, Group<BITS>& g) {
        g.load(dn + static_cast<size_t>(n) * wpr_dn, it.slice);
    };
    Group<BITS> cur[2], nxt[2];
    int n0 = tid;
    if (n0 < K) load_row(n0, cur[0]);
    if (n0 + kThreads < K) load_row(n0 + kThreads, cur[1]);
    for (; n0 < K; n0 += 2 * kThreads) {
        const int n2 = n0 + 2 * kThreads;
        if (n2 < K) load_row(n2, nxt[0]);
        if (n2 + kThreads < K) load_row(n2 + kThreads, nxt[1]);
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            const int n = n0 + r * kThreads;
            if (n < K) {
                float d[TP];
#pragma unroll
                for (int t = 0; t < TP; ++t) d[t] = 0.f;
                const float s = __bfloat162float(__ldg(sdn + n));
                accumulate_group<BITS, TP>(cur[r], sm.h_s, sm.h_all, s, d);
#pragma unroll
                for (int t = 0; t < TP; ++t)
                    if (t < it.ne)
                        atomicAdd(yacc + static_cast<size_t>(it.toks[t]) * K + n, d[t] * wfold[t]);
            }
        }
        cur[0] = nxt[0];
        cur[1] = nxt[1];
    }
}

template <int BITS, bool FOLD>
__global__ void __launch_bounds__(kThreads, 2)
    w1moe_fused_kernel(const __nv_bfloat16* __restrict__ x, const int32_t* __restrict__ ids,
                       const __nv_bfloat16* __restrict__ wts,
                       const uint32_t* __restrict__ packed_gu,
                       const __nv_bfloat16* __restrict__ scales_gu,
                       const uint32_t* __restrict__ packed_dn,
                       const __nv_bfloat16* __restrict__ scales_dn, float* __restrict__ yacc, int T,
                       int topk, int K, int I, int n_items) {
    __shared__ Smem sm;
    const int n_slices = I / kG;
    Group<BITS> ring[kPF];

    int slot = 0;
    resolve_item(sm, sm.item[0], blockIdx.x, n_items, n_slices, T, topk, ids, wts);
    if (sm.item[0].id >= n_items) return;
    prefetch_gate_up<BITS>(packed_gu, sm.item[0], K, I, ring);

    for (;;) {
        const Item& it = sm.item[slot];
        Item& nx = sm.item[slot ^ 1];
        int next_valid = 0;
        if (FOLD) {
            // Resolve the next item now so its rows can be prefetched before the down pass.
            resolve_item(sm, nx, it.id + gridDim.x, n_items, n_slices, T, topk, ids, wts);
            next_valid = nx.id < n_items;
        }
        switch (it.ne) {
            case 1:
                run_item<BITS, 1, FOLD>(sm, it, x, packed_gu, scales_gu, packed_dn, scales_dn, yacc,
                                        K, I, ring, &nx, next_valid);
                break;
            case 2:
                run_item<BITS, 2, FOLD>(sm, it, x, packed_gu, scales_gu, packed_dn, scales_dn, yacc,
                                        K, I, ring, &nx, next_valid);
                break;
            case 3:
            case 4:
                run_item<BITS, 4, FOLD>(sm, it, x, packed_gu, scales_gu, packed_dn, scales_dn, yacc,
                                        K, I, ring, &nx, next_valid);
                break;
            default:
                run_item<BITS, 8, FOLD>(sm, it, x, packed_gu, scales_gu, packed_dn, scales_dn, yacc,
                                        K, I, ring, &nx, next_valid);
                break;
        }
        if (!FOLD) {
            __syncthreads();
            resolve_item(sm, nx, it.id + gridDim.x, n_items, n_slices, T, topk, ids, wts);
            if (nx.id >= n_items) return;
            prefetch_gate_up<BITS>(packed_gu, nx, K, I, ring);
        } else if (!next_valid) {
            return;
        }
        __syncthreads();
        slot ^= 1;
    }
}

// ---- variant 3: the fused step on the tensor cores -----------------------------------------

// Variant 2 measured 57 us at T = 1 against the 19.5 us of its bytes on the 5090 (Nsight: 108
// blocks of 8 warps on 170 SMs, one instruction every 10 cycles per warp, the long-scoreboard
// stall half the samples): every thread walks a whole 5,120-k row on the CUDA cores, about
// 15,000 dependent instructions. This rung keeps the work item, (expert, 128 intermediate
// columns), and gives it 16 warps on w1gemm variant 3's lane-word body:
//   gate|up  warp w streams one 16-column strip of the slice (gate strips 0..7, up 8..15)
//            over the whole K with an 8-deep ring of words and a 4-deep ring of the
//            activations of the expert's tokens (one m16n8k16 B tile, up to 8 tokens);
//   SwiGLU   the strips' accumulators in shared memory, h = bf16(silu(bf16 g) * bf16 u) as
//            bf16 [token][128], the sorted path's rounding;
//   down     the slice is one 128-k group of every down row, so the 5,120 outputs are 320
//            one-group strips, 20 per warp with the next 8 in flight, B the h tile from
//            shared memory, the routing weight applied as each sum is added into y (fp32
//            atomics, as variants 1 and 2).
// One block per item, no persistence: an item a block does not own (its expert came earlier
// in the routing) exits after the scan.
constexpr int kV3Warps = 16;

template <int BITS>
__global__ void __launch_bounds__(kV3Warps * 32)
    w1moe_mma_kernel(const __nv_bfloat16* __restrict__ x, const int32_t* __restrict__ ids,
                     const __nv_bfloat16* __restrict__ wts, const uint32_t* __restrict__ packed_gu,
                     const __nv_bfloat16* __restrict__ scales_gu,
                     const uint32_t* __restrict__ packed_dn,
                     const __nv_bfloat16* __restrict__ scales_dn, float* __restrict__ yacc, int T,
                     int topk, int K, int I) {
    constexpr int D = 8, AD = 4;
    __shared__ float gu_s[2][kG][kMaxTok];                    // gate, up accumulators [col][token]
    __shared__ __align__(16) __nv_bfloat16 h_s[kMaxTok][kG];  // the slice of h, [token][i]
    __shared__ int toks_s[kMaxTok];
    __shared__ float wts_s[kMaxTok];
    __shared__ int ne_s, e_s;

    const int n_slices = I / kG;
    const int pair = blockIdx.x / n_slices, slice = blockIdx.x % n_slices;
    const int n_pairs = T * topk;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int g = lane >> 2, c = lane & 3;
    if (threadIdx.x < 32) {
        const int e = ids[pair];
        int first = n_pairs;
        for (int j = lane; j < n_pairs; j += 32)
            if (ids[j] == e) first = min(first, j);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) first = min(first, __shfl_xor_sync(kFullMask, first, o));
        if (lane == 0) {
            int ne = 0;
            if (first == pair)
                for (int j = 0; j < n_pairs; ++j)
                    if (ids[j] == e) {
                        toks_s[ne] = j / topk;
                        wts_s[ne] = __bfloat162float(wts[j]);
                        ++ne;
                    }
            ne_s = ne;
            e_s = e;
        }
    }
    __syncthreads();
    const int ne = ne_s, e = e_s;
    if (ne == 0) return;

    // ---- gate|up: warp w's strip over the whole K ----
    {
        const int G = K / kG, N = 2 * I;
        const int wpr = w1::words_per_row(K, BITS);
        const int col0 = (w < 8 ? 0 : I) + slice * kG + 16 * (w & 7);
        const uint32_t* pe = packed_gu + static_cast<size_t>(e) * N * wpr;
        const unsigned short* sb = reinterpret_cast<const unsigned short*>(scales_gu) +
                                   static_cast<size_t>(e) * G * N + col0 + g;
        // Lanes past the expert's last token read its last token: their sums go to columns of
        // the tile the SwiGLU ignores, and the load needs no predicate (see w1gemm.cu).
        const uint4* arow =
            reinterpret_cast<const uint4*>(x + static_cast<size_t>(toks_s[min(g, ne - 1)]) * K) +
            4 * c;
        w1::LaneWords<BITS> wr[D][2];
        unsigned sr[D];
        uint4 act[AD][1][4];
        auto load_w = [&](int slot, int grp) {
            wr[slot][0].load(pe, col0 + g, grp, wpr, c);
            wr[slot][1].load(pe, col0 + g + 8, grp, wpr, c);
            const size_t srow = static_cast<size_t>(grp) * N;
            sr[slot] = static_cast<unsigned>(__ldg(sb + srow)) |
                       (static_cast<unsigned>(__ldg(sb + srow + 8)) << 16);
        };
        auto load_a = [&](int buf, int grp) {
#pragma unroll
            for (int q = 0; q < 4; ++q) act[buf][0][q] = __ldg(arow + grp * 16 + q);
        };
        float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
#pragma unroll
        for (int d = 0; d < D; ++d)
            if (d < G) load_w(d, d);
#pragma unroll
        for (int d = 0; d < AD - 1; ++d)
            if (d < G) load_a(d, d);
        for (int i0 = 0; i0 < G; i0 += D) {
#pragma unroll
            for (int d = 0; d < D; ++d) {
                const int i = i0 + d;
                if (i < G) {
                    if (i + AD - 1 < G) load_a((d + AD - 1) % AD, i + AD - 1);
                    const unsigned s2 = sr[d];
                    w1::lane_group_mma<BITS, 1>(acc, wr[d][0], wr[d][1], w1::splat(s2 & 0xFFFFu),
                                                w1::splat(s2 >> 16), act[d % AD]);
                    if (i + D < G) load_w(d, i + D);
                }
            }
        }
        float(*dst)[kMaxTok] = gu_s[w < 8 ? 0 : 1];
        const int cl = 16 * (w & 7) + g;
        dst[cl][2 * c] = acc[0][0];
        dst[cl][2 * c + 1] = acc[0][1];
        dst[cl + 8][2 * c] = acc[0][2];
        dst[cl + 8][2 * c + 1] = acc[0][3];
    }
    __syncthreads();

    // ---- SwiGLU into the bf16 h tile ----
    for (int idx = threadIdx.x; idx < kG * kMaxTok; idx += kV3Warps * 32) {
        const int t = idx / kG, i = idx % kG;
        float h = 0.f;
        if (t < ne) {
            const float gv = bf16_round(gu_s[0][i][t]);
            const float uv = bf16_round(gu_s[1][i][t]);
            h = silu(gv) * uv;
        }
        h_s[t][i] = __float2bfloat16(h);
    }
    __syncthreads();

    // ---- down: 16-output strips of one group each, h as the B tile ----
    {
        const int wpr = w1::words_per_row(I, BITS);
        const uint32_t* pe = packed_dn + static_cast<size_t>(e) * K * wpr;
        const unsigned short* sb = reinterpret_cast<const unsigned short*>(scales_dn) +
                                   (static_cast<size_t>(e) * n_slices + slice) * K + g;
        uint4 act[1][4];
#pragma unroll
        for (int q = 0; q < 4; ++q)
            act[0][q] = *reinterpret_cast<const uint4*>(&h_s[g][32 * c + 8 * q]);
        const int strips = K / 16;
        const int ns = strips > w ? (strips - w + kV3Warps - 1) / kV3Warps : 0;
        w1::LaneWords<BITS> wr[D][2];
        unsigned sr[D];
        auto load_w = [&](int slot, int j) {
            const int n0 = (w + j * kV3Warps) * 16;
            wr[slot][0].load(pe, n0 + g, slice, wpr, c);
            wr[slot][1].load(pe, n0 + g + 8, slice, wpr, c);
            sr[slot] = static_cast<unsigned>(__ldg(sb + n0)) |
                       (static_cast<unsigned>(__ldg(sb + n0 + 8)) << 16);
        };
        const int t0 = 2 * c, t1 = 2 * c + 1;
        const float w0 = t0 < ne ? wts_s[t0] : 0.f, w1v = t1 < ne ? wts_s[t1] : 0.f;
        float* y0 = t0 < ne ? yacc + static_cast<size_t>(toks_s[t0]) * K : nullptr;
        float* y1 = t1 < ne ? yacc + static_cast<size_t>(toks_s[t1]) * K : nullptr;
#pragma unroll
        for (int d = 0; d < D; ++d)
            if (d < ns) load_w(d, d);
        for (int j0 = 0; j0 < ns; j0 += D) {
#pragma unroll
            for (int d = 0; d < D; ++d) {
                const int j = j0 + d;
                if (j < ns) {
                    float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
                    const unsigned s2 = sr[d];
                    w1::lane_group_mma<BITS, 1>(acc, wr[d][0], wr[d][1], w1::splat(s2 & 0xFFFFu),
                                                w1::splat(s2 >> 16), act);
                    if (j + D < ns) load_w(d, j + D);
                    const int n = (w + j * kV3Warps) * 16 + g;
                    if (y0) {
                        atomicAdd(y0 + n, w0 * acc[0][0]);
                        atomicAdd(y0 + n + 8, w0 * acc[0][2]);
                    }
                    if (y1) {
                        atomicAdd(y1 + n, w1v * acc[0][1]);
                        atomicAdd(y1 + n + 8, w1v * acc[0][3]);
                    }
                }
            }
        }
    }
}

__global__ void f32_to_bf16_kernel(const float* __restrict__ src, __nv_bfloat16* __restrict__ dst,
                                   int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = __float2bfloat16(src[i]);
}

// A stream-ordered pool of this file's own that keeps what it was given. The device's
// default pool returns its memory to the driver whenever the device synchronizes (release
// threshold 0), so after every sync a cudaMallocAsync maps pages again: 65 us of host time
// per call on the 5090, in front of a 57 us kernel, which the bench's events counted. One
// pool per device, created on first use, never destroyed (it holds one call's scratch).
cudaMemPool_t scratch_pool() {
    static std::mutex mu;
    static cudaMemPool_t pools[64] = {};
    int dev = 0;
    SPARK_CUDA_CHECK(cudaGetDevice(&dev));
    SPARK_REQUIRE(dev >= 0 && dev < 64, "w1moe_decode: device index out of range");
    const std::lock_guard<std::mutex> lock(mu);
    if (!pools[dev]) {
        cudaMemPoolProps props = {};
        props.allocType = cudaMemAllocationTypePinned;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = dev;
        SPARK_CUDA_CHECK(cudaMemPoolCreate(&pools[dev], &props));
        uint64_t keep = UINT64_MAX;
        SPARK_CUDA_CHECK(
            cudaMemPoolSetAttribute(pools[dev], cudaMemPoolAttrReleaseThreshold, &keep));
    }
    return pools[dev];
}

// Stream-ordered scratch memory, freed when the launcher returns (the free is queued after
// the kernels, so the buffer lives until they finish).
struct Scratch {
    void* p = nullptr;
    cudaStream_t stream;
    Scratch(size_t bytes, cudaStream_t s) : stream(s) {
        SPARK_CUDA_CHECK(cudaMallocFromPoolAsync(&p, bytes, scratch_pool(), stream));
    }
    ~Scratch() {
        if (p) cudaFreeAsync(p, stream);
    }
};

template <int BITS, bool FOLD>
void launch_fused(const __nv_bfloat16* x, const int32_t* ids, const __nv_bfloat16* wts,
                  const uint32_t* packed_gu, const __nv_bfloat16* scales_gu,
                  const uint32_t* packed_dn, const __nv_bfloat16* scales_dn, float* yacc, int T,
                  int topk, int K, int I, cudaStream_t stream) {
    const int n_items = T * topk * (I / kG);
    static const int per_sm = [] {
        int v = 1;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &v, w1moe_fused_kernel<BITS, FOLD>, kThreads, 0));
        return std::max(1, v);
    }();
    const int grid = std::min(n_items, per_sm * num_sms());
    w1moe_fused_kernel<BITS, FOLD><<<grid, kThreads, 0, stream>>>(
        x, ids, wts, packed_gu, scales_gu, packed_dn, scales_dn, yacc, T, topk, K, I, n_items);
    SPARK_CHECK_LAUNCH();
}

}  // namespace

int w1moe_decode_num_variants() {
    return 4;
}

bool w1moe_decode_supports(int T, int topk, int K, int I, int bits, int variant) {
    return variant >= 0 && variant < w1moe_decode_num_variants() && (bits == 1 || bits == 2) &&
           T >= 1 && T <= kMaxTok && topk >= 1 && T * topk <= kMaxPairs && K > 0 && I > 0 &&
           K % kG == 0 && I % kG == 0 && K / kG <= kMaxGroups && K % 256 == 0;
}

void w1moe_decode_bf16(const __nv_bfloat16* x, const int32_t* expert_ids,
                       const __nv_bfloat16* expert_weights, const uint32_t* packed_gate_up,
                       const __nv_bfloat16* scales_gate_up, const uint32_t* packed_down,
                       const __nv_bfloat16* scales_down, __nv_bfloat16* y, int T, int topk, int E,
                       int K, int I, int bits, int variant, cudaStream_t stream) {
    SPARK_REQUIRE(variant >= 0 && variant < w1moe_decode_num_variants(),
                  "w1moe_decode: bad variant");
    SPARK_REQUIRE(w1moe_decode_supports(T, topk, K, I, bits, variant),
                  "w1moe_decode: needs bits 1 or 2, 1 <= T <= 8, T * topk <= 64, K % 256 == 0, "
                  "K <= 16384, I % 128 == 0");
    SPARK_REQUIRE(E >= 1, "w1moe_decode: E must be >= 1");
    SPARK_REQUIRE(x && expert_ids && expert_weights && packed_gate_up && scales_gate_up &&
                      packed_down && scales_down && y,
                  "w1moe_decode: null pointer");
    SPARK_REQUIRE(is_aligned16(x) && is_aligned16(packed_gate_up) && is_aligned16(packed_down),
                  "w1moe_decode: x and the packed weights must be 16-byte aligned");
    const int n_pairs = T * topk;
    if (variant == 0) {
        const size_t gu_elems = static_cast<size_t>(n_pairs) * 2 * I;
        const size_t h_elems = static_cast<size_t>(n_pairs) * I;
        Scratch scratch((gu_elems + h_elems) * sizeof(__nv_bfloat16), stream);
        auto* gu = static_cast<__nv_bfloat16*>(scratch.p);
        __nv_bfloat16* h = gu + gu_elems;
        v0_gate_up_kernel<<<dim3(cdiv(2 * I, 256), n_pairs), 256, 0, stream>>>(
            x, expert_ids, packed_gate_up, scales_gate_up, gu, topk, K, I, bits);
        SPARK_CHECK_LAUNCH();
        v0_swiglu_kernel<<<cdiv(static_cast<int>(h_elems), 256), 256, 0, stream>>>(gu, h, n_pairs,
                                                                                   I);
        SPARK_CHECK_LAUNCH();
        v0_down_kernel<<<dim3(cdiv(K, 256), T), 256, 0, stream>>>(
            h, expert_ids, expert_weights, packed_down, scales_down, y, topk, K, I, bits);
        SPARK_CHECK_LAUNCH();
        return;
    }
    const size_t y_elems = static_cast<size_t>(T) * K;
    Scratch scratch(y_elems * sizeof(float), stream);
    auto* yacc = static_cast<float*>(scratch.p);
    SPARK_CUDA_CHECK(cudaMemsetAsync(yacc, 0, y_elems * sizeof(float), stream));
    if (variant >= 3) {
        const int n_items = n_pairs * (I / kG);
        if (bits == 1)
            w1moe_mma_kernel<1><<<n_items, kV3Warps * 32, 0, stream>>>(
                x, expert_ids, expert_weights, packed_gate_up, scales_gate_up, packed_down,
                scales_down, yacc, T, topk, K, I);
        else
            w1moe_mma_kernel<2><<<n_items, kV3Warps * 32, 0, stream>>>(
                x, expert_ids, expert_weights, packed_gate_up, scales_gate_up, packed_down,
                scales_down, yacc, T, topk, K, I);
        SPARK_CHECK_LAUNCH();
    } else if (variant == 1) {
        if (bits == 1)
            launch_fused<1, false>(x, expert_ids, expert_weights, packed_gate_up, scales_gate_up,
                                   packed_down, scales_down, yacc, T, topk, K, I, stream);
        else
            launch_fused<2, false>(x, expert_ids, expert_weights, packed_gate_up, scales_gate_up,
                                   packed_down, scales_down, yacc, T, topk, K, I, stream);
    } else {
        if (bits == 1)
            launch_fused<1, true>(x, expert_ids, expert_weights, packed_gate_up, scales_gate_up,
                                  packed_down, scales_down, yacc, T, topk, K, I, stream);
        else
            launch_fused<2, true>(x, expert_ids, expert_weights, packed_gate_up, scales_gate_up,
                                  packed_down, scales_down, yacc, T, topk, K, I, stream);
    }
    f32_to_bf16_kernel<<<cdiv(static_cast<int>(y_elems), 256), 256, 0, stream>>>(
        yacc, y, static_cast<int>(y_elems));
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
