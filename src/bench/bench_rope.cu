// Benchmark: RoPE on q and k plus the K/V cache append from a fused q|k|v projection.
//   ./bench_rope [--b=<B> --s=<S> --hq=<H_q> --hkv=<H_kv> --d=<D> --pos0=<P> --cap=<C>]
//   [--iters=<N>]
// Default shapes are Llama-3-8B's heads (32 query, 8 K/V, 128 wide): a 4096-token prefill,
// an 8192-token prefill, and decode steps of 1 and 8 sequences against a 4096-token cache.
// Validates against a CPU double-precision reference; exits 1 on mismatch.
// Each shape also runs the paged form (rope_append_paged_bf16, row "rope_paged"): the same
// B * S tokens packed, positions pos0 + s, k and v into 16-token pages of a shuffled pool.
// And the paged form with Qwen3's QK norm: fused into the append ("rope_paged_qknorm") and
// as its own launch in place on qkv before the plain append ("rope_paged_qknorm_split", the
// norm kernel below). The default shapes add Qwen3-30B-A3B's heads (32 query, 4 K/V, 128).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "spark/kernels.h"

using namespace spark::bench;

namespace {

using spark::bf16x8;

__device__ __forceinline__ float round_bf16(float x) {
    return __bfloat162float(__float2bfloat16_rn(x));
}

// The QK norm as a launch of its own, in place on the q and k heads of qkv [T, (H_q + 2
// H_kv) * D]: the baseline for the fused form. The fused kernel's thread split (D/16 lanes a
// head, 8 elements at c and 8 at c + D/2 each) and its arithmetic, so the two give the same
// bits. D a power of two <= 512.
__global__ void qk_norm_inplace_kernel(__nv_bfloat16* qkv, const __nv_bfloat16* qn,
                                       const __nv_bfloat16* kn, int H_q, int H_kv, int D, float eps,
                                       int64_t total) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int g = D / 16, half = D / 2;
    const int64_t row = i / g;
    const int c = static_cast<int>(i - row * g) * 8;
    const int64_t tok = row / (H_q + H_kv);
    const int h = static_cast<int>(row - tok * (H_q + H_kv));
    __nv_bfloat16* src = qkv + tok * (H_q + 2 * H_kv) * D + static_cast<int64_t>(h) * D + c;
    const __nv_bfloat16* w = h < H_q ? qn : kn;
    bf16x8 x1 = *reinterpret_cast<const bf16x8*>(src);
    bf16x8 x2 = *reinterpret_cast<const bf16x8*>(src + half);
    float a[8], b[8], ss = 0.f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float2 u = __bfloat1622float2(x1.h[k]);
        const float2 v = __bfloat1622float2(x2.h[k]);
        a[2 * k] = u.x;
        a[2 * k + 1] = u.y;
        b[2 * k] = v.x;
        b[2 * k + 1] = v.y;
        ss += u.x * u.x + u.y * u.y + v.x * v.x + v.y * v.y;
    }
    const unsigned lane = threadIdx.x & 31u;
    const unsigned mask = g == 32 ? 0xffffffffu : ((1u << g) - 1u) << (lane & ~(g - 1u));
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        if (o < g) ss += __shfl_xor_sync(mask, ss, o);
    }
    const float r = rsqrtf(ss * (1.0f / static_cast<float>(2 * half)) + eps);
    const bf16x8 w1 = *reinterpret_cast<const bf16x8*>(w + c);
    const bf16x8 w2 = *reinterpret_cast<const bf16x8*>(w + c + half);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float2 wa = __bfloat1622float2(w1.h[k]);
        const float2 wb = __bfloat1622float2(w2.h[k]);
        x1.h[k] = __floats2bfloat162_rn(round_bf16(a[2 * k] * r) * wa.x,
                                        round_bf16(a[2 * k + 1] * r) * wa.y);
        x2.h[k] = __floats2bfloat162_rn(round_bf16(b[2 * k] * r) * wb.x,
                                        round_bf16(b[2 * k + 1] * r) * wb.y);
    }
    *reinterpret_cast<bf16x8*>(src) = x1;
    *reinterpret_cast<bf16x8*>(src + half) = x2;
}

struct Shape {
    int B, S, H_q, H_kv, D, pos0, cap;
};

std::string shape_name(const Shape& s) {
    return "b" + std::to_string(s.B) + "_s" + std::to_string(s.S) + "_hq" + std::to_string(s.H_q) +
           "_hkv" + std::to_string(s.H_kv) + "_d" + std::to_string(s.D) + "_pos" +
           std::to_string(s.pos0);
}

bool run_shape(const Shape& sh, int iters, cudaStream_t stream) {
    const int width = (sh.H_q + 2 * sh.H_kv) * sh.D;
    const size_t tokens = static_cast<size_t>(sh.B) * sh.S;
    const size_t n_qkv = tokens * width;
    const size_t n_q = tokens * sh.H_q * sh.D;
    const size_t n_cache = static_cast<size_t>(sh.B) * sh.H_kv * sh.cap * sh.D;
    const size_t n_tab = static_cast<size_t>(sh.pos0 + sh.S) * sh.D;

    std::vector<float> h_qkv(n_qkv);
    fill_uniform(h_qkv, -2.0f, 2.0f, 1);
    std::vector<__nv_bfloat16> t_qkv(n_qkv);
    for (size_t i = 0; i < n_qkv; ++i) {
        t_qkv[i] = __float2bfloat16(h_qkv[i]);
        h_qkv[i] = __bfloat162float(t_qkv[i]);
    }
    // rotate-half tables: theta 500000, column d and d + D/2 share a frequency
    std::vector<float> h_cos(n_tab), h_sin(n_tab);
    for (int p = 0; p < sh.pos0 + sh.S; ++p) {
        for (int d = 0; d < sh.D; ++d) {
            const int f = d % (sh.D / 2);
            const double inv = std::pow(500000.0, -2.0 * f / sh.D);
            h_cos[static_cast<size_t>(p) * sh.D + d] = static_cast<float>(std::cos(p * inv));
            h_sin[static_cast<size_t>(p) * sh.D + d] = static_cast<float>(std::sin(p * inv));
        }
    }

    // CPU reference in double: q [B, H_q, S, D], k and v caches [B, H_kv, cap, D]
    std::vector<float> ref_q(n_q), ref_k(n_cache, 0.0f), ref_v(n_cache, 0.0f);
    for (int b = 0; b < sh.B; ++b) {
        for (int s = 0; s < sh.S; ++s) {
            const float* row = h_qkv.data() + (static_cast<size_t>(b) * sh.S + s) * width;
            const int pos = sh.pos0 + s;
            const float* cs = h_cos.data() + static_cast<size_t>(pos) * sh.D;
            const float* sn = h_sin.data() + static_cast<size_t>(pos) * sh.D;
            for (int h = 0; h < sh.H_q + sh.H_kv; ++h) {
                const float* x = row + h * sh.D;
                float* out =
                    h < sh.H_q
                        ? ref_q.data() + ((static_cast<size_t>(b) * sh.H_q + h) * sh.S + s) * sh.D
                        : ref_k.data() +
                              ((static_cast<size_t>(b) * sh.H_kv + h - sh.H_q) * sh.cap + pos) *
                                  sh.D;
                for (int d = 0; d < sh.D / 2; ++d) {
                    const double x1 = x[d], x2 = x[d + sh.D / 2];
                    out[d] = static_cast<float>(x1 * cs[d] - x2 * sn[d]);
                    out[d + sh.D / 2] = static_cast<float>(x2 * cs[d] + x1 * sn[d]);
                }
            }
            for (int vh = 0; vh < sh.H_kv; ++vh) {
                const float* x = row + (sh.H_q + sh.H_kv + vh) * sh.D;
                float* out =
                    ref_v.data() + ((static_cast<size_t>(b) * sh.H_kv + vh) * sh.cap + pos) * sh.D;
                for (int d = 0; d < sh.D; ++d) out[d] = x[d];
            }
        }
    }

    __nv_bfloat16 *d_qkv = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_cos = nullptr, *d_sin = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&d_qkv, n_qkv * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_q, n_q * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_k, n_cache * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_v, n_cache * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_cos, n_tab * sizeof(float)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_sin, n_tab * sizeof(float)));
    SPARK_CUDA_CHECK(
        cudaMemcpy(d_qkv, t_qkv.data(), n_qkv * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(
        cudaMemcpy(d_cos, h_cos.data(), n_tab * sizeof(float), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(
        cudaMemcpy(d_sin, h_sin.data(), n_tab * sizeof(float), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemset(d_k, 0, n_cache * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMemset(d_v, 0, n_cache * sizeof(__nv_bfloat16)));

    auto launch = [&] {
        spark::rope_append_bf16(d_qkv, d_cos, d_sin, d_q, d_k, d_v, sh.B, sh.S, sh.H_q, sh.H_kv,
                                sh.D, sh.pos0, sh.cap, stream);
    };
    launch();
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));

    auto fetch = [&](const __nv_bfloat16* d, size_t n) {
        std::vector<__nv_bfloat16> t(n);
        SPARK_CUDA_CHECK(
            cudaMemcpy(t.data(), d, n * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        std::vector<float> f(n);
        for (size_t i = 0; i < n; ++i) f[i] = __bfloat162float(t[i]);
        return f;
    };
    const std::vector<float> got_q = fetch(d_q, n_q), got_k = fetch(d_k, n_cache),
                             got_v = fetch(d_v, n_cache);
    ErrorStats err = compare(got_q.data(), ref_q.data(), n_q);
    const ErrorStats ek = compare(got_k.data(), ref_k.data(), n_cache);
    const ErrorStats ev = compare(got_v.data(), ref_v.data(), n_cache);
    err.max_abs = std::max(err.max_abs, std::max(ek.max_abs, ev.max_abs));
    err.max_rel = std::max(err.max_rel, std::max(ek.max_rel, ev.max_rel));
    // one bf16 rounding of an fp32 product sum; |x| <= 2 so 2e-2 absolute covers it
    const bool ok = err.max_abs <= 2e-2;

    const Timing t = time_kernel(launch, stream, 10, iters);
    // traffic: the qkv rows read once, q and the k / v slots written once
    const double moved_gb = 2.0 * static_cast<double>(n_qkv) * sizeof(__nv_bfloat16) / 1e9;

    // The paged form on the same tokens: token (b, s) is row b * S + s, its k and v go to row
    // pos % 16 of page pages[b][pos / 16], the pages of the pool shuffled.
    constexpr int kPage = 16;
    const int per_seq = (sh.pos0 + sh.S + kPage - 1) / kPage;
    const int npages = sh.B * per_seq;
    std::vector<int> perm(npages);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), std::mt19937(3));
    std::vector<int> h_pos(tokens), h_slot(tokens);
    for (int b = 0; b < sh.B; ++b)
        for (int s = 0; s < sh.S; ++s) {
            const int pos = sh.pos0 + s, i = b * sh.S + s;
            h_pos[i] = pos;
            h_slot[i] = perm[b * per_seq + pos / kPage] * kPage + pos % kPage;
        }
    const size_t n_pool = static_cast<size_t>(npages) * sh.H_kv * kPage * sh.D;
    __nv_bfloat16 *d_pk = nullptr, *d_pv = nullptr, *d_pq = nullptr;
    int *d_pos = nullptr, *d_slot = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&d_pk, n_pool * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_pv, n_pool * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_pq, n_q * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_pos, tokens * sizeof(int)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_slot, tokens * sizeof(int)));
    SPARK_CUDA_CHECK(cudaMemcpy(d_pos, h_pos.data(), tokens * sizeof(int), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(
        cudaMemcpy(d_slot, h_slot.data(), tokens * sizeof(int), cudaMemcpyHostToDevice));
    auto launch_paged = [&] {
        spark::rope_append_paged_bf16(d_qkv, d_cos, d_sin, d_pos, d_slot, d_pq, d_pk, d_pv,
                                      static_cast<int>(tokens), sh.H_q, sh.H_kv, sh.D, kPage,
                                      stream);
    };
    launch_paged();
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::vector<float> pq = fetch(d_pq, n_q), pk = fetch(d_pk, n_pool),
                             pv = fetch(d_pv, n_pool);
    double perr = 0;
    for (int b = 0; b < sh.B; ++b)
        for (int s = 0; s < sh.S; ++s) {
            const int i = b * sh.S + s, pos = sh.pos0 + s;
            for (int h = 0; h < sh.H_q; ++h)
                for (int d = 0; d < sh.D; ++d)
                    perr = std::max(
                        perr,
                        std::fabs(static_cast<double>(
                            pq[(static_cast<size_t>(i) * sh.H_q + h) * sh.D + d] -
                            ref_q[((static_cast<size_t>(b) * sh.H_q + h) * sh.S + s) * sh.D + d])));
            for (int h = 0; h < sh.H_kv; ++h) {
                const size_t dst = ((static_cast<size_t>(h_slot[i] / kPage) * sh.H_kv + h) * kPage +
                                    h_slot[i] % kPage) *
                                   sh.D;
                const size_t src = ((static_cast<size_t>(b) * sh.H_kv + h) * sh.cap + pos) * sh.D;
                for (int d = 0; d < sh.D; ++d) {
                    perr = std::max(perr,
                                    std::fabs(static_cast<double>(pk[dst + d] - ref_k[src + d])));
                    perr = std::max(perr,
                                    std::fabs(static_cast<double>(pv[dst + d] - ref_v[src + d])));
                }
            }
        }
    const bool pok = perr <= 2e-2;
    const Timing tp = time_kernel(launch_paged, stream, 10, iters);

    // The QK norm. Reference: Qwen3RMSNorm on each q and k head of the bf16 inputs (fp32 x
    // times the rsqrt, to bf16, times the weight, to bf16), then the rotation in double.
    constexpr float kEps = 1e-6f;
    std::vector<float> h_qn(sh.D), h_kn(sh.D);
    fill_uniform(h_qn, 0.8f, 1.2f, 5);
    fill_uniform(h_kn, 0.8f, 1.2f, 6);
    std::vector<__nv_bfloat16> t_qn(sh.D), t_kn(sh.D);
    for (int d = 0; d < sh.D; ++d) {
        t_qn[d] = __float2bfloat16(h_qn[d]);
        h_qn[d] = __bfloat162float(t_qn[d]);
        t_kn[d] = __float2bfloat16(h_kn[d]);
        h_kn[d] = __bfloat162float(t_kn[d]);
    }
    auto bf = [](float x) { return __bfloat162float(__float2bfloat16(x)); };
    const int heads = sh.H_q + sh.H_kv;
    std::vector<float> ref_n(tokens * heads * sh.D), nx(sh.D);  // [token, q | k head, D]
    for (size_t i = 0; i < tokens; ++i) {
        const float* cs = h_cos.data() + static_cast<size_t>(h_pos[i]) * sh.D;
        const float* sn = h_sin.data() + static_cast<size_t>(h_pos[i]) * sh.D;
        for (int h = 0; h < heads; ++h) {
            const float* x = h_qkv.data() + i * width + static_cast<size_t>(h) * sh.D;
            const float* w = h < sh.H_q ? h_qn.data() : h_kn.data();
            double ss = 0;
            for (int d = 0; d < sh.D; ++d) ss += static_cast<double>(x[d]) * x[d];
            const float r = static_cast<float>(1.0 / std::sqrt(ss / sh.D + kEps));
            for (int d = 0; d < sh.D; ++d) nx[d] = bf(bf(x[d] * r) * w[d]);
            float* out = ref_n.data() + (i * heads + h) * sh.D;
            for (int d = 0; d < sh.D / 2; ++d) {
                const double x1 = nx[d], x2 = nx[d + sh.D / 2];
                out[d] = static_cast<float>(x1 * cs[d] - x2 * sn[d]);
                out[d + sh.D / 2] = static_cast<float>(x2 * cs[d] + x1 * sn[d]);
            }
        }
    }
    __nv_bfloat16 *d_qn = nullptr, *d_kn = nullptr, *d_qkvn = nullptr;
    __nv_bfloat16 *d_sq = nullptr, *d_sk = nullptr, *d_sv = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&d_qn, sh.D * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_kn, sh.D * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_qkvn, n_qkv * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_sq, n_q * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_sk, n_pool * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMalloc(&d_sv, n_pool * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(
        cudaMemcpy(d_qn, t_qn.data(), sh.D * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(
        cudaMemcpy(d_kn, t_kn.data(), sh.D * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(
        cudaMemcpy(d_qkvn, d_qkv, n_qkv * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
    SPARK_CUDA_CHECK(cudaMemset(d_pk, 0, n_pool * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMemset(d_pv, 0, n_pool * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMemset(d_sk, 0, n_pool * sizeof(__nv_bfloat16)));
    SPARK_CUDA_CHECK(cudaMemset(d_sv, 0, n_pool * sizeof(__nv_bfloat16)));
    auto launch_fused = [&] {
        spark::rope_append_paged_bf16(d_qkv, d_cos, d_sin, d_pos, d_slot, d_pq, d_pk, d_pv,
                                      static_cast<int>(tokens), sh.H_q, sh.H_kv, sh.D, kPage,
                                      stream, d_qn, d_kn, kEps);
    };
    const int64_t norm_items = static_cast<int64_t>(tokens) * heads * (sh.D / 16);
    // In place, so the timed repeats normalize an already normalized qkv: the same work.
    auto launch_split = [&] {
        qk_norm_inplace_kernel<<<static_cast<unsigned>((norm_items + 255) / 256), 256, 0, stream>>>(
            d_qkvn, d_qn, d_kn, sh.H_q, sh.H_kv, sh.D, kEps, norm_items);
        spark::rope_append_paged_bf16(d_qkvn, d_cos, d_sin, d_pos, d_slot, d_sq, d_sk, d_sv,
                                      static_cast<int>(tokens), sh.H_q, sh.H_kv, sh.D, kPage,
                                      stream);
    };
    launch_fused();
    launch_split();
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    const std::vector<float> nq = fetch(d_pq, n_q), nk = fetch(d_pk, n_pool),
                             nv = fetch(d_pv, n_pool);
    const std::vector<float> sq = fetch(d_sq, n_q), sk = fetch(d_sk, n_pool),
                             sv = fetch(d_sv, n_pool);
    double nerr = 0;
    for (size_t i = 0; i < tokens; ++i) {
        for (int h = 0; h < sh.H_q; ++h)
            for (int d = 0; d < sh.D; ++d)
                nerr = std::max(nerr,
                                std::fabs(static_cast<double>(nq[(i * sh.H_q + h) * sh.D + d] -
                                                              ref_n[(i * heads + h) * sh.D + d])));
        for (int h = 0; h < sh.H_kv; ++h) {
            const size_t dst = ((static_cast<size_t>(h_slot[i] / kPage) * sh.H_kv + h) * kPage +
                                h_slot[i] % kPage) *
                               sh.D;
            for (int d = 0; d < sh.D; ++d)
                nerr =
                    std::max(nerr, std::fabs(static_cast<double>(
                                       nk[dst + d] - ref_n[(i * heads + sh.H_q + h) * sh.D + d])));
        }
    }
    size_t diff = 0;
    for (size_t i = 0; i < n_q; ++i) diff += nq[i] != sq[i];
    for (size_t i = 0; i < n_pool; ++i) diff += (nk[i] != sk[i]) + (nv[i] != sv[i]);
    // the 2e-2 above plus one bf16 step of a normalized value (|x| <= 2.1 here) for the rare
    // rsqrt that rounds the other way; fused and split are the same arithmetic, the same bits
    const bool nok = nerr <= 4e-2 && diff == 0;
    std::fprintf(stderr, "%s: qk norm fused vs split, %zu values differ\n", shape_name(sh).c_str(),
                 diff);
    const Timing tn = time_kernel(launch_fused, stream, 10, iters);
    const Timing ts = time_kernel(launch_split, stream, 10, iters);
    Row rn, rs;
    rn.kernel = "rope_paged_qknorm";
    rs.kernel = "rope_paged_qknorm_split";
    for (Row* r : {&rn, &rs}) {
        r->dtype = "bf16";
        r->variant = 0;
        r->shape = shape_name(sh);
        r->max_abs_err = nerr;
        r->ok = nok;
        r->ref_ms = tp.median_ms;  // the plain append
    }
    rn.median_ms = tn.median_ms;
    rn.min_ms = tn.min_ms;
    rn.gbps = moved_gb / (tn.median_ms * 1e-3);
    rs.median_ms = ts.median_ms;
    rs.min_ms = ts.min_ms;
    rs.gbps = moved_gb / (ts.median_ms * 1e-3);
    SPARK_CUDA_CHECK(cudaFree(d_qn));
    SPARK_CUDA_CHECK(cudaFree(d_kn));
    SPARK_CUDA_CHECK(cudaFree(d_qkvn));
    SPARK_CUDA_CHECK(cudaFree(d_sq));
    SPARK_CUDA_CHECK(cudaFree(d_sk));
    SPARK_CUDA_CHECK(cudaFree(d_sv));
    Row rp;
    rp.kernel = "rope_paged";
    rp.dtype = "bf16";
    rp.variant = 0;
    rp.shape = shape_name(sh);
    rp.median_ms = tp.median_ms;
    rp.min_ms = tp.min_ms;
    rp.gbps = moved_gb / (tp.median_ms * 1e-3);
    rp.max_abs_err = perr;
    rp.ok = pok;
    SPARK_CUDA_CHECK(cudaFree(d_pk));
    SPARK_CUDA_CHECK(cudaFree(d_pv));
    SPARK_CUDA_CHECK(cudaFree(d_pq));
    SPARK_CUDA_CHECK(cudaFree(d_pos));
    SPARK_CUDA_CHECK(cudaFree(d_slot));

    Row r;
    r.kernel = "rope";
    r.dtype = "bf16";
    r.variant = 0;
    r.shape = shape_name(sh);
    r.median_ms = t.median_ms;
    r.min_ms = t.min_ms;
    r.gbps = moved_gb / (t.median_ms * 1e-3);
    r.max_abs_err = err.max_abs;
    r.max_rel_err = err.max_rel;
    r.ok = ok;
    print_row(r);
    print_row(rp);
    print_row(rn);
    print_row(rs);

    SPARK_CUDA_CHECK(cudaFree(d_qkv));
    SPARK_CUDA_CHECK(cudaFree(d_q));
    SPARK_CUDA_CHECK(cudaFree(d_k));
    SPARK_CUDA_CHECK(cudaFree(d_v));
    SPARK_CUDA_CHECK(cudaFree(d_cos));
    SPARK_CUDA_CHECK(cudaFree(d_sin));
    return ok && pok && nok;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Args args(argc, argv);
        print_device_banner();
        const int iters = args.geti("iters", 100);
        std::vector<Shape> shapes;
        if (args.has("s")) {
            const int S = args.geti("s", 1);
            const int pos0 = args.geti("pos0", 0);
            shapes.push_back({args.geti("b", 1), S, args.geti("hq", 32), args.geti("hkv", 8),
                              args.geti("d", 128), pos0, args.geti("cap", pos0 + S)});
        } else {
            shapes = {{1, 4096, 32, 8, 128, 0, 4096}, {1, 8192, 32, 8, 128, 0, 8192},
                      {1, 1, 32, 8, 128, 4095, 4096}, {8, 1, 32, 8, 128, 4095, 4096},
                      {1, 4096, 32, 4, 128, 0, 4096}, {1, 1, 32, 4, 128, 4095, 4096},
                      {8, 1, 32, 4, 128, 4095, 4096}, {64, 1, 32, 4, 128, 4095, 4096}};
        }
        cudaStream_t stream;
        SPARK_CUDA_CHECK(cudaStreamCreate(&stream));
        print_header();
        bool ok = true;
        for (const Shape& s : shapes) ok = run_shape(s, iters, stream) && ok;
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
