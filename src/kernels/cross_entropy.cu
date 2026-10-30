// Linear cross-entropy for a large vocabulary: loss = CE(h w^T, y) with h [T, D] and the
// lm_head weight w [V, D] (nn.Linear's layout), without torch's [T, V] logits and their
// fp32 copies. The kernels here are the pieces; python/spark_kernels/ops.py puts them
// together as a ladder (docs/design/cross_entropy.md):
//
//   v1  Liger's recipe on these kernels: a chunk of rows at a time, the logits by ce_gemm
//       (LOGITS), one row kernel that reads a logits row and writes its loss and, in place,
//       its gradient row (softmax minus one-hot, times the reduction scale), then dh and dw
//       by hgemm;
//   v2  the forward without the logits: ce_gemm (STATS) reduces each 128 x 128 tile in the
//       epilogue to a per-row max and sum of exponentials, writes those [T][V/128] partials
//       and the target's logit, and ce_combine merges them into the log-sum-exp. The
//       backward recomputes a chunk of logits by ce_gemm (GRAD), whose epilogue turns each
//       tile into softmax minus one-hot times the scale before the one rounding to bf16;
//   v3  variant 2's forward and a backward with no chunk: ce_bwd_kernel recomputes each
//       128 x 128 tile of the gradient and multiplies it straight into dh and dw through L2
//       reductions (the last section of this file).
//
// ce_gemm is grouped_gemm's bf16 kernel with one expert: logits = h w^T with both operands
// K-contiguous, a persistent grid of one 288-thread block per SM, one producer lane issuing
// two TMA boxes per stage, eight consumer warps on mma.sync.m16n8k16, the last partial wave
// plus one wave split along K Stream-K style and summed through hgemm's deterministic
// chain fixup. Only the epilogue is new, and it runs once per tile, on the finished fp32
// sums (the piece that ends a tile's K chain), so every mode gives the same bits every run.

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>

#include "fp8gemm_tile.cuh"
#include "hgemm_streamk.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

namespace ce {

constexpr float kLog2e = 1.4426950408889634f;

// (max, sum) merge of the online softmax; an empty side is (-inf, 0).
__device__ __forceinline__ void merge(float& m, float& s, float m2, float s2) {
    const float mx = fmaxf(m, m2);
    if (mx == -INFINITY) return;
    s = s * exp2f((m - mx) * kLog2e) + s2 * exp2f((m2 - mx) * kLog2e);
    m = mx;
}

// ---- the v1 row kernel ------------------------------------------------------------------
//
// One block of ROW_THREADS per row, a persistent grid of one block per SM, so 170 rows are
// in flight: 170 x 256 KB at V = 128256, 44 MB, which leaves the second read of a row to L2.
// Pass 1 is the online (max, sum) over 16-byte vectors; thread 0 reads the target's logit
// before the reduction, since pass 2 overwrites the row in place. Pass 2 writes
// (exp(x - lse) - [col == y]) * scale over the logits. A row that does not count is zeroed
// without being read.
constexpr int ROW_THREADS = 1024;

__device__ __forceinline__ void block_merge(float& m, float& s, float2* sh) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float m2 = __shfl_xor_sync(kFullMask, m, o);
        const float s2 = __shfl_xor_sync(kFullMask, s, o);
        merge(m, s, m2, s2);
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) sh[wid] = make_float2(m, s);
    __syncthreads();
    if (wid == 0) {
        const int nw = blockDim.x >> 5;
        float2 v = lane < nw ? sh[lane] : make_float2(-INFINITY, 0.f);
        m = v.x;
        s = v.y;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float m2 = __shfl_xor_sync(kFullMask, m, o);
            const float s2 = __shfl_xor_sync(kFullMask, s, o);
            merge(m, s, m2, s2);
        }
        if (lane == 0) sh[32] = make_float2(m, s);
    }
    __syncthreads();
    m = sh[32].x;
    s = sh[32].y;
    __syncthreads();  // sh is reused by the next row
}

__global__ void __launch_bounds__(ROW_THREADS)
    ce_rows_kernel(__nv_bfloat16* __restrict__ logits, const int64_t* __restrict__ targets,
                   int rows, int V, int ignore_index, const float* __restrict__ scale,
                   float* __restrict__ loss, float* __restrict__ lse_out, bool write_grad) {
    __shared__ float2 sh[33];
    const int nvec = V / 8;
    const float sc = write_grad ? __ldg(scale) : 0.f;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        bf16x8* x = reinterpret_cast<bf16x8*>(logits + static_cast<size_t>(row) * V);
        const long long y = targets[row];
        const bool counts = y != ignore_index && y >= 0 && y < V;
        if (!counts) {  // uniform across the block
            if (write_grad) {
                bf16x8 z;
#pragma unroll
                for (int e = 0; e < 4; ++e) z.h[e] = __floats2bfloat162_rn(0.f, 0.f);
                for (int i = threadIdx.x; i < nvec; i += blockDim.x) x[i] = z;
            }
            if (threadIdx.x == 0) {
                loss[row] = 0.f;
                lse_out[row] = 0.f;
            }
            continue;
        }
        float xy = 0.f;
        if (threadIdx.x == 0) xy = __bfloat162float(logits[static_cast<size_t>(row) * V + y]);
        float m = -INFINITY, s = 0.f;
        for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
            const bf16x8 v = x[i];
            float f[8];
            float lm = -INFINITY;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 q = __bfloat1622float2(v.h[e]);
                f[2 * e] = q.x;
                f[2 * e + 1] = q.y;
                lm = fmaxf(lm, fmaxf(q.x, q.y));
            }
            if (lm > m) {
                s *= exp2f((m - lm) * kLog2e);
                m = lm;
            }
            const float mb = m * kLog2e;
#pragma unroll
            for (int e = 0; e < 8; ++e) s += exp2f(fmaf(f[e], kLog2e, -mb));
        }
        block_merge(m, s, sh);
        const float l = m + logf(s);
        if (threadIdx.x == 0) {
            loss[row] = l - xy;
            lse_out[row] = l;
        }
        if (!write_grad) continue;
        const float lb = l * kLog2e;
        const int yv = static_cast<int>(y);
        for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
            bf16x8 v = x[i];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 q = __bfloat1622float2(v.h[e]);
                float p0 = exp2f(fmaf(q.x, kLog2e, -lb));
                float p1 = exp2f(fmaf(q.y, kLog2e, -lb));
                if (i * 8 + 2 * e == yv) p0 -= 1.f;
                if (i * 8 + 2 * e + 1 == yv) p1 -= 1.f;
                v.h[e] = __floats2bfloat162_rn(p0 * sc, p1 * sc);
            }
            x[i] = v;
        }
    }
}

// ---- transpose of a column slab, zero-padded --------------------------------------------
//
// out[c][r] = x[r][c0 + c] for r < rows and 0 for rows <= r < ld, c < ncols: the gradient
// chunk [C][V] turned into the [V-slab][C rounded up to 64] A operand of dw += g^T h, which
// hgemm takes only K-contiguous. 64 x 64 tiles through shared memory, rows padded to 66 so
// a column read hits 32 different banks.
constexpr int TP = 64;

__global__ void __launch_bounds__(256)
    transpose_pad_kernel(const __nv_bfloat16* __restrict__ x, int rows, int ldx,
                         __nv_bfloat16* __restrict__ out, int ncols, int ld) {
    __shared__ __nv_bfloat16 tile[TP][TP + 2];
    const int c0 = blockIdx.x * TP;
    const int r0 = blockIdx.y * TP;
    const int tx = threadIdx.x & (TP - 1);
    const int ty = threadIdx.x / TP;
    const __nv_bfloat16 zero = __float2bfloat16(0.f);
#pragma unroll 4
    for (int i = ty; i < TP; i += 256 / TP) {
        const int r = r0 + i, c = c0 + tx;
        tile[i][tx] = (r < rows && c < ncols) ? x[static_cast<size_t>(r) * ldx + c] : zero;
    }
    __syncthreads();
#pragma unroll 4
    for (int i = ty; i < TP; i += 256 / TP) {
        const int c = c0 + i, r = r0 + tx;
        if (c < ncols && r < ld) out[static_cast<size_t>(c) * ld + r] = tile[tx][i];
    }
}

// ---- the lm_head GEMM ------------------------------------------------------------------

using hgemm_sk::ld_acquire_gpu;
using hgemm_sk::slot_idx;
using hgemm_sk::st_release_gpu;

constexpr int BM = 128, BN = 128;
constexpr int WARPS_M = 2, WARPS_N = 4;
constexpr int CONSUMER_WARPS = WARPS_M * WARPS_N;
constexpr int CONSUMERS = CONSUMER_WARPS * 32;
constexpr int THREADS = CONSUMERS + 32;
constexpr int ITEM_DEPTH = 2;     // pieces the producer may publish ahead of the consumers
constexpr int FIXUP_BARRIER = 1;  // named barrier for the 256 consumer threads
constexpr int EPI_BARRIER = 2;    // and one for the STATS epilogue's exchange
constexpr int MIN_SHARE = 32;     // least k-steps per Stream-K block (hgemm_sk::kMinShare)
constexpr int BKB = 128;          // bytes of K per stage: 64 bf16
constexpr int STAGES = 2;         // grouped_gemm's and hgemm v6's bf16 choice

using C = fp8gemm_tile::Cfg<BM, BN, BKB, STAGES, WARPS_M, WARPS_N>;
using Acc = C::Acc;

// The STATS epilogue's cross-warp exchange: two buffers (alternate tiles) of [WARPS_N][BM]
// (max, sum) pairs, one per warp column per tile row.
constexpr int RED_PAIRS = 2 * WARPS_N * BM;

constexpr int smem_bytes() {
    return (STAGES * C::STAGE + ITEM_DEPTH * 16 + (2 * STAGES + 2 * ITEM_DEPTH) * 8 +
            RED_PAIRS * 8 + 1023) /
           1024 * 1024;
}
static_assert(smem_bytes() <= 101376, "two 32 KB stages and the exchange fit");

struct Params {
    int M, N;      // T rows of h, V rows of w
    int tiles_m;   // cdiv(M, BM)
    int tiles_n;   // N / BN
    int KT;        // k-steps per tile
    int grid;      // resident blocks
    float* slots;  // 2 * grid Stream-K tiles of BM x BN fp32
    unsigned long long* flags;
    unsigned epoch;
    __nv_bfloat16* out;  // LOGITS: the logits, GRAD: the gradient, [M][N]
    float2* part;        // STATS: [M][tiles_n] (max, sum of exp(x - max)) per row per tile
    float* tgt_logit;    // STATS: [M], the target column's logit
    const int64_t* targets;
    int ignore_index;
    const float* lse;    // GRAD: [M]
    const float* scale;  // GRAD: [1], d loss / d loss_row for a row that counts
};

struct Split {
    int dp, sk, ranges;
};
__device__ __forceinline__ Split split_work(int tiles, int grid, int KT) {
    Split s;
    const int rem = tiles % grid;
    s.sk = rem == 0 ? 0 : rem + (tiles >= grid ? grid : 0);
    s.dp = tiles - s.sk;
    const long long iters = static_cast<long long>(s.sk) * KT;
    s.ranges = static_cast<int>(
        min(static_cast<long long>(grid), max(static_cast<long long>(s.sk), iters / MIN_SHARE)));
    return s;
}
__device__ __forceinline__ int range_start(int sk_iters, int ranges, int c) {
    return static_cast<int>(static_cast<long long>(c) * sk_iters / ranges);
}

// One 32-byte k-step of a bf16 stage (grouped_gemm.cu's bf16_step).
template <int KK>
__device__ __forceinline__ void bf16_step(const unsigned char* as, const unsigned char* bs, int wm,
                                          int wn, int lane, Acc& acc) {
    constexpr int BK = C::BK, MT = C::MT, NT = C::NT, WROWS = C::WROWS, WCOLS = C::WCOLS;
    unsigned afrag[MT][4];
    unsigned bfrag[NT][2];
    const int a_row = lane & 15;
    const int a_ch = KK / 16 + (lane >> 4);
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int row = wm * WROWS + mi * 16 + a_row;
        ldmatrix_x4(afrag[mi], as + row * BK + fp8gemm_tile::swz<BK>(row, a_ch) * 16);
    }
    const int b_row = (lane & 7) + ((lane >> 4) & 1) * 8;
    const int b_ch = KK / 16 + ((lane >> 3) & 1);
#pragma unroll
    for (int nj = 0; nj < NT; nj += 2) {
        const int row = wn * WCOLS + nj * 8 + b_row;
        unsigned r[4];
        ldmatrix_x4(r, bs + row * BK + fp8gemm_tile::swz<BK>(row, b_ch) * 16);
        bfrag[nj][0] = r[0];
        bfrag[nj][1] = r[1];
        bfrag[nj + 1][0] = r[2];
        bfrag[nj + 1][1] = r[3];
    }
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int nj = 0; nj < NT; ++nj) mma_bf16_16816(acc[mi][nj], afrag[mi], bfrag[nj]);
}
__device__ __forceinline__ void bf16_stage(const unsigned char* as, const unsigned char* bs, int wm,
                                           int wn, int lane, Acc& acc) {
    bf16_step<0>(as, bs, wm, wn, lane, acc);
    bf16_step<32>(as, bs, wm, wn, lane, acc);
    bf16_step<64>(as, bs, wm, wn, lane, acc);
    bf16_step<96>(as, bs, wm, wn, lane, acc);
}

// STATS epilogue. Lane (g, t) of warp (wm, wn) holds rows wm*64 + mi*16 + g (+8) and columns
// wn*32 + nj*8 + 2t (+1) of the tile: 8 values per row. The four lanes of a quad share rows,
// so two shuffles give the warp's (max, sum) over its 32 columns; the four warp columns meet
// in shared memory and one thread per row merges them and writes the tile's pair. The lane
// that holds the target column writes the target's logit; exactly one lane in the grid does.
__device__ __forceinline__ void epi_stats(const Acc& acc, const Params& p, int row0, int nt,
                                          float2* red, int wm, int wn, int lane, int tid) {
    constexpr int MT = C::MT, NT = C::NT;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int bn = nt * BN;
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            float m = -INFINITY;
#pragma unroll
            for (int nj = 0; nj < NT; ++nj)
                m = fmaxf(m, fmaxf(acc[mi][nj][2 * h], acc[mi][nj][2 * h + 1]));
            m = fmaxf(m, __shfl_xor_sync(kFullMask, m, 1));
            m = fmaxf(m, __shfl_xor_sync(kFullMask, m, 2));
            const float mb = m * kLog2e;
            float s = 0.f;
#pragma unroll
            for (int nj = 0; nj < NT; ++nj)
                s += exp2f(fmaf(acc[mi][nj][2 * h], kLog2e, -mb)) +
                     exp2f(fmaf(acc[mi][nj][2 * h + 1], kLog2e, -mb));
            s += __shfl_xor_sync(kFullMask, s, 1);
            s += __shfl_xor_sync(kFullMask, s, 2);
            const int r = wm * C::WROWS + mi * 16 + g + 8 * h;
            if (t == 0) red[wn * BM + r] = make_float2(m, s);
            const int row = row0 + r;
            if (row < p.M) {
                const long long c = __ldg(p.targets + row) - (bn + wn * C::WCOLS + 2 * t);
#pragma unroll
                for (int nj = 0; nj < NT; ++nj) {
                    if (c == nj * 8) p.tgt_logit[row] = acc[mi][nj][2 * h];
                    if (c == nj * 8 + 1) p.tgt_logit[row] = acc[mi][nj][2 * h + 1];
                }
            }
        }
    }
    named_barrier_sync(EPI_BARRIER, CONSUMERS);
    if (tid < BM && row0 + tid < p.M) {
        float m = red[tid].x, s = red[tid].y;
#pragma unroll
        for (int w = 1; w < WARPS_N; ++w) merge(m, s, red[w * BM + tid].x, red[w * BM + tid].y);
        p.part[static_cast<size_t>(row0 + tid) * p.tiles_n + nt] = make_float2(m, s);
    }
}

// GRAD epilogue: g = (exp(x - lse) - [col == y]) * scale, 0 on a row that does not count,
// rounded once to bf16 and stored like the plain epilogue (two columns per 4-byte store).
__device__ __forceinline__ void epi_grad(const Acc& acc, const Params& p, int row0, int bn, int wm,
                                         int wn, int lane) {
    constexpr int MT = C::MT, NT = C::NT;
    const int g = lane >> 2;
    const int c2 = (lane & 3) * 2;
    const float scale = __ldg(p.scale);
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = row0 + wm * C::WROWS + mi * 16 + g + 8 * h;
            if (row >= p.M) continue;
            const long long y = __ldg(p.targets + row);
            const bool counts = y != p.ignore_index && y >= 0 && y < p.N;
            const float coef = counts ? scale : 0.f;
            const float lb = __ldg(p.lse + row) * kLog2e;
            __nv_bfloat16* out = p.out + static_cast<size_t>(row) * p.N;
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                const int col = bn + wn * C::WCOLS + nj * 8 + c2;
                const float p0 = exp2f(fmaf(acc[mi][nj][2 * h], kLog2e, -lb)) - (col == y);
                const float p1 = exp2f(fmaf(acc[mi][nj][2 * h + 1], kLog2e, -lb)) - (col + 1 == y);
                *reinterpret_cast<__nv_bfloat162*>(out + col) =
                    __floats2bfloat162_rn(p0 * coef, p1 * coef);
            }
        }
    }
}

template <int MODE>
__global__ void __launch_bounds__(THREADS, 1)
    ce_gemm_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
                   Params p) {
    constexpr int A_STAGE = C::A_STAGE, STAGE_BYTES = C::STAGE;
    constexpr int MT = C::MT, NT = C::NT, WROWS = C::WROWS;

    extern __shared__ __align__(1024) unsigned char smem[];
    int4* items = reinterpret_cast<int4*>(smem + STAGES * STAGE_BYTES);
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(items + ITEM_DEPTH);
    uint64_t* empty_bar = full_bar + STAGES;
    uint64_t* item_full = empty_bar + STAGES;
    uint64_t* item_empty = item_full + ITEM_DEPTH;
    float2* red = reinterpret_cast<float2*>(item_empty + ITEM_DEPTH);
    if (smem_u32(smem) % 1024 != 0) __trap();

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full_bar[s], 1);
            mbar_init(&empty_bar[s], CONSUMER_WARPS);
        }
#pragma unroll
        for (int i = 0; i < ITEM_DEPTH; ++i) {
            mbar_init(&item_full[i], 1);
            mbar_init(&item_empty[i], CONSUMER_WARPS);
        }
        fence_mbar_init();
    }
    __syncthreads();  // the only block-wide barrier in the kernel

    const int tiles = p.tiles_m * p.tiles_n;
    const Split sp = split_work(tiles, p.grid, p.KT);
    const int sk_iters = sp.sk * p.KT;

    if (warp == CONSUMER_WARPS) {
        if (lane == 0) {
            prefetch_tensormap(&tmA);
            prefetch_tensormap(&tmB);
            unsigned g_kt = 0;
            unsigned n_pub = 0;
            auto piece = [&](int tile, int kb, int ke) {
                const int r = n_pub % ITEM_DEPTH;
                const unsigned use = n_pub / ITEM_DEPTH;
                if (n_pub >= ITEM_DEPTH) mbar_wait(&item_empty[r], (use - 1) & 1);
                items[r] = make_int4(tile, kb, ke, 0);
                mbar_arrive(&item_full[r]);
                ++n_pub;
                if (tile < 0) return;
                // m-tile fastest: the tiles in flight share w's rows (the vocabulary slice)
                // and walk h, which is the smaller operand and stays in L2.
                const int row0 = (tile % p.tiles_m) * BM;
                const int bn = (tile / p.tiles_m) * BN;
                for (int kt = kb; kt < ke; ++kt, ++g_kt) {
                    const int s = g_kt % STAGES;
                    const unsigned use_s = g_kt / STAGES;
                    if (g_kt >= STAGES) mbar_wait(&empty_bar[s], (use_s - 1) & 1);
                    mbar_arrive_expect_tx(&full_bar[s], STAGE_BYTES);
                    unsigned char* as = smem + s * STAGE_BYTES;
                    tma_load_2d(as, &tmA, &full_bar[s], kt * BKB, row0);
                    tma_load_2d(as + A_STAGE, &tmB, &full_bar[s], kt * BKB, bn);
                }
            };
            for (int t = blockIdx.x; t < sp.dp; t += p.grid) piece(t, 0, p.KT);
            if (static_cast<int>(blockIdx.x) < sp.ranges) {
                const int it_begin = range_start(sk_iters, sp.ranges, blockIdx.x);
                int it = range_start(sk_iters, sp.ranges, blockIdx.x + 1);
                while (it > it_begin) {
                    const int t = (it - 1) / p.KT;
                    const int k0 = t * p.KT;
                    const int s = max(it_begin, k0);
                    piece(sp.dp + t, s - k0, it - k0);
                    it = s;
                }
            }
            piece(-1, 0, 0);
        }
        return;
    }

    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;
    unsigned g_kt = 0;
    unsigned n_item = 0;
    unsigned n_fin = 0;
    Acc acc;

    for (;;) {
        const int r = n_item % ITEM_DEPTH;
        mbar_wait(&item_full[r], (n_item / ITEM_DEPTH) & 1);
        const int4 itm = items[r];
        __syncwarp();
        if (lane == 0) mbar_arrive(&item_empty[r]);
        ++n_item;
        if (itm.x < 0) break;
        const int tile = itm.x, kb = itm.y, ke = itm.z;
        const int row0 = (tile % p.tiles_m) * BM;
        const int nt = tile / p.tiles_m;
        const int bn = nt * BN;
        const int m_valid = p.M - row0;

#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;

        for (int kt = kb; kt < ke; ++kt, ++g_kt) {
            const int s = g_kt % STAGES;
            mbar_wait(&full_bar[s], (g_kt / STAGES) & 1);
            const unsigned char* as = smem + s * STAGE_BYTES;
            bf16_stage(as, as + A_STAGE, wm, wn, lane, acc);
            fence_proxy_async_smem();
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty_bar[s]);
        }

        bool finish = true;
        if (kb > 0 || ke < p.KT) {
            const int tile_r = tile - sp.dp;
            float4* slot =
                reinterpret_cast<float4*>(p.slots + static_cast<size_t>(tile_r) * (BM * BN));
            unsigned long long* flag = p.flags + tile_r;
            const unsigned long long tag = static_cast<unsigned long long>(p.epoch) << 32;
            if (kb > 0) {
                if (tid == 0) {
                    while (ld_acquire_gpu(flag) != (tag | static_cast<unsigned>(kb))) {
                    }
                }
                named_barrier_sync(FIXUP_BARRIER, CONSUMERS);
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    if (wm * WROWS + mi * 16 >= m_valid) continue;
#pragma unroll
                    for (int nj = 0; nj < NT; ++nj) {
                        const float4 v = __ldcg(slot + slot_idx<MT, NT>(warp, lane, mi, nj));
                        acc[mi][nj][0] += v.x;
                        acc[mi][nj][1] += v.y;
                        acc[mi][nj][2] += v.z;
                        acc[mi][nj][3] += v.w;
                    }
                }
            }
            if (ke < p.KT) {
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    if (wm * WROWS + mi * 16 >= m_valid) continue;
#pragma unroll
                    for (int nj = 0; nj < NT; ++nj)
                        slot[slot_idx<MT, NT>(warp, lane, mi, nj)] = make_float4(
                            acc[mi][nj][0], acc[mi][nj][1], acc[mi][nj][2], acc[mi][nj][3]);
                }
                __threadfence();
                named_barrier_sync(FIXUP_BARRIER, CONSUMERS);
                if (tid == 0) st_release_gpu(flag, tag | static_cast<unsigned>(ke));
                finish = false;
            }
        }
        if (!finish) continue;
        if constexpr (MODE == CE_GEMM_LOGITS) {
            fp8gemm_tile::store_bf16<C>(acc, 1.f, p.out, p.M, p.N, row0, bn);
        } else if constexpr (MODE == CE_GEMM_STATS) {
            epi_stats(acc, p, row0, nt, red + (n_fin & 1) * (WARPS_N * BM), wm, wn, lane, tid);
        } else {
            epi_grad(acc, p, row0, bn, wm, wn, lane);
        }
        ++n_fin;
    }
}

// ---- host side of ce_gemm ---------------------------------------------------------------

struct Workspace {
    float* slots = nullptr;
    unsigned long long* flags = nullptr;
    size_t tiles = 0;
    unsigned epoch = 0;
};

template <int MODE>
int resident_blocks() {
    constexpr int bytes = smem_bytes();
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(ce_gemm_kernel<MODE>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, ce_gemm_kernel<MODE>, THREADS, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <int MODE>
void launch(const __nv_bfloat16* h, const __nv_bfloat16* w, int M, int N, int K, Params p,
            cudaStream_t stream) {
    constexpr int bytes = smem_bytes();
    const int grid = resident_blocks<MODE>();
    const int kbytes = K * 2;
    const CUtensorMap tmA =
        make_tensor_map_2d_u8(h, M, kbytes, BM, BKB, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap tmB =
        make_tensor_map_2d_u8(w, N, kbytes, BN, BKB, CU_TENSOR_MAP_SWIZZLE_128B);

    static Workspace ws;
    const size_t need = 2 * static_cast<size_t>(grid);
    if (ws.tiles < need) {
        if (ws.slots) SPARK_CUDA_CHECK(cudaFree(ws.slots));
        if (ws.flags) SPARK_CUDA_CHECK(cudaFree(ws.flags));
        ws.tiles = need;
        SPARK_CUDA_CHECK(cudaMalloc(&ws.slots, ws.tiles * BM * BN * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMalloc(&ws.flags, ws.tiles * sizeof(unsigned long long)));
        SPARK_CUDA_CHECK(cudaMemset(ws.flags, 0, ws.tiles * sizeof(unsigned long long)));
    }
    if (++ws.epoch == 0) ws.epoch = 1;

    p.M = M;
    p.N = N;
    p.tiles_m = cdiv(M, BM);
    p.tiles_n = N / BN;
    p.KT = kbytes / BKB;
    p.grid = grid;
    p.slots = ws.slots;
    p.flags = ws.flags;
    p.epoch = ws.epoch;
    ce_gemm_kernel<MODE><<<grid, THREADS, bytes, stream>>>(tmA, tmB, p);
    SPARK_CHECK_LAUNCH();
}

// ---- combine: the per-tile partials of a row into its log-sum-exp and loss ---------------

__global__ void ce_combine_kernel(const float2* __restrict__ part, const float* __restrict__ tgt,
                                  const int64_t* __restrict__ targets, int M, int tiles_n, int N,
                                  int ignore_index, float* __restrict__ lse,
                                  float* __restrict__ loss) {
    const int row = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= M) return;
    const float2* pr = part + static_cast<size_t>(row) * tiles_n;
    float m = -INFINITY, s = 0.f;
    for (int j = lane; j < tiles_n; j += 32) {
        const float2 v = __ldg(pr + j);
        merge(m, s, v.x, v.y);
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float m2 = __shfl_xor_sync(kFullMask, m, o);
        const float s2 = __shfl_xor_sync(kFullMask, s, o);
        merge(m, s, m2, s2);
    }
    if (lane == 0) {
        const float l = m + logf(s);
        const long long y = targets[row];
        const bool counts = y != ignore_index && y >= 0 && y < N;
        lse[row] = l;
        loss[row] = counts ? l - tgt[row] : 0.f;
    }
}

// ---- the chunk-free backward (variant 3) ------------------------------------------------
//
// One kernel walks the [T, V] gradient in 128 x 128 tiles and never stores it, the way Cut
// Cross-Entropy does. Per tile a block recomputes the logits from h and w (the forward's lse
// per row is kept, so the softmax of the tile is exp(x - lse)), turns them into the bf16
// gradient tile G in shared memory, then streams the same h and w boxes a second time and
// multiplies, per 32-column stage of D,
//
//   dh[t-tile][d-stage] += G   @ w[v-tile][d-stage]      (A = G, straight from smem)
//   dw[v-tile][d-stage] += G^T @ h[t-tile][d-stage]      (A = G^T, ldmatrix.trans on the same smem)
//
// so three GEMMs' worth of mma per tile, variant 1's count, and nothing of size T x V. The
// second pass is warp-asynchronous: each warp owns 16 rows of the [128 x 32] fp32 partial,
// writes them to its own 2 KB of shared memory and adds them into global memory with one
// cp.reduce.async.bulk.tensor, waiting only on its own previous reduce (a first version
// staged the whole partial through one buffer behind four block barriers per stage and
// spent half its cycles in them). The reduce runs at L2: measured on the RTX 5090, that
// adds 2.9 TB/s of fp32 while the targets sit in L2, against 1.65 TB/s for red.v4.f32 and
// 0.4 for a scalar red, and 0.9 TB/s for any of them once the targets leave L2
// (docs/design/cross_entropy.md). Keeping the targets in L2 is the schedule's whole job:
//
// * dh is accumulated in fp32 for one band of T rows at a time (the caller zeroes it and
//   casts it to bf16 after the band), 32 MB at 2048 rows;
// * dw never exists in fp32 as a whole. Tiles are walked t-fastest inside a vocabulary strip,
//   so the strips in flight are few (170 blocks over 16 t-tiles is 11 strips), and each strip
//   accumulates into one of a few recycled fp32 strips of [128][D] (16 by default, 32 MB).
//   The block that finishes a strip's last tile folds the slot into bf16 dw (a plain store
//   on the first band, an add on the others, one rounding per band as variant 1 does per
//   chunk), zeroes it and releases it; a block starting strip v waits for strip v - slots to
//   be folded, 125 us per block per band measured;
// * tiles are handed out from a global counter as blocks free up, not round-robin: a block
//   that folded a strip or waited on a slot just takes fewer tiles. With a static schedule
//   every such delay was permanent, the late block's later strips folded late, the strips
//   that needed those slots waited, and a 2048-row band spent 8 to 12 ms per block spinning;
// * the reduction targets and the zeroed slots carry an evict_last L2 policy and the fold's
//   dw traffic evict_first. The operand stream turns the 96 MB L2 over in about a
//   millisecond, and without the policies a slot's zeroed lines were gone by the time the
//   strip that reused it arrived, so its reductions missed to DRAM: 56 ms per band
//   against 31 with them.
//
// The stages are 32 columns of D (64 bytes) so that the gradient tile (32 KB), three stages
// (48 KB) and the warps' staging boxes (16 KB) fit in the 99 KB a block may take. The
// reduces run through the L2 atomics in an order that depends on timing, so dh and dw are
// not bit for bit reproducible across runs (variants 1 and 2 are).

namespace bwd {

constexpr int BM = 128, BN = 128;  // the gradient tile: rows of h by rows of w
constexpr int BKB = 64;            // bytes of D per stage
constexpr int BK = BKB / 2;        // 32 columns of D per stage
constexpr int STAGES = 3;
constexpr int WARPS_M = 2, WARPS_N = 4;  // the recompute's warp layout, ce_gemm's
constexpr int CONSUMER_WARPS = WARPS_M * WARPS_N;
constexpr int CONSUMERS = CONSUMER_WARPS * 32;
constexpr int THREADS = CONSUMERS + 32;
constexpr int SLOTS = 16;  // recycled fp32 strips of dw by default, 32 MB at D = 4096
constexpr int P_NT = 4;    // the second pass: 8 x 1 warps of 16 x 32
constexpr int P_ROWS = BM / CONSUMER_WARPS;  // 16 rows of the [128 x 32] partial per warp
constexpr int G_ROW = BN * 2;                // bytes per row of the bf16 gradient tile: 16 chunks
constexpr int G_BYTES = BM * G_ROW;          // 32 KB
constexpr int P_ROW = BK * 4;                // bytes per row of the fp32 partial: 8 chunks
constexpr int P_BYTES = P_ROWS * P_ROW;      // 2 KB per warp
constexpr int BAR_EPI = 1;                   // the one named barrier, over the consumer threads
constexpr int FOLD_BUFS = 4;                 // chunks of a strip fold in flight
constexpr int ITEM_DEPTH = 2;                // tiles the producer may hand over ahead

using Cm = fp8gemm_tile::Cfg<BM, BN, BKB, STAGES, WARPS_M, WARPS_N>;
using AccM = Cm::Acc;
using AccP = float[P_NT][4];

constexpr int G_OFF = STAGES * Cm::STAGE;  // 48 KB
constexpr int P_OFF = G_OFF + G_BYTES;     // 80 KB, 1024-aligned for the 128B swizzle
constexpr int BAR_OFF = P_OFF + CONSUMER_WARPS * P_BYTES;
constexpr int smem_bytes() {
    return (BAR_OFF + (2 * STAGES + FOLD_BUFS + 2 * ITEM_DEPTH) * 8 + ITEM_DEPTH * 4 + 16 + 1023) /
           1024 * 1024;
}
static_assert(smem_bytes() <= 101376, "three 16 KB stages, the gradient tile and the staging fit");
static_assert(P_OFF % 1024 == 0 && P_BYTES % 1024 == 0, "every warp's staging is 1024-aligned");

struct __align__(8) bf16x4 {
    __nv_bfloat162 h[2];
};

struct Params {
    int M;  // rows of h in this band
    int tiles_m, tiles_n, KT;
    int grid;
    int D;
    const int64_t* targets;  // [M]
    const float* lse;        // [M]
    const float* scale;      // [1]
    int ignore_index;
    __nv_bfloat16* dw;  // [V][D]
    float* slots;       // [nslots * 128][D] fp32, zero between uses
    int* done;          // [tiles_n] tiles of the strip completed
    int* folded;        // [tiles_n] 1 once the strip is in dw and its slot is zero again
    int* next_tile;     // [1] the tile counter: blocks take tiles in order as they free up
    int first;          // first band: dw = slot, else dw += slot
    int nslots;         // recycled fp32 strips the slots buffer holds
};

__device__ __forceinline__ int ld_acquire_gpu_s32(const int* p) {
    int v;
    asm volatile("ld.acquire.gpu.global.s32 %0, [%1];\n" : "=r"(v) : "l"(p) : "memory");
    return v;
}
__device__ __forceinline__ void st_release_gpu_s32(int* p, int v) {
    asm volatile("st.release.gpu.global.s32 [%0], %1;\n" ::"l"(p), "r"(v) : "memory");
}
__device__ __forceinline__ void fence_proxy_async_global() {
    asm volatile("fence.proxy.async.global;\n" ::: "memory");
}
// L2 policies: the reduction targets (the dh band and the live slots) are the lines that
// must survive the operand stream, which turns the 96 MB L2 over in about a millisecond
// at this kernel's rate; the fold's dw traffic has no reuse inside a band.
__device__ __forceinline__ unsigned long long policy_evict_last() {
    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(pol));
    return pol;
}
__device__ __forceinline__ unsigned long long policy_evict_first() {
    unsigned long long pol;
    asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(pol));
    return pol;
}
// The [16 x 32] fp32 box at `src` (128B-swizzled rows) added into the map at (x, y).
__device__ __forceinline__ void tma_reduce_add_2d(const CUtensorMap* map, const void* src, int x,
                                                  int y, unsigned long long pol) {
    asm volatile(
        "cp.reduce.async.bulk.tensor.2d.global.shared::cta.add.tile.bulk_group.L2::cache_hint "
        "[%0, {%1, %2}], [%3], %4;\n" ::"l"(map),
        "r"(x), "r"(y), "r"(smem_u32(src)), "l"(pol)
        : "memory");
    asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_read() {
    asm volatile("cp.async.bulk.wait_group.read 0;\n" ::: "memory");
}
__device__ __forceinline__ void bulk_wait_all() {
    asm volatile("cp.async.bulk.wait_group 0;\n" ::: "memory");
}

// One 32-byte k-step of the recompute on a 64-byte stage (ce::bf16_step on bwd::Cm).
template <int KK>
__device__ __forceinline__ void step(const unsigned char* as, const unsigned char* bs, int wm,
                                     int wn, int lane, AccM& acc) {
    constexpr int SBK = Cm::BK, MT = Cm::MT, NT = Cm::NT, WROWS = Cm::WROWS, WCOLS = Cm::WCOLS;
    unsigned afrag[MT][4];
    unsigned bfrag[NT][2];
    const int a_row = lane & 15;
    const int a_ch = KK / 16 + (lane >> 4);
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int row = wm * WROWS + mi * 16 + a_row;
        ldmatrix_x4(afrag[mi], as + row * SBK + fp8gemm_tile::swz<SBK>(row, a_ch) * 16);
    }
    const int b_row = (lane & 7) + ((lane >> 4) & 1) * 8;
    const int b_ch = KK / 16 + ((lane >> 3) & 1);
#pragma unroll
    for (int nj = 0; nj < NT; nj += 2) {
        const int row = wn * WCOLS + nj * 8 + b_row;
        unsigned r[4];
        ldmatrix_x4(r, bs + row * SBK + fp8gemm_tile::swz<SBK>(row, b_ch) * 16);
        bfrag[nj][0] = r[0];
        bfrag[nj][1] = r[1];
        bfrag[nj + 1][0] = r[2];
        bfrag[nj + 1][1] = r[3];
    }
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int nj = 0; nj < NT; ++nj) mma_bf16_16816(acc[mi][nj], afrag[mi], bfrag[nj]);
}

// The gradient tile into shared memory: g = bf16((exp(x - lse) - [col == y]) * scale), 0 on
// a row that does not count or past M. Row r of G is 256 bytes, 16-byte chunk c stored at
// chunk c ^ (r & 7), so that the 8 rows of an ldmatrix matrix, with or without .trans, land
// in 8 different bank groups.
__device__ __forceinline__ void grad_to_smem(const AccM& acc, const Params& p, unsigned char* G,
                                             int row0, int bn, int wm, int wn, int lane) {
    constexpr int MT = Cm::MT, NT = Cm::NT;
    const int g = lane >> 2;
    const int t = lane & 3;
    const float scale = __ldg(p.scale);
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int r = wm * Cm::WROWS + mi * 16 + g + 8 * h;
            const int row = row0 + r;
            const bool in = row < p.M;
            const long long y = in ? __ldg(p.targets + row) : -1;
            const bool counts = in && y != p.ignore_index && y >= 0 && y < p.tiles_n * BN;
            const float coef = counts ? scale : 0.f;
            const float lb = in ? __ldg(p.lse + row) * kLog2e : 0.f;
            unsigned char* grow = G + r * G_ROW;
#pragma unroll
            for (int nj = 0; nj < NT; ++nj) {
                const int c = wn * Cm::WCOLS + nj * 8 + 2 * t;  // column within the tile
                const int col = bn + c;
                const float p0 = exp2f(fmaf(acc[mi][nj][2 * h], kLog2e, -lb)) - (col == y);
                const float p1 = exp2f(fmaf(acc[mi][nj][2 * h + 1], kLog2e, -lb)) - (col + 1 == y);
                const int ch = (c >> 3) ^ (r & 7);
                *reinterpret_cast<__nv_bfloat162*>(grow + ch * 16 + (c & 7) * 2) =
                    __floats2bfloat162_rn(p0 * coef, p1 * coef);
            }
        }
    }
}

// One of the two products of a stage, this warp's 16 rows of it: P[16 x 32] =
// A[16 x 128] B[128 x 32], A from the gradient tile (TRANS: as G^T) and B from the stage's h
// or w box ([128 rows of k][32 of d], 64-byte rows, the TMA 64B swizzle, read with
// ldmatrix.trans).
template <bool TRANS>
__device__ __forceinline__ void product(const unsigned char* G, const unsigned char* box, int warp,
                                        int lane, AccP& acc) {
    constexpr int SBK = Cm::BK;
#pragma unroll
    for (int nj = 0; nj < P_NT; ++nj)
#pragma unroll
        for (int q = 0; q < 4; ++q) acc[nj][q] = 0.f;
#pragma unroll
    for (int ks = 0; ks < BN / 16; ++ks) {
        unsigned afrag[4];
        unsigned bfrag[P_NT][2];
        if constexpr (!TRANS) {
            // rows of G are the product's rows: 16 rows, chunks 2 ks and 2 ks + 1
            const int row = warp * P_ROWS + (lane & 15);
            const int ch = 2 * ks + (lane >> 4);
            ldmatrix_x4(afrag, G + row * G_ROW + (ch ^ (row & 7)) * 16);
        } else {
            // columns of G are the product's rows: matrices (v 0-7 | 8-15) x (t 0-7 | 8-15)
            // read transposed from rows t of G
            const int row = ks * 16 + (lane & 7) + 8 * (lane >> 4);
            const int ch = 2 * warp + ((lane >> 3) & 1);
            ldmatrix_x4_trans(afrag, G + row * G_ROW + (ch ^ (row & 7)) * 16);
        }
#pragma unroll
        for (int nj = 0; nj < P_NT; nj += 2) {
            const int krow = ks * 16 + (lane & 15);
            const int ch = nj + (lane >> 4);
            unsigned r[4];
            ldmatrix_x4_trans(r, box + krow * SBK + fp8gemm_tile::swz<SBK>(krow, ch) * 16);
            bfrag[nj][0] = r[0];
            bfrag[nj][1] = r[1];
            bfrag[nj + 1][0] = r[2];
            bfrag[nj + 1][1] = r[3];
        }
#pragma unroll
        for (int nj = 0; nj < P_NT; ++nj) mma_bf16_16816(acc[nj], afrag, bfrag[nj]);
    }
}

// The warp's partial into its staging box: 16 rows of 32 floats, 16-byte chunk c of row r at
// chunk c ^ (r & 7), the TMA 128B swizzle. Lane (g, t) holds columns 2t, 2t+1 of rows g and
// g + 8 of each n8 tile, one float2 store each. The swizzle sends lanes t and t ^ 2 of one
// row pair to the same banks, so the store goes out as two half-warp stores, each of which
// covers the 32 banks once.
__device__ __forceinline__ void stage_partial(const AccP& acc, unsigned char* P, int lane) {
    const int g = lane >> 2;
    const int t = lane & 3;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
        if ((t >> 1) != half) continue;
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = g + 8 * h;
#pragma unroll
            for (int nj = 0; nj < P_NT; ++nj) {
                const int col = nj * 8 + 2 * t;
                const int ch = (col >> 2) ^ (row & 7);
                *reinterpret_cast<float2*>(P + row * P_ROW + ch * 16 + (col & 3) * 4) =
                    make_float2(acc[nj][2 * h], acc[nj][2 * h + 1]);
            }
        }
    }
}

// Bulk copies with an L2 policy.
__device__ __forceinline__ void bulk_load_hint(void* smem_dst, const void* gmem_src, unsigned bytes,
                                               uint64_t* bar, unsigned long long pol) {
    asm volatile(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], "
        "[%1], %2, [%3], %4;\n" ::"r"(smem_u32(smem_dst)),
        "l"(gmem_src), "r"(bytes), "r"(smem_u32(bar)), "l"(pol)
        : "memory");
}
__device__ __forceinline__ void bulk_store_hint(void* gmem_dst, const void* smem_src,
                                                unsigned bytes, unsigned long long pol) {
    asm volatile(
        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [%0], [%1], %2, %3;\n" ::"l"(
            gmem_dst),
        "r"(smem_u32(smem_src)), "r"(bytes), "l"(pol)
        : "memory");
}

// The finished strip: dw[v-strip] (+)= the slot, rounded once to bf16, and the slot zeroed,
// 6 MB of traffic per strip by one block. It runs on the bulk copy engine, FOLD_CHUNK
// floats at a time through the gradient tile's and the staging's shared memory (both idle
// once the tile is done): a bulk load of the slot chunk and of the dw chunk, the add and
// the rounding by the threads in place, a bulk store of the bf16 chunk to dw and of the
// zeroed chunk back to the slot, FOLD_BUFS chunks in flight with the next chunk's loads
// issued before this chunk's wait: 110 us per strip measured, 0.6 ms of a block's band.
constexpr int FOLD_CHUNK = 2048;  // floats: 8 KB fp32, 4 KB bf16
constexpr int FOLD_F32 = FOLD_CHUNK * 4;
constexpr int FOLD_BF16 = FOLD_CHUNK * 2;
static_assert(FOLD_BUFS * FOLD_F32 <= G_BYTES && FOLD_BUFS * FOLD_BF16 <= CONSUMER_WARPS * P_BYTES,
              "the fold's chunks fit in the gradient tile and the staging");

__device__ __forceinline__ void fold_strip(const Params& p, int nt, int tid, unsigned char* G,
                                           unsigned char* P, uint64_t* fold_bar) {
    const int slot = nt % p.nslots;
    float* s = p.slots + static_cast<size_t>(slot) * BN * p.D;
    __nv_bfloat16* dw = p.dw + static_cast<size_t>(nt) * BN * p.D;
    const int chunks = BN * p.D / FOLD_CHUNK;
    const unsigned long long evict_first = policy_evict_first();
    const unsigned long long evict_last = policy_evict_last();
    auto load = [&](int c) {
        const int buf = c % FOLD_BUFS;
        mbar_arrive_expect_tx(&fold_bar[buf], FOLD_F32 + (p.first ? 0 : FOLD_BF16));
        bulk_load(G + buf * FOLD_F32, s + static_cast<size_t>(c) * FOLD_CHUNK, FOLD_F32,
                  &fold_bar[buf]);
        if (!p.first)
            bulk_load_hint(P + buf * FOLD_BF16, dw + static_cast<size_t>(c) * FOLD_CHUNK, FOLD_BF16,
                           &fold_bar[buf], evict_first);
    };
    if (tid == 0) load(0);
    for (int c = 0; c < chunks; ++c) {
        const int buf = c % FOLD_BUFS;
        float4* f = reinterpret_cast<float4*>(G + buf * FOLD_F32);
        bf16x4* h = reinterpret_cast<bf16x4*>(P + buf * FOLD_BF16);
        if (tid == 0 && c + 1 < chunks) {
            // the next chunk's buffer was last stored from by chunk c + 1 - FOLD_BUFS: let at
            // most FOLD_BUFS - 2 store groups stay pending, so that one has been read
            asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(FOLD_BUFS - 2) : "memory");
            load(c + 1);
        }
        mbar_wait(&fold_bar[buf], (c / FOLD_BUFS) & 1);
#pragma unroll
        for (int j = 0; j < FOLD_CHUNK / 4 / CONSUMERS; ++j) {
            const int i = tid + j * CONSUMERS;
            float4 v = f[i];
            if (!p.first) {
                const bf16x4 prev = h[i];
                const float2 a = __bfloat1622float2(prev.h[0]);
                const float2 b = __bfloat1622float2(prev.h[1]);
                v.x += a.x;
                v.y += a.y;
                v.z += b.x;
                v.w += b.y;
            }
            bf16x4 o;
            o.h[0] = __floats2bfloat162_rn(v.x, v.y);
            o.h[1] = __floats2bfloat162_rn(v.z, v.w);
            h[i] = o;
            f[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        fence_proxy_async_smem();
        named_barrier_sync(BAR_EPI, CONSUMERS);
        if (tid == 0) {
            bulk_store_hint(dw + static_cast<size_t>(c) * FOLD_CHUNK, h, FOLD_BF16, evict_first);
            bulk_store_hint(s + static_cast<size_t>(c) * FOLD_CHUNK, f, FOLD_F32, evict_last);
            asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
        }
    }
    if (tid == 0) bulk_wait_all();
}

__global__ void __launch_bounds__(THREADS, 1)
    ce_bwd_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmB,
                  const __grid_constant__ CUtensorMap tmDh,
                  const __grid_constant__ CUtensorMap tmDw, Params p) {
    constexpr int A_STAGE = Cm::A_STAGE, STAGE_BYTES = Cm::STAGE;

    extern __shared__ __align__(1024) unsigned char smem[];
    unsigned char* G = smem + G_OFF;
    unsigned char* P = smem + P_OFF;
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(smem + BAR_OFF);
    uint64_t* empty_bar = full_bar + STAGES;
    uint64_t* fold_bar = empty_bar + STAGES;
    uint64_t* item_full = fold_bar + FOLD_BUFS;
    uint64_t* item_empty = item_full + ITEM_DEPTH;
    int* items = reinterpret_cast<int*>(item_empty + ITEM_DEPTH);
    int* fold_flag = items + ITEM_DEPTH;
    if (smem_u32(smem) % 1024 != 0) __trap();

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full_bar[s], 1);
            mbar_init(&empty_bar[s], CONSUMER_WARPS);
        }
#pragma unroll
        for (int i = 0; i < FOLD_BUFS; ++i) mbar_init(&fold_bar[i], 1);
#pragma unroll
        for (int i = 0; i < ITEM_DEPTH; ++i) {
            mbar_init(&item_full[i], 1);
            mbar_init(&item_empty[i], CONSUMER_WARPS);
        }
        fence_mbar_init();
    }
    __syncthreads();

    const int tiles = p.tiles_m * p.tiles_n;

    if (warp == CONSUMER_WARPS) {
        if (lane == 0) {
            prefetch_tensormap(&tmA);
            prefetch_tensormap(&tmB);
            unsigned g_kt = 0;
            unsigned n_pub = 0;
            for (;;) {
                // Tiles are taken in order from a global counter as blocks free up: a block
                // that waited on a slot or folded a strip just takes fewer tiles. (A static
                // round-robin made every wait permanent: the late block's later strips
                // folded late, the strips a pool later waited on them, and a 2048-row band
                // spent 8 to 12 ms per block in that spin.)
                const int tile = atomicAdd(p.next_tile, 1);
                const int r = n_pub % ITEM_DEPTH;
                const unsigned use = n_pub / ITEM_DEPTH;
                if (n_pub >= ITEM_DEPTH) mbar_wait(&item_empty[r], (use - 1) & 1);
                items[r] = tile;
                mbar_arrive(&item_full[r]);
                ++n_pub;
                if (tile >= tiles) break;
                const int row0 = (tile % p.tiles_m) * BM;
                const int bn = (tile / p.tiles_m) * BN;
                for (int pass = 0; pass < 2; ++pass) {
                    for (int kt = 0; kt < p.KT; ++kt, ++g_kt) {
                        const int s = g_kt % STAGES;
                        const unsigned use = g_kt / STAGES;
                        if (g_kt >= STAGES) mbar_wait(&empty_bar[s], (use - 1) & 1);
                        mbar_arrive_expect_tx(&full_bar[s], STAGE_BYTES);
                        unsigned char* as = smem + s * STAGE_BYTES;
                        tma_load_2d(as, &tmA, &full_bar[s], kt * BKB, row0);
                        tma_load_2d(as + A_STAGE, &tmB, &full_bar[s], kt * BKB, bn);
                    }
                }
            }
        }
        return;
    }

    if (tid == 0) {
        prefetch_tensormap(&tmDh);
        prefetch_tensormap(&tmDw);
    }
    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;
    unsigned char* Pw = P + warp * P_BYTES;
    const unsigned long long keep = policy_evict_last();
    unsigned g_kt = 0;
    unsigned n_item = 0;
    AccM acc;
    AccP pa;

    for (;;) {
        const int r = n_item % ITEM_DEPTH;
        mbar_wait(&item_full[r], (n_item / ITEM_DEPTH) & 1);
        const int tile = items[r];
        __syncwarp();
        if (lane == 0) mbar_arrive(&item_empty[r]);
        ++n_item;
        if (tile >= tiles) break;
        const int row0 = (tile % p.tiles_m) * BM;
        const int nt = tile / p.tiles_m;
        const int bn = nt * BN;
        const int slot_row = (nt % p.nslots) * BN;

        // pass 1: the logits tile
#pragma unroll
        for (int i = 0; i < Cm::MT; ++i)
#pragma unroll
            for (int j = 0; j < Cm::NT; ++j)
#pragma unroll
                for (int q = 0; q < 4; ++q) acc[i][j][q] = 0.f;
        for (int kt = 0; kt < p.KT; ++kt, ++g_kt) {
            const int s = g_kt % STAGES;
            mbar_wait(&full_bar[s], (g_kt / STAGES) & 1);
            const unsigned char* as = smem + s * STAGE_BYTES;
            step<0>(as, as + A_STAGE, wm, wn, lane, acc);
            step<32>(as, as + A_STAGE, wm, wn, lane, acc);
            fence_proxy_async_smem();
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty_bar[s]);
        }
        // Every warp finished reading the previous tile's G before the barrier that ended
        // it, so G can be overwritten now. The strip's slot must have been folded and
        // zeroed by its previous user before any warp adds into it.
        grad_to_smem(acc, p, G, row0, bn, wm, wn, lane);
        if (tid == 0 && nt >= p.nslots) {
            while (ld_acquire_gpu_s32(p.folded + nt - p.nslots) == 0) {
            }
        }
        named_barrier_sync(BAR_EPI, CONSUMERS);

        // pass 2: the same stages again, two products per stage, every warp on its own
        for (int kt = 0; kt < p.KT; ++kt, ++g_kt) {
            const int s = g_kt % STAGES;
            mbar_wait(&full_bar[s], (g_kt / STAGES) & 1);
            const unsigned char* as = smem + s * STAGE_BYTES;
            // dh[t-tile] += G w[v-tile]
            product<false>(G, as + A_STAGE, warp, lane, pa);
            if (lane == 0) bulk_wait_read();
            __syncwarp();
            stage_partial(pa, Pw, lane);
            fence_proxy_async_smem();
            __syncwarp();
            if (lane == 0) tma_reduce_add_2d(&tmDh, Pw, kt * BK, row0 + warp * P_ROWS, keep);
            // dw[v-tile] += G^T h[t-tile]
            product<true>(G, as, warp, lane, pa);
            fence_proxy_async_smem();
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty_bar[s]);
            if (lane == 0) bulk_wait_read();
            __syncwarp();
            stage_partial(pa, Pw, lane);
            fence_proxy_async_smem();
            __syncwarp();
            if (lane == 0) tma_reduce_add_2d(&tmDw, Pw, kt * BK, slot_row + warp * P_ROWS, keep);
        }

        // The tile is in dh and the slot once every warp's reduces have landed; the strip's
        // last tile folds the slot.
        if (lane == 0) {
            bulk_wait_all();
            __threadfence();
        }
        named_barrier_sync(BAR_EPI, CONSUMERS);
        if (tid == 0) {
            const int n = atomicAdd(p.done + nt, 1) + 1;
            *fold_flag = n == p.tiles_m;
        }
        named_barrier_sync(BAR_EPI, CONSUMERS);
        if (*fold_flag) {
            fold_strip(p, nt, tid, G, P, fold_bar);
            __threadfence();
            named_barrier_sync(BAR_EPI, CONSUMERS);
            if (tid == 0) {
                fence_proxy_async_global();
                st_release_gpu_s32(p.folded + nt, 1);
            }
        }
    }
}

int resident_blocks() {
    constexpr int bytes = smem_bytes();
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(ce_bwd_kernel,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, ce_bwd_kernel, THREADS, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

}  // namespace bwd

}  // namespace ce

}  // namespace

void cross_entropy_rows_bf16(__nv_bfloat16* logits, const int64_t* targets, int rows, int V,
                             int ignore_index, const float* scale, float* loss, float* lse,
                             cudaStream_t stream) {
    SPARK_REQUIRE(V > 0 && V % 8 == 0, "cross_entropy_rows: V must be a multiple of 8");
    SPARK_REQUIRE(is_aligned16(logits), "cross_entropy_rows: logits must be 16-byte aligned");
    if (rows <= 0) return;
    const int grid = std::min(rows, num_sms());
    ce::ce_rows_kernel<<<grid, ce::ROW_THREADS, 0, stream>>>(logits, targets, rows, V, ignore_index,
                                                             scale, loss, lse, scale != nullptr);
    SPARK_CHECK_LAUNCH();
}

void transpose_pad_bf16(const __nv_bfloat16* x, int rows, int ldx, __nv_bfloat16* out, int ncols,
                        int ld, cudaStream_t stream) {
    SPARK_REQUIRE(ld >= rows && ncols <= ldx, "transpose_pad: bad shape");
    if (ncols <= 0 || ld <= 0) return;
    const dim3 grid(cdiv(ncols, ce::TP), cdiv(ld, ce::TP));
    ce::transpose_pad_kernel<<<grid, 256, 0, stream>>>(x, rows, ldx, out, ncols, ld);
    SPARK_CHECK_LAUNCH();
}

bool ce_gemm_supports(int V, int D) {
    return V >= ce::BN && V % ce::BN == 0 && D > 0 && D % 64 == 0;
}

void ce_gemm_bf16(const __nv_bfloat16* h, const __nv_bfloat16* w, int T, int V, int D, int mode,
                  const CeGemmEpilogue& ep, cudaStream_t stream) {
    SPARK_REQUIRE(ce_gemm_supports(V, D),
                  "ce_gemm: requires V a multiple of 128 and D a multiple of 64");
    SPARK_REQUIRE(h && w, "ce_gemm: null pointer");
    SPARK_REQUIRE(is_aligned16(h) && is_aligned16(w), "ce_gemm: h and w must be 16-byte aligned");
    if (T <= 0) return;
    ce::Params p{};
    p.out = ep.out;
    p.part = reinterpret_cast<float2*>(ep.partials);
    p.tgt_logit = ep.target_logit;
    p.targets = ep.targets;
    p.ignore_index = ep.ignore_index;
    p.lse = ep.lse;
    p.scale = ep.scale;
    switch (mode) {
        case CE_GEMM_LOGITS:
            SPARK_REQUIRE(ep.out, "ce_gemm (logits): needs out");
            ce::launch<CE_GEMM_LOGITS>(h, w, T, V, D, p, stream);
            break;
        case CE_GEMM_STATS:
            SPARK_REQUIRE(ep.partials && ep.target_logit && ep.targets,
                          "ce_gemm (stats): needs partials, target_logit and targets");
            ce::launch<CE_GEMM_STATS>(h, w, T, V, D, p, stream);
            break;
        case CE_GEMM_GRAD:
            SPARK_REQUIRE(ep.out && ep.targets && ep.lse && ep.scale,
                          "ce_gemm (grad): needs out, targets, lse and scale");
            ce::launch<CE_GEMM_GRAD>(h, w, T, V, D, p, stream);
            break;
        default:
            SPARK_REQUIRE(false, "ce_gemm: unknown mode");
    }
}

void ce_combine(const float* partials, const float* target_logit, const int64_t* targets, int T,
                int V, int ignore_index, float* lse, float* loss, cudaStream_t stream) {
    if (T <= 0) return;
    constexpr int threads = 256;
    const int rows_per_block = threads / 32;
    ce::ce_combine_kernel<<<cdiv(T, rows_per_block), threads, 0, stream>>>(
        reinterpret_cast<const float2*>(partials), target_logit, targets, T, V / ce::BN, V,
        ignore_index, lse, loss);
    SPARK_CHECK_LAUNCH();
}

int ce_backward_slots() {
    return ce::bwd::SLOTS;
}

void ce_backward_bf16(const __nv_bfloat16* h, const __nv_bfloat16* w, int T, int V, int D,
                      const CeBackwardArgs& a, cudaStream_t stream) {
    using namespace ce::bwd;
    SPARK_REQUIRE(ce_gemm_supports(V, D),
                  "ce_backward: requires V a multiple of 128 and D a multiple of 64");
    SPARK_REQUIRE(h && w && a.targets && a.lse && a.scale && a.dh && a.slots && a.flags && a.dw,
                  "ce_backward: null pointer");
    SPARK_REQUIRE(is_aligned16(h) && is_aligned16(w) && is_aligned16(a.dh) &&
                      is_aligned16(a.slots) && is_aligned16(a.dw),
                  "ce_backward: operands must be 16-byte aligned");
    if (T <= 0) return;
    const int grid = resident_blocks();
    const int kbytes = D * 2;
    const int tiles_m = cdiv(T, BM);
    const CUtensorMap tmA = make_tensor_map_2d_u8(h, T, kbytes, BM, BKB, CU_TENSOR_MAP_SWIZZLE_64B);
    const CUtensorMap tmB = make_tensor_map_2d_u8(w, V, kbytes, BN, BKB, CU_TENSOR_MAP_SWIZZLE_64B);
    const CUtensorMap tmDh = make_tensor_map_2d(CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, a.dh,
                                                static_cast<uint64_t>(tiles_m) * BM, D, P_ROWS, BK,
                                                CU_TENSOR_MAP_SWIZZLE_128B);
    SPARK_REQUIRE(a.slots_n >= 1, "ce_backward: at least one slot");
    const CUtensorMap tmDw = make_tensor_map_2d(CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4, a.slots,
                                                static_cast<uint64_t>(a.slots_n) * BN, D, P_ROWS,
                                                BK, CU_TENSOR_MAP_SWIZZLE_128B);
    Params p{};
    p.M = T;
    p.tiles_m = tiles_m;
    p.tiles_n = V / BN;
    p.KT = D / BK;
    p.grid = grid;
    p.D = D;
    p.targets = a.targets;
    p.lse = a.lse;
    p.scale = a.scale;
    p.ignore_index = a.ignore_index;
    p.dw = a.dw;
    p.slots = a.slots;
    p.done = a.flags;
    p.folded = a.flags + p.tiles_n;
    p.next_tile = a.flags + 2 * p.tiles_n;
    p.first = a.first ? 1 : 0;
    p.nslots = a.slots_n;
    ce_bwd_kernel<<<grid, THREADS, smem_bytes(), stream>>>(tmA, tmB, tmDh, tmDw, p);
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
