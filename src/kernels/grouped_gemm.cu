// Grouped GEMM for MoE prefill: every expert's GEMM in one persistent launch.
//
//   C[r, :] = s_a * s_b[e] * A[r, :] * W[e]^T    for r in offsets[e] .. offsets[e+1]-1
//
// A is [T, K] with its rows sorted by expert (offsets int32 [E + 1] on the device, the
// w1gemm_moe convention: offsets[0] == 0, offsets[E] == T, empty experts allowed), W is
// [E, N, K] (each expert's weight as nn.Linear stores it, K contiguous), C is [T, N] bf16,
// fp32 accumulate. Two element types: bf16 (s_a = s_b = 1) and e4m3 with a per-tensor
// activation scale and one weight scale per expert, through the block-scaled
// kind::mxf8f6f4 mma with unit block scales (fp8gemm's instruction).
//
// The host never sees the per-expert row counts. One launch of a grid of resident blocks
// (one 288-thread block per SM) does all of it:
//   * the tile list is built on the device: the producer warp reads offsets, takes
//     ceil(rows_e / 128) m-tiles per expert and scans them into a tile prefix in shared
//     memory, before the block's only __syncthreads. Every block computes the same prefix,
//     so every block derives the same schedule with no global coordination. The tiles of
//     expert e are its m-tiles times N / 128 n-tiles, m-tile fastest, so the tiles that run
//     at the same time share an expert's weight columns in L2;
//   * the schedule is variant 4/6's Stream-K hybrid: with G resident blocks and `tiles`
//     tiles, the last partial wave plus one full wave are cut into G equal ranges of
//     (tile, k-step) work and summed through the deterministic K-ordered chain fixup of
//     hgemm_streamk.cuh (one fp32 slot and one epoch-tagged flag per Stream-K tile); every
//     earlier tile is whole and goes to block tile % G. So an expert with 3 rows costs one
//     tile of k-steps, not an SM idling while a large expert finishes;
//   * the mainloop is fp8gemm variant 2's (fp8gemm_tma.cu) and hgemm variant 6's: one
//     producer lane publishes each piece through an int4 ring and issues two TMA boxes per
//     stage, 128 rows by BKB bytes of A from a 2-D map over [T, K] and 128 rows by BKB bytes
//     of the expert's weight from one 3-D map over [E][N][K] with the expert as the outer
//     coordinate; eight consumer warps run fp8gemm_tile's ldmatrix + mma.sync body on the
//     landed stage. Both operands are K-contiguous for both types, so a stage is the same
//     bytes either way: for bf16 a 32-byte k-step is one m16n8k16 instead of one m16n8k32
//     on the same fragments (bf16_stage below), everything else is shared.
// A tile's 128 rows start at its expert's first row plus 128 * m-tile and may run into the
// next expert's rows (or past T, where the copy engine zero-fills): those rows are computed
// and never stored. The epilogue stores rows < offsets[e+1] only.
//
// Two additions for whole MoE layers (docs/design/moe_layer.md): the tile height is a
// template parameter, 128 rows or, for bf16 experts of a few hundred rows or fewer, 64 or 16
// (the host picks from T / E; the 128-row tile of an 8-row expert is 16 times its MMA work,
// which at small T bounds the GEMM before the weight stream does), and a SwiGLU epilogue for a
// weight whose rows alternate gate and up, which writes silu(gate) * up, N / 2 columns.
//
// Host rules: N % 128 == 0, K % 64 == 0 (bf16) or K % 128 == 0 (fp8), 1 <= E <= 512,
// 16-byte aligned A and W.

#include <cuda_fp8.h>

#include <algorithm>
#include <cstdlib>

#include "fp8gemm_tile.cuh"
#include "hgemm_streamk.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

namespace gg {

using hgemm_sk::ld_acquire_gpu;
using hgemm_sk::slot_idx;
using hgemm_sk::st_release_gpu;

constexpr int BN = 128;
constexpr int CONSUMER_WARPS = 8;
constexpr int CONSUMERS = CONSUMER_WARPS * 32;
constexpr int THREADS = CONSUMERS + 32;
constexpr int MAX_E = 512;
constexpr int ITEM_DEPTH = 2;     // pieces the producer may publish ahead of the consumers
constexpr int FIXUP_BARRIER = 1;  // named barrier id for the 256 consumer threads
constexpr int MIN_SHARE = 32;     // least k-steps per Stream-K block (hgemm_sk::kMinShare)
constexpr int MIN_SHARE_16 = 8;   // ... for the 16-row tile, whose fp32 slot is only 8 KB

// BM is the tile height in rows (128, or 64 and 16 for experts of few rows), BKB the stage
// depth in bytes of K (64 or 128): 32 or 64 bf16, 64 or 128 e4m3. The eight consumer warps
// are 2 x 4 over a tile of 64 rows or more (64 x 32 or 32 x 32 each) and 1 x 8 over a
// 16-row tile (16 x 16 each, one m16 fragment by two n8).
template <int BM>
constexpr int warps_m() {
    return BM >= 64 ? 2 : 1;
}
template <int BM, int BKB, int STAGES>
using TileCfg =
    fp8gemm_tile::Cfg<BM, BN, BKB, STAGES, warps_m<BM>(), CONSUMER_WARPS / warps_m<BM>()>;

// Dynamic shared memory: the stages (1 KB aligned, the swizzle is keyed on the absolute
// address), the item ring, the barriers, then the m-tile prefix (MAX_E + 1 ints).
template <int BM, int BKB, int STAGES>
constexpr int smem_bytes() {
    return (STAGES * TileCfg<BM, BKB, STAGES>::STAGE + ITEM_DEPTH * 16 +
            (2 * STAGES + 2 * ITEM_DEPTH) * 8 + (MAX_E + 1) * 4 + 1023) /
           1024 * 1024;
}
static_assert(smem_bytes<128, 128, 3>() <= 101376, "three 32 KB stages and the prefix fit");
static_assert(smem_bytes<64, 128, 3>() <= 101376, "three 24 KB stages and the prefix fit");
static_assert(smem_bytes<16, 128, 4>() <= 101376, "four 18 KB stages and the prefix fit");

struct Params {
    const int32_t* offsets;
    int E, N;
    int tiles_n;                // N / BN
    int KT;                     // k-steps per tile
    int grid;                   // resident blocks in the launch
    float* slots;               // 2 * grid Stream-K tiles of BM x BN fp32
    unsigned long long* flags;  // per Stream-K tile: (epoch << 32) | k-steps accumulated
    unsigned epoch;
    const float* scale_a;  // fp8: [1] or null (1)
    const float* scale_b;  // fp8: [E] or null (1)
};

// The work split every block derives from the tile count: tiles [0, dp) are whole and
// belong to block tile % grid; the sk tiles after them are cut into `ranges` equal spans of
// k-steps, one per block 0 .. ranges-1. With tiles % grid == 0 there is no Stream-K region.
struct Split {
    int dp, sk, ranges;
};
__device__ __forceinline__ Split split_work(int tiles, int grid, int KT, int min_share) {
    Split s;
    const int rem = tiles % grid;
    s.sk = rem == 0 ? 0 : rem + (tiles >= grid ? grid : 0);
    s.dp = tiles - s.sk;
    const long long iters = static_cast<long long>(s.sk) * KT;
    s.ranges = static_cast<int>(
        min(static_cast<long long>(grid), max(static_cast<long long>(s.sk), iters / min_share)));
    return s;
}
__device__ __forceinline__ int range_start(int sk_iters, int ranges, int c) {
    return static_cast<int>(static_cast<long long>(c) * sk_iters / ranges);
}

// Tile -> (expert, m-tile, n-tile). Expert e owns tiles [tp[e] * tiles_n, tp[e+1] * tiles_n),
// m-tile fastest. e is the largest with tp[e] <= tile / tiles_n, which skips empty experts.
__device__ __forceinline__ int tile_expert(const int* tp, int E, int tiles_n, int tile) {
    const int q = tile / tiles_n;
    int lo = 0, hi = E - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (tp[mid] <= q)
            lo = mid;
        else
            hi = mid - 1;
    }
    return lo;
}
__device__ __forceinline__ void tile_coords(const int* tp, int tiles_n, int tile, int e, int& mt,
                                            int& nt) {
    const int local = tile - tp[e] * tiles_n;
    const int mte = tp[e + 1] - tp[e];
    mt = local % mte;
    nt = local / mte;
}

// One 32-byte k-step of a bf16 stage: fp8gemm_tile::mma_step's ldmatrix addressing (it works
// on bytes: an 8x8 b16 matrix is 8 rows by 16 bytes, which is 8 bf16 k here and the m16n8k16
// fragment word of lane (g, t), k = 2t, 2t+1 of row g), one m16n8k16 per tile pair.
template <class C, int KK>
__device__ __forceinline__ void bf16_step(const unsigned char* as, const unsigned char* bs, int wm,
                                          int wn, int lane, typename C::Acc& acc) {
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
template <class C>
__device__ __forceinline__ void bf16_stage(const unsigned char* as, const unsigned char* bs, int wm,
                                           int wn, int lane, typename C::Acc& acc) {
    bf16_step<C, 0>(as, bs, wm, wn, lane, acc);
    bf16_step<C, 32>(as, bs, wm, wn, lane, acc);
    if constexpr (C::BK >= 128) {
        bf16_step<C, 64>(as, bs, wm, wn, lane, acc);
        bf16_step<C, 96>(as, bs, wm, wn, lane, acc);
    }
}

// The SwiGLU epilogue, for a weight whose rows alternate gate and up (row 2j = gate_j, row
// 2j + 1 = up_j, interleave_experts): the accumulator gives lane (g, c) columns 2c and 2c + 1
// of each n8 tile, one gate and its up, so the lane forms silu(gate) * up in fp32 from the
// unrounded sums and the output has N / 2 columns. hgemm_epilogue.cuh's SwiGLU store: lanes
// 2q and 2q + 1 hold output columns j and j + 1 of the same two rows, one shuffle swaps a
// value, and the even lane stores the upper row's pair, the odd lane the lower row's.
__device__ __forceinline__ float silu_f(float x) {
    return __fdividef(x, 1.0f + __expf(-x));
}
template <class C>
__device__ __forceinline__ void store_swiglu(const typename C::Acc& acc, float scale,
                                             __nv_bfloat16* __restrict__ out, int M, int ldc,
                                             int bm, int bn) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int wm = warp / C::WN;
    const int wn = warp % C::WN;
    const int g = lane >> 2;
    const bool odd = lane & 1;
#pragma unroll
    for (int mi = 0; mi < C::MT; ++mi) {
#pragma unroll
        for (int nj = 0; nj < C::NT; ++nj) {
            const float* v = acc[mi][nj];
            const float y0 = silu_f(v[0] * scale) * (v[1] * scale);  // row
            const float y1 = silu_f(v[2] * scale) * (v[3] * scale);  // row + 8
            const float recv = __shfl_xor_sync(kFullMask, odd ? y0 : y1, 1);
            const int row = bm + wm * C::WROWS + mi * 16 + g + (odd ? 8 : 0);
            const int j = (bn + wn * C::WCOLS + nj * 8 + (lane & 3) * 2) / 2 - (odd ? 1 : 0);
            if (row < M)
                *reinterpret_cast<__nv_bfloat162*>(out + static_cast<size_t>(row) * ldc + j) =
                    __floats2bfloat162_rn(odd ? recv : y0, odd ? y1 : recv);
        }
    }
}

template <bool FP8, int BM, int BKB, int STAGES, bool SWIGLU>
__global__ void __launch_bounds__(THREADS, 1)
    grouped_kernel(const __grid_constant__ CUtensorMap tmA, const __grid_constant__ CUtensorMap tmW,
                   __nv_bfloat16* __restrict__ Cout, Params p) {
    using C = TileCfg<BM, BKB, STAGES>;
    static_assert(BKB == 64 || BKB == 128, "the TMA swizzles cover 64 B and 128 B rows");
    constexpr int A_STAGE = C::A_STAGE, STAGE_BYTES = C::STAGE;
    constexpr int MT = C::MT, NT = C::NT, WROWS = C::WROWS;

    extern __shared__ __align__(1024) unsigned char smem[];
    int4* items = reinterpret_cast<int4*>(smem + STAGES * STAGE_BYTES);
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(items + ITEM_DEPTH);
    uint64_t* empty_bar = full_bar + STAGES;
    uint64_t* item_full = empty_bar + STAGES;
    uint64_t* item_empty = item_full + ITEM_DEPTH;
    int* tp = reinterpret_cast<int*>(item_empty + ITEM_DEPTH);  // [E + 1] m-tile prefix
    if (smem_u32(smem) % 1024 != 0) __trap();

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int E = p.E;

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
    if (warp == CONSUMER_WARPS) {
        // The tile list: m-tiles per expert, scanned 32 experts at a time.
        int run = 0;
        for (int base = 0; base < E; base += 32) {
            const int e = base + lane;
            int m = 0;
            if (e < E) m = cdiv(__ldg(p.offsets + e + 1) - __ldg(p.offsets + e), BM);
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int v = __shfl_up_sync(kFullMask, m, o);
                if (lane >= o) m += v;
            }
            if (e < E) tp[e + 1] = run + m;
            run += __shfl_sync(kFullMask, m, 31);
        }
        if (lane == 0) tp[0] = 0;
    }
    __syncthreads();  // the only block-wide barrier in the kernel

    const int tiles = tp[E] * p.tiles_n;
    const Split sp = split_work(tiles, p.grid, p.KT, BM == 16 ? MIN_SHARE_16 : MIN_SHARE);
    const int sk_iters = sp.sk * p.KT;

    if (warp == CONSUMER_WARPS) {
        // Producer warp: lane 0 owns the schedule and the loads.
        if (lane == 0) {
            prefetch_tensormap(&tmA);
            prefetch_tensormap(&tmW);
            unsigned g_kt = 0;   // k-tiles issued so far, across pieces
            unsigned n_pub = 0;  // pieces published so far
            auto piece = [&](int tile, int kb, int ke) {
                int e = 0;
                if (tile >= 0) e = tile_expert(tp, E, p.tiles_n, tile);
                const int r = n_pub % ITEM_DEPTH;
                const unsigned use = n_pub / ITEM_DEPTH;
                if (n_pub >= ITEM_DEPTH) mbar_wait(&item_empty[r], (use - 1) & 1);
                items[r] = make_int4(tile, kb, ke, e);
                mbar_arrive(&item_full[r]);
                ++n_pub;
                if (tile < 0) return;
                int mt, nt;
                tile_coords(tp, p.tiles_n, tile, e, mt, nt);
                const int row0 = __ldg(p.offsets + e) + mt * BM;
                const int bn = nt * BN;
                for (int kt = kb; kt < ke; ++kt, ++g_kt) {
                    const int s = g_kt % STAGES;
                    const unsigned use_s = g_kt / STAGES;
                    if (g_kt >= STAGES) mbar_wait(&empty_bar[s], (use_s - 1) & 1);
                    mbar_arrive_expect_tx(&full_bar[s], STAGE_BYTES);
                    unsigned char* as = smem + s * STAGE_BYTES;
                    tma_load_2d(as, &tmA, &full_bar[s], kt * BKB, row0);  // BM rows x BKB
                    tma_load_3d(as + A_STAGE, &tmW, &full_bar[s], kt * BKB, bn, e);
                }
            };
            for (int t = blockIdx.x; t < sp.dp; t += p.grid) piece(t, 0, p.KT);
            if (static_cast<int>(blockIdx.x) < sp.ranges) {
                // Static ranges walked from the top: the piece that starts a tile (and never
                // waits) comes first, the piece that ends one last, as in hgemm variant 6.
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
            piece(-1, 0, 0);  // end of work
        }
        return;
    }

    // Consumer warps.
    const int wm = warp / C::WN;
    const int wn = warp % C::WN;
    const float sa = (FP8 && p.scale_a) ? *p.scale_a : 1.f;
    unsigned g_kt = 0;
    unsigned n_item = 0;
    typename C::Acc acc;

    for (;;) {
        const int r = n_item % ITEM_DEPTH;
        mbar_wait(&item_full[r], (n_item / ITEM_DEPTH) & 1);
        const int4 itm = items[r];
        __syncwarp();
        if (lane == 0) mbar_arrive(&item_empty[r]);
        ++n_item;
        if (itm.x < 0) break;
        const int tile = itm.x, kb = itm.y, ke = itm.z, e = itm.w;
        int mt, nt;
        tile_coords(tp, p.tiles_n, tile, e, mt, nt);
        const int row0 = __ldg(p.offsets + e) + mt * BM;
        const int row_end = __ldg(p.offsets + e + 1);
        const int m_valid = row_end - row0;
        const int bn = nt * BN;

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
            if constexpr (FP8) {
                fp8gemm_tile::MxFrag<C> none;
                fp8gemm_tile::mma_stage<C, false>(as, as + A_STAGE, wm, wn, lane, acc, none);
            } else {
                bf16_stage<C>(as, as + A_STAGE, wm, wn, lane, acc);
            }
            fence_proxy_async_smem();
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty_bar[s]);
        }

        // Chain fixup for Stream-K tiles, hgemm variant 6's, with the producer out of it.
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
        if (finish) {
            const float scale = FP8 ? sa * (p.scale_b ? __ldg(p.scale_b + e) : 1.f) : 1.f;
            // Rows past the expert's last row (row >= row_end) are skipped one by one.
            if constexpr (SWIGLU)
                store_swiglu<C>(acc, scale, Cout, row_end, p.N / 2, row0, bn);
            else
                fp8gemm_tile::store_bf16<C>(acc, scale, Cout, row_end, p.N, row0, bn);
        }
    }
}

// ---- host side --------------------------------------------------------------------------

struct Workspace {
    float* slots = nullptr;
    unsigned long long* flags = nullptr;
    size_t tiles = 0;
    unsigned epoch = 0;
};

template <bool FP8, int BM, int BKB, int STAGES, bool SWIGLU>
int resident_blocks() {
    constexpr int bytes = smem_bytes<BM, BKB, STAGES>();
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(grouped_kernel<FP8, BM, BKB, STAGES, SWIGLU>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, grouped_kernel<FP8, BM, BKB, STAGES, SWIGLU>, THREADS, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <bool FP8, int BKB, int STAGES, bool SWIGLU = false, int BM = 128>
void launch(const void* A, const void* W, const int32_t* offsets, __nv_bfloat16* Cout, int T, int E,
            int N, int K, const float* scale_a, const float* scale_b, cudaStream_t stream) {
    constexpr int bytes = smem_bytes<BM, BKB, STAGES>();
    const int grid = resident_blocks<FP8, BM, BKB, STAGES, SWIGLU>();
    const int kbytes = K * (FP8 ? 1 : 2);
    constexpr CUtensorMapSwizzle swz =
        BKB == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
    // Both maps move raw bytes: a bf16 row of K values is 2K bytes, and the swizzle is
    // defined on bytes. W's map is [E][N][2K or K] with a box of 128 rows of one expert.
    const CUtensorMap tmA = make_tensor_map_2d_u8(A, T, kbytes, BM, BKB, swz);
    const CUtensorMap tmW = make_tensor_map_3d_u8(W, kbytes, N, E, BKB, BN, swz);

    // The Stream-K region is at most the last partial wave plus one wave: 2 * grid tiles.
    static Workspace w;
    const size_t need = 2 * static_cast<size_t>(grid);
    if (w.tiles < need) {
        if (w.slots) SPARK_CUDA_CHECK(cudaFree(w.slots));
        if (w.flags) SPARK_CUDA_CHECK(cudaFree(w.flags));
        w.tiles = need;
        SPARK_CUDA_CHECK(cudaMalloc(&w.slots, w.tiles * BM * BN * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMalloc(&w.flags, w.tiles * sizeof(unsigned long long)));
        SPARK_CUDA_CHECK(cudaMemset(w.flags, 0, w.tiles * sizeof(unsigned long long)));  // once
    }
    if (++w.epoch == 0) w.epoch = 1;

    Params p;
    p.offsets = offsets;
    p.E = E;
    p.N = N;
    p.tiles_n = N / BN;
    p.KT = kbytes / BKB;
    p.grid = grid;
    p.slots = w.slots;
    p.flags = w.flags;
    p.epoch = w.epoch;
    p.scale_a = scale_a;
    p.scale_b = scale_b;
    grouped_kernel<FP8, BM, BKB, STAGES, SWIGLU>
        <<<grid, THREADS, bytes, stream>>>(tmA, tmW, Cout, p);
    SPARK_CHECK_LAUNCH();
}

// Pipeline configurations (BKB bytes of K per stage, stages). SPARK_GROUPED_CONFIG=<index>
// forces one. The defaults are what the plain GEMMs settled on: hgemm variant 6's two 64-k
// bf16 stages and fp8gemm variant 2's three 128-byte stages for many waves.
using LaunchFn = void (*)(const void*, const void*, const int32_t*, __nv_bfloat16*, int, int, int,
                          int, const float*, const float*, cudaStream_t);
template <bool FP8>
constexpr LaunchFn CONFIGS[] = {
    launch<FP8, 128, 2>,  // 0: 64 KB
    launch<FP8, 128, 3>,  // 1: 96 KB
    launch<FP8, 64, 4>,   // 2: 64 KB in 16 KB stages
    launch<FP8, 64, 6>,   // 3: 96 KB in 16 KB stages
};
constexpr int NUM_CONFIGS = static_cast<int>(sizeof(CONFIGS<false>) / sizeof(CONFIGS<false>[0]));
constexpr int CONFIG_BF16 = 0;
constexpr int CONFIG_FP8 = 1;

int forced_config() {
    static int idx = -2;
    if (idx == -2) {
        idx = -1;
        if (const char* e = std::getenv("SPARK_GROUPED_CONFIG")) {
            const int v = std::atoi(e);
            if (v >= 0 && v < NUM_CONFIGS) idx = v;
        }
    }
    return idx;
}

// The tile height, from the rows per expert on average, which the host has without the
// counts (T rows over E experts): a 128-row tile of an expert with 8 rows is 16 times the
// MMA work of its rows, and at a few rows per expert that padding, not the weight stream, is
// what bounds the bf16 GEMM (docs/design/moe_layer.md). fp8 keeps the 128-row tile: its mma
// is four times the rate, and the 64-row tile measured 7 to 20% slower on four of the five
// fp8 shapes of scripts/bench_grouped.py it would have taken. SPARK_GROUPED_BM=16|64|128 forces one
// for both types; SPARK_GROUPED_CONFIG (the 128-row pipelines) implies 128.
constexpr int SMALL_ROWS = 24;  // average rows per expert up to which the 16-row tile runs
constexpr int MID_ROWS = 512;   // ... and the 64-row tile (swept: docs/design/moe_layer.md)
int pick_bm(int T, int E, bool fp8) {
    static int forced = -2;
    if (forced == -2) {
        forced = -1;
        if (const char* e = std::getenv("SPARK_GROUPED_BM")) {
            const int v = std::atoi(e);
            if (v == 16 || v == 64 || v == 128) forced = v;
        }
    }
    if (forced > 0) return forced;
    if (fp8 || forced_config() >= 0) return 128;
    const int avg = cdiv(T, E);
    return avg <= SMALL_ROWS ? 16 : avg <= MID_ROWS ? 64 : 128;
}

// One call: the tile height, then the pipeline (the 16-row tile streams 16 KB of weight per
// 18 KB stage and keeps four in flight, the 64-row tile three stages of 24 KB).
template <bool FP8, bool SWIGLU>
void run(const void* A, const void* W, const int32_t* offsets, __nv_bfloat16* C, int T, int E,
         int N, int K, const float* scale_a, const float* scale_b, cudaStream_t stream) {
    const int bm = pick_bm(T, E, FP8);
    if (bm == 16) {
        launch<FP8, 128, 4, SWIGLU, 16>(A, W, offsets, C, T, E, N, K, scale_a, scale_b, stream);
    } else if (bm == 64) {
        launch<FP8, 128, 3, SWIGLU, 64>(A, W, offsets, C, T, E, N, K, scale_a, scale_b, stream);
    } else if constexpr (SWIGLU) {
        launch<FP8, 128, FP8 ? 3 : 2, true>(A, W, offsets, C, T, E, N, K, scale_a, scale_b, stream);
    } else {
        int cfg = forced_config();
        if (cfg < 0) cfg = FP8 ? CONFIG_FP8 : CONFIG_BF16;
        CONFIGS<FP8>[cfg](A, W, offsets, C, T, E, N, K, scale_a, scale_b, stream);
    }
}

}  // namespace gg

}  // namespace

bool grouped_gemm_supports(int E, int N, int K, bool fp8) {
    return E >= 1 && E <= gg::MAX_E && N >= gg::BN && N % gg::BN == 0 && K > 0 &&
           K % (fp8 ? 128 : 64) == 0;
}

void grouped_gemm_bf16(const __nv_bfloat16* A, const __nv_bfloat16* W, const int32_t* offsets,
                       __nv_bfloat16* C, int T, int E, int N, int K, cudaStream_t stream) {
    SPARK_REQUIRE(grouped_gemm_supports(E, N, K, false),
                  "grouped_gemm (bf16): requires 1 <= E <= 512, N % 128 == 0 and K % 64 == 0");
    SPARK_REQUIRE(A && W && offsets && C, "grouped_gemm: null pointer");
    SPARK_REQUIRE(is_aligned16(A) && is_aligned16(W),
                  "grouped_gemm: A and W must be 16-byte aligned");
    if (T <= 0) return;
    gg::run<false, false>(A, W, offsets, C, T, E, N, K, nullptr, nullptr, stream);
}

void grouped_gemm_fp8(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* W, const int32_t* offsets,
                      __nv_bfloat16* C, int T, int E, int N, int K, const float* scale_a,
                      const float* scale_b, cudaStream_t stream) {
    SPARK_REQUIRE(grouped_gemm_supports(E, N, K, true),
                  "grouped_gemm (fp8): requires 1 <= E <= 512, N % 128 == 0 and K % 128 == 0");
    SPARK_REQUIRE(A && W && offsets && C, "grouped_gemm: null pointer");
    SPARK_REQUIRE(is_aligned16(A) && is_aligned16(W),
                  "grouped_gemm: A and W must be 16-byte aligned");
    if (T <= 0) return;
    gg::run<true, false>(A, W, offsets, C, T, E, N, K, scale_a, scale_b, stream);
}

// The SwiGLU form runs the default pipeline of each type at every tile height
// (SPARK_GROUPED_CONFIG does not apply): C is [T, N / 2], C[r, j] = silu(G) * U with G and U
// rows 2j and 2j + 1 of the expert's weight, both from fp32 sums, scaled (fp8) before the
// activation.
void grouped_gemm_swiglu_bf16(const __nv_bfloat16* A, const __nv_bfloat16* W,
                              const int32_t* offsets, __nv_bfloat16* C, int T, int E, int N, int K,
                              cudaStream_t stream) {
    SPARK_REQUIRE(grouped_gemm_supports(E, N, K, false),
                  "grouped_gemm (bf16): requires 1 <= E <= 512, N % 128 == 0 and K % 64 == 0");
    SPARK_REQUIRE(A && W && offsets && C, "grouped_gemm: null pointer");
    SPARK_REQUIRE(is_aligned16(A) && is_aligned16(W),
                  "grouped_gemm: A and W must be 16-byte aligned");
    if (T <= 0) return;
    gg::run<false, true>(A, W, offsets, C, T, E, N, K, nullptr, nullptr, stream);
}

void grouped_gemm_swiglu_fp8(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* W, const int32_t* offsets,
                             __nv_bfloat16* C, int T, int E, int N, int K, const float* scale_a,
                             const float* scale_b, cudaStream_t stream) {
    SPARK_REQUIRE(grouped_gemm_supports(E, N, K, true),
                  "grouped_gemm (fp8): requires 1 <= E <= 512, N % 128 == 0 and K % 128 == 0");
    SPARK_REQUIRE(A && W && offsets && C, "grouped_gemm: null pointer");
    SPARK_REQUIRE(is_aligned16(A) && is_aligned16(W),
                  "grouped_gemm: A and W must be 16-byte aligned");
    if (T <= 0) return;
    gg::run<true, true>(A, W, offsets, C, T, E, N, K, scale_a, scale_b, stream);
}

}  // namespace spark
