// fp8 (e4m3) GEMM, variant 2: TMA loads and a warp-specialized mbarrier pipeline.
//
//   C[M,N] = scale_a * scale_b * A[M,K] * Bt[N,K]^T     e4m3 in, bf16 out, fp32 accumulate
//
// hgemm variant 5 (src/kernels/hgemm_tma.cu) on 8-bit operands. Same 128x128 block tile, 2x4
// consumer warp grid, ldmatrix + mma.sync.m16n8k32 k-loop body (fp8gemm_tile.cuh), register
// epilogue with the scale and tail split-K as variant 1; one producer warp issues the copies:
//   * both operands are K-contiguous, so a stage is two boxes of the same shape, 128 rows by
//     BK bytes of A and 128 rows by BK bytes of Bt, one cp.async.bulk.tensor each. The copy
//     engine's 128-byte swizzle (BK = 128) XORs the 16-byte chunk with row % 8 and the 64-byte
//     one (BK = 64) with (row / 2) % 4, which are fp8gemm_tile::swz<128> and swz<64>: the box
//     inner dimension is one swizzle span, and 8-bit elements change nothing about that, the
//     swizzle is defined on bytes;
//   * each stage has a "full" mbarrier (producer arrives with the byte count, the copy engine
//     completes it) and an "empty" mbarrier (each consumer warp arrives once it has read the
//     stage), and nothing in the k-loop is a block-wide barrier.
// Host guarantees M % 128 == 0, N % 128 == 0, K % BK == 0 and a grid of at least one wave.
//
// MX mode: the consumer warps fetch their scale chunks (fp8gemm_tile.cuh, MxChunk) from
// global memory one group of stages ahead, as variant 1 does; the producer and the stages
// are unchanged.
//
// Configurations 11 to 19 run a second kernel (fp8gemm_cl_kernel, per-tensor mode): thread
// block clusters whose blocks share A or Bt panels through multicast TMA loads, a 128x256
// tile, and fp4gemm variant 2's unrolled consumer loop. Description above the kernel.

#include <cuda_fp8.h>

#include <algorithm>
#include <cstdlib>

#include "fp8gemm_tile.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

namespace v2 {

using fp8gemm_tile::MxScales;

constexpr int BM = 128, BN = 128;
constexpr int TAIL_BARRIER = 1;  // named barrier id for the consumer threads

// The consumer warp grid is WARPS x WARPS (2x2 warps of 64x64, or 2x4 of 64x32 as hgemm v5),
// plus one producer warp.
template <int BK, int STAGES, int WARPS_M, int WARPS_N>
using TileCfg = fp8gemm_tile::Cfg<BM, BN, BK, STAGES, WARPS_M, WARPS_N>;

// Stages at the front of the dynamic region (1 KB aligned, since both swizzles are keyed on
// the absolute shared address), barriers after them, the total rounded to a whole KB so the
// second block on an SM starts aligned too.
template <int BK, int STAGES>
constexpr int smem_bytes() {
    return (STAGES * (BM + BN) * BK + 2 * STAGES * 8 + 16 + 1023) / 1024 * 1024;
}

struct Sched {
    int tiles_n;
    int dp_tiles;
    int split;
    float* ws;
    int* counters;
    int nomc = 0;  // cluster kernel, experiment: every block copies its whole stage
};

template <int BK, int STAGES, int WARPS_M, int WARPS_N, int MIN_BLOCKS, bool MX, int SFW>
__global__ void __launch_bounds__((WARPS_M * WARPS_N + 1) * 32, MIN_BLOCKS)
    fp8gemm_v2_kernel(const __grid_constant__ CUtensorMap tmA,
                      const __grid_constant__ CUtensorMap tmB, __nv_bfloat16* __restrict__ Cout,
                      int M, int N, int K, const float* __restrict__ scale_a,
                      const float* __restrict__ scale_b, MxScales mx, Sched sched) {
    using C = TileCfg<BK, STAGES, WARPS_M, WARPS_N>;
    static_assert(BK == 64 || BK == 128, "the TMA swizzles cover 64 B and 128 B rows");
    constexpr int CONSUMER_WARPS = WARPS_M * WARPS_N;
    constexpr int CONSUMERS = CONSUMER_WARPS * 32;
    constexpr int A_STAGE = C::A_STAGE, STAGE_BYTES = C::STAGE;
    constexpr int WROWS = C::WROWS, WCOLS = C::WCOLS, MT = C::MT, NT = C::NT;

    extern __shared__ __align__(1024) unsigned char smem[];
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(smem + STAGES * STAGE_BYTES);
    uint64_t* empty_bar = full_bar + STAGES;
    int* s_last = reinterpret_cast<int*>(empty_bar + STAGES);
    if (smem_u32(smem) % 1024 != 0) __trap();

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int KT = K / BK;
    int tile, kt_begin, kt_end, slice = 0;
    if (static_cast<int>(blockIdx.x) < sched.dp_tiles) {
        tile = blockIdx.x;
        kt_begin = 0;
        kt_end = KT;
    } else {
        const int r = blockIdx.x - sched.dp_tiles;
        tile = sched.dp_tiles + r / sched.split;
        slice = r % sched.split;
        kt_begin = static_cast<int>(static_cast<long long>(slice) * KT / sched.split);
        kt_end = static_cast<int>(static_cast<long long>(slice + 1) * KT / sched.split);
    }
    if constexpr (MX) {  // slices start on a scale-chunk boundary (host: split <= KT / SPC)
        constexpr int SPC = fp8gemm_tile::MxChunk<C, SFW>::SPC;
        kt_begin = kt_begin / SPC * SPC;
        if (slice + 1 < sched.split) kt_end = kt_end / SPC * SPC;
    }
    const int bm = (tile / sched.tiles_n) * BM;
    const int bn = (tile % sched.tiles_n) * BN;
    const int nkt = kt_end - kt_begin;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full_bar[s], 1);
            mbar_init(&empty_bar[s], CONSUMER_WARPS);
        }
        fence_mbar_init();
    }
    __syncthreads();

    if (warp == CONSUMER_WARPS) {
        // Producer warp, lane 0. Stage s serves k-tiles s, s+STAGES, ...; its n-th use waits
        // for the consumers' (n-1)-th release, then registers STAGE_BYTES on "full".
        if (lane == 0) {
            prefetch_tensormap(&tmA);
            prefetch_tensormap(&tmB);
            for (int kt = 0; kt < nkt; ++kt) {
                const int s = kt % STAGES;
                const unsigned use = kt / STAGES;
                if (kt >= STAGES) mbar_wait(&empty_bar[s], (use - 1) & 1);
                mbar_arrive_expect_tx(&full_bar[s], STAGE_BYTES);
                const int k0 = (kt_begin + kt) * BK;
                unsigned char* as = smem + s * STAGE_BYTES;
                tma_load_2d(as, &tmA, &full_bar[s], k0, bm);            // (k, m): 128 rows x BK
                tma_load_2d(as + A_STAGE, &tmB, &full_bar[s], k0, bn);  // (k, n): 128 rows x BK
            }
        }
        return;
    }

    // Consumer warps: variant 1's k-loop body on the stage the producer has filled.
    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;
    const float scale = *scale_a * *scale_b;

    typename C::Acc acc;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    auto stage = [&](int kt, const fp8gemm_tile::MxFrag<C>& words) {
        const int s = kt % STAGES;
        mbar_wait(&full_bar[s], (kt / STAGES) & 1);
        const unsigned char* as = smem + s * STAGE_BYTES;
        const unsigned char* bs = as + A_STAGE;
        fp8gemm_tile::mma_stage<C, MX>(as, bs, wm, wn, lane, acc, words);
        // The ldmatrix reads went through the generic proxy and the refill comes through the
        // async proxy: fence, then one arrive per warp hands the stage back.
        fence_proxy_async_smem();
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty_bar[s]);
    };
    if constexpr (!MX) {
        fp8gemm_tile::MxFrag<C> none;
        for (int kt = 0; kt < nkt; ++kt) stage(kt, none);
    } else {
        // Variant 1's group loop (fp8gemm_tile.cuh): the next group's scale chunks are
        // fetched while this group's SPC stages run.
        mx.a += static_cast<size_t>(bm) * mx.ld;
        mx.b += static_cast<size_t>(bn) * mx.ld;
        fp8gemm_tile::mx_loop<C, SFW>(
            nkt, lane, stage, [&](fp8gemm_tile::MxChunk<C, SFW>& ch, int kt) {
                fp8gemm_tile::mx_fetch<C, SFW>(ch, mx, kt_begin + kt, wm, wn, lane, BM);
            });
    }

    if (tile < sched.dp_tiles) {
        fp8gemm_tile::store_bf16<C>(acc, scale, Cout, M, N, bm, bn);
        return;
    }

    // Tail tile: this K-slice goes into the fp32 workspace with L2 atomics; the last slice
    // scales and converts. The producer warp has left, so the consumers use a named barrier.
    const int g = lane >> 2;
    const int c2 = (lane & 3) * 2;
    float* wt = sched.ws + static_cast<size_t>(tile - sched.dp_tiles) * BM * BN;
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int nj = 0; nj < NT; ++nj) {
            const int r0 = wm * WROWS + mi * 16 + g;
            const int c0 = wn * WCOLS + nj * 8 + c2;
            atomicAdd(wt + r0 * BN + c0, acc[mi][nj][0]);
            atomicAdd(wt + r0 * BN + c0 + 1, acc[mi][nj][1]);
            atomicAdd(wt + (r0 + 8) * BN + c0, acc[mi][nj][2]);
            atomicAdd(wt + (r0 + 8) * BN + c0 + 1, acc[mi][nj][3]);
        }
    }
    __threadfence();
    named_barrier_sync(TAIL_BARRIER, CONSUMERS);
    if (tid == 0) {
        *s_last = atomicAdd(sched.counters + (tile - sched.dp_tiles), 1) == sched.split - 1;
    }
    named_barrier_sync(TAIL_BARRIER, CONSUMERS);
    if (!*s_last) return;
    __threadfence();
    for (int i = tid; i < BM * BN / 2; i += CONSUMERS) {
        const int r = (2 * i) / BN;
        const int c = (2 * i) % BN;
        const float2 v = __ldcg(reinterpret_cast<const float2*>(wt + r * BN + c));
        *reinterpret_cast<__nv_bfloat162*>(Cout + static_cast<size_t>(bm + r) * N + bn + c) =
            __floats2bfloat162_rn(v.x * scale, v.y * scale);
    }
}

// ---- host side --------------------------------------------------------------------------

struct Workspace {
    float* ws = nullptr;
    int* counters = nullptr;
    size_t tiles = 0;
};

template <int BK, int STAGES, int WARPS_M, int WARPS_N, int MIN_BLOCKS, bool MX, int SFW>
int resident_blocks() {
    constexpr int bytes = smem_bytes<BK, STAGES>();
    constexpr int threads = (WARPS_M * WARPS_N + 1) * 32;
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(
            fp8gemm_v2_kernel<BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS, MX, SFW>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, fp8gemm_v2_kernel<BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS, MX, SFW>, threads,
            bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <int BK, int STAGES, int WARPS_M, int WARPS_N, int MIN_BLOCKS, bool MX, int SFW>
void launch_sfw(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M, int N,
                int K, const float* scale_a, const float* scale_b, MxScales mx,
                cudaStream_t stream) {
    constexpr int bytes = smem_bytes<BK, STAGES>();
    constexpr int THREADS = (WARPS_M * WARPS_N + 1) * 32;
    const int resident = resident_blocks<BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS, MX, SFW>();

    // Both boxes: 128 rows by BK bytes, swizzled over the BK-byte row. The maps are 128 bytes
    // of host arithmetic each and are rebuilt per call (27 ns each, measured for hgemm v5).
    constexpr CUtensorMapSwizzle swz =
        BK == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
    const CUtensorMap tmA = make_tensor_map_2d_u8(A, M, K, BM, BK, swz);
    const CUtensorMap tmB = make_tensor_map_2d_u8(Bt, N, K, BN, BK, swz);

    Sched s;
    s.tiles_n = N / BN;
    const int tiles = (M / BM) * s.tiles_n;
    const int KT = K / BK;
    const int tail = tiles % resident;
    // MX slices are whole scale-chunk groups (fp8gemm_tile::MxChunk).
    const int max_split =
        MX ? std::max(1,
                      KT / fp8gemm_tile::MxChunk<TileCfg<BK, STAGES, WARPS_M, WARPS_N>, SFW>::SPC)
           : KT;
    s.split = tail > 0 ? std::min(max_split, resident / tail) : 1;
    if (s.split <= 1) {
        s.dp_tiles = tiles;
        s.split = 1;
        s.ws = nullptr;
        s.counters = nullptr;
        fp8gemm_v2_kernel<BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS, MX, SFW>
            <<<tiles, THREADS, bytes, stream>>>(tmA, tmB, Cout, M, N, K, scale_a, scale_b, mx, s);
        return;
    }
    s.dp_tiles = tiles - tail;
    static Workspace w;
    if (w.tiles < static_cast<size_t>(tail)) {
        if (w.ws) SPARK_CUDA_CHECK(cudaFree(w.ws));
        if (w.counters) SPARK_CUDA_CHECK(cudaFree(w.counters));
        w.tiles = tail;
        SPARK_CUDA_CHECK(cudaMalloc(&w.ws, w.tiles * BM * BN * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMalloc(&w.counters, w.tiles * sizeof(int)));
    }
    s.ws = w.ws;
    s.counters = w.counters;
    SPARK_CUDA_CHECK(
        cudaMemsetAsync(w.ws, 0, static_cast<size_t>(tail) * BM * BN * sizeof(float), stream));
    SPARK_CUDA_CHECK(
        cudaMemsetAsync(w.counters, 0, static_cast<size_t>(tail) * sizeof(int), stream));
    const int grid = s.dp_tiles + tail * s.split;
    fp8gemm_v2_kernel<BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS, MX, SFW>
        <<<grid, THREADS, bytes, stream>>>(tmA, tmB, Cout, M, N, K, scale_a, scale_b, mx, s);
}

// MX mode fetches 16-byte scale chunks when every row stride is 16-byte aligned (K % 512
// == 0), else 8-byte ones; per-tensor mode has one instantiation.
template <int BK, int STAGES, int WARPS_M, int WARPS_N, int MIN_BLOCKS, bool MX>
void launch(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M, int N,
            int K, const float* scale_a, const float* scale_b, MxScales mx, cudaStream_t stream) {
    if (MX && K % 512 == 0)
        launch_sfw<BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS, MX, 4>(A, Bt, Cout, M, N, K, scale_a,
                                                                    scale_b, mx, stream);
    else
        launch_sfw<BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS, MX, 2>(A, Bt, Cout, M, N, K, scale_a,
                                                                    scale_b, mx, stream);
}

// ---- the cluster kernel (per-tensor mode) ------------------------------------------------
//
// The same producer / consumer pipeline, with three changes aimed at the joules per FLOP of
// the large shapes, where the kernel runs at the 600 W cap:
//   * the consumer loop of fp4gemm variant 2: unrolled by the ring depth (every stage address
//     and barrier an immediate, the parity one register flipped per round) and no
//     fence.proxy.async before the release (a read followed by an async-proxy write is ordered
//     by the arrive's release and the producer's acquire);
//   * a BN of 128 or 256 (128x256 with 2x4 warps of 64x64: 8 ldmatrix per 32 mma instead of
//     6 per 16, and a quarter less L2 traffic per FLOP);
//   * a CLM x CLN cluster of blocks over a (CLM * 128) x (CLN * BN) cluster tile. Block (ci, cj)
//     (cluster rank ci + CLM * cj) computes the 128 x BN tile (ci, cj). The CLN blocks of a
//     cluster row need the same A rows: each copies 128 / CLN of them and multicasts the box to
//     the whole row; likewise each of the CLM blocks of a cluster column copies BN / CLM rows of
//     Bt for the column. Every stage still lands whole in every block, so the full barrier
//     expects the same bytes, but the L2 -> SM traffic for A drops CLN-fold and for Bt
//     CLM-fold. A block's stage is written by the producers of its row and its column, so
//     each consumer warp releases a stage to all CLM + CLN - 1 of them (a remote arrive per
//     block), and a producer waits for that many warps' releases before it refills a stage.
// Host guarantees M % (128 * CLM) == 0, N % (BN * CLN) == 0, K % BK == 0.

template <int BN_, int BK, int STAGES>
constexpr int smem_bytes_cl() {
    return (STAGES * (BM + BN_) * BK + 2 * STAGES * 8 + 16 + 1023) / 1024 * 1024;
}

constexpr int kMaxSplitCl = 8;  // fp4gemm variant 2's cap on tail slices

template <int BN_, int BK, int STAGES, int WARPS_M, int WARPS_N, int CLM, int CLN>
__global__ void __launch_bounds__((WARPS_M * WARPS_N + 1) * 32, 1)
    fp8gemm_cl_kernel(const __grid_constant__ CUtensorMap tmA,
                      const __grid_constant__ CUtensorMap tmB, __nv_bfloat16* __restrict__ Cout,
                      int M, int N, int K, const float* __restrict__ scale_a,
                      const float* __restrict__ scale_b, Sched sched) {
    using C = fp8gemm_tile::Cfg<BM, BN_, BK, STAGES, WARPS_M, WARPS_N>;
    constexpr int BN = BN_;
    static_assert(BK == 64 || BK == 128, "the TMA swizzles cover 64 B and 128 B rows");
    constexpr int CL = CLM * CLN;
    static_assert(CL >= 1 && CL <= 8, "portable cluster sizes");
    constexpr int CONSUMER_WARPS = WARPS_M * WARPS_N;
    constexpr int CONSUMERS = CONSUMER_WARPS * 32;
    constexpr int A_STAGE = C::A_STAGE, STAGE_BYTES = C::STAGE;
    constexpr int WROWS = C::WROWS, WCOLS = C::WCOLS, MT = C::MT, NT = C::NT;
    constexpr int A_ROWS = BM / CLN;  // rows of A this block copies for its cluster row
    constexpr int B_ROWS = BN / CLM;  // rows of Bt this block copies for its cluster column
    // Each slice starts on a swizzle span (the swizzle is keyed on the absolute address).
    static_assert(A_ROWS * BK % 1024 == 0 && B_ROWS * BK % 1024 == 0, "slices of whole spans");
    constexpr int PEERS = CLM + CLN - 1;  // blocks that write into this block's stages
    static_assert(PEERS <= 32, "one lane per peer");

    extern __shared__ __align__(1024) unsigned char smem[];
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(smem + STAGES * STAGE_BYTES);
    uint64_t* empty_bar = full_bar + STAGES;
    int* s_last = reinterpret_cast<int*>(empty_bar + STAGES);
    if (smem_u32(smem) % 1024 != 0) __trap();

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    unsigned rank = 0;
    if constexpr (CL > 1) rank = cluster_ctarank();
    const int ci = static_cast<int>(rank) % CLM;
    const int cj = static_cast<int>(rank) / CLM;

    // Variant 2's schedule on cluster tiles: every block of a cluster takes the same K slice.
    const int cid = static_cast<int>(blockIdx.x) / CL;
    const int KT = K / BK;
    int ctile, kt_begin, kt_end;
    if (cid < sched.dp_tiles) {
        ctile = cid;
        kt_begin = 0;
        kt_end = KT;
    } else {
        const int r = cid - sched.dp_tiles;
        ctile = sched.dp_tiles + r / sched.split;
        const int slice = r % sched.split;
        kt_begin = static_cast<int>(static_cast<long long>(slice) * KT / sched.split);
        kt_end = static_cast<int>(static_cast<long long>(slice + 1) * KT / sched.split);
    }
    const int bm = (ctile / sched.tiles_n) * (BM * CLM) + ci * BM;
    const int bn = (ctile % sched.tiles_n) * (BN * CLN) + cj * BN;
    const int nkt = kt_end - kt_begin;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full_bar[s], 1);
            mbar_init(&empty_bar[s], CONSUMER_WARPS * PEERS);
        }
        fence_mbar_init();
    }
    // The peers' barriers must be initialized before any multicast or remote arrive.
    if constexpr (CL > 1)
        cluster_sync();
    else
        __syncthreads();

    if (warp == CONSUMER_WARPS) {
        // Producer warp, lane 0. Stage s serves k-tiles s, s+STAGES, ...; its n-th use waits
        // for the (n-1)-th release by every consumer warp of every block it writes into.
        if (lane == 0) {
            prefetch_tensormap(&tmA);
            prefetch_tensormap(&tmB);
            unsigned short mask_a = 0, mask_b = 0;  // the cluster row, the cluster column
#pragma unroll
            for (int j = 0; j < CLN; ++j)
                mask_a |= static_cast<unsigned short>(1u << (ci + CLM * j));
#pragma unroll
            for (int i = 0; i < CLM; ++i)
                mask_b |= static_cast<unsigned short>(1u << (i + CLM * cj));
            const int a_row = bm + cj * A_ROWS;  // this block's slice of the shared A tile
            const int b_row = bn + ci * B_ROWS;
            for (int kt = 0; kt < nkt; ++kt) {
                const int s = kt % STAGES;
                const unsigned use = kt / STAGES;
                if (kt >= STAGES) {
                    if constexpr (CL > 1)
                        mbar_wait_cluster(&empty_bar[s], (use - 1) & 1);
                    else
                        mbar_wait(&empty_bar[s], (use - 1) & 1);
                }
                mbar_arrive_expect_tx(&full_bar[s], STAGE_BYTES);
                const int k0 = (kt_begin + kt) * BK;
                unsigned char* as = smem + s * STAGE_BYTES;
                unsigned char* a_dst = as + cj * A_ROWS * BK;
                unsigned char* b_dst = as + A_STAGE + ci * B_ROWS * BK;
                if (CL > 1 && sched.nomc) {
                    const unsigned short self = static_cast<unsigned short>(1u << rank);
#pragma unroll
                    for (int j = 0; j < CLN; ++j) {
                        if (sched.nomc == 2)
                            tma_load_2d_multicast(as + j * A_ROWS * BK, &tmA, &full_bar[s], k0,
                                                  bm + j * A_ROWS, self);
                        else
                            tma_load_2d(as + j * A_ROWS * BK, &tmA, &full_bar[s], k0,
                                        bm + j * A_ROWS);
                    }
#pragma unroll
                    for (int i = 0; i < CLM; ++i) {
                        if (sched.nomc == 2)
                            tma_load_2d_multicast(as + A_STAGE + i * B_ROWS * BK, &tmB,
                                                  &full_bar[s], k0, bn + i * B_ROWS, self);
                        else
                            tma_load_2d(as + A_STAGE + i * B_ROWS * BK, &tmB, &full_bar[s], k0,
                                        bn + i * B_ROWS);
                    }
                    continue;
                }
                if constexpr (CLN > 1)
                    tma_load_2d_multicast(a_dst, &tmA, &full_bar[s], k0, a_row, mask_a);
                else
                    tma_load_2d(a_dst, &tmA, &full_bar[s], k0, a_row);
                if constexpr (CLM > 1)
                    tma_load_2d_multicast(b_dst, &tmB, &full_bar[s], k0, b_row, mask_b);
                else
                    tma_load_2d(b_dst, &tmB, &full_bar[s], k0, b_row);
            }
        }
        if constexpr (CL > 1) {
            __syncwarp();
            cluster_sync();
        }
        return;
    }

    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;
    const float scale = *scale_a * *scale_b;

    // The block this lane releases a stage to (lanes 0 .. PEERS-1): the CLN blocks of the
    // cluster row, then the other CLM - 1 blocks of the cluster column.
    unsigned peer = rank;
    if constexpr (CL > 1) {
        if (lane < CLN) {
            peer = ci + CLM * lane;
        } else if (lane < PEERS) {
            const int i = lane - CLN;
            peer = (i < ci ? i : i + 1) + CLM * cj;
        }
    }

    typename C::Acc acc;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    const fp8gemm_tile::MxFrag<C> none{};
    auto consume = [&](const unsigned char* as, uint64_t* full, uint64_t* empty, unsigned parity) {
        if constexpr (CL > 1)
            mbar_wait_poll(full, parity);
        else
            mbar_wait(full, parity);
        fp8gemm_tile::mma_stage<C, false>(as, as + A_STAGE, wm, wn, lane, acc, none);
        __syncwarp();
        if constexpr (CL > 1) {
            if (lane < PEERS) mbar_arrive_cluster(empty, peer);
        } else {
            if (lane == 0) mbar_arrive(empty);
        }
    };
    int kt = 0;
    unsigned parity = 0;
    for (; kt + STAGES <= nkt; kt += STAGES, parity ^= 1u) {
        fp8gemm_tile::static_for<STAGES>([&](auto st) {
            constexpr int S = decltype(st)::value;
            consume(smem + S * STAGE_BYTES, &full_bar[S], &empty_bar[S], parity);
        });
    }
    for (int s = 0; kt < nkt; ++kt, ++s)
        consume(smem + s * STAGE_BYTES, &full_bar[s], &empty_bar[s], parity);

    // Nobody in the cluster copies into this block or arrives on its barriers past here.
    if constexpr (CL > 1) cluster_sync();

    if (ctile < sched.dp_tiles) {
        fp8gemm_tile::store_bf16<C>(acc, scale, Cout, M, N, bm, bn);
        return;
    }

    // Tail tile: the workspace holds one BM x BN tile per block of each tail cluster tile;
    // the last slice to arrive converts it, zeroes it again and resets its counter, so the
    // launcher never clears it (fp4gemm variant 2).
    const int g = lane >> 2;
    const int c2 = (lane & 3) * 2;
    const int wt_idx = (ctile - sched.dp_tiles) * CL + static_cast<int>(rank);
    float* wt = sched.ws + static_cast<size_t>(wt_idx) * BM * BN;
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int nj = 0; nj < NT; ++nj) {
            const int r0 = wm * WROWS + mi * 16 + g;
            const int c0 = wn * WCOLS + nj * 8 + c2;
            atomicAdd(wt + r0 * BN + c0, acc[mi][nj][0]);
            atomicAdd(wt + r0 * BN + c0 + 1, acc[mi][nj][1]);
            atomicAdd(wt + (r0 + 8) * BN + c0, acc[mi][nj][2]);
            atomicAdd(wt + (r0 + 8) * BN + c0 + 1, acc[mi][nj][3]);
        }
    }
    __threadfence();
    named_barrier_sync(TAIL_BARRIER, CONSUMERS);
    if (tid == 0) *s_last = atomicAdd(sched.counters + wt_idx, 1) == sched.split - 1;
    named_barrier_sync(TAIL_BARRIER, CONSUMERS);
    if (!*s_last) return;
    __threadfence();
    for (int i = tid; i < BM * BN / 2; i += CONSUMERS) {
        const int r = (2 * i) / BN;
        const int c = (2 * i) % BN;
        float2* p = reinterpret_cast<float2*>(wt + r * BN + c);
        const float2 v = __ldcg(p);
        __stcg(p, make_float2(0.f, 0.f));
        *reinterpret_cast<__nv_bfloat162*>(Cout + static_cast<size_t>(bm + r) * N + bn + c) =
            __floats2bfloat162_rn(v.x * scale, v.y * scale);
    }
    if (tid == 0) sched.counters[wt_idx] = 0;
}

// Zeroed once when it grows; the kernel leaves it zeroed.
struct WorkspaceCl {
    float* ws = nullptr;
    int* counters = nullptr;
    size_t tiles = 0;
    void reserve(size_t n, size_t tile_elems, cudaStream_t stream) {
        if (tiles >= n) return;
        if (ws) SPARK_CUDA_CHECK(cudaFree(ws));
        if (counters) SPARK_CUDA_CHECK(cudaFree(counters));
        tiles = n;
        SPARK_CUDA_CHECK(cudaMalloc(&ws, tiles * tile_elems * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMalloc(&counters, tiles * sizeof(int)));
        SPARK_CUDA_CHECK(cudaMemsetAsync(ws, 0, tiles * tile_elems * sizeof(float), stream));
        SPARK_CUDA_CHECK(cudaMemsetAsync(counters, 0, tiles * sizeof(int), stream));
    }
};

// Clusters resident at once (cudaOccupancyMaxActiveClusters: a cluster lives on one GPC, so
// this can be less than SMs / CL), or 0 if the device cannot launch this cluster shape.
template <int BN_, int BK, int STAGES, int WARPS_M, int WARPS_N, int CLM, int CLN>
int resident_clusters() {
    constexpr int bytes = smem_bytes_cl<BN_, BK, STAGES>();
    constexpr int threads = (WARPS_M * WARPS_N + 1) * 32;
    constexpr int CL = CLM * CLN;
    static int resident = -1;
    if (resident < 0) {
        auto kernel = fp8gemm_cl_kernel<BN_, BK, STAGES, WARPS_M, WARPS_N, CLM, CLN>;
        SPARK_CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        resident = 0;
        int dev = 0, cluster_ok = 0;
        SPARK_CUDA_CHECK(cudaGetDevice(&dev));
        SPARK_CUDA_CHECK(cudaDeviceGetAttribute(&cluster_ok, cudaDevAttrClusterLaunch, dev));
        if (CL == 1) {
            int per_sm = 0;
            SPARK_CUDA_CHECK(
                cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, bytes));
            resident = per_sm * num_sms();
        } else if (cluster_ok) {
            cudaLaunchConfig_t cfg = {};
            cudaLaunchAttribute attr[1];
            attr[0].id = cudaLaunchAttributeClusterDimension;
            attr[0].val.clusterDim.x = CL;
            attr[0].val.clusterDim.y = 1;
            attr[0].val.clusterDim.z = 1;
            cfg.gridDim = dim3(CL * num_sms());
            cfg.blockDim = dim3(threads);
            cfg.dynamicSmemBytes = bytes;
            cfg.attrs = attr;
            cfg.numAttrs = 1;
            int n = 0;
            if (cudaOccupancyMaxActiveClusters(&n, kernel, &cfg) == cudaSuccess) resident = n;
            (void)cudaGetLastError();  // clear what an unsupported cluster shape reports
        }
        if (std::getenv("SPARK_FP8GEMM_VERBOSE"))
            std::fprintf(stderr, "fp8gemm cluster config 128x%d BK %d x%d, %dx%d: %d resident\n",
                         BN_, BK, STAGES, CLM, CLN, resident);
    }
    return resident;
}

// Returns false (nothing launched) if the shape does not tile or the cluster cannot launch.
template <int BN_, int BK, int STAGES, int WARPS_M, int WARPS_N, int CLM, int CLN>
bool launch_cl(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M, int N,
               int K, const float* scale_a, const float* scale_b, cudaStream_t stream) {
    constexpr int BN = BN_, CL = CLM * CLN;
    constexpr int bytes = smem_bytes_cl<BN_, BK, STAGES>();
    constexpr int THREADS = (WARPS_M * WARPS_N + 1) * 32;
    if (M % (BM * CLM) != 0 || N % (BN * CLN) != 0 || K % BK != 0) return false;
    const int resident = resident_clusters<BN_, BK, STAGES, WARPS_M, WARPS_N, CLM, CLN>();
    if (resident <= 0) return false;

    constexpr CUtensorMapSwizzle swz =
        BK == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
    const CUtensorMap tmA = make_tensor_map_2d_u8(A, M, K, BM / CLN, BK, swz);
    const CUtensorMap tmB = make_tensor_map_2d_u8(Bt, N, K, BN / CLM, BK, swz);

    Sched s;
    s.tiles_n = N / (BN * CLN);
    s.nomc = std::getenv("SPARK_FP8GEMM_NOMC") ? std::atoi(std::getenv("SPARK_FP8GEMM_NOMC")) : 0;
    const int ctiles = (M / (BM * CLM)) * s.tiles_n;
    const int KT = K / BK;
    const int tail = ctiles % resident;
    s.split = tail > 0 ? std::min({KT, resident / tail, kMaxSplitCl}) : 1;
    s.ws = nullptr;
    s.counters = nullptr;
    if (s.split <= 1) {
        s.split = 1;
        s.dp_tiles = ctiles;
    } else {
        s.dp_tiles = ctiles - tail;
        static WorkspaceCl w;
        w.reserve(static_cast<size_t>(tail) * CL, static_cast<size_t>(BM) * BN, stream);
        s.ws = w.ws;
        s.counters = w.counters;
    }
    const int grid = (s.dp_tiles + (ctiles - s.dp_tiles) * s.split) * CL;

    cudaLaunchConfig_t cfg = {};
    cudaLaunchAttribute attr[1];
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CL;
    attr[0].val.clusterDim.y = 1;
    attr[0].val.clusterDim.z = 1;
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(THREADS);
    cfg.dynamicSmemBytes = bytes;
    cfg.stream = stream;
    cfg.attrs = attr;
    cfg.numAttrs = 1;
    SPARK_CUDA_CHECK(
        cudaLaunchKernelEx(&cfg, fp8gemm_cl_kernel<BN_, BK, STAGES, WARPS_M, WARPS_N, CLM, CLN>,
                           tmA, tmB, Cout, M, N, K, scale_a, scale_b, s));
    return true;
}

// A cluster configuration as a CONFIGS entry: MX mode, and any shape or device the cluster
// kernel does not take, runs the default configuration of variant 2 instead.
template <int BN_, int BK, int STAGES, int WARPS_M, int WARPS_N, int CLM, int CLN, bool MX>
void launch_cl_or_v2(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M,
                     int N, int K, const float* scale_a, const float* scale_b, MxScales mx,
                     cudaStream_t stream) {
    if constexpr (!MX) {
        if (launch_cl<BN_, BK, STAGES, WARPS_M, WARPS_N, CLM, CLN>(A, Bt, Cout, M, N, K, scale_a,
                                                                   scale_b, stream))
            return;
        launch<128, 3, 2, 4, 1, false>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
    } else {
        launch<64, 4, 2, 4, 1, true>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
    }
}

// Pipeline configurations (BK in bytes, stages, consumer warp grid, blocks per SM the compiler
// is asked to fit). SPARK_FP8GEMM_V2_CONFIG=<index> forces one; the sweep is in
// docs/design/fp8gemm.md.
using LaunchFn = void (*)(const unsigned char*, const unsigned char*, __nv_bfloat16*, int, int, int,
                          const float*, const float*, MxScales, cudaStream_t);
template <bool MX>
constexpr LaunchFn CONFIGS[] = {
    launch<128, 2, 2, 4, 1, MX>,  // 0: 64 KB of smem, 2x4 warps, one block per SM
    launch<128, 3, 2, 4, 1, MX>,  // 1: 96 KB, 2x4 warps, one block per SM
    launch<64, 3, 2, 4, 2, MX>,   // 2: hgemm v3's tile and depth, 48 KB, two blocks per SM
    launch<64, 2, 2, 4, 2, MX>,   // 3: 32 KB, two blocks per SM
    launch<64, 4, 2, 4, 1, MX>,   // 4: 64 KB, one block per SM
    launch<128, 3, 2, 2, 1, MX>,  // 5: 96 KB, 2x2 warps of 64x64, one block per SM
    launch<128, 2, 2, 2, 1, MX>,  // 6: 64 KB, 2x2 warps of 64x64
    launch<64, 3, 2, 2, 2, MX>,   // 7: 48 KB, 2x2 warps of 64x64, two blocks per SM
    launch<64, 2, 2, 2, 2, MX>,   // 8: 32 KB, 2x2 warps of 64x64, two blocks per SM
    launch<64, 5, 2, 4, 1, MX>,   // 9: 80 KB, five stages of 64 B, one block per SM
    launch<64, 6, 2, 4, 1, MX>,   // 10: 96 KB, six stages of 64 B, one block per SM
    // The cluster kernel (per-tensor mode only; MX runs config 4). Tile, BK, stages, warps,
    // cluster CLM x CLN:
    launch_cl_or_v2<128, 128, 3, 2, 4, 1, 1, MX>,  // 11: config 1 with the unrolled loop
    launch_cl_or_v2<128, 128, 3, 2, 4, 2, 1, MX>,  // 12: 2x1 cluster, Bt multicast
    launch_cl_or_v2<128, 128, 3, 2, 4, 1, 2, MX>,  // 13: 1x2 cluster, A multicast
    launch_cl_or_v2<128, 128, 3, 2, 4, 2, 2, MX>,  // 14: 2x2 cluster, both
    launch_cl_or_v2<128, 128, 3, 2, 4, 4, 1, MX>,  // 15: 4x1 cluster, Bt multicast
    launch_cl_or_v2<256, 128, 2, 2, 4, 1, 1, MX>,  // 16: 128x256, 96 KB, 2x4 warps of 64x64
    launch_cl_or_v2<256, 64, 4, 2, 4, 1, 1, MX>,   // 17: 128x256, four 64-byte stages
    launch_cl_or_v2<256, 128, 2, 2, 4, 2, 1, MX>,  // 18: 128x256, 2x1 cluster
    launch_cl_or_v2<256, 64, 4, 2, 4, 2, 1, MX>,   // 19: 128x256, 64-byte stages, 2x1 cluster
};
constexpr int NUM_CONFIGS = static_cast<int>(sizeof(CONFIGS<false>) / sizeof(CONFIGS<false>[0]));
// Grids of more than two blocks per SM: the 128x256 tile with 64x64 warp tiles (config 16),
// 754 against config 1's 675 TFLOPS at 8192^3, at the clock cuBLASLt holds under 600 W
// (docs/design/fp8gemm.md). It runs config 1 when N % 256 != 0.
constexpr int CONFIG_WAVES = 16;
constexpr int CONFIG_SMALL = 3;  // up to two blocks per SM: two resident blocks, no tail
// MX mode: the consumer warps carry the scale fetches and shuffles, and a deeper pipeline of
// 64-byte stages hides what that costs them better than three 128-byte stages (the sweep in
// docs/design/fp8gemm.md); the two-block configuration loses on every MX shape.
constexpr int CONFIG_MX = 4;

int forced_config() {
    static int idx = -2;
    if (idx == -2) {
        idx = -1;
        if (const char* e = std::getenv("SPARK_FP8GEMM_V2_CONFIG")) {
            const int v = std::atoi(e);
            if (v >= 0 && v < NUM_CONFIGS) idx = v;
        }
    }
    return idx;
}

void launch_auto(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M, int N,
                 int K, const float* scale_a, const float* scale_b, MxScales mx,
                 cudaStream_t stream) {
    int cfg = forced_config();
    if (cfg < 0) {
        const int tiles = (M / BM) * (N / BN);
        cfg = mx.a != nullptr ? CONFIG_MX : tiles <= 2 * num_sms() ? CONFIG_SMALL : CONFIG_WAVES;
    }
    if (mx.a != nullptr)
        CONFIGS<true>[cfg](A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
    else
        CONFIGS<false>[cfg](A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
}

}  // namespace v2

}  // namespace

bool fp8gemm_tma_supports(int M, int N, int K) {
    return M % v2::BM == 0 && N % v2::BN == 0 && K % 128 == 0 &&
           (M / v2::BM) * (N / v2::BN) >= num_sms();
}

void fp8gemm_tma(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* Bt, __nv_bfloat16* C, int M, int N,
                 int K, const float* scale_a, const float* scale_b, fp8gemm_tile::MxScales mx,
                 cudaStream_t stream) {
    SPARK_REQUIRE(fp8gemm_tma_supports(M, N, K),
                  "fp8gemm variant 2: requires M % 128 == 0, N % 128 == 0, K % 128 == 0 and at "
                  "least one 128x128 tile per SM");
    v2::launch_auto(reinterpret_cast<const unsigned char*>(A),
                    reinterpret_cast<const unsigned char*>(Bt), C, M, N, K, scale_a, scale_b, mx,
                    stream);
}

}  // namespace spark
