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

#include <cuda_fp8.h>

#include <algorithm>
#include <cstdlib>

#include "fp8gemm_tile.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

namespace v2 {

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
};

template <int BK, int STAGES, int WARPS_M, int WARPS_N, int MIN_BLOCKS>
__global__ void __launch_bounds__((WARPS_M * WARPS_N + 1) * 32, MIN_BLOCKS)
    fp8gemm_v2_kernel(const __grid_constant__ CUtensorMap tmA,
                      const __grid_constant__ CUtensorMap tmB, __nv_bfloat16* __restrict__ Cout,
                      int M, int N, int K, const float* __restrict__ scale_a,
                      const float* __restrict__ scale_b, Sched sched) {
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

    for (int kt = 0; kt < nkt; ++kt) {
        const int s = kt % STAGES;
        mbar_wait(&full_bar[s], (kt / STAGES) & 1);
        const unsigned char* as = smem + s * STAGE_BYTES;
        const unsigned char* bs = as + A_STAGE;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 32)
            fp8gemm_tile::mma_step<C>(as, bs, kk, wm, wn, lane, acc);
        // The ldmatrix reads went through the generic proxy and the refill comes through the
        // async proxy: fence, then one arrive per warp hands the stage back.
        fence_proxy_async_smem();
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty_bar[s]);
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

template <int BK, int STAGES, int WARPS_M, int WARPS_N, int MIN_BLOCKS>
int resident_blocks() {
    constexpr int bytes = smem_bytes<BK, STAGES>();
    constexpr int threads = (WARPS_M * WARPS_N + 1) * 32;
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(
            cudaFuncSetAttribute(fp8gemm_v2_kernel<BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS>,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, fp8gemm_v2_kernel<BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS>, threads, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <int BK, int STAGES, int WARPS_M, int WARPS_N, int MIN_BLOCKS>
void launch(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M, int N,
            int K, const float* scale_a, const float* scale_b, cudaStream_t stream) {
    constexpr int bytes = smem_bytes<BK, STAGES>();
    constexpr int THREADS = (WARPS_M * WARPS_N + 1) * 32;
    const int resident = resident_blocks<BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS>();

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
    s.split = tail > 0 ? std::min(KT, resident / tail) : 1;
    if (s.split <= 1) {
        s.dp_tiles = tiles;
        s.split = 1;
        s.ws = nullptr;
        s.counters = nullptr;
        fp8gemm_v2_kernel<BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS>
            <<<tiles, THREADS, bytes, stream>>>(tmA, tmB, Cout, M, N, K, scale_a, scale_b, s);
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
    fp8gemm_v2_kernel<BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS>
        <<<grid, THREADS, bytes, stream>>>(tmA, tmB, Cout, M, N, K, scale_a, scale_b, s);
}

// Pipeline configurations (BK in bytes, stages, consumer warp grid, blocks per SM the compiler
// is asked to fit). SPARK_FP8GEMM_V2_CONFIG=<index> forces one; the sweep is in
// docs/design/fp8gemm.md.
using LaunchFn = void (*)(const unsigned char*, const unsigned char*, __nv_bfloat16*, int, int, int,
                          const float*, const float*, cudaStream_t);
constexpr LaunchFn CONFIGS[] = {
    launch<128, 2, 2, 4, 1>,  // 0: 64 KB of smem, 2x4 warps, one block per SM
    launch<128, 3, 2, 4, 1>,  // 1: 96 KB, 2x4 warps, one block per SM
    launch<64, 3, 2, 4, 2>,   // 2: hgemm v3's tile and depth, 48 KB, two blocks per SM
    launch<64, 2, 2, 4, 2>,   // 3: 32 KB, two blocks per SM
    launch<64, 4, 2, 4, 1>,   // 4: 64 KB, one block per SM
    launch<128, 3, 2, 2, 1>,  // 5: 96 KB, 2x2 warps of 64x64, one block per SM
    launch<128, 2, 2, 2, 1>,  // 6: 64 KB, 2x2 warps of 64x64
    launch<64, 3, 2, 2, 2>,   // 7: 48 KB, 2x2 warps of 64x64, two blocks per SM
    launch<64, 2, 2, 2, 2>,   // 8: 32 KB, 2x2 warps of 64x64, two blocks per SM
};
constexpr int NUM_CONFIGS = static_cast<int>(sizeof(CONFIGS) / sizeof(CONFIGS[0]));
constexpr int CONFIG_WAVES = 1;  // grids of more than two blocks per SM
constexpr int CONFIG_SMALL = 3;  // up to two blocks per SM: two resident blocks, no tail

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
                 int K, const float* scale_a, const float* scale_b, cudaStream_t stream) {
    int cfg = forced_config();
    if (cfg < 0) {
        const int tiles = (M / BM) * (N / BN);
        cfg = tiles <= 2 * num_sms() ? CONFIG_SMALL : CONFIG_WAVES;
    }
    CONFIGS[cfg](A, Bt, Cout, M, N, K, scale_a, scale_b, stream);
}

}  // namespace v2

}  // namespace

bool fp8gemm_tma_supports(int M, int N, int K) {
    return M % v2::BM == 0 && N % v2::BN == 0 && K % 128 == 0 &&
           (M / v2::BM) * (N / v2::BN) >= num_sms();
}

void fp8gemm_tma(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* Bt, __nv_bfloat16* C, int M, int N,
                 int K, const float* scale_a, const float* scale_b, cudaStream_t stream) {
    SPARK_REQUIRE(fp8gemm_tma_supports(M, N, K),
                  "fp8gemm variant 2: requires M % 128 == 0, N % 128 == 0, K % 128 == 0 and at "
                  "least one 128x128 tile per SM");
    v2::launch_auto(reinterpret_cast<const unsigned char*>(A),
                    reinterpret_cast<const unsigned char*>(Bt), C, M, N, K, scale_a, scale_b,
                    stream);
}

}  // namespace spark
