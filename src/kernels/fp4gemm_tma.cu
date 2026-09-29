// fp4 (e2m1) GEMM, variant 2: TMA loads and a warp-specialized mbarrier pipeline.
//
//   C[M,N] = scale_a * scale_b * sum_k (sfa A)[m][k] (sfb Bt)[n][k]     NVFP4 or MXFP4
//
// fp8gemm variant 2 (src/kernels/fp8gemm_tma.cu) on fp4 operands. One producer lane fills
// each stage with copies on one "full" mbarrier: two tensor-map boxes of BM and BN rows by BK
// bytes (A and Bt, swizzled by the copy engine exactly as fp8gemm_tile::swz expects), and one
// 1-D bulk copy per 128-row block of each operand for the stage's scale tiles, which in the
// blocked layout are one contiguous run of TILES x 512 bytes (fp4gemm_tile.cuh). The consumer
// warps run fp4gemm_tile::mma_stage on the stage, in a k-loop unrolled by the ring depth, and
// release it on its "empty" mbarrier. Register epilogue with the per-tensor scales and the
// tail split-K of variant 1. The shipped tile is 128x128 (the sweep in
// docs/design/fp4gemm.md); the host guarantees M % BM == 0, N % BN == 0, K % 256 == 0 and a
// grid of at least one wave of 128x128 tiles.

#include <algorithm>
#include <cstdlib>

#include "fp4gemm_tile.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

namespace v2 {

using fp4gemm_tile::MXFP4;
using fp4gemm_tile::NVFP4;
using fp4gemm_tile::Scales;

constexpr int BM = 128, BN = 128;  // the smallest tile variant 2 runs, for the shape rules
constexpr int TAIL_BARRIER = 1;
constexpr int kMaxSplit = 8;

template <int BM_, int BN_, int BK, int STAGES, int WARPS_M, int WARPS_N, int FMT>
using TileCfg = fp4gemm_tile::Cfg<BM_, BN_, BK, STAGES, WARPS_M, WARPS_N, FMT>;

// Stages at the front of the dynamic region (1 KB aligned: the swizzles are keyed on the
// absolute address, and every stage is a whole number of KB), barriers after them.
template <class C>
constexpr int smem_bytes() {
    return (C::SMEM + 2 * C::STAGES * 8 + 16 + 1023) / 1024 * 1024;
}

struct Sched {
    int tiles_n;
    int dp_tiles;
    int split;
    float* ws;
    int* counters;
};

template <int BM_, int BN_, int BK, int STAGES, int WARPS_M, int WARPS_N, int MIN_BLOCKS, int FMT>
__global__ void __launch_bounds__((WARPS_M * WARPS_N + 1) * 32, MIN_BLOCKS)
    fp4gemm_v2_kernel(const __grid_constant__ CUtensorMap tmA,
                      const __grid_constant__ CUtensorMap tmB, __nv_bfloat16* __restrict__ Cout,
                      int M, int N, int K, const float* __restrict__ scale_a,
                      const float* __restrict__ scale_b, Scales sf, Sched sched) {
    using C = TileCfg<BM_, BN_, BK, STAGES, WARPS_M, WARPS_N, FMT>;
    constexpr int BM = BM_, BN = BN_;
    static_assert(BK == 64 || BK == 128, "the TMA swizzles cover 64 B and 128 B rows");
    static_assert(C::STAGE % 1024 == 0, "every stage starts on a swizzle-span boundary");
    constexpr int CONSUMER_WARPS = WARPS_M * WARPS_N;
    constexpr int CONSUMERS = CONSUMER_WARPS * 32;
    constexpr int A_STAGE = C::A_STAGE, STAGE_BYTES = C::STAGE;
    constexpr int WROWS = C::WROWS, WCOLS = C::WCOLS, MT = C::MT, NT = C::NT;
    constexpr int TILES = C::TILES;
    constexpr int SF_BYTES = TILES * fp4gemm_tile::kSfTile;  // per operand per stage

    extern __shared__ __align__(1024) unsigned char smem[];
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(smem + STAGES * STAGE_BYTES);
    uint64_t* empty_bar = full_bar + STAGES;
    int* s_last = reinterpret_cast<int*>(empty_bar + STAGES);
    if (smem_u32(smem) % 1024 != 0) __trap();

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int KT = K / 2 / BK;
    int tile, kt_begin, kt_end;
    if (static_cast<int>(blockIdx.x) < sched.dp_tiles) {
        tile = blockIdx.x;
        kt_begin = 0;
        kt_end = KT;
    } else {
        const int r = blockIdx.x - sched.dp_tiles;
        tile = sched.dp_tiles + r / sched.split;
        const int slice = r % sched.split;
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
        // Producer warp, lane 0: stage s serves k-tiles s, s + STAGES, ...
        if (lane == 0) {
            prefetch_tensormap(&tmA);
            prefetch_tensormap(&tmB);
            const unsigned char* sa =
                fp4gemm_tile::sf_src<BM_>(sf.a, sf.ktiles, bm, kt_begin * TILES);
            const unsigned char* sb =
                fp4gemm_tile::sf_src<BN_>(sf.b, sf.ktiles, bn, kt_begin * TILES);
            const size_t rb_step = static_cast<size_t>(sf.ktiles) * fp4gemm_tile::kSfTile;
            for (int kt = 0; kt < nkt; ++kt) {
                const int s = kt % STAGES;
                const unsigned use = kt / STAGES;
                if (kt >= STAGES) mbar_wait(&empty_bar[s], (use - 1) & 1);
                mbar_arrive_expect_tx(&full_bar[s], STAGE_BYTES);
                const int k0 = (kt_begin + kt) * BK;
                unsigned char* as = smem + s * STAGE_BYTES;
                tma_load_2d(as, &tmA, &full_bar[s], k0, bm);
                tma_load_2d(as + A_STAGE, &tmB, &full_bar[s], k0, bn);
                // The scale tiles: one contiguous run of TILES x 512 bytes per 128-row block.
#pragma unroll
                for (int rb = 0; rb < C::SA_RB; ++rb)
                    bulk_load(as + C::SA_OFF + rb * C::SA_RB_STRIDE,
                              sa + rb * rb_step + static_cast<size_t>(kt) * SF_BYTES, SF_BYTES,
                              &full_bar[s]);
#pragma unroll
                for (int rb = 0; rb < C::SB_RB; ++rb)
                    bulk_load(as + C::SB_OFF + rb * C::SB_RB_STRIDE,
                              sb + rb * rb_step + static_cast<size_t>(kt) * SF_BYTES, SF_BYTES,
                              &full_bar[s]);
            }
        }
        return;
    }

    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;
    const float scale = *scale_a * *scale_b;
    const int qa0 = (bm / 32) & 3;  // 0: bm is a multiple of 128
    const int qb0 = (bn / 32) & 3;

    typename C::Acc acc;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    // One stage: wait for the copies, run the mma, hand the stage back with one arrive per
    // warp. The arrive has release semantics and the producer waits on it before the next
    // copy into the stage, which orders this warp's ldmatrix reads before that write; no
    // proxy fence is needed for a read followed by an async-proxy write (the fence and the
    // MEMBAR it compiles to cost 6 instructions per stage and a stall).
    auto consume = [&](const unsigned char* as, uint64_t* full, uint64_t* empty, unsigned parity) {
        mbar_wait(full, parity);
        fp4gemm_tile::mma_stage<C>(as, as + A_STAGE, as + C::SA_OFF, as + C::SB_OFF, wm, wn, lane,
                                   qa0, qb0, acc);
        __syncwarp();
        if (lane == 0) mbar_arrive(empty);
    };
    // The k-loop unrolled by the ring depth, so that every stage address and barrier is an
    // immediate offset and the parity one register flipped per round; the last partial
    // round runs the same body with computed addresses.
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

    if (tile < sched.dp_tiles) {
        fp8gemm_tile::store_bf16<C>(acc, scale, Cout, M, N, bm, bn);
        return;
    }

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
        float2* p = reinterpret_cast<float2*>(wt + r * BN + c);
        const float2 v = __ldcg(p);
        __stcg(p, make_float2(0.f, 0.f));  // leave the workspace zeroed for the next launch
        *reinterpret_cast<__nv_bfloat162*>(Cout + static_cast<size_t>(bm + r) * N + bn + c) =
            __floats2bfloat162_rn(v.x * scale, v.y * scale);
    }
    if (tid == 0) sched.counters[tile - sched.dp_tiles] = 0;
}

// ---- host side --------------------------------------------------------------------------

template <int BM_, int BN_, int BK, int STAGES, int WARPS_M, int WARPS_N, int MIN_BLOCKS, int FMT>
int resident_blocks() {
    using C = TileCfg<BM_, BN_, BK, STAGES, WARPS_M, WARPS_N, FMT>;
    constexpr int bytes = smem_bytes<C>();
    constexpr int threads = (WARPS_M * WARPS_N + 1) * 32;
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(
            fp4gemm_v2_kernel<BM_, BN_, BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS, FMT>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, fp4gemm_v2_kernel<BM_, BN_, BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS, FMT>,
            threads, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <int BM_, int BN_, int BK, int STAGES, int WARPS_M, int WARPS_N, int MIN_BLOCKS, int FMT>
void launch(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M, int N,
            int K, const float* scale_a, const float* scale_b, Scales sf, cudaStream_t stream) {
    using C = TileCfg<BM_, BN_, BK, STAGES, WARPS_M, WARPS_N, FMT>;
    constexpr int BM = BM_, BN = BN_;
    constexpr int bytes = smem_bytes<C>();
    constexpr int THREADS = (WARPS_M * WARPS_N + 1) * 32;
    const int resident = resident_blocks<BM_, BN_, BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS, FMT>();

    // Both boxes: 128 rows by BK bytes of the packed operands, swizzled over the row.
    constexpr CUtensorMapSwizzle swz =
        BK == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
    const CUtensorMap tmA = make_tensor_map_2d_u8(A, M, K / 2, BM, BK, swz);
    const CUtensorMap tmB = make_tensor_map_2d_u8(Bt, N, K / 2, BN, BK, swz);

    Sched s;
    s.tiles_n = N / BN;
    const int tiles = (M / BM) * s.tiles_n;
    const int KT = K / 2 / BK;
    const int tail = tiles % resident;
    // At most kMaxSplit slices: a slice costs a block's prologue and 64 KB of atomics whatever
    // its length, and splitting the 4 tail tiles of 4096^3 32 ways ran 2 to 3% slower than 4
    // or 8 ways (the sweep in docs/design/fp4gemm.md).
    s.split = tail > 0 ? std::min({KT, resident / tail, kMaxSplit}) : 1;
    if (fp4gemm_tile::max_split_override() > 0)
        s.split = std::min(s.split, fp4gemm_tile::max_split_override());
    if (s.split <= 1) {
        s.dp_tiles = tiles;
        s.split = 1;
        s.ws = nullptr;
        s.counters = nullptr;
        fp4gemm_v2_kernel<BM_, BN_, BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS, FMT>
            <<<tiles, THREADS, bytes, stream>>>(tmA, tmB, Cout, M, N, K, scale_a, scale_b, sf, s);
        return;
    }
    s.dp_tiles = tiles - tail;
    static fp4gemm_tile::Workspace w;
    w.reserve(tail, BM * BN, stream);
    s.ws = w.ws;
    s.counters = w.counters;
    const int grid = s.dp_tiles + tail * s.split;
    fp4gemm_v2_kernel<BM_, BN_, BK, STAGES, WARPS_M, WARPS_N, MIN_BLOCKS, FMT>
        <<<grid, THREADS, bytes, stream>>>(tmA, tmB, Cout, M, N, K, scale_a, scale_b, sf, s);
}

// Pipeline configurations (BK in bytes of K per stage, stages, consumer warp grid, blocks per
// SM the compiler is asked to fit). SPARK_FP4GEMM_V2_CONFIG=<index> forces one.
using LaunchFn = void (*)(const unsigned char*, const unsigned char*, __nv_bfloat16*, int, int, int,
                          const float*, const float*, Scales, cudaStream_t);
struct Config {
    LaunchFn nv, mx;
    int bm, bn;
};
#define SPARK_FP4_V2_CONFIG(BM_, BN_, BK, ST, WM, WN, MB)                                      \
    {launch<BM_, BN_, BK, ST, WM, WN, MB, NVFP4>, launch<BM_, BN_, BK, ST, WM, WN, MB, MXFP4>, \
     BM_, BN_}
constexpr Config CONFIGS[] = {
    SPARK_FP4_V2_CONFIG(128, 128, 128, 2, 2, 4, 1),  // 0: 72 KB (NVFP4), 2x4 warps of 64x32
    SPARK_FP4_V2_CONFIG(128, 128, 64, 4, 2, 4, 1),   // 1: 72 KB, four 64-byte stages
    SPARK_FP4_V2_CONFIG(128, 128, 64, 5, 2, 4, 1),   // 2: 90 KB, five 64-byte stages
    SPARK_FP4_V2_CONFIG(128, 128, 64, 2, 2, 4, 2),   // 3: 36 KB, two blocks per SM
    SPARK_FP4_V2_CONFIG(128, 128, 64, 4, 2, 2, 1),   // 4: 72 KB, 2x2 warps of 64x64
    SPARK_FP4_V2_CONFIG(128, 128, 128, 2, 2, 2, 1),  // 5: 72 KB, 2x2 warps of 64x64
    SPARK_FP4_V2_CONFIG(128, 128, 64, 3, 2, 4, 1),   // 6: 54 KB, three 64-byte stages
    SPARK_FP4_V2_CONFIG(128, 256, 64, 3, 2, 4, 1),   // 7: 128x256, 81 KB, 2x4 warps of 64x64
    SPARK_FP4_V2_CONFIG(128, 256, 64, 2, 2, 4, 1),   // 8: 128x256, 54 KB
    SPARK_FP4_V2_CONFIG(256, 128, 64, 3, 4, 2, 1),   // 9: 256x128, 81 KB, 4x2 warps of 64x64
};
#undef SPARK_FP4_V2_CONFIG
constexpr int NUM_CONFIGS = static_cast<int>(sizeof(CONFIGS) / sizeof(CONFIGS[0]));
constexpr int CONFIG_WAVES = 1;
constexpr int CONFIG_SMALL = 3;

int forced_config() {
    static int idx = -2;
    if (idx == -2) {
        idx = -1;
        if (const char* e = std::getenv("SPARK_FP4GEMM_V2_CONFIG")) {
            const int v = std::atoi(e);
            if (v >= 0 && v < NUM_CONFIGS) idx = v;
        }
    }
    return idx;
}

}  // namespace v2

}  // namespace

bool fp4gemm_tma_supports(int M, int N, int K) {
    return M % v2::BM == 0 && N % v2::BN == 0 && K % 256 == 0 &&
           (M / v2::BM) * (N / v2::BN) >= num_sms();
}

void fp4gemm_tma(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* C, int M, int N,
                 int K, const float* scale_a, const float* scale_b, fp4gemm_tile::Scales sf,
                 int format, cudaStream_t stream) {
    int cfg = v2::forced_config();
    if (cfg >= 0 && (M % v2::CONFIGS[cfg].bm != 0 || N % v2::CONFIGS[cfg].bn != 0)) cfg = -1;
    if (cfg < 0) {
        const int tiles = (M / v2::BM) * (N / v2::BN);
        cfg = tiles <= 2 * num_sms() ? v2::CONFIG_SMALL : v2::CONFIG_WAVES;
    }
    const v2::Config& c = v2::CONFIGS[cfg];
    (format == FP4_MXFP4 ? c.mx : c.nv)(A, Bt, C, M, N, K, scale_a, scale_b, sf, stream);
}

}  // namespace spark
