// fp8 (e4m3) GEMM, variant 3: variant 2's TMA mainloop on hgemm's Stream-K schedule, with the
// block tile picked per shape.
//
//   C[M,N] = scale_a * scale_b * A[M,K] * Bt[N,K]^T     e4m3 in, bf16 out, fp32 accumulate
//
// hgemm variant 6 (src/kernels/hgemm_tma_sk.cu) on 8-bit operands. A persistent grid of
// blocks, one producer warp and eight consumer warps each, walks the pieces of the
// (tile, k-step) space that the Stream-K plan hands out (hgemm_streamk.cuh): static ranges up
// to a few waves or whenever A and Bt fit in L2, whole tiles from a queue plus shrinking
// K-passes above that, the grouped raster, and the K-ordered fixup through one fp32 slot and
// one epoch-tagged flag per Stream-K tile. No memset, no atomics on data, the same bits every
// run, unlike variants 1 and 2, whose split-K tail adds fp32 partials with atomics.
//
// The producer lane owns the schedule and publishes each piece (tile, kt_begin, kt_end)
// through a two-deep ring in shared memory before it issues the piece's stages; the stage
// counters run on across pieces, so it loads the next piece while the consumers finish the
// last one's fixup or epilogue. A stage is two boxes, BM rows of A and BN rows of Bt by BK
// bytes, the copy engine's swizzle being fp8gemm_tile::swz as in variant 2, and the consumer
// warps run fp8gemm_tile::mma_stage on it. Rows past M are zero-filled by the copy engine (an
// out-of-range box row) and skipped by the epilogue, so any M >= 1 works; N % BN == 0 and
// K % BK == 0 are the tile's requirements.
//
// What variant 3 runs for a shape is one of the TMA tiles in TILES when the route forces it
// (fp8gemm_v3_set_route), and otherwise the route before it: variant 2 where it takes the
// shape, else variant 1, which also has the decode kernels.
// MX mode always takes the route before it: a Stream-K piece can start at any k-step, and the
// MX k-loop runs whole groups of scale chunks.

#include <algorithm>
#include <atomic>
#include <cstdlib>

#include "fp8gemm_tile.cuh"
#include "hgemm_streamk.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

bool fp8gemm_tma_supports(int M, int N, int K);  // variant 2, fp8gemm_tma.cu

namespace {

namespace v3 {

using hgemm_sk::ld_acquire_gpu;
using hgemm_sk::Params;
using hgemm_sk::pass_range;
using hgemm_sk::range_start;
using hgemm_sk::raster;
using hgemm_sk::slot_idx;
using hgemm_sk::st_release_gpu;

constexpr int ITEM_DEPTH = 2;     // pieces the producer may publish ahead of the consumers
constexpr int FIXUP_BARRIER = 1;  // named barrier id for the consumer threads

// The dynamic region: the stages (1 KB aligned, the swizzle is keyed on the absolute
// address), the item ring, then full/empty barriers per stage and per ring slot, rounded to a
// whole KB so a second block on the SM starts aligned too.
template <class C>
constexpr int smem_bytes() {
    return (C::SMEM + ITEM_DEPTH * 16 + (2 * C::STAGES + 2 * ITEM_DEPTH) * 8 + 1023) / 1024 * 1024;
}

template <int BM, int BN, int BK, int STAGES, int WM, int WN, int MIN_BLOCKS>
__global__ void __launch_bounds__((WM * WN + 1) * 32, MIN_BLOCKS)
    fp8gemm_v3_kernel(const __grid_constant__ CUtensorMap tmA,
                      const __grid_constant__ CUtensorMap tmB, __nv_bfloat16* __restrict__ Cout,
                      int M, int N, int K, const float* __restrict__ scale_a,
                      const float* __restrict__ scale_b, Params p) {
    using C = fp8gemm_tile::Cfg<BM, BN, BK, STAGES, WM, WN>;
    static_assert(BK == 64 || BK == 128, "the TMA swizzles cover 64 B and 128 B rows");
    static_assert(C::A_STAGE % 1024 == 0 && C::STAGE % 1024 == 0, "stages on swizzle spans");
    constexpr int CONSUMER_WARPS = WM * WN;
    constexpr int CONSUMERS = CONSUMER_WARPS * 32;
    constexpr int A_STAGE = C::A_STAGE, STAGE_BYTES = C::STAGE;
    constexpr int WROWS = C::WROWS, MT = C::MT, NT = C::NT;

    extern __shared__ __align__(1024) unsigned char smem[];
    int4* items = reinterpret_cast<int4*>(smem + STAGES * STAGE_BYTES);
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(items + ITEM_DEPTH);
    uint64_t* empty_bar = full_bar + STAGES;
    uint64_t* item_full = empty_bar + STAGES;
    uint64_t* item_empty = item_full + ITEM_DEPTH;
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

    if (warp == CONSUMER_WARPS) {
        // Producer warp: lane 0 owns the schedule and the loads.
        if (lane == 0) {
            prefetch_tensormap(&tmA);
            prefetch_tensormap(&tmB);
            unsigned g_kt = 0;   // k-tiles issued so far, across pieces
            unsigned n_pub = 0;  // pieces published so far
            auto piece = [&](int tile, int kb, int ke) {
                const int r = n_pub % ITEM_DEPTH;
                const unsigned use = n_pub / ITEM_DEPTH;
                if (n_pub >= ITEM_DEPTH) mbar_wait(&item_empty[r], (use - 1) & 1);
                items[r] = make_int4(tile, kb, ke, 0);
                mbar_arrive(&item_full[r]);
                ++n_pub;
                if (tile < 0) return;
                int tm, tn;
                raster(p, tile, tm, tn);
                const int bm = tm * BM, bn = tn * BN;
                for (int kt = kb; kt < ke; ++kt, ++g_kt) {
                    const int s = g_kt % STAGES;
                    const unsigned use_s = g_kt / STAGES;
                    if (g_kt >= STAGES) mbar_wait(&empty_bar[s], (use_s - 1) & 1);
                    mbar_arrive_expect_tx(&full_bar[s], STAGE_BYTES);
                    unsigned char* as = smem + s * STAGE_BYTES;
                    tma_load_2d(as, &tmA, &full_bar[s], kt * BK, bm);            // BM rows x BK
                    tma_load_2d(as + A_STAGE, &tmB, &full_bar[s], kt * BK, bn);  // BN rows x BK
                }
            };

            if (p.passes == 0) {
                // Static ranges, walked from the top: the piece that starts a tile (and never
                // waits) first, the piece that ends one last.
                const int it_begin = range_start(p, blockIdx.x);
                int e = range_start(p, blockIdx.x + 1);
                while (e > it_begin) {
                    const int tile = (e - 1) / p.KT;
                    const int k0 = tile * p.KT;
                    const int s = max(it_begin, k0);
                    piece(tile, s - k0, e - k0);
                    e = s;
                }
            } else {
                // Queue: whole tiles first, then the Stream-K region pass by pass.
                const int sk_tiles = p.tiles - p.dp_tiles;
                const int n_items = p.dp_tiles + sk_tiles * p.passes;
                for (;;) {
                    const int item = static_cast<int>(atomicAdd(p.queue, 1u));
                    if (item >= n_items) break;
                    if (item < p.dp_tiles) {
                        piece(item, 0, p.KT);
                    } else {
                        const int q = item - p.dp_tiles;
                        int kb, ke;
                        pass_range(p, q / sk_tiles, kb, ke);
                        piece(p.dp_tiles + q % sk_tiles, kb, ke);
                    }
                }
                // The last block out resets the queue for the next launch.
                __threadfence();
                if (atomicAdd(p.queue + 1, 1u) == static_cast<unsigned>(p.grid) - 1) {
                    p.queue[0] = 0u;
                    p.queue[1] = 0u;
                    __threadfence();
                }
            }
            piece(-1, 0, 0);  // end of work
        }
        return;
    }

    // Consumer warps. A warp holds its ring slot until the piece is stored and reads the
    // piece from it twice, before and after the k-loop, so that nothing but the accumulators
    // and the stage counter is live across the loop: the 128x256 tile has 128 fp32
    // accumulators per lane and 168 registers, and kept live, the piece and its tile
    // coordinates spilled into the k-loop.
    const int wm = warp / WN;
    const int wn = warp % WN;
    unsigned g_kt = 0;    // k-tiles consumed so far, across pieces
    unsigned n_item = 0;  // pieces taken from the ring so far
    typename C::Acc acc;
    const fp8gemm_tile::MxFrag<C> none{};

    // One k-tile out of stage S (a constant in the unrolled loop below).
    auto consume = [&](int S, unsigned parity) {
        mbar_wait(&full_bar[S], parity);
        const unsigned char* as = smem + S * STAGE_BYTES;
        fp8gemm_tile::mma_stage<C, false>(as, as + A_STAGE, wm, wn, lane, acc, none);
        // ldmatrix read the stage through the generic proxy, the refill comes through the
        // async proxy: fence, then one arrive per warp hands the stage back.
        fence_proxy_async_smem();
        __syncwarp();
        if (lane == 0) mbar_arrive(&empty_bar[S]);
    };

    for (;;) {
        const int r = n_item % ITEM_DEPTH;
        mbar_wait(&item_full[r], (n_item / ITEM_DEPTH) & 1);
        int n = items[r].z - items[r].y;
        if (items[r].x < 0) break;  // the end of work stays in its slot: nobody writes it again

#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;
        // Whole rounds of the ring unrolled, so every stage address and barrier is an
        // immediate; a piece that starts or ends inside a round runs those k-tiles one by one.
        while (n > 0) {
            if (n >= STAGES && g_kt % STAGES == 0) {
                const unsigned parity = (g_kt / STAGES) & 1;
                fp8gemm_tile::static_for<STAGES>([&](auto st) { consume(st.value, parity); });
                g_kt += STAGES;
                n -= STAGES;
            } else {
                consume(static_cast<int>(g_kt % STAGES), (g_kt / STAGES) & 1);
                ++g_kt;
                --n;
            }
        }

        const int4 it = items[r];
        __syncwarp();  // every lane has its copy before the warp hands the ring slot back
        if (lane == 0) mbar_arrive(&item_empty[r]);
        ++n_item;
        const int tile = it.x, kb = it.y, ke = it.z;
        int tm, tn;
        raster(p, tile, tm, tn);
        const int bm = tm * BM, bn = tn * BN;
        const int m_valid = M - bm;

        // Fixup (hgemm variant 6's): pieces of a tile are summed in K order through its slot.
        // Unscaled fp32 partials; the scale is applied once by the piece that stores.
        bool finish = true;
        if (kb > 0 || ke < p.KT) {
            const int tile_r = tile - p.dp_tiles;
            float4* slot =
                reinterpret_cast<float4*>(p.slots + static_cast<size_t>(tile_r) * (BM * BN));
            unsigned long long* flag = p.flags + tile_r;
            const unsigned long long tag = static_cast<unsigned long long>(p.epoch) << 32;
            if (kb > 0) {
                // Everything before kb is in the slot once the flag says so.
                if (tid == 0) {
                    while (ld_acquire_gpu(flag) != (tag | static_cast<unsigned>(kb))) {
                    }
                }
                named_barrier_sync(FIXUP_BARRIER, CONSUMERS);
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    if (wm * WROWS + mi * 16 >= m_valid) continue;  // the whole m16 tile is past M
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
                // Hand the running sum to the piece that continues the tile.
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
        if (finish) fp8gemm_tile::store_bf16<C>(acc, *scale_a * *scale_b, Cout, M, N, bm, bn);
    }
}

// ---- host side --------------------------------------------------------------------------

// Schedule knobs. The k-step here is 128 bytes of K, twice hgemm's, so the k-step counts are
// half of hgemm variant 6's (docs/design/fp8gemm.md, "Model shapes").
struct Knobs {
    int min_share = 16;    // static mode: least k-steps per block
    int min_pass = 4;      // queue mode: no K-pass shorter than this
    int static_waves = 2;  // static ranges up to this many waves (always when in L2)
    int group = -1;        // raster group: > 0 fixed, -1 the L2-footprint rule
};

// Sweep overrides, read once: SPARK_FP8GEMM_V3_MIN_SHARE, _MIN_PASS, _STATIC_WAVES and
// _GROUP. SPARK_FP8GEMM_V3_ROUTE is the initial route
// (fp8gemm_v3_set_route).
const Knobs& knobs() {
    static Knobs k;
    static bool read = false;
    if (!read) {
        read = true;
        auto geti = [](const char* name, int fallback) {
            const char* s = std::getenv(name);
            return s ? std::atoi(s) : fallback;
        };
        k.min_share = std::max(1, geti("SPARK_FP8GEMM_V3_MIN_SHARE", k.min_share));
        k.min_pass = geti("SPARK_FP8GEMM_V3_MIN_PASS", k.min_pass);
        k.static_waves = geti("SPARK_FP8GEMM_V3_STATIC_WAVES", k.static_waves);
        k.group = geti("SPARK_FP8GEMM_V3_GROUP", k.group);
    }
    return k;
}

std::atomic<int>& route_slot() {
    static std::atomic<int> slot([] {
        const char* s = std::getenv("SPARK_FP8GEMM_V3_ROUTE");
        return s ? std::atoi(s) : static_cast<int>(FP8GEMM_V3_ROUTE_DEFAULT);
    }());
    return slot;
}

// hgemm_sk::raster_group with one-byte elements: the largest power-of-two group of tile rows
// whose A rows fit in L2 next to the Bt rows of the tiles in flight.
int raster_group(int grid, int tiles_m, int BM, int BN, int K) {
    const double a_row = static_cast<double>(BM) * K;
    const double b_col = static_cast<double>(BN) * K;
    const double budget = hgemm_sk::kL2Share * static_cast<double>(hgemm_sk::l2_bytes());
    int best_fit = 0, best_any = 1;
    double smallest = -1.0;
    for (int g = 1; g <= tiles_m; g *= 2) {
        const double footprint = g * a_row + cdiv(grid, g) * b_col;
        if (footprint <= budget) best_fit = g;
        if (smallest < 0.0 || footprint < smallest) {
            smallest = footprint;
            best_any = g;
        }
    }
    return best_fit > 0 ? best_fit : best_any;
}

// hgemm_sk::plan for one-byte operands: static ranges when A and Bt fit in L2 together or
// the grid is at most static_waves waves, else the queue; the raster group by footprint.
void plan(Params& p, hgemm_sk::Workspace& w, int M, int N, int K, int BM, int BN, int BK,
          int resident, const Knobs& k) {
    p.tiles_m = cdiv(M, BM);
    p.tiles_n = N / BN;
    p.tiles = p.tiles_m * p.tiles_n;
    p.KT = K / BK;
    const long long total = static_cast<long long>(p.tiles) * p.KT;
    const double operands = static_cast<double>(M) * K + static_cast<double>(K) * N;
    const bool in_l2 = operands <= hgemm_sk::kL2Share * static_cast<double>(hgemm_sk::l2_bytes());
    const bool use_static = in_l2 || p.tiles <= static_cast<long long>(resident) * k.static_waves;
    int sk_tiles;
    if (use_static) {
        p.grid = static_cast<int>(
            std::min<long long>(resident, std::max<long long>(p.tiles, total / k.min_share)));
        p.dp_tiles = 0;
        p.passes = 0;
        p.sk_iters = p.tiles * p.KT;
        sk_tiles = p.tiles;
    } else {
        p.grid = resident;
        const int tail = p.tiles % p.grid;
        // The tail and one full wave, or every tile when there are fewer (static_waves = 0).
        sk_tiles = std::min(p.tiles, tail + (tail > 0 ? p.grid : 0));
        p.dp_tiles = p.tiles - sk_tiles;
        p.passes = 1;
        while ((p.KT >> p.passes) > k.min_pass) ++p.passes;
        p.sk_iters = 0;
    }
    const int g = k.group > 0 ? k.group : raster_group(p.grid, p.tiles_m, BM, BN, K);
    p.group = std::max(1, std::min(g, p.tiles_m));

    if (w.tiles < static_cast<size_t>(sk_tiles)) {
        if (w.slots) SPARK_CUDA_CHECK(cudaFree(w.slots));
        if (w.flags) SPARK_CUDA_CHECK(cudaFree(w.flags));
        w.tiles = std::max<size_t>(sk_tiles, 2 * static_cast<size_t>(resident));
        SPARK_CUDA_CHECK(cudaMalloc(&w.slots, w.tiles * BM * BN * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMalloc(&w.flags, w.tiles * sizeof(unsigned long long)));
        SPARK_CUDA_CHECK(cudaMemset(w.flags, 0, w.tiles * sizeof(unsigned long long)));  // once
    }
    if (!w.queue) {
        SPARK_CUDA_CHECK(cudaMalloc(&w.queue, 2 * sizeof(unsigned)));
        SPARK_CUDA_CHECK(cudaMemset(w.queue, 0, 2 * sizeof(unsigned)));  // once
    }
    if (++w.epoch == 0) w.epoch = 1;
    p.slots = w.slots;
    p.flags = w.flags;
    p.queue = w.queue;
    p.epoch = w.epoch;
}

template <int BM, int BN, int BK, int STAGES, int WM, int WN, int MIN_BLOCKS>
int resident_blocks() {
    using C = fp8gemm_tile::Cfg<BM, BN, BK, STAGES, WM, WN>;
    constexpr int bytes = smem_bytes<C>();
    constexpr int threads = (WM * WN + 1) * 32;
    static int resident = 0;
    if (resident == 0) {
        auto kernel = fp8gemm_v3_kernel<BM, BN, BK, STAGES, WM, WN, MIN_BLOCKS>;
        SPARK_CUDA_CHECK(
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, threads, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <int BM, int BN, int BK, int STAGES, int WM, int WN, int MIN_BLOCKS>
void launch(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M, int N,
            int K, const float* scale_a, const float* scale_b, cudaStream_t stream) {
    using C = fp8gemm_tile::Cfg<BM, BN, BK, STAGES, WM, WN>;
    constexpr int bytes = smem_bytes<C>();
    constexpr int threads = (WM * WN + 1) * 32;
    constexpr CUtensorMapSwizzle swz =
        BK == 64 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
    const CUtensorMap tmA = make_tensor_map_2d_u8(A, M, K, BM, BK, swz);
    const CUtensorMap tmB = make_tensor_map_2d_u8(Bt, N, K, BN, BK, swz);
    static hgemm_sk::Workspace w;  // one per tile configuration
    Params p;
    plan(p, w, M, N, K, BM, BN, BK, resident_blocks<BM, BN, BK, STAGES, WM, WN, MIN_BLOCKS>(),
         knobs());
    fp8gemm_v3_kernel<BM, BN, BK, STAGES, WM, WN, MIN_BLOCKS>
        <<<p.grid, threads, bytes, stream>>>(tmA, tmB, Cout, M, N, K, scale_a, scale_b, p);
}

using LaunchFn = void (*)(const unsigned char*, const unsigned char*, __nv_bfloat16*, int, int, int,
                          const float*, const float*, cudaStream_t);

struct TileConfig {
    const char* name;
    LaunchFn launch;
    int (*resident)();
    int bm, bn, bk;
};
#define SPARK_V3_TILE(name, BM, BN, BK, ST, WM, WN, MB)                                            \
    {                                                                                              \
        name, launch<BM, BN, BK, ST, WM, WN, MB>, resident_blocks<BM, BN, BK, ST, WM, WN, MB>, BM, \
            BN, BK                                                                                 \
    }

// The tiles, all eight consumer warps and one producer warp. 0 and 1 are variant 2's
// configurations 16 and 1 (the 128x256 tile with 64x64 warp tiles, and 128x128 with 64x32);
// 2 is configuration 3, two blocks per SM; the 64-wide tiles have 32x32 warp tiles and four
// 24 KB stages at one block per SM.
const TileConfig TILES[] = {
    SPARK_V3_TILE("tma 128x256 bk128 s2", 128, 256, 128, 2, 2, 4, 1),   // 0: 97 KB
    SPARK_V3_TILE("tma 128x128 bk128 s3", 128, 128, 128, 3, 2, 4, 1),   // 1: 97 KB
    SPARK_V3_TILE("tma 128x128 bk64 s2 x2", 128, 128, 64, 2, 2, 4, 2),  // 2: 33 KB, two per SM
    SPARK_V3_TILE("tma 128x64 bk128 s4", 128, 64, 128, 4, 4, 2, 1),     // 3: 97 KB
    SPARK_V3_TILE("tma 64x128 bk128 s4", 64, 128, 128, 4, 2, 4, 1),     // 4: 97 KB
};
#undef SPARK_V3_TILE
constexpr int NUM_TILES = static_cast<int>(sizeof(TILES) / sizeof(TILES[0]));

// What variant 3 runs for a shape: the route before it (variant 2 or 1, in fp8gemm.cu), or a
// tile of TILES.
struct Route {
    enum Kind { OLD, TILE } kind;
    int tile;
};

bool tile_fits(const TileConfig& t, int N, int K) {
    return N % t.bn == 0 && K % t.bk == 0;
}

Route choose(int M, int N, int K) {
    const int r = route_slot().load(std::memory_order_relaxed);
    if (r == FP8GEMM_V3_ROUTE_LEGACY || M <= 64) return {Route::OLD, 0};
    if (r >= 0 && r < NUM_TILES) {
        if (tile_fits(TILES[r], N, K)) return {Route::TILE, r};
        return {Route::OLD, 0};
    }
    return {Route::OLD, 0};
}

}  // namespace v3

}  // namespace

bool fp8gemm_tma_sk(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* Bt, __nv_bfloat16* C, int M, int N,
                    int K, const float* scale_a, const float* scale_b, cudaStream_t stream) {
    const v3::Route r = v3::choose(M, N, K);
    if (r.kind == v3::Route::OLD) return false;
    v3::TILES[r.tile].launch(reinterpret_cast<const unsigned char*>(A),
                             reinterpret_cast<const unsigned char*>(Bt), C, M, N, K, scale_a,
                             scale_b, stream);
    return true;
}

void fp8gemm_v3_set_route(int route) {
    v3::route_slot().store(route, std::memory_order_relaxed);
}

int fp8gemm_v3_route() {
    return v3::route_slot().load(std::memory_order_relaxed);
}

int fp8gemm_v3_num_tiles() {
    return v3::NUM_TILES;
}

const char* fp8gemm_v3_route_name(int M, int N, int K) {
    if (M <= 0 || N % 64 != 0 || K % 64 != 0) return "unsupported";
    const v3::Route r = v3::choose(M, N, K);
    if (r.kind == v3::Route::OLD) {
        if (M <= 64) return "v1 decode";
        return fp8gemm_tma_supports(M, N, K) ? "v2" : "v1";
    }
    return v3::TILES[r.tile].name;
}

}  // namespace spark
