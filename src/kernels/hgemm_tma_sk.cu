// hgemm variant 6: variant 5's TMA mainloop on variant 4's Stream-K schedule.
//
//   C[M,N] = A[M,K] * B[K,N]      row-major, bf16 in/out, fp32 accumulate
//
// A persistent grid of 288-thread blocks (one or two per SM) walks the pieces that variant
// 4's plan hands out (hgemm_streamk.cuh): static (tile, k-range) ranges up to a few waves,
// whole tiles from a queue plus geometrically shrinking K-passes above that, the grouped
// raster, and the K-ordered chain fixup through one fp32 slot and one epoch-tagged flag per
// tile, no memset and the same bits every run. The mainloop is variant 5's
// (hgemm_tma_tile.cuh): one producer lane issues three cp.async.bulk.tensor boxes per stage,
// eight consumer warps wait on the stage's "full" mbarrier, run ldmatrix + mma.sync, fence
// the proxy and arrive on "empty". No __syncthreads anywhere after the barrier init.
//
// What is new is the handoff across pieces. The producer lane owns the schedule: it walks
// the block's static range or takes items from the queue, and publishes each piece (tile,
// kt_begin, kt_end) through a two-deep ring of int4 slots in shared memory, each ring slot
// with its own full/empty mbarrier pair, before it issues the piece's first stage. The
// consumers read the ring in order, so both sides walk the same sequence without a block
// barrier; a tile of -1 ends it. Stage and parity bookkeeping is one running k-tile counter
// per side that never resets between pieces, so the pipeline stays full across a piece
// boundary: while the consumers add the slot, store bf16 or publish their partial for piece
// i, the producer is already filling stages for piece i+1, up to STAGES k-tiles ahead,
// bounded by the "empty" barriers. That is the prologue-hiding variant 4 could not do.
//
// Proxy fences. The TMA writes shared memory through the async proxy; the one generic-proxy
// access to a stage is the consumers' ldmatrix, fenced before the release as in variant 5.
// The fixup reads and writes its slot in global memory straight from registers (__ldcg and
// float4 stores) and the epilogue stores bf16 from registers, so neither touches shared
// memory and neither needs a fence. The item ring is written and read through the generic
// proxy on both sides and ordered by its mbarriers.
//
// Requires N % 64 == 0, K % 64 == 0, any M >= 1. The TMA tile runs when N % 128 == 0 and each
// block's share of the (tile, k-step) space is at least kMinShare k-steps; rows past M are
// zero-filled by the copy engine (an out-of-range box row) and skipped by the epilogue.
// Decode shapes (M <= 64) go to hgemm_decode.cu and everything else to variant 4's 64-row
// tiles, the same routing variant 4 does itself. That was the route until the model-shape
// sweep (HGEMM_V6_ROUTE_LEGACY now). The kernel is templated on the block tile
// (hgemm_tma_tile.cuh), and the default route picks per shape among 128x128, 128x64 and
// 64x128 TMA tiles and variant 4: route_rule below, the measurements in
// docs/design/hgemm.md, "Model shapes".
//
// The fused epilogue (hgemm_epilogue.cuh) is applied by whichever piece stores a tile: a
// whole tile, or the piece that ends a Stream-K chain after it has added the slot. The
// partials that pass through the slot are plain fp32 sums of A B, so bias, activation,
// residual and the gate/up product all happen once, on the finished sum. The FUSED
// instantiation is separate from the plain one so the plain kernel is unchanged.

#include <atomic>
#include <cstdlib>

#include "hgemm_epilogue.cuh"
#include "hgemm_internal.cuh"
#include "hgemm_streamk.cuh"
#include "hgemm_tile.cuh"  // declares hgemm_streamk_bf16
#include "hgemm_tma_tile.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

namespace v6 {

using namespace hgemm_tma_tile;
using hgemm_sk::ld_acquire_gpu;
using hgemm_sk::Params;
using hgemm_sk::pass_range;
using hgemm_sk::range_start;
using hgemm_sk::raster;
using hgemm_sk::slot_idx;
using hgemm_sk::st_release_gpu;

constexpr int ITEM_DEPTH = 2;     // pieces the producer may publish ahead of the consumers
constexpr int FIXUP_BARRIER = 1;  // named barrier id for the 256 consumer threads

// Layout of the dynamic region: the stages (1 KB aligned, see hgemm_tma_tile.cuh), then the
// item ring (16 B each), then the barriers: full/empty per stage, full/empty per ring slot,
// and one "done" barrier for the serialized-prologue experiment.
template <class T, int BK, int STAGES>
constexpr int smem_bytes() {
    return round_kb(STAGES * T::template stage_bytes<BK>() + ITEM_DEPTH * 16 +
                    (2 * STAGES + 2 * ITEM_DEPTH + 1) * 8);
}

// T is the block tile (hgemm_tma_tile.cuh): 128x128 by default, the others for the shapes in
// the "Model shapes" section of docs/design/hgemm.md. The names below shadow the 128x128
// ones at namespace scope.
template <class T, int BK, int STAGES, int MIN_BLOCKS, bool FUSED>
__global__ void __launch_bounds__(T::THREADS, MIN_BLOCKS)
    hgemm_v6_kernel(const __grid_constant__ CUtensorMap tmA,
                    const __grid_constant__ CUtensorMap tmB, __nv_bfloat16* __restrict__ C, int M,
                    int N, int K, Params p, int serial, hgemm_epi::Ep ep) {
    static_assert(BK == 32 || BK == 64, "the swizzle helpers assume 64 B or 128 B rows of A");
    constexpr int BM = T::BM, BN = T::BN, WM = T::WM, WN = T::WN, MT = T::MT, NT = T::NT;
    constexpr int WARPS_N = T::WARPS_N;
    constexpr int CONSUMER_WARPS = T::CONSUMER_WARPS, CONSUMERS = T::CONSUMERS;
    using Acc = typename T::Acc;
    constexpr int STAGE_BYTES = T::template stage_bytes<BK>();

    extern __shared__ __align__(1024) unsigned char smem[];
    int4* items = reinterpret_cast<int4*>(smem + STAGES * STAGE_BYTES);
    uint64_t* full_bar = reinterpret_cast<uint64_t*>(items + ITEM_DEPTH);
    uint64_t* empty_bar = full_bar + STAGES;
    uint64_t* item_full = empty_bar + STAGES;
    uint64_t* item_empty = item_full + ITEM_DEPTH;
    uint64_t* done_bar = item_empty + ITEM_DEPTH;
    // The swizzle the copy engine applies is keyed on the absolute smem address.
    if (smem_u32(smem) % 1024 != 0) __trap();

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < STAGES; ++s) {
            mbar_init(&full_bar[s], 1);                // the producer's arrive.expect_tx
            mbar_init(&empty_bar[s], CONSUMER_WARPS);  // one arrive per consumer warp
        }
#pragma unroll
        for (int i = 0; i < ITEM_DEPTH; ++i) {
            mbar_init(&item_full[i], 1);                // the producer's publish
            mbar_init(&item_empty[i], CONSUMER_WARPS);  // one arrive per warp once it has read
        }
        mbar_init(done_bar, CONSUMER_WARPS);  // one arrive per warp after a piece's epilogue
        fence_mbar_init();
    }
    __syncthreads();  // the only block-wide barrier in the kernel

    if (warp == CONSUMER_WARPS) {
        // Producer warp: lane 0 owns the schedule and the loads, the rest of the warp leaves.
        if (lane == 0) {
            prefetch_tensormap(&tmA);
            prefetch_tensormap(&tmB);
            unsigned g_kt = 0;   // k-tiles issued so far, across pieces
            unsigned n_pub = 0;  // pieces published so far
            // Publish (tile, kb, ke) through the ring, then issue its k-tiles. Ring slot r is
            // used for pieces r, r+DEPTH, ...; its n-th use waits for the consumers' n-1-th
            // release, exactly like a stage. Stage g_kt % STAGES is used for the
            // g_kt / STAGES-th time whichever piece it belongs to.
            auto piece = [&](int tile, int kb, int ke) {
                const int r = n_pub % ITEM_DEPTH;
                const unsigned use = n_pub / ITEM_DEPTH;
                if (n_pub >= ITEM_DEPTH) mbar_wait(&item_empty[r], (use - 1) & 1);
                items[r] = make_int4(tile, kb, ke, 0);
                mbar_arrive(&item_full[r]);
                ++n_pub;
                if (tile < 0) return;
                // Experiment (SPARK_HGEMM_V6_SERIAL=1): hold the next piece's first stage
                // until the consumers have finished the previous piece's epilogue, which is
                // what a per-piece launch or variant 4's prologue amounts to.
                if (serial && n_pub >= 2) mbar_wait(done_bar, (n_pub - 2) & 1);
                int tm, tn;
                raster(p, tile, tm, tn);
                const int bm = tm * BM, bn = tn * BN;
                for (int kt = kb; kt < ke; ++kt, ++g_kt) {
                    const int s = g_kt % STAGES;
                    const unsigned use_s = g_kt / STAGES;
                    if (g_kt >= STAGES) mbar_wait(&empty_bar[s], (use_s - 1) & 1);
                    T::template issue_stage<BK>(smem + s * STAGE_BYTES, &tmA, &tmB, &full_bar[s],
                                                kt * BK, bm, bn);
                }
            };

            if (p.passes == 0) {
                // Static ranges, walked from the top: the piece that starts a tile (and
                // never waits) comes first, the piece that ends one last. dp_tiles is 0, so
                // a tile's raster index is its slot index.
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
                // Queue: whole tiles first, then the Stream-K region pass by pass, tile
                // interleaved, so a pass's predecessor in K was taken a whole region earlier.
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
                // The last block out resets the queue for the next launch. Every block has
                // made its final, failing grab before it counts itself done.
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

    // Consumer warps.
    const int wm = warp / WARPS_N;
    const int wn = warp % WARPS_N;
    const int g = lane >> 2;
    const int c2 = (lane & 3) * 2;
    unsigned g_kt = 0;    // k-tiles consumed so far, across pieces
    unsigned n_item = 0;  // pieces taken from the ring so far
    Acc acc;

    for (;;) {
        const int r = n_item % ITEM_DEPTH;
        mbar_wait(&item_full[r], (n_item / ITEM_DEPTH) & 1);
        const int4 it = items[r];
        __syncwarp();  // every lane has its copy before the warp hands the ring slot back
        if (lane == 0) mbar_arrive(&item_empty[r]);
        ++n_item;
        if (it.x < 0) break;
        const int tile = it.x, kb = it.y, ke = it.z;
        int tm, tn;
        raster(p, tile, tm, tn);
        const int bm = tm * BM, bn = tn * BN;
        const int m_valid = M - bm;

        // A piece that ends its tile applies the epilogue; its bias and residual are loaded
        // now, before the k-loop, so their DRAM round trip hides under the mma work (40
        // registers, dead again after the store).
        const hgemm_epi::Geo geo{M, N, bm + wm * WM + g, bn + wn * WN + c2};
        hgemm_epi::Pre<MT, NT> pre;
        if constexpr (FUSED) {
            if (ke == p.KT) hgemm_epi::prefetch(pre, ep, geo);
        }

        T::zero_acc(acc);
        for (int kt = kb; kt < ke; ++kt, ++g_kt) {
            const int s = g_kt % STAGES;
            mbar_wait(&full_bar[s], (g_kt / STAGES) & 1);
            T::template consume_stage<BK>(acc, smem + s * STAGE_BYTES, wm, wn, lane);
            // ldmatrix read the stage through the generic proxy; the refill comes through the
            // async proxy. The fence orders the reads before the release, one arrive per warp.
            fence_proxy_async_smem();
            __syncwarp();
            if (lane == 0) mbar_arrive(&empty_bar[s]);
        }

        // Fixup, as in variant 4, with a named barrier in place of __syncthreads: the
        // producer lane is not part of it (it is already loading the next piece).
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
                    if (wm * WM + mi * 16 >= m_valid) continue;  // the whole m16 tile is past M
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
                    if (wm * WM + mi * 16 >= m_valid) continue;
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
            // Epilogue from registers: each lane owns (row g, cols 2c..2c+1) and (row g+8,
            // same) of every m16n8 fragment. M % 16 == 0 is not required: row and row + 8
            // are checked one by one. The fused form (hgemm_epilogue.cuh) is warp-uniform.
            if constexpr (FUSED)
                hgemm_epi::store_warp_tile_fused(acc, pre, C, geo, ep);
            else
                hgemm_epi::store_warp_tile<false, MT, NT>(acc, C, M, N, geo.row0, geo.col0, ep);
        }
        if (serial) {
            __syncwarp();
            if (lane == 0) mbar_arrive(done_bar);
        }
    }
}

// ---- host side --------------------------------------------------------------------------

// Schedule knobs (swept on the RTX 5090, see docs/design/hgemm.md, "Variant 6"). Against
// variant 4's: static ranges for any number of waves when A and B fit in L2 (they beat the
// queue by 3.5% at 4096^3 and by 3% at twelve waves), and the L2-footprint raster group
// (G = 16 instead of 8 at 8192^3 on 170 blocks, 243 against 241 TFLOPS; G = 32 at
// 4096x11008x4096, 243 against 239).
constexpr hgemm_sk::Knobs kKnobs = {hgemm_sk::kMinShare, hgemm_sk::kMinPass, hgemm_sk::kStaticWaves,
                                    /*static_waves_l2=*/0,
                                    /*group=*/-1};

// The per-shape rule's share floor for a problem that no tile can give every block
// kMinShare k-steps of: the smallest tile, cut into pieces of at least this many k-steps, so
// that the grid covers more of the card (a 128 x 1024 x 4096 projection is 16 tiles of
// 128 x 64 and 1,024 k-steps: 32 blocks at kMinShare, 128 at 8).
constexpr int kMinShareSmall = 8;

// Sweep overrides, read once: SPARK_HGEMM_V6_CONFIG forces a pipeline configuration (index
// into CONFIGS below), SPARK_HGEMM_V6_GROUP the raster group, SPARK_HGEMM_V6_STATIC_WAVES
// and SPARK_HGEMM_V6_STATIC_WAVES_L2 the static/queue crossover, SPARK_HGEMM_V6_MIN_SHARE
// the least k-steps per block, SPARK_HGEMM_V6_MIN_SHARE_SMALL the rule's floor above,
// SPARK_HGEMM_V6_MIN_PASS the shortest K-pass, SPARK_HGEMM_V6_SERIAL=1 the
// serialized-prologue experiment. SPARK_HGEMM_V6_ROUTE is the initial tile route
// (hgemm_v6_set_route).
struct Env {
    int cfg = -1;
    int serial = 0;
    int min_share_small = kMinShareSmall;
    hgemm_sk::Knobs knobs = kKnobs;
};
const Env& env() {
    static Env e;
    static bool read = false;
    if (!read) {
        read = true;
        auto geti = [](const char* name, int fallback) {
            const char* s = std::getenv(name);
            return s ? std::atoi(s) : fallback;
        };
        e.cfg = geti("SPARK_HGEMM_V6_CONFIG", -1);
        e.serial = geti("SPARK_HGEMM_V6_SERIAL", 0);
        e.min_share_small = geti("SPARK_HGEMM_V6_MIN_SHARE_SMALL", e.min_share_small);
        e.knobs.group = geti("SPARK_HGEMM_V6_GROUP", e.knobs.group);
        e.knobs.static_waves = geti("SPARK_HGEMM_V6_STATIC_WAVES", e.knobs.static_waves);
        e.knobs.static_waves_l2 = geti("SPARK_HGEMM_V6_STATIC_WAVES_L2", e.knobs.static_waves_l2);
        e.knobs.min_share = geti("SPARK_HGEMM_V6_MIN_SHARE", e.knobs.min_share);
        e.knobs.min_pass = geti("SPARK_HGEMM_V6_MIN_PASS", e.knobs.min_pass);
    }
    return e;
}

// The tile route, settable at run time (hgemm_v6_set_route) so one bench process can sweep it.
std::atomic<int>& route_slot() {
    static std::atomic<int> slot([] {
        const char* s = std::getenv("SPARK_HGEMM_V6_ROUTE");
        return s ? std::atoi(s) : static_cast<int>(HGEMM_V6_ROUTE_DEFAULT);
    }());
    return slot;
}

template <class T, int BK, int STAGES, int MIN_BLOCKS, bool FUSED>
int resident_blocks() {
    constexpr int bytes = smem_bytes<T, BK, STAGES>();
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(hgemm_v6_kernel<T, BK, STAGES, MIN_BLOCKS, FUSED>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, hgemm_v6_kernel<T, BK, STAGES, MIN_BLOCKS, FUSED>, T::THREADS, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <class T, int BK, int STAGES, int MIN_BLOCKS, bool FUSED>
void launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K,
            cudaStream_t stream, const HgemmEpilogue& ep, const hgemm_sk::Knobs& knobs) {
    constexpr int bytes = smem_bytes<T, BK, STAGES>();
    CUtensorMap tmA, tmB;
    T::template make_maps<BK>(A, B, M, N, K, tmA, tmB);
    static hgemm_sk::Workspace w;  // one per tile, pipeline configuration and instantiation
    Params p;
    hgemm_sk::plan(p, w, M, N, K, T::BM, T::BN, BK,
                   resident_blocks<T, BK, STAGES, MIN_BLOCKS, FUSED>(), knobs);
    hgemm_v6_kernel<T, BK, STAGES, MIN_BLOCKS, FUSED><<<p.grid, T::THREADS, bytes, stream>>>(
        tmA, tmB, C, M, N, K, p, env().serial, hgemm_epi::make(ep));
}

using LaunchFn = void (*)(const __nv_bfloat16*, const __nv_bfloat16*, __nv_bfloat16*, int, int,
                          int, cudaStream_t, const HgemmEpilogue&, const hgemm_sk::Knobs&);

// A tile and pipeline configuration with a plain and a fused instantiation; the plain one's
// residency is what the choice below is made from (the two agree: smem sets it).
struct Config {
    const char* name;
    LaunchFn launch;
    LaunchFn launch_fused;
    int (*resident)();
    int bm, bn, bk;
};
#define SPARK_V6_CONFIG(name, T, BK, ST, MB)                                         \
    {                                                                                \
        name, launch<T, BK, ST, MB, false>, launch<T, BK, ST, MB, true>,              \
            resident_blocks<T, BK, ST, MB, false>, T::BM, T::BN, BK                  \
    }

// Pipeline configurations of the 128x128 tile (BK, stages, blocks per SM), the same six as
// variant 5. Index 0 wins or ties on every shape (docs/design/hgemm.md, "Variant 6"): the
// two-block configurations spill inside the k-loop at the 96 registers the ninth warp leaves
// them.
const Config CONFIGS[] = {
    SPARK_V6_CONFIG("tma 128x128 bk64 s2", Tile128, 64, 2, 1),     // 0: 64 KB, one per SM
    SPARK_V6_CONFIG("tma 128x128 bk64 s3", Tile128, 64, 3, 1),     // 1: 96 KB, one per SM
    SPARK_V6_CONFIG("tma 128x128 bk32 s3 x2", Tile128, 32, 3, 2),  // 2: 48 KB, two per SM
    SPARK_V6_CONFIG("tma 128x128 bk32 s2 x2", Tile128, 32, 2, 2),  // 3: 32 KB, two per SM
    SPARK_V6_CONFIG("tma 128x128 bk32 s3", Tile128, 32, 3, 1),     // 4: like 2 without the cap
    SPARK_V6_CONFIG("tma 128x128 bk32 s6", Tile128, 32, 6, 1),     // 5: 96 KB, one per SM
};
constexpr int NUM_CONFIGS = static_cast<int>(sizeof(CONFIGS) / sizeof(CONFIGS[0]));
constexpr int CONFIG_DEFAULT = 0;

// Tiles for the model shapes, all BK = 64 and 288 threads (docs/design/hgemm.md, "Model
// shapes"). 0 is CONFIGS[0]. 128x256 has a 64x64 warp tile, 128 fp32 accumulators per lane,
// and two 48 KB stages. The 64-row and 64-column tiles have a 32x32 warp tile; with a 24 KB
// stage they fit four stages at one block per SM, or two stages at two blocks per SM (2 x 49
// KB plus the reserved KB per block is the 5090's 100 KB per SM exactly), where their 32
// accumulators leave room under the 96-register cap that made the 128x128 two-block
// configurations spill.
using Tile128x256 = Tile<128, 256, 2, 4>;
using Tile128x64 = Tile<128, 64, 4, 2>;
using Tile64x128 = Tile<64, 128, 2, 4>;
const Config TILES[] = {
    SPARK_V6_CONFIG("tma 128x128 s2", Tile128, 64, 2, 1),       // 0
    SPARK_V6_CONFIG("tma 128x256 s2", Tile128x256, 64, 2, 1),   // 1
    SPARK_V6_CONFIG("tma 128x64 s4", Tile128x64, 64, 4, 1),     // 2
    SPARK_V6_CONFIG("tma 64x128 s4", Tile64x128, 64, 4, 1),     // 3
    SPARK_V6_CONFIG("tma 128x64 s2 x2", Tile128x64, 64, 2, 2),  // 4
    SPARK_V6_CONFIG("tma 64x128 s2 x2", Tile64x128, 64, 2, 2),  // 5
};
#undef SPARK_V6_CONFIG
constexpr int NUM_TILES = static_cast<int>(sizeof(TILES) / sizeof(TILES[0]));
constexpr int kTile128 = 0, kTile128x256 = 1, kTile128x64 = 2, kTile64x128 = 3;

// k-steps per block of a configuration on this shape, -1 when N does not tile.
long long share_of(const Config& c, int M, int N, int K) {
    if (N % c.bn != 0) return -1;
    return static_cast<long long>(cdiv(M, c.bm)) * (N / c.bn) * (K / c.bk) / c.resident();
}

// What variant 6 runs for a shape.
struct Route {
    enum Kind { DECODE, V4, CONFIG, TILE } kind;
    int index;      // into CONFIGS or TILES
    int min_share;  // the static schedule's least k-steps per block
};

// The per-shape rule, the default route since the model-shape sweep (docs/design/hgemm.md,
// "Model shapes"). One quantity decides it besides the share: how many tiles there are
// against the blocks in flight. With fewer tiles than a quarter of the resident blocks
// every tile is cut four or more ways, and the chain of fixups costs more than the bigger
// tile saves.
//   1. 128x128 when every block gets kMinShare k-steps and there are at least resident / 4
//      tiles (the old route, minus the shapes with too few tiles: 128x4096x14336 is 32 tiles
//      and ran 104.5% of cuBLAS on 128x128, 114.6% on 128x64).
//   2. Otherwise the 64-wide tile with fewer zero-filled rows past M (128x64 on a tie, or
//      when N % 128 != 0), when it has at least resident / 4 tiles: with kMinShare if its
//      blocks get that many k-steps, else with a floor of kMinShareSmall so the grid covers
//      the card (512x1024x4096: 117% of cuBLAS at a floor of 8, 102% at 32).
//   3. Fewer tiles than that even at 64 wide (128x1024x4096 is 16): variant 4's 64x64
//      cp.async tiles, two blocks per SM and four times the tiles (80.8% of cuBLAS there,
//      69.1% on 128x64).
// 128x256 is not in the rule: at 168 registers it spills and lost to 128x128 on every
// shape it took but the largest lm_head ones (within 0.5%). It stays as a forced tile.
Route route_rule(int M, int N, int K) {
    const Env& e = env();
    const int quarter = TILES[kTile128].resident() / 4;
    auto tiles = [&](const Config& c) { return cdiv(M, c.bm) * (N / c.bn); };
    const Config& big = CONFIGS[CONFIG_DEFAULT];
    if (N % big.bn == 0 && tiles(big) >= quarter && share_of(big, M, N, K) >= e.knobs.min_share)
        return {Route::CONFIG, CONFIG_DEFAULT, e.knobs.min_share};
    const int small = (N % 128 != 0 || cdiv(M, 128) * 128 <= cdiv(M, 64) * 64) ? kTile128x64
                                                                                 : kTile64x128;
    if (tiles(TILES[small]) < quarter) return {Route::V4, 0, 0};
    if (share_of(TILES[small], M, N, K) >= e.knobs.min_share)
        return {Route::TILE, small, e.knobs.min_share};
    // The floor only helps when K can be cut: with fewer than two floors of k-steps per tile
    // and fewer tiles than blocks, a piece is most of a tile plus a fixup, and variant 4's
    // whole 64x64 tiles fill the card instead (256x2048x768: 125.5% of cuBLAS on variant 4,
    // 102.3% on 128x64 at a floor of 8).
    if (K / TILES[small].bk < 2 * e.min_share_small && tiles(TILES[small]) < 4 * quarter)
        return {Route::V4, 0, 0};
    return {Route::TILE, small, e.min_share_small};
}

Route choose(int M, int N, int K) {
    const Env& e = env();
    if (M <= 64)  // decode: stream B (hgemm_decode.cu), or variant 4's 64-row tile
        return {hgemm_decode::supports(M, N, K) ? Route::DECODE : Route::V4, 0, 0};
    const int r = route_slot().load(std::memory_order_relaxed);
    if (r >= 0 && r < NUM_TILES && N % TILES[r].bn == 0)
        return {Route::TILE, r, e.knobs.min_share};
    if (r != HGEMM_V6_ROUTE_LEGACY && e.cfg < 0) return route_rule(M, N, K);
    // The route before the model-shape sweep (and the one SPARK_HGEMM_V6_CONFIG sweeps):
    // the 128x128 tile when each block's share of the (tile, k-step) space is at least
    // kMinShare k-steps, variant 4's smaller tiles below that.
    int cfg = e.cfg;
    if (cfg < 0 || cfg >= NUM_CONFIGS) cfg = CONFIG_DEFAULT;
    if (share_of(CONFIGS[cfg], M, N, K) < e.knobs.min_share) return {Route::V4, 0, 0};
    return {Route::CONFIG, cfg, e.knobs.min_share};
}

void launch_auto(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                 int K, cudaStream_t stream, const HgemmEpilogue& ep) {
    const Route r = choose(M, N, K);
    if (r.kind == Route::DECODE) {
        hgemm_decode::launch(A, B, C, M, N, K, stream, ep);
        return;
    }
    if (r.kind == Route::V4) {
        hgemm_streamk_bf16(A, B, C, M, N, K, stream, ep);
        return;
    }
    const Config& c = r.kind == Route::CONFIG ? CONFIGS[r.index] : TILES[r.index];
    hgemm_sk::Knobs knobs = env().knobs;
    knobs.min_share = r.min_share;
    if (hgemm_epi::is_plain(ep))
        c.launch(A, B, C, M, N, K, stream, ep, knobs);
    else
        c.launch_fused(A, B, C, M, N, K, stream, ep, knobs);
}

const char* route_name(int M, int N, int K) {
    const Route r = choose(M, N, K);
    switch (r.kind) {
        case Route::DECODE:
            return "decode";
        case Route::V4:
            return "v4 cp.async";
        case Route::CONFIG:
            return CONFIGS[r.index].name;
        default:
            if (r.min_share != env().knobs.min_share)
                return r.index == kTile64x128 ? "tma 64x128 s4, small share"
                                              : "tma 128x64 s4, small share";
            return TILES[r.index].name;
    }
}

}  // namespace v6

}  // namespace

bool hgemm_tma_sk_supports(int M, int N, int K) {
    return M >= 1 && N % 64 == 0 && K % 64 == 0;  // variant 4's rules
}

void hgemm_tma_sk(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                  int K, cudaStream_t stream, const HgemmEpilogue& ep) {
    SPARK_REQUIRE(hgemm_tma_sk_supports(M, N, K),
                  "hgemm variant 6: requires N % 64 == 0 and K % 64 == 0");
    SPARK_REQUIRE(is_aligned16(A) && is_aligned16(B),
                  "hgemm variant 6: A and B must be 16-byte aligned");
    v6::launch_auto(A, B, C, M, N, K, stream, ep);
}

void hgemm_v6_set_route(int route) {
    v6::route_slot().store(route, std::memory_order_relaxed);
}

int hgemm_v6_route() {
    return v6::route_slot().load(std::memory_order_relaxed);
}

int hgemm_v6_num_tiles() {
    return v6::NUM_TILES;
}

const char* hgemm_v6_route_name(int M, int N, int K) {
    if (!hgemm_tma_sk_supports(M, N, K)) return "unsupported";
    return v6::route_name(M, N, K);
}

}  // namespace spark
