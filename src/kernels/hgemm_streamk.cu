// hgemm variant 4: the variant 3 tile (hgemm_tile.cuh) on a persistent Stream-K schedule.
//
//   C[M,N] = A[M,K] * B[K,N]      row-major, bf16 in/out, fp32 accumulate
//
// What changes against variant 3 is only who does which work. A persistent grid of at most
// `resident` blocks (2 per SM for the 128x128 tile, 340 on the RTX 5090) walks the tiles in
// a grouped raster order (G tile rows column-major, so the tiles in flight share A rows and
// B columns and fit in L2). Two schedules, picked per call:
//
//   * static ranges (up to two waves, or four when both operands fit in L2): the flattened
//     (tile, k-step) space is cut into equal contiguous ranges, one per block. Every block
//     gets the same number of k-steps, so there is no quantization at all; a tile is
//     finished by two or three blocks.
//   * queue (more waves): whole tiles are handed out from a queue (one atomicAdd per tile,
//     so a block on a slow SM simply takes fewer), and the last wave plus the tail is handed
//     out from the same queue in K-passes of geometrically shrinking length (half the tile,
//     a quarter, ... the last one 9 to 16 k-steps), so the kernel ends within a few k-steps
//     of work on every SM instead of within one tile.
//
// Both use the same fixup. Each Stream-K tile has one fp32 slot and one flag that says how
// many k-steps have been accumulated into the slot. A piece that does not start the tile
// waits (acquire) until the flag reads its kt_begin, then adds the slot into its
// accumulators; a piece that does not end the tile writes the slot and publishes its kt_end
// (release); the piece that ends the tile stores bf16. Partials are always summed in K
// order, so the result is the same bits every run, whichever block computed which piece.
// The flag carries a per-launch epoch in its high word, so a stale value can never match
// and nothing is cleared between launches; the queue is reset by the last block out.
//
// A piece only ever waits for the piece before it in K, which was handed out earlier: in
// queue mode by a block that is running because it took it from the queue, in static mode
// by the block with the next lower index, which was dispatched no later than this one and
// is resident because the grid never exceeds the resident block count. Chains end at the
// piece that starts the tile, which never waits, so there is no cycle.
//
// Requires N % 64 == 0, K % 64 == 0, any M >= 1 (the same rules as variant 3: rows past M
// are zero-filled by the tile and skipped by the epilogue).

#include <algorithm>

#include "hgemm_internal.cuh"
#include "hgemm_tile.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

namespace sk {

using hgemm_tile::THREADS;

struct Params {
    int tiles_m, tiles_n, tiles;
    int KT;        // k-steps per tile
    int group;     // tile rows per raster group (1: plain row-major over N)
    int dp_tiles;  // queue mode: tiles [0, dp_tiles) in raster order are whole; the rest are
                   // the Stream-K region. Static mode: 0.
    int passes;    // queue mode: K-passes per Stream-K tile; 0 = static ranges
    int sk_iters;  // static mode: tiles * KT, cut into `grid` equal ranges
    int grid;      // blocks in the launch, all resident at once
    float* slots;  // (tiles - dp_tiles) x BM x BN fp32, one per Stream-K tile
    unsigned long long* flags;  // per Stream-K tile: (epoch << 32) | k-steps accumulated
    unsigned* queue;            // [0]: next work item, [1]: blocks done
    unsigned epoch;
};

__device__ __forceinline__ void st_release_gpu(unsigned long long* p, unsigned long long v) {
    asm volatile("st.release.gpu.global.u64 [%0], %1;\n" ::"l"(p), "l"(v) : "memory");
}
__device__ __forceinline__ unsigned long long ld_acquire_gpu(const unsigned long long* p) {
    unsigned long long v;
    asm volatile("ld.acquire.gpu.global.u64 %0, [%1];\n" : "=l"(v) : "l"(p) : "memory");
    return v;
}

// Grouped rasterization: tile indices walk G tile rows column-major before moving to the
// next G rows. Consecutive indices then share B columns (same tn) across G tiles and A
// rows across the tiles of one group, instead of sharing A only along a whole tile row.
__device__ __forceinline__ void raster(const Params& p, int tile, int& tm, int& tn) {
    const int per_group = p.group * p.tiles_n;
    const int g = tile / per_group;
    const int first = g * p.group;
    const int gsz = min(p.group, p.tiles_m - first);
    const int r = tile - g * per_group;
    tm = first + r % gsz;
    tn = r / gsz;
}

// Queue mode, pass j of a Stream-K tile: k-steps [KT - KT/2^j, KT - KT/2^(j+1)), the last
// pass running to KT. Half the tile, then a quarter, ... so the last passes are short (the
// kernel ends within one of them on every SM) while a tile has only `passes` chain links,
// 4 at K = 4096, 5 at 8192, 6 at 11008.
__device__ __forceinline__ void pass_range(const Params& p, int j, int& kb, int& ke) {
    kb = j == 0 ? 0 : p.KT - (p.KT >> j);
    ke = j + 1 == p.passes ? p.KT : p.KT - (p.KT >> (j + 1));
}

// Static mode: first iteration owned by block c; ranges are [start(c), start(c+1)).
__device__ __forceinline__ int range_start(const Params& p, int c) {
    return static_cast<int>(static_cast<long long>(c) * p.sk_iters / p.grid);
}

// A partial tile in its slot, in fragment order: lane-contiguous float4s, so every warp
// writes and reads 512-byte runs and a lane finds its own accumulator elements at the same
// place whichever block wrote them. Rows past M are skipped on both sides.
template <class Cfg>
__device__ __forceinline__ int slot_idx(int warp, int lane, int mi, int nj) {
    return ((warp * Cfg::MT + mi) * Cfg::NT + nj) * 32 + lane;
}

template <int BM, int BN, int BK, int STAGES>
__global__ void __launch_bounds__(THREADS)
    hgemm_v4_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ B,
                    __nv_bfloat16* __restrict__ C, int M, int N, int K, Params p) {
    using Cfg = hgemm_tile::Cfg<BM, BN, BK, STAGES>;
    constexpr int WM = Cfg::WM, MT = Cfg::MT, NT = Cfg::NT;
    extern __shared__ __align__(128) unsigned char smem_raw[];
    __shared__ int s_item;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / hgemm_tile::WARPS_N;
    const int cta = blockIdx.x;
    typename Cfg::Acc acc;

    // One piece of Stream-K tile `tile_r` (index within the region): k-steps [kt_begin,
    // kt_end). Leaves the pipeline's smem free for the next call.
    auto piece = [&](int tile_r, int kt_begin, int kt_end) {
        int tm, tn;
        raster(p, p.dp_tiles + tile_r, tm, tn);
        const int bm = tm * BM, bn = tn * BN;
        const int m_valid = M - bm;
        hgemm_tile::mainloop<Cfg>(A, B, N, K, bm, bn, m_valid, kt_begin, kt_end - kt_begin,
                                  smem_raw, acc);
        if (kt_begin > 0 || kt_end < p.KT) {
            float4* slot =
                reinterpret_cast<float4*>(p.slots + static_cast<size_t>(tile_r) * (BM * BN));
            unsigned long long* flag = p.flags + tile_r;
            const unsigned long long tag = static_cast<unsigned long long>(p.epoch) << 32;
            if (kt_begin > 0) {
                // Everything before kt_begin is in the slot once the flag says so.
                if (tid == 0) {
                    while (ld_acquire_gpu(flag) != (tag | static_cast<unsigned>(kt_begin))) {
                    }
                }
                __syncthreads();
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    if (wm * WM + mi * 16 >= m_valid) continue;  // the whole m16 tile is past M
#pragma unroll
                    for (int nj = 0; nj < NT; ++nj) {
                        const float4 v = __ldcg(slot + slot_idx<Cfg>(warp, lane, mi, nj));
                        acc[mi][nj][0] += v.x;
                        acc[mi][nj][1] += v.y;
                        acc[mi][nj][2] += v.z;
                        acc[mi][nj][3] += v.w;
                    }
                }
            }
            if (kt_end < p.KT) {
                // Hand the running sum to the piece that continues the tile.
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    if (wm * WM + mi * 16 >= m_valid) continue;
#pragma unroll
                    for (int nj = 0; nj < NT; ++nj)
                        slot[slot_idx<Cfg>(warp, lane, mi, nj)] = make_float4(
                            acc[mi][nj][0], acc[mi][nj][1], acc[mi][nj][2], acc[mi][nj][3]);
                }
                __threadfence();
                __syncthreads();
                if (tid == 0) st_release_gpu(flag, tag | static_cast<unsigned>(kt_end));
                return;
            }
        }
        hgemm_tile::store_bf16<Cfg>(acc, C, M, N, bm, bn);
        __syncthreads();  // the pipeline's last stage is free before the next prologue
    };

    if (p.passes == 0) {
        // Static ranges, walked from the top: the piece that starts a tile (and never waits)
        // comes first, the piece that ends one (and may wait for the block below) last.
        const int it_begin = range_start(p, cta);
        int e = range_start(p, cta + 1);
        while (e > it_begin) {
            const int tile_r = (e - 1) / p.KT;
            const int tile_k0 = tile_r * p.KT;
            const int s = max(it_begin, tile_k0);
            piece(tile_r, s - tile_k0, e - tile_k0);
            e = s;
        }
        return;
    }

    // Queue: whole tiles first, then the Stream-K region pass by pass. Passes are handed
    // out tile-interleaved (every tile's first pass, then every tile's second, ...), so a
    // pass's predecessor in K was taken a whole region earlier and has almost always been
    // published by the time it is needed.
    const int sk_tiles = p.tiles - p.dp_tiles;
    const int items = p.dp_tiles + sk_tiles * p.passes;
    for (;;) {
        if (tid == 0) s_item = static_cast<int>(atomicAdd(p.queue, 1u));
        __syncthreads();
        const int item = s_item;
        if (item >= items) break;
        if (item < p.dp_tiles) {
            int tm, tn;
            raster(p, item, tm, tn);
            const int bm = tm * BM, bn = tn * BN;
            hgemm_tile::mainloop<Cfg>(A, B, N, K, bm, bn, M - bm, 0, p.KT, smem_raw, acc);
            hgemm_tile::store_bf16<Cfg>(acc, C, M, N, bm, bn);
            __syncthreads();
        } else {
            const int q = item - p.dp_tiles;
            int kb, ke;
            pass_range(p, q / sk_tiles, kb, ke);
            piece(q % sk_tiles, kb, ke);
        }
    }
    // The last block out resets the queue for the next launch. Every block has made its
    // final, failing grab before it counts itself done, so nothing touches the queue after
    // the reset until the next launch.
    if (tid == 0) {
        __threadfence();
        if (atomicAdd(p.queue + 1, 1u) == static_cast<unsigned>(p.grid) - 1) {
            p.queue[0] = 0u;
            p.queue[1] = 0u;
            __threadfence();
        }
    }
}

// Schedule constants (swept on the RTX 5090, see docs/design/hgemm.md).
constexpr int kMinShare = 32;      // static mode: least k-steps per block; tile choice threshold
constexpr int kMinPass = 8;        // queue mode: no pass shorter than this; the last is 9 to 16
constexpr int kStaticWaves = 2;    // static ranges up to this many waves ...
constexpr int kStaticWavesL2 = 4;  // ... or this many when A and B fit in L2 together
constexpr double kL2Share = 0.75;  // "fit": A + B <= this share of the L2

inline size_t l2_bytes() {
    static size_t bytes = 0;
    if (bytes == 0) {
        int dev = 0, l2 = 0;
        SPARK_CUDA_CHECK(cudaGetDevice(&dev));
        SPARK_CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
        bytes = l2 > 0 ? static_cast<size_t>(l2) : size_t{96} << 20;
    }
    return bytes;
}

struct Workspace {
    float* slots = nullptr;
    unsigned long long* flags = nullptr;
    unsigned* queue = nullptr;
    size_t tiles = 0;  // Stream-K tiles the slots and flags can hold
    unsigned epoch = 0;
};

template <int BM, int BN, int BK, int STAGES>
int resident_blocks() {
    constexpr int bytes = hgemm_tile::smem_bytes<BM, BN, BK, STAGES>();
    static int resident = 0;  // blocks resident per GPU; also the > 48 KB smem opt-in
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(hgemm_v4_kernel<BM, BN, BK, STAGES>,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, hgemm_v4_kernel<BM, BN, BK, STAGES>, THREADS, bytes));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <int BM, int BN, int BK, int STAGES>
void launch(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N, int K,
            cudaStream_t stream) {
    constexpr int bytes = hgemm_tile::smem_bytes<BM, BN, BK, STAGES>();
    const int resident = resident_blocks<BM, BN, BK, STAGES>();

    Params p;
    p.tiles_m = cdiv(M, BM);
    p.tiles_n = N / BN;
    p.tiles = p.tiles_m * p.tiles_n;
    p.KT = K / BK;
    const long long total = static_cast<long long>(p.tiles) * p.KT;
    const double operands = 2.0 * (static_cast<double>(M) * K + static_cast<double>(K) * N);
    const int static_waves = operands <= kL2Share * l2_bytes() ? kStaticWavesL2 : kStaticWaves;

    int sk_tiles;
    if (p.tiles <= static_cast<long long>(resident) * static_waves) {
        // Static ranges. Never more blocks than there are pieces of kMinShare k-steps (a
        // shorter piece costs more in fixup traffic than it does in mma work), but never
        // fewer than there are tiles: with one block per tile there are no partials at all.
        p.grid = static_cast<int>(
            std::min<long long>(resident, std::max<long long>(p.tiles, total / kMinShare)));
        p.dp_tiles = 0;
        p.passes = 0;
        p.sk_iters = p.tiles * p.KT;
        sk_tiles = p.tiles;
    } else {
        // Queue: whole tiles, then the tail and one full wave in geometric passes.
        p.grid = resident;
        const int tail = p.tiles % p.grid;
        sk_tiles = tail + (tail > 0 ? p.grid : 0);
        p.dp_tiles = p.tiles - sk_tiles;
        p.passes = 1;
        while ((p.KT >> p.passes) > kMinPass) ++p.passes;
        p.sk_iters = 0;
    }

    // Raster group: G tile rows per group, about the square root of the blocks in flight,
    // so a wave touches G rows of A and grid/G columns of B.
    int g = 1;
    while ((g * 2) * (g * 2) <= p.grid) g *= 2;
    p.group = std::max(1, std::min(g, p.tiles_m));

    static Workspace w;
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

    hgemm_v4_kernel<BM, BN, BK, STAGES><<<p.grid, THREADS, bytes, stream>>>(A, B, C, M, N, K, p);
}

constexpr int BK = 32;
constexpr int STAGES = 3;

// Tile selection. The 128x128 tile has the best smem-side intensity and wins whenever each
// block's share of the (tile, k-step) space is long enough to amortize the pipeline and
// the fixup; below that the 64-row tiles. Decode shapes (M <= 64, bound by streaming B) go
// to the same weight-streaming kernel as variant 3 (hgemm_decode.cu): it streams B at the
// single-launch floor, which the 64x64x64 tile on this schedule does not quite reach
// (1,352 against 1,459 GB/s at 64x4096x4096).
void launch_auto(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M, int N,
                 int K, cudaStream_t stream) {
    auto share = [&](int bm, int bn, int bk, int resident) {  // k-steps per block
        return static_cast<long long>(cdiv(M, bm)) * (N / bn) * (K / bk) / resident;
    };
    if (M <= 64 && hgemm_decode::supports(M, N, K)) {
        hgemm_decode::launch(A, B, C, M, N, K, stream);
    } else if (M <= 64) {
        launch<64, 64, 64, 4>(A, B, C, M, N, K, stream);
    } else if (N % 128 == 0 &&
               share(128, 128, BK, resident_blocks<128, 128, BK, STAGES>()) >= kMinShare) {
        launch<128, 128, BK, STAGES>(A, B, C, M, N, K, stream);
    } else if (N % 128 == 0 &&
               share(64, 128, BK, resident_blocks<64, 128, BK, STAGES>()) >= kMinShare) {
        launch<64, 128, BK, STAGES>(A, B, C, M, N, K, stream);
    } else {
        launch<64, 64, 64, 3>(A, B, C, M, N, K, stream);
    }
}

}  // namespace sk

}  // namespace

void hgemm_streamk_bf16(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C, int M,
                        int N, int K, cudaStream_t stream) {
    sk::launch_auto(A, B, C, M, N, K, stream);
}

}  // namespace spark
