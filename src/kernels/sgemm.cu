// fp32 GEMM optimization ladder for Blackwell sm_12x: RTX 5090 (sm_120), GB10 (sm_121).
//
//   C[M,N] = A[M,K] * B[K,N]   (row-major, fp32 in/out, fp32 accumulate)
//
// variant 0: naive, one thread per C element
// variant 1: shared-memory tiled, 32x32x32 tile, 1024 threads, one C element per thread
// variant 2: register-tiled 128x128x8 tile, 256 threads, 8x8 micro-tile per thread,
//            float4 global loads, A tile stored transposed in smem
// variant 3: variant 2 + two-stage cp.async double buffering (A tile stored untransposed
//            with a +4 pad so the strided fragment reads are bank-conflict free)
// variant 4: register-prefetch double buffering (next tile loaded into registers during
//            the FMAs, transposed on the way into smem), 128x128x16 tile
// variant 5: variant 4 with a 256x128 tile and 16x8 micro-tiles: 0.75 instead of 1 byte of
//            shared memory read per FMA, which is the binding limit on sm_120 (128 FMA and
//            128 B of LDS per SM per clock)
// variant 6: tensor cores, TF32: 128x128x16 block tile of mma.sync.m16n8k8 + ldmatrix out of
//            XOR-swizzled shared memory, 3-stage cp.async ring; operands rounded to tf32
//            (cvt.rna) as the fragments are loaded, one mma per product (sgemm_tf32.cu)
// variant 7: the same tile with each operand split into big + small tf32 parts and three mmas
//            per product (3xTF32), which brings the error back to the fp32 class
// Variants 4 to 7 split the tiles of the last partial wave along K (fp32 atomics into C).
//
// All variants accept arbitrary M, N, K >= 1 and guard every global access.
// See docs/design/sgemm.md for the reasoning behind each rung. Tile sizes are GB10-derived
// (48 SMs, 273 GB/s); re-tune on the RTX 5090 (170 SMs, 1,792 GB/s GDDR7).

#include <algorithm>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {
namespace {

// ---------------------------------------------------------------------------
// variant 0: naive
// ---------------------------------------------------------------------------
__global__ void sgemm_naive_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                   float* __restrict__ C, int M, int N, int K) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    const float* a_row = A + static_cast<size_t>(row) * K;
    for (int k = 0; k < K; ++k) {
        acc += a_row[k] * B[static_cast<size_t>(k) * N + col];
    }
    C[static_cast<size_t>(row) * N + col] = acc;
}

// ---------------------------------------------------------------------------
// variant 1: shared-memory tiled, one output per thread
// ---------------------------------------------------------------------------
constexpr int kTile1 = 32;

__global__ void __launch_bounds__(kTile1* kTile1)
    sgemm_smem_kernel(const float* __restrict__ A, const float* __restrict__ B,
                      float* __restrict__ C, int M, int N, int K) {
    __shared__ float As[kTile1][kTile1];
    __shared__ float Bs[kTile1][kTile1];

    const int tx = threadIdx.x;  // 0..31, column within tile
    const int ty = threadIdx.y;  // 0..31, row within tile
    const int row = blockIdx.y * kTile1 + ty;
    const int col = blockIdx.x * kTile1 + tx;

    float acc = 0.0f;
    const int num_tiles = cdiv(K, kTile1);
    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * kTile1 + tx;  // k index for the A element this thread loads
        const int b_row = t * kTile1 + ty;  // k index for the B element this thread loads
        As[ty][tx] = (row < M && a_col < K) ? A[static_cast<size_t>(row) * K + a_col] : 0.0f;
        Bs[ty][tx] = (b_row < K && col < N) ? B[static_cast<size_t>(b_row) * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < kTile1; ++k) {
            acc += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();
    }
    if (row < M && col < N) {
        C[static_cast<size_t>(row) * N + col] = acc;
    }
}

// ---------------------------------------------------------------------------
// Shared helpers for the register-tiled variants
// ---------------------------------------------------------------------------

// Guarded 4-wide load of A[row, k0..k0+3] (row-major, leading dim K).
// Uses a single float4 when the chunk is fully in-bounds and 16-byte aligned.
__device__ __forceinline__ void load_a_chunk(const float* __restrict__ A, int M, int K, int row,
                                             int k0, float (&v)[4]) {
    if (row < M && k0 + 3 < K && (K & 3) == 0) {
        const float4 t = *reinterpret_cast<const float4*>(A + static_cast<size_t>(row) * K + k0);
        v[0] = t.x;
        v[1] = t.y;
        v[2] = t.z;
        v[3] = t.w;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int k = k0 + i;
            v[i] = (row < M && k < K) ? A[static_cast<size_t>(row) * K + k] : 0.0f;
        }
    }
}

// Guarded 4-wide load of B[k, col0..col0+3] (row-major, leading dim N).
__device__ __forceinline__ void load_b_chunk(const float* __restrict__ B, int K, int N, int k,
                                             int col0, float (&v)[4]) {
    if (k < K && col0 + 3 < N && (N & 3) == 0) {
        const float4 t = *reinterpret_cast<const float4*>(B + static_cast<size_t>(k) * N + col0);
        v[0] = t.x;
        v[1] = t.y;
        v[2] = t.z;
        v[3] = t.w;
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = col0 + i;
            v[i] = (k < K && c < N) ? B[static_cast<size_t>(k) * N + c] : 0.0f;
        }
    }
}

// Guarded store of 4 consecutive C elements C[row, col0..col0+3].
__device__ __forceinline__ void store_c_chunk(float* __restrict__ C, int M, int N, int row,
                                              int col0, const float (&v)[4]) {
    if (row >= M) return;
    if (col0 + 3 < N && (N & 3) == 0) {
        *reinterpret_cast<float4*>(C + static_cast<size_t>(row) * N + col0) =
            make_float4(v[0], v[1], v[2], v[3]);
    } else {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int c = col0 + i;
            if (c < N) C[static_cast<size_t>(row) * N + c] = v[i];
        }
    }
}

// Thread -> micro-tile mapping used by variants 2-5.
// 256 threads form a (BM/TM) x (BN/TN) grid of TM x TN micro-tiles over a BM x BN block tile.
// A thread's rows come in groups of 4: {q * kQuarterM + ty*4 + i}, q = 0..TM/4-1, i = 0..3
// (for the 128x128 / 8x8 case: {ty*4 + i} and {64 + ty*4 + i}); columns likewise in two
// groups of 4. The split layout makes a warp's float4 reads of the A and B tiles contiguous
// (conflict-free) and turns the C stores into float4 stores.
template <int BM, int BN, int TM, int TN>
struct MicroTile {
    static_assert(TM % 4 == 0 && TN == 8, "layout below assumes TM x 8 micro-tiles, TM % 4 == 0");
    static constexpr int kThreadsY = BM / TM;               // 16
    static constexpr int kThreadsX = BN / TN;               // 16
    static constexpr int kThreads = kThreadsY * kThreadsX;  // 256
    static constexpr int kQuarterM = BM / (TM / 4);         // row-group stride (128x128 / 8x8: 64)
    static constexpr int kHalfN = BN / 2;

    __device__ static __forceinline__ int row_of(int ty, int i) {
        return (i / 4) * kQuarterM + ty * 4 + (i % 4);
    }
    __device__ static __forceinline__ int col_of(int tx, int j) {
        return (j < 4) ? (tx * 4 + j) : (kHalfN + tx * 4 + (j - 4));
    }
};

// ---------------------------------------------------------------------------
// variant 2: register-tiled, A stored transposed in smem
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__(MicroTile<BM, BN, TM, TN>::kThreads)
    sgemm_regtile_kernel(const float* __restrict__ A, const float* __restrict__ B,
                         float* __restrict__ C, int M, int N, int K) {
    using MT = MicroTile<BM, BN, TM, TN>;
    static_assert(BM * BK == MT::kThreads * 4, "A tile must be exactly one float4 per thread");
    static_assert(BK * BN == MT::kThreads * 4, "B tile must be exactly one float4 per thread");

    __shared__ __align__(16) float As[BK][BM];  // transposed: As[k][m]
    __shared__ __align__(16) float Bs[BK][BN];  // natural:    Bs[k][n]

    const int tid = threadIdx.x;
    const int ty = tid / MT::kThreadsX;
    const int tx = tid % MT::kThreadsX;
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    // Global -> smem load assignment: one 4-wide chunk of A and one of B per thread.
    constexpr int kAChunksPerRow = BK / 4;   // 2
    constexpr int kBChunksPerRow = BN / 4;   // 32
    const int a_row = tid / kAChunksPerRow;  // 0..127
    const int a_k0 = (tid % kAChunksPerRow) * 4;
    const int b_k = tid / kBChunksPerRow;  // 0..7
    const int b_col0 = (tid % kBChunksPerRow) * 4;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    const int num_tiles = cdiv(K, BK);
    for (int t = 0; t < num_tiles; ++t) {
        const int k_base = t * BK;

        float av[4];
        float bv[4];
        load_a_chunk(A, M, K, bm + a_row, k_base + a_k0, av);
        load_b_chunk(B, K, N, k_base + b_k, bn + b_col0, bv);

        // A goes in transposed so that the k-loop below reads contiguous float4s of m.
#pragma unroll
        for (int i = 0; i < 4; ++i) As[a_k0 + i][a_row] = av[i];
        *reinterpret_cast<float4*>(&Bs[b_k][b_col0]) = make_float4(bv[0], bv[1], bv[2], bv[3]);
        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM];
            float b[TN];
            const float4 a0 = *reinterpret_cast<const float4*>(&As[k][ty * 4]);
            const float4 a1 = *reinterpret_cast<const float4*>(&As[k][MT::kQuarterM + ty * 4]);
            a[0] = a0.x;
            a[1] = a0.y;
            a[2] = a0.z;
            a[3] = a0.w;
            a[4] = a1.x;
            a[5] = a1.y;
            a[6] = a1.z;
            a[7] = a1.w;
            const float4 b0 = *reinterpret_cast<const float4*>(&Bs[k][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4*>(&Bs[k][MT::kHalfN + tx * 4]);
            b[0] = b0.x;
            b[1] = b0.y;
            b[2] = b0.z;
            b[3] = b0.w;
            b[4] = b1.x;
            b[5] = b1.y;
            b[6] = b1.z;
            b[7] = b1.w;
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }

    // Epilogue: two float4 stores per micro-tile row.
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = bm + MT::row_of(ty, i);
        float lo[4] = {acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
        float hi[4] = {acc[i][4], acc[i][5], acc[i][6], acc[i][7]};
        store_c_chunk(C, M, N, row, bn + MT::col_of(tx, 0), lo);
        store_c_chunk(C, M, N, row, bn + MT::col_of(tx, 4), hi);
    }
}

// ---------------------------------------------------------------------------
// variant 3: register-tiled + cp.async double buffering
//
// cp.async copies global -> shared without passing through registers, so it cannot
// transpose. The A tile is therefore kept untransposed (As[m][k]) and padded to
// BK + 4 floats per row: 12-float (48 B) row stride keeps every chunk 16-byte aligned
// and makes the 8 strided fragment reads per k hit 8 distinct banks.
// ---------------------------------------------------------------------------
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__(MicroTile<BM, BN, TM, TN>::kThreads)
    sgemm_regtile_cpasync_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ C, int M, int N, int K) {
    using MT = MicroTile<BM, BN, TM, TN>;
    static_assert(BM * BK == MT::kThreads * 4, "A tile must be exactly one float4 per thread");
    static_assert(BK * BN == MT::kThreads * 4, "B tile must be exactly one float4 per thread");
    constexpr int kStages = 2;
    constexpr int kAPad = BK + 4;  // 12 floats = 48 bytes per row

    __shared__ __align__(16) float As[kStages][BM][kAPad];  // natural: As[s][m][k]
    __shared__ __align__(16) float Bs[kStages][BK][BN];     // natural: Bs[s][k][n]

    const int tid = threadIdx.x;
    const int ty = tid / MT::kThreadsX;
    const int tx = tid % MT::kThreadsX;
    const int bm = blockIdx.y * BM;
    const int bn = blockIdx.x * BN;

    constexpr int kAChunksPerRow = BK / 4;  // 2
    constexpr int kBChunksPerRow = BN / 4;  // 32
    const int a_row = tid / kAChunksPerRow;
    const int a_k0 = (tid % kAChunksPerRow) * 4;
    const int b_k = tid / kBChunksPerRow;
    const int b_col0 = (tid % kBChunksPerRow) * 4;

    const int num_tiles = cdiv(K, BK);

    // Issue the global->shared copy of tile `t` into stage `s`.
    // Fully in-bounds, 16-byte aligned chunks use cp.async; edge chunks use guarded
    // scalar loads through registers (zero-filled).
    auto issue_load = [&](int t, int s) {
        const int k_base = t * BK;
        // ---- A chunk: row (bm + a_row), k in [k_base + a_k0, +4)
        {
            const int grow = bm + a_row;
            const int gk = k_base + a_k0;
            float* dst = &As[s][a_row][a_k0];
            if (grow < M && gk + 3 < K && (K & 3) == 0) {
                cp_async_16(dst, A + static_cast<size_t>(grow) * K + gk);
            } else {
                float v[4];
                load_a_chunk(A, M, K, grow, gk, v);
#pragma unroll
                for (int i = 0; i < 4; ++i) dst[i] = v[i];
            }
        }
        // ---- B chunk: k = k_base + b_k, cols [bn + b_col0, +4)
        {
            const int gk = k_base + b_k;
            const int gcol = bn + b_col0;
            float* dst = &Bs[s][b_k][b_col0];
            if (gk < K && gcol + 3 < N && (N & 3) == 0) {
                cp_async_16(dst, B + static_cast<size_t>(gk) * N + gcol);
            } else {
                float v[4];
                load_b_chunk(B, K, N, gk, gcol, v);
#pragma unroll
                for (int i = 0; i < 4; ++i) dst[i] = v[i];
            }
        }
    };

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    // Prologue: stage 0 holds tile 0.
    issue_load(0, 0);
    cp_async_commit();

    for (int t = 0; t < num_tiles; ++t) {
        const int cur = t & 1;
        if (t + 1 < num_tiles) {
            issue_load(t + 1, cur ^ 1);
            cp_async_commit();
            cp_async_wait<1>();  // tile t has landed (for this thread); tile t+1 may be in flight
        } else {
            cp_async_wait<0>();
        }
        __syncthreads();  // make every thread's copies of tile t visible

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM];
            float b[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) a[i] = As[cur][MT::row_of(ty, i)][k];
            const float4 b0 = *reinterpret_cast<const float4*>(&Bs[cur][k][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4*>(&Bs[cur][k][MT::kHalfN + tx * 4]);
            b[0] = b0.x;
            b[1] = b0.y;
            b[2] = b0.z;
            b[3] = b0.w;
            b[4] = b1.x;
            b[5] = b1.y;
            b[6] = b1.z;
            b[7] = b1.w;
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();  // everyone is done reading stage `cur` before it is refilled
    }

#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = bm + MT::row_of(ty, i);
        float lo[4] = {acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
        float hi[4] = {acc[i][4], acc[i][5], acc[i][6], acc[i][7]};
        store_c_chunk(C, M, N, row, bn + MT::col_of(tx, 0), lo);
        store_c_chunk(C, M, N, row, bn + MT::col_of(tx, 4), hi);
    }
}

// ---------------------------------------------------------------------------
// variant 4: register-prefetch double buffering, A transposed in smem, BK = 16
//
// cp.async (variant 3) cannot transpose, which forces the strided scalar reads of A in the
// inner loop. Prefetching the next tile into registers instead (issued before the current
// tile's FMAs, stored to smem after them) keeps the global loads in flight across the
// compute just the same, and the register hop is exactly where the transpose happens for
// free. The result is variant 2's inner loop (4 x LDS.128 per k per thread) with the
// latency hiding of variant 3, and BK = 16 halves the barrier count per FLOP.
// ---------------------------------------------------------------------------
// XOR swizzle of the m index of the transposed A tile, keyed by k / 4. Without it the four
// k-chunks of one row that neighbouring threads store land in the same bank (a row of As is
// BM = 128 floats = 4 x 32 banks) and every transposed store is a 4-way conflict. The XOR
// operand is a multiple of 8, so the float4 reads in the inner loop stay contiguous and
// aligned; k is a compile-time constant there and the XOR folds into the address.
__device__ __forceinline__ int swz_a(int k, int m) {
    return m ^ ((k >> 2) << 3);
}

// Work assignment for the prefetch kernels (see `launch_prefetch`): blocks [0, dp_tiles) own
// one whole output tile each; the blocks after that split the remaining tiles `split` ways
// along K and add their partial sums straight into C (zeroed beforehand) with fp32 atomics.
struct Sched {
    int tiles_n;   // tiles along N (tile = tm * tiles_n + tn)
    int dp_tiles;  // tiles handled whole
    int split;     // K-slices per tail tile (1: no tail)
};

// kMinBlocks caps registers so that many blocks fit per SM: for the 128x128 / 8x8 tile the
// natural 153 registers would leave one 8-warp block per SM and the issue slot idle a third
// of the time, and 128 costs nothing; the 256x128 / 16x8 tile (variant 5) needs ~245.
template <int BM, int BN, int BK, int TM, int TN, int kMinBlocks>
__global__ void __launch_bounds__(MicroTile<BM, BN, TM, TN>::kThreads, kMinBlocks)
    sgemm_regtile_prefetch_kernel(const float* __restrict__ A, const float* __restrict__ B,
                                  float* __restrict__ C, int M, int N, int K, Sched sched) {
    using MT = MicroTile<BM, BN, TM, TN>;
    constexpr int kAChunks = (BM * BK / 4) / MT::kThreads;  // float4 chunks of A per thread
    constexpr int kBChunks = (BK * BN / 4) / MT::kThreads;
    static_assert(kAChunks * MT::kThreads * 4 == BM * BK, "A tile must divide evenly");
    static_assert(kBChunks * MT::kThreads * 4 == BK * BN, "B tile must divide evenly");
    constexpr int kAChunksPerRow = BK / 4;
    constexpr int kBChunksPerRow = BN / 4;

    __shared__ __align__(16) float As[BK][BM];  // transposed: As[k][m]
    __shared__ __align__(16) float Bs[BK][BN];  // natural:    Bs[k][n]

    const int tid = threadIdx.x;
    const int ty = tid / MT::kThreadsX;
    const int tx = tid % MT::kThreadsX;

    const int KT = cdiv(K, BK);
    int tile, t_begin, t_end;
    if (static_cast<int>(blockIdx.x) < sched.dp_tiles) {
        tile = blockIdx.x;
        t_begin = 0;
        t_end = KT;
    } else {
        const int r = blockIdx.x - sched.dp_tiles;
        tile = sched.dp_tiles + r / sched.split;
        const int slice = r % sched.split;
        t_begin = static_cast<int>(static_cast<long long>(slice) * KT / sched.split);
        t_end = static_cast<int>(static_cast<long long>(slice + 1) * KT / sched.split);
    }
    const int bm = (tile / sched.tiles_n) * BM;
    const int bn = (tile % sched.tiles_n) * BN;

    float pa[kAChunks][4];
    float pb[kBChunks][4];
    auto gload = [&](int t) {
        const int k_base = t * BK;
#pragma unroll
        for (int i = 0; i < kAChunks; ++i) {
            const int c = tid + i * MT::kThreads;
            load_a_chunk(A, M, K, bm + c / kAChunksPerRow, k_base + (c % kAChunksPerRow) * 4,
                         pa[i]);
        }
#pragma unroll
        for (int i = 0; i < kBChunks; ++i) {
            const int c = tid + i * MT::kThreads;
            load_b_chunk(B, K, N, k_base + c / kBChunksPerRow, bn + (c % kBChunksPerRow) * 4,
                         pb[i]);
        }
    };
    auto sstore = [&]() {
#pragma unroll
        for (int i = 0; i < kAChunks; ++i) {
            const int c = tid + i * MT::kThreads;
            const int row = c / kAChunksPerRow;
            const int k0 = (c % kAChunksPerRow) * 4;
#pragma unroll
            for (int e = 0; e < 4; ++e) As[k0 + e][swz_a(k0 + e, row)] = pa[i][e];
        }
#pragma unroll
        for (int i = 0; i < kBChunks; ++i) {
            const int c = tid + i * MT::kThreads;
            *reinterpret_cast<float4*>(&Bs[c / kBChunksPerRow][(c % kBChunksPerRow) * 4]) =
                make_float4(pb[i][0], pb[i][1], pb[i][2], pb[i][3]);
        }
    };

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.0f;

    gload(t_begin);
    sstore();
    __syncthreads();

    for (int t = t_begin; t < t_end; ++t) {
        const bool more = t + 1 < t_end;
        if (more) gload(t + 1);  // in flight while the FMAs below run

#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float a[TM];
            float b[TN];
#pragma unroll
            for (int q = 0; q < TM / 4; ++q) {
                const float4 aq =
                    *reinterpret_cast<const float4*>(&As[k][swz_a(k, q * MT::kQuarterM + ty * 4)]);
                a[4 * q] = aq.x;
                a[4 * q + 1] = aq.y;
                a[4 * q + 2] = aq.z;
                a[4 * q + 3] = aq.w;
            }
            const float4 b0 = *reinterpret_cast<const float4*>(&Bs[k][tx * 4]);
            const float4 b1 = *reinterpret_cast<const float4*>(&Bs[k][MT::kHalfN + tx * 4]);
            b[0] = b0.x;
            b[1] = b0.y;
            b[2] = b0.z;
            b[3] = b0.w;
            b[4] = b1.x;
            b[5] = b1.y;
            b[6] = b1.z;
            b[7] = b1.w;
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();  // everyone is done reading the tile before it is overwritten
        if (more) {
            sstore();
            __syncthreads();
        }
    }

    if (tile < sched.dp_tiles) {
#pragma unroll
        for (int i = 0; i < TM; ++i) {
            const int row = bm + MT::row_of(ty, i);
            float lo[4] = {acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
            float hi[4] = {acc[i][4], acc[i][5], acc[i][6], acc[i][7]};
            store_c_chunk(C, M, N, row, bn + MT::col_of(tx, 0), lo);
            store_c_chunk(C, M, N, row, bn + MT::col_of(tx, 4), hi);
        }
        return;
    }
    // Tail tile: this block holds one K-slice of the sum; the tile of C was zeroed on the
    // stream before the launch and every slice adds into it (the adds happen at L2).
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int row = bm + MT::row_of(ty, i);
        if (row >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int col = bn + MT::col_of(tx, j);
            if (col < N) atomicAdd(C + static_cast<size_t>(row) * N + col, acc[i][j]);
        }
    }
}

// Wave quantization: with P blocks resident at once, tiles past the last full wave would
// run alone for a whole wave (4096^3 with 256x128 tiles: 512 tiles on 170 SMs, 3.01 waves
// in the time of 4). Those tiles are split `split` ways along K over the idle blocks
// instead (split-K on the tail only), so the extra wave lasts 1/split of a tile.
template <int BM, int BN, int BK, int TM, int TN, int kMinBlocks>
void launch_prefetch(const float* A, const float* B, float* C, int M, int N, int K,
                     cudaStream_t stream) {
    using MT = MicroTile<BM, BN, TM, TN>;
    static int resident = 0;
    if (resident == 0) {
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, sgemm_regtile_prefetch_kernel<BM, BN, BK, TM, TN, kMinBlocks>, MT::kThreads,
            0));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    Sched s;
    s.tiles_n = cdiv(N, BN);
    const int tiles = cdiv(M, BM) * s.tiles_n;
    const int tail = tiles % resident;
    s.split = tail > 0 ? std::min(cdiv(K, BK), resident / tail) : 1;
    if (s.split <= 1) {
        s.dp_tiles = tiles;
        s.split = 1;
    } else {
        s.dp_tiles = tiles - tail;
        for (int t = s.dp_tiles; t < tiles; ++t) {  // zero the tail tiles of C
            const int bm = (t / s.tiles_n) * BM;
            const int bn = (t % s.tiles_n) * BN;
            SPARK_CUDA_CHECK(cudaMemset2DAsync(
                C + static_cast<size_t>(bm) * N + bn, static_cast<size_t>(N) * sizeof(float), 0,
                static_cast<size_t>(std::min(BN, N - bn)) * sizeof(float), std::min(BM, M - bm),
                stream));
        }
    }
    const int grid = s.dp_tiles + (tiles - s.dp_tiles) * s.split;
    sgemm_regtile_prefetch_kernel<BM, BN, BK, TM, TN, kMinBlocks>
        <<<grid, MT::kThreads, 0, stream>>>(A, B, C, M, N, K, s);
}

}  // namespace

// Variants 6 and 7 (sgemm_tf32.cu): the tensor-core tile, one or three tf32 passes.
void sgemm_tf32(const float* A, const float* B, float* C, int M, int N, int K, bool three_pass,
                cudaStream_t stream);

// ---------------------------------------------------------------------------
// Host entry point
// ---------------------------------------------------------------------------
int sgemm_num_variants() {
    return 8;
}

int sgemm_default_variant() {
    return 5;
}

void sgemm(const float* A, const float* B, float* C, int M, int N, int K, int variant,
           cudaStream_t stream) {
    SPARK_REQUIRE(A != nullptr && B != nullptr && C != nullptr, "sgemm: null pointer");
    SPARK_REQUIRE(M >= 1 && N >= 1 && K >= 1, "sgemm: M, N, K must be >= 1");
    SPARK_REQUIRE(variant >= 0 && variant < sgemm_num_variants(), "sgemm: bad variant");

    switch (variant) {
        case 0: {
            const dim3 block(16, 16);
            const dim3 grid(cdiv(N, 16), cdiv(M, 16));
            sgemm_naive_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            break;
        }
        case 1: {
            const dim3 block(kTile1, kTile1);
            const dim3 grid(cdiv(N, kTile1), cdiv(M, kTile1));
            sgemm_smem_kernel<<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            break;
        }
        case 2: {
            constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
            const dim3 block(MicroTile<BM, BN, TM, TN>::kThreads);
            const dim3 grid(cdiv(N, BN), cdiv(M, BM));
            sgemm_regtile_kernel<BM, BN, BK, TM, TN><<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            break;
        }
        case 3: {
            constexpr int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;
            const dim3 block(MicroTile<BM, BN, TM, TN>::kThreads);
            const dim3 grid(cdiv(N, BN), cdiv(M, BM));
            sgemm_regtile_cpasync_kernel<BM, BN, BK, TM, TN>
                <<<grid, block, 0, stream>>>(A, B, C, M, N, K);
            break;
        }
        case 4:
            launch_prefetch<128, 128, 16, 8, 8, 2>(A, B, C, M, N, K, stream);
            break;
        case 5:
            launch_prefetch<256, 128, 16, 16, 8, 1>(A, B, C, M, N, K, stream);
            break;
        case 6:
            sgemm_tf32(A, B, C, M, N, K, /*three_pass=*/false, stream);
            break;
        case 7:
            sgemm_tf32(A, B, C, M, N, K, /*three_pass=*/true, stream);
            break;
        default:
            break;
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
