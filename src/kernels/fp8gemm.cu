// fp8 (e4m3) tensor-core GEMM ladder for Blackwell sm_12x: RTX 5090 (sm_120), GB10 (sm_121).
//
//   C[M,N] = scale_a * scale_b * A[M,K] * Bt[N,K]^T
//
// A and Bt are e4m3 with K contiguous (Bt is B transposed, the "TN" layout cuBLASLt requires
// for fp8), the scales are per-tensor fp32 on the device, C is bf16, accumulation is fp32 on
// mma.sync.m16n8k32. Both operands K-contiguous is what makes the fragments fall out of a plain
// ldmatrix for A and B alike (fp8gemm_tile.cuh, docs/design/fp8gemm.md).
//
//   variant 0: one warp per 16x8 C tile, fragments loaded from global memory as 4-byte words.
//   variant 1: 128x128x64 block tile, mma.sync + ldmatrix out of XOR-swizzled smem, 3-stage
//              cp.async pipeline, register epilogue with the scale, split-K on the last
//              partial wave, smaller tiles for small grids, one CTA per 16/32/64-row strip
//              of Bt for decode shapes (M <= 64).
//   variant 2: variant 1's 128x128 tile fed by TMA through a warp-specialized mbarrier
//              pipeline (fp8gemm_tma.cu).
//
// MX mode (fp8gemm_mx): the same three variants with a ue8m0 scale per row per 32 k on both
// operands, sfa[M][K/32] and sfb[N][K/32], applied by the block-scaled mma itself
// (fp8gemm_tile.cuh, MxFrag / MxChunk). Requires K % 256 == 0 so that a row's scales for
// two 128-k stages are one aligned 8-byte chunk.

#include <cuda_fp8.h>

#include <algorithm>
#include <cstdlib>

#include "fp8gemm_tile.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

// Variant 2 lives in fp8gemm_tma.cu.
bool fp8gemm_tma_supports(int M, int N, int K);
void fp8gemm_tma(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* Bt, __nv_bfloat16* C, int M, int N,
                 int K, const float* scale_a, const float* scale_b, fp8gemm_tile::MxScales mx,
                 cudaStream_t stream);

namespace {

using fp8gemm_tile::Cfg;
using fp8gemm_tile::MxScales;

// ---------------------------------------------------------------------------------------
// Variant 0. Block = 4 warps covering 16 rows x 32 columns; each warp walks K in steps of 32
// loading its fragment words straight from global memory: lane (g, t) needs k 4t..4t+3 of row
// g (and g+8, and +16), which is one aligned 4-byte load per register. Host guarantees
// M % 16 == 0, N % 32 == 0, K % 32 == 0.
// ---------------------------------------------------------------------------------------
constexpr int V0_THREADS = 128;
constexpr int V0_ROWS = 16, V0_COLS = 32;

// In MX mode lane (g, t) also reads, per k32 step, the scale byte of row g + 8 (t & 1) and of
// column g for that k-block, straight from the scale tensors (the lanes the instruction reads
// with thread-id 0: fp8gemm_tile.cuh). This is the reference the smem variants are checked
// against: nothing about the fragment or word layout is assumed beyond the probe.
template <bool MX>
__global__ void __launch_bounds__(V0_THREADS)
    fp8gemm_v0_kernel(const unsigned char* __restrict__ A, const unsigned char* __restrict__ Bt,
                      __nv_bfloat16* __restrict__ C, int M, int N, int K,
                      const float* __restrict__ scale_a, const float* __restrict__ scale_b,
                      MxScales mx) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int row = blockIdx.y * V0_ROWS;
    const int col = blockIdx.x * V0_COLS + warp * 8;
    (void)M;

    const unsigned char* a0 = A + static_cast<size_t>(row + g) * K + 4 * t;
    const unsigned char* a1 = a0 + static_cast<size_t>(8) * K;
    const unsigned char* b0 = Bt + static_cast<size_t>(col + g) * K + 4 * t;
    const unsigned char* sa = mx.a + static_cast<size_t>(row + g + 8 * (t & 1)) * mx.ld;
    const unsigned char* sb = mx.b + static_cast<size_t>(col + g) * mx.ld;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < K; k0 += 32) {
        unsigned a[4], b[2];
        a[0] = *reinterpret_cast<const unsigned*>(a0 + k0);
        a[1] = *reinterpret_cast<const unsigned*>(a1 + k0);
        a[2] = *reinterpret_cast<const unsigned*>(a0 + k0 + 16);
        a[3] = *reinterpret_cast<const unsigned*>(a1 + k0 + 16);
        b[0] = *reinterpret_cast<const unsigned*>(b0 + k0);
        b[1] = *reinterpret_cast<const unsigned*>(b0 + k0 + 16);
        if constexpr (MX)
            mma_e4m3_16832_mx<0>(acc, a, b, sa[k0 >> 5], sb[k0 >> 5]);
        else
            mma_e4m3_16832(acc, a, b);
    }
    const float s = *scale_a * *scale_b;
    __nv_bfloat16* c0 = C + static_cast<size_t>(row + g) * N + col + 2 * t;
    *reinterpret_cast<__nv_bfloat162*>(c0) = __floats2bfloat162_rn(acc[0] * s, acc[1] * s);
    *reinterpret_cast<__nv_bfloat162*>(c0 + static_cast<size_t>(8) * N) =
        __floats2bfloat162_rn(acc[2] * s, acc[3] * s);
}

// ---------------------------------------------------------------------------------------
// Variant 1: the bf16 variant 3 design on 8-bit operands. The tile (swizzle, pipeline, mma
// loop, register epilogue) is fp8gemm_tile.cuh; this file hands out (tile, K-range) work.
// ---------------------------------------------------------------------------------------
namespace v1 {

// Work assignment, as hgemm variant 3's: blocks [0, dp_tiles) each own one full output tile;
// blocks from dp_tiles on split the remaining tiles `split` ways along K, accumulate their
// K-slice into the fp32 workspace `ws` with atomics, and the last slice to finish a tile
// applies the scale and converts it to bf16 (`counters` is one arrival counter per tail tile,
// zeroed with `ws` before the launch).
struct Sched {
    int tiles_n;   // tiles along N (tile index = tm * tiles_n + tn)
    int dp_tiles;  // tiles handled whole
    int split;     // K-slices per tail tile (1: no tail)
    float* ws;     // (tiles - dp_tiles) x BM x BN fp32
    int* counters;
};

template <class C, bool MX, int SFW>
__global__ void __launch_bounds__(C::THREADS)
    fp8gemm_v1_kernel(const unsigned char* __restrict__ A, const unsigned char* __restrict__ Bt,
                      __nv_bfloat16* __restrict__ Cout, int M, int N, int K,
                      const float* __restrict__ scale_a, const float* __restrict__ scale_b,
                      MxScales mx, Sched sched) {
    constexpr int BM = C::BM, BN = C::BN, BK = C::BK, THREADS = C::THREADS;
    constexpr int WROWS = C::WROWS, WCOLS = C::WCOLS, MT = C::MT, NT = C::NT;
    extern __shared__ __align__(128) unsigned char smem_raw[];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / C::WN;
    const int wn = warp % C::WN;

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
    if constexpr (MX) {
        // Slices start on a scale-chunk boundary (the host caps split at KT / SPC, so none
        // is empty).
        constexpr int SPC = fp8gemm_tile::MxChunk<C, SFW>::SPC;
        kt_begin = kt_begin / SPC * SPC;
        if (slice + 1 < sched.split) kt_end = kt_end / SPC * SPC;
    }
    const int bm = (tile / sched.tiles_n) * BM;
    const int bn = (tile % sched.tiles_n) * BN;
    const int m_valid = M - bm;  // rows of this tile that exist
    const float scale = *scale_a * *scale_b;

    typename C::Acc acc;
    fp8gemm_tile::mainloop<C, MX, SFW>(A, Bt, K, bm, bn, m_valid, kt_begin, kt_end - kt_begin,
                                       smem_raw, acc, mx);

    if (tile < sched.dp_tiles) {
        fp8gemm_tile::store_bf16<C>(acc, scale, Cout, M, N, bm, bn);
        return;
    }

    // Tail tile: accumulate this K-slice into the fp32 workspace tile. The adds are performed
    // at L2, so no ordering between the slices is needed; the scale waits for the sum.
    const int g = lane >> 2;
    const int c2 = (lane & 3) * 2;
    float* wt = sched.ws + static_cast<size_t>(tile - sched.dp_tiles) * BM * BN;
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int nj = 0; nj < NT; ++nj) {
            const int r0 = wm * WROWS + mi * 16 + g;
            const int c0 = wn * WCOLS + nj * 8 + c2;
            if (r0 >= m_valid) continue;
            atomicAdd(wt + r0 * BN + c0, acc[mi][nj][0]);
            atomicAdd(wt + r0 * BN + c0 + 1, acc[mi][nj][1]);
            if (r0 + 8 >= m_valid) continue;
            atomicAdd(wt + (r0 + 8) * BN + c0, acc[mi][nj][2]);
            atomicAdd(wt + (r0 + 8) * BN + c0 + 1, acc[mi][nj][3]);
        }
    }
    // Last slice to arrive converts the finished tile. The fence orders every thread's adds
    // before the counter; the smem flag broadcasts the outcome to the block.
    __shared__ int s_last;
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        s_last = atomicAdd(sched.counters + (tile - sched.dp_tiles), 1) == sched.split - 1;
    }
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    const int rows_here = m_valid < BM ? m_valid : BM;
    for (int i = tid; i < rows_here * BN / 2; i += THREADS) {
        const int r = (2 * i) / BN;
        const int c = (2 * i) % BN;
        const float2 v = __ldcg(reinterpret_cast<const float2*>(wt + r * BN + c));  // L2, not L1
        *reinterpret_cast<__nv_bfloat162*>(Cout + static_cast<size_t>(bm + r) * N + bn + c) =
            __floats2bfloat162_rn(v.x * scale, v.y * scale);
    }
}

// The split-K workspace: one per configuration and device, grown on demand, zeroed with
// cudaMemsetAsync on the stream before a launch that needs it.
struct Workspace {
    float* ws = nullptr;
    int* counters = nullptr;
    size_t tiles = 0;
};

// Resident blocks per GPU for one configuration (and mode: the MX kernel holds two stages of
// scale words in registers and may fit fewer blocks); also the > 48 KB smem opt-in.
template <class C, bool MX, int SFW>
int resident_blocks() {
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(
            fp8gemm_v1_kernel<C, MX, SFW>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, fp8gemm_v1_kernel<C, MX, SFW>, C::THREADS, C::SMEM));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

// The scale chunk width of MX mode for this K (fp8gemm_tile::MxChunk): 16-byte chunks when
// every row stride is 16-byte aligned, else 8. Per-tensor mode is the SFW = 2 instantiation,
// whose code does not depend on it.
constexpr int sfw_for(int K) {
    return K % 512 == 0 ? 4 : 2;
}
// Split-K slices of MX mode are whole chunk groups: at most KT / SPC of them.
template <class C, bool MX, int SFW>
constexpr int max_split(int KT) {
    if constexpr (MX) return std::max(1, KT / fp8gemm_tile::MxChunk<C, SFW>::SPC);
    return KT;
}

// `split` K-slices for the `tail` last tiles in row-major tile order (whole tiles first).
// split == 1 is a plain launch of `tiles` blocks with no workspace.
template <class C, bool MX, int SFW>
void launch(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M, int N,
            int K, const float* scale_a, const float* scale_b, MxScales mx, int tail, int split,
            cudaStream_t stream) {
    constexpr int BM = C::BM, BN = C::BN;
    (void)resident_blocks<C, MX, SFW>();  // the smem opt-in, if the caller has not done it
    Sched s;
    s.tiles_n = N / BN;
    const int tiles = cdiv(M, BM) * s.tiles_n;
    if (split <= 1 || tail <= 0) {
        s.dp_tiles = tiles;
        s.split = 1;
        s.ws = nullptr;
        s.counters = nullptr;
        fp8gemm_v1_kernel<C, MX, SFW>
            <<<tiles, C::THREADS, C::SMEM, stream>>>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, s);
        return;
    }
    s.dp_tiles = tiles - tail;
    s.split = split;
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
    fp8gemm_v1_kernel<C, MX, SFW>
        <<<grid, C::THREADS, C::SMEM, stream>>>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, s);
}

// Wave quantization (hgemm variant 3's rule): the tiles past the last full wave of resident
// blocks are split along K over the otherwise idle blocks, `split = min(KT, resident / tail)`.
template <class C, bool MX, int SFW>
void launch_tiles_sfw(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M,
                      int N, int K, const float* scale_a, const float* scale_b, MxScales mx,
                      cudaStream_t stream) {
    const int resident = resident_blocks<C, MX, SFW>();
    const int tiles = cdiv(M, C::BM) * (N / C::BN);
    const int KT = K / C::BK;
    const int tail = tiles % resident;
    const int split = tail > 0 ? std::min(max_split<C, MX, SFW>(KT), resident / tail) : 1;
    launch<C, MX, SFW>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, tail, split, stream);
}

// MX mode picks the chunk width from K; per-tensor mode has one instantiation.
template <class C, bool MX>
void launch_tiles(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M,
                  int N, int K, const float* scale_a, const float* scale_b, MxScales mx,
                  cudaStream_t stream) {
    if (MX && sfw_for(K) == 4)
        launch_tiles_sfw<C, MX, 4>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
    else
        launch_tiles_sfw<C, MX, 2>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
}

// Decode shapes (M <= 64): one CTA per BN-row strip of Bt, no split-K unless a narrow N
// leaves fewer than kMinCtas strips (measured for the bf16 decode kernel: 64 CTAs with 24 KB
// of B in flight each stream the weights at the copy roof, and every split-K launch pays a
// flat 2 us for the reduction chain). The first of two configurations runs if its strips fit
// in one wave of resident CTAs, otherwise the second, which has more slots.
constexpr int kMinCtas = 64;

template <class C, bool MX>
bool fits(int N) {
    return N % C::BN == 0 && N / C::BN <= resident_blocks<C, MX, 2>();
}

template <class C, bool MX, int SFW>
void launch_decode_sfw(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M,
                       int N, int K, const float* scale_a, const float* scale_b, MxScales mx,
                       cudaStream_t stream) {
    const int strips = N / C::BN;
    const int KT = K / C::BK;
    const int split = std::min(max_split<C, MX, SFW>(KT), std::max(1, cdiv(kMinCtas, strips)));
    launch<C, MX, SFW>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, split > 1 ? strips : 0, split,
                       stream);
}

template <class C, bool MX>
void launch_decode(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M,
                   int N, int K, const float* scale_a, const float* scale_b, MxScales mx,
                   cudaStream_t stream) {
    if (MX && sfw_for(K) == 4)
        launch_decode_sfw<C, MX, 4>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
    else
        launch_decode_sfw<C, MX, 2>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
}

// Large shapes: the 128x128 tile when its grid is at least one wave, else 64x128, else 64x64
// with a 128-byte BK (half the per-step pipeline overhead when a block owns few K-steps; 64
// bytes when K is not a multiple of 128). The 128x128 tile runs 2x2 warps of 64x64 (4 x 8
// accumulator tiles, 203 registers, 128 threads, two blocks per SM): at the full fp8 rate a
// 64x32 warp tile's 6 ldmatrix per 16 mma cost 1.5 to 2% against 8 per 32 (the sweep in
// docs/design/fp8gemm.md).
using Big = Cfg<128, 128, 64, 3, 2, 2>;    // 48 KB, two blocks per SM
using Half = Cfg<64, 128, 64, 3, 2, 4>;    // 36 KB
using Small = Cfg<64, 64, 128, 3, 2, 4>;   // 48 KB
using Small64 = Cfg<64, 64, 64, 3, 2, 4>;  // 24 KB, for K % 128 != 0

// Alternatives for the 128x128 tile, for a re-sweep: SPARK_FP8GEMM_V1_CONFIG=<index> forces
// one (the sweep is in docs/design/fp8gemm.md). Index 0 is the shipped `Big`.
using BigFn = void (*)(const unsigned char*, const unsigned char*, __nv_bfloat16*, int, int, int,
                       const float*, const float*, MxScales, cudaStream_t);
template <bool MX>
constexpr BigFn BIG_CONFIGS[] = {
    launch_tiles<Big, MX>,                          // 0: BK 64, 3 stages, 2x2 warps of 64x64, 48 KB
    launch_tiles<Cfg<128, 128, 64, 3, 2, 4>, MX>,   // 1: 2x4 warps of 64x32 (hgemm v3's grid)
    launch_tiles<Cfg<128, 128, 128, 2, 2, 4>, MX>,  // 2: BK 128, 2 stages, 64 KB, one block per SM
    launch_tiles<Cfg<128, 128, 128, 3, 2, 4>, MX>,  // 3: BK 128, 3 stages, 96 KB
    launch_tiles<Cfg<128, 128, 64, 4, 2, 4>, MX>,   // 4: BK 64, 4 stages, 64 KB
    launch_tiles<Cfg<128, 128, 128, 2, 2, 2>, MX>,  // 5: 2x2 warps of 64x64, BK 128, 64 KB
};
constexpr int NUM_BIG_CONFIGS =
    static_cast<int>(sizeof(BIG_CONFIGS<false>) / sizeof(BIG_CONFIGS<false>[0]));

int forced_big_config() {
    static int idx = -2;
    if (idx == -2) {
        idx = 0;
        if (const char* e = std::getenv("SPARK_FP8GEMM_V1_CONFIG")) {
            const int v = std::atoi(e);
            if (v >= 0 && v < NUM_BIG_CONFIGS) idx = v;
        }
    }
    return idx;
}

// Decode configurations per row tile, (first, second) as described at launch_decode, with
// BK = 128 bytes (K % 128 == 0) or 64 (any other K % 64 == 0). Stage bytes are BK x (BM + BN).
template <int BK, bool MX>
void launch_decode_auto(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M,
                        int N, int K, const float* scale_a, const float* scale_b, MxScales mx,
                        cudaStream_t stream) {
    using D16a = Cfg<16, 64, BK, 4, 1, 4>;  // BK 128: 40 KB, 2 per SM, 340 slots
    using D16b = Cfg<16, 32, BK, 4, 1, 2>;  // 24 KB, 4 per SM, 680 slots
    using D32a = Cfg<32, 32, BK, 4, 2, 2>;  // 32 KB, 3 per SM, 510 slots
    using D32b = Cfg<32, 64, BK, 4, 2, 4>;  // 48 KB, 2 per SM, 340 slots
    using D64a = Cfg<64, 32, BK, 4, 4, 2>;  // 48 KB, 2 per SM, 340 slots
    using D64b = Cfg<64, 64, BK, 3, 2, 4>;  // 48 KB, 2 per SM, 340 slots
    if (M <= 16) {
        if (fits<D16a, MX>(N))
            launch_decode<D16a, MX>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
        else
            launch_decode<D16b, MX>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
    } else if (M <= 32) {
        if (fits<D32a, MX>(N))
            launch_decode<D32a, MX>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
        else
            launch_decode<D32b, MX>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
    } else {
        if (fits<D64a, MX>(N))
            launch_decode<D64a, MX>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
        else
            launch_decode<D64b, MX>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
    }
}

// MX mode requires K % 256 == 0 (fp8gemm_mx_supports), so the tiles that only serve
// K % 128 != 0 (Small64, the BK = 64 decode set) are not instantiated for it.
template <bool MX>
void launch_auto(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M, int N,
                 int K, const float* scale_a, const float* scale_b, MxScales mx,
                 cudaStream_t stream) {
    auto tiles = [&](int bm, int bn) { return cdiv(M, bm) * (N / bn); };
    if (M <= 64) {
        if (K % 128 == 0)
            launch_decode_auto<128, MX>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
        else if constexpr (!MX)
            launch_decode_auto<64, MX>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
    } else if (N % 128 == 0 && tiles(128, 128) >= resident_blocks<Big, MX, 2>()) {
        BIG_CONFIGS<MX>[forced_big_config()](A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
    } else if (N % 128 == 0 && tiles(64, 128) >= resident_blocks<Half, MX, 2>()) {
        launch_tiles<Half, MX>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
    } else if (K % 128 == 0) {
        launch_tiles<Small, MX>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
    } else if constexpr (!MX) {
        launch_tiles<Small64, MX>(A, Bt, Cout, M, N, K, scale_a, scale_b, mx, stream);
    }
}

}  // namespace v1

// Whether this build's device code has the block-scaled instruction (an sm_120a / sm_121a
// build): the macro only exists in the device pass, so a kernel reports it.
__global__ void mx_available_kernel(int* out) {
    *out = SPARK_HAS_MX_MMA;
}

// A device-resident 1.0f for callers of the MX mode that pass no per-tensor scales.
__device__ const float kUnitScale = 1.0f;

// The common launcher: MX mode when mx.a != nullptr.
void run(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* Bt, __nv_bfloat16* C, int M, int N, int K,
         const float* scale_a, const float* scale_b, MxScales mx, int variant,
         cudaStream_t stream) {
    const unsigned char* a = reinterpret_cast<const unsigned char*>(A);
    const unsigned char* b = reinterpret_cast<const unsigned char*>(Bt);
    const bool MX = mx.a != nullptr;
    switch (variant) {
        case 0: {
            const dim3 grid(N / V0_COLS, M / V0_ROWS);
            if (MX)
                fp8gemm_v0_kernel<true>
                    <<<grid, V0_THREADS, 0, stream>>>(a, b, C, M, N, K, scale_a, scale_b, mx);
            else
                fp8gemm_v0_kernel<false>
                    <<<grid, V0_THREADS, 0, stream>>>(a, b, C, M, N, K, scale_a, scale_b, mx);
            break;
        }
        case 1:
            if (MX)
                v1::launch_auto<true>(a, b, C, M, N, K, scale_a, scale_b, mx, stream);
            else
                v1::launch_auto<false>(a, b, C, M, N, K, scale_a, scale_b, mx, stream);
            break;
        case 2:
            fp8gemm_tma(A, Bt, C, M, N, K, scale_a, scale_b, mx, stream);
            break;
        default:
            break;
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace

int fp8gemm_num_variants() {
    return 3;
}

bool fp8gemm_supports(int M, int N, int K, int variant) {
    if (variant < 0 || variant >= fp8gemm_num_variants()) return false;
    if (M <= 0 || N <= 0 || K <= 0) return false;
    if (N % 64 != 0 || K % 64 != 0) return false;
    if (variant == 0) return M % 16 == 0;
    if (variant == 2) return fp8gemm_tma_supports(M, N, K);
    return true;  // variant 1 zero-fills rows past M
}

bool fp8gemm_mx_supports(int M, int N, int K, int variant) {
    return K % 256 == 0 && fp8gemm_supports(M, N, K, variant);
}

bool fp8gemm_mx_available() {
    static int available = -1;
    if (available < 0) {
        int* d = nullptr;
        SPARK_CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
        mx_available_kernel<<<1, 1>>>(d);
        SPARK_CHECK_LAUNCH();
        int h = 0;
        SPARK_CUDA_CHECK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
        SPARK_CUDA_CHECK(cudaFree(d));
        available = h ? 1 : 0;
    }
    return available == 1;
}

void fp8gemm(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* Bt, __nv_bfloat16* C, int M, int N, int K,
             const float* scale_a, const float* scale_b, int variant, cudaStream_t stream) {
    SPARK_REQUIRE(
        A != nullptr && Bt != nullptr && C != nullptr && scale_a != nullptr && scale_b != nullptr,
        "fp8gemm: null pointer");
    SPARK_REQUIRE(M > 0 && N > 0 && K > 0, "fp8gemm: M, N, K must be positive");
    SPARK_REQUIRE(N % 64 == 0 && K % 64 == 0, "fp8gemm: N, K must be multiples of 64");
    SPARK_REQUIRE(variant >= 0 && variant < fp8gemm_num_variants(), "fp8gemm: unknown variant");
    SPARK_REQUIRE(is_aligned16(A) && is_aligned16(Bt), "fp8gemm: A, Bt must be 16-byte aligned");
    SPARK_REQUIRE(variant != 0 || M % 16 == 0, "fp8gemm variant 0: M must be a multiple of 16");
    run(A, Bt, C, M, N, K, scale_a, scale_b, MxScales{}, variant, stream);
}

void fp8gemm_mx(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* Bt, __nv_bfloat16* C, int M, int N,
                int K, const float* scale_a, const float* scale_b, const unsigned char* sfa,
                const unsigned char* sfb, int variant, cudaStream_t stream) {
    SPARK_REQUIRE(A != nullptr && Bt != nullptr && C != nullptr && sfa != nullptr && sfb != nullptr,
                  "fp8gemm_mx: null pointer");
    SPARK_REQUIRE((scale_a == nullptr) == (scale_b == nullptr),
                  "fp8gemm_mx: scale_a and scale_b must both be given or both be null");
    SPARK_REQUIRE(M > 0 && N > 0 && K > 0, "fp8gemm_mx: M, N, K must be positive");
    SPARK_REQUIRE(N % 64 == 0 && K % 256 == 0,
                  "fp8gemm_mx: N must be a multiple of 64 and K of 256");
    SPARK_REQUIRE(variant >= 0 && variant < fp8gemm_num_variants(), "fp8gemm_mx: unknown variant");
    SPARK_REQUIRE(is_aligned16(A) && is_aligned16(Bt), "fp8gemm_mx: A, Bt must be 16-byte aligned");
    SPARK_REQUIRE(is_aligned16(sfa) && is_aligned16(sfb),
                  "fp8gemm_mx: sfa, sfb must be 16-byte aligned");
    SPARK_REQUIRE(variant != 0 || M % 16 == 0, "fp8gemm_mx variant 0: M must be a multiple of 16");
    SPARK_REQUIRE(fp8gemm_mx_available(),
                  "fp8gemm_mx: this build has no block-scaled mma (compile for sm_120a / sm_121a)");
    if (scale_a == nullptr) {
        void* unit = nullptr;
        SPARK_CUDA_CHECK(cudaGetSymbolAddress(&unit, kUnitScale));
        scale_a = scale_b = static_cast<const float*>(unit);
    }
    MxScales mx;
    mx.a = sfa;
    mx.b = sfb;
    mx.ld = K / 32;
    run(A, Bt, C, M, N, K, scale_a, scale_b, mx, variant, stream);
}

}  // namespace spark
