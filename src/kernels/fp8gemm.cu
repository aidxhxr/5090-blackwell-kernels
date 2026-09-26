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

#include <cuda_fp8.h>

#include <algorithm>
#include <cstdlib>

#include "fp8gemm_tile.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

using fp8gemm_tile::Cfg;

// ---------------------------------------------------------------------------------------
// Variant 0. Block = 4 warps covering 16 rows x 32 columns; each warp walks K in steps of 32
// loading its fragment words straight from global memory: lane (g, t) needs k 4t..4t+3 of row
// g (and g+8, and +16), which is one aligned 4-byte load per register. Host guarantees
// M % 16 == 0, N % 32 == 0, K % 32 == 0.
// ---------------------------------------------------------------------------------------
constexpr int V0_THREADS = 128;
constexpr int V0_ROWS = 16, V0_COLS = 32;

__global__ void __launch_bounds__(V0_THREADS)
    fp8gemm_v0_kernel(const unsigned char* __restrict__ A, const unsigned char* __restrict__ Bt,
                      __nv_bfloat16* __restrict__ C, int M, int N, int K,
                      const float* __restrict__ scale_a, const float* __restrict__ scale_b) {
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
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < K; k0 += 32) {
        unsigned a[4], b[2];
        a[0] = *reinterpret_cast<const unsigned*>(a0 + k0);
        a[1] = *reinterpret_cast<const unsigned*>(a1 + k0);
        a[2] = *reinterpret_cast<const unsigned*>(a0 + k0 + 16);
        a[3] = *reinterpret_cast<const unsigned*>(a1 + k0 + 16);
        b[0] = *reinterpret_cast<const unsigned*>(b0 + k0);
        b[1] = *reinterpret_cast<const unsigned*>(b0 + k0 + 16);
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

template <class C>
__global__ void __launch_bounds__(C::THREADS)
    fp8gemm_v1_kernel(const unsigned char* __restrict__ A, const unsigned char* __restrict__ Bt,
                      __nv_bfloat16* __restrict__ Cout, int M, int N, int K,
                      const float* __restrict__ scale_a, const float* __restrict__ scale_b,
                      Sched sched) {
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
    const int bm = (tile / sched.tiles_n) * BM;
    const int bn = (tile % sched.tiles_n) * BN;
    const int m_valid = M - bm;  // rows of this tile that exist
    const float scale = *scale_a * *scale_b;

    typename C::Acc acc;
    fp8gemm_tile::mainloop<C>(A, Bt, K, bm, bn, m_valid, kt_begin, kt_end - kt_begin, smem_raw,
                              acc);

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

// Resident blocks per GPU for one configuration; also the > 48 KB smem opt-in.
template <class C>
int resident_blocks() {
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(
            fp8gemm_v1_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, fp8gemm_v1_kernel<C>, C::THREADS, C::SMEM));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

// `split` K-slices for the `tail` last tiles in row-major tile order (whole tiles first).
// split == 1 is a plain launch of `tiles` blocks with no workspace.
template <class C>
void launch(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M, int N,
            int K, const float* scale_a, const float* scale_b, int tail, int split,
            cudaStream_t stream) {
    constexpr int BM = C::BM, BN = C::BN;
    (void)resident_blocks<C>();  // the smem opt-in, if the caller has not done it
    Sched s;
    s.tiles_n = N / BN;
    const int tiles = cdiv(M, BM) * s.tiles_n;
    if (split <= 1 || tail <= 0) {
        s.dp_tiles = tiles;
        s.split = 1;
        s.ws = nullptr;
        s.counters = nullptr;
        fp8gemm_v1_kernel<C>
            <<<tiles, C::THREADS, C::SMEM, stream>>>(A, Bt, Cout, M, N, K, scale_a, scale_b, s);
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
    fp8gemm_v1_kernel<C>
        <<<grid, C::THREADS, C::SMEM, stream>>>(A, Bt, Cout, M, N, K, scale_a, scale_b, s);
}

// Wave quantization (hgemm variant 3's rule): the tiles past the last full wave of resident
// blocks are split along K over the otherwise idle blocks, `split = min(KT, resident / tail)`.
template <class C>
void launch_tiles(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M,
                  int N, int K, const float* scale_a, const float* scale_b, cudaStream_t stream) {
    const int resident = resident_blocks<C>();
    const int tiles = cdiv(M, C::BM) * (N / C::BN);
    const int KT = K / C::BK;
    const int tail = tiles % resident;
    const int split = tail > 0 ? std::min(KT, resident / tail) : 1;
    launch<C>(A, Bt, Cout, M, N, K, scale_a, scale_b, tail, split, stream);
}

// Decode shapes (M <= 64): one CTA per BN-row strip of Bt, no split-K unless a narrow N
// leaves fewer than kMinCtas strips (measured for the bf16 decode kernel: 64 CTAs with 24 KB
// of B in flight each stream the weights at the copy roof, and every split-K launch pays a
// flat 2 us for the reduction chain). The first of two configurations runs if its strips fit
// in one wave of resident CTAs, otherwise the second, which has more slots.
constexpr int kMinCtas = 64;

template <class C>
bool fits(int N) {
    return N % C::BN == 0 && N / C::BN <= resident_blocks<C>();
}

template <class C>
void launch_decode(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M,
                   int N, int K, const float* scale_a, const float* scale_b, cudaStream_t stream) {
    const int strips = N / C::BN;
    const int KT = K / C::BK;
    const int split = std::min(KT, std::max(1, cdiv(kMinCtas, strips)));
    launch<C>(A, Bt, Cout, M, N, K, scale_a, scale_b, split > 1 ? strips : 0, split, stream);
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
                       const float*, const float*, cudaStream_t);
constexpr BigFn BIG_CONFIGS[] = {
    launch_tiles<Big>,                          // 0: BK 64, 3 stages, 2x2 warps of 64x64, 48 KB
    launch_tiles<Cfg<128, 128, 64, 3, 2, 4>>,   // 1: 2x4 warps of 64x32 (hgemm v3's grid), 48 KB
    launch_tiles<Cfg<128, 128, 128, 2, 2, 4>>,  // 2: BK 128, 2 stages, 64 KB, one block per SM
    launch_tiles<Cfg<128, 128, 128, 3, 2, 4>>,  // 3: BK 128, 3 stages, 96 KB
    launch_tiles<Cfg<128, 128, 64, 4, 2, 4>>,   // 4: BK 64, 4 stages, 64 KB
    launch_tiles<Cfg<128, 128, 128, 2, 2, 2>>,  // 5: 2x2 warps of 64x64, BK 128, 64 KB
};
constexpr int NUM_BIG_CONFIGS = static_cast<int>(sizeof(BIG_CONFIGS) / sizeof(BIG_CONFIGS[0]));

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
template <int BK>
void launch_decode_auto(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M,
                        int N, int K, const float* scale_a, const float* scale_b,
                        cudaStream_t stream) {
    using D16a = Cfg<16, 64, BK, 4, 1, 4>;  // BK 128: 40 KB, 2 per SM, 340 slots
    using D16b = Cfg<16, 32, BK, 4, 1, 2>;  // 24 KB, 4 per SM, 680 slots
    using D32a = Cfg<32, 32, BK, 4, 2, 2>;  // 32 KB, 3 per SM, 510 slots
    using D32b = Cfg<32, 64, BK, 4, 2, 4>;  // 48 KB, 2 per SM, 340 slots
    using D64a = Cfg<64, 32, BK, 4, 4, 2>;  // 48 KB, 2 per SM, 340 slots
    using D64b = Cfg<64, 64, BK, 3, 2, 4>;  // 48 KB, 2 per SM, 340 slots
    if (M <= 16) {
        if (fits<D16a>(N))
            launch_decode<D16a>(A, Bt, Cout, M, N, K, scale_a, scale_b, stream);
        else
            launch_decode<D16b>(A, Bt, Cout, M, N, K, scale_a, scale_b, stream);
    } else if (M <= 32) {
        if (fits<D32a>(N))
            launch_decode<D32a>(A, Bt, Cout, M, N, K, scale_a, scale_b, stream);
        else
            launch_decode<D32b>(A, Bt, Cout, M, N, K, scale_a, scale_b, stream);
    } else {
        if (fits<D64a>(N))
            launch_decode<D64a>(A, Bt, Cout, M, N, K, scale_a, scale_b, stream);
        else
            launch_decode<D64b>(A, Bt, Cout, M, N, K, scale_a, scale_b, stream);
    }
}

void launch_auto(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M, int N,
                 int K, const float* scale_a, const float* scale_b, cudaStream_t stream) {
    auto tiles = [&](int bm, int bn) { return cdiv(M, bm) * (N / bn); };
    if (M <= 64) {
        if (K % 128 == 0)
            launch_decode_auto<128>(A, Bt, Cout, M, N, K, scale_a, scale_b, stream);
        else
            launch_decode_auto<64>(A, Bt, Cout, M, N, K, scale_a, scale_b, stream);
    } else if (N % 128 == 0 && tiles(128, 128) >= resident_blocks<Big>()) {
        BIG_CONFIGS[forced_big_config()](A, Bt, Cout, M, N, K, scale_a, scale_b, stream);
    } else if (N % 128 == 0 && tiles(64, 128) >= resident_blocks<Half>()) {
        launch_tiles<Half>(A, Bt, Cout, M, N, K, scale_a, scale_b, stream);
    } else if (K % 128 == 0) {
        launch_tiles<Small>(A, Bt, Cout, M, N, K, scale_a, scale_b, stream);
    } else {
        launch_tiles<Small64>(A, Bt, Cout, M, N, K, scale_a, scale_b, stream);
    }
}

}  // namespace v1

}  // namespace

// Variant 2 lives in fp8gemm_tma.cu.
bool fp8gemm_tma_supports(int M, int N, int K);
void fp8gemm_tma(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* Bt, __nv_bfloat16* C, int M, int N,
                 int K, const float* scale_a, const float* scale_b, cudaStream_t stream);

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

void fp8gemm(const __nv_fp8_e4m3* A, const __nv_fp8_e4m3* Bt, __nv_bfloat16* C, int M, int N, int K,
             const float* scale_a, const float* scale_b, int variant, cudaStream_t stream) {
    SPARK_REQUIRE(
        A != nullptr && Bt != nullptr && C != nullptr && scale_a != nullptr && scale_b != nullptr,
        "fp8gemm: null pointer");
    SPARK_REQUIRE(M > 0 && N > 0 && K > 0, "fp8gemm: M, N, K must be positive");
    SPARK_REQUIRE(N % 64 == 0 && K % 64 == 0, "fp8gemm: N, K must be multiples of 64");
    SPARK_REQUIRE(variant >= 0 && variant < fp8gemm_num_variants(), "fp8gemm: unknown variant");
    SPARK_REQUIRE(is_aligned16(A) && is_aligned16(Bt), "fp8gemm: A, Bt must be 16-byte aligned");
    const unsigned char* a = reinterpret_cast<const unsigned char*>(A);
    const unsigned char* b = reinterpret_cast<const unsigned char*>(Bt);

    switch (variant) {
        case 0: {
            SPARK_REQUIRE(M % 16 == 0, "fp8gemm variant 0: M must be a multiple of 16");
            const dim3 grid(N / V0_COLS, M / V0_ROWS);
            fp8gemm_v0_kernel<<<grid, V0_THREADS, 0, stream>>>(a, b, C, M, N, K, scale_a, scale_b);
            break;
        }
        case 1:
            v1::launch_auto(a, b, C, M, N, K, scale_a, scale_b, stream);
            break;
        case 2:
            fp8gemm_tma(A, Bt, C, M, N, K, scale_a, scale_b, stream);
            break;
        default:
            break;
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
