// fp4 (e2m1) block-scaled tensor-core GEMM ladder for Blackwell sm_12x: NVFP4 and MXFP4.
//
//   C[M,N] = scale_a * scale_b * sum_k (sfa[m][k/V] A[m][k]) (sfb[n][k/V] Bt[n][k])
//
// A [M, K] and Bt [N, K] are e2m1 packed two per byte with K contiguous (the TN layout of the
// fp8 GEMM, and of cuBLASLt's fp4 path), sfa and sfb are block scales in the blocked layout
// (fp4gemm_tile.cuh): e4m3 per 16 k for NVFP4 (V = 16), ue8m0 per 32 k for MXFP4 (V = 32).
// scale_a and scale_b are per-tensor fp32 scalars on the device (NVFP4's second-level scale),
// C is bf16, accumulation is fp32 inside mma.sync.m16n8k64.kind::mxf4nvf4, which applies the
// block scales itself.
//
//   variant 0: one warp per 16x8 C tile, fragments and scale words loaded straight from
//              global memory.
//   variant 1: fp8gemm variant 1's design on fp4 operands: 128x128 block tile, ldmatrix +
//              mma.sync out of XOR-swizzled smem with each stage's scale tiles copied behind
//              its operand tiles by the same cp.async pipeline, register epilogue, split-K on
//              the last partial wave, smaller tiles for small grids, one CTA per strip of Bt
//              for decode shapes (M <= 64).
//   variant 2: the 128x128 tile fed by TMA through a warp-specialized mbarrier pipeline,
//              the scale tiles by bulk copies on the same barriers (fp4gemm_tma.cu).

#include <algorithm>
#include <cstdlib>

#include "fp4gemm_tile.cuh"
#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

// Variant 2 lives in fp4gemm_tma.cu.
bool fp4gemm_tma_supports(int M, int N, int K);
void fp4gemm_tma(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* C, int M, int N,
                 int K, const float* scale_a, const float* scale_b, fp4gemm_tile::Scales sf,
                 int format, cudaStream_t stream);

namespace {

using fp4gemm_tile::Cfg;
using fp4gemm_tile::MXFP4;
using fp4gemm_tile::NVFP4;
using fp4gemm_tile::Scales;

// ---------------------------------------------------------------------------------------
// Variant 0. Block = 4 warps covering 16 rows x 32 columns; each warp walks K in k64 steps
// (32 bytes), loading the fragment words from global memory as fp8gemm variant 0 does, and
// the scale words from the blocked layout: lane (g, t) supplies row g + 8 (t & 1) of A and
// column g of Bt (the lanes the instruction reads with thread-id 0), one aligned 4-byte word
// each. Host guarantees M % 16 == 0, N % 64 == 0, K % 256 == 0.
// ---------------------------------------------------------------------------------------
constexpr int V0_THREADS = 128;
constexpr int V0_ROWS = 16, V0_COLS = 32;

// Byte offset of the scale word of row r for k64 step s in the blocked layout.
template <int FMT>
__device__ __forceinline__ size_t sf_word(int r, int s, int ktiles) {
    constexpr int SPT = FMT == NVFP4 ? 1 : 2;
    return (static_cast<size_t>(r / 128) * ktiles + s / SPT) * fp4gemm_tile::kSfTile +
           (r % 32) * 16 + ((r / 32) % 4) * 4;
}

template <int FMT>
__global__ void __launch_bounds__(V0_THREADS)
    fp4gemm_v0_kernel(const unsigned char* __restrict__ A, const unsigned char* __restrict__ Bt,
                      __nv_bfloat16* __restrict__ C, int M, int N, int K,
                      const float* __restrict__ scale_a, const float* __restrict__ scale_b,
                      Scales sf) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g = lane >> 2;
    const int t = lane & 3;
    const int row = blockIdx.y * V0_ROWS;
    const int col = blockIdx.x * V0_COLS + warp * 8;
    const int Kb = K / 2;
    (void)M;

    const unsigned char* a0 = A + static_cast<size_t>(row + g) * Kb + 4 * t;
    const unsigned char* a1 = a0 + static_cast<size_t>(8) * Kb;
    const unsigned char* b0 = Bt + static_cast<size_t>(col + g) * Kb + 4 * t;
    const int ra = row + g + 8 * (t & 1);
    const int rb = col + g;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < K / 64; ++s) {
        const int k0 = s * 32;
        unsigned a[4], b[2];
        a[0] = *reinterpret_cast<const unsigned*>(a0 + k0);
        a[1] = *reinterpret_cast<const unsigned*>(a1 + k0);
        a[2] = *reinterpret_cast<const unsigned*>(a0 + k0 + 16);
        a[3] = *reinterpret_cast<const unsigned*>(a1 + k0 + 16);
        b[0] = *reinterpret_cast<const unsigned*>(b0 + k0);
        b[1] = *reinterpret_cast<const unsigned*>(b0 + k0 + 16);
        const unsigned sa =
            *reinterpret_cast<const unsigned*>(sf.a + sf_word<FMT>(ra, s, sf.ktiles));
        const unsigned sb =
            *reinterpret_cast<const unsigned*>(sf.b + sf_word<FMT>(rb, s, sf.ktiles));
        if constexpr (FMT == NVFP4)
            mma_e2m1_16864_nvf4<0, 0>(acc, a, b, sa, sb);
        else if (s & 1)
            mma_e2m1_16864_mxf4<2, 0, 0>(acc, a, b, sa, sb);
        else
            mma_e2m1_16864_mxf4<0, 0, 0>(acc, a, b, sa, sb);
    }
    const float scale = *scale_a * *scale_b;
    __nv_bfloat16* c0 = C + static_cast<size_t>(row + g) * N + col + 2 * t;
    *reinterpret_cast<__nv_bfloat162*>(c0) = __floats2bfloat162_rn(acc[0] * scale, acc[1] * scale);
    *reinterpret_cast<__nv_bfloat162*>(c0 + static_cast<size_t>(8) * N) =
        __floats2bfloat162_rn(acc[2] * scale, acc[3] * scale);
}

// ---------------------------------------------------------------------------------------
// Variant 1: fp8gemm variant 1's work assignment on the fp4 tile (fp4gemm_tile.cuh).
// ---------------------------------------------------------------------------------------
namespace v1 {

// Blocks [0, dp_tiles) own one output tile each; blocks from dp_tiles on split the remaining
// tiles `split` ways along K into the fp32 workspace, the last slice to arrive converting.
struct Sched {
    int tiles_n;
    int dp_tiles;
    int split;
    float* ws;
    int* counters;
};

template <class C>
__global__ void __launch_bounds__(C::THREADS)
    fp4gemm_v1_kernel(const unsigned char* __restrict__ A, const unsigned char* __restrict__ Bt,
                      __nv_bfloat16* __restrict__ Cout, int M, int N, int K,
                      const float* __restrict__ scale_a, const float* __restrict__ scale_b,
                      Scales sf, Sched sched) {
    constexpr int BM = C::BM, BN = C::BN, BK = C::BK, THREADS = C::THREADS;
    constexpr int WROWS = C::WROWS, WCOLS = C::WCOLS, MT = C::MT, NT = C::NT;
    extern __shared__ __align__(128) unsigned char smem_raw[];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / C::WN;
    const int wn = warp % C::WN;

    const int Kb = K / 2;
    const int KT = Kb / BK;
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
    const int m_valid = M - bm;
    const float scale = *scale_a * *scale_b;

    typename C::Acc acc;
    fp4gemm_tile::mainloop<C>(A, Bt, Kb, bm, bn, m_valid, kt_begin, kt_end - kt_begin, smem_raw,
                              acc, sf);

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
            if (r0 >= m_valid) continue;
            atomicAdd(wt + r0 * BN + c0, acc[mi][nj][0]);
            atomicAdd(wt + r0 * BN + c0 + 1, acc[mi][nj][1]);
            if (r0 + 8 >= m_valid) continue;
            atomicAdd(wt + (r0 + 8) * BN + c0, acc[mi][nj][2]);
            atomicAdd(wt + (r0 + 8) * BN + c0 + 1, acc[mi][nj][3]);
        }
    }
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
        float2* p = reinterpret_cast<float2*>(wt + r * BN + c);
        const float2 v = __ldcg(p);
        __stcg(p, make_float2(0.f, 0.f));  // the workspace stays zeroed (fp4gemm_tile::Workspace)
        *reinterpret_cast<__nv_bfloat162*>(Cout + static_cast<size_t>(bm + r) * N + bn + c) =
            __floats2bfloat162_rn(v.x * scale, v.y * scale);
    }
    if (tid == 0) sched.counters[tile - sched.dp_tiles] = 0;
}

template <class C>
int resident_blocks() {
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(cudaFuncSetAttribute(
            fp4gemm_v1_kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, fp4gemm_v1_kernel<C>, C::THREADS, C::SMEM));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <class C>
void launch(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M, int N,
            int K, const float* scale_a, const float* scale_b, Scales sf, int tail, int split,
            cudaStream_t stream) {
    constexpr int BM = C::BM, BN = C::BN;
    (void)resident_blocks<C>();
    Sched s;
    s.tiles_n = N / BN;
    const int tiles = cdiv(M, BM) * s.tiles_n;
    if (split <= 1 || tail <= 0) {
        s.dp_tiles = tiles;
        s.split = 1;
        s.ws = nullptr;
        s.counters = nullptr;
        fp4gemm_v1_kernel<C>
            <<<tiles, C::THREADS, C::SMEM, stream>>>(A, Bt, Cout, M, N, K, scale_a, scale_b, sf, s);
        return;
    }
    s.dp_tiles = tiles - tail;
    s.split = split;
    static fp4gemm_tile::Workspace w;
    w.reserve(tail, BM * BN, stream);
    s.ws = w.ws;
    s.counters = w.counters;
    const int grid = s.dp_tiles + tail * s.split;
    fp4gemm_v1_kernel<C>
        <<<grid, C::THREADS, C::SMEM, stream>>>(A, Bt, Cout, M, N, K, scale_a, scale_b, sf, s);
}

// Wave quantization: the tiles past the last full wave split along K over the idle blocks.
constexpr int kMaxSplit = 8;
template <class C>
void launch_tiles(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M,
                  int N, int K, const float* scale_a, const float* scale_b, Scales sf,
                  cudaStream_t stream) {
    const int resident = resident_blocks<C>();
    const int tiles = cdiv(M, C::BM) * (N / C::BN);
    const int KT = K / 2 / C::BK;
    const int tail = tiles % resident;
    // At most kMaxSplit slices, as variant 2 (fp4gemm_tma.cu).
    int split = tail > 0 ? std::min({KT, resident / tail, kMaxSplit}) : 1;
    if (fp4gemm_tile::max_split_override() > 0)
        split = std::min(split, fp4gemm_tile::max_split_override());
    launch<C>(A, Bt, Cout, M, N, K, scale_a, scale_b, sf, tail, split, stream);
}

// Decode shapes (M <= 64): one CTA per BN-row strip of Bt, split-K only when the strips are
// fewer than kMinCtas (fp8gemm variant 1's rule).
constexpr int kMinCtas = 64;

template <class C>
bool fits(int N) {
    return N % C::BN == 0 && N / C::BN <= resident_blocks<C>();
}

template <class C>
void launch_decode(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M,
                   int N, int K, const float* scale_a, const float* scale_b, Scales sf,
                   cudaStream_t stream) {
    const int strips = N / C::BN;
    const int KT = K / 2 / C::BK;
    const int split = std::min(KT, std::max(1, cdiv(kMinCtas, strips)));
    launch<C>(A, Bt, Cout, M, N, K, scale_a, scale_b, sf, split > 1 ? strips : 0, split, stream);
}

// Tile configurations (BK in bytes: 128 bytes = 256 e2m1 values = 4 k64 steps per stage).
template <int FMT>
struct Tiles {
    using Big = Cfg<128, 128, 64, 2, 2, 2, FMT>;  // BK 64, 2 stages, 2x2 warps of 64x64, 36 KB
    using Half = Cfg<64, 128, 64, 4, 2, 4, FMT>;
    using Small = Cfg<64, 64, 128, 3, 2, 4, FMT>;
    using D16a = Cfg<16, 64, 128, 4, 1, 4, FMT>;
    using D16b = Cfg<16, 32, 128, 4, 1, 2, FMT>;
    using D32a = Cfg<32, 32, 128, 4, 2, 2, FMT>;
    using D32b = Cfg<32, 64, 128, 4, 2, 4, FMT>;
    using D64a = Cfg<64, 32, 128, 4, 4, 2, FMT>;
    using D64b = Cfg<64, 64, 128, 3, 2, 4, FMT>;
};

// Alternatives for the 128x128 tile: SPARK_FP4GEMM_V1_CONFIG=<index> forces one.
using BigFn = void (*)(const unsigned char*, const unsigned char*, __nv_bfloat16*, int, int, int,
                       const float*, const float*, Scales, cudaStream_t);
template <int FMT>
constexpr BigFn BIG_CONFIGS[] = {
    launch_tiles<typename Tiles<FMT>::Big>,          // 0: BK 64, 2 stages, 2x2 of 64x64, 36 KB
    launch_tiles<Cfg<128, 128, 64, 2, 2, 4, FMT>>,   // 1: BK 64, 2 stages, 2x4 of 64x32, 36 KB
    launch_tiles<Cfg<128, 128, 64, 4, 2, 4, FMT>>,   // 2: BK 64, 4 stages, 2x4, 72 KB
    launch_tiles<Cfg<128, 128, 128, 2, 2, 4, FMT>>,  // 3: BK 128, 2 stages, 2x4, 72 KB
    launch_tiles<Cfg<128, 128, 64, 3, 2, 4, FMT>>,   // 4: BK 64, 3 stages, 2x4, 54 KB
    launch_tiles<Cfg<128, 128, 64, 4, 2, 2, FMT>>,   // 5: BK 64, 4 stages, 2x2 of 64x64, 72 KB
};
constexpr int NUM_BIG_CONFIGS =
    static_cast<int>(sizeof(BIG_CONFIGS<NVFP4>) / sizeof(BIG_CONFIGS<NVFP4>[0]));

int forced_big_config() {
    static int idx = -2;
    if (idx == -2) {
        idx = 0;
        if (const char* e = std::getenv("SPARK_FP4GEMM_V1_CONFIG")) {
            const int v = std::atoi(e);
            if (v >= 0 && v < NUM_BIG_CONFIGS) idx = v;
        }
    }
    return idx;
}

template <int FMT>
void launch_auto(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* Cout, int M, int N,
                 int K, const float* scale_a, const float* scale_b, Scales sf,
                 cudaStream_t stream) {
    using T = Tiles<FMT>;
    auto tiles = [&](int bm, int bn) { return cdiv(M, bm) * (N / bn); };
    if (M <= 16) {
        if (fits<typename T::D16a>(N))
            launch_decode<typename T::D16a>(A, Bt, Cout, M, N, K, scale_a, scale_b, sf, stream);
        else
            launch_decode<typename T::D16b>(A, Bt, Cout, M, N, K, scale_a, scale_b, sf, stream);
    } else if (M <= 32) {
        if (fits<typename T::D32a>(N))
            launch_decode<typename T::D32a>(A, Bt, Cout, M, N, K, scale_a, scale_b, sf, stream);
        else
            launch_decode<typename T::D32b>(A, Bt, Cout, M, N, K, scale_a, scale_b, sf, stream);
    } else if (M <= 64) {
        if (fits<typename T::D64a>(N))
            launch_decode<typename T::D64a>(A, Bt, Cout, M, N, K, scale_a, scale_b, sf, stream);
        else
            launch_decode<typename T::D64b>(A, Bt, Cout, M, N, K, scale_a, scale_b, sf, stream);
    } else if (N % 128 == 0 && tiles(128, 128) >= resident_blocks<typename T::Big>()) {
        BIG_CONFIGS<FMT>[forced_big_config()](A, Bt, Cout, M, N, K, scale_a, scale_b, sf, stream);
    } else if (N % 128 == 0 && tiles(64, 128) >= resident_blocks<typename T::Half>()) {
        launch_tiles<typename T::Half>(A, Bt, Cout, M, N, K, scale_a, scale_b, sf, stream);
    } else {
        launch_tiles<typename T::Small>(A, Bt, Cout, M, N, K, scale_a, scale_b, sf, stream);
    }
}

}  // namespace v1

// Whether this build's device code has the fp4 instruction (an sm_120a / sm_121a build).
__global__ void fp4_available_kernel(int* out) {
    *out = SPARK_HAS_MX_MMA;
}

__device__ const float kUnitScale = 1.0f;

}  // namespace

int fp4gemm_num_variants() {
    return 3;
}

size_t fp4_scale_bytes(int rows, int K, int format) {
    const size_t padded = static_cast<size_t>(cdiv(rows, 128)) * 128;
    return padded * static_cast<size_t>(K / (format == FP4_MXFP4 ? 32 : 16));
}

bool fp4gemm_supports(int M, int N, int K, int variant) {
    if (variant < 0 || variant >= fp4gemm_num_variants()) return false;
    if (M <= 0 || N <= 0 || K <= 0) return false;
    if (N % 64 != 0 || K % 256 != 0) return false;
    if (variant == 0) return M % 16 == 0;
    if (variant == 2) return fp4gemm_tma_supports(M, N, K);
    return true;
}

bool fp4gemm_available() {
    static int available = -1;
    if (available < 0) {
        int* d = nullptr;
        SPARK_CUDA_CHECK(cudaMalloc(&d, sizeof(int)));
        fp4_available_kernel<<<1, 1>>>(d);
        SPARK_CHECK_LAUNCH();
        int h = 0;
        SPARK_CUDA_CHECK(cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost));
        SPARK_CUDA_CHECK(cudaFree(d));
        available = h ? 1 : 0;
    }
    return available == 1;
}

void fp4gemm(const unsigned char* A, const unsigned char* Bt, __nv_bfloat16* C, int M, int N, int K,
             const unsigned char* sfa, const unsigned char* sfb, const float* scale_a,
             const float* scale_b, int format, int variant, cudaStream_t stream) {
    SPARK_REQUIRE(A != nullptr && Bt != nullptr && C != nullptr && sfa != nullptr && sfb != nullptr,
                  "fp4gemm: null pointer");
    SPARK_REQUIRE((scale_a == nullptr) == (scale_b == nullptr),
                  "fp4gemm: scale_a and scale_b must both be given or both be null");
    SPARK_REQUIRE(format == FP4_NVFP4 || format == FP4_MXFP4, "fp4gemm: unknown format");
    SPARK_REQUIRE(M > 0 && N > 0 && K > 0, "fp4gemm: M, N, K must be positive");
    SPARK_REQUIRE(N % 64 == 0 && K % 256 == 0, "fp4gemm: N must be a multiple of 64 and K of 256");
    SPARK_REQUIRE(variant >= 0 && variant < fp4gemm_num_variants(), "fp4gemm: unknown variant");
    SPARK_REQUIRE(is_aligned16(A) && is_aligned16(Bt) && is_aligned16(sfa) && is_aligned16(sfb),
                  "fp4gemm: A, Bt, sfa, sfb must be 16-byte aligned");
    SPARK_REQUIRE(variant != 0 || M % 16 == 0, "fp4gemm variant 0: M must be a multiple of 16");
    SPARK_REQUIRE(variant != 2 || fp4gemm_tma_supports(M, N, K),
                  "fp4gemm variant 2: requires M % 128 == 0, N % 128 == 0 and at least one "
                  "128x128 tile per SM");
    SPARK_REQUIRE(fp4gemm_available(),
                  "fp4gemm: this build has no fp4 mma (compile for sm_120a / sm_121a)");
    if (scale_a == nullptr) {
        void* unit = nullptr;
        SPARK_CUDA_CHECK(cudaGetSymbolAddress(&unit, kUnitScale));
        scale_a = scale_b = static_cast<const float*>(unit);
    }
    Scales sf;
    sf.a = sfa;
    sf.b = sfb;
    sf.ktiles = K / (format == FP4_MXFP4 ? 128 : 64);
    const bool mx = format == FP4_MXFP4;
    switch (variant) {
        case 0: {
            const dim3 grid(N / V0_COLS, M / V0_ROWS);
            if (mx)
                fp4gemm_v0_kernel<MXFP4>
                    <<<grid, V0_THREADS, 0, stream>>>(A, Bt, C, M, N, K, scale_a, scale_b, sf);
            else
                fp4gemm_v0_kernel<NVFP4>
                    <<<grid, V0_THREADS, 0, stream>>>(A, Bt, C, M, N, K, scale_a, scale_b, sf);
            break;
        }
        case 1:
            if (mx)
                v1::launch_auto<MXFP4>(A, Bt, C, M, N, K, scale_a, scale_b, sf, stream);
            else
                v1::launch_auto<NVFP4>(A, Bt, C, M, N, K, scale_a, scale_b, sf, stream);
            break;
        case 2:
            fp4gemm_tma(A, Bt, C, M, N, K, scale_a, scale_b, sf, format, stream);
            break;
        default:
            break;
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
