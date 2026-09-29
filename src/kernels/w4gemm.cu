// W4A16 GEMM: C[M,N] = A[M,K] * dequant(W)[K,N], bf16 activations, int4 weights with a bf16
// scale (and optionally a zero point) per 128 k of a column, fp32 accumulation.
// Variants (the ladder of include/spark/kernels.h):
//   0  one thread per output, scalar dequant from the packed layout
//   1  one warp per 16-column strip and 8 tokens, the packed weights and the activations
//      straight into registers, lop3 dequant into the mma.sync A fragment
//   2  M <= 16: variant 1's step in independent warps that split K inside a block, with a
//      ring of weight loads several groups deep and the activations one group ahead;
//      M > 16: a block tile with warps along N and M sharing activations through a
//      multi-stage cp.async pipeline in shared memory, on a Stream-K schedule
// The packed layout, the k order inside a group and the dequant: w4gemm_internal.cuh.
// Design and measurements: docs/design/w4gemm.md.
#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "spark/common.cuh"
#include "spark/kernels.h"
#include "w4gemm_internal.cuh"

namespace spark {

namespace {

constexpr int kG = kW4GroupSize;

// 16 bytes of weights, read once per call: no L1 allocation.
__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

__device__ __forceinline__ unsigned word_of(const uint4& v, int j) {
    return j == 0 ? v.x : j == 1 ? v.y : j == 2 ? v.z : v.w;
}

// ---- variant 0: one thread per output element -------------------------------------------

__global__ void w4_naive_kernel(const __nv_bfloat16* __restrict__ A,
                                const uint32_t* __restrict__ packed,
                                const __nv_bfloat16* __restrict__ scales,
                                const uint8_t* __restrict__ zeros, __nv_bfloat16* __restrict__ C,
                                int M, int N, int K) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;
    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        size_t word;
        int p;
        w4::locate(n, k, K, word, p);
        const int q = (packed[word] >> (4 * p)) & 0xF;
        const size_t sidx = static_cast<size_t>(k / kG) * N + n;
        const int z = zeros ? zeros[sidx] : 8;
        const float w = __bfloat162float(
            __float2bfloat16(static_cast<float>(q - z) * __bfloat162float(scales[sidx])));
        acc = fmaf(__bfloat162float(A[static_cast<size_t>(m) * K + k]), w, acc);
    }
    C[static_cast<size_t>(m) * N + n] = __float2bfloat16(acc);
}

// ---- pieces shared by variants 1 and 2 ----------------------------------------------------

// bf16x2 (scale, scale) and (128 + z, 128 + z) for the two rows g and g + 8 a lane dequantizes.
struct RowPairs {
    unsigned ss_lo, ss_hi, zz_lo, zz_hi;
};

// The eight k16 steps of one 128-k group for one 16-column block: step j's A fragment
// (weights) is word j % 4 of unit j / 4, its B fragments (NT token tiles) come from the
// 16-byte activation load i = j / 2 (low half for even j, high half for odd j).
template <int NT>
__device__ __forceinline__ void group_mma(float (&acc)[NT][4], const uint4 (&w)[2],
                                          const RowPairs& rp, const uint4 (&act)[NT][4]) {
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        unsigned a[4];
        w4::dequant(word_of(w[j / 4], j % 4), rp.zz_lo, rp.zz_hi, rp.ss_lo, rp.ss_hi, a);
#pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            const uint4& v = act[nt][j / 2];
            const unsigned b[2] = {j % 2 == 0 ? v.x : v.z, j % 2 == 0 ? v.y : v.w};
            mma_bf16_16816(acc[nt], a, b);
        }
    }
}

// Store one m16n8 accumulator tile: rows are output columns n0 + g and n0 + g + 8, columns
// tokens m0 + 2c and m0 + 2c + 1.
__device__ __forceinline__ void store_tile(const float (&d)[4], __nv_bfloat16* C, int M, int N,
                                           int n0, int m0, int lane) {
    const int g = lane >> 2, c = lane & 3;
    const int m = m0 + 2 * c;
    if (m < M) {
        C[static_cast<size_t>(m) * N + n0 + g] = __float2bfloat16(d[0]);
        C[static_cast<size_t>(m) * N + n0 + g + 8] = __float2bfloat16(d[2]);
    }
    if (m + 1 < M) {
        C[static_cast<size_t>(m + 1) * N + n0 + g] = __float2bfloat16(d[1]);
        C[static_cast<size_t>(m + 1) * N + n0 + g + 8] = __float2bfloat16(d[3]);
    }
}

// ---- variant 1: registers only, one group at a time ----------------------------------------

// Block of 4 warps, each on its own 16-column strip and one 8-token chunk (blockIdx.y), over
// the whole K. Per 128-k group: two 16-byte weight loads, two scales, four 16-byte
// activation loads, eight dequants and eight mma.sync. Nothing is loaded ahead.
__global__ void __launch_bounds__(128)
    w4_reg_kernel(const __nv_bfloat16* __restrict__ A, const uint4* __restrict__ packed,
                  const __nv_bfloat16* __restrict__ scales, const uint8_t* __restrict__ zeros,
                  __nv_bfloat16* __restrict__ C, int M, int N, int K) {
    const int lane = threadIdx.x & 31;
    const int t = blockIdx.x * 4 + (threadIdx.x >> 5);  // 16-column strip
    if (t >= N / 16) return;
    const int m0 = blockIdx.y * 8;
    const int g = lane >> 2, c = lane & 3;
    const int n0 = t * 16;
    const int units = K / 64;
    const bool has_row = m0 + g < M;
    const uint4* arow =
        reinterpret_cast<const uint4*>(A + static_cast<size_t>(has_row ? m0 + g : 0) * K) + c;
    const unsigned short* sc = reinterpret_cast<const unsigned short*>(scales);

    float acc[1][4] = {{0.f, 0.f, 0.f, 0.f}};
    for (int grp = 0; grp < K / kG; ++grp) {
        const uint4* wp = packed + w4::block_index(t, 2 * grp, units) * 32 + lane;
        const uint4 w[2] = {__ldg(wp), __ldg(wp + 32)};
        const size_t srow = static_cast<size_t>(grp) * N + n0;
        RowPairs rp{w4::splat(__ldg(sc + srow + g)), w4::splat(__ldg(sc + srow + g + 8)),
                    w4::kSymZeroPair, w4::kSymZeroPair};
        if (zeros) {
            rp.zz_lo = w4::zero_pair(__ldg(zeros + srow + g));
            rp.zz_hi = w4::zero_pair(__ldg(zeros + srow + g + 8));
        }
        uint4 act[1][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
            act[0][i] = has_row ? __ldg(arow + grp * 16 + 4 * i) : make_uint4(0, 0, 0, 0);
        group_mma<1>(acc, w, rp, act);
    }
    store_tile(acc[0], C, M, N, n0, m0, lane);
}

// ---- variant 2 ------------------------------------------------------------------------------

// Where a block's partial sums go when no single block covers a tile: an fp32 workspace
// (atomics) and a per-tile count of the groups added so far. The block that completes the
// count converts the tile to bf16 and puts the workspace and the counter back to zero, so
// both are zero between launches and no memset precedes a call.
struct Fixup {
    float* ws;      // M x N
    int* counters;  // one per tile
};

// After this block's fp32 adds for a tile of rows m0..m0+rows, columns n0..n0+bn: count
// `len` groups and, if they complete the tile's G, convert it. Every thread of the block
// calls it; `s_last` is a __shared__ int.
__device__ __forceinline__ void fixup_tile(const Fixup& f, __nv_bfloat16* C, int N, int tile,
                                           int len, int G, int m0, int rows, int n0, int bn,
                                           int* s_last) {
    __threadfence();
    __syncthreads();
    if (threadIdx.x == 0) *s_last = atomicAdd(f.counters + tile, len) + len == G;
    __syncthreads();
    if (!*s_last) return;
    __threadfence();
    for (int i = threadIdx.x; i < rows * (bn / 4); i += blockDim.x) {
        const int r = i / (bn / 4);
        const int c = n0 + (i % (bn / 4)) * 4;
        float4* wsp = reinterpret_cast<float4*>(f.ws + static_cast<size_t>(m0 + r) * N + c);
        const float4 v = __ldcg(wsp);  // from L2, where the atomics landed
        __nv_bfloat162* out =
            reinterpret_cast<__nv_bfloat162*>(C + static_cast<size_t>(m0 + r) * N + c);
        out[0] = __floats2bfloat162_rn(v.x, v.y);
        out[1] = __floats2bfloat162_rn(v.z, v.w);
        __stcg(wsp, make_float4(0.f, 0.f, 0.f, 0.f));
    }
    if (threadIdx.x == 0) atomicExch(f.counters + tile, 0);
}

// Output element e of lane ln of a warp tile: e = (mt * NT + nt) * 4 + q; the accumulator
// tile's rows are columns g and g + 8, its columns tokens 2c and 2c + 1.
template <int NT>
__device__ __forceinline__ void element_coords(int e, int ln, int& dn, int& dm) {
    const int mt = e / (4 * NT), nt = (e / 4) % NT, q = e % 4;
    dn = mt * 16 + (ln >> 2) + (q >> 1) * 8;
    dm = nt * 8 + 2 * (ln & 3) + (q & 1);
}

// ---- variant 2, M <= 16: independent warps, everything in registers ----------------------

// Block: WK warps on one strip of 16 MTW columns and one chunk of TW <= 8 NT tokens, over the
// groups of one K slice (blockIdx = (slice * strips + strip) * chunks + chunk). Warp wk takes
// the slice's groups wk, wk + WK, ... so the block's warps read consecutive 1 KB blocks of
// the strip at any moment. Per warp: the weights and scales of the next D groups are loaded
// into a register ring, the activations of the next group into a second buffer, and no
// barrier is taken until the partial sums are added across the WK warps at the end. Every
// load is the lane's own fragment, so nothing goes through shared memory on the way in.
struct WarpParams {
    const __nv_bfloat16* A;
    const uint4* B;
    const __nv_bfloat16* S;
    const uint8_t* Z;
    __nv_bfloat16* C;
    int M, N, K;
    int chunks, split;
    Fixup fix;
};

template <int MTW, int NT, int TW, int WK, int D, bool ASYM>
__global__ void __launch_bounds__(WK * 32) w4_warp_kernel(WarpParams p) {
    static_assert(NT == 1 || NT == 2, "tokens per block");
    static_assert(TW <= 8 * NT && (NT == 1 || TW == 16), "");
    static_assert(D % 2 == 0, "the activation double buffer alternates with the ring slot");
    constexpr int BN = 16 * MTW;
    constexpr int E = MTW * NT * 4;
    __shared__ float red[WK > 1 ? WK * E * 32 : 1];
    __shared__ int s_last;

    const int lane = threadIdx.x & 31, wk = threadIdx.x >> 5;
    const int g = lane >> 2, c = lane & 3;
    const int M = p.M, N = p.N, K = p.K;
    const int G = K / kG, units = K / 64;
    const int strips = N / BN;
    const int chunk = blockIdx.x % p.chunks;
    const int strip = (blockIdx.x / p.chunks) % strips;
    const int slice = blockIdx.x / (p.chunks * strips);
    const int gb = static_cast<int>(static_cast<long long>(slice) * G / p.split);
    const int ge = static_cast<int>(static_cast<long long>(slice + 1) * G / p.split);
    const int ng = ge - gb > wk ? (ge - gb - wk + WK - 1) / WK : 0;  // this warp's groups
    const int m0 = chunk * TW, n0 = strip * BN;

    // Per-lane bases. Token rows past M (or past TW) load nothing and multiply zeros.
    const uint4* wbase = p.B + w4::block_index(strip * MTW, 0, units) * 32 + lane;
    const size_t wstrip = static_cast<size_t>(units) * 32;  // uint4s per 16-column strip
    const unsigned short* sbase = reinterpret_cast<const unsigned short*>(p.S) + n0 + g;
    const uint8_t* zbase = p.Z + n0 + g;
    const uint4* abase[NT];
    bool arow[NT];
#pragma unroll
    for (int nt = 0; nt < NT; ++nt) {
        const int tw = nt * 8 + g;
        arow[nt] = tw < TW && m0 + tw < M;
        abase[nt] =
            reinterpret_cast<const uint4*>(p.A + static_cast<size_t>(arow[nt] ? m0 + tw : 0) * K) +
            c;
    }

    uint4 wr[D][MTW][2];
    unsigned short sr[D][MTW][2];
    uint8_t zr[D][MTW][2];
    uint4 ar[2][NT][4];
    auto load_w = [&](int slot, int grp) {
#pragma unroll
        for (int mt = 0; mt < MTW; ++mt) {
            const uint4* wp = wbase + mt * wstrip + static_cast<size_t>(2 * grp) * 32;
            wr[slot][mt][0] = ldg_stream(wp);
            wr[slot][mt][1] = ldg_stream(wp + 32);
            const size_t srow = static_cast<size_t>(grp) * N + mt * 16;
            sr[slot][mt][0] = __ldg(sbase + srow);
            sr[slot][mt][1] = __ldg(sbase + srow + 8);
            if constexpr (ASYM) {
                zr[slot][mt][0] = __ldg(zbase + srow);
                zr[slot][mt][1] = __ldg(zbase + srow + 8);
            }
        }
    };
    auto load_a = [&](int buf, int grp) {
#pragma unroll
        for (int nt = 0; nt < NT; ++nt)
#pragma unroll
            for (int i = 0; i < 4; ++i)
                ar[buf][nt][i] =
                    arow[nt] ? __ldg(abase[nt] + grp * 16 + 4 * i) : make_uint4(0, 0, 0, 0);
    };

    float acc[MTW][NT][4];
#pragma unroll
    for (int i = 0; i < MTW; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

#pragma unroll
    for (int d = 0; d < D; ++d)
        if (d < ng) load_w(d, gb + wk + d * WK);
    if (ng > 0) load_a(0, gb + wk);

    for (int i0 = 0; i0 < ng; i0 += D) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
            const int i = i0 + d;
            if (i < ng) {
                if (i + 1 < ng) load_a((d + 1) % 2, gb + wk + (i + 1) * WK);
#pragma unroll
                for (int mt = 0; mt < MTW; ++mt) {
                    RowPairs rp{w4::splat(sr[d][mt][0]), w4::splat(sr[d][mt][1]), w4::kSymZeroPair,
                                w4::kSymZeroPair};
                    if constexpr (ASYM) {
                        rp.zz_lo = w4::zero_pair(zr[d][mt][0]);
                        rp.zz_hi = w4::zero_pair(zr[d][mt][1]);
                    }
                    group_mma<NT>(acc[mt], wr[d][mt], rp, ar[d % 2]);
                }
                if (i + D < ng) load_w(d, gb + wk + (i + D) * WK);
            }
        }
    }

    // Sum the WK warps' partials; thread i then owns element (i / 32) of lane i % 32.
    const bool full = p.split == 1;
    auto emit = [&](int e, int ln, float v) {
        int dn, dm;
        element_coords<NT>(e, ln, dn, dm);
        const int m = m0 + dm;
        if (dm >= TW || m >= M) return;
        const size_t idx = static_cast<size_t>(m) * N + n0 + dn;
        if (full)
            p.C[idx] = __float2bfloat16(v);
        else
            atomicAdd(p.fix.ws + idx, v);
    };
    if constexpr (WK > 1) {
#pragma unroll
        for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int q = 0; q < 4; ++q)
                    red[(wk * E + (mt * NT + nt) * 4 + q) * 32 + lane] = acc[mt][nt][q];
        __syncthreads();
        for (int i = threadIdx.x; i < E * 32; i += WK * 32) {
            float v = 0.f;
#pragma unroll
            for (int k = 0; k < WK; ++k) v += red[k * E * 32 + i];
            emit(i / 32, i % 32, v);
        }
    } else {
#pragma unroll
        for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                for (int q = 0; q < 4; ++q) emit((mt * NT + nt) * 4 + q, lane, acc[mt][nt][q]);
    }
    if (!full)
        fixup_tile(p.fix, p.C, N, strip * p.chunks + chunk, ge - gb, G, m0, min(TW, M - m0), n0, BN,
                   &s_last);
}

// The work of one call is tiles x G iterations, a tile being BN columns x BM tokens
// (tile = strip * chunks + chunk) and an iteration one 128-k group of it. Every block owns a
// contiguous range of iterations, it = tile * G + group:
//   split > 0  block = slice * tiles + tile runs its tile's groups [slice G / split,
//              (slice + 1) G / split): one tile per block, K split `split` ways;
//   split == 0 Stream-K: gridDim.x resident blocks divide the tiles x G iterations evenly
//              in tile-major order, so a block may finish one tile, run whole ones and start
//              another.
// A tile that no single block covers goes through the Fixup.
struct PipeParams {
    const __nv_bfloat16* A;
    const uint4* B;
    const __nv_bfloat16* S;
    const uint8_t* Z;
    __nv_bfloat16* C;
    int M, N, K;
    int chunks;
    int split;
    Fixup fix;
};

// Block tile: WN warps along N, each on MTW 16-column blocks (BN = 16 MTW WN columns), times
// WM warps along M, each on TW tokens in NT n8 tiles (BM = TW WM), times WK warps along K,
// each on one group of a stage (a stage is up to WK groups of one tile). The WM warps of a
// column dequantize the same weights; the WK warps' partial sums are added in shared memory
// at the end. TW = 8 NT, except that one n8 tile (NT = 1) may hold 1, 2 or 4 tokens: the
// stage then holds only those rows and the lanes of the other rows feed zeros. A stage holds
// BM x 128 WK activations, WN WK MTW KB of weights, WK x BN scales (and zeros).
template <int MTW, int NT, int TW, int WM, int WN, int WK, int STAGES, bool ASYM>
struct PipeCfg {
    static_assert(NT == 1 ? (TW == 1 || TW == 2 || TW == 4 || TW == 8) && WM == 1 : TW == 8 * NT,
                  "tokens per warp");
    static constexpr int kThreads = WM * WN * WK * 32;
    static constexpr int kBN = 16 * MTW * WN;
    static constexpr int kBM = TW * WM;
    static constexpr int kKS = kG * WK;
    static constexpr int kABytes = kBM * kKS * 2;
    static constexpr int kBBytes = WN * WK * MTW * 1024;
    static constexpr int kSBytes = WK * kBN * 2;
    static constexpr int kZBytes = ASYM ? WK * kBN : 0;
    static constexpr int kStage = kABytes + kBBytes + kSBytes + kZBytes;
    static constexpr int kPipe = STAGES * kStage;
    static constexpr int kRed = WK > 1 ? kThreads * MTW * NT * 4 * 4 : 0;  // fp32 partials
    static constexpr int kSmem = kPipe > kRed ? kPipe : kRed;
};

// Physical 16-byte chunk of logical chunk `ch` in activation row `row` of a stage. A lane
// reads chunk 4i + c of row g (w4gemm_internal.cuh), so the eight lanes of a quarter warp
// touch chunks 4i..4i+3 of two adjacent rows; flipping bit 2 on odd rows puts those eight
// in eight different 16-byte bank groups.
__device__ __forceinline__ int swz(int row, int ch) {
    return ch ^ ((row & 1) << 2);
}

template <int MTW, int NT, int TW, int WM, int WN, int WK, int STAGES, bool ASYM>
__global__ void __launch_bounds__(WM * WN * WK * 32) w4_pipe_kernel(PipeParams p) {
    using C_ = PipeCfg<MTW, NT, TW, WM, WN, WK, STAGES, ASYM>;
    constexpr int THREADS = C_::kThreads, BN = C_::kBN, BM = C_::kBM, KS = C_::kKS;
    constexpr int CPR = KS / 8;  // 16-byte chunks per activation row of a stage
    constexpr int E = MTW * NT * 4;

    extern __shared__ __align__(128) unsigned char smem[];
    __shared__ int s_last;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wn = warp % WN;
    const int wm = (warp / WN) % WM;
    const int wk = warp / (WN * WM);
    const int g = lane >> 2, c = lane & 3;
    const int M = p.M, N = p.N, K = p.K;
    const int G = K / kG;
    const int units = K / 64;
    const int tiles = (N / BN) * p.chunks;
    int it0, it1;
    if (p.split > 0) {
        const int tile = blockIdx.x % tiles, slice = blockIdx.x / tiles;
        it0 = tile * G + static_cast<int>(static_cast<long long>(slice) * G / p.split);
        it1 = tile * G + static_cast<int>(static_cast<long long>(slice + 1) * G / p.split);
    } else {
        const long long total = static_cast<long long>(tiles) * G;
        it0 = static_cast<int>(total * blockIdx.x / gridDim.x);
        it1 = static_cast<int>(total * (blockIdx.x + 1) / gridDim.x);
    }

    // A stage holds up to WK groups of one tile: a cursor moves by min(WK, groups left in the
    // tile and in the range). The load cursor runs STAGES - 1 stages ahead of the compute one
    // with the same rule; neither divides after the start.
    struct Cursor {
        int it, tile, g, len;
    };
    auto start = [&](int it) {
        Cursor cu{it, it / G, it % G, 0};
        cu.len = min(WK, min(it1 - it, G - cu.g));
        return cu;
    };
    auto advance = [&](Cursor& cu) {
        cu.it += cu.len;
        cu.g += cu.len;
        if (cu.g == G) {
            cu.g = 0;
            ++cu.tile;
        }
        cu.len = min(WK, min(it1 - cu.it, G - cu.g));
    };
    auto tile_origin = [&](int tile, int& m0, int& n0) {
        const int strip = p.chunks == 1 ? tile : tile / p.chunks;
        m0 = (tile - strip * p.chunks) * BM;
        n0 = strip * BN;
    };

    auto a_stage = [&](int s) { return smem + s * C_::kStage; };
    auto b_stage = [&](int s) { return smem + s * C_::kStage + C_::kABytes; };
    auto s_stage = [&](int s) { return smem + s * C_::kStage + C_::kABytes + C_::kBBytes; };
    auto z_stage = [&](int s) {
        return smem + s * C_::kStage + C_::kABytes + C_::kBBytes + C_::kSBytes;
    };

    // Activation rows past M are not loaded. Their smem holds whatever the slot held before,
    // which only reaches the mma columns of tokens that are never stored: a token's outputs
    // depend on its own row alone.
    auto load_stage = [&](int s, const Cursor& cu) {
        int m0, n0;
        tile_origin(cu.tile, m0, n0);
        const int rows = min(BM, M - m0), gs = cu.g, gcount = cu.len;
        unsigned char* as = a_stage(s);
        for (int i = tid; i < rows * CPR; i += THREADS) {
            const int row = i / CPR, ch = i % CPR;
            if (ch < gcount * 16)
                cp_async_16(as + (row * CPR + swz(row, ch)) * 16,
                            p.A + static_cast<size_t>(m0 + row) * K + gs * kG + ch * 8);
        }
        // Weights: cell (wn, wk, mt) is the two 512 B blocks of group gs + wk in strip
        // n0 / 16 + wn MTW + mt.
        unsigned char* bs = b_stage(s);
        constexpr int BCHUNKS = WN * WK * MTW * 64;
        for (int i = tid; i < BCHUNKS; i += THREADS) {
            const int cell = i / 64, within = i % 64;
            const int mt = cell % MTW, cwk = (cell / MTW) % WK, cwn = cell / (MTW * WK);
            if (cwk < gcount) {
                const int t = n0 / 16 + cwn * MTW + mt;
                cp_async_16(bs + i * 16,
                            p.B + w4::block_index(t, 2 * (gs + cwk), units) * 32 + within);
            }
        }
        unsigned char* ss = s_stage(s);
        constexpr int SCH = BN / 8;  // 16-byte chunks per row of BN bf16 scales
        for (int i = tid; i < gcount * SCH; i += THREADS)
            cp_async_16(ss + i * 16,
                        p.S + static_cast<size_t>(gs + i / SCH) * N + n0 + (i % SCH) * 8);
        if constexpr (ASYM) {
            unsigned char* zs = z_stage(s);
            constexpr int ZCH = BN / 16;
            for (int i = tid; i < gcount * ZCH; i += THREADS)
                cp_async_16(zs + i * 16,
                            p.Z + static_cast<size_t>(gs + i / ZCH) * N + n0 + (i % ZCH) * 16);
        }
    };

    float acc[MTW][NT][4];
    auto zero_acc = [&] {
#pragma unroll
        for (int i = 0; i < MTW; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;
    };
    zero_acc();

    Cursor ld = start(it0);
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (ld.it < it1) {
            load_stage(s, ld);
            advance(ld);
        }
        cp_async_commit();
    }

    // Store (or add into the workspace) element e of lane ln of the warps at (wm_, wn_).
    auto emit = [&](int wm_, int wn_, int e, int ln, float v, int m0, int n0, bool full) {
        int dn, dm;
        element_coords<NT>(e, ln, dn, dm);
        const int m = m0 + wm_ * TW + dm;
        if (dm >= TW || m >= M) return;
        const size_t idx = static_cast<size_t>(m) * N + n0 + wn_ * MTW * 16 + dn;
        if (full)
            p.C[idx] = __float2bfloat16(v);
        else
            atomicAdd(p.fix.ws + idx, v);
    };
    // The end of a tile's run of groups in this block: store it, or add it into the
    // workspace and let the block that completes the tile's group count convert it.
    auto flush = [&](int tile, int it_begin, int it_end) {
        int m0, n0;
        tile_origin(tile, m0, n0);
        const bool full = it_begin == tile * G && it_end == (tile + 1) * G;
        if constexpr (WK > 1) {
            // Only the split > 0 schedule runs WK > 1, and then this is the block's last
            // stage: the pipeline smem is free once the (empty) groups in flight are waited.
            cp_async_wait<0>();
            __syncthreads();
            float* red = reinterpret_cast<float*>(smem);
#pragma unroll
            for (int mt = 0; mt < MTW; ++mt)
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
#pragma unroll
                    for (int q = 0; q < 4; ++q)
                        red[((warp * E) + (mt * NT + nt) * 4 + q) * 32 + lane] = acc[mt][nt][q];
            __syncthreads();
            for (int i = tid; i < WM * WN * E * 32; i += THREADS) {
                const int ln = i & 31, e = (i >> 5) % E, wmn = (i >> 5) / E;
                float v = 0.f;
#pragma unroll
                for (int k = 0; k < WK; ++k) v += red[(((k * WM * WN + wmn) * E) + e) * 32 + ln];
                emit(wmn / WN, wmn % WN, e, ln, v, m0, n0, full);
            }
        } else {
#pragma unroll
            for (int e = 0; e < E; ++e)
                emit(wm, wn, e, lane, acc[e / (4 * NT)][(e / 4) % NT][e % 4], m0, n0, full);
        }
        zero_acc();
        if (!full)
            fixup_tile(p.fix, p.C, N, tile, it_end - it_begin, G, m0, min(BM, M - m0), n0, BN,
                       &s_last);
    };

    int seg_begin = it0;
    int s = 0;  // ring slot of the compute cursor
    for (Cursor cu = start(it0); cu.it < it1;) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (ld.it < it1) {
            load_stage(s == 0 ? STAGES - 1 : s - 1, ld);
            advance(ld);
        }
        cp_async_commit();  // always, so the group count stays uniform
        if (wk < cu.len) {
            const unsigned char* as = a_stage(s);
            const uint4* bs =
                reinterpret_cast<const uint4*>(b_stage(s)) + ((wn * WK + wk) * MTW) * 64 + lane;
            const unsigned short* ss =
                reinterpret_cast<const unsigned short*>(s_stage(s)) + wk * BN + wn * MTW * 16 + g;
            // Activations: lane (g, c) of token tile nt reads chunk 16 wk + 4i + c of row
            // wm TW + 8 nt + g for steps 2i and 2i + 1 (w4gemm_internal.cuh); rows past TW
            // are zeros.
            uint4 act[NT][4];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                const int tw = nt * 8 + g;
                const int row = wm * TW + tw;
#pragma unroll
                for (int i = 0; i < 4; ++i)
                    act[nt][i] = tw < TW
                                     ? *reinterpret_cast<const uint4*>(
                                           as + (row * CPR + swz(row, wk * 16 + 4 * i + c)) * 16)
                                     : make_uint4(0, 0, 0, 0);
            }
#pragma unroll
            for (int mt = 0; mt < MTW; ++mt) {
                RowPairs rp{w4::splat(ss[mt * 16]), w4::splat(ss[mt * 16 + 8]), w4::kSymZeroPair,
                            w4::kSymZeroPair};
                if constexpr (ASYM) {
                    const unsigned char* zs = z_stage(s) + wk * BN + wn * MTW * 16 + g;
                    rp.zz_lo = w4::zero_pair(zs[mt * 16]);
                    rp.zz_hi = w4::zero_pair(zs[mt * 16 + 8]);
                }
                const uint4 w[2] = {bs[mt * 64], bs[mt * 64 + 32]};
                group_mma<NT>(acc[mt], w, rp, act);
            }
        }
        const int tile = cu.tile;
        const bool seg_done = cu.g + cu.len == G || cu.it + cu.len == it1;
        advance(cu);
        s = s + 1 == STAGES ? 0 : s + 1;
        if (seg_done) {
            flush(tile, seg_begin, cu.it);
            seg_begin = cu.it;
        }
    }
    cp_async_wait<0>();
}

// ---- host side ---------------------------------------------------------------------------

struct Args {
    const __nv_bfloat16* A;
    const uint4* B;
    const __nv_bfloat16* S;
    const uint8_t* Z;
    __nv_bfloat16* C;
    int M, N, K;
    cudaStream_t stream;
};

// The workspace and counters of the Fixup: M x N fp32 and one counter per tile, zeroed when
// allocated and left zero by every launch. One per process, grown on demand.
Fixup workspace(size_t floats, size_t count) {
    static float* ws = nullptr;
    static int* counters = nullptr;
    static size_t have_floats = 0, have_count = 0;
    if (have_floats < floats) {
        if (ws) SPARK_CUDA_CHECK(cudaFree(ws));
        have_floats = std::max(floats, have_floats * 2);
        SPARK_CUDA_CHECK(cudaMalloc(&ws, have_floats * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMemset(ws, 0, have_floats * sizeof(float)));
    }
    if (have_count < count) {
        if (counters) SPARK_CUDA_CHECK(cudaFree(counters));
        have_count = std::max(count, have_count * 2);
        SPARK_CUDA_CHECK(cudaMalloc(&counters, have_count * sizeof(int)));
        SPARK_CUDA_CHECK(cudaMemset(counters, 0, have_count * sizeof(int)));
    }
    return {ws, counters};
}

// The small-M kernel: `split` K slices per (strip, chunk), 0 = pick one so that the grid
// reaches kMinBlocks blocks.
constexpr int kMinBlocks = 128;

template <int MTW, int NT, int TW, int WK, int D, bool ASYM>
int warp_slots() {
    static int resident = 0;
    if (resident == 0) {
        int per_sm = 0;
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &per_sm, w4_warp_kernel<MTW, NT, TW, WK, D, ASYM>, WK * 32, 0));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

template <int MTW, int NT, int TW, int WK, int D, bool ASYM>
void launch_warp(const Args& a, int split) {
    constexpr int BN = 16 * MTW;
    const int strips = a.N / BN;
    const int chunks = cdiv(a.M, TW);
    const int G = a.K / kG;
    WarpParams p;
    p.A = a.A;
    p.B = a.B;
    p.S = a.S;
    p.Z = a.Z;
    p.C = a.C;
    p.M = a.M;
    p.N = a.N;
    p.K = a.K;
    p.chunks = chunks;
    if (split <= 0) split = cdiv(kMinBlocks, strips * chunks);
    p.split = std::max(1, std::min(split, G));
    p.fix = Fixup{nullptr, nullptr};
    if (p.split > 1)
        p.fix = workspace(static_cast<size_t>(a.M) * a.N, static_cast<size_t>(strips) * chunks);
    w4_warp_kernel<MTW, NT, TW, WK, D, ASYM>
        <<<strips * chunks * p.split, WK * 32, 0, a.stream>>>(p);
    SPARK_CHECK_LAUNCH();
}

template <int MTW, int NT, int TW, int WM, int WN, int WK, int STAGES, bool ASYM>
int pipe_slots() {
    using C_ = PipeCfg<MTW, NT, TW, WM, WN, WK, STAGES, ASYM>;
    auto* fn = w4_pipe_kernel<MTW, NT, TW, WM, WN, WK, STAGES, ASYM>;
    static int resident = 0;
    if (resident == 0) {
        SPARK_CUDA_CHECK(
            cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, C_::kSmem));
        int per_sm = 0;
        SPARK_CUDA_CHECK(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, fn, C_::kThreads, C_::kSmem));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

// The pipelined kernel: `split` K slices per tile, or 0 for Stream-K over every resident
// block (WK = 1 only; with warps splitting K a block runs one tile, and split 0 picks the
// slices that bring the grid to kMinBlocks).
template <int MTW, int NT, int TW, int WM, int WN, int WK, int STAGES, bool ASYM>
void launch_pipe(const Args& a, int split) {
    using C_ = PipeCfg<MTW, NT, TW, WM, WN, WK, STAGES, ASYM>;
    const int resident = pipe_slots<MTW, NT, TW, WM, WN, WK, STAGES, ASYM>();
    const int chunks = cdiv(a.M, C_::kBM);
    const int tiles = (a.N / C_::kBN) * chunks;
    const int G = a.K / kG;
    PipeParams p;
    p.A = a.A;
    p.B = a.B;
    p.S = a.S;
    p.Z = a.Z;
    p.C = a.C;
    p.M = a.M;
    p.N = a.N;
    p.K = a.K;
    p.chunks = chunks;
    int grid;
    if (split > 0 || WK > 1) {
        if (split <= 0) split = cdiv(kMinBlocks, tiles);
        p.split = std::max(1, std::min(split, G));
        grid = tiles * p.split;
    } else {
        p.split = 0;
        grid = static_cast<int>(std::min<long long>(resident, static_cast<long long>(tiles) * G));
    }
    p.fix = workspace(static_cast<size_t>(a.M) * a.N, static_cast<size_t>(tiles));
    w4_pipe_kernel<MTW, NT, TW, WM, WN, WK, STAGES, ASYM>
        <<<grid, C_::kThreads, C_::kSmem, a.stream>>>(p);
    SPARK_CHECK_LAUNCH();
}

// One entry per block shape of variant 2: the columns and tokens one block covers, whether
// it is a Stream-K grid (which always fits the resident blocks), its resident blocks per GPU
// and its launcher for symmetric and asymmetric weights.
struct Entry {
    int bn, bm;
    bool streamk;
    int (*slots)();
    void (*sym)(const Args&, int);
    void (*asym)(const Args&, int);
};
template <int MTW, int NT, int TW, int WK, int D>
constexpr Entry warp_entry() {
    return {16 * MTW,
            TW,
            false,
            &warp_slots<MTW, NT, TW, WK, D, true>,
            &launch_warp<MTW, NT, TW, WK, D, false>,
            &launch_warp<MTW, NT, TW, WK, D, true>};
}
template <int MTW, int NT, int TW, int WM, int WN, int WK, int STAGES>
constexpr Entry pipe_entry() {
    return {16 * MTW * WN,
            TW * WM,
            WK == 1,
            &pipe_slots<MTW, NT, TW, WM, WN, WK, STAGES, true>,
            &launch_pipe<MTW, NT, TW, WM, WN, WK, STAGES, false>,
            &launch_pipe<MTW, NT, TW, WM, WN, WK, STAGES, true>};
}

const Entry kEntries[] = {
    // independent warps, registers only: (MTW, NT, TW, WK, D)
    warp_entry<1, 1, 8, 4, 4>(),   // 0: M <= 8, when the pipelined blocks do not fit
    warp_entry<1, 2, 16, 4, 4>(),  // 1: M <= 16
    warp_entry<2, 2, 16, 8, 2>(),  // 2: M <= 16, long K
    warp_entry<1, 1, 8, 1, 4>(),   // 3: M <= 8, one warp per strip (wide N)
    warp_entry<1, 2, 16, 1, 4>(),  // 4: M <= 16, one warp per strip (wide N)
    // pipelined, warps along K: (MTW, NT, TW, WM, WN, WK, STAGES)
    pipe_entry<1, 1, 1, 1, 1, 4, 6>(),  // 5: M = 1
    pipe_entry<1, 1, 4, 1, 1, 4, 4>(),  // 6: M <= 4
    pipe_entry<1, 1, 8, 1, 1, 4, 4>(),  // 7: M <= 8
    // pipelined, warps along N and M, Stream-K
    pipe_entry<2, 2, 16, 2, 4, 1, 4>(),  // 8: 128 x 32
    pipe_entry<1, 4, 32, 1, 4, 1, 4>(),  // 9: 64 x 32
    pipe_entry<2, 4, 32, 1, 4, 1, 3>(),  // 10: 128 x 32
    pipe_entry<1, 4, 32, 2, 4, 1, 3>(),  // 11: 64 x 64
    pipe_entry<2, 4, 32, 2, 4, 1, 3>(),  // 12: 128 x 64
};
constexpr int kNumEntries = sizeof(kEntries) / sizeof(kEntries[0]);

// Tuning hook: SPARK_W4_CFG="entry[,split]" forces a block shape (and the K split of the
// one-tile-per-block shapes).
// Read once per process: the sweeps of docs/design/w4gemm.md were run through it.
bool env_override(int& idx, int& split) {
    static const char* e = std::getenv("SPARK_W4_CFG");
    if (!e || !*e) return false;
    idx = std::atoi(e);
    const char* comma = std::strchr(e, ',');
    split = comma ? std::atoi(comma + 1) : 0;
    return idx >= 0 && idx < kNumEntries;
}

// Blocks of one launch of entry e without a forced split (Stream-K grids always fit).
int blocks_of(const Entry& e, const Args& a) {
    const int tiles = (a.N / e.bn) * cdiv(a.M, e.bm);
    return tiles * std::max(1, cdiv(kMinBlocks, tiles));
}

// The block shape for a call, from the measurements of docs/design/w4gemm.md: a preferred
// shape per token count, used when its blocks fit in one wave of resident blocks (384 blocks
// on 340 slots would run a second wave for 44 of them), else a fallback that is faster when
// neither fits; either must divide N, and 16-column blocks divide every legal N.
int pick(const Args& a) {
    const int G = a.K / kG;
    int pref, fallback;
    // Four one-warp strips per SM already keep the weights streaming: no K split at all.
    const bool wide = a.N / 16 >= 4 * num_sms();
    if (a.M <= 1) {
        pref = wide ? 3 : 5;  // pipelined, 4 warps along K, one token row staged
        fallback = wide ? 3 : 0;
    } else if (a.M <= 8) {
        pref = wide ? 3 : 7;      // pipelined, 4 warps along K, 8 rows staged
        fallback = wide ? 3 : 0;  // independent warps, 8 tokens
    } else if (a.M <= 16) {
        pref = fallback = wide ? 4 : G >= 64 ? 2 : 1;  // long K: 8 warps along K
    } else {
        pref = fallback = a.M <= 32 ? 9 : 10;  // Stream-K tiles
    }
    const Entry& e = kEntries[pref];
    if (a.N % e.bn == 0 && (e.streamk || blocks_of(e, a) <= e.slots())) return pref;
    if (a.N % kEntries[fallback].bn == 0) return fallback;
    return a.M <= 8 ? 0 : 1;
}

void launch_v2(const Args& a) {
    int idx = 0, split = 0;
    if (!env_override(idx, split)) idx = pick(a);
    const Entry& e = kEntries[idx];
    SPARK_REQUIRE(a.N % e.bn == 0, "w4gemm: N is not a multiple of the block width");
    (a.Z ? e.asym : e.sym)(a, split);
}

}  // namespace

bool w4gemm_supports(int M, int N, int K, int variant) {
    return variant >= 0 && variant < w4gemm_num_variants() && M >= 1 && N > 0 && K > 0 &&
           N % 16 == 0 && K % kG == 0;
}

int w4gemm_num_variants() {
    return 3;
}

void w4gemm_bf16(const __nv_bfloat16* A, const int32_t* packed, const __nv_bfloat16* scales,
                 const uint8_t* zeros, __nv_bfloat16* C, int M, int N, int K, int variant,
                 cudaStream_t stream) {
    SPARK_REQUIRE(variant >= 0 && variant < w4gemm_num_variants(), "w4gemm: bad variant");
    SPARK_REQUIRE(w4gemm_supports(M, N, K, variant),
                  "w4gemm: needs M >= 1, N % 16 == 0 and K % 128 == 0");
    SPARK_REQUIRE(A && packed && scales && C, "w4gemm: null pointer");
    SPARK_REQUIRE(is_aligned16(A) && is_aligned16(packed) && is_aligned16(scales) &&
                      (zeros == nullptr || is_aligned16(zeros)),
                  "w4gemm: A, packed, scales and zeros must be 16-byte aligned");
    const auto* pk = reinterpret_cast<const uint4*>(packed);
    switch (variant) {
        case 0: {
            const dim3 grid(cdiv(N, 128), M);
            w4_naive_kernel<<<grid, 128, 0, stream>>>(A, reinterpret_cast<const uint32_t*>(packed),
                                                      scales, zeros, C, M, N, K);
            SPARK_CHECK_LAUNCH();
            break;
        }
        case 1: {
            const dim3 grid(cdiv(N / 16, 4), cdiv(M, 8));
            w4_reg_kernel<<<grid, 128, 0, stream>>>(A, pk, scales, zeros, C, M, N, K);
            SPARK_CHECK_LAUNCH();
            break;
        }
        default:
            launch_v2(Args{A, pk, scales, zeros, C, M, N, K, stream});
    }
}

}  // namespace spark
