// sgemm variants 6 and 7: the fp32 GEMM on the tensor cores (docs/design/sgemm.md).
//
//   variant 6, TF32:   each fp32 operand is rounded to tf32 (11 significant bits, cvt.rna)
//                      as its fragment is loaded and one mma.sync.m16n8k8 per product does
//                      the rest. The precision contract of cuBLAS under
//                      CUBLAS_TF32_TENSOR_OP_MATH and of torch with allow_tf32: 2^-11
//                      relative error per operand.
//   variant 7, 3xTF32: each operand is split as x = big + small with big = tf32(x) and
//                      small = tf32(x - big); big*big + big*small + small*big is
//                      accumulated, three mmas per product. The dropped small*small term
//                      and the split residual are 2^-22 relative: fp32 class.
//
// Both run the cp.async block tile of hgemm variant 3 (hgemm_tile.cuh) on 32-bit elements:
// BM x 128 x BK, 8 warps as 2 (M) x 4 (N), a (BM/2) x 32 warp tile of m16n8 accumulators,
// XOR-swizzled shared memory, a STAGES-deep cp.async ring, a register epilogue, and split-K
// over the tiles of the last partial wave with fp32 atomics into a zeroed C (variants 4 and
// 5 do the same). Rows past M are zero-filled on the way in. Any M, N, K >= 1: a problem
// whose rows are not 16-byte multiples (N % 4 or K % 4 != 0) or whose pointers are not
// 16-byte aligned takes a guarded register fill instead of cp.async and scalar stores.
//
// The B fragment is read with one 128-bit LDS per lane instead of eight 32-bit ones by
// permuting the columns each n8 tile of a warp owns: mma n-tile j, column g of the warp's
// 32-wide strip holds global column n0 + 4g + j, so lane (g, c) reads the float4 at
// Bs[k][n0 + 4g] and gets b0 of all four tiles at once. The permutation is undone in the
// epilogue, where it turns the two fp32 per lane per tile into two float4 stores.

#include <algorithm>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {
namespace tf32_tile {

constexpr int THREADS = 256;

// Physical 16-byte chunk for logical (row, chunk) of an A tile whose rows are BK fp32 long.
// Rows of 64 B (BK=16): chunk ^= (row/2)%4; rows of 128 B (BK=32): chunk ^= row%8. The
// eight rows an ldmatrix matrix touches then sit in eight different bank groups.
template <int BK>
__device__ __forceinline__ int swz_a(int row, int chunk) {
    if constexpr (BK == 16)
        return chunk ^ ((row >> 1) & 3);
    else
        return chunk ^ (row & 7);
}
// B rows are 512 B (BN = 128 fp32). A lane quartet reads k rows 4h + c, c = 0..3, of two
// neighbouring chunks (2g, 2g + 1) per LDS.128 phase; XOR-ing the chunk with 2 (k % 8)
// spreads those eight accesses over the eight bank groups.
__device__ __forceinline__ int swz_b(int krow, int chunk) {
    return chunk ^ ((krow & 7) << 1);
}

// Eight warps as WARPS_M x WARPS_N over a BM x BN block tile.
template <int BM_, int BN_, int BK_, int STAGES_, int WARPS_M_ = 2, int WARPS_N_ = 4>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_;
    static constexpr int WARPS_M = WARPS_M_, WARPS_N = WARPS_N_;
    static_assert(WARPS_M * WARPS_N * 32 == THREADS, "eight warps");
    static_assert(BK == 16 || BK == 32, "swizzle assumes 64 B or 128 B rows of A");
    static_assert(BN == 128, "B rows are 512 B: the swizzle and the raster assume BN = 128");
    static constexpr int WM = BM / WARPS_M;  // 32, 64 or 128
    static constexpr int WN = BN / WARPS_N;  // 32 or 64
    static_assert(WM % 16 == 0 && WN % 32 == 0,
                  "the B fragment load is one float4 per lane per group of four n8 tiles");
    static constexpr int MT = WM / 16;    // m16 tiles per warp
    static constexpr int NT = WN / 8;     // n8 tiles per warp: 4 or 8
    static constexpr int NQ = NT / 4;     // float4 B loads per k half-step
    static constexpr int A_CPR = BK / 4;  // 16-byte chunks per smem row of A
    static constexpr int B_CPR = BN / 4;
    static constexpr int A_ITERS = (BM * A_CPR) / THREADS;  // chunks per thread per stage
    static constexpr int B_ITERS = (BK * B_CPR) / THREADS;
    static_assert(A_ITERS * THREADS == BM * A_CPR && B_ITERS * THREADS == BK * B_CPR, "");
    static constexpr int A_STAGE = BM * BK;  // elements
    static constexpr int B_STAGE = BK * BN;
    static constexpr int SMEM = STAGES * (A_STAGE + B_STAGE) * static_cast<int>(sizeof(float));
    using Acc = float[MT][NT][4];
};

// Global -> shared copy of one BK-wide K-slab into `as` / `bs`. ALIGNED: every 16-byte chunk
// is in or out of the matrix as a whole (K % 4 == 0, N % 4 == 0, aligned bases), so it is a
// cp.async with zero-fill for the ones out. Otherwise: guarded scalar loads through registers
// and a plain shared store (the stage is free, see the barrier in the mainloop).
template <class C, bool ALIGNED>
__device__ __forceinline__ void load_stage(const float* __restrict__ A, const float* __restrict__ B,
                                           int M, int N, int K, int bm, int bn, int k0, float* as,
                                           float* bs) {
    constexpr int BK = C::BK, BN = C::BN;
    const int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < C::A_ITERS; ++i) {
        const int c = tid + i * THREADS;
        const int row = c / C::A_CPR;
        const int ch = c % C::A_CPR;
        const int grow = bm + row;
        const int gk = k0 + ch * 4;
        float* dst = as + row * BK + swz_a<BK>(row, ch) * 4;
        if constexpr (ALIGNED) {
            const bool ok = grow < M && gk < K;
            cp_async_16_zfill(dst, A + (ok ? static_cast<size_t>(grow) * K + gk : 0), ok);
        } else {
            float v[4];
#pragma unroll
            for (int e = 0; e < 4; ++e)
                v[e] = (grow < M && gk + e < K) ? A[static_cast<size_t>(grow) * K + gk + e] : 0.f;
            *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
        }
    }
#pragma unroll
    for (int i = 0; i < C::B_ITERS; ++i) {
        const int c = tid + i * THREADS;
        const int row = c / C::B_CPR;
        const int ch = c % C::B_CPR;
        const int gk = k0 + row;
        const int gcol = bn + ch * 4;
        float* dst = bs + row * BN + swz_b(row, ch) * 4;
        if constexpr (ALIGNED) {
            const bool ok = gk < K && gcol < N;
            cp_async_16_zfill(dst, B + (ok ? static_cast<size_t>(gk) * N + gcol : 0), ok);
        } else {
            float v[4];
#pragma unroll
            for (int e = 0; e < 4; ++e)
                v[e] = (gk < K && gcol + e < N) ? B[static_cast<size_t>(gk) * N + gcol + e] : 0.f;
            *reinterpret_cast<float4*>(dst) = make_float4(v[0], v[1], v[2], v[3]);
        }
    }
}

// x = big + small: big is x rounded to tf32; the residual x - big is exact in fp32 (it has
// at most 13 significant bits) and small is that residual rounded to tf32, so
// |x - big - small| <= 2^-22 |x|. Variant 6 keeps only big.
template <bool THREE>
__device__ __forceinline__ void split_tf32(float x, unsigned& big, unsigned& small) {
    big = f32_to_tf32(x);
    if constexpr (THREE) small = f32_to_tf32(x - __uint_as_float(big));
}

// Accumulates A[bm.., kt_begin*BK .. (kt_begin+nkt)*BK) x B[same K, bn..] into `acc`, which
// it zeroes first. The pipeline is drained on return.
template <class C, bool THREE, bool ALIGNED>
__device__ __forceinline__ void mainloop(const float* __restrict__ A, const float* __restrict__ B,
                                         int M, int N, int K, int bm, int bn, int kt_begin, int nkt,
                                         unsigned char* smem, typename C::Acc& acc) {
    constexpr int BK = C::BK, BN = C::BN, STAGES = C::STAGES;
    constexpr int WM = C::WM, WN = C::WN, MT = C::MT, NT = C::NT, NQ = C::NQ;
    constexpr int A_STAGE = C::A_STAGE, B_STAGE = C::B_STAGE;

    float* As = reinterpret_cast<float*>(smem);
    float* Bs = As + STAGES * A_STAGE;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int wm = warp / C::WARPS_N;
    const int wn = warp % C::WARPS_N;

#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    // Prologue: the first STAGES-1 slabs are in flight before any compute starts.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nkt)
            load_stage<C, ALIGNED>(A, B, M, N, K, bm, bn, (kt_begin + s) * BK, As + s * A_STAGE,
                                   Bs + s * B_STAGE);
        cp_async_commit();
    }

    // ldmatrix.x4 on b16 pairs: matrix i (lanes 8i..8i+7 give its row addresses) is rows
    // 0-7 / 8-15 of the m16 tile at k chunk 0 / 1 of the k8 step, so register i lands as
    // a[i] of the m16n8k8 fragment: (row g, k c), (row g+8, k c), (row g, k c+4), (row g+8,
    // k c+4) with g = lane/4, c = lane%4. Each lane's 32-bit register is one tf32.
    const int a_row_in_tile = lane & 15;
    const int a_kchunk = lane >> 4;
    // B: lane (g, c) reads the float4 at k row c (b0) or c+4 (b1), chunk 8 q + g of the
    // warp's strip, for each group q of four n8 tiles.
    const int b_tig = lane & 3;
    const int b_chunk = wn * (WN / 4) + (lane >> 2);

    for (int kt = 0; kt < nkt; ++kt) {
        cp_async_wait<STAGES - 2>();  // slab kt has landed (for this thread)
        __syncthreads();              // ... for every thread; and stage (kt-1)%STAGES is free
        {
            const int nk = kt + STAGES - 1;
            if (nk < nkt)
                load_stage<C, ALIGNED>(A, B, M, N, K, bm, bn, (kt_begin + nk) * BK,
                                       As + (nk % STAGES) * A_STAGE, Bs + (nk % STAGES) * B_STAGE);
            cp_async_commit();  // always commit so the group count stays uniform
        }
        const float* as = As + (kt % STAGES) * A_STAGE;
        const float* bs = Bs + (kt % STAGES) * B_STAGE;

#pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            unsigned abig[MT][4], asml[MT][4];
            unsigned bbig[NT][2], bsml[NT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = wm * WM + mi * 16 + a_row_in_tile;
                const int ch = kk / 4 + a_kchunk;
                unsigned raw[4];
                ldmatrix_x4(raw, as + row * BK + swz_a<BK>(row, ch) * 4);
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    split_tf32<THREE>(__uint_as_float(raw[e]), abig[mi][e], asml[mi][e]);
            }
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int krow = kk + h * 4 + b_tig;
                    const float4 v = *reinterpret_cast<const float4*>(
                        bs + krow * BN + swz_b(krow, b_chunk + 8 * q) * 4);
                    split_tf32<THREE>(v.x, bbig[4 * q][h], bsml[4 * q][h]);
                    split_tf32<THREE>(v.y, bbig[4 * q + 1][h], bsml[4 * q + 1][h]);
                    split_tf32<THREE>(v.z, bbig[4 * q + 2][h], bsml[4 * q + 2][h]);
                    split_tf32<THREE>(v.w, bbig[4 * q + 3][h], bsml[4 * q + 3][h]);
                }
            }
            // 3xTF32: the two small cross terms first, the big product last, so the small
            // corrections are not swallowed by fp32 accumulation of a partial sum that is
            // already 2^24 times larger than they are.
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
#pragma unroll
                for (int nj = 0; nj < NT; ++nj) {
                    if constexpr (THREE) {
                        mma_tf32_1688(acc[mi][nj], asml[mi], bbig[nj]);
                        mma_tf32_1688(acc[mi][nj], abig[mi], bsml[nj]);
                    }
                    mma_tf32_1688(acc[mi][nj], abig[mi], bbig[nj]);
                }
        }
    }
    cp_async_wait<0>();
}

// The same slab through registers instead of cp.async: guarded loads into `pa` / `pb`.
template <class C, bool ALIGNED>
__device__ __forceinline__ void gload(const float* __restrict__ A, const float* __restrict__ B,
                                      int M, int N, int K, int bm, int bn, int k0,
                                      float (&pa)[C::A_ITERS][4], float (&pb)[C::B_ITERS][4]) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < C::A_ITERS; ++i) {
        const int c = tid + i * THREADS;
        const int grow = bm + c / C::A_CPR;
        const int gk = k0 + (c % C::A_CPR) * 4;
        if constexpr (ALIGNED) {
            const float4 v =
                (grow < M && gk < K)
                    ? *reinterpret_cast<const float4*>(A + static_cast<size_t>(grow) * K + gk)
                    : make_float4(0.f, 0.f, 0.f, 0.f);
            pa[i][0] = v.x, pa[i][1] = v.y, pa[i][2] = v.z, pa[i][3] = v.w;
        } else {
#pragma unroll
            for (int e = 0; e < 4; ++e)
                pa[i][e] =
                    (grow < M && gk + e < K) ? A[static_cast<size_t>(grow) * K + gk + e] : 0.f;
        }
    }
#pragma unroll
    for (int i = 0; i < C::B_ITERS; ++i) {
        const int c = tid + i * THREADS;
        const int gk = k0 + c / C::B_CPR;
        const int gcol = bn + (c % C::B_CPR) * 4;
        if constexpr (ALIGNED) {
            const float4 v =
                (gk < K && gcol < N)
                    ? *reinterpret_cast<const float4*>(B + static_cast<size_t>(gk) * N + gcol)
                    : make_float4(0.f, 0.f, 0.f, 0.f);
            pb[i][0] = v.x, pb[i][1] = v.y, pb[i][2] = v.z, pb[i][3] = v.w;
        } else {
#pragma unroll
            for (int e = 0; e < 4; ++e)
                pb[i][e] =
                    (gk < K && gcol + e < N) ? B[static_cast<size_t>(gk) * N + gcol + e] : 0.f;
        }
    }
}

// Round the prefetched slab to tf32 and store it, swizzled, into stage (`as`, `bs`). Every
// element is converted exactly once here, by the thread that fetched it; the fragment loads
// below then feed the mma straight from ldmatrix / LDS, as the bf16 tile does.
template <class C>
__device__ __forceinline__ void sstore_tf32(const float (&pa)[C::A_ITERS][4],
                                            const float (&pb)[C::B_ITERS][4], float* as,
                                            float* bs) {
    const int tid = threadIdx.x;
#pragma unroll
    for (int i = 0; i < C::A_ITERS; ++i) {
        const int c = tid + i * THREADS;
        const int row = c / C::A_CPR;
        const int ch = c % C::A_CPR;
        *reinterpret_cast<uint4*>(as + row * C::BK + swz_a<C::BK>(row, ch) * 4) =
            make_uint4(f32_to_tf32(pa[i][0]), f32_to_tf32(pa[i][1]), f32_to_tf32(pa[i][2]),
                       f32_to_tf32(pa[i][3]));
    }
#pragma unroll
    for (int i = 0; i < C::B_ITERS; ++i) {
        const int c = tid + i * THREADS;
        const int row = c / C::B_CPR;
        const int ch = c % C::B_CPR;
        *reinterpret_cast<uint4*>(bs + row * C::BN + swz_b(row, ch) * 4) =
            make_uint4(f32_to_tf32(pb[i][0]), f32_to_tf32(pb[i][1]), f32_to_tf32(pb[i][2]),
                       f32_to_tf32(pb[i][3]));
    }
}

// Variant 4's register prefetch on the tensor-core tile (TF32 only). Slab kt+1 is loaded
// into registers before the mmas of slab kt and stored to shared memory, rounded to tf32,
// after them, so the global loads are in flight for a whole slab of tensor work and the
// rounding happens once per element instead of once per fragment (4x for A, 2x for B in
// the cp.async loop, plus the register moves that put converted values into mma operand
// quads). A two-stage ring and one __syncthreads per slab: the barrier at the top of
// iteration kt both publishes the stores of slab kt (made at the end of kt-1) and frees
// the stage those of slab kt+1 overwrite (read during kt-1).
template <class C, bool ALIGNED>
__device__ __forceinline__ void mainloop_prefetch(const float* __restrict__ A,
                                                  const float* __restrict__ B, int M, int N, int K,
                                                  int bm, int bn, int kt_begin, int nkt,
                                                  unsigned char* smem, typename C::Acc& acc) {
    constexpr int BK = C::BK, BN = C::BN, STAGES = C::STAGES;
    constexpr int WM = C::WM, WN = C::WN, MT = C::MT, NT = C::NT, NQ = C::NQ;
    constexpr int A_STAGE = C::A_STAGE, B_STAGE = C::B_STAGE;
    static_assert(STAGES == 2, "one slab in registers, one barrier per slab: a two-stage ring");

    float* As = reinterpret_cast<float*>(smem);
    float* Bs = As + STAGES * A_STAGE;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int wm = warp / C::WARPS_N;
    const int wn = warp % C::WARPS_N;

#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    float pa[C::A_ITERS][4], pb[C::B_ITERS][4];
    if (nkt > 0) {
        gload<C, ALIGNED>(A, B, M, N, K, bm, bn, kt_begin * BK, pa, pb);
        sstore_tf32<C>(pa, pb, As, Bs);
    }

    const int a_row_in_tile = lane & 15;
    const int a_kchunk = lane >> 4;
    const int b_tig = lane & 3;
    const int b_chunk = wn * (WN / 4) + (lane >> 2);

    for (int kt = 0; kt < nkt; ++kt) {
        __syncthreads();
        const bool more = kt + 1 < nkt;
        if (more) gload<C, ALIGNED>(A, B, M, N, K, bm, bn, (kt_begin + kt + 1) * BK, pa, pb);
        const float* as = As + (kt % STAGES) * A_STAGE;
        const float* bs = Bs + (kt % STAGES) * B_STAGE;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 8) {
            unsigned afrag[MT][4];
            unsigned bfrag[NT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = wm * WM + mi * 16 + a_row_in_tile;
                const int ch = kk / 4 + a_kchunk;
                ldmatrix_x4(afrag[mi], as + row * BK + swz_a<BK>(row, ch) * 4);
            }
#pragma unroll
            for (int q = 0; q < NQ; ++q) {
#pragma unroll
                for (int h = 0; h < 2; ++h) {
                    const int krow = kk + h * 4 + b_tig;
                    const uint4 v = *reinterpret_cast<const uint4*>(
                        bs + krow * BN + swz_b(krow, b_chunk + 8 * q) * 4);
                    bfrag[4 * q][h] = v.x;
                    bfrag[4 * q + 1][h] = v.y;
                    bfrag[4 * q + 2][h] = v.z;
                    bfrag[4 * q + 3][h] = v.w;
                }
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
#pragma unroll
                for (int nj = 0; nj < NT; ++nj) mma_tf32_1688(acc[mi][nj], afrag[mi], bfrag[nj]);
        }
        if (more)
            sstore_tf32<C>(pa, pb, As + ((kt + 1) % STAGES) * A_STAGE,
                           Bs + ((kt + 1) % STAGES) * B_STAGE);
    }
}

// Epilogue. Lane (g, c) of warp (wm, wn) holds, for m16 tile mi, rows g and g+8 of the
// tile; d[0] of n-tile j (of the warp's group q of four) is the mma's column 2c, which is
// global column n0 + 32q + 8c + j, and d[1] its column 2c+1, global n0 + 32q + 8c + 4 + j.
// Gathering d[e] across the four n-tiles of a group therefore gives four consecutive
// columns: two float4 stores per row per lane per group.
// ATOMIC: this block holds one K-slice of a tail tile; C was zeroed before the launch and
// every slice adds into it (the adds happen at L2).
template <class C, bool ALIGNED, bool ATOMIC>
__device__ __forceinline__ void store_c(const typename C::Acc& acc, float* __restrict__ Cp, int M,
                                        int N, int bm, int bn) {
    constexpr int WM = C::WM, WN = C::WN, MT = C::MT, NQ = C::NQ;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int wm = warp / C::WARPS_N;
    const int wn = warp % C::WARPS_N;
    const int g = lane >> 2;
    const int c = lane & 3;
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const int row = bm + wm * WM + mi * 16 + g + h * 8;
            if (row >= M) continue;
            float* crow = Cp + static_cast<size_t>(row) * N;
#pragma unroll
            for (int ql = 0; ql < 2 * NQ; ++ql) {
                const int q = ql >> 1, lo = ql & 1;
                const int col0 = bn + wn * WN + 32 * q + 8 * c + 4 * lo;
                float v[4];
#pragma unroll
                for (int j = 0; j < 4; ++j) v[j] = acc[mi][4 * q + j][h * 2 + lo];
                if constexpr (ATOMIC) {
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        if (col0 + e < N) atomicAdd(crow + col0 + e, v[e]);
                } else if (ALIGNED && col0 + 3 < N) {
                    *reinterpret_cast<float4*>(crow + col0) = make_float4(v[0], v[1], v[2], v[3]);
                } else {
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        if (col0 + e < N) crow[col0 + e] = v[e];
                }
            }
        }
    }
}

// Work assignment (see `launch`): blocks [0, dp_tiles) own one whole output tile each; the
// blocks after that split the remaining tiles `split` ways along K. Tiles are numbered in
// bands of `group` tile rows walked column by column, so the ~340 tiles in flight at once
// cover a group x (340 / group) rectangle of C instead of a few whole rows of tiles: the A
// rows and B columns they share then fit in L2 (8192^3 in fp32 is 256 MB per operand; a
// row-major wave re-reads all of B from DRAM, 3.3 GB per GEMM).
struct Sched {
    int tiles_m, tiles_n;  // tiles along M and N
    int group;             // tile rows per band
    int dp_tiles;          // tiles handled whole
    int split;             // K-slices per tail tile (1: no tail)
};

__device__ __forceinline__ void tile_coords(const Sched& s, int tile, int& tm, int& tn) {
    const int band = tile / (s.group * s.tiles_n);
    const int first_m = band * s.group;
    const int rows = min(s.group, s.tiles_m - first_m);
    const int in_band = tile - band * s.group * s.tiles_n;
    tm = first_m + in_band % rows;
    tn = in_band / rows;
}

// Zeroes the tail tiles of C (tiles dp_tiles.. of the schedule) before their K-slices add
// into them: one block per tile, one launch, instead of a cudaMemset2DAsync per tile (which
// cost 1024^3 with its 128 tail tiles 0.2 ms of launch overhead).
template <int BM, int BN>
__global__ void __launch_bounds__(THREADS)
    zero_tail_kernel(float* __restrict__ Cp, int M, int N, Sched sched) {
    int tm, tn;
    tile_coords(sched, sched.dp_tiles + blockIdx.x, tm, tn);
    const int bm = tm * BM, bn = tn * BN;
    const int rows = min(BM, M - bm), cols = min(BN, N - bn);
    for (int i = threadIdx.x; i < rows * cols; i += THREADS)
        Cp[static_cast<size_t>(bm + i / cols) * N + bn + i % cols] = 0.f;
}

// PREFETCH selects the register-prefetch mainloop over the cp.async one (an ablation kept
// for the doc: it measured 2 to 3% slower). MIN_BLOCKS is the register cap (2 fits the
// prefetch loop's extra slab registers into two blocks per SM; the cp.async loop needs none).
template <class C, bool THREE, bool ALIGNED, bool PREFETCH, int MIN_BLOCKS>
__global__ void __launch_bounds__(THREADS, MIN_BLOCKS)
    sgemm_tf32_kernel(const float* __restrict__ A, const float* __restrict__ B,
                      float* __restrict__ Cp, int M, int N, int K, Sched sched) {
    static_assert(!(THREE && PREFETCH),
                  "3xTF32 splits per fragment (two tiles per slab in smem otherwise)");
    extern __shared__ __align__(16) unsigned char smem[];
    const int KT = cdiv(K, C::BK);
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
    int tm, tn;
    tile_coords(sched, tile, tm, tn);
    const int bm = tm * C::BM;
    const int bn = tn * C::BN;

    typename C::Acc acc;
    if constexpr (PREFETCH)
        mainloop_prefetch<C, ALIGNED>(A, B, M, N, K, bm, bn, kt_begin, kt_end - kt_begin, smem,
                                      acc);
    else
        mainloop<C, THREE, ALIGNED>(A, B, M, N, K, bm, bn, kt_begin, kt_end - kt_begin, smem, acc);
    if (tile < sched.dp_tiles)
        store_c<C, ALIGNED, false>(acc, Cp, M, N, bm, bn);
    else
        store_c<C, ALIGNED, true>(acc, Cp, M, N, bm, bn);
}

template <class C, bool THREE, bool ALIGNED, bool PREFETCH, int MIN_BLOCKS>
int resident_blocks() {
    static int resident = 0;  // blocks resident per GPU; also the > 48 KB smem opt-in
    if (resident == 0) {
        auto* kernel = sgemm_tf32_kernel<C, THREE, ALIGNED, PREFETCH, MIN_BLOCKS>;
        if (C::SMEM > 48 * 1024) {
            SPARK_CUDA_CHECK(
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
        }
        int per_sm = 0;
        SPARK_CUDA_CHECK(
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, THREADS, C::SMEM));
        resident = (per_sm > 0 ? per_sm : 1) * num_sms();
    }
    return resident;
}

// Wave quantization, handled as in sgemm variants 4 and 5: the tiles past the last full
// wave are split `split` ways along K over the otherwise idle blocks, their tiles of C are
// zeroed on the stream and every slice adds its partial sums in with fp32 atomics.
constexpr int GROUP = 16;  // tile rows per band of the raster (the sweep is in the doc)

template <class C, bool THREE, bool ALIGNED, bool PREFETCH, int MIN_BLOCKS>
void launch(const float* A, const float* B, float* Cp, int M, int N, int K, cudaStream_t stream) {
    const int resident = resident_blocks<C, THREE, ALIGNED, PREFETCH, MIN_BLOCKS>();
    Sched s;
    s.tiles_m = cdiv(M, C::BM);
    s.tiles_n = cdiv(N, C::BN);
    s.group = GROUP;
    const int tiles = s.tiles_m * s.tiles_n;
    const int tail = tiles % resident;
    s.split = tail > 0 ? std::min(cdiv(K, C::BK), resident / tail) : 1;
    if (s.split <= 1) {
        s.dp_tiles = tiles;
        s.split = 1;
    } else {
        s.dp_tiles = tiles - tail;
        zero_tail_kernel<C::BM, C::BN><<<tail, THREADS, 0, stream>>>(Cp, M, N, s);
    }
    const int grid = s.dp_tiles + (tiles - s.dp_tiles) * s.split;
    sgemm_tf32_kernel<C, THREE, ALIGNED, PREFETCH, MIN_BLOCKS>
        <<<grid, THREADS, C::SMEM, stream>>>(A, B, Cp, M, N, K, s);
}

// Tiles: 128x128x16, three cp.async stages, 48 KB of shared memory, two blocks per SM, for
// both variants; a 64x128 tile when the 128-row one cannot fill a wave of blocks, as hgemm
// variant 3 does. The sweep over tiles, depths, warp grids and the register-prefetch loop
// (PREFETCH = true, a two-stage ring) is in docs/design/sgemm.md; none of them beat this.
using Big = Cfg<128, 128, 16, 3>;
using Small = Cfg<64, 128, 16, 3>;

template <bool THREE, bool ALIGNED>
void launch_auto(const float* A, const float* B, float* Cp, int M, int N, int K,
                 cudaStream_t stream) {
    const int tiles_big = cdiv(M, Big::BM) * cdiv(N, Big::BN);
    if (tiles_big >= resident_blocks<Big, THREE, ALIGNED, false, 1>())
        launch<Big, THREE, ALIGNED, false, 1>(A, B, Cp, M, N, K, stream);
    else
        launch<Small, THREE, ALIGNED, false, 1>(A, B, Cp, M, N, K, stream);
}

}  // namespace tf32_tile

// Entry point for sgemm variants 6 (three_pass = false) and 7 (true); dispatched to by sgemm.
void sgemm_tf32(const float* A, const float* B, float* C, int M, int N, int K, bool three_pass,
                cudaStream_t stream) {
    using namespace tf32_tile;
    const bool aligned =
        (K & 3) == 0 && (N & 3) == 0 && is_aligned16(A) && is_aligned16(B) && is_aligned16(C);
    if (three_pass) {
        if (aligned)
            launch_auto<true, true>(A, B, C, M, N, K, stream);
        else
            launch_auto<true, false>(A, B, C, M, N, K, stream);
    } else {
        if (aligned)
            launch_auto<false, true>(A, B, C, M, N, K, stream);
        else
            launch_auto<false, false>(A, B, C, M, N, K, stream);
    }
}

}  // namespace spark
