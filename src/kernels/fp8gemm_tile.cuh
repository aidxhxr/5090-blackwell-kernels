// The mma.sync.m16n8k32 + ldmatrix block tile of the fp8 GEMM (fp8gemm.cu, variant 1 and its
// decode configurations). Private to src/kernels.
//
//   C[M,N] = s * A[M,K] * Bt[N,K]^T       A, Bt e4m3 with K contiguous, fp32 accumulate
//
// Both operands are K-contiguous, so one layout, one copy routine and one swizzle serve both:
// a stage holds BM rows of A and BN rows of Bt, every row BK bytes (BK e4m3 values) as
// BK/16 chunks of 16 bytes, XOR-swizzled by row. ldmatrix is a 16-bit instruction and does
// not know the element type: one of its 8x8 b16 matrices is 8 rows by 16 bytes, which for
// e4m3 is 8 rows by 16 k, and the 32-bit word it hands lane (g, t) holds b16 elements 2t and
// 2t+1 of row g, i.e. e4m3 k = 4t..4t+3 of that row, lowest k in the low byte. That is the
// m16n8k32 fragment word for lane (g, t) (docs/design/fp8gemm.md). So an A fragment is one
// ldmatrix.x4 over a 16-row by 32-byte block, addressed exactly like the bf16 tile's 16x16,
// and the B fragments of two n8 tiles are one ldmatrix.x4 over two 8-row by 32-byte blocks of
// Bt, without .trans: a row of Bt is one n at ascending k, which is the "col" operand's order.
#pragma once

#include <cuda_fp8.h>

#include "spark/common.cuh"

namespace spark::fp8gemm_tile {

// Physical 16-byte chunk for logical (row, chunk) of a smem tile whose rows are BK bytes, so
// that the 8 rows an ldmatrix touches fall in 8 different bank groups. Rows of 32 B (2 chunks):
// chunk ^= (row/4) % 2; 64 B (4 chunks): chunk ^= (row/2) % 4; 128 B and up: chunk ^= row % 8.
// The 64 B and 128 B forms are what the TMA unit's 64B and 128B swizzles compute.
template <int BK>
__device__ __forceinline__ int swz(int row, int chunk) {
    if constexpr (BK == 32)
        return chunk ^ ((row >> 2) & 1);
    else if constexpr (BK == 64)
        return chunk ^ ((row >> 1) & 3);
    else
        return chunk ^ (row & 7);
}

// Compile-time layout of one tile configuration: block tile BM x BN x BK (bytes of K per
// stage), STAGES-deep cp.async pipeline, WM x WN warps each owning BM/WM rows by BN/WN columns.
template <int BM_, int BN_, int BK_, int STAGES_, int WM_, int WN_>
struct Cfg {
    static constexpr int BM = BM_, BN = BN_, BK = BK_, STAGES = STAGES_, WM = WM_, WN = WN_;
    static constexpr int THREADS = WM * WN * 32;
    static constexpr int WROWS = BM / WM;  // rows per warp
    static constexpr int WCOLS = BN / WN;  // columns per warp
    static constexpr int MT = WROWS / 16;  // m16 tiles per warp
    static constexpr int NT = WCOLS / 8;   // n8 tiles per warp
    static_assert(MT >= 1 && MT * 16 * WM == BM, "rows per warp must be a multiple of 16");
    static_assert(NT >= 2 && NT % 2 == 0 && NT * 8 * WN == BN,
                  "ldmatrix.x4 fills the B fragments of two n8 tiles at a time");
    static_assert(BK == 32 || BK == 64 || BK % 128 == 0, "rows of 32 B, 64 B or 128 B+");
    static constexpr int CPR = BK / 16;  // 16-byte chunks per row
    static constexpr int A_CHUNKS = BM * CPR;
    static constexpr int B_CHUNKS = BN * CPR;
    static constexpr int A_ITERS = cdiv(A_CHUNKS, THREADS);  // chunks per thread per stage
    static constexpr int B_ITERS = cdiv(B_CHUNKS, THREADS);
    static constexpr int A_STAGE = BM * BK;  // bytes
    static constexpr int B_STAGE = BN * BK;
    static constexpr int STAGE = A_STAGE + B_STAGE;
    static constexpr int SMEM = STAGES * STAGE;
    using Acc = float[MT][NT][4];
};

// One k32 step of the warp tile out of a stage whose A rows start at `as` and Bt rows at `bs`
// (both BK bytes per row, swizzled): MT ldmatrix.x4 for A, NT/2 for B, MT x NT mma.
template <class C>
__device__ __forceinline__ void mma_step(const unsigned char* as, const unsigned char* bs, int kk,
                                         int wm, int wn, int lane, typename C::Acc& acc) {
    constexpr int BK = C::BK, MT = C::MT, NT = C::NT, WROWS = C::WROWS, WCOLS = C::WCOLS;
    unsigned afrag[MT][4];
    unsigned bfrag[NT][2];
    // A: matrices (rows 0-7, k 0-15), (rows 8-15, k 0-15), (rows 0-7, k 16-31), (rows 8-15,
    // k 16-31) of the m16 x k32 block: lane l addresses row l % 16, chunk l / 16.
    const int a_row = lane & 15;
    const int a_ch = kk / 16 + (lane >> 4);
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int row = wm * WROWS + mi * 16 + a_row;
        ldmatrix_x4(afrag[mi], as + row * BK + swz<BK>(row, a_ch) * 16);
    }
    // B: matrices (tile nj, k 0-15), (nj, k 16-31), (nj+1, k 0-15), (nj+1, k 16-31): lane l
    // addresses row l % 8 of tile nj + l / 16, chunk (l / 8) % 2.
    const int b_row = (lane & 7) + ((lane >> 4) & 1) * 8;
    const int b_ch = kk / 16 + ((lane >> 3) & 1);
#pragma unroll
    for (int nj = 0; nj < NT; nj += 2) {
        const int row = wn * WCOLS + nj * 8 + b_row;
        unsigned r[4];
        ldmatrix_x4(r, bs + row * BK + swz<BK>(row, b_ch) * 16);
        bfrag[nj][0] = r[0];
        bfrag[nj][1] = r[1];
        bfrag[nj + 1][0] = r[2];
        bfrag[nj + 1][1] = r[3];
    }
#pragma unroll
    for (int mi = 0; mi < MT; ++mi)
#pragma unroll
        for (int nj = 0; nj < NT; ++nj) mma_e4m3_16832(acc[mi][nj], afrag[mi], bfrag[nj]);
}

// Accumulates A[bm.., kt_begin*BK .. (kt_begin+nkt)*BK) x Bt[bn.., same K]^T into `acc`,
// which it zeroes first. Rows of A past m_valid are zero-filled (cp.async with a source size
// of 0) and Bt rows bn..bn+BN must exist (N % BN == 0). `smem` is the STAGES-deep tile buffer
// (C::SMEM bytes). The pipeline is drained on return.
template <class C>
__device__ __forceinline__ void mainloop(const unsigned char* __restrict__ A,
                                         const unsigned char* __restrict__ Bt, int K, int bm,
                                         int bn, int m_valid, int kt_begin, int nkt,
                                         unsigned char* smem, typename C::Acc& acc) {
    constexpr int BM = C::BM, BN = C::BN, BK = C::BK, STAGES = C::STAGES, THREADS = C::THREADS;
    constexpr int CPR = C::CPR, A_CHUNKS = C::A_CHUNKS, B_CHUNKS = C::B_CHUNKS;
    constexpr int A_ITERS = C::A_ITERS, B_ITERS = C::B_ITERS;
    constexpr int A_STAGE = C::A_STAGE, STAGE = C::STAGE;
    constexpr int MT = C::MT, NT = C::NT;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / C::WN;
    const int wn = warp % C::WN;

    const unsigned char* Ab = A + static_cast<size_t>(bm) * K + static_cast<size_t>(kt_begin) * BK;
    const unsigned char* Bb = Bt + static_cast<size_t>(bn) * K + static_cast<size_t>(kt_begin) * BK;

    auto load_stage = [&](int stage, int k0) {
        unsigned char* as = smem + stage * STAGE;
        unsigned char* bs = as + A_STAGE;
#pragma unroll
        for (int i = 0; i < A_ITERS; ++i) {
            const int c = tid + i * THREADS;
            if (A_CHUNKS % THREADS == 0 || c < A_CHUNKS) {
                const int row = c / CPR;
                const int ch = c % CPR;
                const bool ok = row < m_valid;
                cp_async_16_zfill(as + row * BK + swz<BK>(row, ch) * 16,
                                  Ab + static_cast<size_t>(ok ? row : 0) * K + k0 + ch * 16, ok);
            }
        }
#pragma unroll
        for (int i = 0; i < B_ITERS; ++i) {
            const int c = tid + i * THREADS;
            if (B_CHUNKS % THREADS == 0 || c < B_CHUNKS) {
                const int row = c / CPR;
                const int ch = c % CPR;
                cp_async_16(bs + row * BK + swz<BK>(row, ch) * 16,
                            Bb + static_cast<size_t>(row) * K + k0 + ch * 16);
            }
        }
    };
    (void)BM;
    (void)BN;

#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

    // Prologue: the first STAGES-1 tiles are in flight before any compute starts.
#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nkt) load_stage(s, s * BK);
        cp_async_commit();
    }

    for (int kt = 0; kt < nkt; ++kt) {
        cp_async_wait<STAGES - 2>();  // tile kt has landed (for this thread)
        __syncthreads();              // ... for every thread; and stage (kt-1)%STAGES is free
        {
            const int nk = kt + STAGES - 1;
            if (nk < nkt) load_stage(nk % STAGES, nk * BK);
            cp_async_commit();  // always commit so the group count stays uniform
        }
        const unsigned char* as = smem + (kt % STAGES) * STAGE;
        const unsigned char* bs = as + A_STAGE;
#pragma unroll
        for (int kk = 0; kk < BK; kk += 32) mma_step<C>(as, bs, kk, wm, wn, lane, acc);
    }
    cp_async_wait<0>();
}

// Epilogue: each lane owns (row g, cols 2c..2c+1) and (row g+8, same cols) of every 16x8
// tile; the scale is applied in fp32 and the pair rounded once to bf16, two per store,
// straight from registers. Rows past M are skipped one at a time, so any M works.
template <class C>
__device__ __forceinline__ void store_bf16(const typename C::Acc& acc, float scale,
                                           __nv_bfloat16* __restrict__ C_, int M, int N, int bm,
                                           int bn) {
    constexpr int WROWS = C::WROWS, WCOLS = C::WCOLS, MT = C::MT, NT = C::NT;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int wm = warp / C::WN;
    const int wn = warp % C::WN;
    const int g = lane >> 2;
    const int c2 = (lane & 3) * 2;
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int nj = 0; nj < NT; ++nj) {
            const int row = bm + wm * WROWS + mi * 16 + g;
            const int col = bn + wn * WCOLS + nj * 8 + c2;
            __nv_bfloat16* p0 = C_ + static_cast<size_t>(row) * N + col;
            __nv_bfloat16* p1 = p0 + static_cast<size_t>(8) * N;
            if (row < M)
                *reinterpret_cast<__nv_bfloat162*>(p0) =
                    __floats2bfloat162_rn(acc[mi][nj][0] * scale, acc[mi][nj][1] * scale);
            if (row + 8 < M)
                *reinterpret_cast<__nv_bfloat162*>(p1) =
                    __floats2bfloat162_rn(acc[mi][nj][2] * scale, acc[mi][nj][3] * scale);
        }
    }
}

}  // namespace spark::fp8gemm_tile
