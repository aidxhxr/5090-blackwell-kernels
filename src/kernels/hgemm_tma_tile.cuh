// The TMA-fed block tile shared by hgemm variant 5 (hgemm_tma.cu) and variant 6
// (hgemm_tma_sk.cu). Private to src/kernels. The two kernels differ in who computes which
// (tile, K-range) piece; the stage geometry, the swizzle, what the producer lane issues per
// stage and what a consumer warp does with a landed stage are the same and live here once.
//
// A stage is a BM x BK box of A followed by BN / 64 boxes of BK x 64 of B (128 B rows, one
// swizzle span each), all written by the copy engine with the map's swizzle: the 64-byte
// swizzle for A at BK = 32 (chunk ^= (row/2)%4), the 128-byte one otherwise (chunk ^= row%8).
// Both are keyed on the absolute shared address, so a stage must start on a 1 KB boundary.
// The consumer's ldmatrix addressing is variant 3's, unchanged; the only difference is that
// a warp's columns of B live in box (wn * WN) / 64 at chunk ((wn * WN) % 64) / 8 + nj + lane/16.
//
// The geometry is a template, Tile<BM, BN, WARPS_M, WARPS_N>, always eight consumer warps and
// one producer warp. Variant 5 and variant 6's default use Tile<128, 128, 2, 4> (warp tile 64 x
// 32), which is also what the names at namespace scope below refer to, so variant 5 reads as
// it did. Variant 6 adds tiles for the shapes the 128 x 128 grid cannot fill (128 x 64 and 64
// x 128, warp tile 32 x 32) and for wide N (128 x 256, warp tile 64 x 64: half the ldmatrix
// per mma.sync); see docs/design/hgemm.md, "Model shapes".
#pragma once

#include "spark/common.cuh"

namespace spark::hgemm_tma_tile {

constexpr int B_BOX_N = 64;  // columns per TMA box of B: 128 B, one swizzle span

// Round a shared-memory total up to a whole KB, so that with two blocks on an SM the second
// block's allocation (this plus the reserved KB) also starts on a 1 KB boundary, whichever
// address the swizzle is keyed on.
constexpr int round_kb(int bytes) {
    return (bytes + 1023) / 1024 * 1024;
}

// Physical 16-byte chunk for logical (row, chunk) of an A row of BK bf16, as the copy engine
// wrote it. Same function as variant 3's swz_a.
template <int BK>
__device__ __forceinline__ int swz_a(int row, int chunk) {
    if constexpr (BK == 32)
        return chunk ^ ((row >> 1) & 3);
    else
        return chunk ^ (row & 7);
}
// A B box row is 128 B (8 chunks) and the 128-byte swizzle XORs its chunk with row%8.
__device__ __forceinline__ int swz_b(int row, int chunk) {
    return chunk ^ (row & 7);
}

template <int BM_, int BN_, int WARPS_M_, int WARPS_N_>
struct Tile {
    static constexpr int BM = BM_, BN = BN_;
    static constexpr int WARPS_M = WARPS_M_, WARPS_N = WARPS_N_;
    static constexpr int CONSUMER_WARPS = WARPS_M * WARPS_N;
    static constexpr int CONSUMERS = CONSUMER_WARPS * 32;
    static constexpr int THREADS = CONSUMERS + 32;  // plus the producer warp
    static constexpr int WM = BM / WARPS_M;
    static constexpr int WN = BN / WARPS_N;
    static constexpr int MT = WM / 16;  // m16 tiles per warp
    static constexpr int NT = WN / 8;   // n8 tiles per warp
    static constexpr int B_BOXES = BN / B_BOX_N;
    static_assert(CONSUMER_WARPS == 8, "the named barriers and the ring count eight warps");
    static_assert(BM % (16 * WARPS_M) == 0 && BM <= 256, "an A box is at most 256 rows");
    static_assert(WN % 16 == 0, "one ldmatrix.x4.trans loads two n8 tiles");
    static_assert(BN % B_BOX_N == 0, "B is loaded as whole 64-column boxes, so a warp's 16-column "
                                     "ldmatrix pairs never straddle two of them");

    using Acc = float[MT][NT][4];

    // Bytes of A + B per stage: a multiple of 1 KB for every tile here and both BK.
    template <int BK>
    static constexpr int stage_bytes() {
        return (BM * BK + BK * BN) * static_cast<int>(sizeof(__nv_bfloat16));
    }

    // Host: the two tensor maps. A is K-innermost with a box of BK columns x BM rows,
    // swizzled over the box width; B is N-innermost with a box of 64 columns x BK rows. Rows
    // past M and columns past N are zero-filled by the copy engine and still count toward the
    // barrier. Encoding is host arithmetic on 128 bytes, 27 ns per call, so maps are rebuilt
    // per call.
    template <int BK>
    static void make_maps(const __nv_bfloat16* A, const __nv_bfloat16* B, int M, int N, int K,
                          CUtensorMap& tmA, CUtensorMap& tmB) {
        constexpr CUtensorMapSwizzle a_swz =
            BK == 32 ? CU_TENSOR_MAP_SWIZZLE_64B : CU_TENSOR_MAP_SWIZZLE_128B;
        tmA = make_tensor_map_2d_bf16(A, M, K, BM, BK, a_swz);
        tmB = make_tensor_map_2d_bf16(B, K, N, BK, B_BOX_N, CU_TENSOR_MAP_SWIZZLE_128B);
    }

    // Producer lane: register the stage's bytes on `full` and issue its 1 + BN / 64 boxes,
    // k-tile [k0, k0 + BK) of the tile at (bm, bn), into `stage`.
    template <int BK>
    static __device__ __forceinline__ void issue_stage(unsigned char* stage, const CUtensorMap* tmA,
                                                       const CUtensorMap* tmB, uint64_t* full,
                                                       int k0, int bm, int bn) {
        constexpr int A_BYTES = BM * BK * static_cast<int>(sizeof(__nv_bfloat16));
        constexpr int B_BOX_BYTES = BK * B_BOX_N * static_cast<int>(sizeof(__nv_bfloat16));
        mbar_arrive_expect_tx(full, stage_bytes<BK>());
        unsigned char* bs = stage + A_BYTES;
        tma_load_2d(stage, tmA, full, k0, bm);  // (k, m): BM rows x BK
#pragma unroll
        for (int b = 0; b < B_BOXES; ++b)  // (n, k): BK rows x 64
            tma_load_2d(bs + b * B_BOX_BYTES, tmB, full, bn + b * B_BOX_N, k0);
    }

    static __device__ __forceinline__ void zero_acc(Acc& acc) {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;
    }

    // Consumer warp (wm, wn): variant 3's ldmatrix + mma.sync body over one landed stage. The
    // caller has waited on the stage's "full" barrier; afterwards it fences the proxy,
    // __syncwarp()s and arrives on "empty".
    template <int BK>
    static __device__ __forceinline__ void consume_stage(Acc& acc, const unsigned char* stage,
                                                         int wm, int wn, int lane) {
        constexpr int A_STAGE = BM * BK;     // elements
        constexpr int B_BOX = BK * B_BOX_N;  // elements per B box
        const __nv_bfloat16* as = reinterpret_cast<const __nv_bfloat16*>(stage);
        const __nv_bfloat16* bs = as + A_STAGE;

        const int a_row_in_tile = lane & 15;
        const int a_kchunk = lane >> 4;
        const int b_krow_in_step = lane & 15;
        const int b_nchunk = lane >> 4;

#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            unsigned afrag[MT][4];
            unsigned bfrag[NT][2];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                const int row = wm * WM + mi * 16 + a_row_in_tile;
                const int ch = kk / 8 + a_kchunk;
                ldmatrix_x4(afrag[mi], as + row * BK + swz_a<BK>(row, ch) * 8);
            }
#pragma unroll
            for (int nj = 0; nj < NT; nj += 2) {
                const int krow = kk + b_krow_in_step;
                const int col = wn * WN + nj * 8;  // first of the 16 columns, within the tile
                const __nv_bfloat16* box = bs + (col / B_BOX_N) * B_BOX;
                const int ch = (col % B_BOX_N) / 8 + b_nchunk;  // chunk within the box
                unsigned r[4];
                ldmatrix_x4_trans(r, box + krow * B_BOX_N + swz_b(krow, ch) * 8);
                bfrag[nj][0] = r[0];
                bfrag[nj][1] = r[1];
                bfrag[nj + 1][0] = r[2];
                bfrag[nj + 1][1] = r[3];
            }
#pragma unroll
            for (int mi = 0; mi < MT; ++mi)
#pragma unroll
                for (int nj = 0; nj < NT; ++nj) mma_bf16_16816(acc[mi][nj], afrag[mi], bfrag[nj]);
        }
    }
};

// ---- the 128 x 128 tile under its old names (variant 5, variant 6's default) --------------

using Tile128 = Tile<128, 128, 2, 4>;
constexpr int BM = Tile128::BM, BN = Tile128::BN;
constexpr int WARPS_M = Tile128::WARPS_M, WARPS_N = Tile128::WARPS_N;
constexpr int CONSUMER_WARPS = Tile128::CONSUMER_WARPS;
constexpr int CONSUMERS = Tile128::CONSUMERS;  // 256 threads
constexpr int THREADS = Tile128::THREADS;
constexpr int WM = Tile128::WM;  // 64
constexpr int WN = Tile128::WN;  // 32
constexpr int MT = Tile128::MT;
constexpr int NT = Tile128::NT;
using Acc = Tile128::Acc;

template <int BK>
constexpr int stage_bytes() {
    return Tile128::stage_bytes<BK>();
}
template <int BK>
inline void make_maps(const __nv_bfloat16* A, const __nv_bfloat16* B, int M, int N, int K,
                      CUtensorMap& tmA, CUtensorMap& tmB) {
    Tile128::make_maps<BK>(A, B, M, N, K, tmA, tmB);
}
template <int BK>
__device__ __forceinline__ void issue_stage(unsigned char* stage, const CUtensorMap* tmA,
                                            const CUtensorMap* tmB, uint64_t* full, int k0, int bm,
                                            int bn) {
    Tile128::issue_stage<BK>(stage, tmA, tmB, full, k0, bm, bn);
}
__device__ __forceinline__ void zero_acc(Acc& acc) {
    Tile128::zero_acc(acc);
}
template <int BK>
__device__ __forceinline__ void consume_stage(Acc& acc, const unsigned char* stage, int wm, int wn,
                                              int lane) {
    Tile128::consume_stage<BK>(acc, stage, wm, wn, lane);
}

}  // namespace spark::hgemm_tma_tile
