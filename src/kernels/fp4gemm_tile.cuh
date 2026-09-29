// The mma.sync.m16n8k64 + ldmatrix block tile of the fp4 GEMM (fp4gemm.cu, variant 1 and its
// decode configurations; fp4gemm_tma.cu, variant 2). Private to src/kernels.
//
//   C[M,N] = s * sum_k (sfa[m][k/V] A[m][k]) (sfb[n][k/V] Bt[n][k])
//
// A and Bt are e2m1 packed two to a byte, K contiguous (element k in the low nibble of byte
// k/2 for even k), with an e4m3 scale per 16 k (NVFP4, V = 16) or a ue8m0 scale per 32 k
// (MXFP4, V = 32) on both operands, applied by the block-scaled mma itself.
//
// Operands. The m16n8k64 fragments are the m16n8k32 fp8 fragments with two e2m1 values per
// byte (common.cuh), so a stage of A or Bt is the fp8 tile byte for byte: BM rows of BK bytes
// (2 BK values), XOR-swizzled 16-byte chunks, one ldmatrix.x4 per 16 rows by 32 bytes. The
// swizzle, the geometry and the epilogue are fp8gemm_tile's; a k64 step is 32 bytes of K,
// the same bytes as the fp8 tile's k32 step, and does twice the multiply-adds.
//
// Scales. They come in the blocked layout cuBLASLt and torch use for block-scaled GEMMs
// (fp4_scale_bytes in kernels.h): tiles of 128 rows by 4 scale bytes, 512 bytes each, the row
// tiles outermost and the K tiles inside them, and in a tile the 4 bytes of row r at byte
// (r % 32) * 16 + (r / 32) * 4. For NVFP4 those 4 bytes are the scales of one k64 step, which
// is exactly the scale register the instruction takes from one lane, and for MXFP4 they are
// two steps' (byte-id 0 and 2). So a stage's scales are STEPS / SPT consecutive 512-byte tiles
// per 128 rows, one contiguous run in memory that cp.async or a bulk copy moves as is, and
// every scale register is one aligned 4-byte word of shared memory, with no repacking:
// the fp8 MX mode, whose scales are row-major, spends a shuffle per tile per stage instead.
//
// Which words a lane loads. The instruction reads, for thread-id T, row g of an A tile from
// lane 4g + 2T and row g + 8 from lane 4g + 2T + 1, and column g of a B tile from lane
// 4g + T. So a lane (g, c) holds in one register the row g + 8 (c & 1) of A tile 2p + (c >> 1)
// (the pair of tiles 2p, 2p + 1 read with thread-id 0 and 1) and the column g of B tile
// 2p + (c & 1). In the blocked layout the rows r and r + 32 are adjacent words, and so are
// A tiles p and p + 2 of a 64-row warp tile, and B tiles p and p + 4 of a 64-column one:
// one 8-byte load then fills two registers. Per k64 step a 64x32 warp tile issues 6 ldmatrix,
// 3 scale loads and 16 mma; a 64x64 one 8 ldmatrix, 3 scale loads and 32 mma.
#pragma once

#include <algorithm>
#include <cstdlib>

#include "fp8gemm_tile.cuh"
#include "spark/common.cuh"

namespace spark::fp4gemm_tile {

using fp8gemm_tile::swz;

constexpr int NVFP4 = 0;
constexpr int MXFP4 = 1;
constexpr int kSfTile = 512;  // bytes of one 128-row by 4-byte scale tile

// One tile configuration: fp8gemm_tile::Cfg's operand geometry (BK in bytes of K per stage,
// 2 BK e2m1 values) plus each stage's scale tiles after the two operand tiles. An operand of
// R >= 128 rows spans R / 128 row blocks of the blocked layout; its stage region is laid out
// [row block][scale tile][512 bytes], so that each row block's tiles, which are consecutive
// in memory, land in one contiguous run (one bulk copy). R < 32 rows use R 16-byte rows of
// their one tile (the rows of the other row blocks' words are not needed).
template <int BM_, int BN_, int BK_, int STAGES_, int WM_, int WN_, int FMT_>
struct Cfg : fp8gemm_tile::Cfg<BM_, BN_, BK_, STAGES_, WM_, WN_> {
    using Base = fp8gemm_tile::Cfg<BM_, BN_, BK_, STAGES_, WM_, WN_>;
    static constexpr int FMT = FMT_;
    static constexpr int STEPS = BK_ / 32;               // k64 steps per stage
    static constexpr int SPT = FMT == NVFP4 ? 1 : 2;     // k64 steps per scale tile
    static constexpr int TILES = STEPS / SPT;            // scale tiles per stage and row block
    static constexpr int SA_ROWS = BM_ < 32 ? BM_ : 32;  // 16-byte rows of a tile in use
    static constexpr int SB_ROWS = BN_ < 32 ? BN_ : 32;
    static constexpr int SA_RB = BM_ > 128 ? BM_ / 128 : 1;  // row blocks
    static constexpr int SB_RB = BN_ > 128 ? BN_ / 128 : 1;
    static constexpr int SA_TILE = SA_ROWS * 16;
    static constexpr int SB_TILE = SB_ROWS * 16;
    static constexpr int SA_RB_STRIDE = TILES * SA_TILE;  // bytes between row blocks
    static constexpr int SB_RB_STRIDE = TILES * SB_TILE;
    static constexpr int SA_OFF = Base::A_STAGE + Base::B_STAGE;  // in a stage
    static constexpr int SB_OFF = SA_OFF + SA_RB * SA_RB_STRIDE;
    // Stages of the 128-row tiles start on 1 KB boundaries, which the TMA swizzle of
    // variant 2 needs (MXFP4's half-size scale tiles would otherwise leave 512 bytes over).
    static constexpr int STAGE_ALIGN = BM_ >= 128 ? 1024 : 16;
    static constexpr int STAGE =
        (SB_OFF + SB_RB * SB_RB_STRIDE + STAGE_ALIGN - 1) / STAGE_ALIGN * STAGE_ALIGN;
    static constexpr int SMEM = STAGES_ * STAGE;
    // 16-byte copies per stage (the cp.async mainloop)
    static constexpr int SA_CHUNKS = SA_RB * TILES * SA_ROWS;
    static constexpr int SF_CHUNKS = SA_CHUNKS + SB_RB * TILES * SB_ROWS;
    static constexpr int SF_ITERS = cdiv(SF_CHUNKS, Base::THREADS);
    static_assert(BK_ >= 64 && STEPS % SPT == 0, "a stage is whole scale tiles");
    static_assert(BM_ % 16 == 0 && (BM_ <= 32 || BM_ % 32 == 0), "row blocks of 16, 32 or 32n");
    static_assert(BM_ <= 128 || BM_ % 128 == 0, "BM above 128 is whole row blocks");
    static_assert(BN_ <= 128 || BN_ % 128 == 0, "BN above 128 is whole row blocks");
};

// The scale tensors as the launchers hand them down: the blocked layout, `ktiles` scale tiles
// along K (K / 64 for NVFP4, K / 128 for MXFP4).
struct Scales {
    const unsigned char* a = nullptr;
    const unsigned char* b = nullptr;
    int ktiles = 0;
};

// Address of the scale bytes a block of R rows starting at row0 needs for scale tile `t` along
// K: the whole 512-byte tile when R >= 32 (row0 is then a multiple of 32), else the R 16-byte
// rows of it that hold rows row0 .. row0 + R - 1.
template <int R>
__device__ __forceinline__ const unsigned char* sf_src(const unsigned char* sf, int ktiles,
                                                       int row0, int t) {
    const size_t tile = static_cast<size_t>(row0 / 128) * ktiles + t;
    return sf + tile * kSfTile + (R < 32 ? (row0 % 32) * 16 : 0);
}

// Byte offset, in an operand's scale region of R rows, of the word of block-relative row
// `rel` (within one scale tile; add t * TILE for tile t). q0 = (row0 / 32) % 4 of the block's
// first row: blocks of 32 or 64 rows start inside a row block.
template <int R, int RB_STRIDE>
__device__ __forceinline__ int sf_word_off(int rel, int q0) {
    if constexpr (R < 32) {
        return rel * 16 + q0 * 4;
    } else {
        const int r = rel % 128;
        return (rel / 128) * RB_STRIDE + (r % 32) * 16 + ((q0 + r / 32) & 3) * 4;
    }
}

// The scale registers of one scale tile: A pairs (tiles 2p, 2p+1) and B pairs.
template <class C>
struct SfRegs {
    static constexpr int NA = C::MT >= 2 ? C::MT / 2 : 1;
    static constexpr int NB = C::NT / 2;
    unsigned a[NA];
    unsigned b[NB];
};

// Loads the lane's scale registers from one scale tile in shared memory. `sa` and `sb` point
// at the tile (row block 0) of this step; qa0 / qb0 = (row0 / 32) % 4 of the block's first row
// of A / Bt.
template <class C>
__device__ __forceinline__ void load_sf(SfRegs<C>& r, const unsigned char* sa,
                                        const unsigned char* sb, int wm, int wn, int lane, int qa0,
                                        int qb0) {
    constexpr int MT = C::MT, NT = C::NT, WROWS = C::WROWS, WCOLS = C::WCOLS;
    constexpr int BM = C::BM, BN = C::BN, SAS = C::SA_RB_STRIDE, SBS = C::SB_RB_STRIDE;
    const int g = lane >> 2;
    const int c = lane & 3;
    // A: lane (g, c) holds row g + 8 (c & 1) of tile 2p + (c >> 1).
    if constexpr (MT == 4 && BM >= 32) {  // tiles 2 and 3 are rows + 32: the next word
        const int rel = wm * WROWS + (c >> 1) * 16 + g + 8 * (c & 1);
        const uint2 v = *reinterpret_cast<const uint2*>(sa + sf_word_off<BM, SAS>(rel, qa0));
        r.a[0] = v.x;
        r.a[1] = v.y;
    } else {
#pragma unroll
        for (int p = 0; p < SfRegs<C>::NA; ++p) {
            const int rel = wm * WROWS + (MT >= 2 ? (2 * p + (c >> 1)) * 16 : 0) + g + 8 * (c & 1);
            r.a[p] = *reinterpret_cast<const unsigned*>(sa + sf_word_off<BM, SAS>(rel, qa0));
        }
    }
    // B: lane (g, c) holds column g of tile 2p + (c & 1).
    if constexpr (NT == 8) {  // pairs p and p + 2 are columns + 32: the next word
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int rel = wn * WCOLS + (2 * p + (c & 1)) * 8 + g;
            const uint2 v = *reinterpret_cast<const uint2*>(sb + sf_word_off<BN, SBS>(rel, qb0));
            r.b[p] = v.x;
            r.b[p + 2] = v.y;
        }
    } else {
#pragma unroll
        for (int p = 0; p < NT / 2; ++p) {
            const int rel = wn * WCOLS + (2 * p + (c & 1)) * 8 + g;
            r.b[p] = *reinterpret_cast<const unsigned*>(sb + sf_word_off<BN, SBS>(rel, qb0));
        }
    }
    static_assert(MT == 1 || MT == 2 || MT == 4, "warp tiles of 16, 32 or 64 rows");
    static_assert(NT == 2 || NT == 4 || NT == 8, "warp tiles of 16, 32 or 64 columns");
}

// One mma with the thread-ids of tile (mi, nj) and the byte-id of the step as immediates.
template <int FMT, int BID>
__device__ __forceinline__ void mma(float (&d)[4], const unsigned (&a)[4], const unsigned (&b)[2],
                                    unsigned sa, unsigned sb, int ta, int tb) {
    // ta and tb are constants once the caller's loops are unrolled; the branches fold.
    if constexpr (FMT == NVFP4) {
        if (ta == 0 && tb == 0) mma_e2m1_16864_nvf4<0, 0>(d, a, b, sa, sb);
        if (ta == 0 && tb == 1) mma_e2m1_16864_nvf4<0, 1>(d, a, b, sa, sb);
        if (ta == 1 && tb == 0) mma_e2m1_16864_nvf4<1, 0>(d, a, b, sa, sb);
        if (ta == 1 && tb == 1) mma_e2m1_16864_nvf4<1, 1>(d, a, b, sa, sb);
    } else {
        if (ta == 0 && tb == 0) mma_e2m1_16864_mxf4<BID, 0, 0>(d, a, b, sa, sb);
        if (ta == 0 && tb == 1) mma_e2m1_16864_mxf4<BID, 0, 1>(d, a, b, sa, sb);
        if (ta == 1 && tb == 0) mma_e2m1_16864_mxf4<BID, 1, 0>(d, a, b, sa, sb);
        if (ta == 1 && tb == 1) mma_e2m1_16864_mxf4<BID, 1, 1>(d, a, b, sa, sb);
    }
}

// One k64 step (32 bytes of K at byte KK of the stage) of the warp tile: MT ldmatrix.x4 for
// A, NT/2 for Bt (fp8gemm_tile::mma_step's addressing), MT x NT mma with the scale registers
// of the step's scale tile.
template <class C, int KK>
__device__ __forceinline__ void mma_step(const unsigned char* as, const unsigned char* bs, int wm,
                                         int wn, int lane, typename C::Acc& acc,
                                         const SfRegs<C>& sf) {
    constexpr int BK = C::BK, MT = C::MT, NT = C::NT, WROWS = C::WROWS, WCOLS = C::WCOLS;
    constexpr int BID = C::FMT == MXFP4 ? 2 * ((KK / 32) % 2) : 0;
    unsigned afrag[MT][4];
    unsigned bfrag[NT][2];
    const int a_row = lane & 15;
    const int a_ch = KK / 16 + (lane >> 4);
#pragma unroll
    for (int mi = 0; mi < MT; ++mi) {
        const int row = wm * WROWS + mi * 16 + a_row;
        ldmatrix_x4(afrag[mi], as + row * BK + swz<BK>(row, a_ch) * 16);
    }
    const int b_row = (lane & 7) + ((lane >> 4) & 1) * 8;
    const int b_ch = KK / 16 + ((lane >> 3) & 1);
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
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int nj = 0; nj < NT; ++nj) {
            mma<C::FMT, BID>(acc[mi][nj], afrag[mi], bfrag[nj], sf.a[MT >= 2 ? mi / 2 : 0],
                             sf.b[nj / 2], MT >= 2 ? mi & 1 : 0, nj & 1);
        }
    }
}

// The STEPS k64 steps of one stage whose operand tiles are at `as` / `bs` and whose scale
// tiles are at `sfa` / `sfb`: the scale registers are loaded once per scale tile (every step
// for NVFP4, every second step for MXFP4).
template <class C>
__device__ __forceinline__ void mma_stage(const unsigned char* as, const unsigned char* bs,
                                          const unsigned char* sfa, const unsigned char* sfb,
                                          int wm, int wn, int lane, int qa0, int qb0,
                                          typename C::Acc& acc) {
    SfRegs<C> sf;
    fp8gemm_tile::static_for<C::STEPS>([&](auto s) {
        constexpr int S = decltype(s)::value;
        if constexpr (S % C::SPT == 0)
            load_sf<C>(sf, sfa + (S / C::SPT) * C::SA_TILE, sfb + (S / C::SPT) * C::SB_TILE, wm, wn,
                       lane, qa0, qb0);
        mma_step<C, S * 32>(as, bs, wm, wn, lane, acc, sf);
    });
}

// Accumulates the block's tile over k-tiles [kt_begin, kt_begin + nkt) (BK bytes each) into
// `acc`, which it zeroes first: the cp.async pipeline of fp8gemm_tile::mainloop with each
// stage's scale tiles copied behind its operand tiles. Rows of A past m_valid are zero-filled
// (their scale rows exist: the blocked layout pads to 128 rows). Drains the pipeline on return.
template <class C>
__device__ __forceinline__ void mainloop(const unsigned char* __restrict__ A,
                                         const unsigned char* __restrict__ Bt, int Kb, int bm,
                                         int bn, int m_valid, int kt_begin, int nkt,
                                         unsigned char* smem, typename C::Acc& acc, Scales sf) {
    constexpr int BK = C::BK, STAGES = C::STAGES, THREADS = C::THREADS;
    constexpr int CPR = C::CPR, A_CHUNKS = C::A_CHUNKS, B_CHUNKS = C::B_CHUNKS;
    constexpr int A_ITERS = C::A_ITERS, B_ITERS = C::B_ITERS;
    constexpr int A_STAGE = C::A_STAGE, STAGE = C::STAGE;
    constexpr int MT = C::MT, NT = C::NT, TILES = C::TILES;
    constexpr int SA_ROWS = C::SA_ROWS, SB_ROWS = C::SB_ROWS;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int wm = warp / C::WN;
    const int wn = warp % C::WN;
    const int qa0 = (bm / 32) & 3;
    const int qb0 = (bn / 32) & 3;

    const unsigned char* Ab = A + static_cast<size_t>(bm) * Kb + static_cast<size_t>(kt_begin) * BK;
    const unsigned char* Bb =
        Bt + static_cast<size_t>(bn) * Kb + static_cast<size_t>(kt_begin) * BK;
    const unsigned char* Sa = sf_src<C::BM>(sf.a, sf.ktiles, bm, kt_begin * TILES);
    const unsigned char* Sb = sf_src<C::BN>(sf.b, sf.ktiles, bn, kt_begin * TILES);

    auto load_stage = [&](int stage, int kt) {
        unsigned char* as = smem + stage * STAGE;
        unsigned char* bs = as + A_STAGE;
        const int k0 = kt * BK;
#pragma unroll
        for (int i = 0; i < A_ITERS; ++i) {
            const int c = tid + i * THREADS;
            if (A_CHUNKS % THREADS == 0 || c < A_CHUNKS) {
                const int row = c / CPR;
                const int ch = c % CPR;
                const bool ok = row < m_valid;
                cp_async_16_zfill(as + row * BK + swz<BK>(row, ch) * 16,
                                  Ab + static_cast<size_t>(ok ? row : 0) * Kb + k0 + ch * 16, ok);
            }
        }
#pragma unroll
        for (int i = 0; i < B_ITERS; ++i) {
            const int c = tid + i * THREADS;
            if (B_CHUNKS % THREADS == 0 || c < B_CHUNKS) {
                const int row = c / CPR;
                const int ch = c % CPR;
                cp_async_16(bs + row * BK + swz<BK>(row, ch) * 16,
                            Bb + static_cast<size_t>(row) * Kb + k0 + ch * 16);
            }
        }
        // Scale tiles, [row block][tile][16-byte row] for A, then the same for Bt; each row
        // block's tiles are consecutive in memory.
#pragma unroll
        for (int i = 0; i < C::SF_ITERS; ++i) {
            const int c = tid + i * THREADS;
            if (C::SF_CHUNKS % THREADS == 0 || c < C::SF_CHUNKS) {
                if (c < C::SA_CHUNKS) {
                    const int rb = c / (TILES * SA_ROWS), t = (c / SA_ROWS) % TILES,
                              r = c % SA_ROWS;
                    cp_async_16(
                        as + C::SA_OFF + rb * C::SA_RB_STRIDE + t * C::SA_TILE + r * 16,
                        Sa + (static_cast<size_t>(rb) * sf.ktiles + kt * TILES + t) * kSfTile +
                            r * 16);
                } else {
                    const int c2 = c - C::SA_CHUNKS;
                    const int rb = c2 / (TILES * SB_ROWS), t = (c2 / SB_ROWS) % TILES,
                              r = c2 % SB_ROWS;
                    cp_async_16(
                        as + C::SB_OFF + rb * C::SB_RB_STRIDE + t * C::SB_TILE + r * 16,
                        Sb + (static_cast<size_t>(rb) * sf.ktiles + kt * TILES + t) * kSfTile +
                            r * 16);
                }
            }
        }
        (void)bs;
    };

#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[i][j][e] = 0.f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s) {
        if (s < nkt) load_stage(s, s);
        cp_async_commit();
    }
    for (int kt = 0; kt < nkt; ++kt) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        {
            const int nk = kt + STAGES - 1;
            if (nk < nkt) load_stage(nk % STAGES, nk);
            cp_async_commit();
        }
        const unsigned char* as = smem + (kt % STAGES) * STAGE;
        mma_stage<C>(as, as + A_STAGE, as + C::SA_OFF, as + C::SB_OFF, wm, wn, lane, qa0, qb0, acc);
    }
    cp_async_wait<0>();
}

// The split-K workspace of the tail tiles, one per kernel configuration and device: an fp32 tile
// and an arrival counter per tail tile. It cleans up after itself: the last slice to arrive at
// a tile converts it, writes zeros back over it and resets the counter, so the buffers are
// zeroed once when they are allocated and no launch needs a memset (two memsets were two
// launches of 2 to 3 us each in front of a 120 us GEMM at 4096^3).
struct Workspace {
    float* ws = nullptr;
    int* counters = nullptr;
    size_t tiles = 0;

    void reserve(int tail, int tile_elems, cudaStream_t stream) {
        if (tiles >= static_cast<size_t>(tail)) return;
        if (ws) SPARK_CUDA_CHECK(cudaFree(ws));
        if (counters) SPARK_CUDA_CHECK(cudaFree(counters));
        tiles = tail;
        SPARK_CUDA_CHECK(cudaMalloc(&ws, tiles * tile_elems * sizeof(float)));
        SPARK_CUDA_CHECK(cudaMalloc(&counters, tiles * sizeof(int)));
        SPARK_CUDA_CHECK(cudaMemsetAsync(ws, 0, tiles * tile_elems * sizeof(float), stream));
        SPARK_CUDA_CHECK(cudaMemsetAsync(counters, 0, tiles * sizeof(int), stream));
    }
};

// SPARK_FP4GEMM_MAX_SPLIT=<n> caps the K-slices per tail tile (a sweep knob; 0 = no cap).
inline int max_split_override() {
    static int v = -1;
    if (v < 0) {
        const char* e = std::getenv("SPARK_FP4GEMM_MAX_SPLIT");
        v = e ? std::max(0, std::atoi(e)) : 0;
    }
    return v;
}

}  // namespace spark::fp4gemm_tile
