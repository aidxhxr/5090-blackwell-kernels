// The mma.sync.m16n8k32 + ldmatrix block tile of the fp8 GEMM (fp8gemm.cu, variant 1 and its
// decode configurations). Private to src/kernels.
//
//   C[M,N] = s * A[M,K] * Bt[N,K]^T       A, Bt e4m3 with K contiguous, fp32 accumulate
//
// and, in MX mode, with a ue8m0 scale per row per 32 k on both operands (sfa[M][K/32],
// sfb[N][K/32]) fed to the block-scaled mma through its scale operands (MxFrag below).
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

#include <type_traits>

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

// The MX scale tensors, as the launchers hand them down: sfa[M][K/32] and sfb[N][K/32] in
// ue8m0 with row stride `ld` = K / 32 bytes. Null `a` means per-tensor mode.
struct MxScales {
    const unsigned char* a = nullptr;
    const unsigned char* b = nullptr;
    int ld = 0;
};

// The scale words of one stage, held the way the block-scaled mma reads them. The
// instruction takes one .b32 register per lane per operand and reads, with thread-id 0, the
// scale of row g from lane 4g, of row g+8 from lane 4g+1 and of column g from lane 4g (g =
// lane / 4; the other lanes are ignored), byte-id selecting one of the register's four bytes.
// A row's scales lie along k in memory, so the 4-byte word at byte 4w of a row holds the
// scales of k-blocks 4w .. 4w+3 = k 128w .. 128w+127, one BK = 128 stage (or two BK = 64
// stages), and byte-id = kk / 32 is the k32 step within the stage. So lane (g, c) holds, per
// stage, the word of row g + 8 (c & 1) of each of its MT A tiles and of column g of each of
// its NT Bt tiles.
template <class C>
struct MxFrag {
    unsigned a[C::MT];
    unsigned b[C::NT];
};

// Where the words come from. A row's scales for SFW consecutive words (SFW * 128 k, SPC
// stages) are one SFW * 4-byte chunk, and the warp fetches the chunks of its WROWS rows of A
// and WCOLS columns of Bt with one 8- or 16-byte load per lane per 32 rows, straight from
// global memory, one group of SPC stages ahead (two register sets, used alternately). Each
// stage then takes its words out of the chunks with one shuffle per tile: lane (g, c) reads
// word i of the chunk that lane (16 mi + g + 8 (c & 1)) % 32 loaded for A tile mi, and of
// the chunk lane 8 nj + g loaded for Bt tile nj, one stage ahead of the mma that uses them.
// A 32-byte sector of a row serves 8 words, and the warps of a block that share rows (WN of
// them for A, WM for Bt) hit L1 on each other's sectors within a group. Fetching per stage
// instead (one 4-byte load per row per lane) measured 12 sectors per request, a 61% L1 hit
// rate, 29% more L2 sectors than the tiles themselves and the tensor pipe at 52%; having
// every lane fetch its own rows' chunks (no shuffles, MT + NT loads per group) took the
// registers of the two sets to 64 and more and spilled the two-block configurations, a
// win on variant 2's one-block configurations and a loss everywhere else
// (docs/design/fp8gemm.md). The words do not go through shared memory
// because the 3-stage 128-byte configuration of variant 2 is at 97 KB of the 99 KB a block
// may take, and the decode configurations fill their SM at 2 to 4 blocks.
//
// SFW is 4 (16-byte chunks) when K % 512 == 0, so that every row stride is 16-byte aligned,
// else 2 (K % 256 == 0). Split-K slices start on a group boundary (the launchers align them).
template <class C, int SFW>
struct MxChunk {
    static constexpr int SPC = SFW * 128 / C::BK;  // stages per chunk
    static_assert(SFW * 128 % C::BK == 0 && (SFW == 2 || SFW == 4), "chunks of 8 or 16 bytes");
    static constexpr int A_LOADS = cdiv(C::WROWS, 32);
    static constexpr int B_LOADS = cdiv(C::WCOLS, 32);
    unsigned a[A_LOADS][SFW];
    unsigned b[B_LOADS][SFW];
};

template <int SFW>
__device__ __forceinline__ void mx_load_chunk(unsigned (&w)[SFW], const unsigned char* p) {
    if constexpr (SFW == 4) {
        const uint4 v = *reinterpret_cast<const uint4*>(p);
        w[0] = v.x;
        w[1] = v.y;
        w[2] = v.z;
        w[3] = v.w;
    } else {
        const uint2 v = *reinterpret_cast<const uint2*>(p);
        w[0] = v.x;
        w[1] = v.y;
    }
}

// Fetches the chunks of the group of SPC stages that starts at k-tile kt (a multiple of
// SPC). `mx` points at the tile's first rows; A rows past m_valid read row 0 (their data is
// zero-filled), and a warp tile narrower than 32 rows or columns has its spare lanes read a
// row again rather than branch.
template <class C, int SFW>
__device__ __forceinline__ void mx_fetch(MxChunk<C, SFW>& ch, const MxScales& mx, int kt, int wm,
                                         int wn, int lane, int m_valid) {
    using Ch = MxChunk<C, SFW>;
    const int off = kt * (C::BK / 32);  // byte offset of this group's first k-block
#pragma unroll
    for (int j = 0; j < Ch::A_LOADS; ++j) {
        const int r = wm * C::WROWS + min(j * 32 + lane, C::WROWS - 1);
        mx_load_chunk<SFW>(ch.a[j], mx.a + static_cast<size_t>(r < m_valid ? r : 0) * mx.ld + off);
    }
#pragma unroll
    for (int j = 0; j < Ch::B_LOADS; ++j) {
        const int n = wn * C::WCOLS + min(j * 32 + lane, C::WCOLS - 1);
        mx_load_chunk<SFW>(ch.b[j], mx.b + static_cast<size_t>(n) * mx.ld + off);
    }
}

// The words of stage I (0 .. SPC-1) of the group: word I * BK / 128 of the chunks, and for
// BK = 64 the second stage of a word shifts its two bytes down so that byte-id = kk / 32
// holds for both.
template <class C, int SFW, int I>
__device__ __forceinline__ void mx_words(MxFrag<C>& f, const MxChunk<C, SFW>& ch, int lane) {
    constexpr int WI = I * C::BK / 128;
    constexpr int SHIFT = C::BK == 128 ? 0 : 16 * (I & 1);
    const int g = lane >> 2;
    const int c = lane & 3;
#pragma unroll
    for (int mi = 0; mi < C::MT; ++mi) {
        const int src = 16 * (mi & 1) + g + 8 * (c & 1);
        f.a[mi] = __shfl_sync(kFullMask, ch.a[mi / 2][WI], src) >> SHIFT;
    }
#pragma unroll
    for (int nj = 0; nj < C::NT; ++nj) {
        f.b[nj] = __shfl_sync(kFullMask, ch.b[nj / 4][WI], 8 * (nj & 3) + g) >> SHIFT;
    }
}

// Runs `fn(std::integral_constant<int, i>{})` for i in [0, N): the stage index inside a
// group has to be a compile-time constant (it selects the chunk word and the byte-id).
template <int N, class F>
__device__ __forceinline__ void static_for(F&& fn) {
    if constexpr (N > 0) {
        static_for<N - 1>(fn);
        fn(std::integral_constant<int, N - 1>{});
    }
}

// The MX k-loop shared by the cp.async and TMA mainloops: `stage(kt, words)` runs one stage,
// `fetch(chunk, kt)` starts one group's chunk loads. The next group's chunks are fetched while
// this group runs, then copied over (12 moves per group). Shuffling each stage's words out
// one stage ahead of its mma, or alternating two register sets instead of copying, measured
// 3 to 8% slower on variant 2 (docs/design/fp8gemm.md).
template <class C, int SFW, class Stage, class Fetch>
__device__ __forceinline__ void mx_loop(int nkt, int lane, Stage&& stage, Fetch&& fetch) {
    using Ch = MxChunk<C, SFW>;
    constexpr int SPC = Ch::SPC;
    Ch cur, next;
    if (nkt > 0) fetch(cur, 0);
    for (int kt = 0; kt < nkt; kt += SPC) {
        if (kt + SPC < nkt) fetch(next, kt + SPC);
        static_for<SPC>([&](auto i) {
            if (kt + i.value < nkt) {
                MxFrag<C> f;
                mx_words<C, SFW, i.value>(f, cur, lane);
                stage(kt + i.value, f);
            }
        });
        cur = next;
    }
}

// One k32 step of the warp tile out of a stage whose A rows start at `as` and Bt rows at `bs`
// (both BK bytes per row, swizzled): MT ldmatrix.x4 for A, NT/2 for B, MT x NT mma. In MX
// mode the mma takes the stage's scale words, byte KK / 32 of them; KK is a compile-time
// constant so that the byte selector is the immediate the instruction wants.
template <class C, bool MX, int KK>
__device__ __forceinline__ void mma_step(const unsigned char* as, const unsigned char* bs, int wm,
                                         int wn, int lane, typename C::Acc& acc,
                                         const MxFrag<C>& mx) {
    constexpr int kk = KK;
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
    for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
        for (int nj = 0; nj < NT; ++nj) {
            if constexpr (MX)
                mma_e4m3_16832_mx<KK / 32>(acc[mi][nj], afrag[mi], bfrag[nj], mx.a[mi], mx.b[nj]);
            else
                mma_e4m3_16832(acc[mi][nj], afrag[mi], bfrag[nj]);
        }
    }
}

// The BK / 32 k32 steps of one stage.
template <class C, bool MX>
__device__ __forceinline__ void mma_stage(const unsigned char* as, const unsigned char* bs, int wm,
                                          int wn, int lane, typename C::Acc& acc,
                                          const MxFrag<C>& mx) {
    mma_step<C, MX, 0>(as, bs, wm, wn, lane, acc, mx);
    mma_step<C, MX, 32>(as, bs, wm, wn, lane, acc, mx);
    if constexpr (C::BK >= 128) {
        mma_step<C, MX, 64>(as, bs, wm, wn, lane, acc, mx);
        mma_step<C, MX, 96>(as, bs, wm, wn, lane, acc, mx);
    }
    static_assert(C::BK == 64 || C::BK == 128, "one stage is two or four k32 steps");
}

// Accumulates A[bm.., kt_begin*BK .. (kt_begin+nkt)*BK) x Bt[bn.., same K]^T into `acc`,
// which it zeroes first. Rows of A past m_valid are zero-filled (cp.async with a source size
// of 0) and Bt rows bn..bn+BN must exist (N % BN == 0). `smem` is the STAGES-deep tile buffer
// (C::SMEM bytes). The pipeline is drained on return. In MX mode `mx` holds the scale
// tensors (whole, not offset), kt_begin is a multiple of the group length and the k-loop
// runs a group of SPC stages per iteration with the next group's chunks in flight.
template <class C, bool MX, int SFW>
__device__ __forceinline__ void mainloop(const unsigned char* __restrict__ A,
                                         const unsigned char* __restrict__ Bt, int K, int bm,
                                         int bn, int m_valid, int kt_begin, int nkt,
                                         unsigned char* smem, typename C::Acc& acc, MxScales mx) {
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

    // One stage: wait for its tile, refill the stage freed by the previous one, run the mma.
    auto stage = [&](int kt, const MxFrag<C>& words) {
        cp_async_wait<STAGES - 2>();  // tile kt has landed (for this thread)
        __syncthreads();              // ... for every thread; and stage (kt-1)%STAGES is free
        {
            const int nk = kt + STAGES - 1;
            if (nk < nkt) load_stage(nk % STAGES, nk * BK);
            cp_async_commit();  // always commit so the group count stays uniform
        }
        const unsigned char* as = smem + (kt % STAGES) * STAGE;
        const unsigned char* bs = as + A_STAGE;
        mma_stage<C, MX>(as, bs, wm, wn, lane, acc, words);
    };

    if constexpr (!MX) {
        MxFrag<C> none;
        for (int kt = 0; kt < nkt; ++kt) stage(kt, none);
    } else {
        mx.a += static_cast<size_t>(bm) * mx.ld;
        mx.b += static_cast<size_t>(bn) * mx.ld;
        mx_loop<C, SFW>(nkt, lane, stage, [&](MxChunk<C, SFW>& ch, int kt) {
            mx_fetch<C, SFW>(ch, mx, kt_begin + kt, wm, wn, lane, m_valid);
        });
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
