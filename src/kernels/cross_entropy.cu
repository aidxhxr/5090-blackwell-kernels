// Linear cross-entropy for a large vocabulary: loss = CE(h w^T, y) with h [T, D] and the
// lm_head weight w [V, D] (nn.Linear's layout), without torch's [T, V] logits and their
// fp32 copies (docs/design/cross_entropy.md). This file holds the pieces; the ladder that
// puts them together is linear_cross_entropy in python/spark_kernels/ops.py.
//
// The row kernel is the step after the logits of one chunk of rows: it reads a logits row
// and writes the row's loss and, in place, its gradient row (softmax minus one-hot, times
// the reduction scale). The transpose turns a slab of that gradient into the K-contiguous
// operand of dw += g^T h.

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

namespace ce {

constexpr float kLog2e = 1.4426950408889634f;

// (max, sum) merge of the online softmax; an empty side is (-inf, 0).
__device__ __forceinline__ void merge(float& m, float& s, float m2, float s2) {
    const float mx = fmaxf(m, m2);
    if (mx == -INFINITY) return;
    s = s * exp2f((m - mx) * kLog2e) + s2 * exp2f((m2 - mx) * kLog2e);
    m = mx;
}

// ---- the v1 row kernel ------------------------------------------------------------------
//
// One block of ROW_THREADS per row, a persistent grid of one block per SM, so 170 rows are
// in flight: 170 x 256 KB at V = 128256, 44 MB, which leaves the second read of a row to L2.
// Pass 1 is the online (max, sum) over 16-byte vectors; thread 0 reads the target's logit
// before the reduction, since pass 2 overwrites the row in place. Pass 2 writes
// (exp(x - lse) - [col == y]) * scale over the logits. A row that does not count is zeroed
// without being read.
constexpr int ROW_THREADS = 1024;

__device__ __forceinline__ void block_merge(float& m, float& s, float2* sh) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float m2 = __shfl_xor_sync(kFullMask, m, o);
        const float s2 = __shfl_xor_sync(kFullMask, s, o);
        merge(m, s, m2, s2);
    }
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    if (lane == 0) sh[wid] = make_float2(m, s);
    __syncthreads();
    if (wid == 0) {
        const int nw = blockDim.x >> 5;
        float2 v = lane < nw ? sh[lane] : make_float2(-INFINITY, 0.f);
        m = v.x;
        s = v.y;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float m2 = __shfl_xor_sync(kFullMask, m, o);
            const float s2 = __shfl_xor_sync(kFullMask, s, o);
            merge(m, s, m2, s2);
        }
        if (lane == 0) sh[32] = make_float2(m, s);
    }
    __syncthreads();
    m = sh[32].x;
    s = sh[32].y;
    __syncthreads();  // sh is reused by the next row
}

__global__ void __launch_bounds__(ROW_THREADS)
    ce_rows_kernel(__nv_bfloat16* __restrict__ logits, const int64_t* __restrict__ targets,
                   int rows, int V, int ignore_index, const float* __restrict__ scale,
                   float* __restrict__ loss, float* __restrict__ lse_out, bool write_grad) {
    __shared__ float2 sh[33];
    const int nvec = V / 8;
    const float sc = write_grad ? __ldg(scale) : 0.f;
    for (int row = blockIdx.x; row < rows; row += gridDim.x) {
        bf16x8* x = reinterpret_cast<bf16x8*>(logits + static_cast<size_t>(row) * V);
        const long long y = targets[row];
        const bool counts = y != ignore_index && y >= 0 && y < V;
        if (!counts) {  // uniform across the block
            if (write_grad) {
                bf16x8 z;
#pragma unroll
                for (int e = 0; e < 4; ++e) z.h[e] = __floats2bfloat162_rn(0.f, 0.f);
                for (int i = threadIdx.x; i < nvec; i += blockDim.x) x[i] = z;
            }
            if (threadIdx.x == 0) {
                loss[row] = 0.f;
                lse_out[row] = 0.f;
            }
            continue;
        }
        float xy = 0.f;
        if (threadIdx.x == 0) xy = __bfloat162float(logits[static_cast<size_t>(row) * V + y]);
        float m = -INFINITY, s = 0.f;
        for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
            const bf16x8 v = x[i];
            float f[8];
            float lm = -INFINITY;
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 q = __bfloat1622float2(v.h[e]);
                f[2 * e] = q.x;
                f[2 * e + 1] = q.y;
                lm = fmaxf(lm, fmaxf(q.x, q.y));
            }
            if (lm > m) {
                s *= exp2f((m - lm) * kLog2e);
                m = lm;
            }
            const float mb = m * kLog2e;
#pragma unroll
            for (int e = 0; e < 8; ++e) s += exp2f(fmaf(f[e], kLog2e, -mb));
        }
        block_merge(m, s, sh);
        const float l = m + logf(s);
        if (threadIdx.x == 0) {
            loss[row] = l - xy;
            lse_out[row] = l;
        }
        if (!write_grad) continue;
        const float lb = l * kLog2e;
        const int yv = static_cast<int>(y);
        for (int i = threadIdx.x; i < nvec; i += blockDim.x) {
            bf16x8 v = x[i];
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const float2 q = __bfloat1622float2(v.h[e]);
                float p0 = exp2f(fmaf(q.x, kLog2e, -lb));
                float p1 = exp2f(fmaf(q.y, kLog2e, -lb));
                if (i * 8 + 2 * e == yv) p0 -= 1.f;
                if (i * 8 + 2 * e + 1 == yv) p1 -= 1.f;
                v.h[e] = __floats2bfloat162_rn(p0 * sc, p1 * sc);
            }
            x[i] = v;
        }
    }
}

// ---- transpose of a column slab, zero-padded --------------------------------------------
//
// out[c][r] = x[r][c0 + c] for r < rows and 0 for rows <= r < ld, c < ncols: the gradient
// chunk [C][V] turned into the [V-slab][C rounded up to 64] A operand of dw += g^T h, which
// hgemm takes only K-contiguous. 64 x 64 tiles through shared memory, rows padded to 66 so
// a column read hits 32 different banks.
constexpr int TP = 64;

__global__ void __launch_bounds__(256)
    transpose_pad_kernel(const __nv_bfloat16* __restrict__ x, int rows, int ldx,
                         __nv_bfloat16* __restrict__ out, int ncols, int ld) {
    __shared__ __nv_bfloat16 tile[TP][TP + 2];
    const int c0 = blockIdx.x * TP;
    const int r0 = blockIdx.y * TP;
    const int tx = threadIdx.x & (TP - 1);
    const int ty = threadIdx.x / TP;
    const __nv_bfloat16 zero = __float2bfloat16(0.f);
#pragma unroll 4
    for (int i = ty; i < TP; i += 256 / TP) {
        const int r = r0 + i, c = c0 + tx;
        tile[i][tx] = (r < rows && c < ncols) ? x[static_cast<size_t>(r) * ldx + c] : zero;
    }
    __syncthreads();
#pragma unroll 4
    for (int i = ty; i < TP; i += 256 / TP) {
        const int c = c0 + i, r = r0 + tx;
        if (c < ncols && r < ld) out[static_cast<size_t>(c) * ld + r] = tile[tx][i];
    }
}

}  // namespace ce

}  // namespace

void cross_entropy_rows_bf16(__nv_bfloat16* logits, const int64_t* targets, int rows, int V,
                             int ignore_index, const float* scale, float* loss, float* lse,
                             cudaStream_t stream) {
    SPARK_REQUIRE(V > 0 && V % 8 == 0, "cross_entropy_rows: V must be a multiple of 8");
    SPARK_REQUIRE(is_aligned16(logits), "cross_entropy_rows: logits must be 16-byte aligned");
    if (rows <= 0) return;
    const int grid = std::min(rows, num_sms());
    ce::ce_rows_kernel<<<grid, ce::ROW_THREADS, 0, stream>>>(logits, targets, rows, V, ignore_index,
                                                             scale, loss, lse, scale != nullptr);
    SPARK_CHECK_LAUNCH();
}

void transpose_pad_bf16(const __nv_bfloat16* x, int rows, int ldx, __nv_bfloat16* out, int ncols,
                        int ld, cudaStream_t stream) {
    SPARK_REQUIRE(ld >= rows && ncols <= ldx, "transpose_pad: bad shape");
    if (ncols <= 0 || ld <= 0) return;
    const dim3 grid(cdiv(ncols, ce::TP), cdiv(ld, ce::TP));
    ce::transpose_pad_kernel<<<grid, 256, 0, stream>>>(x, rows, ldx, out, ncols, ld);
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
