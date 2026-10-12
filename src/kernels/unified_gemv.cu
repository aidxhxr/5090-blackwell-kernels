// The decode GEMV over weights in any kind of memory, and the CPU-GPU signal pair.
//
// y[N] = W[N, K] . A[K] in bf16 with fp32 accumulation, W row-major with K contiguous (the
// layout a Linear's weight is stored in, so no transpose for the experiment). A and y are in
// device memory; W may be in any UnifiedKind (unified.cu): the kernel does not know and does
// not care, which is the point. At batch 1 the whole op is the weight stream, so the GB/s
// the bench computes from N * K * 2 bytes over the time is the GB/s a decode step would see
// if its weights lived in that kind of memory.
//
// One warp per strip of rows; a lane reads 16 bytes (8 k) per row per step, so a warp reads
// 512 contiguous bytes of a row per load instruction. The ladder:
//   0  one warp per row, one 16-byte load per lane per step: the baseline, one load in
//      flight per lane and a grid of N / 4 blocks
//   1  one warp per 4 rows: the four rows' loads issued together and the next step's four
//      already in flight while the current ones are multiplied, 8 loads per lane in flight
//   2  variant 1's warp body on a persistent grid sized from the SM count times the
//      occupancy the kernel reaches, each warp walking row groups with a stride, so a
//      14336-row weight on 48 SMs and a 2304-row expert on 170 SMs both launch the right
//      number of blocks
// K % 8 == 0 and 16-byte aligned A and W; N any.
//
// The signals (unified_poll_flag, unified_set_flag, unified_ping_pong) are the latency half
// of the experiment: how long after the CPU writes a word does a running kernel see it, and
// the other way round, without a cudaMemcpy and without a kernel launch in between. The
// contract is in kernels.h; in short: the flag lives in memory both sides can see at once
// (host_mapped, managed with concurrentManagedAccess, or system memory), every access is a
// system-scope atomic (cuda::atomic_ref<int, cuda::thread_scope_system> here,
// __atomic_load_n / __atomic_store_n with acquire / release on the host), and every spin is
// bounded so a flag that never arrives fails a row instead of hanging the card.
#include <cuda/atomic>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

constexpr int kGemvWarps = 4;  // warps per block
constexpr int kGemvBlock = kGemvWarps * 32;
constexpr int kRowsPerWarp = 4;  // variants 1 and 2

__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// fp32 dot of two 16-byte words of bf16.
__device__ __forceinline__ float dot8(const uint4& a, const uint4& w, float acc) {
    const __nv_bfloat162* pa = reinterpret_cast<const __nv_bfloat162*>(&a);
    const __nv_bfloat162* pw = reinterpret_cast<const __nv_bfloat162*>(&w);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 fa = __bfloat1622float2(pa[i]);
        const float2 fw = __bfloat1622float2(pw[i]);
        acc = fmaf(fa.x, fw.x, acc);
        acc = fmaf(fa.y, fw.y, acc);
    }
    return acc;
}

// Variant 0's warp body: one row, one load per lane per step.
__device__ __forceinline__ void gemv_row(const __nv_bfloat16* __restrict__ A,
                                         const __nv_bfloat16* __restrict__ W,
                                         __nv_bfloat16* __restrict__ y, int K, int row, int lane) {
    const int K8 = K / 8;
    const uint4* a8 = reinterpret_cast<const uint4*>(A);
    const uint4* w8 = reinterpret_cast<const uint4*>(W + static_cast<size_t>(row) * K);
    float acc = 0.f;
    for (int i = lane; i < K8; i += 32) acc = dot8(__ldg(a8 + i), ldg_stream(w8 + i), acc);
    acc = warp_reduce_sum(acc);
    if (lane == 0) y[row] = __float2bfloat16(acc);
}

// Variants 1 and 2's warp body: rows row0 .. row0 + 3, the next step's four loads issued
// before the current step's multiplies. Rows past N read row N - 1 again (valid memory) and
// are not written.
__device__ __forceinline__ void gemv_rows4(const __nv_bfloat16* __restrict__ A,
                                           const __nv_bfloat16* __restrict__ W,
                                           __nv_bfloat16* __restrict__ y, int N, int K, int row0,
                                           int lane) {
    const int K8 = K / 8;
    const uint4* a8 = reinterpret_cast<const uint4*>(A);
    const uint4* w8[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row = min(row0 + r, N - 1);
        w8[r] = reinterpret_cast<const uint4*>(W + static_cast<size_t>(row) * K);
    }
    float acc[kRowsPerWarp] = {0.f, 0.f, 0.f, 0.f};
    int i = lane;
    if (i < K8) {
        uint4 cur[kRowsPerWarp];
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) cur[r] = ldg_stream(w8[r] + i);
        for (;;) {
            const int j = i + 32;
            const bool more = j < K8;
            uint4 nxt[kRowsPerWarp];
            if (more) {
#pragma unroll
                for (int r = 0; r < kRowsPerWarp; ++r) nxt[r] = ldg_stream(w8[r] + j);
            }
            const uint4 a = __ldg(a8 + i);
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) acc[r] = dot8(a, cur[r], acc[r]);
            if (!more) break;
#pragma unroll
            for (int r = 0; r < kRowsPerWarp; ++r) cur[r] = nxt[r];
            i = j;
        }
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const float s = warp_reduce_sum(acc[r]);
        if (lane == 0 && row0 + r < N) y[row0 + r] = __float2bfloat16(s);
    }
}

__global__ void __launch_bounds__(kGemvBlock)
    gemv_v0_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ W,
                   __nv_bfloat16* __restrict__ y, int N, int K) {
    const int row = blockIdx.x * kGemvWarps + (threadIdx.x >> 5);
    if (row >= N) return;
    gemv_row(A, W, y, K, row, threadIdx.x & 31);
}

__global__ void __launch_bounds__(kGemvBlock)
    gemv_v1_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ W,
                   __nv_bfloat16* __restrict__ y, int N, int K) {
    const int group = blockIdx.x * kGemvWarps + (threadIdx.x >> 5);
    const int row0 = group * kRowsPerWarp;
    if (row0 >= N) return;
    gemv_rows4(A, W, y, N, K, row0, threadIdx.x & 31);
}

// Persistent: warp g of the grid takes row groups g, g + warps, g + 2 warps, ...
__global__ void __launch_bounds__(kGemvBlock)
    gemv_v2_kernel(const __nv_bfloat16* __restrict__ A, const __nv_bfloat16* __restrict__ W,
                   __nv_bfloat16* __restrict__ y, int N, int K) {
    const int warps = gridDim.x * kGemvWarps;
    const int groups = (N + kRowsPerWarp - 1) / kRowsPerWarp;
    for (int g = blockIdx.x * kGemvWarps + (threadIdx.x >> 5); g < groups; g += warps)
        gemv_rows4(A, W, y, N, K, g * kRowsPerWarp, threadIdx.x & 31);
}

// Blocks per SM the persistent kernel can keep resident, asked of the runtime once.
int persistent_blocks_per_sm() {
    static int per_sm = 0;
    if (per_sm == 0) {
        SPARK_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, gemv_v2_kernel,
                                                                        kGemvBlock, 0));
        if (per_sm <= 0) per_sm = 8;
    }
    return per_sm;
}

// ---- the signals -------------------------------------------------------------------------

using sys_atomic = cuda::atomic_ref<int, cuda::thread_scope_system>;

__global__ void poll_flag_kernel(int* flag, int expect, int* out_iters, long long max_spins) {
    sys_atomic f(*flag);
    long long it = 0;
    while (f.load(cuda::memory_order_acquire) != expect) {
        if (++it >= max_spins) {
            *out_iters = -1;
            return;
        }
    }
    *out_iters = static_cast<int>(it > INT_MAX ? INT_MAX : it);
}

__global__ void set_flag_kernel(int* flag, int value) {
    sys_atomic f(*flag);
    f.store(value, cuda::memory_order_release);
}

// rounds trips: wait for req == i, answer ack = i. out_iters gets the mean spins per round,
// or -1 if a round timed out (the host sees ack stop advancing and times out as well).
__global__ void ping_pong_kernel(int* req, int* ack, int rounds, int* out_iters,
                                 long long max_spins) {
    sys_atomic r(*req), a(*ack);
    long long total = 0;
    for (int i = 1; i <= rounds; ++i) {
        long long it = 0;
        while (r.load(cuda::memory_order_acquire) != i) {
            if (++it >= max_spins) {
                *out_iters = -1;
                return;
            }
        }
        total += it;
        a.store(i, cuda::memory_order_release);
    }
    const long long mean = rounds > 0 ? total / rounds : 0;
    *out_iters = static_cast<int>(mean > INT_MAX ? INT_MAX : mean);
}

}  // namespace

int unified_gemv_num_variants() {
    return 3;
}

void unified_gemv_bf16(const __nv_bfloat16* A, const __nv_bfloat16* W, __nv_bfloat16* y, int N,
                       int K, int variant, cudaStream_t stream) {
    SPARK_REQUIRE(A && W && y, "unified_gemv: null pointer");
    SPARK_REQUIRE(N > 0 && K > 0 && K % 8 == 0, "unified_gemv: needs N > 0 and K % 8 == 0");
    SPARK_REQUIRE(variant >= 0 && variant < unified_gemv_num_variants(),
                  "unified_gemv: unknown variant");
    SPARK_REQUIRE(is_aligned16(A) && is_aligned16(W),
                  "unified_gemv: A and W must be 16-byte aligned");
    switch (variant) {
        case 0:
            gemv_v0_kernel<<<cdiv(N, kGemvWarps), kGemvBlock, 0, stream>>>(A, W, y, N, K);
            break;
        case 1:
            gemv_v1_kernel<<<cdiv(cdiv(N, kRowsPerWarp), kGemvWarps), kGemvBlock, 0, stream>>>(
                A, W, y, N, K);
            break;
        default: {
            const int need = cdiv(cdiv(N, kRowsPerWarp), kGemvWarps);
            const int want = num_sms() * persistent_blocks_per_sm();
            gemv_v2_kernel<<<std::min(need, want), kGemvBlock, 0, stream>>>(A, W, y, N, K);
            break;
        }
    }
    SPARK_CHECK_LAUNCH();
}

void unified_poll_flag(volatile int* flag, int expect, int* out_iters, long long max_spins,
                       cudaStream_t stream) {
    SPARK_REQUIRE(flag != nullptr && out_iters != nullptr, "unified_poll_flag: null pointer");
    SPARK_REQUIRE(max_spins > 0, "unified_poll_flag: max_spins must be > 0");
    poll_flag_kernel<<<1, 1, 0, stream>>>(const_cast<int*>(flag), expect, out_iters, max_spins);
    SPARK_CHECK_LAUNCH();
}

void unified_set_flag(volatile int* flag, int value, cudaStream_t stream) {
    SPARK_REQUIRE(flag != nullptr, "unified_set_flag: null pointer");
    set_flag_kernel<<<1, 1, 0, stream>>>(const_cast<int*>(flag), value);
    SPARK_CHECK_LAUNCH();
}

void unified_ping_pong(volatile int* req, volatile int* ack, int rounds, int* out_iters,
                       long long max_spins, cudaStream_t stream) {
    SPARK_REQUIRE(req != nullptr && ack != nullptr && out_iters != nullptr,
                  "unified_ping_pong: null pointer");
    SPARK_REQUIRE(rounds > 0 && max_spins > 0,
                  "unified_ping_pong: rounds and max_spins must be > 0");
    ping_pong_kernel<<<1, 1, 0, stream>>>(const_cast<int*>(req), const_cast<int*>(ack), rounds,
                                          out_iters, max_spins);
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
