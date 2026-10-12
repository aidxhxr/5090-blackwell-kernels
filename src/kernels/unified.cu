// Unified memory experiments: can a GPU kernel stream weights that live outside cudaMalloc
// memory at the same GB/s, and how fast does a word written by the CPU reach a kernel?
//
// The GB10 (DGX Spark) has one 128 GB pool of LPDDR5X that the Grace cores and the Blackwell
// GPU share over a coherent fabric, and no PCIe copy between them. The RTX 5090 is the usual
// discrete card: 32 GB of GDDR7 behind PCIe 5.0 x16, 64 GB/s each way, and the same CUDA calls
// mean a very different thing there. Nothing in this file has run on either machine yet; it is
// the experiment, written so the same binary builds and runs on both and the design note can
// put the two columns side by side (docs/design/unified.md).
//
// Four kinds of memory a kernel can read (UnifiedKind, unified_alloc):
//   device       cudaMalloc. The baseline every other kind is measured against.
//   managed      cudaMallocManaged. Pages migrate on demand between the host and the device
//                on the 5090; on the GB10 there is only one physical pool, so "migration" is
//                a page-table operation at most. cudaMemAdvise and cudaMemPrefetchAsync apply.
//   host_mapped  cudaHostAlloc(cudaHostAllocMapped). Pinned host pages the GPU reads in place:
//                over PCIe on the 5090, over the fabric on the GB10. Never migrates.
//   system       posix_memalign. Plain pageable memory, the kind every mmap'd checkpoint and
//                every torch CPU tensor already is. When the device reports
//                pageableMemoryAccess (HMM on Linux, or ATS on Grace parts) the GPU can read
//                it with no CUDA call at all; otherwise unified_alloc falls back to
//                cudaHostRegister and records that it did, so the row says which path ran.
//
// Two kernels and a pair of signals:
//   unified_read_bf16     streams a bf16 array into a per-block fp32 sink, so the bytes are
//                         really read and the sum can be checked against the CPU. The ladder:
//                         0  one bf16 per thread per step, grid-stride
//                         1  one 16-byte load per thread per step, grid-stride
//                         2  four independent 16-byte loads in flight per thread per step
//                         The grid is sized from the SM count, not from n, so the sink is a
//                         fixed unified_read_sink_floats() entries for any n.
//   unified_gemv_bf16     the decode GEMV, in unified_gemv.cu (the second half of this
//                         experiment); the signals are there too.
//
// Why the loads-in-flight rung matters more here than on the 5090: Little's law. At 273 GB/s
// and a few hundred ns of latency to LPDDR5X across the fabric, a few hundred KB must be in
// flight across the chip; 48 SMs x 2048 threads x 16 bytes is 1.5 MB with one load per thread,
// which looks like enough, until an occupancy of 50% and the latency of a far page halve and
// double the two sides. On the 5090 there are 170 SMs to hide the same latency behind. The
// three rungs say where each machine stops being latency bound.
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>

#include "spark/common.cuh"
#include "spark/kernels.h"

namespace spark {

namespace {

constexpr int kBlock = 256;
constexpr int kBlocksPerSm = 8;

// 16 bytes, read once, no L1 allocation: a streaming read that does not evict the other
// kernels' working sets.
__device__ __forceinline__ uint4 ldg_stream(const uint4* p) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
                 : "l"(p));
    return v;
}

// The fp32 sum of the eight bf16 in a 16-byte word.
__device__ __forceinline__ float sum8(const uint4& v) {
    const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&v);
    float s = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float2 f = __bfloat1622float2(h[i]);
        s += f.x + f.y;
    }
    return s;
}

// Every thread's partial into one float per block. The sink is overwritten, not accumulated,
// so no memset is needed between launches and the host sums gridDim.x floats.
__device__ __forceinline__ void block_sink(float acc, float* __restrict__ sink) {
    __shared__ float red[32];
    acc = block_reduce_sum(acc, red);
    if (threadIdx.x == 0) sink[blockIdx.x] = acc;
}

// Variant 0: one bf16 per thread per step. A warp moves 64 bytes per load instruction, so
// the LSU issue rate, not DRAM, is what this rung measures; the same baseline as
// bandwidth_copy variant 0, grid-stride so the sink stays fixed.
__global__ void __launch_bounds__(kBlock)
    read_scalar_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ sink, int64_t n) {
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    float acc = 0.f;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += stride)
        acc += __bfloat162float(x[i]);
    block_sink(acc, sink);
}

// The n % 8 tail of the vectorized rungs: at most 7 elements, read by one thread.
__device__ __forceinline__ float tail_sum(const __nv_bfloat16* __restrict__ x, int64_t from,
                                          int64_t n) {
    float s = 0.f;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        for (int64_t i = from; i < n; ++i) s += __bfloat162float(x[i]);
    return s;
}

// Variant 1: one 16-byte load per thread per step, grid-stride. One load in flight per
// thread between two dependent adds.
__global__ void __launch_bounds__(kBlock)
    read_vec_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ sink, int64_t n) {
    const int64_t n8 = n / 8;
    const uint4* x8 = reinterpret_cast<const uint4*>(x);
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    float acc = tail_sum(x, n8 * 8, n);
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n8;
         i += stride)
        acc += sum8(ldg_stream(x8 + i));
    block_sink(acc, sink);
}

// Variant 2: four independent 16-byte loads issued before any of them is consumed, so each
// thread has 64 bytes in flight and a block 16 KB; the four partial sums stay separate so the
// adds do not serialize on one accumulator either.
__global__ void __launch_bounds__(kBlock)
    read_vec4_kernel(const __nv_bfloat16* __restrict__ x, float* __restrict__ sink, int64_t n) {
    const int64_t n8 = n / 8;
    const uint4* x8 = reinterpret_cast<const uint4*>(x);
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    float acc0 = tail_sum(x, n8 * 8, n), acc1 = 0.f, acc2 = 0.f, acc3 = 0.f;
    int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n8; i += 4 * stride) {
        const uint4 a = ldg_stream(x8 + i);
        const uint4 b = ldg_stream(x8 + i + stride);
        const uint4 c = ldg_stream(x8 + i + 2 * stride);
        const uint4 d = ldg_stream(x8 + i + 3 * stride);
        acc0 += sum8(a);
        acc1 += sum8(b);
        acc2 += sum8(c);
        acc3 += sum8(d);
    }
    for (; i < n8; i += stride) acc0 += sum8(ldg_stream(x8 + i));
    block_sink((acc0 + acc1) + (acc2 + acc3), sink);
}

int attr(cudaDeviceAttr a, int device) {
    int v = 0;
    SPARK_CUDA_CHECK(cudaDeviceGetAttribute(&v, a, device));
    return v;
}

int current_device() {
    int dev = 0;
    SPARK_CUDA_CHECK(cudaGetDevice(&dev));
    return dev;
}

size_t round_up_page(size_t bytes) {
    const size_t page = 4096;
    return (bytes + page - 1) / page * page;
}

#if CUDART_VERSION >= 13000
// CUDA 13 moved cudaMemAdvise and cudaMemPrefetchAsync to a cudaMemLocation; a device id or
// the host as one.
cudaMemLocation location_of(int device) {
    cudaMemLocation loc{};
    if (device == cudaCpuDeviceId) {
        loc.type = cudaMemLocationTypeHost;
        loc.id = 0;
    } else {
        loc.type = cudaMemLocationTypeDevice;
        loc.id = device;
    }
    return loc;
}
cudaError_t mem_advise(const void* p, size_t bytes, cudaMemoryAdvise advice, int device) {
    return cudaMemAdvise(p, bytes, advice, location_of(device));
}
cudaError_t mem_prefetch(const void* p, size_t bytes, int device, cudaStream_t stream) {
    return cudaMemPrefetchAsync(p, bytes, location_of(device), 0, stream);
}
#else
cudaError_t mem_advise(const void* p, size_t bytes, cudaMemoryAdvise advice, int device) {
    return cudaMemAdvise(p, bytes, advice, device);
}
cudaError_t mem_prefetch(const void* p, size_t bytes, int device, cudaStream_t stream) {
    return cudaMemPrefetchAsync(p, bytes, device, stream);
}
#endif

}  // namespace

// ---- the device's view of unified memory ----------------------------------------------------

UnifiedDeviceCaps unified_device_caps(int device) {
    if (device < 0) device = current_device();
    UnifiedDeviceCaps c;
    c.device = device;
    c.sm_count = attr(cudaDevAttrMultiProcessorCount, device);
    c.unified_addressing = attr(cudaDevAttrUnifiedAddressing, device);
    c.integrated = attr(cudaDevAttrIntegrated, device);
    c.can_map_host_memory = attr(cudaDevAttrCanMapHostMemory, device);
    c.pageable_memory_access = attr(cudaDevAttrPageableMemoryAccess, device);
    c.pageable_memory_access_uses_host_page_tables =
        attr(cudaDevAttrPageableMemoryAccessUsesHostPageTables, device);
    c.concurrent_managed_access = attr(cudaDevAttrConcurrentManagedAccess, device);
    c.direct_managed_mem_access_from_host =
        attr(cudaDevAttrDirectManagedMemAccessFromHost, device);
    c.host_native_atomic_supported = attr(cudaDevAttrHostNativeAtomicSupported, device);
    c.can_use_host_pointer_for_registered_mem =
        attr(cudaDevAttrCanUseHostPointerForRegisteredMem, device);
    return c;
}

const char* unified_kind_name(UnifiedKind kind) {
    switch (kind) {
        case UnifiedKind::device:
            return "device";
        case UnifiedKind::managed:
            return "managed";
        case UnifiedKind::host_mapped:
            return "host_mapped";
        case UnifiedKind::system:
            return "system";
    }
    return "?";
}

const char* unified_advice_name(UnifiedAdvice advice) {
    switch (advice) {
        case UnifiedAdvice::none:
            return "none";
        case UnifiedAdvice::preferred_gpu:
            return "preferred_gpu";
        case UnifiedAdvice::preferred_cpu:
            return "preferred_cpu";
        case UnifiedAdvice::accessed_by_gpu:
            return "accessed_by_gpu";
        case UnifiedAdvice::read_mostly:
            return "read_mostly";
        case UnifiedAdvice::prefetch_gpu:
            return "prefetch_gpu";
        case UnifiedAdvice::prefetch_cpu:
            return "prefetch_cpu";
    }
    return "?";
}

// ---- allocation ----------------------------------------------------------------------------

UnifiedAlloc unified_alloc(UnifiedKind kind, size_t bytes) {
    SPARK_REQUIRE(bytes > 0, "unified_alloc: bytes must be > 0");
    UnifiedAlloc a;
    a.kind = kind;
    a.bytes = bytes;
    a.device = current_device();
    switch (kind) {
        case UnifiedKind::device: {
            SPARK_CUDA_CHECK(cudaMalloc(&a.ptr, bytes));
            a.host_ptr = nullptr;  // the host cannot dereference it; copy in and out
            break;
        }
        case UnifiedKind::managed: {
            SPARK_CUDA_CHECK(cudaMallocManaged(&a.ptr, bytes, cudaMemAttachGlobal));
            a.host_ptr = a.ptr;
            break;
        }
        case UnifiedKind::host_mapped: {
            SPARK_CUDA_CHECK(cudaHostAlloc(&a.host_ptr, bytes,
                                           cudaHostAllocMapped | cudaHostAllocPortable));
            // Under unified addressing the two are the same pointer; ask anyway.
            SPARK_CUDA_CHECK(cudaHostGetDevicePointer(&a.ptr, a.host_ptr, 0));
            break;
        }
        case UnifiedKind::system: {
            const size_t rounded = round_up_page(bytes);
            void* p = nullptr;
            const int rc = posix_memalign(&p, 4096, rounded);
            SPARK_REQUIRE(rc == 0 && p != nullptr, "unified_alloc: posix_memalign failed");
            // Touch every page from the host so the pages exist before the GPU sees them: the
            // first-touch policy is part of what is being measured, so it has to be the same
            // every run.
            std::memset(p, 0, rounded);
            a.host_ptr = p;
            if (attr(cudaDevAttrPageableMemoryAccess, a.device)) {
                // HMM / ATS: the GPU dereferences the host's pointer as it is.
                a.ptr = p;
                a.registered = false;
            } else {
                // No pageable access: pin and map the pages the old way, which makes this
                // kind the same thing as host_mapped with a different allocator.
                SPARK_CUDA_CHECK(cudaHostRegister(p, rounded,
                                                  cudaHostRegisterPortable | cudaHostRegisterMapped));
                SPARK_CUDA_CHECK(cudaHostGetDevicePointer(&a.ptr, p, 0));
                a.registered = true;
            }
            break;
        }
    }
    return a;
}

void unified_free(UnifiedAlloc& a) {
    if (a.ptr == nullptr && a.host_ptr == nullptr) return;
    switch (a.kind) {
        case UnifiedKind::device:
        case UnifiedKind::managed:
            SPARK_CUDA_CHECK(cudaFree(a.ptr));
            break;
        case UnifiedKind::host_mapped:
            SPARK_CUDA_CHECK(cudaFreeHost(a.host_ptr));
            break;
        case UnifiedKind::system:
            if (a.registered) SPARK_CUDA_CHECK(cudaHostUnregister(a.host_ptr));
            std::free(a.host_ptr);
            break;
    }
    a.ptr = nullptr;
    a.host_ptr = nullptr;
    a.bytes = 0;
}

// ---- advice --------------------------------------------------------------------------------

bool unified_advise(void* ptr, size_t bytes, UnifiedAdvice advice, int device,
                    cudaStream_t stream) {
    SPARK_REQUIRE(ptr != nullptr && bytes > 0, "unified_advise: null pointer or zero bytes");
    if (device < 0) device = current_device();
    if (advice == UnifiedAdvice::none) return true;
    cudaError_t err = cudaSuccess;
    switch (advice) {
        case UnifiedAdvice::preferred_gpu:
            err = mem_advise(ptr, bytes, cudaMemAdviseSetPreferredLocation, device);
            break;
        case UnifiedAdvice::preferred_cpu:
            err = mem_advise(ptr, bytes, cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId);
            break;
        case UnifiedAdvice::accessed_by_gpu:
            err = mem_advise(ptr, bytes, cudaMemAdviseSetAccessedBy, device);
            break;
        case UnifiedAdvice::read_mostly:
            err = mem_advise(ptr, bytes, cudaMemAdviseSetReadMostly, device);
            break;
        case UnifiedAdvice::prefetch_gpu:
            err = mem_prefetch(ptr, bytes, device, stream);
            break;
        case UnifiedAdvice::prefetch_cpu:
            err = mem_prefetch(ptr, bytes, cudaCpuDeviceId, stream);
            break;
        case UnifiedAdvice::none:
            break;
    }
    if (err == cudaSuccess) return true;
    // The runtime refuses advice on memory it does not manage (cudaMalloc, cudaHostAlloc,
    // and plain pages on a machine without HMM). That is an answer, not a failure: clear the
    // sticky error and tell the caller the advice did not apply.
    if (err == cudaErrorInvalidValue || err == cudaErrorInvalidDevice ||
        err == cudaErrorNotSupported) {
        (void)cudaGetLastError();
        return false;
    }
    SPARK_CUDA_CHECK(err);
    return false;
}

// ---- the streaming read ----------------------------------------------------------------------

int unified_read_num_variants() {
    return 3;
}

int unified_read_sink_floats() {
    return num_sms() * kBlocksPerSm;
}

void unified_read_bf16(const __nv_bfloat16* x, float* sink, int64_t n, int variant,
                       cudaStream_t stream) {
    SPARK_REQUIRE(x != nullptr && sink != nullptr, "unified_read: null pointer");
    SPARK_REQUIRE(n >= 0, "unified_read: n must be >= 0");
    SPARK_REQUIRE(variant >= 0 && variant < unified_read_num_variants(),
                  "unified_read: unknown variant");
    const unsigned grid = static_cast<unsigned>(unified_read_sink_floats());
    if (variant == 0) {
        read_scalar_kernel<<<grid, kBlock, 0, stream>>>(x, sink, n);
    } else {
        SPARK_REQUIRE(is_aligned16(x), "unified_read: vectorized variants need a 16-byte aligned x");
        if (variant == 1)
            read_vec_kernel<<<grid, kBlock, 0, stream>>>(x, sink, n);
        else
            read_vec4_kernel<<<grid, kBlock, 0, stream>>>(x, sink, n);
    }
    SPARK_CHECK_LAUNCH();
}

}  // namespace spark
