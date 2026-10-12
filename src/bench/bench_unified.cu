// Benchmark: what a kernel gets from each kind of memory on a unified-memory machine, and what
// a CPU-to-GPU handoff of a few bytes costs without a copy.
//
//   ./bench_unified                                   # every kind, every advice, 1 GB
//   ./bench_unified --kinds=managed,system --gb=4     # the two kinds the GB10 is about
//   ./bench_unified --mb=256 --iters=10 --trips=200   # a quick pass
//   ./bench_unified --advice=none,prefetch_gpu --skip-signal
//
// For each memory kind (device, managed, host_mapped, system; docs/design/unified.md) and each
// advice setting the runtime accepts for it (none, preferred_gpu, preferred_cpu,
// accessed_by_gpu, read_mostly, prefetch_gpu) the bench fills a buffer of --gb GB from the
// host, then:
//   read   checks every unified_read_bf16 variant's sum against the CPU's double sum and times
//          it; the first launch after the fill is timed on its own (first_ms on stderr), since
//          on a discrete card that launch is the page migration
//   gemv   checks every unified_gemv_bf16 variant against cuBLAS on a device copy of the same
//          weight and times it, the weight rotated over the copies the buffer holds so the
//          bytes come from memory, not L2; shapes: the Llama-3-8B decode projections
//          (4096x4096, 14336x4096, 4096x14336) and a DeepSeek V4.1 Flash expert (2304x5120)
//   signal (host_mapped, managed, system) a persistent kernel answering --trips request /
//          acknowledge rounds, the host timing each round trip with std::chrono; the baseline
//          is a 4-byte cudaMemcpyAsync up and down with a stream sync, which is what a decode
//          loop pays today to read a token back
// Every JSON row carries kind, advice, the path the system kind took (hmm or registered), the
// bytes read and the device's unified-memory attributes. gbps = bytes read once / time;
// ref_ms is the first-touch launch for read rows, cuBLAS on the device copy for gemv rows
// and the copy path for the signal rows.
// Nothing here has been run: no GPU in reach while it was written.
//
// stdout: one JSON object per row (collected by scripts/run_all_benches.sh)
// stderr: a human table and the notes

#include <cublas_v2.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#include "bench_common.hpp"
#include "spark/kernels.h"

#define CUBLAS_CHECK(expr)                                                                      \
    do {                                                                                        \
        cublasStatus_t _st = (expr);                                                            \
        if (_st != CUBLAS_STATUS_SUCCESS) {                                                     \
            std::fprintf(stderr, "cuBLAS error %d at %s:%d\n", static_cast<int>(_st), __FILE__, \
                         __LINE__);                                                             \
            std::exit(1);                                                                       \
        }                                                                                       \
    } while (0)

namespace {

using spark::UnifiedAdvice;
using spark::UnifiedAlloc;
using spark::UnifiedKind;

constexpr long long kMaxSpins = 20'000'000;  // per round; bounds a flag that never arrives
constexpr double kHostTimeoutS = 5.0;         // per round, host side

struct Options {
    std::vector<UnifiedKind> kinds;
    std::vector<UnifiedAdvice> advice;
    size_t bytes = size_t{1} << 30;
    int iters = 20;
    int trips = 1000;
    bool read = true, gemv = true, signal = true;
};

std::string caps_json;  // "\"caps\":{...}" for every row
std::string path_of(const UnifiedAlloc& a) {
    if (a.kind != UnifiedKind::system) return "-";
    return a.registered ? "registered" : "hmm";
}

std::string make_caps_json(const spark::UnifiedDeviceCaps& c) {
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "\"caps\":{\"sm_count\":%d,\"unified_addressing\":%d,\"integrated\":%d,"
                  "\"pageable_memory_access\":%d,\"pageable_uses_host_page_tables\":%d,"
                  "\"concurrent_managed_access\":%d,\"direct_managed_mem_access_from_host\":%d,"
                  "\"host_native_atomic_supported\":%d,"
                  "\"can_use_host_pointer_for_registered_mem\":%d}",
                  c.sm_count, c.unified_addressing, c.integrated, c.pageable_memory_access,
                  c.pageable_memory_access_uses_host_page_tables, c.concurrent_managed_access,
                  c.direct_managed_mem_access_from_host, c.host_native_atomic_supported,
                  c.can_use_host_pointer_for_registered_mem);
    return buf;
}

void print_unified_header() {
    std::fprintf(stderr, "%-14s %-11s %-15s %-3s %-18s %10s %10s %9s %10s %10s %s\n", "kernel",
                 "kind", "advice", "var", "shape", "median_ms", "min_ms", "GB/s", "ref_ms",
                 "max_abs", "ok");
}

// bench_common's Row plus the fields this bench is about; same table, same JSON keys plus
// kind, advice, path, bytes and caps.
void print_unified_row(const spark::bench::Row& r, const std::string& kind,
                       const std::string& advice, const std::string& path, size_t bytes) {
    std::fprintf(stderr, "%-14s %-11s %-15s %-3d %-18s %10.4f %10.4f %9.1f %10.4f %10.2e %s\n",
                 r.kernel.c_str(), kind.c_str(), advice.c_str(), r.variant, r.shape.c_str(),
                 r.median_ms, r.min_ms, r.gbps, r.ref_ms, r.max_abs_err, r.ok ? "yes" : "NO");
    std::printf(
        "{\"device\":\"%s\",\"kernel\":\"%s\",\"dtype\":\"%s\",\"variant\":%d,\"shape\":\"%s\","
        "\"kind\":\"%s\",\"advice\":\"%s\",\"path\":\"%s\",\"bytes\":%zu,"
        "\"median_ms\":%.6f,\"min_ms\":%.6f,\"gbps\":%.3f,\"tflops\":%.4f,\"ref_ms\":%.6f,"
        "\"max_abs_err\":%.6e,\"max_rel_err\":%.6e,\"ok\":%s,%s}\n",
        spark::bench::config().device.c_str(), r.kernel.c_str(), r.dtype.c_str(), r.variant,
        r.shape.c_str(), kind.c_str(), advice.c_str(), path.c_str(), bytes, r.median_ms, r.min_ms,
        r.gbps, r.tflops, r.ref_ms, r.max_abs_err, r.max_rel_err, r.ok ? "true" : "false",
        caps_json.c_str());
    std::fflush(stdout);
}

template <typename T>
T* dmalloc(size_t n) {
    T* p = nullptr;
    SPARK_CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    return p;
}

std::vector<std::string> split(const std::string& s) {
    std::vector<std::string> out;
    size_t b = 0;
    while (b <= s.size()) {
        size_t e = s.find(',', b);
        if (e == std::string::npos) e = s.size();
        if (e > b) out.push_back(s.substr(b, e - b));
        b = e + 1;
    }
    return out;
}

UnifiedKind kind_of(const std::string& s) {
    for (int k = 0; k < spark::kUnifiedNumKinds; ++k)
        if (s == spark::unified_kind_name(static_cast<UnifiedKind>(k)))
            return static_cast<UnifiedKind>(k);
    throw std::invalid_argument("--kinds: unknown kind " + s);
}

UnifiedAdvice advice_of(const std::string& s) {
    for (int a = 0; a < spark::kUnifiedNumAdvice; ++a)
        if (s == spark::unified_advice_name(static_cast<UnifiedAdvice>(a)))
            return static_cast<UnifiedAdvice>(a);
    throw std::invalid_argument("--advice: unknown advice " + s);
}

// The buffer's contents, built once on the host: values in [-1, 1) from a hash of the index,
// with their exact double sum and the sum of their magnitudes for the tolerance.
struct Staging {
    std::vector<__nv_bfloat16> data;
    double sum = 0.0, sum_abs = 0.0;
};

Staging make_staging(size_t n) {
    Staging s;
    s.data.resize(n);
    for (size_t i = 0; i < n; ++i) {
        const uint32_t h = static_cast<uint32_t>(i) * 2654435761u;
        const float v = static_cast<float>((h >> 12) % 2001u) / 1000.0f - 1.0f;
        const __nv_bfloat16 b = __float2bfloat16(v);
        s.data[i] = b;
        const double d = __bfloat162float(b);
        s.sum += d;
        s.sum_abs += std::fabs(d);
    }
    return s;
}

// A buffer of one kind filled from the staging copy, with the advice applied (or refused).
struct Buffer {
    UnifiedAlloc a;
    bool advised = true;
    const __nv_bfloat16* x() const { return static_cast<const __nv_bfloat16*>(a.ptr); }
    const void* host_src() const { return a.kind == UnifiedKind::device ? a.ptr : a.host_ptr; }
};

Buffer make_buffer(UnifiedKind kind, const Staging& st, UnifiedAdvice advice, int device,
                   cudaStream_t stream) {
    Buffer b;
    const size_t bytes = st.data.size() * sizeof(__nv_bfloat16);
    b.a = spark::unified_alloc(kind, bytes);
    if (kind == UnifiedKind::device) {
        SPARK_CUDA_CHECK(cudaMemcpy(b.a.ptr, st.data.data(), bytes, cudaMemcpyHostToDevice));
    } else {
        // Written by the CPU, as a checkpoint loader would: on a discrete card the managed
        // pages are now resident on the host and the GPU's first touch will move them.
        std::memcpy(b.a.host_ptr, st.data.data(), bytes);
    }
    b.advised = spark::unified_advise(b.a.ptr, bytes, advice, device, stream);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    return b;
}

// ---- read ----------------------------------------------------------------------------------

bool run_read(const Buffer& b, const Staging& st, const std::string& advice, int iters,
              cudaStream_t stream) {
    const int64_t n = static_cast<int64_t>(st.data.size());
    const size_t bytes = st.data.size() * sizeof(__nv_bfloat16);
    const int sink_n = spark::unified_read_sink_floats();
    float* d_sink = dmalloc<float>(sink_n);
    std::vector<float> h_sink(sink_n);
    const std::string kind = spark::unified_kind_name(b.a.kind);
    const std::string shape = std::to_string(bytes >> 20) + "MB";
    const double tol = 1e-5 * st.sum_abs + 1.0;
    bool all_ok = true;

    // The first touch after the host wrote the buffer, on its own.
    cudaEvent_t e0, e1;
    SPARK_CUDA_CHECK(cudaEventCreate(&e0));
    SPARK_CUDA_CHECK(cudaEventCreate(&e1));
    SPARK_CUDA_CHECK(cudaEventRecord(e0, stream));
    spark::unified_read_bf16(b.x(), d_sink, n, spark::unified_read_num_variants() - 1, stream);
    SPARK_CUDA_CHECK(cudaEventRecord(e1, stream));
    SPARK_CUDA_CHECK(cudaEventSynchronize(e1));
    float first_ms = 0.f;
    SPARK_CUDA_CHECK(cudaEventElapsedTime(&first_ms, e0, e1));
    SPARK_CUDA_CHECK(cudaEventDestroy(e0));
    SPARK_CUDA_CHECK(cudaEventDestroy(e1));
    std::fprintf(stderr, "  %s/%s: first touch %.3f ms (%.1f GB/s)\n", kind.c_str(),
                 advice.c_str(), first_ms, bytes / (first_ms * 1e-3) / 1e9);

    for (int v = 0; v < spark::unified_read_num_variants(); ++v) {
        spark::unified_read_bf16(b.x(), d_sink, n, v, stream);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        SPARK_CUDA_CHECK(
            cudaMemcpy(h_sink.data(), d_sink, sink_n * sizeof(float), cudaMemcpyDeviceToHost));
        double got = 0.0;
        for (float f : h_sink) got += f;
        const double err = std::fabs(got - st.sum);
        const bool ok = err <= tol;
        if (!ok)
            std::fprintf(stderr, "  FAIL read %s/%s variant %d: sum %.6f, cpu %.6f, tol %.3f\n",
                         kind.c_str(), advice.c_str(), v, got, st.sum, tol);
        const auto t = spark::bench::time_kernel(
            [&] { spark::unified_read_bf16(b.x(), d_sink, n, v, stream); }, stream, 3, iters);
        spark::bench::Row r;
        r.kernel = "unified_read";
        r.dtype = "bf16";
        r.variant = v;
        r.shape = shape;
        r.median_ms = t.median_ms;
        r.min_ms = t.min_ms;
        r.gbps = static_cast<double>(bytes) / (t.median_ms * 1e-3) / 1e9;
        r.ref_ms = first_ms;
        r.max_abs_err = err;
        r.max_rel_err = err / std::max(1.0, std::fabs(st.sum));
        r.ok = ok;
        print_unified_row(r, kind, advice, path_of(b.a), bytes);
        all_ok = all_ok && ok;
    }
    SPARK_CUDA_CHECK(cudaFree(d_sink));
    return all_ok;
}

// ---- gemv ----------------------------------------------------------------------------------

struct GemvShape {
    int N, K;
    const char* what;
};

// y[N] = W[N, K] . a[K] through column-major cuBLAS: C(N x 1) = W^T(N x K, W stored K x N
// column-major with ld K) * a(K x 1).
void cublas_gemv(cublasHandle_t h, const __nv_bfloat16* W, const __nv_bfloat16* A,
                 __nv_bfloat16* y, int N, int K) {
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, 1, K, &alpha, W, CUDA_R_16BF, K, A,
                              CUDA_R_16BF, K, &beta, y, CUDA_R_16BF, N, CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT));
}

std::vector<float> to_host_f32(const __nv_bfloat16* d, size_t n) {
    std::vector<__nv_bfloat16> h(n);
    SPARK_CUDA_CHECK(cudaMemcpy(h.data(), d, n * 2, cudaMemcpyDeviceToHost));
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i) out[i] = __bfloat162float(h[i]);
    return out;
}

bool run_gemv(cublasHandle_t handle, const Buffer& b, size_t bytes, const std::string& advice,
              const GemvShape& s, int iters, cudaStream_t stream) {
    const size_t nW = static_cast<size_t>(s.N) * s.K;
    if (nW * 2 > bytes) {
        std::fprintf(stderr, "  skip gemv %dx%d: needs %zu MB, buffer has %zu\n", s.N, s.K,
                     (nW * 2) >> 20, bytes >> 20);
        return true;
    }
    const int copies = static_cast<int>(bytes / (nW * 2));
    const std::string kind = spark::unified_kind_name(b.a.kind);
    const std::string shape = std::to_string(s.N) + "x" + std::to_string(s.K);

    // The activation, random bf16 on the device; the reference on a device copy of W.
    std::vector<float> hA(s.K);
    spark::bench::fill_uniform(hA, -1.0f, 1.0f, 1234 + s.K);
    std::vector<__nv_bfloat16> hAb(s.K);
    for (int k = 0; k < s.K; ++k) hAb[k] = __float2bfloat16(hA[k]);
    __nv_bfloat16* dA = dmalloc<__nv_bfloat16>(s.K);
    __nv_bfloat16* dW = dmalloc<__nv_bfloat16>(nW);
    __nv_bfloat16* dy = dmalloc<__nv_bfloat16>(s.N);
    __nv_bfloat16* dref = dmalloc<__nv_bfloat16>(s.N);
    SPARK_CUDA_CHECK(cudaMemcpy(dA, hAb.data(), s.K * 2, cudaMemcpyHostToDevice));
    SPARK_CUDA_CHECK(cudaMemcpy(dW, b.host_src(), nW * 2, cudaMemcpyDefault));
    cublas_gemv(handle, dW, dA, dref, s.N, s.K);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    const auto ref = to_host_f32(dref, s.N);
    double max_ref = 0.0;
    for (float f : ref) max_ref = std::max(max_ref, std::fabs(static_cast<double>(f)));
    // bf16 output on both sides, fp32 sums in different orders: one bf16 ulp of the largest
    // output plus a little.
    const double tol = 1e-2 * max_ref + 1e-2;
    const auto t_ref = spark::bench::time_kernel(
        [&] { cublas_gemv(handle, dW, dA, dref, s.N, s.K); }, stream, 3, iters);

    bool all_ok = true;
    int turn = 0;
    for (int v = 0; v < spark::unified_gemv_num_variants(); ++v) {
        SPARK_CUDA_CHECK(cudaMemset(dy, 0, s.N * 2));
        spark::unified_gemv_bf16(dA, b.x(), dy, s.N, s.K, v, stream);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto got = to_host_f32(dy, s.N);
        const auto err = spark::bench::compare(got.data(), ref.data(), s.N);
        const bool ok = err.max_abs <= tol;
        if (!ok)
            std::fprintf(stderr, "  FAIL gemv %s/%s variant %d %s: max_abs %.4e tol %.4e\n",
                         kind.c_str(), advice.c_str(), v, shape.c_str(), err.max_abs, tol);
        const auto t = spark::bench::time_kernel(
            [&] {
                const __nv_bfloat16* W = b.x() + static_cast<size_t>(turn % copies) * nW;
                spark::unified_gemv_bf16(dA, W, dy, s.N, s.K, v, stream);
                ++turn;
            },
            stream, 3, iters);
        spark::bench::Row r;
        r.kernel = "unified_gemv";
        r.dtype = "bf16";
        r.variant = v;
        r.shape = shape;
        r.median_ms = t.median_ms;
        r.min_ms = t.min_ms;
        r.gbps = static_cast<double>(nW * 2) / (t.median_ms * 1e-3) / 1e9;
        r.tflops = 2.0 * nW / (t.median_ms * 1e-3) / 1e12;
        r.ref_ms = t_ref.median_ms;
        r.max_abs_err = err.max_abs;
        r.max_rel_err = err.max_rel;
        r.ok = ok;
        print_unified_row(r, kind, advice, path_of(b.a), nW * 2);
        all_ok = all_ok && ok;
    }
    std::fprintf(stderr, "    %s: cuBLAS on the device copy %.1f us, weight rotated over %d copies\n",
                 s.what, t_ref.median_ms * 1e3, copies);
    SPARK_CUDA_CHECK(cudaFree(dA));
    SPARK_CUDA_CHECK(cudaFree(dW));
    SPARK_CUDA_CHECK(cudaFree(dy));
    SPARK_CUDA_CHECK(cudaFree(dref));
    return all_ok;
}

// ---- signal --------------------------------------------------------------------------------

using SteadyClock = std::chrono::steady_clock;

double median_us(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

// The copy path a decode loop uses today: 4 bytes up, 4 bytes down, a stream sync.
double copy_round_trip_us(int trips, cudaStream_t stream) {
    int* d_flag = dmalloc<int>(1);
    int h_in = 0, h_out = 0;
    std::vector<double> us;
    us.reserve(trips);
    for (int i = 0; i < trips; ++i) {
        h_in = i;
        const auto t0 = SteadyClock::now();
        SPARK_CUDA_CHECK(cudaMemcpyAsync(d_flag, &h_in, 4, cudaMemcpyHostToDevice, stream));
        SPARK_CUDA_CHECK(cudaMemcpyAsync(&h_out, d_flag, 4, cudaMemcpyDeviceToHost, stream));
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto t1 = SteadyClock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        if (h_out != i) std::fprintf(stderr, "  copy round trip: read back %d, wrote %d\n", h_out, i);
    }
    SPARK_CUDA_CHECK(cudaFree(d_flag));
    return median_us(us);
}

bool run_signal(UnifiedKind kind, const std::string& advice_name, bool advise_cpu, int trips,
                double copy_us, int device, cudaStream_t stream) {
    const std::string kind_s = spark::unified_kind_name(kind);
    UnifiedAlloc f = spark::unified_alloc(kind, 4096);
    int* h = static_cast<int*>(f.host_ptr);
    int* d = static_cast<int*>(f.ptr);
    std::memset(h, 0, 4096);
    // req, ack and the iteration count 128 bytes apart: three lines, no false sharing.
    volatile int* d_req = d;
    volatile int* d_ack = d + 32;
    int* d_out = d + 64;
    int* h_req = h;
    int* h_ack = h + 32;
    int* h_out = h + 64;
    if (advise_cpu) {
        // Pages pinned to the host with a GPU mapping: no migration under the spin.
        spark::unified_advise(f.ptr, 4096, UnifiedAdvice::preferred_cpu, device, stream);
        spark::unified_advise(f.ptr, 4096, UnifiedAdvice::accessed_by_gpu, device, stream);
    }

    // One launch-and-see trip: the kernel's launch plus the time until it sees the flag.
    std::vector<double> launch_us;
    for (int i = 0; i < std::max(1, trips / 10); ++i) {
        __atomic_store_n(h_req, 0, __ATOMIC_RELEASE);
        const auto t0 = SteadyClock::now();
        spark::unified_poll_flag(d_req, 1, d_out, kMaxSpins, stream);
        __atomic_store_n(h_req, 1, __ATOMIC_RELEASE);
        SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
        const auto t1 = SteadyClock::now();
        launch_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    {
        spark::bench::Row r;
        r.kernel = "poll_flag";
        r.dtype = "int";
        r.variant = 0;
        r.shape = "launch+1 trip";
        r.median_ms = median_us(launch_us) * 1e-3;
        r.min_ms = launch_us.front() * 1e-3;
        r.ref_ms = copy_us * 1e-3;
        r.ok = __atomic_load_n(h_out, __ATOMIC_ACQUIRE) >= 0;
        print_unified_row(r, kind_s, advice_name, path_of(f), 4);
    }

    // The persistent round trips.
    __atomic_store_n(h_req, 0, __ATOMIC_RELEASE);
    __atomic_store_n(h_ack, 0, __ATOMIC_RELEASE);
    __atomic_store_n(h_out, 0, __ATOMIC_RELEASE);
    spark::unified_ping_pong(d_req, d_ack, trips, d_out, kMaxSpins, stream);
    std::vector<double> us;
    us.reserve(trips);
    bool timed_out = false;
    for (int i = 1; i <= trips && !timed_out; ++i) {
        const auto t0 = SteadyClock::now();
        __atomic_store_n(h_req, i, __ATOMIC_RELEASE);
        while (__atomic_load_n(h_ack, __ATOMIC_ACQUIRE) != i) {
            const double waited =
                std::chrono::duration<double>(SteadyClock::now() - t0).count();
            if (waited > kHostTimeoutS) {
                timed_out = true;
                break;
            }
        }
        const auto t1 = SteadyClock::now();
        us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
    if (timed_out)
        std::fprintf(stderr,
                     "  FAIL ping_pong %s/%s: the GPU did not answer within %.0f s (the kernel "
                     "returns after %lld spins; see the memory rules in kernels.h)\n",
                     kind_s.c_str(), advice_name.c_str(), kHostTimeoutS, kMaxSpins);
    SPARK_CUDA_CHECK(cudaStreamSynchronize(stream));
    const int spins = __atomic_load_n(h_out, __ATOMIC_ACQUIRE);
    const bool ok = !timed_out && spins >= 0;
    spark::bench::Row r;
    r.kernel = "ping_pong";
    r.dtype = "int";
    r.variant = 0;
    r.shape = "trips=" + std::to_string(trips);
    r.median_ms = median_us(us) * 1e-3;
    r.min_ms = (us.empty() ? 0.0 : us.front()) * 1e-3;
    r.ref_ms = copy_us * 1e-3;
    r.max_abs_err = spins < 0 ? 1.0 : 0.0;
    r.ok = ok;
    print_unified_row(r, kind_s, advice_name, path_of(f), 8);
    std::fprintf(stderr, "    round trip median %.2f us, min %.2f us, copy path %.2f us, %d GPU spins per trip\n",
                 r.median_ms * 1e3, r.min_ms * 1e3, copy_us, spins);
    spark::unified_free(f);
    return ok;
}

}  // namespace

int run(int argc, char** argv) {
    spark::bench::Args args(argc, argv);
    spark::bench::print_device_banner();
    int device = 0;
    SPARK_CUDA_CHECK(cudaGetDevice(&device));
    const spark::UnifiedDeviceCaps caps = spark::unified_device_caps(device);
    caps_json = make_caps_json(caps);
    std::fprintf(stderr,
                 "unified memory: pageableMemoryAccess %d (uses host page tables %d), "
                 "concurrentManagedAccess %d, directManagedMemAccessFromHost %d, "
                 "hostNativeAtomicSupported %d, integrated %d\n",
                 caps.pageable_memory_access, caps.pageable_memory_access_uses_host_page_tables,
                 caps.concurrent_managed_access, caps.direct_managed_mem_access_from_host,
                 caps.host_native_atomic_supported, caps.integrated);

    Options o;
    o.iters = args.geti("iters", 20);
    o.trips = args.geti("trips", 1000);
    if (args.has("mb"))
        o.bytes = static_cast<size_t>(args.geti64("mb", 1024)) << 20;
    else
        o.bytes = static_cast<size_t>(args.geti64("gb", 1)) << 30;
    SPARK_REQUIRE(o.bytes >= (size_t{1} << 20), "--gb / --mb: at least 1 MB");
    o.read = !args.has("skip-read");
    o.gemv = !args.has("skip-gemv");
    o.signal = !args.has("skip-signal");
    for (const auto& k : split(args.get("kinds", "device,managed,host_mapped,system")))
        o.kinds.push_back(kind_of(k));
    for (const auto& a : split(args.get(
             "advice", "none,preferred_gpu,preferred_cpu,accessed_by_gpu,read_mostly,prefetch_gpu")))
        o.advice.push_back(advice_of(a));

    cudaStream_t stream;
    SPARK_CUDA_CHECK(cudaStreamCreate(&stream));
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    const std::vector<GemvShape> shapes = {
        {4096, 4096, "Llama-3-8B o_proj / q_proj"},
        {14336, 4096, "Llama-3-8B gate or up"},
        {4096, 14336, "Llama-3-8B down"},
        {2304, 5120, "DeepSeek V4.1 Flash expert gate or up"},
    };

    print_unified_header();
    bool all_ok = true;
    if (o.read || o.gemv) {
        std::fprintf(stderr, "filling %zu MB on the host once...\n", o.bytes >> 20);
        const Staging st = make_staging(o.bytes / sizeof(__nv_bfloat16));
        for (UnifiedKind kind : o.kinds) {
            for (UnifiedAdvice adv : o.advice) {
                const std::string adv_s = spark::unified_advice_name(adv);
                // Device and host_mapped memory take no advice; only their "none" row exists.
                if (adv != UnifiedAdvice::none &&
                    (kind == UnifiedKind::device || kind == UnifiedKind::host_mapped))
                    continue;
                Buffer b = make_buffer(kind, st, adv, device, stream);
                if (!b.advised) {
                    std::fprintf(stderr, "  %s: the runtime refused %s on this memory (path %s); no row\n",
                                 spark::unified_kind_name(kind), adv_s.c_str(),
                                 path_of(b.a).c_str());
                    spark::unified_free(b.a);
                    continue;
                }
                if (o.read) all_ok = run_read(b, st, adv_s, o.iters, stream) && all_ok;
                if (o.gemv)
                    for (const auto& s : shapes)
                        all_ok = run_gemv(handle, b, o.bytes, adv_s, s, o.iters, stream) && all_ok;
                spark::unified_free(b.a);
            }
        }
    }
    if (o.signal) {
        const double copy_us = copy_round_trip_us(o.trips, stream);
        std::fprintf(stderr, "signal: cudaMemcpyAsync 4 B up + 4 B down + sync, median %.2f us\n",
                     copy_us);
        for (UnifiedKind kind : o.kinds) {
            if (kind == UnifiedKind::device) continue;  // the host cannot write it in place
            if (kind == UnifiedKind::managed && !caps.concurrent_managed_access) {
                std::fprintf(stderr, "  managed: no concurrentManagedAccess, a host touch during "
                                     "the kernel would fault; skipped\n");
                continue;
            }
            all_ok = run_signal(kind, "none", false, o.trips, copy_us, device, stream) && all_ok;
            if (kind == UnifiedKind::managed)
                all_ok = run_signal(kind, "preferred_cpu+accessed_by", true, o.trips, copy_us,
                                    device, stream) &&
                         all_ok;
        }
    }
    CUBLAS_CHECK(cublasDestroy(handle));
    SPARK_CUDA_CHECK(cudaStreamDestroy(stream));
    if (!all_ok) {
        std::fprintf(stderr, "CORRECTNESS FAILURE\n");
        return 1;
    }
    return 0;
}

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "bench_unified: %s\n", e.what());
        return 2;
    }
}
