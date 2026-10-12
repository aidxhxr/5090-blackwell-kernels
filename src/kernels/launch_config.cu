// The device shape behind launch_config.cuh and its public view in kernels.h.
//
// device_shape() asks the runtime for the SM count, the L2 size, the memory clock and the
// bus width of the current device once and keeps them per device. The spec bandwidth is
// 2 x clock x width / 8 (the runtime reports half the data rate of a DDR-type memory as the
// clock; that gives 1,792 GB/s for the 5090's GDDR7 and should give 273 GB/s for the GB10's
// LPDDR5X, to be checked on the Spark). When the attributes read 0 the named constants are
// used, picked by the SM count. The bytes in flight are the 5090's measured 1.5 MB on a
// 170-SM device and bandwidth x kGB10LoadedLatencyNs anywhere else; the GB10 branch is an
// assumption until the latency is measured.
#include <atomic>
#include <cstdlib>
#include <mutex>

#include "launch_config.cuh"
#include "spark/kernels.h"

namespace spark::launch {

namespace {

constexpr int kMaxDevices = 16;

int read_forced_sms() {
    const char* e = std::getenv("SPARK_FORCE_SMS");
    if (!e || !*e) return 0;
    const int n = std::atoi(e);
    return n > 0 ? n : 0;
}

int read_env_split() {
    const char* e = std::getenv("SPARK_DECODE_SPLIT");
    if (!e || !*e) return 0;
    const int n = std::atoi(e);
    return n > 0 ? n : 0;
}

int attribute(cudaDeviceAttr attr, int dev) {
    int v = 0;
    if (cudaDeviceGetAttribute(&v, attr, dev) != cudaSuccess) {
        (void)cudaGetLastError();  // leave the context clean; the fallback below is used
        return 0;
    }
    return v > 0 ? v : 0;
}

DeviceShape query(int dev) {
    DeviceShape s{};
    s.sm_count = attribute(cudaDevAttrMultiProcessorCount, dev);
    const int l2 = attribute(cudaDevAttrL2CacheSize, dev);
    s.l2_bytes = l2;
    s.memory_clock_khz = attribute(cudaDevAttrMemoryClockRate, dev);
    s.bus_width_bits = attribute(cudaDevAttrGlobalMemoryBusWidth, dev);
    s.forced = false;

    // Named fallbacks: the GB10 when the device looks like one, the 5090 otherwise (also the
    // choice num_sms() makes in common.cuh).
    const bool looks_gb10 =
        s.sm_count == kGB10.sm_count || (s.sm_count == 0 && l2 == kGB10.l2_bytes);
    const DeviceShape& named = looks_gb10 ? kGB10 : kRTX5090;
    if (s.sm_count <= 0) s.sm_count = named.sm_count;
    if (s.l2_bytes <= 0) s.l2_bytes = named.l2_bytes;
    if (s.memory_clock_khz > 0 && s.bus_width_bits > 0) {
        s.spec_gbps = 2.0 * static_cast<double>(s.memory_clock_khz) * 1e3 *
                      (static_cast<double>(s.bus_width_bits) / 8.0) / 1e9;
    } else {
        s.memory_clock_khz = named.memory_clock_khz;
        s.bus_width_bits = named.bus_width_bits;
        s.spec_gbps = named.spec_gbps;
    }
    // Bytes in flight: the measurement where there is one, Little's law with the assumed
    // latency where there is not.
    if (s.sm_count == kRTX5090.sm_count)
        s.bytes_in_flight = kRTX5090.bytes_in_flight;
    else
        s.bytes_in_flight = bytes_in_flight_for(s.spec_gbps, kGB10LoadedLatencyNs);

    const int forced = read_forced_sms();
    if (forced > 0) {
        s.sm_count = forced;
        s.forced = true;
    }
    return s;
}

struct Cache {
    std::once_flag once[kMaxDevices];
    DeviceShape shape[kMaxDevices];
};

Cache& cache() {
    static Cache c;
    return c;
}

std::atomic<int>& forced_split_slot() {
    static std::atomic<int> v{-1};  // -1: SPARK_DECODE_SPLIT not read yet
    return v;
}

}  // namespace

const DeviceShape& device_shape() {
    int dev = 0;
    SPARK_CUDA_CHECK(cudaGetDevice(&dev));
    if (dev < 0 || dev >= kMaxDevices) dev = 0;
    Cache& c = cache();
    std::call_once(c.once[dev], [&] { c.shape[dev] = query(dev); });
    return c.shape[dev];
}

const DeviceShape& device_shape_named(int which) {
    if (which == DECODE_DEVICE_GB10) return kGB10;
    if (which == DECODE_DEVICE_RTX5090) return kRTX5090;
    return device_shape();
}

int forced_split_k() {
    std::atomic<int>& v = forced_split_slot();
    int cur = v.load(std::memory_order_relaxed);
    if (cur < 0) {
        int expected = -1;
        v.compare_exchange_strong(expected, read_env_split(), std::memory_order_relaxed);
        cur = v.load(std::memory_order_relaxed);
    }
    return cur;
}

}  // namespace spark::launch

// ---- public view (include/spark/kernels.h) -----------------------------------------------

namespace spark {

int device_shape_sm_count() {
    return launch::device_shape().sm_count;
}

int64_t device_shape_l2_bytes() {
    return launch::device_shape().l2_bytes;
}

double device_shape_spec_gbps() {
    return launch::device_shape().spec_gbps;
}

int64_t device_shape_bytes_in_flight() {
    return launch::device_shape().bytes_in_flight;
}

bool device_shape_forced() {
    return launch::device_shape().forced;
}

int decode_split_k(int64_t weight_bytes, int tiles, int max_split, int64_t per_cta_in_flight,
                   int slots, int device) {
    return launch::split_k_for(weight_bytes, tiles, max_split, per_cta_in_flight, slots,
                               launch::device_shape_named(device));
}

int decode_need_ctas(int64_t per_cta_in_flight, int device) {
    return launch::ctas_for_bandwidth(per_cta_in_flight, launch::device_shape_named(device));
}

bool decode_fits_l2(int64_t bytes, int device) {
    return launch::fits_l2(bytes, launch::device_shape_named(device));
}

void decode_force_split_k(int split) {
    launch::forced_split_slot().store(split > 0 ? split : 0, std::memory_order_relaxed);
}

int decode_forced_split_k() {
    return launch::forced_split_k();
}

}  // namespace spark
