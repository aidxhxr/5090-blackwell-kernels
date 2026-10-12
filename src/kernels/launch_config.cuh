// Launch geometry from the device, not from the RTX 5090's constants. Private to src/kernels;
// the public view (what the kernels decided) is the device_shape_* / decode_* functions at
// the end of include/spark/kernels.h, implemented in launch_config.cu.
//
// Every decode kernel here (hgemm_decode, the small-M shapes of w4gemm, w1gemm) is a weight
// stream: M <= 16 rows of activations against a weight that is read from DRAM once. Their
// grids were sized on the RTX 5090, 170 SMs at a measured 1.53 TB/s copy roof, and the
// numbers that came out (64 CTAs for hgemm_decode, 128 blocks for w4gemm, "4 x SMs strips
// is wide") were written down as constants. The GB10 has 48 SMs, 273 GB/s of LPDDR5X and a
// 24 MB L2. This header turns those constants back into the quantity they stand for:
//
//     bytes in flight = bandwidth x loaded latency            (Little's law)
//
// A stream saturates DRAM once the bytes outstanding across the device cover the bandwidth
// times the latency a load sees under load. On the 5090 that was measured, not computed: 64
// CTAs with 24 KB each in flight stream B at the roof and 32 do not (docs/design/hgemm.md,
// "Bytes in flight"), so the device needs about 1.5 MB outstanding, which is 878 ns of loaded
// latency at the 1,792 GB/s spec and 1.03 us at the measured roof. Nobody has published the
// GB10's loaded latency. kGB10LoadedLatencyNs below is an assumption, the design note
// (docs/design/decode_spark.md) tabulates 400, 600 and 800 ns, and the first thing
// bench_decode_spark --sweep-split measures is whether the CTA count it implies is right.
//
// Nothing here has run on a GB10. On a device that reports 170 SMs every rule below gives
// the old constant back exactly; each call site says which constant and how.
#pragma once

#include <algorithm>
#include <cstdint>

#include "spark/common.cuh"

namespace spark::launch {

struct DeviceShape {
    int sm_count;
    int64_t l2_bytes;
    int memory_clock_khz;      // cudaDevAttrMemoryClockRate, 0 when the runtime reports nothing
    int bus_width_bits;        // cudaDevAttrGlobalMemoryBusWidth, 0 likewise
    double spec_gbps;          // DRAM spec: 2 x clock x width / 8 from the attributes, or the
                               // named number when they are missing
    int64_t bytes_in_flight;   // what the whole device must keep outstanding to stream at
                               // spec_gbps: measured on the 5090, bandwidth x assumed latency
                               // elsewhere
    bool forced;               // SPARK_FORCE_SMS was set: sm_count is the override, the rest
                               // is the real device
};

// GB/s x ns is bytes: 1e9 B/s x 1e-9 s.
constexpr int64_t bytes_in_flight_for(double gbps, int latency_ns) {
    return static_cast<int64_t>(gbps * latency_ns);
}

// The GB10's loaded DRAM latency is the one number on this page nobody has measured. 800 ns
// is the conservative end of the table in decode_spark.md (LPDDR5X behind a unified-memory
// fabric; GDDR7 on the 5090 came out at 878 ns at spec from the 64-CTA measurement, and the
// 5090 has 3.6x the SMs to spread the queue over). Measured on the Spark, it replaces this.
constexpr int kGB10LoadedLatencyNs = 800;

// RTX 5090: 170 SMs, 96 MB L2, GDDR7 at 28 Gbps on 512 bits (the runtime reports half the
// data rate as the memory clock, 14,001 MHz, so 2 x 14.001e9 x 64 B = 1,792 GB/s). Bytes in
// flight: the 64 CTAs x 24 KB of hgemm.md, measured.
constexpr DeviceShape kRTX5090 = {170, int64_t{96} << 20, 14001000, 512, 1792.0,
                                  int64_t{64} * 24 * 1024, false};

// GB10: 48 SMs, 24 MB L2, LPDDR5X at 8,533 MT/s on 256 bits (273 GB/s; the clock attribute
// is expected to read 4,266.5 MHz by the same half-rate convention, unverified). Bytes in
// flight from Little's law at the assumed latency: 273 x 800 = 218,400 B.
constexpr DeviceShape kGB10 = {48, int64_t{24} << 20, 4266500, 256, 273.0,
                               bytes_in_flight_for(273.0, kGB10LoadedLatencyNs), false};

// Bytes in flight per SM, the number a kernel's tile and stage count has to supply when
// every SM holds one CTA. Derivation: bandwidth x latency / SMs. GB10 at 800 ns: 218,400 /
// 48 = 4,550 B, under five 16-byte loads per lane of one warp per SM. RTX 5090: 1,572,864 /
// 170 = 9,252 B (hgemm.md's "9 KB per SM if it were spread over all 170"). A 4-stage
// 16x64x64 decode CTA holds 24 KB, so on either device one CTA per SM is more than enough and
// the question is never the pipeline depth, it is how many SMs get a CTA at all.
constexpr int64_t kBytesInFlightPerSM = kGB10.bytes_in_flight / kGB10.sm_count;
constexpr int64_t kBytesInFlightPerSM_RTX5090 = kRTX5090.bytes_in_flight / kRTX5090.sm_count;
static_assert(kBytesInFlightPerSM == 4550, "the derivation in the comment above");
static_assert(kBytesInFlightPerSM_RTX5090 == 9252, "the derivation in the comment above");

// The shape of the current device, cached per device after the first call. SPARK_FORCE_SMS=n
// (read once per process) replaces the SM count so a 5090 can run the 48-SM grid for a
// sanity pass: the kernels then size their grids and split-K for n SMs while the device
// itself is unchanged, so times mean nothing and correctness means everything.
const DeviceShape& device_shape();

// The named shape for the host-side tests and the bench: 1 = GB10, 2 = RTX 5090 (the
// DecodeDevice values of kernels.h); anything else is the current device.
const DeviceShape& device_shape_named(int which);

inline int64_t bytes_in_flight_per_sm(const DeviceShape& d) {
    return cdiv64(d.bytes_in_flight, d.sm_count);
}

// CTAs a weight stream needs when each keeps per_cta bytes in flight: the device's bytes in
// flight divided up, never fewer than one. RTX 5090 at 24 KB per CTA: 64, the hgemm_decode
// constant. GB10 at 24 KB and 800 ns: 9.
inline int ctas_for_bandwidth(int64_t per_cta, const DeviceShape& d) {
    const int64_t per = std::max<int64_t>(1, per_cta);
    return static_cast<int>(std::max<int64_t>(1, cdiv64(d.bytes_in_flight, per)));
}

// Waves of a grid over `slots` resident CTAs, and how full the last one is.
inline double waves(int64_t ctas, int slots) {
    return slots > 0 ? static_cast<double>(ctas) / slots : 0.0;
}
inline double last_wave_fill(int64_t ctas, int slots) {
    if (slots <= 0 || ctas <= 0) return 0.0;
    const int64_t full = cdiv64(ctas, slots) * slots;
    return static_cast<double>(ctas) / static_cast<double>(full);
}

// A weight is L2-resident across two launches if it leaves room for the activations, the
// output and the other operands that pass through: kL2ResidentShare of the L2, the share
// hgemm_streamk.cuh uses for the same question on the GEMM operands.
constexpr double kL2ResidentShare = 0.75;
inline bool fits_l2(int64_t bytes, const DeviceShape& d) {
    return bytes > 0 &&
           static_cast<double>(bytes) <= kL2ResidentShare * static_cast<double>(d.l2_bytes);
}

// The bench's knob: a forced K split for every decode kernel that has one (0 = the rule).
// Set by spark::decode_force_split_k, initial value from SPARK_DECODE_SPLIT. The kernels
// clamp it to their own legal range.
int forced_split_k();

// What one decode launch decided, for the launchers and for the bench's report.
struct DecodePlan {
    int split;         // K slices per tile
    int need_ctas;     // CTAs the bandwidth rule asked for
    int ctas;          // tiles x split
    double waves;      // over the resident slots
    bool l2_resident;  // the weight fits kL2ResidentShare of the L2
    double floor_us;   // weight_bytes at spec_gbps
};

// The split-K rule. `tiles` CTAs each stream their share of a weight of weight_bytes keeping
// per_cta_in_flight bytes outstanding; `slots` CTAs are resident at once (occupancy x SMs);
// the split may go up to max_split (the k-steps per tile).
//
//   1. Bytes in flight first: split = cdiv(ctas_for_bandwidth, tiles), clamped to [1,
//      max_split]. On 170 SMs with the 5090's measured 1.5 MB this is the old
//      cdiv(kMinCtas, strips) with kMinCtas = 64 for a 24 KB CTA and cdiv(kMinBlocks, tiles)
//      with kMinBlocks = 128 for a 12 KB one, so the 5090 grids do not move.
//   2. Waves second, and only when the bandwidth rule itself asks for more CTAs than one
//      wave holds (need_ctas > slots): then the tail wave is paid for anyway and a slightly
//      larger split that fills it (kWaveFill of the last wave, or the best fill up to twice
//      the split) is taken. A grid that is wide because it has many strips, not because the
//      bandwidth rule asked, is left alone: hgemm_decode's tile choice already handles that
//      wave. On 170 SMs the bandwidth rule never exceeds one wave (64 or 128 CTAs against at
//      least 170 slots), so this clause never fires there and the behaviour is the old one
//      exactly. On 48 SMs at 800 ns it asks for 9 to 18 CTAs, so it does not fire there
//      either: the Spark question is whether 9 CTAs on 48 SMs really stream at 273 GB/s
//      (Little's law says yes if the latency is 800 ns) or whether one CTA per SM is needed
//      for the issue rate. --sweep-split answers it.
//   3. A forced split (forced_split_k) replaces the rule, clamped to [1, max_split].
constexpr double kWaveFill = 0.875;

inline DecodePlan plan_decode(int64_t weight_bytes, int tiles, int max_split,
                              int64_t per_cta_in_flight, int slots, const DeviceShape& d) {
    DecodePlan p{};
    tiles = std::max(1, tiles);
    max_split = std::max(1, max_split);
    p.need_ctas = ctas_for_bandwidth(per_cta_in_flight, d);
    int split = std::clamp(cdiv(p.need_ctas, tiles), 1, max_split);
    if (slots > 0 && p.need_ctas > slots) {
        int best = split;
        double best_fill = last_wave_fill(static_cast<int64_t>(tiles) * split, slots);
        const int hi = std::min(max_split, 2 * split);
        for (int s = split; s <= hi; ++s) {
            const double f = last_wave_fill(static_cast<int64_t>(tiles) * s, slots);
            if (f >= kWaveFill) {
                best = s;
                break;
            }
            if (f > best_fill) {
                best = s;
                best_fill = f;
            }
        }
        split = best;
    }
    const int forced = forced_split_k();
    if (forced > 0) split = std::clamp(forced, 1, max_split);
    p.split = split;
    p.ctas = tiles * split;
    p.waves = waves(p.ctas, slots);
    p.l2_resident = fits_l2(weight_bytes, d);
    p.floor_us = d.spec_gbps > 0 ? static_cast<double>(weight_bytes) / (d.spec_gbps * 1e3) : 0.0;
    return p;
}

inline int split_k_for(int64_t weight_bytes, int tiles, int max_split, int64_t per_cta_in_flight,
                       int slots, const DeviceShape& d) {
    return plan_decode(weight_bytes, tiles, max_split, per_cta_in_flight, slots, d).split;
}

}  // namespace spark::launch
