// PyTorch bindings for the CPU-side ops of include/spark/cpu.h (the Arm cores of the GB10).
// Registered into the one spark_kernels._C module by bindings.cpp through
// register_cpu_bindings. Every tensor must live on the CPU: on the GB10 a CUDA tensor's
// storage is the same LPDDR5X, but these functions dereference the pointer on a core, so
// the API asks for .is_cpu() and leaves the "which side touches it" decision to the caller
// (a .cpu() on the Spark is a copy today; a true zero-copy view is a later experiment).
#include <torch/extension.h>

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "spark/cpu.h"

namespace {

using at::Tensor;

void check_cpu_contig(const Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cpu(), name, " must be a CPU tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.numel() > 0, name, " must be non-empty");
}

const spark::cpu::bf16* bf16_ptr(const Tensor& t) {
    return reinterpret_cast<const spark::cpu::bf16*>(t.data_ptr<at::BFloat16>());
}
spark::cpu::bf16* bf16_ptr_mut(Tensor& t) {
    return reinterpret_cast<spark::cpu::bf16*>(t.data_ptr<at::BFloat16>());
}

// a [K] or [M, K] bf16 @ the sign / ternary weight (packed [N, K bits / 32] int32, scales
// [K / 128, N] bf16) -> [N] or [M, N] bf16, one GEMV per row of a.
Tensor cpu_w1gemv(const Tensor& a, const Tensor& packed, const Tensor& scales, int64_t bits,
                  int64_t threads, int64_t variant) {
    check_cpu_contig(a, "a");
    check_cpu_contig(packed, "packed");
    check_cpu_contig(scales, "scales");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16, "a must be bfloat16");
    TORCH_CHECK(packed.scalar_type() == at::kInt, "packed must be int32");
    TORCH_CHECK(scales.scalar_type() == at::kBFloat16, "scales must be bfloat16");
    TORCH_CHECK(bits == 1 || bits == 2, "bits must be 1 or 2");
    TORCH_CHECK(a.dim() == 1 || a.dim() == 2, "a must be [K] or [M, K]");
    TORCH_CHECK(packed.dim() == 2 && scales.dim() == 2, "packed and scales must be 2-D");
    const int64_t K = a.size(-1);
    const int64_t N = packed.size(0);
    TORCH_CHECK(K % spark::cpu::kW1GroupSize == 0, "K must be a multiple of 128, got ", K);
    TORCH_CHECK(packed.size(1) == K * bits / 32, "packed must be [N, K * bits / 32]");
    TORCH_CHECK(scales.size(0) == K / spark::cpu::kW1GroupSize && scales.size(1) == N,
                "scales must be [K / 128, N]");
    const int64_t M = a.dim() == 1 ? 1 : a.size(0);
    Tensor y = at::empty(a.dim() == 1 ? std::vector<int64_t>{N} : std::vector<int64_t>{M, N},
                         a.options());
    try {
        for (int64_t m = 0; m < M; ++m) {
            spark::cpu::w1gemv_bf16(bf16_ptr(a) + m * K,
                                    reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>()),
                                    bf16_ptr(scales), bf16_ptr_mut(y) + m * N, static_cast<int>(N),
                                    static_cast<int>(K), static_cast<int>(bits),
                                    static_cast<int>(threads), static_cast<int>(variant));
        }
    } catch (const std::invalid_argument& e) {
        throw py::value_error(e.what());
    }
    return y;
}

// logits [E] or [T, E] fp32, optional bias [E] -> (ids [T, k] int32, weights [T, k] fp32),
// spark_kernels.moe.route's contract for one token at a time.
std::tuple<Tensor, Tensor> cpu_route(const Tensor& logits, int64_t top_k,
                                     const std::optional<Tensor>& bias, double scale) {
    check_cpu_contig(logits, "logits");
    TORCH_CHECK(logits.scalar_type() == at::kFloat, "logits must be float32");
    TORCH_CHECK(logits.dim() == 1 || logits.dim() == 2, "logits must be [E] or [T, E]");
    const int64_t E = logits.size(-1);
    const int64_t T = logits.dim() == 1 ? 1 : logits.size(0);
    TORCH_CHECK(top_k >= 1 && top_k <= E, "top_k must be in [1, E]");
    const float* bias_ptr = nullptr;
    if (bias.has_value()) {
        check_cpu_contig(*bias, "bias");
        TORCH_CHECK(bias->scalar_type() == at::kFloat && bias->numel() == E, "bias must be [E] float32");
        bias_ptr = bias->data_ptr<float>();
    }
    const auto shape = logits.dim() == 1 ? std::vector<int64_t>{top_k} : std::vector<int64_t>{T, top_k};
    Tensor ids = at::empty(shape, logits.options().dtype(at::kInt));
    Tensor weights = at::empty(shape, logits.options());
    for (int64_t t = 0; t < T; ++t) {
        spark::cpu::route_sigmoid_topk(logits.data_ptr<float>() + t * E, bias_ptr, static_cast<int>(E),
                                       static_cast<int>(top_k), static_cast<float>(scale),
                                       ids.data_ptr<int32_t>() + t * top_k,
                                       weights.data_ptr<float>() + t * top_k);
    }
    return {ids, weights};
}

// logits [V] or [B, V] bf16 -> int64 [] or [B]; one (seed, offset + b) draw per row.
Tensor cpu_sample(const Tensor& logits, double temperature, int64_t top_k, double top_p,
                  int64_t seed, int64_t offset) {
    check_cpu_contig(logits, "logits");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16, "logits must be bfloat16");
    TORCH_CHECK(logits.dim() == 1 || logits.dim() == 2, "logits must be [V] or [B, V]");
    const int64_t V = logits.size(-1);
    const int64_t B = logits.dim() == 1 ? 1 : logits.size(0);
    Tensor out = at::empty(logits.dim() == 1 ? std::vector<int64_t>{} : std::vector<int64_t>{B},
                           logits.options().dtype(at::kLong));
    for (int64_t b = 0; b < B; ++b) {
        out.data_ptr<int64_t>()[b] = spark::cpu::sample_bf16(
            bf16_ptr(logits) + b * V, static_cast<int>(V), static_cast<float>(temperature),
            static_cast<int>(top_k), static_cast<float>(top_p), static_cast<uint64_t>(seed),
            static_cast<uint64_t>(offset + b));
    }
    return out;
}

double cpu_bandwidth_probe(int64_t bytes, int64_t threads, int64_t iters) {
    TORCH_CHECK(bytes > 0 && threads > 0 && iters > 0, "bytes, threads and iters must be positive");
    return spark::cpu::bandwidth_probe(static_cast<size_t>(bytes), static_cast<int>(threads),
                                       static_cast<int>(iters));
}

int64_t cpu_w1gemv_num_variants() { return spark::cpu::w1gemv_num_variants(); }
bool cpu_w1gemv_supports(int64_t variant) { return spark::cpu::w1gemv_supports(static_cast<int>(variant)); }

}  // namespace

void register_cpu_bindings(py::module_& m) {
    m.def("cpu_w1gemv", &cpu_w1gemv,
          "on the CPU: a [K] or [M, K] bf16 @ the sign or ternary weight (packed, scales) -> [N] "
          "or [M, N] bf16, fp32 accumulate, rows split over `threads`",
          py::arg("a"), py::arg("packed"), py::arg("scales"), py::arg("bits") = 1,
          py::arg("threads") = 1, py::arg("variant") = -1);
    m.def("cpu_w1gemv_num_variants", &cpu_w1gemv_num_variants,
          "number of CPU GEMV rungs (0 scalar, 1 NEON fp32, 2 NEON int8 dot)");
    m.def("cpu_w1gemv_supports", &cpu_w1gemv_supports,
          "whether this build of the extension has the given CPU GEMV rung", py::arg("variant"));
    m.def("cpu_route", &cpu_route,
          "on the CPU: sigmoid gating over fp32 logits [E] or [T, E] -> (expert ids int32, gate "
          "weights fp32), the top_k of sigmoid + bias, weights the unbiased sigmoid renormalized "
          "to sum `scale`",
          py::arg("logits"), py::arg("top_k"), py::arg("bias") = py::none(), py::arg("scale") = 1.0);
    m.def("cpu_sample", &cpu_sample,
          "on the CPU: one token per bf16 logit row; temperature <= 0 is argmax, then top-k, "
          "top-p and a splitmix64 draw from (seed, offset + row)",
          py::arg("logits"), py::arg("temperature"), py::arg("top_k") = 0, py::arg("top_p") = 1.0,
          py::arg("seed") = 0, py::arg("offset") = 0);
    m.def("cpu_bandwidth_probe", &cpu_bandwidth_probe,
          "stream a private buffer of `bytes` with 16-byte NEON loads on `threads` threads; the "
          "median GB/s over `iters` sweeps",
          py::arg("bytes"), py::arg("threads"), py::arg("iters") = 5);
}
