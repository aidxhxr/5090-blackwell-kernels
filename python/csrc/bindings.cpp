// PyTorch bindings for spark-kernels.
//
// Every op validates on the host and dispatches to the launchers declared in
// include/spark/kernels.h on the current PyTorch CUDA stream. Problems caught by the
// TORCH_CHECKs here raise RuntimeError in Python; the launchers' own std::invalid_argument
// (an explicitly requested variant that cannot take the input) becomes ValueError; CUDA
// failures raise RuntimeError.
//
// `variant = -1` means "the fastest variant that accepts this input". That is
// num_variants() - 1 except where the top rung has requirements the rung below does not
// (swiglu: 16-byte aligned storage; hgemm: N and K multiples of 64), in which case the default
// steps down one rung. An explicitly requested variant is never substituted.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "spark/kernels.h"

namespace {

using at::Tensor;

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------
void check_cuda_contig(const Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
    TORCH_CHECK(t.numel() > 0, name, " must be non-empty");
}

void check_float_or_bf16(const Tensor& t, const char* name) {
    TORCH_CHECK(t.scalar_type() == at::kFloat || t.scalar_type() == at::kBFloat16, name,
                " must be float32 or bfloat16, got ", t.scalar_type());
}

int resolve_variant(int variant, int num) {
    if (variant < 0) return num - 1;
    TORCH_CHECK(variant < num, "variant ", variant, " out of range [0, ", num, ")");
    return variant;
}

// The vectorized kernels read 16 bytes per lane; a tensor whose storage offset is not a
// multiple of 16 bytes (e.g. a slice of a flat buffer) cannot feed them.
bool aligned16(const Tensor& t) {
    return reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

// Row ops treat the last dim as the row; all leading dims are flattened.
struct RowShape {
    int rows;
    int cols;
};
RowShape row_shape(const Tensor& t) {
    TORCH_CHECK(t.dim() >= 1, "expected at least 1 dimension");
    const int64_t cols = t.size(-1);
    const int64_t rows = t.numel() / cols;
    TORCH_CHECK(rows <= INT32_MAX && cols <= INT32_MAX, "tensor too large for int32 indexing");
    return {static_cast<int>(rows), static_cast<int>(cols)};
}

const __nv_bfloat16* bf16_ptr(const Tensor& t) {
    return reinterpret_cast<const __nv_bfloat16*>(t.data_ptr<at::BFloat16>());
}
__nv_bfloat16* bf16_ptr_mut(Tensor& t) {
    return reinterpret_cast<__nv_bfloat16*>(t.data_ptr<at::BFloat16>());
}

cudaStream_t current_stream(const Tensor& t) {
    return at::cuda::getCurrentCUDAStream(t.device().index()).stream();
}

// ---------------------------------------------------------------------------
// Ops
// ---------------------------------------------------------------------------
Tensor rmsnorm(const Tensor& x, const Tensor& w, double eps, int variant) {
    check_cuda_contig(x, "x");
    check_cuda_contig(w, "w");
    check_float_or_bf16(x, "x");
    TORCH_CHECK(w.scalar_type() == x.scalar_type(), "w must have the same dtype as x");
    const RowShape s = row_shape(x);
    TORCH_CHECK(w.dim() == 1 && w.size(0) == s.cols, "w must have shape [cols] = [", s.cols, "]");
    const c10::cuda::CUDAGuard guard(x.device());
    Tensor out = at::empty_like(x);
    const int v = resolve_variant(variant, spark::rmsnorm_num_variants());
    cudaStream_t stream = current_stream(x);
    if (x.scalar_type() == at::kFloat) {
        spark::rmsnorm_f32(x.data_ptr<float>(), w.data_ptr<float>(), out.data_ptr<float>(), s.rows,
                           s.cols, static_cast<float>(eps), v, stream);
    } else {
        spark::rmsnorm_bf16(bf16_ptr(x), bf16_ptr(w), bf16_ptr_mut(out), s.rows, s.cols,
                            static_cast<float>(eps), v, stream);
    }
    return out;
}

// resid += x; out = rmsnorm(resid) * w. resid is modified in place. bf16 only.
Tensor add_rmsnorm_(const Tensor& x, Tensor resid, const Tensor& w, double eps) {
    check_cuda_contig(x, "x");
    check_cuda_contig(resid, "resid");
    check_cuda_contig(w, "w");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "add_rmsnorm_ supports bfloat16 only");
    TORCH_CHECK(resid.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16,
                "resid and w must be bfloat16");
    TORCH_CHECK(resid.sizes() == x.sizes(), "resid must have the same shape as x");
    const RowShape s = row_shape(x);
    TORCH_CHECK(w.dim() == 1 && w.size(0) == s.cols, "w must have shape [cols] = [", s.cols, "]");
    const c10::cuda::CUDAGuard guard(x.device());
    Tensor out = at::empty_like(x);
    spark::add_rmsnorm_bf16(bf16_ptr(x), bf16_ptr_mut(resid), bf16_ptr(w), bf16_ptr_mut(out),
                            s.rows, s.cols, static_cast<float>(eps), current_stream(x));
    return out;
}

Tensor swiglu(const Tensor& gate, const Tensor& up, int variant) {
    check_cuda_contig(gate, "gate");
    check_cuda_contig(up, "up");
    check_float_or_bf16(gate, "gate");
    TORCH_CHECK(up.scalar_type() == gate.scalar_type(), "up must have the same dtype as gate");
    TORCH_CHECK(up.sizes() == gate.sizes(), "gate and up must have the same shape");
    const c10::cuda::CUDAGuard guard(gate.device());
    Tensor out = at::empty_like(gate);
    int v = resolve_variant(variant, spark::swiglu_num_variants());
    if (variant < 0 && !(aligned16(gate) && aligned16(up) && aligned16(out))) v = 0;
    cudaStream_t stream = current_stream(gate);
    const int64_t n = gate.numel();
    if (gate.scalar_type() == at::kFloat) {
        spark::swiglu_f32(gate.data_ptr<float>(), up.data_ptr<float>(), out.data_ptr<float>(), n, v,
                          stream);
    } else {
        spark::swiglu_bf16(bf16_ptr(gate), bf16_ptr(up), bf16_ptr_mut(out), n, v, stream);
    }
    return out;
}

Tensor softmax(const Tensor& x, int variant) {
    check_cuda_contig(x, "x");
    check_float_or_bf16(x, "x");
    const RowShape s = row_shape(x);
    const c10::cuda::CUDAGuard guard(x.device());
    Tensor out = at::empty_like(x);
    const int v = resolve_variant(variant, spark::softmax_num_variants());
    cudaStream_t stream = current_stream(x);
    if (x.scalar_type() == at::kFloat) {
        spark::softmax_f32(x.data_ptr<float>(), out.data_ptr<float>(), s.rows, s.cols, v, stream);
    } else {
        spark::softmax_bf16(bf16_ptr(x), bf16_ptr_mut(out), s.rows, s.cols, v, stream);
    }
    return out;
}

struct GemmShape {
    int M, N, K;
};
GemmShape gemm_shape(const Tensor& a, const Tensor& b) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2, "gemm expects 2-D tensors");
    TORCH_CHECK(a.size(1) == b.size(0), "inner dimensions mismatch: a is ", a.size(0), "x",
                a.size(1), ", b is ", b.size(0), "x", b.size(1));
    TORCH_CHECK(a.size(0) <= INT32_MAX && b.size(1) <= INT32_MAX && a.size(1) <= INT32_MAX,
                "gemm dims too large for int32");
    return {static_cast<int>(a.size(0)), static_cast<int>(b.size(1)), static_cast<int>(a.size(1))};
}

Tensor sgemm(const Tensor& a, const Tensor& b, int variant) {
    check_cuda_contig(a, "a");
    check_cuda_contig(b, "b");
    TORCH_CHECK(a.scalar_type() == at::kFloat && b.scalar_type() == at::kFloat,
                "sgemm expects float32 inputs");
    const GemmShape s = gemm_shape(a, b);
    const c10::cuda::CUDAGuard guard(a.device());
    Tensor c = at::empty({s.M, s.N}, a.options());
    const int v = resolve_variant(variant, spark::sgemm_num_variants());
    spark::sgemm(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), s.M, s.N, s.K, v,
                 current_stream(a));
    return c;
}

Tensor hgemm(const Tensor& a, const Tensor& b, int variant) {
    check_cuda_contig(a, "a");
    check_cuda_contig(b, "b");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16,
                "hgemm expects bfloat16 inputs");
    const GemmShape s = gemm_shape(a, b);
    const c10::cuda::CUDAGuard guard(a.device());
    Tensor c = at::empty({s.M, s.N}, a.options());
    int v = resolve_variant(variant, spark::hgemm_num_variants());
    while (variant < 0 && v > 0 && !spark::hgemm_supports(s.M, s.N, s.K, v)) --v;
    spark::hgemm_bf16(bf16_ptr(a), bf16_ptr(b), bf16_ptr_mut(c), s.M, s.N, s.K, v,
                      current_stream(a));
    return c;
}

// C = scale_a * scale_b * a @ b_t^T with a [M, K] and b_t [N, K] both float8_e4m3fn and K
// contiguous (the layout torch._scaled_mm and cuBLASLt take), per-tensor fp32 scales on the
// device, bf16 out. The default variant is the highest that accepts the shape.
Tensor fp8gemm(const Tensor& a, const Tensor& b_t, const Tensor& scale_a, const Tensor& scale_b,
               int variant) {
    check_cuda_contig(a, "a");
    check_cuda_contig(b_t, "b_t");
    TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fn && b_t.scalar_type() == at::kFloat8_e4m3fn,
                "fp8gemm expects float8_e4m3fn inputs");
    TORCH_CHECK(a.dim() == 2 && b_t.dim() == 2, "fp8gemm expects 2-D tensors");
    TORCH_CHECK(a.size(1) == b_t.size(1), "inner dimensions mismatch: a is ", a.size(0), "x",
                a.size(1), ", b_t is ", b_t.size(0), "x", b_t.size(1), " (b_t is [N, K])");
    TORCH_CHECK(a.size(0) <= INT32_MAX && b_t.size(0) <= INT32_MAX && a.size(1) <= INT32_MAX,
                "fp8gemm dims too large for int32");
    for (const auto* sp : {&scale_a, &scale_b}) {
        const Tensor& sc = *sp;
        TORCH_CHECK(sc.is_cuda() && sc.scalar_type() == at::kFloat && sc.numel() == 1,
                    "scale_a and scale_b must be float32 CUDA tensors with one element");
        TORCH_CHECK(sc.device() == a.device(), "scales must be on a's device");
    }
    TORCH_CHECK(aligned16(a) && aligned16(b_t), "fp8gemm needs 16-byte aligned a and b_t storage");
    const int M = static_cast<int>(a.size(0)), N = static_cast<int>(b_t.size(0)),
              K = static_cast<int>(a.size(1));
    const c10::cuda::CUDAGuard guard(a.device());
    Tensor c = at::empty({M, N}, a.options().dtype(at::kBFloat16));
    int v = resolve_variant(variant, spark::fp8gemm_num_variants());
    while (variant < 0 && v > 0 && !spark::fp8gemm_supports(M, N, K, v)) --v;
    spark::fp8gemm(reinterpret_cast<const __nv_fp8_e4m3*>(a.data_ptr()),
                   reinterpret_cast<const __nv_fp8_e4m3*>(b_t.data_ptr()), bf16_ptr_mut(c), M, N, K,
                   scale_a.data_ptr<float>(), scale_b.data_ptr<float>(), v, current_stream(a));
    return c;
}

// O = softmax(q k^T / sqrt(D)) v for q = [B, H_q, S_q, D] and k, v = [B, H_kv, S_kv, D] bf16
// tensors, H_q a multiple of H_kv (grouped-query attention: query head h reads k/v head
// h / (H_q / H_kv)); every variant takes any S_q, S_kv >= 1 and D in {64, 128}. The default
// steps down from the top rung if a future rung ever refuses a shape, as hgemm's does.
Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal, int variant) {
    check_cuda_contig(q, "q");
    check_cuda_contig(k, "k");
    check_cuda_contig(v, "v");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && k.scalar_type() == at::kBFloat16 &&
                    v.scalar_type() == at::kBFloat16,
                "attention expects bfloat16 q, k, v");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "attention expects [B, H, S, D] tensors");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3),
                "q and k must agree on B and D (q is ", q.sizes(), ", k is ", k.sizes(), ")");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "q must have a multiple of k's heads (GQA), got ",
                q.size(1), " query heads and ", k.size(1), " k/v heads");
    TORCH_CHECK(q.size(3) == 64 || q.size(3) == 128, "attention supports D = 64 or 128, got ",
                q.size(3));
    TORCH_CHECK(
        q.size(2) <= INT32_MAX && k.size(2) <= INT32_MAX && q.size(0) * q.size(1) <= INT32_MAX,
        "attention dims too large for int32");
    TORCH_CHECK(aligned16(q) && aligned16(k) && aligned16(v),
                "attention needs 16-byte aligned q, k, v storage");
    const c10::cuda::CUDAGuard guard(q.device());
    Tensor out = at::empty_like(q);
    int var = resolve_variant(variant, spark::attention_num_variants());
    while (variant < 0 && var > 0 &&
           !spark::attention_supports(static_cast<int>(q.size(2)), static_cast<int>(k.size(2)),
                                      static_cast<int>(q.size(3)), var))
        --var;
    spark::attention_bf16(
        bf16_ptr(q), bf16_ptr(k), bf16_ptr(v), bf16_ptr_mut(out), static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)), static_cast<int>(k.size(1)), static_cast<int>(q.size(2)),
        static_cast<int>(k.size(2)), static_cast<int>(q.size(3)), causal, var, current_stream(q));
    return out;
}

int num_variants(const std::string& name) {
    if (name == "rmsnorm") return spark::rmsnorm_num_variants();
    if (name == "swiglu") return spark::swiglu_num_variants();
    if (name == "softmax") return spark::softmax_num_variants();
    if (name == "sgemm") return spark::sgemm_num_variants();
    if (name == "hgemm") return spark::hgemm_num_variants();
    if (name == "fp8gemm") return spark::fp8gemm_num_variants();
    if (name == "attention") return spark::attention_num_variants();
    if (name == "bandwidth") return spark::bandwidth_num_variants();
    throw std::invalid_argument("unknown kernel: " + name);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "spark-kernels: hand-written CUDA kernels for LLM inference on sm_120 / sm_121";
    m.def("rmsnorm", &rmsnorm, "RMSNorm over the last dim", py::arg("x"), py::arg("w"),
          py::arg("eps") = 1e-6, py::arg("variant") = -1);
    m.def("add_rmsnorm_", &add_rmsnorm_, "resid += x; return rmsnorm(resid) * w (bf16, in place)",
          py::arg("x"), py::arg("resid"), py::arg("w"), py::arg("eps") = 1e-6);
    m.def("swiglu", &swiglu, "silu(gate) * up", py::arg("gate"), py::arg("up"),
          py::arg("variant") = -1);
    m.def("softmax", &softmax, "softmax over the last dim (fp32 math)", py::arg("x"),
          py::arg("variant") = -1);
    m.def("sgemm", &sgemm, "fp32 GEMM: a @ b", py::arg("a"), py::arg("b"), py::arg("variant") = -1);
    m.def("hgemm", &hgemm, "bf16 tensor-core GEMM: a @ b", py::arg("a"), py::arg("b"),
          py::arg("variant") = -1);
    m.def("fp8gemm", &fp8gemm,
          "fp8 (e4m3) tensor-core GEMM: scale_a * scale_b * a @ b_t^T, b_t is [N, K], bf16 out",
          py::arg("a"), py::arg("b_t"), py::arg("scale_a"), py::arg("scale_b"),
          py::arg("variant") = -1);
    m.def("attention", &attention,
          "softmax(q k^T / sqrt(D)) v over [B, H, S, D] bf16 tensors (k, v may have fewer heads)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false,
          py::arg("variant") = -1);
    m.def("num_variants", &num_variants, "number of implementation variants for a kernel",
          py::arg("name"));
}
