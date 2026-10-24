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
// steps down one rung, and except for sgemm, whose variants 6 and 7 trade fp32 accuracy for
// the tensor cores (TF32) and are opt-in. An explicitly requested variant is never substituted.
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

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

// A 2-D tensor whose rows are contiguous but not adjacent: one half of a fused gate|up
// projection, gu[:, :I] or gu[:, I:].
bool strided_rows(const Tensor& t) {
    return t.is_cuda() && t.dim() == 2 && t.stride(1) == 1 && t.stride(0) >= t.size(1) &&
           !t.is_contiguous();
}

// gate and up as the two halves of one [rows, 2 * cols] GEMM output (or any 2-D tensors with
// contiguous rows), read in place through the strided kernel; the output is contiguous.
Tensor swiglu_strided(const Tensor& gate, const Tensor& up) {
    TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have the same shape");
    TORCH_CHECK(gate.stride(1) == 1 && up.stride(1) == 1, "swiglu needs contiguous rows");
    TORCH_CHECK(gate.numel() > 0, "gate must be non-empty");
    const c10::cuda::CUDAGuard guard(gate.device());
    Tensor out = at::empty(gate.sizes(), gate.options());
    cudaStream_t stream = current_stream(gate);
    const int64_t rows = gate.size(0), cols = gate.size(1);
    if (gate.scalar_type() == at::kFloat) {
        spark::swiglu_strided_f32(gate.data_ptr<float>(), up.data_ptr<float>(),
                                  out.data_ptr<float>(), rows, cols, gate.stride(0), up.stride(0),
                                  stream);
    } else {
        spark::swiglu_strided_bf16(bf16_ptr(gate), bf16_ptr(up), bf16_ptr_mut(out), rows, cols,
                                   gate.stride(0), up.stride(0), stream);
    }
    return out;
}

Tensor swiglu(const Tensor& gate, const Tensor& up, int variant) {
    check_float_or_bf16(gate, "gate");
    TORCH_CHECK(up.scalar_type() == gate.scalar_type(), "up must have the same dtype as gate");
    if (variant < 0 && (strided_rows(gate) || strided_rows(up)) && gate.is_cuda() && up.is_cuda() &&
        gate.dim() == 2 && up.dim() == 2 && gate.stride(1) == 1 && up.stride(1) == 1) {
        return swiglu_strided(gate, up);
    }
    check_cuda_contig(gate, "gate");
    check_cuda_contig(up, "up");
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

// The per-row parameters of sample: [B] on the device of logits, the given dtype.
void check_row_param(const Tensor& t, const char* name, at::ScalarType dtype, int64_t B,
                     const Tensor& logits) {
    check_cuda_contig(t, name);
    TORCH_CHECK(t.scalar_type() == dtype, name, " must be ", dtype, ", got ", t.scalar_type());
    TORCH_CHECK(t.dim() == 1 && t.size(0) == B, name, " must be [B] = [", B, "]");
    TORCH_CHECK(t.device() == logits.device(), name, " must be on the device of logits");
}

// One token per row of logits [B, V] bf16 (spark::sample_bf16 states the rule): per-row
// temperature, top_p (float32), top_k (int32), seed and offset (int64), all [B] on the device;
// each row's offset is advanced by one in place.
Tensor sample(const Tensor& logits, const Tensor& temperature, const Tensor& top_k,
              const Tensor& top_p, const Tensor& seed, Tensor offset, int variant) {
    check_cuda_contig(logits, "logits");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16 && logits.dim() == 2,
                "sample expects logits as [B, V] bfloat16");
    const int64_t B = logits.size(0), V = logits.size(1);
    TORCH_CHECK(B <= INT32_MAX && V <= (int64_t{1} << 23), "sample: B or V too large");
    check_row_param(temperature, "temperature", at::kFloat, B, logits);
    check_row_param(top_k, "top_k", at::kInt, B, logits);
    check_row_param(top_p, "top_p", at::kFloat, B, logits);
    check_row_param(seed, "seed", at::kLong, B, logits);
    check_row_param(offset, "offset", at::kLong, B, logits);
    const c10::cuda::CUDAGuard guard(logits.device());
    Tensor out = at::empty({B}, logits.options().dtype(at::kLong));
    const int v = resolve_variant(variant, spark::sample_num_variants());
    spark::sample_bf16(
        bf16_ptr(logits), static_cast<int>(B), static_cast<int>(V), temperature.data_ptr<float>(),
        top_k.data_ptr<int>(), top_p.data_ptr<float>(), seed.data_ptr<int64_t>(),
        offset.data_ptr<int64_t>(), out.data_ptr<int64_t>(), v, current_stream(logits));
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
    // -1 is variant 5, the top fp32 rung. Variants 6 and 7 run on the tensor cores in TF32
    // (one and three passes) and are a different precision contract, so they are opt-in,
    // as torch's allow_tf32 is.
    const int v = variant < 0 ? spark::sgemm_default_variant()
                              : resolve_variant(variant, spark::sgemm_num_variants());
    spark::sgemm(a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), s.M, s.N, s.K, v,
                 current_stream(a));
    return c;
}

// The activation of the fused epilogue, by name; None or "none" is the identity.
int hgemm_act(const std::optional<std::string>& act) {
    if (!act || *act == "none") return spark::HGEMM_ACT_NONE;
    if (*act == "silu") return spark::HGEMM_ACT_SILU;
    if (*act == "gelu") return spark::HGEMM_ACT_GELU;
    if (*act == "relu") return spark::HGEMM_ACT_RELU;
    throw std::invalid_argument("hgemm: act must be None, \"silu\", \"gelu\" or \"relu\", got \"" +
                                *act + "\"");
}

// c = act(a @ b + bias) + residual, one kernel, the epilogue applied in fp32 before the one
// rounding to bf16 (variants 4 and 6; the launcher throws std::invalid_argument, ValueError
// here, for the others). `out` is written in place when given; out=residual is the in-place
// accumulate c += act(a @ b + bias). With swiglu, b is [K, 2N] with gate_j and up_j in columns
// 2j and 2j+1 (spark_kernels.interleave_gate_up) and the result is [M, N]:
// silu(gate) * up (+ residual).
Tensor hgemm(const Tensor& a, const Tensor& b, int variant, const std::optional<Tensor>& bias,
             const std::optional<std::string>& act, const std::optional<Tensor>& residual,
             bool swiglu, const std::optional<Tensor>& out) {
    check_cuda_contig(a, "a");
    check_cuda_contig(b, "b");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && b.scalar_type() == at::kBFloat16,
                "hgemm expects bfloat16 inputs");
    const GemmShape s = gemm_shape(a, b);
    TORCH_CHECK(!swiglu || s.N % 2 == 0,
                "hgemm_swiglu: b must have an even number of columns "
                "(interleaved gate/up pairs), got ",
                s.N);
    const int n_out = swiglu ? s.N / 2 : s.N;
    spark::HgemmEpilogue ep;
    ep.act = hgemm_act(act);
    ep.swiglu = swiglu;
    if (bias) {
        check_cuda_contig(*bias, "bias");
        TORCH_CHECK(bias->scalar_type() == at::kBFloat16, "bias must be bfloat16");
        TORCH_CHECK(bias->dim() == 1 && bias->size(0) == s.N, "bias must have shape [", s.N,
                    "] (one entry per column of b), got ", bias->sizes());
        TORCH_CHECK(bias->device() == a.device(), "bias must be on a's device");
        ep.bias = bf16_ptr(*bias);
    }
    if (residual) {
        check_cuda_contig(*residual, "residual");
        TORCH_CHECK(residual->scalar_type() == at::kBFloat16, "residual must be bfloat16");
        TORCH_CHECK(residual->dim() == 2 && residual->size(0) == s.M && residual->size(1) == n_out,
                    "residual must have shape [", s.M, ", ", n_out, "], got ", residual->sizes());
        TORCH_CHECK(residual->device() == a.device(), "residual must be on a's device");
        ep.residual = bf16_ptr(*residual);
    }
    const c10::cuda::CUDAGuard guard(a.device());
    Tensor c;
    if (out) {
        c = *out;
        check_cuda_contig(c, "out");
        TORCH_CHECK(c.scalar_type() == at::kBFloat16, "out must be bfloat16");
        TORCH_CHECK(c.dim() == 2 && c.size(0) == s.M && c.size(1) == n_out, "out must have shape [",
                    s.M, ", ", n_out, "], got ", c.sizes());
        TORCH_CHECK(c.device() == a.device(), "out must be on a's device");
    } else {
        c = at::empty({s.M, n_out}, a.options());
    }
    int v = variant < 0 ? spark::hgemm_default_variant()
                        : resolve_variant(variant, spark::hgemm_num_variants());
    while (variant < 0 && v > 0 && !spark::hgemm_supports(s.M, s.N, s.K, v)) --v;
    spark::hgemm_bf16(bf16_ptr(a), bf16_ptr(b), bf16_ptr_mut(c), s.M, s.N, s.K, v,
                      current_stream(a), ep);
    return c;
}

// C = scale_a * scale_b * a @ b_t^T with a [M, K] and b_t [N, K] both float8_e4m3fn and K
// contiguous (the layout torch._scaled_mm and cuBLASLt take), per-tensor fp32 scales on the
// device, bf16 out. With sfa and sfb (MX mode: uint8 ue8m0 scales, [M, K/32] and [N, K/32],
// one per 32 consecutive k of a row) every product is also scaled by its two block scales,
// inside the block-scaled mma; the per-tensor scales may then be None (1.0). The default
// variant is the highest that accepts the shape.
Tensor fp8gemm(const Tensor& a, const Tensor& b_t, const std::optional<Tensor>& scale_a,
               const std::optional<Tensor>& scale_b, int variant, const std::optional<Tensor>& sfa,
               const std::optional<Tensor>& sfb) {
    check_cuda_contig(a, "a");
    check_cuda_contig(b_t, "b_t");
    TORCH_CHECK(a.scalar_type() == at::kFloat8_e4m3fn && b_t.scalar_type() == at::kFloat8_e4m3fn,
                "fp8gemm expects float8_e4m3fn inputs");
    TORCH_CHECK(a.dim() == 2 && b_t.dim() == 2, "fp8gemm expects 2-D tensors");
    TORCH_CHECK(a.size(1) == b_t.size(1), "inner dimensions mismatch: a is ", a.size(0), "x",
                a.size(1), ", b_t is ", b_t.size(0), "x", b_t.size(1), " (b_t is [N, K])");
    TORCH_CHECK(a.size(0) <= INT32_MAX && b_t.size(0) <= INT32_MAX && a.size(1) <= INT32_MAX,
                "fp8gemm dims too large for int32");
    const bool mx = sfa.has_value() || sfb.has_value();
    TORCH_CHECK(!mx || (sfa.has_value() && sfb.has_value()),
                "fp8gemm: sfa and sfb must both be given (MX mode) or both be None");
    TORCH_CHECK(mx || (scale_a.has_value() && scale_b.has_value()),
                "fp8gemm: scale_a and scale_b are required without block scales");
    TORCH_CHECK(scale_a.has_value() == scale_b.has_value(),
                "fp8gemm: scale_a and scale_b must both be given or both be None");
    const float* sa = nullptr;
    const float* sb = nullptr;
    if (scale_a.has_value()) {
        for (const Tensor* sp : {&*scale_a, &*scale_b}) {
            const Tensor& sc = *sp;
            TORCH_CHECK(sc.is_cuda() && sc.scalar_type() == at::kFloat && sc.numel() == 1,
                        "scale_a and scale_b must be float32 CUDA tensors with one element");
            TORCH_CHECK(sc.device() == a.device(), "scales must be on a's device");
        }
        sa = scale_a->data_ptr<float>();
        sb = scale_b->data_ptr<float>();
    }
    TORCH_CHECK(aligned16(a) && aligned16(b_t), "fp8gemm needs 16-byte aligned a and b_t storage");
    const int M = static_cast<int>(a.size(0)), N = static_cast<int>(b_t.size(0)),
              K = static_cast<int>(a.size(1));
    if (mx) {
        TORCH_CHECK(K % 256 == 0, "fp8gemm with block scales needs K a multiple of 256, got ", K);
        const std::pair<const Tensor*, const char*> sfs[2] = {{&*sfa, "sfa"}, {&*sfb, "sfb"}};
        const int64_t rows[2] = {M, N};
        for (int i = 0; i < 2; ++i) {
            const Tensor& sf = *sfs[i].first;
            check_cuda_contig(sf, sfs[i].second);
            TORCH_CHECK(sf.scalar_type() == at::kByte, sfs[i].second,
                        " must be uint8 (ue8m0: exponent + 127)");
            TORCH_CHECK(sf.dim() == 2 && sf.size(0) == rows[i] && sf.size(1) == K / 32,
                        sfs[i].second, " must be [", rows[i], ", ", K / 32, "], got ", sf.sizes());
            TORCH_CHECK(sf.device() == a.device(), sfs[i].second, " must be on a's device");
            TORCH_CHECK(aligned16(sf), sfs[i].second, " needs 16-byte aligned storage");
        }
    }
    const c10::cuda::CUDAGuard guard(a.device());
    Tensor c = at::empty({M, N}, a.options().dtype(at::kBFloat16));
    int v = resolve_variant(variant, spark::fp8gemm_num_variants());
    const auto supports = [&](int vv) {
        return mx ? spark::fp8gemm_mx_supports(M, N, K, vv) : spark::fp8gemm_supports(M, N, K, vv);
    };
    while (variant < 0 && v > 0 && !supports(v)) --v;
    const auto* pa = reinterpret_cast<const __nv_fp8_e4m3*>(a.data_ptr());
    const auto* pb = reinterpret_cast<const __nv_fp8_e4m3*>(b_t.data_ptr());
    if (mx) {
        spark::fp8gemm_mx(pa, pb, bf16_ptr_mut(c), M, N, K, sa, sb,
                          static_cast<const unsigned char*>(sfa->data_ptr()),
                          static_cast<const unsigned char*>(sfb->data_ptr()), v, current_stream(a));
    } else {
        spark::fp8gemm(pa, pb, bf16_ptr_mut(c), M, N, K, sa, sb, v, current_stream(a));
    }
    return c;
}

// fp8gemm variant 3's tile route (spark::fp8gemm_v3_set_route): -1 the per-shape rule, -2 the
// route before variant 3, n >= 0 TMA tile n forced. Returns the previous route, so a test can
// put it back. Process-wide, like the environment variable it overrides.
int fp8gemm_v3_set_route(int route) {
    const int prev = spark::fp8gemm_v3_route();
    TORCH_CHECK(route >= -2 && route < spark::fp8gemm_v3_num_tiles(),
                "fp8gemm_v3_set_route: ", route, " is not -1, -2 or a tile in [0, ",
                spark::fp8gemm_v3_num_tiles(), ")");
    spark::fp8gemm_v3_set_route(route);
    return prev;
}

// ---- fp4 (e2m1): NVFP4 and MXFP4 -------------------------------------------------------
int fp4_format(const std::string& fmt) {
    if (fmt == "nvfp4") return spark::FP4_NVFP4;
    if (fmt == "mxfp4") return spark::FP4_MXFP4;
    TORCH_CHECK(false, "fp4 format must be 'nvfp4' or 'mxfp4', got '", fmt, "'");
    return -1;
}

// Packed e2m1 operands are uint8 or torch.float4_e2m1fn_x2 (two values per byte, the element
// with the lower index in the low nibble); block scales are uint8 or the float8 type of the
// format (float8_e4m3fn for NVFP4, float8_e8m0fnu for MXFP4), all read as bytes.
void check_fp4_bytes(const Tensor& t, const char* name) {
    check_cuda_contig(t, name);
    TORCH_CHECK(t.scalar_type() == at::kByte || t.scalar_type() == at::kFloat4_e2m1fn_x2, name,
                " must be uint8 or float4_e2m1fn_x2 (packed e2m1), got ", t.scalar_type());
    TORCH_CHECK(t.dim() == 2, name, " must be 2-D [rows, K / 2]");
    TORCH_CHECK(aligned16(t), name, " needs 16-byte aligned storage");
}
void check_fp4_scales(const Tensor& t, const char* name, int rows, int K, int format,
                      const Tensor& like) {
    check_cuda_contig(t, name);
    TORCH_CHECK(t.scalar_type() == at::kByte ||
                    (format == spark::FP4_NVFP4 && t.scalar_type() == at::kFloat8_e4m3fn) ||
                    (format == spark::FP4_MXFP4 && t.scalar_type() == at::kFloat8_e8m0fnu),
                name, " must be uint8 or the format's float8 scale type, got ", t.scalar_type());
    const auto want = static_cast<int64_t>(spark::fp4_scale_bytes(rows, K, format));
    TORCH_CHECK(t.numel() == want, name, " must hold the blocked scale layout of ", rows, " rows (",
                want, " bytes, rows padded to 128), got ", t.numel());
    TORCH_CHECK(t.device() == like.device(), name, " must be on a's device");
    TORCH_CHECK(aligned16(t), name, " needs 16-byte aligned storage");
}

// (q, sf, scale) for x [rows, K] bf16: q [rows, K/2] uint8 packed e2m1, sf the block scales in
// the blocked layout (a flat uint8 tensor of fp4_scale_bytes), scale the per-tensor decode
// scale (NVFP4: the given one, or max|x| / (6 * 448) computed on the device; None for MXFP4).
std::tuple<Tensor, Tensor, std::optional<Tensor>> fp4_quantize(const Tensor& x,
                                                               const std::string& fmt,
                                                               const std::optional<Tensor>& scale) {
    check_cuda_contig(x, "x");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "fp4_quantize expects bfloat16 x");
    TORCH_CHECK(x.dim() == 2, "fp4_quantize expects a 2-D [rows, K] tensor");
    TORCH_CHECK(aligned16(x), "fp4_quantize needs 16-byte aligned x storage");
    const int format = fp4_format(fmt);
    TORCH_CHECK(x.size(0) <= INT32_MAX && x.size(1) <= INT32_MAX, "fp4_quantize dims too large");
    const int rows = static_cast<int>(x.size(0)), K = static_cast<int>(x.size(1));
    TORCH_CHECK(K % (format == spark::FP4_MXFP4 ? 128 : 64) == 0,
                "fp4_quantize needs K a multiple of 64 (nvfp4) or 128 (mxfp4), got ", K);
    const c10::cuda::CUDAGuard guard(x.device());
    std::optional<Tensor> s;
    if (format == spark::FP4_NVFP4) {
        if (scale.has_value()) {
            TORCH_CHECK(scale->is_cuda() && scale->scalar_type() == at::kFloat &&
                            scale->numel() == 1 && scale->device() == x.device(),
                        "scale must be a float32 CUDA tensor with one element on x's device");
            s = scale->reshape({}).contiguous();
        }
    }
    // no scale given: max|x| / (6 * 448) on the device, an amax pass into a zeroed cell
    Tensor work;
    if (format == spark::FP4_NVFP4 && !scale.has_value()) {
        work = at::zeros({2}, x.options().dtype(at::kFloat));  // [amax bits, scale]
        s = work.narrow(0, 1, 1).reshape({});
    }
    Tensor q = at::empty({rows, K / 2}, x.options().dtype(at::kByte));
    Tensor sf = at::empty({static_cast<int64_t>(spark::fp4_scale_bytes(rows, K, format))},
                          x.options().dtype(at::kByte));
    spark::fp4_quantize(bf16_ptr(x), static_cast<unsigned char*>(q.data_ptr()),
                        static_cast<unsigned char*>(sf.data_ptr()), rows, K,
                        s.has_value() ? s->data_ptr<float>() : nullptr, format, current_stream(x),
                        work.defined() ? static_cast<unsigned*>(work.data_ptr()) : nullptr,
                        work.defined() ? s->data_ptr<float>() : nullptr);
    return {q, sf, s};
}

// bf16 [rows, K] -> (e4m3 [rows, K], scales): one fp32 scale ("tensor", shape [1]), one per
// row ("row", [rows, 1]) or one e8m0 byte per 32 values ("mx", [rows, K / 32] uint8).
std::tuple<Tensor, Tensor> fp8_quantize(const Tensor& x, const std::string& mode) {
    check_cuda_contig(x, "x");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16, "fp8_quantize expects bfloat16 x");
    TORCH_CHECK(x.dim() == 2, "fp8_quantize expects a 2-D [rows, K] tensor");
    TORCH_CHECK(aligned16(x), "fp8_quantize needs 16-byte aligned x storage");
    TORCH_CHECK(x.size(0) <= INT32_MAX && x.size(1) <= INT32_MAX, "fp8_quantize dims too large");
    const int rows = static_cast<int>(x.size(0)), K = static_cast<int>(x.size(1));
    int m;
    if (mode == "tensor") {
        m = spark::FP8Q_TENSOR;
    } else if (mode == "row") {
        m = spark::FP8Q_ROW;
    } else {
        TORCH_CHECK(mode == "mx", "fp8_quantize mode must be 'tensor', 'row' or 'mx', got ", mode);
        m = spark::FP8Q_MX;
    }
    TORCH_CHECK(K % (m == spark::FP8Q_MX ? 32 : 8) == 0,
                "fp8_quantize needs K a multiple of 8 (32 for mx), got ", K);
    TORCH_CHECK(rows > 0 && K > 0, "fp8_quantize needs a non-empty x");
    const c10::cuda::CUDAGuard guard(x.device());
    Tensor q = at::empty({rows, K}, x.options().dtype(at::kFloat8_e4m3fn));
    Tensor scale;
    Tensor work;
    if (m == spark::FP8Q_TENSOR) {
        work = at::zeros({2}, x.options().dtype(at::kFloat));  // [amax bits, scale]
        scale = work.narrow(0, 1, 1);
    } else if (m == spark::FP8Q_ROW) {
        scale = at::empty({rows, 1}, x.options().dtype(at::kFloat));
    } else {
        scale = at::empty({rows, K / 32}, x.options().dtype(at::kByte));
    }
    spark::fp8_quantize(bf16_ptr(x), static_cast<unsigned char*>(q.data_ptr()), scale.data_ptr(),
                        work.defined() ? static_cast<unsigned*>(work.data_ptr()) : nullptr, rows, K,
                        m, current_stream(x));
    return {q, scale};
}

// C = scale_a * scale_b * sum_k (sfa a)(sfb b_t) with a [M, K/2] and b_t [N, K/2] packed
// e2m1, block scales in the blocked layout, per-tensor fp32 scales optional (1.0), bf16 out.
Tensor fp4gemm(const Tensor& a, const Tensor& b_t, const Tensor& sfa, const Tensor& sfb,
               const std::optional<Tensor>& scale_a, const std::optional<Tensor>& scale_b,
               const std::string& fmt, int variant) {
    check_fp4_bytes(a, "a");
    check_fp4_bytes(b_t, "b_t");
    const int format = fp4_format(fmt);
    TORCH_CHECK(a.size(1) == b_t.size(1), "inner dimensions mismatch: a is ", a.size(0), "x",
                a.size(1), " bytes, b_t is ", b_t.size(0), "x", b_t.size(1),
                " (both [rows, K / 2])");
    TORCH_CHECK(b_t.device() == a.device(), "b_t must be on a's device");
    TORCH_CHECK(a.size(0) <= INT32_MAX && b_t.size(0) <= INT32_MAX && 2 * a.size(1) <= INT32_MAX,
                "fp4gemm dims too large for int32");
    const int M = static_cast<int>(a.size(0)), N = static_cast<int>(b_t.size(0)),
              K = static_cast<int>(2 * a.size(1));
    TORCH_CHECK(scale_a.has_value() == scale_b.has_value(),
                "fp4gemm: scale_a and scale_b must both be given or both be None");
    const float* sa = nullptr;
    const float* sb = nullptr;
    if (scale_a.has_value()) {
        for (const Tensor* sp : {&*scale_a, &*scale_b}) {
            TORCH_CHECK(sp->is_cuda() && sp->scalar_type() == at::kFloat && sp->numel() == 1 &&
                            sp->device() == a.device(),
                        "scale_a and scale_b must be float32 CUDA tensors with one element");
        }
        sa = scale_a->data_ptr<float>();
        sb = scale_b->data_ptr<float>();
    }
    check_fp4_scales(sfa, "sfa", M, K, format, a);
    check_fp4_scales(sfb, "sfb", N, K, format, a);
    const c10::cuda::CUDAGuard guard(a.device());
    Tensor c = at::empty({M, N}, a.options().dtype(at::kBFloat16));
    int v = resolve_variant(variant, spark::fp4gemm_num_variants());
    while (variant < 0 && v > 0 && !spark::fp4gemm_supports(M, N, K, v)) --v;
    spark::fp4gemm(static_cast<const unsigned char*>(a.data_ptr()),
                   static_cast<const unsigned char*>(b_t.data_ptr()), bf16_ptr_mut(c), M, N, K,
                   static_cast<const unsigned char*>(sfa.data_ptr()),
                   static_cast<const unsigned char*>(sfb.data_ptr()), sa, sb, format, v,
                   current_stream(a));
    return c;
}

// ---- W4A16 ----------------------------------------------------------------------------------

// Round-to-nearest int4 quantization of w [K, N] bf16 (the [K, N] layout hgemm takes), one
// scale per 128 k of a column: (qweight [K/8, N] int32 in GPTQ's packing, scales [K/128, N]
// bf16, zeros [K/128, N] uint8 or None). The formulas are in include/spark/kernels.h.
std::tuple<Tensor, Tensor, std::optional<Tensor>> w4_quantize(const Tensor& w, bool asym) {
    check_cuda_contig(w, "w");
    TORCH_CHECK(w.scalar_type() == at::kBFloat16, "w4_quantize expects a bfloat16 weight");
    TORCH_CHECK(w.dim() == 2, "w4_quantize expects a 2-D [K, N] weight");
    const int64_t K = w.size(0), N = w.size(1);
    TORCH_CHECK(K % spark::kW4GroupSize == 0 && N % 16 == 0 && K <= INT32_MAX && N <= INT32_MAX,
                "w4_quantize needs K a multiple of 128 and N a multiple of 16, got ", K, "x", N);
    const c10::cuda::CUDAGuard guard(w.device());
    Tensor q = at::empty({K / 8, N}, w.options().dtype(at::kInt));
    Tensor sc = at::empty({K / spark::kW4GroupSize, N}, w.options());
    std::optional<Tensor> z;
    if (asym) z = at::empty({K / spark::kW4GroupSize, N}, w.options().dtype(at::kByte));
    spark::w4_quantize_bf16(bf16_ptr(w), q.data_ptr<int32_t>(), bf16_ptr_mut(sc),
                            z ? z->data_ptr<uint8_t>() : nullptr, static_cast<int>(K),
                            static_cast<int>(N), current_stream(w));
    return {q, sc, z};
}

// qweight [K/8, N] int32 -> the kernel layout, int32 [N/16, 2K].
Tensor w4_repack(const Tensor& qweight) {
    check_cuda_contig(qweight, "qweight");
    TORCH_CHECK(qweight.scalar_type() == at::kInt && qweight.dim() == 2,
                "w4_repack expects a 2-D int32 qweight [K/8, N]");
    const int64_t K = qweight.size(0) * 8, N = qweight.size(1);
    TORCH_CHECK(K % 64 == 0 && N % 16 == 0 && K <= INT32_MAX && N <= INT32_MAX,
                "w4_repack needs K a multiple of 64 and N a multiple of 16, got ", K, "x", N);
    const c10::cuda::CUDAGuard guard(qweight.device());
    Tensor packed = at::empty({N / 16, 2 * K}, qweight.options());
    spark::w4_repack(qweight.data_ptr<int32_t>(), packed.data_ptr<int32_t>(), static_cast<int>(K),
                     static_cast<int>(N), current_stream(qweight));
    return packed;
}

// a [M, K] bf16 times the weight in `packed` [N/16, 2K] int32 (w4_repack), `scales`
// [K/128, N] bf16 and `zeros` [K/128, N] uint8 (None: symmetric) -> [M, N] bf16.
Tensor w4gemm(const Tensor& a, const Tensor& packed, const Tensor& scales,
              const std::optional<Tensor>& zeros, int variant) {
    check_cuda_contig(a, "a");
    check_cuda_contig(packed, "packed");
    check_cuda_contig(scales, "scales");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && a.dim() == 2,
                "w4gemm expects a 2-D bfloat16 a");
    TORCH_CHECK(packed.scalar_type() == at::kInt && packed.dim() == 2,
                "w4gemm expects packed as the int32 [N/16, 2K] tensor from w4_repack");
    const int64_t M = a.size(0), K = a.size(1), N = packed.size(0) * 16;
    TORCH_CHECK(packed.size(1) == 2 * K, "packed is [", packed.size(0), ", ", packed.size(1),
                "], expected [N/16, 2K] = [", packed.size(0), ", ", 2 * K, "] for K = ", K);
    TORCH_CHECK(K % spark::kW4GroupSize == 0, "w4gemm needs K a multiple of 128, got ", K);
    TORCH_CHECK(M <= INT32_MAX && N <= INT32_MAX && K <= INT32_MAX, "w4gemm dims too large");
    TORCH_CHECK(scales.scalar_type() == at::kBFloat16 && scales.dim() == 2 &&
                    scales.size(0) == K / spark::kW4GroupSize && scales.size(1) == N,
                "scales must be bfloat16 [", K / spark::kW4GroupSize, ", ", N, "], got ",
                scales.sizes());
    const uint8_t* zp = nullptr;
    if (zeros) {
        check_cuda_contig(*zeros, "zeros");
        TORCH_CHECK(zeros->scalar_type() == at::kByte && zeros->dim() == 2 &&
                        zeros->size(0) == K / spark::kW4GroupSize && zeros->size(1) == N,
                    "zeros must be uint8 [", K / spark::kW4GroupSize, ", ", N, "], got ",
                    zeros->sizes());
        TORCH_CHECK(zeros->device() == a.device(), "zeros must be on a's device");
        zp = zeros->data_ptr<uint8_t>();
    }
    TORCH_CHECK(packed.device() == a.device() && scales.device() == a.device(),
                "packed and scales must be on a's device");
    TORCH_CHECK(
        aligned16(a) && aligned16(packed) && aligned16(scales) && (!zeros || aligned16(*zeros)),
        "w4gemm needs 16-byte aligned a, packed, scales and zeros storage");
    const c10::cuda::CUDAGuard guard(a.device());
    Tensor c = at::empty({M, N}, a.options());
    const int v = resolve_variant(variant, spark::w4gemm_num_variants());
    spark::w4gemm_bf16(bf16_ptr(a), packed.data_ptr<int32_t>(), bf16_ptr(scales), zp,
                       bf16_ptr_mut(c), static_cast<int>(M), static_cast<int>(N),
                       static_cast<int>(K), v, current_stream(a));
    return c;
}

// ---- W1A16 (sign and ternary weights) -------------------------------------------------------

void check_w1_bits(int bits, const char* op) {
    TORCH_CHECK(bits == 1 || bits == 2, op, " takes bits 1 (sign) or 2 (ternary), got ", bits);
}

// w [K, N] bf16 -> (packed [N, K * bits / 32] int32, scales [K/128, N] bf16): one scale per
// 128 k of a column, s = bf16(mean|w|); bits == 1 codes w >= 0, bits == 2 codes w against
// +-s/2. k ascends from the lowest bits of the lowest word of its row (kernels.h).
std::tuple<Tensor, Tensor> w1_quantize(const Tensor& w, int bits) {
    check_cuda_contig(w, "w");
    check_w1_bits(bits, "w1_quantize");
    TORCH_CHECK(w.scalar_type() == at::kBFloat16, "w1_quantize expects a bfloat16 weight");
    TORCH_CHECK(w.dim() == 2, "w1_quantize expects a 2-D [K, N] weight");
    const int64_t K = w.size(0), N = w.size(1);
    TORCH_CHECK(K % spark::kW1GroupSize == 0 && N % 16 == 0 && K <= INT32_MAX && N <= INT32_MAX,
                "w1_quantize needs K a multiple of 128 and N a multiple of 16, got ", K, "x", N);
    const c10::cuda::CUDAGuard guard(w.device());
    Tensor packed = at::empty({N, K * bits / 32}, w.options().dtype(at::kInt));
    Tensor sc = at::empty({K / spark::kW1GroupSize, N}, w.options());
    spark::w1_quantize_bf16(bf16_ptr(w), reinterpret_cast<uint32_t*>(packed.data_ptr<int32_t>()),
                            bf16_ptr_mut(sc), static_cast<int>(K), static_cast<int>(N), bits,
                            current_stream(w));
    return {packed, sc};
}

// The checks w1gemm and w1gemm_moe share on one expert's (packed [N, K * bits / 32], scales
// [K/128, N]) pair, given the trailing two dims of each; returns N.
int64_t check_w1_weight(const Tensor& a, const Tensor& packed, const Tensor& scales, int bits,
                        int64_t K, const char* op) {
    check_cuda_contig(packed, "packed");
    check_cuda_contig(scales, "scales");
    TORCH_CHECK(packed.scalar_type() == at::kInt, op,
                " expects packed as the int32 tensor "
                "from w1_quantize, got ",
                packed.scalar_type());
    const int64_t N = packed.size(-2);
    TORCH_CHECK(packed.size(-1) == K * bits / 32, "packed is ", packed.sizes(),
                ", expected a trailing [N, K * bits / 32] = [N, ", K * bits / 32, "] for K = ", K,
                ", bits = ", bits);
    TORCH_CHECK(K % spark::kW1GroupSize == 0, op, " needs K a multiple of 128, got ", K);
    TORCH_CHECK(N % 16 == 0, op, " needs N a multiple of 16, got ", N);
    TORCH_CHECK(N <= INT32_MAX && K <= INT32_MAX, op, " dims too large");
    TORCH_CHECK(scales.scalar_type() == at::kBFloat16 &&
                    scales.size(-2) == K / spark::kW1GroupSize && scales.size(-1) == N,
                "scales must be bfloat16 with trailing [", K / spark::kW1GroupSize, ", ", N,
                "], got ", scales.sizes());
    TORCH_CHECK(packed.device() == a.device() && scales.device() == a.device(),
                "packed and scales must be on a's device");
    TORCH_CHECK(aligned16(a) && aligned16(packed) && aligned16(scales), op,
                " needs 16-byte aligned a, packed and scales storage");
    return N;
}

// a [M, K] bf16 times the sign or ternary weight (packed [N, K * bits / 32] int32, scales
// [K/128, N] bf16 from w1_quantize) -> [M, N] bf16, fp32 accumulate.
Tensor w1gemm(const Tensor& a, const Tensor& packed, const Tensor& scales, int bits, int variant) {
    check_cuda_contig(a, "a");
    check_w1_bits(bits, "w1gemm");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && a.dim() == 2,
                "w1gemm expects a 2-D bfloat16 a");
    TORCH_CHECK(packed.dim() == 2 && scales.dim() == 2,
                "w1gemm expects 2-D packed and scales (w1gemm_moe takes the stacked ones)");
    const int64_t M = a.size(0), K = a.size(1);
    const int64_t N = check_w1_weight(a, packed, scales, bits, K, "w1gemm");
    TORCH_CHECK(M <= INT32_MAX, "w1gemm dims too large");
    const c10::cuda::CUDAGuard guard(a.device());
    Tensor c = at::empty({M, N}, a.options());
    const int v = resolve_variant(variant, spark::w1gemm_num_variants());
    spark::w1gemm_bf16(bf16_ptr(a), reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>()),
                       bf16_ptr(scales), bf16_ptr_mut(c), static_cast<int>(M), static_cast<int>(N),
                       static_cast<int>(K), bits, v, current_stream(a));
    return c;
}

// Grouped over E experts: a [T, K] bf16 with its rows sorted by expert, expert e owning rows
// offsets[e]..offsets[e+1]-1 (offsets int32 [E+1] on the device, offsets[E] == T, empty
// experts allowed); packed [E, N, K * bits / 32] int32 and scales [E, K/128, N] bf16 are the
// per-expert w1_quantize outputs stacked -> [T, N] bf16.
Tensor w1gemm_moe(const Tensor& a, const Tensor& packed, const Tensor& scales,
                  const Tensor& offsets, int bits, int variant) {
    check_cuda_contig(a, "a");
    check_cuda_contig(offsets, "offsets");
    check_w1_bits(bits, "w1gemm_moe");
    TORCH_CHECK(a.scalar_type() == at::kBFloat16 && a.dim() == 2,
                "w1gemm_moe expects a 2-D bfloat16 a [T, K]");
    TORCH_CHECK(packed.dim() == 3 && scales.dim() == 3,
                "w1gemm_moe expects packed [E, N, K * bits / 32] and scales [E, K/128, N]");
    const int64_t E = packed.size(0);
    TORCH_CHECK(scales.size(0) == E, "scales has ", scales.size(0), " experts, packed has ", E);
    const int64_t T = a.size(0), K = a.size(1);
    const int64_t N = check_w1_weight(a, packed, scales, bits, K, "w1gemm_moe");
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.dim() == 1 && offsets.size(0) == E + 1,
                "offsets must be int32 [E + 1] = [", E + 1, "], got ", offsets.scalar_type(), " ",
                offsets.sizes());
    TORCH_CHECK(offsets.device() == a.device(), "offsets must be on a's device");
    TORCH_CHECK(T <= INT32_MAX && E <= INT32_MAX, "w1gemm_moe dims too large");
    const c10::cuda::CUDAGuard guard(a.device());
    Tensor c = at::empty({T, N}, a.options());
    const int v = resolve_variant(variant, spark::w1gemm_num_variants());
    spark::w1gemm_moe_bf16(
        bf16_ptr(a), reinterpret_cast<const uint32_t*>(packed.data_ptr<int32_t>()),
        bf16_ptr(scales), offsets.data_ptr<int32_t>(), bf16_ptr_mut(c), static_cast<int>(E),
        static_cast<int>(N), static_cast<int>(K), bits, v, current_stream(a), static_cast<int>(T));
    return c;
}

// Grouped GEMM over E experts for MoE prefill: a [T, K] with its rows sorted by expert
// (offsets int32 [E + 1] on the device, the w1gemm_moe convention), w [E, N, K] (each
// expert's weight as nn.Linear stores it) -> [T, N] bf16, row t = a[t] @ w[e].T. bf16 a and
// w, or float8_e4m3fn a and w with scale_a (one element) and scale_b (one per expert, or
// one for all) float32 on the device, either optional (1.0).
Tensor grouped_gemm(const Tensor& a, const Tensor& w, const Tensor& offsets,
                    const std::optional<Tensor>& scale_a, const std::optional<Tensor>& scale_b) {
    check_cuda_contig(a, "a");
    check_cuda_contig(w, "w");
    check_cuda_contig(offsets, "offsets");
    const bool fp8 = a.scalar_type() == at::kFloat8_e4m3fn;
    TORCH_CHECK((fp8 && w.scalar_type() == at::kFloat8_e4m3fn) ||
                    (a.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16),
                "grouped_gemm expects a and w both bfloat16 or both float8_e4m3fn, got ",
                a.scalar_type(), " and ", w.scalar_type());
    TORCH_CHECK(a.dim() == 2 && w.dim() == 3, "grouped_gemm expects a [T, K] and w [E, N, K]");
    const int64_t T = a.size(0), K = a.size(1), E = w.size(0), N = w.size(1);
    TORCH_CHECK(w.size(2) == K, "w is ", w.sizes(), ", expected [E, N, ", K, "]");
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.dim() == 1 && offsets.size(0) == E + 1,
                "offsets must be int32 [E + 1] = [", E + 1, "], got ", offsets.scalar_type(), " ",
                offsets.sizes());
    TORCH_CHECK(w.device() == a.device() && offsets.device() == a.device(),
                "w and offsets must be on a's device");
    TORCH_CHECK(T <= INT32_MAX && K <= INT32_MAX && N <= INT32_MAX && E <= INT32_MAX,
                "grouped_gemm dims too large");
    TORCH_CHECK(spark::grouped_gemm_supports(static_cast<int>(E), static_cast<int>(N),
                                             static_cast<int>(K), fp8),
                "grouped_gemm needs 1 <= E <= 512, N a multiple of 128 and K a multiple of ",
                fp8 ? 128 : 64, ", got E = ", E, ", N = ", N, ", K = ", K);
    TORCH_CHECK(aligned16(a) && aligned16(w), "grouped_gemm needs 16-byte aligned a and w");
    const float* sa = nullptr;
    const float* sb = nullptr;
    Tensor sb_full;
    if (fp8) {
        if (scale_a.has_value()) {
            TORCH_CHECK(scale_a->is_cuda() && scale_a->scalar_type() == at::kFloat &&
                            scale_a->numel() == 1 && scale_a->device() == a.device(),
                        "scale_a must be a float32 tensor with one element on a's device");
            sa = scale_a->data_ptr<float>();
        }
        if (scale_b.has_value()) {
            TORCH_CHECK(scale_b->is_cuda() && scale_b->scalar_type() == at::kFloat &&
                            (scale_b->numel() == 1 || scale_b->numel() == E) &&
                            scale_b->device() == a.device(),
                        "scale_b must be float32 with 1 or E elements on a's device");
            sb_full = scale_b->reshape({-1}).expand({E}).contiguous();
            sb = sb_full.data_ptr<float>();
        }
    } else {
        TORCH_CHECK(!scale_a.has_value() && !scale_b.has_value(),
                    "grouped_gemm: scales are for float8_e4m3fn inputs only");
    }
    const c10::cuda::CUDAGuard guard(a.device());
    Tensor c = at::empty({T, N}, a.options().dtype(at::kBFloat16));
    if (fp8) {
        spark::grouped_gemm_fp8(reinterpret_cast<const __nv_fp8_e4m3*>(a.data_ptr()),
                                reinterpret_cast<const __nv_fp8_e4m3*>(w.data_ptr()),
                                offsets.data_ptr<int32_t>(), bf16_ptr_mut(c), static_cast<int>(T),
                                static_cast<int>(E), static_cast<int>(N), static_cast<int>(K), sa,
                                sb, current_stream(a));
    } else {
        spark::grouped_gemm_bf16(bf16_ptr(a), bf16_ptr(w), offsets.data_ptr<int32_t>(),
                                 bf16_ptr_mut(c), static_cast<int>(T), static_cast<int>(E),
                                 static_cast<int>(N), static_cast<int>(K), current_stream(a));
    }
    return c;
}

// ---- Linear cross-entropy pieces (src/kernels/cross_entropy.cu) ---------------------------
// h [T, D] and w [V, D] bf16, targets int64 [T]. The ladder that puts them together is
// linear_cross_entropy in python/spark_kernels/ops.py.
struct CeShape {
    int T, V, D;
};
CeShape ce_check(const Tensor& h, const Tensor& w, const Tensor* targets, const char* who) {
    check_cuda_contig(h, "h");
    check_cuda_contig(w, "w");
    TORCH_CHECK(h.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16, who,
                " expects bfloat16 h and w");
    TORCH_CHECK(h.dim() == 2 && w.dim() == 2 && h.size(1) == w.size(1), who,
                " expects h [T, D] and w [V, D], got ", h.sizes(), " and ", w.sizes());
    TORCH_CHECK(w.device() == h.device(), who, ": w must be on h's device");
    TORCH_CHECK(h.size(0) <= INT32_MAX && w.size(0) <= INT32_MAX && h.size(1) <= INT32_MAX, who,
                ": dims too large");
    const CeShape s{static_cast<int>(h.size(0)), static_cast<int>(w.size(0)),
                    static_cast<int>(h.size(1))};
    TORCH_CHECK(spark::ce_gemm_supports(s.V, s.D), who,
                " needs V a multiple of 128 and D a multiple of 64, got V = ", s.V, ", D = ", s.D);
    TORCH_CHECK(aligned16(h) && aligned16(w), who, " needs 16-byte aligned h and w");
    if (targets) {
        check_cuda_contig(*targets, "targets");
        TORCH_CHECK(
            targets->scalar_type() == at::kLong && targets->dim() == 1 && targets->size(0) == s.T,
            who, " expects int64 targets [", s.T, "], got ", targets->scalar_type(), " ",
            targets->sizes());
        TORCH_CHECK(targets->device() == h.device(), who, ": targets must be on h's device");
    }
    return s;
}
Tensor ce_out(const std::optional<Tensor>& out, const Tensor& h, int T, int V) {
    if (!out) return at::empty({T, V}, h.options());
    check_cuda_contig(*out, "out");
    TORCH_CHECK(out->scalar_type() == at::kBFloat16 && out->dim() == 2 && out->size(0) == T &&
                    out->size(1) == V && out->device() == h.device(),
                "out must be bfloat16 [", T, ", ", V, "] on h's device, got ", out->sizes());
    TORCH_CHECK(aligned16(*out), "out must be 16-byte aligned");
    return *out;
}
const float* ce_f32(const Tensor& t, int64_t n, const Tensor& h, const char* name) {
    check_cuda_contig(t, name);
    TORCH_CHECK(t.scalar_type() == at::kFloat && t.numel() == n && t.device() == h.device(), name,
                " must be float32 with ", n, " elements on h's device");
    return t.data_ptr<float>();
}

// logits = h @ w.T, bf16 [T, V], into `out` when given.
Tensor ce_logits(const Tensor& h, const Tensor& w, const std::optional<Tensor>& out) {
    const CeShape s = ce_check(h, w, nullptr, "ce_logits");
    const c10::cuda::CUDAGuard guard(h.device());
    Tensor c = ce_out(out, h, s.T, s.V);
    spark::CeGemmEpilogue ep;
    ep.out = bf16_ptr_mut(c);
    spark::ce_gemm_bf16(bf16_ptr(h), bf16_ptr(w), s.T, s.V, s.D, spark::CE_GEMM_LOGITS, ep,
                        current_stream(h));
    return c;
}

// (lse, loss), float32 [T] each, from the GEMM's per-tile (max, sum) partials; no logits.
std::tuple<Tensor, Tensor> ce_stats(const Tensor& h, const Tensor& w, const Tensor& targets,
                                    int64_t ignore_index) {
    const CeShape s = ce_check(h, w, &targets, "ce_stats");
    const c10::cuda::CUDAGuard guard(h.device());
    const auto f32 = h.options().dtype(at::kFloat);
    Tensor part = at::empty({s.T, s.V / 128, 2}, f32);
    Tensor tgt = at::empty({s.T}, f32);
    Tensor lse = at::empty({s.T}, f32);
    Tensor loss = at::empty({s.T}, f32);
    spark::CeGemmEpilogue ep;
    ep.partials = part.data_ptr<float>();
    ep.target_logit = tgt.data_ptr<float>();
    ep.targets = targets.data_ptr<int64_t>();
    ep.ignore_index = static_cast<int>(ignore_index);
    cudaStream_t stream = current_stream(h);
    spark::ce_gemm_bf16(bf16_ptr(h), bf16_ptr(w), s.T, s.V, s.D, spark::CE_GEMM_STATS, ep, stream);
    spark::ce_combine(ep.partials, ep.target_logit, ep.targets, s.T, s.V, ep.ignore_index,
                      lse.data_ptr<float>(), loss.data_ptr<float>(), stream);
    return {lse, loss};
}

// d loss / d logits = (softmax - one-hot) * scale, bf16 [T, V], recomputed from h and w and
// the forward's lse; scale is float32 [1] on the device.
Tensor ce_grad(const Tensor& h, const Tensor& w, const Tensor& targets, const Tensor& lse,
               const Tensor& scale, int64_t ignore_index, const std::optional<Tensor>& out) {
    const CeShape s = ce_check(h, w, &targets, "ce_grad");
    const c10::cuda::CUDAGuard guard(h.device());
    Tensor c = ce_out(out, h, s.T, s.V);
    spark::CeGemmEpilogue ep;
    ep.out = bf16_ptr_mut(c);
    ep.targets = targets.data_ptr<int64_t>();
    ep.ignore_index = static_cast<int>(ignore_index);
    ep.lse = ce_f32(lse, s.T, h, "lse");
    ep.scale = ce_f32(scale, 1, h, "scale");
    spark::ce_gemm_bf16(bf16_ptr(h), bf16_ptr(w), s.T, s.V, s.D, spark::CE_GEMM_GRAD, ep,
                        current_stream(h));
    return c;
}

// The fused row kernel on bf16 logits [rows, V]: (loss, lse) float32 [rows]; with `scale`
// (float32 [1] on the device) the logits are overwritten by the gradient rows.
std::tuple<Tensor, Tensor> ce_rows(const Tensor& logits_in, const Tensor& targets,
                                   int64_t ignore_index, const std::optional<Tensor>& scale) {
    Tensor logits = logits_in;
    check_cuda_contig(logits, "logits");
    check_cuda_contig(targets, "targets");
    TORCH_CHECK(logits.scalar_type() == at::kBFloat16 && logits.dim() == 2,
                "ce_rows expects bfloat16 logits [rows, V]");
    const int64_t rows = logits.size(0), V = logits.size(1);
    TORCH_CHECK(rows <= INT32_MAX && V <= INT32_MAX, "ce_rows: dims too large");
    TORCH_CHECK(V % 8 == 0 && aligned16(logits), "ce_rows needs V % 8 == 0 and aligned logits");
    TORCH_CHECK(targets.scalar_type() == at::kLong && targets.dim() == 1 &&
                    targets.size(0) == rows && targets.device() == logits.device(),
                "ce_rows expects int64 targets [", rows, "] on the logits' device");
    const c10::cuda::CUDAGuard guard(logits.device());
    const auto f32 = logits.options().dtype(at::kFloat);
    Tensor loss = at::empty({rows}, f32);
    Tensor lse = at::empty({rows}, f32);
    const float* sc = scale ? ce_f32(*scale, 1, logits, "scale") : nullptr;
    spark::cross_entropy_rows_bf16(bf16_ptr_mut(logits), targets.data_ptr<int64_t>(),
                                   static_cast<int>(rows), static_cast<int>(V),
                                   static_cast<int>(ignore_index), sc, loss.data_ptr<float>(),
                                   lse.data_ptr<float>(), current_stream(logits));
    return {loss, lse};
}

// out[c - c0][r] = x[r][c] for c in [c0, c1), r < rows, and 0 for rows <= r < ld: a column
// slab of x [rows, cols] transposed into out [c1 - c0, ld].
Tensor ce_transpose(const Tensor& x, int64_t c0, int64_t c1, int64_t ld, const Tensor& out_in) {
    Tensor out = out_in;
    check_cuda_contig(x, "x");
    check_cuda_contig(out, "out");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16 &&
                    x.dim() == 2 && out.dim() == 2,
                "ce_transpose expects bfloat16 matrices");
    TORCH_CHECK(0 <= c0 && c0 < c1 && c1 <= x.size(1) && ld >= x.size(0),
                "ce_transpose: bad column range or ld");
    TORCH_CHECK(out.size(0) == c1 - c0 && out.size(1) == ld && out.device() == x.device(),
                "ce_transpose: out must be [", c1 - c0, ", ", ld, "] on x's device");
    TORCH_CHECK(x.numel() <= INT64_MAX / 2 && x.size(0) <= INT32_MAX && x.size(1) <= INT32_MAX &&
                    ld <= INT32_MAX,
                "ce_transpose: dims too large");
    const c10::cuda::CUDAGuard guard(x.device());
    spark::transpose_pad_bf16(bf16_ptr(x) + c0, static_cast<int>(x.size(0)),
                              static_cast<int>(x.size(1)), bf16_ptr_mut(out),
                              static_cast<int>(c1 - c0), static_cast<int>(ld), current_stream(x));
    return out;
}

// ---- MoE layer pieces: grouped_gemm's SwiGLU form, routing, permute, combine -------------

// grouped_gemm with the SwiGLU in the epilogue: w [E, 2I, K] with rows alternating gate and
// up (interleave_experts) -> [T, I] bf16, silu(a w_gate^T) * (a w_up^T) from fp32 sums.
Tensor grouped_gemm_swiglu(const Tensor& a, const Tensor& w, const Tensor& offsets,
                           const std::optional<Tensor>& scale_a,
                           const std::optional<Tensor>& scale_b) {
    check_cuda_contig(a, "a");
    check_cuda_contig(w, "w");
    check_cuda_contig(offsets, "offsets");
    const bool fp8 = a.scalar_type() == at::kFloat8_e4m3fn;
    TORCH_CHECK((fp8 && w.scalar_type() == at::kFloat8_e4m3fn) ||
                    (a.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16),
                "grouped_gemm_swiglu expects a and w both bfloat16 or both float8_e4m3fn");
    TORCH_CHECK(a.dim() == 2 && w.dim() == 3,
                "grouped_gemm_swiglu expects a [T, K] and w [E, 2I, K]");
    const int64_t T = a.size(0), K = a.size(1), E = w.size(0), N = w.size(1);
    TORCH_CHECK(w.size(2) == K, "w is ", w.sizes(), ", expected [E, 2I, ", K, "]");
    TORCH_CHECK(offsets.scalar_type() == at::kInt && offsets.dim() == 1 && offsets.size(0) == E + 1,
                "offsets must be int32 [E + 1]");
    TORCH_CHECK(w.device() == a.device() && offsets.device() == a.device(),
                "w and offsets must be on a's device");
    TORCH_CHECK(T <= INT32_MAX && K <= INT32_MAX && N <= INT32_MAX, "grouped_gemm dims too large");
    TORCH_CHECK(spark::grouped_gemm_supports(static_cast<int>(E), static_cast<int>(N),
                                             static_cast<int>(K), fp8),
                "grouped_gemm_swiglu needs 1 <= E <= 512, 2I a multiple of 128 and K a multiple "
                "of ",
                fp8 ? 128 : 64, ", got E = ", E, ", 2I = ", N, ", K = ", K);
    TORCH_CHECK(aligned16(a) && aligned16(w), "grouped_gemm_swiglu needs 16-byte aligned a and w");
    const float* sa = nullptr;
    const float* sb = nullptr;
    Tensor sb_full;
    if (fp8) {
        if (scale_a.has_value()) {
            TORCH_CHECK(scale_a->is_cuda() && scale_a->scalar_type() == at::kFloat &&
                            scale_a->numel() == 1 && scale_a->device() == a.device(),
                        "scale_a must be a float32 tensor with one element on a's device");
            sa = scale_a->data_ptr<float>();
        }
        if (scale_b.has_value()) {
            TORCH_CHECK(scale_b->is_cuda() && scale_b->scalar_type() == at::kFloat &&
                            (scale_b->numel() == 1 || scale_b->numel() == E) &&
                            scale_b->device() == a.device(),
                        "scale_b must be float32 with 1 or E elements on a's device");
            sb_full = scale_b->reshape({-1}).expand({E}).contiguous();
            sb = sb_full.data_ptr<float>();
        }
    } else {
        TORCH_CHECK(!scale_a.has_value() && !scale_b.has_value(),
                    "grouped_gemm_swiglu: scales are for float8_e4m3fn inputs only");
    }
    const c10::cuda::CUDAGuard guard(a.device());
    Tensor c = at::empty({T, N / 2}, a.options().dtype(at::kBFloat16));
    if (fp8) {
        spark::grouped_gemm_swiglu_fp8(
            reinterpret_cast<const __nv_fp8_e4m3*>(a.data_ptr()),
            reinterpret_cast<const __nv_fp8_e4m3*>(w.data_ptr()), offsets.data_ptr<int32_t>(),
            bf16_ptr_mut(c), static_cast<int>(T), static_cast<int>(E), static_cast<int>(N),
            static_cast<int>(K), sa, sb, current_stream(a));
    } else {
        spark::grouped_gemm_swiglu_bf16(bf16_ptr(a), bf16_ptr(w), offsets.data_ptr<int32_t>(),
                                        bf16_ptr_mut(c), static_cast<int>(T), static_cast<int>(E),
                                        static_cast<int>(N), static_cast<int>(K),
                                        current_stream(a));
    }
    return c;
}

// The routing's outputs, allocated and filled by one moe_route launch: ids [T, k] int32,
// weights [T, k] fp32, offsets [E + 1] int32 and rank [T, k] int32 (made into pos by whoever
// runs next).
struct MoeRouting {
    Tensor ids, weights, offsets, pos;
    spark::MoeRoutePlan plan;
    int T, E, k;
};

MoeRouting run_moe_route(const Tensor& logits, int64_t top_k, bool renormalize) {
    check_cuda_contig(logits, "logits");
    check_float_or_bf16(logits, "logits");
    TORCH_CHECK(logits.dim() == 2, "logits must be [T, E]");
    const int64_t T = logits.size(0), E = logits.size(1);
    TORCH_CHECK(T * top_k <= INT32_MAX, "moe_route: too many tokens");
    TORCH_CHECK(spark::moe_route_supports(static_cast<int>(E), static_cast<int>(top_k)),
                "moe_route needs 1 <= top_k <= min(E, 16) and E <= 512, got E = ", E,
                ", top_k = ", top_k);
    const c10::cuda::CUDAGuard guard(logits.device());
    const auto i32 = logits.options().dtype(at::kInt);
    MoeRouting r;
    r.T = static_cast<int>(T);
    r.E = static_cast<int>(E);
    r.k = static_cast<int>(top_k);
    r.ids = at::empty({T, top_k}, i32);
    r.weights = at::empty({T, top_k}, logits.options().dtype(at::kFloat));
    r.offsets = at::empty({E + 1}, i32);
    r.pos = at::empty({T, top_k}, i32);
    r.plan = spark::moe_route(logits.data_ptr(), logits.scalar_type() == at::kFloat, r.T, r.E, r.k,
                              renormalize, r.ids.data_ptr<int32_t>(), r.weights.data_ptr<float>(),
                              r.pos.data_ptr<int32_t>(), r.offsets.data_ptr<int32_t>(),
                              current_stream(logits));
    return r;
}

// Softmax top-k routing: (ids, weights, offsets, pos), pos [T, k] the row of expert order each
// (token, slot) goes to. Two launches (the routing, then the positions).
std::tuple<Tensor, Tensor, Tensor, Tensor> moe_route(const Tensor& logits, int64_t top_k,
                                                     bool renormalize) {
    MoeRouting r = run_moe_route(logits, top_k, renormalize);
    spark::moe_positions(r.ids.data_ptr<int32_t>(), r.pos.data_ptr<int32_t>(), r.plan, r.T, r.E,
                         r.k, current_stream(logits));
    return {r.ids, r.weights, r.offsets, r.pos};
}

// The routing and the rows in expert order: (xs [T * k, D], ids, weights, offsets, pos), two
// launches (the routing, then the permute, which finishes pos as it moves the rows).
std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor> moe_dispatch(const Tensor& x,
                                                                const Tensor& logits, int64_t top_k,
                                                                bool renormalize) {
    check_cuda_contig(x, "x");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2, "x must be bfloat16 [T, D]");
    TORCH_CHECK(logits.dim() == 2 && logits.size(0) == x.size(0),
                "logits must be [T, E] with x's T");
    TORCH_CHECK(x.device() == logits.device(), "x and logits must be on one device");
    const int64_t D = x.size(1);
    TORCH_CHECK(spark::moe_permute_supports(static_cast<int>(D), static_cast<int>(top_k)),
                "moe_dispatch needs D a multiple of 8, at most 8192, and top_k <= 16");
    TORCH_CHECK(aligned16(x), "moe_dispatch needs 16-byte aligned x");
    MoeRouting r = run_moe_route(logits, top_k, renormalize);
    const c10::cuda::CUDAGuard guard(x.device());
    Tensor xs = at::empty({x.size(0) * top_k, D}, x.options());
    spark::moe_permute_bf16(bf16_ptr(x), r.ids.data_ptr<int32_t>(), r.pos.data_ptr<int32_t>(),
                            r.plan, bf16_ptr_mut(xs), r.T, r.E, r.k, static_cast<int>(D),
                            current_stream(x));
    return {xs, r.ids, r.weights, r.offsets, r.pos};
}

// out [T, D] = sum over s of weights[t, s] * y[pos[t, s]], fp32 in slot order, one rounding.
Tensor moe_combine(const Tensor& y, const Tensor& pos, const Tensor& weights) {
    check_cuda_contig(y, "y");
    check_cuda_contig(pos, "pos");
    check_cuda_contig(weights, "weights");
    TORCH_CHECK(y.scalar_type() == at::kBFloat16 && y.dim() == 2, "y must be bfloat16 [T * k, D]");
    TORCH_CHECK(pos.scalar_type() == at::kInt && pos.dim() == 2, "pos must be int32 [T, k]");
    TORCH_CHECK(weights.scalar_type() == at::kFloat && weights.sizes() == pos.sizes(),
                "weights must be float32 with pos's shape");
    TORCH_CHECK(y.device() == pos.device() && y.device() == weights.device(),
                "y, pos and weights must be on one device");
    const int64_t T = pos.size(0), k = pos.size(1), D = y.size(1);
    TORCH_CHECK(y.size(0) == T * k, "y has ", y.size(0), " rows, pos says ", T * k);
    TORCH_CHECK(spark::moe_permute_supports(static_cast<int>(D), static_cast<int>(k)),
                "moe_combine needs D a multiple of 8, at most 8192, and k <= 16");
    TORCH_CHECK(aligned16(y), "moe_combine needs 16-byte aligned y");
    const c10::cuda::CUDAGuard guard(y.device());
    Tensor out = at::empty({T, D}, y.options());
    spark::moe_combine_bf16(bf16_ptr(y), pos.data_ptr<int32_t>(), weights.data_ptr<float>(),
                            bf16_ptr_mut(out), static_cast<int>(T), static_cast<int>(k),
                            static_cast<int>(D), current_stream(y));
    return out;
}

// The rows each K/V head is allocated when k and v are the leading S_kv rows of every head of
// a [B, H_kv, cap, D] cache (`cache[:, :, :length]`, the filled part of a cache with spare
// capacity), which the forward reads in place; 0 for packed k and v. Anything else is not a
// layout the kernels take.
int64_t kv_capacity(const Tensor& k, const Tensor& v) {
    if (k.is_contiguous() && v.is_contiguous()) return 0;
    TORCH_CHECK(k.strides() == v.strides(), "k and v must have the same strides, got ", k.strides(),
                " and ", v.strides());
    const int64_t D = k.size(3), S = k.size(2);
    const auto st = k.strides();
    TORCH_CHECK(
        st[3] == 1 && st[2] == D && st[1] >= S * D && st[1] % D == 0 && st[0] == k.size(1) * st[1],
        "k and v must be contiguous, or the first S_kv rows of every head of a "
        "[B, H_kv, capacity, D] cache (k[:, :, :S_kv]); got shape ",
        k.sizes(), " with strides ", k.strides());
    return st[1] / D;
}

// The checks every attention entry point shares: q = [B, H_q, S_q, D] and k, v =
// [B, H_kv, S_kv, D] bf16 CUDA tensors, H_q a multiple of H_kv (grouped-query attention:
// query head h reads k/v head h / (H_q / H_kv)), D in {64, 128}, 16-byte aligned. q is
// contiguous; k and v are contiguous or, when `cache_kv`, the leading rows of a cache
// (kv_capacity above).
void check_attention_inputs(const Tensor& q, const Tensor& k, const Tensor& v,
                            bool cache_kv = false) {
    check_cuda_contig(q, "q");
    if (cache_kv) {
        for (const auto& [t, name] : {std::pair{&k, "k"}, std::pair{&v, "v"}}) {
            TORCH_CHECK(t->is_cuda(), name, " must be a CUDA tensor");
            TORCH_CHECK(t->numel() > 0, name, " must be non-empty");
        }
    } else {
        check_cuda_contig(k, "k");
        check_cuda_contig(v, "v");
    }
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
    if (cache_kv) {
        const int64_t cap = kv_capacity(k, v);
        TORCH_CHECK(
            cap <= INT32_MAX && cap * k.size(3) * k.size(1) * k.size(0) < (int64_t{1} << 40),
            "attention: the K/V cache is too large");
    }
}

// The forward rung for `variant` (-1: the top rung that accepts the shape).
int attention_variant(const Tensor& q, const Tensor& k, int variant) {
    int var = resolve_variant(variant, spark::attention_num_variants());
    while (variant < 0 && var > 0 &&
           !spark::attention_supports(static_cast<int>(q.size(2)), static_cast<int>(k.size(2)),
                                      static_cast<int>(q.size(3)), var))
        --var;
    return var;
}

// O = softmax(q k^T / sqrt(D)) v; every variant takes any S_q, S_kv >= 1. With `lse` the
// kernel also writes the natural-log log-sum-exp of every row of scaled scores. k and v may
// be the filled part of a cache with spare capacity (kv_capacity), read where it is.
void run_attention(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out, bool causal,
                   int variant, float* lse) {
    const c10::cuda::CUDAGuard guard(q.device());
    spark::attention_bf16(bf16_ptr(q), bf16_ptr(k), bf16_ptr(v), bf16_ptr_mut(out),
                          static_cast<int>(q.size(0)), static_cast<int>(q.size(1)),
                          static_cast<int>(k.size(1)), static_cast<int>(q.size(2)),
                          static_cast<int>(k.size(2)), static_cast<int>(q.size(3)), causal,
                          attention_variant(q, k, variant), current_stream(q), lse,
                          static_cast<int>(kv_capacity(k, v)));
}

Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal, int variant) {
    check_attention_inputs(q, k, v, /*cache_kv=*/true);
    Tensor out = at::empty_like(q);
    run_attention(q, k, v, out, causal, variant, nullptr);
    return out;
}

// The forward for training: (O, lse) with lse fp32 [B, H_q, S_q], what attention_bwd takes.
std::tuple<Tensor, Tensor> attention_fwd(const Tensor& q, const Tensor& k, const Tensor& v,
                                         bool causal, int variant) {
    check_attention_inputs(q, k, v, /*cache_kv=*/true);
    Tensor out = at::empty_like(q);
    Tensor lse = at::empty({q.size(0), q.size(1), q.size(2)}, q.options().dtype(at::kFloat));
    run_attention(q, k, v, out, causal, variant, lse.data_ptr<float>());
    return {out, lse};
}

// (dq, dk, dv) of O = attention(q, k, v) for the upstream gradient d_out, from the forward's
// output `out` and log-sum-exp `lse`. dk and dv have k's shape: under GQA they are summed over
// the query heads of each group.
std::tuple<Tensor, Tensor, Tensor> attention_bwd(const Tensor& q, const Tensor& k, const Tensor& v,
                                                 const Tensor& out, const Tensor& d_out,
                                                 const Tensor& lse, bool causal, int variant,
                                                 bool deterministic) {
    check_attention_inputs(q, k, v);
    check_cuda_contig(out, "out");
    check_cuda_contig(d_out, "d_out");
    check_cuda_contig(lse, "lse");
    TORCH_CHECK(out.sizes() == q.sizes() && d_out.sizes() == q.sizes(),
                "out and d_out must have q's shape ", q.sizes());
    TORCH_CHECK(out.scalar_type() == at::kBFloat16 && d_out.scalar_type() == at::kBFloat16,
                "attention_bwd expects bfloat16 out and d_out");
    TORCH_CHECK(lse.scalar_type() == at::kFloat &&
                    lse.sizes() == at::IntArrayRef({q.size(0), q.size(1), q.size(2)}),
                "lse must be float32 [B, H_q, S_q]");
    TORCH_CHECK(aligned16(out) && aligned16(d_out), "attention_bwd needs 16-byte aligned storage");
    const int var = resolve_variant(variant, spark::attention_bwd_num_variants());
    const c10::cuda::CUDAGuard guard(q.device());
    Tensor dq = at::empty_like(q), dk = at::empty_like(k), dv = at::empty_like(v);
    spark::attention_bwd_bf16(
        bf16_ptr(q), bf16_ptr(k), bf16_ptr(v), bf16_ptr(out), bf16_ptr(d_out),
        lse.data_ptr<float>(), bf16_ptr_mut(dq), bf16_ptr_mut(dk), bf16_ptr_mut(dv),
        static_cast<int>(q.size(0)), static_cast<int>(q.size(1)), static_cast<int>(k.size(1)),
        static_cast<int>(q.size(2)), static_cast<int>(k.size(2)), static_cast<int>(q.size(3)),
        causal, deterministic, var, current_stream(q));
    return {dq, dk, dv};
}

// The fp8 forward: q = [B, H_q, S_q, D], k, v = [B, H_kv, S_kv, D] float8_e4m3fn, with fp32
// descale factors q_scale, k_scale, v_scale on the device: one element each (per tensor) or
// B * H elements each (per head, b-major, as [B, H]). bf16 out. The default variant is the
// highest.
Tensor attention_fp8(const Tensor& q, const Tensor& k, const Tensor& v, const Tensor& q_scale,
                     const Tensor& k_scale, const Tensor& v_scale, bool causal, int variant) {
    check_cuda_contig(q, "q");
    check_cuda_contig(k, "k");
    check_cuda_contig(v, "v");
    TORCH_CHECK(q.scalar_type() == at::kFloat8_e4m3fn && k.scalar_type() == at::kFloat8_e4m3fn &&
                    v.scalar_type() == at::kFloat8_e4m3fn,
                "attention_fp8 expects float8_e4m3fn q, k, v");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.dim() == 4,
                "attention_fp8 expects [B, H, S, D] tensors");
    TORCH_CHECK(k.sizes() == v.sizes(), "k and v must have the same shape");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(3) == k.size(3),
                "q and k must agree on B and D (q is ", q.sizes(), ", k is ", k.sizes(), ")");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "q must have a multiple of k's heads (GQA), got ",
                q.size(1), " query heads and ", k.size(1), " k/v heads");
    TORCH_CHECK(q.size(3) == 64 || q.size(3) == 128, "attention_fp8 supports D = 64 or 128, got ",
                q.size(3));
    TORCH_CHECK(
        q.size(2) <= INT32_MAX && k.size(2) <= INT32_MAX && q.size(0) * q.size(1) <= INT32_MAX,
        "attention_fp8 dims too large for int32");
    TORCH_CHECK(aligned16(q) && aligned16(k) && aligned16(v),
                "attention_fp8 needs 16-byte aligned q, k, v storage");
    const bool per_head = q_scale.numel() != 1 || k_scale.numel() != 1 || v_scale.numel() != 1;
    const std::pair<const Tensor*, const char*> scales[3] = {
        {&q_scale, "q_scale"}, {&k_scale, "k_scale"}, {&v_scale, "v_scale"}};
    const int64_t heads[3] = {q.size(0) * q.size(1), k.size(0) * k.size(1), k.size(0) * k.size(1)};
    for (int i = 0; i < 3; ++i) {
        const Tensor& sc = *scales[i].first;
        TORCH_CHECK(sc.is_cuda() && sc.scalar_type() == at::kFloat && sc.is_contiguous(),
                    scales[i].second, " must be a contiguous float32 CUDA tensor");
        TORCH_CHECK(sc.device() == q.device(), scales[i].second, " must be on q's device");
        TORCH_CHECK(sc.numel() == (per_head ? heads[i] : 1), scales[i].second, " must have ",
                    per_head ? heads[i] : 1,
                    " elements (the scales are all per tensor or all per (b, head)), got ",
                    sc.numel());
    }
    const c10::cuda::CUDAGuard guard(q.device());
    Tensor out = at::empty(q.sizes(), q.options().dtype(at::kBFloat16));
    int var = resolve_variant(variant, spark::attention_fp8_num_variants());
    while (variant < 0 && var > 0 &&
           !spark::attention_fp8_supports(static_cast<int>(q.size(2)), static_cast<int>(k.size(2)),
                                          static_cast<int>(q.size(3)), var))
        --var;
    spark::attention_fp8(reinterpret_cast<const __nv_fp8_e4m3*>(q.data_ptr()),
                         reinterpret_cast<const __nv_fp8_e4m3*>(k.data_ptr()),
                         reinterpret_cast<const __nv_fp8_e4m3*>(v.data_ptr()), bf16_ptr_mut(out),
                         static_cast<int>(q.size(0)), static_cast<int>(q.size(1)),
                         static_cast<int>(k.size(1)), static_cast<int>(q.size(2)),
                         static_cast<int>(k.size(2)), static_cast<int>(q.size(3)),
                         q_scale.data_ptr<float>(), k_scale.data_ptr<float>(),
                         v_scale.data_ptr<float>(), per_head, causal, var, current_stream(q));
    return out;
}

// q = rope(qkv[..., :H_q D]) as [B, H_q, S, D]; the caches [B, H_kv, cap, D] get rope(k) and
// v at positions pos0..pos0+S-1, in place. cos and sin are [>= pos0 + S, D] fp32 tables in
// the rotate-half layout.
Tensor rope_append_(const Tensor& qkv, const Tensor& cos, const Tensor& sin, Tensor k_cache,
                    Tensor v_cache, int64_t pos0, int64_t H_q, int64_t H_kv) {
    check_cuda_contig(qkv, "qkv");
    check_cuda_contig(cos, "cos");
    check_cuda_contig(sin, "sin");
    check_cuda_contig(k_cache, "k_cache");
    check_cuda_contig(v_cache, "v_cache");
    TORCH_CHECK(qkv.scalar_type() == at::kBFloat16 && k_cache.scalar_type() == at::kBFloat16 &&
                    v_cache.scalar_type() == at::kBFloat16,
                "rope_append_ expects bfloat16 qkv and caches");
    TORCH_CHECK(cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat,
                "rope_append_ expects float32 cos and sin tables");
    TORCH_CHECK(qkv.dim() == 3, "qkv must be [B, S, (H_q + 2 H_kv) * D]");
    TORCH_CHECK(k_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
                "k_cache and v_cache must be [B, H_kv, cap, D] with the same shape");
    TORCH_CHECK(cos.dim() == 2 && cos.sizes() == sin.sizes(), "cos and sin must be [P, D]");
    const int64_t B = qkv.size(0), S = qkv.size(1), D = k_cache.size(3), cap = k_cache.size(2);
    TORCH_CHECK(H_q >= 1 && H_kv >= 1 && H_q % H_kv == 0, "H_q must be a multiple of H_kv");
    TORCH_CHECK(qkv.size(2) == (H_q + 2 * H_kv) * D,
                "qkv's last dim must be (H_q + 2 H_kv) * D = ", (H_q + 2 * H_kv) * D, ", got ",
                qkv.size(2));
    TORCH_CHECK(k_cache.size(0) == B && k_cache.size(1) == H_kv,
                "caches must be [B, H_kv, cap, D]");
    TORCH_CHECK(cos.size(1) == D, "cos and sin must be [P, D] with D = ", D);
    TORCH_CHECK(D % 16 == 0, "D must be a multiple of 16");
    TORCH_CHECK(pos0 >= 0 && pos0 + S <= cap, "the cache (capacity ", cap, ") cannot take ", S,
                " tokens at position ", pos0);
    TORCH_CHECK(cos.size(0) >= pos0 + S, "cos and sin cover ", cos.size(0), " positions, need ",
                pos0 + S);
    TORCH_CHECK(B * S * (H_q + 2 * H_kv) * D < (int64_t{1} << 40), "rope_append_ dims too large");
    const c10::cuda::CUDAGuard guard(qkv.device());
    Tensor q = at::empty({B, H_q, S, D}, qkv.options());
    spark::rope_append_bf16(bf16_ptr(qkv), cos.data_ptr<float>(), sin.data_ptr<float>(),
                            bf16_ptr_mut(q), bf16_ptr_mut(k_cache), bf16_ptr_mut(v_cache),
                            static_cast<int>(B), static_cast<int>(S), static_cast<int>(H_q),
                            static_cast<int>(H_kv), static_cast<int>(D), static_cast<int>(pos0),
                            static_cast<int>(cap), current_stream(qkv));
    return q;
}

// ---- the paged K/V cache (docs/design/serving.md) ---------------------------------------

void check_int32(const Tensor& t, const char* name) {
    check_cuda_contig(t, name);
    TORCH_CHECK(t.scalar_type() == at::kInt, name, " must be int32");
}

// The caches [num_pages, H_kv, page, D], both bf16 or both float8_e4m3fn, the same shape,
// page a power of two. Returns true for e4m3, whose per-head scales (k_scale, v_scale:
// float32 [H_kv], the layer's) must then be given; a bf16 cache takes none.
bool check_cache(const Tensor& k_cache, const Tensor& v_cache, const std::optional<Tensor>& k_scale,
                 const std::optional<Tensor>& v_scale) {
    check_cuda_contig(k_cache, "k_cache");
    check_cuda_contig(v_cache, "v_cache");
    const bool fp8 = k_cache.scalar_type() == at::kFloat8_e4m3fn;
    TORCH_CHECK((fp8 || k_cache.scalar_type() == at::kBFloat16) &&
                    v_cache.scalar_type() == k_cache.scalar_type(),
                "k_cache and v_cache must both be bfloat16 or both float8_e4m3fn");
    TORCH_CHECK(k_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
                "k_cache and v_cache must be [num_pages, H_kv, page, D] with the same shape");
    const int64_t page = k_cache.size(2);
    TORCH_CHECK(page >= 1 && (page & (page - 1)) == 0,
                "the page size (k_cache.size(2)) must be a "
                "power of two, got ",
                page);
    TORCH_CHECK(aligned16(k_cache) && aligned16(v_cache), "the caches need 16-byte alignment");
    if (!fp8) {
        TORCH_CHECK(!k_scale.has_value() && !v_scale.has_value(),
                    "k_scale and v_scale are for float8_e4m3fn caches only");
        return false;
    }
    TORCH_CHECK(k_scale.has_value() && v_scale.has_value(),
                "float8_e4m3fn caches need k_scale and v_scale");
    for (const Tensor* t : {&*k_scale, &*v_scale}) {
        check_cuda_contig(*t, "k_scale and v_scale");
        TORCH_CHECK(t->scalar_type() == at::kFloat && t->numel() == k_cache.size(1),
                    "k_scale and v_scale must be float32 with one element per kv head (",
                    k_cache.size(1), ")");
        TORCH_CHECK(t->device() == k_cache.device(), "the scales must be on the caches' device");
    }
    return true;
}

const __nv_fp8_e4m3* fp8_ptr(const Tensor& t) {
    return reinterpret_cast<const __nv_fp8_e4m3*>(t.data_ptr());
}

// q = rope(qkv[:, :H_q D]) as [T, H_q, D]; token t's rope(k) and v go to slot slots[t] of the
// paged caches (skipped when slots[t] < 0), rotated at position positions[t].
// With float8_e4m3fn caches k and v are stored as e4m3(x / scale) of their kv head.
Tensor rope_append_paged_(const Tensor& qkv, const Tensor& cos, const Tensor& sin,
                          const Tensor& positions, const Tensor& slots, Tensor k_cache,
                          Tensor v_cache, int64_t H_q, int64_t H_kv,
                          const std::optional<Tensor>& k_scale,
                          const std::optional<Tensor>& v_scale) {
    check_cuda_contig(qkv, "qkv");
    check_cuda_contig(cos, "cos");
    check_cuda_contig(sin, "sin");
    check_int32(positions, "positions");
    check_int32(slots, "slots");
    const bool fp8 = check_cache(k_cache, v_cache, k_scale, v_scale);
    TORCH_CHECK(qkv.scalar_type() == at::kBFloat16, "rope_append_paged_ expects bfloat16 qkv");
    TORCH_CHECK(cos.scalar_type() == at::kFloat && sin.scalar_type() == at::kFloat,
                "rope_append_paged_ expects float32 cos and sin tables");
    TORCH_CHECK(qkv.dim() == 2, "qkv must be [T, (H_q + 2 H_kv) * D]");
    const int64_t T = qkv.size(0), D = k_cache.size(3);
    TORCH_CHECK(H_q >= 1 && H_kv >= 1 && H_q % H_kv == 0, "H_q must be a multiple of H_kv");
    TORCH_CHECK(k_cache.size(1) == H_kv, "the caches must have H_kv heads");
    TORCH_CHECK(qkv.size(1) == (H_q + 2 * H_kv) * D,
                "qkv's last dim must be (H_q + 2 H_kv) * D = ", (H_q + 2 * H_kv) * D, ", got ",
                qkv.size(1));
    TORCH_CHECK(positions.numel() == T && slots.numel() == T,
                "positions and slots need one entry per token");
    TORCH_CHECK(cos.dim() == 2 && cos.sizes() == sin.sizes() && cos.size(1) == D,
                "cos and sin must be [P, D]");
    TORCH_CHECK(D % 16 == 0, "D must be a multiple of 16");
    TORCH_CHECK(T * (H_q + 2 * H_kv) * D < (int64_t{1} << 40), "rope_append_paged_ too large");
    const c10::cuda::CUDAGuard guard(qkv.device());
    Tensor q = at::empty({T, H_q, D}, qkv.options());
    if (fp8) {
        spark::rope_append_paged_fp8(
            bf16_ptr(qkv), cos.data_ptr<float>(), sin.data_ptr<float>(), positions.data_ptr<int>(),
            slots.data_ptr<int>(), bf16_ptr_mut(q),
            reinterpret_cast<__nv_fp8_e4m3*>(k_cache.data_ptr()),
            reinterpret_cast<__nv_fp8_e4m3*>(v_cache.data_ptr()), k_scale->data_ptr<float>(),
            v_scale->data_ptr<float>(), static_cast<int>(T), static_cast<int>(H_q),
            static_cast<int>(H_kv), static_cast<int>(D), static_cast<int>(k_cache.size(2)),
            current_stream(qkv));
        return q;
    }
    spark::rope_append_paged_bf16(
        bf16_ptr(qkv), cos.data_ptr<float>(), sin.data_ptr<float>(), positions.data_ptr<int>(),
        slots.data_ptr<int>(), bf16_ptr_mut(q), bf16_ptr_mut(k_cache), bf16_ptr_mut(v_cache),
        static_cast<int>(T), static_cast<int>(H_q), static_cast<int>(H_kv), static_cast<int>(D),
        static_cast<int>(k_cache.size(2)), current_stream(qkv));
    return q;
}

// o [B, H_q, D] = attention of each sequence's one query token over its seq_lens[b] keys in
// the paged cache (bf16, or float8_e4m3fn with its scales).
Tensor paged_decode(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                    const Tensor& block_table, const Tensor& seq_lens, int variant,
                    const std::optional<Tensor>& k_scale, const std::optional<Tensor>& v_scale) {
    check_cuda_contig(q, "q");
    const bool fp8 = check_cache(k_cache, v_cache, k_scale, v_scale);
    check_int32(block_table, "block_table");
    check_int32(seq_lens, "seq_lens");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && q.dim() == 3,
                "paged_decode expects q as [B, H_q, D] bfloat16");
    const int64_t B = q.size(0), H_q = q.size(1), D = q.size(2), H_kv = k_cache.size(1);
    TORCH_CHECK(k_cache.size(3) == D, "q and the caches must agree on D");
    TORCH_CHECK(H_q % H_kv == 0, "q must have a multiple of the caches' heads");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == B,
                "block_table must be [B, max_pages]");
    TORCH_CHECK(seq_lens.numel() == B, "seq_lens must have B entries");
    TORCH_CHECK(aligned16(q), "paged_decode needs 16-byte aligned q");
    const c10::cuda::CUDAGuard guard(q.device());
    Tensor out = at::empty_like(q);
    const int var = variant < 0 ? spark::paged_decode_default_variant()
                                : resolve_variant(variant, spark::paged_decode_num_variants());
    if (fp8) {
        spark::paged_decode_fp8(
            bf16_ptr(q), fp8_ptr(k_cache), fp8_ptr(v_cache), k_scale->data_ptr<float>(),
            v_scale->data_ptr<float>(), block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
            bf16_ptr_mut(out), static_cast<int>(B), static_cast<int>(H_q), static_cast<int>(H_kv),
            static_cast<int>(D), static_cast<int>(k_cache.size(2)),
            static_cast<int>(block_table.size(1)), var, current_stream(q));
        return out;
    }
    spark::paged_decode_bf16(
        bf16_ptr(q), bf16_ptr(k_cache), bf16_ptr(v_cache), block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(), bf16_ptr_mut(out), static_cast<int>(B), static_cast<int>(H_q),
        static_cast<int>(H_kv), static_cast<int>(D), static_cast<int>(k_cache.size(2)),
        static_cast<int>(block_table.size(1)), var, current_stream(q));
    return out;
}

// o [T, H_q, D] = attention of the packed new tokens (sequence b's at rows
// cu_seqlens_q[b] .. cu_seqlens_q[b+1]-1) over their sequences' keys in the paged cache.
Tensor attention_varlen(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                        const Tensor& cu_seqlens_q, const Tensor& seq_lens,
                        const Tensor& block_table, bool causal, int variant,
                        const std::optional<Tensor>& k_scale,
                        const std::optional<Tensor>& v_scale) {
    check_cuda_contig(q, "q");
    const bool fp8 = check_cache(k_cache, v_cache, k_scale, v_scale);
    check_int32(cu_seqlens_q, "cu_seqlens_q");
    check_int32(seq_lens, "seq_lens");
    check_int32(block_table, "block_table");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && q.dim() == 3,
                "attention_varlen expects q as [T, H_q, D] bfloat16");
    const int64_t T = q.size(0), H_q = q.size(1), D = q.size(2), H_kv = k_cache.size(1);
    const int64_t B = seq_lens.numel();
    TORCH_CHECK(k_cache.size(3) == D, "q and the caches must agree on D");
    TORCH_CHECK(H_q % H_kv == 0, "q must have a multiple of the caches' heads");
    TORCH_CHECK(cu_seqlens_q.numel() == B + 1, "cu_seqlens_q must have B + 1 entries");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == B,
                "block_table must be [B, max_pages]");
    TORCH_CHECK(aligned16(q), "attention_varlen needs 16-byte aligned q");
    TORCH_CHECK(T * H_q <= INT32_MAX, "attention_varlen dims too large for int32");
    const c10::cuda::CUDAGuard guard(q.device());
    Tensor out = at::empty_like(q);
    const int var = variant < 0 ? spark::attention_varlen_default_variant()
                                : resolve_variant(variant, spark::attention_varlen_num_variants());
    if (fp8) {
        spark::attention_varlen_fp8(
            bf16_ptr(q), fp8_ptr(k_cache), fp8_ptr(v_cache), k_scale->data_ptr<float>(),
            v_scale->data_ptr<float>(), cu_seqlens_q.data_ptr<int>(), seq_lens.data_ptr<int>(),
            block_table.data_ptr<int>(), bf16_ptr_mut(out), static_cast<int>(B),
            static_cast<int>(T), static_cast<int>(H_q), static_cast<int>(H_kv), static_cast<int>(D),
            static_cast<int>(k_cache.size(2)), static_cast<int>(block_table.size(1)), causal, var,
            current_stream(q));
        return out;
    }
    spark::attention_varlen_bf16(
        bf16_ptr(q), bf16_ptr(k_cache), bf16_ptr(v_cache), cu_seqlens_q.data_ptr<int>(),
        seq_lens.data_ptr<int>(), block_table.data_ptr<int>(), bf16_ptr_mut(out),
        static_cast<int>(B), static_cast<int>(T), static_cast<int>(H_q), static_cast<int>(H_kv),
        static_cast<int>(D), static_cast<int>(k_cache.size(2)),
        static_cast<int>(block_table.size(1)), causal, var, current_stream(q));
    return out;
}

// ---- sliding-window attention and attention sinks (docs/design/attention.md) ------------
// The three attention ops above with a window of `window` keys (0: none; causal, aligned to the
// last key) and per-head sink logits (float32 [H_q] on q's device, or none). The inputs are
// checked by the same rules as the plain ops.

int window_arg(int64_t window) {
    TORCH_CHECK(window >= 0 && window <= INT32_MAX, "window must be >= 0, got ", window);
    return static_cast<int>(window);
}

const float* sinks_ptr(const std::optional<Tensor>& sinks, const Tensor& q, int64_t H_q) {
    if (!sinks.has_value()) return nullptr;
    const Tensor& s = *sinks;
    check_cuda_contig(s, "sinks");
    TORCH_CHECK(s.scalar_type() == at::kFloat && s.dim() == 1 && s.size(0) == H_q,
                "sinks must be float32 [H_q] with H_q = ", H_q, ", got ", s.sizes(), " ",
                s.scalar_type());
    TORCH_CHECK(s.device() == q.device(), "sinks must be on q's device");
    return s.data_ptr<float>();
}

Tensor attention_window(const Tensor& q, const Tensor& k, const Tensor& v, bool causal,
                        int64_t window, const std::optional<Tensor>& sinks, int variant) {
    check_attention_inputs(q, k, v, /*cache_kv=*/true);
    const int w = window_arg(window);
    TORCH_CHECK(w == 0 || q.size(2) <= k.size(2),
                "a window needs S_q <= S_kv (the queries are the last S_q positions), got S_q = ",
                q.size(2), ", S_kv = ", k.size(2));
    const float* sp = sinks_ptr(sinks, q, q.size(1));
    const c10::cuda::CUDAGuard guard(q.device());
    Tensor out = at::empty_like(q);
    spark::attention_window_bf16(bf16_ptr(q), bf16_ptr(k), bf16_ptr(v), bf16_ptr_mut(out),
                                 static_cast<int>(q.size(0)), static_cast<int>(q.size(1)),
                                 static_cast<int>(k.size(1)), static_cast<int>(q.size(2)),
                                 static_cast<int>(k.size(2)), static_cast<int>(q.size(3)), causal,
                                 w, sp, attention_variant(q, k, variant), current_stream(q),
                                 nullptr, static_cast<int>(kv_capacity(k, v)));
    return out;
}

Tensor paged_decode_window(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                           const Tensor& block_table, const Tensor& seq_lens, int64_t window,
                           const std::optional<Tensor>& sinks, int variant,
                           const std::optional<Tensor>& k_scale,
                           const std::optional<Tensor>& v_scale) {
    check_cuda_contig(q, "q");
    const bool fp8 = check_cache(k_cache, v_cache, k_scale, v_scale);
    check_int32(block_table, "block_table");
    check_int32(seq_lens, "seq_lens");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && q.dim() == 3,
                "paged_decode expects q as [B, H_q, D] bfloat16");
    const int64_t B = q.size(0), H_q = q.size(1), D = q.size(2), H_kv = k_cache.size(1);
    TORCH_CHECK(k_cache.size(3) == D, "q and the caches must agree on D");
    TORCH_CHECK(H_q % H_kv == 0, "q must have a multiple of the caches' heads");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == B,
                "block_table must be [B, max_pages]");
    TORCH_CHECK(seq_lens.numel() == B, "seq_lens must have B entries");
    TORCH_CHECK(aligned16(q), "paged_decode needs 16-byte aligned q");
    const int w = window_arg(window);
    const float* sp = sinks_ptr(sinks, q, H_q);
    const c10::cuda::CUDAGuard guard(q.device());
    Tensor out = at::empty_like(q);
    // -1 is the default rung, not the top one: variant 2 (batch invariant) has no window.
    const int var = variant < 0 ? spark::paged_decode_default_variant()
                                : resolve_variant(variant, spark::paged_decode_num_variants());
    if (fp8) {
        spark::paged_decode_window_fp8(
            bf16_ptr(q), fp8_ptr(k_cache), fp8_ptr(v_cache), k_scale->data_ptr<float>(),
            v_scale->data_ptr<float>(), block_table.data_ptr<int>(), seq_lens.data_ptr<int>(),
            bf16_ptr_mut(out), static_cast<int>(B), static_cast<int>(H_q), static_cast<int>(H_kv),
            static_cast<int>(D), static_cast<int>(k_cache.size(2)),
            static_cast<int>(block_table.size(1)), w, sp, var, current_stream(q));
        return out;
    }
    spark::paged_decode_window_bf16(
        bf16_ptr(q), bf16_ptr(k_cache), bf16_ptr(v_cache), block_table.data_ptr<int>(),
        seq_lens.data_ptr<int>(), bf16_ptr_mut(out), static_cast<int>(B), static_cast<int>(H_q),
        static_cast<int>(H_kv), static_cast<int>(D), static_cast<int>(k_cache.size(2)),
        static_cast<int>(block_table.size(1)), w, sp, var, current_stream(q));
    return out;
}

Tensor attention_varlen_window(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                               const Tensor& cu_seqlens_q, const Tensor& seq_lens,
                               const Tensor& block_table, bool causal, int64_t window,
                               const std::optional<Tensor>& sinks, int variant,
                               const std::optional<Tensor>& k_scale,
                               const std::optional<Tensor>& v_scale) {
    check_cuda_contig(q, "q");
    const bool fp8 = check_cache(k_cache, v_cache, k_scale, v_scale);
    check_int32(cu_seqlens_q, "cu_seqlens_q");
    check_int32(seq_lens, "seq_lens");
    check_int32(block_table, "block_table");
    TORCH_CHECK(q.scalar_type() == at::kBFloat16 && q.dim() == 3,
                "attention_varlen expects q as [T, H_q, D] bfloat16");
    const int64_t T = q.size(0), H_q = q.size(1), D = q.size(2), H_kv = k_cache.size(1);
    const int64_t B = seq_lens.numel();
    TORCH_CHECK(k_cache.size(3) == D, "q and the caches must agree on D");
    TORCH_CHECK(H_q % H_kv == 0, "q must have a multiple of the caches' heads");
    TORCH_CHECK(cu_seqlens_q.numel() == B + 1, "cu_seqlens_q must have B + 1 entries");
    TORCH_CHECK(block_table.dim() == 2 && block_table.size(0) == B,
                "block_table must be [B, max_pages]");
    TORCH_CHECK(aligned16(q), "attention_varlen needs 16-byte aligned q");
    TORCH_CHECK(T * H_q <= INT32_MAX, "attention_varlen dims too large for int32");
    const int w = window_arg(window);
    const float* sp = sinks_ptr(sinks, q, H_q);
    const c10::cuda::CUDAGuard guard(q.device());
    Tensor out = at::empty_like(q);
    // -1 is the default rung, not the top one: variant 2 (batch invariant) has no window.
    const int var = variant < 0 ? spark::attention_varlen_default_variant()
                                : resolve_variant(variant, spark::attention_varlen_num_variants());
    if (fp8) {
        spark::attention_varlen_window_fp8(
            bf16_ptr(q), fp8_ptr(k_cache), fp8_ptr(v_cache), k_scale->data_ptr<float>(),
            v_scale->data_ptr<float>(), cu_seqlens_q.data_ptr<int>(), seq_lens.data_ptr<int>(),
            block_table.data_ptr<int>(), bf16_ptr_mut(out), static_cast<int>(B),
            static_cast<int>(T), static_cast<int>(H_q), static_cast<int>(H_kv), static_cast<int>(D),
            static_cast<int>(k_cache.size(2)), static_cast<int>(block_table.size(1)), causal, w, sp,
            var, current_stream(q));
        return out;
    }
    spark::attention_varlen_window_bf16(
        bf16_ptr(q), bf16_ptr(k_cache), bf16_ptr(v_cache), cu_seqlens_q.data_ptr<int>(),
        seq_lens.data_ptr<int>(), block_table.data_ptr<int>(), bf16_ptr_mut(out),
        static_cast<int>(B), static_cast<int>(T), static_cast<int>(H_q), static_cast<int>(H_kv),
        static_cast<int>(D), static_cast<int>(k_cache.size(2)),
        static_cast<int>(block_table.size(1)), causal, w, sp, var, current_stream(q));
    return out;
}

// ---------------------------------------------------------------------------
// Launch configuration (src/kernels/launch_config.cu): what the decode kernels decided
// ---------------------------------------------------------------------------
int decode_device_of(const std::string& device) {
    if (device.empty() || device == "current") return spark::DECODE_DEVICE_CURRENT;
    if (device == "gb10") return spark::DECODE_DEVICE_GB10;
    if (device == "rtx5090") return spark::DECODE_DEVICE_RTX5090;
    throw std::invalid_argument("device must be '', 'gb10' or 'rtx5090', got '" + device + "'");
}

py::dict device_shape() {
    py::dict d;
    d["sm_count"] = spark::device_shape_sm_count();
    d["l2_bytes"] = spark::device_shape_l2_bytes();
    d["spec_gbps"] = spark::device_shape_spec_gbps();
    d["bytes_in_flight"] = spark::device_shape_bytes_in_flight();
    d["forced"] = spark::device_shape_forced();
    return d;
}

int split_k_for(int64_t weight_bytes, int tiles, int max_split, int64_t per_cta_in_flight,
                int slots, const std::string& device) {
    TORCH_CHECK(tiles >= 1 && max_split >= 1 && per_cta_in_flight >= 1 && slots >= 0,
                "split_k_for: tiles, max_split and per_cta_in_flight must be >= 1, slots >= 0");
    return spark::decode_split_k(weight_bytes, tiles, max_split, per_cta_in_flight, slots,
                                 decode_device_of(device));
}

int need_ctas(int64_t per_cta_in_flight, const std::string& device) {
    TORCH_CHECK(per_cta_in_flight >= 1, "need_ctas: per_cta_in_flight must be >= 1");
    return spark::decode_need_ctas(per_cta_in_flight, decode_device_of(device));
}

bool fits_l2(int64_t bytes, const std::string& device) {
    return spark::decode_fits_l2(bytes, decode_device_of(device));
}

int num_variants(const std::string& name) {
    if (name == "rmsnorm") return spark::rmsnorm_num_variants();
    if (name == "swiglu") return spark::swiglu_num_variants();
    if (name == "softmax") return spark::softmax_num_variants();
    if (name == "sample") return spark::sample_num_variants();
    if (name == "sgemm") return spark::sgemm_num_variants();
    if (name == "hgemm") return spark::hgemm_num_variants();
    if (name == "fp8gemm") return spark::fp8gemm_num_variants();
    if (name == "fp4gemm") return spark::fp4gemm_num_variants();
    if (name == "attention") return spark::attention_num_variants();
    if (name == "attention_fp8") return spark::attention_fp8_num_variants();
    if (name == "attention_bwd") return spark::attention_bwd_num_variants();
    if (name == "w4gemm") return spark::w4gemm_num_variants();
    if (name == "w1gemm") return spark::w1gemm_num_variants();
    if (name == "bandwidth") return spark::bandwidth_num_variants();
    if (name == "paged_decode") return spark::paged_decode_num_variants();
    if (name == "attention_varlen") return spark::attention_varlen_num_variants();
    throw std::invalid_argument("unknown kernel: " + name);
}

// The fused decode-step over the routed experts (w1moe_decode_bf16): x [T, K] bf16,
// expert_ids int32 [T, topk], expert_weights bf16 [T, topk], the gate|up experts (packed
// [E, 2I, K * bits / 32], scales [E, K/128, 2I]) and the down experts (packed [E, K, I * bits /
// 32], scales [E, I/128, K]) -> y [T, K] bf16. T <= 8.
Tensor w1moe_decode(const Tensor& x, const Tensor& expert_ids, const Tensor& expert_weights,
                    const Tensor& packed_gu, const Tensor& scales_gu, const Tensor& packed_dn,
                    const Tensor& scales_dn, int bits, int variant) {
    check_cuda_contig(x, "x");
    check_cuda_contig(expert_ids, "expert_ids");
    check_cuda_contig(expert_weights, "expert_weights");
    check_w1_bits(bits, "w1moe_decode");
    TORCH_CHECK(x.scalar_type() == at::kBFloat16 && x.dim() == 2,
                "w1moe_decode expects a 2-D bfloat16 x [T, K]");
    TORCH_CHECK(
        packed_gu.dim() == 3 && scales_gu.dim() == 3 && packed_dn.dim() == 3 &&
            scales_dn.dim() == 3,
        "w1moe_decode expects stacked expert weights [E, N, K * bits / 32] / [E, K/128, N]");
    const int64_t T = x.size(0), K = x.size(1);
    const int64_t E = packed_gu.size(0);
    TORCH_CHECK(scales_gu.size(0) == E && packed_dn.size(0) == E && scales_dn.size(0) == E,
                "w1moe_decode: the four expert tensors must hold the same number of experts");
    const int64_t N_gu = check_w1_weight(x, packed_gu, scales_gu, bits, K, "w1moe_decode");
    TORCH_CHECK(N_gu % 2 == 0, "w1moe_decode: the gate|up weight needs an even column count");
    const int64_t I = N_gu / 2;
    TORCH_CHECK(packed_dn.scalar_type() == at::kInt && packed_dn.size(1) == K &&
                    packed_dn.size(2) == I * bits / 32,
                "w1moe_decode: down packed must be int32 [E, K, I * bits / 32] = [", E, ", ", K,
                ", ", I * bits / 32, "], got ", packed_dn.sizes());
    TORCH_CHECK(scales_dn.scalar_type() == at::kBFloat16 &&
                    scales_dn.size(1) == I / spark::kW1GroupSize && scales_dn.size(2) == K,
                "w1moe_decode: down scales must be bfloat16 [E, I/128, K], got ",
                scales_dn.sizes());
    check_cuda_contig(packed_dn, "packed_down");
    check_cuda_contig(scales_dn, "scales_down");
    TORCH_CHECK(
        expert_ids.scalar_type() == at::kInt && expert_ids.dim() == 2 && expert_ids.size(0) == T,
        "w1moe_decode: expert_ids must be int32 [T, topk], got ", expert_ids.scalar_type(), " ",
        expert_ids.sizes());
    const int64_t topk = expert_ids.size(1);
    TORCH_CHECK(expert_weights.scalar_type() == at::kBFloat16 &&
                    expert_weights.sizes() == expert_ids.sizes(),
                "w1moe_decode: expert_weights must be bfloat16 [T, topk]");
    TORCH_CHECK(T >= 1 && T <= 8 && T * topk <= 64,
                "w1moe_decode takes 1 <= T <= 8 tokens and "
                "T * topk <= 64, got T = ",
                T, ", topk = ", topk);
    TORCH_CHECK(K % 256 == 0 && K <= 16384 && I % spark::kW1GroupSize == 0,
                "w1moe_decode needs K % 256 == 0, K <= 16384 and I % 128 == 0, got K = ", K,
                ", I = ", I);
    TORCH_CHECK(expert_ids.device() == x.device() && expert_weights.device() == x.device() &&
                    packed_dn.device() == x.device() && scales_dn.device() == x.device(),
                "w1moe_decode: every tensor must be on x's device");
    TORCH_CHECK(aligned16(packed_dn), "w1moe_decode needs 16-byte aligned down packed storage");
    const c10::cuda::CUDAGuard guard(x.device());
    Tensor y = at::empty({T, K}, x.options());
    const int v = resolve_variant(variant, spark::w1moe_decode_num_variants());
    spark::w1moe_decode_bf16(
        bf16_ptr(x), expert_ids.data_ptr<int32_t>(), bf16_ptr(expert_weights),
        reinterpret_cast<const uint32_t*>(packed_gu.data_ptr<int32_t>()), bf16_ptr(scales_gu),
        reinterpret_cast<const uint32_t*>(packed_dn.data_ptr<int32_t>()), bf16_ptr(scales_dn),
        bf16_ptr_mut(y), static_cast<int>(T), static_cast<int>(topk), static_cast<int>(E),
        static_cast<int>(K), static_cast<int>(I), bits, v, current_stream(x));
    return y;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "spark-kernels: hand-written CUDA kernels for LLM inference on sm_120 / sm_121";
    m.def("rmsnorm", &rmsnorm, "RMSNorm over the last dim", py::arg("x"), py::arg("w"),
          py::arg("eps") = 1e-6, py::arg("variant") = -1);
    m.def("add_rmsnorm_", &add_rmsnorm_, "resid += x; return rmsnorm(resid) * w (bf16, in place)",
          py::arg("x"), py::arg("resid"), py::arg("w"), py::arg("eps") = 1e-6);
    m.def("swiglu", &swiglu,
          "silu(gate) * up (2-D gate and up with contiguous rows may be strided: the halves "
          "of a fused gate|up projection)",
          py::arg("gate"), py::arg("up"), py::arg("variant") = -1);
    m.def("softmax", &softmax, "softmax over the last dim (fp32 math)", py::arg("x"),
          py::arg("variant") = -1);
    m.def("sample", &sample,
          "one token per row of [B, V] bf16 logits with per-row temperature, top_k, top_p, seed "
          "and offset device tensors (offset advanced in place); int64 [B]",
          py::arg("logits"), py::arg("temperature"), py::arg("top_k"), py::arg("top_p"),
          py::arg("seed"), py::arg("offset"), py::arg("variant") = -1);
    m.def("sgemm", &sgemm, "fp32 GEMM: a @ b (variants 6 and 7: tensor cores in TF32 / 3xTF32)",
          py::arg("a"), py::arg("b"), py::arg("variant") = -1);
    m.def("hgemm", &hgemm,
          "bf16 tensor-core GEMM with a fused epilogue: act(a @ b + bias) + residual, or "
          "silu(gate) * up from an interleaved gate/up b (swiglu=True)",
          py::arg("a"), py::arg("b"), py::arg("variant") = -1, py::arg("bias") = py::none(),
          py::arg("act") = py::none(), py::arg("residual") = py::none(), py::arg("swiglu") = false,
          py::arg("out") = py::none());
    m.def("fp8gemm", &fp8gemm,
          "fp8 (e4m3) tensor-core GEMM: scale_a * scale_b * a @ b_t^T, b_t is [N, K], bf16 out; "
          "sfa [M, K/32] and sfb [N, K/32] uint8 ue8m0 add a scale per 32 k (MXFP8)",
          py::arg("a"), py::arg("b_t"), py::arg("scale_a") = py::none(),
          py::arg("scale_b") = py::none(), py::arg("variant") = -1, py::arg("sfa") = py::none(),
          py::arg("sfb") = py::none());
    m.def("fp8gemm_v3_set_route", &fp8gemm_v3_set_route,
          "fp8gemm variant 3's route: -1 the per-shape rule, -2 the route before it, n a forced "
          "TMA tile; returns the previous one",
          py::arg("route"));
    m.def(
        "fp8gemm_v3_route_name",
        [](int M, int N, int K) { return std::string(spark::fp8gemm_v3_route_name(M, N, K)); },
        "what fp8gemm variant 3 runs for an M x N x K problem under the current route",
        py::arg("M"), py::arg("N"), py::arg("K"));
    m.def("fp8gemm_v3_num_tiles", &spark::fp8gemm_v3_num_tiles,
          "number of TMA tiles fp8gemm variant 3 can be forced to");
    m.def("fp4gemm", &fp4gemm,
          "fp4 (e2m1) block-scaled GEMM, NVFP4 or MXFP4: scale_a * scale_b * a @ b_t^T with a "
          "[M, K/2] and b_t [N, K/2] packed e2m1 and block scales in the blocked layout, bf16 out",
          py::arg("a"), py::arg("b_t"), py::arg("sfa"), py::arg("sfb"),
          py::arg("scale_a") = py::none(), py::arg("scale_b") = py::none(),
          py::arg("fmt") = "nvfp4", py::arg("variant") = -1);
    m.def("fp4_quantize", &fp4_quantize,
          "bf16 [rows, K] -> (packed e2m1 [rows, K/2], blocked block scales, per-tensor scale)",
          py::arg("x"), py::arg("fmt") = "nvfp4", py::arg("scale") = py::none());
    m.def("fp8_quantize", &fp8_quantize,
          "bf16 [rows, K] -> (e4m3 [rows, K], power-of-two scales per tensor or row, or MX "
          "e8m0 scales per 32)",
          py::arg("x"), py::arg("mode") = "tensor");
    m.def("w4_quantize", &w4_quantize,
          "round-to-nearest int4 quantization of a [K, N] bf16 weight, one scale per 128 k: "
          "(qweight [K/8, N] int32, scales [K/128, N] bf16, zeros [K/128, N] uint8 or None)",
          py::arg("w"), py::arg("asym") = false);
    m.def("w4_repack", &w4_repack, "qweight [K/8, N] int32 -> the w4gemm layout [N/16, 2K]",
          py::arg("qweight"));
    m.def("w4gemm", &w4gemm,
          "a [M, K] bf16 @ the int4 weight (packed, scales, zeros) -> [M, N] bf16, fp32 accumulate",
          py::arg("a"), py::arg("packed"), py::arg("scales"), py::arg("zeros") = py::none(),
          py::arg("variant") = -1);
    m.def("w1_quantize", &w1_quantize,
          "sign (bits=1) or ternary (bits=2) quantization of a [K, N] bf16 weight, one scale "
          "per 128 k: (packed [N, K * bits / 32] int32, scales [K/128, N] bf16)",
          py::arg("w"), py::arg("bits") = 1);
    m.def("w1gemm", &w1gemm,
          "a [M, K] bf16 @ the sign or ternary weight (packed, scales) -> [M, N] bf16, fp32 "
          "accumulate",
          py::arg("a"), py::arg("packed"), py::arg("scales"), py::arg("bits") = 1,
          py::arg("variant") = -1);
    m.def("w1gemm_moe", &w1gemm_moe,
          "w1gemm grouped over E experts: a [T, K] sorted by expert with offsets int32 [E + 1], "
          "packed [E, N, K * bits / 32], scales [E, K/128, N] -> [T, N] bf16",
          py::arg("a"), py::arg("packed"), py::arg("scales"), py::arg("offsets"),
          py::arg("bits") = 1, py::arg("variant") = -1);
    m.def("grouped_gemm", &grouped_gemm,
          "grouped GEMM for MoE prefill: a [T, K] sorted by expert with offsets int32 [E + 1], "
          "w [E, N, K] -> [T, N] bf16, one launch; bf16, or e4m3 with scale_a [1] and scale_b "
          "[E] or [1]",
          py::arg("a"), py::arg("w"), py::arg("offsets"), py::arg("scale_a") = py::none(),
          py::arg("scale_b") = py::none());
    m.def("ce_logits", &ce_logits, "logits = h @ w.T, h [T, D] and w [V, D] bf16 -> [T, V] bf16",
          py::arg("h"), py::arg("w"), py::arg("out") = py::none());
    m.def("ce_stats", &ce_stats,
          "cross-entropy forward without the logits: (lse, loss) float32 [T] from h, w and "
          "int64 targets",
          py::arg("h"), py::arg("w"), py::arg("targets"), py::arg("ignore_index") = -100);
    m.def("ce_grad", &ce_grad,
          "(softmax(h @ w.T) - one_hot(targets)) * scale as bf16 [T, V], from the forward's lse",
          py::arg("h"), py::arg("w"), py::arg("targets"), py::arg("lse"), py::arg("scale"),
          py::arg("ignore_index") = -100, py::arg("out") = py::none());
    m.def("ce_rows", &ce_rows,
          "cross-entropy rows of bf16 logits: (loss, lse) float32; with scale the logits are "
          "overwritten by (softmax - one_hot) * scale",
          py::arg("logits"), py::arg("targets"), py::arg("ignore_index") = -100,
          py::arg("scale") = py::none());
    m.def("ce_transpose", &ce_transpose,
          "columns [c0, c1) of x [rows, cols] bf16 transposed into out [c1 - c0, ld], zero "
          "past rows",
          py::arg("x"), py::arg("c0"), py::arg("c1"), py::arg("ld"), py::arg("out"));
    m.def("grouped_gemm_swiglu", &grouped_gemm_swiglu,
          "grouped_gemm with the SwiGLU in the epilogue: w [E, 2I, K] with rows alternating gate "
          "and up -> [T, I] bf16",
          py::arg("a"), py::arg("w"), py::arg("offsets"), py::arg("scale_a") = py::none(),
          py::arg("scale_b") = py::none());
    m.def("moe_route", &moe_route,
          "softmax top-k routing of logits [T, E]: (ids [T, k] int32, weights [T, k] fp32, "
          "offsets [E + 1] int32, pos [T, k] int32, the row of expert order)",
          py::arg("logits"), py::arg("top_k"), py::arg("renormalize") = true);
    m.def("moe_dispatch", &moe_dispatch,
          "moe_route plus the rows of x [T, D] in expert order: (xs [T * k, D], ids, weights, "
          "offsets, pos)",
          py::arg("x"), py::arg("logits"), py::arg("top_k"), py::arg("renormalize") = true);
    m.def("moe_combine", &moe_combine,
          "out [T, D] = sum over s of weights[t, s] * y[pos[t, s]], fp32, fixed order",
          py::arg("y"), py::arg("pos"), py::arg("weights"));
    m.def("attention", &attention,
          "softmax(q k^T / sqrt(D)) v over [B, H, S, D] bf16 tensors (k, v may have fewer heads)",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false,
          py::arg("variant") = -1);
    m.def("attention_fp8", &attention_fp8,
          "softmax(q k^T / sqrt(D)) v over [B, H, S, D] float8_e4m3fn tensors with fp32 descale "
          "factors (per tensor or per (b, head)), bf16 out",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("q_scale"), py::arg("k_scale"),
          py::arg("v_scale"), py::arg("causal") = false, py::arg("variant") = -1);
    m.def("attention_fwd", &attention_fwd,
          "attention forward for training: (out, lse), lse the fp32 [B, H_q, S_q] log-sum-exp of "
          "each row of scaled scores",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false,
          py::arg("variant") = -1);
    m.def("attention_bwd", &attention_bwd,
          "attention backward: (dq, dk, dv) from q, k, v, the forward's out and lse and d_out; "
          "dk and dv are summed over each K/V head's query heads",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("out"), py::arg("d_out"),
          py::arg("lse"), py::arg("causal") = false, py::arg("variant") = -1,
          py::arg("deterministic") = false);
    m.def("rope_append_", &rope_append_,
          "RoPE on the q and k columns of a fused qkv projection, k and v appended to the caches "
          "in place; returns q as [B, H_q, S, D]",
          py::arg("qkv"), py::arg("cos"), py::arg("sin"), py::arg("k_cache"), py::arg("v_cache"),
          py::arg("pos0"), py::arg("H_q"), py::arg("H_kv"));
    m.def("rope_append_paged_", &rope_append_paged_,
          "RoPE on the q and k columns of packed tokens, k and v written to their slots of the "
          "paged caches (bf16, or float8_e4m3fn divided by k_scale / v_scale [H_kv]); returns q "
          "as [T, H_q, D]",
          py::arg("qkv"), py::arg("cos"), py::arg("sin"), py::arg("positions"), py::arg("slots"),
          py::arg("k_cache"), py::arg("v_cache"), py::arg("H_q"), py::arg("H_kv"),
          py::arg("k_scale") = py::none(), py::arg("v_scale") = py::none());
    m.def("paged_decode", &paged_decode,
          "one query token per sequence against a paged K/V cache: q [B, H_q, D], block_table "
          "[B, max_pages], seq_lens [B]; float8_e4m3fn caches take k_scale and v_scale [H_kv]",
          py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("block_table"),
          py::arg("seq_lens"), py::arg("variant") = -1, py::arg("k_scale") = py::none(),
          py::arg("v_scale") = py::none());
    m.def("attention_varlen", &attention_varlen,
          "packed prompts (q [T, H_q, D], cu_seqlens_q [B + 1]) against a paged K/V cache, "
          "causal mask aligned bottom-right; float8_e4m3fn caches take k_scale and v_scale",
          py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("cu_seqlens_q"),
          py::arg("seq_lens"), py::arg("block_table"), py::arg("causal") = true,
          py::arg("variant") = -1, py::arg("k_scale") = py::none(),
          py::arg("v_scale") = py::none());
    // ---- sliding-window attention and attention sinks --------------------------------------
    m.def("attention_window", &attention_window,
          "attention with a causal window of `window` keys aligned to the last key (0: none) "
          "and sink logits float32 [H_q] added to every softmax denominator",
          py::arg("q"), py::arg("k"), py::arg("v"), py::arg("causal") = false,
          py::arg("window") = 0, py::arg("sinks") = py::none(), py::arg("variant") = -1);
    m.def("paged_decode_window", &paged_decode_window,
          "paged_decode with each token seeing the last `window` keys of its sequence and sink "
          "logits float32 [H_q]",
          py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("block_table"),
          py::arg("seq_lens"), py::arg("window") = 0, py::arg("sinks") = py::none(),
          py::arg("variant") = -1, py::arg("k_scale") = py::none(),
          py::arg("v_scale") = py::none());
    m.def("attention_varlen_window", &attention_varlen_window,
          "attention_varlen with a causal window of `window` keys and sink logits float32 [H_q]",
          py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("cu_seqlens_q"),
          py::arg("seq_lens"), py::arg("block_table"), py::arg("causal") = true,
          py::arg("window") = 0, py::arg("sinks") = py::none(), py::arg("variant") = -1,
          py::arg("k_scale") = py::none(), py::arg("v_scale") = py::none());
    m.def("num_variants", &num_variants, "number of implementation variants for a kernel",
          py::arg("name"));
    // The CPU-side ops for the GB10's Arm cores (python/csrc/cpu_bindings.cpp, no CUDA).
    void register_cpu_bindings(py::module_&);
    register_cpu_bindings(m);

    // ---- unified memory experiments (docs/design/unified.md) -----------------------------
    // The weight or the array may be a CUDA tensor, a pinned CPU tensor (cudaHostAlloc memory,
    // readable by the GPU in place under unified addressing), or a plain CPU tensor on a device
    // that reports pageableMemoryAccess (HMM / ATS). The launch goes on the current stream of
    // the current CUDA device; a plain CPU tensor on a device without pageable access is
    // refused here rather than faulting in the kernel.
    m.def(
        "unified_device_caps",
        []() {
            const spark::UnifiedDeviceCaps c = spark::unified_device_caps(-1);
            py::dict d;
            d["device"] = c.device;
            d["sm_count"] = c.sm_count;
            d["unified_addressing"] = static_cast<bool>(c.unified_addressing);
            d["integrated"] = static_cast<bool>(c.integrated);
            d["can_map_host_memory"] = static_cast<bool>(c.can_map_host_memory);
            d["pageable_memory_access"] = static_cast<bool>(c.pageable_memory_access);
            d["pageable_memory_access_uses_host_page_tables"] =
                static_cast<bool>(c.pageable_memory_access_uses_host_page_tables);
            d["concurrent_managed_access"] = static_cast<bool>(c.concurrent_managed_access);
            d["direct_managed_mem_access_from_host"] =
                static_cast<bool>(c.direct_managed_mem_access_from_host);
            d["host_native_atomic_supported"] = static_cast<bool>(c.host_native_atomic_supported);
            d["can_use_host_pointer_for_registered_mem"] =
                static_cast<bool>(c.can_use_host_pointer_for_registered_mem);
            return d;
        },
        "cudaDeviceGetAttribute's view of unified memory on the current device, as a dict");
    m.def(
        "unified_read_sum",
        [](const Tensor& x, int variant) {
            TORCH_CHECK(x.is_contiguous() && x.numel() > 0, "x must be contiguous and non-empty");
            TORCH_CHECK(x.scalar_type() == at::kBFloat16, "unified_read_sum expects bfloat16");
            const bool pageable = spark::unified_device_caps(-1).pageable_memory_access != 0;
            TORCH_CHECK(x.is_cuda() || x.is_pinned() || pageable,
                        "x is a pageable CPU tensor and this device has no pageableMemoryAccess; "
                        "pass a CUDA tensor or one made with pin_memory=True");
            const int v = resolve_variant(variant, spark::unified_read_num_variants());
            const at::Device dev =
                x.is_cuda() ? x.device() : at::Device(at::kCUDA, at::cuda::current_device());
            const c10::cuda::CUDAGuard guard(dev);
            Tensor sink = at::empty({spark::unified_read_sink_floats()},
                                    at::TensorOptions().dtype(at::kFloat).device(dev));
            spark::unified_read_bf16(reinterpret_cast<const __nv_bfloat16*>(x.data_ptr()),
                                     sink.data_ptr<float>(), x.numel(), v, current_stream(sink));
            return sink;
        },
        "stream a bf16 tensor (CUDA, pinned CPU, or plain CPU with pageableMemoryAccess) into "
        "per-block fp32 partial sums on the current CUDA device; sum them for the total",
        py::arg("x"), py::arg("variant") = -1);
    m.def(
        "unified_gemv",
        [](const Tensor& a, const Tensor& w, int variant) {
            check_cuda_contig(a, "a");
            TORCH_CHECK(a.scalar_type() == at::kBFloat16 && w.scalar_type() == at::kBFloat16,
                        "unified_gemv expects bfloat16 a and w");
            TORCH_CHECK(a.dim() == 1 || (a.dim() == 2 && a.size(0) == 1),
                        "a must be [K] or [1, K], got ", a.sizes());
            TORCH_CHECK(w.dim() == 2 && w.is_contiguous() && w.numel() > 0,
                        "w must be a contiguous [N, K] (K contiguous)");
            const int64_t K = a.size(-1), N = w.size(0);
            TORCH_CHECK(w.size(1) == K, "w is ", w.sizes(), ", expected [N, ", K, "]");
            TORCH_CHECK(K % 8 == 0 && N <= INT32_MAX && K <= INT32_MAX,
                        "unified_gemv needs K % 8 == 0, got K = ", K);
            TORCH_CHECK(aligned16(a) && aligned16(w), "a and w need 16-byte aligned storage");
            const bool pageable = spark::unified_device_caps(-1).pageable_memory_access != 0;
            TORCH_CHECK(w.is_cuda() || w.is_pinned() || pageable,
                        "w is a pageable CPU tensor and this device has no pageableMemoryAccess; "
                        "pass a CUDA tensor or one made with pin_memory=True");
            TORCH_CHECK(!w.is_cuda() || w.device() == a.device(), "w must be on a's device");
            const c10::cuda::CUDAGuard guard(a.device());
            Tensor y = a.dim() == 1 ? at::empty({N}, a.options()) : at::empty({1, N}, a.options());
            const int v = resolve_variant(variant, spark::unified_gemv_num_variants());
            spark::unified_gemv_bf16(
                bf16_ptr(a), reinterpret_cast<const __nv_bfloat16*>(w.data_ptr()), bf16_ptr_mut(y),
                static_cast<int>(N), static_cast<int>(K), v, current_stream(a));
            return y;
        },
        "decode GEMV: a [K] or [1, K] bf16 on the device times w [N, K] bf16 in any memory the "
        "GPU can read (CUDA, pinned CPU, or plain CPU with pageableMemoryAccess) -> [N] or "
        "[1, N] bf16, fp32 accumulate",
        py::arg("a"), py::arg("w"), py::arg("variant") = -1);
    m.def(
        "unified_num_variants",
        [](const std::string& name) {
            if (name == "unified_read") return spark::unified_read_num_variants();
            if (name == "unified_gemv") return spark::unified_gemv_num_variants();
            throw std::invalid_argument("unknown unified kernel: " + name);
        },
        "number of variants of unified_read or unified_gemv", py::arg("name"));
    m.def("w1moe_decode", &w1moe_decode,
          "the decode step over the routed 1-bit / ternary experts: x [T, K] bf16 (T <= 8), "
          "expert_ids int32 [T, topk], expert_weights bf16 [T, topk], the stacked gate|up and "
          "down experts (packed, scales) -> y [T, K] bf16, the SwiGLU intermediate kept on chip",
          py::arg("x"), py::arg("expert_ids"), py::arg("expert_weights"), py::arg("packed_gate_up"),
          py::arg("scales_gate_up"), py::arg("packed_down"), py::arg("scales_down"),
          py::arg("bits") = 1, py::arg("variant") = -1);
    m.def("w1moe_decode_num_variants", &spark::w1moe_decode_num_variants,
          "number of implementation variants of w1moe_decode");
    m.def("device_shape", &device_shape,
          "what the decode kernels decided about the current device: sm_count, l2_bytes, "
          "spec_gbps, bytes_in_flight, forced (SPARK_FORCE_SMS)");
    m.def("split_k_for", &split_k_for,
          "the decode kernels' K split for `tiles` CTAs over a weight of weight_bytes, each "
          "keeping per_cta_in_flight bytes outstanding, `slots` resident; device '' (current), "
          "'gb10' or 'rtx5090' (the named shapes need no GPU)",
          py::arg("weight_bytes"), py::arg("tiles"), py::arg("max_split"),
          py::arg("per_cta_in_flight"), py::arg("slots"), py::arg("device") = "");
    m.def("need_ctas", &need_ctas,
          "CTAs the bandwidth rule asks for at per_cta_in_flight bytes each, on a device",
          py::arg("per_cta_in_flight"), py::arg("device") = "");
    m.def("fits_l2", &fits_l2, "whether a weight of `bytes` is L2-resident on a device",
          py::arg("bytes"), py::arg("device") = "");
    m.def("decode_force_split_k", &spark::decode_force_split_k,
          "force the K split of every decode kernel that has one (0 = the rule)", py::arg("split"));

    // hgemm variant 6's route and its split-K switch (tests/test_gemm.py, the sweeps).
    m.def(
        "hgemm_v6_route_name",
        [](int M, int N, int K) { return std::string(spark::hgemm_v6_route_name(M, N, K)); },
        "what hgemm variant 6 runs for an M x N x K problem under the current route", py::arg("M"),
        py::arg("N"), py::arg("K"));
    m.def("hgemm_v6_set_split", &spark::hgemm_v6_set_split,
          "hgemm variant 6's split-K: -1 the rule, 0 off, n >= 2 forces n K-slices",
          py::arg("split"));
    m.def("hgemm_v6_split", &spark::hgemm_v6_split, "hgemm variant 6's split-K setting");
}
