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
    int v = resolve_variant(variant, spark::hgemm_num_variants());
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
                        work.defined() ? static_cast<unsigned*>(work.data_ptr()) : nullptr, rows,
                        K, m, current_stream(x));
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

// The checks every attention entry point shares: q = [B, H_q, S_q, D] and k, v =
// [B, H_kv, S_kv, D] contiguous bf16 CUDA tensors, H_q a multiple of H_kv (grouped-query
// attention: query head h reads k/v head h / (H_q / H_kv)), D in {64, 128}, 16-byte aligned.
void check_attention_inputs(const Tensor& q, const Tensor& k, const Tensor& v) {
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
// kernel also writes the natural-log log-sum-exp of every row of scaled scores.
void run_attention(const Tensor& q, const Tensor& k, const Tensor& v, Tensor& out, bool causal,
                   int variant, float* lse) {
    const c10::cuda::CUDAGuard guard(q.device());
    spark::attention_bf16(bf16_ptr(q), bf16_ptr(k), bf16_ptr(v), bf16_ptr_mut(out),
                          static_cast<int>(q.size(0)), static_cast<int>(q.size(1)),
                          static_cast<int>(k.size(1)), static_cast<int>(q.size(2)),
                          static_cast<int>(k.size(2)), static_cast<int>(q.size(3)), causal,
                          attention_variant(q, k, variant), current_stream(q), lse);
}

Tensor attention(const Tensor& q, const Tensor& k, const Tensor& v, bool causal, int variant) {
    check_attention_inputs(q, k, v);
    Tensor out = at::empty_like(q);
    run_attention(q, k, v, out, causal, variant, nullptr);
    return out;
}

// The forward for training: (O, lse) with lse fp32 [B, H_q, S_q], what attention_bwd takes.
std::tuple<Tensor, Tensor> attention_fwd(const Tensor& q, const Tensor& k, const Tensor& v,
                                         bool causal, int variant) {
    check_attention_inputs(q, k, v);
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

// The caches [num_pages, H_kv, page, D] bf16, the same shape, page a power of two.
void check_cache(const Tensor& k_cache, const Tensor& v_cache) {
    check_cuda_contig(k_cache, "k_cache");
    check_cuda_contig(v_cache, "v_cache");
    TORCH_CHECK(k_cache.scalar_type() == at::kBFloat16 && v_cache.scalar_type() == at::kBFloat16,
                "k_cache and v_cache must be bfloat16");
    TORCH_CHECK(k_cache.dim() == 4 && k_cache.sizes() == v_cache.sizes(),
                "k_cache and v_cache must be [num_pages, H_kv, page, D] with the same shape");
    const int64_t page = k_cache.size(2);
    TORCH_CHECK(page >= 1 && (page & (page - 1)) == 0,
                "the page size (k_cache.size(2)) must be a "
                "power of two, got ",
                page);
    TORCH_CHECK(aligned16(k_cache) && aligned16(v_cache), "the caches need 16-byte alignment");
}

// q = rope(qkv[:, :H_q D]) as [T, H_q, D]; token t's rope(k) and v go to slot slots[t] of the
// paged caches (skipped when slots[t] < 0), rotated at position positions[t].
Tensor rope_append_paged_(const Tensor& qkv, const Tensor& cos, const Tensor& sin,
                          const Tensor& positions, const Tensor& slots, Tensor k_cache,
                          Tensor v_cache, int64_t H_q, int64_t H_kv) {
    check_cuda_contig(qkv, "qkv");
    check_cuda_contig(cos, "cos");
    check_cuda_contig(sin, "sin");
    check_int32(positions, "positions");
    check_int32(slots, "slots");
    check_cache(k_cache, v_cache);
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
    spark::rope_append_paged_bf16(
        bf16_ptr(qkv), cos.data_ptr<float>(), sin.data_ptr<float>(), positions.data_ptr<int>(),
        slots.data_ptr<int>(), bf16_ptr_mut(q), bf16_ptr_mut(k_cache), bf16_ptr_mut(v_cache),
        static_cast<int>(T), static_cast<int>(H_q), static_cast<int>(H_kv), static_cast<int>(D),
        static_cast<int>(k_cache.size(2)), current_stream(qkv));
    return q;
}

// o [B, H_q, D] = attention of each sequence's one query token over its seq_lens[b] keys in
// the paged cache.
Tensor paged_decode(const Tensor& q, const Tensor& k_cache, const Tensor& v_cache,
                    const Tensor& block_table, const Tensor& seq_lens, int variant) {
    check_cuda_contig(q, "q");
    check_cache(k_cache, v_cache);
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
    const int var = resolve_variant(variant, spark::paged_decode_num_variants());
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
                        const Tensor& block_table, bool causal, int variant) {
    check_cuda_contig(q, "q");
    check_cache(k_cache, v_cache);
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
    const int var = resolve_variant(variant, spark::attention_varlen_num_variants());
    spark::attention_varlen_bf16(
        bf16_ptr(q), bf16_ptr(k_cache), bf16_ptr(v_cache), cu_seqlens_q.data_ptr<int>(),
        seq_lens.data_ptr<int>(), block_table.data_ptr<int>(), bf16_ptr_mut(out),
        static_cast<int>(B), static_cast<int>(T), static_cast<int>(H_q), static_cast<int>(H_kv),
        static_cast<int>(D), static_cast<int>(k_cache.size(2)),
        static_cast<int>(block_table.size(1)), causal, var, current_stream(q));
    return out;
}

int num_variants(const std::string& name) {
    if (name == "rmsnorm") return spark::rmsnorm_num_variants();
    if (name == "swiglu") return spark::swiglu_num_variants();
    if (name == "softmax") return spark::softmax_num_variants();
    if (name == "sgemm") return spark::sgemm_num_variants();
    if (name == "hgemm") return spark::hgemm_num_variants();
    if (name == "fp8gemm") return spark::fp8gemm_num_variants();
    if (name == "fp4gemm") return spark::fp4gemm_num_variants();
    if (name == "attention") return spark::attention_num_variants();
    if (name == "attention_fp8") return spark::attention_fp8_num_variants();
    if (name == "attention_bwd") return spark::attention_bwd_num_variants();
    if (name == "w4gemm") return spark::w4gemm_num_variants();
    if (name == "bandwidth") return spark::bandwidth_num_variants();
    if (name == "paged_decode") return spark::paged_decode_num_variants();
    if (name == "attention_varlen") return spark::attention_varlen_num_variants();
    throw std::invalid_argument("unknown kernel: " + name);
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
          "paged caches; returns q as [T, H_q, D]",
          py::arg("qkv"), py::arg("cos"), py::arg("sin"), py::arg("positions"), py::arg("slots"),
          py::arg("k_cache"), py::arg("v_cache"), py::arg("H_q"), py::arg("H_kv"));
    m.def("paged_decode", &paged_decode,
          "one query token per sequence against a paged K/V cache: q [B, H_q, D], block_table "
          "[B, max_pages], seq_lens [B]",
          py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("block_table"),
          py::arg("seq_lens"), py::arg("variant") = -1);
    m.def("attention_varlen", &attention_varlen,
          "packed prompts (q [T, H_q, D], cu_seqlens_q [B + 1]) against a paged K/V cache, "
          "causal mask aligned bottom-right",
          py::arg("q"), py::arg("k_cache"), py::arg("v_cache"), py::arg("cu_seqlens_q"),
          py::arg("seq_lens"), py::arg("block_table"), py::arg("causal") = true,
          py::arg("variant") = -1);
    m.def("num_variants", &num_variants, "number of implementation variants for a kernel",
          py::arg("name"));
}
