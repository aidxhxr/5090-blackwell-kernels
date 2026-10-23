// The projection shapes of real checkpoints that bench_hgemm --models and bench_fp8gemm
// --models run at M = 128 to 4096 tokens, and the two helpers both use: a comma-separated
// list parser for --ms and a device-side max error over a bf16 output (a 128256-column
// lm_head at 4096 tokens is 1 GB of C, too much to bring to the host per row).
//
// A weight is (N, K): K in, N out. bench_hgemm stores it as B [K][N], bench_fp8gemm as
// Bt [N][K] (nn.Linear's layout), the same matrix either way.
#pragma once

#include <cuda_bf16.h>

#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "spark/common.cuh"

namespace spark::bench {

struct ModelWeight {
    const char* model;
    const char* name;
    int N, K;
};

inline const std::vector<ModelWeight>& model_weights() {
    static const std::vector<ModelWeight> w = {
        {"llama3-8b", "q_proj/o_proj", 4096, 4096},
        {"llama3-8b", "k_proj/v_proj", 1024, 4096},
        {"llama3-8b", "qkv fused", 6144, 4096},
        {"llama3-8b", "gate/up", 14336, 4096},
        {"llama3-8b", "down", 4096, 14336},
        {"llama3-8b", "lm_head", 128256, 4096},
        {"llama3-70b", "q_proj/o_proj", 8192, 8192},
        {"llama3-70b", "qkv fused", 10240, 8192},
        {"llama3-70b", "gate/up", 28672, 8192},
        {"llama3-70b", "down", 8192, 28672},
        {"qwen3-8b", "gate/up", 12288, 4096},
        {"qwen3-8b", "down", 4096, 12288},
        {"qwen3-32b", "qkv fused", 10240, 5120},
        {"qwen3-32b", "o_proj", 5120, 8192},
        {"qwen3-32b", "gate/up", 25600, 5120},
        {"qwen3-32b", "down", 5120, 25600},
        {"qwen3-30b-a3b", "expert gate/up", 768, 2048},
        {"qwen3-30b-a3b", "expert down", 2048, 768},
        {"deepseek", "expert up", 2304, 5120},
        {"deepseek", "expert down", 5120, 2304},
    };
    return w;
}

inline std::vector<int> parse_int_list(const std::string& s) {
    std::vector<int> out;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        if (j > i) out.push_back(std::stoi(s.substr(i, j - i)));
        i = j + 1;
    }
    return out;
}

// out[0] = max |got - ref| (a NaN counts as infinite), out[1] = max |ref|, as float bits:
// non-negative floats order like their bit patterns, so atomicMax on unsigned works.
static __global__ void max_err_kernel(const __nv_bfloat16* __restrict__ got,
                                      const __nv_bfloat16* __restrict__ ref, size_t n,
                                      unsigned* out) {
    float d = 0.f, r = 0.f;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        const float g = __bfloat162float(got[i]), x = __bfloat162float(ref[i]);
        const float e = fabsf(g - x);
        d = fmaxf(d, isnan(e) ? INFINITY : e);
        r = fmaxf(r, fabsf(x));
    }
    for (int o = 16; o > 0; o >>= 1) {
        d = fmaxf(d, __shfl_xor_sync(0xffffffffu, d, o));
        r = fmaxf(r, __shfl_xor_sync(0xffffffffu, r, o));
    }
    if ((threadIdx.x & 31) == 0) {
        atomicMax(out, __float_as_uint(d));
        atomicMax(out + 1, __float_as_uint(r));
    }
}

// {max |got - ref|, max |ref|}.
inline std::pair<double, double> max_err_dev(const __nv_bfloat16* got, const __nv_bfloat16* ref,
                                             size_t n, unsigned* scratch, cudaStream_t stream) {
    SPARK_CUDA_CHECK(cudaMemsetAsync(scratch, 0, 2 * sizeof(unsigned), stream));
    max_err_kernel<<<1024, 256, 0, stream>>>(got, ref, n, scratch);
    SPARK_CUDA_CHECK(cudaGetLastError());
    unsigned h[2];
    SPARK_CUDA_CHECK(cudaMemcpy(h, scratch, sizeof(h), cudaMemcpyDeviceToHost));
    float d, r;
    std::memcpy(&d, &h[0], sizeof(float));
    std::memcpy(&r, &h[1], sizeof(float));
    return {d, r};
}

}  // namespace spark::bench
