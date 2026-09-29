"""Build the spark_kernels PyTorch extension.

    pip install -e . --no-build-isolation      # RTX 5090 (CUDA 13, torch with sm_120 support)
    TORCH_CUDA_ARCH_LIST="12.0a" pip install -e . --no-build-isolation  # same thing, explicit
    TORCH_CUDA_ARCH_LIST="12.1a" pip install -e . --no-build-isolation  # DGX Spark (GB10)

The default architecture is compute capability 12.0 in its architecture-specific form
(sm_120a: the RTX 5090 and the other RTX Blackwell cards). The "a" matters for two kernels:
the fp8 GEMM's block-scaled mma.sync is only exposed on sm_120a / sm_121a, and a plain
"12.0" build falls back to the half-rate fp8 instruction (docs/design/fp8gemm.md); the fp4
GEMM has no instruction at all without it and refuses to run (docs/design/fp4gemm.md). Set
TORCH_CUDA_ARCH_LIST to build for something else: "12.1a" for the DGX Spark, "12.0a;12.1a"
for both.
"""

import glob
import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Target architecture. torch's BuildExtension honors TORCH_CUDA_ARCH_LIST and, if it is
# set, appends its own -gencode flags. If it is *not* set we pin the RTX 5090 (sm_120a)
# explicitly so the build does not fall back to "whatever GPU torch detects" heuristics.
# ---------------------------------------------------------------------------
arch_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "").strip()
gencode_flags = []
if not arch_list:
    gencode_flags = ["-gencode", "arch=compute_120a,code=sm_120a"]

nvcc_flags = [
    "-O3",
    "-lineinfo",  # keep source correlation for Nsight Compute
    "--expt-relaxed-constexpr",
    "-std=c++20",  # torch >= 2.9 headers require C++20
] + gencode_flags

# setuptools refuses absolute source paths, so everything is relative to the project root.
sources = [os.path.join("python", "csrc", "bindings.cpp")] + sorted(
    os.path.relpath(p, ROOT) for p in glob.glob(os.path.join(ROOT, "src", "kernels", "*.cu"))
)

ext = CUDAExtension(
    name="spark_kernels._C",
    sources=sources,
    include_dirs=[os.path.join(ROOT, "include")],
    extra_compile_args={"cxx": ["-O3", "-std=c++20"], "nvcc": nvcc_flags},
)

# name, version and the rest of the metadata come from the [project] table in pyproject.toml.
setup(
    packages=["spark_kernels"],
    package_dir={"": "python"},
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension.with_options(use_ninja=True)},
    zip_safe=False,
)
