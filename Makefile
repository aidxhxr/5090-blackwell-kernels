# Convenience wrapper. Real build logic lives in CMakeLists.txt / setup.py.
# ARCH: 120 = RTX 5090 (default), 121 = DGX Spark / GB10, "120;121" = both. The fp8 and fp4
# kernels and bench_peak are built for the matching "a" arch (120a / 121a), where the
# block-scaled fp8 and fp4 mma.sync live; CMakeLists.txt does the mapping.
BUILD ?= build
ARCH  ?= 120

CXX_SOURCES = $(shell find include src python/csrc -type f \
	\( -name '*.cu' -o -name '*.cuh' -o -name '*.h' -o -name '*.hpp' -o -name '*.cpp' \))

.PHONY: all help configure build bench python test lint format ncu results env clean

all: build

help:
	@echo "configure  cmake configure only, for sm_$(ARCH) into $(BUILD)/ (ARCH=121 for the DGX Spark)"
	@echo "build      cmake configure + build for sm_$(ARCH) into $(BUILD)/"
	@echo "bench      run every bench_* binary, write results/*.json and results/env.txt (GPU state)"
	@echo "results    docs/RESULTS.md, results/headline.md, results/roofline.png"
	@echo "python     pip install -e . (PyTorch extension)"
	@echo "test       install the extension, then pytest parity tests for every variant"
	@echo "lint       ruff + clang-format --dry-run, same checks as CI"
	@echo "format     clang-format -i on all C++/CUDA sources"
	@echo "ncu        Nsight Compute reports for the top rungs of every ladder (needs root on GeForce)"
	@echo "env        GPU state that matters for a run: clocks, P-state, power limit, throttling, display"
	@echo "clean      remove build outputs"

configure:
	cmake -S . -B $(BUILD) -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_ARCHITECTURES=$(ARCH)"

build: configure
	cmake --build $(BUILD) -j

bench: build
	./scripts/run_all_benches.sh $(BUILD)

results:
	python3 scripts/make_results_table.py
	python3 scripts/roofline.py

python:
	pip install -e . --no-build-isolation -v

test: python
	pytest -q tests

lint:
	ruff check python scripts tests setup.py
	clang-format --dry-run --Werror $(CXX_SOURCES)

format:
	clang-format -i $(CXX_SOURCES)

env:
	@./scripts/gpu_env.sh

ncu: build
	./scripts/profile_ncu.sh $(BUILD)

clean:
	rm -rf $(BUILD) build_ext *.egg-info python/spark_kernels/*.so
