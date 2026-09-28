#!/usr/bin/env bash
# One-shot pipeline for a fresh session on the GPU box (RTX 5090 by default):
#   build C++ benches -> run them (validates every variant) -> generate tables + roofline
#   -> build the PyTorch extension -> pytest -> torch comparison -> Nsight Compute reports.
# Each stage is optional: pass --skip-python or --skip-ncu to leave those out.
# On the DGX Spark: ARCH=121 TORCH_CUDA_ARCH_LIST=12.1a ./scripts/run_all.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SKIP_PYTHON=0; SKIP_NCU=0
export ARCH="${ARCH:-120}"
for a in "$@"; do
  case "$a" in
    --skip-python) SKIP_PYTHON=1 ;;
    --skip-ncu) SKIP_NCU=1 ;;
    *) echo "unknown arg: $a" >&2; exit 2 ;;
  esac
done

step() { printf '\n\033[1;32m==> %s\033[0m\n' "$*"; }

step "toolchain"
nvcc --version | tail -1
cmake --version | head -1

step "build (sm_$ARCH)"
make build "ARCH=$ARCH"

step "benchmarks (also validates every variant against cuBLAS / CPU reference; GPU state -> results/env.txt)"
make bench

step "results tables + roofline"
make results
echo; cat results/headline.md || true

if [ "$SKIP_PYTHON" -eq 0 ]; then
  step "PyTorch extension"
  python3 -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cc', torch.cuda.get_device_capability())"
  pip install -e . --no-build-isolation -q
  step "pytest"
  pytest -q tests
  step "torch comparison"
  python3 scripts/bench_torch.py
  python3 scripts/bench_layer.py
  make results
fi

if [ "$SKIP_NCU" -eq 0 ]; then
  step "Nsight Compute"
  make ncu || echo "ncu failed; if it is a permissions error see the comment in scripts/profile_ncu.sh"
fi

step "done"
echo "Next: paste results/headline.md into README.md, commit results/roofline.png and results/env.txt, fill the tables in docs/design/*.md."
