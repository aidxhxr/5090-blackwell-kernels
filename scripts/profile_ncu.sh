#!/usr/bin/env bash
# Profile the top two rungs of each ladder with Nsight Compute and dump the "details" page
# as text.
#   ./scripts/profile_ncu.sh [build_dir]
#
# Permissions: ncu needs access to GPU performance counters, which the driver restricts to
# admins by default (GeForce boards like the RTX 5090 and DGX OS alike). Either run with
# sudo, or allow all users once:
#   echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/ncu.conf
#   sudo update-initramfs -u && sudo reboot
# If the RTX 5090 also drives a display, close GPU-heavy desktop apps first: their launches
# share the counters and show up as noise in the DRAM/SM throughput metrics.
set -euo pipefail

BUILD_DIR="${1:-build}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS="$ROOT/results"
mkdir -p "$RESULTS"

NCU="${NCU:-ncu}"
command -v "$NCU" >/dev/null || { echo "ncu not found; it ships with the CUDA 13 toolkit (/usr/local/cuda/bin)" >&2; exit 1; }

# name | binary + args | kernel-name regex (our kernels all live in namespace spark)
profile() {
  local name="$1"; shift
  local regex="$1"; shift
  echo "=== ncu: $name" >&2
  "$NCU" --set full \
         --kernel-name "regex:$regex" \
         --launch-skip 0 --launch-count 1 \
         -f -o "$RESULTS/ncu_$name" \
         "$@" >/dev/null
  "$NCU" --import "$RESULTS/ncu_$name.ncu-rep" --page details > "$RESULTS/ncu_$name.txt"
  echo "    -> $RESULTS/ncu_$name.ncu-rep, $RESULTS/ncu_$name.txt" >&2
}

profile attention_v4 'attention_v4_kernel' "$BUILD_DIR/bench_attention" --variant=4 --b=1 --h=32 --s=4096 --d=128 --causal=1 --iters=1 --warmup=0
profile attention_v3 'attention_v2_kernel' "$BUILD_DIR/bench_attention" --variant=3 --b=1 --h=32 --s=4096 --d=128 --causal=1 --iters=1 --warmup=0
profile attention_v1 'attention_v1_kernel' "$BUILD_DIR/bench_attention" --variant=1 --b=1 --h=32 --s=4096 --d=128 --causal=1 --iters=1 --warmup=0
profile hgemm_v6   'hgemm_v6' "$BUILD_DIR/bench_hgemm" --variant=6 --m=8192 --n=8192 --k=8192 --iters=1 --warmup=0
profile attention_decode 'attention_decode_kernel' "$BUILD_DIR/bench_attention" --variant=3 --b=1 --hq=32 --hkv=8 --sq=1 --skv=131072 --d=128 --iters=1 --warmup=0
profile fp8gemm_v2 'fp8gemm_v2' "$BUILD_DIR/bench_fp8gemm" --variant=2 --m=8192 --n=8192 --k=8192 --iters=1 --warmup=0
profile fp8gemm_v1 'fp8gemm_v1' "$BUILD_DIR/bench_fp8gemm" --variant=1 --m=8192 --n=8192 --k=8192 --iters=1 --warmup=0
profile hgemm_v5   'hgemm_v5' "$BUILD_DIR/bench_hgemm" --variant=5 --m=8192 --n=8192 --k=8192 --iters=1 --warmup=0
profile hgemm_v4   'hgemm_v4' "$BUILD_DIR/bench_hgemm" --variant=4 --m=8192 --n=8192 --k=8192 --iters=1 --warmup=0
profile hgemm_v3   'hgemm_v3' "$BUILD_DIR/bench_hgemm" --variant=3 --m=8192 --n=8192 --k=8192 --iters=1 --warmup=0
profile hgemm_v0   'hgemm_v0' "$BUILD_DIR/bench_hgemm" --variant=0 --m=4096 --n=4096 --k=4096 --iters=1 --warmup=0
profile sgemm_v4   'sgemm_regtile_prefetch' "$BUILD_DIR/bench_sgemm" --variant=4 --m=4096 --n=4096 --k=4096 --iters=1 --warmup=0
profile sgemm_v3   'sgemm_regtile_cpasync'  "$BUILD_DIR/bench_sgemm" --variant=3 --m=4096 --n=4096 --k=4096 --iters=1 --warmup=0
profile rmsnorm_v4 'rmsnorm_reg'   "$BUILD_DIR/bench_rmsnorm" --rows=16384 --cols=8192 --iters=1 --warmup=0
profile rmsnorm_v3 'rmsnorm_block' "$BUILD_DIR/bench_rmsnorm" --rows=16384 --cols=8192 --iters=1 --warmup=0
profile softmax_v3 'softmax_reg'   "$BUILD_DIR/bench_softmax" --cols=16384 --iters=1 --warmup=0
profile softmax_v2 'softmax_block' "$BUILD_DIR/bench_softmax" --cols=16384 --iters=1 --warmup=0

echo "open the .ncu-rep files in the Nsight Compute GUI, or grep the .txt dumps for" >&2
echo "'Achieved Occupancy', 'DRAM Throughput', 'Shared Memory Bank Conflicts'." >&2
