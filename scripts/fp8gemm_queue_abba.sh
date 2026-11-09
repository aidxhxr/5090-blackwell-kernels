#!/usr/bin/env bash
# The sweeps behind fp8gemm variant 3's whole-queue rule (docs/design/fp8gemm.md, "Whole
# tiles from the queue"): the model shapes four times in ABBA order, A the Stream-K region
# on every queue-mode shape (SPARK_FP8GEMM_V3_WHOLE_QUEUE=0, rows "MxNxK@<label>-sk") and B
# another setting of the knob, every row timed next to cuBLASLt on the same bytes.
#   ./scripts/fp8gemm_queue_abba.sh <out_dir> [tensor|mx-tiled] [B's knob] [bench args]
#   ./scripts/fp8gemm_queue_abba.sh sweep/mx mx-tiled       # A1 B1 B2 A2 .jsonl and .err
#   ./scripts/fp8gemm_queue_abba.sh sweep/wq tensor 1       # B forced whole, rows "@v3-wq"
#   ./scripts/fp8gemm_queue_abba.sh sweep/lm tensor -1 --models=llama3-8b --iters=200
# B's knob defaults to the rule (-1, the variable unset). Bench args go first, so they win
# over the defaults here (bench_common.hpp's Args takes the first of a name).
# Run it in one hold of whatever keeps other work off the GPU: the two orders only cancel the
# drift of the card (clocks, power, other processes) when nothing else changes in between.
set -euo pipefail

OUT="${1:?usage: fp8gemm_queue_abba.sh <out_dir> [tensor|mx-tiled] [knob] [bench args]}"
MODE="${2:-tensor}"
KNOB="${3:-}"
EXTRA=("${@:4}")
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BENCH="${BENCH:-$ROOT/build/bench_fp8gemm}"
mkdir -p "$OUT"

case "$MODE" in
  tensor) ARGS=(--models --mode=tensor --iters=20) ;;
  mx-tiled) ARGS=(--models --mode=mx --layout=tiled --iters=20) ;;
  *) echo "mode: tensor or mx-tiled" >&2; exit 1 ;;
esac
BTAG=""
[ "$KNOB" = 1 ] && BTAG="-wq"
[ "$KNOB" = -1 ] && KNOB=""

# pass <name> <tag> [ENV=value]: one sweep of the shapes into <name>.jsonl, the table
# and the exit code into <name>.err.
pass() {
  local name="$1" tag="$2"; shift 2
  local rc=0
  env "$@" "$BENCH" ${EXTRA[@]+"${EXTRA[@]}"} "${ARGS[@]}" ${tag:+--tag="$tag"} >"$OUT/$name.jsonl" 2>"$OUT/$name.err" || rc=$?
  echo "exit $rc" >>"$OUT/$name.err"
}

B_ENV=(X=0)
[ -n "$KNOB" ] && B_ENV=(SPARK_FP8GEMM_V3_WHOLE_QUEUE="$KNOB")
pass A1 -sk SPARK_FP8GEMM_V3_WHOLE_QUEUE=0
pass B1 "$BTAG" "${B_ENV[@]}"
pass B2 "$BTAG" "${B_ENV[@]}"
pass A2 -sk SPARK_FP8GEMM_V3_WHOLE_QUEUE=0
