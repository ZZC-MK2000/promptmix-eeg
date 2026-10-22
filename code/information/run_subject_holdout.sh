#!/usr/bin/env bash
# Subject-held-out retraining of the stage-1 alignment head (clean cross-subject control).
#
# Code: vendor/mainv2/src/align/nmedt_align_v2.py
# Data: $MAINV2_DIR/data (read-only caches; not published)
# Each fold is launched through launch_guard.py (one instance per tag); a driver lock
# prevents two copies of this loop from running at once.
#
# Set MAINV2_DIR to the workspace that holds data/ and results/ if it is not the
# vendored copy. Set PYTHON to the interpreter with torch/transformers.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_M2="$ROOT/../mainv2"
if [ ! -d "$DEFAULT_M2/data" ]; then DEFAULT_M2="$ROOT/vendor/mainv2"; fi
MAINV2_DIR="${MAINV2_DIR:-$DEFAULT_M2}"
PY="${PYTHON:-python}"
OUT="$ROOT/results/subject_holdout"
GUARD="$ROOT/code/information/launch_guard.py"
mkdir -p "$OUT" "$ROOT/results/.locks"

DRIVER_LOCK="$ROOT/results/.locks/SH_folds_driver.lock"
if ! mkdir "$DRIVER_LOCK" 2>/dev/null; then
  echo "REFUSED: driver lock exists ($DRIVER_LOCK); another fold loop is running." >&2
  exit 2
fi
trap 'rmdir "$DRIVER_LOCK" 2>/dev/null || true' EXIT

TEST=("02,03,04,05" "06,07,08,09" "10,11,12,13" "14,15,16,17" "19,20,21,23")
VAL=("06,07" "04,05" "06,07" "08,09" "10,11")

cd "$MAINV2_DIR"
for f in 0 1 2 3 4; do
  echo "===== fold $f  test=${TEST[$f]}  val=${VAL[$f]}  $(date) ====="
  "$PY" "$GUARD" --tag "SH_f${f}_stock_s0" \
    --log "$OUT/fold${f}.log" -- \
    "$PY" src/align/nmedt_align_v2.py \
    --tag "SH_f${f}_stock_s0" --cache h14_a --front enc --seed 0 --epochs 25 \
    --micro 16 --accum 4 --lr-enc 1e-4 --lr-head 1e-3 \
    --ce 1 --clip-w 1 --dense 0.5 --dense-target mert --mert-layer -1 \
    --pool --ema 0.999 --init stock \
    --heldout-subjects "${TEST[$f]}" --val-subjects "${VAL[$f]}" \
    --save-emb "$OUT/eeg_emb_aldm_SH_f${f}_s0.npy"
done
echo "ALL FOLDS DONE $(date)"
