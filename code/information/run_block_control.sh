#!/usr/bin/env bash
# Control for the subject-held-out retrain: SAME cache/init/recipe, but the default
# 60-s block split (subjects appear in train and test). This isolates the split effect
# from the stock-vs-indomain init effect.
#
# Single-instance: launched through launch_guard.py (rule 38).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEFAULT_M2="$ROOT/../mainv2"
if [ ! -d "$DEFAULT_M2/data" ]; then DEFAULT_M2="$ROOT/vendor/mainv2"; fi
MAINV2_DIR="${MAINV2_DIR:-$DEFAULT_M2}"
PY="${PYTHON:-python}"
OUT="$ROOT/results/subject_holdout"
GUARD="$ROOT/code/information/launch_guard.py"
mkdir -p "$OUT"
cd "$MAINV2_DIR"
exec "$PY" "$GUARD" --tag SH_ctrl_blocksplit_stock_s0 \
  --log "$OUT/control_run.log" -- \
  "$PY" src/align/nmedt_align_v2.py \
  --tag SH_ctrl_blocksplit_stock_s0 --cache h14_a --front enc --seed 0 --epochs 25 \
  --micro 16 --accum 4 --lr-enc 1e-4 --lr-head 1e-3 \
  --ce 1 --clip-w 1 --dense 0.5 --dense-target mert --mert-layer -1 \
  --pool --ema 0.999 --init stock \
  --save-emb "$OUT/eeg_emb_aldm_ctrl_blocksplit_stock_s0.npy"
