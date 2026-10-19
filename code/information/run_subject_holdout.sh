#!/usr/bin/env bash
# Subject-held-out retraining of the stage-1 alignment head (clean cross-subject control).
#
# Each fold is launched through launch_guard.py (rule 38: one instance per tag), and a
# driver lock prevents two copies of this loop from running at once.
set -euo pipefail
PY=/c/Users/18439/.conda/envs/clabram312/python.exe
cd /d/Projects/CLaBraM/mainv2
OUT=/d/Projects/CLaBraM/promptmix-eeg/results/subject_holdout
GUARD=/d/Projects/CLaBraM/promptmix-eeg/code/information/launch_guard.py
mkdir -p "$OUT" /d/Projects/CLaBraM/promptmix-eeg/results/.locks

DRIVER_LOCK=/d/Projects/CLaBraM/promptmix-eeg/results/.locks/SH_folds_driver.lock
if ! mkdir "$DRIVER_LOCK" 2>/dev/null; then
  echo "REFUSED: driver lock exists ($DRIVER_LOCK); another fold loop is running." >&2
  exit 2
fi
trap 'rmdir "$DRIVER_LOCK" 2>/dev/null || true' EXIT

TEST=("02,03,04,05" "06,07,08,09" "10,11,12,13" "14,15,16,17" "19,20,21,23")
VAL=("06,07" "04,05" "06,07" "08,09" "10,11")

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
