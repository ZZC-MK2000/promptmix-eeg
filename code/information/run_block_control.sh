#!/usr/bin/env bash
# Control for the subject-held-out retrain: SAME cache/init/recipe, but the default
# 60-s block split (subjects appear in train and test). This isolates the split effect
# from the stock-vs-indomain init effect.
#
# Single-instance: launched through launch_guard.py (rule 38).
set -euo pipefail
PY=/c/Users/18439/.conda/envs/clabram312/python.exe
cd /d/Projects/CLaBraM/mainv2
OUT=/d/Projects/CLaBraM/promptmix-eeg/results/subject_holdout
GUARD=/d/Projects/CLaBraM/promptmix-eeg/code/information/launch_guard.py
mkdir -p "$OUT"
exec "$PY" "$GUARD" --tag SH_ctrl_blocksplit_stock_s0 \
  --log "$OUT/control_run.log" -- \
  "$PY" src/align/nmedt_align_v2.py \
  --tag SH_ctrl_blocksplit_stock_s0 --cache h14_a --front enc --seed 0 --epochs 25 \
  --micro 16 --accum 4 --lr-enc 1e-4 --lr-head 1e-3 \
  --ce 1 --clip-w 1 --dense 0.5 --dense-target mert --mert-layer -1 \
  --pool --ema 0.999 --init stock \
  --save-emb "$OUT/eeg_emb_aldm_ctrl_blocksplit_stock_s0.npy"
