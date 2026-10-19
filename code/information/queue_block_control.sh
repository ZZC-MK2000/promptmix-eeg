#!/usr/bin/env bash
# Queue the block-split control after the 5 subject-held-out folds finish.
# Waits for the last fold's embeddings, then runs run_block_control.sh.
set -euo pipefail
OUT=/d/Projects/CLaBraM/promptmix-eeg/results/subject_holdout
SENTINEL="$OUT/eeg_emb_aldm_SH_f4_s0.npy"
for i in $(seq 1 1440); do            # up to 24 h
  if [ -f "$SENTINEL" ]; then
    sleep 120                          # let the training process release the GPU
    echo "queue: fold 4 finished, launching block-split control $(date)"
    bash /d/Projects/CLaBraM/promptmix-eeg/code/information/run_block_control.sh \
      2>&1 | tee "$OUT/control_run.log"
    echo "queue: control done $(date)"
    exit 0
  fi
  sleep 60
done
echo "queue: timeout waiting for $SENTINEL" >&2
exit 1
