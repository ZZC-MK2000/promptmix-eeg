# Datasheet — Auditory EEG Bench v0.1

## Motivation

Naturalistic auditory EEG is evaluated with incompatible splits, preprocessing and
readouts. AEB fixes a six-dataset training cache and a small set of listener-aware
evaluation splits so that representation quality can be compared across models.

## Composition

| Dataset | Role | Subjects | Windows (8 s) | Hours | Source / access |
|---|---|---:|---:|---:|---|
| CIRE | train | 35 | 8,400 | 18.7 | local provider; license unverified |
| KUL AAD | train | 16 | 11,968 | 26.6 | KU Leuven AAD; license unverified |
| DTU AAD | train | 18 | 6,480 | 14.4 | DTU AAD; license unverified |
| MUSIN-G | train | 20 | 3,760 | 8.4 | OpenNeuro ds003774; CC0 |
| ds002725 | train | 21 | 4,991 | 11.1 | OpenNeuro ds002725; CC0 |
| SparrKULee | train | 85 | 33,509 | 74.5 | KU Leuven; CC BY-NC 4.0; not redistributed |
| **total train** | | **195** | **69,108** | **153.6** | |
| NMED-T | test | 20 | 7,020 | 15.6 | NMED-T; research use; stimuli copyrighted |

SparrKULee contributes train/val files only; its test split is never used in the training
cache. NMED-T is never used for encoder or fine-tune training; its probe is fit on the
training half of each trial-wise split.

## Preprocessing

- All training data: 19 standard 10–20 channels, 200 Hz, non-overlapping 8-s windows.
- SparrKULee: BioSemi64 order → 19 channels, resampled 120 → 200 Hz. The channel order is
  an assumption; it is recorded in `configs/datasets.yaml` and the cache manifest.
- Inputs are per-window, per-channel z-scored and clipped to ±10 for the reference models.
- No raw EEG, audio or checkpoints are included in this repository.

## Splits

- `splits/nmedt_folds.json`: 3 seeds × {random rows, 5-window blocks, subjects}.
- `splits/leave_one_dataset_out.json`: one split per held-out dataset.
- `manifests/split_audit.json`: row/unit/dataset overlap checks.

## Uses

Intended: listener-aware representation comparison, leakage auditing, and reproducible
baselines on the target task. Not intended: clinical diagnosis, subject identification,
or claims about raw-data redistribution.

## Limitations

- 153.6 h is small compared with large EEG foundation-model corpora.
- CIRE/KUL/DTU licenses could not be verified from the local checkout.
- Sparr channel mapping is an assumption.
- The supervised reference budget is fixed but not tuned; see the release blockers.
- NMED-T has only 20 listeners and 10 songs; subject-level CIs are correspondingly wide.
