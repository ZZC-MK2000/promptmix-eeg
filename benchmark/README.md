# Auditory EEG Bench (AEB) — v0.1

Repository: https://github.com/ZZC-MK2000/auditory-eeg-bench

**A benchmark for naturalistic auditory EEG decoding, not an encoder paper.**

AEB fixes the data cache, split suite, model zoo, controls and reporting format so that
"good auditory EEG representation" can be compared without changing the protocol between
papers. The reference task is NMED-T 10-way song identification; the stress test is
leave-one-dataset-out (LODO) over six auditory datasets.

> **Status.** v0.1 contains the fixed split files, a leakage audit, frozen-probe reference
> results for LaBraM/BIOT/CBraMod/band-power and a compact spatial-spectral encoder, and
> from-scratch supervised reference results. The unified 3-seed fine-tune grid and the full
> LODO leaderboard are the next release; this repository does not claim they are complete.

## What is fixed

| Component | Definition |
|---|---|
| Training cache | six datasets, 19 channels, 200 Hz, non-overlapping 8-s windows |
| Scale | **69,108 windows, 195 dataset-subject entries, 153.6 h** |
| Target task | NMED-T, 20 subjects × 10 songs, 10-way window identification |
| Primary readout | frozen features + per-subject logistic probe (trial-wise 50/50) |
| Secondary readout | supervised fine-tuning, same per-subject trial-wise splits |
| Split suite | random rows / held-out blocks / held-out subjects / leave-one-dataset-out |
| Controls | zero EEG, shuffled EEG, answer-conditioned query, adjacent-window upper bound |
| Reporting | chance/majority, mean ± SD, subject-level paired tests, 95% CI |

## Split suite

All split files use row indices into local caches; no raw EEG is redistributed.
`nmedt_folds.json` has 3 seeds for each of random rows, blocks and subjects.
`leave_one_dataset_out.json` has one split per held-out dataset.

| Split | Unit | Expected leakage | Train / val / test (seed 0) |
|---|---|---|---:|
| random rows | window | subject + adjacent windows (intentional upper bound) | 4,914 / 1,053 / 1,053 |
| held-out blocks | 5-window contiguous block | no block overlap; same subjects | 4,883 / 1,071 / 1,066 |
| held-out subjects | subject | no subject overlap | 4,914 / 1,053 / 1,053 |
| leave-one-dataset-out | dataset | no dataset/subject overlap | varies |

The audit is in `manifests/split_audit.json`. Random-row subject overlap is 20/20 by
design; block splits also keep subjects in training; subject splits have zero subject
overlap. LODO splits have zero dataset and zero subject overlap.

## Reference results

Frozen features, per-subject trial-wise logistic probe, 5 split repetitions (chance 0.10):

| Model | Family | Params | 10-way | Note |
|---|---|---:|---:|---|
| Band-power 82×22 | feature | — | **0.840** | strongest closed-set label baseline |
| Encoder (6 datasets) | frozen encoder | 1.14M | **0.607** | target-unseen encoder |
| BIOT | pretrained FM | 3.19M | 0.381 | 19→18-channel interpolation |
| LaBraM-200 | pretrained FM | — | 0.358 | frozen 200-d |
| CBraMod | pretrained FM | 4.92M | 0.250 | 19-channel |
| EEGPT | pretrained FM | — | excluded | 62-channel checkpoint; no comparable 19-ch projection |
| EEGNet | supervised | 9K | 0.118 | from scratch, per-subject |
| ShallowConvNet | supervised | 72K | 0.258 | from scratch, per-subject |
| Deep4Net | supervised | 305K | 0.107 | from scratch, per-subject |
| EEG-Conformer | supervised | 1.19M | 0.142 | from scratch, per-subject |

Supervised models were also trained pooled across subjects (0.120–0.231). They are
reference points, not tuned upper bounds: the per-subject training half has ~17
windows/class and EEGNet reaches 1.00 train / 0.14 test on a representative subject.
A unified 3-seed, tuned fine-tune grid is listed as a release blocker below.

## Controls

- **Random rows vs held-out blocks vs held-out subjects** (band-power): .896 / .873 / .106.
- **Answer-conditioned query** (released MindMix implementation, KUL S1): real, zero and
  shuffled EEG all report 1.000 pair accuracy; label-free pairwise .503 and EEG-only .501.
- **Zero/shuffled EEG** must be reported for every new query-conditioned method.
- **Adjacent windows** are retained only as an intentional upper bound, never as a
  cross-subject result.

## Data and licenses

The dataset table and redistribution rules are in `DATASHEET.md` and `DATA_LICENSES.md`.
In short: OpenNeuro ds002725 and MUSIN-G are CC0; SparrKULee is CC BY-NC 4.0 and is not
redistributed (download script and split files only); CIRE/KUL/DTU licenses must be
confirmed with their providers before any raw-data release. NMED-T stimuli are
copyrighted; only derived features and aggregate results are released.

## Reproduce

```bash
# 1. put local caches at the paths in configs/datasets.yaml
python code/audit_splits.py
python code/run_probe.py --features frozen_features.npz --split splits/nmedt_folds.json
python code/run_finetune.py --cache /path/to/nmed_cache --model eegnet --seeds 0 1 2
python code/make_leaderboard.py --results results/reference.json --out leaderboard/index.html
```

The cache builder and dataset access notes are in `DATASHEET.md`; raw EEG and
checkpoints are not in this repository.

## Release blockers before calling the benchmark final

1. Unified 3-seed grid for every frozen and fine-tune model; report both separately.
2. Tuned supervised budget (validation-based early stopping or a preregistered grid).
3. LODO results for the full model zoo.
4. EEGPT: either a documented 62→19 channel projection or a formal exclusion in the
   leaderboard (currently excluded).
5. Subject-level paired tests and 95% CIs for every main-table claim.

## Citation

If you use AEB, cite the dataset sources and this benchmark release. See `CITATION.cff`.
