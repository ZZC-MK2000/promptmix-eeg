# Reproducibility guide

## Environment

- Python 3.12; PyTorch 2.11+cu128; `braindecode>=1.8.1`; scikit-learn, numpy, scipy.
- Hugging Face mirror for BIOT/CBraMod weights:
  `HF_ENDPOINT=https://hf-mirror.com`.
- Local caches are **not** in the repository. See `configs/datasets.yaml` for paths.

## Fixed artifacts

- `splits/nmedt_folds.json` — 3 seeds × random/blocks/subjects.
- `splits/leave_one_dataset_out.json` — one split per held-out dataset.
- `manifests/coverage.json` — subjects/windows/hours.
- `manifests/split_audit.json` — leakage checks.
- `results/reference.json` — reference leaderboard values and caveats.

## Commands

```bash
python code/build_splits.py                 # regenerate splits from local caches
python code/audit_splits.py                 # verify leakage properties
python code/make_leaderboard.py             # rebuild leaderboard/index.html
python code/run_probe.py --features F.npz --split splits/nmedt_folds.json
python code/run_finetune.py --cache CACHE --model eegnet --seeds 0 1 2
```

## Reporting rules

1. Report random rows, blocks and subjects separately.
2. Report chance and the best trivial baseline beside every model.
3. Report frozen probe and fine-tuning in separate tables.
4. Use subject-level paired tests and 95% bootstrap CIs.
5. Do not tune on the test split; use the validation indices in the split files.

## Known gaps before v1.0

- 3-seed unified grid for all models, including tuned supervised baselines.
- LODO results for the full zoo.
- EEGPT channel projection or formal exclusion (excluded in v0.1).
- Downstream audio demo is illustrative only; no MOS.
