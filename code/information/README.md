# Cross-subject / listener analysis code

Code for the listener-bottleneck results on the webpage:

- information ladder and split control
- per-subject oracle Procrustes (the subject fiber)
- k-shot calibration and TTA baselines
- subject-held-out retraining and the block-split control
- calibrated retrieval, song-description selection, genre selection
- Qwen / tag description and generation pilots

**Data, derived caches, checkpoints and results are not in this repository.**
See [`docs/DATA_AND_MODELS.md`](../../docs/DATA_AND_MODELS.md) for the expected paths,
environment variables and the privacy/copyright boundary.

## Script map

| script | result / claim |
|---|---|
| `info_ladder.py` | information ladder: representation × split × metric (Fig 14) |
| `oracle_alignment.py` | per-subject Procrustes, 0.089 → 0.706 (Fig 16) |
| `k_shot_calibration.py`, `k_shot_calibration_v2.py` | k-shot anchors, hybrid selection (Fig 15) |
| `tta_baselines.py` | EA / CORAL / z-score / nearest-centroid controls |
| `codebook_select.py` | codebook coverage null (library saturation) |
| `run_subject_holdout.sh`, `eval_subject_holdout.py` | 5-fold subject-held-out retrain + eval (0.106 ± 0.020) |
| `run_block_control.sh`, `queue_block_control.sh` | stock block-split control |
| `bandpower_clap_k_shot.py` | calibrated real-clip retrieval (.413 / .504) |
| `mainline_k_shot.py`, `mainline_linear_probe.py`, `k_shot_mainline.py`, `k_shot_fold_mainline.py` | mainline k-shot / probe controls |
| `calib_head_pilot.py`, `calib_head_5fold.py`, `calib_head_laion_5fold.py` | calibration-aware head; clip-only vs calibrated |
| `calib_head_genre.py` | 12-way genre readout; k-shot does **not** help (κ .40 vs .045) |
| `desc_selection_nmedt.py`, `desc_selection.py`, `desc_selection_captions.py` | 10/14-way description selection (tag .350, Qwen .201) |
| `clap_descriptions.py` | frozen-CLAP tag descriptions |
| `genre_prompts_qwen.py`, `genre_selection_qwen.py` | Qwen genre prompts; CLAP prompt matching negative |
| `caption_qwen.py`, `caption_lp.py`, `make_caption_audio.py` | free-text captions from stimulus audio |
| `desc_generate.py`, `desc_controls.py`, `desc_fixedseed.py`, `desc_samegenre.py`, `desc_margin_all.py` | generation arms, fixed-seed / same-genre / mismatch controls |
| `pilot_qwen_two_paths.py` | three-task pilot (identification / genre generation / description selection) |
| `launch_guard.py`, `test_launch_guard.sh` | one-instance launcher, lock and RAM preflight |

## Data and trainer boundary

- `run_subject_holdout.sh` calls the vendored trainer at
  `vendor/mainv2/src/align/nmedt_align_v2.py`. The public drivers use `MAINV2_DIR`
  (default: sibling `mainv2` if it has `data/`, else `vendor/mainv2`); set `MAINV2_DIR` to your original workspace if the
  data caches live there.
- The stage-1 trainer needs the LaBraM code and `labram-base.pth`; set `LABRAM_DIR` /
  `LABRAM_CKPT`. These are third-party assets and are not redistributed.
- Raw NMED-T / NMED-H EEG and audio are not published. The analysis scripts expect
  `data/eegcond` and `vendor/mainv2/data` caches; see `docs/DATA_AND_MODELS.md`.
- The analysis scripts use the workspace paths in their `ROOT` / `M2` constants. On a
  fresh checkout, point them at your data workspace (or keep the original sibling
  `mainv2/` layout) before running.
- `caption_lp.py` uses the vendored `lpmc_model/` code (LP-MusicCaps, MIT); the model
  weights are external. `caption_qwen.py` expects a local Qwen2-Audio checkpoint.
