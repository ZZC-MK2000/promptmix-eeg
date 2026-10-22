[![live demo](https://img.shields.io/badge/live%20demo-GitHub%20Pages-blue)](https://zzc-mk2000.github.io/promptmix-eeg/)

> **Live demo:** https://zzc-mk2000.github.io/promptmix-eeg/ — the current three-task story,
> k-shot song-vs-genre evidence, retrieval, and playable frozen-MusicGen clips.
> Stimulus audio is withheld for copyright.

# PromptMix-EEG

**Brain signals → frozen MusicGen, with the listener as the bottleneck.**

The current version has three downstream tasks:

1. **Song identification / retrieval** — a calibrated EEG query retrieves a real audio clip
   (subject-held-out bandpower retrieval **.413 at k=10**, **.504 at k=100**; chance .10).
   The mainline A→L bridge reaches **84.9%** 14-way on **held-out segments** (subjects remain
   in training); that is a different, easier protocol.
2. **Genre-conditioned generation** — the shared 12-way classifier selects a genre prompt
   (**.742 acc, κ .399**) and a frozen MusicGen samples at 32 kHz. k-shot does **not** help
   this task (κ ≤ .045), and zero-shot CLAP prompt matching scores .056 against a .672
   majority floor.
3. **Song-level description selection** — the k-shot calibrated query selects among 14 song
   descriptions (**.350** with discriminative tags, **.201** with Qwen free-text captions;
   10-way, floor .119). This is **text selection, not song-level audio generation**: the
   generated audio remains genre-level (true song prompt vs same-genre control: +.034,
   t = 1.02, n.s.). Song-level delivery is retrieval.

The listener story: zero-shot cross-subject song decoding is at chance; a per-subject fiber
estimated from **10 labelled windows** lifts bandpower song decoding to **.454**, and the
same k-shot calibration is what unlocks the song-level description selection above.

## Repository layout

```
promptmix-eeg/
├── index.html                  # static GitHub Pages demo (current version only)
├── assets/fig/                 # published figures (TNR, shared palette)
├── code/
│   ├── information/            # cross-subject listener analysis + selection/retrieval
│   ├── figures/                # figure scripts (CPU; see render_all.py)
│   ├── classifier/ · radii/ · alignment/ · delivery/ · evaluation/ · shared/
├── vendor/mainv2/              # vendored stage-1 trainer + local import closure
├── docs/DATA_AND_MODELS.md     # expected data caches, model weights, env vars
├── results/public/             # small JSONs used by the webpage figures
├── requirements.txt
└── README.md
```

## Code map

- **Listener / k-shot:** `code/information/info_ladder.py`, `oracle_alignment.py`,
  `k_shot_calibration_v2.py`, `tta_baselines.py`, `codebook_select.py`
- **Subject-held-out retrain + block control:**
  `code/information/run_subject_holdout.sh`, `run_block_control.sh`,
  `eval_subject_holdout.py`; trainer in `vendor/mainv2/`
- **Calibration heads / genre:** `calib_head_pilot.py`, `calib_head_5fold.py`,
  `calib_head_laion_5fold.py`, `calib_head_genre.py`
- **Retrieval / selection:** `bandpower_clap_k_shot.py`, `mainline_k_shot.py`,
  `mainline_linear_probe.py`, `desc_selection_nmedt.py`, `desc_selection_captions.py`,
  `clap_descriptions.py`, `genre_prompts_qwen.py`, `genre_selection_qwen.py`
- **Generation pilot:** `desc_generate.py`, `desc_controls.py`, `desc_fixedseed.py`,
  `desc_samegenre.py`, `desc_margin_all.py`, `pilot_qwen_two_paths.py`,
  `caption_qwen.py`, `caption_lp.py`, `make_caption_audio.py`
- **Figures:** `code/figures/render_all.py` (CPU; Fig 0 is hand-supplied and excluded)

See [`code/information/README.md`](code/information/README.md) for the script-to-claim map.

## What is and is not in this repository

**Included:** code, figure scripts, published figures, the small `results/public/` JSONs
used by the new webpage figures, and the requirements.

**Not included:** raw NMED-T / NMED-H EEG or audio; derived embedding caches (`data/eegcond`,
`vendor/mainv2/data`); trained checkpoints (`eeg_to_clap_proj.pt`, LaBraM, CLAP, MERT,
MusicGen, Qwen2-Audio); and the raw experiment result JSONs. They are subject to dataset
terms, copyright, privacy, model licenses, or size.

The subject-held-out trainer is vendored; it needs the data caches and the external LaBraM
code/checkpoint. See [`docs/DATA_AND_MODELS.md`](docs/DATA_AND_MODELS.md) for the exact
layout and environment variables (`MAINV2_DIR`, `LABRAM_DIR`, `LABRAM_CKPT`, ...).

## Quick start

```bash
pip install -r requirements.txt
# place data caches and model weights as documented in docs/DATA_AND_MODELS.md
export MAINV2_DIR=/path/to/mainv2-with-caches
export LABRAM_DIR=/path/to/LaBraM-main
export LABRAM_CKPT=$LABRAM_DIR/checkpoints/labram-base.pth

python code/information/info_ladder.py
python code/information/oracle_alignment.py
python code/information/k_shot_calibration_v2.py
python code/figures/build_public_fig_data.py   # rebuild the public figure JSONs
python code/figures/render_all.py               # rebuild figures (Fig 0 excluded)
```

## Honest boundaries

- 84.9% / 88.2% are **segment-held-out** (subjects remain in training), not subject-held-out.
- The 12-way genre classifier is majority-heavy; **κ**, not accuracy, is the honest statistic.
- k-shot helps song-level description selection, not genre selection.
- Free-text Qwen captions do not beat discriminative tags for selection.
- Generated audio is **genre-level**; **song-level audio generation is not solved**.
- No human MOS has been run; retrieval of real clips is the song-level deliverable.
