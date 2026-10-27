[![live page](https://img.shields.io/badge/live-GitHub%20Pages-blue)](https://zzc-mk2000.github.io/promptmix-eeg/)

# PromptMix-EEG

**Beyond generic EEG foundation models: a spatial-spectral encoder for naturalistic auditory EEG.**

Live page: <https://zzc-mk2000.github.io/promptmix-eeg/>

## Current version

A compact spatial-spectral encoder (1.14M parameters) is trained on six auditory EEG datasets
(CIRE, KUL, DTU, MUSIN-G, ds002725, SparrKULee): **195 dataset-subject entries, 69,108
eight-second windows, 153.6 hours**. NMED-T is never used for encoder training.

Encoder objective: K=32 learnable spatial filters, one-second spectral tokens, a four-layer
temporal Transformer, masked spectral reconstruction, cross-subject contrastive learning, and a
VICReg-style variance-covariance regularizer. CLAP alignment (frozen audio/text towers) uses
window InfoNCE, per-second dense InfoNCE, a song-ID auxiliary, and a subspace placement anchor.

## Main numbers

| Method | NMED 10-way | CLAP trial cos/top-1 | CLAP subject cos/top-1 |
|---|---:|---:|---:|
| LaBraM-200 (frozen) | .358 | .729 / .265 | .784 / .109 |
| Band-power 82×22 | **.840** | **.796 / .678** | .678 / .108 |
| Encoder (6 datasets) | .607 ± .006 | .764 / .435 | .780 / .099 |

**Only the encoder is target-unseen.** The linear and CLAP probes are fit on NMED-T training
halves; this is not a zero-shot pipeline. Subject-held-out top-1 is at chance for all methods.
Band-power remains the stronger closed-set label baseline.



## Audio demo

The page restores the downstream frozen-MusicGen demo: 14 heard/generated pairs, 8
oracle/calibrated/zero-shot rows, and 12 native prompt samples. It is illustrative and is
**not** part of the six-dataset encoder benchmark; no new MOS or retrieval claim is attached.
Assets live under `assets/audio/` with `assets/pairs.json` and `assets/calibrated_pairs.json`.

## Other architectures

Four standard supervised EEG models were trained from scratch on the same per-subject trial-wise
splits (5 repetitions): EEGNet .118/.134, ShallowConvNet .258/.231, Deep4Net .107/.133,
EEG-Conformer .142/.120 (per-subject/pooled). They remain below the frozen encoder (.607) and
band-power (.840). The per-subject training set has ~17 windows per class; EEGNet reaches 1.00 train
but 0.14 test on a representative subject, so the baselines overfit. BIOT/CBraMod/EEGPT pretrained
weights were not obtainable locally; no random-init versions are presented as foundation-model results.


## Pretrained foundation-model comparison

Frozen features + the same per-subject trial-wise logistic probe: BIOT **.381**, CBraMod **.250**,
LaBraM-200 **.358**, our encoder **.607**, band-power **.840**. BIOT uses 19→18-channel interpolation;
CBraMod runs at 19 channels. EEGPT is not reported because its released checkpoint uses a 62-channel
token layout and a 19-channel adaptation would require a new channel projection, not a comparable
frozen-feature run. See `fig26_foundation_comparison.png`.

## Protocol audit

- Split ladder (NMED-T, band-power): random rows .896 / held-out blocks .873 / held-out subjects .106.
- A short-window study reports 88.69% under a within-participant random split (1-s inputs,
  random time chunks); this is within-listener interpolation, not cross-subject evidence.
- Answer-conditioned audit: real, zeroed and shuffled EEG all report 1.000 pair accuracy in the
  released implementation, while label-free pairwise and EEG-only variants remain near .50.

These are protocol observations, not allegations of intent. The page lists the minimum controls:
listener-level splits, block-held-out controls, zero/shuffled EEG controls, and label-free
pairwise / EEG-only variants for answer-conditioned queries.

## Figure and code map

- `index.html` — current results page only.
- `assets/fig/fig20_protocol_pitfalls.png` — split ladder + answer-conditioned query audit.
- `assets/fig/fig22_method_losses.png` — encoder + CLAP alignment objectives.
- `assets/fig/fig23_main_results.png` — NMED-T main results.
- `assets/fig/fig24_ablations.png` — CLAP loss arms + subject-held-out controls.
- `assets/fig/fig25_model_comparison.png` — supervised EEG baselines vs frozen encoders.
- `assets/fig/fig26_foundation_comparison.png` — BIOT/CBraMod/LaBraM vs our encoder.
- `assets/audio/` — restored downstream demo audio (pairs, calibrated/oracle/zero, prompt library).
- `code/auditory_encoder/` — encoder, six-dataset cache builder, evaluators, VQ pilot.
- `code/figures/make_fig_*.py` — figure scripts.
- Working manuscript is kept locally and is not published with the page.

## Boundaries

- No cross-subject top-1 claim; held-out listeners remain at chance.
- No phase or raw-waveform claim; raw-patch/codebook pilots are negative.
- No data-scaling claim at 153.6 hours.
- SparrKULee 64→19 channel mapping assumes BioSemi64 order; documented in code.
- No raw EEG, checkpoints, or dataset audio are published.
