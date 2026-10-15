[![live demo](https://img.shields.io/badge/live%20demo-GitHub%20Pages-blue)](https://zzc-mk2000.github.io/promptmix-eeg/)

> **Live demo (static GitHub Pages):** https://zzc-mk2000.github.io/promptmix-eeg/ — method, figures, and generated audio samples.

# PromptMix-EEG

**Frozen-Generator Music Synthesis from Brain Signals via Prompt-Library Mixture Weights.**
Two decoupled parts: **Alignment** (EEG→laion-CLAP bridge → *retrieve the heard song*, 84.9% 14-way) and **Generation** (tiny adapter → mixture weights over a frozen native-prompt library → **frozen** MusicGen 32 kHz → *pleasant music by construction*).

Renamed from `CLaBraM` (the upstream EEG-encoder's name, no longer apt). This folder is the **self-contained project root** — code, data, results, docs, listening clips are all real files here (no symlinks/junctions). `../mainv3` is kept as the original working tree / legacy fallback. The large legacy fine-tune checkpoints (`gen_eegcond*/*.pt`, ~3 GB) were intentionally NOT copied (reproducible by training; not part of the PromptMix-EEG deliverable) — everything else (embeddings, JSONs, classifier weights, all listenable wavs) is present.

## Structure (one glance)

```
promptmix-eeg/
├── code/                      # all experiment code, grouped by function
│   ├── radii/                 # the three-radius pre-gate (coverage / tolerance / label-validity)
│   │     library_radius.py · validate_mode_a_labels.py · compat_matrix_b.py · coverage_k24.py · library_b_experiment.py
│   ├── classifier/            # the trainable projection layer (EEG -> prompt weights)
│   │     finetune_prompt_classifier.py · compute_true_oracle(_fast).py · eval_true_oracle.py · bootstrap_genre_ci.py
│   ├── delivery/              # turn EEG -> listenable frozen-native audio
│   │     deliver_mode_a.py · deliver_mode_b.py · deliver_library_b.py · genre_gallery.py
│   ├── evaluation/            # musicality / quality metrics (incl. honest negatives)
│   │     learned_musicality.py · musicality_metric.py · mixed_eval.py · score_listen.py · score_identity.py
│   ├── alignment/             # Part 1: retrieval of the heard song
│   │     retrieval_eval.py · retrieval_audio.py · second_ridge_*.py · bridge_gap.py
│   ├── figures/               # paper figures
│   │     make_figures.py · make_figures_tsne.py · make_figures_spectro.py · make_fig_architecture.py
│   ├── shared/                # helpers imported across groups (embed_text/gen_from_enc, CondWidthModel)
│   │     arms_2x2.py · pilot_multitoken.py
│   └── legacy/                # earlier fine-tune arms (ep16/mixed2) kept for contrast, not the main route
├── results/                   # real artifacts (JSONs, embeddings, classifier weights, wavs)
│   ├── gen_eegcond/library_radius/   # coverage + tolerance radii (radius.json, library clips)
│   ├── coverage_k24/                 # K=24 library-size ablation
│   ├── library_b/                    # corpus-matched library + compat map + gallery + classifier.pt
│   ├── prompt_classifier/            # cond_oracle classifier + label-validity + true_oracle labels
│   ├── prompt_classifier_true/       # true_oracle classifier + kappa/CI eval + genre_ci.json
│   ├── retrieval_line/               # Part-1 alignment retrieval (song_hit 0.849)
│   └── gen_eegcond_{k8,mixed,mixed2}/# legacy fine-tune arms
├── listening/                 # THE deliverable: paired GEN/REF wavs to judge "pleasant"
│   ├── LISTENING_MANIFEST.md  #    every clip paired with the true heard window + cos
│   ├── gallery/               #    one native clip per genre (12)
│   ├── modeA_corpus_matched/  #    EEG->genre->frozen sample (recommended)
│   ├── modeA_original/ · modeB_mix/
├── docs/                      # write-ups
│   ├── outline_promptmix_eeg.md   # top-venue paper outline (two-part, three-radius, real numbers)
│   ├── RESULTS_SUMMARY.md         # achievements summary
│   └── PROJECT_NAME.md            # rename rationale
├── env/
│   └── ENVIRONMENT.md         # measured toolchain versions + install + Windows gotchas
├── data/                      # real EEG corpus (cond/codes/group) + fma8s distractors (EEG corpus, codes)
├── listening/ · docs/ · env/    # real files (see above)
└── README.md                  # this file
```

## Headline numbers (all verified against source JSONs)
- Alignment: song retrieval **0.849** (chance/constant/shuffled ≈ 0.07); **hard pool 4146 → 0.266 (1103× chance)**.
- Coverage radius: mixture **0.863 → 0.879 (corpus-matched) → 0.890 (K=36)**; **greedy best-12-of-36 = 0.882** (selection matters more than size); trivial baseline 0.70–0.73 (absolute value is laion-clustering-inflated).
- Tolerance radius: embedding interpolation off-manifold for **39% of all 66 pairs** → never interpolate conditions; mix natively.
- EEG→song-genre (direct classifier): **κ 0.551 [0.528, 0.574]** held-out subjects; contrastive bridge lifts the argmax readout 16.7%→28.4%.
- Delivery: frozen native samples, un-clipped; cos-to-heard 0.536 (mode A) / 0.607 (B-mix). End-to-end demo: `code/delivery/demo_eeg_to_music.py`.
- Musicality: **MFCC-Fréchet frozen 3444 vs fine-tuned 29472 (8.6× closer to real timbre)** — the texture metric that resolves the CLAP-FD tie. Human MOS remains the final gate.

## Quick start
See `env/ENVIRONMENT.md` for setup; run entry points listed there. All code reads/writes real files under this folder (`data/`, `results/`) — fully self-contained, no links.
