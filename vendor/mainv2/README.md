# Vendored `mainv2` trainer code

Snapshot (2026-10-22) of the stage-1 alignment trainer used by
`code/information/run_subject_holdout.sh` and `run_block_control.sh`.

Files:

```
src/align/nmedt_align_v2.py          # stage-1 trainer (subject-held-out / block splits)
src/encoder/encoder_labram.py        # frozen LaBraM-Base wrapper (needs external LaBraM code)
src/measurement/gate_nmedt_labram.py # cache + target harness imported by the trainer
src/measurement/gate_nmedt_nway.py   # Head / InfoNCE / padding helpers
src/audio_target/clap_eval.py        # CLAP audio evaluator
src/audio_target/codec.py            # EnCodec helpers
src/generate/musicgen_clap_finetune.py
src/generate/train_eeg_to_clap_proj.py
src/generate/train_eeg_labram_to_laion.py
```

These are code only. **No data, caches, checkpoints, or results are in this directory.**
See [`docs/DATA_AND_MODELS.md`](../../docs/DATA_AND_MODELS.md).

## Expected layout

```
vendor/mainv2/data/          # read-only caches (gate_nmedh14_ov2_a.npz etc.)
vendor/mainv2/results/       # eeg_to_clap_proj.pt, align_ladder_result.json
vendor/models/musicgen-small # or set MUSICGEN_DIR
vendor/LaBraM-main/          # external LaBraM code + checkpoints/labram-base.pth
data/eegcond/                # NMED-T/NMED-H caches (cond/codes/group)
```

## Environment

- `MAINV2_DIR` — workspace containing `data/` and `results/`; defaults to the sibling `mainv2` if it has `data/`,
  else this directory. Set it to your original `mainv2` if the caches live there.
- `MAINV2_DATA` / `MAINV2_RESULTS` — override the cache/result directories.
- `NMEDT_ROOT` — raw NMED-T audio/EEG root, if a script needs it.
- `LABRAM_DIR` / `LABRAM_CKPT` — LaBraM code and `labram-base.pth`.
- `MUSICGEN_DIR`, `FMA_DIR` — generation/retrieval assets.

The trainer still uses the original import layout (`sys.path` to `src/encoder`,
`src/measurement`, ...); run it from this directory, as the drivers do.
