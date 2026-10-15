# PromptMix-EEG — Environment & Dependencies

Measured on the working machine (2026-10-15). Reproduce with conda env `clabram312`.

## Toolchain (measured)
| Component | Version |
|---|---|
| Python | 3.12.14 |
| PyTorch | 2.11.0+cu128 (CUDA available) |
| CUDA toolkit | 12.8 (required for RTX 50xx / sm_120) |
| transformers | 5.17.0 |
| numpy | 2.5.3 |
| librosa | 1.0.0 |
| scikit-learn | 1.9.1 |
| soundfile | 0.14.0 |
| matplotlib | 3.11.2 |
| scipy | 1.18.1 |

## Hardware
- GPU: NVIDIA RTX 50xx-class (Blackwell, sm_120), 16 GB VRAM. **Must use CUDA 12.8 (cu128)** builds; cu126/older will not run sm_120.
- OS: Windows 10/11. Shell: Git Bash for commands; PowerShell for junctions.

## Install
```bash
conda create -n clabram312 python=3.12 -y
conda activate clabram312
pip install torch --index-url https://download.pytorch.org/whl/cu128   # cu128 for RTX50xx
pip install "transformers>=5.17" numpy librosa scikit-learn soundfile matplotlib scipy
```

## Models / data (external, frozen)
- **MusicGen** (facebook/musicgen-medium or -small) — frozen generator; loaded via `transformers`. Path referenced as `models/musicgen-small` under the repo parent.
- **LaBraM** EEG encoder — frozen; produces the 768-d EEG embedding.
- **laion CLAP** — audio/text embedding (512-d) used for the bridge, oracle labels, and metrics.
- **EEG corpus**: 68 subjects, 36036 kept 8-s windows (`data/eegcond/{cond,codes,group}.npy`). Subject-level train/val split (10% val groups).
- **FMA-8s index** (optional, for legacy fine-tune arms only).

## Windows-specific gotchas (learned the hard way)
- Git Bash's bare `python` is a system 3.14 — **always** invoke the conda env: `/c/Users/18439/.conda/envs/clabram312/python.exe`.
- `torch.cuda.is_available()` can lie on mis-built torch; verify with a real tensor `.to('cuda')`.
- EnCodec `audio_encoder.decode()` **rejects batched codes** under transformers 5.17 ("Expected one frame, got B") and needs **int64** codes (int16 → "ShortTensor" embedding error). Decode one row at a time, cast `.astype(np.int64)`.
- DataLoader `num_workers>0` deadlocks on Windows — keep `num_workers=0`.
- PowerShell 5.1 reads BOM-less `.ps1`/`.md` as GBK → Chinese shows mojibake in console; files themselves are UTF-8 and fine.
- Directory junctions (`New-Item -ItemType Junction`) are used to expose `data/` and `results/` here without duplicating ~GB; removing a junction with `rmdir` does NOT delete the target.

## Run entry points (from `promptmix-eeg/`)
```bash
PY=/c/Users/18439/.conda/envs/clabram312/python.exe
$PY code/radii/library_radius.py            # coverage + tolerance radii (offline, no EEG)
$PY code/radii/compat_matrix_b.py           # full 66-pair tolerance map
$PY code/classifier/finetune_prompt_classifier.py --target true_oracle --outdir results/prompt_classifier_true
$PY code/classifier/eval_true_oracle.py     # kappa / balanced acc / baselines
$PY code/delivery/deliver_library_b.py      # listenable EEG-driven clips (corpus-matched lib)
$PY code/delivery/genre_gallery.py          # one native clip per genre
$PY code/figures/make_figures.py            # paper figures
```
