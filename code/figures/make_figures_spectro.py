"""Fig 7 (CPU): qualitative spectrogram panel -- real heard window vs delivered native clips
(mode A / B-cat / B-mix) vs a KNOWN off-manifold interpolation clip (jazz<->choir a=0.25, zcr 0.46).
Visual evidence that the delivered clips have musical time-frequency structure (harmonics/ridges)
while the interpolation clip is noise-like. Uses existing wavs only.

Run: python src/train/make_figures_spectro.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import save

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
SR = 32000

# pick representative clips
def first(pat):
    g = sorted(glob.glob(str(ROOT / pat)))
    return g[0] if g else None

clips = [
    ("real heard window", first("results/library_b/listen/ref__*.wav")),
    ("mode A (native, EEG genre)", first("results/library_b/listen/genB__*.wav")),
    ("B-cat (native sample)", first("results/prompt_classifier_true/listen_B/B-cat__*.wav")),
    ("B-mix (waveform blend)", first("results/prompt_classifier_true/listen_B/B-mix__*.wav")),
    ("interpolation a=.25 (OFF-manifold)", first("results/gen_eegcond/library_radius/sweep_p8_10_a025.wav")),
]

fig, axes = plt.subplots(len(clips), 1, figsize=(8.4, 2.1 * len(clips)))
for ax, (label, path) in zip(np.atleast_1d(axes), clips):
    if path is None:
        ax.text(0.5, 0.5, f"{label}: (no file)", ha="center"); ax.axis("off"); continue
    w, sr = sf.read(path)
    w = np.asarray(w)
    if w.ndim > 1:
        w = w.mean(1)
    ax.specgram(w, Fs=sr, NFFT=1024, noverlap=512, cmap="magma", vmin=-80, vmax=-10)
    zc = float(np.mean(np.abs(np.diff(np.sign(w))))) / 2
    # anchored to the axes (not a fixed point offset) so a long label cannot run under the panel
    ax.text(-0.02, 0.5, label, transform=ax.transAxes, ha="right", va="center", fontsize=8)
    ax.set_yticks([]); ax.set_xticks([])
    ax.text(0.99, 0.06, f"zcr={zc:.3f}", transform=ax.transAxes, ha="right", color="w", fontsize=8)
fig.suptitle("Spectrograms: delivered native clips show musical structure, interpolation is noise-like",
             fontsize=9)
fig.subplots_adjust(left=0.26, right=0.99, top=0.95, bottom=0.02, hspace=0.45)
save(fig, "fig7_spectrograms.png", dpi=140)
for label, path in clips:
    print(f"  {label:38} {Path(path).name if path else 'MISSING'}")
