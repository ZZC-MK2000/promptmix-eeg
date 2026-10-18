"""Shared, clipping-safe saving for the paper figures.

matplotlib's default savefig bbox is the figure canvas, so any title or axis label longer than
the axes is silently cut off at the edge. Every figure here is saved with bbox_inches="tight"
(the canvas grows to contain all artists) and mirrored to paper/figures so the two copies
cannot drift.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
DESTS = (ROOT / "assets" / "fig", ROOT.parent / "paper" / "figures")
TITLE_FS = 10.5

# Serif body for every figure: English + numerals in Times New Roman, math in the
# Times-compatible STIX face. The DejaVu Serif tail is a glyph fallback only.
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
})

# One coherent, colour-blind-safe palette across all figures. Semantics:
#   green = frozen / ours / safe      blue = trainable / main series
#   orange/amber = fine-tuned arms    red = off-manifold / danger
#   grey = trivial baseline           ink = near-black text
PALETTE = {
    "green": "#2e7d32",
    "green_dark": "#1b5e20",
    "blue": "#1565c0",
    "blue_dark": "#0d47a1",
    "orange": "#ef6c00",
    "amber": "#e6a817",
    "red": "#c62828",
    "purple": "#6a1b9a",
    "grey": "#9e9e9e",
    "grey_light": "#cfd8dc",
    "ink": "#263238",
}


def save(fig, name, dpi=150):
    for d in DESTS:
        d.mkdir(parents=True, exist_ok=True)
        fig.savefig(d / name, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print("wrote", DESTS[0] / name)


def fit_text(fig, patch, text, margin=0.9, min_fs=6.0, step=0.5):
    """Shrink `text` until its rendered box fits inside `patch` (in pixels). Returns the size used."""
    renderer = fig.canvas.get_renderer()
    box = patch.get_window_extent(renderer=renderer)
    for _ in range(40):
        t = text.get_window_extent(renderer=renderer)
        if t.width <= box.width * margin and t.height <= box.height * margin:
            return text.get_fontsize()
        fs = text.get_fontsize() - step
        if fs < min_fs:
            text.set_fontsize(min_fs)
            return min_fs
        text.set_fontsize(fs)
    return text.get_fontsize()
