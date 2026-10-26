"""Method and loss diagram: spatial-spectral encoder + frozen CLAP alignment."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from figstyle import PALETTE, fit_text, save  # noqa: E402


def box(ax, fig, x, y, w, h, text, fc, fs=11, ec="#37474f", lw=1.1):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)
    fig.canvas.draw(); fit_text(fig, p, t, margin=0.92)
    return p


def arrow(ax, x1, y1, x2, y2, color="#37474f", lw=1.4, style="-|>"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=12, color=color, lw=lw))


def main():
    fig, ax = plt.subplots(figsize=(13.2, 7.2))
    ax.set_xlim(0, 13.2); ax.set_ylim(0, 7.2); ax.axis("off")

    ax.text(0.1, 6.95, "Spatial-spectral EEG encoder", fontsize=12, weight="bold",
            color=PALETTE["blue_dark"], va="top")
    ax.text(8.0, 6.95, "Frozen CLAP semantic space", fontsize=12, weight="bold",
            color=PALETTE["green_dark"], va="top")

    # encoder chain
    box(ax, fig, 0.2, 5.2, 1.25, 0.8, "8-s EEG\n64 ch", "#eceff1")
    box(ax, fig, 1.8, 5.2, 1.55, 0.8, "learnable\nspatial filters", "#bbdefb")
    box(ax, fig, 3.7, 5.2, 1.55, 0.8, "1-s spectral\ntokens", "#bbdefb")
    box(ax, fig, 5.6, 5.1, 1.75, 1.0, "temporal\nTransformer", "#90caf9")
    box(ax, fig, 7.65, 5.15, 1.15, 0.9, "content\n$z$", "#64b5f6")
    for a, b in [(1.45, 1.8), (3.35, 3.7), (5.25, 5.6), (7.35, 7.65)]:
        arrow(ax, a, 5.6, b, 5.6)
    ax.text(2.55, 4.95, "PC1-style, K=32", fontsize=8.5, ha="center", color="#37474f")
    ax.text(4.47, 4.95, "5 bands / 1 s", fontsize=8.5, ha="center", color="#37474f")
    ax.text(6.47, 4.9, "8 tokens", fontsize=8.5, ha="center", color="#37474f")

    # encoder losses
    enc = FancyBboxPatch((0.2, 2.25), 8.6, 2.35, boxstyle="round,pad=0.02",
                         fc="#e3f2fd", ec=PALETTE["blue"], lw=1.4)
    ax.add_patch(enc)
    ax.text(0.4, 4.35, "Encoder objective (trainable)", fontsize=10.5, weight="bold",
            color=PALETTE["blue_dark"])
    box(ax, fig, 0.45, 3.5, 2.4, 0.65, "masked spectral\nreconstruction  $L_{rec}$", "#ffffff", fs=9.5)
    box(ax, fig, 3.1, 3.5, 2.4, 0.65, "cross-subject\nInfoNCE  $L_{cross}$", "#ffffff", fs=9.5)
    box(ax, fig, 5.75, 3.5, 2.55, 0.65, "VICReg variance/\ncovariance  $L_{vic}$", "#ffffff", fs=9.5)
    ax.text(0.45, 2.55, "VICReg prevents dimensional collapse; no label or target audio is used here.",
            fontsize=8.3, color="#37474f")

    # CLAP side
    box(ax, fig, 9.2, 5.2, 1.6, 0.8, "frozen CLAP\naudio encoder", "#c8e6c9", fs=9.5)
    box(ax, fig, 11.15, 5.2, 1.6, 0.8, "frozen CLAP\ntext encoder", "#c8e6c9", fs=9.5)
    box(ax, fig, 9.2, 4.1, 1.6, 0.7, "audio target\n$a$", "#e8f5e9", fs=9.5)
    box(ax, fig, 11.15, 4.1, 1.6, 0.7, "text anchor\n$t$", "#e8f5e9", fs=9.5)
    box(ax, fig, 10.2, 3.0, 1.75, 0.65, "EEG projection\n$\\phi(z)$", "#fff3e0", fs=9.5)
    arrow(ax, 9.2, 5.6, 9.2, 4.8); arrow(ax, 11.15, 5.6, 11.15, 4.8)
    arrow(ax, 8.8, 5.5, 10.2, 3.65, color=PALETTE["blue"])

    clap = FancyBboxPatch((8.9, 0.75), 4.1, 2.0, boxstyle="round,pad=0.02",
                          fc="#e8f5e9", ec=PALETTE["green"], lw=1.4)
    ax.add_patch(clap)
    ax.text(9.1, 2.5, "CLAP alignment objective", fontsize=10.5, weight="bold",
            color=PALETTE["green_dark"])
    ax.text(9.1, 2.2, "window InfoNCE  $L_{clip}$", fontsize=9.5)
    ax.text(9.1, 1.9, "per-second dense  $L_{dense}$", fontsize=9.5)
    ax.text(9.1, 1.6, "song-ID CE  $L_{CE}$", fontsize=9.5)
    ax.text(9.1, 1.3, "subspace anchor  $L_{pos}$", fontsize=9.5)
    ax.text(9.1, 0.95, "audio/text towers frozen", fontsize=8.3, color=PALETTE["green_dark"])

    ax.text(0.1, 0.35, "Evaluation: NMED-T is never used for encoder or CLAP-head training.",
            fontsize=9, color=PALETTE["red"])
    save(fig, "fig22_method_losses.png")


if __name__ == "__main__":
    main()
