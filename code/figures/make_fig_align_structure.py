"""Fig 13 (CPU): the stage-1 alignment structure — the project's core design.

One frozen EEG encoder is anchored to the heard audio at TWO granularities at once:
  window level  : InfoNCE against the audio's CLAP embedding, extra negatives = same-song windows
  second level  : InfoNCE of the 8 per-second states against MERT's per-second embeddings
  plus a song-label CE anchor.  Grey = frozen, blue = trainable, orange = loss, green = shipped.

Run: python code/figures/make_fig_align_structure.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import fit_text, PALETTE, save

FROZEN, TRAIN, LOSS, IO, OUT = "#eceff1", "#d6e4ff", "#ffe0b2", "#fff9c4", "#c8e6c9"


def box(ax, fig, x, y, w, h, text, fc, fs=10.5, tc="black", ec="#333"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec=ec, lw=1.2)
    ax.add_patch(p)
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)
    fig.canvas.draw()
    fit_text(fig, p, t)
    return p


def arrow(ax, x1, y1, x2, y2, color="#333", lw=1.5, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 color=color, lw=lw, linestyle=ls))


def loss_node(ax, fig, cx, cy, r, text, fs=10):
    ax.add_patch(Circle((cx, cy), r, fc=LOSS, ec=PALETTE["orange"], lw=1.6))
    t = ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color="#7a3000")
    fig.canvas.draw()
    fit_text(fig, Circle((cx, cy), r * 1.7), t)


def main():
    fig, ax = plt.subplots(figsize=(14.5, 6.9))
    ax.set_xlim(-0.5, 14.8); ax.set_ylim(-0.3, 6.75); ax.axis("off")

    # ---- row 1: window-level anchor -------------------------------------------------
    box(ax, fig, 0.15, 4.45, 1.55, 0.95, "8-s EEG window\n82 channels", IO)
    box(ax, fig, 2.0, 4.45, 1.7, 0.95, "frozen LaBraM\nEEG encoder", FROZEN)
    box(ax, fig, 4.0, 4.45, 1.9, 0.95, "mean over 8 s\n+ head → 512-d unit", TRAIN)
    loss_node(ax, fig, 7.0, 4.92, 0.58, "window\nInfoNCE ×1.0")
    box(ax, fig, 8.0, 4.45, 1.5, 0.95, "512-d unit\nt_win", FROZEN)
    box(ax, fig, 9.8, 4.45, 1.7, 0.95, "frozen CLAP\naudio tower", FROZEN)
    box(ax, fig, 11.8, 4.45, 1.3, 0.95, "same 8-s\naudio", IO)
    arrow(ax, 1.70, 4.92, 2.00, 4.92); arrow(ax, 3.70, 4.92, 4.00, 4.92)
    arrow(ax, 5.90, 4.92, 6.42, 4.92); arrow(ax, 8.00, 4.92, 7.58, 4.92)
    arrow(ax, 9.80, 4.92, 9.50, 4.92); arrow(ax, 11.80, 4.92, 11.50, 4.92)
    ax.text(7.0, 4.05, "negatives = batch + other windows of the SAME song",
            ha="center", fontsize=9, color=PALETTE["orange"])

    # ---- row 2: second-level (MERT dense) anchor -------------------------------------
    box(ax, fig, 2.0, 2.45, 1.7, 0.95, "per-second states\n8 × 200", FROZEN)
    box(ax, fig, 4.0, 2.45, 1.6, 0.95, "proj → 64\n(LN + Linear)", TRAIN)
    loss_node(ax, fig, 7.0, 2.92, 0.58, "per-second\nInfoNCE ×0.5")
    box(ax, fig, 8.0, 2.45, 1.3, 0.95, "proj → 64", TRAIN)
    box(ax, fig, 9.8, 2.45, 1.7, 0.95, "frozen MERT\nmusic SSL", FROZEN)
    arrow(ax, 2.85, 4.45, 2.85, 3.40, ls=(0, (4, 3)))
    arrow(ax, 3.70, 2.92, 4.00, 2.92); arrow(ax, 5.60, 2.92, 6.42, 2.92)
    arrow(ax, 8.00, 2.92, 7.58, 2.92); arrow(ax, 9.80, 2.92, 9.30, 2.92)
    ax.plot([12.45, 12.45], [4.45, 2.92], ls=(0, (4, 3)), color="#333", lw=1.5)
    arrow(ax, 12.45, 2.92, 11.50, 2.92, ls=(0, (4, 3)))
    ax.text(7.0, 2.05, "second t of EEG ↔ second t of MERT; every other second in the batch is a negative",
            ha="center", fontsize=9, color=PALETTE["orange"])

    # ---- row 3: song-label anchor ----------------------------------------------------
    box(ax, fig, 2.0, 1.15, 1.7, 0.80, "pooled features\n(same encoder)", FROZEN, fs=9.5)
    box(ax, fig, 4.0, 1.15, 1.4, 0.80, "linear\nclassifier", TRAIN, fs=9.5)
    loss_node(ax, fig, 7.0, 1.55, 0.50, "song CE ×1.0", fs=9.5)
    box(ax, fig, 8.2, 1.15, 1.3, 0.80, "song\nlabel", IO, fs=9.5)
    arrow(ax, 2.85, 2.45, 2.85, 1.95, ls=(0, (4, 3)))
    arrow(ax, 3.70, 1.55, 4.00, 1.55); arrow(ax, 5.40, 1.55, 6.50, 1.55)
    arrow(ax, 8.20, 1.55, 7.50, 1.55)

    # ---- shipped output --------------------------------------------------------------
    box(ax, fig, 2.0, 0.05, 4.6, 0.80, "40,336 × 512 unit EEG embeddings\n(AudioLDM-CLAP space)",
        OUT, fs=10, ec=PALETTE["green"])
    box(ax, fig, 7.0, 0.05, 1.8, 0.80, "ridge →\nlaion-CLAP", TRAIN, fs=10)
    box(ax, fig, 9.2, 0.05, 3.4, 0.80, "Part 1 retrieval query\nPart 2 prompt weights", OUT,
        fs=10, ec=PALETTE["green"])
    arrow(ax, 6.60, 0.45, 7.00, 0.45); arrow(ax, 8.80, 0.45, 9.20, 0.45)
    ax.plot([4.95, 4.95, -0.25, -0.25], [5.40, 6.05, 6.05, 0.45], ls=(0, (4, 3)),
            color=PALETTE["green"], lw=1.4)
    arrow(ax, -0.25, 0.45, 1.98, 0.45, color=PALETTE["green"], lw=1.4, ls=(0, (4, 3)))
    ax.text(2.3, 6.22, "trained head output z_win, saved per window", fontsize=9, color=PALETTE["green"])

    ax.set_title("Stage-1 alignment: one frozen EEG encoder, anchored to the heard audio at two granularities\n"
                 "grey = frozen · blue = trainable · orange = loss · green = shipped",
                 fontsize=12.5, pad=12)
    save(fig, "fig13_align_structure.png")


if __name__ == "__main__":
    main()
