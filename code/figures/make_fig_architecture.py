"""Fig 0 (CPU): the two-part architecture diagram for the paper.
Part 1 (Alignment): EEG -> frozen LaBraM -> ridge bridge -> laion-CLAP -> RETRIEVE heard song.
Part 2 (Generation): bridge(EEG) -> tiny projection (mixture weights over a frozen 12-prompt
library) -> pick/mix prompts -> FROZEN MusicGen native sampling -> pleasant 32 kHz audio.
Drawn with matplotlib boxes/arrows only (no external deps). Every label is auto-shrunk to fit
its box, so no wording can spill past a border.

Run: python code/figures/make_fig_architecture.py
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import fit_text, save


def box(ax, fig, x, y, w, h, text, fc, fs=9.5, tc="black"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012", fc=fc, ec="#333", lw=1.2)
    ax.add_patch(p)
    t = ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc)
    fig.canvas.draw()
    used = fit_text(fig, p, t)
    if used < fs:
        print(f"  fitted {text.splitlines()[0]!r:34} {fs} -> {used}")
    return p


def header(ax, fig, panel, cx, y, text, color, fs=10.5):
    t = ax.text(cx, y, text, ha="center", va="center", fontsize=fs, color=color, weight="bold")
    fig.canvas.draw()
    used = fit_text(fig, panel, t, margin=0.94)
    if used < fs:
        print(f"  fitted header {text[:28]!r} {fs} -> {used}")


def arrow(ax, x1, y1, x2, y2, color="#333", style="-|>", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw))


def main():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")

    # shared input chain
    box(ax, fig, 0.2, 2.7, 1.4, 1.0, "8-s EEG\nwindow", "#e8f0fe")
    box(ax, fig, 2.0, 2.7, 1.7, 1.0, "frozen\nLaBraM encoder", "#d2e3fc")
    box(ax, fig, 4.1, 2.7, 1.9, 1.0, "ridge bridge\n→ laion-CLAP cond", "#d2e3fc")
    arrow(ax, 1.6, 3.2, 2.0, 3.2); arrow(ax, 3.7, 3.2, 4.1, 3.2)

    # Part 1: alignment / retrieval
    p1 = FancyBboxPatch((6.5, 3.9), 5.3, 1.9, boxstyle="round,pad=0.02",
                        fc="#e6f4ea", ec="#2b7a2b", lw=1.6)
    ax.add_patch(p1)
    header(ax, fig, p1, 9.15, 5.45, "PART 1 — ALIGNMENT   (retrieve the heard song)", "#1b5e20")
    box(ax, fig, 6.7, 4.05, 1.7, 0.95, "kNN / rank in\nlaion space", "#c8e6c9")
    box(ax, fig, 8.6, 4.05, 1.5, 0.95, "match real\nsong", "#c8e6c9")
    box(ax, fig, 10.3, 4.05, 1.3, 0.95, "song-ID\n84%", "#a5d6a7", tc="#1b5e20")
    arrow(ax, 8.4, 4.53, 8.6, 4.53); arrow(ax, 10.1, 4.53, 10.3, 4.53)
    arrow(ax, 6.0, 3.45, 6.7, 4.53, color="#2b7a2b")

    # Part 2: generation
    p2 = FancyBboxPatch((6.5, 0.35), 5.3, 3.05, boxstyle="round,pad=0.02",
                        fc="#fce8e6", ec="#b03a2e", lw=1.6)
    ax.add_patch(p2)
    header(ax, fig, p2, 9.15, 3.1, "PART 2 — GENERATION   (pleasant music, by construction)",
           "#8e2b1f")
    box(ax, fig, 6.7, 1.85, 1.6, 0.9, "tiny projection\n(134K) → weights", "#f5b7b1", fs=8.5)
    box(ax, fig, 8.5, 1.85, 1.5, 0.9, "frozen 12-prompt\nlibrary", "#f5b7b1", fs=8.5)
    box(ax, fig, 10.2, 1.85, 1.4, 0.9, "FROZEN\nMusicGen", "#e6b0aa", fs=8.5)
    box(ax, fig, 10.2, 0.6, 1.4, 0.85, "32 kHz\nnative audio", "#f9e79f")
    arrow(ax, 8.3, 2.3, 8.5, 2.3); arrow(ax, 10.0, 2.3, 10.2, 2.3)
    arrow(ax, 10.9, 1.85, 10.9, 1.45)
    arrow(ax, 6.0, 2.95, 6.7, 2.3, color="#b03a2e")
    note = ax.text(6.7, 1.35, "argmax / B-cat / top-k mix\n(no condition interpolation)",
                   fontsize=7.5, color="#8e2b1f", ha="left", va="center")
    fig.canvas.draw()
    fit_text(fig, p2, note, margin=0.62)

    ax.set_title("PromptMix-EEG: decoupled alignment (retrieval) and generation (frozen prompt-mixing)",
                 fontsize=11, pad=14)
    save(fig, "fig0_architecture.png")


if __name__ == "__main__":
    main()
