"""Fig 0 (CPU): the two-part architecture diagram for the paper.
Part 1 (Alignment): EEG -> frozen LaBraM -> ridge bridge -> laion-CLAP -> RETRIEVE heard song.
Part 2 (Generation): bridge(EEG) -> tiny projection (mixture weights over a frozen 12-prompt
library) -> pick/mix prompts -> FROZEN MusicGen native sampling -> pleasant 32 kHz audio.
Drawn with matplotlib boxes/arrows only (no external deps).

Run: python src/train/make_fig_architecture.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG = "D:/Projects/CLaBraM/paper/figures"


def box(ax, x, y, w, h, text, fc, fs=9, tc="black"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                fc=fc, ec="#333", lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, wrap=True)


def arrow(ax, x1, y1, x2, y2, color="#333", style="-|>", lw=1.6):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14,
                                 color=color, lw=lw))


def main():
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.set_xlim(0, 11); ax.set_ylim(0, 5.6); ax.axis("off")

    # shared input
    box(ax, 0.2, 2.5, 1.3, 0.9, "8-s EEG\nwindow", "#e8f0fe", fs=9)
    box(ax, 1.9, 2.5, 1.5, 0.9, "frozen\nLaBraM encoder", "#d2e3fc", fs=9)
    box(ax, 3.8, 2.5, 1.5, 0.9, "ridge bridge\n-> laion-CLAP c", "#d2e3fc", fs=9)
    arrow(ax, 1.5, 2.95, 1.9, 2.95); arrow(ax, 3.4, 2.95, 3.8, 2.95)

    # Part 1 (alignment / retrieval) -- top branch
    ax.add_patch(FancyBboxPatch((5.8, 3.6), 5.0, 1.7, boxstyle="round,pad=0.02",
                                fc="#e6f4ea", ec="#2b7a2b", lw=1.6))
    ax.text(6.0, 5.05, "PART 1 — ALIGNMENT  (goal: retrieve the heard song)", fontsize=10,
            color="#1b5e20", weight="bold")
    box(ax, 6.0, 3.9, 1.6, 0.9, "kNN / rank in\nlaion space", "#c8e6c9", fs=9)
    box(ax, 7.9, 3.9, 1.5, 0.9, "match real\nsong", "#c8e6c9", fs=9)
    box(ax, 9.6, 3.9, 1.1, 0.9, "song-ID\n84%", "#a5d6a7", fs=9, tc="#1b5e20")
    arrow(ax, 7.6, 4.35, 7.9, 4.35); arrow(ax, 9.4, 4.35, 9.6, 4.35)
    arrow(ax, 5.3, 3.2, 6.0, 4.35, color="#2b7a2b")

    # Part 2 (generation) -- bottom branch
    ax.add_patch(FancyBboxPatch((5.8, 0.3), 5.0, 2.7, boxstyle="round,pad=0.02",
                                fc="#fce8e6", ec="#b03a2e", lw=1.6))
    ax.text(6.0, 2.75, "PART 2 — GENERATION  (goal: pleasant music, by construction)", fontsize=10,
            color="#8e2b1f", weight="bold")
    box(ax, 6.0, 1.7, 1.5, 0.8, "tiny projection\n(134K) -> weights", "#f5b7b1", fs=8)
    box(ax, 7.7, 1.7, 1.4, 0.8, "frozen 12-prompt\nlibrary", "#f5b7b1", fs=8)
    box(ax, 9.3, 1.7, 1.3, 0.8, "FROZEN\nMusicGen", "#e6b0aa", fs=8)
    box(ax, 9.3, 0.5, 1.3, 0.8, "32 kHz\nnative audio", "#f9e79f", fs=9)
    arrow(ax, 7.5, 2.1, 7.7, 2.1); arrow(ax, 9.1, 2.1, 9.3, 2.1)
    arrow(ax, 9.95, 1.7, 9.95, 1.3)
    arrow(ax, 5.3, 2.7, 6.0, 2.1, color="#b03a2e")
    ax.text(6.05, 1.35, "argmax / B-cat / top-k mix\n(no condition interpolation)", fontsize=7.5,
            color="#8e2b1f")

    ax.set_title("PromptMix-EEG: decoupled alignment (retrieval) and generation (frozen prompt-mixing)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG + "/fig0_architecture.png", dpi=150)
    print("wrote", FIG + "/fig0_architecture.png")


if __name__ == "__main__":
    main()
