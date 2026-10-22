"""Fig 18: three downstream paths under subject shift.

Song identification/retrieval (song-level), genre-conditioned generation (shared
classifier + Qwen genre prompt), and song-level description selection (k-shot text +
frozen generator; not audio reconstruction).
Numbers come from results/public/fig_three_paths.json.

Run: python code/figures/make_fig_three_paths.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import PALETTE, save
MUT = "#5a6478"

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
D = json.load(open(ROOT / "results" / "public" / "fig_three_paths.json", encoding="utf-8"))


def card(ax, x, y, w, h, title, color, lines, tag):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
                         linewidth=1.4, edgecolor=color, facecolor="white", zorder=2)
    ax.add_patch(box)
    ax.add_patch(FancyBboxPatch((x, y + h - 0.095), w, 0.095,
                                boxstyle="round,pad=0.012,rounding_size=0.018",
                                linewidth=0, facecolor=color, alpha=0.13, zorder=3))
    ax.text(x + 0.02, y + h - 0.032, title, fontsize=10.2, fontweight="bold", color=color,
            va="center", zorder=4)
    ax.text(x + 0.022, y + h - 0.068, tag, fontsize=8.2, color=color,
            va="center", zorder=4)
    yy = y + h - 0.125
    for i, (txt, bold) in enumerate(lines):
        ax.text(x + 0.022, yy, txt, fontsize=9.4 if bold else 8.8,
                fontweight="bold" if bold else "normal",
                color=PALETTE["ink"] if bold else MUT,
                va="top", zorder=4, linespacing=1.35)
        yy -= 0.085 if bold else 0.07


def main():
    fig, ax = plt.subplots(figsize=(13.4, 5.5))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    ax.text(0.5, 0.965, "Same listener, same frozen generator: three downstream tasks",
            ha="center", va="top", fontsize=14.5, fontweight="bold")
    ax.text(0.5, 0.915, "Only song identification / retrieval is song-level under subject shift",
            ha="center", va="top", fontsize=10, color=MUT)

    # input
    in_box = FancyBboxPatch((0.015, 0.35), 0.14, 0.30, boxstyle="round,pad=0.012,rounding_size=0.018",
                            linewidth=1.5, edgecolor=PALETTE["ink"], facecolor="#eef1f5", zorder=2)
    ax.add_patch(in_box)
    ax.text(0.085, 0.585, "8-s EEG window", ha="center", fontsize=11, fontweight="bold")
    ax.text(0.085, 0.535, "+ k-shot calibration", ha="center", fontsize=9.6)
    ax.text(0.085, 0.485, "10–300 labelled", ha="center", fontsize=9.0, color=MUT)
    ax.text(0.085, 0.445, "windows per listener", ha="center", fontsize=9.0, color=MUT)
    ax.text(0.085, 0.395, "held-out subjects", ha="center", fontsize=8.4, style="italic", color=PALETTE["ink"])

    centers = [(0.33, PALETTE["green"], "1 · Song identification", "song-level", [
        ("Calibrated query", False),
        (f"bandpower  k=10  {D['retrieval']['bp_10']:.3f}", True),
        (f"bandpower  k=100 {D['retrieval']['bp_100']:.3f}", True),
        (f"CLAP query {D['retrieval']['clap_10']:.3f} / {D['retrieval']['clap_100']:.3f}", False),
        (f"chance {D['retrieval_chance']:.2f}", False),
        ("→ nearest real clip", False),
    ]),
    (0.59, PALETTE["blue"], "2 · Genre-conditioned generation", "genre-level", [
        ("Shared 12-way classifier", False),
        (f"{D['genre']['mainline_acc']:.3f} acc · κ {D['genre']['mainline_kappa']:.3f}", True),
        ("→ Qwen genre prompt", False),
        ("→ frozen MusicGen", False),
        (f"k-shot κ {D['genre']['kshot_kappa']:.3f} (no gain)", False),
        (f"CLAP prompt {D['genre']['clap_prompt_acc']:.3f} vs {D['genre']['clap_majority']:.3f}", False),
    ]),
    (0.85, PALETTE["orange"], "3 · Song-level description selection", "text only", [
        ("k-shot description selection", False),
        (f"{D['song']['tag_k100']:.3f} tag · {D['song']['qwen_k100']:.3f} Qwen", True),
        (f"floor {D['song']['floor']:.3f}", False),
        ("→ frozen MusicGen", False),
        (f"generation +{D['song']['generation']['oracle_minus_samegenre']:.3f}, n.s.", False),
        ("→ retrieve a real clip instead", False),
    ])]

    ax.plot([0.155, 0.19], [0.50, 0.50], color=MUT, lw=1.6, zorder=1)
    ax.plot([0.19, 0.19], [0.50, 0.82], color=MUT, lw=1.6, zorder=1)
    ax.plot([0.19, 0.85], [0.82, 0.82], color=MUT, lw=1.6, zorder=1)
    for cx, color, title, tag, lines in centers:
        x, w = cx - 0.1175, 0.235
        card(ax, x, 0.20, w, 0.55, title, color, lines, tag)
        ax.add_patch(FancyArrowPatch((cx, 0.82), (cx, 0.755),
                                     arrowstyle="-|>", mutation_scale=16, lw=2.0,
                                     color=color, shrinkA=1, shrinkB=1, zorder=1))

    ax.text(0.5, 0.10,
            "Genre-conditioned generation: the shared classifier selects the prompt; k-shot is not used.\n"
            "Song-level description selection: k-shot selects the description; the text→audio interface remains genre-level.",
            ha="center", va="center", fontsize=9.2, color="#5a6478", linespacing=1.5)
    fig.tight_layout()
    save(fig, "fig18_three_paths.png")


if __name__ == "__main__":
    main()
