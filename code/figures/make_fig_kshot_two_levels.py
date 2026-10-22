"""Fig 17: k-shot is song-specific, not genre-specific.

Reads results/public/fig_kshot_two_levels.json (built by build_public_fig_data.py).
Left: 10-way song-description selection (tag vs Qwen), 5-fold held-out subjects.
Right: 12-way genre selection; the shared classifier is κ .399, k-shot stays ≤ .045,
and CLAP text-prompt matching is below the majority floor.

Run: python code/figures/make_fig_kshot_two_levels.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import PALETTE, TITLE_FS, save

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
D = json.load(open(ROOT / "results" / "public" / "fig_kshot_two_levels.json", encoding="utf-8"))
X = list(range(len(D["k"])))
XT = [str(k) for k in D["k"]]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), gridspec_kw={"width_ratios": [1.05, 1]})
    ax = axes[0]
    ax.plot(X, D["song"]["tag"], "-o", color=PALETTE["blue"], lw=2.2, ms=6,
            label="discriminative tag descriptions")
    ax.plot(X, D["song"]["qwen"], "-s", color=PALETTE["orange"], lw=2.2, ms=6,
            label="Qwen free-text captions")
    ax.axhline(D["song"]["majority"], ls="--", c=PALETTE["grey"], lw=1.1,
               label=f"majority floor {D['song']['majority']:.3f}")
    ax.axhline(D["song"]["chance"], ls=":", c=PALETTE["ink"], lw=1.1,
               label=f"chance {D['song']['chance']:.2f}")
    for y, c in ((D["song"]["tag"][2], PALETTE["blue"]), (D["song"]["qwen"][2], PALETTE["orange"])):
        ax.annotate(f"{y:.3f}", (X[2], y), textcoords="offset points", xytext=(8, 6),
                    fontsize=9, color=c, fontweight="bold")
    ax.set_xticks(X); ax.set_xticklabels(XT)
    ax.set_xlabel("k = labelled calibration windows per listener")
    ax.set_ylabel("10-way description top-1 (5-fold held-out subjects)")
    ax.set_ylim(0.07, 0.40)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7.6, loc="lower right")
    ax.set_title("Song level: k-shot unlocks the description", fontsize=TITLE_FS)

    ax = axes[1]
    ax.plot(X, D["genre"]["head_calib_kappa"], "-o", color=PALETTE["purple"], lw=2.1, ms=6,
            label="k-shot head + classifier")
    ax.plot(X, D["genre"]["clip_calib_kappa"], "-s", color=PALETTE["grey"], lw=1.5, ms=5,
            label="k-shot clip-only head")
    ax.axhline(D["genre"]["mainline_kappa"], ls="--", c=PALETTE["green"], lw=1.4,
               label=f"shared classifier {D['genre']['mainline_kappa']:.3f}")
    ax.set_xticks(X); ax.set_xticklabels(XT)
    ax.set_xlabel("k = labelled calibration windows per listener")
    ax.set_ylabel("Cohen's κ (12-way genre, 5 folds)")
    ax.set_ylim(-0.01, 0.46)
    ax.grid(alpha=0.22)
    ax.legend(fontsize=7.4, loc="upper right")
    ax.text(0.03, 0.10,
            f"CLAP matching of 12 Qwen genre prompts:\n"
            f"{D['genre']['clap_prompt_acc'][2]:.3f} acc vs "
            f"{D['genre']['clap_majority']:.3f} majority",
            transform=ax.transAxes, fontsize=8.0, color=PALETTE["red"],
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=PALETTE["red"], alpha=0.9))
    ax.set_title("Genre level: k-shot does not move κ", fontsize=TITLE_FS)

    fig.suptitle("K-shot calibration is song-specific, not genre-specific",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig17_kshot_two_levels.png")


if __name__ == "__main__":
    main()
