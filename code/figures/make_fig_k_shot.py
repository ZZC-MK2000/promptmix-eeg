"""Fig 15 (CPU): k-shot subject calibration, random vs active anchor selection.

Reads results/k_shot/k_shot_v2.json. Two panels (bandpower / LaBraM-200), log-x k,
lines for random, closest, FPS, hybrid, plus zero-shot / oracle / within-subject.

Run: python code/figures/make_fig_k_shot.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import PALETTE, TITLE_FS, save

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
D = json.load(open(ROOT / "results" / "k_shot" / "k_shot_v2.json", encoding="utf-8"))
KS = [int(k) for k in D["protocol"]["k"]]
PANELS = [("bandpower", "bandpower (frozen)"), ("labram", "LaBraM-200 (frozen)")]
LINES = [("random", "random anchors", PALETTE["grey"]),
         ("dopt_closest", "closest-to-centroid", PALETTE["blue"]),
         ("dopt_fps", "FPS within class", PALETTE["purple"]),
         ("dopt_hybrid", "hybrid (ours)", PALETTE["green"])]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3), sharey=True)
    for ax, (feat, title) in zip(axes, PANELS):
        f = D["features"][feat]
        for key, lab, col in LINES:
            y = [f["curve"][str(k)][key]["acc"] for k in KS]
            ax.plot(KS, y, "-o", color=col, lw=2 if key == "dopt_hybrid" else 1.5,
                    ms=5, label=lab)
        b = f["baselines"]
        ax.axhline(b["zero"], ls=":", c=PALETTE["ink"], lw=1.1, label=f"zero-shot {b['zero']:.3f}")
        ax.axhline(b["oracle"], ls="--", c=PALETTE["orange"], lw=1.2,
                   label=f"oracle (all fit rows) {b['oracle']:.3f}")
        ax.axhline(b["within"], ls="-.", c=PALETTE["grey"], lw=1.0,
                   label=f"within-subject {b['within']:.3f}")
        ax.set_xscale("log")
        ax.set_xticks(KS); ax.set_xticklabels([str(k) for k in KS], fontsize=8)
        ax.set_xlabel("k = labelled calibration windows")
        ax.set_ylim(0, 0.8)
        ax.set_title(title, fontsize=TITLE_FS)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("10-way song accuracy (held-out segments)")
    axes[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle("k-shot subject calibration: one window per song already beats chance; "
                 "active anchors win at small k", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig15_k_shot.png")


if __name__ == "__main__":
    main()
