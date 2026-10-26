"""Ablations and scaling: CLAP losses, subject-held-out wall, data scale."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from figstyle import PALETTE, save  # noqa: E402


def main():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.3),
                             gridspec_kw={"wspace": 0.34})

    # (a) CLAP loss arms on held-out blocks
    arms = ["InfoNCE", "+ dense", "+ song CE"]
    vals = [0.2231, 0.2516, 0.5079]
    ax = axes[0]
    ax.bar(range(3), vals, color=[PALETTE["blue"], PALETTE["green"], PALETTE["orange"]],
           width=0.6, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(3)); ax.set_xticklabels(arms, fontsize=8.5)
    ax.set_ylim(0, 0.65); ax.set_ylabel("CLAP 10-way, held-out blocks")
    ax.set_title("a  CLAP objective terms", fontsize=10.5, weight="bold", loc="left")
    ax.text(2.0, 0.60, "subspace anchor: .685\n(no anchor: .707)", fontsize=7.8,
            ha="right", color="#555")
    ax.spines[["top", "right"]].set_visible(False)

    # (b) subject-held-out folds
    folds = [0.1175, 0.0826, 0.1303, 0.1090, 0.0883]
    ax = axes[1]
    ax.bar(range(5), folds, color=PALETTE["red"], width=0.62, edgecolor="white")
    ax.axhline(0.10, color="#555", ls="--", lw=0.8, label="chance")
    for i, v in enumerate(folds):
        ax.text(i, v + 0.006, f"{v:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(5)); ax.set_xticklabels([f"fold {i}" for i in range(5)], fontsize=8.5)
    ax.set_ylim(0, 0.18); ax.set_ylabel("CLAP 10-way")
    ax.set_title("b  Held-out subjects: chance", fontsize=10.5, weight="bold", loc="left")
    ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    save(fig, "fig24_ablations.png")


if __name__ == "__main__":
    main()
