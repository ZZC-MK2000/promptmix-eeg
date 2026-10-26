"""Protocol pitfalls: split ladder + MindMix-style answer-conditioned query audit."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
from figstyle import PALETTE, save  # noqa: E402

LADDER = ROOT / "promptmix-eeg" / "results" / "info_ladder" / "info_ladder.json"
AUDIT = ROOT / "_private" / "mindmix_audit_2026-10-24" / "MINDMIX_AUDIT.md"


def panel_split(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 6.2); ax.axis("off")
    ax.text(0.05, 5.85, "a  What the split actually holds out", fontsize=11,
            weight="bold", color=PALETTE["ink"], va="top")
    rows = [
        ("Adjacent windows", ["T", "E", "T", "E", "T", "E", "T", "E", "T", "E"], "same subject"),
        ("Held-out blocks", ["T"] * 7 + ["E"] * 3, "same subject"),
        ("Held-out subjects", ["T"] * 7 + ["E"] * 3, "different subject"),
    ]
    for r, (name, seq, note) in enumerate(rows):
        y = 4.55 - r * 1.35
        ax.text(0.05, y + 0.55, name, fontsize=9.5, weight="bold", va="center")
        ax.text(0.05, y + 0.22, note, fontsize=8, color="#555", va="center")
        for j, v in enumerate(seq):
            fc = "#c8e6c9" if v == "T" else "#ffcdd2"
            ax.add_patch(Rectangle((2.55 + j * 0.55, y), 0.5, 0.55, fc=fc,
                                   ec="#555", lw=0.7))
        ax.text(8.15, y + 0.28, "train / test",
                fontsize=8, color="#555", va="center")
    ax.text(2.55, 0.15, "green = train   red = test", fontsize=8, color="#555")


def panel_ladder(ax):
    d = json.load(open(LADDER, encoding="utf-8"))["nmedt"]
    splits = ["random", "segment", "subject"]
    names = ["bandpower", "labram", "aldm"]
    labels = ["Bandpower", "LaBraM", "EEG-ALDM"]
    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]]
    x = np.arange(len(splits)); w = 0.25
    for i, (n, lab, c) in enumerate(zip(names, labels, colors)):
        vals = [d[f"split_{s}"][n]["song"]["acc"] for s in splits]
        ax.bar(x + (i - 1) * w, vals, w, color=c, label=lab, edgecolor="white")
        for xx, v in zip(x + (i - 1) * w, vals):
            ax.text(xx, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.axhline(0.10, color="#555", ls="--", lw=0.8, label="chance (0.10)")
    ax.set_xticks(x); ax.set_xticklabels(["random rows", "held-out blocks", "held-out subjects"])
    ax.set_ylim(0, 1.0); ax.set_ylabel("NMED-T 10-way accuracy")
    ax.set_title("b  Most of the gap is subject leakage", fontsize=11, weight="bold", loc="left")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)


def panel_clap(ax):
    d = json.load(open(ROOT / "mainv2" / "results" / "align_ladder_result.json",
                       encoding="utf-8"))
    block = [v["ten_way"] for k, arr in d.items() for v in arr
             if k == "SH_ctrl_blocksplit_stock_s0"]
    subj = [v["ten_way"] for k, arr in d.items() for v in arr
            if k.startswith("SH_f") and "stock" in k]
    block = block[0] if block else np.nan
    ax.bar([0, 1], [block, np.mean(subj)], color=[PALETTE["green"], PALETTE["red"]],
           width=0.55, edgecolor="white")
    ax.scatter([1] * len(subj), subj, color="white", edgecolor=PALETTE["red"], zorder=3, s=28)
    ax.axhline(0.10, color="#555", ls="--", lw=0.8, label="chance (0.10)")
    ax.text(0, block + 0.03, f"{block:.2f}", ha="center", fontsize=9)
    ax.text(1, np.mean(subj) + 0.03, f"{np.mean(subj):.2f}", ha="center", fontsize=9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["held-out blocks", "held-out subjects"])
    ax.set_ylim(0, 0.8); ax.set_ylabel("CLAP 10-way"); ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("c  A strong within-subject aligner\ndoes not transfer across listeners",
                 fontsize=10.5, weight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)


def panel_audit(ax):
    vals = [1.0, 1.0, 1.0, 0.505, 0.509]
    labs = ["reported\nprotocol", "zero\nEEG", "shuffled\nEEG", "correct\npairwise", "EEG-only\nquery"]
    cols = [PALETTE["red"], PALETTE["red"], PALETTE["red"], PALETTE["grey"], PALETTE["grey"]]
    ax.bar(range(5), vals, color=cols, width=0.65, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.025, f"{v:.2f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="#555", ls="--", lw=0.8, label="pair chance (0.50)")
    ax.set_xticks(range(5)); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylim(0, 1.12); ax.set_ylabel("pair accuracy"); ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    ax.set_title("d  Answer-conditioned query: target audio is in the input",
                 fontsize=10.5, weight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)


def main():
    fig = plt.figure(figsize=(13.2, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0], hspace=0.38, wspace=0.25)
    ax0 = fig.add_subplot(gs[0, 0]); panel_split(ax0)
    ax1 = fig.add_subplot(gs[0, 1]); panel_ladder(ax1)
    ax2 = fig.add_subplot(gs[1, 0]); panel_clap(ax2)
    ax3 = fig.add_subplot(gs[1, 1]); panel_audit(ax3)
    save(fig, "fig20_protocol_pitfalls.png")


if __name__ == "__main__":
    main()
