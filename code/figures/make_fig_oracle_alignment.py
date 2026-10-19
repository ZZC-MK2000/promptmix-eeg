"""Fig 16 (CPU): the subject fiber — zero-shot vs oracle Procrustes and centroid geometry.

Reads results/oracle_alignment/oracle_alignment.json.

Run: python code/figures/make_fig_oracle_alignment.py
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
D = json.load(open(ROOT / "results" / "oracle_alignment" / "oracle_alignment.json", encoding="utf-8"))
REPS = [("bandpower", "band-power", PALETTE["green"]),
        ("labram", "LaBraM-200", PALETTE["blue"]),
        ("aldm", "alignment emb.\n(transductive)", PALETTE["grey"])]


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    ax = axes[0]
    w = 0.26
    x = np.arange(len(REPS))
    for j, (key, lab) in enumerate([("zero_shot_acc", "zero-shot"),
                                     ("procrustes_aligned_acc", "oracle Procrustes"),
                                     ("within_subject_acc", "within-subject")]):
        vals = [D["features"][k][key] for k, _, _ in REPS]
        cols = [PALETTE["grey"], PALETTE["green"], PALETTE["blue"]][j]
        ax.bar(x + (j - 1) * w, vals, w, color=cols, label=lab)
        for i, v in enumerate(vals):
            ax.text(i + (j - 1) * w, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.axhline(0.1, ls=":", c=PALETTE["ink"], lw=1, label="chance 0.10")
    ax.set_xticks(x); ax.set_xticklabels([l for _, l, _ in REPS], fontsize=8.5)
    ax.set_ylabel("10-way song accuracy"); ax.set_ylim(0, 1)
    ax.set_title("A · one per-subject rotation recovers band-power decoding", fontsize=TITLE_FS)
    ax.legend(fontsize=7.5, ncol=2)

    ax = axes[1]
    for j, (key, lab, col) in enumerate(REPS):
        before = D["features"][key]["centroid_cos_before"]
        after = D["features"][key]["centroid_cos_after"]
        ax.bar(j - 0.18, before, 0.34, color=col, alpha=0.35,
               label="before" if j == 0 else None)
        ax.bar(j + 0.18, after, 0.34, color=col,
               label="after Procrustes" if j == 0 else None)
        ax.text(j - 0.18, before + 0.02, f"{before:.2f}", ha="center", fontsize=7.5)
        ax.text(j + 0.18, after + 0.02, f"{after:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(REPS))); ax.set_xticklabels([l for _, l, _ in REPS], fontsize=8.5)
    ax.set_ylabel("class-centroid cosine"); ax.set_ylim(0, 1.05)
    ax.set_title("B · the same songs occupy the same geometry up to a rotation", fontsize=TITLE_FS)
    ax.legend(fontsize=8)

    fig.suptitle("The subject fiber is real in band-power, partial in LaBraM-200", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "fig16_oracle_alignment.png")


if __name__ == "__main__":
    main()
