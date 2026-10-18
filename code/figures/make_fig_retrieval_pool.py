"""Fig 9 (CPU): retrieval top-1 as the candidate pool grows from the 14 true songs to thousands
of FMA distractors, against the 1/pool random baseline.

Run: python code/figures/make_fig_retrieval_pool.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
sys.path.insert(0, str(Path(__file__).parent))
from figstyle import TITLE_FS, PALETTE, save


def main():
    r = json.load(open(ROOT / "results" / "retrieval_hard" / "retrieval_hard.json", encoding="utf-8"))
    pts = sorted(r["curve"], key=lambda d: d["pool"])
    pools = [p["pool"] for p in pts]
    top1 = [p["top1"] for p in pts]
    rand = [p["random"] for p in pts]
    best_ratio = max(t / d for t, d in zip(top1, rand))

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    ax.plot(pools, top1, "-o", color=PALETTE["blue"], lw=2, ms=6, label="EEG top-1 song retrieval")
    ax.plot(pools, rand, "--", color=PALETTE["grey"], lw=1.4, label="random chance 1/pool")
    for p, t in zip(pools, top1):
        ax.annotate(f"{t:.2f}", (p, t), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    ax.set_xscale("log"); ax.set_ylim(0, 1.0)
    ax.set_xlabel("candidate pool size (true songs + FMA distractors)")
    ax.set_ylabel("top-1 accuracy")
    ax.set_title(f"Retrieval survives a big pool: {top1[-1]:.1%} over {pools[-1]:,} candidates\n"
                 f"({best_ratio:.0f}x random), {r['n_queries']:,} held-out queries, "
                 f"{r['n_subjects']} subjects", fontsize=TITLE_FS)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); save(fig, "fig9_retrieval_pool.png")
    print("pools", pools, "top1", [round(t, 4) for t in top1])


if __name__ == "__main__":
    main()
