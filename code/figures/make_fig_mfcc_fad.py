"""Fig 10 (CPU): texture-aware Fréchet distance (20 MFCCs) from each generation arm to the real
heard windows. This is the metric that separates frozen from fine-tuned where the semantic
CLAP-Fréchet distance tied them.

Run: python code/figures/make_fig_mfcc_fad.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
sys.path.insert(0, str(Path(__file__).parent))
from figstyle import TITLE_FS, save

ARMS = [("frozen_native", "frozen native\n(ours)", "#2b7a2b"),
        ("finetuned", "fine-tuned\n(LoRA)", "#e0a526"),
        ("interpolation", "T5 interpolation\n(off-manifold)", "#c0392b")]


def main():
    d = json.load(open(ROOT / "results" / "mfcc_fad.json", encoding="utf-8"))
    vals = [d[k]["fd_to_real"] for k, _, _ in ARMS]
    ratio = vals[1] / vals[0]

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    bars = ax.bar(range(len(ARMS)), vals, color=[c for _, _, c in ARMS])
    for b, k, v in zip(bars, [k for k, _, _ in ARMS], vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                f"{v:,.0f} (n={d[k]['n']})", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(ARMS)))
    ax.set_xticklabels([lab for _, lab, _ in ARMS], fontsize=9)
    ax.set_ylabel("MFCC-Fréchet distance to real music  (lower = closer)")
    ax.set_ylim(0, max(vals) * 1.15)
    ax.set_title(f"Texture metric separates the arms:\nfrozen is {ratio:.1f}x closer to real than "
                 f"fine-tuned", fontsize=TITLE_FS)
    fig.tight_layout(); save(fig, "fig10_mfcc_fad.png")
    print("fd", dict(zip([k for k, _, _ in ARMS], [round(v, 1) for v in vals])), "ratio", round(ratio, 2))


if __name__ == "__main__":
    main()
