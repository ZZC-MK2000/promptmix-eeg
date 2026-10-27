"""Pretrained foundation-model features vs our encoder on NMED-T."""
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

D = json.load(open(ROOT / "results" / "compare_foundation_models" / "summary.json"))


def main():
    rows = [
        ("Band-power 82x22", D["reference"]["bandpower_82x22"], PALETTE["grey"]),
        ("Encoder (6 datasets)", D["reference"]["encoder_6datasets"], PALETTE["blue"]),
        ("BIOT (pretrained)", D["models"]["biot"]["logistic"], PALETTE["green"]),
        ("LaBraM-200 (frozen)", D["reference"]["labram_200"], PALETTE["orange"]),
        ("CBraMod (pretrained)", D["models"]["cbramod"]["logistic"], PALETTE["purple"]),
    ]
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    y = np.arange(len(rows))
    vals = [v for _, v, _ in rows]
    cols = [c for _, _, c in rows]
    ax.barh(y, vals, color=cols, height=0.62, edgecolor="white")
    for i, v in enumerate(vals):
        ax.text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=9)
    ax.axvline(0.10, color="#555", ls="--", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels([n for n, _, _ in rows], fontsize=9)
    ax.invert_yaxis(); ax.set_xlim(0, 0.95)
    ax.set_xlabel("NMED-T 10-way logistic accuracy (frozen features)")
    ax.set_title("Pretrained foundation models vs the six-dataset encoder", fontsize=11,
                 weight="bold", loc="left")
    ax.text(0.11, 4.34, "chance", fontsize=7.5, color="#555", va="center")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig26_foundation_comparison.png")


if __name__ == "__main__":
    main()
