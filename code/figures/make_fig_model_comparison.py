"""Supervised EEG baselines vs frozen encoders on NMED-T trial-wise splits."""
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

D = json.load(open(ROOT / "results" / "compare_nmed_models" / "summary.json"))


def main():
    names = ["EEGNet", "ShallowConvNet", "Deep4Net", "EEG-Conformer"]
    keys = ["eegnet", "shallow", "deep4", "conformer"]
    x = np.arange(len(names)); w = 0.34
    per = [D["models"][k]["per_subject"]["mean"] for k in keys]
    per_sd = [D["models"][k]["per_subject"]["sd"] for k in keys]
    pool = [D["models"][k]["pooled"]["mean"] for k in keys]
    pool_sd = [D["models"][k]["pooled"]["sd"] for k in keys]

    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.bar(x - w / 2, per, w, yerr=per_sd, color=PALETTE["blue"], label="per-subject",
           error_kw={"lw": 0.8, "capsize": 2})
    ax.bar(x + w / 2, pool, w, yerr=pool_sd, color=PALETTE["green"], label="pooled training",
           error_kw={"lw": 0.8, "capsize": 2})
    for i in range(len(names)):
        ax.text(x[i] - w / 2, per[i] + 0.02, f"{per[i]:.2f}", ha="center", fontsize=8)
        ax.text(x[i] + w / 2, pool[i] + 0.02, f"{pool[i]:.2f}", ha="center", fontsize=8)

    band = D["frozen"]["bandpower_82x22"]["acc"]
    enc = D["frozen"]["encoder_6datasets"]["acc"]
    lab = D["frozen"]["labram"]["acc"]
    ax.axhline(band, color=PALETTE["grey"], ls="--", lw=1.0, label=f"Bandpower {band:.2f}")
    ax.axhline(enc, color=PALETTE["blue"], ls="-.", lw=1.0, label=f"Encoder (6 datasets) {enc:.2f}")
    ax.axhline(lab, color=PALETTE["orange"], ls=":", lw=1.0, label=f"LaBraM {lab:.2f}")
    ax.axhline(0.10, color="#555", ls="--", lw=0.7, label="chance")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.0); ax.set_ylabel("NMED-T 10-way accuracy")
    ax.set_title("Supervised baselines vs frozen encoders (same trial-wise splits)",
                 fontsize=11, weight="bold", loc="left")
    ax.legend(frameon=False, fontsize=7.5, ncol=2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig25_model_comparison.png")


if __name__ == "__main__":
    main()
