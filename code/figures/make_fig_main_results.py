"""Main results: NMED-T retrieval and CLAP semantic geometry."""
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

R = ROOT / "results"


def load(path):
    return json.load(open(path, encoding="utf-8"))


def main():
    v2 = load(R / "spatial_spectral_k32" / "trialwise_nmed_logistic.json")["features"]
    v2t = load(R / "spatial_spectral_k32" / "semantic_trial.json")["features"]
    v2s = load(R / "spatial_spectral_k32" / "semantic_subject.json")["features"]
    six = [load(R / "six_dataset" / f"s{i}" / "trialwise_nmed.json")["features"]
           for i in range(3)]
    sixt = [load(R / "six_dataset" / f"s{i}" / "semantic_trial.json")["features"]
            for i in range(3)]
    sixs = [load(R / "six_dataset" / f"s{i}" / "semantic_subject.json")["features"]
            for i in range(3)]
    six_log = np.array([x["spatial_spectral"]["logistic"] for x in six])
    six_t_cos = np.array([x["spatial_spectral"]["cos"] for x in sixt])
    six_t_top = np.array([x["spatial_spectral"]["top1"] for x in sixt])
    six_s_cos = np.array([x["spatial_spectral"]["cos"] for x in sixs])
    six_s_top = np.array([x["spatial_spectral"]["top1"] for x in sixs])

    methods = ["LaBraM", "Bandpower 82x22", "Encoder (6 datasets)"]
    cols = [PALETTE["orange"], PALETTE["grey"], PALETTE["blue"], PALETTE["green"]]
    log = [v2["labram"]["logistic"], v2["bandpower"]["logistic"], six_log.mean()]
    log_sd = [v2["labram"]["logistic_sd"], v2["bandpower"]["logistic_sd"], six_log.std(ddof=1)]
    tcos = [v2t["labram"]["cos"], v2t["bandpower"]["cos"], six_t_cos.mean()]
    scos = [v2s["labram"]["cos"], v2s["bandpower"]["cos"], six_s_cos.mean()]
    ttop = [v2t["labram"]["top1"], v2t["bandpower"]["top1"], six_t_top.mean()]
    stop = [v2s["labram"]["top1"], v2s["bandpower"]["top1"], six_s_top.mean()]

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3),
                             gridspec_kw={"wspace": 0.32})
    ax = axes[0]
    y = np.arange(len(methods))
    ax.barh(y, log, xerr=log_sd, color=cols, height=0.62,
            error_kw={"lw": 0.8, "capsize": 2})
    for i, v in enumerate(log):
        ax.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8.5)
    ax.axvline(0.10, color="#555", ls="--", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(methods, fontsize=8.6)
    ax.set_xlim(0, 1.06); ax.set_xlabel("NMED-T 10-way logistic accuracy")
    ax.set_title("a  Closed-set decoding", fontsize=10.5, weight="bold", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    x = np.arange(len(methods)); w = 0.36
    ax.bar(x - w / 2, tcos, w, color=cols, alpha=0.9, label="trial-wise")
    ax.bar(x + w / 2, scos, w, color=cols, alpha=0.45, hatch="//",
           edgecolor="white", label="held-out subjects")
    for i in range(len(methods)):
        ax.text(x[i] - w / 2, tcos[i] + 0.005, f"{tcos[i]:.2f}",
                ha="center", fontsize=7.6)
        ax.text(x[i] + w / 2, scos[i] + 0.005, f"{scos[i]:.2f}",
                ha="center", fontsize=7.6)
    ax.set_xticks(x); ax.set_xticklabels(["LaBraM", "Bandpower", "Encoder"],
                                         fontsize=8.2)
    ax.set_ylim(0.6, 0.85); ax.set_ylabel("CLAP cosine")
    ax.set_title("b  Semantic geometry under subject shift", fontsize=10.5,
                 weight="bold", loc="left")
    ax.legend(frameon=False, fontsize=7.2, loc="upper center",
              bbox_to_anchor=(0.5, -0.18), ncol=2)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[2]
    ax.bar(x - w / 2, ttop, w, color=cols, alpha=0.9)
    ax.bar(x + w / 2, stop, w, color=cols, alpha=0.45, hatch="//", edgecolor="white")
    for i in range(len(methods)):
        ax.text(x[i] - w / 2, ttop[i] + 0.008, f"{ttop[i]:.2f}",
                ha="center", fontsize=7.6)
        ax.text(x[i] + w / 2, stop[i] + 0.008, f"{stop[i]:.2f}",
                ha="center", fontsize=7.6)
    ax.axhline(0.10, color="#555", ls="--", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(["LaBraM", "Bandpower", "Encoder"],
                                         fontsize=8.2)
    ax.set_ylim(0, 0.78); ax.set_ylabel("CLAP top-1")
    ax.set_title("c  Retrieval top-1: subject shift is the wall", fontsize=10.5,
                 weight="bold", loc="left")
    ax.axhline(0.10, color="#555", ls="--", lw=0.8, label="chance")
    ax.legend(frameon=False, fontsize=7.2, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig23_main_results.png")


if __name__ == "__main__":
    main()
