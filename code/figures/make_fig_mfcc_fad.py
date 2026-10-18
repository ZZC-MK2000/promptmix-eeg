"""Fig 10 (CPU): texture-aware Fréchet distance (20 MFCCs) from each generation arm to the real
heard windows, read from results/mfcc_fad_paired/paired.json.

That file is produced by code/evaluation/mfcc_fad_paired.py, which puts every arm on the same 60
stratified rows and gives each arm exactly one checkpoint. The earlier figure blended four
checkpoints into the "fine-tuned" arm and used a one-subject prefix for the frozen arm, which
inflated the ratio to 8.6x / 9.9x.

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
from figstyle import TITLE_FS, PALETTE, save

ARMS = [("frozen_native", "frozen native\n(ours)", PALETTE["green"]),
        ("finetuned_s12000", "fine-tuned\nmixed s12000", PALETTE["orange"]),
        ("finetuned_ep16", "fine-tuned\nk8 ep16", PALETTE["amber"])]


def main():
    d = json.load(open(ROOT / "results" / "mfcc_fad_paired" / "paired.json", encoding="utf-8"))
    arms = [(k, lab, c) for k, lab, c in ARMS if k in d["arms"]]
    vals = [d["arms"][k]["fd_to_real"] for k, _, _ in arms]
    ratio = vals[1] / vals[0]

    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    bars = ax.bar(range(len(arms)), vals, color=[c for _, _, c in arms])
    for b, (k, _, _), v in zip(bars, arms, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + max(vals) * 0.02,
                f"{v:,.0f} (n={d['arms'][k]['n']})", ha="center", fontsize=9)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([lab for _, lab, _ in arms], fontsize=9)
    ax.set_ylabel("MFCC-Fréchet distance to real music  (lower = closer)")
    ax.set_ylim(0, max(vals) * 1.16)
    p0 = d["paired"][0]
    ax.set_title(f"One checkpoint per arm, same {d['n_rows']} stratified windows\n"
                 f"frozen {ratio:.1f}x closer on Fréchet, but per-window the fine-tuned arm is not worse",
                 fontsize=TITLE_FS)
    ax.annotate(f"paired per-window Mahalanobis: {p0['mean_delta']:+.1f} "
                f"(t={p0['t']:.2f}, CI [{p0['ci95'][0]:+.1f},{p0['ci95'][1]:+.1f}])",
                xy=(0.5, -0.34), xycoords="axes fraction", ha="center", fontsize=8, color=PALETTE["ink"])
    fig.tight_layout(); save(fig, "fig10_mfcc_fad.png")
    print("fd", {k: round(v["fd_to_real"], 1) for k, v in d["arms"].items()}, "ratio", round(ratio, 2))


if __name__ == "__main__":
    main()
