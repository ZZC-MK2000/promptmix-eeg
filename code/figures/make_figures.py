"""Generate paper figures from existing JSON results (CPU only, no GPU, English labels to avoid
missing-CJK-font tofu). Writes PNGs to paper/figures/.

Sources:
  results/gen_eegcond/library_radius/radius.json   (coverage + tolerance)
  results/prompt_classifier/true_oracle_meta.json  (oracle label distribution / collapse)
  results/prompt_classifier_true/eval_true_oracle.json  (classifier pred-vs-true genre dist)

Run: python src/train/make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
FIG = ROOT.parent / "paper" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
R = json.load(open(ROOT / "results" / "gen_eegcond" / "library_radius" / "radius.json"))

# ---- Fig 1: coverage radius (per-target mean +- SE) ----
labels = ["Convex\nmixture", "Best single\nclip", "Best prompt\nmean", "Uniform over\nprompts",
          "Global mean\n(trivial)"]
keys = ["mixture_over_library", "best_single_clip", "best_single_prompt_mean",
        "uniform_over_prompts", "global_mean"]
means = [np.mean(R[k]["per_target"]) for k in keys]
ses = [np.std(R[k]["per_target"], ddof=1) / np.sqrt(len(R[k]["per_target"])) for k in keys]
fig, ax = plt.subplots(figsize=(6.2, 3.4))
cols = ["#2b7a2b", "#5aa9e6", "#5aa9e6", "#c0c0c0", "#c0c0c0"]
ax.bar(range(len(means)), means, yerr=ses, capsize=4, color=cols)
ax.axhline(means[4], ls="--", c="grey", lw=1, label=f"trivial {means[4]:.3f}")
for i, m in enumerate(means):
    ax.text(i, m + 0.015, f"{m:.3f}", ha="center", fontsize=9)
ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("CLAP cosine to held-out real window"); ax.set_ylim(0.5, 0.95)
ax.set_title("Coverage radius: library convex hull spans real 8-s windows")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIG / "fig1_coverage_radius.png", dpi=150); plt.close(fig)

# ---- Fig 2: tolerance radius (zcr vs alpha per pair) ----
pairs = sorted(set(tuple(s["pair"]) for s in R["sweep"]))
names = {0: "piano", 1: "indian", 2: "ambient", 3: "folk", 4: "dance", 5: "lofi",
         6: "orchestral", 7: "percussion", 8: "jazz", 9: "rock", 10: "choir", 11: "chiptune"}
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for pa, c in zip(pairs, ["#d62728", "#1f77b4", "#2ca02c"]):
    xs = [s["alpha"] for s in R["sweep"] if tuple(s["pair"]) == pa]
    ys = [s["zcr"] for s in R["sweep"] if tuple(s["pair"]) == pa]
    ax.plot(xs, ys, "-o", color=c, lw=2, ms=5,
            label=f"{names[pa[0]]} <-> {names[pa[1]]}")
ax.axhspan(0.02, 0.05, color="green", alpha=0.12, label="frozen native band")
ax.axhline(R["zcr_truth"], ls="--", c="k", lw=1, label=f"real music {R['zcr_truth']:.3f}")
ax.set_xlabel("interpolation weight alpha"); ax.set_ylabel("zero-crossing rate")
ax.set_title("Tolerance radius: T5-embedding interpolation goes off-manifold")
ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig2_tolerance_radius.png", dpi=150); plt.close(fig)

# ---- Fig 3: oracle label distribution / chiptune collapse ----
try:
    meta = json.load(open(ROOT / "results" / "prompt_classifier" / "true_oracle_meta.json"))
    dist = np.array(meta["distribution"])
except Exception:
    dist = None
try:
    ev = json.load(open(ROOT / "results" / "prompt_classifier_true" / "eval_true_oracle.json"))
    pred_dist = np.array(ev["pred_dist"]); true_dist = np.array(ev["true_dist"])
except Exception:
    pred_dist = true_dist = None

if dist is not None or true_dist is not None:
    d = dist if dist is not None else true_dist
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    x = np.arange(len(d))
    ax.bar(x, d, color=["#d62728" if v > 0.3 * d.sum() else "#999" for v in d])
    if pred_dist is not None:
        ax.plot(x, pred_dist, "-o", color="#1f77b4", lw=1.5, ms=4, label="classifier predicted")
        ax.plot(x, d, "-s", color="#2ca02c", lw=1.5, ms=4, label="true-audio oracle")
        ax.legend(fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(list(names.values()), rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("# held-out windows")
    ax.set_title("Oracle genre distribution: collapse onto chiptune (library-corpus mismatch)")
    fig.tight_layout(); fig.savefig(FIG / "fig3_oracle_collapse.png", dpi=150); plt.close(fig)

print("figures written to", FIG)
for f in sorted(FIG.glob("*.png")):
    print("  ", f.name)
