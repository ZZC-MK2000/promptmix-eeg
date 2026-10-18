"""Generate paper figures from existing JSON results (CPU only, no GPU, English labels to avoid
missing-CJK-font tofu). Writes PNGs to paper/figures/.

Sources:
  results/coverage_unified/table.json              (coverage radius, one convention — run
                                                    code/radii/coverage_unified.py first)
  results/gen_eegcond/library_radius/radius.json   (tolerance sweep, zcr of real music)
  results/prompt_classifier/true_oracle_meta.json  (oracle label distribution / collapse)
  results/prompt_classifier_true/eval_true_oracle.json  (classifier pred-vs-true genre dist)

Run: python code/figures/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import TITLE_FS, PALETTE, save

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
R = json.load(open(ROOT / "results" / "gen_eegcond" / "library_radius" / "radius.json"))

# ---- Fig 1: coverage radius, from the single unified table (convention: 6 clips per prompt) ----
COV = json.load(open(ROOT / "results" / "coverage_unified" / "table.json", encoding="utf-8"))
LIB = {m["label"]: m for m in COV["libraries"]}
MAIN = LIB["corpus-matched 12-prompt library"]
ON = [m for m in COV["libraries"]
      if m["on_convention"] and not m["label"].startswith("original")]
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.8, 6.8))
labels = ["Convex\nmixture", "Best single\nclip", "Best prompt\nmean", "Uniform over\nprompts",
          "Global mean\n(trivial)"]
keys = ["mixture", "best_single_clip", "best_prompt_mean", "uniform_over_prompts", "global_mean"]
means = [MAIN[k] for k in keys]
ses = [MAIN["se"][k] for k in keys]
ax.bar(range(5), means, yerr=ses, capsize=4,
       color=[PALETTE["green"], PALETTE["blue"], PALETTE["blue"],
              PALETTE["grey_light"], PALETTE["grey_light"]])
ax.axhline(means[4], ls="--", c=PALETTE["grey"], lw=1, label=f"trivial {means[4]:.3f}")
for i, m in enumerate(means):
    ax.text(i, m + 0.012, f"{m:.3f}", ha="center", fontsize=10)
ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("CLAP cosine to held-out real window"); ax.set_ylim(0.5, 0.99)
ax.set_title(f"main library = {MAIN['prompts']} prompts x {MAIN['clips_per_prompt']} clips "
             f"= {MAIN['atoms']} clips", fontsize=10.5)
ax.legend(fontsize=9)
for m, c, mk in zip(ON, [PALETTE["green"], PALETTE["blue"]], ["o", "^"]):
    ax2.plot(m["prompts"], m["mixture"], mk, color=c, ms=9, label=m["label"].replace(" library", ""))
    ax2.annotate(f"{m['mixture']:.3f}", (m["prompts"], m["mixture"]), textcoords="offset points",
                 xytext=(0, 10), ha="center", fontsize=9, color=c)
ax2.plot([12, 24], [MAIN["mixture"], LIB["corpus-matched 24-prompt library"]["mixture"]],
         "-", c=PALETTE["green"], lw=1.2, alpha=0.55)
ax2.set_xlabel(f"prompts in the library (each {COV['libraries'][0]['clips_per_prompt']} clips)")
ax2.set_ylabel("convex-mixture cosine"); ax2.set_ylim(0.80, 0.94)
ax2.set_xticks(sorted({m["prompts"] for m in ON}))
ax2.set_title("growing the library, clips per prompt fixed", fontsize=10.5)
ax2.legend(fontsize=9, loc="lower right")
fig.suptitle(f"Coverage radius under one convention — {COV['targets']} held-out real 8-s windows, "
             f"6 clips per prompt", fontsize=TITLE_FS)
fig.tight_layout(rect=[0, 0, 1, 0.955]); save(fig, "fig1_coverage_radius.png")

# ---- Fig 2: tolerance radius (zcr vs alpha per pair) ----
pairs = sorted(set(tuple(s["pair"]) for s in R["sweep"]))
names = {0: "piano", 1: "indian", 2: "ambient", 3: "folk", 4: "dance", 5: "lofi",
         6: "orchestral", 7: "percussion", 8: "jazz", 9: "rock", 10: "choir", 11: "chiptune"}
fig, ax = plt.subplots(figsize=(5.6, 3.4))
for pa, c in zip(pairs, [PALETTE["blue"], PALETTE["orange"], PALETTE["purple"]]):
    xs = [s["alpha"] for s in R["sweep"] if tuple(s["pair"]) == pa]
    ys = [s["zcr"] for s in R["sweep"] if tuple(s["pair"]) == pa]
    ax.plot(xs, ys, "-o", color=c, lw=2, ms=5,
            label=f"{names[pa[0]]} <-> {names[pa[1]]}")
ax.axhspan(0.02, 0.05, color=PALETTE["green"], alpha=0.12, label="frozen native band")
ax.axhline(R["zcr_truth"], ls="--", c=PALETTE["ink"], lw=1, label=f"real music {R['zcr_truth']:.3f}")
ax.set_xlabel("interpolation weight alpha"); ax.set_ylabel("zero-crossing rate")
ax.set_title("Tolerance radius: T5-embedding interpolation goes off-manifold", fontsize=TITLE_FS)
ax.legend(fontsize=7)
fig.tight_layout(); save(fig, "fig2_tolerance_radius.png")

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
    ax.bar(x, d, color=[PALETTE["red"] if v > 0.3 * d.sum() else PALETTE["grey_light"] for v in d])
    if pred_dist is not None:
        ax.plot(x, pred_dist, "-o", color=PALETTE["blue"], lw=1.5, ms=4, label="classifier predicted")
        ax.plot(x, d, "-s", color=PALETTE["green"], lw=1.5, ms=4, label="true-audio oracle")
        ax.legend(fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(list(names.values()), rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("# held-out windows")
    ax.set_title("Oracle genre distribution: collapse onto chiptune (library-corpus mismatch)",
                 fontsize=TITLE_FS)
    fig.tight_layout(); save(fig, "fig3_oracle_collapse.png")

print("fig1-3 written")
