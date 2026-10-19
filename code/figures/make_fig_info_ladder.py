"""Fig 14 (CPU): the information ladder.

Reads results/info_ladder/info_ladder.json and draws four panels:
  A  split ladder      song accuracy under random / held-out-segment / held-out-subject
  B  representation    subject-held-out song accuracy per EEG representation (+ audio ceiling)
  C  continuous        cosine to true audio-CLAP vs to the song centroid (subject-held-out)
  D  label validity    the 12-class oracle collapses into one class on NMED-T

Run: python code/figures/make_fig_info_ladder.py
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
RES = json.load(open(ROOT / "results" / "info_ladder" / "info_ladder.json", encoding="utf-8"))

REPS = [("bandpower", "bandpower", PALETTE["grey"]),
        ("labram", "LaBraM", PALETTE["blue"]),
        ("aldm", "EEG-ALDM", PALETTE["purple"]),
        ("bridge", "bridge (ours)", PALETTE["green"])]
SPLITS = [("split_random", "random rows"), ("split_segment", "held-out\nsegments"),
          ("split_subject", "held-out\nsubjects")]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.2))
    ax = axes[0][0]
    w = 0.2
    for j, (key, lab, col) in enumerate(REPS):
        vals = [RES["nmedt"][s][key]["song"]["acc"] for s, _ in SPLITS]
        ax.bar(np.arange(3) + (j - 1.5) * w, vals, w, color=col, label=lab)
    ax.axhline(0.1, ls="--", c=PALETTE["ink"], lw=1, label="chance 0.10")
    ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in SPLITS], fontsize=9)
    ax.set_ylabel("10-way song accuracy"); ax.set_ylim(0, 1)
    ax.set_title("A · split ladder: within-subject → cross-subject", fontsize=TITLE_FS)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[0][1]
    vals = [RES["nmedt"]["split_subject"][k]["song"]["acc"] for k, _, _ in REPS]
    ax.bar(range(len(REPS)), vals, color=[c for _, _, c in REPS])
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8.5)
    ceil = RES["nmedt"]["audio_clap_ceiling"]["subject"]["acc"]
    ax.axhline(ceil, ls="--", c=PALETTE["orange"], lw=1.2, label=f"audio-CLAP ceiling {ceil:.3f}")
    ax.axhline(0.1, ls=":", c=PALETTE["ink"], lw=1, label="chance 0.10")
    ax.set_xticks(range(len(REPS))); ax.set_xticklabels([l for _, l, _ in REPS], fontsize=8.5)
    ax.set_ylim(0, 1); ax.set_ylabel("held-out-subject song accuracy")
    ax.set_title("B · representation ladder (probe CV)", fontsize=TITLE_FS)
    ax.legend(fontsize=8)
    ax.text(0.02, 0.04, "aldm/bridge encoder was trained on all subjects → transductive",
            transform=ax.transAxes, fontsize=7.5, color=PALETTE["orange"])

    ax = axes[1][0]
    w = 0.36
    for j, (key, lab, col) in enumerate(REPS):
        full = RES["nmedt"]["split_subject"][key]["continuous"]["cos_target"]
        base = RES["nmedt"]["split_subject"][key]["continuous"]["cos_global_mean_baseline"]
        ax.bar(j - w / 2, full, w, color=col, label="cos to true audio" if j == 0 else None)
        ax.bar(j + w / 2, base, w, color=col, alpha=0.35, hatch="//",
               label="trivial global-mean" if j == 0 else None)
    ax.set_xticks(range(len(REPS))); ax.set_xticklabels([l for _, l, _ in REPS], fontsize=8.5)
    ax.set_ylabel("cosine (held-out subjects)"); ax.set_ylim(0, 1)
    ax.set_title("C · continuous: mostly the trivial baseline", fontsize=TITLE_FS)
    ax.legend(fontsize=8)
    cont = RES["nmedt"]["split_subject"]["bridge"]["continuous"]
    txt = (f"bridge excess {cont['excess_cos_over_trivial']:+.3f} · "
           f"within-song residual {cont['cos_residual_within_song']:.3f}")
    if cont.get("within_song_top1") is not None:
        txt += f" · segment top-1 {cont['within_song_top1']:.3f} (ch {cont['within_song_chance']:.3f})"
    ax.text(0.02, 0.04, txt, transform=ax.transAxes, fontsize=7.5, color=PALETTE["ink"])

    ax = axes[1][1]
    dist = np.array(RES["label_validity"]["oracle_genre_distribution_nmedt"], float)
    dist /= dist.sum()
    ax.bar(range(len(dist)), dist, color=[PALETTE["red"] if v > 0.3 else PALETTE["grey_light"]
                                          for v in dist])
    ax.set_xticks(range(len(dist)))
    ax.set_xticklabels(["bolly", "ind-ball", "pop", "r&b", "edm", "acous", "funk",
                        "reggae", "amb", "rock", "trap", "orch"], rotation=45,
                       ha="right", fontsize=7.5)
    ax.set_ylabel("share of NMED-T rows"); ax.set_ylim(0, 0.8)
    ax.set_title("D · why genre is not a rung: the oracle collapses", fontsize=TITLE_FS)
    ax.text(0.98, 0.92, f"max class {dist.max():.0%}", transform=ax.transAxes,
            ha="right", fontsize=9, color=PALETTE["red"])

    fig.suptitle("Information ladder: within-subject song identity is strong; frozen features fall to "
                 "chance on unseen subjects", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save(fig, "fig14_info_ladder.png")


if __name__ == "__main__":
    main()
