"""Fig 4 (CPU): EEG -> corpus-matched genre-prompt confusion matrix on held-out subjects.

Re-runs results/library_b/classifier.pt over the same group-level 10% split used at evaluation
(seed 0) and asserts the recomputed overall accuracy equals the checkpoint's entry in
results/prompt_classifier_true/genre_ci.json, so the figure cannot drift from the reported number.

Run: python code/figures/make_fig_genre_confusion.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
sys.path.insert(0, str(ROOT / "code" / "classifier"))
sys.path.insert(0, str(Path(__file__).parent))
from figstyle import TITLE_FS, save
from finetune_prompt_classifier import PromptClassifier

K = 12
SHORT = ["bolly-dance", "ind-ballad", "pop", "r&b", "edm", "acoustic",
         "funk", "reggaeton", "ambient", "rock", "trap", "orch-cue"]


def main():
    ci = json.load(open(ROOT / "results" / "prompt_classifier_true" / "genre_ci.json",
                        encoding="utf-8"))["corpus-matched lib"]
    cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)
    group = np.load(ROOT / "data" / "eegcond" / "group.npy").astype(np.int64)
    oracle = np.load(ROOT / "results" / "library_b" / "oracle_all.npy").astype(np.int64)
    uniq = np.unique(group); rng = np.random.default_rng(0)
    val_g = set(uniq[rng.permutation(len(uniq))[:max(1, int(round(len(uniq) * 0.1)))]].tolist())
    va = np.array([g in val_g for g in group])
    cond_v, y_v = cond[va], oracle[va]

    ck = torch.load(ROOT / "results" / "library_b" / "classifier.pt", map_location="cpu",
                    weights_only=True)
    model = PromptClassifier(512, ck["args"].get("hidden", 256), K, 0.1)
    model.load_state_dict(ck["model"]); model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(cond_v)).argmax(1).numpy()

    overall = float((pred == y_v).mean())
    assert len(y_v) == ci["n"], f"split does not match the reported evaluation: {len(y_v)} vs {ci['n']}"
    assert abs(overall - ci["overall"]) < 1e-6, \
        f"figure would not match the paper number: {overall:.6f} vs {ci['overall']:.6f}"
    conf = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_v, pred):
        conf[t, p] += 1

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(conf, cmap="Blues")
    thresh = conf.max() * 0.55
    for i in range(K):
        for j in range(K):
            if conf[i, j] > 0:
                ax.text(j, i, f"{conf[i, j]:,}", ha="center", va="center", fontsize=7.5,
                        color="white" if conf[i, j] > thresh else "#222")
    ax.set_xticks(range(K)); ax.set_xticklabels(SHORT, rotation=55, ha="right", fontsize=8)
    ax.set_yticks(range(K)); ax.set_yticklabels(SHORT, fontsize=8)
    ax.set_xlabel("predicted genre prompt", fontsize=9.5)
    ax.set_ylabel("true-audio oracle genre prompt", fontsize=9.5)
    ax.set_title(f"EEG → genre confusion, held-out subjects (corpus-matched library)\n"
                 f"overall {overall:.3f}  ·  Cohen's κ {ci['kappa']:.3f} "
                 f"[{ci['kappa_ci'][0]:.2f}, {ci['kappa_ci'][1]:.2f}]", fontsize=TITLE_FS)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("held-out windows", fontsize=8.5)
    fig.tight_layout(); save(fig, "fig4_genre_confusion.png")
    print(f"overall={overall:.4f} matches genre_ci.json  row-sums={conf.sum(1).tolist()}")


if __name__ == "__main__":
    main()
