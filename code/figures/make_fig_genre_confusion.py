"""Fig 4 (CPU): EEG -> genre-prompt confusion on held-out subjects (5 folds).

Uses the exact mainline protocol: per fold, LogisticRegression on `cond` vs the
corpus-matched library_b/oracle_all labels, training subjects excluded; predictions
aggregated over the five held-out-subject folds. This replaces the earlier group-level
10% split, which kept subjects in training and reported κ .563.

Run: python code/figures/make_fig_genre_confusion.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, balanced_accuracy_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
GATE = ROOT.parent / "mainv2" / "data" / "gate_nmedh14_ov2_a.npz"
sys.path.insert(0, str(Path(__file__).parent))
from figstyle import TITLE_FS, save

FOLDS = {
    0: {"test": ["02", "03", "04", "05"], "val": ["06", "07"]},
    1: {"test": ["06", "07", "08", "09"], "val": ["04", "05"]},
    2: {"test": ["10", "11", "12", "13"], "val": ["06", "07"]},
    3: {"test": ["14", "15", "16", "17"], "val": ["08", "09"]},
    4: {"test": ["19", "20", "21", "23"], "val": ["10", "11"]},
}

K = 12
SHORT = ["bolly-dance", "ind-ballad", "pop", "r&b", "edm", "acoustic",
         "funk", "reggaeton", "ambient", "rock", "trap", "orch-cue"]


def main():
    z = np.load(GATE, allow_pickle=True)
    cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)
    aldm = np.load(ROOT.parent / "mainv2" / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    w = torch.load(ROOT.parent / "mainv2" / "results" / "eeg_to_clap_proj.pt",
                   map_location="cpu", weights_only=True)
    ca = aldm @ w["weight"].numpy().astype(np.float32).T + w["bias"].numpy().astype(np.float32)
    ca /= np.linalg.norm(ca, axis=1, keepdims=True) + 1e-8
    key = {ca[i].tobytes(): i for i in range(len(ca))}
    idx = np.array([key[cond[j].tobytes()] for j in range(len(cond))])
    subj = z["subject"][idx].astype(str)
    ds = z["dataset"][idx].astype(int)
    al = z["aligned8"][idx].astype(bool)
    oracle = np.load(ROOT / "results" / "library_b" / "oracle_all.npy").astype(np.int64)

    ys, ps = [], []
    for f, meta in FOLDS.items():
        train_subs = [s for s in np.unique(subj)
                      if s not in meta["test"] and s not in meta["val"] and s in set(subj[ds == 0])]
        train_rows = np.where((ds == 0) & np.isin(subj, train_subs))[0]
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(cond[train_rows], oracle[train_rows])
        for s in meta["test"]:
            rt = np.where(al & (subj == s))[0]
            if len(rt) < 100:
                continue
            ys.append(oracle[rt]); ps.append(clf.predict(cond[rt]))
    y = np.concatenate(ys); pred = np.concatenate(ps)
    overall = float((y == pred).mean())
    kappa = float(cohen_kappa_score(y, pred))
    bal = float(balanced_accuracy_score(y, pred))
    conf = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y, pred):
        conf[t, p] += 1
    print(f"held-out subjects, 5-fold: n={len(y)} overall={overall:.4f} "
          f"balanced={bal:.4f} kappa={kappa:.4f}", flush=True)

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
    ax.set_ylabel("true genre prompt", fontsize=9.5)
    ax.set_title(f"EEG → genre prompt, held-out subjects (5-fold)\n"
                 f"overall {overall:.3f}  ·  Cohen's κ {kappa:.3f}  ·  balanced {bal:.3f}",
                 fontsize=TITLE_FS)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03).set_label("held-out windows", fontsize=8.5)
    fig.tight_layout(); save(fig, "fig4_genre_confusion.png")
    print("wrote fig4_genre_confusion.png")


if __name__ == "__main__":
    main()
