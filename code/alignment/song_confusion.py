"""Fig 11: 14-way SONG-retrieval confusion matrix for the ALIGNMENT part.

For each held-out-subject EEG window, retrieve the nearest of the 14 song centroids (laion-CLAP of
the heard audio) via the ridge bridge, and build a 14x14 confusion matrix (true song x retrieved
song). This is the alignment-side counterpart to fig4 (which is the genre confusion). Cell colours
are row-normalised (recall per true song); printed numbers are raw window counts.

Needs the mainv2 workspace for the gate_nmedh14 npz + the aldm EEG embeddings (external to this repo).

Run: python code/alignment/song_confusion.py   (from promptmix-eeg/, CPU)
"""
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
sys.path.insert(0, str(ROOT / "code" / "figures"))
from figstyle import TITLE_FS, save


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    group = np.load(DATA / "group.npy").astype(np.int64)
    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    target = z["target"].astype(np.float32); song = z["song"].astype(int)
    eeg_aldm = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    proj = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu", weights_only=True)
    W = proj["weight"].numpy().astype(np.float32); b = proj["bias"].numpy().astype(np.float32)
    cond_all = unit(eeg_aldm @ W.T + b).astype(np.float32)
    key = {cond_all[i].tobytes(): i for i in range(len(cond_all))}
    q_song = np.full(len(cond), -1, np.int64)
    for j in range(len(cond)):
        i = key.get(cond[j].tobytes(), -1)
        if i >= 0:
            q_song[j] = song[i]
    ok = q_song >= 0

    uniq = np.unique(group); rng = np.random.default_rng(0)
    vg = set(uniq[rng.permutation(len(uniq))[:max(1, int(round(len(uniq) * 0.1)))]].tolist())
    va = np.array([g in vg for g in group]) & ok
    Q = unit(cond[va]); y = q_song[va]

    cent = unit(np.stack([unit(target[song == s]).mean(0) for s in range(14)]))
    pred = (Q @ cent.T).argmax(1)
    C = np.zeros((14, 14), int)
    for t, p in zip(y, pred):
        C[t, p] += 1
    acc = float((pred == y).mean())
    print(f"held-out windows={len(y)}  14-way song-retrieval top-1 acc={acc:.4f}")
    print("per-song recall:", [round(float(C[i, i] / max(C[i].sum(), 1)), 3) for i in range(14)])

    Cn = C / np.maximum(C.sum(1, keepdims=True), 1)
    labels = [f"T{i}" if i <= 9 else f"H{i-10}" for i in range(14)]
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.imshow(Cn, cmap="Blues", vmin=0, vmax=1)
    for i in range(14):
        for j in range(14):
            if C[i, j] > 0:
                ax.text(j, i, str(C[i, j]), ha="center", va="center", fontsize=7,
                        color=("white" if Cn[i, j] > 0.5 else "black"))
    ax.set_xticks(range(14)); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(14)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("retrieved song"); ax.set_ylabel("true (heard) song")
    ax.set_title(f"14-way song retrieval, held-out subjects: top-1 {acc:.3f}\n"
                 f"T0-T9 = NMED-T songs, H0-H3 = NMED-H (Bollywood) songs", fontsize=TITLE_FS)
    fig.tight_layout(); save(fig, "fig11_song_confusion.png")


if __name__ == "__main__":
    main()
