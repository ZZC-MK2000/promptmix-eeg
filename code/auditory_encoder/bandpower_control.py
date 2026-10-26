"""Bandpower control on the CIRE held-out subjects (same 10-way protocol)."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Corpus  # noqa: E402

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def bandpower(windows, rows):
    x = np.stack([windows[i][:, :1000] for i in rows]).astype(np.float32)
    x = x - x.mean(-1, keepdims=True)
    p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
    f = np.fft.rfftfreq(x.shape[-1], d=1 / 200)
    feats = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        feats.append(np.log1p(p[..., m].mean(-1)))
    return np.concatenate(feats, axis=1)  # [n, 19*5]


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/Projects/CLaBraM/mainv2/data/auditory_corpus")
    ap.add_argument("--out", default="D:/Projects/CLaBraM/promptmix-eeg/results/auditory_encoder/bandpower_control.json")
    args = ap.parse_args()
    corpus = Corpus(Path(args.root))
    held = [s for s in corpus.subjects if s.startswith("CIRE:")][:5]
    train = [s for s in corpus.train_subjects if s.startswith("CIRE:")]
    stimuli = corpus.ds_stims["cire"][:10]
    rows, labels = [], []
    for s in stimuli:
        for u in train[:20]:
            r = corpus.rows.get((u, s))
            if r:
                rows.append(int(r[0])); labels.append(s)
    Xtr = bandpower(corpus.windows, rows)
    book = {}
    for s in stimuli:
        m = np.array([l == s for l in labels])
        book[s] = unit(Xtr[m].mean(0))
    C = np.stack([book[s] for s in stimuli])
    res = {"zero": [], "subject_probe": None}
    for u in held:
        rr = corpus.rows_for(u, stimuli)
        if rr is None:
            continue
        Xq = bandpower(corpus.windows, rr)
        pred = (unit(Xq) @ C.T).argmax(1)
        res["zero"].append(float((pred == np.arange(len(stimuli))).mean()))
    # subject probe on training subjects (within-training sanity)
    if train:
        rows_u, y = [], []
        for i, u in enumerate(train):
            for s in stimuli:
                r = corpus.rows.get((u, s))
                if r:
                    rows_u.append(int(r[0])); y.append(i)
        if len(set(y)) > 2:
            Xp = bandpower(corpus.windows, rows_u)
            clf = LogisticRegression(max_iter=2000)
            res["subject_probe"] = float(cross_val_score(clf, Xp, y, cv=5).mean())
    res["zero_mean"] = float(np.mean(res["zero"])) if res["zero"] else float("nan")
    res["chance"] = 0.1
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
