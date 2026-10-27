"""Frozen-feature probe runner for AEB.

Input npz keys: features [N,D] (or X), song [N], subject [N].
For each split in nmedt_folds.json, fit a per-subject logistic/centroid probe on
the training rows and evaluate the held-out rows. Use the same feature matrix
for every model; this is the primary AEB readout.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    X = d["features"] if "features" in d else d["X"]
    return np.asarray(X, np.float32), d["song"].astype(int), d["subject"].astype(str)


def per_subject(X, y, subj, tr, te, kind="logistic"):
    accs = []
    for s in sorted(set(subj.tolist())):
        m_tr = tr[subj[tr] == s]; m_te = te[subj[te] == s]
        if len(m_tr) == 0 or len(m_te) == 0:
            continue
        if kind == "logistic":
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
            clf.fit(X[m_tr], y[m_tr]); pred = clf.predict(X[m_te])
        else:
            C = np.stack([X[m_tr][y[m_tr] == c].mean(0) for c in range(10)])
            pred = np.argmax(X[m_te] @ C.T, axis=1)
        accs.append(float((pred == y[m_te]).mean()))
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--split", default="splits/nmedt_folds.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    X, song, subj = load_npz(args.features)
    d = json.load(open(args.split))
    res = {"features": args.features, "split": args.split, "models": {}}
    for name, seeds in d["splits"].items():
        res["models"][name] = []
        for sp in seeds:
            tr = np.asarray(sp["train"]); te = np.asarray(sp["test"])
            l = per_subject(X, song, subj, tr, te, "logistic")
            c = per_subject(X, song, subj, tr, te, "centroid")
            res["models"][name].append({
                "seed": sp["seed"],
                "logistic_mean": float(np.mean(l)), "logistic_sd": float(np.std(l, ddof=1)),
                "centroid_mean": float(np.mean(c)), "per_subject_logistic": l,
                "chance": 0.10})
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    for name, vals in res["models"].items():
        print(name, np.mean([v["logistic_mean"] for v in vals]),
              np.mean([v["centroid_mean"] for v in vals]))


if __name__ == "__main__":
    main()
