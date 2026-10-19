"""TTA / domain-adaptation baselines for cross-subject song decoding (CPU).

The paper must show that k-shot Procrustes calibration is not just "fine-tune on the
target subject". This script runs the standard alternatives on the SAME split:

  zero            no adaptation
  subj_zscore     per-subject feature z-score (uses unlabeled fit rows)
  ea_pca          Euclidean alignment in a 100-d PCA space (unlabeled fit rows)
  coral_pca       CORAL covariance alignment in the same space (unlabeled fit rows)
  probe_k         nearest-class-centroid trained on k labelled fit rows (supervised TTA)

For EA/CORAL the shared classifier is trained on the training subjects' aligned fit rows
and evaluated on the held-out subject's aligned test rows. PCA is unsupervised and fit on
all NMED-T rows (transductive, standard for TTA); labels are never used in it.

Run: python code/information/tta_baselines.py     (CPU)
"""
import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
OUT = ROOT / "results" / "tta"
SEED = 0
KS = [10, 20, 50, 100, 200]
N_DIM = 100
SHRINK = 1e-2


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def whiten(X, mu, cov):
    d = cov.shape[0]
    cov = (1 - SHRINK) * cov + SHRINK * (np.trace(cov) / d) * np.eye(d)
    evals, evecs = np.linalg.eigh(cov)
    evals = np.maximum(evals, 1e-8)
    W = (evecs / np.sqrt(evals)) @ evecs.T
    return (X - mu) @ W


def coral(X, mu_s, cov_s, mu_t, cov_t):
    d = cov_s.shape[0]
    cov_s = (1 - SHRINK) * cov_s + SHRINK * (np.trace(cov_s) / d) * np.eye(d)
    cov_t = (1 - SHRINK) * cov_t + SHRINK * (np.trace(cov_t) / d) * np.eye(d)

    def sqrtm(C):
        e, V = np.linalg.eigh(C); e = np.maximum(e, 1e-8)
        return (V * np.sqrt(e)) @ V.T

    W = np.linalg.inv(sqrtm(cov_s)) @ sqrtm(cov_t)
    return (X - mu_s) @ W + mu_t


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT.parent / "mainv2" / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)
    aldm = np.load(ROOT.parent / "mainv2" / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    import torch
    w = torch.load(ROOT.parent / "mainv2" / "results" / "eeg_to_clap_proj.pt",
                   map_location="cpu", weights_only=True)
    ca = unit(aldm @ w["weight"].numpy().astype(np.float32).T + w["bias"].numpy().astype(np.float32))
    key = {ca[i].tobytes(): i for i in range(len(ca))}
    idx = np.array([key[cond[j].tobytes()] for j in range(len(cond))])
    ds = z["dataset"][idx]; song = z["song"][idx].astype(int)
    subj = z["subject"][idx].astype(str)
    grp = np.load(ROOT / "data" / "eegcond" / "group.npy").astype(int)
    m = ds == 0
    songs = np.unique(song[m])
    song, subj, grp = song[m], subj[m], grp[m]
    feats = {"bandpower": z["bandpower"][idx][m].astype(np.float32),
             "labram": z["labram"][idx][m].astype(np.float32)}
    subjects = np.unique(subj)
    rng = np.random.default_rng(SEED)

    res = {"protocol": {"corpus": "NMED-T 20 subjects x 10 songs",
                        "split": "per-subject 50/50 by audio segment",
                        "pca_dim": N_DIM, "shrinkage": SHRINK,
                        "note": "EA/CORAL use unlabelled fit rows; probe_k uses k labels"},
           "features": {}}

    for name, X in feats.items():
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        pca = PCA(n_components=N_DIM, random_state=SEED).fit(X)
        Xp = pca.transform(X).astype(np.float64)
        split = {}
        for s in subjects:
            rows = np.where(subj == s)[0]
            gs = np.unique(grp[rows]); rng.shuffle(gs)
            fg = set(gs[:len(gs) // 2])
            split[s] = (rows[np.isin(grp[rows], list(fg))],
                        rows[~np.isin(grp[rows], list(fg))])
        rec = {"zero": [], "subj_zscore": [], "ea_pca": [], "coral_pca": [],
               **{f"probe_{k}": [] for k in KS}}
        for s in subjects:
            rf, rt = split[s]
            if len(rf) < 200 or len(rt) < 200:
                continue
            yte = song[rt]
            tr_fit = np.concatenate([split[t][0] for t in subjects if t != s])
            # zero-shot: shared logistic regression on other subjects' FIT rows
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xp[tr_fit], song[tr_fit])
            rec["zero"].append(float((clf.predict(Xp[rt]) == yte).mean()))
            # per-subject z-score
            Xs = Xp.copy()
            mu_s, sd_s = Xs[rf].mean(0), Xs[rf].std(0) + 1e-6
            Xs = (Xs - mu_s) / sd_s
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xs[tr_fit], song[tr_fit])
            rec["subj_zscore"].append(float((clf.predict(Xs[rt]) == yte).mean()))
            # EA in PCA space: whiten each subject, shared classifier
            mu_t, cov_t = Xp[tr_fit].mean(0), np.cov(Xp[tr_fit], rowvar=False)
            Xa = Xp.copy()
            for t in subjects:
                rf_t, _ = split[t]
                mu, cov = Xp[rf_t].mean(0), np.cov(Xp[rf_t], rowvar=False)
                Xa[rf_t] = whiten(Xp[rf_t], mu, cov)
                if t == s:
                    Xa[rt] = whiten(Xp[rt], mu, cov)
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xa[tr_fit], song[tr_fit])
            rec["ea_pca"].append(float((clf.predict(Xa[rt]) == yte).mean()))
            # CORAL: align each subject to the training mean/cov
            Xc = Xp.copy()
            for t in subjects:
                rf_t, _ = split[t]
                mu, cov = Xp[rf_t].mean(0), np.cov(Xp[rf_t], rowvar=False)
                Xc[rf_t] = coral(Xp[rf_t], mu, cov, mu_t, cov_t)
                if t == s:
                    Xc[rt] = coral(Xp[rt], mu, cov, mu_t, cov_t)
            clf = LogisticRegression(max_iter=1000, C=1.0).fit(Xc[tr_fit], song[tr_fit])
            rec["coral_pca"].append(float((clf.predict(Xc[rt]) == yte).mean()))
            # supervised k-shot nearest-centroid on the subject's own labels
            for k in KS:
                accs = []
                for _ in range(5):
                    pick = rf[rng.choice(len(rf), k, replace=False)]
                    present = [c for c in songs if (song[pick] == c).any()]
                    cents = np.stack([Xp[pick][song[pick] == c].mean(0) for c in present])
                    pred = np.array(present)[(unit(Xp[rt]) @ unit(cents).T).argmax(1)]
                    accs.append(float((pred == yte).mean()))
                rec[f"probe_{k}"].append(float(np.mean(accs)))
        res["features"][name] = {k: float(np.mean(v)) for k, v in rec.items()}
        print(f"\n{name}")
        for k, v in res["features"][name].items():
            print(f"  {k:16} {v:.4f}")

    json.dump(res, open(OUT / "tta_baselines.json", "w", encoding="utf-8"), indent=2)
    print("\nwrote", OUT / "tta_baselines.json")


if __name__ == "__main__":
    main()
