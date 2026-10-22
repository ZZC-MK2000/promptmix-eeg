"""Clean mainline k-shot: held-out-subject encoder output + bridge refit per fold.

The mainline `cond` is transductive (the alignment encoder saw all subjects), so its
zero-shot 0.799 on a within-subject split is not a cross-subject number. This script uses
the five subject-held-out checkpoints saved by run_subject_holdout.sh, refits the A→L ridge
on the training subjects of each fold, and evaluates the mainline 14-way retrieval plus
k-shot calibration on the held-out subjects.

Protocol per fold:
  - E = eeg_emb_aldm_SH_f<k>_s0.npy  (head output; the encoder never saw the test subjects)
  - fit Ridge(E[train subjects] -> target) on that fold's training subjects
  - cond = unit(E @ W.T + b)
  - gallery = 14 audio-CLAP centroids from target (mainline retrieval protocol)
  - per held-out subject: 50/50 segment split; anchors from the fit half; Procrustes on
    the 10 NMED-T centroids; evaluate 14-way and 10-way on the test half.

Run: python code/information/k_shot_fold_mainline.py     (CPU)
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = ROOT.parent / "mainv2"
OUT = ROOT / "results" / "k_shot"
FOLDS = {
    0: {"test": ["02", "03", "04", "05"], "val": ["06", "07"]},
    1: {"test": ["06", "07", "08", "09"], "val": ["04", "05"]},
    2: {"test": ["10", "11", "12", "13"], "val": ["06", "07"]},
    3: {"test": ["14", "15", "16", "17"], "val": ["08", "09"]},
    4: {"test": ["19", "20", "21", "23"], "val": ["10", "11"]},
}
KS = [10, 20, 30, 50, 75, 100, 150, 200, 300]
N_RANDOM = 10
SEED = 0


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def procrustes(A, B, w):
    s = np.sqrt(w / w.sum())
    Aw, Bw = A * s[:, None], B * s[:, None]
    Ua, Sa, Vta = np.linalg.svd(Aw, full_matrices=False)
    Ub, Sb, Vtb = np.linalg.svd(Bw, full_matrices=False)
    M = (Sa[:, None] * (Ua.T @ Ub)) * Sb[None, :]
    P, _, Qt = np.linalg.svd(M)
    return (Vta.T @ P) @ (Vtb.T @ Qt.T).T


def fit_align(Cs, Cg, w):
    ms = np.average(Cs, axis=0, weights=w)
    mg = np.average(Cg, axis=0, weights=w)
    return procrustes(Cs - ms, Cg - mg, w), ms, mg


def d_optimal_counts(G, k):
    C = len(G)
    counts = np.zeros(C, dtype=int)
    if k >= C:
        counts[:] = 1
    eps = 1e-6
    for _ in range(k - counts.sum()):
        best_c, best = -1, -np.inf
        for c in range(C):
            cand = counts.copy(); cand[c] += 1
            A = (np.sqrt(cand)[:, None] * G * np.sqrt(cand)[None, :]) + eps * np.eye(C)
            _, logdet = np.linalg.slogdet(A)
            if logdet > best:
                best, best_c = logdet, c
        counts[best_c] += 1
    return counts


def fps_select(rows_c, X, n):
    if n >= len(rows_c):
        return rows_c
    U = unit(X[rows_c]); cen = unit(U.mean(0))
    sel = [int(np.argmax(U @ cen))]
    dist = 1.0 - U @ U[sel[0]]
    for _ in range(1, n):
        i = int(np.argmax(dist)); sel.append(i)
        dist = np.minimum(dist, 1.0 - U @ U[i])
    return rows_c[np.array(sel)]


def pick_rows(rows_c, X, n, rng):
    if n >= len(rows_c):
        return rows_c
    m = min(n, 2)
    cen = X[rows_c].mean(0)
    order = np.argsort(-(unit(X[rows_c]) @ unit(cen)))
    sel = list(order[:m])
    if n > m:
        U = unit(X[rows_c])
        dist = np.min(1.0 - U @ U[sel].T, axis=1)
        for _ in range(n - m):
            i = int(np.argmax(dist)); sel.append(i)
            dist = np.minimum(dist, 1.0 - U @ U[i])
    return rows_c[np.array(sel)]


def centroids(X, y, classes):
    return np.stack([X[y == c].mean(0) for c in classes])


def decode(aligned, G):
    return (aligned @ unit(G).T).argmax(1)


def score(aligned, G, y, classes):
    return float((np.array(classes)[decode(aligned, G)] == y).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    song = z["song"].astype(int)
    ds = z["dataset"].astype(int)
    subj = z["subject"].astype(str)
    seg2 = z["seg2"].astype(int)
    seg_key = song * 1000 + seg2
    target = z["target"].astype(np.float32)
    G14 = unit(np.stack([unit(target[song == s]).mean(0) for s in range(14)]))
    G10 = G14[:10]
    Gmat = (G10 - G10.mean(0)) @ (G10 - G10.mean(0)).T
    rng = np.random.default_rng(SEED)

    rec = {"baselines": {}, "curve": {str(k): {} for k in KS}}
    base = {b: [] for b in ("zero14", "zero10", "within10", "oracle14", "oracle10")}
    print("fold  subject  n_fit n_test  zero14 zero10 within10 oracle14 oracle10")
    for f, meta in FOLDS.items():
        E = np.load(ROOT / "results" / "subject_holdout" / f"eeg_emb_aldm_SH_f{f}_s0.npy").astype(np.float32)
        test = set(meta["test"]); val = set(meta["val"])
        train_rows = np.where(~np.isin(subj, list(test | val)))[0]
        rg = Ridge(alpha=1e3).fit(E[train_rows], target[train_rows])
        cond = unit(E @ rg.coef_.T + rg.intercept_)
        for s in meta["test"]:
            rows = np.where(subj == s)[0]
            gs = np.unique(seg_key[rows]); rng.shuffle(gs)
            fg = set(gs[:len(gs) // 2])
            rf = rows[np.isin(seg_key[rows], list(fg))]
            rt = rows[~np.isin(seg_key[rows], list(fg))]
            if len(rf) < 100 or len(rt) < 100:
                continue
            yte = song[rt]
            base["zero14"].append(score(cond[rt], G14, yte, range(14)))
            base["zero10"].append(score(cond[rt], G10, yte, range(10)))
            Cown = centroids(cond[rf], song[rf], range(10))
            base["within10"].append(score(cond[rt], Cown, yte, range(10)))
            Rall, ms, mg = fit_align(Cown, G10, np.ones(10))
            base["oracle14"].append(score((cond[rt] - ms) @ Rall + mg, G14, yte, range(14)))
            base["oracle10"].append(score((cond[rt] - ms) @ Rall + mg, G10, yte, range(10)))
            for k in KS:
                avail = np.array([len(rf[song[rf] == c]) for c in range(10)])
                for mode in ("random", "active"):
                    reps = N_RANDOM if mode == "random" else 1
                    a14, a10 = [], []
                    for _ in range(reps):
                        if mode == "random":
                            ap = rf[rng.choice(len(rf), k, replace=False)]
                            ay = song[ap]
                            pres = [c for c in range(10) if (ay == c).any()]
                        else:
                            counts = np.minimum(d_optimal_counts(Gmat, k), avail)
                            picks = []
                            for c in range(10):
                                if counts[c]:
                                    picks.append(pick_rows(rf[song[rf] == c], cond, counts[c], rng))
                            ap = np.concatenate(picks); ay = song[ap]
                            pres = [c for c in range(10) if (ay == c).any()]
                        Cs = centroids(cond[ap], ay, pres)
                        Cg = G10[pres]
                        wg = np.array([(ay == c).sum() for c in pres], float)
                        R, ms, mg = fit_align(Cs, Cg, wg)
                        aligned = (cond[rt] - ms) @ R + mg
                        a14.append(score(aligned, G14, yte, range(14)))
                        a10.append(score(aligned, G10, yte, range(10)))
                    rec["curve"][str(k)].setdefault(f"{mode}14", []).append(float(np.mean(a14)))
                    rec["curve"][str(k)].setdefault(f"{mode}10", []).append(float(np.mean(a10)))
            print(f"{f:4d}  {s:>6}  {len(rf):5d} {len(rt):6d}  "
                  f"{base['zero14'][-1]:.3f}  {base['zero10'][-1]:.3f}  "
                  f"{base['within10'][-1]:.3f}  {base['oracle14'][-1]:.3f}  {base['oracle10'][-1]:.3f}")
        del E, cond

    out = {"protocol": {
        "folds": FOLDS, "k": KS, "anchor": "random (10) vs active hybrid",
        "bridge": "A->L ridge refit on the fold's training subjects",
        "gallery": "14 audio-CLAP centroids from target (mainline protocol)",
        "note": "the encoder never saw the test subjects; this is the clean mainline k-shot"},
        "baselines": {k: float(np.mean(v)) for k, v in base.items()},
        "curve": {}}
    for k in KS:
        out["curve"][str(k)] = {key: {"acc": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
                                for key, v in rec["curve"][str(k)].items()}
    print("\nbaselines", out["baselines"])
    for k in KS:
        print(f"k={k:3d}  " + "  ".join(f"{key}={v['acc']:.3f}"
                                        for key, v in out["curve"][str(k)].items()))
    json.dump(out, open(OUT / "k_shot_fold_mainline.json", "w", encoding="utf-8"), indent=2)
    print("wrote", OUT / "k_shot_fold_mainline.json")


if __name__ == "__main__":
    main()
