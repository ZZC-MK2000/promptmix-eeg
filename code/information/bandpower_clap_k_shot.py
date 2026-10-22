"""C: does the band-power fiber survive the bridge into the main pipeline's query space?

The band-power k-shot calibration works (0.454@10 -> 0.721@300). The main pipeline needs a
laion-CLAP query. Here we fit a ridge band-power -> laion-CLAP on the training subjects,
apply the subject's k-shot Procrustes in band-power space, map through the ridge, and decode
against the laion-CLAP song centroids of the training subjects.

Arms per k:
  bp_space     calibrated band-power -> nearest band-power centroid (reference)
  clap_space   calibrated band-power -> ridge -> nearest laion centroid (the pipeline query)
  zero         no calibration -> ridge -> nearest laion centroid

Run: python code/information/bandpower_clap_k_shot.py     (CPU)
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
OUT = ROOT / "results" / "bandpower_clap_k_shot"
FOLDS = {
    0: {"test": ["02", "03", "04", "05"], "val": ["06", "07"]},
    1: {"test": ["06", "07", "08", "09"], "val": ["04", "05"]},
    2: {"test": ["10", "11", "12", "13"], "val": ["06", "07"]},
    3: {"test": ["14", "15", "16", "17"], "val": ["08", "09"]},
    4: {"test": ["19", "20", "21", "23"], "val": ["10", "11"]},
}
KS = [10, 20, 30, 50, 75, 100, 150, 200, 300]
ALPHA = 1e3


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def procrustes(A, B, w=None):
    if w is None:
        w = np.ones(len(A)) / len(A)
    s = np.sqrt(w / w.sum())
    Aw, Bw = A * s[:, None], B * s[:, None]
    Ua, Sa, Vta = np.linalg.svd(Aw, full_matrices=False)
    Ub, Sb, Vtb = np.linalg.svd(Bw, full_matrices=False)
    M = (Sa[:, None] * (Ua.T @ Ub)) * Sb[None, :]
    P, _, Qt = np.linalg.svd(M)
    return (Vta.T @ P) @ (Vtb.T @ Qt.T).T


def balanced_counts(k, c):
    counts = np.full(c, k // c)
    counts[:k % c] += 1
    return counts


def fps_select(rows_c, X, n):
    if n >= len(rows_c):
        return rows_c
    U = unit(X[rows_c])
    sel = [int(np.argmax(U @ unit(U.mean(0))))]
    dist = 1.0 - U @ U[sel[0]]
    for _ in range(1, n):
        i = int(np.argmax(dist)); sel.append(i)
        dist = np.minimum(dist, 1.0 - U @ U[i])
    return rows_c[np.array(sel)]


def pick_rows(rows_c, X, n):
    if n >= len(rows_c):
        return rows_c
    if n <= 2:
        cen = X[rows_c].mean(0)
        order = np.argsort(-(unit(X[rows_c]) @ unit(cen)))
        return rows_c[order[:n]]
    return fps_select(rows_c, X, n)


def centroids(X, y, songs):
    return unit(np.stack([X[y == s].mean(0) for s in songs]))


def decode(aligned, C):
    return (aligned @ unit(C).T).argmax(1)


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
    ds = z["dataset"][idx]; m = ds == 0
    song = z["song"][idx][m].astype(int)
    subj = z["subject"][idx][m].astype(str)
    grp = np.load(ROOT / "data" / "eegcond" / "group.npy").astype(int)[m]
    bp = z["bandpower"][idx][m].astype(np.float32)
    bp = (bp - bp.mean(0)) / (bp.std(0) + 1e-6)
    clap = np.load(ROOT / "results" / "prompt_classifier" / "true_emb_all.npy").astype(np.float32)[m]
    clap = unit(clap)
    songs = np.arange(10)
    # rebuild the same NMED-T-only fit/test split per subject
    rng = np.random.default_rng(0)
    split = {}
    for s in np.unique(subj):
        rows = np.where(subj == s)[0]
        gs = np.unique(grp[rows]); rng.shuffle(gs)
        fg = set(gs[:len(gs) // 2])
        split[s] = (rows[np.isin(grp[rows], list(fg))], rows[~np.isin(grp[rows], list(fg))])

    res = {"protocol": {"corpus": "NMED-T 20 subjects x 10 songs",
                        "bridge": f"ridge band-power -> laion-CLAP, alpha={ALPHA}",
                        "calibration": "per-subject Procrustes in band-power space",
                        "decoder": "nearest laion-CLAP song centroid"},
           "folds": {}, "curve": {}}
    per = {k: {"bp": [], "clap": []} for k in KS}
    zero = []

    for f, meta in FOLDS.items():
        train_subj = [s for s in np.unique(subj) if s not in meta["test"] and s not in meta["val"]]
        tr = np.where(np.isin(subj, train_subj))[0]
        rg = Ridge(alpha=ALPHA).fit(bp[tr], clap[tr])
        Cb = centroids(bp[tr], song[tr], songs)
        Cl = centroids(clap[tr], song[tr], songs)
        fc = {k: {"bp": [], "clap": []} for k in KS}
        fz = []
        for s in meta["test"]:
            rf, rt = split[s]
            if len(rf) < 100 or len(rt) < 100:
                continue
            yte = song[rt]
            fz.append(float((songs[decode(unit(rg.predict(bp[rt])), Cl)] == yte).mean()))
            for k in KS:
                counts = np.minimum(balanced_counts(k, 10),
                                    np.array([len(rf[song[rf] == c]) for c in songs]))
                picks = [pick_rows(rf[song[rf] == c], bp, counts[ci])
                         for ci, c in enumerate(songs) if counts[ci]]
                ap = np.concatenate(picks); ay = song[ap]
                present = np.array([c for c in songs if (ay == c).any()])
                Cs = centroids(bp[ap], ay, present)
                Cba = Cb[present]
                wgt = np.array([(ay == c).sum() for c in present], float)
                cs_m, cb_m = np.average(Cs, axis=0, weights=wgt), np.average(Cba, axis=0, weights=wgt)
                R = procrustes(Cs - cs_m, Cba - cb_m, wgt)
                A = (bp[rt] - cs_m) @ R
                fc[k]["bp"].append(float((songs[decode(unit(A), Cb)] == yte).mean()))
                fc[k]["clap"].append(float((songs[decode(unit(rg.predict(A)), Cl)] == yte).mean()))
        res["folds"][f] = {"zero_clap": float(np.mean(fz)),
                           "curve": {str(k): {"bp": float(np.mean(fc[k]["bp"])),
                                              "clap": float(np.mean(fc[k]["clap"]))} for k in KS}}
        zero += fz
        for k in KS:
            per[k]["bp"] += fc[k]["bp"]; per[k]["clap"] += fc[k]["clap"]
        print(f"fold {f}: zero_clap {res['folds'][f]['zero_clap']:.3f} | "
              + " ".join(f"k{k}:bp {np.mean(fc[k]['bp']):.3f}/clap {np.mean(fc[k]['clap']):.3f}" for k in (10, 100, 300)))

    for k in KS:
        res["curve"][str(k)] = {"bp_space": float(np.mean(per[k]["bp"])),
                                "clap_space": float(np.mean(per[k]["clap"]))}
    res["baselines"] = {"zero_clap": float(np.mean(zero))}
    json.dump(res, open(OUT / "bridge.json", "w", encoding="utf-8"), indent=2)
    print(f"\nzero (no calibration, ridge->CLAP): {res['baselines']['zero_clap']:.4f}")
    print("  k     band-power space   laion-CLAP query")
    for k in KS:
        r = res["curve"][str(k)]
        print(f"  {k:3d}   {r['bp_space']:.3f}              {r['clap_space']:.3f}")
    print("wrote", OUT / "bridge.json")


if __name__ == "__main__":
    main()
