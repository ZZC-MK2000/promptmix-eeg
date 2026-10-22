"""Main-pipeline linear probes on the subject-held-out fold embeddings.

Question: is the 0.25-0.29 oracle ceiling of the nearest-centroid decoder a decoder
artifact, or does the trained embedding itself lack cross-subject song information?

Arms (5 folds, held-out subjects never seen by the encoder):
  probe14        logistic regression on all training subjects -> 14 song classes
  probe10        logistic regression on NMED-T training subjects -> 10 song classes
  probe10_cal    train each NMED-T subject's Procrustes to the shared codebook, then fit
                 the probe on the calibrated training rows
  probe10_cal_k  evaluate probe10_cal on the held-out subject after k-shot calibration
  local_ridge_k  RidgeClassifier on the held-out subject's own k anchors (TTA control)

Run: python code/information/mainline_linear_probe.py     (CPU)
"""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
OUT = ROOT / "results" / "mainline_linear_probe"
GATE = ROOT.parent / "mainv2" / "data" / "gate_nmedh14_ov2_a.npz"
FOLDS = {
    0: {"test": ["02", "03", "04", "05"], "val": ["06", "07"]},
    1: {"test": ["06", "07", "08", "09"], "val": ["04", "05"]},
    2: {"test": ["10", "11", "12", "13"], "val": ["06", "07"]},
    3: {"test": ["14", "15", "16", "17"], "val": ["08", "09"]},
    4: {"test": ["19", "20", "21", "23"], "val": ["10", "11"]},
}
KS = [10, 20, 30, 50, 75, 100, 150, 200, 300]


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(GATE, allow_pickle=True)
    song = z["song"].astype(int); subj = z["subject"].astype(str)
    al = z["aligned8"].astype(bool); ds = z["dataset"].astype(int)
    grp = np.load(ROOT / "data" / "eegcond" / "group.npy").astype(int)
    songs10 = np.arange(10)
    rng = np.random.default_rng(0)
    res = {"protocol": {
        "embedding": "stage-1 head output, 5 subject-held-out folds (stock init)",
        "train": "training subjects of the same fold; test = held-out subjects",
        "calibration": "per-subject Procrustes to the shared codebook",
    }, "folds": {}, "curve": {}}
    per = {k: [] for k in KS}
    agg = {a: [] for a in ("probe14", "probe10", "probe10_cal_oracle")}
    local = {k: [] for k in KS}

    for f, meta in FOLDS.items():
        E = unit(np.load(ROOT / "results" / "subject_holdout" / f"eeg_emb_aldm_SH_f{f}_s0.npy").astype(np.float32))
        train_subj = [s for s in np.unique(subj) if s not in meta["test"] and s not in meta["val"]]
        train_nmed = [s for s in train_subj if s in set(subj[ds == 0])]
        tr_all = np.where(al & np.isin(subj, train_subj))[0]
        tr_t = np.where(al & np.isin(subj, train_nmed))[0]
        Cg10 = centroids(E[tr_t], song[tr_t], songs10)

        p14 = LogisticRegression(max_iter=2000, C=1.0).fit(E[tr_all], song[tr_all])
        p10 = LogisticRegression(max_iter=2000, C=1.0).fit(E[tr_t], song[tr_t])

        # calibrate each NMED-T training subject (all their aligned8 rows)
        Ecal = E.copy()
        for t in train_nmed:
            rt = np.where(al & (subj == t))[0]
            Ct = centroids(E[rt], song[rt], songs10)
            m = Ct.mean(0)
            R = procrustes(Ct - m, Cg10 - Cg10.mean(0))
            Ecal[rt] = (E[rt] - m) @ R
        p10c = LogisticRegression(max_iter=2000, C=1.0).fit(Ecal[tr_t], song[tr_t])

        fold = {"probe14": [], "probe10": [], "probe10_cal_oracle": [],
                "curve": {k: [] for k in KS}, "local_ridge": {k: [] for k in KS}}
        for s in meta["test"]:
            rows = np.where(al & (subj == s))[0]
            gs = np.unique(grp[rows]); rng.shuffle(gs)
            fg = set(gs[:len(gs) // 2])
            rf = rows[np.isin(grp[rows], list(fg))]
            rt = rows[~np.isin(grp[rows], list(fg))]
            if len(rf) < 100 or len(rt) < 100:
                continue
            yte = song[rt]
            fold["probe14"].append(float((p14.predict(E[rt]) == yte).mean()))
            fold["probe10"].append(float((p10.predict(E[rt]) == yte).mean()))
            # oracle calibration (all fit rows) + calibrated probe
            present_s = np.unique(song[rf])
            Ct = centroids(E[rf], song[rf], present_s)
            Cga_o = Cg10[present_s]
            w_o = np.array([(song[rf] == c).sum() for c in present_s], float)
            cs_o, cg_o = np.average(Ct, axis=0, weights=w_o), np.average(Cga_o, axis=0, weights=w_o)
            R_o = procrustes(Ct - cs_o, Cga_o - cg_o, w_o)
            fold["probe10_cal_oracle"].append(
                float((p10c.predict((E[rt] - cs_o) @ R_o) == yte).mean()))
            for k in KS:
                counts = np.minimum(balanced_counts(k, 10),
                                    np.array([len(rf[song[rf] == c]) for c in songs10]))
                picks = [pick_rows(rf[song[rf] == c], E, counts[ci])
                         for ci, c in enumerate(songs10) if counts[ci]]
                ap = np.concatenate(picks); ay = song[ap]
                present = np.array([c for c in songs10 if (ay == c).any()])
                Cs = centroids(E[ap], ay, present)
                Cga = Cg10[present]
                w = np.array([(ay == c).sum() for c in present], float)
                cs_m, cg_m = np.average(Cs, axis=0, weights=w), np.average(Cga, axis=0, weights=w)
                R = procrustes(Cs - cs_m, Cga - cg_m, w)
                Etest = (E[rt] - cs_m) @ R
                fold["curve"][k].append(float((p10c.predict(Etest) == yte).mean()))
                # local ridge on the subject's own anchors (TTA control)
                if k >= 50:
                    rc = RidgeClassifier(alpha=10.0).fit(E[ap], ay)
                    fold["local_ridge"][k].append(float((rc.predict(E[rt]) == yte).mean()))
        res["folds"][f] = {
            "probe14": float(np.mean(fold["probe14"])), "probe10": float(np.mean(fold["probe10"])),
            "probe10_cal_oracle": float(np.mean(fold["probe10_cal_oracle"])),
            "curve": {str(k): float(np.mean(v)) for k, v in fold["curve"].items() if v},
            "local_ridge": {str(k): float(np.mean(v)) for k, v in fold["local_ridge"].items() if v},
        }
        for a in agg:
            agg[a] += fold[a]
        for k in KS:
            per[k] += fold["curve"][k]
            local[k] += fold["local_ridge"][k]
        print(f"fold {f}: probe14 {res['folds'][f]['probe14']:.3f} "
              f"probe10 {res['folds'][f]['probe10']:.3f} "
              f"probe10_cal_oracle {res['folds'][f]['probe10_cal_oracle']:.3f}")

    for k in KS:
        res["curve"][str(k)] = {"calibrated_probe10": float(np.mean(per[k]))}
        if local[k]:
            res["curve"][str(k)]["local_ridge10"] = float(np.mean(local[k]))
    res["baselines"] = {a: float(np.mean(v)) for a, v in agg.items()}
    json.dump(res, open(OUT / "probe.json", "w", encoding="utf-8"), indent=2)
    b = res["baselines"]
    print(f"\nMAIN PIPELINE PROBES (held-out subjects)")
    print(f"  probe14 {b['probe14']:.4f} | probe10 {b['probe10']:.4f} | "
          f"probe10_cal_oracle {b['probe10_cal_oracle']:.4f}")
    print("  k     calibrated_probe10   local_ridge10")
    for k in KS:
        r = res["curve"][str(k)]
        print(f"  {k:3d}   {r['calibrated_probe10']:.3f}              "
              f"{r.get('local_ridge10', float('nan')):.3f}")
    print("wrote", OUT / "probe.json")


if __name__ == "__main__":
    main()
