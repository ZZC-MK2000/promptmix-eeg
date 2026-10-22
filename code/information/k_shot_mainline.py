"""k-shot calibration on the mainline representations and retrieval protocol.

The earlier k-shot experiment (code/information/k_shot_calibration_v2.py) used 10-way
NMED-T song classification on band-power and LaBraM-200. The mainline pipeline is different:
query = bridge(EEG) in laion-CLAP; gallery = 14 audio-CLAP song centroids; metric = 14-way
song retrieval. This script puts the three requested representations on the same per-subject
50/50 segment protocol:

  labram     frozen LaBraM-200 (10-way NMED-T, EEG-space centroids)
  aldm       trained alignment embedding (10-way NMED-T, EEG-space centroids)  [transductive]
  mainline   bridge(EEG) = cond (laion-CLAP); gallery = 14 target centroids; 14-way + 10-way

Calibration: weighted orthogonal Procrustes with translation on the subject's class
centroids, fitted from k labelled anchors; anchors are random or active (D-optimal class
counts + hybrid within-class: closest for n_c<=2, FPS above).

Run: python code/information/k_shot_mainline.py     (CPU)
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = ROOT.parent / "mainv2"
OUT = ROOT / "results" / "k_shot"
SEED = 0
KS = [10, 20, 30, 50, 75, 100, 150, 200, 300]
N_RANDOM = 10


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
    U = unit(X[rows_c])
    cen = unit(U.mean(0))
    sel = [int(np.argmax(U @ cen))]
    dist = 1.0 - U @ U[sel[0]]
    for _ in range(1, n):
        i = int(np.argmax(dist)); sel.append(i)
        dist = np.minimum(dist, 1.0 - U @ U[i])
    return rows_c[np.array(sel)]


def pick_rows(rows_c, X, n, mode, rng):
    if n >= len(rows_c):
        return rows_c
    if mode == "closest":
        cen = X[rows_c].mean(0)
        order = np.argsort(-(unit(X[rows_c]) @ unit(cen)))
        return rows_c[order[:n]]
    if mode == "uniform":
        return rows_c[rng.choice(len(rows_c), n, replace=False)]
    if mode == "fps":
        return fps_select(rows_c, X, n)
    if mode == "hybrid":
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
    raise ValueError(mode)


def centroids(X, y, classes):
    return np.stack([X[y == c].mean(0) for c in classes])


def decode(aligned, G):
    return (unit(aligned) @ unit(G).T).argmax(1)


def score(aligned, G, y, classes):
    return float((np.array(classes)[decode(aligned, G)] == y).mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)
    aldm = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    w = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu", weights_only=True)
    ca = unit(aldm @ w["weight"].numpy().astype(np.float32).T + w["bias"].numpy().astype(np.float32))
    key = {ca[i].tobytes(): i for i in range(len(ca))}
    idx = np.array([key[cond[j].tobytes()] for j in range(len(cond))])
    song = z["song"][idx].astype(int)
    ds = z["dataset"][idx].astype(int)
    subj = z["subject"][idx].astype(str)
    grp = np.load(ROOT / "data" / "eegcond" / "group.npy").astype(int)
    target = z["target"].astype(np.float32)
    song_all = z["song"].astype(int)
    # mainline gallery: 14 audio-CLAP centroids from target (the retrieval-line protocol)
    G14 = unit(np.stack([unit(target[song_all == s]).mean(0) for s in range(14)]))
    G10 = G14[:10]

    m = ds == 0
    subj_t = subj[m]
    song_t = song[m]
    grp_t = grp[m]
    feats = {
        "labram": z["labram"][idx][m].astype(np.float64),
        "aldm": aldm[idx][m].astype(np.float64),
        "mainline": cond[m].astype(np.float64),   # already unit, laion-CLAP
    }
    subjects = np.unique(subj_t)
    rng = np.random.default_rng(SEED)
    res = {"protocol": {
        "corpus": "NMED-T 20 subjects x 10 songs; per-subject 50/50 by audio segment",
        "labram": "frozen LaBraM-200; 10-way; global EEG centroids from other subjects",
        "aldm": "trained alignment embedding (transductive); 10-way; global EEG centroids",
        "mainline": "bridge(EEG)=cond in laion-CLAP; gallery=14 audio-CLAP centroids; 14-way and 10-way",
        "anchors": "random (10 repeats) vs active D-optimal + hybrid within-class",
        "k": KS}, "features": {}}

    for name, X in feats.items():
        if name != "mainline":
            X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        split = {}
        for s in subjects:
            rows = np.where(subj_t == s)[0]
            gs = np.unique(grp_t[rows]); rng.shuffle(gs)
            fg = set(gs[:len(gs) // 2])
            rf = rows[np.isin(grp_t[rows], list(fg))]
            rt = rows[~np.isin(grp_t[rows], list(fg))]
            split[s] = (rf, rt)

        rec = {"baselines": {}, "curve": {str(k): {} for k in KS}}
        base = {b: [] for b in
                (["zero10", "within10", "oracle10"] if name != "mainline"
                 else ["zero14", "zero10", "within10", "oracle14", "oracle10"])}
        for s in subjects:
            rf, rt = split[s]
            if len(rf) < 200 or len(rt) < 200:
                continue
            yte = song_t[rt]
            if name == "mainline":
                gal10, gal14 = G10, G14
                base["zero14"].append(score(cond[rt], gal14, yte, range(14)))
                base["zero10"].append(score(cond[rt], gal10, yte, range(10)))
                Cown = centroids(X[rf], song_t[rf], range(10))
                base["within10"].append(score(X[rt], Cown, yte, range(10)))
                Call = Cown
                Rall, ms, mg = fit_align(Call, gal10, np.ones(10))
                base["oracle14"].append(score((X[rt] - ms) @ Rall + mg, gal14, yte, range(14)))
                base["oracle10"].append(score((X[rt] - ms) @ Rall + mg, gal10, yte, range(10)))
                codebook = gal10
            else:
                Cg = centroids(X[np.where(subj_t != s)[0]], song_t[np.where(subj_t != s)[0]], range(10))
                base["zero10"].append(score(X[rt], Cg, yte, range(10)))
                Cown = centroids(X[rf], song_t[rf], range(10))
                base["within10"].append(score(X[rt], Cown, yte, range(10)))
                Rall, ms, mg = fit_align(Cown, Cg, np.ones(10))
                base["oracle10"].append(score((X[rt] - ms) @ Rall + mg, Cg, yte, range(10)))
                codebook = Cg
            codebook_gram = (codebook - codebook.mean(0)) @ (codebook - codebook.mean(0)).T

            for k in KS:
                avail = np.array([len(rf[song_t[rf] == c]) for c in range(10)])
                for mode in ("random", "active"):
                    reps = N_RANDOM if mode == "random" else 1
                    accs = {g: [] for g in (["10"] if name != "mainline" else ["14", "10"])}
                    for _ in range(reps):
                        if mode == "random":
                            ap = rf[rng.choice(len(rf), k, replace=False)]
                            ay = song_t[ap]
                            pres = [c for c in range(10) if (ay == c).any()]
                        else:
                            counts = np.minimum(d_optimal_counts(codebook_gram, k), avail)
                            picks = []
                            for c in range(10):
                                if counts[c] == 0:
                                    continue
                                picks.append(pick_rows(rf[song_t[rf] == c], X, counts[c], "hybrid", rng))
                            ap = np.concatenate(picks); ay = song_t[ap]
                            pres = [c for c in range(10) if (ay == c).any()]
                        Cs = centroids(X[ap], ay, pres)
                        Cg_sub = (gal10 if name == "mainline" else codebook)[pres]
                        wg = np.array([(ay == c).sum() for c in pres], float)
                        R, ms, mg = fit_align(Cs, Cg_sub, wg)
                        aligned = (X[rt] - ms) @ R + mg
                        if name == "mainline":
                            accs["14"].append(score(aligned, gal14, yte, range(14)))
                        accs["10"].append(score(aligned, (gal10 if name == "mainline" else codebook),
                                                yte, range(10)))
                    for g in accs:
                        rec["curve"][str(k)].setdefault(f"{mode}{g}", []).append(float(np.mean(accs[g])))

        rec["baselines"] = {b: float(np.mean(v)) for b, v in base.items()}
        res["features"][name] = rec

    # aggregate across subjects
    out = {"protocol": res["protocol"], "features": {}}
    for name, rec in res["features"].items():
        f = {"baselines": rec["baselines"], "curve": {}}
        for k in KS:
            f["curve"][str(k)] = {}
            for key, vals in rec["curve"][str(k)].items():
                f["curve"][str(k)][key] = {"acc": float(np.mean(vals)),
                                           "sd": float(np.std(vals, ddof=1))}
        out["features"][name] = f
        b = f["baselines"]
        print(f"\n{name}: " + "  ".join(f"{k}={v:.3f}" for k, v in b.items()))
        for k in KS:
            row = f["curve"][str(k)]
            print(f"  k={k:3d}  " + "  ".join(f"{key}={v['acc']:.3f}" for key, v in row.items()))

    json.dump(out, open(OUT / "k_shot_mainline.json", "w", encoding="utf-8"), indent=2)
    print("\nwrote", OUT / "k_shot_mainline.json")


if __name__ == "__main__":
    main()
