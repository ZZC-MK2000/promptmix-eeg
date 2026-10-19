"""k-shot calibration v2: within-class diversity (farthest-point sampling).

v1 picked the windows *closest* to each class centroid. That is biased at large k
(it over-represents low-noise samples), which is why the random control caught up.
v2 separates the two design factors:

  class allocation   D-optimal (log-det)  vs  balanced (round-robin)
  within-class pick  closest  vs  uniform  vs  farthest-point sampling (FPS)

Strategies:
  random               k windows uniformly from the subject's fit pool
  dopt_closest         D-optimal counts + closest-to-centroid
  dopt_fps             D-optimal counts + FPS
  dopt_uniform         D-optimal counts + uniform within class
  balanced_fps         balanced counts + FPS
  balanced_uniform     balanced counts + uniform within class

Evaluation is unchanged: orthogonal Procrustes on the anchors, nearest global
centroid on the held-out audio segments of the same subject.

Run: python code/information/k_shot_calibration_v2.py     (CPU)
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
OUT = ROOT / "results" / "k_shot"
SEED = 0
KS = [10, 20, 30, 50, 75, 100, 150, 200, 300]
N_RANDOM = 10
N_UNIFORM = 5
STRATS = ["random", "dopt_closest", "dopt_fps", "dopt_uniform", "dopt_hybrid",
          "balanced_fps", "balanced_uniform", "balanced_hybrid"]


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


def balanced_counts(G, C, k, rng):
    if k < C:
        return d_optimal_counts(G, k)
    counts = np.full(C, k // C)
    order = rng.permutation(C)
    counts[order[:k % C]] += 1
    return counts


def fps_select(rows_c, X, n):
    """Fewest-point sampling: start at the closest-to-centroid row, then maximise
    the minimum cosine distance to the selected set."""
    if n >= len(rows_c):
        return rows_c
    U = unit(X[rows_c])
    cen = unit(U.mean(0))
    sel = [int(np.argmax(U @ cen))]
    dist = 1.0 - U @ U[sel[0]]
    for _ in range(1, n):
        i = int(np.argmax(dist))
        sel.append(i)
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
        # small n_c: closest-to-centroid (noise reduction); larger n_c: FPS diversity
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


def centroids(X, y, songs):
    return np.stack([X[y == s].mean(0) for s in songs])


def decode(aligned, Cg):
    return (unit(aligned) @ unit(Cg).T).argmax(1)


def evaluate(R, Cg_centred, Cs_mean, Xte, yte, songs):
    aligned = (Xte - Cs_mean) @ R
    return float((songs[decode(aligned, Cg_centred)] == yte).mean())


def within_class_diversity(rows, X):
    if len(rows) < 2:
        return 0.0
    U = unit(X[rows])
    C = U @ U.T
    n = len(U)
    return float((C.sum() - n) / (n * (n - 1)))


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
    feats = {"bandpower": z["bandpower"][idx][m].astype(np.float64),
             "labram": z["labram"][idx][m].astype(np.float64)}
    subjects = np.unique(subj)
    rng = np.random.default_rng(SEED)
    res = {"protocol": {"corpus": "NMED-T 20 subjects x 10 songs",
                        "split": "per-subject 50/50 by audio segment",
                        "strategies": STRATS, "k": KS,
                        "n_random": N_RANDOM, "n_uniform": N_UNIFORM},
           "features": {}}

    for name, X in feats.items():
        X = (X - X.mean(0)) / (X.std(0) + 1e-6)
        split = {}
        for s in subjects:
            rows = np.where(subj == s)[0]
            gs = np.unique(grp[rows]); rng.shuffle(gs)
            fg = set(gs[:len(gs) // 2])
            rf = rows[np.isin(grp[rows], list(fg))]
            rt = rows[~np.isin(grp[rows], list(fg))]
            split[s] = (rf, rt)

        curves = {k: {st: [] for st in STRATS} for k in KS}
        cov = {k: {st: [] for st in STRATS} for k in KS}
        div = {k: {st: [] for st in STRATS} for k in KS}
        base = {"zero": [], "within": [], "oracle": []}

        for s in subjects:
            rf, rt = split[s]
            if len(rf) < 200 or len(rt) < 200:
                continue
            ro = np.where(subj != s)[0]
            Cg = centroids(X[ro], song[ro], songs)
            G = (Cg - Cg.mean(0)) @ (Cg - Cg.mean(0)).T
            yte = song[rt]
            base["zero"].append(float((songs[decode(X[rt], Cg)] == yte).mean()))
            Cown = centroids(X[rf], song[rf], songs)
            base["within"].append(float((songs[decode(X[rt], Cown)] == yte).mean()))
            call_mean, cg_mean = Cown.mean(0), Cg.mean(0)
            Rall = procrustes(Cown - call_mean, Cg - cg_mean)
            base["oracle"].append(evaluate(Rall, Cg - cg_mean, call_mean, X[rt], yte, songs))

            for k in KS:
                avail = np.array([len(rf[song[rf] == c]) for c in songs])
                for st in STRATS:
                    if st == "random":
                        accs, covs, divs = [], [], []
                        for _ in range(N_RANDOM):
                            rp = rf[rng.choice(len(rf), k, replace=False)]
                            ry = song[rp]
                            pres = np.array([c for c in songs if (ry == c).any()])
                            Cs = centroids(X[rp], ry, pres)
                            Cga = Cg[[list(songs).index(c) for c in pres]]
                            wg = np.array([(ry == c).sum() for c in pres], float)
                            cs_m, cg_m = np.average(Cs, axis=0, weights=wg), np.average(Cga, axis=0, weights=wg)
                            R = procrustes(Cs - cs_m, Cga - cg_m, wg)
                            accs.append(evaluate(R, Cg - cg_m, cs_m, X[rt], yte, songs))
                            covs.append(len(pres))
                            divs.append(float(np.mean([within_class_diversity(rp[ry == c], X)
                                                       for c in pres])))
                        curves[k][st].append(float(np.mean(accs)))
                        cov[k][st].append(float(np.mean(covs)))
                        div[k][st].append(float(np.mean(divs)))
                        continue

                    fam, mode = st.split("_", 1)
                    counts = (d_optimal_counts(G, k) if fam == "dopt"
                              else balanced_counts(G, len(songs), k, rng))
                    counts = np.minimum(counts, avail)
                    reps = N_UNIFORM if mode == "uniform" else 1
                    accs, covs, divs = [], [], []
                    for _ in range(reps):
                        picks = []
                        for ci, c in enumerate(songs):
                            if counts[ci] == 0:
                                continue
                            picks.append(pick_rows(rf[song[rf] == c], X, counts[ci], mode, rng))
                        ap = np.concatenate(picks); ay = song[ap]
                        pres = np.array([c for c in songs if (ay == c).any()])
                        Cs = centroids(X[ap], ay, pres)
                        Cga = Cg[[list(songs).index(c) for c in pres]]
                        wg = np.array([(ay == c).sum() for c in pres], float)
                        cs_m, cg_m = np.average(Cs, axis=0, weights=wg), np.average(Cga, axis=0, weights=wg)
                        R = procrustes(Cs - cs_m, Cga - cg_m, wg)
                        accs.append(evaluate(R, Cg - cg_m, cs_m, X[rt], yte, songs))
                        covs.append(len(pres))
                        divs.append(float(np.mean([within_class_diversity(ap[ay == c], X)
                                                   for c in pres])))
                    curves[k][st].append(float(np.mean(accs)))
                    cov[k][st].append(float(np.mean(covs)))
                    div[k][st].append(float(np.mean(divs)))

        out = {"baselines": {k: float(np.mean(v)) for k, v in base.items()}, "curve": {}}
        for k in KS:
            out["curve"][str(k)] = {}
            for st in STRATS:
                v = np.array(curves[k][st])
                out["curve"][str(k)][st] = {
                    "acc": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "class_coverage": float(np.mean(cov[k][st])),
                    "within_class_cos": float(np.mean(div[k][st])),
                }
        res["features"][name] = out
        b = out["baselines"]
        print(f"\n{name}: zero {b['zero']:.3f} | within {b['within']:.3f} | oracle {b['oracle']:.3f}")
        print("  k    " + "  ".join(f"{st:16}" for st in STRATS))
        for k in KS:
            row = "  ".join(f"{out['curve'][str(k)][st]['acc']:.3f}/{out['curve'][str(k)][st]['within_class_cos']:.2f}"
                            for st in STRATS)
            print(f"  {k:3d}  {row}")

    json.dump(res, open(OUT / "k_shot_v2.json", "w", encoding="utf-8"), indent=2)
    print("\nwrote", OUT / "k_shot_v2.json")


if __name__ == "__main__":
    main()
