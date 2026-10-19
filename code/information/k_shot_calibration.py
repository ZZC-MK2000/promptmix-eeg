"""k-shot subject calibration + active (D-optimal) anchor selection.

Follows docs/identifiability_proposition.md:

  - shared codebook Z (class centroids from the other subjects),
  - per-subject orthogonal Procrustes fit on k labelled anchor windows,
  - active anchors: D-optimal class allocation (maximise log-det of the weighted
    codebook covariance), then per class pick the windows closest to the subject's
    class centroid;
  - random anchors: k windows sampled uniformly from the subject's fit pool (the
    control; it may miss classes at small k).

Evaluation: held-out audio segments of the same subject, nearest global centroid.
Baselines: zero-shot, within-subject, oracle (all fit rows).

Run: python code/information/k_shot_calibration.py     (CPU)
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
OUT = ROOT / "results" / "k_shot"
SEED = 0
KS = [10, 20, 30, 50, 75, 100, 150, 200, 300]
N_RANDOM = 10


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def procrustes(A, B, w=None):
    """Weighted orthogonal Procrustes: min ||diag(sqrt(w)) (A R - B)||_F."""
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
    """Greedy D-optimal class counts: maximise log-det of the weighted Gram."""
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


def lam_min(G, counts):
    A = (np.sqrt(counts)[:, None] * G * np.sqrt(counts)[None, :])
    return float(np.linalg.eigvalsh(A)[0])


def centroids(X, y, songs):
    return np.stack([X[y == s].mean(0) for s in songs])


def decode(aligned, Cg):
    return (unit(aligned) @ unit(Cg).T).argmax(1)


def evaluate(R, Cg, Cs_mean, Xte, yte, songs):
    aligned = (Xte - Cs_mean) @ R
    return float((songs[decode(aligned, Cg)] == yte).mean())


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
                        "anchors": "active=D-optimal class allocation; random=uniform",
                        "decoder": "nearest global centroid after Procrustes",
                        "k": KS, "n_random": N_RANDOM},
           "features": {}}

    for name, X in feats.items():
        mu, sd = X.mean(0), X.std(0) + 1e-6
        X = (X - mu) / sd
        # subject -> fit/test rows and per-subject class centroids
        split = {}
        for s in subjects:
            rows = np.where(subj == s)[0]
            gs = np.unique(grp[rows]); rng.shuffle(gs)
            fg = set(gs[:len(gs) // 2])
            rf = rows[np.isin(grp[rows], list(fg))]
            rt = rows[~np.isin(grp[rows], list(fg))]
            split[s] = (rf, rt)
        curves = {k: {"random": [], "active": []} for k in KS}
        cover = {k: {"random": [], "active": []} for k in KS}
        lam = {k: {"random": [], "active": []} for k in KS}
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
            Call = centroids(X[rf], song[rf], songs)
            call_mean, cg_mean = Call.mean(0), Cg.mean(0)
            Rall = procrustes(Call - call_mean, Cg - cg_mean)
            base["oracle"].append(evaluate(Rall, Cg - cg_mean, call_mean, X[rt], yte, songs))
            for k in KS:
                # ---- active ----
                counts = d_optimal_counts(G, k)
                avail = np.array([len(rf[song[rf] == c]) for c in songs])
                counts = np.minimum(counts, avail)
                picks = []
                for ci, c in enumerate(songs):
                    rows_c = rf[song[rf] == c]
                    if len(rows_c) == 0 or counts[ci] == 0:
                        continue
                    cen = X[rows_c].mean(0)
                    order = np.argsort(-(unit(X[rows_c]) @ unit(cen)))
                    picks.append(rows_c[order[:counts[ci]]])
                ap = np.concatenate(picks); ay = song[ap]
                present = np.array([c for c in songs if (ay == c).any()])
                Cs = centroids(X[ap], ay, present); Cga = Cg[[list(songs).index(c) for c in present]]
                wgt = np.array([(ay == c).sum() for c in present], float)
                cs_mean = np.average(Cs, axis=0, weights=wgt)
                cga_mean = np.average(Cga, axis=0, weights=wgt)
                R = procrustes(Cs - cs_mean, Cga - cga_mean, wgt)
                acc_a = evaluate(R, Cg - cga_mean, cs_mean, X[rt], yte, songs)
                curves[k]["active"].append(acc_a)
                cover[k]["active"].append(len(present))
                lam[k]["active"].append(lam_min(G, counts))
                # ---- random control ----
                accs, covs, lams = [], [], []
                for _ in range(N_RANDOM):
                    rp = rf[rng.choice(len(rf), k, replace=False)]
                    ry = song[rp]
                    pres = np.array([c for c in songs if (ry == c).any()])
                    Csr = centroids(X[rp], ry, pres)
                    Cgr = Cg[[list(songs).index(c) for c in pres]]
                    wg = np.array([(ry == c).sum() for c in pres], float)
                    csr_mean = np.average(Csr, axis=0, weights=wg)
                    cgr_mean = np.average(Cgr, axis=0, weights=wg)
                    Rr = procrustes(Csr - csr_mean, Cgr - cgr_mean, wg)
                    accs.append(evaluate(Rr, Cg - cgr_mean, csr_mean, X[rt], yte, songs))
                    covs.append(len(pres))
                    cnt = np.array([(ry == c).sum() for c in songs])
                    lams.append(lam_min(G, cnt))
                curves[k]["random"].append(float(np.mean(accs)))
                cover[k]["random"].append(float(np.mean(covs)))
                lam[k]["random"].append(float(np.mean(lams)))
        res["features"][name] = {
            "baselines": {k: float(np.mean(v)) for k, v in base.items()},
            "curve": {str(k): {"random": float(np.mean(curves[k]["random"])),
                               "active": float(np.mean(curves[k]["active"])),
                               "random_sd": float(np.std(curves[k]["random"], ddof=1)),
                               "active_sd": float(np.std(curves[k]["active"], ddof=1)),
                               "random_class_coverage": float(np.mean(cover[k]["random"])),
                               "active_class_coverage": float(np.mean(cover[k]["active"])),
                               "random_lam_min": float(np.mean(lam[k]["random"])),
                               "active_lam_min": float(np.mean(lam[k]["active"]))}
                       for k in KS},
        }
        b = res["features"][name]["baselines"]; c = res["features"][name]["curve"]
        print(f"\n{name}: zero {b['zero']:.3f} | within {b['within']:.3f} | oracle {b['oracle']:.3f}")
        print("  k     random  active  | cov r/a | lam_min r/a")
        for k in KS:
            r = c[str(k)]
            print(f"  {k:3d}   {r['random']:.3f}   {r['active']:.3f}  | "
                  f"{r['random_class_coverage']:.1f}/{r['active_class_coverage']:.1f} | "
                  f"{r['random_lam_min']:.2f}/{r['active_lam_min']:.2f}")

    json.dump(res, open(OUT / "k_shot.json", "w", encoding="utf-8"), indent=2)
    print("\nwrote", OUT / "k_shot.json")


if __name__ == "__main__":
    main()
