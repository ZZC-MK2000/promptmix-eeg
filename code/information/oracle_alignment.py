"""Feasibility diagnostic for the shared-content + subject-fiber decomposition.

Before writing identifiability theory, test its necessary condition:

  if a per-subject ORACLE alignment (fit on half of that subject's own windows with
  song labels) cannot lift cross-subject song decoding above chance, then a linear /
  orthogonal subject transform cannot be the missing piece and the theory is moot.

Protocol (NMED-T, 20 subjects x 10 songs):
  - split each subject's audio segments (group) 50/50 into fit/test;
  - class centroids C_s from the subject's FIT rows; global centroids C_g from the
    OTHER subjects' fit rows;
  - Procrustes: rotate C_s onto C_g (orthogonal, closed form), apply the same rotation
    to the subject's TEST rows, decode by nearest C_g;
  - controls: zero-shot (no rotation), within-subject (nearest C_s), and the
    class-centroid consistency before/after rotation.

Run: python code/information/oracle_alignment.py     (CPU)
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
OUT = ROOT / "results" / "oracle_alignment"
SEED = 0


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def fit_map(X, y, songs, n_dim):
    return np.stack([X[y == s].mean(0) for s in songs])


def procrustes(A, B):
    """Orthogonal R minimizing ||A R - B||_F, both centred by class mean.

    A, B are [k, d] with k << d (k = number of songs), so the full d x d SVD is
    avoided: the solution lives in the k-dim row spaces.
    """
    Ua, Sa, Vta = np.linalg.svd(A, full_matrices=False)   # Ua [k,k], Vta [k,d]
    Ub, Sb, Vtb = np.linalg.svd(B, full_matrices=False)
    M = (Sa[:, None] * (Ua.T @ Ub)) * Sb[None, :]         # [k,k]
    P, _, Qt = np.linalg.svd(M)
    U = Vta.T @ P                                         # [d,k]
    V = Vtb.T @ Qt.T                                      # [d,k]
    return U @ V.T                                        # [d,d], rank <= k


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(ROOT.parent / "mainv2" / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    # kept rows -> gate rows (same byte-identical bridge trick)
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

    feats = {"bandpower": z["bandpower"][idx][m].astype(np.float32),
             "labram": z["labram"][idx][m].astype(np.float32),
             "aldm": aldm[idx][m]}
    song, subj, grp = song[m], subj[m], grp[m]
    rng = np.random.default_rng(SEED)
    subjects = np.unique(subj)
    res = {"n_subjects": int(len(subjects)), "n_songs": int(len(songs)),
           "split": "per-subject 50/50 by audio segment", "features": {}}

    for name, X in feats.items():
        X = X.astype(np.float64)
        mu, sd = X.mean(0), X.std(0) + 1e-6
        X = (X - mu) / sd
        acc_zero, acc_align, acc_within, cons_before, cons_after = [], [], [], [], []
        for s in subjects:
            rows = np.where(subj == s)[0]
            gs = np.unique(grp[rows]); rng.shuffle(gs)
            fit_g = set(gs[:len(gs) // 2])
            rf = rows[np.isin(grp[rows], list(fit_g))]
            rt = rows[~np.isin(grp[rows], list(fit_g))]
            if len(rf) < 50 or len(rt) < 50:
                continue
            # class centroids (both need every song present)
            Cs = fit_map(X[rf], song[rf], songs, X.shape[1])
            ro = np.where(subj != s)[0]
            Cg = fit_map(X[ro], song[ro], songs, X.shape[1])
            # centre by class mean, rotate Cs onto Cg
            As, Bg = Cs - Cs.mean(0), Cg - Cg.mean(0)
            R = procrustes(As, Bg)
            Xt = X[rt] - Cs.mean(0)                      # match the Procrustes centring
            aligned = Xt @ R
            # decode by cosine to the global centroids
            zero = (unit(X[rt]) @ unit(Cg).T).argmax(1)
            alig = (unit(aligned) @ unit(Cg - Cg.mean(0)).T).argmax(1)
            within = (unit(X[rt]) @ unit(Cs).T).argmax(1)
            yt = song[rt]
            acc_zero.append(float((songs[zero] == yt).mean()))
            acc_align.append(float((songs[alig] == yt).mean()))
            acc_within.append(float((songs[within] == yt).mean()))
            # centroid consistency: mean cosine per class before/after rotation
            cons_before.append(float((unit(As) * unit(Bg)).sum(1).mean()))
            cons_after.append(float((unit(As @ R) * unit(Bg)).sum(1).mean()))
        res["features"][name] = {
            "zero_shot_acc": float(np.mean(acc_zero)),
            "procrustes_aligned_acc": float(np.mean(acc_align)),
            "within_subject_acc": float(np.mean(acc_within)),
            "centroid_cos_before": float(np.mean(cons_before)),
            "centroid_cos_after": float(np.mean(cons_after)),
            "per_subject_aligned": [round(x, 3) for x in acc_align],
            "chance": 0.1,
        }
        print(f"{name:10} zero-shot {np.mean(acc_zero):.3f} | Procrustes {np.mean(acc_align):.3f} "
              f"| within-subject {np.mean(acc_within):.3f} | centroid cos "
              f"{np.mean(cons_before):.3f} -> {np.mean(cons_after):.3f}")

    json.dump(res, open(OUT / "oracle_alignment.json", "w", encoding="utf-8"), indent=2)
    print("wrote", OUT / "oracle_alignment.json")


if __name__ == "__main__":
    main()
