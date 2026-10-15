"""Discrimination, not cosine: does the second ridge keep, or destroy, the per-group signal?

Raw cos is useless here (laion unit vectors cluster: predicting the training-mean direction
scores 0.82 with zero skill). So score retrieval: for each held-out group, rank its own
stimulus vector among all held-out group vectors using the given condition. Report top-1 and
mean rank against chance, plus margin (own minus mean of others) with paired t = margin in
units of its own standard error.

Compares: identity bridge(EEG) | second ridge at several strengths | constant predictor
(degenerate, should collapse to chance) | bridge(EEG) before normalisation.

Run: python src/train/second_ridge_retrieval.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def ridge(X, Y, lam):
    A = X.T @ X + lam * np.eye(X.shape[1], dtype=np.float64)
    return np.linalg.solve(A, X.T @ Y.astype(np.float64))


def score(P, T, chance_note):
    """P: per-group condition vectors; T: per-group true vectors; both [m,512]."""
    m = P.shape[0]
    S = P @ T.T
    own = np.diag(S)
    ranks, margins = [], []
    for i in range(m):
        order = np.argsort(-S[i])
        ranks.append(int(np.where(order == i)[0][0]) + 1)
        margins.append(S[i, i] - np.delete(S[i], i).mean())
    margins = np.array(margins)
    top1 = sum(r == 1 for r in ranks) / m
    t = margins.mean() / (margins.std(ddof=1) / np.sqrt(m) + 1e-12)
    print(f"  {chance_note:<34} top-1={top1:.3f} (chance {1/m:.3f})  "
          f"平均秩={np.mean(ranks):.2f} (chance {(m+1)/2:.1f})  "
          f"margin={margins.mean():+.4f} t={t:+.2f}")
    return dict(top1=top1, rank=float(np.mean(ranks)), margin=float(margins.mean()), t=float(t))


def main():
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    group = np.load(DATA / "group.npy").astype(np.int64)
    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    target = z["target"].astype(np.float32)
    eeg = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    proj = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu",
                      weights_only=True)
    W, b = proj["weight"].numpy(), proj["bias"].numpy()
    orig = unit(eeg @ W.T + b).astype(np.float32)
    j, mapped = 0, []
    for i in range(len(cond)):
        k = cond[i].tobytes()
        while j < len(orig) and orig[j].tobytes() != k:
            j += 1
        mapped.append(j); j += 1
    tgt = unit(target[np.array(mapped)])

    uniq = np.unique(group)
    rng = np.random.default_rng(0)
    n_val = max(1, int(round(len(uniq) * 0.1)))
    val_g = set(uniq[rng.permutation(len(uniq))[:n_val]].tolist())
    va = np.array([g in val_g for g in group])
    tr, hv = ~va, va

    # collapse to one row per held-out group (targets are identical inside a group)
    gvals = np.unique(group[hv])
    rows = [np.where(hv & (group == g))[0][0] for g in gvals]
    rows = np.array(rows)
    T = tgt[rows]
    C = unit(cond)[rows]
    print(f"{len(rows)} held-out groups used as the candidate pool")

    print("\n=== 区分能力（这才是生成器能吃下去的东西）===")
    score(C, T, "identity：直接用桥输出")
    mu = tgt[tr].mean(0); mu = mu / np.linalg.norm(mu)
    score(np.tile(mu, (len(rows), 1)), T, "常数：全预测训练均值方向")
    for lam in [1e-4, 1e-2, 1.0, 10.0, 100.0]:
        B = ridge(unit(cond)[tr].astype(np.float64), tgt[tr], lam)
        P = unit(C.astype(np.float64) @ B)
        score(P, T, f"第二次岭回归 λ={lam}")


if __name__ == "__main__":
    main()
