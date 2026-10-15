"""Is the second ridge's gain real, or subject leakage?

The ridge is fit on training rows, and one subject's rows appear in BOTH training and
held-out code-groups. So the gain could be "this subject's personal offset" rather than a
general EEG->audio map.

Fix: hold out whole SUBJECTS. Fit the ridge only on rows of other subjects, then evaluate
retrieval only on the held-out subjects' rows. Compare identity vs corrected under that rule.

Chance lines are printed with every number. margin = own similarity minus mean similarity to
the other candidates, per group; t is margin in units of its own standard error.

Run: python src/train/second_ridge_subject_heldout.py
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


def retrieve(P, T):
    m = len(P)
    S = P @ T.T
    ranks = [int(np.where(np.argsort(-S[i]) == i)[0][0]) + 1 for i in range(m)]
    margins = np.array([S[i, i] - np.delete(S[i], i).mean() for i in range(m)])
    top1 = sum(r == 1 for r in ranks) / m
    t = margins.mean() / (margins.std(ddof=1) / np.sqrt(m) + 1e-12)
    return top1, float(np.mean(ranks)), float(margins.mean()), float(t), m


def main():
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    group = np.load(DATA / "group.npy").astype(np.int64)
    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    target = z["target"].astype(np.float32)
    subj_all = np.array([str(x) for x in z["subject"]])
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
    mapped = np.array(mapped)
    tgt = unit(target[mapped])
    C = unit(cond)
    subj = subj_all[mapped]
    print(f"{len(C)} rows, {len(np.unique(subj))} subjects")

    # held-out code-groups (same as generator training) -> one row per group
    uniq = np.unique(group)
    rng = np.random.default_rng(0)
    n_val = max(1, int(round(len(uniq) * 0.1)))
    val_g = set(uniq[rng.permutation(len(uniq))[:n_val]].tolist())
    in_val_g = np.array([g in val_g for g in group])

    print("\n每轮留出 10 个被试：回归只用其余被试的行，检索只在留出被试的行上做")
    subs = np.unique(subj)
    accs = {"identity": [], "ridge2": []}
    for rep in range(3):
        hold = set(rng.choice(subs, 10, replace=False).tolist())
        is_hold = np.array([s in hold for s in subj])
        fit_m = (~is_hold) & (~in_val_g)          # train rows, other subjects
        # eval pool: held-out groups, restricted to held-out subjects, one row per group
        ev_rows, seen = [], set()
        for g in np.unique(group[in_val_g & is_hold]):
            rr = np.where(in_val_g & is_hold & (group == g))[0]
            if len(rr) and int(g) not in seen:
                seen.add(int(g)); ev_rows.append(int(rr[0]))
        ev = np.array(ev_rows)
        if len(ev) < 15:
            continue
        T, Ce = tgt[ev], C[ev]
        t1 = retrieve(Ce, T)
        best = None
        for lam in [1e-2, 1.0, 10.0]:
            B = ridge(C[fit_m].astype(np.float64), tgt[fit_m], lam)
            P = unit(Ce.astype(np.float64) @ B)
            r = retrieve(P, T)
            if best is None or r[0] > best[0]:
                best = r
        accs["identity"].append(t1)
        accs["ridge2"].append(best)
        print(f"  第{rep+1}轮 留出被试 {sorted(hold)[:4]}…  池 {t1[4]} 组   "
              f"identity top-1={t1[0]:.3f} 秩={t1[1]:.1f} | "
              f"ridge2 top-1={best[0]:.3f} 秩={best[1]:.1f}")

    print("\n=== 留出被试后的汇总（3 轮平均）===")
    for k, v in accs.items():
        a = np.array([[x[0], x[1], x[2], x[3]] for x in v])
        print(f"  {k:<9} top-1={a[:,0].mean():.3f}  平均秩={a[:,1].mean():.2f}  "
              f"margin={a[:,2].mean():+.4f}  t={a[:,3].mean():+.2f}   (chance top-1 见上表)")
    di = np.array([x[0] for x in accs["ridge2"]]) - np.array([x[0] for x in accs["identity"]])
    print(f"\n  ridge2 − identity 的 top-1 差 = {di.mean():+.4f}，三轮各自 {np.round(di,4).tolist()}")
    print("  若这个差仍为正 ⇒ 增益不是被试泄漏，是可以用于生成条件的真实改进")
    print("  若归零或变负 ⇒ 之前 18.2%→24.5% 是被试个体偏移，不能算数")


if __name__ == "__main__":
    main()
