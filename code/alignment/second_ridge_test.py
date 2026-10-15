"""Does a second ridge (bridge(EEG) -> real-audio laion vector) actually help?

The existing bridge is already a ridge fit, so composing another ridge with it cannot beat the
best linear map of the same inputs -- what a second ridge CAN still do is (a) re-shrink/rotate
the unit-normalised output toward the true audio-vector distribution and (b) correct the fit if
the original bridge was trained against a different target definition.

Judged on held-out code-groups, against three references that must be reported together:
  identity      : use bridge(EEG) as-is
  constant      : predict the training-set mean vector  (laion unit vectors cluster hard, so
                  this scores high even with zero skill -- the trap we have been burned by)
  audio2audio   : a real audio vector vs another real audio vector = the scale reference
Also: cos between the corrected vector and the plain bridge output (how much changed).

Run: python src/train/second_ridge_test.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
sys.path.insert(0, str(M2 / "src" / "generate"))


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def ridge_fit(X, Y, lam):
    d = X.shape[1]
    A = X.T @ X + lam * np.eye(d, dtype=np.float64)
    Bv = np.linalg.solve(A, X.T @ Y.astype(np.float64))
    return Bv


def main():
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    group = np.load(DATA / "group.npy").astype(np.int64)

    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    target = z["target"].astype(np.float32)          # real-audio laion CLAP per cache row
    eeg = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    proj = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu",
                      weights_only=True)
    W, b = proj["weight"].numpy(), proj["bias"].numpy()
    v = eeg @ W.T + b
    orig = unit(v).astype(np.float32)
    j, mapped = 0, []
    for i in range(len(cond)):
        key = cond[i].tobytes()
        while j < len(orig) and orig[j].tobytes() != key:
            j += 1
        mapped.append(j); j += 1
    tgt = target[np.array(mapped)]                   # aligned to cond rows
    print(f"rows {len(cond)}  target aligned")

    # held-out groups, same split as the generator training
    uniq = np.unique(group)
    rng = np.random.default_rng(0)
    n_val = max(1, int(round(len(uniq) * 0.1)))
    val_g = set(uniq[rng.permutation(len(uniq))[:n_val]].tolist())
    va = np.array([g in val_g for g in group])
    tr, hv = ~va, va
    print(f"train rows {tr.sum()}  held-out rows {hv.sum()}  held-out groups {len(set(group[hv]))}")

    Cu, Tu = unit(cond), unit(tgt)
    mu_tr = Tu[tr].mean(0); mu_tr = mu_tr / np.linalg.norm(mu_tr)

    def cos_rows(P, T):
        return float((P * T).sum(1).mean())

    print(f"\n参照（全部在留 out 的组上）:")
    print(f"  常数基线（预测训练均值方向）      cos={cos_rows(np.tile(mu_tr, (hv.sum(), 1)), Tu[hv]):.4f}")
    print(f"  音频↔音频（真向量两两）            cos={float((Tu[hv] @ Tu[hv].T)[np.triu_indices(hv.sum(),1)].mean()):.4f}")
    print(f"  直接用桥输出 identity              cos={cos_rows(Cu[hv], Tu[hv]):.4f}")

    print(f"\n第二次岭回归 cond -> 真音频 laion（不同正则强度）:")
    best = None
    for lam in [1e-4, 1e-2, 1.0, 10.0, 100.0, 1e3]:
        B = ridge_fit(Cu[tr].astype(np.float64), Tu[tr], lam)
        P = unit(Cu[hv] @ B)
        c = cos_rows(P, Tu[hv])
        shift = float((P * Cu[hv]).sum(1).mean())
        print(f"  λ={lam:<8} cos(修正后, 真音频)={c:.4f}   cos(修正后, 原桥输出)={shift:.4f}")
        if best is None or c > best[1]:
            best = (lam, c, shift, P)

    lam, c, shift, P = best
    d = P - Cu[hv]
    print(f"\n判读: 最好 λ={lam} 让 cos 从 identity 到 {c:.4f}，"
          f"增益 {c - cos_rows(Cu[hv], Tu[hv]):+.4f}；"
          f"修正后与原向量余弦 {shift:.4f}（=改动幅度），"
          f"平均欧氏位移 {float(np.linalg.norm(d, axis=1).mean()):.4f}")
    print("若增益小而改动大，说明这一步只是把向量搬动了一圈而没多拿到信息；")
    print("若 cos 仍远低于音频↔音频，瓶颈就在 8 秒脑电本身可预测性，不在映射层数。")

    np.save(ROOT / "results" / "gen_eegcond" / "cond_ridge2_heldout.npy",
            np.concatenate([Cu[hv], P], axis=0))
    np.save(ROOT / "results" / "gen_eegcond" / "cond_ridge2_groups.npy", group[hv])


if __name__ == "__main__":
    main()
