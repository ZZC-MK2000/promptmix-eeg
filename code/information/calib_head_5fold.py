"""Five-fold feasibility of the calibration-aware head (band-power / LaBraM-200).

Extends calib_head_pilot.py to all five subject-held-out folds. Per fold: train two heads
on the training subjects (clip_only vs clip + differentiable-Procrustes calibration CE),
evaluate on the four held-out subjects with the same k-shot protocol, and aggregate.

Run: python code/information/calib_head_5fold.py --features bandpower
     python code/information/calib_head_5fold.py --features labram
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from calib_head_pilot import (DEV, GATE, KS, OUT, procrustes_torch, train_head,
                              unit, unit_np, pick_rows_eval)  # noqa: E402

FOLDS = {
    0: {"test": ["02", "03", "04", "05"], "val": ["06", "07"]},
    1: {"test": ["06", "07", "08", "09"], "val": ["04", "05"]},
    2: {"test": ["10", "11", "12", "13"], "val": ["06", "07"]},
    3: {"test": ["14", "15", "16", "17"], "val": ["08", "09"]},
    4: {"test": ["19", "20", "21", "23"], "val": ["10", "11"]},
}


def evaluate_fold(head, X, P, song, subj, seg, al, test_subjects, seed=0):
    head.eval()
    P = unit(P.to(DEV))
    Pn = unit_np(P.cpu().numpy())
    rng = np.random.default_rng(seed)
    acc = {k: [] for k in KS}
    with torch.no_grad():
        Z = unit(head(X.to(DEV))).cpu().numpy()
    for s in test_subjects:
        rows = np.where(al & (subj == s))[0]
        segs = np.unique(seg[rows]); rng.shuffle(segs)
        fg = set(segs[:len(segs) // 2])
        rf = rows[np.isin(seg[rows], list(fg))]
        rt = rows[~np.isin(seg[rows], list(fg))]
        if len(rf) < 100 or len(rt) < 100:
            continue
        yte = song[rt]
        for k in KS:
            if k == 0:
                pred = (unit_np(Z[rt]) @ Pn.T).argmax(1)
            else:
                picks = []
                for c in range(10):
                    rc = rf[song[rf] == c]
                    if len(rc) == 0:
                        continue
                    picks.append(pick_rows_eval(rc, Z, min(k // 10, len(rc))))
                ap = np.concatenate(picks); ay = song[ap]
                present = np.array([c for c in range(10) if (ay == c).any()])
                Cs = unit_np(np.stack([Z[ap][ay == c].mean(0) for c in present]))
                Pp = Pn[present]
                A = torch.tensor(Cs - Cs.mean(0), dtype=torch.float32, device=DEV)
                B = torch.tensor(Pp - Pp.mean(0), dtype=torch.float32, device=DEV)
                R = procrustes_torch(A, B).cpu().numpy()
                test = (Z[rt] - Cs.mean(0)) @ R
                pred = (unit_np(test) @ Pn.T).argmax(1)
            acc[k].append(float((pred == yte).mean()))
    return {k: float(np.mean(v)) for k, v in acc.items()}, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="bandpower", choices=["labram", "bandpower"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(GATE, allow_pickle=True)
    X = torch.tensor(z[args.features].astype(np.float32))
    T = torch.tensor(z["target"].astype(np.float32))
    P = torch.tensor(z["proto"][:10].astype(np.float32))
    song = z["song"].astype(int); subj = z["subject"].astype(str)
    seg = z["seg2"].astype(int); ds = z["dataset"].astype(int); al = z["aligned8"].astype(bool)

    res = {"protocol": {"features": f"frozen {args.features}", "head": f"linear {X.shape[1]}->512, L2",
                        "folds": 5, "target": "AudioLDM-CLAP gate target",
                        "epochs": 30, "batch": "4 subjects x 10 songs x (1 support + 1 query)"},
           "folds": {}, "aggregate": {}}
    per = {name: {k: [] for k in KS} for name in ("clip_only", "calib")}
    for f, meta in FOLDS.items():
        train_subs = [s for s in np.unique(subj)
                      if s not in meta["test"] and s not in meta["val"] and s in set(subj[ds == 0])]
        train_rows = np.where((ds == 0) & np.isin(subj, train_subs))[0]
        sg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for r in train_rows:
            sg[subj[r]][song[r]][seg[r]].append(r)
        res["folds"][f] = {}
        for name, use_calib in (("clip_only", False), ("calib", True)):
            print(f"fold {f} training {name} ...", flush=True)
            head = train_head(X, T, P, sg, train_subs, use_calib, seed=f)
            torch.save(head.state_dict(), OUT / f"{name}_{args.features}_f{f}.pt")
            acc, _ = evaluate_fold(head, X, P, song, subj, seg, al, meta["test"], seed=f)
            res["folds"][f][name] = acc
            for k in KS:
                per[name][k].append(acc[k])
            print(f"fold {f} {name}: " + " ".join(f"k{k} {acc[k]:.3f}" for k in KS), flush=True)

    for name in per:
        res["aggregate"][name] = {
            str(k): {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
            for k, v in per[name].items()}
    out = OUT / f"{args.features}_5fold.json"
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2)
    print("\n=== 5-fold aggregate ===")
    for name in ("clip_only", "calib"):
        print(f"{name:10} " + " ".join(
            f"k{k} {res['aggregate'][name][str(k)]['mean']:.3f}"
            f"±{res['aggregate'][name][str(k)]['sd']:.3f}" for k in KS))
    print("wrote", out)


if __name__ == "__main__":
    main()
