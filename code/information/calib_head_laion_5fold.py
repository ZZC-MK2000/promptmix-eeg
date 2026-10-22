"""Five-fold calibration-aware head on band-power with a laion-CLAP target.

The pilot above uses the AudioLDM-CLAP gate target. The main pipeline's prompt classifier
and generator use laion-CLAP, so this version trains the same head on the laion audio
embeddings (results/prompt_classifier/true_emb_all.npy, aligned to the 36,036 kept rows) and
evaluates k-shot song retrieval against the laion song centroids. Its head output is a
drop-in query for the 12-prompt classifier / frozen generator.

Run: python code/information/calib_head_laion_5fold.py
"""
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
from calib_head_5fold import FOLDS, evaluate_fold  # noqa: E402

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(GATE, allow_pickle=True)
    cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)
    aldm = np.load(ROOT.parent / "mainv2" / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    w = torch.load(ROOT.parent / "mainv2" / "results" / "eeg_to_clap_proj.pt",
                   map_location="cpu", weights_only=True)
    ca = unit_np(aldm @ w["weight"].numpy().astype(np.float32).T + w["bias"].numpy().astype(np.float32))
    key = {ca[i].tobytes(): i for i in range(len(ca))}
    idx = np.array([key[cond[j].tobytes()] for j in range(len(cond))])
    assert (idx >= 0).all() and len(np.unique(idx)) == len(idx)
    T = torch.tensor(unit_np(np.load(ROOT / "results" / "prompt_classifier" / "true_emb_all.npy").astype(np.float32)))
    X = torch.tensor(z["bandpower"][idx].astype(np.float32))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    song = z["song"][idx].astype(int); subj = z["subject"][idx].astype(str)
    seg = z["seg2"][idx].astype(int); ds = z["dataset"][idx].astype(int)
    al = z["aligned8"][idx].astype(bool)

    res = {"protocol": {"features": "frozen band-power 1804-d, standardised",
                        "target": "laion-CLAP (true_emb_all, kept rows)",
                        "head": "linear 1804->512, L2", "folds": 5,
                        "evaluation": "nearest laion song centroid (10-way)"},
           "folds": {}, "aggregate": {}}
    per = {name: {k: [] for k in KS} for name in ("clip_only", "calib")}
    for f, meta in FOLDS.items():
        train_subs = [s for s in np.unique(subj)
                      if s not in meta["test"] and s not in meta["val"] and s in set(subj[ds == 0])]
        train_rows = np.where((ds == 0) & np.isin(subj, train_subs))[0]
        sg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for r in train_rows:
            sg[subj[r]][song[r]][seg[r]].append(r)
        tr_t = torch.as_tensor(train_rows)
        P = unit(torch.stack([T[tr_t][torch.as_tensor(song[train_rows] == c)].mean(0) for c in range(10)]))
        res["folds"][f] = {}
        for name, use_calib in (("clip_only", False), ("calib", True)):
            print(f"fold {f} training {name} (laion target) ...", flush=True)
            head = train_head(X, T, P, sg, train_subs, use_calib, seed=f)
            torch.save(head.state_dict(), OUT / f"{name}_bandpower_laion_f{f}.pt")
            acc, _ = evaluate_fold(head, X, P, song, subj, seg, al, meta["test"], seed=f)
            res["folds"][f][name] = acc
            for k in KS:
                per[name][k].append(acc[k])
            print(f"fold {f} {name}: " + " ".join(f"k{k} {acc[k]:.3f}" for k in KS), flush=True)
    for name in per:
        res["aggregate"][name] = {str(k): {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
                                  for k, v in per[name].items()}
    out = OUT / "bandpower_laion_5fold.json"
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2)
    print("\n=== laion-target 5-fold ===")
    for name in ("clip_only", "calib"):
        print(f"{name:10} " + " ".join(
            f"k{k} {res['aggregate'][name][str(k)]['mean']:.3f}"
            f"±{res['aggregate'][name][str(k)]['sd']:.3f}" for k in KS))
    print("wrote", out)


if __name__ == "__main__":
    main()
