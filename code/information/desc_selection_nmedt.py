"""Description selection restricted to the 10 NMED-T candidates.

The held-out calibration subjects are NMED-T, which contains songs 0--9 only; songs
10--13 (NMED-H) can never be correct. The 14-candidate accuracy therefore has an
effective uniform chance of 1/10, and the majority-class floor is ~0.105, not 0.071.
This script reports both candidate sets and the constant-query / majority floors.

Run (CPU is fine): python code/information/desc_selection_nmedt.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from calib_head_pilot import DEV, GATE, OUT, unit_np  # noqa: E402
from calib_head_5fold import FOLDS  # noqa: E402
from calib_head_genre import embed, fit_rotation, load_head  # noqa: E402

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
DESC = ROOT / "results" / "descriptions" / "descriptions.json"
KS = (0, 10, 100, 300)


def main():
    d = json.load(open(DESC, encoding="utf-8"))
    D = unit_np(np.array(d["embedding_specific"], dtype=np.float32))
    D10 = D[:10]
    z = np.load(GATE, allow_pickle=True)
    cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)
    aldm = np.load(ROOT.parent / "mainv2" / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    w = torch.load(ROOT.parent / "mainv2" / "results" / "eeg_to_clap_proj.pt",
                   map_location="cpu", weights_only=True)
    ca = unit_np(aldm @ w["weight"].numpy().astype(np.float32).T + w["bias"].numpy().astype(np.float32))
    key = {ca[i].tobytes(): i for i in range(len(ca))}
    idx = np.array([key[cond[j].tobytes()] for j in range(len(cond))])
    X = torch.tensor(z["bandpower"][idx].astype(np.float32))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    T = unit_np(np.load(ROOT / "results" / "prompt_classifier" / "true_emb_all.npy").astype(np.float32))
    song = z["song"][idx].astype(int); subj = z["subject"][idx].astype(str)
    seg = z["seg2"][idx].astype(int); ds = z["dataset"][idx].astype(int)
    al = z["aligned8"][idx].astype(bool)

    arms = [f"k{k}" for k in KS]
    per = {a: [] for a in arms}
    per10 = {a: [] for a in arms}
    floors = []
    for f, meta in FOLDS.items():
        train_subs = [s for s in np.unique(subj)
                      if s not in meta["test"] and s not in meta["val"] and s in set(subj[ds == 0])]
        train_rows = np.where((ds == 0) & np.isin(subj, train_subs))[0]
        Pn = unit_np(np.stack([T[train_rows][song[train_rows] == c].mean(0) for c in range(10)]))
        head = load_head(OUT / f"calib_bandpower_laion_f{f}.pt", X.shape[1])
        Z = embed(head, X)
        fv = {a: [] for a in arms}; fv10 = {a: [] for a in arms}
        counts = np.zeros(10, int); n = 0
        rng = np.random.default_rng(f)
        for s in meta["test"]:
            rows = np.where(al & (subj == s))[0]
            segs = np.unique(seg[rows]); rng.shuffle(segs)
            fg = set(segs[:len(segs) // 2])
            rf = rows[np.isin(seg[rows], list(fg))]
            rt = rows[~np.isin(seg[rows], list(fg))]
            if len(rf) < 100 or len(rt) < 100:
                continue
            yte = song[rt]
            counts += np.bincount(yte, minlength=10); n += len(rt)
            fv["k0"].append(float(((Z[rt] @ D.T).argmax(1) == yte).mean()))
            fv10["k0"].append(float(((Z[rt] @ D10.T).argmax(1) == yte).mean()))
            for k in KS[1:]:
                R, m = fit_rotation(Z, rf, song, Pn, k=k)
                q = (Z[rt] - m) @ R
                fv[f"k{k}"].append(float(((q @ D.T).argmax(1) == yte).mean()))
                fv10[f"k{k}"].append(float(((q @ D10.T).argmax(1) == yte).mean()))
        for a in arms:
            per[a].append(float(np.mean(fv[a])))
            per10[a].append(float(np.mean(fv10[a])))
        floors.append(float(counts.max() / n))
        print(f"fold {f}: 14cand " + " ".join(f"{a} {np.mean(fv[a]):.3f}" for a in arms) +
              " | 10cand " + " ".join(f"{a} {np.mean(fv10[a]):.3f}" for a in arms) +
              f" | majority {counts.max()/n:.3f}", flush=True)

    res = {"protocol": {"test_songs": "NMED-T songs 0-9; NMED-H songs 10-13 not in held-out subjects",
                        "chance_14cand": 1 / 14, "chance_10cand": 0.10,
                        "majority_floor_mean": float(np.mean(floors))},
           "aggregate_14cand": {a: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
                                for a, v in per.items()},
           "aggregate_10cand": {a: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
                                for a, v in per10.items()},
           "folds_majority": floors}
    json.dump(res, open(OUT / "desc_selection_nmedt.json", "w", encoding="utf-8"), indent=2)
    print("\n=== 14 vs 10 candidates (5-fold) ===")
    for a in arms:
        print(f"{a:5} 14cand {res['aggregate_14cand'][a]['mean']:.3f}  "
              f"10cand {res['aggregate_10cand'][a]['mean']:.3f}")
    print("majority floor", res["protocol"]["majority_floor_mean"])
    print("wrote", OUT / "desc_selection_nmedt.json")


if __name__ == "__main__":
    main()
