"""14-way description selection with the calibrated laion query, plus shuffled control.

Descriptions: the frozen-CLAP "specific" tags in results/descriptions/descriptions.json
(audio-only, pre-registered vocabulary). The query is the calibration head output, calibrated
on the held-out subject's k anchors to the laion song codebook.

Arms:
  k=0        uncalibrated head output -> argmax cos(query, description)
  k=10/100/300  k-shot Procrustes -> same selection
  shuffled   the same, but the 14 descriptions are permuted (control)

Run: python code/information/desc_selection.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from calib_head_pilot import DEV, GATE, KS, OUT, unit_np  # noqa: E402
from calib_head_5fold import FOLDS  # noqa: E402
from calib_head_genre import embed, fit_rotation, load_head  # noqa: E402

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
DESC = ROOT / "results" / "descriptions" / "descriptions.json"
ARMS = ["k0", "k10", "k100", "k300", "shuffled_k10", "shuffled_k100", "shuffled_k300"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = json.load(open(DESC, encoding="utf-8"))
    D = unit_np(np.array(d["embedding_specific"], dtype=np.float32))      # 14 x 512
    Dsh = D[np.random.default_rng(0).permutation(14)]                  # shuffled control
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

    res = {"protocol": {"descriptions": "specific (frozen CLAP tags)", "n": 14,
                        "chance": 1 / 14, "query": "calibration head output (laion), k-shot Procrustes",
                        "shuffled": "14 descriptions permuted"},
           "folds": {}, "aggregate": {}}
    per = {a: [] for a in ARMS}
    for f, meta in FOLDS.items():
        train_subs = [s for s in np.unique(subj)
                      if s not in meta["test"] and s not in meta["val"] and s in set(subj[ds == 0])]
        train_rows = np.where((ds == 0) & np.isin(subj, train_subs))[0]
        Pn = unit_np(np.stack([T[train_rows][song[train_rows] == c].mean(0) for c in range(10)]))
        head = load_head(OUT / f"calib_bandpower_laion_f{f}.pt", X.shape[1])
        Z = embed(head, X)
        fv = {a: [] for a in ARMS}
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
            fv["k0"].append(float(((Z[rt] @ D.T).argmax(1) == yte).mean()))
            for k in (10, 100, 300):
                R, m = fit_rotation(Z, rf, song, Pn, k=k)
                q = (Z[rt] - m) @ R
                fv[f"k{k}"].append(float(((q @ D.T).argmax(1) == yte).mean()))
                fv[f"shuffled_k{k}"].append(float(((q @ Dsh.T).argmax(1) == yte).mean()))
        res["folds"][f] = {a: float(np.mean(v)) for a, v in fv.items() if v}
        for a in ARMS:
            if fv[a]:
                per[a].append(float(np.mean(fv[a])))
        print(f"fold {f}: " + " ".join(f"{a} {np.mean(fv[a]) if fv[a] else float('nan'):.3f}"
                                       for a in ARMS), flush=True)

    res["aggregate"] = {a: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
                        for a, v in per.items() if v}
    res["descriptor_only"] = d["stats_specific"]
    json.dump(res, open(OUT / "desc_selection.json", "w", encoding="utf-8"), indent=2)
    print("\n=== 14-way description selection (5-fold mean; chance 0.071) ===")
    for a in ARMS:
        if a in res["aggregate"]:
            g = res["aggregate"][a]
            print(f"{a:14} {g['mean']:.3f}±{g['sd']:.3f}")
    print("descriptor-only self-consistency", round(d["stats_specific"]["self_consistency_audio_text_top1"], 3))
    print("wrote", OUT / "desc_selection.json")


if __name__ == "__main__":
    main()
