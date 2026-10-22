"""Genre-prompt readout of the calibration-aware head (laion-CLAP target).

Loads the five laion-target heads from calib_head_laion_5fold.py and compares, on held-out
subjects:

  mainline        the existing laion bridge output `cond` (transductive reference)
  head_zero       calibration head, no calibration; classifier trained on uncalibrated outputs
  head_calib_k    calibration head, k-shot Procrustes; classifier trained on calibrated outputs
  clip_zero       the clip-only ablation, no calibration
  clip_calib_k    the clip-only ablation, k-shot calibration

Metrics: accuracy, balanced accuracy, Cohen's kappa (the oracle labels are skewed).

Run: python code/information/calib_head_genre.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from calib_head_pilot import (DEV, GATE, KS, OUT, procrustes_torch, unit,
                              unit_np, pick_rows_eval)  # noqa: E402
from calib_head_5fold import FOLDS  # noqa: E402

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
ARMS = ["mainline", "head_zero", "head_calib_10", "head_calib_100", "head_calib_300",
        "clip_zero", "clip_calib_10", "clip_calib_100", "clip_calib_300"]


def load_head(path, d_in):
    h = nn.Linear(d_in, 512)
    h.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return h.to(DEV).eval()


@torch.no_grad()
def embed(head, X):
    return unit(head(X.to(DEV))).cpu().numpy()


def fit_rotation(Z, rows, song, Pn, k=None):
    """Fit one subject's Procrustes on `rows`; return (R, center) for applying elsewhere."""
    picks = []
    for c in range(10):
        rc = rows[song[rows] == c]
        if len(rc) == 0:
            continue
        picks.append(rc if k is None else pick_rows_eval(rc, Z, min(k // 10, len(rc))))
    ap = np.concatenate(picks); ay = song[ap]
    present = np.array([c for c in range(10) if (ay == c).any()])
    Cs = unit_np(np.stack([Z[ap][ay == c].mean(0) for c in present]))
    Pp = Pn[present]
    A = torch.tensor(Cs - Cs.mean(0), dtype=torch.float32, device=DEV)
    B = torch.tensor(Pp - Pp.mean(0), dtype=torch.float32, device=DEV)
    return procrustes_torch(A, B).cpu().numpy(), Cs.mean(0)


def metrics(y, p):
    return {"acc": float((p == y).mean()), "bal": float(balanced_accuracy_score(y, p)),
            "kappa": float(cohen_kappa_score(y, p))}


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
    X = torch.tensor(z["bandpower"][idx].astype(np.float32))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    T = unit_np(np.load(ROOT / "results" / "prompt_classifier" / "true_emb_all.npy").astype(np.float32))
    oracle = np.load(ROOT / "results" / "library_b" / "oracle_all.npy").astype(int)
    song = z["song"][idx].astype(int); subj = z["subject"][idx].astype(str)
    seg = z["seg2"][idx].astype(int); ds = z["dataset"][idx].astype(int)
    al = z["aligned8"][idx].astype(bool)

    res = {"protocol": {"labels": "corpus-matched 12-class oracle (skewed; kappa is the honest statistic)",
                        "target": "laion-CLAP", "folds": 5,
                        "calibration": "per-subject Procrustes to the laion song codebook"},
           "folds": {}, "aggregate": {}}
    per = {a: {"acc": [], "bal": [], "kappa": []} for a in ARMS}

    for f, meta in FOLDS.items():
        train_subs = [s for s in np.unique(subj)
                      if s not in meta["test"] and s not in meta["val"] and s in set(subj[ds == 0])]
        train_rows = np.where((ds == 0) & np.isin(subj, train_subs))[0]
        Pn = unit_np(np.stack([T[train_rows][song[train_rows] == c].mean(0) for c in range(10)]))
        hcal = load_head(OUT / f"calib_bandpower_laion_f{f}.pt", X.shape[1])
        hclip = load_head(OUT / f"clip_only_bandpower_laion_f{f}.pt", X.shape[1])
        Zcal = embed(hcal, X); Zclip = embed(hclip, X)

        # calibrate the training subjects with all their rows
        Zcal_tr = Zcal.copy(); Zclip_tr = Zclip.copy()
        rng = np.random.default_rng(f)
        for t in train_subs:
            r_t = np.where(al & (subj == t))[0]
            if len(r_t) < 100:
                r_t = np.where(subj == t)[0]
            R, m = fit_rotation(Zcal, r_t, song, Pn)
            Zcal_tr[r_t] = (Zcal[r_t] - m) @ R
            R, m = fit_rotation(Zclip, r_t, song, Pn)
            Zclip_tr[r_t] = (Zclip[r_t] - m) @ R

        clf = {
            "mainline": LogisticRegression(max_iter=2000, C=1.0).fit(cond[train_rows], oracle[train_rows]),
            "head_zero": LogisticRegression(max_iter=2000, C=1.0).fit(Zcal[train_rows], oracle[train_rows]),
            "clip_zero": LogisticRegression(max_iter=2000, C=1.0).fit(Zclip[train_rows], oracle[train_rows]),
        }
        for k in (10, 100, 300):
            clf[f"head_calib_{k}"] = LogisticRegression(max_iter=2000, C=1.0).fit(Zcal_tr[train_rows], oracle[train_rows])
            clf[f"clip_calib_{k}"] = LogisticRegression(max_iter=2000, C=1.0).fit(Zclip_tr[train_rows], oracle[train_rows])

        fv = {a: {"acc": [], "bal": [], "kappa": []} for a in ARMS}
        for s in meta["test"]:
            rows = np.where(al & (subj == s))[0]
            segs = np.unique(seg[rows]); rng.shuffle(segs)
            fg = set(segs[:len(segs) // 2])
            rf = rows[np.isin(seg[rows], list(fg))]
            rt = rows[~np.isin(seg[rows], list(fg))]
            if len(rf) < 100 or len(rt) < 100:
                continue
            yte = oracle[rt]
            preds = {
                "mainline": clf["mainline"].predict(cond[rt]),
                "head_zero": clf["head_zero"].predict(Zcal[rt]),
                "clip_zero": clf["clip_zero"].predict(Zclip[rt]),
            }
            for k in (10, 100, 300):
                R, m = fit_rotation(Zcal, rf, song, Pn, k=k)
                preds[f"head_calib_{k}"] = clf[f"head_calib_{k}"].predict((Zcal[rt] - m) @ R)
                R, m = fit_rotation(Zclip, rf, song, Pn, k=k)
                preds[f"clip_calib_{k}"] = clf[f"clip_calib_{k}"].predict((Zclip[rt] - m) @ R)
            for a, p in preds.items():
                mm = metrics(yte, p)
                for x in ("acc", "bal", "kappa"):
                    fv[a][x].append(mm[x])
        res["folds"][f] = {a: {x: float(np.mean(v)) for x, v in fv[a].items()} for a in ARMS}
        for a in ARMS:
            for x in ("acc", "bal", "kappa"):
                per[a][x].append(float(np.mean(fv[a][x])))
        print(f"fold {f}: " + " | ".join(
            f"{a} acc {np.mean(fv[a]['acc']):.3f} k {np.mean(fv[a]['kappa']):.3f}" for a in ARMS), flush=True)

    res["aggregate"] = {a: {x: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))}
                            for x, v in per[a].items()} for a in ARMS}
    out = OUT / "genre_readout.json"
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2)
    print("\n=== genre readout (5-fold mean) ===")
    for a in ARMS:
        g = res["aggregate"][a]
        print(f"{a:16} acc {g['acc']['mean']:.3f}±{g['acc']['sd']:.3f} "
              f"bal {g['bal']['mean']:.3f} kappa {g['kappa']['mean']:.3f}")
    print("wrote", out)


if __name__ == "__main__":
    main()
