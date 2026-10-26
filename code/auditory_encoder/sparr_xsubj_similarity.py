"""How much same-time stimulus signal is shared across subjects?

For one story, compare feature vectors at the same audio time across subjects
against time-shifted pairs. Above-zero delta means a listener-independent
stimulus component exists; near-zero means cross-subject alignment has little
to learn from this feature.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats
from scipy.signal import welch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from sparr_scale import FS, WIN, build_model, list_files  # noqa: E402

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def windows(path, n_win):
    a = np.load(path)
    starts = np.linspace(0, a.shape[0] - WIN, n_win).astype(int)
    xs = []
    for st in starts:
        x = np.asarray(a[st:st + WIN], dtype=np.float32).T
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
        xs.append(np.clip(x, -10, 10))
    return xs


def bandpower(x):
    f, p = welch(x, fs=FS, nperseg=min(256, x.shape[1]), axis=1)
    return np.concatenate([np.log(p[:, (f >= lo) & (f < hi)].mean(1) + 1e-8)
                           for lo, hi in BANDS])


def similarity(F, shift):
    same, diff = [], []
    for i in range(len(F)):
        for j in range(i + 1, len(F)):
            for w in range(F.shape[1]):
                same.append(float((F[i, w] * F[j, w]).mean()))
                diff.append(float((F[i, w] * F[j, (w + shift) % F.shape[1]]).mean()))
    same, diff = np.array(same), np.array(diff)
    d = same - diff
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    return {"same": float(same.mean()), "shifted": float(diff.mean()),
            "delta": float(d.mean()), "ci95": [float(ci[0]), float(ci[1])],
            "t": float(stats.ttest_rel(same, diff).statistic), "n_pairs": int(len(d))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/datasets/sparrkulee/split_data")
    ap.add_argument("--story", default="audiobook_1")
    ap.add_argument("--split", default="train")
    ap.add_argument("--subjects", type=int, default=20)
    ap.add_argument("--start-subject", type=int, default=0,
                    help="offset into the sorted subject list (use to select held-out subjects)")
    ap.add_argument("--exclude-pool", type=int, default=0,
                    help="exclude N training subjects after the first 8 test subjects")
    ap.add_argument("--n-win", type=int, default=12)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=str(ROOT / "results" / "sparr_xsubj_similarity" / "bp.json"))
    args = ap.parse_args()

    all_recs = list_files(args.root)
    if args.exclude_pool:
        all_subs = sorted({r["subject"] for r in all_recs})
        excl = set(all_subs[8:8 + args.exclude_pool])
        all_recs = [r for r in all_recs if r["subject"] not in excl]
    recs = [r for r in all_recs
            if r["split"] == args.split and r["story"] == args.story]
    subs = sorted({r["subject"] for r in recs})[args.start_subject:
                                                  args.start_subject + args.subjects]
    recs = [r for r in recs if r["subject"] in set(subs)]
    print(f"subjects {len(recs)} story {args.story}", flush=True)

    model = None
    if args.ckpt:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        model = build_model(argparse.Namespace(**ck["args"]))
        model.load_state_dict(ck["model"])
        model.eval()

    feat = {"bp": [], "z": []}
    with torch.no_grad():
        for r in recs:
            xs = windows(r["path"], args.n_win)
            feat["bp"].append(np.stack([bandpower(x) for x in xs]))
            if model is not None:
                z, _, _ = model(torch.tensor(np.stack(xs), dtype=torch.float32))
                feat["z"].append(z.numpy())
    out = {"story": args.story, "split": args.split, "subjects": subs,
           "n_win": args.n_win, "ckpt": args.ckpt, "metrics": {}}
    for k, v in feat.items():
        if not v:
            continue
        F = np.stack(v)
        F = (F - F.mean(1, keepdims=True)) / (F.std(1, keepdims=True) + 1e-6)
        out["metrics"][k] = similarity(F, max(1, args.n_win // 2))
        print(k, json.dumps(out["metrics"][k]), flush=True)
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
