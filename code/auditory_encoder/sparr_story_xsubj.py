"""Cross-subject story classification on SparrKULee.

Train a linear story classifier on encoder content from training subjects and
test on held-out subjects, restricted to stories heard by both groups. Tests
whether content carries stimulus-locked information that transfers across
listeners (a stricter protocol than within-subject retrieval).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from sparr_scale import build_model, list_files  # noqa: E402
from sparr_story_probe import extract_windows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/datasets/sparrkulee/split_data")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subjects", type=int, default=60)
    ap.add_argument("--test-subjects", type=int, default=8)
    ap.add_argument("--n-win", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "results" / "sparr_story_xsubj" / "probe"))
    args = ap.parse_args()

    recs = [r for r in list_files(args.root) if r["split"] == "test"]
    subs = sorted({r["subject"] for r in recs})
    test_subs = set(subs[:args.test_subjects])
    pool = [s for s in subs if s not in test_subs][:args.subjects]
    tr_recs = [r for r in recs if r["subject"] in set(pool)]
    te_recs = [r for r in recs if r["subject"] in test_subs]

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cargs = argparse.Namespace(**ck["args"])
    model = build_model(cargs)
    model.load_state_dict(ck["model"])
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    tr = extract_windows(model, tr_recs, n_win=args.n_win)
    te = extract_windows(model, te_recs, n_win=args.n_win)
    stories = sorted({r["story"] for r in te})
    tr = [r for r in tr if r["story"] in set(stories)]
    te = [r for r in te if r["story"] in set(stories)]
    sid = {s: i for i, s in enumerate(stories)}
    ytr = np.array([sid[r["story"]] for r in tr])
    yte = np.array([sid[r["story"]] for r in te])

    out = {"ckpt": str(args.ckpt), "params": params, "subjects": len(pool),
           "test_subjects": len(test_subs), "stories": stories,
           "n_train_windows": len(tr), "n_test_windows": len(te),
           "chance": 1.0 / len(stories),
           "majority": float(np.bincount(ytr).max() / len(ytr)), "methods": {}}
    for key in ("z", "bp"):
        Xtr = np.stack([r[key] for r in tr]); Xte = np.stack([r[key] for r in te])
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, C=1.0))
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        acc = float((pred == yte).mean())
        per_class = {stories[c]: float((pred[yte == c] == c).mean())
                     for c in range(len(stories)) if (yte == c).any()}
        out["methods"][key] = {"acc": acc, "per_class": per_class}
        print(f"{key}: acc {acc:.3f} chance {out['chance']:.3f} "
              f"majority {out['majority']:.3f}", flush=True)
    p = Path(args.out); p.mkdir(parents=True, exist_ok=True)
    (p / "eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", p / "eval.json")


if __name__ == "__main__":
    main()
