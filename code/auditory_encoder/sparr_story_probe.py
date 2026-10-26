"""Within-subject story retrieval on SparrKULee test split.

Each subject listened to ~8 audiobook stories. For every non-overlapping 8-s
window we extract encoder content (or a bandpower baseline), then classify a
held-out window's story by nearest neighbour among the subject's other windows.
Chance is 1 / n_stories for that subject. This is a trial-wise (within-subject)
probe, complementing the subject-held-out envelope stress test.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from sparr_scale import FS, PATCH, WIN, build_model, list_files  # noqa: E402

BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def load_windows(rec, n_win=6):
    a = np.load(rec["path"], mmap_mode="r")
    total = a.shape[0]
    starts = [i * WIN for i in range(n_win) if i * WIN + WIN <= total]
    if not starts:
        return [], []
    xs = []
    for s in starts:
        x = np.asarray(a[s:s + WIN, :], dtype=np.float32).T
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
        xs.append(np.clip(x, -10, 10))
    return xs, starts


def bandpower_feat(x):
    from scipy.signal import welch
    f, p = welch(x, fs=FS, nperseg=min(256, x.shape[1]), axis=1)
    feats = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        feats.append(np.log(p[:, m].mean(1) + 1e-8))
    return np.concatenate(feats)


@torch.no_grad()
def extract_windows(model, recs, n_win=6):
    device = next(model.parameters()).device
    rows = []
    for r in recs:
        xs, starts = load_windows(r, n_win)
        if not xs:
            continue
        xb = torch.tensor(np.stack(xs), dtype=torch.float32, device=device)
        z, _, _ = model(xb)
        z = z.cpu().numpy()
        for k, st in enumerate(starts):
            rows.append({"subject": r["subject"], "story": r["story"], "start": int(st),
                         "z": z[k], "bp": bandpower_feat(xs[k])})
    return rows


def _loo_once(Z, y, n_classes):
    pred = []
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        cents = np.stack([Z[m & (y == c)].mean(0) if (m & (y == c)).any()
                          else np.full(Z.shape[1], np.inf) for c in range(n_classes)])
        pred.append(int(np.argmin(((cents - Z[i]) ** 2).sum(1))))
    return np.array(pred)


def loo_accuracy(rows, key, null_reps=0, seed=0):
    subj = sorted({r["subject"] for r in rows})
    per_subject = {}
    for s in subj:
        rs = [r for r in rows if r["subject"] == s]
        if len(rs) < 4:
            continue
        stories = sorted({r["story"] for r in rs})
        if len(stories) < 2:
            continue
        sid = {st: i for i, st in enumerate(stories)}
        Z = StandardScaler().fit_transform(np.stack([r[key] for r in rs]))
        y = np.array([sid[r["story"]] for r in rs])
        acc = float((_loo_once(Z, y, len(stories)) == y).mean())
        per_subject[s] = {"acc": acc, "n_windows": len(rs), "n_stories": len(stories),
                          "chance": 1.0 / len(stories)}
    accs = np.array([v["acc"] for v in per_subject.values()])
    chances = np.array([v["chance"] for v in per_subject.values()])
    out = {"mean_acc": float(accs.mean()) if len(accs) else float("nan"),
           "mean_chance": float(chances.mean()) if len(chances) else float("nan"),
           "n_subjects": int(len(accs)), "per_subject": per_subject}
    if null_reps:
        rng = np.random.default_rng(seed)
        null = []
        for _ in range(null_reps):
            accs_n = []
            for s in per_subject:
                rs = [r for r in rows if r["subject"] == s]
                stories = sorted({r["story"] for r in rs})
                sid = {st: i for i, st in enumerate(stories)}
                Z = StandardScaler().fit_transform(np.stack([r[key] for r in rs]))
                y = rng.permutation(np.array([sid[r["story"]] for r in rs]))
                accs_n.append(float((_loo_once(Z, y, len(stories)) == y).mean()))
            null.append(np.mean(accs_n))
        out["null_mean"] = float(np.mean(null))
        out["null_p95"] = float(np.percentile(null, 95))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/datasets/sparrkulee/split_data")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--test-subjects", type=int, default=8)
    ap.add_argument("--test-offset", type=int, default=0,
                    help="index of the first held-out subject in the sorted list")
    ap.add_argument("--n-win", type=int, default=6)
    ap.add_argument("--null-reps", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "results" / "sparr_story_probe" / "probe"))
    args = ap.parse_args()

    recs = [r for r in list_files(args.root) if r["split"] == "test"]
    subs = sorted({r["subject"] for r in recs})[args.test_offset:
                                                  args.test_offset + args.test_subjects]
    recs = [r for r in recs if r["subject"] in set(subs)]
    print(f"subjects {len(subs)} files {len(recs)}", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cargs = argparse.Namespace(**ck["args"])
    if ck.get("arch") == "patch":
        from patch_encoder import PatchEncoder
        model = PatchEncoder(n_ch=64, patch=120, n_time=WIN // 120,
                             d=cargs.d, depth=cargs.depth, heads=cargs.heads,
                             codebook=cargs.codebook)
    else:
        model = build_model(cargs)
    model.load_state_dict(ck["model"])
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    rows = extract_windows(model, recs, n_win=args.n_win)
    out = {"ckpt": str(args.ckpt), "params": params, "n_windows": len(rows),
           "subjects": len(subs), "methods": {}}
    for key in ("z", "bp"):
        out["methods"][key] = loo_accuracy(rows, key, null_reps=args.null_reps)
        m = out["methods"][key]
        extra = (f" null {m['null_mean']:.3f} p95 {m['null_p95']:.3f}"
                 if "null_mean" in m else "")
        print(f"{key}: acc {m['mean_acc']:.3f} chance {m['mean_chance']:.3f} "
              f"n_subj {m['n_subjects']}{extra}", flush=True)
    p = Path(args.out); p.mkdir(parents=True, exist_ok=True)
    (p / "eval.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", p / "eval.json")


if __name__ == "__main__":
    main()
