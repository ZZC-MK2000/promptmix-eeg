"""SparrKULee scaling pilot: data scale and model size vs held-out story ID.

The downloaded split_data files are [T,64] EEG at ~120 Hz. We train the spatial-
spectral encoder with masked spectral + VICReg (no labels), then freeze it and fit a
10-way story classifier on training subjects; evaluate on held-out subjects.

This is a pilot for the scaling question, not a final speech benchmark.
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from spatial_spectral import SpatialSpectralEncoder, masked_spectral_loss, vicreg_terms  # noqa: E402

FS = 120.0
WIN = 960  # 8 s at 120 Hz
PATCH = 120  # 1 s


def parse(path):
    m = re.match(r"(train|val|test)_-_(sub-\d+)_-_((?:audiobook|podcast)_\d+)(?:_\d+)?_-_eeg\.npy$",
                 Path(path).name)
    if not m:
        return None
    return {"path": str(path), "split": m.group(1), "subject": m.group(2),
            "story": m.group(3)}


def list_files(root):
    out = []
    for p in sorted(Path(root).glob("*_eeg.npy")):
        rec = parse(p)
        if rec:
            out.append(rec)
    return out


class SparrWindows(Dataset):
    def __init__(self, records, per_file=4, win=WIN, seed=0, labeled=False):
        self.records = records
        self.per_file = per_file
        self.win = win
        self.seed = seed
        self.labeled = labeled
        self.stories = sorted({r["story"] for r in records})
        self.sid = {s: i for i, s in enumerate(self.stories)}

    def __len__(self):
        return len(self.records) * self.per_file

    def __getitem__(self, i):
        r = self.records[i // self.per_file]
        a = np.load(r["path"], mmap_mode="r")
        total = a.shape[0]
        if total <= self.win:
            start = 0
        else:
            start = int(np.random.default_rng(self.seed + i).integers(0, total - self.win + 1))
        x = np.asarray(a[start:start + self.win, :], dtype=np.float32).T  # [64,T]
        if x.shape[1] < self.win:
            x = np.pad(x, ((0, 0), (0, self.win - x.shape[1])))
        x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
        x = torch.from_numpy(np.clip(x, -10, 10))
        if self.labeled:
            return x, self.sid[r["story"]]
        return x


def build_model(args):
    return SpatialSpectralEncoder(n_ch=64, d=args.d, depth=args.depth, heads=args.heads,
                                  n_filters=args.n_filters, patch=PATCH, sfreq=FS,
                                  spectral_skip=True)


def train(args, train_records):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = SparrWindows(train_records, per_file=args.per_file, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    model = build_model(args).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.1)
    it = iter(dl); t0 = time.time(); logs = []
    for step in range(1, args.steps + 1):
        try:
            x = next(it)
        except StopIteration:
            it = iter(dl); x = next(it)
        x = x.to(device)
        mask = torch.rand(x.shape[0], WIN // PATCH, device=device) < 0.25
        none = ~mask.any(1)
        if none.any():
            mask[none, 0] = True
        model.train()
        content, recon, feats = model(x, mask)
        loss = masked_spectral_loss(recon, feats, mask) + 0.05 * vicreg_terms(content)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 100 == 0 or step == 1:
            rec = {"step": step, "loss": float(loss), "sec": round(time.time() - t0, 1)}
            logs.append(rec); print(json.dumps(rec), flush=True)
    ck = {"model": model.state_dict(), "args": vars(args), "arch": "v2_sparr",
          "params": n_params}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    path = out / f"encoder_s{args.steps}.pt"
    torch.save(ck, path)
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "logs": logs}, indent=2),
                                        encoding="utf-8")
    print("params", n_params, "wrote", path, flush=True)
    return model, n_params


@torch.no_grad()
def extract(model, records, per_file=2, seed=0, bs=64):
    device = next(model.parameters()).device
    ds = SparrWindows(records, per_file=per_file, seed=seed, labeled=True)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=0)
    Z, y = [], []
    model.eval()
    for x, lab in dl:
        z, _, _ = model(x.to(device))
        Z.append(z.cpu()); y.append(lab)
    return torch.cat(Z).numpy(), torch.cat(y).numpy(), ds.stories


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/datasets/sparrkulee/split_data")
    ap.add_argument("--out", default=str(ROOT / "results" / "sparr_scale" / "pilot"))
    ap.add_argument("--subjects", type=int, default=10)
    ap.add_argument("--test-subjects", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--per-file", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-filters", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    recs = list_files(args.root)
    subs = sorted({r["subject"] for r in recs})
    if len(subs) < args.test_subjects + 2:
        raise SystemExit(f"only {len(subs)} subjects downloaded")
    test_subs = subs[:args.test_subjects]
    pool = [s for s in subs if s not in test_subs][:args.subjects]
    train_recs = [r for r in recs if r["subject"] in set(pool) and r["split"] in ("train", "val")]
    test_recs = [r for r in recs if r["subject"] in set(test_subs) and r["split"] == "test"]
    if not train_recs:
        raise SystemExit("no train/val files downloaded yet; wait for the download or use --allow-all")
    print(f"files {len(recs)} train_subj {len(pool)} test_subj {len(test_subs)} "
          f"train_files {len(train_recs)} test_files {len(test_recs)}", flush=True)
    model, params = train(args, train_recs)
    Ztr, ytr, stories = extract(model, train_recs, per_file=2, seed=1)
    Zte, yte_raw, stories_te = extract(model, test_recs, per_file=2, seed=2)
    # map test story names to the training label space
    te_map = np.array([stories.index(s) if s in stories else -1 for s in stories_te])
    yte = te_map[yte_raw]
    # align classes to training stories
    keep_tr = np.isin(ytr, range(len(stories)))
    keep_te = np.isin(yte, range(len(stories)))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    clf.fit(Ztr[keep_tr], ytr[keep_tr])
    pred = clf.predict(Zte[keep_te])
    acc = float((pred == yte[keep_te]).mean()) if keep_te.any() else float("nan")
    maj = float(np.bincount(ytr[keep_tr]).max() / keep_tr.sum()) if keep_tr.any() else float("nan")
    res = {"args": vars(args), "params": params, "subjects": len(subs),
           "train_subjects": len(pool), "test_subjects": len(test_subs),
           "train_files": len(train_recs), "test_files": len(test_recs),
           "n_stories": len(stories), "chance": 1.0 / len(stories),
           "majority": maj, "story_acc": acc,
           "train_windows": int(keep_tr.sum()), "test_windows": int(keep_te.sum())}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
