"""Same-protocol supervised baselines on NMED-T (per-subject trial-wise 50/50).

The encoder comparison in the paper is frozen-feature + logistic. Here each
baseline is trained from scratch on the same per-subject, per-song train half
and evaluated on the held-out half. Models are standard EEG architectures from
braindecode; BIOT is included as an architecture control only (pretrained
weights are not available locally / HF is unreachable).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code" / "information"))
from eval_finetune_local75 import nmed_windows  # noqa: E402

from braindecode.models import (  # noqa: E402
    BIOT, CBraMod, Deep4Net, EEGConformer, EEGNet, ShallowFBCSPNet,
)


def make_model(name, n_ch=19, n_times=1600, n_out=10):
    kw = dict(n_chans=n_ch, n_times=n_times, sfreq=200,
              input_window_seconds=n_times / 200, n_outputs=n_out)
    if name == "eegnet":
        return EEGNet(**kw)
    if name == "shallow":
        return ShallowFBCSPNet(**kw)
    if name == "deep4":
        return Deep4Net(**kw)
    if name == "conformer":
        return EEGConformer(**kw)
    if name == "biot":
        return BIOT(**kw)
    if name == "cbramod":
        return CBraMod(**kw)
    raise ValueError(name)


def norm(x):
    x = x.float()
    x = x - x.mean(-1, keepdim=True)
    return torch.clamp(x / (x.std(-1, keepdim=True) + 1e-6), -10, 10)


def split_subject(song, subj, s, rng):
    rs = np.where(subj == s)[0]
    tr, te, ytr, yte = [], [], [], []
    for c in range(10):
        rr = rs[song[rs] == c]
        if len(rr) < 4:
            continue
        rr = rng.permutation(rr)
        h = len(rr) // 2
        tr.append(rr[:h]); te.append(rr[h:])
        ytr += [c] * len(rr[:h]); yte += [c] * len(rr[h:])
    if len(set(ytr)) < 10:
        return None
    return np.concatenate(tr), np.concatenate(te), np.array(ytr), np.array(yte)


def train_one(X, song, subj, s, seed, name, device, epochs, batch):
    rng = np.random.default_rng(seed)
    sp = split_subject(song, subj, s, rng)
    if sp is None:
        return None
    tr, te, ytr, yte = sp
    model = make_model(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    Xtr = torch.from_numpy(X[tr]).float(); Xte = torch.from_numpy(X[te]).float()
    ytr_t = torch.from_numpy(ytr).long(); yte_t = torch.from_numpy(yte).long()
    model.train()
    n = len(tr)
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            xb = norm(Xtr[idx].to(device)); yb = ytr_t[idx].to(device)
            loss = ce(model(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    pred = []
    with torch.no_grad():
        for i in range(0, len(te), batch):
            xb = norm(Xte[i:i + batch].to(device))
            pred.append(model(xb).argmax(1).cpu())
    return float(accuracy_score(yte, torch.cat(pred).numpy()))


def train_pooled(X, song, subj, seed, name, device, epochs, batch):
    rng = np.random.default_rng(seed)
    trs, tes, ytrs, ytes = [], [], [], []
    for s in sorted(set(subj.tolist())):
        sp = split_subject(song, subj, s, rng)
        if sp is None:
            continue
        tr, te, ytr, yte = sp
        trs.append(tr); tes.append(te); ytrs.append(ytr); ytes.append(yte)
    tr = np.concatenate(trs); te = np.concatenate(tes)
    ytr = np.concatenate(ytrs); yte = np.concatenate(ytes)
    model = make_model(name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    Xtr = torch.from_numpy(X[tr]).float(); Xte = torch.from_numpy(X[te]).float()
    ytr_t = torch.from_numpy(ytr).long(); yte_t = torch.from_numpy(yte).long()
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(len(tr))
        for i in range(0, len(tr), batch):
            idx = perm[i:i + batch]
            xb = norm(Xtr[idx].to(device)); yb = ytr_t[idx].to(device)
            loss = ce(model(xb), yb)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval(); pred = []
    with torch.no_grad():
        for i in range(0, len(te), batch):
            pred.append(model(norm(Xte[i:i + batch].to(device))).argmax(1).cpu())
    return float(accuracy_score(yte, torch.cat(pred).numpy()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="eegnet,shallow,deep4,conformer")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--pooled", action="store_true",
                    help="train one model on all subjects' train halves per repetition")
    ap.add_argument("--out", default=str(ROOT / "results" / "compare_nmed_models" / "results.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, song, subj = nmed_windows()
    X = X.astype(np.float32)
    subs = sorted(set(subj.tolist()))
    print(f"X {X.shape} subjects {len(subs)} models {args.models}", flush=True)
    out = {"protocol": {"dataset": "NMED-T", "split": "per-subject per-song 50/50",
                        "chance": 0.10, "reps": args.reps, "epochs": args.epochs,
                        "note": "from-scratch supervised; same splits as frozen probes" if not args.pooled else "from-scratch supervised, pooled over training halves"},
           "models": {}}
    for name in args.models.split(","):
        t0 = time.time(); vals = []
        for r in range(args.reps):
            if args.pooled:
                a = train_pooled(X, song, subj, 1000 * r, name, device,
                                 args.epochs, args.batch)
                vals.append(float(a))
            else:
                accs = []
                for s in subs:
                    a = train_one(X, song, subj, s, 1000 * r + int(s), name,
                                  device, args.epochs, args.batch)
                    if a is not None:
                        accs.append(a)
                vals.append(float(np.mean(accs)))
            print(f"{name} rep {r}: {vals[-1]:.4f} ({time.time()-t0:.0f}s)", flush=True)
        out["models"][name] = {"mean": float(np.mean(vals)),
                               "sd": float(np.std(vals, ddof=1)),
                               "reps": vals,
                               "params": sum(p.numel() for p in make_model(name).parameters())}
        print(f"{name}: {out['models'][name]['mean']:.4f} ± {out['models'][name]['sd']:.4f}",
              flush=True)
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
