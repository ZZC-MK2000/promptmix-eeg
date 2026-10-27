"""From-scratch supervised reference runner for AEB.

Input npz: X [N,19,1600] float, song [N], subject [N].
Models: eegnet, shallow, deep4, conformer (braindecode). Training is per subject
on the split's train rows; the split's test rows are never used for gradients.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from braindecode.models import Deep4Net, EEGConformer, EEGNet, ShallowFBCSPNet


def model(name):
    kw = dict(n_chans=19, n_times=1600, sfreq=200, input_window_seconds=8, n_outputs=10)
    return {"eegnet": EEGNet, "shallow": ShallowFBCSPNet,
            "deep4": Deep4Net, "conformer": EEGConformer}[name](**kw)


def norm(x):
    x = x.float()
    x = x - x.mean(-1, keepdim=True)
    return torch.clamp(x / (x.std(-1, keepdim=True) + 1e-6), -10, 10)


def fit(X, song, subj, tr, te, name, device, epochs, batch, seed):
    torch.manual_seed(seed)
    m = model(name).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss()
    accs = []
    for s in sorted(set(subj[tr].tolist())):
        a = tr[subj[tr] == s]; b = te[subj[te] == s]
        if len(a) == 0 or len(b) == 0 or len(set(song[a].tolist())) < 10:
            continue
        m.train()
        for ep in range(epochs):
            p = torch.randperm(len(a))
            for i in range(0, len(a), batch):
                idx = p[i:i + batch]
                xb = norm(torch.from_numpy(X[a[idx]]).to(device))
                yb = torch.from_numpy(song[a[idx]]).long().to(device)
                loss = ce(m(xb), yb)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        m.eval(); pred = []
        with torch.no_grad():
            for i in range(0, len(b), batch):
                pred.append(m(norm(torch.from_numpy(X[b[i:i + batch]]).to(device))).argmax(1).cpu())
        accs.append(float((torch.cat(pred).numpy() == song[b]).mean()))
    return accs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="splits/nmedt_folds.json")
    ap.add_argument("--model", default="eegnet")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    d = np.load(args.data, allow_pickle=True)
    X = d["X"]; song = d["song"].astype(int); subj = d["subject"].astype(str)
    split = json.load(open(args.split))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    res = {"model": args.model, "splits": []}
    for name, seeds in split["splits"].items():
        for sp in seeds:
            accs = fit(X, song, subj, np.asarray(sp["train"]), np.asarray(sp["test"]),
                       args.model, device, args.epochs, args.batch, sp["seed"])
            res["splits"].append({"split": name, "seed": sp["seed"],
                                  "mean": float(np.mean(accs)), "per_subject": accs})
            print(name, sp["seed"], np.mean(accs), flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
