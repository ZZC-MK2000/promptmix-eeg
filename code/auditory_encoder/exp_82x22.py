"""82-channel x 22-band input diagnostic (NMED-T trial-wise).

Input: gate_nmedt10 bandpower, 82 channels x 22 two-Hz bands (1-45 Hz), 1804-d.
Protocol: per subject x song 50/50; centroids from train half; test half.
Compare:
  - centroid on 1804-d (the current strong baseline)
  - logistic regression on 1804-d
  - channel-attention encoder on [82,22] (82 channel tokens, 22-d band vector)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT.parent / "mainv2" / "data" / "gate_nmedt10.npz"
OUT = ROOT / "results" / "exp_82x22"
OUT.mkdir(parents=True, exist_ok=True)


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def splits(song, subj, reps=5):
    """Per-subject splits: a centroid/classifier is fit for each subject separately."""
    for r in range(reps):
        rng = np.random.default_rng(r)
        out = []
        for s in np.unique(subj):
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
                continue
            out.append((np.concatenate(tr), np.concatenate(te), np.array(ytr), np.array(yte)))
        yield out


def centroid_acc(X, song, subj, reps=5):
    accs = []
    for subjects in splits(song, subj, reps):
        pred, true = [], []
        for tr, te, ytr, yte in subjects:
            C = np.stack([X[tr][ytr == c].mean(0) for c in range(10)])
            pred += (unit(X[te]) @ unit(C).T).argmax(1).tolist()
            true += yte.tolist()
        accs.append(float((np.array(pred) == np.array(true)).mean()))
    return float(np.mean(accs)), float(np.std(accs))


class BandSpatialEncoder(nn.Module):
    def __init__(self, n_ch=82, n_bands=22, d=64, heads=4, depth=2, n_classes=10, drop=0.2):
        super().__init__()
        self.proj = nn.Linear(n_bands, d)
        self.ch_emb = nn.Parameter(torch.zeros(1, n_ch, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.ch_emb, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, "gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, n_classes)

    def forward(self, x):
        tok = self.proj(x) + self.ch_emb
        tok = torch.cat([self.cls.expand(tok.shape[0], -1, -1), tok], 1)
        h = self.blocks(tok)
        return self.head(self.norm(h[:, 0]))


def neural_acc(X, song, subj, reps=5, epochs=40, bs=64, seed=0):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accs = []
    for r, subjects in enumerate(splits(song, subj, reps)):
        pred, true = [], []
        for si, (tr, te, ytr, yte) in enumerate(subjects):
            sc = StandardScaler().fit(X[tr])
            Xtr = sc.transform(X[tr]).reshape(-1, 82, 22).astype(np.float32)
            Xte = sc.transform(X[te]).reshape(-1, 82, 22).astype(np.float32)
            torch.manual_seed(seed + 100 * r + si)
            model = BandSpatialEncoder().to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
            Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, device=device)
            Xte_t = torch.tensor(Xte, device=device); yte_t = torch.tensor(yte, device=device)
            for ep in range(epochs):
                model.train()
                perm = torch.randperm(len(Xtr_t), device=device)
                for a in range(0, len(perm), bs):
                    idx = perm[a:a + bs]
                    loss = F.cross_entropy(model(Xtr_t[idx]), ytr_t[idx])
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pred += model(Xte_t).argmax(1).tolist()
            true += yte.tolist()
        accs.append(float((np.array(pred) == np.array(true)).mean()))
        print(f"  neural split {r}: {accs[-1]:.3f}", flush=True)
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    args = ap.parse_args()
    z = np.load(GATE, allow_pickle=True)
    X = z["bandpower"].astype(np.float32)
    song = z["song"].astype(int); subj = z["subject"].astype(str)
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    res = {"protocol": {"input": "82 channels x 22 two-Hz bands (1804-d)",
                        "split": "per-subject per-song 50/50 trial-wise",
                        "chance": 0.10, "reps": args.reps}, "results": {}}
    print("centroid", flush=True)
    res["results"]["centroid_1804"] = dict(zip(("acc", "sd"),
                                               centroid_acc(X, song, subj, args.reps)))
    print("logistic", flush=True)
    log_acc = []
    for subjects in splits(song, subj, args.reps):
        pred, true = [], []
        for tr, te, ytr, yte in subjects:
            clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, C=1.0))
            clf.fit(X[tr], ytr)
            pred += clf.predict(X[te]).tolist(); true += yte.tolist()
        log_acc.append(float((np.array(pred) == np.array(true)).mean()))
    res["results"]["logistic_1804"] = {"acc": float(np.mean(log_acc)),
                                       "sd": float(np.std(log_acc))}
    print("channel-attention", flush=True)
    res["results"]["channel_attention_82x22"] = dict(zip(("acc", "sd"),
                                                         neural_acc(X, song, subj, args.reps,
                                                                    args.epochs)))
    print(json.dumps(res, indent=2))
    (OUT / "results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", OUT / "results.json")


if __name__ == "__main__":
    main()
