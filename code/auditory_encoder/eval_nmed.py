"""NMED-T 10-way zero-shot / fiber-oracle for the new encoder vs bandpower."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
sys.path.insert(0, str(ROOT / "code" / "auditory_encoder"))
sys.path.insert(0, str(ROOT / "code" / "information"))
from model import AuditoryEncoder, procrustes  # noqa: E402
from eval_finetune_local75 import nmed_windows  # noqa: E402

M2 = ROOT.parent / "mainv2"


def zscore(x):
    x = np.asarray(x, dtype=np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    return np.clip((x - mu) / sd, -10, 10).astype(np.float32)


@torch.no_grad()
def encode(model, X, bs=64):
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i + bs], device=dev)
        z, _, _ = model(x)
        out.append(z.cpu())
    return torch.cat(out, 0)


def eval_rep(Z, song, subject, folds=5):
    Z = F.normalize(Z, dim=-1)
    subs = sorted(set(subject))
    rng = np.random.default_rng(0); rng.shuffle(subs)
    zero, oracle = [], []
    for f in np.array_split(subs, folds):
        te = np.array([s in set(f) for s in subject]); tr = ~te
        C = torch.stack([Z[tr & (song == c)].mean(0) for c in range(10)])
        C = F.normalize(C, dim=-1)
        pred = (Z[te] @ C.T).argmax(1)
        zero.append(float((pred == torch.tensor(song[te])).float().mean()))
        for s in f:
            rows = np.where(te & (subject == s))[0]
            a, q = [], []
            for c in range(10):
                r = rows[song[rows] == c]
                if len(r) < 2:
                    a = None; break
                a.append(r[:len(r) // 2]); q.append(r[len(r) // 2:])
            if a is None:
                continue
            aa = torch.stack([Z[r].mean(0) for r in a])
            qq = torch.cat([Z[r] for r in q])
            yy = torch.tensor([c for c in range(10) for _ in q[c]])
            R = procrustes(aa, C)
            pr = (F.normalize((qq - aa.mean(0)) @ R, dim=-1) @ C.T).argmax(1)
            oracle.append(float((pr == yy).float().mean()))
    return {"zero": float(np.mean(zero)), "oracle": float(np.mean(oracle))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "results/auditory_encoder/pilot_3k/encoder_3000.pt"))
    ap.add_argument("--out", default=str(ROOT / "results/auditory_encoder/nmed_eval.json"))
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    m = AuditoryEncoder(d=a.get("d", 128), depth=a.get("depth", 4), heads=a.get("heads", 4),
                        spectral=a.get("spectral", False), spectral_only=a.get("spectral_only", False))
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("subj_head")}
    m.load_state_dict(sd, strict=False); m.eval()
    X, song, subject = nmed_windows()
    X = np.stack([zscore(x) for x in X])
    Z = encode(m, X)
    out = {"encoder": eval_rep(Z, song, subject)}
    z = np.load(M2 / "data" / "gate_nmedt10.npz", allow_pickle=True)
    bp = z["bandpower"].astype(np.float32)
    bp = (bp - bp.mean(0)) / (bp.std(0) + 1e-6)
    out["bandpower"] = eval_rep(torch.tensor(bp), song, subject)
    print(json.dumps(out, indent=2))
    Path(args.out).write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
