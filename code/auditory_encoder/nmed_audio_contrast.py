"""NMED EEG-audio contrastive adaptation of a pretrained auditory encoder.

The external-corpus encoder is frozen as the EEG branch; a 128->512 projection
head is trained with InfoNCE to the ALDM audio target on NMED-H, then evaluated
with the mainline fixed ALDM->laion bridge (song hit / held-subject cosine).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = ROOT.parent / "mainv2"
sys.path.insert(0, str(M2 / "src" / "measurement"))
sys.path.insert(0, str(ROOT / "code" / "information"))
from gate_nmedt_labram import Head, info_nce, l2  # noqa: E402
from train_finetuned_aldm import eval_mainline  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ZA = M2 / "data" / "gate_nmedh14_ov2_a.npz"
ZL = M2 / "data" / "gate_nmedh14_ov2_lfix.npz"
DATA = ROOT / "data" / "eegcond"


def splits(d, seed=0):
    blocks = np.asarray(d["block"])
    ub = np.unique(blocks)
    rng = np.random.default_rng(seed)
    pm = rng.permutation(len(ub))
    n_tr, n_va = int(0.70 * len(ub)), int(0.10 * len(ub))
    b_tr, b_va, b_te = ub[pm[:n_tr]], ub[pm[n_tr:n_tr + n_va]], ub[pm[n_tr + n_va:]]
    al = np.asarray(d["aligned8"]).astype(bool)
    tr = np.where(np.isin(blocks, b_tr))[0]
    va = np.where(np.isin(blocks, b_va) & al)[0]
    te = np.where(np.isin(blocks, b_te) & al)[0]
    return tr, va, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "auditory_encoder"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)

    E = np.load(args.emb).astype(np.float32)
    za = np.load(ZA, allow_pickle=True)
    zl = np.load(ZL, allow_pickle=True)
    T = za["target"].astype(np.float32)
    proto = l2(torch.tensor(za["proto"].astype(np.float32), device=DEV))
    song = za["song"].astype(int)
    tr, va, te = splits(za, args.seed)
    print(f"E {E.shape} train {len(tr)} val {len(va)} test {len(te)}", flush=True)

    Et = torch.tensor(E, device=DEV); Tt = torch.tensor(T, device=DEV)
    head = Head(E.shape[1]).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
    best = {"acc": -1, "state": None, "epoch": 0}
    for ep in range(args.epochs):
        head.train()
        p = torch.randperm(len(tr), device=DEV)
        for i in range(0, len(tr), 256):
            j = torch.tensor(tr, device=DEV)[p[i:i + 256]]
            loss = info_nce(l2(head(Et[j])), Tt[j])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        head.eval()
        with torch.no_grad():
            z = l2(head(Et[torch.tensor(va, device=DEV)]))
            pred = (z @ proto.T).argmax(1).cpu().numpy()
        acc = float((pred == song[va]).mean())
        if acc > best["acc"]:
            best = {"acc": acc, "state": {k: v.detach().clone() for k, v in head.state_dict().items()},
                    "epoch": ep + 1, "loss": float(loss)}
        if (ep + 1) % 5 == 0:
            print(f"ep {ep+1:2d} val14 {acc:.4f} best {best['acc']:.4f}", flush=True)
    head.load_state_dict(best["state"])
    with torch.no_grad():
        Z = l2(head(Et)).cpu().numpy()
        z = l2(head(Et[torch.tensor(te, device=DEV)]))
        pred = (z @ proto.T).argmax(1).cpu().numpy()
    te14 = float((pred == song[te]).mean())
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    group = np.load(DATA / "group.npy").astype(np.int64)
    res = {"emb": args.emb, "best_epoch": best["epoch"], "val14": best["acc"],
           "test14": te14, "mainline": eval_mainline(Z, zl, cond, group)}
    tag = Path(args.emb).stem
    np.save(out / f"{tag}_head.npy", Z)
    (out / f"{tag}_audio_contrast.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
