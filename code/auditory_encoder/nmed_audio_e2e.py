"""End-to-end NMED EEG-audio contrastive adaptation of the auditory encoder."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = ROOT.parent / "mainv2"
sys.path.insert(0, str(ROOT / "code" / "auditory_encoder"))
sys.path.insert(0, str(M2 / "src" / "measurement"))
sys.path.insert(0, str(ROOT / "code" / "information"))
from model import AuditoryEncoder  # noqa: E402
from gate_nmedt_labram import Head, info_nce, l2  # noqa: E402
from train_finetuned_aldm import eval_mainline  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
ZA = M2 / "data" / "gate_nmedh14_ov2_a.npz"
ZL = M2 / "data" / "gate_nmedh14_ov2_lfix.npz"
DATA = ROOT / "data" / "eegcond"


def splits(d, seed=0):
    blocks = np.asarray(d["block"]); ub = np.unique(blocks)
    rng = np.random.default_rng(seed); pm = rng.permutation(len(ub))
    n_tr, n_va = int(0.70 * len(ub)), int(0.10 * len(ub))
    b_tr, b_va, b_te = ub[pm[:n_tr]], ub[pm[n_tr:n_tr + n_va]], ub[pm[n_tr + n_va:]]
    al = np.asarray(d["aligned8"]).astype(bool)
    return (np.where(np.isin(blocks, b_tr))[0],
            np.where(np.isin(blocks, b_va) & al)[0],
            np.where(np.isin(blocks, b_te) & al)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--x", default=str(ROOT / "results/auditory_encoder/nmedh19.npy"))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=str(ROOT / "results/auditory_encoder"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--enc-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    X = np.load(args.x, mmap_mode="r")
    za = np.load(ZA, allow_pickle=True); zl = np.load(ZL, allow_pickle=True)
    T = torch.tensor(za["target"].astype(np.float32), device=DEV)
    proto = l2(torch.tensor(za["proto"].astype(np.float32), device=DEV))
    song = za["song"].astype(int)
    tr, va, te = splits(za, args.seed)
    print(f"X {X.shape} train {len(tr)} val {len(va)}", flush=True)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    enc = AuditoryEncoder(d=a.get("d", 128), depth=a.get("depth", 4), heads=a.get("heads", 4),
                          spectral=a.get("spectral", False), spectral_only=a.get("spectral_only", False)).to(DEV)
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("subj_head")}
    enc.load_state_dict(sd, strict=False)
    head = Head(a.get("d", 128)).to(DEV)
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": args.enc_lr},
                             {"params": head.parameters(), "lr": args.head_lr}], weight_decay=1e-2)
    best = {"acc": -1, "enc": None, "head": None, "epoch": 0}
    for ep in range(args.epochs):
        enc.train(); head.train()
        p = np.random.default_rng(args.seed + ep).permutation(len(tr))
        for i in range(0, len(tr), args.bs):
            j = tr[p[i:i + args.bs]]
            x = torch.tensor(np.asarray(X[j], dtype=np.float32), device=DEV)
            z, _, _ = enc(x)
            loss = info_nce(l2(head(z)), T[torch.tensor(j, device=DEV)])
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(head.parameters()), 1.0)
            opt.step()
        enc.eval(); head.eval()
        with torch.no_grad():
            zv = []
            for i in range(0, len(va), 128):
                x = torch.tensor(np.asarray(X[va[i:i + 128]], dtype=np.float32), device=DEV)
                zv.append(l2(head(enc(x)[0])))
            pred = (torch.cat(zv) @ proto.T).argmax(1).cpu().numpy()
        acc = float((pred == song[va]).mean())
        print(f"ep {ep+1} val14 {acc:.4f}", flush=True)
        if acc > best["acc"]:
            best = {"acc": acc, "epoch": ep + 1,
                    "enc": {k: v.detach().clone() for k, v in enc.state_dict().items()},
                    "head": {k: v.detach().clone() for k, v in head.state_dict().items()}}
    enc.load_state_dict(best["enc"]); head.load_state_dict(best["head"])
    enc.eval(); head.eval()
    with torch.no_grad():
        Z = []
        for i in range(0, len(X), 128):
            x = torch.tensor(np.asarray(X[i:i + 128], dtype=np.float32), device=DEV)
            Z.append(l2(head(enc(x)[0])))
        Z = torch.cat(Z).cpu().numpy()
        zq = Z[te]
        pred = (torch.tensor(zq, device=DEV) @ proto.T).argmax(1).cpu().numpy()
    te14 = float((pred == song[te]).mean())
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    group = np.load(DATA / "group.npy").astype(np.int64)
    res = {"ckpt": args.ckpt, "best_epoch": best["epoch"], "val14": best["acc"],
           "test14": te14, "mainline": eval_mainline(Z, zl, cond, group)}
    tag = Path(args.ckpt).parent.name
    np.save(out / f"{tag}_e2e_head.npy", Z)
    torch.save({"enc": best["enc"], "head": best["head"], "args": vars(args)},
               out / f"{tag}_e2e.pt")
    (out / f"{tag}_e2e.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2), flush=True)


if __name__ == "__main__":
    main()
