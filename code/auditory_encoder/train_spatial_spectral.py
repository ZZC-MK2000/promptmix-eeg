"""Self-supervised training for the spatial-spectral encoder.

Objective: masked spectral modeling on 1-s tokens + a small VICReg term on the content
embedding. No labels are used. The corpus windows are [N,19,1600] @200 Hz.

Run:
  python train_spatial_spectral.py --steps 200 --batch 64 --out results/spatial_spectral
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from spatial_spectral import SpatialSpectralEncoder, masked_spectral_loss, vicreg_terms

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = ROOT.parent / "mainv2" / "data" / "auditory_corpus"


class WindowDataset(Dataset):
    def __init__(self, root, indices=None):
        self.windows = np.load(root / "windows.npy", mmap_mode="r")
        self.indices = np.arange(len(self.windows)) if indices is None else np.asarray(indices)
        self.n_ch = self.windows.shape[1]
        self.n_t = self.windows.shape[2]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x = np.asarray(self.windows[int(self.indices[i])], dtype=np.float32)
        mu = x.mean(-1, keepdims=True)
        sd = x.std(-1, keepdims=True) + 1e-6
        x = np.clip((x - mu) / sd, -10.0, 10.0)
        return torch.from_numpy(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--out", default=str(ROOT / "results" / "spatial_spectral"))
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--mask-ratio", type=float, default=0.25)
    ap.add_argument("--vicreg-w", type=float, default=0.1)
    ap.add_argument("--holdout", default="", help="comma-separated subject ids to exclude")
    ap.add_argument("--smoke", action="store_true", help="use random tensors only")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.smoke:
        ds = None
        n_ch, n_t = 19, 1600
    else:
        root = Path(args.root)
        meta = np.load(root / "meta.npz", allow_pickle=True)
        subj = np.array([str(x) for x in meta["subject"]])
        hold = set(s for s in args.holdout.split(",") if s)
        idx = np.where(~np.isin(subj, list(hold)))[0] if hold else None
        ds = WindowDataset(root, idx)
        n_ch, n_t = ds.n_ch, ds.n_t
        print(f"windows {len(ds)} (holdout {len(hold)} subjects) n_ch={n_ch} n_t={n_t}", flush=True)

    model = SpatialSpectralEncoder(n_ch=n_ch, d=128, depth=4, heads=4,
                                   n_filters=8, patch=200, sfreq=200.0).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params {n_params/1e6:.2f}M device={device}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.1)
    loader = None if ds is None else DataLoader(ds, batch_size=args.batch, shuffle=True,
                                                num_workers=0, drop_last=True)
    it = iter(loader) if loader is not None else None
    t0 = time.time()
    logs = []
    for step in range(1, args.steps + 1):
        if it is None:
            x = torch.randn(args.batch, n_ch, n_t)
        else:
            try:
                x = next(it)
            except StopIteration:
                it = iter(loader); x = next(it)
        x = x.to(device)
        mask = torch.rand(x.shape[0], n_t // 200, device=device) < args.mask_ratio
        # guarantee at least one masked token per sample
        none = ~mask.any(1)
        if none.any():
            mask[none, 0] = True
        model.train()
        content, recon, feats = model(x, mask)
        loss_rec = masked_spectral_loss(recon, feats, mask)
        loss_var = vicreg_terms(content)
        loss = loss_rec + args.vicreg_w * loss_var
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 10 == 0 or step == 1:
            rec = {"step": step, "loss": float(loss), "rec": float(loss_rec),
                   "var": float(loss_var), "sec": round(time.time() - t0, 1)}
            logs.append(rec); print(json.dumps(rec), flush=True)
    ckpt = {"model": model.state_dict(), "args": vars(args), "params": n_params}
    path = out / f"encoder_{args.steps}.pt"
    torch.save(ckpt, path)
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "logs": logs}, indent=2),
                                        encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
