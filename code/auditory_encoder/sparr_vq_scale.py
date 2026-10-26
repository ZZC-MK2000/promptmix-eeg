"""Small-scale VQ-codebook pretraining vs continuous spectral reconstruction.

LaBraM-style hypothesis: predict a discrete code for each masked raw 1-s patch
instead of reconstructing the continuous signal/spectrum. Here a VQ tokenizer is
trained on raw 64-channel x 1-s patches; its codes are frozen targets for the
spatial-spectral encoder. Same data/split/steps as sparr_env_scale.py.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from sparr_env_scale import EnvWindows  # noqa: E402
from sparr_scale import WIN, build_model, list_files  # noqa: E402
from spatial_spectral import vicreg_terms  # noqa: E402

N_PATCH = WIN // 120


class VQTokenizer(nn.Module):
    def __init__(self, patch_dim=64 * 120, h=128, d=64, k=128):
        super().__init__()
        h = max(32, h)
        d = max(16, d)
        self.enc = nn.Sequential(nn.Linear(patch_dim, h), nn.GELU(),
                                 nn.Linear(h, d))
        self.code = nn.Parameter(torch.randn(k, d) * 0.02)
        self.dec = nn.Sequential(nn.Linear(d, h), nn.GELU(),
                                 nn.Linear(h, patch_dim))
        self.k = k

    def forward(self, x):
        z = self.enc(x)
        dist = (z.pow(2).sum(1, keepdim=True)
                - 2 * z @ self.code.T + self.code.pow(2).sum(1)[None])
        idx = dist.argmin(1)
        zq = self.code[idx]
        rec = self.dec(zq + (z - zq).detach())
        return rec, z, zq, idx


def patches_of(x, per_channel=False):
    # x: [N,64,WIN] -> [N*8, 64*120] or [N*8*64, 120]
    n, c, t = x.shape
    p = x.reshape(n, c, N_PATCH, t // N_PATCH).permute(0, 2, 1, 3)
    if per_channel:
        return p.reshape(n * N_PATCH * c, t // N_PATCH)
    return p.reshape(n * N_PATCH, -1)


class ChannelCodeHead(nn.Module):
    def __init__(self, d, dc, n_ch, k):
        super().__init__()
        self.proj = nn.Linear(d, dc)
        self.ch = nn.Parameter(torch.zeros(n_ch, dc))
        self.code = nn.Parameter(torch.randn(k, dc) * 0.02)

    def forward(self, tokens):
        # tokens [B,N,d] -> [B,N,C,K]
        z = self.proj(tokens)
        return (z[:, :, None, :] + self.ch[None, None]) @ self.code.T


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/datasets/sparrkulee/split_data")
    ap.add_argument("--out", default=str(ROOT / "results" / "sparr_vq" / "pilot"))
    ap.add_argument("--subjects", type=int, default=10)
    ap.add_argument("--test-subjects", type=int, default=25)
    ap.add_argument("--test-offset", type=int, default=60)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--tok-steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--tok-batch", type=int, default=128)
    ap.add_argument("--per-file", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tok-lr", type=float, default=1e-3)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-filters", type=int, default=32)
    ap.add_argument("--codebook", type=int, default=128)
    ap.add_argument("--per-channel", action="store_true")
    ap.add_argument("--kmeans", action="store_true",
                    help="use a fixed k-means codebook instead of a learned VQ tokenizer")
    ap.add_argument("--spectral-w", type=float, default=0.0)
    ap.add_argument("--train-splits", default="train")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    splits = set(args.train_splits.split(","))
    recs = list_files(args.root)
    subs = sorted({r["subject"] for r in recs})
    test_subs = set(subs[args.test_offset:args.test_offset + args.test_subjects])
    pool = [s for s in subs if s not in test_subs][:args.subjects]
    train_recs = [r for r in recs if r["subject"] in set(pool) and r["split"] in splits]
    print(f"train files {len(train_recs)} subjects {len(pool)} test {len(test_subs)}",
          flush=True)

    ds = EnvWindows(train_recs, per_file=args.per_file, seed=args.seed)
    X = torch.stack([it[0] for it in ds.items])  # [N,64,WIN]
    P = patches_of(X, per_channel=args.per_channel)
    print("windows", len(ds), "patches", tuple(P.shape), flush=True)

    # 1) build the codebook (learned VQ tokenizer or fixed k-means)
    tok = None
    if args.kmeans:
        from sklearn.cluster import MiniBatchKMeans
        km = MiniBatchKMeans(n_clusters=args.codebook, batch_size=1024,
                             n_init=3, random_state=args.seed)
        lab = km.fit_predict(P.numpy())
        if args.per_channel:
            codes = torch.tensor(lab, dtype=torch.long).reshape(len(ds), N_PATCH, -1)
        else:
            codes = torch.tensor(lab, dtype=torch.long).reshape(len(ds), N_PATCH)
        uniq = int(len(set(lab.tolist())))
        print("kmeans codebook used", uniq, "/", args.codebook, flush=True)
    else:
        if args.per_channel:
            tok = VQTokenizer(patch_dim=WIN // N_PATCH, h=64, d=32, k=args.codebook).to(device)
        else:
            tok = VQTokenizer(patch_dim=64 * (WIN // N_PATCH), h=128, d=64,
                              k=args.codebook).to(device)
        opt = torch.optim.AdamW(tok.parameters(), lr=args.tok_lr, weight_decay=1e-2)
        t0 = time.time()
        for step in range(1, args.tok_steps + 1):
            idx = torch.randint(0, len(P), (args.tok_batch,))
            xb = P[idx].to(device)
            rec, z, zq, idx = tok(xb)
            rec_loss = F.mse_loss(rec, xb)
            cb_loss = F.mse_loss(zq, z.detach())
            commit = F.mse_loss(z, zq.detach())
            usage = torch.bincount(idx, minlength=args.codebook).float()
            usage = usage / usage.sum()
            div = (usage * (usage + 1e-8).log()).sum()  # negative entropy
            loss = rec_loss + cb_loss + 0.25 * commit + 0.01 * div
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            if step % 200 == 0 or step == 1:
                print(json.dumps({"tok_step": step, "rec": float(rec_loss),
                                  "cb": float(cb_loss), "commit": float(commit),
                                  "used": int((usage > 0).sum()),
                                  "sec": round(time.time() - t0, 1)}), flush=True)

        with torch.no_grad():
            codes = []
            for i in range(0, len(P), 512):
                _, _, _, idx = tok(P[i:i + 512].to(device))
                codes.append(idx.cpu())
            if args.per_channel:
                codes = torch.cat(codes).reshape(len(ds), N_PATCH, -1)
            else:
                codes = torch.cat(codes).reshape(len(ds), N_PATCH)
            uniq = int(codes.unique().numel())
    print("codebook used", uniq, "/", args.codebook, flush=True)

    # 2) train encoder to predict masked code indices
    model = build_model(args).to(device)
    if args.per_channel:
        head = ChannelCodeHead(args.d, 64, codes.shape[-1], args.codebook).to(device)
        y = codes.reshape(len(ds), N_PATCH, -1)
    else:
        head = nn.Sequential(nn.LayerNorm(args.d), nn.Linear(args.d, 2 * args.d),
                             nn.GELU(), nn.Linear(2 * args.d, args.codebook)).to(device)
        y = codes
    params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()),
                            lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps, pct_start=0.1)
    dl = DataLoader(TensorDataset(X, codes), batch_size=args.batch, shuffle=True,
                    num_workers=0, drop_last=True)
    it = iter(dl); t0 = time.time(); logs = []
    for step in range(1, args.steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl); x, y = next(it)
        x, y = x.to(device), y.to(device)
        mask = torch.rand(x.shape[0], N_PATCH, device=device) < 0.25
        none = ~mask.any(1)
        if none.any():
            mask[none, 0] = True
        model.train(); head.train()
        content, recon, feats, tokens = model(x, mask, return_tokens=True)
        logits = head(tokens)
        ce = F.cross_entropy(logits[mask].reshape(-1, args.codebook),
                             y[mask].reshape(-1))
        loss = ce + 0.05 * vicreg_terms(content)
        if args.spectral_w:
            loss = loss + args.spectral_w * F.mse_loss(recon[mask], feats[mask])
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                acc = float((logits[mask].reshape(-1, args.codebook).argmax(1)
                             == y[mask].reshape(-1)).float().mean())
            rec = {"step": step, "loss": float(loss), "ce": float(ce),
                   "code_acc": acc, "sec": round(time.time() - t0, 1)}
            logs.append(rec); print(json.dumps(rec), flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args),
                "arch": "v2_sparr", "params": params},
               out / f"encoder_s{args.steps}.pt")
    if tok is not None:
        torch.save({"tok": tok.state_dict(), "args": vars(args)},
                   out / "tokenizer.pt")
    np.save(out / "codes.npy", codes.numpy())
    (out / "train_log.json").write_text(json.dumps({
        "args": vars(args), "codebook_used": uniq, "n_patches": int(len(P)),
        "logs": logs}, indent=2), encoding="utf-8")
    print("params", params, "codebook_used", uniq,
          "wrote", out / f"encoder_s{args.steps}.pt", flush=True)
