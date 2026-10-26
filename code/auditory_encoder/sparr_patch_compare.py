"""Same raw-patch token architecture, two pretraining targets: VQ code vs raw recon."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from patch_encoder import PatchEncoder, patch_targets  # noqa: E402
from sparr_env_scale import EnvWindows  # noqa: E402
from sparr_scale import WIN, list_files  # noqa: E402
from spatial_spectral import vicreg_terms  # noqa: E402

N_PATCH = WIN // 120


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/datasets/sparrkulee/split_data")
    ap.add_argument("--out", default=str(ROOT / "results" / "sparr_patch_compare" / "pilot"))
    ap.add_argument("--subjects", type=int, default=10)
    ap.add_argument("--test-subjects", type=int, default=25)
    ap.add_argument("--test-offset", type=int, default=60)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--per-file", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--codebook", type=int, default=256)
    ap.add_argument("--target", choices=("code", "recon"), required=True)
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
    ds = EnvWindows(train_recs, per_file=args.per_file, seed=args.seed)
    X = torch.stack([it[0] for it in ds.items])
    print(f"train files {len(train_recs)} windows {len(ds)} target {args.target}", flush=True)

    Y = patch_targets(X)  # [N, 8*64, 120]
    if args.target == "code":
        P = Y.reshape(-1, Y.shape[-1]).numpy()
        km = MiniBatchKMeans(n_clusters=args.codebook, batch_size=1024,
                             n_init=3, random_state=args.seed)
        lab = km.fit_predict(P)
        codes = torch.tensor(lab, dtype=torch.long).reshape(len(ds), -1)
        print("codebook used", len(set(lab.tolist())), "/", args.codebook, flush=True)
    else:
        codes = None

    model = PatchEncoder(n_ch=X.shape[1], patch=WIN // N_PATCH, n_time=N_PATCH,
                         d=args.d, depth=args.depth, heads=args.heads,
                         codebook=args.codebook).to(device)
    params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr,
                                                total_steps=args.steps, pct_start=0.1)
    data = TensorDataset(X, Y if codes is None else codes)
    dl = DataLoader(data, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    it = iter(dl); t0 = time.time(); logs = []
    for step in range(1, args.steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl); x, y = next(it)
        x, y = x.to(device), y.to(device)
        mask = torch.rand(x.shape[0], N_PATCH * x.shape[1], device=device) < 0.25
        none = ~mask.any(1)
        if none.any():
            mask[none, 0] = True
        model.train()
        content, recon, logits = model(x, mask)
        if args.target == "code":
            loss_t = F.cross_entropy(logits[mask], y[mask])
        else:
            loss_t = F.mse_loss(recon[mask], y[mask])
        loss = loss_t + 0.05 * vicreg_terms(content)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 200 == 0 or step == 1:
            with torch.no_grad():
                if args.target == "code":
                    acc = float((logits[mask].argmax(1) == y[mask]).float().mean())
                else:
                    acc = float(((recon[mask] - y[mask]) ** 2).mean())
            rec = {"step": step, "loss": float(loss), "target": float(loss_t),
                   "metric": acc, "sec": round(time.time() - t0, 1)}
            logs.append(rec); print(json.dumps(rec), flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args),
                "arch": "patch", "params": params},
               out / f"encoder_s{args.steps}.pt")
    (out / "train_log.json").write_text(json.dumps({
        "args": vars(args), "params": params, "logs": logs}, indent=2), encoding="utf-8")
    print("params", params, "wrote", out / f"encoder_s{args.steps}.pt", flush=True)


if __name__ == "__main__":
    main()
