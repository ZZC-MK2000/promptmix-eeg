"""SparrKULee scaling: subject-held-out envelope prediction from encoder content.

Uses the preprocessed test split (same 85 subjects; subject-disjoint evaluation).
For each 8-s window, the target is the speech envelope downsampled to 16 bins.
Encoder is trained unsupervised (masked spectral + VICReg); a ridge probe maps
content -> envelope on training subjects and is tested on held-out subjects.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from sparr_scale import WIN, list_files, build_model, masked_spectral_loss, vicreg_terms  # noqa: E402
from torch.utils.data import DataLoader, Dataset

ENV_BINS = 16


def env_target(env_path, start):
    a = np.load(env_path, mmap_mode="r")  # [T,1]
    seg = np.asarray(a[start:start + WIN, 0], dtype=np.float32)
    if len(seg) < WIN:
        seg = np.pad(seg, (0, WIN - len(seg)))
    seg = seg.reshape(ENV_BINS, -1).mean(1)
    return seg


@torch.no_grad()
def extract_env(model, records, per_file=2, seed=0, bs=64):
    device = next(model.parameters()).device
    rows = []
    for i, r in enumerate(records):
        p = Path(r["path"])
        envp = Path(str(p).replace("_eeg.npy", "_envelope.npy"))
        if not envp.exists():
            continue
        a = np.load(p, mmap_mode="r")
        total = a.shape[0]
        starts = np.linspace(0, max(0, total - WIN), per_file + 1).astype(int)[:per_file]
        for j, start in enumerate(starts):
            x = np.asarray(a[start:start + WIN, :], dtype=np.float32).T
            if x.shape[1] < WIN:
                continue
            x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
            x = np.clip(x, -10, 10)
            rows.append((x, env_target(envp, start), r["subject"], i))
    Z, Y = [], []
    for st in range(0, len(rows), bs):
        xb = torch.tensor(np.stack([r[0] for r in rows[st:st + bs]]), dtype=torch.float32, device=device)
        z, _, _ = model(xb)
        Z.append(z.cpu().numpy())
        Y.append(np.stack([r[1] for r in rows[st:st + bs]]))
    return np.concatenate(Z), np.concatenate(Y), [r[2] for r in rows]


class EnvWindows(Dataset):
    def __init__(self, records, per_file=4, seed=0):
        self.items = []
        for idx, r in enumerate(records):
            p = Path(r["path"])
            envp = Path(str(p).replace("_eeg.npy", "_envelope.npy"))
            a = np.load(p)
            total = a.shape[0]
            for j in range(per_file):
                i = idx * per_file + j
                start = int(np.random.default_rng(seed + i).integers(0, max(1, total - WIN + 1)))
                x = np.asarray(a[start:start + WIN, :], dtype=np.float32).T
                if x.shape[1] < WIN:
                    x = np.pad(x, ((0, 0), (0, WIN - x.shape[1])))
                x = (x - x.mean(1, keepdims=True)) / (x.std(1, keepdims=True) + 1e-6)
                if envp.exists():
                    y = env_target(envp, start)
                    y = (y - y.mean()) / (y.std() + 1e-6)
                else:
                    y = np.zeros(ENV_BINS, dtype=np.float32)
                self.items.append((torch.from_numpy(np.clip(x, -10, 10)),
                                   torch.tensor(y, dtype=torch.float32)))
            if getattr(a, "_mmap", None) is not None:
                a._mmap.close()
            del a

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def train_env(args, records):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = EnvWindows(records, per_file=args.per_file, seed=args.seed)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    model = build_model(args).to(device)
    head = None if args.ssl_only else nn.Linear(args.d, ENV_BINS).to(device)
    extra = [] if head is None else list(head.parameters())
    params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in extra)
    opt = torch.optim.AdamW(list(model.parameters()) + extra,
                            lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.1)
    it = iter(dl); t0 = 0.0
    import time
    t0 = time.time(); logs = []
    for step in range(1, args.steps + 1):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(dl); x, y = next(it)
        x, y = x.to(device), y.to(device)
        mask = torch.rand(x.shape[0], WIN // 120, device=device) < 0.25
        none = ~mask.any(1)
        if none.any():
            mask[none, 0] = True
        model.train()
        if head is not None:
            head.train()
        content, recon, feats = model(x, mask)
        l_ssl = masked_spectral_loss(recon, feats, mask) + 0.05 * vicreg_terms(content)
        if head is None:
            loss = l_ssl; l_env = torch.zeros((), device=device)
        else:
            l_env = F.mse_loss(head(content), y)
            loss = l_ssl + 0.5 * l_env
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + extra, 1.0)
        opt.step(); sched.step()
        if step % 200 == 0 or step == 1:
            rec = {"step": step, "loss": float(loss), "ssl": float(l_ssl),
                   "env": float(l_env), "sec": round(time.time() - t0, 1)}
            logs.append(rec); print(json.dumps(rec), flush=True)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ck = {"model": model.state_dict(), "args": vars(args), "arch": "v2_sparr",
          "params": params}
    if head is not None:
        ck["env_head"] = head.state_dict()
    torch.save(ck, out / f"encoder_s{args.steps}.pt")
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "logs": logs}, indent=2),
                                        encoding="utf-8")
    print("params", params, "wrote", out / f"encoder_s{args.steps}.pt", flush=True)
    return model, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/datasets/sparrkulee/split_data")
    ap.add_argument("--out", default=str(ROOT / "results" / "sparr_env_scale" / "pilot"))
    ap.add_argument("--subjects", type=int, default=10)
    ap.add_argument("--test-subjects", type=int, default=8)
    ap.add_argument("--test-offset", type=int, default=0,
                    help="index of the first held-out subject in the sorted list")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--per-file", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--n-filters", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ssl-only", action="store_true",
                    help="train masked-spectral/VICReg only; skip envelope head and probe")
    ap.add_argument("--train-splits", default="train",
                    help="comma-separated splits used for unsupervised training")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    train_splits = set(args.train_splits.split(","))
    recs = list_files(args.root)
    subs = sorted({r["subject"] for r in recs})
    if len(subs) < args.test_subjects + 2:
        raise SystemExit(f"only {len(subs)} subjects downloaded")
    test_subs = subs[args.test_offset:args.test_offset + args.test_subjects]
    pool = [s for s in subs if s not in test_subs][:args.subjects]
    train_recs = [r for r in recs if r["subject"] in set(pool) and r["split"] in train_splits]
    test_recs = [r for r in recs if r["subject"] in set(test_subs) and r["split"] == "test"]
    if not train_recs:
        raise SystemExit(f"no files for train splits {sorted(train_splits)}")
    if not args.ssl_only:
        train_recs = [r for r in train_recs
                      if Path(str(r["path"]).replace("_eeg.npy", "_envelope.npy")).exists()]
        test_recs = [r for r in test_recs
                     if Path(str(r["path"]).replace("_eeg.npy", "_envelope.npy")).exists()]
    print(f"subjects {len(subs)} train {len(pool)} test {len(test_subs)} "
          f"train_files {len(train_recs)} test_files {len(test_recs)}", flush=True)
    model, params = train_env(args, train_recs)
    if args.ssl_only:
        print(json.dumps({"ssl_only": True, "params": params,
                          "train_subjects": len(pool), "test_subjects": len(test_subs),
                          "train_files": len(train_recs), "test_files": len(test_recs)}))
        return
    Ztr, Ytr, Str = extract_env(model, train_recs, per_file=2, seed=1)
    Zte, Yte, Ste = extract_env(model, test_recs, per_file=2, seed=2)
    reg = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(Ztr, Ytr)
    pred = reg.predict(Zte)
    rs = [float(np.corrcoef(pred[:, k], Yte[:, k])[0, 1]) for k in range(Yte.shape[1])]
    ss_res = float(((pred - Yte) ** 2).sum()); ss_tot = float(((Yte - Ytr.mean(0)) ** 2).sum())
    res = {"args": vars(args), "params": params,
           "train_subjects": len(pool), "test_subjects": len(test_subs),
           "train_windows": int(len(Ztr)), "test_windows": int(len(Zte)),
           "env_r_mean": float(np.nanmean(rs)), "env_r_by_bin": rs,
           "r2_mean_baseline": 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"),
           "subjects_total": len(subs)}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    (out / "eval.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "env_r_by_bin"}, indent=2))


if __name__ == "__main__":
    main()
