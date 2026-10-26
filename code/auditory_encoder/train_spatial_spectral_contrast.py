"""Contrastive + masked-spectral training for the spatial-spectral encoder.

Uses the existing auditory corpus episodes (subjects sharing stimuli) to learn a
cross-subject stimulus embedding, plus masked spectral reconstruction and VICReg.
No target-subject labels are used at test time.

Run:
  python train_spatial_spectral_contrast.py --steps 500 --out results/spatial_spectral_contrast
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from data import Corpus  # noqa: E402
from spatial_spectral import SpatialSpectralEncoder, masked_spectral_loss, vicreg_terms  # noqa: E402


def norm_windows(x):
    x = x.float()
    mu = x.mean(-1, keepdim=True)
    sd = x.std(-1, keepdim=True) + 1e-6
    return torch.clamp((x - mu) / sd, -10.0, 10.0)


def stratified_subjects(corpus, n, seed):
    by = defaultdict(list)
    for u in corpus.train_subjects:
        by[corpus.subj_ds[u]].append(u)
    ds = sorted(by)
    total = sum(len(v) for v in by.values())
    # episode sampler needs 4 CIRE subjects and 2 for the other datasets
    floor = {d: (4 if d == "cire" else 2) for d in ds}
    alloc = {d: max(floor[d], int(n * len(by[d]) / total)) for d in ds}
    while sum(alloc.values()) > n:
        d = max((d for d in ds if alloc[d] > 1), key=lambda x: alloc[x])
        alloc[d] -= 1
    while sum(alloc.values()) < n:
        d = max((d for d in ds if alloc[d] < len(by[d])),
                key=lambda x: n * len(by[x]) / total - alloc[x])
        alloc[d] += 1
    rng = np.random.default_rng(seed)
    out = []
    for d in ds:
        out += list(rng.choice(by[d], alloc[d], replace=False))
    return sorted(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT.parent / "mainv2" / "data" / "auditory_corpus"))
    ap.add_argument("--out", default=str(ROOT / "results" / "spatial_spectral_contrast"))
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--rec-w", type=float, default=0.1)
    ap.add_argument("--con-w", type=float, default=1.0)
    ap.add_argument("--vicreg-w", type=float, default=0.05)
    ap.add_argument("--audio-w", type=float, default=0.1)
    ap.add_argument("--n-filters", type=int, default=8)
    ap.add_argument("--spectral-skip", action="store_true")
    ap.add_argument("--temp", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subjects", type=int, default=0,
                    help="use a stratified subset of N training subjects (0 = all)")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    corpus = Corpus(Path(args.root))
    audio_t, audio_m = None, None
    if args.audio_w > 0 and (Path(args.root) / "audio_targets.npy").exists():
        audio_t = np.load(Path(args.root) / "audio_targets.npy", mmap_mode="r")
        audio_m = np.load(Path(args.root) / "audio_mask.npy")
    if args.subjects:
        corpus.train_subjects = stratified_subjects(corpus, args.subjects, args.seed)
    print(f"windows {len(corpus.windows)} subjects {len(corpus.train_subjects)} "
          f"datasets {corpus.datasets}", flush=True)
    model = SpatialSpectralEncoder(n_ch=19, d=128, depth=4, heads=4,
                                   n_filters=args.n_filters, patch=200, sfreq=200,
                                   spectral_skip=args.spectral_skip).to(device)
    print("params", sum(p.numel() for p in model.parameters()), flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.1)
    local = ["kul", "dtu", "musin-g", "ds002725", "sparr"]
    logs = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        group = "cire" if rng.random() < 0.5 else "local"
        p_local = np.full(len(local), 1.0 / len(local))
        ds = "cire" if group == "cire" else local[int(rng.choice(len(local), p=p_local))]
        k, n_sup, n_q = (4, 16, 16) if ds == "cire" else (2, 6, 6)
        if ds == "musin-g":
            n_sup = n_q = 5
        try:
            ep = corpus.sample_episode(rng, ds, corpus.train_subjects, k, n_sup, n_q)
        except RuntimeError:
            continue
        xs = norm_windows(torch.tensor(np.concatenate([ep["x_sup"], ep["x_q"]]), device=device))
        labels = np.concatenate([np.tile(np.arange(n_sup), k),
                                 np.tile(np.arange(n_sup, n_sup + n_q), k)])
        labels = torch.tensor(labels, device=device)
        b = xs.shape[0]
        mask = torch.rand(b, 8, device=device) < 0.25
        none = ~mask.any(1)
        if none.any():
            mask[none, 0] = True
        model.train()
        content, recon, feats = model(xs, mask)
        l_rec = masked_spectral_loss(recon, feats, mask)
        z = F.normalize(content, dim=-1)
        logits = z @ z.T / args.temp
        l_con = F.cross_entropy(logits, labels)
        l_var = vicreg_terms(content)
        l_audio = torch.zeros((), device=device)
        if audio_t is not None:
            sel = audio_m[ep["rows_q"]]
            if sel.any():
                sel_t = torch.tensor(sel, device=device)
                pred = F.normalize(model.audio_head(content[k * n_sup:][sel_t]), dim=-1)
                tgt = F.normalize(torch.tensor(np.asarray(audio_t[ep["rows_q"][sel]]), device=device), dim=-1)
                l_audio = (1.0 - (pred * tgt).sum(-1)).mean()
        loss = (args.rec_w * l_rec + args.con_w * l_con + args.vicreg_w * l_var
                + args.audio_w * l_audio)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 20 == 0 or step == 1:
            rec = {"step": step, "loss": float(loss), "rec": float(l_rec), "con": float(l_con),
                   "audio": float(l_audio), "var": float(l_var),
                   "sec": round(time.time() - t0, 1), "ds": ds}
            logs.append(rec); print(json.dumps(rec), flush=True)
    ck = {"model": model.state_dict(), "args": vars(args),
          "params": sum(p.numel() for p in model.parameters())}
    path = out / f"encoder_{args.steps}.pt"
    torch.save(ck, path)
    (out / "train_log.json").write_text(json.dumps({"args": vars(args), "logs": logs}, indent=2),
                                        encoding="utf-8")
    print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
