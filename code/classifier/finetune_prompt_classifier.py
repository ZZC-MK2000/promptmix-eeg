"""Mode (A): EEG -> 12-way prompt classifier -> frozen MusicGen native sampling.

Architecture:
  Input:  cond.npy (N, 512)  -- bridge(EEG) output, laion-CLAP space
  Head:   MLP 512 -> 256 -> 12 (logits)
  Loss:   cross-entropy against oracle prompt labels
  Oracle: y_r = argmax_k cos(means[k], cond[r]) where means[k] is the mean CLAP embedding
          of the 6 library clips generated from prompt k (from library_radius results).
  Inference: argmax(logits) -> pick that prompt's T5 embedding -> sample from frozen MusicGen.

Why this design:
  - library_radius showed coverage radius 0.8633 (mixture) vs 0.7542 (best single clip) vs
    0.7040 (trivial). Mixture is better, but tolerance radius showed T5-embedding interpolation
    produces noise (jazz<->choir alpha=0.25 zcr=0.46). So we CANNOT interpolate T5 embeddings.
  - Mode (A) sacrifices coverage (0.7542 instead of 0.8633) for texture safety: every delivered
    audio is a native sample from the frozen MusicGen, zcr ~ 0.02-0.05 (real music 0.0556).
  - The classifier is tiny (~130K params) and trains in <1 hour on a single GPU.

Run:  python src/train/finetune_prompt_classifier.py --steps 3000
Smoke: python src/train/finetune_prompt_classifier.py --steps 6 --val-rows 32
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
DATA = ROOT / "data" / "eegcond"
LIB = ROOT / "results" / "gen_eegcond" / "library_radius"
OUT = ROOT / "results" / "prompt_classifier"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))

PROMPTS = [
    "a warm melodic pop piece with soft piano and light drums",
    "upbeat indian film music with strings and tabla",
    "ambient electronic texture with pads and gentle beat",
    "acoustic guitar folk melody, calm and bright",
    "driving electronic dance track with a four-on-the-floor kick and bright synths",
    "lo-fi hip hop beat with dusty drums and mellow keys",
    "sweeping orchestral film score with strings and brass swells",
    "sparse percussion groove with congas and shakers",
    "smooth jazz trio with upright bass, brushed drums and electric piano",
    "distorted electric guitar rock riff with a live drum kit",
    "ethereal choir pad with slow harmonic movement",
    "bouncy chiptune melody with square waves and fast arpeggios",
]
K = len(PROMPTS)


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class PromptClassifier(nn.Module):
    def __init__(self, in_dim=512, hidden=256, k=K, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, k))

    def forward(self, x):
        return self.net(x)


def compute_oracle_labels(cond, lib_emb, lib_owner):
    """For each row in cond, find the prompt whose mean library embedding is closest (cosine).

    cond: (N, 512) bridge(EEG) output
    lib_emb: (72, 512) library clip embeddings (12 prompts x 6 clips)
    lib_owner: (72,) prompt index for each clip
    Returns: (N,) int64 oracle labels in [0, K)
    """
    # Compute mean embedding per prompt
    means = np.stack([unit(lib_emb[lib_owner == k].mean(0)) for k in range(K)])  # (K, 512)
    # Cosine similarity: (N, K)
    cos = unit(cond) @ means.T
    return cos.argmax(1).astype(np.int64), cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--target", default="cond_oracle",
                    choices=["cond_oracle", "true_oracle"],
                    help="cond_oracle=argmax over bridge(EEG) [self-consistent but song-meaningless, "
                         "16.7% agreement with truth]; true_oracle=argmax over decoded true audio "
                         "[the song-matching prompt -- requires compute_true_oracle.py first]")
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-rows", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n"); log.flush()

    # Load data
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    group = np.load(DATA / "group.npy").astype(np.int64)
    lib_emb = np.load(LIB / "library_emb.npy").astype(np.float32)
    lib_owner = np.load(LIB / "library_owner.npy").astype(np.int64)

    say(f"[{datetime.now():%m-%d %H:%M}] cond {cond.shape} | lib_emb {lib_emb.shape} | "
        f"K={K} prompts | target={args.target}")

    # Compute oracle labels
    t0 = time.time()
    if args.target == "true_oracle":
        # Song-matching prompt: argmax over decoded TRUE audio embeddings.
        to_path = OUT / "true_oracle_all.npy"
        if not to_path.exists():
            raise SystemExit(f"{to_path} missing -- run compute_true_oracle.py first.")
        labels = np.load(to_path).astype(np.int64)
        assert len(labels) == len(cond), f"{len(labels)} labels vs {len(cond)} cond rows"
        # cos stats vs true embeddings if available
        te_path = OUT / "true_emb_all.npy"
        if te_path.exists():
            te = np.load(te_path).astype(np.float32)
            means = np.stack([unit(lib_emb[lib_owner == k].mean(0)) for k in range(K)])
            cos_matrix = unit(te) @ means.T
        else:
            cos_matrix = None
        say(f"Loaded TRUE-audio oracle labels from {to_path.name}. Distribution: "
            f"{np.bincount(labels, minlength=K).tolist()}")
    else:
        labels, cos_matrix = compute_oracle_labels(cond, lib_emb, lib_owner)
        say(f"Oracle labels (cond_oracle) computed in {time.time()-t0:.1f}s. Distribution: "
            f"{np.bincount(labels, minlength=K).tolist()}")
    if cos_matrix is not None:
        say(f"Oracle cos: mean={cos_matrix.max(1).mean():.4f} "
            f"min={cos_matrix.max(1).min():.4f} max={cos_matrix.max(1).max():.4f}")

    # Split by group (subject) to prevent leakage
    uniq = np.unique(group)
    rng = np.random.default_rng(args.seed)
    n_val = max(1, int(round(len(uniq) * 0.1)))
    val_g = set(uniq[rng.permutation(len(uniq))[:n_val]].tolist())
    va = np.array([g in val_g for g in group])
    tr_idx = np.where(~va)[0]
    val_idx = np.where(va)[0][:args.val_rows]

    train_ds = TensorDataset(torch.from_numpy(cond[tr_idx]),
                             torch.from_numpy(labels[tr_idx]))
    val_ds = TensorDataset(torch.from_numpy(cond[val_idx]),
                           torch.from_numpy(labels[val_idx]))
    torch.manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                          num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    say(f"Train {len(tr_idx)} | Val {len(val_idx)} | n_groups train={len(np.unique(group[tr_idx]))} "
        f"val={len(np.unique(group[val_idx]))}")

    # Model
    model = PromptClassifier(in_dim=cond.shape[1], hidden=args.hidden, k=K,
                             dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    say(f"PromptClassifier: {n_params:,} params (hidden={args.hidden}, dropout={args.dropout})")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss()

    # Train loop
    it = iter(train_dl)
    ema_loss = None
    t0 = time.time()
    model.train()
    for step in range(1, args.steps + 1):
        try:
            emb, y = next(it)
        except StopIteration:
            it = iter(train_dl)
            emb, y = next(it)
        emb, y = emb.to(device), y.to(device)
        logits = model(emb)
        loss = crit(logits, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        lv = float(loss.item())
        ema_loss = lv if ema_loss is None else 0.95 * ema_loss + 0.05 * lv

        if step % args.log_every == 0:
            acc = float((logits.argmax(1) == y).float().mean())
            say(f"step {step}/{args.steps}  loss ema={ema_loss:.4f}  batch acc={acc:.3f}  "
                f"[{time.time()-t0:.0f}s]")

        if step % args.save_every == 0 or step == args.steps:
            model.eval()
            vs, vn, vc = 0.0, 0, 0
            with torch.no_grad():
                for emb, y in val_dl:
                    emb, y = emb.to(device), y.to(device)
                    logits = model(emb)
                    l = crit(logits, y)
                    vs += float(l.item()) * len(emb)
                    vc += int((logits.argmax(1) == y).sum())
                    vn += len(emb)
            vloss = vs / max(vn, 1)
            vacc = vc / max(vn, 1)
            el = time.time() - t0
            rem = el / step * (args.steps - step)
            ckpt = {"step": step, "model": model.state_dict(),
                    "val_loss": vloss, "val_acc": vacc,
                    "args": vars(args), "labels": labels[:100].tolist()}
            torch.save(ckpt, out / f"ckpt_s{step}.pt")
            say(f"  == step {step} val_loss={vloss:.4f} val_acc={vacc:.4f}  "
                f"存 ckpt_s{step}.pt  [{el:.0f}s | 还需 {rem/60:.0f} 分钟 → "
                f"{(datetime.now()+timedelta(seconds=rem)):%m-%d %H:%M}] ==")
            model.train()

    # Save final
    final = {"step": args.steps, "model": model.state_dict(),
             "args": vars(args), "prompts": PROMPTS,
             "label_distribution": np.bincount(labels, minlength=K).tolist(),
             "oracle_cos_mean": float(cos_matrix.max(1).mean())}
    torch.save(final, out / "final.pt")
    json.dump({"steps": args.steps, "seconds": round(time.time()-t0, 1),
               "args": vars(args), "label_distribution": np.bincount(labels, minlength=K).tolist(),
               "oracle_cos_mean": float(cos_matrix.max(1).mean())},
              open(out / "train_config.json", "w"), indent=2)
    say(f"DONE steps={args.steps} [{time.time()-t0:.0f}s]")
    log.close()


if __name__ == "__main__":
    main()
