"""Train the compact auditory encoder with cross-subject stimulus contrast."""
import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import Corpus, eval_subject  # noqa: E402
from model import AuditoryEncoder, fiber_apply, vicreg  # noqa: E402


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def default_holdout(corpus):
    subj_ds = {}
    for u, d in zip(corpus.subject, corpus.dataset):
        subj_ds.setdefault(u, d)
    held = [s for s in corpus.subjects if s.startswith("CIRE:")][:5]
    for ds in ("kul", "dtu", "musin-g", "ds002725"):
        ss = [s for s in corpus.subjects if subj_ds.get(s) == ds]
        if ss:
            held.append(ss[0])
    return held


def choose_stimuli(corpus, subject, ds, n=10, oracle=False):
    out = []
    for s in corpus.ds_stims[ds]:
        rows = corpus.all_rows_for(subject, [s])
        if rows is None:
            continue
        if oracle and len(rows[0]) < 2:
            continue
        out.append(s)
        if len(out) == n:
            break
    return out


def run_eval(model, corpus, held, device, step):
    res = {"step": step, "cire": [], "dtu": [], "musin": []}
    for u in held:
        if u.startswith("CIRE:"):
            st = choose_stimuli(corpus, u, "cire", 10, oracle=False)
            if len(st) == 10:
                r = eval_subject(model, corpus, u, st, corpus.train_subjects, device)
                if r:
                    res["cire"].append(r["zero"])
        for ds, key in (("dtu", "dtu"), ("musin-g", "musin")):
            st = choose_stimuli(corpus, u, ds, 10, oracle=True)
            if len(st) == 10:
                r = eval_subject(model, corpus, u, st, corpus.train_subjects, device)
                if r and r["oracle"] == r["oracle"]:
                    res[key].append((r["zero"], r["oracle"]))
    out = {"step": step,
           "cire_zero": float(np.mean(res["cire"])) if res["cire"] else float("nan"),
           "dtu_zero": float(np.mean([x[0] for x in res["dtu"]])) if res["dtu"] else float("nan"),
           "dtu_oracle": float(np.mean([x[1] for x in res["dtu"]])) if res["dtu"] else float("nan"),
           "musin_zero": float(np.mean([x[0] for x in res["musin"]])) if res["musin"] else float("nan"),
           "musin_oracle": float(np.mean([x[1] for x in res["musin"]])) if res["musin"] else float("nan")}
    print("EVAL", json.dumps(out), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/Projects/CLaBraM/mainv2/data/auditory_corpus")
    ap.add_argument("--out", default="D:/Projects/CLaBraM/promptmix-eeg/results/auditory_encoder")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--holdout", default="")
    ap.add_argument("--spectral", action="store_true")
    ap.add_argument("--spectral-only", action="store_true")
    ap.add_argument("--no-fiber", action="store_true")
    ap.add_argument("--audio-w", type=float, default=1.0)
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    corpus = Corpus(Path(args.root), held_out=args.holdout.split(",") if args.holdout else ())
    held = args.holdout.split(",") if args.holdout else default_holdout(corpus)
    corpus = Corpus(Path(args.root), held_out=held)
    audio_t = np.load(Path(args.root) / "audio_targets.npy", mmap_mode="r")
    audio_m = np.load(Path(args.root) / "audio_mask.npy")
    print(f"windows {len(corpus.windows)} subjects {len(corpus.subjects)} "
          f"train {len(corpus.train_subjects)} held {held} audio {int(audio_m.sum())}", flush=True)

    model = AuditoryEncoder(d=args.d, depth=args.depth, heads=args.heads,
                            spectral=args.spectral, spectral_only=args.spectral_only).to(device)
    model.subj_head = nn.Sequential(nn.Linear(args.d, 128), nn.GELU(),
                                    nn.Linear(128, len(corpus.subjects))).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                pct_start=0.1)
    local_ds = ["kul", "dtu", "musin-g"]
    logs, evals = [], []
    for step in range(1, args.steps + 1):
        model.train()
        grp = "cire" if rng.random() < 0.5 else "local"
        ds = "cire" if grp == "cire" else local_ds[int(rng.choice(len(local_ds), p=[0.5, 0.25, 0.25]))]
        k, n_sup, n_q = (4, 16, 16) if ds == "cire" else (2, 6, 6)
        if ds == "musin-g":
            n_sup = n_q = 5
        try:
            ep = corpus.sample_episode(rng, ds, corpus.train_subjects, k, n_sup, n_q)
        except RuntimeError as e:
            print("skip", e, flush=True); continue
        xs = torch.tensor(np.concatenate([ep["x_sup"], ep["x_q"]]), device=device)
        subj_ids = torch.tensor(ep["subj"], device=device)
        b = xs.shape[0]
        mask = torch.rand(b, 152, device=device) < 0.5
        z, recon, target = model(xs, mask)
        z_sup, z_q = z[:k * n_sup], z[k * n_sup:]
        za = z_q if args.no_fiber else fiber_apply(z_sup, z_q, n_sup, n_q)[0]
        ha = model.proj(za)
        labels = torch.arange(n_q, device=device).repeat(k)
        logits = ha @ ha.T / 0.1
        l_stim = F.cross_entropy(logits, labels)
        l_rec = F.mse_loss(recon[mask], target[mask])
        l_var, l_cov, var_raw, cov_raw = vicreg(torch.cat([z_sup, z_q], 0))
        l_audio = torch.tensor(0.0, device=device)
        sel = audio_m[ep["rows_q"]]
        if sel.any():
            sel_t = torch.tensor(sel, device=device)
            pred = F.normalize(model.audio_head(za[sel_t]), dim=-1)
            tgt = F.normalize(torch.tensor(np.asarray(audio_t[ep["rows_q"][sel]]), device=device), dim=-1)
            l_audio = (1.0 - (pred * tgt).sum(-1)).mean()
        lam = min(0.5, 0.5 * step / 1000)
        subj_logits = model.subj_head(GRL.apply(ha, lam))
        l_subj = F.cross_entropy(subj_logits, subj_ids.repeat_interleave(n_q))
        loss = l_stim + 0.5 * l_rec + l_var + l_cov + 0.1 * l_subj + args.audio_w * l_audio
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 50 == 0:
            with torch.no_grad():
                acc = float((logits.argmax(1) == labels).float().mean())
                sacc = float((subj_logits.argmax(1) == subj_ids.repeat_interleave(n_q)).float().mean())
            rec = {"step": step, "loss": float(loss.detach()), "stim": float(l_stim.detach()), "acc": acc,
                   "recon": float(l_rec.detach()), "audio": float(l_audio.detach()),
                   "var": float(var_raw.detach()), "cov": float(cov_raw.detach()),
                   "subj_acc": sacc, "ds": ds}
            logs.append(rec)
            print("TRAIN", json.dumps(rec), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            ev = run_eval(model, corpus, held, device, step)
            evals.append(ev)
            torch.save({"model": model.state_dict(), "args": vars(args), "held": held,
                        "step": step}, out / f"encoder_{step}.pt")
            (out / "metrics.json").write_text(json.dumps({"logs": logs, "evals": evals}, indent=2),
                                              encoding="utf-8")
    print("done", flush=True)


if __name__ == "__main__":
    main()
