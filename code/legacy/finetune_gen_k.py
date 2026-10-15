"""Full run: EEG-conditioned generator with a K-token cross-attention condition.

Pilot (8000 steps/arm, identical seed & budget) showed condition WIDTH is the bottleneck:
  k=1: window top-1 13.3%, paired t +1.78, GEN-v-GEN 0.774 > own 0.713 (collapsed)
  k=8: window top-1 26.7%, paired t +4.03, own 0.721 > GEN-v-GEN 0.711 (no longer collapsed)
and k=8 beat the 32200-step k=1 run on both, so this is the width, not the extra training.

Reuse: CondWidthModel + split from pilot_multitoken.py (same code path the pilot validated).

Cost discipline: the previous run spent ~80% of wall clock on an end-of-epoch
full-set teacher-forced top-1 (chunk=64 allocated ~0.8 GB of logits on a 16 GB card).
Here validation is a small loss-only subsample by default; the real readout is the
identity test, not that statistic.

Run:  python src/train/finetune_gen_k.py --k 8 --epochs 30
Resume: python src/train/finetune_gen_k.py --k 8 --epochs 30 --resume <ckpt.pt>
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
from pilot_multitoken import CondWidthModel, DATA   # validated in the pilot


def load_state(model, path):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    if int(ck.get("k", -1)) != model.k:
        raise SystemExit(f"ckpt k={ck.get('k')} != --k {model.k}")
    model.m.clap_proj.load_state_dict(ck["clap_proj"])
    model.m.mg.decoder.load_state_dict(ck["decoder_lora"], strict=False)
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-proj", type=float, default=3e-4)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--val-rows", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--outdir", default=str(ROOT / "results" / "gen_eegcond_k8"))
    ap.add_argument("--save-every", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=0, help="0 = full epochs")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    OUT = Path(args.outdir)
    OUT.mkdir(parents=True, exist_ok=True)
    LOGH = open(OUT / "train.log", "a", encoding="utf-8")

    def say(msg):
        line = f"{msg}"
        print(line, flush=True)
        LOGH.write(line + "\n")
        LOGH.flush()

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy")
    group = np.load(DATA / "group.npy").astype(np.int64)
    cond_t = torch.from_numpy(cond)
    code_t = torch.from_numpy(codes).long()

    uniq = np.unique(group)
    rng = np.random.default_rng(args.seed)
    n_val_g = max(1, int(round(len(uniq) * 0.1)))
    val_g = set(uniq[rng.permutation(len(uniq))[:n_val_g]].tolist())
    va = np.array([g in val_g for g in group])
    tr_idx = np.where(~va)[0]
    va_idx = np.where(va)[0]
    say(f"[{datetime.now():%m-%d %H:%M}] device={device} k={args.k} "
        f"train rows {len(tr_idx)} val groups {len(set(group[va_idx]))}")

    tr_ds = torch.utils.data.TensorDataset(cond_t[tr_idx], code_t[tr_idx])
    vsub = va_idx[:args.val_rows] if args.val_rows else va_idx
    va_ds = torch.utils.data.TensorDataset(cond_t[vsub], code_t[vsub])
    torch.manual_seed(args.seed)
    dl = DataLoader(tr_ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    drop_last=True)
    vdl = DataLoader(va_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = CondWidthModel(device, k_tokens=args.k)
    if args.resume:
        ck = load_state(model, args.resume)
        say(f"resumed {Path(args.resume).name} (k={ck['k']})")
    proj_p = list(model.m.clap_proj.parameters())
    lora_p = [p for p in model.m.mg.decoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": proj_p, "lr": args.lr_proj},
                             {"params": lora_p, "lr": args.lr_lora}], weight_decay=args.wd)

    steps_per_ep = len(tr_idx) // args.batch
    say(f"steps/epoch = {steps_per_ep}")
    best = float("inf")
    t0 = time.time()
    steps = 0
    for ep in range(args.epochs):
        model.m.train()
        ema = None
        for emb, cd in dl:
            loss = model.loss(cd, emb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj_p + lora_p, 1.0)
            opt.step()
            lv = float(loss.item())
            del loss
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            steps += 1
            if steps % 200 == 0:
                say(f"ep{ep} step {steps} (in-ep {steps % steps_per_ep}) loss={lv:.4f} "
                    f"ema={ema:.4f} [{time.time()-t0:.0f}s]")
            if args.max_steps and steps >= args.max_steps:
                break

        model.m.eval()
        vsum, vn = 0.0, 0
        with torch.no_grad():
            for emb, cd in vdl:
                l = model.loss(cd, emb)
                vsum += float(l.item()) * len(emb)
                vn += len(emb)
        vloss = vsum / max(vn, 1)
        el = time.time() - t0
        remain = el / (ep + 1) * (args.epochs - ep - 1)
        say(f"== epoch {ep} val_loss={vloss:.4f} ema_train={ema:.4f} "
            f"[{el:.0f}s | 预计还需 {remain/3600:.1f}h "
            f"-> {(datetime.now()+timedelta(seconds=remain)):%m-%d %H:%M}] ==")

        st = model.state()
        st.update({"epoch": ep, "val_loss": vloss, "k": args.k})
        if args.max_steps:                       # smoke run: don't write real ckpts
            break
        torch.save(st, OUT / f"ckpt_ep{ep}.pt")
        if vloss < best:
            best = vloss
            torch.save(st, OUT / "best.pt")
            say(f"  saved best.pt (val_loss={vloss:.4f})")

    json.dump({"best_val_loss": best, "steps": steps, "k": args.k,
               "seconds": round(time.time() - t0, 1), "args": vars(args)},
              open(OUT / "train_config.json", "w"), indent=2)
    say(f"DONE steps={steps} best_val_loss={best:.4f} [{time.time()-t0:.0f}s]")
    LOGH.close()


if __name__ == "__main__":
    main()
