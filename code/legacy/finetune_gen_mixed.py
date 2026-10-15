"""Mixed-condition fine-tune: keep the EEG adaptation, re-learn to READ laion-space conditions.

Why this design (all measured earlier):
  - the condition port carries strong song-level identity (margin = 13 standard errors at the
    embedding level) but the generator only passes 5.5, so the loss is inside the generator;
  - roughness is NOT from sampling (5 recipes identical), NOT from LoRA damaging the prior
    (removing LoRA doubles the buzz), NOT from training length (flat across checkpoints);
  - our EEG corpus has only 1103 distinct 8s targets. FMA gives 4132 MORE (condition -> codes)
    pairs from very different music, which is what teaches a decoder that laion-space
    vectors are legitimate conditions.

Start from ckpt_ep16 (already EEG-adapted) and train on a mix: p_eeg of the batches are
EEG pairs, the rest are FMA audio-laion pairs. Learning rates are 1/3 of the original to
reduce interference. Both pools' losses are logged SEPARATELY so that interference (FMA
helping while EEG loss rises) is visible immediately, not discovered at the end.

Run:  python src/train/finetune_gen_mixed.py --steps 12000 --p-eeg 0.5
Smoke: python src/train/finetune_gen_mixed.py --steps 6 --save-every 6 --val-rows 32
"""
import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
DATA = ROOT / "data" / "eegcond"
FMA8 = ROOT / "data" / "fma8s"
OUT = ROOT / "results" / "gen_eegcond_mixed"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
from pilot_multitoken import CondWidthModel   # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--p-eeg", type=float, default=0.5)
    ap.add_argument("--lr-proj", type=float, default=1e-4)
    ap.add_argument("--lr-lora", type=float, default=3e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warm-start", default=str(ROOT / "results" / "gen_eegcond_k8" / "ckpt_ep16.pt"))
    ap.add_argument("--save-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--val-rows", type=int, default=1200)
    ap.add_argument("--outdir", default=str(OUT))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train.log", "a", encoding="utf-8")

    def say(s):
        print(s, flush=True)
        log.write(s + "\n"); log.flush()

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy")
    group = np.load(DATA / "group.npy").astype(np.int64)
    fcond = np.load(FMA8 / "cond.npy").astype(np.float32)
    fcodes = np.load(FMA8 / "codes.npy")

    uniq = np.unique(group)
    rng = np.random.default_rng(args.seed)
    n_val = max(1, int(round(len(uniq) * 0.1)))
    val_g = set(uniq[rng.permutation(len(uniq))[:n_val]].tolist())
    va = np.array([g in val_g for g in group])
    tr = np.where(~va)[0]
    vi = np.where(va)[0][:args.val_rows]

    eeg_ds = TensorDataset(torch.from_numpy(cond[tr]), torch.from_numpy(codes[tr]).long())
    fma_ds = TensorDataset(torch.from_numpy(fcond), torch.from_numpy(fcodes).long())
    val_ds = TensorDataset(torch.from_numpy(cond[vi]), torch.from_numpy(codes[vi]).long())
    torch.manual_seed(args.seed)
    eeg_dl = DataLoader(eeg_ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    fma_dl = DataLoader(fma_ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    say(f"[{datetime.now():%m-%d %H:%M}] EEG train {len(tr)} | FMA {len(fma_ds)} | "
        f"val rows {len(vi)} | p_eeg={args.p_eeg} k={args.k}")

    model = CondWidthModel(device, k_tokens=args.k)
    if args.warm_start:
        ck = torch.load(args.warm_start, map_location="cpu", weights_only=True)
        model.m.clap_proj.load_state_dict(ck["clap_proj"])
        model.m.mg.decoder.load_state_dict(ck["decoder_lora"], strict=False)
        say(f"warm start {Path(args.warm_start).name} epoch={ck.get('epoch')} "
            f"val_loss={ck.get('val_loss'):.4f}")
    else:
        say("cold start: pristine pretrained decoder + freshly initialised clap_proj")

    proj_p = list(model.m.clap_proj.parameters())
    lora_p = [p for p in model.m.mg.decoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": proj_p, "lr": args.lr_proj},
                            {"params": lora_p, "lr": args.lr_lora}], weight_decay=args.wd)

    it_e, it_f = iter(eeg_dl), iter(fma_dl)
    gen = np.random.default_rng(args.seed + 1)
    eema = fema = None
    t0 = time.time()
    model.m.train()
    for step in range(1, args.steps + 1):
        use_eeg = gen.random() < args.p_eeg
        if use_eeg:
            try:
                emb, cd = next(it_e)
            except StopIteration:
                it_e = iter(eeg_dl); emb, cd = next(it_e)
        else:
            try:
                emb, cd = next(it_f)
            except StopIteration:
                it_f = iter(fma_dl); emb, cd = next(it_f)
        loss = model.loss(cd, emb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proj_p + lora_p, 1.0)
        opt.step()
        lv = float(loss.item())
        del loss
        if use_eeg:
            eema = lv if eema is None else 0.95 * eema + 0.05 * lv
        else:
            fema = lv if fema is None else 0.95 * fema + 0.05 * lv

        fmt = lambda v: f"{v:.4f}" if v is not None else "  --  "
        if step % args.log_every == 0:
            say(f"step {step}/{args.steps}  EEG池loss ema={fmt(eema)}  FMA池loss ema={fmt(fema)}  "
                f"[{time.time()-t0:.0f}s]")
        if step % args.save_every == 0 or step == args.steps:
            model.m.eval()
            vs, vn = 0.0, 0
            with torch.no_grad():
                for emb, cd in val_dl:
                    l = model.loss(cd, emb)
                    vs += float(l.item()) * len(emb); vn += len(emb)
            vloss = vs / max(vn, 1)
            el = time.time() - t0
            rem = el / step * (args.steps - step)
            st = model.state(); st.update({"step": step, "val_loss": vloss, "k": args.k})
            torch.save(st, out / f"ckpt_s{step}.pt")
            say(f"  == step {step} EEG留out val_loss={vloss:.4f}  存 ckpt_s{step}.pt  "
                f"[{el:.0f}s | 还需 {rem/60:.0f} 分钟 → "
                f"{(datetime.now()+timedelta(seconds=rem)):%m-%d %H:%M}] ==")
            model.m.train()

    json.dump({"steps": args.steps, "p_eeg": args.p_eeg, "seconds": round(time.time()-t0, 1),
               "args": vars(args)}, open(out / "train_config.json", "w"), indent=2)
    say(f"DONE steps={args.steps} [{time.time()-t0:.0f}s]")
    log.close()


if __name__ == "__main__":
    main()
