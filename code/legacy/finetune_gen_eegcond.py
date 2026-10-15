"""mainv3: fine-tune ONLY the MusicGen generator so its condition INPUT is bridge(EEG).

Pipeline stays EEG -> aldm align -> ridge bridge -> laion-CLAP space -> generator.
The generator is fed the exact vector used at inference (bridge(EEG), a unit-norm
laion-space 512-d), and supervised to reproduce the Encodec codes of the corresponding
8s stimulus audio. Small gap -> rewarded, large gap -> penalised (delay-pattern CE).

We reuse mainv2's MusicGenCLAP (with the _delayed_inputs fix + LoRA) unchanged.
Only the decoder LoRA adapters + clap_proj are trained; audio/text encoders frozen.

Data (from src/prepare/build_eegcond_pairs.py):
  cond.npy   [N,512]  float32  bridge(EEG)           = model input
  codes.npy  [N,4,400] int16   matched-audio codes   = target
  group.npy  [N]      int      unique-target group id (split by this, no leakage)

Run (GPU):  python src/train/finetune_gen_eegcond.py --epochs 20 --batch-size 16
Smoke(CPU): python src/train/finetune_gen_eegcond.py --max-steps 6 --epochs 1
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
OUTDIR = ROOT / "results" / "gen_eegcond"
sys.path.insert(0, str(M2 / "src" / "generate"))


class PairDataset(Dataset):
    def __init__(self, cond, codes):
        self.cond = torch.from_numpy(cond)                    # [N,512] f32
        self.codes = torch.from_numpy(codes).long()           # [N,4,400] i64

    def __len__(self):
        return len(self.cond)

    def __getitem__(self, i):
        return self.cond[i], self.codes[i]


def split_by_group(group, val_frac=0.1, seed=0):
    uniq = np.unique(group)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_groups = set(uniq[order[:n_val]].tolist())
    is_val = np.array([g in val_groups for g in group.tolist()])
    return ~is_val, is_val


def count_top1_acc(model, codes, cond, chunk=64):
    """Teacher-forced top-1 on the delay-pattern layout, computed directly from
    logits (independent of HF's internal label shifting): logits[b,c,pos] predicts
    the delayed token D[b,c,pos+1] = codes[c, pos+1-c] when pos+1>=c."""
    model.eval()
    correct = total = 0
    from transformers.modeling_outputs import BaseModelOutput
    with torch.no_grad():
        for a in range(0, len(codes), chunk):
            cb = codes[a:a + chunk].to(model.device)
            eb = cond[a:a + chunk].to(model.device)
            dec_input, _ = model._delayed_inputs(cb)          # [B*C, T-1]
            cross = model.clap_proj(eb).unsqueeze(1)
            out = model.mg(encoder_outputs=BaseModelOutput(last_hidden_state=cross),
                           decoder_input_ids=dec_input, return_dict=True)
            B, C, T = cb.shape
            pred = out.logits.reshape(B, C, T - 1, -1).argmax(-1)   # [B,C,T-1]
            tgt = torch.full((B, C, T - 1), -1, dtype=cb.dtype, device=cb.device)
            for c in range(C):
                p0 = max(0, c - 1)                             # first valid pos
                src = torch.arange(p0 + 1 - c, T - c, device=cb.device)
                tgt[:, c, p0:] = cb[:, c, src]
            mask = tgt >= 0
            correct += (pred[mask] == tgt[mask]).sum().item()
            total += int(mask.sum().item())
    return correct / max(total, 1)


def train(args):
    device = "cuda" if (torch.cuda.is_available() and not args.cpu) else "cpu"
    print(f"Device: {device}")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int16)
    group = np.load(DATA / "group.npy").astype(np.int64)
    print(f"cond {cond.shape}  codes {codes.shape}  groups {int(group.max())+1}")

    tr_mask, va_mask = split_by_group(group, val_frac=args.val_frac, seed=args.seed)
    print(f"train rows {int(tr_mask.sum())}  val rows {int(va_mask.sum())}  "
          f"(val groups {len(np.unique(group[va_mask]))})")

    # cap val for speed if huge
    va_idx = np.where(va_mask)[0]
    if len(va_idx) > args.val_max:
        va_idx = va_idx[np.random.default_rng(args.seed).permutation(len(va_idx))[:args.val_max]]
    val_ds = PairDataset(cond[va_idx], codes[va_idx])

    tr_idx = np.where(tr_mask)[0]
    if args.max_train:
        tr_idx = tr_idx[np.random.default_rng(args.seed).permutation(len(tr_idx))[:args.max_train]]
    train_ds = PairDataset(cond[tr_idx], codes[tr_idx])
    # num_workers=0: Windows DataLoader spawn can deadlock; pure in-memory indexing is cheap.
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=0, drop_last=True, pin_memory=(device == "cuda"))
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    from musicgen_clap_finetune import MusicGenCLAP, _save_ckpt
    model = MusicGenCLAP(device=device, use_lora=args.lora,
                         lora_r=args.lora_r, lora_alpha=args.lora_alpha)

    proj_params = list(model.clap_proj.parameters())
    lora_params = [p for p in model.mg.decoder.parameters() if p.requires_grad]
    print(f"trainable: proj {sum(p.numel() for p in proj_params)/1e6:.2f}M  "
          f"lora {sum(p.numel() for p in lora_params)/1e6:.2f}M")

    groups = [{"params": proj_params, "lr": args.lr_proj}]
    if lora_params:
        groups.append({"params": lora_params, "lr": args.lr_lora})
    opt = torch.optim.AdamW(groups, weight_decay=args.wd)

    # warm-up: only clap_proj learns first (random proj feeds garbage cross-attn;
    # letting LoRA chase that moves it the wrong way). Then unfreeze adapters.
    lora_frozen = False
    if args.warmup_proj_steps > 0 and lora_params:
        for p in lora_params:
            p.requires_grad = False
        lora_frozen = True

    steps = 0
    global_best = float("inf")
    t0 = time.time()
    ema = None
    stop = False
    for epoch in range(args.epochs):
        model.train()
        for cond_b, code_b in train_dl:
            if lora_frozen and steps >= args.warmup_proj_steps:
                for p in lora_params:
                    p.requires_grad = True
                lora_frozen = False
                print(f"  [{steps} steps] proj warm-up done, LoRA unfrozen")

            loss = model.forward_from_codes(code_b, cond_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(proj_params + lora_params, args.clip)
            opt.step()
            lv = loss.item()
            ema = lv if ema is None else 0.9 * ema + 0.1 * lv
            del loss
            steps += 1

            if steps % args.log_every == 0:
                print(f"ep{epoch} step{steps}  loss={lv:.4f} ema={ema:.4f}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)

            if args.max_steps and steps >= args.max_steps:
                stop = True
                break
        if stop:
            break

        # ── validation ──
        model.eval()
        vloss, vn = 0.0, 0
        with torch.no_grad():
            for cond_b, code_b in val_dl:
                l = model.forward_from_codes(code_b, cond_b)
                vloss += l.item() * len(cond_b)
                vn += len(cond_b)
        vloss /= max(vn, 1)
        acc = count_top1_acc(model, val_ds.codes, val_ds.cond) if not args.no_acc else 0.0
        print(f"== epoch {epoch}  val_loss={vloss:.4f}  val_top1={acc:.4f}  "
              f"ema_train={ema:.4f}  [{time.time()-t0:.0f}s] ==", flush=True)

        _save_ckpt(model, OUTDIR / f"ckpt_ep{epoch}.pt", epoch, vloss)
        if vloss < global_best:
            global_best = vloss
            _save_ckpt(model, OUTDIR / "best.pt", epoch, vloss)
            print(f"  saved best.pt (val_loss={vloss:.4f})")
        if args.max_steps:
            break

    json.dump({"best_val_loss": global_best, "steps": steps,
               "seconds": round(time.time() - t0, 1), "args": vars(args)},
              open(OUTDIR / "train_config.json", "w"), indent=2)
    print(f"DONE steps={steps} best_val_loss={global_best:.4f} [{time.time()-t0:.0f}s]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr-proj", type=float, default=3e-4)
    p.add_argument("--lr-lora", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--warmup-proj-steps", type=int, default=300)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--val-max", type=int, default=4096)
    p.add_argument("--max-train", type=int, default=0, help="0 = all")
    p.add_argument("--max-steps", type=int, default=0, help="0 = until epoch end")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lora", action="store_true", default=True)
    p.add_argument("--no-lora", dest="lora", action="store_false")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--no-acc", action="store_true")
    p.add_argument("--cpu", action="store_true")
    train(p.parse_args())


if __name__ == "__main__":
    main()
