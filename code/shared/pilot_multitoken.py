"""Pilot: is the single-token condition channel the bottleneck?

Same 512-d bridge(EEG) input, only the cross-attention WIDTH changes:
  k=1 -> clap_proj emits one 768-d KV token (what mainv2/mainv3 train today)
  k=K -> clap_proj emits K KV tokens

Both arms restart from the pretrained decoder with identical seeds and identical step
budgets, then are scored on the SAME 30 held-out targets already on disk from
results/gen_eegcond/identity (TRUE_*.wav), so the comparison is apples-to-apples.

This cannot add mutual information -- it only makes the existing condition easier to
read. If K helps, the channel was the limit; if K does nothing, the limit is the
information in a single 8s EEG embedding and the fix is finer temporal EEG features.

Run one arm:  python src/train/pilot_multitoken.py --k 1 --steps 2000
Both + score:  python src/train/pilot_multitoken.py --both --steps 2000
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
RES = ROOT / "results" / "gen_eegcond"
IDDIR = RES / "identity"
sys.path.insert(0, str(M2 / "src" / "generate"))


class CondWidthModel:
    """Wraps mainv2's MusicGenCLAP but replaces clap_proj with a K-token emitter."""

    def __init__(self, device, k_tokens, lora_r=16, lora_alpha=32):
        from musicgen_clap_finetune import MusicGenCLAP
        self.m = MusicGenCLAP(device=device, use_lora=True,
                              lora_r=lora_r, lora_alpha=lora_alpha)
        t5 = self.m.mg.config.text_encoder.d_model
        self.k = k_tokens
        self.m.clap_proj = nn.Sequential(
            nn.Linear(512, t5), nn.GELU(), nn.Linear(t5, t5 * k_tokens))
        for p in self.m.clap_proj.parameters():
            p.requires_grad = True
        self.m.clap_proj.to(device)
        self.device = device

    def cond(self, emb):                                   # [B,512] -> [B,K,768]
        out = self.m.clap_proj(emb)
        return out.view(out.shape[0], self.k, -1)

    def loss(self, codes, emb):
        from transformers.modeling_outputs import BaseModelOutput
        dec_input, labels = self.m._delayed_inputs(codes.to(self.m.device))
        enc = BaseModelOutput(last_hidden_state=self.cond(emb.to(self.m.device)))
        return self.m.mg(encoder_outputs=enc, decoder_input_ids=dec_input,
                         labels=labels, return_dict=True).loss

    @torch.no_grad()
    def generate(self, emb, max_new_tokens=400, temperature=0.9, top_k=None, top_p=None):
        from transformers.modeling_outputs import BaseModelOutput
        if emb.ndim == 1:
            emb = emb.unsqueeze(0)
        enc = BaseModelOutput(last_hidden_state=self.cond(emb.to(self.m.device)))
        kw = {}
        if top_k is not None:
            kw["top_k"] = top_k
        if top_p is not None:
            kw["top_p"] = top_p
        out = self.m.mg.generate(encoder_outputs=enc, max_new_tokens=max_new_tokens,
                                 temperature=temperature, do_sample=True, **kw)
        return np.asarray(out.cpu()).squeeze()

    def state(self):
        sd = {"clap_proj": {k: v.cpu() for k, v in self.m.clap_proj.state_dict().items()},
              "k": self.k}
        sd["decoder_lora"] = {k: v.cpu() for k, v in self.m.mg.decoder.state_dict().items()
                              if "lora_" in k}
        return sd


def dataloaders(batch, seed):
    from torch.utils.data import TensorDataset, DataLoader
    cond = torch.from_numpy(np.load(DATA / "cond.npy").astype(np.float32))
    codes = torch.from_numpy(np.load(DATA / "codes.npy")).long()
    group = np.load(DATA / "group.npy").astype(np.int64)
    uniq = np.unique(group)
    rng = np.random.default_rng(seed)
    val_g = set(uniq[rng.permutation(len(uniq))[:max(1, int(round(len(uniq) * 0.1)))]].tolist())
    va = np.array([g in val_g for g in group])
    tr = TensorDataset(cond[~va], codes[~va])
    torch.manual_seed(seed)                        # same shuffle order for both arms
    return DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0, drop_last=True), va


def train_arm(k, steps, batch, lr_proj, lr_lora, device, seed=0):
    print(f"\n===== arm k={k}: training {steps} steps =====", flush=True)
    model = CondWidthModel(device, k_tokens=k)
    dl, _ = dataloaders(batch, seed)
    proj_p = list(model.m.clap_proj.parameters())
    lora_p = [p for p in model.m.mg.decoder.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": proj_p, "lr": lr_proj},
                             {"params": lora_p, "lr": lr_lora}], weight_decay=0.01)
    it = iter(dl)
    ema, t0 = None, time.time()
    model.m.train()
    for s in range(1, steps + 1):
        try:
            emb, codes = next(it)
        except StopIteration:
            it = iter(dl)
            emb, codes = next(it)
        loss = model.loss(codes, emb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(proj_p + lora_p, 1.0)
        opt.step()
        lv = float(loss.item())
        del loss
        ema = lv if ema is None else 0.9 * ema + 0.1 * lv
        if s % 100 == 0 or s == steps:
            print(f"  k={k} step {s}/{steps} loss={lv:.4f} ema={ema:.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
    print(f"  arm k={k} done in {time.time()-t0:.0f}s, final ema={ema:.4f}", flush=True)
    return model, ema


def score(model, rows, cond, device, tokens=400, temp=0.9):
    """Generate on the same held-out targets and score vs TRUE_*.wav already on disk."""
    import soundfile as sf
    from musicgen_clap_finetune import load_clap, clap_embed_audio

    wav_sr = 32000
    clap, proc, csr = load_clap(device)
    unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    tps = sorted(IDDIR.glob("TRUE_*.wav"))
    rowkeys = [p.stem.split("row")[-1] for p in tps]

    def load32(p):
        w, s = sf.read(str(p), dtype="float32")
        return (w.mean(1) if w.ndim > 1 else w).astype(np.float32)

    T = unit(np.stack([clap_embed_audio(clap, proc, wav_sr, load32(p), device) for p in tps]))
    tt_off = float((T @ T.T)[np.triu_indices(len(tps), 1)].mean())

    G, graw = [], []
    for r in rows:
        w = np.asarray(model.generate(torch.from_numpy(cond[r]), max_new_tokens=tokens,
                                      temperature=temp)).reshape(-1)
        graw.append(w)
        G.append(clap_embed_audio(clap, proc, wav_sr, w, device))
        (IDDIR / f"pilot_k{model.k}_row{r:05d}.wav").parent.mkdir(exist_ok=True)
        sf.write(str(IDDIR / f"pilot_k{model.k}_row{r:05d}.wav"), w, wav_sr)
    G = unit(np.stack(G))
    idx = [rowkeys.index(f"{r:05d}") for r in rows]
    S = G @ T.T
    own = np.array([S[i, idx[i]] for i in range(len(rows))])
    others = np.stack([np.delete(S[i], idx[i]).mean() for i in range(len(rows))])
    ranks = [int(np.where(np.argsort(-S[i]) == idx[i])[0][0]) + 1 for i in range(len(rows))]
    GG = G @ G.T
    gg_off = float(GG[np.triu_indices(len(rows), 1)].mean())
    n = len(rows)
    spec = json.loads((IDDIR / "test_spec.json").read_text(encoding="utf-8"))
    row_song = {f"{int(rr):05d}": s for rr, s in zip(spec["pick"], spec["songs"])}
    songs = np.array([row_song[f"{r:05d}"] for r in rows])            # the picks
    # reference pool may be larger than the picks; index songs by COLUMN of S
    songs_ref = np.array([row_song[rk] for rk in rowkeys])
    p = np.bincount(songs, minlength=14) / n
    chance_song = float((p * p).sum())
    d = own - others
    t = float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
    out = dict(k=model.k, n=n, window_top1=sum(r == 1 for r in ranks) / n,
               mean_rank=float(np.mean(ranks)), margin=float(d.mean()), t=t,
               song_top1=sum(1 for i in range(n)
                             if songs_ref[int(np.argmax(S[i]))] == songs[i]) / n,
               chance_window=1 / n, chance_song=chance_song,
               own=float(own.mean()), others=float(others.mean()),
               gen_gen=gg_off, true_true=tt_off, ema_loss=None)
    print(f"\n  SCORE k={model.k}: window top1={out['window_top1']:.3f} "
          f"(chance {out['chance_window']:.3f})  mean rank={out['mean_rank']:.2f}  "
          f"margin={out['margin']:+.4f} t={out['t']:+.2f}  "
          f"song top1={out['song_top1']:.3f} (chance {out['chance_song']:.3f})  "
          f"GEN-v-GEN={out['gen_gen']:.4f} own={out['own']:.4f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--both", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-proj", type=float, default=3e-4)
    ap.add_argument("--lr-lora", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-score", type=int, default=30)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    _, va = dataloaders(args.batch, 0)
    spec = json.loads((IDDIR / "test_spec.json").read_text(encoding="utf-8"))
    rows = spec["pick"][:args.n_score]
    print(f"scoring on the same {len(rows)} held-out targets as the ep7 identity test")

    ks = [1, args.k] if args.both else [args.k]
    results = []
    for k in ks:
        model, ema = train_arm(k, args.steps, args.batch, args.lr_proj, args.lr_lora, device)
        r = score(model, rows, cond, device)
        r["ema_loss"] = ema
        torch.save(model.state(), RES / f"pilot_k{k}_pt{args.steps}.pt")
        results.append(r)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    (RES / "pilot_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\n===== PILOT SUMMARY (identical budget, only condition width differs) =====")
    hdr = f"{'k':>3} {'ema_loss':>9} {'win top1':>9} {'chance':>7} {'mean rank':>10} " \
          f"{'margin':>8} {'t':>6} {'song top1':>10} {'chance':>7} {'GEN-v-GEN':>10} {'own':>7}"
    print(hdr)
    for r in results:
        print(f"{r['k']:>3} {r['ema_loss']:>9.4f} {r['window_top1']:>9.3f} "
              f"{r['chance_window']:>7.3f} {r['mean_rank']:>10.2f} {r['margin']:>+8.4f} "
              f"{r['t']:>+6.2f} {r['song_top1']:>10.3f} {r['chance_song']:>7.3f} "
              f"{r['gen_gen']:>10.4f} {r['own']:>7.4f}")
    if len(results) == 2:
        a, b = results
        print(f"\n判读: k={b['k']} 相对 k=1 窗级 top-1 {b['window_top1']-a['window_top1']:+.3f}, "
              f"t 提升 {b['t']-a['t']:+.2f}; 若 t 和 top-1 都没过随机水平多少, "
              f"说明瓶颈在信息量而不是通道宽度")


if __name__ == "__main__":
    main()
