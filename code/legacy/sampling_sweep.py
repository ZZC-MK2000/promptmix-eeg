"""Sampling-recipe sweep: does MusicGen's official sampling make EEG-driven music better?

Finding that motivated this: musicgen-small's generation_config has top_k/top_p/temperature
= None, and our calls pass only temperature -> every step samples the FULL 2048-way codebook
with no truncation. MusicGen's official recipe uses top_k=250. Truncation is the cheapest
"sound better" lever and needs NO retraining.

For each recipe we report both sides of the trade-off on the same 60 held-out targets:
  musicality proxy -> you judge by ear (wavs are written for identical rows across recipes)
  condition faithfulness -> margin = own - others averaged over targets, with its standard
      error SE = std/ sqrt(n) and t = margin/SE (|t|>=2 counts as solid), plus top-1 and
      mean rank against their chance lines.

Run: python src/train/sampling_sweep.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
OUT = ROOT / "results" / "gen_eegcond" / "sampling"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

WAV_SR = 32000

RECIPES = [
    dict(name="无截断T0.9(现状)", temperature=0.9, top_k=None, top_p=None),
    dict(name="T1.0/k250(MusicGen官方)", temperature=1.0, top_k=250, top_p=None),
    dict(name="T0.85/k250", temperature=0.85, top_k=250, top_p=None),
    dict(name="T0.85/k50", temperature=0.85, top_k=50, top_p=None),
    dict(name="T0.7/k250/p0.9", temperature=0.7, top_k=250, top_p=0.9),
]


def pick_rows(group, n, seed=0, val_frac=0.1):
    uniq = np.unique(group)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_g = set(uniq[rng.permutation(len(uniq))[:n_val]].tolist())
    va = np.where(np.array([g in val_g for g in group]))[0]
    seen, one = set(), []
    for r in va:
        if int(group[r]) not in seen:
            seen.add(int(group[r]))
            one.append(int(r))
    one = np.array(one)
    return one[np.linspace(0, len(one) - 1, n, dtype=int)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--ckpt", default=str(ROOT / "results" / "gen_eegcond_k8" / "ckpt_ep16.pt"))
    ap.add_argument("--listen-rows", type=int, default=8, help="rows whose wav is kept per recipe")
    ap.add_argument("--only", default="", help="comma list of recipe indices")
    args = ap.parse_args()

    from pilot_multitoken import CondWidthModel
    from finetune_gen_k import load_state
    from musicgen_clap_finetune import load_clap, clap_embed_audio, MusicGenCLAP
    import soundfile as sf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int64)
    group = np.load(DATA / "group.npy").astype(np.int64)
    rows = pick_rows(group, args.n)
    n = len(rows)

    clap, proc, csr = load_clap(device)
    unit = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    base = MusicGenCLAP(device=device, use_lora=False)
    T = unit(np.stack([
        clap_embed_audio(clap, proc, WAV_SR,
                         np.asarray(base.mg.audio_encoder.decode(
                             torch.from_numpy(codes[r]).unsqueeze(0).unsqueeze(0).to(device),
                             audio_scales=[None]).audio_values.cpu()).reshape(-1), device)
        for r in rows]))
    del base
    if device == "cuda":
        torch.cuda.empty_cache()

    model = CondWidthModel(device, k_tokens=args.k)
    load_state(model, args.ckpt)
    model.m.eval()
    print(f"{n} held-out targets; model {Path(args.ckpt).name} k={args.k}")
    print(f"chance lines: top-1 = 1/{n} = {1/n:.3f}   mean rank = {(n+1)/2:.1f}")

    idxs = list(range(len(RECIPES))) if not args.only else [int(x) for x in args.only.split(",")]
    results = []
    for i in idxs:
        rec = RECIPES[i]
        t0 = time.time()
        G, keep = [], []
        for j, r in enumerate(rows):
            w = np.asarray(model.generate(torch.from_numpy(cond[r]),
                                          max_new_tokens=args.tokens,
                                          temperature=rec["temperature"],
                                          top_k=rec["top_k"], top_p=rec["top_p"])).reshape(-1)
            G.append(clap_embed_audio(clap, proc, WAV_SR, w, device))
            if j < args.listen_rows:
                tag = rec["name"].replace("/", "_").replace("(", "").replace(")", "")
                sf.write(str(OUT / f"{i}_{tag}_row{r:05d}.wav"), w, WAV_SR)
                keep.append((r, float(np.sqrt(np.mean(w ** 2))), float(np.abs(w).max())))
        Gm = unit(np.stack(G))
        S = Gm @ T.T
        own = np.array([S[j, j] for j in range(n)])
        others = np.array([np.delete(S[j], j).mean() for j in range(n)])
        d = own - others
        se = float(d.std(ddof=1) / np.sqrt(n))
        ranks = [int(np.where(np.argsort(-S[j]) == j)[0][0]) + 1 for j in range(n)]
        GG = Gm @ Gm.T
        out = dict(idx=i, **rec, margin=float(d.mean()), se=se, t=float(d.mean() / se),
                   top1=sum(1 for r_ in ranks if r_ == 1) / n,
                   mean_rank=float(np.mean(ranks)),
                   own=float(own.mean()), gg=float(GG[np.triu_indices(n, 1)].mean()),
                   rms=[round(k[1], 4) for k in keep],
                   peak=[round(k[2], 3) for k in keep])
        results.append(out)
        clip = float(np.mean(out["peak"]))
        print(f"  [{i}] {rec['name']:<26} margin={out['margin']:+.4f}±{se:.4f} t={out['t']:+.2f} "
              f"top1={out['top1']:.3f} rank={out['mean_rank']:.2f} GEN-v-GEN={out['gg']:.4f} "
              f"峰值均={clip:.2f} [{'削波风险' if clip > 1.0 else '无削波'}] "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del G, Gm, S
        if device == "cuda":
            torch.cuda.empty_cache()

    json.dump(results, open(OUT / "recipes.json", "w"), indent=2)
    print("\n判读: margin 随截断变紧通常会降(更靠音乐先验、更少用条件)——两个数要一起看，"
          "好不好听得你在同样的行号上对比 {i}_<配方>_row<行号>.wav")
    print(f"试听文件在 {OUT}")


if __name__ == "__main__":
    main()
