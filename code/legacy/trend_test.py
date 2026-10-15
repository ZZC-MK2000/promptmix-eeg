"""Settle whether condition specificity declines late in training.

Correct statistic: per-target margin_i = cos(gen_i,true_i) - mean_j!=i cos(gen_i,true_j),
then a PAIRED test of margin_i(A) - margin_i(B) across the same targets. Comparing two
independent margin SEs (as ckpt_sweep did) is far weaker because target difficulty cancels
in the paired difference.

Pool is enlarged to --n targets (default 60), one row per code-group so no two targets
share the same stimulus audio.

Run: python src/train/trend_test.py --n 60 --eps 8,16,23,29
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
RES = ROOT / "results" / "gen_eegcond_k8"
DATA = ROOT / "data" / "eegcond"
OUTDIR = ROOT / "results" / "gen_eegcond" / "trend"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

WAV_SR = 32000


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


def paired(a, b, labels):
    d = a - b
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    wins = int((d > 0).sum())
    print(f"  配对 {labels}: Δmargin={d.mean():+.4f} ± {d.std(ddof=1)/np.sqrt(len(d)):.4f} "
          f"t={t:+.2f}  {labels.split(' vs ')[0]} 更优的行 {wins}/{len(d)}")
    return float(t), float(d.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--eps", default="8,16,23,29")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--reps", type=int, default=1)
    args = ap.parse_args()

    from pilot_multitoken import CondWidthModel
    from finetune_gen_k import load_state
    from musicgen_clap_finetune import load_clap, clap_embed_audio, MusicGenCLAP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUTDIR.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int64)
    group = np.load(DATA / "group.npy").astype(np.int64)
    rows = pick_rows(group, args.n)
    print(f"{len(rows)} held-out targets (one per code-group)")

    # song labels for the song-level chance line
    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    song_all = z["song"].astype(int)
    eeg = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    proj = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu",
                      weights_only=True)
    W = proj["weight"].numpy(); b = proj["bias"].numpy()
    v = eeg @ W.T + b
    orig = (v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)
    j, mapped = 0, []
    for i in range(len(cond)):
        key = cond[i].tobytes()
        while j < len(orig) and orig[j].tobytes() != key:
            j += 1
        mapped.append(j); j += 1
    songs = song_all[np.array(mapped)][rows]
    p = np.bincount(songs, minlength=14) / len(rows)
    chance_song = float((p * p).sum())
    print(f"chance: window {1/len(rows):.3f}  same-song {chance_song:.3f}")

    clap, proc, csr = load_clap(device)
    unit = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    base = MusicGenCLAP(device=device, use_lora=False)
    TRUE = []
    for r in rows:
        w = base.mg.audio_encoder.decode(
            torch.from_numpy(codes[r]).unsqueeze(0).unsqueeze(0).to(device),
            audio_scales=[None]).audio_values
        w = np.asarray(w.cpu()).reshape(-1)
        TRUE.append(w)
        sf.write(str(OUTDIR / f"TRUE_row{r:05d}.wav"), w, WAV_SR)
    Tembed = unit(np.stack([clap_embed_audio(clap, proc, WAV_SR, w, device) for w in TRUE]))
    del base
    if device == "cuda":
        torch.cuda.empty_cache()

    per_ep = {}
    for ep in [int(x) for x in args.eps.split(",")]:
        ck = RES / f"ckpt_ep{ep}.pt"
        if not ck.exists():
            print(f"ep{ep}: missing"); continue
        t0 = time.time()
        model = CondWidthModel(device, k_tokens=args.k)
        load_state(model, str(ck)); model.m.eval()
        acc = []
        for rep in range(args.reps):
            G = []
            for r in rows:
                w = np.asarray(model.generate(torch.from_numpy(cond[r]),
                                              max_new_tokens=args.tokens,
                                              temperature=args.temp)).reshape(-1)
                G.append(clap_embed_audio(clap, proc, WAV_SR, w, device))
                if rep == 0:
                    sf.write(str(OUTDIR / f"gen_ep{ep}_row{r:05d}.wav"), w, WAV_SR)
            acc.append(unit(np.stack(G)))
        Gm = unit(np.mean(np.stack(acc), axis=0)) if args.reps > 1 else acc[0]
        S = Gm @ Tembed.T
        n = len(rows)
        own = np.array([S[i, i] for i in range(n)])
        others = np.array([np.delete(S[i], i).mean() for i in range(n)])
        margins = own - others
        ranks = [int(np.where(np.argsort(-S[i]) == i)[0][0]) + 1 for i in range(n)]
        GG = Gm @ Gm.T
        rec = dict(ep=ep, steps=(ep + 1) * 4027,
                   margin=float(margins.mean()),
                   margin_se=float(margins.std(ddof=1) / np.sqrt(n)),
                   t_spec=float(margins.mean() / (margins.std(ddof=1) / np.sqrt(n))),
                   window_top1=sum(1 for r_ in ranks if r_ == 1) / n,
                   mean_rank=float(np.mean(ranks)),
                   song_top1=sum(1 for i in range(n)
                                 if songs[int(np.argmax(S[i]))] == songs[i]) / n,
                   own=float(own.mean()), others=float(others.mean()),
                   gen_gen=float(GG[np.triu_indices(n, 1)].mean()))
        per_ep[ep] = margins
        print(f"ep{ep:>3} margin={rec['margin']:+.4f}±{rec['margin_se']:.4f} "
              f"t={rec['t_spec']:+.2f} top1={rec['window_top1']:.3f} "
              f"rank={rec['mean_rank']:.2f} song={rec['song_top1']:.3f} "
              f"gg={rec['gen_gen']:.4f} [{time.time()-t0:.0f}s]", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        json.dump({str(k): [float(x) for x in v] for k, v in per_ep.items()},
                  open(OUTDIR / "per_target_margins.json", "w"))

    print("\n===== 逐条配对（同一批目标，消掉目标难度）=====")
    ks = sorted(per_ep)
    for a, b in [(x, ks[-1]) for x in ks[:-1]]:
        paired(per_ep[a], per_ep[b], f"ep{a} vs ep{b}")
    if len(ks) >= 2:
        ms = [float(np.mean(per_ep[k])) for k in ks]
        sl = np.polyfit(ks, ms, 1)[0]
        print(f"\nmargin 随 epoch 的斜率 = {sl:+.5f} /epoch  "
              f"({'单调下滑' if all(b < a for a, b in zip(ms, ms[1:])) else '非单调'})")


if __name__ == "__main__":
    main()
