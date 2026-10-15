"""Evaluate mixed-condition checkpoints on the same axis we calibrated everything else on.

For each checkpoint, with the SAME fixed sampling recipe (temperature 1.0, top_k 250 -- the
best of the five tested) and the SAME 60 held-out stimulus windows:
  condition faithfulness: margin = cos(gen_i, true_i) minus the mean cos to the other 59,
      averaged over the 60; SE = its standard error = std/sqrt(60); t = margin/SE (|t|>=2
      counts); plus 60-way top-1, mean rank, and the song-level hit rate
  musicality proxy: zero-crossing rate (share of adjacent samples that flip sign; higher =
      buzzier) against the REAL stimulus windows decoded through the same codec, which sits at
      ~0.0556 -- that is the fair bar, not text-prompted soft piano
  output diversity: GEN-v-GEN, mean pairwise cosine among generated clips

ep16 is evaluated in the same pass so the comparison cannot be contaminated by recipe drift.

Run: python src/train/mixed_eval.py --ckpts ep16,4000,8000,12000
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
K8 = ROOT / "results" / "gen_eegcond_k8"
MIX = ROOT / "results" / "gen_eegcond_mixed"
OUT = MIX / "eval"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

SR = 32000
RECIPE = dict(temperature=1.0, top_k=250)


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


def zcr(w):
    return float(np.mean(np.abs(np.diff(np.sign(w))))) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="ep16,4000,8000,12000")
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--device", default="")
    ap.add_argument("--reps", type=int, default=1, help="decodes averaged per target")
    ap.add_argument("--outdir", default="", help="where wav/eval.json go; default MIX/eval")
    args = ap.parse_args()

    from pilot_multitoken import CondWidthModel
    from musicgen_clap_finetune import load_clap, clap_embed_audio, MusicGenCLAP
    import soundfile as sf

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.outdir) if args.outdir else OUT
    out_dir.mkdir(parents=True, exist_ok=True)

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int64)
    group = np.load(DATA / "group.npy").astype(np.int64)
    rows = pick_rows(group, args.n)
    n = len(rows)
    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    song_all = z["song"].astype(int)
    eeg = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    proj = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu",
                      weights_only=True)
    W, b = proj["weight"].numpy(), proj["bias"].numpy()
    o = eeg @ W.T + b
    orig = (o / (np.linalg.norm(o, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)
    j, mapped = 0, []
    for i in range(len(cond)):
        kk = cond[i].tobytes()
        while j < len(orig) and orig[j].tobytes() != kk:
            j += 1
        mapped.append(j); j += 1
    songs = song_all[np.array(mapped)][rows]
    p = np.bincount(songs, minlength=14) / n
    chance_song = float((p * p).sum())

    clap, proc, csr = load_clap(device)
    unit = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    base = MusicGenCLAP(device=device, use_lora=False)
    TRUE = []
    for r in rows:
        oc = base.mg.audio_encoder.decode(
            torch.from_numpy(codes[r]).unsqueeze(0).unsqueeze(0).to(device),
            audio_scales=[None]).audio_values
        TRUE.append(np.asarray(oc.cpu()).reshape(-1))
    Tembed = unit(np.stack([clap_embed_audio(clap, proc, SR, w, device) for w in TRUE]))
    z_true = float(np.mean([zcr(w) for w in TRUE]))
    del base
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"{n} held-out targets; 真音参照过零率={z_true:.4f}（这是公平基准）; "
          f"chance top-1={1/n:.3f} 同歌随机={chance_song:.3f}; 配方 {RECIPE}")

    out = []
    per_target = {}
    per_hit = {}
    sims = {}
    names = []
    for spec in args.ckpts.split(","):
        spec = spec.strip()
        if spec == "ep16":
            path, name = K8 / "ckpt_ep16.pt", "ep16"
        elif spec.endswith(".pt") or "/" in spec:
            path, name = Path(spec), Path(spec).stem
        else:
            path, name = MIX / f"ckpt_s{spec}.pt", spec
        if not path.exists():
            print(f"  {name}: 缺文件，跳过"); continue
        t0 = time.time()
        model = CondWidthModel(device, k_tokens=args.k)
        ck = torch.load(str(path), map_location="cpu", weights_only=True)
        model.m.clap_proj.load_state_dict(ck["clap_proj"])
        model.m.mg.decoder.load_state_dict(ck["decoder_lora"], strict=False)
        model.m.eval()
        acc_emb, zs, pk = [], [], []
        for rep in range(max(1, args.reps)):
            G = []
            for q, r in enumerate(rows):
                w = np.asarray(model.generate(torch.from_numpy(cond[r]),
                                              max_new_tokens=args.tokens, **RECIPE)).reshape(-1)
                G.append(clap_embed_audio(clap, proc, SR, w, device))
                if rep == 0:
                    zs.append(zcr(w)); pk.append(float(np.abs(w).max()))
                    if q < 8:
                        sf.write(str(out_dir / f"gen_{name}_row{r:05d}.wav"), w, SR)
            acc_emb.append(np.stack(G))
        Gm = unit(np.mean(np.stack(acc_emb), axis=0))   # average the re-decodes
        names.append(name)
        S = Gm @ Tembed.T
        own = np.array([S[i, i] for i in range(n)])
        others = np.array([np.delete(S[i], i).mean() for i in range(n)])
        d = own - others
        per_target[name] = d.copy()
        se = float(d.std(ddof=1) / np.sqrt(n))
        ranks = [int(np.where(np.argsort(-S[i]) == i)[0][0]) + 1 for i in range(n)]
        GG = Gm @ Gm.T
        song_hit = np.array([songs[int(np.argmax(S[i]))] == songs[i] for i in range(n)])
        win_hit = np.array([r_ == 1 for r_ in ranks])
        per_hit[name] = (win_hit, song_hit)
        sims[name] = S
        rec = dict(tag=name, steps=int(name) if name.isdigit() else None,
                   margin=float(d.mean()), se=se, t=float(d.mean() / se),
                   top1=sum(r_ == 1 for r_ in ranks) / n,
                   mean_rank=float(np.mean(ranks)),
                   song=float(song_hit.mean()),
                   own=float(own.mean()), others=float(others.mean()),
                   gg=float(GG[np.triu_indices(n, 1)].mean()),
                   zcr=float(np.mean(zs)), zcr_med=float(np.median(zs)),
                   peak=float(np.mean(pk)))
        out.append(rec)
        print(f"  {name:<12} margin={rec['margin']:+.4f}±{se:.4f} t={rec['t']:+.2f} "
              f"top1={rec['top1']:.3f} 秩={rec['mean_rank']:.1f} 同歌={rec['song']:.3f} | "
              f"过零率={rec['zcr']:.4f}(中位{rec['zcr_med']:.4f}) 峰值={rec['peak']:.2f} "
              f"GENvGEN={rec['gg']:.4f} [{time.time()-t0:.0f}s]", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n===== 逐条配对（同一批目标上两两比较 margin 之差）=====")
    ks = [x for x in names if x in per_target]

    def mcnemar_p(b, c):
        # 精确 McNemar：只看两臂判决不一致的那些目标，b/c 之差是否超出 50/50
        from math import comb
        m = b + c
        if m == 0:
            return 1.0
        lo = min(b, c)
        p = sum(comb(m, i) for i in range(lo + 1)) * 0.5 ** m
        return min(1.0, 2 * p)

    pair = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            a, b = per_target[ks[i]], per_target[ks[j]]
            dd = a - b
            tt = float(dd.mean() / (dd.std(ddof=1) / np.sqrt(len(dd))))
            wins = int((dd > 0).sum())
            print(f"  {ks[i]} − {ks[j]}: Δmargin={dd.mean():+.4f} 配对t={tt:+.2f} "
                  f"{ks[i]} 更好的目标 {wins}/{len(dd)}")
            pair.append({"a": ks[i], "b": ks[j], "delta": float(dd.mean()), "t": tt,
                         "wins_a": wins})

    print("\n===== 逐条配对（命中类指标用精确 McNemar，只看两臂判决不一致的目标）=====")
    hits = []
    for i in range(len(ks)):
        for j in range(i + 1, len(ks)):
            row = {"a": ks[i], "b": ks[j]}
            for lbl, idx in (("窗top1", 0), ("同歌", 1)):
                ha, hb = per_hit[ks[i]][idx], per_hit[ks[j]][idx]
                bb = int((ha & ~hb).sum())          # a 命中、b 漏
                cc = int((~ha & hb).sum())          # b 命中、a 漏
                p = mcnemar_p(bb, cc)
                print(f"  {ks[i]} vs {ks[j]} {lbl}: 只有{ks[i]}中 {bb} 条 / 只有{ks[j]}中 {cc} 条 "
                      f"精确 McNemar p={p:.4f}")
                row[lbl] = {"only_a": bb, "only_b": cc, "p": p}
            hits.append(row)

    import json
    json.dump({"true_zcr_reference": z_true, "chance_top1": 1 / n, "chance_song": chance_song,
               "recipe": RECIPE, "rows": [int(x) for x in rows], "results": out,
               "paired": pair, "paired_hits": hits,
               "per_target_margin": {k: list(map(float, v)) for k, v in per_target.items()},
               "per_target_hits": {k: {"win": list(map(bool, v[0])),
                                       "song": list(map(bool, v[1]))} for k, v in per_hit.items()},
               "similarity": {k: [[float(x) for x in r] for r in v] for k, v in sims.items()},
               "reps": args.reps},
              open(out_dir / "eval.json", "w"), indent=2)
    print(f"\n判读: 过零率越接近 {z_true:.4f} 越像真音乐；margin 的 t 不掉到 2 以下说明混合没丢掉脑电贴合。")
    print(f"A/B 音频在 {out_dir}（每个 ckpt 同样 8 个行号）")


if __name__ == "__main__":
    main()
