"""Tolerance-radius probe for the "frozen generator + adapter over a prompt library" route.

No EEG here and no weight changes: the frozen MusicGen is only ever driven by its own native
text prompts. This measures the two numbers that decide whether that route is worth building:

  1) 覆盖半径 (library span): how close can a MIXTURE over the library's own generated clips get
     to a real held-out song window in CLAP space? Compared against the best single prompt and
     against trivial references (uniform over prompts, global mean). If the mixture is no better
     than the best single prompt, "mixture weights" buys nothing and the deliverable collapses to
     "classify the EEG into one of K prompts".
  2) 容错半径 (tolerance radius): interpolate between two native prompt embeddings at alpha and
     watch the texture (zero-crossing rate). The frozen model's own output sits at ~0.02-0.05; the
     alpha at which it blows past the real-music range is how far off-manifold a condition can go.

Both are measured with the SAME recipe as arms_2x2 (temperature 0.9, no top-k, 400 tokens) so the
numbers are comparable to the 0.0319 (frozen) / 0.0797 (fine-tuned) reference points.

Run: python src/train/library_radius.py
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
OUT = ROOT / "results" / "gen_eegcond" / "library_radius"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

SR = 32000
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
PAIRS = [(0, 4), (2, 6), (8, 10)]      # piano<->dance, ambient<->orchestral, jazz<->choir
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def zcr(w):
    return float(np.mean(np.abs(np.diff(np.sign(w))))) / 2


def proj_simplex(v):
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    return np.maximum(v - (css[rho] - 1) / (rho + 1), 0)


def fit_simplex(L, t, iters=600):
    """min ||L.T w - t||^2 s.t. w >= 0, sum w = 1 (projected gradient).

    L is (n_candidates, d): w lives on the candidates, the mixture is L.T @ w.
    """
    d, m = L.shape[1], L.shape[0]
    eta = 1.0 / (np.linalg.norm(L, 2) ** 2 + 1e-8)
    w = np.full(m, 1.0 / m)
    for _ in range(iters):
        r = L.T @ w - t
        w = proj_simplex(w - eta * (L @ r))
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=int, default=6, help="clips generated per prompt")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--targets", type=int, default=60)
    ap.add_argument("--sweep-reps", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from musicgen_clap_finetune import load_clap, clap_embed_audio, MusicGenCLAP
    from mixed_eval import pick_rows
    from arms_2x2 import embed_text, gen_from_enc
    import soundfile as sf

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    top_k = args.top_k or None
    t0 = time.time()

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int64)
    group = np.load(DATA / "group.npy").astype(np.int64)
    rows = pick_rows(group, args.targets)
    n = len(rows)

    clap, proc, csr = load_clap(device)
    base = MusicGenCLAP(device=device, use_lora=False)      # untouched decoder
    mg = base.mg
    mg.eval()

    TRUE, z_true = [], []
    for r in rows:
        oc = mg.audio_encoder.decode(torch.from_numpy(codes[r]).unsqueeze(0).unsqueeze(0).to(device),
                                    audio_scales=[None]).audio_values
        w = np.asarray(oc.cpu()).reshape(-1)
        TRUE.append(w); z_true.append(zcr(w))
    Tembed = unit(np.stack([clap_embed_audio(clap, proc, SR, w, device) for w in TRUE]))
    z_true_mean = float(np.mean(z_true))
    print(f"{n} 条留出真音窗（过码本解码回来）平均过零率={z_true_mean:.4f}；"
          f"库={len(PROMPTS)} 条提示 × {args.clips} 条；采样 temperature={args.temperature} "
          f"top_k={top_k} tokens={args.tokens}", flush=True)

    lib_emb, lib_z, lib_owner = [], [], []
    for k, prompt in enumerate(PROMPTS):
        h = embed_text(mg, prompt, device)
        for j in range(args.clips):
            w = gen_from_enc(mg, h.unsqueeze(0), args.tokens, args.temperature, top_k,
                             seed=args.seed + 1000 * k + j)
            lib_emb.append(clap_embed_audio(clap, proc, SR, w, device))
            lib_z.append(zcr(w)); lib_owner.append(k)
            if j < 2:
                sf.write(str(OUT / f"lib_prompt{k}_clip{j}.wav"), w, SR)
            print(f"  库 prompt{k} clip{j} 过零率={lib_z[-1]:.4f} 峰值={float(np.abs(w).max()):.2f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        print(f"  库 prompt{k} 平均过零率={float(np.mean(lib_z[-args.clips:])):.4f}", flush=True)

    Lcand = unit(np.stack(lib_emb))                        # (K*C, 512)
    owner = np.array(lib_owner)
    means = unit(np.stack([Lcand[owner == k].mean(0) for k in range(len(PROMPTS))]))
    np.save(OUT / "library_emb.npy", Lcand)
    np.save(OUT / "library_owner.npy", owner)
    np.save(OUT / "true_emb.npy", Tembed)
    print(f"库矩阵已落盘 {Lcand.shape}（库平均过零率={float(np.mean(lib_z)):.4f}）", flush=True)

    def cos_to(a, b):
        return float(a @ b)

    mix_c, best_c, bestmean_c, uni_c, glob_c = [], [], [], [], []
    glob = unit(Lcand.mean(0, keepdims=True))[0]
    uni = unit(means.mean(0, keepdims=True))[0]
    for i in range(n):
        t = Tembed[i]
        w = fit_simplex(Lcand, t)
        mix_c.append(cos_to(unit(Lcand.T @ w), t))
        best_c.append(float((Lcand @ t).max()))
        bestmean_c.append(float((means @ t).max()))
        uni_c.append(cos_to(uni, t))
        glob_c.append(cos_to(glob, t))

    def summ(name, v):
        v = np.array(v)
        se = float(v.std(ddof=1) / np.sqrt(len(v)))
        print(f"  {name:<28} 平均余弦={v.mean():.4f}±{se:.4f}", flush=True)
        return {"mean": float(v.mean()), "se": se, "per_target": list(map(float, v))}

    print("\n===== 覆盖半径：库能贴真歌窗多近（CLAP 余弦，越高越像）=====")
    rec = {
        "mixture_over_library": summ("凸组合（学习权重）", mix_c),
        "best_single_clip": summ("最好的单条库片段", best_c),
        "best_single_prompt_mean": summ("最好的单条提示均值", bestmean_c),
        "uniform_over_prompts": summ("12 条提示等权（平凡参照）", uni_c),
        "global_mean": summ("全库均值（最平凡参照）", glob_c),
        "zcr_truth": z_true_mean,
        "zcr_library": float(np.mean(lib_z)),
    }
    d = np.array(mix_c) - np.array(best_c)
    se = float(d.std(ddof=1) / np.sqrt(n))
    print(f"\n  凸组合 − 最好单片段：Δ余弦={d.mean():+.4f}±{se:.4f} t={d.mean()/se:+.2f} "
          f"（同一批目标逐条配对）")
    rec["delta_mix_vs_best"] = {"delta": float(d.mean()), "se": se, "t": float(d.mean() / se),
                                "wins": int((d > 0).sum()), "n": n}

    print("\n===== 容错半径：两条原生提示的嵌入按 α 插值，纹理何时崩 =====")
    sweep = []
    for (ka, kb) in PAIRS:
        ha, hb = embed_text(mg, PROMPTS[ka], device), embed_text(mg, PROMPTS[kb], device)
        Lp = min(ha.shape[0], hb.shape[0])
        for al in ALPHAS:
            h = al * hb[:Lp] + (1 - al) * ha[:Lp]
            zs, embs = [], []
            for j in range(args.sweep_reps):
                w = gen_from_enc(mg, h.unsqueeze(0), args.tokens, args.temperature, top_k,
                                 seed=args.seed + j)
                zs.append(zcr(w)); embs.append(clap_embed_audio(clap, proc, SR, w, device))
                if j == 0:
                    sf.write(str(OUT / f"sweep_p{ka}_{kb}_a{int(al*100):03d}.wav"), w, SR)
            e = unit(np.stack(embs)).mean(0)
            e = e / (np.linalg.norm(e) + 1e-8)
            sweep.append({"pair": [ka, kb], "alpha": al, "zcr": float(np.mean(zs)),
                          "cos_to_prompt_mean": float(max(e @ means[ka], e @ means[kb]))})
            print(f"  提示{ka}↔{kb} α={al:.2f} 过零率={np.mean(zs):.4f} "
                  f"（真音 {z_true_mean:.4f}；冻结原生提示约 0.02-0.05）", flush=True)
        del ha, hb
    rec["sweep"] = sweep
    rec["prompts"] = PROMPTS
    rec["rows"] = [int(x) for x in rows]
    rec["recipe"] = {"temperature": args.temperature, "top_k": top_k, "tokens": args.tokens}
    json.dump(rec, open(OUT / "radius.json", "w"), indent=2)
    np.save(OUT / "library_emb.npy", Lcand)
    np.save(OUT / "library_owner.npy", owner)
    np.save(OUT / "true_emb.npy", Tembed)
    print(f"\ndone -> {OUT}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
