"""Library-size ablation: does a BIGGER corpus-matched library (K=24 vs K=12) close the coverage
gap that the K=12 library leaves open? Re-measures the coverage radius (convex mixture vs the 60
held-out real windows) with 24 prompts x 6 clips and compares to the K=12 number (0.879).

Reports mixture / best-single-clip / best-prompt-mean / uniform / global-mean, and the top1-top2
margin (label separability). CPU-light except generation (~24*6 clips, a few minutes GPU).

Run: python src/train/coverage_k24.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
LIB = ROOT / "results" / "gen_eegcond" / "library_radius"
OUT = ROOT / "results" / "coverage_k24"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

SR = 32000
# 12 from library_b + 12 new corpus-relevant, distinct genres
PROMPTS24 = [
    "upbeat indian bollywood dance song with dhol and bright synths",
    "melodic indian film ballad with lush strings and bamboo flute",
    "catchy pop song with airy female vocals and glossy production",
    "smooth r&b groove with warm bass and crisp rimshot drums",
    "energetic electronic dance music with a heavy four-on-the-floor kick",
    "mellow acoustic pop with fingerpicked guitar and soft brushes",
    "funky groove with slap bass and punchy brass stabs",
    "latin reggaeton beat with tropical synths and dembow rhythm",
    "dreamy ambient synth pad with slow evolving textures",
    "classic rock with a driving electric guitar riff and live drums",
    "hip hop trap beat with boomy 808 bass and rattling hi-hats",
    "cinematic orchestral action cue with pounding war drums",
    "punjabi bhangra rhythm with energetic dhol and claps",
    "sufi qawwali with harmonium, clapping and soaring vocals",
    "retro synthwave with arpeggiated analog leads",
    "drum and bass breakbeat with a rolling sub bass",
    "bossa nova with nylon guitar and soft brushed drums",
    "afrobeat with interlocking guitars and talking drums",
    "lofi chillhop with jazzy chords and vinyl crackle",
    "epic hybrid trailer score with braams and choir",
    "indie folk with harmonized vocals and banjo",
    "deep house with mellow pads and a steady kick",
    "flamenco guitar with handclaps and cajon",
    "bright k-pop dance with punchy brass stabs",
]
K = len(PROMPTS24)


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def proj_simplex(v):
    u = np.sort(v)[::-1]; css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    return np.maximum(v - (css[rho] - 1) / (rho + 1), 0)


def fit_simplex(L, t, iters=600):
    m = L.shape[0]; eta = 1.0 / (np.linalg.norm(L, 2) ** 2 + 1e-8)
    w = np.full(m, 1.0 / m)
    for _ in range(iters):
        r = L.T @ w - t
        w = proj_simplex(w - eta * (L @ r))
    return w


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    from musicgen_clap_finetune import MusicGenCLAP, load_clap, clap_embed_audio
    from arms_2x2 import embed_text, gen_from_enc

    t0 = time.time()
    base = MusicGenCLAP(device=device, use_lora=False); mg = base.mg; mg.eval()
    clap, proc, csr = load_clap(device)

    clips, owner = [], []
    for k, p in enumerate(PROMPTS24):
        h = embed_text(mg, p, device)
        for j in range(6):
            w = gen_from_enc(mg, h.unsqueeze(0), 400, 0.9, None, seed=555 + 1000 * k + j)
            clips.append(clap_embed_audio(clap, proc, SR, w, device)); owner.append(k)
        if k % 6 == 0:
            print(f"  gen prompt{k}/{K} [{time.time()-t0:.0f}s]", flush=True)
    Lcand = unit(np.stack(clips).astype(np.float32))            # (144,512)
    owner = np.array(owner)
    means = unit(np.stack([Lcand[owner == k].mean(0) for k in range(K)]))
    np.save(OUT / "library_emb.npy", Lcand)

    Tembed = unit(np.load(LIB / "true_emb.npy").astype(np.float32))   # (60,512)
    glob = unit(Lcand.mean(0, keepdims=True))[0]
    uni = unit(means.mean(0, keepdims=True))[0]
    mix, best, bmean = [], [], []
    for t in Tembed:
        w = fit_simplex(Lcand, t)
        mix.append(float(unit(Lcand.T @ w) @ t)); best.append(float((Lcand @ t).max()))
        bmean.append(float((means @ t).max()))
    mix, best, bmean = np.array(mix), np.array(best), np.array(bmean)
    uni_c = float((Tembed @ uni).mean()); glob_c = float((Tembed @ glob).mean())
    print(f"\n=== K=24 coverage (60 real windows) ===")
    print(f"  mixture={mix.mean():.4f}  best_single={best.mean():.4f}  best_prompt_mean={bmean.mean():.4f}")
    print(f"  uniform={uni_c:.4f}  global={glob_c:.4f}")
    print(f"  vs K=12 (library_b): mixture 0.8790, best_single 0.7969, global 0.7300")
    d = mix - best
    print(f"  mixture - best_single = {d.mean():+.4f}  t={d.mean()/(d.std(ddof=1)/np.sqrt(len(d))):+.2f}")
    json.dump({"K": K, "mixture": float(mix.mean()), "best_single": float(best.mean()),
               "best_prompt_mean": float(bmean.mean()), "uniform": uni_c, "global": glob_c,
               "mix_minus_best": float(d.mean())}, open(OUT / "coverage_k24.json", "w"), indent=2)
    print(f"\nsaved -> {OUT/'coverage_k24.json'}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
