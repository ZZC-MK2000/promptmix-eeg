"""#2: does a BIGGER library improve coverage? K=36 corpus-relevant prompts, 3 clips each. Reports:
  - coverage radius at 12 / 24 / 36 prompts (convex mixture vs the 60 held-out real windows)
  - GREEDY best-12-of-36 subset by coverage (facility-location style): directly attacks the
    "the library leaves part of the real cloud unreached" problem by choosing prompts that spread.
  - saves the 36-prompt clip embeddings for later reuse.

Note: this run uses 3 clips per prompt, so it is NOT on the same axis as the 6-clips-per-prompt
numbers; code/radii/coverage_unified.py keeps them apart.

Run: python code/radii/coverage_k48.py   (from promptmix-eeg/, GPU)
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
OUT = ROOT / "results" / "coverage_k48"
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(M2 / "src" / "generate"))
SR = 32000

PROMPTS = [
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
    "bollywood item number with heavy dhol and synth brass",
    "ghazal with soft strings and expressive male vocals",
    "south indian film music with fast percussion and flute",
    "devotional bhajan with harmonium and tabla",
    "retro 90s bollywood dance with orchestral hits",
    "punjabi folk with dhol and energetic group vocals",
    "trance build with pulsing synth bass and arpeggios",
    "future bass with supersaw chords and vocal chops",
    "synthpop with gated reverb drums and bright leads",
    "indie rock with jangly guitars and steady drums",
    "heavy metal with palm-muted riffs and double kick drums",
    "blues rock with slide guitar and a shuffle groove",
    "soul ballad with warm horns and organ",
    "gospel choir with clapping and Hammond organ",
    "latin salsa with brass mambo and congas",
    "caribbean dancehall with heavy bass and skank guitar",
    "middle eastern darbuka groove with oud",
    "japanese koto and shamisen traditional melody",
    "celtic folk with fiddle and bodhran",
    "cinematic piano and strings emotional theme",
    "epic orchestral choir battle theme",
    "lofi study beat with mellow Rhodes and tape hiss",
    "neo-soul with jazzy guitar and dusty drums",
    "ambient drone with evolving pads and no beat",
]
K = len(PROMPTS)
CLIPS = 3


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def proj_simplex(v):
    u = np.sort(v)[::-1]; css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    return np.maximum(v - (css[rho] - 1) / (rho + 1), 0)


def fit_simplex(L, t, iters=400):
    m = L.shape[0]; eta = 1.0 / (np.linalg.norm(L, 2) ** 2 + 1e-8); w = np.full(m, 1.0 / m)
    for _ in range(iters):
        r = L.T @ w - t; w = proj_simplex(w - eta * (L @ r))
    return w


def coverage(L, T):
    return float(np.mean([unit(L.T @ fit_simplex(L, t)) @ t for t in T]))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    from musicgen_clap_finetune import MusicGenCLAP, load_clap, clap_embed_audio
    from arms_2x2 import embed_text, gen_from_enc
    t0 = time.time()
    base = MusicGenCLAP(device=device, use_lora=False); mg = base.mg; mg.eval()
    clap, proc, csr = load_clap(device)
    clips, owner = [], []
    for k, p in enumerate(PROMPTS):
        h = embed_text(mg, p, device)
        for j in range(CLIPS):
            w = gen_from_enc(mg, h.unsqueeze(0), 400, 0.9, None, seed=321 + 1000 * k + j)
            clips.append(clap_embed_audio(clap, proc, SR, w, device)); owner.append(k)
        if k % 8 == 0:
            print(f"  gen {k}/{K} [{time.time()-t0:.0f}s]", flush=True)
    L = unit(np.stack(clips).astype(np.float32)); owner = np.array(owner)
    np.save(OUT / "library_emb.npy", L); np.save(OUT / "library_owner.npy", owner)
    T = unit(np.load(LIB / "true_emb.npy").astype(np.float32))

    def sub(kk):
        mask = owner < kk
        return L[mask]
    c12, c24, c48 = coverage(sub(12), T), coverage(sub(24), T), coverage(L, T)
    print(f"\ncoverage by K: 12={c12:.4f}  24={c24:.4f}  48={c48:.4f}", flush=True)

    # greedy best-12-of-48 by coverage (facility location on prompt groups)
    chosen = []
    remaining = list(range(K))
    for _ in range(12):
        best, bestc = None, -1
        for kk in remaining:
            sel = chosen + [kk]
            mask = np.isin(owner, sel); c = coverage(L[mask], T)
            if c > bestc:
                bestc, best = c, kk
        chosen.append(best); remaining.remove(best)
    gcov = coverage(L[np.isin(owner, chosen)], T)
    print(f"greedy best-12-of-48: prompts {sorted(chosen)} coverage={gcov:.4f}", flush=True)
    json.dump({"K": K, "cov12": c12, "cov24": c24, "cov48": c48,
               "greedy12": sorted(chosen), "greedy12_cov": gcov, "prompts": PROMPTS},
              open(OUT / "coverage_k48.json", "w"), indent=2)
    print(f"saved -> {OUT/'coverage_k48.json'}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
