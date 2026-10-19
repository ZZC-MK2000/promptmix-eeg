"""Coverage radius of an EXPANDED frozen prompt library, measured on generated audio.

The text-only pre-gate (code/radii/prompt_expansion_probe.py) said adding 10 genuinely new genres
from MUSIN-G beats adding 10 paraphrases of the existing prompts by 4x. This is the real
measurement, on generated audio, so the numbers are directly comparable to the published curve
(existing library 0.879, K=24 0.878, K=36 0.890 from coverage_k24/coverage_k48).

Recipe is copied exactly from library_b_experiment.py: 6 clips per prompt, max_new_tokens=400,
temperature=0.9, seed=1234+1000*k+j, embedded with laion-CLAP at 32 kHz. Because the seeds are
pinned, the existing 12 prompts must reproduce results/library_b/library_emb.npy bit-for-bit --
that is asserted before anything is trusted, so a silent recipe drift cannot contaminate the
comparison.

Measured on the SAME 60 held-out real windows library_b used (comparable), and repeated on 2000
windows (stability). Arms:
  existing 12 prompts
  existing 12 + 10 new genres (22)
  greedy 12-of-22 (facility location on the real windows)

Run: python code/radii/coverage_expanded.py        (GPU)
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
B12 = ROOT / "results" / "library_b"
OUT = ROOT / "results" / "coverage_expanded"
for _s in ("shared", "radii", "classifier", "delivery", "evaluation", "alignment", "figures", "legacy"):
    sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

SR = 32000
CLIPS_PER_PROMPT = 6
PROMPTS_OLD = json.load(open(B12 / "summary.json", encoding="utf-8"))["prompts"]
PROMPTS_NEW = [
    "hindustani classical raga with sitar and tabla, slow alapa improvisation",
    "indian folk song with dholak and harmonium and group vocals",
    "indian semi-classical thumri with expressive female vocals and light tabla",
    "goth rock with dark detuned guitar and heavy reverb on the drums",
    "progressive instrumental rock in odd time signatures with dual guitars",
    "deep house groove with muffled kick, sub bass and warm synth stabs",
    "new age instrumental with soft pads and gentle piano",
    "soft jazz trio with brushed drums upright bass and mellow piano",
    "indie rock with jangly clean guitars and loose live drums",
    "electronic track with glitching synths, textured pads and no drums",
]


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


def coverage(T, lib_emb, owner, k):
    """Same basis as library_b_experiment: the simplex is fitted over the INDIVIDUAL CLIPS, not
    over the prompt means. Fitting over the means is a strictly weaker library and does not
    reproduce the published 0.8790, which is how this got caught."""
    means = unit(np.stack([lib_emb[owner == j].mean(0) for j in range(k)]))
    cos = T @ means.T
    mix, best, bmean = [], [], []
    for t in T:
        w = fit_simplex(lib_emb, t)
        mix.append(float(unit(lib_emb.T @ w) @ t))
        best.append(float((lib_emb @ t).max()))
        bmean.append(float((means @ t).max()))
    out = {}
    out["convex_mixture"] = float(np.mean(mix))
    out["best_single_clip"] = float(np.mean(best))
    out["best_prompt_mean"] = float(np.mean(bmean))
    out["uniform_over_prompts"] = float((T @ unit(means.sum(0))).mean())
    out["trivial_library_mean"] = float((T @ unit(lib_emb.mean(0))).mean())
    out["trivial_target_mean"] = float((T @ unit(T.mean(0))).mean())
    out["oracle_prompt_dist"] = np.bincount(cos.argmax(1), minlength=k).tolist()
    return out


def greedy_subset(T, lib_emb, owner, k, want):
    """facility location: add the prompt whose clip set raises the running max-cos most."""
    per = [unit(lib_emb[owner == j].mean(0)) for j in range(k)]
    chosen, cur = [], np.full(len(T), -2.0)
    for _ in range(want):
        best, bj = None, -1
        for j in range(k):
            if j in chosen:
                continue
            g = np.maximum(cur, T @ per[j])
            v = float(g.mean())
            if best is None or v > best:
                best, bj = v, j
        chosen.append(bj); cur = np.maximum(cur, T @ per[bj])
    return chosen


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "log.txt", "a", encoding="utf-8")

    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from musicgen_clap_finetune import MusicGenCLAP, load_clap, clap_embed_audio
    from arms_2x2 import embed_text, gen_from_enc

    prompts = PROMPTS_OLD + PROMPTS_NEW
    K = len(prompts)
    cache = OUT / "library_emb.npy"
    if cache.exists():                      # the 132 generations are expensive; reuse them
        lib = unit(np.load(cache).astype(np.float32))
        owner = np.load(OUT / "library_owner.npy").astype(int)
        say(f"loaded {lib.shape[0]} cached clip embeddings ({lib.shape[1]}-d), no generation")
    else:
        base = MusicGenCLAP(device=device, use_lora=False); mg = base.mg; mg.eval()
        clap, proc, csr = load_clap(device)
        emb = []
        for k, p in enumerate(prompts):
            h = embed_text(mg, p, device)
            for j in range(CLIPS_PER_PROMPT):
                w = gen_from_enc(mg, h.unsqueeze(0), 400, 0.9, None, seed=1234 + 1000 * k + j)
                w = np.asarray(w, np.float32)
                if w.ndim > 1:
                    w = w.reshape(-1)
                emb.append(np.asarray(clap_embed_audio(clap, proc, SR, w, device)).reshape(-1).astype(np.float32))
            say(f"  prompt{k:02d} '{p[:38]}' [{time.time()-t0:.0f}s]")
        lib = unit(np.stack(emb).astype(np.float32))
        owner = np.repeat(np.arange(K), CLIPS_PER_PROMPT)
        np.save(OUT / "library_emb.npy", lib); np.save(OUT / "library_owner.npy", owner)
        (OUT / "prompts.json").write_text(json.dumps(prompts, indent=2), encoding="utf-8")
    assert lib.shape[0] == K * CLIPS_PER_PROMPT, f"{lib.shape[0]} clips for {K} prompts"

    # self-check: the first 72 must equal the stored library_b embeddings (same seeds, same recipe)
    old = unit(np.load(B12 / "library_emb.npy").astype(np.float32))
    d = np.abs(lib[:72] - old).max()
    cos_old = float(np.diag(lib[:72] @ old.T).mean())
    say(f"reproduction check vs results/library_b: max|Δ|={d:.2e} mean cos={cos_old:.6f}")
    assert d < 1e-4, f"recipe drift: regenerated library differs from library_b (max|Δ|={d:.2e})"

    T60 = unit(np.load(LIB / "true_emb.npy").astype(np.float32))
    Tall = unit(np.load(ROOT / "results" / "prompt_classifier" / "true_emb_all.npy").astype(np.float32))[::18][:2000]
    say(f"targets: {T60.shape[0]} held-out windows (comparable to fig1) and {Tall.shape[0]} windows (stability)")

    res = {"clips_per_prompt": CLIPS_PER_PROMPT, "reproduction_max_abs_delta": float(d),
           "windows": {"n60": int(T60.shape[0]), "nwide": int(Tall.shape[0])}, "arms": {}}
    for name, k in [("existing_12_prompts", 12), ("existing_plus_10_new_genres_22", K)]:
        # clip indices, not prompt indices: each prompt owns CLIPS_PER_PROMPT consecutive rows
        sel = np.arange(k * CLIPS_PER_PROMPT)
        sub = lib[sel]; own = np.repeat(np.arange(k), CLIPS_PER_PROMPT)
        res["arms"][name] = {"n_prompts": k,
                             "on60": coverage(T60, sub, own, k),
                             "on2000": coverage(Tall, sub, own, k)}
        r = res["arms"][name]
        say(f"  {name:32} convex {r['on60']['convex_mixture']:.4f} / "
            f"best-single-clip {r['on60']['best_single_clip']:.4f} / "
            f"trivial(target mean) {r['on60']['trivial_target_mean']:.4f}   "
            f"(n=2000: {r['on2000']['convex_mixture']:.4f})")

    pick = greedy_subset(T60, lib, owner, K, 12)
    say(f"  greedy 12-of-22 selected prompt indices: {sorted(pick)}")
    sel = np.concatenate([np.where(owner == j)[0] for j in pick])
    res["arms"]["greedy_12_of_22"] = {"picked": pick,
                                      "on60": coverage(T60, lib[sel], np.repeat(np.arange(12), CLIPS_PER_PROMPT), 12),
                                      "on2000": coverage(Tall, lib[sel], np.repeat(np.arange(12), CLIPS_PER_PROMPT), 12)}
    say(f"  greedy_12_of_22                convex {res['arms']['greedy_12_of_22']['on60']['convex_mixture']:.4f}")

    a = res["arms"]["existing_12_prompts"]["on60"]["convex_mixture"]
    b = res["arms"]["existing_plus_10_new_genres_22"]["on60"]["convex_mixture"]
    c = res["arms"]["greedy_12_of_22"]["on60"]["convex_mixture"]
    res["summary"] = {"existing12": a, "expanded22": b, "greedy12of22": c,
                      "gain_22_over_12": b - a, "gain_greedy12_over_12": c - a}
    say(f"SUMMARY convex mixture: existing 12 = {a:.4f} | 22 prompts = {b:.4f} "
        f"(+{b-a:.4f}) | greedy 12-of-22 = {c:.4f} (+{c-a:.4f})")
    json.dump(res, open(OUT / "coverage_expanded.json", "w", encoding="utf-8"), indent=2)
    say(f"saved -> {OUT/'coverage_expanded.json'}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
