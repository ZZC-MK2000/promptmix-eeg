"""Learned-metric musicality comparison (the gap left by zcr/spectral-flatness both failing).

In laion-CLAP embedding space (PCA-reduced to 32-d for stable covariance), measure how far each
generation approach sits from the REAL-MUSIC distribution (the 36036 decoded true windows):
  - frozen native prompt samples (library_b clips)      -> the PromptMix-EEG delivery
  - fine-tuned generator output (mixed2 eval wavs)       -> LoRA-on-MusicGen (the route we avoid)
  - T5-embedding interpolation clips (library_radius sweep) -> off-manifold condition
Report Frechet distance to real + mean cosine to the real centroid. Lower FD / higher cos = closer
to real music. Tests the paper's central claim: freezing keeps output in the real-music manifold
better than fine-tuning or interpolation.

Run: python src/train/learned_musicality.py
"""
import glob
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
sys.path.insert(0, str(M2 / "src" / "generate"))
SR = 32000


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def frechet(a, b):
    # a,b: (n,d) already PCA-reduced; Gaussian FD = ||ma-mb||^2 + Tr(Ca+Cb-2(CaCb)^.5)
    ma, mb = a.mean(0), b.mean(0)
    Ca, Cb = np.cov(a, rowvar=False), np.cov(b, rowvar=False)
    diff = ma - mb
    # matrix sqrt via eigen of CaCb (symmetrized product)
    ev = np.linalg.eigvals(Ca @ Cb)
    ev = np.clip(ev.real, 0, None)
    tr = np.trace(Ca) + np.trace(Cb) - 2 * ev.sum()
    return float(diff @ diff + max(tr, 0))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from musicgen_clap_finetune import load_clap, clap_embed_audio
    import soundfile as sf

    real = np.load(ROOT / "results/prompt_classifier/true_emb_all.npy").astype(np.float32)  # (36036,512)
    frozen = np.load(ROOT / "results/library_b/library_emb.npy").astype(np.float32)          # (72,512) native clips

    def embed_wavs(paths):
        out = []
        for p in paths:
            w, sr = sf.read(p); w = np.asarray(w)
            if w.ndim > 1: w = w.mean(1)
            out.append(clap_embed_audio(clap, proc, SR, w.astype(np.float32), device))
        return np.stack(out).astype(np.float32)

    clap, proc, csr = load_clap(device)
    ft_paths = sorted(glob.glob(str(ROOT / "results/gen_eegcond_mixed2/eval/gen_*.wav")))[:60]
    if not ft_paths:
        ft_paths = sorted(glob.glob(str(ROOT / "results/gen_eegcond_mixed/eval/gen_*.wav")))[:60]
    ip_paths = sorted(glob.glob(str(ROOT / "results/gen_eegcond/library_radius/sweep_*.wav")))
    finetuned = embed_wavs(ft_paths)
    interp = embed_wavs(ip_paths)
    print(f"n: real={len(real)} frozen_native={len(frozen)} finetuned={len(finetuned)} interp={len(interp)}",
          flush=True)

    # PCA to 32-d fit on real
    from sklearn.decomposition import PCA
    pca = PCA(n_components=32).fit(unit(real))
    R = pca.transform(unit(real))
    groups = {"frozen_native": frozen, "finetuned(mixed2)": finetuned, "interpolation": interp}
    print("\n=== distance to REAL-MUSIC distribution (laion-CLAP, PCA-32) ===")
    print(f"{'approach':22} {'Frechet->real':>14} {'mean-cos->real-centroid':>24}")
    rc = unit(real.mean(0))
    res = {}
    for name, E in groups.items():
        Z = pca.transform(unit(E))
        fd = frechet(Z, R)
        mc = float(np.mean(unit(E) @ rc))
        res[name] = {"frechet_to_real": fd, "mean_cos_to_real_centroid": mc}
        print(f"{name:22} {fd:>14.2f} {mc:>24.4f}")
    # also real-vs-real split as FD floor (two halves of real)
    h1, h2 = R[:len(R)//2], R[len(R)//2:]
    floor = frechet(h1, h2)
    print(f"\nFD floor (real-vs-real halves) = {floor:.2f}  (any group near this = indistinguishable from real)")
    best = min(res, key=lambda k: res[k]["frechet_to_real"])
    print(f"\nclosest to real music: {best}")
    import json
    json.dump({"floor_real_vs_real": floor, "groups": res,
               "note": "learned CLAP-space metric; complements the failed zcr/spectral-flatness proxies"},
              open(ROOT / "results/library_b/learned_musicality.json", "w"), indent=2)
    print("saved -> results/library_b/learned_musicality.json")


if __name__ == "__main__":
    main()
