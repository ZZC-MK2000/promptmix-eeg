"""Objective musicality-adjacent metric that is NEITHER zcr (confounded by bright genres) NOR CLAP
cos (inflated by laion clustering): spectral flatness. Low flatness => tonal/harmonic (music-like);
high flatness => noise-like. Compare groups on the SAME scale:
  real heard windows  vs  delivered native clips (mode A / B-cat)  vs  B-mix blends  vs  known
  off-manifold interpolation clips (should be noise-like).
If delivered native clips sit near real music and far from the interpolation-noise group, that is
objective evidence the frozen-prompt delivery is musical, independent of human listening.

Run: python src/train/musicality_metric.py
"""
import glob
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
SR = 32000


def flatness(path):
    import librosa
    w, sr = sf.read(path)
    w = np.asarray(w)
    if w.ndim > 1:
        w = w.mean(1)
    if sr != SR:
        w = librosa.resample(w, orig_sr=sr, target_sr=SR)
    sf_ = librosa.feature.spectral_flatness(y=w, n_fft=2048)[0]
    return float(np.median(sf_))


def collect(pat, cap=40):
    fs = sorted(glob.glob(str(ROOT / pat)))[:cap]
    vals = []
    for f in fs:
        try:
            vals.append(flatness(f))
        except Exception:
            pass
    return np.array(vals)


def main():
    groups = {
        "real heard windows": "results/library_b/listen/ref__*.wav",
        "delivered mode A (native)": "results/library_b/listen/genB__*.wav",
        "delivered B-cat (native)": "results/prompt_classifier_true/listen_B/B-cat__*.wav",
        "delivered B-mix (blend)": "results/prompt_classifier_true/listen_B/B-mix__*.wav",
        "interpolation OFF-manifold (noise)": "results/gen_eegcond/library_radius/sweep_p*_a0[25]*.wav",
    }
    print("spectral flatness (lower = more tonal/music-like; higher = noise-like)")
    print("=" * 66)
    res = {}
    for name, pat in groups.items():
        v = collect(pat)
        if len(v) == 0:
            print(f"  {name:38} (no files)")
            continue
        res[name] = {"median": float(np.median(v)), "mean": float(v.mean()), "n": int(len(v))}
        print(f"  {name:38} median={np.median(v):.4f} mean={v.mean():.4f} n={len(v)}")
    # verdict
    real = res.get("real heard windows", {}).get("median")
    a = res.get("delivered mode A (native)", {}).get("median")
    bc = res.get("delivered B-cat (native)", {}).get("median")
    noise = res.get("interpolation OFF-manifold (noise)", {}).get("median")
    print("\n" + "=" * 66)
    if None not in (real, a, bc, noise):
        print(f"  delivered (A {a:.4f}, B-cat {bc:.4f}) vs real {real:.4f} vs noise {noise:.4f}")
        near_real = min(abs(a - real), abs(bc - real))
        near_noise = min(abs(a - noise), abs(bc - noise))
        if near_real < near_noise:
            print("  -> delivered clips are CLOSER to real music than to noise on this zcr/CLAP-independent metric.")
        else:
            print("  -> delivered clips are closer to noise; investigate.")
    import json
    json.dump(res, open(ROOT / "results/library_b/musicality_flatness.json", "w"), indent=2)
    print(f"\nsaved -> results/library_b/musicality_flatness.json")


if __name__ == "__main__":
    main()
