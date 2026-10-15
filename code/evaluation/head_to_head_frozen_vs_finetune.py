"""Fix weakness #7: a FAIR head-to-head of frozen-prompt-mixing vs the fine-tuned generator, on the
SAME held-out EEG windows, scored by CLAP cosine to the actually-heard audio (semantic relevance)
plus texture (zcr, peak) — same metric, same windows, so the comparison is apples-to-apples.

Arms compared (each delivers audio for some held-out rows; we intersect on shared rows):
  frozen_modeA_true   : deliver_mode_a (EEG->song-matched genre prompt, frozen sampling)
  frozen_modeB_mix    : deliver_mode_b top-3 waveform blend
  frozen_libB         : deliver_library_b (corpus-matched lib)
  finetuned_mixed2    : the LoRA+projection fine-tuned generator (the route we argue against)

Run: python code/evaluation/head_to_head_frozen_vs_finetune.py   (from promptmix-eeg/, GPU)
"""
import glob
import re
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
sys.path.insert(0, str(M2 / "src" / "generate"))
SR = 32000


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def zcr(w):
    return float(np.mean(np.abs(np.diff(np.sign(w))))) / 2


ARMS = {
    "frozen_modeA_true": "results/prompt_classifier_true/listen/gen__true_oracle__row*_p*.wav",
    "frozen_modeB_mix": "results/prompt_classifier_true/listen_B/B-mix__row*.wav",
    "frozen_libB": "results/library_b/listen/genB__row*_p*.wav",
    "finetuned_mixed2": "results/gen_eegcond_mixed/eval/gen_*_row*.wav",
}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from musicgen_clap_finetune import load_clap, clap_embed_audio
    clap, proc, csr = load_clap(device)
    true_emb = unit(np.load(ROOT / "results/prompt_classifier/true_emb_all.npy").astype(np.float32))

    t0 = time.time()
    data = {}
    for name, pat in ARMS.items():
        rows = {}
        for f in sorted(glob.glob(str(ROOT / pat))):
            m = re.search(r"row(\d+)", Path(f).name)
            if not m:
                continue
            r = int(m.group(1))
            if r >= len(true_emb):
                continue
            w, sr = sf.read(f); w = np.asarray(w)
            if w.ndim > 1:
                w = w.mean(1)
            e = unit(clap_embed_audio(clap, proc, SR, w.astype(np.float32), device))
            rows[r] = (float(e @ true_emb[r]), zcr(w), float(np.abs(w).max()))
        data[name] = rows
        print(f"{name:20} n={len(rows)}", flush=True)

    # shared rows across frozen_modeA_true and finetuned_mixed2 (the core comparison)
    core = [a for a in ("frozen_modeA_true", "frozen_modeB_mix", "frozen_libB", "finetuned_mixed2")
            if data.get(a)]
    print(f"\n=== per-arm on their own rows ===")
    print(f"{'arm':22} {'n':>4} {'cos->heard':>11} {'zcr':>7} {'peak':>6}")
    for a in core:
        v = list(data[a].values())
        if not v:
            continue
        cos = np.array([x[0] for x in v]); z = np.array([x[1] for x in v]); pk = np.array([x[2] for x in v])
        print(f"{a:22} {len(v):>4} {cos.mean():>11.4f} {z.mean():>7.4f} {pk.mean():>6.3f}")

    # strict head-to-head: rows present in BOTH a frozen arm and finetuned
    ft = data.get("finetuned_mixed2", {})
    for fa in ("frozen_modeA_true", "frozen_libB"):
        shared = sorted(set(ft) & set(data.get(fa, {})))
        if len(shared) >= 3:
            fc = np.mean([data[fa][r][0] for r in shared]); ftc = np.mean([ft[r][0] for r in shared])
            fz = np.mean([data[fa][r][1] for r in shared]); ftz = np.mean([ft[r][1] for r in shared])
            print(f"\n[same {len(shared)} windows] {fa} vs finetuned_mixed2:")
            print(f"   cos->heard: {fc:.4f} vs {ftc:.4f}   zcr: {fz:.4f} vs {ftz:.4f}")
            print(f"   -> frozen {'HIGHER' if fc>ftc else 'lower'} semantic relevance; "
                  f"texture {'cleaner' if fz<ftz else 'rougher'} (zcr, genre-caveat applies)")
    import json
    json.dump({a: {"n": len(v), "rows": {str(k): list(map(float, val)) for k, val in v.items()}}
               for a, v in data.items()},
              open(ROOT / "results/head_to_head.json", "w"), indent=2)
    print(f"\nsaved -> results/head_to_head.json  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
