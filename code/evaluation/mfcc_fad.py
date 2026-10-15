"""Texture-aware musicality metric (fixes the gap where CLAP-FD tied frozen & fine-tuned).

VGGish weights aren't available offline, so we use a FAD-style Fréchet distance over MFCC features
(20-dim mean + covariance per clip set) -- MFCCs capture timbre/texture, the axis CLAP (semantic)
missed. Compare each generation approach to the REAL-heard-audio distribution:
  real            : decoded true windows (reference)
  frozen_native   : mode-A corpus-matched prompt samples (the delivery)
  finetuned       : mixed/eval LoRA outputs (the route we argue against)
  interpolation   : off-manifold T5-embedding clips (known-bad)
Lower FD = timbral statistics closer to real music. n reported per arm (small-n inflates FD).

Run: python code/evaluation/mfcc_fad.py   (from promptmix-eeg/, GPU for decode/generate)
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
DATA = Path("D:/Projects/CLaBraM/promptmix-eeg/data/eegcond")
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "radii"))
sys.path.insert(0, str(ROOT / "code" / "classifier"))
sys.path.insert(0, str(M2 / "src" / "generate"))
SR = 32000


def mfccs(w):
    import librosa
    w = np.asarray(w, dtype=np.float32)
    if w.ndim > 1:
        w = w.mean(1)
    m = librosa.feature.mfcc(y=w, sr=SR, n_mfcc=20)     # (20, frames)
    return m.mean(1), np.cov(m)                          # per-clip mean vector + cov


def frechet(m1, c1, m2, c2):
    diff = m1 - m2
    covmean, _ = _sqrtm(c1 @ c2)
    return float(diff @ diff + np.trace(c1) + np.trace(c2) - 2 * np.real(np.trace(covmean)))


def _sqrtm(m):
    ev, V = np.linalg.eig(m)
    sq = np.real(V @ np.diag(np.sqrt(np.clip(np.real(ev), 0, None))) @ np.linalg.inv(V))
    return sq, 0


def stack_feats(wavs):
    M = np.stack([mfccs(w)[0] for w in wavs])
    C = np.mean([mfccs(w)[1] for w in wavs], axis=0)     # pooled within-clip cov
    C += 1e-3 * np.eye(M.shape[1])                        # shrinkage for small n
    return M.mean(0), C + np.cov(M.T), len(wavs)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()
    from musicgen_clap_finetune import MusicGenCLAP
    from arms_2x2 import embed_text, gen_from_enc
    from library_b_experiment import PROMPTS_B
    from finetune_prompt_classifier import PromptClassifier

    base = MusicGenCLAP(device=device, use_lora=False); mg = base.mg; mg.eval()
    codes = np.load(DATA / "codes.npy").astype(np.int64)
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    group = np.load(DATA / "group.npy").astype(np.int64)
    uniq = np.unique(group); rng = np.random.default_rng(0)
    vg = set(uniq[rng.permutation(len(uniq))[:max(1, int(round(len(uniq) * 0.1)))]].tolist())
    va = np.where(np.array([g in vg for g in group]))[0][:60]

    # real: decode held-out windows
    real_w = []
    with torch.no_grad():
        for r in va:
            oc = mg.audio_encoder.decode(torch.from_numpy(codes[r]).unsqueeze(0).unsqueeze(0).to(device),
                                         audio_scales=[None]).audio_values
            real_w.append(np.asarray(oc.cpu()).reshape(-1))
    print(f"real clips={len(real_w)} [{time.time()-t0:.0f}s]", flush=True)

    # frozen native: mode-A corpus-matched prompt per held-out window
    ck = torch.load(ROOT / "results/library_b/classifier.pt", map_location="cpu", weights_only=True)
    clf = PromptClassifier(512, 256, len(PROMPTS_B), 0.1); clf.load_state_dict(ck["model"]); clf.eval()
    with torch.no_grad():
        pred = clf(torch.from_numpy(cond[va])).argmax(1).cpu().numpy()
    ph = {k: embed_text(mg, PROMPTS_B[k], device) for k in range(len(PROMPTS_B))}
    froz_w = []
    for i, r in enumerate(va):
        w = gen_from_enc(mg, ph[int(pred[i])].unsqueeze(0), 400, 0.9, None, seed=99 + int(r))
        froz_w.append(w)
    print(f"frozen clips={len(froz_w)} [{time.time()-t0:.0f}s]", flush=True)

    def load_wavs(pat):
        out = []
        for f in sorted(glob.glob(str(ROOT / pat)))[:60]:
            w, sr = sf.read(f); w = np.asarray(w)
            out.append(w.mean(1) if w.ndim > 1 else w)
        return out
    ft_w = load_wavs("results/gen_eegcond_mixed/eval/gen_*_row*.wav")
    ip_w = load_wavs("results/gen_eegcond/library_radius/sweep_*.wav")

    mr, cr, nr = stack_feats(real_w)
    print(f"\n=== MFCC-Fréchet distance to REAL audio (lower = closer timbre) ===")
    print(f"{'arm':16} {'n':>4} {'FD->real':>10}")
    res = {"real_n": nr}
    for name, ws in [("frozen_native", froz_w), ("finetuned", ft_w), ("interpolation", ip_w)]:
        if not ws:
            print(f"{name:16}    0        n/a"); continue
        m, c, n = stack_feats(ws)
        fd = frechet(m, c, mr, cr)
        res[name] = {"n": n, "fd_to_real": fd}
        print(f"{name:16} {n:>4} {fd:>10.2f}", flush=True)
    import json
    json.dump(res, open(ROOT / "results/mfcc_fad.json", "w"), indent=2)
    print(f"\nsaved -> results/mfcc_fad.json  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
