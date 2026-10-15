"""Genre gallery: one frozen-native MusicGen sample per corpus-matched prompt, so a listener can
hear the FULL pleasant range the system can produce (the EEG-driven deliveries skew to a few
genres; this is the unbiased showcase). Every clip is a native sample of the frozen generator under
its own prompt -> pleasant by construction. Writes gallery/<idx>_<slug>.wav + a manifest.

Run: python src/train/genre_gallery.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
OUT = ROOT / "results" / "library_b" / "gallery"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

SR = 32000


def soft_limit(w, thr=0.95):
    return np.tanh(w / thr) * thr


def zcr(w):
    return float(np.mean(np.abs(np.diff(np.sign(w))))) / 2


def main():
    from library_b_experiment import PROMPTS_B
    from musicgen_clap_finetune import MusicGenCLAP
    from arms_2x2 import embed_text, gen_from_enc
    import soundfile as sf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    base = MusicGenCLAP(device=device, use_lora=False); mg = base.mg; mg.eval()

    lines = ["# 曲风画廊 — 每条提示一个冻结原生样本（系统能产出的全部曲风，供试听）", ""]
    for k, p in enumerate(PROMPTS_B):
        h = embed_text(mg, p, device)
        w = gen_from_enc(mg, h.unsqueeze(0), 400, 0.9, None, seed=2025 + k)
        w = soft_limit(w)
        slug = p.split()[0] + p.split()[1][:4] if len(p.split()) > 1 else f"p{k}"
        path = OUT / f"{k:02d}_{slug}.wav"
        sf.write(str(path), w, SR)
        lines.append(f"- **prompt {k}** — \"{p}\"")
        lines.append(f"  - {path}  (zcr={zcr(w):.3f} peak={np.abs(w).max():.2f} dur={len(w)/SR:.1f}s)")
        print(f"  {k:02d} {p[:44]:44} zcr={zcr(w):.3f} peak={np.abs(w).max():.2f} [{time.time()-t0:.0f}s]",
              flush=True)
    lines += ["", "> 全部是冻结 MusicGen 在原生文本提示下的采样，悦耳由构造保证；zcr 只作参考不是判据。"]
    (OUT / "GALLERY.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\ngallery -> {OUT}  ({len(PROMPTS_B)} clips)  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
