"""Dose test: how much does each fine-tune move the decoder off its native text-conditioned
output? Every arm is driven by the SAME four official text prompts, same sampling, same seed as
arms_2x2.py -- so the only thing that changes is which weights are loaded.

This is the control for the user's claim "fine-tuning destroys musicality": with the condition
held native (in-distribution text), any roughness has to come from the weights, not from the
condition. Adding the FMA-only arm (fine-tuned on real-audio conditions, never saw EEG) splits
the cause in two: any codebook fine-tune vs EEG-conditioned fine-tune specifically.

Runs on CPU on purpose so it can sit beside a training job without competing for VRAM.

Run: python src/train/text_prompt_dose.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
OUT = ROOT / "results" / "gen_eegcond" / "text_dose"

ARMS = [
    ("pretrained",            None),
    ("fma_s20000",            "results/gen_fmaonly/ckpt_s20000.pt"),
    ("eeg_k8_ep2",            "results/gen_eegcond_k8/ckpt_ep2.pt"),
    ("eeg_k8_ep16",           "results/gen_eegcond_k8/ckpt_ep16.pt"),
    ("eeg_k8_ep29",           "results/gen_eegcond_k8/ckpt_ep29.pt"),
    ("mixed_s12000",          "results/gen_eegcond_mixed/ckpt_s12000.pt"),
    ("fma2eeg_s8000",         "results/gen_fma2eeg/ckpt_s8000.pt"),
]


def main():
    import soundfile as sf
    from arms_2x2 import PROMPTS, embed_text, gen_from_enc
    from pilot_multitoken import CondWidthModel
    from finetune_gen_k import load_state
    from musicgen_clap_finetune import MusicGenCLAP

    device = "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    tokens, temperature, top_k, seed = 400, 0.9, None, 1234
    print(f"CPU; 固定采样 temperature={temperature} top_k={top_k} tokens={tokens} seed={seed}"
          f"（与 arms2x2 完全一致）", flush=True)

    rows = []
    for name, ck in ARMS:
        t0 = time.time()
        if ck is None:
            mg = MusicGenCLAP(device=device, use_lora=False).mg
        else:
            wrap = CondWidthModel(device, k_tokens=8)
            load_state(wrap, str(ROOT / ck))
            mg = wrap.m.mg
        mg.eval()
        zs, pks = [], []
        for i, prompt in enumerate(PROMPTS):
            h = embed_text(mg, prompt, device)
            w = gen_from_enc(mg, h.unsqueeze(0), tokens, temperature, top_k, seed)
            sf.write(str(OUT / f"{name}__prompt{i}.wav"), w, 32000)
            z = float(np.mean(np.abs(np.diff(np.sign(w))))) / 2
            pk = float(np.abs(w).max())
            zs.append(z); pks.append(pk)
            print(f"  {name:<16} prompt{i} 过零率={z:.4f} 峰值={pk:.2f}", flush=True)
        rows.append((name, float(np.mean(zs)), float(np.mean(pks))))
        print(f"  {name:<16} 均值 过零率={np.mean(zs):.4f} 峰值={np.mean(pks):.2f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del mg
        if ck is not None:
            del wrap

    print("\n===== 同一组原生文本提示下的纹理（越接近 0.0319 的未微调水平越干净）=====")
    for name, z, pk in rows:
        print(f"  {name:<16} 过零率={z:.4f}  峰值={pk:.2f}")
    print(f"wav 在 {OUT}；每个 arm 4 条，提示与采样完全相同")


if __name__ == "__main__":
    main()
