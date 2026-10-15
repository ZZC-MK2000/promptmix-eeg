"""Decisive 2x2: is the roughness from sampling, from our fine-tune, or from the EEG condition?

Arms (same machine, same Encodec codec, same sampling settings):
  weights : pretrained (untouched decoder) | finetuned (ckpt_ep16: LoRA + trained clap_proj)
  condition: official TEXT prompt via the T5 text encoder | our bridge(EEG) via clap_proj

Reading:
  text+pretrained good, text+finetuned good, eeg+finetuned rough  -> the EEG condition path
  text+finetuned rough while text+pretrained good                 -> our LoRA training degraded music
  text+pretrained itself rough                                    -> ceiling of this 300M + Encodec stack
Sampling (top_k etc) is held fixed across all four, so it cannot explain any difference here.

Run: python src/train/arms_2x2.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
OUT = ROOT / "results" / "gen_eegcond" / "arms2x2"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

WAV_SR = 32000
PROMPTS = [
    "a warm melodic pop piece with soft piano and light drums",
    "upbeat indian film music with strings and tabla",
    "ambient electronic texture with pads and gentle beat",
    "acoustic guitar folk melody, calm and bright",
]
# held-out target rows used by the earlier identity tests, so the EEG arm is out-of-sample
EEG_ROWS = [202, 457, 681, 889]


def embed_text(mg, text, device):
    """One prompt at a time: batch=1 means no padding, so the absent attention_mask
    cannot silently handicap the official text arm."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(M2.parent / "models" / "musicgen-small"))
    ids = tok([text], return_tensors="pt").to(device)
    with torch.no_grad():
        h = mg.text_encoder(input_ids=ids.input_ids,
                            attention_mask=ids.attention_mask).last_hidden_state
    return h[0]


def gen_from_enc(mg, enc_hidden, tokens, temperature, top_k, seed):
    from transformers.modeling_outputs import BaseModelOutput
    torch.manual_seed(seed)
    enc = BaseModelOutput(last_hidden_state=enc_hidden)
    out = mg.generate(encoder_outputs=enc, max_new_tokens=tokens,
                      temperature=temperature, top_k=top_k, do_sample=True)
    return np.asarray(out.cpu()).reshape(-1)


def main():
    from pilot_multitoken import CondWidthModel
    from finetune_gen_k import load_state
    from musicgen_clap_finetune import MusicGenCLAP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    tokens, temperature, top_k = 400, 0.9, None   # identical for every arm
    print(f"fixed sampling for all arms: temperature={temperature} top_k={top_k} "
          f"tokens={tokens} (~{tokens/50:.1f}s)")

    cond = np.load(DATA / "cond.npy").astype(np.float32)

    combos = []
    for wt in ("pretrained", "finetuned_ep16"):
        for cd in ("text", "eeg"):
            if cd == "eeg" and wt == "pretrained":
                continue      # a random/untrained clap_proj EEG arm exists elsewhere; skip noise
            combos.append((wt, cd))

    for wt, cd in combos:
        t0 = __import__("time").time()
        if wt == "pretrained":
            wrap = MusicGenCLAP(device=device, use_lora=False)   # untouched decoder
            k = None
        else:
            wrap = CondWidthModel(device, k_tokens=8)
            load_state(wrap, str(ROOT / "results/gen_eegcond_k8/ckpt_ep16.pt"))
            k = 8
        mg = wrap.m.mg if k else wrap.mg
        mg.eval()

        if cd == "text":
            items = [(f"prompt{i}", embed_text(mg, t, device))
                     for i, t in enumerate(PROMPTS)]
        else:
            emb = torch.from_numpy(cond[EEG_ROWS])
            H = wrap.cond(emb.to(device))
            items = [(f"row{r}", H[i]) for i, r in enumerate(EEG_ROWS)]

        for name, h in items:
            w = gen_from_enc(mg, h.unsqueeze(0) if h.ndim == 2 else h,
                             tokens, temperature, top_k, seed=1234)
            p = OUT / f"{wt}__{cd}__{name}.wav"
            sf.write(str(p), w, WAV_SR)
            rms = float(np.sqrt(np.mean(w ** 2)))
            peak = float(np.abs(w).max())
            zcr = float(np.mean(np.abs(np.diff(np.sign(w))))) / 2
            print(f"  {wt:<15}{cd:<5}{name:<10} len={len(w)/WAV_SR:.2f}s rms={rms:.4f} "
                  f"peak={peak:.2f} 过零率={zcr:.3f}", flush=True)
        print(f"[{wt}/{cd}] done [{__import__('time').time()-t0:.0f}s]\n")
        del wrap, mg
        if device == "cuda":
            torch.cuda.empty_cache()

    print(f"文件在 {OUT}，命名 权重__条件类型__编号.wav；四种组合采样参数完全相同")
    print("提示: 过零率(每秒符号翻转次数, 越高越'毛刺')只是客观辅助, 判决靠耳朵")


if __name__ == "__main__":
    main()
