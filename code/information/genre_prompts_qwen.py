"""Qwen-written genre prompts for the 12-genre conditioned-generation path.

The 12 genre identities are the existing library prompts; only the wording is rewritten
by Qwen2-Audio (text-only, no audio) so that the genre-level and song-level prompt sets
share one author and one style. The song prompts are the Qwen captions from
caption_qwen.py. Output includes laion-CLAP text embeddings for selection.

Run: python code/information/genre_prompts_qwen.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
OUT = ROOT / "results" / "descriptions"
MODEL = ROOT.parent / "models" / "qwen2-audio-7b-ms"
CLAP_ID = "laion/clap-htsat-unfused"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

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
]

INSTRUCTION = (
    "You are a music prompt writer. Rewrite the following genre description as one clear, "
    "self-contained text-to-music prompt of 25 to 40 words. Keep the genre/style, the main "
    "instruments, the tempo/rhythm, the mood, and the production. Do not name any song, artist, "
    "album, film, or lyrics. Do not add a specific melody. Output only the rewritten prompt."
)


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2AudioForConditionalGeneration)
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        str(MODEL), quantization_config=bnb, device_map="auto", torch_dtype=torch.bfloat16).eval()
    proc = AutoProcessor.from_pretrained(str(MODEL))

    rows = []
    for i, p in enumerate(PROMPTS):
        text = f"{INSTRUCTION}\n\nGenre description: {p}"
        conv = [{"role": "user", "content": [{"type": "text", "text": text}]}]
        chat = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=chat, return_tensors="pt")
        inp = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=90, do_sample=False)
        q = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        rows.append({"index": i, "original": p, "qwen_prompt": q})
        print(f"{i:2d} {q}", flush=True)
    json.dump(rows, open(OUT / "genre_prompts_qwen.json", "w", encoding="utf-8"), indent=2)

    from transformers import ClapModel, ClapProcessor
    cp = ClapProcessor.from_pretrained(CLAP_ID)
    clap = ClapModel.from_pretrained(CLAP_ID).eval().to(DEV)
    with torch.no_grad():
        tinp = cp(text=[r["qwen_prompt"] for r in rows], return_tensors="pt", padding=True).to(DEV)
        E = unit(clap.get_text_features(**tinp).pooler_output.cpu().numpy().astype(np.float32))
    P = E @ E.T
    iu = np.triu_indices(12, 1)
    stats = {"pairwise_text_cos_mean": float(P[iu].mean()),
             "pairwise_text_cos_min": float(P[iu].min()),
             "pairwise_text_cos_max": float(P[iu].max()),
             "unique": len({r["qwen_prompt"] for r in rows})}
    for r, e in zip(rows, E):
        r["embedding"] = e.tolist()
    json.dump({"stats": stats, "prompts": rows},
              open(OUT / "genre_prompts_qwen.json", "w", encoding="utf-8"), indent=2)
    print("stats", {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()})
    print("wrote", OUT / "genre_prompts_qwen.json")


if __name__ == "__main__":
    main()
