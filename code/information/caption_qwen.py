"""Free-text music captions with Qwen2-Audio-7B-Instruct (4-bit).

Audio: three centroid-representative 8 s windows per song
(results/descriptions/caption_audio_centroid).
Prompt bans titles/artists/albums/films; genres and instruments are allowed.
Embeddings: laion/clap-htsat-unfused text tower, same space as tags/LP captions.

Run: python code/information/caption_qwen.py
"""
import argparse
import json
import re
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
OUT = ROOT / "results" / "descriptions"
MODEL = ROOT.parent / "models" / "qwen2-audio-7b-ms"
CLAP_ID = "laion/clap-htsat-unfused"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

PROMPT = (
    "You are an acoustic music analyst. Listen to the audio and write ONE concise caption "
    "(maximum 35 words) describing only what you hear: style/genre, mood, tempo, prominent "
    "instruments, vocal type, and production. You may name genres and instruments, but do NOT "
    "name the song, artist, album, film, or lyrics, and do not guess a title. If you recognize "
    "the track, ignore that knowledge and describe the acoustics only. Output only the caption."
)

# capitalized tokens that are useful genre/instrument/region words, not titles/artists
ALLOW = {
    "Bollywood", "Indian", "Hindustani", "Latin", "Western", "African", "Afropop",
    "EDM", "R&B", "Hip", "Hop", "Rock", "Pop", "Jazz", "Soul", "Funk", "Techno",
    "House", "Reggae", "Reggaeton", "Classical", "Orchestral", "Ambient", "Folk",
    "Trap", "Disco", "Gospel", "Metal", "Punk", "Country", "Blues", "Dancehall",
    "Arabic", "Chinese", "Japanese", "Korean", "English", "Spanish", "Tamil", "Hindi",
}


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def leak_flags(cap):
    flags = []
    for sent in re.split(r"(?<=[.!?])\s+", cap):
        toks = re.findall(r"\b[A-Z][A-Za-z'&.-]{1,}\b", sent)
        if toks and sent.lstrip().startswith(toks[0]):
            toks = toks[1:]          # sentence-initial capital is not a proper noun
        flags += [t for t in toks if t not in ALLOW]
    if re.search(r"\b(by|titled|called|from the film|soundtrack|artist)\b", cap, re.I):
        flags.append("TITLE_PATTERN")
    return flags


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path, default=OUT / "caption_audio_centroid")
    ap.add_argument("--model-dir", type=Path, default=MODEL)
    ap.add_argument("--prefix", default="captions_qwen")
    ap.add_argument("--max-new-tokens", type=int, default=80)
    ap.add_argument("--sample", action="store_true")
    args = ap.parse_args()

    from transformers import (AutoProcessor, BitsAndBytesConfig,
                              Qwen2AudioForConditionalGeneration)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16,
                             bnb_4bit_use_double_quant=True)
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        str(args.model_dir), quantization_config=bnb, device_map="auto",
        torch_dtype=torch.bfloat16)
    model.eval()
    proc = AutoProcessor.from_pretrained(str(args.model_dir))
    sr = int(proc.feature_extractor.sampling_rate)
    print("loaded Qwen2-Audio; input sr", sr, "device", model.device, flush=True)

    manifest = json.load(open(args.audio_dir / "manifest.json", encoding="utf-8"))
    rows = []
    for m in manifest:
        wav, _ = librosa.load(m["path"], sr=sr, mono=True)
        conv = [{"role": "user", "content": [
            {"type": "audio", "audio_url": m["path"]},
            {"type": "text", "text": PROMPT},
        ]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[wav], sampling_rate=sr, return_tensors="pt", padding=True)
        inp = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inp.items()}
        gen_kw = {"max_new_tokens": args.max_new_tokens, "do_sample": args.sample}
        if args.sample:
            gen_kw.update({"temperature": 0.7, "top_p": 0.9})
        with torch.no_grad():
            out = model.generate(**inp, **gen_kw)
        cap = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        flags = leak_flags(cap)
        rows.append({**m, "caption": cap, "leak_flags": flags})
        print(f"song {m['song']:2d} seg {m['seg']:3d}: {cap}  flags={flags}", flush=True)
    json.dump(rows, open(OUT / f"{args.prefix}.json", "w", encoding="utf-8"), indent=2)

    from transformers import ClapModel, ClapProcessor
    cp = ClapProcessor.from_pretrained(CLAP_ID)
    clap = ClapModel.from_pretrained(CLAP_ID).eval().to(DEV)
    with torch.no_grad():
        tinp = cp(text=[r["caption"] for r in rows], return_tensors="pt", padding=True).to(DEV)
        E = clap.get_text_features(**tinp).pooler_output.cpu().numpy().astype(np.float32)
    E = unit(E)

    songs = []
    for s in range(14):
        idx = [i for i, r in enumerate(rows) if r["song"] == s]
        es = E[idx]
        sim = es @ es.T
        med = idx[int(np.argmax(sim.mean(1)))]
        concat = " ".join(rows[i]["caption"].rstrip(".") + "." for i in idx)
        with torch.no_grad():
            tinp = cp(text=[concat], return_tensors="pt", padding=True).to(DEV)
            ec = unit(clap.get_text_features(**tinp).pooler_output.cpu().numpy().astype(np.float32))[0]
        songs.append({"song": s, "captions": [rows[i]["caption"] for i in idx],
                      "medoid_caption": rows[med]["caption"],
                      "medoid_embedding": E[med].tolist(),
                      "concat_caption": concat, "concat_embedding": ec.tolist(),
                      "leak_flags": sorted({f for i in idx for f in rows[i]["leak_flags"]})})
    pair = np.stack([unit(np.array(r["medoid_embedding"], np.float32)) for r in songs])
    P = pair @ pair.T
    iu = np.triu_indices(14, 1)
    stats = {"pairwise_text_cos_mean": float(P[iu].mean()),
             "pairwise_text_cos_min": float(P[iu].min()),
             "pairwise_text_cos_max": float(P[iu].max()),
             "n_leak_flags": int(sum(bool(r["leak_flags"]) for r in songs))}
    print("\nmedoid captions:")
    for r in songs:
        print(f"  song {r['song']:2d}: {r['medoid_caption']}  flags={r['leak_flags']}")
    print("stats", {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()})
    json.dump({"stats": stats, "songs": songs},
              open(OUT / f"{args.prefix}_song.json", "w", encoding="utf-8"), indent=2)
    print("wrote", OUT / f"{args.prefix}_song.json")


if __name__ == "__main__":
    main()
