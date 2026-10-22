"""Free-text music captions with LP-MusicCaps for the 14 stimuli.

Audio: three decoded 8 s windows per song (results/descriptions/caption_audio).
Model: LP-MusicCaps (BartCaptionModel, 148M params) pretrained on MusicCaps-style
captions; only acoustic/style content, no titles or artists.
Embeddings: laion/clap-htsat-unfused text tower, same space as the tag descriptions.

Run: python code/information/caption_lp.py
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
from safetensors.torch import load_file

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from lpmc_model.bart import BartCaptionModel  # noqa: E402

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
AUDIO = ROOT / "results" / "descriptions" / "caption_audio"
OUT = ROOT / "results" / "descriptions"
WEIGHTS = ROOT.parent / "models" / "lp-music-caps" / "model.safetensors"
BART_DIR = ROOT.parent / "models" / "bart-base-tokenizer"
CLAP_ID = "laion/clap-htsat-unfused"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SR = 16000
N_SAMPLES = 10 * SR


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


@torch.no_grad()
def generate_caption(model, wav):
    x = np.zeros(N_SAMPLES, np.float32)
    x[:min(len(wav), N_SAMPLES)] = wav[:N_SAMPLES]
    embs = model.audio_encoder(torch.from_numpy(x).unsqueeze(0).to(DEV))
    out = model.bart.generate(inputs_embeds=embs, max_new_tokens=48, num_beams=3,
                              min_length=5, repetition_penalty=1.1)
    return model.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-dir", type=Path, default=AUDIO)
    ap.add_argument("--prefix", default="captions_lp")
    args = ap.parse_args()
    manifest = json.load(open(args.audio_dir / "manifest.json", encoding="utf-8"))
    model = BartCaptionModel(bart_type=str(BART_DIR)).to(DEV).eval()
    sd = load_file(str(WEIGHTS))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print("missing", len(missing), missing[:5])
    print("unexpected", len(unexpected), unexpected[:5])
    model.eval()

    rows = []
    for m in manifest:
        wav, _ = librosa.load(m["path"], sr=SR, mono=True)
        cap = generate_caption(model, wav.astype(np.float32))
        rows.append({**m, "caption": cap})
        print(f"song {m['song']:2d} seg {m['seg']:3d}: {cap}", flush=True)
    json.dump(rows, open(OUT / f"{args.prefix}.json", "w", encoding="utf-8"), indent=2)

    # text embeddings with the same laion CLAP text tower as the tag descriptions
    from transformers import ClapModel, ClapProcessor
    proc = ClapProcessor.from_pretrained(CLAP_ID)
    clap = ClapModel.from_pretrained(CLAP_ID).eval().to(DEV)
    caps = [r["caption"] for r in rows]
    with torch.no_grad():
        inp = proc(text=caps, return_tensors="pt", padding=True).to(DEV)
        E = clap.get_text_features(**inp).pooler_output.cpu().numpy().astype(np.float32)
    E = unit(E)

    song_rows = []
    for s in range(14):
        idx = [i for i, r in enumerate(rows) if r["song"] == s]
        es = E[idx]
        sim = es @ es.T
        med = idx[int(np.argmax(sim.mean(1)))]
        concat = " ".join(rows[i]["caption"].rstrip(".") + "." for i in idx)
        with torch.no_grad():
            inp = proc(text=[concat], return_tensors="pt", padding=True).to(DEV)
            ec = unit(clap.get_text_features(**inp).pooler_output.cpu().numpy().astype(np.float32))[0]
        song_rows.append({"song": s, "captions": [rows[i]["caption"] for i in idx],
                          "medoid_caption": rows[med]["caption"],
                          "medoid_embedding": es[idx.index(med)].tolist(),
                          "concat_caption": concat, "concat_embedding": ec.tolist()})
    pair = np.stack([unit(np.array(r["medoid_embedding"], np.float32)) for r in song_rows])
    P = pair @ pair.T
    iu = np.triu_indices(14, 1)
    stats = {"pairwise_text_cos_mean": float(P[iu].mean()),
             "pairwise_text_cos_min": float(P[iu].min()),
             "pairwise_text_cos_max": float(P[iu].max()),
             "n_unique_medoids": len({r["medoid_caption"] for r in song_rows})}
    print("\nmedoid captions:")
    for r in song_rows:
        print(f"  song {r['song']:2d}: {r['medoid_caption']}")
    print("pairwise text cos", {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()})
    json.dump({"stats": stats, "songs": song_rows},
              open(OUT / f"{args.prefix}_song.json", "w", encoding="utf-8"), indent=2)
    print("wrote", OUT / f"{args.prefix}_song.json")


if __name__ == "__main__":
    main()
