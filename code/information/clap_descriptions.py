"""Frozen CLAP tagger: audio-only descriptions for the 14 songs.

No official semantic descriptions exist in NMED-T/NMED-H (only titles/artists/BPM, which are
unique identifiers and unfair). This script builds 14 descriptions from a PRE-REGISTERED
fixed vocabulary using the frozen laion CLAP text tower:

  - audio side : 14 song centroids from true_emb_all (laion-CLAP, audio only; no EEG/labels)
  - text side  : the fixed vocabulary below, embedded with laion/clap-htsat-unfused
  - topk       : top-8 tags by cosine to the song centroid
  - specific   : top-8 tags by (cos(song, tag) - mean over songs), with cos > 0.10
                 (discriminative tags; still from the fixed vocabulary)

Outputs results/descriptions/descriptions.json with the tag lists, full description text,
text embeddings, pairwise text cosine, tag overlap, and the audio->text self-consistency
(which is a sanity check, not an independent ceiling: the tags were selected from the audio).

Run: python code/information/clap_descriptions.py
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
OUT = ROOT / "results" / "descriptions"
GATE = ROOT.parent / "mainv2" / "data" / "gate_nmedh14_ov2_a.npz"
CLAP_ID = "laion/clap-htsat-unfused"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
K = 8


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


# Pre-registered vocabulary: genres, instruments, mood, tempo/energy, production.
# No titles, artists, or any unique identifier.
VOCAB = [
    # genres / styles
    "bollywood film music", "indian classical", "hindustani classical", "indian folk",
    "indian film ballad", "pop", "indie pop", "rock", "classic rock", "indie rock",
    "goth rock", "electronic", "edm", "deep house", "hip hop", "trap", "r&b", "soul",
    "funk", "latin", "reggaeton", "jazz", "acoustic", "folk", "classical", "orchestral",
    "ambient", "new age", "world music", "devotional", "film score", "chiptune",
    # instruments / voices
    "sitar", "tabla", "dhol", "dholak", "harmonium", "bamboo flute", "strings",
    "guitar", "electric guitar", "acoustic guitar", "bass guitar", "slap bass",
    "piano", "synthesizer", "synth pads", "brass", "drums", "live drums", "808 bass",
    "hi-hats", "female vocals", "male vocals", "choir", "percussion", "hand percussion",
    # mood
    "energetic", "calm", "dreamy", "dark", "uplifting", "melancholic", "romantic",
    "aggressive", "meditative", "playful", "cinematic", "hypnotic",
    # tempo / energy
    "fast tempo", "slow tempo", "mid-tempo", "high energy", "low energy",
    "driving beat", "four-on-the-floor", "syncopated groove", "groovy",
    # production / texture
    "glossy production", "lo-fi", "heavy reverb", "distorted", "warm", "bright",
    "airy", "punchy", "minimal", "dense mix",
]


def load_song_centroids():
    z = np.load(GATE, allow_pickle=True)
    cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)
    aldm = np.load(ROOT.parent / "mainv2" / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    w = torch.load(ROOT.parent / "mainv2" / "results" / "eeg_to_clap_proj.pt",
                   map_location="cpu", weights_only=True)
    ca = unit(aldm @ w["weight"].numpy().astype(np.float32).T + w["bias"].numpy().astype(np.float32))
    key = {ca[i].tobytes(): i for i in range(len(ca))}
    idx = np.array([key[cond[j].tobytes()] for j in range(len(cond))])
    T = unit(np.load(ROOT / "results" / "prompt_classifier" / "true_emb_all.npy").astype(np.float32))
    song = z["song"][idx].astype(int)
    C = unit(np.stack([T[song == s].mean(0) for s in range(14)]))
    return C, song


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    from transformers import ClapModel, ClapProcessor
    proc = ClapProcessor.from_pretrained(CLAP_ID)
    model = ClapModel.from_pretrained(CLAP_ID).eval().to(DEV)
    C, _ = load_song_centroids()

    with torch.no_grad():
        inp = proc(text=VOCAB, return_tensors="pt", padding=True).to(DEV)
        txt = model.get_text_features(**inp).pooler_output.cpu().numpy().astype(np.float32)
    txt = unit(txt)
    sim = C @ txt.T                                   # 14 x vocab

    def desc_from(idx_list):
        return ", ".join(VOCAB[i] for i in idx_list)

    topk, spec = [], []
    mean_sim = sim.mean(0, keepdims=True)
    for s in range(14):
        order = np.argsort(-sim[s])[:K]
        topk.append([int(i) for i in order])
        order_s = np.argsort(-(sim[s] - mean_sim[0]))
        picked = [int(i) for i in order_s if sim[s, i] > 0.10][:K]
        if len(picked) < K:                            # fill with top cosine if needed
            picked += [int(i) for i in np.argsort(-sim[s]) if int(i) not in picked][:K - len(picked)]
        spec.append(picked)

    desc_topk = [desc_from(v) for v in topk]
    desc_spec = [desc_from(v) for v in spec]
    with torch.no_grad():
        Dt = model.get_text_features(**proc(text=desc_topk, return_tensors="pt", padding=True).to(DEV)).pooler_output.cpu().numpy()
        Ds = model.get_text_features(**proc(text=desc_spec, return_tensors="pt", padding=True).to(DEV)).pooler_output.cpu().numpy()
    Dt, Ds = unit(Dt.astype(np.float32)), unit(Ds.astype(np.float32))

    def stats(D):
        pair = D @ D.T
        off = pair[~np.eye(14, dtype=bool)]
        acc = float(((D @ C.T).argmax(1) == np.arange(14)).mean())
        return {"pairwise_text_cos_mean": float(off.mean()),
                "pairwise_text_cos_min": float(off.min()),
                "self_consistency_audio_text_top1": acc}

    overlap = []
    for i in range(14):
        for j in range(i + 1, 14):
            overlap.append(len(set(spec[i]) & set(spec[j])))
    res = {
        "protocol": {
            "annotator": f"frozen laion CLAP text tower ({CLAP_ID}); audio-only",
            "audio_source": "true_emb_all (laion-CLAP of the heard stimulus); no EEG/labels used",
            "vocabulary": "pre-registered fixed list; no titles/artists/unique identifiers",
            "top_k": K,
            "specific_rule": "top-k of cos(song,tag) - mean_songs(cos), require cos > 0.10",
        },
        "vocab": VOCAB,
        "descriptions_topk": desc_topk,
        "descriptions_specific": desc_spec,
        "tags_topk": topk,
        "tags_specific": spec,
        "embedding_topk": Dt.tolist(),
        "embedding_specific": Ds.tolist(),
        "stats_topk": stats(Dt),
        "stats_specific": stats(Ds),
        "tag_overlap_mean_specific": float(np.mean(overlap)),
        "tag_overlap_max_specific": int(np.max(overlap)),
        "n_unique_tags_specific": int(len(set(i for v in spec for i in v))),
    }
    json.dump(res, open(OUT / "descriptions.json", "w", encoding="utf-8"), indent=2)
    print("=== specific descriptions (used for selection) ===")
    for s, d in enumerate(desc_spec):
        print(f"  song {s:2d}: {d}")
    print("\ntopk pairwise text cos", round(res["stats_topk"]["pairwise_text_cos_mean"], 3),
          "| self-consistency", round(res["stats_topk"]["self_consistency_audio_text_top1"], 3))
    print("specific pairwise text cos", round(res["stats_specific"]["pairwise_text_cos_mean"], 3),
          "| self-consistency", round(res["stats_specific"]["self_consistency_audio_text_top1"], 3),
          "| tag overlap mean", round(res["tag_overlap_mean_specific"], 1),
          "| unique tags", res["n_unique_tags_specific"])
    print("wrote", OUT / "descriptions.json")


if __name__ == "__main__":
    main()
