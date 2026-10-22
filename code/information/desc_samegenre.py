"""Same-genre control: replace the true description with the most similar *other* song's.

The least-similar wrong description (desc_shuffled) tests "any wrong song"; this arm
tests the harder case that matters for a song-level claim: another song from the same
text neighbourhood. If oracle > same-genre, the generated audio carries song-specific
information beyond genre; if oracle ~= same-genre, the effect is genre-level.

Same 16 windows and fixed seeds as generation_fixedseed_s*.json.

Run: python code/information/desc_samegenre.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from desc_generate import SR, spectral_metrics  # noqa: E402

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = ROOT.parent / "mainv2"
SRC = ROOT / "results" / "descriptions"
OUT = SRC / "gen_fixedseed"
SEEDS = (20261021, 20261022)


def main():
    desc = json.load(open(SRC / "descriptions.json", encoding="utf-8"))
    spec = desc["descriptions_specific"]
    D = np.array(desc["embedding_specific"], dtype=np.float32)
    D /= np.linalg.norm(D, axis=1, keepdims=True) + 1e-9
    S = D @ D.T
    close = {}
    for s in range(len(spec)):
        sims = S[s].copy(); sims[s] = -9
        close[s] = int(np.argmax(sims))
    print("same-genre proxy map:", close, flush=True)

    gen = json.load(open(SRC / "generation.json", encoding="utf-8"))
    rows = sorted({r["row"]: r["true_song"] for r in gen["records"]}.items())
    codes = np.load(ROOT / "data" / "eegcond" / "codes.npy").astype(np.int64)

    sys.path.insert(0, str(M2 / "src" / "generate"))
    sys.path.insert(0, str(ROOT / "code" / "shared"))
    sys.path.insert(0, str(ROOT / "code" / "radii"))
    from musicgen_clap_finetune import MusicGenCLAP
    from arms_2x2 import embed_text, gen_from_enc

    base = MusicGenCLAP(device="cuda", use_lora=False); mg = base.mg; mg.eval()
    phase = {s: embed_text(mg, spec[close[s]], "cuda") for s in close}
    print(f"{len(rows)} rows x {len(SEEDS)} seeds; {len(phase)} distinct prompts", flush=True)

    for seed in SEEDS:
        new = []
        for row, true_song in rows:
            with torch.no_grad():
                ref = mg.audio_encoder.decode(
                    torch.from_numpy(codes[row]).unsqueeze(0).unsqueeze(0).to("cuda"),
                    audio_scales=[None]).audio_values
            ref = np.asarray(ref.cpu()).reshape(-1).astype(np.float32)
            text = spec[close[true_song]]
            wav = np.asarray(gen_from_enc(mg, phase[true_song].unsqueeze(0), 400, 0.9, None,
                                          seed=seed), np.float32).reshape(-1)
            sf.write(OUT / f"s{seed}_desc_samegenre_row{row}.wav", wav, SR)
            new.append({"row": row, "true_song": true_song, "arm": "desc_samegenre",
                        "prompt": text, "seed": seed, **spectral_metrics(wav, ref)})
            print(f"seed {seed} row {row} song {true_song} -> proxy song {close[true_song]}", flush=True)
        base_recs = json.load(open(SRC / f"generation_fixedseed_s{seed}.json",
                                   encoding="utf-8"))
        recs = base_recs["records"] + new
        json.dump({"protocol": {"seeds": list(SEEDS), "seed_fixed": seed,
                                "same_genre_map": close,
                                "note": "oracle/desc_samegenre share windows and seeds"},
                   "records": recs},
                  open(SRC / f"generation_samegenre_s{seed}.json", "w", encoding="utf-8"),
                  indent=2)
        print("wrote", SRC / f"generation_samegenre_s{seed}.json", flush=True)


if __name__ == "__main__":
    main()
