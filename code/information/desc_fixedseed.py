"""Fixed-seed control for the generation pilot: remove the seed--row confound.

The original pilot used seed = 1234 + row. Rows are grouped by song, so a constant
prompt can acquire a positive own-vs-other margin purely from the seed/row structure
(the uncond arm showed +0.048, p=0.006). This script regenerates the *same prompts*
on the *same 16 windows* with seeds that do not depend on the row (two fixed seeds).
Comparing the two runs separates prompt content from seed structure.

Criterion (written before running): the calibrated-description arm has a song-level
effect beyond a constant prompt iff its song-centroid margin exceeds the constant
prompt's margin by > 0 with a paired sign-flip p < 0.05 under the fixed seeds. If the
constant floor collapses and the calibrated arm keeps its +0.067, the row-seeded pilot
was confounded; if the calibrated arm collapses too, the pilot does not support an
end-to-end song-level spectral effect.

Run: python code/information/desc_fixedseed.py
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
ARMS = ["oracle", "head_calib", "head_zero", "mainline", "desc_shuffled", "uncond"]
SEEDS = (20261021, 20261022)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gen = json.load(open(SRC / "generation.json", encoding="utf-8"))
    ctl = json.load(open(SRC / "controls.json", encoding="utf-8"))
    prompts = {}
    for r in gen["records"] + ctl["records"]:
        prompts[(r["row"], r["arm"])] = r["prompt"]
    rows = sorted({k[0] for k in prompts})
    true_song = {r["row"]: r["true_song"] for r in gen["records"]}
    codes = np.load(ROOT / "data" / "eegcond" / "codes.npy").astype(np.int64)

    sys.path.insert(0, str(M2 / "src" / "generate"))
    sys.path.insert(0, str(ROOT / "code" / "shared"))
    sys.path.insert(0, str(ROOT / "code" / "radii"))
    from musicgen_clap_finetune import MusicGenCLAP
    from arms_2x2 import embed_text, gen_from_enc

    base = MusicGenCLAP(device="cuda", use_lora=False); mg = base.mg; mg.eval()
    phase = {}
    for (row, arm), text in prompts.items():
        if text not in phase:
            phase[text] = embed_text(mg, text, "cuda")
    print(f"{len(rows)} rows x {len(ARMS)} arms x {len(SEEDS)} seeds = "
          f"{len(rows)*len(ARMS)*len(SEEDS)} clips; {len(phase)} distinct prompts", flush=True)

    for seed in SEEDS:
        records = []
        for row in rows:
            with torch.no_grad():
                ref = mg.audio_encoder.decode(
                    torch.from_numpy(codes[row]).unsqueeze(0).unsqueeze(0).to("cuda"),
                    audio_scales=[None]).audio_values
            ref = np.asarray(ref.cpu()).reshape(-1).astype(np.float32)
            for arm in ARMS:
                text = prompts[(row, arm)]
                wav = np.asarray(gen_from_enc(mg, phase[text].unsqueeze(0), 400, 0.9, None,
                                              seed=seed), np.float32).reshape(-1)
                sf.write(OUT / f"s{seed}_{arm}_row{row}.wav", wav, SR)
                records.append({"row": row, "true_song": true_song[row], "arm": arm,
                                "prompt": text, "seed": seed,
                                **spectral_metrics(wav, ref)})
            print(f"seed {seed} row {row} song {true_song[row]} done", flush=True)
        json.dump({"protocol": {"seeds": list(SEEDS), "seed_fixed": seed,
                                "note": "same prompts/windows as generation.json; seed independent of row"},
                   "records": records},
                  open(SRC / f"generation_fixedseed_s{seed}.json", "w", encoding="utf-8"),
                  indent=2)
        print("wrote", SRC / f"generation_fixedseed_s{seed}.json", flush=True)


if __name__ == "__main__":
    main()
