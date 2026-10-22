"""Zero-information controls for the fold-0 description-generation pilot.

Two arms, same windows / seeds / generator as desc_generate.py:

  desc_shuffled  each window gets the *least similar* other song's description
                 (song-level text prompt, but for the wrong song)
  uncond         a neutral constant prompt ("music"), per-window seeds only

The mainline arm in generation.json emitted one fixed prompt on 14/16 windows, so its
own-vs-other margin is not a clean zero: the seed is tied to the row id, and row ids are
grouped by song. These two arms separate prompt content from that seed/row structure.

Run: python code/information/desc_controls.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from calib_head_pilot import DEV, GATE, KS, OUT, unit_np  # noqa: E402
from calib_head_5fold import FOLDS  # noqa: E402
from calib_head_genre import embed, fit_rotation, load_head  # noqa: E402
from desc_generate import spectral_metrics  # noqa: E402

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = ROOT.parent / "mainv2"
DESC = ROOT / "results" / "descriptions" / "descriptions.json"
GEN = ROOT / "results" / "descriptions" / "gen"
SR = 32000
N_WINDOWS_PER_SUBJECT = 4
K = 100
ARMS = ["desc_shuffled", "uncond"]


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    desc = json.load(open(DESC, encoding="utf-8"))
    desc_spec = desc["descriptions_specific"]
    D = unit_np(np.array(desc["embedding_specific"], dtype=np.float32))
    z = np.load(GATE, allow_pickle=True)
    cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)
    codes = np.load(ROOT / "data" / "eegcond" / "codes.npy").astype(np.int64)
    aldm = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    w = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu", weights_only=True)
    ca = unit_np(aldm @ w["weight"].numpy().astype(np.float32).T + w["bias"].numpy().astype(np.float32))
    key = {ca[i].tobytes(): i for i in range(len(ca))}
    idx = np.array([key[cond[j].tobytes()] for j in range(len(cond))])
    X = torch.tensor(z["bandpower"][idx].astype(np.float32))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    T = unit_np(np.load(ROOT / "results" / "prompt_classifier" / "true_emb_all.npy").astype(np.float32))
    song = z["song"][idx].astype(int); subj = z["subject"][idx].astype(str)
    seg = z["seg2"][idx].astype(int); ds = z["dataset"][idx].astype(int)
    al = z["aligned8"][idx].astype(bool)

    sys.path.insert(0, str(M2 / "src" / "generate"))
    sys.path.insert(0, str(ROOT / "code" / "shared"))
    sys.path.insert(0, str(ROOT / "code" / "radii"))
    sys.path.insert(0, str(ROOT / "code" / "classifier"))
    from musicgen_clap_finetune import MusicGenCLAP, load_clap, clap_embed_audio
    from arms_2x2 import embed_text, gen_from_enc

    base = MusicGenCLAP(device=DEV, use_lora=False); mg = base.mg; mg.eval()
    clap, proc, _ = load_clap(DEV)
    ph_desc = {i: embed_text(mg, d, DEV) for i, d in enumerate(desc_spec)}
    ph_uncond = embed_text(mg, "music", DEV)

    # each song's least similar other song by frozen text embedding
    Txt = D
    wrong = {}
    for s in range(len(desc_spec)):
        sims = Txt @ Txt[s]
        sims[s] = 2.0
        wrong[s] = int(np.argmin(sims))
    print("least-similar description map:", wrong)

    f = 0
    meta = FOLDS[f]
    train_subs = [s for s in np.unique(subj)
                  if s not in meta["test"] and s not in meta["val"] and s in set(subj[ds == 0])]
    train_rows = np.where((ds == 0) & np.isin(subj, train_subs))[0]
    Pn = unit_np(np.stack([T[train_rows][song[train_rows] == c].mean(0) for c in range(10)]))
    head = load_head(OUT / f"calib_bandpower_laion_f{f}.pt", X.shape[1])
    Z = embed(head, X)
    rng = np.random.default_rng(f)
    records = []
    for s in meta["test"]:
        rows = np.where(al & (subj == s))[0]
        segs = np.unique(seg[rows]); rng.shuffle(segs)
        fg = set(segs[:len(segs) // 2])
        rf = rows[np.isin(seg[rows], list(fg))]
        rt = rows[~np.isin(seg[rows], list(fg))]
        if len(rf) < 100 or len(rt) < 100:
            continue
        R, m = fit_rotation(Z, rf, song, Pn, k=K)
        pick = np.linspace(0, len(rt) - 1, N_WINDOWS_PER_SUBJECT).astype(int)
        for j in pick:
            row = int(rt[j]); true_song = int(song[row])
            with torch.no_grad():
                ref = mg.audio_encoder.decode(
                    torch.from_numpy(codes[row]).unsqueeze(0).unsqueeze(0).to(DEV),
                    audio_scales=[None]).audio_values
            ref = np.asarray(ref.cpu()).reshape(-1).astype(np.float32)
            prompts = {"desc_shuffled": desc_spec[wrong[true_song]], "uncond": "music"}
            phases = {"desc_shuffled": ph_desc[wrong[true_song]], "uncond": ph_uncond}
            for arm in ARMS:
                wav = np.asarray(gen_from_enc(mg, phases[arm].unsqueeze(0), 400, 0.9, None,
                                              seed=1234 + row), np.float32).reshape(-1)
                sf.write(GEN / f"{arm}_row{row}.wav", wav, SR)
                gclap = unit_np(clap_embed_audio(clap, proc, SR, wav, DEV))
                records.append({"row": row, "subject": s, "true_song": true_song, "arm": arm,
                                "prompt": prompts[arm],
                                "clap_to_stimulus": float(gclap @ T[row]),
                                **spectral_metrics(wav, ref)})
            print(f"row {row} subj {s} song {true_song}: wrong desc -> song {wrong[true_song]}", flush=True)

    agg = {}
    for arm in ARMS:
        rs = [r for r in records if r["arm"] == arm]
        agg[arm] = {k: float(np.mean([r[k] for r in rs])) for k in
                    ("clap_to_stimulus", "mel_frechet", "dtw_logmel_norm",
                     "centroid_absdiff", "rolloff_absdiff", "zcr_gen", "zcr_ref")}
        agg[arm]["n"] = len(rs)
    res = {"protocol": {"fold": f, "k": K, "windows_per_subject": N_WINDOWS_PER_SUBJECT,
                        "note": "same rows/seeds as generation.json; arms are zero-information controls",
                        "wrong_description_map": wrong},
           "aggregate": agg, "records": records}
    json.dump(res, open(ROOT / "results" / "descriptions" / "controls.json", "w",
                        encoding="utf-8"), indent=2)
    print("\n=== controls (fold 0) ===")
    for arm in ARMS:
        g = agg[arm]
        print(f"{arm:14} clap_stim {g['clap_to_stimulus']:.3f} melFD {g['mel_frechet']:.0f} "
              f"dtw {g['dtw_logmel_norm']:.2f} centroid {g['centroid_absdiff']:.0f} "
              f"zcr {g['zcr_gen']:.3f}")
    print("wrote", ROOT / "results" / "descriptions" / "controls.json")


if __name__ == "__main__":
    main()
