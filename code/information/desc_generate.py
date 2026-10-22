"""Generation with the selected 14 descriptions + spectral similarity to the stimulus.

The 14-way description selection is 0.295 +- 0.022 (chance 0.071; shuffled 0.03), so the
song-level calibration can drive a song-level text interface. This script generates clips
with the frozen MusicGen for fold 0's held-out subjects and compares four prompt arms:

  oracle      the true song's description
  head_calib  the k-shot calibrated query selects one of the 14 descriptions
  head_zero   the uncalibrated query selects one of the 14 descriptions
  mainline    the existing 12-prompt genre classifier

Metrics per clip: CLAP cosine to the true stimulus; log-mel Frechet distance, DTW, spectral
centroid / rolloff / zcr differences to the decoded stimulus. The stimulus is decoded from
the cached EnCodec codes (NMED-T audio is not published; NMED-H WAVs exist for 4 songs).

Run: python code/information/desc_generate.py
"""
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from scipy.linalg import sqrtm

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from calib_head_pilot import DEV, GATE, KS, OUT, unit_np  # noqa: E402
from calib_head_5fold import FOLDS  # noqa: E402
from calib_head_genre import embed, fit_rotation, load_head  # noqa: E402

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = ROOT.parent / "mainv2"
DESC = ROOT / "results" / "descriptions" / "descriptions.json"
GEN = ROOT / "results" / "descriptions" / "gen"
SR = 32000
N_WINDOWS_PER_SUBJECT = 4
K = 100
ARM_NAMES = ["oracle", "head_calib", "head_zero", "mainline"]


def zcr(w):
    return float(np.mean(np.abs(np.diff(np.sign(w)))) / 2)


def spectral_metrics(gen, ref):
    n = min(len(gen), len(ref))
    gen, ref = gen[:n].astype(np.float32), ref[:n].astype(np.float32)
    mg = librosa.feature.melspectrogram(y=gen, sr=SR, n_fft=1024, hop_length=256, n_mels=64)
    mr = librosa.feature.melspectrogram(y=ref, sr=SR, n_fft=1024, hop_length=256, n_mels=64)
    lg, lr = librosa.power_to_db(mg), librosa.power_to_db(mr)
    mu_g, mu_r = lg.mean(1), lr.mean(1)
    Cg, Cr = np.cov(lg), np.cov(lr)
    covmean = sqrtm(Cg @ Cr)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fd = float(((mu_g - mu_r) ** 2).sum() + np.trace(Cg + Cr - 2 * covmean))
    dtw = float(librosa.sequence.dtw(X=lg, Y=lr, metric="euclidean")[0][-1, -1] /
                (lg.shape[1] + lr.shape[1]))
    sc_g = librosa.feature.spectral_centroid(y=gen, sr=SR).mean()
    sc_r = librosa.feature.spectral_centroid(y=ref, sr=SR).mean()
    ro_g = librosa.feature.spectral_rolloff(y=gen, sr=SR).mean()
    ro_r = librosa.feature.spectral_rolloff(y=ref, sr=SR).mean()
    return {"mel_frechet": fd, "dtw_logmel_norm": dtw,
            "centroid_absdiff": float(abs(sc_g - sc_r)),
            "rolloff_absdiff": float(abs(ro_g - ro_r)),
            "zcr_gen": zcr(gen), "zcr_ref": zcr(ref)}


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
    from library_b_experiment import PROMPTS_B
    from finetune_prompt_classifier import PromptClassifier

    base = MusicGenCLAP(device=DEV, use_lora=False); mg = base.mg; mg.eval()
    clap, proc, _ = load_clap(DEV)
    ck = torch.load(ROOT / "results" / "library_b" / "classifier.pt", map_location="cpu", weights_only=True)
    clf = PromptClassifier(512, 256, len(PROMPTS_B), 0.1)
    clf.load_state_dict(ck["model"]); clf.eval().to(DEV)
    ph_desc = {i: embed_text(mg, d, DEV) for i, d in enumerate(desc_spec)}
    ph_main = {k: embed_text(mg, PROMPTS_B[k], DEV) for k in range(len(PROMPTS_B))}

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
        q = (Z[rt] - m) @ R
        pick = np.linspace(0, len(rt) - 1, N_WINDOWS_PER_SUBJECT).astype(int)
        for j in pick:
            row = int(rt[j]); true_song = int(song[row])
            sel_cal = int(np.argmax(q[j] @ D.T))
            sel_zero = int(np.argmax(Z[row] @ D.T))
            with torch.no_grad():
                logits = clf(torch.from_numpy(cond[row:row + 1]).to(DEV)).float()
            sel_main = int(logits.argmax(1).item())
            prompts = {"oracle": desc_spec[true_song], "head_calib": desc_spec[sel_cal],
                       "head_zero": desc_spec[sel_zero], "mainline": PROMPTS_B[sel_main]}
            with torch.no_grad():
                ref = mg.audio_encoder.decode(
                    torch.from_numpy(codes[row]).unsqueeze(0).unsqueeze(0).to(DEV),
                    audio_scales=[None]).audio_values
            ref = np.asarray(ref.cpu()).reshape(-1).astype(np.float32)
            for arm in ARM_NAMES:
                if arm == "mainline":
                    ph = ph_main[sel_main]
                else:
                    idx_desc = true_song if arm == "oracle" else (sel_cal if arm == "head_calib" else sel_zero)
                    ph = ph_desc[idx_desc]
                wav = np.asarray(gen_from_enc(mg, ph.unsqueeze(0), 400, 0.9, None,
                                              seed=1234 + row), np.float32).reshape(-1)
                sf.write(GEN / f"{arm}_row{row}.wav", wav, SR)
                gclap = unit_np(clap_embed_audio(clap, proc, SR, wav, DEV))
                dc = (1 if arm == "oracle" else
                      int(sel_cal == true_song) if arm == "head_calib" else
                      int(sel_zero == true_song) if arm == "head_zero" else None)
                qq = (T[row] if arm == "oracle" else q[j] if arm == "head_calib" else
                      Z[row] if arm == "head_zero" else cond[row])
                rec = {"row": row, "subject": s, "true_song": true_song, "arm": arm,
                       "prompt": prompts[arm], "desc_correct": dc,
                       "clap_to_stimulus": float(gclap @ T[row]),
                       "clap_to_query": float(gclap @ qq),
                       **spectral_metrics(wav, ref)}
                records.append(rec)
            print(f"row {row} subj {s} song {true_song}: cal {sel_cal} zero {sel_zero} main {sel_main}", flush=True)

    agg = {}
    for arm in ARM_NAMES:
        rs = [r for r in records if r["arm"] == arm]
        agg[arm] = {k: float(np.mean([r[k] for r in rs])) for k in
                    ("clap_to_stimulus", "clap_to_query", "mel_frechet",
                     "dtw_logmel_norm", "centroid_absdiff", "rolloff_absdiff", "zcr_gen", "zcr_ref")}
        dc = [r["desc_correct"] for r in rs if r["desc_correct"] is not None]
        agg[arm]["desc_correct"] = float(np.mean(dc)) if dc else None
        agg[arm]["n"] = len(rs)
    res = {"protocol": {"fold": 0, "k": K, "windows_per_subject": N_WINDOWS_PER_SUBJECT,
                        "stimulus": "EnCodec-decoded codes (NMED-T audio not published)",
                        "generator": "frozen MusicGen-small, 400 tokens, temp 0.9, fixed seed",
                        "selection_accuracy_5fold": 0.295},
           "aggregate": agg, "records": records}
    json.dump(res, open(ROOT / "results" / "descriptions" / "generation.json", "w", encoding="utf-8"), indent=2)
    print("\n=== generation (fold 0) ===")
    for arm in ARM_NAMES:
        g = agg[arm]
        dc = f"{g['desc_correct']:.2f}" if g["desc_correct"] is not None else "  n/a"
        print(f"{arm:12} desc_correct {dc} clap_stim {g['clap_to_stimulus']:.3f} "
              f"melFD {g['mel_frechet']:.0f} dtw {g['dtw_logmel_norm']:.2f} "
              f"centroid {g['centroid_absdiff']:.0f} zcr {g['zcr_gen']:.3f}/{g['zcr_ref']:.3f}")
    print("wrote", ROOT / "results" / "descriptions" / "generation.json")


if __name__ == "__main__":
    main()
