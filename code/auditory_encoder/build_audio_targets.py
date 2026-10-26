"""Build window-level audio targets for the external corpus (CIRE + KUL)."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.optimize import linear_sum_assignment

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = ROOT.parent / "mainv2"
CIRE = Path("D:/Projects/CLaBraM/data_external/cire")


def cire_map():
    ev = pd.read_csv(CIRE / "sub-02_task-CIRE_events.tsv", sep="\t")
    on = ev.iloc[0::2].onset.values; off = ev.iloc[1::2].onset.values
    td = off - on
    fd = {}
    for p in CIRE.glob("*.wav"):
        n = int(re.search(r"(\d+)\.wav$", p.name).group(1))
        fd[n] = sf.info(p).duration
    ids = sorted(fd)
    D = np.abs(td[:, None] - np.array([fd[i] for i in ids])[None, :])
    r, c = linear_sum_assignment(D)
    out = {int(i): int(ids[j]) for i, j in zip(r, c)}
    print(f"CIRE matched {len(out)}/240 trials, mean |dt| {D[r,c].mean()*1000:.2f} ms", flush=True)
    return out


def main():
    meta = np.load(M2 / "data" / "auditory_corpus" / "meta.npz", allow_pickle=True)
    stim = np.array([str(x) for x in meta["stimulus"]])
    ds = np.array([str(x) for x in meta["dataset"]])
    N = len(stim)
    audio = np.full((N, 768), np.nan, np.float32)
    cmap = cire_map()
    # CIRE: trial index -> wav file -> mean wav2vec2 frame
    n_ci = 0
    for i in np.where(ds == "cire")[0]:
        trial = int(stim[i].split(":")[1])
        w = cmap.get(trial)
        if w is None:
            continue
        p = CIRE / f"{w}.npy"
        if not p.exists():
            continue
        a = np.load(p).astype(np.float32)
        audio[i] = a.mean(0)
        n_ci += 1
    # KUL: stimulus -> wav2vec2 vector
    kw = np.load(M2 / "data" / "local75" / "kul_w2v.npy").astype(np.float32)
    ks = np.array([str(x) for x in np.load(M2 / "data" / "local75" / "kul_w2v_stim.npy", allow_pickle=True)])
    kmap = {s: j for j, s in enumerate(ks)}
    n_ku = 0
    for i in np.where(ds == "kul")[0]:
        j = kmap.get(stim[i])
        if j is not None:
            audio[i] = kw[j]; n_ku += 1
    mask = ~np.isnan(audio).any(1)
    np.save(M2 / "data" / "auditory_corpus" / "audio_targets.npy", audio)
    np.save(M2 / "data" / "auditory_corpus" / "audio_mask.npy", mask)
    summary = {"windows": int(N), "with_audio": int(mask.sum()), "cire": n_ci, "kul": n_ku,
               "cire_trial_to_wav": {str(k): v for k, v in sorted(cmap.items())}}
    (M2 / "data" / "auditory_corpus" / "audio_summary.json").write_text(
        json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "cire_trial_to_wav"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
