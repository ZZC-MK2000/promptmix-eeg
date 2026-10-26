"""Append SparrKULee (speech) to the five-dataset auditory corpus.

Sparr 64-channel recordings are mapped to the 19 standard 10-20 channels using
the BioSemi64 order, resampled 120 -> 200 Hz, and cut into non-overlapping 8 s
windows. Only train/val files are used; the Sparr test split is never touched.
"""
import json
import sys
import time
from pathlib import Path

import mne
import numpy as np
from scipy.signal import resample_poly

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from sparr_scale import list_files  # noqa: E402

SRC = ROOT.parent / "mainv2" / "data" / "auditory_corpus"
OUT = ROOT.parent / "mainv2" / "data" / "auditory_corpus_six"
WIN = 1600
CH = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
      "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
BS64 = mne.channels.make_standard_montage("biosemi64").ch_names
IDX19 = [BS64.index(c) for c in CH]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    src_w = np.load(SRC / "windows.npy", mmap_mode="r")
    src_m = np.load(SRC / "meta.npz", allow_pickle=True)
    recs = [r for r in list_files("D:/datasets/sparrkulee/split_data")
            if r["split"] in ("train", "val")]
    counts = []
    for r in recs:
        with open(r["path"], "rb") as f:
            ver = np.lib.format.read_magic(f)
            sh, _, _ = (np.lib.format.read_array_header_1_0(f) if ver == (1, 0)
                        else np.lib.format.read_array_header_2_0(f))
        n200 = int(round(sh[0] * 200 / 120))
        counts.append(n200 // WIN)
    n_sp = int(sum(counts))
    n_tot = len(src_w) + n_sp
    print(f"source {len(src_w)} + sparr {n_sp} = {n_tot} windows "
          f"({n_tot * 8 / 3600:.1f} h)", flush=True)
    dst = np.lib.format.open_memmap(OUT / "windows.npy", mode="w+",
                                    dtype=np.float16, shape=(n_tot, 19, WIN))
    dst[:len(src_w)] = src_w[:]
    subj = [str(x) for x in src_m["subject"]]
    stim = [str(x) for x in src_m["stimulus"]]
    dset = [str(x) for x in src_m["dataset"]]
    win = [int(x) for x in src_m["win"]]
    t0 = time.time(); pos = len(src_w)
    for i, (r, nw) in enumerate(zip(recs, counts)):
        a = np.load(r["path"])[:, IDX19].T.astype(np.float32)
        x = resample_poly(a, 200, 120, axis=1)
        x = np.clip(x, -300, 300)
        for j in range(nw):
            dst[pos + j] = x[:, j * WIN:(j + 1) * WIN].astype(np.float16)
            subj.append("sparr:" + r["subject"])
            stim.append("sparr:" + r["story"])
            dset.append("sparr")
            win.append(j)
        pos += nw
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(recs)} {time.time()-t0:.0f}s", flush=True)
    dst.flush(); del dst
    np.savez(OUT / "meta.npz",
             subject=np.array(subj), stimulus=np.array(stim),
             dataset=np.array(dset), win=np.array(win, dtype=np.int64))
    summary = {"source_windows": int(len(src_w)), "sparr_windows": n_sp,
               "total_windows": n_tot, "total_hours": n_tot * 8 / 3600,
               "sparr_subjects": len({r["subject"] for r in recs}),
               "sparr_files": len(recs), "channel_map": "BioSemi64->19 10-20",
               "sr": 200, "win_seconds": 8, "split": "train+val"}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
