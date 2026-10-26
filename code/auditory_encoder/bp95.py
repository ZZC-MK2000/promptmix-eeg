"""95-d (19 channels x 5 bands) log bandpower for the corpus and NMED-T."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = ROOT.parent / "mainv2"
sys.path.insert(0, str(ROOT / "code" / "information"))
BANDS = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]


def bp95(X, fs=200):
    out = np.empty((len(X), 19, 5), np.float32)
    for i in range(0, len(X), 256):
        x = np.asarray(X[i:i + 256], np.float32)
        p = np.abs(np.fft.rfft(x, axis=-1)) ** 2
        f = np.fft.rfftfreq(x.shape[-1], d=1 / fs)
        for b, (lo, hi) in enumerate(BANDS):
            m = (f >= lo) & (f < hi)
            out[i:i + len(x), :, b] = np.log1p(p[..., m].mean(-1))
    return out.reshape(len(X), -1)


def main():
    W = np.load(M2 / "data" / "auditory_corpus" / "windows.npy", mmap_mode="r")
    np.save(M2 / "data" / "auditory_corpus" / "bp95.npy", bp95(W))
    from eval_finetune_local75 import nmed_windows
    X, song, subj = nmed_windows()
    np.save(ROOT / "results" / "auditory_encoder" / "nmed_bp95.npy", bp95(X))
    np.savez(ROOT / "results" / "auditory_encoder" / "nmed_bp95_meta.npz",
             song=song, subject=subj)
    print("saved corpus and nmed bp95", flush=True)


if __name__ == "__main__":
    main()
