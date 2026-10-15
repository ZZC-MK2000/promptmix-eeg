"""Build an 8-second FMA index consistent with the EEG windows (for mixed-condition training).

Existing FMA cache is 10 s (codes 4,500 and laion embeddings from a 10 s span). Our EEG pairs
are 8 s (4,400), so mixing them would train the interface on two different spans. Here we
re-cut every segment to exactly 8.0 s and compute BOTH the Encodec codes and the laion CLAP
embedding from that same span, so (condition -> target) pairs are internally consistent.

Output: mainv3/data/fma8s/{cond.npy (M,512) float32, codes.npy (M,4,400) int16, meta.json}

Run: python src/train/build_fma8s_index.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
FMA = M2.parent / "data_external" / "fma_small" / "fma_small"
OUT = ROOT / "data" / "fma8s"
sys.path.insert(0, str(M2 / "src" / "generate"))

SR = 32000
WIN = 8.0
CLAP_SR = 48000


def main():
    from musicgen_clap_finetune import (MusicGenCLAP, load_clap, clap_embed_audio)

    OUT.mkdir(parents=True, exist_ok=True)
    meta = json.loads((M2 / "data" / "fma_prepared" / "fma_clap_meta.json").read_text(encoding="utf-8"))
    print(f"{len(meta)} FMA segments; cutting each to {WIN}s at its start_sec")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mg = MusicGenCLAP(device=device)
    mg.mg.audio_encoder.eval()
    clap, proc, csr = load_clap(device)

    import librosa

    C, K = [], []
    t0 = time.time()
    skipped = 0
    for i, m in enumerate(meta):
        p = FMA / m["file"]
        if not p.exists():
            skipped += 1
            continue
        try:
            w, _ = librosa.load(str(p), sr=SR, mono=True,
                                offset=float(m["start_sec"]), duration=WIN + 0.5)
        except Exception:
            skipped += 1
            continue
        n = int(WIN * SR)
        if len(w) < n:
            w = np.pad(w, (0, n - len(w)))
        seg = w[:n].astype(np.float32)
        if float(np.abs(seg).max()) < 1e-4:
            skipped += 1
            continue
        with torch.no_grad():
            codes = mg.mg.audio_encoder.encode(
                torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).to(device)).audio_codes[0]
        codes = np.asarray(codes.cpu()).reshape(codes.shape[-2], codes.shape[-1]).astype(np.int16)
        if codes.shape[1] != int(WIN * 50):
            skipped += 1
            continue
        # same span, but at the CLAP model's own rate
        w48, _ = librosa.load(str(p), sr=CLAP_SR, mono=True,
                              offset=float(m["start_sec"]), duration=WIN + 0.5)
        w48 = w48[:int(WIN * CLAP_SR)].astype(np.float32)
        if len(w48) < int(4 * CLAP_SR):
            skipped += 1
            continue
        emb = clap_embed_audio(clap, proc, CLAP_SR, w48, device)
        C.append(emb.astype(np.float32))
        K.append(codes)
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  [{i+1}/{len(meta)}] {el:.0f}s  剩 {(el/(i+1))*(len(meta)-i-1)/60:.1f} 分钟",
                  flush=True)

    C = np.stack(C); K = np.stack(K)
    np.save(OUT / "cond.npy", C)
    np.save(OUT / "codes.npy", K)
    (OUT / "meta.json").write_text(json.dumps(
        {"n_segments": int(len(C)), "skipped": int(skipped), "win_sec": WIN,
         "code_frames": int(K.shape[2]), "cond_norm_mean": float(np.linalg.norm(C, axis=1).mean()),
         "seconds": round(time.time() - t0, 1)}, indent=2), encoding="utf-8")
    print(f"saved {OUT}: cond {C.shape} codes {K.shape} skipped={skipped} "
          f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
