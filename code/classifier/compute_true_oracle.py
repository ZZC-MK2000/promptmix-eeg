"""Compute the TRUE-audio oracle prompt label for every row, by decoding the EnCodec codes
back to audio and embedding with laion CLAP -- the same space the library means live in.

Why: finetune_prompt_classifier trained against cond_oracle = argmax_k cos(means[k], bridge(EEG)).
validate_mode_a_labels showed that label agrees with the TRUE-audio argmax only 16.7% (near chance),
because bridge(EEG) has poor ABSOLUTE alignment to the laion audio manifold (cos 0.19 vs 0.75).
The fix is to train EEG -> true_oracle directly. This script produces true_oracle for all rows.

Output: results/prompt_classifier/true_emb_all.npy   (N, 512) laion CLAP of decoded audio
        results/prompt_classifier/true_oracle_all.npy (N,) int64 in [0,12)

Run: python src/train/compute_true_oracle.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
LIB = ROOT / "results" / "gen_eegcond" / "library_radius"
OUT = ROOT / "results" / "prompt_classifier"
sys.path.insert(0, str(M2 / "src" / "generate"))

SR = 32000
K = 12


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    codes = np.load(DATA / "codes.npy")            # (N, 4, T) int16
    N = len(codes)
    print(f"codes {codes.shape}  N={N}  device={device}", flush=True)

    # library means (K, 512)
    lib_emb = np.load(LIB / "library_emb.npy").astype(np.float32)
    lib_owner = np.load(LIB / "library_owner.npy").astype(np.int64)
    means = unit(np.stack([unit(lib_emb[lib_owner == k].mean(0)) for k in range(K)]))

    from musicgen_clap_finetune import MusicGenCLAP, load_clap, clap_embed_audio
    base = MusicGenCLAP(device=device, use_lora=False)
    mg = base.mg
    mg.eval()
    clap, proc, csr = load_clap(device)

    true_emb = np.zeros((N, 512), np.float32)
    # DECODE ONE ROW AT A TIME: batched audio_encoder.decode() raises
    # "Expected one frame, got B" under transformers 5.17 (library_radius decoded per-row and
    # worked). Batching here only speeds the loop bookkeeping, not the model call.
    STEP = 1
    with torch.no_grad():
        for s in range(0, N, STEP):
            e = min(s + STEP, N)
            for j in range(s, e):
                cb = torch.from_numpy(codes[j].astype(np.int64)).unsqueeze(0).unsqueeze(0).to(device)
                audio = mg.audio_encoder.decode(cb, audio_scales=[None]).audio_values
                w = np.asarray(audio.cpu()).reshape(-1)
                true_emb[j] = clap_embed_audio(clap, proc, csr, w, device)
            if (s // STEP) % 500 == 0:
                el = time.time() - t0
                rem = el / max(e, 1) * (N - e)
                print(f"  {e}/{N}  [{el:.0f}s | rem {rem/60:.1f} min]", flush=True)

    true_emb_u = unit(true_emb)
    np.save(OUT / "true_emb_all.npy", true_emb)

    cos = true_emb_u @ means.T                       # (N, K)
    true_oracle = cos.argmax(1).astype(np.int64)
    np.save(OUT / "true_oracle_all.npy", true_oracle)

    best = cos.max(1)
    print(f"\ncos(true_audio, best prompt mean): mean={best.mean():.4f} "
          f"min={best.min():.4f} max={best.max():.4f}", flush=True)
    dist = np.bincount(true_oracle, minlength=K).tolist()
    print(f"true_oracle distribution: {dist}", flush=True)
    print(f"majority-class baseline acc = {max(dist)/N:.4f}", flush=True)
    s = np.sort(cos, axis=1)
    print(f"top1-top2 margin: mean={(s[:,-1]-s[:,-2]).mean():.4f}", flush=True)
    print(f"\ndone -> {OUT}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
