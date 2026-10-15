"""Fast true_oracle: song-matched prompt label for every cond.npy row, WITHOUT decoding audio.

Key insight: build_eegcond_pairs.py produced cond.npy as
    cond_all = unit(eeg_aldm @ W.T + b)        # bridge(EEG), laion 512-d
    cond.npy  = cond_all[keep_rows]            # a SUBSET of the 40336 cache rows
and the cache `target` column IS the laion-CLAP of the matched TRUE audio window (build script
verified recovered-window-CLAP vs target at match cos 0.9611). So true_oracle = argmax_k
cos(means[k], target) is available directly from the cache -- no EnCodec decode needed.

We align cache rows to cond.npy rows by exact float32 byte match, then write
results/prompt_classifier/true_oracle_all.npy (36036,) aligned with cond.npy/group.npy.

Run: python src/train/compute_true_oracle_fast.py
"""
import json
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
K = 12


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    t0 = time.time()
    cond = np.load(DATA / "cond.npy").astype(np.float32)      # (36036, 512)
    group = np.load(DATA / "group.npy").astype(np.int64)
    N = len(cond)

    lib_emb = np.load(LIB / "library_emb.npy").astype(np.float32)
    lib_owner = np.load(LIB / "library_owner.npy").astype(np.int64)
    means = unit(np.stack([unit(lib_emb[lib_owner == k].mean(0)) for k in range(K)]))

    # cache: target = laion CLAP of true audio; eeg_aldm + bridge -> cond_all
    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    target = z["target"].astype(np.float32)                    # (40336, 512)
    eeg_aldm = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    proj = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu", weights_only=True)
    W = proj["weight"].numpy().astype(np.float32); b = proj["bias"].numpy().astype(np.float32)
    cond_all = unit(eeg_aldm @ W.T + b).astype(np.float32)      # (40336, 512)
    print(f"cache {cond_all.shape}  target {target.shape}  cond.npy {cond.shape}", flush=True)

    # exact float32 byte match: cache row -> index
    key_all = {cond_all[i].tobytes(): i for i in range(len(cond_all))}
    idx = np.full(N, -1, np.int64)
    for j in range(N):
        i = key_all.get(cond[j].tobytes(), -1)
        idx[j] = i
    matched = idx >= 0
    cov = float(matched.mean())
    print(f"exact-match coverage of cond.npy rows in cache: {cov:.4f} ({int(matched.sum())}/{N})",
          flush=True)

    # true audio embedding per cond row (from cache target)
    te = np.zeros((N, 512), np.float32)
    te[matched] = target[idx[matched]]
    cos = unit(te) @ means.T
    true_oracle = cos.argmax(1).astype(np.int64)
    true_oracle[~matched] = -1                                  # mark unmatched

    dist = np.bincount(true_oracle[matched], minlength=K).tolist()
    best = cos[matched].max(1)
    majority = max(dist) / int(matched.sum())
    print(f"\ntrue_oracle distribution (12): {dist}", flush=True)
    print(f"majority-class baseline acc = {majority:.4f}", flush=True)
    print(f"cos(true_audio, best prompt mean) mean={best.mean():.4f} "
          f"min={best.min():.4f} max={best.max():.4f}", flush=True)
    srt = np.sort(cos[matched], axis=1)
    print(f"top1-top2 margin mean={(srt[:,-1]-srt[:,-2]).mean():.4f}", flush=True)

    if cov < 0.95:
        print("\n!! coverage < 0.95 — true_oracle_all.npy written with -1 for unmatched; the "
              "classifier loader must drop unmatched rows (see note). Falling back not automated.",
              flush=True)

    np.save(OUT / "true_oracle_all.npy", true_oracle)
    np.save(OUT / "true_emb_all.npy", te)
    json.dump({"coverage": cov, "matched": int(matched.sum()), "N": N,
               "distribution": dist, "majority_baseline": float(majority),
               "cos_mean": float(best.mean()), "top2_margin": float((srt[:,-1]-srt[:,-2]).mean()),
               "seconds": round(time.time()-t0, 1)}, open(OUT / "true_oracle_meta.json", "w"), indent=2)
    print(f"\nsaved -> {OUT/'true_oracle_all.npy'}  [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
