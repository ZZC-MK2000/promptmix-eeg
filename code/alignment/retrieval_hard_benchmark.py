"""Harder retrieval benchmark for the ALIGNMENT part (fixes the "84% is only 14-way" weakness).

Instead of "pick the right song out of 14", rank the true song against a growing pool of DISTRACTOR
songs (real FMA clips) and report top-1 accuracy vs pool size. A number that survives a large pool
is credible; one that collapses tells us the real ceiling.

  query   = bridge(EEG) window, held-out subjects (no subject leakage)
  gallery = 14 true-song centroids (laion-CLAP of the heard audio, per song) + D FMA distractor clips
  metric  = top-1: is the true song the nearest gallery item? vs random baseline 1/(14+D).

Run: python code/alignment/retrieval_hard_benchmark.py   (from promptmix-eeg/)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = Path("D:/Projects/CLaBraM/promptmix-eeg/data/eegcond")
FMA = Path("D:/Projects/CLaBraM/promptmix-eeg/data/fma8s")
OUT = ROOT / "results" / "retrieval_hard"
sys.path.insert(0, str(M2 / "src" / "generate"))


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    # queries: bridge(EEG) + recover song label via exact-bytes match to the cache
    cond = np.load(DATA / "cond.npy").astype(np.float32)          # (36036,512)
    group = np.load(DATA / "group.npy").astype(np.int64)          # subject
    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    target = z["target"].astype(np.float32)                       # (40336,512) laion of true audio
    song = z["song"].astype(int)
    eeg_aldm = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    proj = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu", weights_only=True)
    W = proj["weight"].numpy().astype(np.float32); b = proj["bias"].numpy().astype(np.float32)
    cond_all = unit(eeg_aldm @ W.T + b).astype(np.float32)
    key = {cond_all[i].tobytes(): i for i in range(len(cond_all))}
    q_song = np.full(len(cond), -1, np.int64)
    for j in range(len(cond)):
        i = key.get(cond[j].tobytes(), -1)
        if i >= 0:
            q_song[j] = song[i]
    ok = q_song >= 0
    print(f"queries with song label: {ok.sum()}/{len(cond)}", flush=True)

    # held-out subjects (same 10% group split, seed 0)
    uniq = np.unique(group); rng = np.random.default_rng(0)
    vg = set(uniq[rng.permutation(len(uniq))[:max(1, int(round(len(uniq) * 0.1)))]].tolist())
    va = np.array([g in vg for g in group]) & ok
    Q = unit(cond[va]); y = q_song[va]
    print(f"held-out queries: {len(Q)} rows, {len(set(group[va]))} subjects", flush=True)

    # gallery: 14 true-song centroids from cache audio targets
    songs = np.arange(14)
    cent = unit(np.stack([unit(target[song == s]).mean(0) for s in songs]))   # (14,512)

    # distractors: FMA clips, unit-normalized
    dist = unit(np.load(FMA / "cond.npy").astype(np.float32))                  # (4132,512)

    pools = [0, 36, 86, 486, 2000, len(dist)]
    print(f"\n=== top-1 song retrieval vs candidate-pool size ({len(Q)} held-out queries) ===")
    print(f"{'pool':>6} {'N_distract':>10} {'top1':>8} {'random':>8} {'ratio':>7}")
    rows = []
    for D in pools:
        if D > 0:
            dr = dist[np.random.default_rng(7).permutation(len(dist))[:D]]
            G = np.vstack([cent, dr])
        else:
            G = cent
        sims = Q @ G.T                       # (nq, 14+D)
        pred = sims.argmax(1)
        acc = float((pred == y).mean())     # true song centroid index == its song id (0..13)
        rand = 1.0 / G.shape[0]
        print(f"{G.shape[0]:>6} {D:>10} {acc:>8.4f} {rand:>8.4f} {acc/rand:>7.1f}x", flush=True)
        rows.append({"pool": int(G.shape[0]), "n_distract": int(D), "top1": acc, "random": rand})
    json.dump({"n_queries": int(len(Q)), "n_subjects": int(len(set(group[va]))),
               "curve": rows, "seconds": round(time.time() - t0, 1)},
              open(OUT / "retrieval_hard.json", "w"), indent=2)
    print(f"\nsaved -> {OUT/'retrieval_hard.json'}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
