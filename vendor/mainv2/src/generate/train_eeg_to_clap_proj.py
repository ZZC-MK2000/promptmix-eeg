"""Train EEG→CLAP bridge: aldm CLAP space → laion CLAP space.

MusicGen is fine-tuned with laion CLAP embeddings as conditioning.  The existing EEG
alignment produces embeddings in the AudioLDM-CLAP space (aldm).  At inference we must
map an aldm-space vector into laion-space so the generator sees in-distribution input.

We fit that map by ridge regression on 40336 paired audio embeddings already cached in
gate_nmedh14_ov2_lfix.npz (laion) and gate_nmedh14_ov2_a.npz (aldm) -- both are the
AudioCLAP of the SAME audio windows, so the row-aligned pair is a training example.

Split is held-out by subject (not by row) so the reported val cosine is a real
generalisation estimate.

The docs (retarget_claps.py) already measured that this bridge caps stage-2 retrieval
near 0.5607 for a *perfect* laion input -- so the bridge is known-lossy.  It is still
the cheapest thing to test tonight after MusicGen training ends; a full retrain of the
EEG→laion alignment is a bigger move.

Run:  python src/generate/train_eeg_to_clap_proj.py
Out:  results/eeg_to_clap_proj.pt   (dict with 'weight', 'bias', 'meta')
      results/eeg_to_clap_proj_meta.json
"""
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

NPZ_L = DATA / "gate_nmedh14_ov2_lfix.npz"
NPZ_A = DATA / "gate_nmedh14_ov2_a.npz"
OUT_PT = RESULTS / "eeg_to_clap_proj.pt"
OUT_META = RESULTS / "eeg_to_clap_proj_meta.json"


def main():
    t0 = time.time()
    print(f"Loading paired caches...")
    zl = np.load(NPZ_L, allow_pickle=True)
    za = np.load(NPZ_A, allow_pickle=True)
    X = za["target"].astype(np.float32)   # aldm
    Y = zl["target"].astype(np.float32)   # laion
    subj = zl["subject"]
    song = zl["song"]
    run = zl["run"]
    n = len(X)
    assert X.shape == Y.shape == (n, 512), f"shape mismatch: {X.shape} vs {Y.shape}"
    print(f"  {n} pairs, {len(np.unique(subj))} subjects, {len(np.unique(song))} songs")

    # Held-out subjects (round-robin by sorted unique subject id)
    subjects = np.unique(subj)
    n_held = max(1, int(round(len(subjects) * 0.15)))
    held_set = set(subjects[::max(1, len(subjects)//n_held)][:n_held].tolist())
    tr_mask = np.array([s not in held_set for s in subj])
    va_mask = ~tr_mask
    print(f"  train={tr_mask.sum()}, val={va_mask.sum()} (held subjects: {sorted(held_set)})")

    Xtr, Ytr = X[tr_mask], Y[tr_mask]
    Xva, Yva = X[va_mask], Y[va_mask]

    # Normalise inputs to unit sphere before ridge (helps conditioning)
    def unit(v):
        return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    Xtr_u, Ytr_u = unit(Xtr), unit(Ytr)
    Xva_u, Yva_u = unit(Xva), unit(Yva)

    best = None
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0]:
        Xt = Xtr_u.T
        G = Xt @ Xtr_u + lam * np.eye(512, dtype=np.float32)
        B = Xt @ Ytr_u
        W = np.linalg.solve(G, B).T           # (512, 512) applied as x @ W.T
        pred_va = Xva_u @ W.T
        pred_va = unit(pred_va)
        cos = float((pred_va * Yva_u).sum(-1).mean())
        # Also compute how much the mapping shrinks variance (a rough sanity check)
        pred_norm = float(np.linalg.norm(Xva_u @ W.T, axis=-1).mean())
        print(f"  ridge lam={lam:g}: val cos={cos:.4f}  mean||pred||={pred_norm:.4f}")
        if best is None or cos > best["cos"]:
            best = {"lam": lam, "W": W, "cos": cos, "pred_norm": pred_norm}

    W = best["W"]
    print(f"\nChose lam={best['lam']:g}, val cos={best['cos']:.4f}")

    # Save in the exact format _eeg_to_clap() expects:
    #   eeg @ W.T + b, then L2-normalise
    proj = {
        "weight": torch.from_numpy(W),
        "bias": torch.zeros(512),
    }
    torch.save(proj, OUT_PT)

    meta = {
        "n_pairs": int(n),
        "n_train": int(tr_mask.sum()),
        "n_val": int(va_mask.sum()),
        "held_subjects": sorted([str(s) for s in held_set]),
        "ridge_lam": best["lam"],
        "val_cos_after_bridge": best["cos"],
        "val_pred_norm": best["pred_norm"],
        "raw_cos_no_bridge": float((unit(Xva) * Yva_u).sum(-1).mean()),
        "elapsed_sec": round(time.time() - t0, 1),
        "source_caches": [str(NPZ_L.name), str(NPZ_A.name)],
        "caveat": "aldm↔laion towers near-orthogonal on same audio (mean cos ~0.03); "
                  "bridge is a linear approximation and structurally lossy per "
                  "nmedh14_retarget_claps.py docs (caps 14-way at 0.5607 with perfect input).",
    }
    OUT_META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {OUT_PT}  ({OUT_PT.stat().st_size/1e6:.2f} MB)")
    print(f"Meta:  {OUT_META}")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
