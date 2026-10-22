"""Direct EEG→laion-CLAP alignment (no bridge).

The pipeline used to go:
    EEG → CLaBraM encoder → stage-1 align to AudioLDM CLAP (aldm) → (bridge fit today) → laion CLAP → MusicGen
This script fits a direct map:
    EEG → CLaBraM encoder (cached `labram` in gate_nmedh14_ov2_lfix.npz, 200d) → laion CLAP (512d)

Motivation: nmedh14_retarget_claps.py documents that the aldm↔laion towers are
near-orthogonal (mean cos ~0.03 on same audio), so routing through aldm costs
information. Aligning stage-1 directly to laion is architecturally cleaner.

Fit is ridge regression held-out by subject (matching the bridge's split so the
two are comparable). Emits two artifacts:
  - results/eeg_labram_to_laion_proj.pt   (weight 512x200 + bias 512)
  - results/eeg_labram_to_laion_meta.json

Also prints a head-to-head val cosine comparison against the bridge (which reads
eeg_emb_aldm_s0.npy → aldm→laion linear map).
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data"
RESULTS = ROOT / "results"

Z = DATA / "gate_nmedh14_ov2_lfix.npz"
OUT_PT = RESULTS / "eeg_labram_to_laion_proj.pt"
OUT_META = RESULTS / "eeg_labram_to_laion_meta.json"


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)


def fit_ridge(X, Y, lam):
    """x @ W.T → y; return (W, b)."""
    Xt = X.T
    G = Xt @ X + lam * np.eye(X.shape[1], dtype=np.float32)
    B = Xt @ Y
    W = np.linalg.solve(G, B).T
    return W.astype(np.float32)


def eval_val(Xval, Yval_true_u, W):
    pred = unit(Xval @ W.T)
    return float((pred * Yval_true_u).sum(-1).mean())


def main():
    t0 = time.time()
    z = np.load(Z, allow_pickle=True)
    X = z["labram"].astype(np.float32)          # (40336, 200)
    Y = z["target"].astype(np.float32)          # (40336, 512) laion
    subj = z["subject"]

    subjects = np.unique(subj)
    n_held = max(1, int(round(len(subjects) * 0.15)))
    held_set = set(subjects[::max(1, len(subjects)//n_held)][:n_held].tolist())
    tr = np.array([s not in held_set for s in subj])
    va = ~tr
    print(f"labram {X.shape}, laion {Y.shape}, {tr.sum()} train / {va.sum()} val (held subjects {sorted(held_set)})")

    # Note: labram 200-d is not unit-norm; center+scale lightly so ridge doesn't get lost.
    mu = X.mean(0); sd = X.std(0) + 1e-6
    Xz = (X - mu) / sd
    Xz_tr, Xz_va = Xz[tr], Xz[va]
    Y_u = unit(Y)
    Y_tr_u, Y_va_u = Y_u[tr], Y_u[va]

    best = None
    for lam in [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]:
        W = fit_ridge(Xz_tr, Y_tr_u, lam)
        c = eval_val(Xz_va, Y_va_u, W)
        print(f"  ridge lam={lam:g}: val cos = {c:.4f}")
        if best is None or c > best["cos"]:
            best = {"lam": lam, "W": W, "cos": c}

    W = best["W"]
    print(f"\nBest lam={best['lam']:g}, val cos={best['cos']:.4f}")

    # Comparison: what does the current aldm-bridge pipeline achieve on the same val rows?
    eeg_aldm = np.load(DATA / "eeg_emb_aldm_s0.npy").astype(np.float32)
    proj = torch.load(RESULTS / "eeg_to_clap_proj.pt", map_location="cpu", weights_only=True)
    Wb = proj["weight"].numpy()
    bridge_pred = unit(unit(eeg_aldm[va]) @ Wb.T)
    bridge_cos = float((bridge_pred * Y_va_u).sum(-1).mean())
    print(f"Current bridge path (EEG→aldm→ridge→laion) on same val: cos = {bridge_cos:.4f}")

    torch.save({"weight": torch.from_numpy(W), "bias": torch.zeros(512),
                "mu": torch.from_numpy(mu), "sd": torch.from_numpy(sd)}, OUT_PT)
    meta = {
        "n_pairs": int(tr.sum() + va.sum()),
        "n_train": int(tr.sum()), "n_val": int(va.sum()),
        "held_subjects": sorted([str(s) for s in held_set]),
        "ridge_lam": best["lam"],
        "val_cos_direct": best["cos"],
        "val_cos_bridge_current": bridge_cos,
        "delta_direct_minus_bridge": best["cos"] - bridge_cos,
        "elapsed_sec": round(time.time() - t0, 2),
        "input_field": "labram (200-d CLaBraM encoder output cached in gate_nmedh14_ov2_lfix.npz)",
        "target_field": "target (laion CLAP of paired audio, SR-fixed)",
        "note": "Both val numbers share the same held-out subjects and target rows. "
                "Direct ridge fits a fresh linear map from labram to laion without going through aldm.",
    }
    OUT_META.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {OUT_PT} ({OUT_PT.stat().st_size/1e6:.2f} MB) and {OUT_META}")


if __name__ == "__main__":
    main()
