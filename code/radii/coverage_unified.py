"""One coverage table, one convention: every library is 6 clips per prompt, so atoms = 6 x prompts.

Reads only cached embeddings (no GPU, no generation), recomputes all five coverage readings for each
library, and asserts each matches the number already stored in that run's own JSON -- so this file,
not the four separate scripts, is what the figure and the page quote.

Why it exists: the coverage numbers came from four scripts at two different clips-per-prompt settings
(6 and 3), and the caption had been reading them as one K-sweep. They are not one sweep.

Run: python code/radii/coverage_unified.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
R = ROOT / "results"
OUT = R / "coverage_unified"


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def proj_simplex(v):
    u, css = np.sort(v)[::-1], np.cumsum(np.sort(v)[::-1])
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    return np.maximum(v - (css[rho] - 1) / (rho + 1), 0)


def fit_simplex(L, t, iters=600):
    eta = 1.0 / (np.linalg.norm(L, 2) ** 2 + 1e-8)
    w = np.full(L.shape[0], 1.0 / L.shape[0])
    for _ in range(iters):
        w = proj_simplex(w - eta * (L @ (L.T @ w - t)))
    return w


def measure(L, owner, K, T):
    """L (atoms,512) unit-norm, owner (atoms,) prompt index, T (targets,512) real windows."""
    means = unit(np.stack([L[owner == k].mean(0) for k in range(K)]))
    uni = unit(means.mean(0, keepdims=True))[0]
    glob = unit(L.mean(0, keepdims=True))[0]
    per = {
        "mixture": np.array([float(unit(L.T @ fit_simplex(L, t)) @ t) for t in T]),
        "best_single_clip": np.array([float((L @ t).max()) for t in T]),
        "best_prompt_mean": np.array([float((means @ t).max()) for t in T]),
        "uniform_over_prompts": uni @ T.T,
        "global_mean": glob @ T.T,
    }

    def paired(other):
        d = per["mixture"] - other
        return {"delta": float(d.mean()),
                "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))),
                "wins": int((d > 0).sum()), "n": int(len(d))}

    out = {"atoms": int(L.shape[0]), "prompts": int(K), "clips_per_prompt": int(L.shape[0] // K)}
    out.update({k: float(v.mean()) for k, v in per.items()})
    out["se"] = {k: float(v.std(ddof=1) / np.sqrt(len(v))) for k, v in per.items()}
    out["mixture_vs_best_single"] = paired(per["best_single_clip"])
    out["mixture_vs_uniform"] = paired(per["uniform_over_prompts"])
    out["mixture_vs_global_mean"] = paired(per["global_mean"])
    return out


CONFIGS = [
    ("original 12-prompt library", "gen_eegcond/library_radius", "radius.json", 12,
     lambda d: d["mixture_over_library"]["mean"]),
    ("corpus-matched 12-prompt library", "library_b", "summary.json", 12,
     lambda d: d["coverage"]["mixture"]),
    ("corpus-matched 24-prompt library", "coverage_k24", "coverage_k24.json", 24,
     lambda d: d["mixture"]),
    ("36-prompt pool (3 clips/prompt)", "coverage_k48", "coverage_k48.json", 36,
     lambda d: d["cov48"]),
]


def main():
    T = unit(np.load(R / "gen_eegcond/library_radius/true_emb.npy").astype(np.float32))
    print(f"{len(T)} held-out real 8-s windows as targets, 6 clips per prompt is the convention\n")
    rows = []
    for label, sub, json_name, K, stored_get in CONFIGS:
        d = R / sub
        L = unit(np.load(d / "library_emb.npy").astype(np.float32))
        o_path = d / "library_owner.npy"
        owner = (np.load(o_path).astype(int) if o_path.exists()
                 else np.repeat(np.arange(K), L.shape[0] // K))
        m = measure(L, owner, K, T)
        m["label"] = label
        m["on_convention"] = m["clips_per_prompt"] == 6
        stored = stored_get(json.load(open(d / json_name, encoding="utf-8")))
        assert abs(m["mixture"] - stored) < 2e-3, \
            f"{label}: recomputed {m['mixture']:.4f} != stored {stored:.4f}"
        rows.append(m)
        print(f"  {label:36} atoms={m['atoms']:4} mixture={m['mixture']:.4f} "
              f"== stored {stored:.4f}  OK")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"targets": len(T), "convention": "6 clips per prompt; atoms = 6 x prompts",
               "note": "the 36-prompt pool used 3 clips per prompt and is therefore not on the "
                       "same axis as the others; it is reported but excluded from the sweep.",
               "libraries": rows}, open(OUT / "table.json", "w", encoding="utf-8"), indent=2)
    print(f"\n{'library':38} {'atoms':>6} {'mix':>7} {'best clip':>10} {'prompt mean':>12} "
          f"{'uniform':>8} {'global':>7}  {'vs uniform':>11} {'vs global':>10}")
    for m in rows:
        print(f"{m['label']:38} {m['atoms']:6} {m['mixture']:7.4f} {m['best_single_clip']:10.4f} "
              f"{m['best_prompt_mean']:12.4f} {m['uniform_over_prompts']:8.4f} "
              f"{m['global_mean']:7.4f}  "
              f"+{m['mixture_vs_uniform']['delta']:.3f} t{m['mixture_vs_uniform']['t']:5.1f}  "
              f"+{m['mixture_vs_global_mean']['delta']:.3f} t{m['mixture_vs_global_mean']['t']:5.1f}")


if __name__ == "__main__":
    main()
