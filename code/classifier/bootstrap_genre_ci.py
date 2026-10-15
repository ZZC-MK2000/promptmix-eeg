"""Bootstrap 95% CIs for the EEG->genre classifier's Cohen's kappa and balanced accuracy on
held-out SUBJECTS (resample rows with replacement), for BOTH the original 12-prompt library and the
corpus-matched library. Turns the headline "EEG carries song-genre info" into a claim with error
bars a reviewer can trust. CPU only.

Run: python src/train/bootstrap_genre_ci.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
DATA = ROOT / "data" / "eegcond"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
K = 12


def heldout_split(group, seed=0):
    uniq = np.unique(group); rng = np.random.default_rng(seed)
    vg = set(uniq[rng.permutation(len(uniq))[:max(1, int(round(len(uniq) * 0.1)))]].tolist())
    return np.array([g in vg for g in group])


def kappa(y, p):
    po = (y == p).mean()
    pe = sum(((y == c).mean()) * ((p == c).mean()) for c in range(K))
    return (po - pe) / (1 - pe + 1e-12)


def balanced(y, p):
    rec = [(p[y == c] == c).mean() for c in range(K) if (y == c).sum() > 0]
    return float(np.mean(rec))


def preds_for(ckpt, cond, va):
    from finetune_prompt_classifier import PromptClassifier
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    m = PromptClassifier(512, ck["args"].get("hidden", 256), K, 0.1); m.load_state_dict(ck["model"]); m.eval()
    with torch.no_grad():
        return m(torch.from_numpy(cond[va])).argmax(1).cpu().numpy()


def boot(y, p, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y)
    ks, bs = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb, pb = y[idx], p[idx]
        if len(np.unique(yb)) < 2:
            continue
        ks.append(kappa(yb, pb)); bs.append(balanced(yb, pb))
    return (np.percentile(ks, [2.5, 97.5]), np.percentile(bs, [2.5, 97.5]))


def main():
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    group = np.load(DATA / "group.npy").astype(np.int64)
    va = heldout_split(group)

    cases = [
        ("original 12-prompt lib", ROOT / "results/prompt_classifier_true/final.pt",
         np.load(ROOT / "results/prompt_classifier/true_oracle_all.npy").astype(np.int64)),
        ("corpus-matched lib", ROOT / "results/library_b/classifier.pt",
         np.load(ROOT / "results/library_b/oracle_all.npy").astype(np.int64)),
    ]
    out = {}
    for name, ckpt, oracle in cases:
        p = preds_for(ckpt, cond, va); y = oracle[va]
        k0 = kappa(y, p); b0 = balanced(y, p); acc = float((y == p).mean())
        (klo, khi), (blo, bhi) = boot(y, p)
        print(f"\n=== {name} (held-out rows={len(y)}) ===")
        print(f"  overall acc   = {acc:.4f}")
        print(f"  Cohen's kappa = {k0:.4f}  95% CI [{klo:.4f}, {khi:.4f}]")
        print(f"  balanced acc  = {b0:.4f}  95% CI [{blo:.4f}, {bhi:.4f}]")
        print(f"  chance kappa  = 0;  chance balanced = {1/K:.4f}")
        out[name] = {"overall": acc, "kappa": k0, "kappa_ci": [klo, khi],
                     "balanced": b0, "balanced_ci": [blo, bhi], "n": int(len(y))}
    import json
    json.dump(out, open(ROOT / "results/prompt_classifier_true/genre_ci.json", "w"), indent=2)
    print("\nsaved -> results/prompt_classifier_true/genre_ci.json")


if __name__ == "__main__":
    main()
