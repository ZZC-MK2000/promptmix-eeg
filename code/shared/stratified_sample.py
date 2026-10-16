"""Deterministic, coverage-checked row selection for anything that gets shown or published.

Replaces the `val_rows[:N]` prefix pattern, which silently returns "the first N rows of an array
ordered by subject and run" — i.e. one person, one song, with overlapping windows repeated. That
bug put 14 'heard vs generated' pairs on the demo page that were really 6 audio clips from a single
subject listening to a single song, and it made every own-vs-others statistic on that set invalid
because the baseline contained byte-identical copies of the target.

Selection rule (must be reported alongside any sample chosen here, per discipline rule 31):
  1. restrict to held-out groups (subject-level split, so no training leakage);
  2. one row per group (kills overlapping-window duplicates);
  3. greedy farthest-point spread over the row's true-audio embedding, so the chosen rows are
     maximally different in content rather than adjacent in index.

Run `python code/shared/stratified_sample.py` to print a worked example with its coverage report.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")


def heldout_mask(group, val_frac=0.1, seed=0):
    uniq = np.unique(group)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    vg = set(uniq[rng.permutation(len(uniq))[:n_val]].tolist())
    return np.array([int(g) in vg for g in group])


def diverse_rows(emb, cand, group, k, seed=0):
    """Pick k rows from `cand` (indices), one per group, maximally spread in `emb` (unit-norm)."""
    g_of = {}
    pool = []
    seen = set()
    for r in cand:
        g = int(group[r])
        if g in seen:
            continue
        seen.add(g)
        pool.append(int(r))
    pool = np.array(pool)
    if len(pool) == 0:
        return np.array([], dtype=np.int64)
    k = min(k, len(pool))
    rng = np.random.default_rng(seed)
    # start from a random pool member so the choice is reproducible but not index-biased
    chosen = [int(rng.choice(pool))]
    mind = 1.0 - emb[pool] @ emb[chosen[0]]
    for _ in range(k - 1):
        j = int(np.argmax(mind))
        chosen.append(int(pool[j]))
        mind = np.minimum(mind, 1.0 - emb[pool] @ emb[pool[j]])
    return np.array(sorted(chosen), dtype=np.int64)


def coverage_report(rows, group, emb, label=""):
    rows = np.asarray(rows)
    if len(rows) == 0:
        return {"n": 0}
    C = emb[rows] @ emb[rows].T
    off = C[~np.eye(len(rows), dtype=bool)]
    return {"n": int(len(rows)), "distinct_groups": int(len({int(group[r]) for r in rows})),
            "mean_pairwise_cos_of_stimuli": float(off.mean()), "min_pairwise_cos": float(off.min()),
            "label": label}


def load_val_embeddings():
    """(36036, 512) laion-CLAP of each row's TRUE audio, already in this repo."""
    e = np.load(ROOT / "results" / "prompt_classifier" / "true_emb_all.npy").astype(np.float32)
    return e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-8)


def main():
    group = np.load(ROOT / "data" / "eegcond" / "group.npy").astype(np.int64)
    cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)
    emb = load_val_embeddings()
    va = np.where(heldout_mask(group))[0]
    old = va[:14]
    new = diverse_rows(emb, va, group, 14, seed=0)
    for name, sel in [("val_rows[:14]  (the shipped bug)", old), ("diverse_rows(...,14)  (fixed)", new)]:
        c = coverage_report(sel, group, emb)
        print(f"{name:34} n={c['n']:3} groups={c['distinct_groups']:3} "
              f"stimulus mean pairwise cos={c['mean_pairwise_cos_of_stimuli']:.3f} "
              f"(min {c['min_pairwise_cos']:.3f})")
    print("\nA high 'mean pairwise cos of stimuli' means the showcase is one song on repeat; "
          "that is what the prefix bug produced.")


if __name__ == "__main__":
    sys.exit(main())
