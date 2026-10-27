"""Build the fixed AEB split suite.

Splits are index lists into the local NMED-T 8-s window cache (7020 rows) and
the six-dataset cache (69108 rows). No raw EEG is written. Run from a checkout
with the local caches available; see configs/datasets.yaml.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT = HERE.parent / "splits"
MAN = HERE.parent / "manifests"


def nmed(path):
    d = np.load(path, allow_pickle=True)
    X = d["X"] if "X" in d else d["features"]
    return np.asarray(X), np.asarray(d["song"]), np.asarray(d["subject"]).astype(str)


def split_units(n, seed, frac=(0.70, 0.15, 0.15)):
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    a = int(frac[0] * n)
    b = a + int(frac[1] * n)
    return p[:a], p[a:b], p[b:]


def expand(selected, unit_ids):
    return np.where(np.isin(unit_ids, selected))[0].tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmed", default=str(HERE.parent / "data" / "nmed.npz"))
    ap.add_argument("--six", default=str(HERE.parent / "data" / "auditory_corpus_six"))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True); MAN.mkdir(parents=True, exist_ok=True)

    X, song, subj = nmed(args.nmed)
    subj = subj.astype(str)
    n = len(X)
    # contiguous 5-window blocks within each subject-song stream
    block = np.empty(n, dtype=object)
    for s in sorted(set(subj.tolist())):
        for c in sorted(set(song.tolist())):
            idx = np.where((subj == s) & (song == c))[0]
            for j, i in enumerate(idx):
                block[i] = f"{s}:{c}:{j // 5}"
    block = np.array(block.tolist())
    systems = {"random_rows": np.arange(n), "blocks": block,
               "subjects": subj}
    suite = {"dataset": "NMED-T", "rows": n, "seeds": [0, 1, 2],
             "note": "index lists into the 7020-row 8-s cache; random_rows intentionally leaks adjacent windows/subjects",
             "splits": {}}
    for name, unit_ids in systems.items():
        units = np.unique(unit_ids)
        suite["splits"][name] = []
        for seed in (0, 1, 2):
            tr_u, va_u, te_u = split_units(len(units), seed)
            tr = expand(units[tr_u], unit_ids)
            va = expand(units[va_u], unit_ids)
            te = expand(units[te_u], unit_ids)
            suite["splits"][name].append({"seed": seed,
                                          "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                                          "train": tr, "val": va, "test": te})
    (OUT / "nmedt_folds.json").write_text(json.dumps(suite), encoding="utf-8")

    # leave-one-dataset-out over the six-dataset cache
    meta = np.load(Path(args.six) / "meta.npz", allow_pickle=True)
    ds = np.asarray(meta["dataset"]).astype(str)
    su = np.asarray(meta["subject"]).astype(str)
    lodo = {"datasets": sorted(set(ds.tolist())), "splits": []}
    for hold in sorted(set(ds.tolist())):
        tr = np.where(ds != hold)[0]
        te = np.where(ds == hold)[0]
        train_subs = np.unique(su[tr]); rng = np.random.default_rng(0)
        p = rng.permutation(len(train_subs)); nv = max(1, int(0.1 * len(p)))
        va_subs = set(train_subs[p[:nv]].tolist())
        va = np.array([i for i in tr if su[i] in va_subs])
        tr = np.array([i for i in tr if su[i] not in va_subs])
        lodo["splits"].append({"held_out_dataset": hold,
                               "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                               "train": tr.tolist(), "val": va.tolist(), "test": te.tolist()})
    (OUT / "leave_one_dataset_out.json").write_text(json.dumps(lodo), encoding="utf-8")

    # coverage manifest
    cov = {}
    for d in sorted(set(ds.tolist())):
        m = ds == d
        cov[d] = {"subjects": int(len(set(su[m].tolist()))), "windows": int(m.sum()),
                  "hours": round(int(m.sum()) * 8 / 3600, 1)}
    cov["total"] = {"subjects": int(len(set(su.tolist()))), "windows": int(len(ds)),
                    "hours": round(len(ds) * 8 / 3600, 1)}
    (MAN / "coverage.json").write_text(json.dumps(cov, indent=2), encoding="utf-8")
    print("wrote", OUT / "nmedt_folds.json", OUT / "leave_one_dataset_out.json",
          MAN / "coverage.json", flush=True)


if __name__ == "__main__":
    main()
