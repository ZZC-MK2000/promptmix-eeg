"""Leakage audit for the AEB split suite.

The random-row split is retained as an intentional upper bound and is expected
to leak both subject and adjacent windows; block and subject splits must not
overlap at their declared unit.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SPLITS = HERE.parent / "splits"
OUT = HERE.parent / "manifests" / "split_audit.json"


def nmed(path):
    d = np.load(path, allow_pickle=True)
    return np.asarray(d["song"]), np.asarray(d["subject"]).astype(str)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--nmed", default=str(HERE.parent / "data" / "nmed.npz"))
    args = ap.parse_args()
    song, subj = nmed(args.nmed)
    block = np.empty(len(subj), dtype=object)
    for s in sorted(set(subj.tolist())):
        for c in sorted(set(song.tolist())):
            idx = np.where((subj == s) & (song == c))[0]
            for j, i in enumerate(idx):
                block[i] = f"{s}:{c}:{j // 5}"
    units = {"random_rows": np.arange(len(subj)), "blocks": block, "subjects": subj}
    d = json.load(open(SPLITS / "nmedt_folds.json", encoding="utf-8"))
    audit = {"nmedt": {}, "lodo": {}}
    for name, split_list in d["splits"].items():
        audit["nmedt"][name] = []
        for sp in split_list:
            tr, va, te = map(np.asarray, (sp["train"], sp["val"], sp["test"]))
            u = units[name]
            def overlap(a, b):
                return int(len(set(u[a].tolist()) & set(u[b].tolist())))
            audit["nmedt"][name].append({
                "seed": sp["seed"],
                "n": [len(tr), len(va), len(te)],
                "row_overlap_train_test": int(len(set(tr.tolist()) & set(te.tolist()))),
                "subject_overlap_train_test": int(len(set(subj[tr].tolist()) & set(subj[te].tolist()))),
                "unit_overlap_train_test": overlap(tr, te),
                "unit_overlap_train_val": overlap(tr, va),
            })
    lodo = json.load(open(SPLITS / "leave_one_dataset_out.json", encoding="utf-8"))
    meta = np.load(ROOT / "mainv2" / "data" / "auditory_corpus_six" / "meta.npz",
                   allow_pickle=True)
    ds = np.asarray(meta["dataset"]).astype(str)
    su = np.asarray(meta["subject"]).astype(str)
    for sp in lodo["splits"]:
        tr, va, te = map(np.asarray, (sp["train"], sp["val"], sp["test"]))
        audit["lodo"][sp["held_out_dataset"]] = {
            "n": [len(tr), len(va), len(te)],
            "dataset_overlap_train_test": int(len(set(ds[tr].tolist()) & set(ds[te].tolist()))),
            "subject_overlap_train_test": int(len(set(su[tr].tolist()) & set(su[te].tolist()))),
            "datasets_train": sorted(set(ds[tr].tolist())),
            "dataset_test": sorted(set(ds[te].tolist())),
        }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2)[:3000], flush=True)
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
