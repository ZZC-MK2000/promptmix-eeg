"""Evaluate the subject-held-out retraining folds.

Reads, for each fold, the embeddings saved by run_subject_holdout.sh and scores the
held-out subjects' aligned8 rows against the 14 song prototypes (and the 10 NMED-T
prototypes), with chance and shuffled-pairing controls. Also pulls the CE-head accuracy
from mainv2/results/align_ladder_result.json.

Run after run_subject_holdout.sh:
    python code/information/eval_subject_holdout.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = ROOT.parent / "mainv2"
OUT = ROOT / "results" / "subject_holdout"
GATE = M2 / "data" / "gate_nmedh14_ov2_a.npz"
RES = M2 / "results" / "align_ladder_result.json"
FOLDS = {
    0: {"test": ["02", "03", "04", "05"], "val": ["06", "07"]},
    1: {"test": ["06", "07", "08", "09"], "val": ["04", "05"]},
    2: {"test": ["10", "11", "12", "13"], "val": ["06", "07"]},
    3: {"test": ["14", "15", "16", "17"], "val": ["08", "09"]},
    4: {"test": ["19", "20", "21", "23"], "val": ["10", "11"]},
}


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(GATE, allow_pickle=True)
    song = z["song"].astype(int); subj = z["subject"].astype(str)
    al = z["aligned8"].astype(bool)
    proto = unit(z["proto"].astype(np.float32))
    proto_song = z["proto_songs"].astype(int)
    assert (proto_song == np.arange(len(proto_song))).all(), "proto_songs is not identity"
    heads = json.loads(RES.read_text(encoding="utf-8")) if RES.exists() else {}

    out = {"folds": {}, "pooled": {}}
    pooled = {"y14": [], "p14": [], "y10": [], "p10": []}
    for f, meta in FOLDS.items():
        emb_path = OUT / f"eeg_emb_aldm_SH_f{f}_s0.npy"
        if not emb_path.exists():
            print(f"fold {f}: missing {emb_path}")
            continue
        E = unit(np.load(emb_path).astype(np.float32))
        m = al & np.isin(subj, meta["test"])
        y = song[m]
        q = E[m]
        p14 = (q @ proto.T).argmax(1)
        p10 = (q @ proto[:10].T).argmax(1)
        rng = np.random.default_rng(0)
        shuf14 = float((p14 == rng.permutation(len(proto))[y]).mean())
        shuf10 = float((p10 == rng.permutation(10)[y]).mean())
        acc14 = float((p14 == y).mean()); acc10 = float((p10 == y).mean())
        tag = f"SH_f{f}_stock_s0"
        ce = None
        if tag in heads:
            ce = heads[tag][-1].get("test_clf_acc")
        out["folds"][f] = {"subjects": meta["test"], "n_rows": int(m.sum()),
                           "acc14": acc14, "chance14": 1 / 14, "shuffled14": shuf14,
                           "acc10": acc10, "chance10": 0.1, "shuffled10": shuf10,
                           "ce_head_acc14": ce,
                           "train_seconds": heads.get(tag, [{}])[-1].get("train_seconds")}
        pooled["y14"].append(y); pooled["p14"].append(p14)
        pooled["y10"].append(y); pooled["p10"].append(p10)
        print(f"fold {f} subjects={meta['test']} n={m.sum()} 14-way {acc14:.4f} "
              f"(chance {1/14:.3f}, shuffled {shuf14:.3f}) | 10-way {acc10:.4f} "
              f"(chance 0.100, shuffled {shuf10:.3f}) | CE head {ce}")

    if pooled["y14"]:
        y = np.concatenate(pooled["y14"]); p14 = np.concatenate(pooled["p14"]); p10 = np.concatenate(pooled["p10"])
        out["pooled"] = {"n_rows": int(len(y)), "acc14": float((p14 == y).mean()),
                         "chance14": 1 / 14, "acc10": float((p10 == y).mean()), "chance10": 0.1,
                         "acc14_per_fold_mean": float(np.mean([v["acc14"] for v in out["folds"].values()])),
                         "acc14_per_fold_sd": float(np.std([v["acc14"] for v in out["folds"].values()], ddof=1)),
                         "acc10_per_fold_mean": float(np.mean([v["acc10"] for v in out["folds"].values()])),
                         "acc10_per_fold_sd": float(np.std([v["acc10"] for v in out["folds"].values()], ddof=1))}
        print("POOLED", json.dumps(out["pooled"], indent=1))
    json.dump(out, open(OUT / "summary.json", "w", encoding="utf-8"), indent=2)
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
