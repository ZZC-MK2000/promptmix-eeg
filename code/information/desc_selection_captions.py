"""10-way description selection: free-text captions vs the tag baseline.

Same folds, head checkpoints and calibrated query as desc_selection_nmedt.py. Only the
14 description embeddings change. Reports k=0/10/100/300, the majority floor and the
paired per-fold difference from the tag descriptions.

Run: python code/information/desc_selection_captions.py --desc results/descriptions/captions_lp_song.json --mode medoid
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from calib_head_pilot import GATE, OUT, unit_np  # noqa: E402
from calib_head_5fold import FOLDS  # noqa: E402
from calib_head_genre import embed, fit_rotation, load_head  # noqa: E402

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
TAG = ROOT / "results" / "descriptions" / "descriptions.json"
KS = (0, 10, 100, 300)


def load_tag_D():
    d = json.load(open(TAG, encoding="utf-8"))
    return unit_np(np.array(d["embedding_specific"], dtype=np.float32))


def load_caption_D(path, mode):
    d = json.load(open(path, encoding="utf-8"))
    key = "medoid_embedding" if mode == "medoid" else "concat_embedding"
    return unit_np(np.array([s[key] for s in d["songs"]], dtype=np.float32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", type=Path, required=True)
    ap.add_argument("--mode", choices=["medoid", "concat"], default="medoid")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "descriptions" / "desc_selection_lp.json")
    args = ap.parse_args()

    D = {"tag": load_tag_D(), "caption": load_caption_D(args.desc, args.mode)}
    z = np.load(GATE, allow_pickle=True)
    cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)
    aldm = np.load(ROOT.parent / "mainv2" / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    w = torch.load(ROOT.parent / "mainv2" / "results" / "eeg_to_clap_proj.pt",
                   map_location="cpu", weights_only=True)
    ca = unit_np(aldm @ w["weight"].numpy().astype(np.float32).T + w["bias"].numpy().astype(np.float32))
    key = {ca[i].tobytes(): i for i in range(len(ca))}
    idx = np.array([key[cond[j].tobytes()] for j in range(len(cond))])
    X = torch.tensor(z["bandpower"][idx].astype(np.float32))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    T = unit_np(np.load(ROOT / "results" / "prompt_classifier" / "true_emb_all.npy").astype(np.float32))
    song = z["song"][idx].astype(int); subj = z["subject"][idx].astype(str)
    seg = z["seg2"][idx].astype(int); ds = z["dataset"][idx].astype(int)
    al = z["aligned8"][idx].astype(bool)
    C = unit_np(np.stack([T[song == s].mean(0) for s in range(14)]))

    for name, Dm in D.items():
        Dm10 = Dm[:10]
        acc = C @ Dm.T
        self14 = float((acc.argmax(1) == np.arange(14)).mean())
        self10 = float((acc[:10, :10].argmax(1) == np.arange(10)).mean())
        print(f"{name}: descriptor-only self-consistency 14-way {self14:.3f}, 10-way {self10:.3f}")

    res = {"protocol": {"desc": str(args.desc), "mode": args.mode, "n_candidates": 10,
                        "chance": 0.10, "folds": len(FOLDS)}, "folds": {}, "aggregate": {}}
    per = {name: {k: [] for k in KS} for name in D}
    floors = []
    for f, meta in FOLDS.items():
        train_subs = [s for s in np.unique(subj)
                      if s not in meta["test"] and s not in meta["val"] and s in set(subj[ds == 0])]
        train_rows = np.where((ds == 0) & np.isin(subj, train_subs))[0]
        Pn = unit_np(np.stack([T[train_rows][song[train_rows] == c].mean(0) for c in range(10)]))
        head = load_head(OUT / f"calib_bandpower_laion_f{f}.pt", X.shape[1])
        Z = embed(head, X)
        fv = {name: {k: [] for k in KS} for name in D}
        counts = np.zeros(10, int); n = 0
        rng = np.random.default_rng(f)
        for s in meta["test"]:
            rows = np.where(al & (subj == s))[0]
            segs = np.unique(seg[rows]); rng.shuffle(segs)
            fg = set(segs[:len(segs) // 2])
            rf = rows[np.isin(seg[rows], list(fg))]
            rt = rows[~np.isin(seg[rows], list(fg))]
            if len(rf) < 100 or len(rt) < 100:
                continue
            yte = song[rt]
            counts += np.bincount(yte, minlength=10); n += len(rt)
            for name, Dm in D.items():
                Dm10 = Dm[:10]
                fv[name][0].append(float(((Z[rt] @ Dm10.T).argmax(1) == yte).mean()))
                for k in KS[1:]:
                    R, m = fit_rotation(Z, rf, song, Pn, k=k)
                    q = (Z[rt] - m) @ R
                    fv[name][k].append(float(((q @ Dm10.T).argmax(1) == yte).mean()))
        floors.append(float(counts.max() / n))
        for name in D:
            for k in KS:
                per[name][k].append(float(np.mean(fv[name][k])))
        print(f"fold {f}: " + " ".join(
            f"{name} " + "/".join(f"{np.mean(fv[name][k]):.3f}" for k in KS)
            for name in D) + f" | majority {counts.max()/n:.3f}", flush=True)

    for name in D:
        res["aggregate"][name] = {str(k): {"mean": float(np.mean(per[name][k])),
                                          "sd": float(np.std(per[name][k], ddof=1))}
                                  for k in KS}
        res["aggregate"][name]["per_fold"] = {str(k): per[name][k] for k in KS}
    res["majority_floor"] = {"mean": float(np.mean(floors)), "per_fold": floors}
    k = 100
    diff = np.array(per["caption"][k]) - np.array(per["tag"][k])
    res["caption_minus_tag_k100"] = {"mean": float(diff.mean()),
                                     "sd": float(diff.std(ddof=1)),
                                     "t": float(diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))),
                                     "per_fold": diff.tolist()}
    json.dump(res, open(args.out, "w", encoding="utf-8"), indent=2)
    print("\n=== 10-way selection (5-fold) ===")
    for name in D:
        a = res["aggregate"][name]
        print(f"{name:8} " + "  ".join(f"k{k} {a[str(k)]['mean']:.3f}±{a[str(k)]['sd']:.3f}" for k in KS))
    print("majority floor", round(res["majority_floor"]["mean"], 3))
    print("caption - tag @k100", round(diff.mean(), 3), "t", round(res["caption_minus_tag_k100"]["t"], 2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
