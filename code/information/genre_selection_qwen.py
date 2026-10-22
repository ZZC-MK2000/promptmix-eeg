"""12-way genre selection with Qwen-written genre prompts, same calibrated query.

Labels: results/library_b/oracle_all.npy (mainline 12-class oracle,
compact 36,036 rows). Candidate prompts: the 12 Qwen rewrites, in the same class order.
Compares Qwen vs the original hand-written prompts and reports accuracy, balanced
accuracy, Cohen's kappa and the majority floor.

Run: python code/information/genre_selection_qwen.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from calib_head_pilot import GATE, OUT, unit_np  # noqa: E402
from calib_head_5fold import FOLDS  # noqa: E402
from calib_head_genre import embed, fit_rotation, load_head  # noqa: E402

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
GP = ROOT / "results" / "descriptions" / "genre_prompts_qwen.json"
CLAP_ID = "laion/clap-htsat-unfused"
KS = (0, 10, 100, 300)


def metrics(y, pred):
    acc = float((y == pred).mean())
    recalls = [float((pred[y == c] == c).mean()) for c in np.unique(y)]
    bal = float(np.mean(recalls))
    return {"acc": acc, "balanced": bal, "kappa": float(cohen_kappa_score(y, pred))}


def main():
    d = json.load(open(GP, encoding="utf-8"))
    prompts = d["prompts"]
    Dq = unit_np(np.array([p["embedding"] for p in prompts], dtype=np.float32))
    from transformers import ClapModel, ClapProcessor
    cp = ClapProcessor.from_pretrained(CLAP_ID)
    clap = ClapModel.from_pretrained(CLAP_ID).eval().to("cuda")
    with torch.no_grad():
        inp = cp(text=[p["original"] for p in prompts], return_tensors="pt", padding=True).to("cuda")
        Do = unit_np(clap.get_text_features(**inp).pooler_output.cpu().numpy().astype(np.float32))
    D = {"qwen": Dq, "original": Do}

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
    y_all = np.load(ROOT / "results" / "library_b" / "oracle_all.npy").astype(np.int64)
    song = z["song"][idx].astype(int); subj = z["subject"][idx].astype(str)
    seg = z["seg2"][idx].astype(int); ds = z["dataset"][idx].astype(int)
    al = z["aligned8"][idx].astype(bool)

    res = {"protocol": {"labels": "library_b/oracle_all (mainline 12-class oracle)",
                        "candidates": 12, "chance": 1 / 12, "folds": len(FOLDS)},
           "folds": {}, "aggregate": {}}
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
        all_y = []
        rng = np.random.default_rng(f)
        for s in meta["test"]:
            rows = np.where(al & (subj == s))[0]
            segs = np.unique(seg[rows]); rng.shuffle(segs)
            fg = set(segs[:len(segs) // 2])
            rf = rows[np.isin(seg[rows], list(fg))]
            rt = rows[~np.isin(seg[rows], list(fg))]
            if len(rf) < 100 or len(rt) < 100:
                continue
            y = y_all[rt]
            all_y.append(y)
            for name, Dm in D.items():
                fv[name][0].append(metrics(y, (Z[rt] @ Dm.T).argmax(1)))
                for k in KS[1:]:
                    R, m = fit_rotation(Z, rf, song, Pn, k=k)
                    q = (Z[rt] - m) @ R
                    fv[name][k].append(metrics(y, (q @ Dm.T).argmax(1)))
        ycat = np.concatenate(all_y)
        counts = np.bincount(ycat, minlength=12)
        floors.append(float(counts.max() / counts.sum()))
        for name in D:
            for k in KS:
                per[name][k].append({kk: float(np.mean([x[kk] for x in fv[name][k]])) for kk in
                                     ("acc", "balanced", "kappa")})
        print(f"fold {f}: " + " ".join(
            f"{name} acc/kappa " + "/".join(f"{np.mean([x['acc'] for x in fv[name][k]]):.3f}/"
                                            f"{np.mean([x['kappa'] for x in fv[name][k]]):.3f}" for k in KS)
            for name in D), flush=True)

    for name in D:
        for k in KS:
            arr = per[name][k]
            res["aggregate"][f"{name}_k{k}"] = {kk: float(np.mean([x[kk] for x in arr])) for kk in
                                                ("acc", "balanced", "kappa")}
    res["majority_floor"] = float(np.mean(floors))
    json.dump(res, open(ROOT / "results" / "descriptions" / "genre_selection_qwen.json",
                        "w", encoding="utf-8"), indent=2)
    print("\n=== 12-way genre selection (5-fold) ===")
    for name in D:
        print(name, {k: round(res["aggregate"][f"{name}_k{k}"]["acc"], 3) for k in KS},
              "kappa", {k: round(res["aggregate"][f"{name}_k{k}"]["kappa"], 3) for k in KS})
    print("majority floor", round(res["majority_floor"], 3))
    print("wrote results/descriptions/genre_selection_qwen.json")


if __name__ == "__main__":
    main()
