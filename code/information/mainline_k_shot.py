"""Main-pipeline k-shot calibration on the subject-held-out fold embeddings.

The band-power result (0.454@10 -> 0.721@300) was a proof of concept on a frozen
representation. This runs the same protocol on the MAIN pipeline's embedding: the
stage-1 alignment head trained with the held-out subjects excluded (the five folds in
results/subject_holdout/). Nothing here is transductive: the codebook (global class
centroids) is estimated from the training subjects of the same fold, the held-out
subject contributes only k labelled anchors, and the test rows are the subject's
held-out audio segments.

Primary metric: 14-way / 10-way song retrieval by nearest global centroid.
Secondary: a supervised nearest-centroid probe on the same anchors (TTA control).

Run: python code/information/mainline_k_shot.py
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
OUT = ROOT / "results" / "mainline_k_shot"
GATE = ROOT.parent / "mainv2" / "data" / "gate_nmedh14_ov2_a.npz"
FOLDS = {
    0: {"test": ["02", "03", "04", "05"], "val": ["06", "07"]},
    1: {"test": ["06", "07", "08", "09"], "val": ["04", "05"]},
    2: {"test": ["10", "11", "12", "13"], "val": ["06", "07"]},
    3: {"test": ["14", "15", "16", "17"], "val": ["08", "09"]},
    4: {"test": ["19", "20", "21", "23"], "val": ["10", "11"]},
}
KS = [10, 20, 30, 50, 75, 100, 150, 200, 300]


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def procrustes(A, B, w=None):
    if w is None:
        w = np.ones(len(A)) / len(A)
    s = np.sqrt(w / w.sum())
    Aw, Bw = A * s[:, None], B * s[:, None]
    Ua, Sa, Vta = np.linalg.svd(Aw, full_matrices=False)
    Ub, Sb, Vtb = np.linalg.svd(Bw, full_matrices=False)
    M = (Sa[:, None] * (Ua.T @ Ub)) * Sb[None, :]
    P, _, Qt = np.linalg.svd(M)
    return (Vta.T @ P) @ (Vtb.T @ Qt.T).T


def balanced_counts(k, c):
    counts = np.full(c, k // c)
    counts[:k % c] += 1
    return counts


def fps_select(rows_c, X, n):
    if n >= len(rows_c):
        return rows_c
    U = unit(X[rows_c])
    sel = [int(np.argmax(U @ unit(U.mean(0))))]
    dist = 1.0 - U @ U[sel[0]]
    for _ in range(1, n):
        i = int(np.argmax(dist)); sel.append(i)
        dist = np.minimum(dist, 1.0 - U @ U[i])
    return rows_c[np.array(sel)]


def pick_rows(rows_c, X, n):
    """Hybrid: closest-to-centroid for n<=2, FPS above (the band-power winner)."""
    if n >= len(rows_c):
        return rows_c
    if n <= 2:
        cen = X[rows_c].mean(0)
        order = np.argsort(-(unit(X[rows_c]) @ unit(cen)))
        return rows_c[order[:n]]
    return fps_select(rows_c, X, n)


def centroids(X, y, songs):
    return unit(np.stack([X[y == s].mean(0) for s in songs]))


def decode(aligned, C):
    return (unit(aligned) @ unit(C).T).argmax(1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(GATE, allow_pickle=True)
    song = z["song"].astype(int); subj = z["subject"].astype(str)
    al = z["aligned8"].astype(bool)
    grp = np.load(ROOT / "data" / "eegcond" / "group.npy").astype(int)
    songs = np.arange(14)
    rng = np.random.default_rng(0)

    per = {k: {"main_k": [], "probe_k": [], "acc14": [], "acc10": []} for k in KS}
    base = {"zero14": [], "zero10": [], "oracle14": [], "oracle10": []}
    fold_rec = {}

    for f, meta in FOLDS.items():
        emb = np.load(ROOT / "results" / "subject_holdout" / f"eeg_emb_aldm_SH_f{f}_s0.npy").astype(np.float32)
        E = unit(emb)
        train_subj = [s for s in np.unique(subj) if s not in meta["test"] and s not in meta["val"]]
        tr_rows = np.where(np.isin(subj, train_subj))[0]
        Cg = centroids(E[tr_rows], song[tr_rows], songs)          # shared codebook, training subjects
        fc = {k: {"main_k": [], "probe_k": [], "acc14": [], "acc10": []} for k in KS}
        fb = {"zero14": [], "zero10": [], "oracle14": [], "oracle10": []}

        for s in meta["test"]:
            rows = np.where(al & (subj == s))[0]
            gs = np.unique(grp[rows]); rng.shuffle(gs)
            fg = set(gs[:len(gs) // 2])
            rf = rows[np.isin(grp[rows], list(fg))]
            rt = rows[~np.isin(grp[rows], list(fg))]
            if len(rf) < 100 or len(rt) < 100:
                continue
            yte = song[rt]
            fb["zero14"].append(float((songs[decode(E[rt], Cg)] == yte).mean()))
            fb["zero10"].append(float((songs[decode(E[rt], Cg[:10])] == yte).mean()))
            # oracle: all fit rows (a held-out NMED-T subject has songs 0..9 only)
            present_s = np.unique(song[rf])
            Call = centroids(E[rf], song[rf], present_s)
            Cg_s = Cg[present_s]
            m = Call.mean(0)
            Rall = procrustes(Call - m, Cg_s - Cg_s.mean(0))
            fb["oracle14"].append(float((songs[decode((E[rt] - m) @ Rall, Cg - Cg.mean(0))] == yte).mean()))
            fb["oracle10"].append(float((songs[decode((E[rt] - m) @ Rall, Cg[:10] - Cg[:10].mean(0))] == yte).mean()))

            for k in KS:
                counts = np.minimum(balanced_counts(k, 10), np.array([len(rf[song[rf] == c]) for c in songs[:10]]))
                picks = []
                for ci, c in enumerate(songs[:10]):
                    if counts[ci]:
                        picks.append(pick_rows(rf[song[rf] == c], E, counts[ci]))
                ap = np.concatenate(picks); ay = song[ap]
                present = np.array([c for c in songs[:10] if (ay == c).any()])
                Cs = centroids(E[ap], ay, present)
                Cga = Cg[present]
                w = np.array([(ay == c).sum() for c in present], float)
                cs_m, cg_m = np.average(Cs, axis=0, weights=w), np.average(Cga, axis=0, weights=w)
                R = procrustes(Cs - cs_m, Cga - cg_m, w)
                aligned = (E[rt] - cs_m) @ R
                fc[k]["acc14"].append(float((songs[decode(aligned, Cg - Cg.mean(0))] == yte).mean()))
                fc[k]["acc10"].append(float((songs[decode(aligned, Cg[:10] - Cg[:10].mean(0))] == yte).mean()))
                # TTA control: nearest centroid of the subject's own anchors, no Procrustes
                fc[k]["probe_k"].append(float((songs[decode(E[rt], Cs)] == yte).mean()))
                # oracle-aligned main-pipeline accuracy at this k is the acc above

        fold_rec[f] = {"n_subjects": len(meta["test"]),
                       "zero14": float(np.mean(fb["zero14"])), "zero10": float(np.mean(fb["zero10"])),
                       "oracle14": float(np.mean(fb["oracle14"])), "oracle10": float(np.mean(fb["oracle10"])),
                       "curve": {str(k): {"acc14": float(np.mean(fc[k]["acc14"])),
                                          "acc10": float(np.mean(fc[k]["acc10"])),
                                          "probe10": float(np.mean(fc[k]["probe_k"]))} for k in KS}}
        for k in KS:
            per[k]["acc14"] += fc[k]["acc14"]; per[k]["acc10"] += fc[k]["acc10"]
            per[k]["probe_k"] += fc[k]["probe_k"]
        for key in base:
            base[key] += fb[key]
        print(f"fold {f}: zero14 {fold_rec[f]['zero14']:.4f} zero10 {fold_rec[f]['zero10']:.4f} "
              f"oracle14 {fold_rec[f]['oracle14']:.4f} oracle10 {fold_rec[f]['oracle10']:.4f}")

    res = {"protocol": {
        "embedding": "stage-1 head output from the 5 subject-held-out folds (stock init)",
        "codebook": "global class centroids from the fold's TRAINING subjects",
        "anchors": "balanced counts + hybrid (closest<=2, FPS above)",
        "test": "held-out audio segments of the held-out subject",
        "retrieval": "nearest global centroid (14-way and 10-way)",
    }, "baselines": {k: float(np.mean(v)) for k, v in base.items()},
        "curve": {}, "folds": fold_rec}
    for k in KS:
        res["curve"][str(k)] = {
            "acc14": float(np.mean(per[k]["acc14"])),
            "sd14": float(np.std(per[k]["acc14"], ddof=1)),
            "acc10": float(np.mean(per[k]["acc10"])),
            "sd10": float(np.std(per[k]["acc10"], ddof=1)),
            "probe10": float(np.mean(per[k]["probe_k"])),
        }
    res["bandpower_reference"] = {"note": "frozen band-power, 10-way nearest-centroid",
                                  "path": "results/k_shot/k_shot_v2.json"}
    json.dump(res, open(OUT / "k_shot.json", "w", encoding="utf-8"), indent=2)
    b = res["baselines"]
    print("\nMAIN PIPELINE (held-out subjects)")
    print(f"  zero 14-way {b['zero14']:.4f} | zero 10-way {b['zero10']:.4f} | "
          f"oracle 14-way {b['oracle14']:.4f} | oracle 10-way {b['oracle10']:.4f}")
    print("  k     main14  main10  probe10")
    for k in KS:
        r = res["curve"][str(k)]
        print(f"  {k:3d}   {r['acc14']:.3f}   {r['acc10']:.3f}   {r['probe10']:.3f}")
    print("wrote", OUT / "k_shot.json")


if __name__ == "__main__":
    main()
