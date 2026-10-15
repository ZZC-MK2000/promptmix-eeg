"""Verify the thing that matters: did the model learn to decode musical FEATURES from EEG?

Two deliverables:

1. BLIND MATCHING PACK for the ear: 10 generated clips renamed A..J and their 10 stimulus
   clips renamed 1..10, independently shuffled, so matching is not leaked by filename or
   order. The answer key goes to a separate file to be revealed after listening. Chance = 1/10.

2. FEATURE TRANSFER (objective): per 8s window compute loudness (rms), brightness (spectral
   centroid), roughness (spectral flatness ~1 means noise-like), event density (onsets/sec),
   tonal profile (12-d chroma cosine). Then Spearman rank correlation (order agreement only,
   robust to outliers) between generated and true values across the held-out windows, with a
   2000-draw permutation null giving the percentile of the observed correlation.

Run: python src/train/verify_features.py --ep 23
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
G = ROOT / "results" / "gen_eegcond"
TREND = G / "trend"
PACK = G / "ear_test"


def feats(w, sr):
    import librosa
    w = w.astype(np.float32)
    rms = float(np.sqrt(np.mean(w ** 2)))
    cent = float(np.mean(librosa.feature.spectral_centroid(y=w, sr=sr)))
    flat = float(np.mean(librosa.feature.spectral_flatness(y=w)))
    on = librosa.onset.onset_detect(y=w, sr=sr, backtrack=False)
    dens = len(on) / (len(w) / sr)
    ch = librosa.feature.chroma_cqt(y=w, sr=sr).mean(1)
    ch = ch / (np.linalg.norm(ch) + 1e-9)
    return dict(rms=rms, centroid=cent, flatness=flat, onset_rate=float(dens), chroma=ch)


def load(p, sr=32000):
    import soundfile as sf
    w, s = sf.read(str(p), dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    return w


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    d = ra - ra.mean(), rb - rb.mean()
    num = float((d[0] * d[1]).sum())
    den = float(np.sqrt((d[0] ** 2).sum() * (d[1] ** 2).sum()) + 1e-12)
    return num / den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ep", type=int, default=23)
    ap.add_argument("--compare-ep", type=int, default=16)
    ap.add_argument("--n-listen", type=int, default=10)
    ap.add_argument("--perm", type=int, default=2000)
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    PACK.mkdir(parents=True, exist_ok=True)
    for sub in ("gen_shuffled", "true_shuffled"):
        d = PACK / sub
        d.mkdir(exist_ok=True)
        for f in d.glob("*.wav"):
            f.unlink()

    rows = sorted(p.stem.split("row")[-1] for p in TREND.glob(f"gen_ep{args.ep}_row*.wav"))
    print(f"{len(rows)} held-out windows with both generated(ep{args.ep}) and true audio")

    F_gen, F_true = [], []
    for rk in rows:
        g = load(TREND / f"gen_ep{args.ep}_row{rk}.wav")
        t = load(TREND / f"TRUE_row{rk}.wav")
        F_gen.append(feats(g, 32000))
        F_true.append(feats(t, 32000))

    keys = ["rms", "centroid", "flatness", "onset_rate"]
    print(f"\n=== 特征是否跟着脑电走（n={len(rows)} 条留 out 窗；Spearman 秩相关 + 置换检验）===")
    res = {}
    for k in keys:
        gv = np.array([f[k] for f in F_gen])
        tv = np.array([f[k] for f in F_true])
        rho = spearman(gv, tv)
        null = np.array([spearman(gv, rng.permutation(tv)) for _ in range(args.perm)])
        pct = float((null >= rho).mean())
        band = f"[{np.percentile(null,2.5):+.3f},{np.percentile(null,97.5):+.3f}]"
        print(f"  {k:<11} ρ={rho:+.3f}   置换95%区间{band}   单侧p={pct:.3f}  "
              f"{'显著' if pct < 0.05 else '不显著'}")
        res[k] = dict(rho=float(rho), p=pct, null_hi=float(np.percentile(null, 97.5)),
                      null_lo=float(np.percentile(null, 2.5)))

    cg = np.stack([f["chroma"] for f in F_gen]); ct = np.stack([f["chroma"] for f in F_true])
    cos = (cg * ct).sum(1) / (np.linalg.norm(cg, axis=1) * np.linalg.norm(ct, axis=1) + 1e-9)
    off = []
    for i in range(len(rows)):
        for j in range(len(rows)):
            if i != j:
                off.append(float((cg[i] @ ct[j]) /
                                 (np.linalg.norm(cg[i]) * np.linalg.norm(ct[j]) + 1e-9)))
    off = np.array(off)
    z = float((cos.mean() - off.mean()) / (off.std() + 1e-9))
    print(f"  {'chroma':<11} 同窗余弦={cos.mean():.3f}  乱配余弦={off.mean():.3f}±{off.std():.3f}"
          f"  高出 {z:+.2f} 个乱配标准差  {'显著' if z>=2 else '不显著'}")
    res["chroma"] = dict(own=float(cos.mean()), other=float(off.mean()), z=z)

    # blind matching pack
    pick = list(rng.permutation(len(rows))[:args.n_listen])
    order_g = rng.permutation(len(pick))
    order_t = rng.permutation(len(pick))
    key = {}
    for pos, gi in enumerate(order_g):
        src = TREND / f"gen_ep{args.ep}_row{rows[pick[gi]]}.wav"
        dst = PACK / "gen_shuffled" / f"{chr(65+pos)}.wav"
        shutil.copyfile(src, dst)
    for pos, ti in enumerate(order_t):
        r = rows[pick[ti]]
        src = TREND / f"TRUE_row{r}.wav"
        dst = PACK / "true_shuffled" / f"{pos+1}.wav"
        shutil.copyfile(src, dst)
        key[f"{pos+1}"] = r
    # which letter holds the matching clip for each number
    row2letter = {}
    for pos, gi in enumerate(order_g):
        row2letter[rows[pick[gi]]] = chr(65 + pos)
    answer = {f"{pos+1}": row2letter[key[f"{pos+1}"]] for pos in range(len(pick))}
    (PACK / "ANSWER_KEY_do_not_open_before_listening.json").write_text(
        json.dumps({"number_to_true_row": key, "number_to_matching_letter": answer,
                    "chance": f"1/{len(pick)}", "ep": args.ep}, indent=2), encoding="utf-8")
    print(f"\n盲测包: {PACK}/gen_shuffled/A..{chr(64+len(pick))}.wav 与 "
          f"true_shuffled/1..{len(pick)}.wav")
    print(f"答案在 ANSWER_KEY_do_not_open_before_listening.json（听完再开），随机配对命中率={1/len(pick):.2f}")
    json.dump(res, open(PACK / f"feature_transfer_ep{args.ep}.json", "w"), indent=2)


if __name__ == "__main__":
    main()
