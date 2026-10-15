"""Correct scoring of the saved identity-test clips.

Fixes two bugs in identity_test.py's inline scoring:
  - it indexed the embedding matrix as if it were the similarity matrix
  - it fed 32 kHz MusicGen audio into laion-CLAP while claiming sr=48 kHz, so no
    resample happened (content played 1.5x fast); clips are resampled from their real
    rate here.

Reports window-level retrieval, paired specificity t, song-level hit vs its own chance,
and the gen-vs-gen control that exposes a condition being ignored.

Run: python src/train/score_identity.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
D = ROOT / "results" / "gen_eegcond" / "identity"
sys.path.insert(0, str(M2 / "src" / "generate"))

WAV_SR = 32000          # MusicGen / Encodec output rate


def _load_32k(p):
    import soundfile as sf
    w, s = sf.read(str(p), dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    assert s == WAV_SR, f"{p}: expected {WAV_SR} got {s}"
    return w


def main():
    global D
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(D), help="clip dir to score (identity or identity_kK)")
    args = ap.parse_args()
    D = Path(args.dir)

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from musicgen_clap_finetune import load_clap, clap_embed_audio
    clap, proc, sr = load_clap(dev)
    print(f"CLAP expects {sr} Hz; clips are {WAV_SR} Hz and get resampled properly")

    spec = json.loads((D / "test_spec.json").read_text(encoding="utf-8"))
    row_song = {f"{r:05d}": s for r, s in zip(spec["pick"], spec["songs"])}

    unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    def embed(ps):
        # clap_embed_audio resamples when we tell it the true source rate
        return unit(np.stack([clap_embed_audio(clap, proc, WAV_SR, _load_32k(p), dev)
                              for p in ps]))

    trues = sorted(D.glob("TRUE_*.wav"))
    rows = [p.stem.split("row")[-1] for p in trues]
    n = len(rows)
    T = embed(trues)
    TT = T @ T.T
    tt_off = float(TT[np.triu_indices(n, 1)].mean())
    print(f"\nn={n} distinct held-out target groups")
    print(f"TRUE-v-TRUE off-diagonal mean = {tt_off:.4f}  (CLAP 空间的整体聚簇水平)")

    songs = np.array([row_song[r] for r in rows])
    p = np.bincount(songs, minlength=14) / n
    chance_song = float((p * p).sum())
    print(f"chance: window top-1 = {1/n:.3f}   same-song = {chance_song:.3f}")

    for arm in ("trained", "randomproj"):
        ps = sorted(D.glob(f"gen_{arm}_row*.wav"))
        if len(ps) != n:
            print(f"\n[{arm}] {len(ps)} clips != {n}, skip")
            continue
        idx = [rows.index(p.stem.split("row")[-1]) for p in ps]
        G = embed(ps)
        S = G @ T.T                                    # [n_gen, n_true] cosines
        own = np.array([S[i, idx[i]] for i in range(n)])
        others = np.stack([np.delete(S[i], idx[i]).mean() for i in range(n)])
        GG = G @ G.T
        gg_off = float(GG[np.triu_indices(n, 1)].mean())

        ranks = []
        for i in range(n):
            o = np.argsort(-S[i])
            ranks.append(int(np.where(o == idx[i])[0][0]) + 1)
        top1 = sum(r == 1 for r in ranks)
        ss_hit = sum(1 for i in range(n) if songs[int(np.argmax(S[i]))] == songs[i])
        ss_hit_excl = 0
        for i in range(n):
            j = int(np.argmax(S[i]))
            if j != i and songs[j] == songs[i]:
                ss_hit_excl += 1

        d = own - others
        tstat = d.mean() / (d.std(ddof=1) / np.sqrt(n))
        print(f"\n### arm={arm}")
        print(f"  own  cos(gen_i, true_i)     mean={own.mean():.4f}")
        print(f"  others cos(gen_i, true_j)   mean={others.mean():.4f}")
        print(f"  GEN-v-GEN mutual            mean={gg_off:.4f}")
        print(f"  WINDOW top-1 = {top1}/{n} = {top1/n:.3f}     mean rank = {np.mean(ranks):.2f}")
        print(f"  specificity margin = {d.mean():+.4f}   paired t = {tstat:+.2f}")
        print(f"  SONG top-1 same song = {ss_hit}/{n} = {ss_hit/n:.3f}  (chance {chance_song:.3f}; "
              f"其中 {ss_hit_excl} 个还正好不是自己)")
        print(f"  own - GEN-v-GEN = {own.mean()-gg_off:+.4f}   "
              f"TRUE-v-TRUE = {tt_off:+.4f}")
        print(f"  ranks={ranks}")


if __name__ == "__main__":
    main()
