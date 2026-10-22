"""Song-level CLAP margin for all generation arms, with content-disjoint references.

Two fixes over a naive own-vs-other-15 analysis:
  * the 16 windows cover only 8 songs, so "other 15" contained windows of the query's
    own song (same-song CLAP cos 0.70 vs 0.56 across songs). References are restricted
    to *different* songs;
  * retrieval is over 8 song centroids, not 16 correlated windows.

Primary statistic: song-centroid margin = cos(generated, own-song centroid) minus the
mean cos to the other 7 song centroids. The permutation null shuffles the 8 song labels
(group-preserving, 2000 draws), so it tests whether a generated clip is specifically
aligned with its own song rather than with a generic stimulus-set direction.

Run: python code/information/desc_margin_all.py
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = ROOT.parent / "mainv2"
GEN = ROOT / "results" / "descriptions"
ARMS = ["oracle", "head_calib", "head_zero", "mainline", "desc_shuffled",
        "uncond", "desc_samegenre"]
B = 2000


def unit(a):
    return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, default=GEN / "generation.json")
    ap.add_argument("--gen-dir", type=Path, default=GEN / "gen")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--out", type=Path, default=GEN / "margin_all.json")
    args = ap.parse_args()

    sys.path.insert(0, str(M2 / "src" / "generate"))
    from musicgen_clap_finetune import load_clap, clap_embed_audio

    d = json.load(open(args.records, encoding="utf-8"))
    recs = d["records"]
    row_song = {r["row"]: r["true_song"] for r in recs}
    rows = sorted(row_song)
    songs = sorted(set(row_song.values()))
    T = unit(np.load(ROOT / "results" / "prompt_classifier" / "true_emb_all.npy").astype(np.float32))
    W = unit(np.stack([T[r] for r in rows]))
    song_of = np.array([row_song[r] for r in rows])
    S = unit(np.stack([W[song_of == s].mean(0) for s in songs]))
    ss = S @ S.T
    iu = np.triu_indices(len(songs), 1)
    print(f"windows {len(rows)} songs {len(songs)}; song-centroid cos mean "
          f"{ss[iu].mean():.3f} (min {ss[iu].min():.3f}, max {ss[iu].max():.3f})")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    clap, proc, _ = load_clap(dev)
    per_arm = {}
    for arm in ARMS:
        wavs = [args.gen_dir / f"{args.prefix}{arm}_row{r}.wav" for r in rows]
        if not all(p.exists() for p in wavs):
            print(f"skip {arm}: missing wavs")
            continue
        cen_sims, win_own, win_m, cen_own, cen_m, ranks = [], [], [], [], [], []
        for i, p in enumerate(wavs):
            wav, sr = sf.read(p, dtype="float32")
            e = unit(np.asarray(clap_embed_audio(clap, proc, sr, wav, dev), dtype=np.float32))
            sw = W @ e
            diff = song_of != song_of[i]
            win_own.append(float(sw[i]))
            win_m.append(float(sw[i] - sw[diff].mean()))
            s = S @ e
            cen_sims.append(s)
            j = songs.index(song_of[i])
            cen_own.append(float(s[j]))
            cen_m.append(float(s[j] - np.delete(s, j).mean()))
            ranks.append(int((s > s[j]).sum()) + 1)
        per_arm[arm] = {"cen_sims": np.stack(cen_sims), "win_margin": np.array(win_m),
                        "win_own": np.array(win_own), "cen_own": np.array(cen_own),
                        "cen_margin": np.array(cen_m), "rank": np.array(ranks)}
        n = len(rows)
        m = np.mean(cen_m); se = np.std(cen_m, ddof=1) / np.sqrt(n)
        print(f"{arm:14s} cen-own {np.mean(cen_own):.3f} cen-margin {m:+.3f} "
              f"(t {m/se:+.2f}) | win-own {np.mean(win_own):.3f} "
              f"win-margin {np.mean(win_m):+.3f} | rank1 {int((per_arm[arm]['rank']==1).sum())}/{n} "
              f"mean-rank {np.mean(per_arm[arm]['rank']):.2f}")

    rng = np.random.default_rng(0)
    song_idx = np.array([songs.index(s) for s in song_of])
    out = {"windows": rows, "songs": songs, "row_song": row_song,
           "song_centroid_pair_cos_mean": float(ss[iu].mean()), "arms": {}}
    for arm, a in per_arm.items():
        n = len(rows)
        obs = float(a["cen_margin"].mean())
        cs = a["cen_sims"]
        null = np.empty(B)
        for b in range(B):
            perm = rng.permutation(len(songs))
            pseudo = perm[song_idx]
            own = cs[np.arange(n), pseudo]
            other = (cs.sum(1) - own) / (len(songs) - 1)
            null[b] = (own - other).mean()
        p = (1 + int((null >= obs).sum())) / (1 + B)
        se = float(a["cen_margin"].std(ddof=1) / np.sqrt(n))
        out["arms"][arm] = {
            "n": n, "mean_song_centroid_own": float(a["cen_own"].mean()),
            "mean_song_centroid_margin": obs, "se": se, "t": obs / se,
            "ci95": [obs - 2.131 * se, obs + 2.131 * se],
            "rows_margin_positive": int((a["cen_margin"] > 0).sum()),
            "mean_window_own": float(a["win_own"].mean()),
            "mean_window_margin": float(a["win_margin"].mean()),
            "rank1": int((a["rank"] == 1).sum()), "mean_rank": float(a["rank"].mean()),
            "perm_p_song_label": p,
            "per_row": [{"row": int(rows[i]), "song": int(song_of[i]),
                         "cen_own": float(a["cen_own"][i]),
                         "cen_margin": float(a["cen_margin"][i]),
                         "win_margin": float(a["win_margin"][i]),
                         "rank": int(a["rank"][i])} for i in range(len(rows))],
        }
        print(f"{arm:14s} song-label permutation p = {p:.3f}")

    for x, y in [("head_calib", "head_zero"), ("head_calib", "mainline"),
                 ("oracle", "mainline"), ("oracle", "desc_shuffled"),
                 ("head_calib", "uncond"), ("mainline", "uncond"),
                 ("oracle", "uncond"), ("oracle", "head_calib"),
                 ("head_calib", "desc_shuffled"),
                 ("oracle", "desc_samegenre"), ("head_calib", "desc_samegenre"),
                 ("desc_samegenre", "uncond"), ("desc_samegenre", "desc_shuffled")]:
        if x not in per_arm or y not in per_arm:
            continue
        dm = per_arm[x]["cen_margin"] - per_arm[y]["cen_margin"]
        obs = float(dm.mean()); se = float(dm.std(ddof=1) / np.sqrt(len(dm)))
        cnt = 0
        for b in range(B):
            s = rng.choice([-1.0, 1.0], size=len(dm))
            if abs((s * dm).mean()) >= abs(obs):
                cnt += 1
        out["arms"][f"{x}_minus_{y}"] = {
            "mean_diff": obs, "se": se, "t": obs / se, "perm_p": (1 + cnt) / (1 + B)}
        # effective unit is the song: average rows of a song, exact sign-flip over 8 songs
        dx = np.array([per_arm[x]["cen_margin"][song_of == s].mean() for s in songs])
        dy = np.array([per_arm[y]["cen_margin"][song_of == s].mean() for s in songs])
        dsong = dx - dy
        signs = np.array(list(itertools.product([-1.0, 1.0], repeat=len(dsong))))
        null_song = (signs * dsong).mean(1)
        p_song = (1 + int((np.abs(null_song) >= abs(dsong.mean())).sum())) / (1 + len(null_song))
        out["arms"][f"{x}_minus_{y}"]["song_level_diff"] = float(dsong.mean())
        out["arms"][f"{x}_minus_{y}"]["song_level_p"] = float(p_song)
        out["arms"][f"{x}_minus_{y}"]["song_level_values"] = dsong.tolist()
        print(f"{x} - {y}: cen-margin diff {obs:+.3f} (t {obs/se:+.2f}, "
              f"perm p {out['arms'][f'{x}_minus_{y}']['perm_p']:.3f}; "
              f"song-level {dsong.mean():+.3f}, exact p {p_song:.3f})")

    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
