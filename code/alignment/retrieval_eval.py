"""Retrieval line, measured honestly: how well does EEG pick the right stimulus music?

Corpus = the 1103 stimulus windows whose Encodec codes are already on disk (so every hit can
be turned into real audio with no approximation). Query = bridge(EEG), the same vector that
feeds the generator.

Everything is reported against baselines, because a bare number means nothing in this space:
  song-level 14-way : chance 1/14 = 0.071; known ceiling for this port = 0.8428 (held-out
                      subject diagnostic from the alignment work)
  window-level      : chance 1/1103 = 0.0009, but same-song windows are near-duplicates of
                      each other, so we ALSO report the harder within-song version
  constant baseline : always answer the most frequent song  (defeats inflated song accuracy)
  no-skill baseline : random corpus entry

Temporal smoothing is the lever this line can actually use: EEG is noisy per 8 s window but a
subject listens to one song for minutes, so a majority filter over consecutive windows should
lift song accuracy a lot. That is measured, not assumed.

Run: python src/train/retrieval_eval.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
OUT = ROOT / "results" / "retrieval_line"
sys.path.insert(0, str(M2 / "src" / "generate"))


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def majority(seq, w):
    """1-D majority filter with an odd window w."""
    if w <= 1:
        return list(seq)
    h = w // 2
    out = []
    for i in range(len(seq)):
        seg = seq[max(0, i - h):i + h + 1]
        out.append(Counter(seg).most_common(1)[0][0])
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy")
    group = np.load(DATA / "group.npy").astype(np.int64)

    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    song_all = z["song"].astype(int)
    subj_all = np.array([str(x) for x in z["subject"]])
    run_all = np.array([str(x) for x in z["run"]])
    seg_all = z["seg2"].astype(int)
    target_all = z["target"].astype(np.float32)

    # map kept rows back to original cache rows by an order-preserving byte walk
    eeg = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    proj = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu",
                      weights_only=True)
    W, b = proj["weight"].numpy(), proj["bias"].numpy()
    orig = unit(eeg @ W.T + b).astype(np.float32)
    j, mapped = 0, []
    for i in range(len(cond)):
        k = cond[i].tobytes()
        while j < len(orig) and orig[j].tobytes() != k:
            j += 1
        mapped.append(j); j += 1
    mapped = np.array(mapped)
    q_song, q_subj, q_run, q_seg = song_all[mapped], subj_all[mapped], run_all[mapped], seg_all[mapped]

    # corpus: one entry per code-group (identical target inside a group)
    gvals = np.unique(group)
    first = {int(g): int(np.where(group == g)[0][0]) for g in gvals}
    rows = np.array([first[int(g)] for g in gvals])
    T = unit(target_all[mapped[rows]])           # laion vectors of the stimulus windows
    c_song = q_song[rows]
    print(f"corpus {len(rows)} stimulus windows over {len(np.unique(c_song))} songs; "
          f"queries {len(cond)} EEG windows")

    Q = unit(cond)
    S = Q @ T.T                                   # [queries, corpus]

    top = np.argmax(S, axis=1)
    song_hit = float((c_song[top] == q_song).mean())
    # window-level needs group identity: a query is correct if the retrieved window is the
    # stimulus window that query was cut from -> group id equality
    grp_of_row = group[rows]
    win_hit = float((grp_of_row[top] == group).mean())

    rng = np.random.default_rng(0)
    ridx = rng.integers(0, len(rows), len(cond))      # one random corpus item per query
    rand_hit = float((c_song[ridx] == q_song).mean())
    mode_song = Counter(c_song.tolist()).most_common(1)[0][0]
    const_hit = float((q_song == mode_song).mean())

    print(f"\n=== 逐窗（无平滑）===")
    print(f"  歌级命中 = {song_hit:.4f}   随机基线 {1/14:.4f}   常数基线(总答最多的歌) "
          f"{const_hit:.4f}   已知上限 0.8428")
    print(f"  窗级命中 = {win_hit:.5f}   随机基线 {1/len(rows):.5f}")

    # harder view: among same-song windows only, can it find the right window?
    ws, ch = [], []
    step = max(1, len(cond) // 2000)
    for qi in range(0, len(cond), step):
        if c_song[top[qi]] != q_song[qi]:
            continue
        ii = np.where(c_song == q_song[qi])[0]
        if len(ii) < 4:
            continue
        pick = int(np.argmax(S[qi, ii]))
        ws.append(int(grp_of_row[ii[pick]] == group[qi]))
        ch.append(1 / len(ii))
    if ws:
        print(f"  在答对歌的前提下取到正确窗 = {np.mean(ws):.4f} "
              f"(n={len(ws)}，该歌窗数倒数的均值 = {np.mean(ch):.4f})")

    # temporal smoothing along each subject-run
    print(f"\n=== 时间平滑（同一被试同一次记录的连续窗做多数表决）===")
    keys = {}
    for i in range(len(cond)):
        keys.setdefault((q_subj[i], q_run[i], int(q_song[i])), []).append(i)
    best = None
    for wsize in (1, 3, 5, 9):
        tot = cor = 0
        for kk, idxs in keys.items():
            idxs = np.array(idxs)[np.argsort(q_seg[np.array(idxs)])]
            pred_seq = [int(c_song[top[i]]) for i in idxs]
            sm = majority(pred_seq, wsize)
            tot += len(idxs); cor += sum(int(a == q_song[i]) for a, i in zip(sm, idxs))
        acc = cor / tot
        print(f"  窗口 {wsize} 个窗（{wsize*8} 秒）: 歌级命中 = {acc:.4f}  (+{acc-song_hit:+.4f} vs 不平滑)")
        if best is None or acc > best[1]:
            best = (wsize, acc)
    print(f"  最佳平滑: 窗口 {best[0]} → {best[1]:.4f}")

    json.dump({"corpus_windows": int(len(rows)), "queries": int(len(cond)),
               "song_hit_raw": float(song_hit), "song_hit_chance": 1 / 14,
               "song_hit_constant": const_hit, "song_hit_random_shuffled": rand_hit,
               "window_hit_raw": win_hit, "window_hit_chance": 1 / len(rows),
               "within_song_window_rate": float(np.mean(ws)),
               "best_smooth_window": best[0], "best_smooth_acc": float(best[1]),
               "known_port_ceiling": 0.8428},
              open(OUT / "eval.json", "w"), indent=2)
    print(f"\n数字写入 {OUT/'eval.json'}")


if __name__ == "__main__":
    main()
