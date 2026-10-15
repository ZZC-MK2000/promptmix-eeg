"""Is the failure in the generator or in the representation?

The continuous demo showed zero temporal following within one song. Two very different
causes:
  (A) bridge(EEG) cannot tell two 8s windows of the SAME song apart  -> representation
      ceiling; no generator, however good, could follow the timeline.
  (B) the vectors ARE separable but the generator ignores them        -> fixable.

This tests (A) directly, with no generator at all: use each window's condition vector to
retrieve its own 8s stimulus audio among the windows of that song. Chance = 1/N.
Also reports the audio-side floor: how similar those true windows are to each other, and
a cross-song run as a positive control.

Run: python src/train/eeg_window_separability.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
IDK8 = ROOT / "results" / "gen_eegcond" / "trend"
sys.path.insert(0, str(M2 / "src" / "generate"))

WAV_SR = 32000


def main():
    from musicgen_clap_finetune import load_clap, clap_embed_audio, MusicGenCLAP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int64)
    group = np.load(DATA / "group.npy").astype(np.int64)

    # the same 60 held-out targets the trend test used
    rows = json.loads((IDK8 / "test_spec.json").read_text(encoding="utf-8"))["pick"] \
        if (IDK8 / "test_spec.json").exists() else None
    if rows is None:
        uniq = np.unique(group)
        rng = np.random.default_rng(0)
        n_val = max(1, int(round(len(uniq) * 0.1)))
        val_g = set(uniq[rng.permutation(len(uniq))[:n_val]].tolist())
        va = np.where(np.array([g in val_g for g in group]))[0]
        seen, one = set(), []
        for r in va:
            if int(group[r]) not in seen:
                seen.add(int(group[r])); one.append(int(r))
        rows = np.array(one)[np.linspace(0, len(one) - 1, 60, dtype=int)].tolist()
    print(f"{len(rows)} targets")

    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    song_all = z["song"].astype(int)
    eeg = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    proj = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu",
                      weights_only=True)
    W, b = proj["weight"].numpy(), proj["bias"].numpy()
    v = eeg @ W.T + b
    orig = (v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)
    j, mapped = 0, []
    for i in range(len(cond)):
        key = cond[i].tobytes()
        while j < len(orig) and orig[j].tobytes() != key:
            j += 1
        mapped.append(j); j += 1
    songs_all = song_all[np.array(mapped)]
    songs = np.array([songs_all[r] for r in rows])
    print("song distribution:", np.bincount(songs, minlength=14).tolist())

    clap, proc, csr = load_clap(device)
    base = MusicGenCLAP(device=device, use_lora=False)

    # true-window CLAP embeddings (decode target codes -> audio -> embed)
    A = []
    for r in rows:
        w = np.asarray(base.mg.audio_encoder.decode(
            torch.from_numpy(codes[r]).unsqueeze(0).unsqueeze(0).to(device),
            audio_scales=[None]).audio_values.cpu()).reshape(-1)
        A.append(clap_embed_audio(clap, proc, WAV_SR, w, device))
    del base
    if device == "cuda":
        torch.cuda.empty_cache()

    unit = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)
    C = unit(cond[rows])          # bridge(EEG)  [N,512]
    E = unit(A)                   # true audio   [N,512]

    def report(name, S, idx, chance):
        ranks = [int(np.where(np.argsort(-S[i]) == idx[i])[0][0]) + 1 for i in range(len(idx))]
        n = len(idx)
        top1 = sum(r == 1 for r in ranks) / n
        mr = float(np.mean(ranks))
        own = np.array([S[i, idx[i]] for i in range(n)])
        others = np.array([np.delete(S[i], idx[i]).mean() for i in range(n)])
        d = own - others
        t = float(d.mean() / (d.std(ddof=1) / np.sqrt(n)))
        print(f"  {name:<34} top-1={top1:.3f} (chance {chance:.3f})  "
              f"mean rank={mr:.2f} (chance {(n+1)/2:.1f})  margin={d.mean():+.4f} t={t:+.2f}  n={n}")
        return dict(top1=top1, chance=chance, mean_rank=mr, margin=float(d.mean()), t=t, n=n)

    res = {}
    print("\n=== 跨歌（14 首歌混在一起，全 60 条）===")
    S = C @ E.T
    res["eeg_cross_song"] = report("bridge(EEG) -> 真音窗", S, list(range(len(rows))),
                                  1 / len(rows))

    # within-song subsets: for each song with >=8 windows, retrieve only among that song
    print("\n=== 歌内（同一首歌的窗之间检索）—— 时间跟随的天花板在哪 ===")
    tot_top1 = tot_n = 0
    ms, mt, mm = [], [], []
    for sg in np.unique(songs):
        ii = np.where(songs == sg)[0]
        if len(ii) < 6:
            continue
        sub = S[np.ix_(ii, ii)]
        r_ = report(f"song {sg} 内检索", sub, list(range(len(ii))), 1 / len(ii))
        tot_top1 += r_["top1"] * len(ii); tot_n += len(ii)
        ms.append(r_["top1"]); mt.append(r_["t"]); mm.append(r_["margin"])
    if tot_n:
        print(f"  --- 歌内汇总: top-1={tot_top1/tot_n:.3f}  "
              f"各歌 top-1 均值={np.mean(ms):.3f}  margin 均值={np.mean(mm):+.4f}")

    print("\n=== 正对照：真音自己检索自己（衡量任务本身可不可分）===")
    SAA = E @ E.T
    report("真音 -> 真音（同集合）", SAA, list(range(len(rows))), 1 / len(rows))
    within_audio = []
    for sg in np.unique(songs):
        ii = np.where(songs == sg)[0]
        if len(ii) < 6:
            continue
        sub = SAA[np.ix_(ii, ii)]
        rk = [int(np.where(np.argsort(-sub[i]) == i)[0][0]) + 1 for i in range(len(ii))]
        within_audio.append(sum(r == 1 for r in rk) / len(ii))
    print(f"  歌内 真音检索自己 top-1 均值={np.mean(within_audio):.3f} "
          f"(若这也低，说明这些 8s 窗在 CLAP 空间本就彼此难分，任务本身无解)")

    Path(ROOT / "results" / "gen_eegcond" / "trend" / "separability.json").write_text(
        json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
