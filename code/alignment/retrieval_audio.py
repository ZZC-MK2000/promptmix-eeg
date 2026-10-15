"""Turn the retrieval line into listenable audio: reconstruct what a subject heard, out of
real recorded music (no generation, so fidelity is not the issue -- only selection is).

Two stages, each backed by a measured number:
  1. song track: per 8s window vote the song, then smooth with a 9-window majority filter
     (measured: 84.93% per-window -> 88.41% smoothed, chance 7.14%, constant baseline 7.88%)
  2. within-song window pick: choose the highest-cosine stimulus window inside that song
     (measured: 3.78% exact-window rate given the right song, vs 1.28% by chance = 3x)

Output per subject-run: a continuous real-audio track, the true-window track for A/B, and a
per-window reconciliation file saying what was right and what was substituted.

CPU decode by default so it does not fight the GPU training run.

Run: python src/train/retrieval_audio.py --runs 2
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
OUT = ROOT / "results" / "retrieval_line" / "audio"
sys.path.insert(0, str(M2 / "src" / "generate"))

SR = 32000


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def majority(seq, w):
    h = w // 2
    return [Counter(seq[max(0, i - h):i + h + 1]).most_common(1)[0][0] for i in range(len(seq))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--smooth", type=int, default=9)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    from musicgen_clap_finetune import MusicGenCLAP

    OUT.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int64)
    group = np.load(DATA / "group.npy").astype(np.int64)

    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    song_all = z["song"].astype(int)
    subj_all = np.array([str(x) for x in z["subject"]])
    run_all = np.array([str(x) for x in z["run"]])
    seg_all = z["seg2"].astype(int)
    target_all = z["target"].astype(np.float32)
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
    tgt = unit(target_all[mapped])

    gvals = np.unique(group)
    first = {int(g): int(np.where(group == g)[0][0]) for g in gvals}
    crows = np.array([first[int(g)] for g in gvals])
    T = tgt[crows]
    c_song, c_group = q_song[crows], group[crows]

    Q = unit(cond)
    S = Q @ T.T

    keys = {}
    for i in range(len(cond)):
        keys.setdefault((q_subj[i], q_run[i]), []).append(i)
    # a run's windows may span songs; keep the longest single-song stretch
    cand = []
    for (sb, rn), idxs in keys.items():
        idxs = np.array(idxs)
        for sg in np.unique(q_song[idxs]):
            ii = idxs[q_song[idxs] == sg]
            if len(ii) >= 24:
                cand.append((len(ii), sb, rn, int(sg), ii[np.argsort(q_seg[ii])]))
    cand.sort(key=lambda t: -t[0])
    picked = cand[:args.runs]
    print(f"{len(cand)} candidate runs; taking the {len(picked)} longest")

    mg = MusicGenCLAP(device=args.device, use_lora=False)
    dec_cache = {}

    def decode(gi):
        if gi not in dec_cache:
            r = int(np.where(group == gi)[0][0])
            oc = mg.mg.audio_encoder.decode(
                torch.from_numpy(codes[r]).unsqueeze(0).unsqueeze(0).to(args.device),
                audio_scales=[None]).audio_values
            dec_cache[gi] = np.asarray(oc.cpu()).reshape(-1).astype(np.float32)
        return dec_cache[gi]

    def stitch(ws):
        if not ws:
            return np.zeros(0, np.float32)
        fade = int(0.25 * SR)
        out = np.asarray(ws[0]).reshape(-1)
        for w in ws[1:]:
            w = np.asarray(w).reshape(-1)
            f = min(fade, len(out), len(w))
            ramp = np.linspace(0, 1, f)
            out = np.concatenate([out[:-f], out[-f:] * (1 - ramp) + w[:f] * ramp, w[f:]])
        return out

    summary = []
    for nwin, sb, rn, sg, idxs in picked:
        t0 = time.time()
        raw_vote = [int(c_song[int(np.argmax(S[i]))]) for i in idxs]
        smooth = majority(raw_vote, args.smooth) if args.smooth > 1 else raw_vote
        acc_raw = float(np.mean([v == sg for v in raw_vote]))
        acc_sm = float(np.mean([v == sg for v in smooth]))

        sel_groups, sel_cos = [], []
        for qi, vote in zip(idxs, smooth):
            pool = np.where(c_song == vote)[0]
            if len(pool) == 0:
                pool = np.array([int(np.argmax(S[qi]))])
            pick = int(pool[np.argmax(S[qi, pool])])
            sel_groups.append(int(c_group[pick]))
            sel_cos.append(float(S[qi, pick]))

        gen_w = [decode(g) for g in sel_groups]
        tru_w = [decode(int(group[i])) for i in idxs]
        tag = f"{sb}_{rn}_song{sg}"
        sf.write(str(OUT / f"RETRIEVED_{tag}.wav"), stitch(gen_w), SR)
        sf.write(str(OUT / f"TRUE_{tag}.wav"), stitch(tru_w), SR)
        win_ok = float(np.mean([g == int(group[i]) for g, i in zip(sel_groups, idxs)]))
        rows = [{"i": int(k), "seg": int(q_seg[i]), "true_group": int(group[i]),
                 "retrieved_group": sel_groups[k], "vote_song": int(smooth[k]),
                 "cos": round(sel_cos[k], 4), "song_ok": int(smooth[k] == sg),
                 "window_ok": int(sel_groups[k] == int(group[i]))}
                for k, i in enumerate(idxs)]
        (OUT / f"trace_{tag}.json").write_text(json.dumps(
            {"subject": sb, "run": rn, "song": sg, "windows": int(len(idxs)),
             "song_acc_raw": acc_raw, "song_acc_smoothed": acc_sm,
             "window_acc": win_ok, "seconds": round(len(idxs) * 8),
             "files": [f"RETRIEVED_{tag}.wav", f"TRUE_{tag}.wav"], "rows": rows},
            indent=2), encoding="utf-8")
        print(f"  {tag}: {len(idxs)} 窗 ≈{len(idxs)*8/60:.1f} 分钟  歌级 "
              f"{acc_raw:.3f}→{acc_sm:.3f}(平滑)  窗级 {win_ok:.3f}  [{time.time()-t0:.0f}s]")
        summary.append({"tag": tag, "windows": int(len(idxs)), "song_raw": acc_raw,
                        "song_smooth": acc_sm, "window": win_ok})

    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n音频与对账单在 {OUT}")
    print("每个 run 两条等长音频：RETRIEVED_（按脑电选出的真音乐）与 TRUE_（实际听的），"
          "可直接 A/B；trace_*.json 逐窗写明选对/选错")


if __name__ == "__main__":
    main()
