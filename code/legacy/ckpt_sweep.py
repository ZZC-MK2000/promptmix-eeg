"""Does condition-dependence DECAY with training? Sweep checkpoints on one fixed eval set.

Comparing two isolated points (pilot 8k steps vs full 120k steps) is not enough: each arm
was decoded once at temperature 0.9, and the margin difference is only ~1.3 SE. So score
several checkpoints on the SAME 30 held-out targets and look for a monotone trend.

Signals per checkpoint:
  window top-1 / mean rank / paired t   -> how well a clip finds its own stimulus
  GEN-v-GEN                              -> mutual similarity of outputs (rising = collapse)
  own - GEN-v-GEN                        -> the collapse test (negative = condition ignored)

Run: python src/train/ckpt_sweep.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
RES = ROOT / "results" / "gen_eegcond_k8"
IDK8 = ROOT / "results" / "gen_eegcond" / "identity_k8"
DATA = ROOT / "data" / "eegcond"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

WAV_SR = 32000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", default="2,8,16,23,29")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--reps", type=int, default=1, help="decodes per condition (averages noise)")
    args = ap.parse_args()

    from pilot_multitoken import CondWidthModel
    from finetune_gen_k import load_state
    from musicgen_clap_finetune import load_clap, clap_embed_audio

    device = "cuda" if torch.cuda.is_available() else "cpu"
    import soundfile as sf

    spec = json.loads((IDK8 / "test_spec.json").read_text(encoding="utf-8"))
    rows = spec["pick"]
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    print(f"{len(rows)} fixed held-out targets, songs dist "
          f"{np.bincount(np.array(spec['songs']), minlength=14).tolist()}")

    clap, proc, csr = load_clap(device)
    unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    def load32(p):
        w, s = sf.read(str(p), dtype="float32")
        assert s == WAV_SR, f"{p} rate {s}"
        return (w.mean(1) if w.ndim > 1 else w).astype(np.float32)

    tps = sorted(IDK8.glob("TRUE_*.wav"))
    rowkeys = [p.stem.split("row")[-1] for p in tps]
    T = unit(np.stack([clap_embed_audio(clap, proc, WAV_SR, load32(p), device) for p in tps]))
    idx = [rowkeys.index(f"{r:05d}") for r in rows]
    n = len(rows)
    # map by explicit row->song dict; do NOT rely on pick order matching sorted filenames
    row_song = {f"{int(rr):05d}": s for rr, s in zip(spec["pick"], spec["songs"])}
    songs = np.array([row_song[f"{r:05d}"] for r in rows])
    p_ = np.bincount(songs, minlength=14) / n
    chance_song = float((p_ * p_).sum())
    print(f"chance: window {1/n:.3f}   same-song {chance_song:.3f}")

    out = []
    for ep in [int(x) for x in args.eps.split(",")]:
        ck = RES / f"ckpt_ep{ep}.pt"
        if not ck.exists():
            print(f"ep{ep}: no ckpt, skip")
            continue
        t0 = time.time()
        model = CondWidthModel(device, k_tokens=args.k)
        load_state(model, str(ck))
        model.m.eval()
        G = []
        for rep in range(args.reps):
            for r in rows:
                w = np.asarray(model.generate(torch.from_numpy(cond[r]),
                                              max_new_tokens=args.tokens,
                                              temperature=args.temp)).reshape(-1)
                G.append(clap_embed_audio(clap, proc, WAV_SR, w, device))
                if rep == 0:
                    sf.write(str(IDK8 / f"sweep_ep{ep}_row{r:05d}.wav"), w, WAV_SR)
        Gm = unit(np.stack(G))
        S = Gm @ T.T
        if args.reps > 1:
            S = S.reshape(args.reps, n, n).mean(0)
        own = np.array([S[i, idx[i]] for i in range(n)])
        others = np.array([np.delete(S[i], idx[i]).mean() for i in range(n)])
        ranks = [int(np.where(np.argsort(-S[i]) == idx[i])[0][0]) + 1 for i in range(n)]
        GG = Gm @ Gm.T
        gg = float(GG[np.triu_indices(len(GG), 1)].mean())
        d = own - others
        tstat = float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))
        top1 = sum(r == 1 for r in ranks) / n
        ss = sum(1 for i in range(n) if songs[int(np.argmax(S[i]))] == songs[i]) / n
        rec = dict(ep=ep, steps=(ep + 1) * 4027, window_top1=top1, mean_rank=float(np.mean(ranks)),
                   margin=float(d.mean()), t=tstat, song_top1=ss, own=float(own.mean()),
                   gen_gen=gg, own_minus_gg=float(own.mean() - gg),
                   ranks=ranks, own_per_row=[round(float(x), 4) for x in own],
                   rows=[int(r) for r in rows])
        out.append(rec)
        print(f"ep{ep:>3} (steps {rec['steps']:>7}) win top1={top1:.3f} rank={rec['mean_rank']:.2f} "
              f"margin={rec['margin']:+.4f} t={tstat:+.2f} song={ss:.3f} "
              f"GEN-v-GEN={gg:.4f} own={own.mean():.4f} own-gg={rec['own_minus_gg']:+.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    (RES / f"ckpt_sweep_reps{args.reps}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("\n===== 趋势判读 =====")
    if len(out) >= 3:
        ts = [r["t"] for r in out]
        gs = [r["gen_gen"] for r in out]
        oms = [r["own_minus_gg"] for r in out]
        print(f"配对 t:  {' -> '.join('%+.2f' % x for x in ts)}   "
              f"{'随训练下滑' if ts[-1] < ts[0] else '未随训练下滑'}")
        print(f"GEN-v-GEN: {' -> '.join('%.4f' % x for x in gs)}   "
              f"{'升高=输出趋同(塌缩)' if gs[-1] > gs[0] else '未升高'}")
        print(f"own−GEN-v-GEN: {' -> '.join('%+.4f' % x for x in oms)}   "
              f"{'转负=条件被忽略' if oms[-1] < 0 <= oms[0] else '见上'}")
    print("注: 每个 ckpt 只解码一次时含采样噪声；--reps 2-3 可对每个条件多次解码取平均。")


if __name__ == "__main__":
    main()
