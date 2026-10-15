"""Continuous EEG-driven demo + temporal-tracking test.

Why not just "generate one long stretch and listen": the train/val split is by CODE GROUP
(song, window), so only ~10% of a song's windows are held out. A long in-order stretch is
mostly targets the model memorised, which would flatter it.

So measure temporal tracking instead: S[t,s] = cos(gen_t, true_s). If the EEG condition
actually steers generation, similarity peaks at s=t and decays with |lag|; a model spitting
out an average of the song, or ignoring the condition, gives a flat profile. The
width-matched random-projection arm provides that flat reference.

Deliverables:
  one continuous generated wav + the matching true-stimulus wav (for listening), and the
  lag-decay curve for both arms.

Run: python src/train/continuous_demo.py --n 40
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
DATA = ROOT / "data" / "eegcond"
OUT = ROOT / "results" / "gen_eegcond" / "continuous"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

WAV_SR = 32000
CLIP_S = 8.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40, help="windows in the stretch")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--ckpt", default=str(ROOT / "results" / "gen_eegcond_k8" / "ckpt_ep16.pt"))
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--subject", default="")
    ap.add_argument("--run", default="")
    args = ap.parse_args()

    from pilot_multitoken import CondWidthModel
    from finetune_gen_k import load_state
    from musicgen_clap_finetune import load_clap, clap_embed_audio, MusicGenCLAP

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    tag_ckpt = Path(args.ckpt).stem
    import soundfile as sf

    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    song = z["song"].astype(int)
    seg2 = z["seg2"].astype(int)
    subj = np.array([str(x) for x in z["subject"]])
    run = np.array([str(x) for x in z["run"]])

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int64)
    group = np.load(DATA / "group.npy").astype(np.int64)

    # map cond.npy rows back to original cache rows (order-preserving byte walk)
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
    oidx = np.array(mapped)                     # cond row i  <-  original cache row oidx[i]
    c_song, c_seg, c_subj, c_run = song[oidx], seg2[oidx], subj[oidx], run[oidx]

    # pick the presentation with the most rows; prefer contiguous seg2 coverage
    keys = {}
    for i in range(len(cond)):
        keys.setdefault((c_subj[i], c_run[i], int(c_song[i])), []).append(i)
    cands = sorted(keys.items(), key=lambda kv: -len(kv[1]))
    if args.subject and args.run:
        cands = [kv for kv in cands if kv[0] == (args.subject, args.run) or
                 kv[0][0] == args.subject and kv[0][1] == args.run] or cands
    (sub, rn, sg), idxs = cands[0]
    idxs = np.array(idxs)
    sel = idxs[np.argsort(c_seg[idxs])]        # chronological within the presentation
    take = sel[:args.n]
    print(f"presentation: subject {sub} run {rn} song {sg}  -> {len(take)} windows "
          f"(seg2 {int(c_seg[take].min())}..{int(c_seg[take].max())})")
    uniq_g = np.unique(group[take])
    print(f"其中唯一码组 {len(uniq_g)} 个")

    clap, proc, csr = load_clap(device)
    unit = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    base = MusicGenCLAP(device=device, use_lora=False)
    TRUE_W, TRUE_E = [], []
    for r in take:
        oc = base.mg.audio_encoder.decode(
            torch.from_numpy(codes[r]).unsqueeze(0).unsqueeze(0).to(device),
            audio_scales=[None]).audio_values
        w = np.asarray(oc.cpu()).reshape(-1)
        TRUE_W.append(w)
        TRUE_E.append(clap_embed_audio(clap, proc, WAV_SR, w, device))
    T = unit(np.stack(TRUE_E))
    del base
    if device == "cuda":
        torch.cuda.empty_cache()

    fade = int(0.25 * WAV_SR)
    out_sr = WAV_SR

    def stitch(ws):
        if not ws:
            return np.zeros(0, np.float32)
        out = ws[0]
        for w in ws[1:]:
            f = min(fade, len(out), len(w))
            ramp = np.linspace(0, 1, f)
            out = np.concatenate([out[:-f], out[-f:] * (1 - ramp) + w[:f] * ramp, w[f:]])
        return out

    sf.write(str(OUT / f"TRUE_stretch_{tag_ckpt}_{len(take)}win.wav"), stitch(TRUE_W), out_sr)

    summary = {}
    for arm in ("trained", "randomproj"):
        t0 = time.time()
        if arm == "trained":
            model = CondWidthModel(device, k_tokens=args.k)
            load_state(model, args.ckpt)
            model.m.eval()
            print(f"[{arm}] loaded {Path(args.ckpt).name} k={args.k}")
        else:
            model = CondWidthModel(device, k_tokens=args.k)   # fresh proj, LoRA B=0
            model.m.eval()
            print(f"[{arm}] width-matched control (random proj, LoRA off by init)")
        GEN_W, GEN_E = [], []
        for r in take:
            w = np.asarray(model.generate(torch.from_numpy(cond[r]),
                                          max_new_tokens=args.tokens,
                                          temperature=args.temp)).reshape(-1)
            GEN_W.append(w)
            GEN_E.append(clap_embed_audio(clap, proc, WAV_SR, w, device))
            if arm == "trained":
                sf.write(str(OUT / f"win_{i:02d}_row{int(r)}.wav"), w, WAV_SR)
        G = unit(np.stack(GEN_E))
        S = G @ T.T                                       # [time_gen, time_true]
        n = len(take)
        far_lag = max(3, n // 5)                           # adapt: small smokes have no lag-8
        lags = np.arange(0, min(max(4, far_lag + 1), n))
        prof = []
        for L in lags:
            vals = [S[i, i + L] for i in range(n - L)] + [S[i + L, i] for i in range(n - L)]
            prof.append(float(np.mean(vals)))
        # per-timestamp paired margin: own time vs temporally distant true clips
        marg, own0, farm = [], [], []
        for i in range(n):
            distant = [j for j in range(n) if abs(i - j) >= far_lag]
            if not distant:
                continue
            marg.append(S[i, i] - float(np.mean([S[i, j] for j in distant])))
            own0.append(S[i, i])
            farm.append(float(np.mean([S[i, j] for j in distant])))
        marg = np.array(marg)
        ttrack = float(marg.mean() / (marg.std(ddof=1) / np.sqrt(len(marg)))) if len(marg) > 1 else float("nan")
        row = dict(lag_prof=[round(x, 4) for x in prof], far_lag=int(far_lag),
                   far=float(np.mean(farm)) if farm else float("nan"),
                   diag0=float(np.mean(own0)), temporal_margin=float(marg.mean()),
                   temporal_t=ttrack)
        summary[arm] = row
        print(f"  [{arm}] lag0={prof[0]:.4f} lag1={prof[1]:.4f} lag{far_lag}+={row['far']:.4f}  "
              f"时间裕度={row['temporal_margin']:+.4f} 配对 t={ttrack:+.2f}  "
              f"[{time.time()-t0:.0f}s]", flush=True)
        if arm == "trained":
            sf.write(str(OUT / f"GEN_stretch_{tag_ckpt}_{args.temp}_{len(take)}win.wav"), stitch(GEN_W), out_sr)
            (OUT / "gen_windows.txt").write_text(
                "\n".join(f"{i}\trow{int(r)}\tseg{int(c_seg[r])}" for i, r in enumerate(take)),
                encoding="utf-8")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    a = summary["trained"]
    c = summary["randomproj"]
    print("\n===== 时间跟随性判读 =====")
    print(f"微调:   lag0 {a['diag0']:.4f}  lag{a['far_lag']}+ {a['far']:.4f}  "
          f"时间裕度 {a['temporal_margin']:+.4f}  配对 t {a['temporal_t']:+.2f}")
    print(f"对照臂: lag0 {c['diag0']:.4f}  lag{c['far_lag']}+ {c['far']:.4f}  "
          f"时间裕度 {c['temporal_margin']:+.4f}  配对 t {c['temporal_t']:+.2f}")
    print(f"裕度之差 = {a['temporal_margin']-c['temporal_margin']:+.4f}  "
          f"(微调的 t 明显>0 而对照的 t≈0 => 生成在跟随刺激的时间进程，不是吐歌曲平均)")
    json.dump({"presentation": [sub, rn, int(sg)], "n_windows": int(len(take)),
               "seg_range": [int(c_seg[take].min()), int(c_seg[take].max())],
               **summary}, open(OUT / "result.json", "w"), indent=2)
    print(f"\n连续音频与曲线在 {OUT}")


if __name__ == "__main__":
    main()
