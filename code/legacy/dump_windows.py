"""Dump listenable 8-second windows for one checkpoint, matched to the three-way eval rows.

Rows come from pick_rows(group, --n-pool) -- the same deterministic held-out selection
mixed_eval.py uses -- so the first --count files line up 1:1 with the TRUE_/B_ep16_ clips
already in results/listen_now. Deliberately no CLAP here: this only writes audio, which keeps
VRAM low enough to run beside an evaluation.

Run: python src/train/dump_windows.py --ckpt results/gen_fma2eeg/ckpt_s8000.pt --prefix D_fma2eeg
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))

SR = 32000
RECIPE = dict(temperature=1.0, top_k=250)


def zcr(w):
    return float(np.mean(np.abs(np.diff(np.sign(w))))) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prefix", required=True, help="filename tag, e.g. D_fma2eeg")
    ap.add_argument("--count", type=int, default=8, help="how many windows to write")
    ap.add_argument("--reps", type=int, default=2, help="independent draws per target")
    ap.add_argument("--n-pool", type=int, default=60, help="held-out pool size used by the eval")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--outdir", default=str(ROOT / "results" / "listen_now"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from mixed_eval import pick_rows
    from pilot_multitoken import CondWidthModel
    import soundfile as sf

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    data = ROOT / "data" / "eegcond"
    cond = np.load(data / "cond.npy").astype(np.float32)
    group = np.load(data / "group.npy").astype(np.int64)
    pool = pick_rows(group, args.n_pool)
    rows = pool[:args.count]
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    print(f"目标行号（与评估同一套留 out 选取）: {[int(r) for r in rows]}  配方 {RECIPE}", flush=True)

    t0 = time.time()
    model = CondWidthModel(device, k_tokens=args.k)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    model.m.clap_proj.load_state_dict(ck["clap_proj"])
    model.m.mg.decoder.load_state_dict(ck["decoder_lora"], strict=False)
    model.m.eval()
    print(f"loaded {args.ckpt} [{time.time()-t0:.0f}s]", flush=True)

    for rep in range(max(1, args.reps)):
        zs = []
        for r in rows:
            w = np.asarray(model.generate(torch.from_numpy(cond[int(r)]),
                                          max_new_tokens=args.tokens, **RECIPE)).reshape(-1)
            zs.append(zcr(w))
            if rep == 0:
                p = out / f"{args.prefix}_row{int(r):05d}.wav"
                sf.write(str(p), w, SR)
            print(f"  {args.prefix} rep{rep} row{int(r):05d} 过零率={zs[-1]:.4f} "
                  f"峰值={float(np.abs(w).max()):.2f} 时长={len(w)/SR:.2f}s "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        print(f"  {args.prefix} rep{rep} 平均过零率={np.mean(zs):.4f}"
              f"（真音经码本解码回来的天花板参照≈0.0556，越高越毛刺）", flush=True)
    print(f"done -> {out} [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
