"""Score the CPU listen clips objectively: does generated audio resemble ITS OWN
stimulus more than a random other one?

laion-CLAP unit vectors cluster (random music pairs already score high), so a bare
cosine is meaningless -- the constant-baseline trap. We therefore report matched-pair
cos against the cross-pair (gen vs some OTHER clip's true audio) distribution, per arm.

Run:  python src/train/score_listen.py
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
CLIP = ROOT / "results" / "gen_eegcond" / "listen_cpu"
sys.path.insert(0, str(M2 / "src" / "generate"))


def _load(p, sr):
    import soundfile as sf
    import librosa
    w, s = sf.read(str(p), dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    if s != sr:
        w = librosa.resample(w, orig_sr=s, target_sr=sr)
    return w


def main():
    import torch
    torch.set_num_threads(6)
    from musicgen_clap_finetune import load_clap, clap_embed_audio

    device = "cpu"
    model, proc, sr = load_clap(device)

    gens = sorted(CLIP.glob("gen_*.wav"))
    trues = sorted(CLIP.glob("TRUE_*.wav"))
    if not gens:
        print(f"no gen clips in {CLIP}")
        return
    print(f"{len(gens)} generated, {len(trues)} true references")

    row = lambda p: p.stem.split("row")[-1]
    arm = lambda p: p.stem.split("_")[1]

    temb = {row(p): clap_embed_audio(model, proc, sr, _load(p, sr), device) for p in trues}
    gemb = {(arm(p), row(p)): clap_embed_audio(model, proc, sr, _load(p, sr), device)
            for p in gens}

    unit = lambda v: v / (np.linalg.norm(v) + 1e-9)
    g = {k: unit(v) for k, v in gemb.items()}
    t = {k: unit(v) for k, v in temb.items()}

    tt = [float(t[a] @ t[b]) for i, a in enumerate(sorted(t)) for b in sorted(t)[i + 1:]]
    print(f"(参照) TRUE-v-TRUE mean={np.mean(tt):.4f}  -- 随便两段真音乐之间的相似度"
          if tt else "(参照) 需要 >=2 条真音")

    for a in sorted({k[0] for k in g}):
        rows = [r for (ar, r) in g if ar == a and r in t]
        if not rows:
            continue
        match = [float(g[(a, k)] @ t[k]) for k in rows]
        cross = [float(g[(a, k)] @ t[j]) for k in rows for j in rows if k != j]
        gg = [float(g[(a, p1)] @ g[(a, p2)]) for i, p1 in enumerate(rows)
              for p2 in rows[i + 1:]]
        print(f"\n### arm={a}  n_clips={len(rows)}")
        for k, c in sorted(zip(rows, match)):
            print(f"  row {k}: matched cos={c:.4f}")
        cm = float(np.mean(match))
        if cross:
            xm, xsd = float(np.mean(cross)), float(np.std(cross))
            print(f"  MATCHED mean={cm:.4f}   CROSS mean={xm:.4f} sd={xsd:.4f}"
                  + (f"   GEN-v-GEN mean={np.mean(gg):.4f}" if gg else ""))
            lift = cm - xm
            print(f"  LIFT (matched - cross) = {lift:+.4f}"
                  + (f"  = {lift/(xsd+1e-9):+.2f} cross sd" if xsd > 0 else ""))
            print(f"  matched 高于乱配均值者: {sum(1 for c in match if c > xm)}/{len(rows)}")
        else:
            print(f"  MATCHED mean={cm:.4f}   (只有 1 条, 算不了 LIFT)")

    print("\n判读: LIFT<=0 说明生成的音频跟'自己那段刺激'并不比别的刺激更像 -> 条件没起作用")


if __name__ == "__main__":
    main()
