"""N-way identity retrieval on the listen clips.

Average matched-minus-cross cosine is mushy (CLAP vectors cluster, sd is large). The
hard question is discriminative: given one generated clip, is ITS OWN stimulus the
closest among all N true references? Chance = 1/N. Run this per arm.

Run:  python src/train/retrieval_test.py
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
    model, proc, sr = load_clap("cpu")
    unit = lambda v: v / (np.linalg.norm(v) + 1e-9)

    trues = sorted(CLIP.glob("TRUE_*.wav"))
    rows = [p.stem.split("row")[-1] for p in trues]
    T = np.stack([unit(clap_embed_audio(model, proc, sr, _load(p, sr), "cpu"))
                  for p in trues])
    print(f"{len(rows)} held-out true references")

    for arm in ("trained", "randomproj"):
        ps = sorted(CLIP.glob(f"gen_{arm}_row*.wav"))
        if not ps:
            continue
        idx = [rows.index(p.stem.split("row")[-1]) for p in ps]
        G = np.stack([unit(clap_embed_audio(model, proc, sr, _load(p, sr), "cpu"))
                      for p in ps])
        S = G @ T.T                                   # [n_gen, n_true]
        # rank of the own target (1 = best)
        ranks = []
        for gi, own in enumerate(idx):
            order = np.argsort(-S[gi])
            ranks.append(int(np.where(order == own)[0][0]) + 1)
        n = len(idx)
        mean_rank = float(np.mean(ranks))
        # mean similarity to own vs to the best other
        own_sim = [float(S[i, idx[i]]) for i in range(n)]
        # compare own against the MEAN of the others; the max of others is upward-biased
        # by construction (1 draw vs 11), so it is not a fair reference.
        others_mean = []
        for i in range(n):
            row = np.delete(S[i], idx[i])
            others_mean.append(float(row.mean()))
        margin = float(np.mean(own_sim) - np.mean(others_mean))
        se = float(np.std(np.array(own_sim) - np.array(others_mean), ddof=1) / np.sqrt(n))
        wins = sum(1 for i in range(n) if S[i, idx[i]] == S[i].max())
        print(f"\n### arm={arm}  n={n}   chance top-1 = {1/n:.3f}")
        print(f"  retrieval top-1 = {wins/n:.3f}  ({wins}/{n})")
        print(f"  mean rank of own stimulus = {mean_rank:.2f} (chance {(n+1)/2:.2f}, best 1)")
        print(f"  cos to OWN        mean={np.mean(own_sim):.4f}")
        print(f"  cos to OTHERS-AVG mean={np.mean(others_mean):.4f}  -> margin={margin:+.4f}")
        print(f"  paired SE of margin = {se:.4f}  (margin/SE = {margin/max(se,1e-9):+.2f})")
        print(f"  ranks={ranks}")
        print(f"  判读: top-1 接近 {1/n:.2f} 或 margin/SE<2 => 条件没稳定带出'是哪一段'的信息")


if __name__ == "__main__":
    main()
