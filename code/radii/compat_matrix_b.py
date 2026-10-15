"""Complete the TOLERANCE-RADIUS map: the objective's central gate was "how far can a mixture go
before the frozen generator leaves its manifold". library_radius measured only 3 prompt pairs;
this measures ALL C(12,2)=66 pairs of the corpus-matched library at alpha=0.5 (2 reps), so we get
the full off-manifold map and a "which pairs are safe to interpolate/mix" table.

For each pair (i,j): condition h = 0.5*(h_i + h_j) (T5-embedding interpolation, the mode-C readout
that §6.2 warns against), sample the frozen MusicGen, measure zero-crossing rate. A pair is SAFE if
its interpolated zcr stays near the native band; DANGEROUS if it blows up (off-manifold noise).
This maps the tolerance radius over the whole library, not 3 hand-picked pairs.

Run: python src/train/compat_matrix_b.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
LIBB = ROOT / "results" / "library_b"
OUT = LIBB / "compat"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

SR = 32000
NATIVE_HI = 0.06      # top of the soft frozen-native band
DANGER = 0.15         # zcr above this = off-manifold / rough


def zcr(w):
    return float(np.mean(np.abs(np.diff(np.sign(w))))) / 2


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    from library_b_experiment import PROMPTS_B
    from musicgen_clap_finetune import MusicGenCLAP
    from arms_2x2 import embed_text, gen_from_enc
    import soundfile as sf

    t0 = time.time()
    base = MusicGenCLAP(device=device, use_lora=False); mg = base.mg; mg.eval()
    H = {k: embed_text(mg, PROMPTS_B[k], device) for k in range(len(PROMPTS_B))}
    K = len(PROMPTS_B)
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    print(f"{len(pairs)} pairs, alpha=0.5, 2 reps, temperature=0.9 tokens=400", flush=True)

    rows = []
    for (i, j) in pairs:
        hi, hj = H[i], H[j]
        L = min(hi.shape[0], hj.shape[0])
        h = 0.5 * (hi[:L] + hj[:L])
        zs = []
        for rep in range(2):
            w = gen_from_enc(mg, h.unsqueeze(0), 400, 0.9, None, seed=777 + i * 100 + j + rep)
            zs.append(zcr(w))
            if rep == 0 and zs[-1] > DANGER and len(list(OUT.glob("bad_*.wav"))) < 8:
                sf.write(str(OUT / f"bad_p{i}_{j}.wav"), w, SR)
        z = float(np.mean(zs))
        rows.append({"pair": [i, j], "prompts": [PROMPTS_B[i][:22], PROMPTS_B[j][:22]],
                     "zcr": z, "safe": bool(z <= NATIVE_HI), "dangerous": bool(z > DANGER)})
        flag = "DANGER" if z > DANGER else ("safe" if z <= NATIVE_HI else "mid")
        print(f"  p{i:>2}-p{j:<2} zcr={z:.4f} {flag:6} [{time.time()-t0:.0f}s]", flush=True)

    zc = np.array([r["zcr"] for r in rows])
    nsafe = sum(r["safe"] for r in rows); ndang = sum(r["dangerous"] for r in rows)
    print(f"\n=== tolerance map over {len(pairs)} pairs (corpus-matched lib) ===")
    print(f"  zcr: median={np.median(zc):.4f} mean={zc.mean():.4f} max={zc.max():.4f}")
    print(f"  SAFE (zcr<={NATIVE_HI}): {nsafe}/{len(rows)}   DANGEROUS (zcr>{DANGER}): {ndang}/{len(rows)}")
    print(f"  -> embedding interpolation is off-manifold for {ndang} pairs; waveform mixing (B-mix) "
          f"or categorical (B-cat) avoid this.")
    worst = sorted(rows, key=lambda r: -r["zcr"])[:8]
    print("  worst pairs:")
    for r in worst:
        print(f"    p{r['pair'][0]}-p{r['pair'][1]} zcr={r['zcr']:.4f}  {r['prompts']}")
    json.dump({"n_pairs": len(rows), "zcr_median": float(np.median(zc)), "zcr_mean": float(zc.mean()),
               "zcr_max": float(zc.max()), "n_safe": nsafe, "n_dangerous": ndang,
               "native_hi": NATIVE_HI, "danger": DANGER, "rows": rows},
              open(OUT / "compat.json", "w"), indent=2)
    print(f"\nsaved -> {OUT/'compat.json'}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
