"""Mode (B): the literal "mix between prompts" -- the adapter's simplex weights decide which
prompts to blend and how much. Two realizations, both kept NATIVE per component (no T5-embedding
interpolation, which §6.2 shows goes off-manifold):

  B-cat  : treat weights as a categorical distribution and SAMPLE one prompt per window. Every
           delivered clip is a single native frozen sample -> pleasantness guaranteed; across many
           windows the genre proportions follow the EEG weights.
  B-mix  : sample the top-k prompts, normalize each clip to a common RMS, sum with the (renormalized)
           weights, soft-limit. This is a real waveform blend (layering stems). RISK: independent
           generations have different key/tempo, so the sum can be cacophonous. We measure texture
           and save clips so a human can judge -- pleasantness is the user's stated gate.

Run:
  python src/train/deliver_mode_b.py --arm true_oracle --mode B-cat --rows 12
  python src/train/deliver_mode_b.py --arm true_oracle --mode B-mix --topk 3 --rows 12
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
LIB = ROOT / "results" / "gen_eegcond" / "library_radius"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

SR = 32000
K = 12


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def zcr(w):
    return float(np.mean(np.abs(np.diff(np.sign(w))))) / 2


def soft_limit(w, thr=0.98):
    return np.tanh(w / thr) * thr


def heldout_rows(group, n, seed=0):
    uniq = np.unique(group)
    rng = np.random.default_rng(seed)
    val_g = set(uniq[rng.permutation(len(uniq))[:max(1, int(round(len(uniq) * 0.1)))]].tolist())
    va = np.array([g in val_g for g in group])
    idx = np.where(va)[0]
    return idx[:n] if n else idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="true_oracle", choices=["cond_oracle", "true_oracle"])
    ap.add_argument("--mode", default="B-mix", choices=["B-cat", "B-mix"])
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--tau", type=float, default=0.7, help="softmax temperature on logits")
    ap.add_argument("--rows", type=int, default=12)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_dir = (ROOT / "results" / "prompt_classifier" if args.arm == "cond_oracle"
                else ROOT / "results" / "prompt_classifier_true")
    ckpt_path = ckpt_dir / "final.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"no classifier at {ckpt_path}")
    OUT = ckpt_dir / "listen_B"
    OUT.mkdir(parents=True, exist_ok=True)

    from musicgen_clap_finetune import MusicGenCLAP, load_clap, clap_embed_audio
    from finetune_prompt_classifier import PromptClassifier, PROMPTS
    from arms_2x2 import embed_text, gen_from_enc
    import soundfile as sf

    t0 = time.time()
    print(f"arm={args.arm} mode={args.mode} topk={args.topk} tau={args.tau}", flush=True)

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int64)
    group = np.load(DATA / "group.npy").astype(np.int64)
    rows = heldout_rows(group, args.rows)

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = PromptClassifier(in_dim=512, hidden=ck["args"].get("hidden", 256), k=K, dropout=0.1)
    model.load_state_dict(ck["model"]); model.eval()

    base = MusicGenCLAP(device=device, use_lora=False); mg = base.mg; mg.eval()
    clap, proc, csr = load_clap(device)
    prompt_h = {k: embed_text(mg, PROMPTS[k], device) for k in range(K)}

    zc_mix, cos_mix = [], []
    rec = []
    for r in rows:
        with torch.no_grad():
            logits = model(torch.from_numpy(cond[r:r+1])).float()
        w = torch.softmax(logits / args.tau, dim=1)[0].cpu().numpy()
        order = np.argsort(w)[::-1]

        if args.mode == "B-cat":
            k_pick = int(np.random.default_rng(args.seed + int(r)).choice(K, p=w / w.sum()))
            comps = [(k_pick, 1.0)]
        else:
            kk = order[:args.topk]
            ww = w[kk] / w[kk].sum()
            comps = list(zip(kk.tolist(), ww.tolist()))

        clips = []
        for idx, wi in comps:
            c = gen_from_enc(mg, prompt_h[idx].unsqueeze(0), args.tokens, args.temperature,
                             None, seed=args.seed + int(r) * 7 + idx)
            rms = float(np.sqrt(np.mean(c ** 2))) + 1e-9
            clips.append((c / rms) * wi)     # per-clip RMS-normalized, weighted by mixture weight
        L = min(len(c) for c in clips)
        mix = soft_limit(sum(c[:L] for c in clips))   # waveform blend (B-mix) or single clip (B-cat)
        sf.write(str(OUT / f"{args.mode}__row{r}.wav"), mix, SR)

        e_gen = unit(clap_embed_audio(clap, proc, SR, mix, device))
        oc = mg.audio_encoder.decode(torch.from_numpy(codes[r]).unsqueeze(0).unsqueeze(0).to(device),
                                     audio_scales=[None]).audio_values
        ref = np.asarray(oc.cpu()).reshape(-1)
        e_ref = unit(clap_embed_audio(clap, proc, csr, ref, device))
        zc = zcr(mix); cs = float(e_gen @ e_ref)
        zc_mix.append(zc); cos_mix.append(cs)
        rec.append({"row": int(r), "mode": args.mode,
                    "comps": [(int(i), float(round(wi, 3))) for i, wi in comps],
                    "zcr": zc, "peak": float(np.abs(mix).max()), "cos_to_true": cs,
                    "wav": str(OUT / f"{args.mode}__row{r}.wav")})
        top = ",".join(f"p{i}:{wi:.2f}" for i, wi in comps)
        print(f"  row{r}  zcr={zc:.4f} peak={np.abs(mix).max():.2f} cos={cs:.3f}  [{top}]",
              flush=True)

    zc_mix = np.array(zc_mix); cos_mix = np.array(cos_mix)
    print(f"\n=== {args.mode}: {args.arm} | zcr mean={zc_mix.mean():.4f} "
          f"(native 0.02-0.06) | peak mean={np.mean([x['peak'] for x in rec]):.3f} | "
          f"cos_to_true mean={cos_mix.mean():.4f} ===")
    print("NOTE B-mix sums independent generations: high zcr or clash = cacophony; listen to decide.")
    json.dump({"arm": args.arm, "mode": args.mode, "topk": args.topk, "tau": args.tau,
               "zcr_mean": float(zc_mix.mean()), "cos_to_true_mean": float(cos_mix.mean()),
               "rows": [int(x) for x in rows], "per_row": rec},
              open(OUT / f"report_{args.mode}.json", "w"), indent=2)
    print(f"clips in {OUT}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
