"""Generation-part deliverable: drive the frozen MusicGen from EEG and produce LISTENABLE,
musical audio -- the only hard requirement on the generation side is "pleasant music", NOT
song match (song-retrieval is the ALIGNMENT part's job, already at 84% 14-way via the bridge).

Musicality is guaranteed by construction here: for every held-out EEG window we pick a prompt
from the classifier and sample from the FROZEN MusicGen under that prompt's own native T5
embedding -- so every delivered clip lives on the generator's native manifold (zcr ~ 0.02-0.05,
the frozen-prompt texture), unlike the fine-tuned arms (ep16 0.0699, mixed2 0.0793) and unlike
embedding interpolation which can blow up (jazz<->choir alpha=0.25 -> 0.46).

Two modes delivered side by side so a human can A/B them against the real heard song:
  gen__<arm>__row<r>__p<prompt>.wav      -- the EEG-selected prompt, frozen sampling
  ref__row<r>.wav                         -- the decoded TRUE audio window (what was heard)

Arms:
  cond_oracle classifier (exists now)  -- EEG -> its argmax prompt (self-consistent selection)
  true_oracle classifier (if trained)  -- EEG -> song-matched genre prompt (ties to alignment)

Run:  python src/train/deliver_mode_a.py --rows 12 --arm cond_oracle
      python src/train/deliver_mode_a.py --rows 12 --arm true_oracle
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
PROMPTS = [
    "a warm melodic pop piece with soft piano and light drums",
    "upbeat indian film music with strings and tabla",
    "ambient electronic texture with pads and gentle beat",
    "acoustic guitar folk melody, calm and bright",
    "driving electronic dance track with a four-on-the-floor kick and bright synths",
    "lo-fi hip hop beat with dusty drums and mellow keys",
    "sweeping orchestral film score with strings and brass swells",
    "sparse percussion groove with congas and shakers",
    "smooth jazz trio with upright bass, brushed drums and electric piano",
    "distorted electric guitar rock riff with a live drum kit",
    "ethereal choir pad with slow harmonic movement",
    "bouncy chiptune melody with square waves and fast arpeggios",
]
K = 12


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def zcr(w):
    return float(np.mean(np.abs(np.diff(np.sign(w))))) / 2


def heldout_rows(group, n, seed=0):
    """Rows from val groups (the same 10% split as the classifier) so the EEG is out-of-sample."""
    uniq = np.unique(group)
    rng = np.random.default_rng(seed)
    val_g = set(uniq[rng.permutation(len(uniq))[:max(1, int(round(len(uniq) * 0.1)))]].tolist())
    va = np.array([g in val_g for g in group])
    idx = np.where(va)[0]
    return idx[:n] if n else idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="cond_oracle", choices=["cond_oracle", "true_oracle"])
    ap.add_argument("--rows", type=int, default=12, help="held-out EEG rows to deliver (0=all val rows)")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    ckpt_dir = (ROOT / "results" / "prompt_classifier" if args.arm == "cond_oracle"
                else ROOT / "results" / "prompt_classifier_true")
    ckpt_path = ckpt_dir / "final.pt"
    if not ckpt_path.exists():
        raise SystemExit(f"no classifier at {ckpt_path} (arm={args.arm}). Train it first.")
    OUT = ckpt_dir / "listen"
    OUT.mkdir(parents=True, exist_ok=True)
    top_k = args.top_k or None

    from musicgen_clap_finetune import MusicGenCLAP, load_clap, clap_embed_audio
    from finetune_prompt_classifier import PromptClassifier
    from arms_2x2 import embed_text, gen_from_enc
    import soundfile as sf

    t0 = time.time()
    print(f"arm={args.arm} ckpt={ckpt_path}", flush=True)

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int64)   # EnCodec codebook needs Long/Int, not int16
    group = np.load(DATA / "group.npy").astype(np.int64)
    rows = heldout_rows(group, args.rows)
    print(f"{len(rows)} held-out rows: {rows[:15].tolist()}", flush=True)

    # classifier
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model = PromptClassifier(in_dim=512, hidden=ck["args"].get("hidden", 256), k=K, dropout=0.1)
    model.load_state_dict(ck["model"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(cond[rows]))
    pred = logits.argmax(1).numpy()
    prob = torch.softmax(logits, 1).max(1).values.numpy()

    # frozen generator
    base = MusicGenCLAP(device=device, use_lora=False)
    mg = base.mg; mg.eval()
    clap, proc, csr = load_clap(device)
    lib_emb = np.load(LIB / "library_emb.npy").astype(np.float32)
    lib_owner = np.load(LIB / "library_owner.npy").astype(np.int64)
    means = unit(np.stack([unit(lib_emb[lib_owner == k].mean(0)) for k in range(K)]))

    # pre-embed the 12 native prompts once
    prompt_h = {k: embed_text(mg, PROMPTS[k], device) for k in range(K)}

    zc_gen, zc_ref, cos_ref = [], [], []
    rec = []
    for i, r in enumerate(rows):
        pk = int(pred[i])
        h = prompt_h[pk]
        w = gen_from_enc(mg, h.unsqueeze(0), args.tokens, args.temperature, top_k,
                         seed=args.seed + int(r))
        gp = OUT / f"gen__{args.arm}__row{r}__p{pk}.wav"
        sf.write(str(gp), w, SR)

        oc = mg.audio_encoder.decode(torch.from_numpy(codes[r]).unsqueeze(0).unsqueeze(0).to(device),
                                     audio_scales=[None]).audio_values
        ref = np.asarray(oc.cpu()).reshape(-1)
        rp = OUT / f"ref__row{r}.wav"
        sf.write(str(rp), ref, SR)

        e_gen = unit(clap_embed_audio(clap, proc, SR, w, device))
        e_ref = unit(clap_embed_audio(clap, proc, csr, ref, device))
        cos_gr = float(e_gen @ e_ref)
        zc_gen.append(zcr(w)); zc_ref.append(zcr(ref)); cos_ref.append(cos_gr)
        rec.append({"row": int(r), "pred_prompt": pk, "pred_prompt_text": PROMPTS[pk],
                    "conf": float(prob[i]), "zcr_gen": zcr(w), "zcr_ref": zcr(ref),
                    "cos_gen_vs_trueaudio": cos_gr, "peak_gen": float(np.abs(w).max()),
                    "gen_wav": str(gp), "ref_wav": str(rp)})
        print(f"  row{r} -> p{pk} ({PROMPTS[pk][:28]:28}) conf={prob[i]:.2f} "
              f"zcr_gen={zcr(w):.4f} zcr_ref={zcr(ref):.4f} cos={cos_gr:.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)

    zc_gen = np.array(zc_gen); zc_ref = np.array(zc_ref); cos_ref = np.array(cos_ref)
    print(f"\n=== {args.arm}: delivered music texture ===")
    print(f"  zcr_gen mean={zc_gen.mean():.4f} (frozen native ~0.02-0.05; ep16 0.0699; "
          f"mixed2 0.0793; truth {zc_ref.mean():.4f})")
    print(f"  peak_gen mean={np.mean([x['peak_gen'] for x in rec]):.3f} "
          f"(clipping if >1.0)")
    print(f"  cos(gen, true-audio) mean={cos_ref.mean():.4f} "
          f"-- generation is NOT required to song-match; this is context, not a gate.")
    print(f"  listenable clips in {OUT}")
    json.dump({"arm": args.arm, "n": len(rows), "rows": [int(x) for x in rows],
               "zcr_gen_mean": float(zc_gen.mean()), "zcr_truth_mean": float(zc_ref.mean()),
               "cos_gen_true_mean": float(cos_ref.mean()),
               "chance_cos_note": "generation only requires pleasantness; retrieval (84%) covers song match",
               "per_row": rec}, open(OUT / "listen_report.json", "w"), indent=2)
    print(f"\ndone -> {OUT}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
