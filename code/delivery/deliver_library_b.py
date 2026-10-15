"""Improved flagship listening set: deliver pleasant music using the CORPUS-MATCHED library
(library_b_experiment) instead of the original (whose dominant genre was a chiptune artifact).

Trains + saves the EEG->genre classifier on library_b's oracle_all.npy, then for held-out subjects
picks the argmax corpus-matched prompt and samples the FROZEN MusicGen under that prompt's own
native T5 embedding. Every clip is a native sample (pleasant by construction) but now in a genre
that plausibly matches the Bollywood/film-heavy corpus. Saves gen + true-audio-ref wavs + a report.

Run: python src/train/deliver_library_b.py --rows 14
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
LIBB = ROOT / "results" / "library_b"
OUT = LIBB / "listen"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

SR = 32000
K = 12


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def zcr(w):
    return float(np.mean(np.abs(np.diff(np.sign(w))))) / 2


def soft_limit(w, thr=0.95):
    return np.tanh(w / thr) * thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=14)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    from library_b_experiment import PROMPTS_B
    from musicgen_clap_finetune import MusicGenCLAP, load_clap, clap_embed_audio
    from finetune_prompt_classifier import PromptClassifier
    from arms_2x2 import embed_text, gen_from_enc
    import soundfile as sf

    t0 = time.time()
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int64)
    group = np.load(DATA / "group.npy").astype(np.int64)
    oracle = np.load(LIBB / "oracle_all.npy").astype(np.int64)

    # train + save the corpus-matched classifier (CPU, fast)
    uniq = np.unique(group); rng = np.random.default_rng(0)
    val_g = set(uniq[rng.permutation(len(uniq))[:max(1, int(round(len(uniq) * 0.1)))]].tolist())
    va = np.array([g in val_g for g in group])
    model = PromptClassifier(512, 256, K, 0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    crit = torch.nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(torch.from_numpy(cond[~va]), torch.from_numpy(oracle[~va]))
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    model.train(); it = iter(dl)
    for step in range(1, 3001):
        try: xb, yb = next(it)
        except StopIteration: it = iter(dl); xb, yb = next(it)
        loss = crit(model(xb.to(device)), yb.to(device))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    torch.save({"model": model.state_dict(), "args": {"hidden": 256}, "prompts": PROMPTS_B},
               LIBB / "classifier.pt")
    print(f"classifier trained+saved [{time.time()-t0:.0f}s]", flush=True)

    base = MusicGenCLAP(device=device, use_lora=False); mg = base.mg; mg.eval()
    clap, proc, csr = load_clap(device)
    prompt_h = {k: embed_text(mg, PROMPTS_B[k], device) for k in range(K)}

    val_rows = np.where(va)[0]
    rows = val_rows[:args.rows]
    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(cond[rows]).to(device)).argmax(1).cpu().numpy()

    zc, cs = [], []
    rec = []
    for i, r in enumerate(rows):
        pk = int(pred[i])
        w = gen_from_enc(mg, prompt_h[pk].unsqueeze(0), args.tokens, args.temperature, None,
                         seed=args.seed + int(r))
        w = soft_limit(w)
        gp = OUT / f"genB__row{r}__p{pk}.wav"; sf.write(str(gp), w, SR)
        oc = mg.audio_encoder.decode(torch.from_numpy(codes[r]).unsqueeze(0).unsqueeze(0).to(device),
                                     audio_scales=[None]).audio_values
        ref = np.asarray(oc.cpu()).reshape(-1)
        rp = OUT / f"ref__row{r}.wav"; sf.write(str(rp), ref, SR)
        # ref is 32 kHz: passing csr made CLAP treat it as 48 kHz without resampling, i.e. it
        # scored a 1.5x-fast stimulus and pulled cos_to_true down
        c = float(unit(clap_embed_audio(clap, proc, SR, w, device)) @
                  unit(clap_embed_audio(clap, proc, SR, ref, device)))
        zc.append(zcr(w)); cs.append(c)
        rec.append({"row": int(r), "prompt": pk, "prompt_text": PROMPTS_B[pk],
                    "zcr": zcr(w), "peak": float(np.abs(w).max()), "cos_to_true": c,
                    "gen_wav": str(gp), "ref_wav": str(rp)})
        print(f"  row{r} -> p{pk} ({PROMPTS_B[pk][:32]:32}) zcr={zcr(w):.4f} "
              f"peak={np.abs(w).max():.2f} cos={c:.3f}", flush=True)

    zc = np.array(zc); cs = np.array(cs)
    print(f"\n=== corpus-matched delivery: zcr mean={zc.mean():.4f} peak mean="
          f"{np.mean([x['peak'] for x in rec]):.3f} cos_to_true mean={cs.mean():.4f} ===")
    print(f"prompt usage: {np.bincount(pred, minlength=K).tolist()}")
    json.dump({"zcr_mean": float(zc.mean()), "cos_to_true_mean": float(cs.mean()),
               "pred_dist": np.bincount(pred, minlength=K).tolist(), "per_row": rec},
              open(OUT / "report.json", "w"), indent=2)
    print(f"clips in {OUT}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
