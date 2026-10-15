"""CPU-only listen test for mainv3 EEG-conditioned generator.

Generates from HELD-OUT val-group conditions (bridge(EEG)) so nothing here was
trained on, and decodes the target codes of those same rows as the "real stimulus"
reference, so the user can A/B generated vs true directly.

Runs on CPU: does not touch the GPU training process.

Run:  python src/train/eval_gen_cpu.py [--ckpt results/gen_eegcond/best.pt] [--n 2]
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
OUT = ROOT / "results" / "gen_eegcond" / "listen_cpu"
sys.path.insert(0, str(M2 / "src" / "generate"))


def val_indices(group, seed=0, val_frac=0.1):
    """Replicate finetune_gen_eegcond.split_by_group exactly to get held-out rows."""
    uniq = np.unique(group)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_g = set(uniq[order[:n_val]].tolist())
    return np.where(np.array([g in val_g for g in group]))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "results" / "gen_eegcond" / "best.pt"))
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--arm", default="trained", choices=["trained", "randomproj"],
                    help="randomproj = pretrained decoder + UNTRAINED random clap_proj, "
                         "same conditions: the control that shows whether the condition port "
                         "actually carries song identity")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    dev = "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    import soundfile as sf

    from musicgen_clap_finetune import load_trained_model, MusicGenCLAP, SR

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int64)
    group = np.load(DATA / "group.npy").astype(np.int64)
    va = val_indices(group)
    # spread picks across the held-out set rather than taking adjacent rows
    pick = va[np.linspace(0, len(va) - 1, args.n, dtype=int)]
    print(f"held-out val rows {len(va)}; picking {list(pick)}")

    t0 = time.time()
    if args.arm == "trained":
        model, ck = load_trained_model(args.ckpt, device=dev)
        print(f"loaded {Path(args.ckpt).name} epoch={ck.get('epoch')} "
              f"val_loss={ck.get('val_loss'):.4f}  [{time.time()-t0:.0f}s]")
    else:
        model = MusicGenCLAP(device=dev, use_lora=False)
        model.eval()
        print(f"CONTROL arm: pretrained decoder + freshly-initialised random clap_proj "
              f"(nothing learned)  [{time.time()-t0:.0f}s]")

    mg = model.mg
    tag = args.arm
    for j, i in enumerate(pick):
        c = torch.from_numpy(cond[i])
        t1 = time.time()
        wav = model.generate(c, max_new_tokens=args.tokens, temperature=args.temp)
        wav = np.asarray(wav).reshape(-1)
        gt = f"{OUT}/gen_{tag}_row{i:05d}.wav"
        sf.write(gt, wav, SR)
        rms_g = float(np.sqrt(np.mean(wav ** 2)))
        print(f"  row{i}: GEN len={len(wav)/SR:.2f}s rms={rms_g:.4f} "
              f"peak={np.abs(wav).max():.3f}  [{time.time()-t1:.0f}s] -> {gt}")

        ref = mg.audio_encoder.decode(
            torch.from_numpy(codes[i]).unsqueeze(0).unsqueeze(0),
            audio_scales=[None]).audio_values
        ref = np.asarray(ref).reshape(-1)
        rp = f"{OUT}/TRUE_row{i:05d}.wav"
        sf.write(rp, ref, SR)
        rms_r = float(np.sqrt(np.mean(ref ** 2)))
        print(f"  row{i}: TRUE len={len(ref)/SR:.2f}s rms={rms_r:.4f} "
              f"peak={np.abs(ref).max():.3f} -> {rp}")

    print(f"\nAll in {OUT}  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
