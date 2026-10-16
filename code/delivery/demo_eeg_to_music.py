"""#3: end-to-end demo -- one EEG window in, music out, printing every stage of the pipeline.

  EEG window -> frozen LaBraM -> ridge bridge -> laion-CLAP c (512)
            -> projection layer -> softmax weights over 12 corpus-matched prompts
            -> argmax prompt -> T5-encode -> FROZEN MusicGen sample -> 32 kHz wav

No training here; it runs the full inference path on held-out windows and writes listenable clips
plus a per-window trace (chosen genre, weight vector, cos to the heard audio). This is the
"does the whole thing actually run and produce music" artifact.

Run: python code/delivery/demo_eeg_to_music.py --n 6      (from promptmix-eeg/, GPU)
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
DATA = Path("D:/Projects/CLaBraM/promptmix-eeg/data/eegcond")
OUT = ROOT / "results" / "demo"
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "radii"))
sys.path.insert(0, str(ROOT / "code" / "classifier"))
sys.path.insert(0, str(M2 / "src" / "generate"))
SR = 32000


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def soft_limit(w, thr=0.95):
    return np.tanh(w / thr) * thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--tau", type=float, default=0.7)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from musicgen_clap_finetune import MusicGenCLAP, load_clap, clap_embed_audio
    from arms_2x2 import embed_text, gen_from_enc
    from library_b_experiment import PROMPTS_B
    from finetune_prompt_classifier import PromptClassifier

    base = MusicGenCLAP(device=device, use_lora=False); mg = base.mg; mg.eval()
    clap, proc, csr = load_clap(device)
    ck = torch.load(ROOT / "results/library_b/classifier.pt", map_location="cpu", weights_only=True)
    clf = PromptClassifier(512, 256, len(PROMPTS_B), 0.1); clf.load_state_dict(ck["model"]); clf.eval().to(device)

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int64)
    group = np.load(DATA / "group.npy").astype(np.int64)
    true_emb = unit(np.load(ROOT / "results/prompt_classifier/true_emb_all.npy").astype(np.float32))
    uniq = np.unique(group); rng = np.random.default_rng(0)
    vg = set(uniq[rng.permutation(len(uniq))[:max(1, int(round(len(uniq) * 0.1)))]].tolist())
    from stratified_sample import diverse_rows, load_val_embeddings
    rows = diverse_rows(load_val_embeddings(),
                        np.where(np.array([g in vg for g in group]))[0], group, args.n, seed=0)

    print(f"=== PromptMix-EEG end-to-end demo: {len(rows)} held-out EEG windows ===")
    print(f"pipeline: EEG -> bridge -> projection -> argmax prompt -> FROZEN MusicGen -> 32kHz\n")
    ph = {k: embed_text(mg, PROMPTS_B[k], device) for k in range(len(PROMPTS_B))}
    rec = []
    for r in rows:
        c = unit(cond[r])
        with torch.no_grad():
            logits = clf(torch.from_numpy(cond[r:r+1]).to(device)).float()
        w = torch.softmax(logits / args.tau, dim=1)[0].cpu().numpy()
        k = int(w.argmax())
        audio = soft_limit(gen_from_enc(mg, ph[k].unsqueeze(0), 400, 0.9, None, seed=7 + int(r)))
        p = OUT / f"eeg_row{r}__{PROMPTS_B[k].split()[0]}.wav"; sf_write(p, audio)
        cos = float(unit(clap_embed_audio(clap, proc, SR, audio, device)) @ true_emb[r])
        top = np.argsort(w)[::-1][:3]
        trace = " | ".join(f"{PROMPTS_B[t].split()[0]}:{w[t]:.2f}" for t in top)
        print(f"window {r:>5}: top3 weights [{trace}]  -> chose '{PROMPTS_B[k][:40]}'  cos->heard={cos:.3f}")
        rec.append({"row": int(r), "prompt": k, "prompt_text": PROMPTS_B[k],
                    "weights_top3": [[int(t), float(w[t])] for t in top], "cos_to_heard": cos, "wav": str(p)})
    json.dump(rec, open(OUT / "demo_trace.json", "w"), indent=2)
    print(f"\n{len(rec)} clips + trace in {OUT}  [{time.time()-t0:.0f}s]")


def sf_write(path, w):
    import soundfile as sf
    sf.write(str(path), w, SR)


if __name__ == "__main__":
    main()
