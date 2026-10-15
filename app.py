"""PromptMix-EEG — interactive Gradio demo (the "demo website").

EEG window -> ridge bridge -> projection layer (mixture weights over 12 corpus-matched prompts)
-> chosen genre -> FROZEN MusicGen native sample -> playable 32 kHz clip, shown next to the
TRUE audio the subject actually heard. Pick any held-out subject window and listen to the A/B.

Run locally:
    pip install gradio
    python app.py            # then open the printed http://127.0.0.1:7860
    python app.py --share    # temporary public Gradio Live-Share URL (publishes to the internet!)

To host as a permanent site, push this repo to HuggingFace Spaces (SDK=gradio) -- see README.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "eegcond"
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "radii"))
sys.path.insert(0, str(ROOT / "code" / "classifier"))

SR = 32000
STATE = {}


def _unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def load():
    if STATE:
        return
    from musicgen_clap_finetune import MusicGenCLAP
    from arms_2x2 import embed_text, gen_from_enc
    from library_b_experiment import PROMPTS_B
    from finetune_prompt_classifier import PromptClassifier
    import soundfile as sf
    device = "cuda" if torch.cuda.is_available() else "cpu"
    base = MusicGenCLAP(device=device, use_lora=False); mg = base.mg; mg.eval()
    ck = torch.load(ROOT / "results/library_b/classifier.pt", map_location="cpu", weights_only=True)
    clf = PromptClassifier(512, 256, len(PROMPTS_B), 0.1); clf.load_state_dict(ck["model"]); clf.eval().to(device)
    STATE.update(dict(mg=mg, clf=clf, device=device, prompts=PROMPTS_B, gen_from_enc=gen_from_enc, sf=sf,
                      cond=np.load(DATA / "cond.npy").astype(np.float32),
                      codes=np.load(DATA / "codes.npy").astype(np.int64),
                      group=np.load(DATA / "group.npy").astype(np.int64),
                      ph={k: embed_text(mg, PROMPTS_B[k], device) for k in range(len(PROMPTS_B))}))
    uniq = np.unique(STATE["group"]); rng = np.random.default_rng(0)
    vg = set(uniq[rng.permutation(len(uniq))[:max(1, int(round(len(uniq) * 0.1)))]].tolist())
    STATE["val_rows"] = np.where(np.array([g in vg for g in STATE["group"]]))[0]


def soft_limit(w, thr=0.95):
    return np.tanh(w / thr) * thr


def run(row_i: int, tau: float):
    load()
    rows = STATE["val_rows"]; row = int(rows[int(row_i) % len(rows)])
    cond = STATE["cond"]; clf = STATE["clf"]; device = STATE["device"]
    with torch.no_grad():
        logits = clf(torch.from_numpy(cond[row:row + 1]).to(device)).float()
    w = torch.softmax(logits / float(tau), 1)[0].cpu().numpy()
    k = int(w.argmax())
    audio = soft_limit(STATE["gen_from_enc"](STATE["mg"], STATE["ph"][k].unsqueeze(0), 400, 0.9, None, seed=7 + row))
    oc = STATE["mg"].audio_encoder.decode(torch.from_numpy(STATE["codes"][row]).unsqueeze(0).unsqueeze(0).to(device),
                                          audio_scales=[None]).audio_values
    ref = np.asarray(oc.cpu()).reshape(-1)
    order = np.argsort(w)[::-1][:5]
    bars = "\n".join(f"{w[t]*100:5.1f}%  {STATE['prompts'][t]}" for t in order)
    caption = f"EEG window #{row}  →  chosen genre: \"{STATE['prompts'][k]}\"\n\ntop-5 mixture weights (tau={tau}):\n{bars}"
    return (SR, audio), (SR, ref), caption


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--share", action="store_true"); a = ap.parse_args()
    load()
    import gradio as gr
    n = len(STATE["val_rows"])
    with gr.Blocks(title="PromptMix-EEG demo") as demo:
        gr.Markdown("# PromptMix-EEG — brain → frozen-MusicGen demo\n"
                    "Pick a held-out EEG window; the projection layer chooses a genre prompt and the "
                    "**frozen** MusicGen samples it. Compare the generated clip to the audio the subject actually heard.")
        with gr.Row():
            row = gr.Slider(0, n - 1, value=0, step=1, label=f"held-out EEG window (0–{n-1})")
            tau = gr.Slider(0.25, 4.0, value=0.7, step=0.05, label="mixture temperature τ (higher = more blended)")
        gen = gr.Audio(label="Generated (frozen MusicGen, EEG-chosen prompt)", type="numpy")
        ref = gr.Audio(label="True heard audio (reference)", type="numpy")
        cap = gr.Textbox(label="Pipeline trace", lines=8)
        gr.Button("Generate").click(run, [row, tau], [gen, ref, cap])
    demo.launch(share=a.share)


if __name__ == "__main__":
    main()
