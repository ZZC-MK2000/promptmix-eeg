"""Decode stimulus windows for the free-text captioner.

Two sampling modes:
  spread    three segments spread across the song (intro/middle/outro); kept for
            comparison, but can land on atypical intros/drum solos.
  centroid  three kept+aligned windows whose laion-CLAP embedding is closest to the
            whole-song centroid, from distinct segments/subjects (default).

Uses the compact-to-full row map: codes.npy has 36036 kept rows, not 40336.

Run: python code/information/make_caption_audio.py --mode centroid
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from calib_head_pilot import GATE  # noqa: E402

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = ROOT.parent / "mainv2"
N_PICK = 3


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["centroid", "spread"], default="centroid")
    ap.add_argument("--out-dir", type=Path,
                    default=ROOT / "results" / "descriptions" / "caption_audio_centroid")
    args = ap.parse_args()
    OUT = args.out_dir
    OUT.mkdir(parents=True, exist_ok=True)

    z = np.load(GATE, allow_pickle=True)
    cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)
    codes = np.load(ROOT / "data" / "eegcond" / "codes.npy").astype(np.int64)
    aldm = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    w = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu", weights_only=True)
    ca = aldm @ w["weight"].numpy().astype(np.float32).T + w["bias"].numpy().astype(np.float32)
    ca = unit(ca)
    key = {ca[i].tobytes(): i for i in range(len(ca))}
    idx = np.array([key[cond[j].tobytes()] for j in range(len(cond))])
    T = unit(np.load(ROOT / "results" / "prompt_classifier" / "true_emb_all.npy").astype(np.float32))

    al = z["aligned8"][idx]; song = z["song"][idx]; seg = z["seg2"][idx]; subj = z["subject"][idx]
    sys.path.insert(0, str(M2 / "src" / "generate"))
    sys.path.insert(0, str(ROOT / "code" / "shared"))
    sys.path.insert(0, str(ROOT / "code" / "radii"))
    from musicgen_clap_finetune import MusicGenCLAP

    base = MusicGenCLAP(device="cuda", use_lora=False); mg = base.mg; mg.eval()
    manifest = []
    for s in range(14):
        rows = np.where(al & (song == s))[0]
        if args.mode == "centroid":
            C = unit(T[rows].mean(0))
            order = np.argsort(-(T[rows] @ C))
            ranked = rows[order]
            chosen, used_seg, used_subj = [], set(), set()
            for r in ranked:
                if int(seg[r]) in used_seg or str(subj[r]) in used_subj:
                    continue
                chosen.append(int(r)); used_seg.add(int(seg[r])); used_subj.add(str(subj[r]))
                if len(chosen) == N_PICK:
                    break
            picks = chosen
        else:
            segs = np.unique(seg[rows])
            picks = []
            used_subj = set()
            for j in np.linspace(0, len(segs) - 1, N_PICK).round().astype(int):
                sg = int(segs[j])
                cand = [int(r) for r in rows[seg[rows] == sg] if subj[r] not in used_subj] or [int(rows[seg[rows] == sg][0])]
                picks.append(cand[0]); used_subj.add(str(subj[cand[0]]))
        for r in picks:
            ci = r
            with torch.no_grad():
                wav = mg.audio_encoder.decode(
                    torch.from_numpy(codes[ci]).unsqueeze(0).unsqueeze(0).to("cuda"),
                    audio_scales=[None]).audio_values
            wav = np.asarray(wav.cpu()).reshape(-1).astype(np.float32)
            path = OUT / f"song{s}_seg{int(seg[r])}.wav"
            sf.write(path, wav, 32000)
            manifest.append({"song": s, "seg": int(seg[r]), "row": int(r), "code_idx": ci,
                             "subject": str(subj[r]), "path": str(path),
                             "duration": float(len(wav) / 32000),
                             "cos_to_song_centroid": float(T[r] @ unit(T[rows].mean(0)))})
            print(f"song {s:2d} seg {int(seg[r]):3d} row {r:5d} subj {subj[r]} "
                  f"cos {manifest[-1]['cos_to_song_centroid']:.3f} -> {path.name}", flush=True)
    json.dump(manifest, open(OUT / "manifest.json", "w", encoding="utf-8"), indent=2)
    print("wrote", OUT / "manifest.json")


if __name__ == "__main__":
    main()
