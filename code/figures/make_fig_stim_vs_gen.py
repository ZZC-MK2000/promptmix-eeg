"""Fig 12 + web spectrogram tiles: the heard stimulus window vs the clip the frozen MusicGen made
from that same window's EEG.

Two jobs:
  1. assets/fig/spectro/row<N>_ref.png / _gen.png — one tile per clip, identical time axis (0-8 s),
     identical frequency axis and identical colour scale, for the demo page's comparison table.
     Only the STIMULUS tiles are derived from copyrighted audio, and only as images: the stimulus
     waveform is never published.
  2. results/stim_vs_gen/cos_matrix.json — per-row laion-CLAP cosine between the generated clip and
     its own stimulus (own) vs the mean cosine to the other 13 stimuli (others). "own vs others" is
     the comparator; a bare own=0.48 must not be read against 1.0.

Needs the mainv2 workspace for the laion-CLAP helper (external to this repo).

Run: python code/figures/make_fig_stim_vs_gen.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
LISTEN = ROOT / "results" / "library_b" / "listen"
TILES = ROOT / "assets" / "fig" / "spectro"
OUT = ROOT / "results" / "stim_vs_gen"
sys.path.insert(0, str(Path(__file__).parent))
from figstyle import TITLE_FS, save

NFFT, NOVL, VMIN, VMAX, FMAX = 1024, 512, -80.0, -10.0, 8000


def panel(ax, w, sr):
    """Same windowing, same dB floor, same axis limits everywhere, so tiles stay comparable."""
    ax.specgram(np.asarray(w, np.float32), Fs=sr, NFFT=NFFT, noverlap=NOVL, cmap="magma",
                vmin=VMIN, vmax=VMAX)
    ax.set_ylim(0, FMAX); ax.set_xlim(0, len(w) / sr)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def main():
    rep = json.load(open(LISTEN / "report.json", encoding="utf-8"))
    rows = rep["per_row"]
    n = len(rows)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sys.path.insert(0, str(M2 / "src" / "generate"))
    from musicgen_clap_finetune import clap_embed_audio, load_clap
    model, proc, _ = load_clap(device)

    clips = []
    for r in rows:
        w, sr = sf.read(LISTEN / f"ref__row{r['row']}.wav")
        g, gsr = sf.read(LISTEN / Path(r["gen_wav"]).name)
        clips.append((r, np.asarray(w, np.float32), sr, np.asarray(g, np.float32), gsr))

    def emb(wav, rate):
        e = np.asarray(clap_embed_audio(model, proc, rate, wav, device)).reshape(-1).astype(np.float32)
        return e / (np.linalg.norm(e) + 1e-8)

    R = np.stack([emb(c[1], c[2]) for c in clips])
    G = np.stack([emb(c[3], c[4]) for c in clips])
    C = G @ R.T
    own = np.diag(C).copy()
    others = (C.sum(1) - own) / (n - 1)
    margin = own - others

    # floor: laion-CLAP unit vectors cluster, so establish what an unrelated pair scores at all
    def offdiag(M):
        m = M.copy(); np.fill_diagonal(m, np.nan); return float(np.nanmean(m))
    ref_ref, gen_gen = offdiag(R @ R.T), offdiag(G @ G.T)
    print(f"floor: mean cos between two different stimuli {ref_ref:.4f}, "
          f"between two different generations {gen_gen:.4f}")

    TILES.mkdir(parents=True, exist_ok=True)
    for r, wref, sr, wgen, gsr in clips:
        fig, ax = plt.subplots(figsize=(3.7, 1.25))
        panel(ax, wref, sr); fig.savefig(TILES / f"row{r['row']}_ref.png", dpi=120,
                                         bbox_inches="tight", pad_inches=0.01); plt.close(fig)
        fig, ax = plt.subplots(figsize=(3.7, 1.25))
        panel(ax, wgen, gsr); fig.savefig(TILES / f"row{r['row']}_gen.png", dpi=120,
                                          bbox_inches="tight", pad_inches=0.01); plt.close(fig)

    payload = {
        "n_pairs": n,
        "mean_cos_own": float(own.mean()), "mean_cos_others": float(others.mean()),
        "mean_cos_ref_vs_ref": ref_ref, "mean_cos_gen_vs_gen": gen_gen,
        "mean_margin": float(margin.mean()),
        "margin_t": float(margin.mean() / (margin.std(ddof=1) / np.sqrt(n))),
        "rows_own_above_others": int((margin > 0).sum()),
        "rows": [{"row": int(r["row"]), "prompt": int(r["prompt"]), "prompt_text": r["prompt_text"],
                  "gen_file": Path(r["gen_wav"]).name,
                  "cos_own": float(own[i]), "cos_others": float(others[i]),
                  "margin": float(margin[i]), "zcr": float(r["zcr"])}
                 for i, (r, *_ ) in enumerate(clips)],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(OUT / "cos_matrix.json", "w", encoding="utf-8"), indent=2)
    # the demo page reads the measured numbers straight from this file rather than a hand-typed copy
    web = ROOT / "assets" / "pairs.json"
    web.parent.mkdir(parents=True, exist_ok=True)
    json.dump(payload, open(web, "w", encoding="utf-8"), indent=2)
    print(f"own {own.mean():.4f}  others {others.mean():.4f}  margin {margin.mean():+.4f} "
          f"t={payload['margin_t']:.1f}  own>others in {payload['rows_own_above_others']}/{n}")

    order = np.argsort(own)
    pick = [rows[order[k]]["row"] for k in (0, n // 4, n // 2, (3 * n) // 4, n - 1)]
    by_row = {c[0]["row"]: (j, c) for j, c in enumerate(clips)}
    fig, axes = plt.subplots(len(pick), 2, figsize=(8.8, 1.55 * len(pick)))
    for k, row_no in enumerate(pick):
        j, (r, wref, sr, wgen, gsr) = by_row[row_no]
        for col, (wav, rate) in enumerate([(wref, sr), (wgen, gsr)]):
            ax = axes[k][col]
            panel(ax, wav, rate)
            if k == 0:
                ax.set_title(["heard stimulus (audio withheld: copyright)",
                              "generated from this window's EEG (playable)"][col], fontsize=9)
            if col == 0:
                ax.set_ylabel(f"row {row_no}", fontsize=9)
            else:
                ax.text(1.01, 0.5, f"CLAP cos {own[j]:.2f}\nvs others {others[j]:.2f}",
                        transform=ax.transAxes, ha="left", va="center", fontsize=7.5)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)  ·  0-8 s window  ·  0-8 kHz  ·  colour scale "
                      f"{VMIN:.0f} to {VMAX:.0f} dB", fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.97, 0.92])
    fig.suptitle(f"Heard vs generated for the same 8-s EEG window — 5 rows spanning the "
                 f"cosine range\nmean CLAP cos to own stimulus {own.mean():.2f} vs "
                 f"{others.mean():.2f} to the other {n-1} stimuli", fontsize=TITLE_FS)
    save(fig, "fig12_stim_vs_gen.png")


if __name__ == "__main__":
    main()
