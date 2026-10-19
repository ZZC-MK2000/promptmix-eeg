"""Pre-gate for enlarging the frozen prompt library, CPU-only.

The t-SNE (fig5) says the current 12-prompt library sits in ONE region of the real-window cloud.
This asks whether the 12 human-annotated MUSIN-G genres (Hindustani classical, Indian folk, goth
rock, deep house, soft jazz ...) fill that gap -- BEFORE spending GPU time generating audio for
them, which is what a real coverage measurement needs.

How: laion-CLAP has a text tower, so each prompt can be embedded without generating anything, and
compared against the real windows' AUDIO embeddings cross-modally. Absolute values are therefore
NOT comparable to fig1's 0.863 (that used generated-audio library points); only the comparison
between the arms below is meaningful.

Three arms, same measurement each:
  A  the current 12 prompts                                  (baseline)
  B  12 + 10 genuinely new genres from MUSIN-G               (the proposal)
  C  12 + 10 paraphrases of the existing 12                  (the control)
C exists to answer the only question that matters: if adding 10 near-duplicate prompts helps just
as much, the gain is sampling density, not genre coverage, and the library does not need new styles.

Reported per arm: trivial global-mean, best single prompt, uniform mixture, and the simplex
convex-combination fit (the coverage-radius statistic, from library_radius.fit_simplex).

Run: python code/radii/prompt_expansion_probe.py       (CPU, no GPU, no generation)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
OUT = ROOT / "results" / "prompt_expansion"
sys.path.insert(0, str(ROOT / "code" / "radii"))
sys.path.insert(0, str(M2 / "src" / "generate"))

N_WINDOWS = 2000          # deterministic every-k-th of the 36,036 real windows, not a selection

OLD12 = json.load(open(ROOT / "results" / "library_b" / "summary.json", encoding="utf-8"))["prompts"]

# the 12 MUSIN-G genres (stimuli/Song_Description), minus Electronic Dance and Ambient which the
# current library already covers; each phrased with the instrument/texture words MusicGen responds to
NEW10 = [
    "hindustani classical raga with sitar and tabla, slow alapa improvisation",
    "indian folk song with dholak and harmonium and group vocals",
    "indian semi-classical thumri with expressive female vocals and light tabla",
    "goth rock with dark detuned guitar and heavy reverb on the drums",
    "progressive instrumental rock in odd time signatures with dual guitars",
    "deep house groove with muffled kick, sub bass and warm synth stabs",
    "new age instrumental with soft pads and gentle piano",
    "soft jazz trio with brushed drums upright bass and mellow piano",
    "indie rock with jangly clean guitars and loose live drums",
    "electronic track with glitching synths, textured pads and no drums",
]

# control: same COUNT, same style of phrasing, but no new territory
PARA10 = [
    "bollywood dance track with loud dhol and shiny synths",
    "indian film song with strings and bamboo flute melody",
    "pop song with breathy female vocals and polished mix",
    "r&b groove with round bass and rimshot snare",
    "edm with pounding four-on-the-floor kick and big drop",
    "acoustic song with fingerstyle guitar and soft drums",
    "funk with slap bass and bright brass hits",
    "reggaeton with dembow rhythm and tropical synths",
    "ambient pad with slow evolving texture",
    "rock with distorted guitar riff and live drums",
]


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def embed_texts(model, proc, texts, device):
    with torch.no_grad():
        out = []
        for t in texts:
            inp = proc(text=[t], return_tensors="pt", padding=True)
            inp = {k: v.to(device) for k, v in inp.items() if hasattr(v, "to")}
            e = model.get_text_features(**inp)
            e = e.pooler_output if hasattr(e, "pooler_output") else e
            out.append(np.asarray(e).reshape(-1).astype(np.float32))
    return unit(np.stack(out))


def main():
    from musicgen_clap_finetune import load_clap
    from library_radius import fit_simplex

    device = "cpu"
    clap, proc, _ = load_clap(device)

    W = np.load(ROOT / "results" / "prompt_classifier" / "true_emb_all.npy").astype(np.float32)
    total = len(W)
    W = unit(W)
    stride = max(1, total // N_WINDOWS)
    W = W[::stride][:N_WINDOWS]
    print(f"{len(W)} real held-out windows (every {stride}th of {total})", flush=True)

    arms = {"A_current12": OLD12,
            "B_plus_new_genres": OLD12 + NEW10,
            "C_plus_paraphrases": OLD12 + PARA10}
    E = {k: embed_texts(clap, proc, v, device) for k, v in arms.items()}

    res = {"n_windows": int(len(W)), "arms": {}}
    for k, P in E.items():
        cos = W @ P.T
        best_single = float(cos.max(1).mean())
        uniform = float((W @ unit(P.sum(0))).mean())          # equal-weight mixture of the library
        trivial = float((W @ unit(W.mean(0))).mean())         # fig1's baseline: always answer the
        # corpus average, which learns nothing about the target window
        mix = []
        for i, t in enumerate(W):
            w = fit_simplex(P, t)
            mix.append(float(t @ (P.T @ w) / (np.linalg.norm(t) * np.linalg.norm(P.T @ w) + 1e-9)))
        mix = np.array(mix)
        res["arms"][k] = {"n_prompts": int(P.shape[0]),
                          "trivial_global_mean": trivial,
                          "best_single_prompt": best_single,
                          "uniform_mixture": uniform,
                          "convex_mixture": float(mix.mean()),
                          "convex_minus_trivial": float(mix.mean() - trivial),
                          "convex_minus_best_single": float(mix.mean() - best_single)}
        r = res["arms"][k]
        print(f"{k:22} K={r['n_prompts']:3}  trivial {r['trivial_global_mean']:.4f}  "
              f"best-single {r['best_single_prompt']:.4f}  uniform {r['uniform_mixture']:.4f}  "
              f"convex {r['convex_mixture']:.4f}  (convex-trivial {r['convex_minus_trivial']:+.4f})")

    a, b, c = (res["arms"][x] for x in ("A_current12", "B_plus_new_genres", "C_plus_paraphrases"))
    print(f"\ngain over current:  new genres {b['convex_mixture']-a['convex_mixture']:+.4f} | "
          f"paraphrase control {c['convex_mixture']-a['convex_mixture']:+.4f}")
    res["verdict"] = {
        "new_genre_gain": b["convex_mixture"] - a["convex_mixture"],
        "paraphrase_gain": c["convex_mixture"] - a["convex_mixture"],
        "new_beats_control": (b["convex_mixture"] - a["convex_mixture"]) >
                             1.5 * (c["convex_mixture"] - a["convex_mixture"]),
    }
    print("does the new-genre gain beat the density control by >1.5x? ", res["verdict"]["new_beats_control"])

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "probe.json", "w", encoding="utf-8"), indent=2)
    print(f"\nsaved -> {OUT/'probe.json'}")


if __name__ == "__main__":
    main()
