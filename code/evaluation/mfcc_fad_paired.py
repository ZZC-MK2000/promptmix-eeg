"""Fig 10 done properly: one checkpoint per arm, every arm generated on the SAME stratified rows.

Two defects in the shipped fig10 this replaces:
  1. the "fine-tuned" arm was `glob("gen_*_row*.wav")[:60]` over a directory holding FOUR different
     checkpoints (13x s12000, 9x s8000, 13x ep16, 8x mixed2-step12000) -- an unlabelled blend, not
     a model;
  2. the frozen arm used a `[:60]` prefix of a subject-ordered array (one subject, 6 of 14 songs),
     so the two arms were also computed on different row sets.

Here each arm is a single named checkpoint, all arms share one row list, and the comparison is
reported both distributionally (MFCC-Fréchet distance to the real windows) and per row (Mahalanobis
distance of each clip's MFCC mean to the real Gaussian, paired t-test).

Recipe is copied from code/evaluation/mixed_eval.py unchanged: fine-tuned arms use
temperature 1.0 / top_k 250 / 400 tokens / k=8; the frozen arm uses the corpus-matched prompt chosen
by results/library_b/classifier.pt at temperature 0.9, seed 99+row, exactly as mfcc_fad.py does.

The fine-tuned checkpoints live in ../mainv3/results (deliberately not copied into this repo).

Run: python code/evaluation/mfcc_fad_paired.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
M3 = Path("D:/Projects/CLaBraM/mainv3")
DATA = ROOT / "data" / "eegcond"
OUT = ROOT / "results" / "mfcc_fad_paired"
for _s in ("shared", "radii", "classifier", "delivery", "evaluation"):
    sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

SR = 32000
FT_RECIPE = dict(temperature=1.0, top_k=250)     # from mixed_eval.RECIPE
K_TOKENS = 8
N_ROWS = 60
# name -> checkpoint path; one model per arm, no blending
FINETUNED = {
    "finetuned_s12000": M3 / "results/gen_eegcond_mixed/ckpt_s12000.pt",
    "finetuned_ep16": M3 / "results/gen_eegcond_k8/ckpt_ep16.pt",
}


def main():
    import soundfile as sf
    from mfcc_fad import frechet, stack_feats
    from stratified_sample import heldout_mask, diverse_rows, load_val_embeddings, coverage_report

    OUT.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.time()

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int64)
    group = np.load(DATA / "group.npy").astype(np.int64)
    va = heldout_mask(group)
    rows = diverse_rows(load_val_embeddings(), np.where(va)[0], group, N_ROWS, seed=0)
    cov = coverage_report(rows, group, load_val_embeddings())
    print(f"rows={len(rows)} coverage={cov}", flush=True)
    assert len(rows) == N_ROWS and cov["distinct_groups"] == N_ROWS, "rows must be one per group"

    from musicgen_clap_finetune import MusicGenCLAP
    from library_b_experiment import PROMPTS_B
    from finetune_prompt_classifier import PromptClassifier
    from arms_2x2 import embed_text, gen_from_enc

    base = MusicGenCLAP(device=device, use_lora=False); mg = base.mg; mg.eval()
    real_w = []
    with torch.no_grad():
        for r in rows:
            oc = mg.audio_encoder.decode(torch.from_numpy(codes[r]).unsqueeze(0).unsqueeze(0).to(device),
                                         audio_scales=[None]).audio_values
            real_w.append(np.asarray(oc.cpu()).reshape(-1))
    print(f"real clips={len(real_w)} [{time.time()-t0:.0f}s]", flush=True)

    ck = torch.load(ROOT / "results/library_b/classifier.pt", map_location="cpu", weights_only=True)
    clf = PromptClassifier(512, 256, len(PROMPTS_B), 0.1); clf.load_state_dict(ck["model"]); clf.eval()
    with torch.no_grad():
        pred = clf(torch.from_numpy(cond[rows])).argmax(1).cpu().numpy()
    ph = {k: embed_text(mg, PROMPTS_B[k], device) for k in range(len(PROMPTS_B))}
    froz_w = [gen_from_enc(mg, ph[int(pred[i])].unsqueeze(0), 400, 0.9, None, seed=99 + int(r))
              for i, r in enumerate(rows)]
    print(f"frozen clips={len(froz_w)} [{time.time()-t0:.0f}s]", flush=True)
    del base, mg
    if device == "cuda":
        torch.cuda.empty_cache()

    from pilot_multitoken import CondWidthModel
    arms = {"frozen_native": froz_w}
    for name, path in FINETUNED.items():
        if not path.exists():
            print(f"  {name}: checkpoint missing at {path} -- arm skipped"); continue
        cwm = CondWidthModel(device, k_tokens=K_TOKENS)
        st = torch.load(str(path), map_location="cpu", weights_only=True)
        cwm.m.clap_proj.load_state_dict(st["clap_proj"])
        cwm.m.mg.decoder.load_state_dict(st["decoder_lora"], strict=False)
        cwm.m.eval()
        ws = []
        for r in rows:
            w = np.asarray(cwm.generate(torch.from_numpy(cond[r]), max_new_tokens=400, **FT_RECIPE)).reshape(-1)
            ws.append(w)
            if len(ws) <= 6:
                sf.write(str(OUT / f"{name}_row{int(r):05d}.wav"), w, SR)
        arms[name] = ws
        print(f"{name} clips={len(ws)} [{time.time()-t0:.0f}s]", flush=True)
        del cwm
        if device == "cuda":
            torch.cuda.empty_cache()

    mr, cr, nr = stack_feats(real_w)
    # per-row paired score: Mahalanobis distance of each clip's MFCC mean to the real Gaussian
    from mfcc_fad import mfccs
    inv = np.linalg.pinv(cr)

    def maha(ws):
        means = np.stack([mfccs(w)[0] for w in ws])
        d = means - mr
        return np.einsum("ij,jk,ik->i", d, inv, d)
    scores = {k: maha(v) for k, v in arms.items()}
    scores["real"] = maha(real_w)

    res = {"coverage": cov, "rows": [int(x) for x in rows], "n_rows": len(rows),
           "ft_recipe": FT_RECIPE, "frozen_recipe": {"temperature": 0.9, "tokens": 400, "seed": "99+row"},
           "real_n": nr, "arms": {}}
    print(f"\n{'arm':20} {'n':>4} {'MFCC-FD->real':>14} {'median Mahalanobis':>19}")
    for k in ["real", "frozen_native"] + list(FINETUNED):
        if k == "real":
            ws = real_w
        elif k in arms:
            ws = arms[k]
        else:
            continue
        m, c, n = stack_feats(ws)
        fd = 0.0 if k == "real" else frechet(m, c, mr, cr)
        res["arms"][k] = {"n": n, "fd_to_real": fd, "median_mahalanobis": float(np.median(scores[k]))}
        print(f"{k:20} {n:4} {fd:14.1f} {np.median(scores[k]):19.1f}")

    # paired tests against the frozen arm, row by row
    def paired(a, b):
        d = scores[a] - scores[b]
        rng = np.random.default_rng(0)
        bs = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(4000)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"a": a, "b": b, "n": int(len(d)), "mean_delta": float(d.mean()),
                "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))), "ci95": [float(lo), float(hi)],
                "wins": int((d < 0).sum())}
    res["paired"] = [paired(k, "frozen_native") for k in FINETUNED if k in arms]
    print("\npaired per-row Mahalanobis (negative delta = first arm CLOSER to real than frozen):")
    for p in res["paired"]:
        print(f"  {p['a']} vs {p['b']}: delta {p['mean_delta']:+.1f} t={p['t']:.2f} "
              f"95% CI [{p['ci95'][0]:+.1f},{p['ci95'][1]:+.1f}]  {p['wins']}/{p['n']} closer")
    if "finetuned_s12000" in res["arms"] and "frozen_native" in res["arms"]:
        ratio = res["arms"]["finetuned_s12000"]["fd_to_real"] / res["arms"]["frozen_native"]["fd_to_real"]
        res["frozen_vs_finetuned_s12000_ratio"] = float(ratio)
        print(f"\nfrozen vs fine-tuned(s12000) FD ratio = {ratio:.2f}x")
    json.dump(res, open(OUT / "paired.json", "w", encoding="utf-8"), indent=2)
    print(f"\nsaved -> {OUT/'paired.json'}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
