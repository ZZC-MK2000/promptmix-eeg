"""CRITICAL validity check for Mode (A): does the bridge(EEG)-derived oracle prompt label
actually agree with the prompt that best matches the TRUE audio the subject heard?

Background: finetune_prompt_classifier trained to 94% val_acc predicting
    y_r = argmax_k cos(means[k], cond[r])      # cond = bridge(EEG) output
But cond.npy has cos only 0.164 to library means, while TRUE audio (library_radius
true_emb.npy) has cos 0.747. If bridge output sits in a different region of laion space,
the argmax prompt from bridge may NOT equal the argmax prompt from true audio -- which
would mean Mode (A) delivers the wrong prompt and the whole route is broken.

This script computes, on the SAME 60 held-out rows used by library_radius:
  1. true_oracle[i]  = argmax_k cos(means[k], true_emb[i])      # what we WANT to predict
  2. cond_oracle[i]  = argmax_k cos(means[k], cond[rows[i]])    # what we TRAINED on
  3. clf_pred[i]     = argmax classifier(cond[rows[i]])         # what the model OUTPUTS
  and the agreement rates between them. Pure numpy/torch CPU, no GPU, no generation.

Run: python src/train/validate_mode_a_labels.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
DATA = ROOT / "data" / "eegcond"
LIB = ROOT / "results" / "gen_eegcond" / "library_radius"
CLF = ROOT / "results" / "prompt_classifier"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))

PROMPTS = [
    "warm melodic pop", "indian film", "ambient electronic", "acoustic folk",
    "electronic dance", "lo-fi hip hop", "orchestral score", "sparse percussion",
    "smooth jazz", "rock riff", "ethereal choir", "chiptune",
]
K = 12


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    # Load library means (12, 512)
    lib_emb = np.load(LIB / "library_emb.npy").astype(np.float32)   # (72, 512)
    lib_owner = np.load(LIB / "library_owner.npy").astype(np.int64)  # (72,)
    means = unit(np.stack([unit(lib_emb[lib_owner == k].mean(0)) for k in range(K)]))

    # Load true audio embeddings (60, 512) and the rows they correspond to
    true_emb = np.load(LIB / "true_emb.npy").astype(np.float32)      # (60, 512)
    radius = json.load(open(LIB / "radius.json"))
    rows = np.array(radius["rows"], dtype=np.int64)                  # (60,) into cond.npy
    assert len(rows) == len(true_emb), f"{len(rows)} rows vs {len(true_emb)} true_emb"

    # Load bridge(EEG) conditions for those same rows
    cond = np.load(DATA / "cond.npy").astype(np.float32)             # (36036, 512)
    cond_rows = cond[rows]                                           # (60, 512)

    # Cosine to library means
    cos_true = unit(true_emb) @ means.T     # (60, K)
    cos_cond = unit(cond_rows) @ means.T    # (60, K)

    true_oracle = cos_true.argmax(1)
    cond_oracle = cos_cond.argmax(1)

    # Agreement: does the bridge-derived label match the true-audio label?
    agree = (true_oracle == cond_oracle)
    n = len(rows)
    print("=" * 70)
    print("MODE (A) LABEL VALIDITY CHECK — 60 held-out rows from library_radius")
    print("=" * 70)
    print(f"\ncos(true_audio, best prompt mean):  mean={cos_true.max(1).mean():.4f}  "
          f"min={cos_true.max(1).min():.4f}  max={cos_true.max(1).max():.4f}")
    print(f"cos(bridge_EEG,  best prompt mean):  mean={cos_cond.max(1).mean():.4f}  "
          f"min={cos_cond.max(1).min():.4f}  max={cos_cond.max(1).max():.4f}")

    print(f"\n--- ARGMAX AGREEMENT (the make-or-break number) ---")
    print(f"true_oracle == cond_oracle:  {agree.sum()}/{n} = {agree.mean():.3f}")
    print(f"  (chance = 1/{K} = {1/K:.3f}; majority-class baseline = "
          f"{np.bincount(cond_oracle, minlength=K).max()/n:.3f})")

    # Margin of the argmax: how separated is the best prompt from the 2nd best?
    def top2_margin(cos):
        s = np.sort(cos, axis=1)
        return s[:, -1] - s[:, -2]
    print(f"\ntrue cos top1-top2 margin: mean={top2_margin(cos_true).mean():.4f}")
    print(f"cond cos top1-top2 margin: mean={top2_margin(cos_cond).mean():.4f}")
    print("  (small margin = label is noise-dominated, argmax is unstable)")

    # Per-row detail
    print(f"\n--- PER-ROW (first 20) ---")
    print(f"{'row':>6} {'true_k':>6} {'cond_k':>6} {'agree':>5} "
          f"{'cosT':>7} {'cosC':>7}  true_prompt / cond_prompt")
    for i in range(min(20, n)):
        tk, ck = int(true_oracle[i]), int(cond_oracle[i])
        mark = "OK" if tk == ck else "XX"
        print(f"{rows[i]:>6} {tk:>6} {ck:>6} {mark:>5} "
              f"{cos_true[i,tk]:>7.3f} {cos_cond[i,ck]:>7.3f}  "
              f"{PROMPTS[tk]} / {PROMPTS[ck]}")

    # Confusion: which true-prompt gets mislabeled as which cond-prompt
    print(f"\n--- CONFUSION (true_oracle rows -> cond_oracle cols) ---")
    conf = np.zeros((K, K), int)
    for t, c in zip(true_oracle, cond_oracle):
        conf[t, c] += 1
    print("      " + "".join(f"{c:>5}" for c in range(K)))
    for t in range(K):
        if conf[t].sum() > 0:
            print(f"true{t:>2} " + "".join(f"{conf[t,c]:>5}" for c in range(K))
                  + f"   (n={conf[t].sum()})")

    # Now load the trained classifier and check ITS predictions vs true_oracle
    print(f"\n--- CLASSIFIER PREDICTION vs TRUE ORACLE ---")
    # weights_only=True: final.pt is our own file (state_dict + primitive metadata).
    ckpt = torch.load(CLF / "final.pt", map_location="cpu", weights_only=True)
    from finetune_prompt_classifier import PromptClassifier
    model = PromptClassifier(in_dim=512, hidden=256, k=K, dropout=0.1)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(cond_rows))
    clf_pred = logits.argmax(1).numpy()
    clf_vs_true = (clf_pred == true_oracle)
    clf_vs_cond = (clf_pred == cond_oracle)
    print(f"clf_pred == cond_oracle (train target): {clf_vs_cond.sum()}/{n} = {clf_vs_cond.mean():.3f}")
    print(f"clf_pred == true_oracle (what matters):  {clf_vs_true.sum()}/{n} = {clf_vs_true.mean():.3f}")
    print(f"  (chance = {1/K:.3f})")

    # VERDICT
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    if agree.mean() < 0.3:
        print("BROKEN: bridge(EEG) argmax prompt disagrees with true-audio argmax prompt")
        print(f"  ({agree.mean():.1%} agreement, near chance {1/K:.1%}). Mode (A) as designed")
        print("  would deliver the WRONG prompt. The 94% classifier val_acc is real but")
        print("  measures agreement with a label that does not correspond to the song heard.")
        print("  -> Do NOT generate audio from this classifier expecting song match.")
    elif agree.mean() < 0.6:
        print(f"WEAK: only {agree.mean():.1%} agreement. Mode (A) partially works but the")
        print("  bridge->library alignment is the bottleneck, not the classifier.")
    else:
        print(f"OK: {agree.mean():.1%} agreement. Mode (A) label is meaningful; proceed to")
        print("  generation + listening eval.")

    # Save
    out = {
        "n": int(n),
        "cos_true_best_mean": float(cos_true.max(1).mean()),
        "cos_cond_best_mean": float(cos_cond.max(1).mean()),
        "argmax_agreement": float(agree.mean()),
        "clf_vs_true_oracle": float(clf_vs_true.mean()),
        "clf_vs_cond_oracle": float(clf_vs_cond.mean()),
        "true_top2_margin": float(top2_margin(cos_true).mean()),
        "cond_top2_margin": float(top2_margin(cos_cond).mean()),
        "true_oracle": true_oracle.tolist(),
        "cond_oracle": cond_oracle.tolist(),
        "clf_pred": clf_pred.tolist(),
        "rows": rows.tolist(),
    }
    json.dump(out, open(CLF / "label_validity.json", "w"), indent=2)
    print(f"\nsaved -> {CLF / 'label_validity.json'}")


if __name__ == "__main__":
    main()
