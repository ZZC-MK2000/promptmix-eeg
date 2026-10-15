"""Honest evaluation of the EEG -> true-audio-genre-prompt classifier, accounting for the
label collapse onto chiptune (54.6% of rows). Overall accuracy is inflated by that collapse, so
we report balanced (macro-recall) accuracy and Cohen's kappa against several trivial baselines.

Run: python src/train/eval_true_oracle.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
DATA = ROOT / "data" / "eegcond"
OUT = ROOT / "results" / "prompt_classifier_true"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
K = 12


def main():
    from finetune_prompt_classifier import PromptClassifier

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    group = np.load(DATA / "group.npy").astype(np.int64)
    labels = np.load(ROOT / "results" / "prompt_classifier" / "true_oracle_all.npy").astype(np.int64)
    assert len(labels) == len(cond)
    labels = labels[labels >= 0] if (labels < 0).any() else labels
    cond_m, group_m = cond[:len(labels)], group[:len(labels)]

    # same group-level held-out split as training (seed 0, 10% val groups)
    uniq = np.unique(group_m)
    rng = np.random.default_rng(0)
    val_g = set(uniq[rng.permutation(len(uniq))[:max(1, int(round(len(uniq) * 0.1)))]].tolist())
    va = np.array([g in val_g for g in group_m])
    cond_v, y_v = cond_m[va], labels[va]
    print(f"held-out val rows={va.sum()}  groups={len(set(group_m[va]))}  K={K}", flush=True)

    ck = torch.load(OUT / "final.pt", map_location="cpu", weights_only=True)
    model = PromptClassifier(in_dim=512, hidden=ck["args"].get("hidden", 256), k=K, dropout=0.1)
    model.load_state_dict(ck["model"]); model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(cond_v)).argmax(1).numpy()

    def balanced_acc(y, p):
        rec = [ (p[y == c] == c).mean() for c in range(K) if (y == c).sum() > 0 ]
        return float(np.mean(rec))
    def kappa(y, p):
        n = len(y); po = (y == p).mean()
        pe = sum(((y == c).mean()) * ((p == c).mean()) for c in range(K))
        return float((po - pe) / (1 - pe + 1e-12))

    # baselines on the val set
    prior = np.bincount(y_v, minlength=K)
    maj = int(prior.argmax())
    acc_majority = float((y_v == maj).mean())
    rng2 = np.random.default_rng(1)
    sample_prior = rng2.choice(K, size=len(y_v), p=prior / prior.sum())
    acc_random = float((y_v == sample_prior).mean())

    overall = float((pred == y_v).mean())
    ba = balanced_acc(y_v, pred); kb = kappa(y_v, pred)
    print(f"\npredicted dist: {np.bincount(pred, minlength=K).tolist()}")
    print(f"true     dist: {prior.tolist()}", flush=True)
    print(f"\n=== EEG -> song-matched genre prompt (held-out subjects) ===")
    print(f"  overall acc      = {overall:.4f}")
    print(f"  balanced acc     = {ba:.4f}   (chance 1/K = {1/K:.4f})")
    print(f"  Cohen's kappa    = {kb:.4f}   (0 = no better than chance)")
    print(f"  --- trivial baselines ---")
    print(f"  majority (always '{maj}') = {acc_majority:.4f}")
    print(f"  sample-from-prior        = {acc_random:.4f}")
    print(f"  uniform random           = {1/K:.4f}")

    # per-class recall to see if non-chiptune genres are ever recovered
    print(f"\n  per-class recall:")
    names = ["pop","indian","ambient","folk","dance","lofi","orch","perc","jazz","rock","choir","chip"]
    for c in range(K):
        if prior[c] > 0:
            rec = (pred[y_v == c] == c).mean()
            print(f"    {c:>2} {names[c]:>8}  n={prior[c]:>4}  recall={rec:.3f}")
        else:
            print(f"    {c:>2} {names[c]:>8}  n=   0  recall=  n/a (never a true label)")

    # VERDICT
    print("\n" + "=" * 60)
    if kb >= 0.2 and ba >= 0.2:
        print("GENRE LINK REAL: EEG predicts the song-matched genre prompt well above chance")
        print(f"  (kappa {kb:.3f}, balanced acc {ba:.3f}); note the chiptune label collapse caps overall acc.")
    elif overall > acc_majority + 0.03:
        print("GENRE LINK MARGINAL: beats majority overall but balance/kappa weak — mostly the")
        print("  chiptune mode; the fixed library partitions the corpus poorly.")
    else:
        print("GENRE LINK DEAD: overall acc ≈ majority baseline; EEG does not pick a song-relevant")
        print("  genre. Generation pleasantness is unaffected (native sampling).")
    print("  Either way, the GENERATION requirement (pleasant music) holds by construction:")
    print("  delivered clips are frozen native samples (see deliver_mode_a).")

    import json
    json.dump({"overall": overall, "balanced": ba, "kappa": kb,
               "acc_majority": acc_majority, "acc_random_prior": acc_random, "chance": 1/K,
               "n_val": int(va.sum()), "pred_dist": np.bincount(pred, minlength=K).tolist(),
               "true_dist": prior.tolist()}, open(OUT / "eval_true_oracle.json", "w"), indent=2)
    print(f"\nsaved -> {OUT/'eval_true_oracle.json'}")


if __name__ == "__main__":
    main()
