"""Temperature sweep on the projection layer's mixture weights (held-out subjects).

Key point: mode (A) takes argmax(softmax(logits/tau)) = argmax(logits) -- INVARIANT to tau. So tau
only matters for the mixture modes: B-cat (sample a prompt from the categorical weights) and B-mix
(weighted waveform blend). This sweep characterizes how tau sharpens/softens the weight vector:
entropy, top-1 weight, and effective number of prompts (participation ratio). Higher tau -> flatter.

Run: python code/classifier/tau_sweep.py   (from promptmix-eeg/, CPU)
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
DATA = ROOT / "data" / "eegcond"
sys.path.insert(0, str(ROOT / "code" / "classifier"))


def main():
    from finetune_prompt_classifier import PromptClassifier
    from library_b_experiment import PROMPTS_B
    K = len(PROMPTS_B)
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    group = np.load(DATA / "group.npy").astype(np.int64)
    oracle = np.load(ROOT / "results/library_b/oracle_all.npy").astype(np.int64)
    uniq = np.unique(group); rng = np.random.default_rng(0)
    vg = set(uniq[rng.permutation(len(uniq))[:max(1, int(round(len(uniq) * 0.1)))]].tolist())
    va = np.array([g in vg for g in group])
    ck = torch.load(ROOT / "results/library_b/classifier.pt", map_location="cpu", weights_only=True)
    m = PromptClassifier(512, 256, K, 0.1); m.load_state_dict(ck["model"]); m.eval()
    with torch.no_grad():
        logits = m(torch.from_numpy(cond[va])).float().numpy()
    y = oracle[va]

    print(f"held-out rows={va.sum()}  K={K}")
    print(f"{'tau':>5} {'top1_w':>7} {'entropy(nats)':>14} {'eff#prompts':>12} {'argmax-acc':>11} {'>5% count':>10}")
    for tau in [0.25, 0.5, 1.0, 2.0, 4.0]:
        z = logits / tau
        w = np.exp(z - z.max(1, keepdims=True)); w /= w.sum(1, keepdims=True)
        top1 = w.max(1).mean()
        ent = float(-np.sum(w * np.log(w + 1e-12), axis=1).mean())
        eff = float(np.mean(1.0 / np.sum(w ** 2, axis=1)))          # participation ratio
        acc = float((w.argmax(1) == y).mean())
        n5 = float((w > 0.05).sum(1).mean())
        print(f"{tau:>5} {top1:>7.3f} {ent:>14.3f} {eff:>12.2f} {acc:>11.3f} {n5:>10.2f}")
    print("\nnote: argmax-acc is tau-invariant by construction (mode A unaffected); tau only")
    print("reshapes the weights that B-cat sampling / B-mix blending consume.")
    import json
    json.dump({"taus": [0.25, 0.5, 1.0, 2.0, 4.0], "n_heldout": int(va.sum())},
              open(ROOT / "results/tau_sweep.json", "w"), indent=2)


if __name__ == "__main__":
    main()
