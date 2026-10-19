"""Verifier-guided inference over the frozen generator.

For each held-out EEG window:
  1. the projection head gives a posterior over the 12 prompts; take its top-M prompts;
  2. generate one native clip per prompt with the FROZEN MusicGen (fixed seeds);
  3. embed each clip with laion-CLAP and score it against the EEG query;
  4. select the best-scoring clip (verifier), or use the head's top-1 (no verifier).

Arms:
  head_top1        the projection head's argmax prompt (no verification)
  verifier         argmax cos(clip, EEG query)
  verifier_shuffled  same but the query is taken from another row (control: should not help)
  oracle           argmax cos(clip, TRUE stimulus embedding) -- upper bound, not available at test
  random           a random candidate (control)

Metrics per arm: cos(clip, EEG query) and cos(clip, true stimulus). The second one is the
honest question: does verification pick clips closer to what the subject actually heard?

Run (GPU, after the subject-held-out retrain finishes):
    python code/information/verifier_eval.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = ROOT.parent / "mainv2"
OUT = ROOT / "results" / "verifier"
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT / "code" / "radii"))
sys.path.insert(0, str(ROOT / "code" / "classifier"))
sys.path.insert(0, str(M2 / "src" / "generate"))

SR = 32000
TOPM = 4
TAU = 1.0
SEED_BASE = 1234


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def boot_ci(d, n=4000):
    rng = np.random.default_rng(0)
    b = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n)])
    return [float(x) for x in np.percentile(b, [2.5, 97.5])]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rep = json.load(open(ROOT / "results" / "library_b" / "listen" / "report.json", encoding="utf-8"))
    rows = rep["per_row"]
    stim = np.load(ROOT / "results" / "stim_vs_gen" / "emb.npz")["stim"].astype(np.float32)
    stim = unit(stim)
    assert len(stim) == len(rows), f"{len(stim)} stim embeddings vs {len(rows)} rows"
    cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    from musicgen_clap_finetune import MusicGenCLAP, load_clap, clap_embed_audio
    from arms_2x2 import embed_text, gen_from_enc
    from library_b_experiment import PROMPTS_B
    from finetune_prompt_classifier import PromptClassifier

    base = MusicGenCLAP(device=device, use_lora=False); mg = base.mg; mg.eval()
    clap, proc, _ = load_clap(device)
    ck = torch.load(ROOT / "results" / "library_b" / "classifier.pt",
                    map_location="cpu", weights_only=True)
    clf = PromptClassifier(512, 256, len(PROMPTS_B), 0.1)
    clf.load_state_dict(ck["model"]); clf.eval().to(device)
    ph = {k: embed_text(mg, PROMPTS_B[k], device) for k in range(len(PROMPTS_B))}
    q = unit(cond)

    per_row = []
    for i, r in enumerate(rows):
        row = int(r["row"])
        with torch.no_grad():
            logits = clf(torch.from_numpy(cond[row:row + 1]).to(device)).float()
        w = torch.softmax(logits / TAU, 1)[0].cpu().numpy()
        top = np.argsort(w)[::-1][:TOPM]
        cands = []
        for k in top:
            wav = gen_from_enc(mg, ph[int(k)].unsqueeze(0), 400, 0.9, None,
                               seed=SEED_BASE + 1000 * int(k) + row)
            wav = np.asarray(wav, np.float32).reshape(-1)
            c = np.asarray(clap_embed_audio(clap, proc, SR, wav, device)).reshape(-1).astype(np.float32)
            cands.append(unit(c))
            sf.write(OUT / f"row{row}__p{int(k)}.wav", wav, SR)
        C = np.stack(cands)
        qq = q[row]
        per_row.append({
            "row": row, "prompts": [int(k) for k in top],
            "weights": [float(w[int(k)]) for k in top],
            "head_top1_idx": 0,
            "cos_query": [float(x) for x in (C @ qq)],
            "cos_true": [float(x) for x in (C @ stim[i])],
        })
        print(f"row {row}: top{TOPM}={top.tolist()} "
              f"q={np.round(C @ qq, 3).tolist()} true={np.round(C @ stim[i], 3).tolist()}", flush=True)

    # ---- arms ----
    n = len(per_row)
    order = np.arange(n)
    shuffled = np.roll(order, 1)                 # query from the previous row
    rec = {}
    for arm in ("head_top1", "verifier", "verifier_shuffled", "oracle", "random"):
        chosen = []
        for j, d in enumerate(per_row):
            cq = np.array(d["cos_query"]); ct = np.array(d["cos_true"])
            if arm == "head_top1":
                k = 0
            elif arm == "verifier" or arm == "verifier_shuffled":
                other = per_row[shuffled[j]] if arm == "verifier_shuffled" else d
                k = int(np.argmax(other["cos_query"]))
            elif arm == "oracle":
                k = int(np.argmax(ct))
            else:
                k = int(np.random.default_rng(j).integers(0, len(ct)))
            chosen.append(k)
        cq = np.array([per_row[j]["cos_query"][k] for j, k in enumerate(chosen)])
        ct = np.array([per_row[j]["cos_true"][k] for j, k in enumerate(chosen)])
        rec[arm] = {"cos_query_mean": float(cq.mean()), "cos_query_ci": boot_ci(cq),
                    "cos_true_mean": float(ct.mean()), "cos_true_ci": boot_ci(ct),
                    "chosen": chosen}

    # paired comparison: verifier vs head_top1 on cos_true
    vt = np.array([per_row[j]["cos_true"][k] for j, k in enumerate(rec["verifier"]["chosen"])])
    ht = np.array([per_row[j]["cos_true"][0] for j in range(n)])
    dlt = vt - ht
    rec["verifier_minus_head_top1_cos_true"] = {
        "mean": float(dlt.mean()), "t": float(dlt.mean() / (dlt.std(ddof=1) / np.sqrt(n))),
        "wins": int((dlt > 0).sum()), "n": n, "ci95": boot_ci(dlt)}
    rec["oracle_minus_head_top1_cos_true"] = float(
        rec["oracle"]["cos_true_mean"] - rec["head_top1"]["cos_true_mean"])
    rec["coverage"] = {"n_rows": n, "top_m": TOPM, "prompts": len(PROMPTS_B),
                       "cos_true_own_mean": float((stim @ stim.T)[np.triu_indices(n, 1)].mean())}

    json.dump({"per_row": per_row, "arms": rec, "protocol": {
        "generator": "frozen MusicGen, 400 tokens, temp 0.9, fixed seeds",
        "query": "bridge output (laion-CLAP); verifier scores generated clips against it",
        "true": "laion-CLAP embedding of the heard stimulus (stim_vs_gen/emb.npz)",
        "not_available_at_test": ["oracle", "true stimulus embedding"],
    }}, open(OUT / "verifier_result.json", "w", encoding="utf-8"), indent=2)
    print("\nARMS")
    for arm in ("head_top1", "verifier", "verifier_shuffled", "oracle", "random"):
        print(f"  {arm:18} cos_query {rec[arm]['cos_query_mean']:.4f} "
              f"cos_true {rec[arm]['cos_true_mean']:.4f}")
    print("verifier - head_top1 (cos_true):", rec["verifier_minus_head_top1_cos_true"])
    print("wrote", OUT / "verifier_result.json")


if __name__ == "__main__":
    main()
