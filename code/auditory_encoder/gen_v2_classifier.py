"""Generation pilot v2: direct per-subject description classifier on encoder content.

calibrated: fit a 10-way logistic head on the subject's first-half windows;
zero: fit it on the other subjects' windows;
oracle: true song description.
Then generate 8 windows x 3 arms with frozen MusicGen and report own vs other CLAP.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code" / "information"))
sys.path.insert(0, str(ROOT / "code" / "shared"))
sys.path.insert(0, str(ROOT.parent / "mainv2" / "src" / "generate"))
from spatial_spectral import SpatialSpectralEncoder
from eval_trialwise import encode
from eval_finetune_local75 import nmed_windows
from arms_2x2 import embed_text, gen_from_enc
from musicgen_clap_finetune import MusicGenCLAP, load_clap, clap_embed_audio

GATE = ROOT.parent / "mainv2" / "data" / "gate_nmedt10.npz"
DESC = ROOT / "results" / "descriptions" / "descriptions.json"
OUT = ROOT / "results" / "generation_v2_classifier"
CKPT = ROOT / "results" / "spatial_spectral_k32" / "encoder_500.pt"
SR = 32000
N_SUB = 4
N_WIN = 2


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    import soundfile as sf
    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    model = SpatialSpectralEncoder(n_ch=19, d=128, depth=4, heads=4,
                                   n_filters=a.get("n_filters", 32), patch=200, sfreq=200,
                                   spectral_skip=a.get("spectral_skip", True))
    model.load_state_dict(ck["model"]); model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"; model.to(device)
    X, song, subj = nmed_windows()
    z = np.load(GATE, allow_pickle=True)
    assert np.array_equal(z["song"], song)
    T = unit(z["target"].astype(np.float32))
    d = json.load(open(DESC, encoding="utf-8"))
    prompts = d["descriptions_specific"][:10]
    Z = unit(encode(model, X, device=device))
    song_means = unit(np.stack([T[song == c].mean(0) for c in range(10)]))

    picks = []
    subjects = sorted(set(subj))[:N_SUB]
    for s in subjects:
        rows = np.where(subj == s)[0]
        rf, rt = [], []
        for c in range(10):
            rr = rows[song[rows] == c]
            if len(rr) < 4:
                continue
            rr = np.random.default_rng(0).permutation(rr)
            h = len(rr) // 2
            rf += rr[:h].tolist(); rt += rr[h:].tolist()
        rf, rt = np.array(rf), np.array(rt)
        ro = np.where(subj != s)[0]
        clf_cal = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf_cal.fit(Z[rf], song[rf])
        clf_zero = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf_zero.fit(Z[ro], song[ro])
        pred_cal = clf_cal.predict(Z[rt])
        pred_zero = clf_zero.predict(Z[rt])
        take = np.linspace(0, len(rt) - 1, N_WIN).astype(int)
        for j in take:
            row = int(rt[j]); true_song = int(song[row])
            sel_cal = int(pred_cal[j]); sel_zero = int(pred_zero[j])
            picks.append({"row": row, "subject": str(s), "true_song": true_song,
                          "sel_cal": sel_cal, "sel_zero": sel_zero,
                          "prompt_cal": prompts[sel_cal], "prompt_zero": prompts[sel_zero],
                          "prompt_oracle": prompts[true_song]})

    print(f"selection: calibrated {np.mean([p['sel_cal']==p['true_song'] for p in picks]):.3f} "
          f"zero {np.mean([p['sel_zero']==p['true_song'] for p in picks]):.3f}", flush=True)
    wrap = MusicGenCLAP(device=device, use_lora=False); mg = wrap.mg; mg.eval()
    clap, proc, _ = load_clap(device)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "wav").mkdir(exist_ok=True)
    records = []
    for p in picks:
        row = p["row"]
        for arm, key in (("oracle", "prompt_oracle"), ("calibrated", "prompt_cal"),
                         ("zero", "prompt_zero")):
            ph = embed_text(mg, p[key], device).unsqueeze(0)
            wav = np.asarray(gen_from_enc(mg, ph, 400, 0.9, None, seed=1234 + row),
                             dtype=np.float32).reshape(-1)
            fn = f"{arm}_row{row}.wav"
            sf.write(OUT / "wav" / fn, wav, SR)
            g = unit(clap_embed_audio(clap, proc, SR, wav, device))
            own = float(g @ T[row])
            other = float(np.mean([g @ song_means[c] for c in range(10) if c != p["true_song"]]))
            records.append({**p, "arm": arm, "prompt": p[key], "audio": f"wav/{fn}",
                            "own_cos": own, "other_cos": other, "margin": own - other,
                            "desc_correct": int(
                                (key == "prompt_oracle") or
                                (key == "prompt_cal" and p["sel_cal"] == p["true_song"]) or
                                (key == "prompt_zero" and p["sel_zero"] == p["true_song"]))})
            print(f"{arm:10} row {row} own {own:.3f} other {other:.3f} "
                  f"margin {own-other:+.3f} desc {records[-1]['desc_correct']}", flush=True)
    agg = {}
    for arm in ("oracle", "calibrated", "zero"):
        rs = [r for r in records if r["arm"] == arm]
        agg[arm] = {"n": len(rs),
                    "own_cos": float(np.mean([r["own_cos"] for r in rs])),
                    "other_cos": float(np.mean([r["other_cos"] for r in rs])),
                    "margin": float(np.mean([r["margin"] for r in rs])),
                    "own_gt_other": float(np.mean([r["margin"] > 0 for r in rs])),
                    "desc_correct": float(np.mean([r["desc_correct"] for r in rs]))}
    res = {"protocol": {"encoder": str(CKPT), "selector": "per-subject logistic description head",
                        "n_windows": len(picks), "generator": "frozen MusicGen-small 400 tokens",
                        "own": "cos to true-stimulus CLAP",
                        "other": "mean cos to other 9 song CLAP centroids"},
           "aggregate": agg, "records": records}
    (OUT / "pilot.json").write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                    encoding="utf-8")
    print(json.dumps(agg, indent=2)); print("wrote", OUT / "pilot.json")


if __name__ == "__main__":
    main()
