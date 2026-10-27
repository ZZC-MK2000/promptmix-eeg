"""Frozen pretrained foundation-model features on the same NMED-T trial-wise splits.

BIOT and CBraMod weights are loaded from the Hugging Face mirror; EEGPT is
attempted separately because its released checkpoint uses a 62-channel token
layout. No NMED-T data are used to fit the encoders; only the downstream
logistic/centroid probe is fit on the training halves, exactly as for LaBraM.
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import mne
import numpy as np
import torch

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code" / "information"))
from eval_finetune_local75 import nmed_windows  # noqa: E402
from eval_trialwise import trial_acc, trial_logistic  # noqa: E402

CH = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
      "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]


def norm(x):
    x = x.float()
    x = x - x.mean(-1, keepdim=True)
    return torch.clamp(x / (x.std(-1, keepdim=True) + 1e-6), -10, 10)


def load_model(name):
    from braindecode.models import CBraMod, InterpolatedBIOT
    mont = mne.channels.make_standard_montage("standard_1020")
    chs = [{"ch_name": c, "loc": mont.get_positions()["ch_pos"][c]} for c in CH]
    kw = dict(n_outputs=10, n_times=1600, sfreq=200, input_window_seconds=8)
    if name == "biot":
        return InterpolatedBIOT.from_pretrained(
            "braindecode/biot-pretrained-six-datasets-18chs",
            chs_info=chs, return_feature=True, **kw)
    if name == "cbramod":
        return CBraMod.from_pretrained(
            "braindecode/cbramod-pretrained", n_chans=19,
            return_encoder_output=True, **kw)
    raise ValueError(name)


@torch.no_grad()
def extract(model, X, name, device, bs=64):
    out = []
    model.eval().to(device)
    for i in range(0, len(X), bs):
        xb = norm(torch.from_numpy(X[i:i + bs]).to(device))
        y = model(xb)
        if name == "biot":
            y = y[1]  # (logits, embedding) with return_feature=True
        elif name == "cbramod":
            y = y.mean(dim=(1, 2))  # [B,19,8,200] -> [B,200]
        out.append(y.float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="biot,cbramod")
    ap.add_argument("--out", default=str(ROOT / "results" / "compare_foundation_models" / "results.json"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    X, song, subj = nmed_windows()
    X = X.astype(np.float32)
    res = {"protocol": {"dataset": "NMED-T", "split": "per-subject per-song 50/50",
                        "chance": 0.10, "reps": 5,
                        "note": "frozen pretrained features; probe fit on training halves"},
           "models": {}}
    for name in args.models.split(","):
        model = load_model(name)
        params = sum(p.numel() for p in model.parameters())
        F = extract(model, X, name, device)
        a, sd = trial_acc(F, song, subj)
        la, lsd = trial_logistic(F, song, subj)
        res["models"][name] = {"params": params, "feature_dim": int(F.shape[1]),
                               "centroid": a, "centroid_sd": sd,
                               "logistic": la, "logistic_sd": lsd}
        print(f"{name}: dim {F.shape[1]} params {params} "
              f"centroid {a:.3f} logistic {la:.3f}", flush=True)
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
