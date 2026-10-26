"""Trial-wise 10-way NMED-T retrieval for the spatial-spectral encoder vs bandpower/LaBraM.

Protocol (mainstream, as requested): for each subject and song, split that subject's
windows 50/50; build class centroids from the train half; classify the test half.
Subject is shared between train/test, stimulus is the same 10 songs; this is a
trial-wise/segment-held-out protocol, not cross-subject.
"""
import argparse
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
from spatial_spectral import SpatialSpectralEncoder  # noqa: E402
from eval_finetune_local75 import nmed_windows  # noqa: E402

GATE = ROOT.parent / "mainv2" / "data" / "gate_nmedt10.npz"


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def trial_acc(X, song, subj, reps=5):
    accs = []
    for r in range(reps):
        rng = np.random.default_rng(r)
        pred, true = [], []
        for s in np.unique(subj):
            rs = np.where(subj == s)[0]
            tr, te, ytr, yte = [], [], [], []
            for c in range(10):
                rr = rs[song[rs] == c]
                if len(rr) < 4:
                    continue
                rr = rng.permutation(rr)
                h = len(rr) // 2
                tr.append(rr[:h]); te.append(rr[h:])
                ytr += [c] * len(rr[:h]); yte += [c] * len(rr[h:])
            if len(set(ytr)) < 10:
                continue
            tr = np.concatenate(tr); te = np.concatenate(te)
            C = np.stack([X[tr][np.array(ytr) == c].mean(0) for c in range(10)])
            pr = unit(X[te]) @ unit(C).T
            pred += pr.argmax(1).tolist(); true += yte
        accs.append(float(np.mean(np.array(pred) == np.array(true))))
    return float(np.mean(accs)), float(np.std(accs))


def trial_logistic(X, song, subj, reps=5):
    accs = []
    for r in range(reps):
        rng = np.random.default_rng(r)
        pred, true = [], []
        for s in np.unique(subj):
            rs = np.where(subj == s)[0]
            tr, te, ytr, yte = [], [], [], []
            for c in range(10):
                rr = rs[song[rs] == c]
                if len(rr) < 4:
                    continue
                rr = rng.permutation(rr)
                h = len(rr) // 2
                tr.append(rr[:h]); te.append(rr[h:])
                ytr += [c] * len(rr[:h]); yte += [c] * len(rr[h:])
            if len(set(ytr)) < 10:
                continue
            tr = np.concatenate(tr); te = np.concatenate(te)
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=2000, C=1.0))
            clf.fit(X[tr], np.array(ytr))
            pred += clf.predict(X[te]).tolist(); true += yte
        accs.append(float(np.mean(np.array(pred) == np.array(true))))
    return float(np.mean(accs)), float(np.std(accs))


@torch.no_grad()
def encode(model, X, bs=64, device="cuda"):
    model.eval(); out = []
    for i in range(0, len(X), bs):
        x = torch.tensor(X[i:i + bs], dtype=torch.float32, device=device)
        x = x - x.mean(-1, keepdim=True)
        x = x / (x.std(-1, keepdim=True) + 1e-6)
        x = torch.clamp(x, -10, 10)
        z, _, _ = model(x)
        out.append(z.cpu())
    return torch.cat(out, 0).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "results" / "spatial_spectral_contrast" / "encoder_500.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "spatial_spectral_contrast" / "trialwise_nmed.json"))
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    if ck.get("arch") == "v3":
        from spatial_spectral_v3 import SpatialSpectralV3
        model = SpatialSpectralV3(n_ch=19, d=128, depth=4, heads=4, ch_depth=2,
                                  patch=200, sfreq=200, spectral_skip=True)
    else:
        model = SpatialSpectralEncoder(n_ch=19, d=128, depth=4, heads=4,
                                       n_filters=a.get("n_filters", 8), patch=200, sfreq=200,
                                       spectral_skip=a.get("spectral_skip", False))
    model.load_state_dict(ck["model"]); model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    X, song, subj = nmed_windows()
    zgate = np.load(GATE, allow_pickle=True)
    assert len(zgate["song"]) == len(song) and np.array_equal(zgate["song"], song)
    feats = {"spatial_spectral": encode(model, X, device=device)}
    for name in ("bandpower", "labram"):
        F = zgate[name].astype(np.float32)
        feats[name] = (F - F.mean(0)) / (F.std(0) + 1e-6)
    bp95_path = ROOT / "results" / "auditory_encoder" / "nmed_bp95.npy"
    if bp95_path.exists():
        F = np.load(bp95_path).astype(np.float32)
        feats["bandpower_19x5"] = (F - F.mean(0)) / (F.std(0) + 1e-6)
    res = {"protocol": {"corpus": "NMED-T 20 subjects x 10 songs",
                        "split": "per-subject per-song 50/50 trial-wise",
                        "chance": 0.10, "reps": 5, "encoder_ckpt": args.ckpt},
           "features": {}}
    for name, F in feats.items():
        acc, sd = trial_acc(F, song, subj)
        lacc, lsd = trial_logistic(F, song, subj)
        res["features"][name] = {"acc": acc, "sd": sd,
                                 "logistic": lacc, "logistic_sd": lsd}
        print(f"{name:18} centroid={acc:.3f} ± {sd:.3f} "
              f"logistic={lacc:.3f} ± {lsd:.3f}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
