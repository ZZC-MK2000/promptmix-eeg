"""Trial-wise semantic probe: predict the heard audio's CLAP embedding from EEG features.

For each subject x song, fit a ridge from train-half EEG features to the train-half audio
CLAP target, predict the test half, and report (a) cosine to the true target and (b) 10-way
top-1 to the song CLAP centroids. This is the fair test of the user's point: bandpower can
classify but may lose semantic/audio alignment.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
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


def trial_semantic(X, T, song, subj, reps=5, alpha=10.0):
    cos, top1 = [], []
    for r in range(reps):
        rng = np.random.default_rng(r)
        cs, ts, truth = [], [], []
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
            sc = StandardScaler().fit(X[tr])
            Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
            ytr_t = T[tr]
            C = unit(np.stack([ytr_t[np.array(ytr) == c].mean(0) for c in range(10)]))
            reg = Ridge(alpha=alpha).fit(Xtr, ytr_t)
            pred = unit(reg.predict(Xte))
            true = unit(T[te])
            cs.extend((pred * true).sum(1).tolist())
            ts.extend((pred @ C.T).argmax(1).tolist())
            truth.extend(yte)
        if cs:
            cos.append(float(np.mean(cs))); top1.append(float(np.mean(np.array(ts) == np.array(truth))))
    return float(np.mean(cos)), float(np.std(cos)), float(np.mean(top1)), float(np.std(top1))


def subject_semantic(X, T, song, subj, alpha=10.0):
    cos, top1 = [], []
    for s in np.unique(subj):
        tr = np.where(subj != s)[0]; te = np.where(subj == s)[0]
        if len(te) < 10:
            continue
        sc = StandardScaler().fit(X[tr])
        Xtr = sc.transform(X[tr]); Xte = sc.transform(X[te])
        C = unit(np.stack([T[tr][song[tr] == c].mean(0) for c in range(10)]))
        reg = Ridge(alpha=alpha).fit(Xtr, T[tr])
        pred = unit(reg.predict(Xte)); true = unit(T[te])
        cos.append(float((pred * true).sum(1).mean()))
        top1.append(float(np.mean((pred @ C.T).argmax(1) == song[te])))
    return float(np.mean(cos)), float(np.std(cos)), float(np.mean(top1)), float(np.std(top1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "results" / "spatial_spectral_contrast" / "encoder_500.pt"))
    ap.add_argument("--out", default=str(ROOT / "results" / "spatial_spectral_contrast" / "semantic_nmed.json"))
    ap.add_argument("--protocol", choices=["trial", "subject"], default="trial")
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
    device = "cuda" if torch.cuda.is_available() else "cpu"; model.to(device)
    X, song, subj = nmed_windows()
    z = np.load(GATE, allow_pickle=True)
    assert len(z["song"]) == len(song) and np.array_equal(z["song"], song)
    T = unit(z["target"].astype(np.float32))
    feats = {"spatial_spectral": encode(model, X, device=device)}
    for name in ("bandpower", "labram"):
        F = z[name].astype(np.float32)
        feats[name] = (F - F.mean(0)) / (F.std(0) + 1e-6)
    res = {"protocol": {"corpus": "NMED-T " + args.protocol,
                        "target": "audio CLAP (512-d), ridge",
                        "chance_top1": 0.10, "reps": 5}, "features": {}}
    for name, F in feats.items():
        fn = trial_semantic if args.protocol == "trial" else subject_semantic
        c_mean, c_sd, t_mean, t_sd = fn(F, T, song, subj)
        res["features"][name] = {"cos": c_mean, "cos_sd": c_sd,
                                 "top1": t_mean, "top1_sd": t_sd}
        print(f"{name:18} CLAP cos {c_mean:.3f} ± {c_sd:.3f} | top1 {t_mean:.3f} ± {t_sd:.3f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
