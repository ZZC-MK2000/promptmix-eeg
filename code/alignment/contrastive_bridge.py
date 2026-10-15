"""Fix weakness #6: the ridge bridge has poor ABSOLUTE alignment (cos 0.19 to library means; the
argmax-prompt readout is song-uninformative, 16.7%). Retrain the EEG->laion bridge with a
CONTRASTIVE (InfoNCE) objective so EEG embeddings land near their audio, then test whether the
absolute-alignment readout is fixed.

Data (from the mainv2 cache): labram EEG emb (768) and audio laion-CLAP (512) per window, plus
subject + song. Split by SUBJECT (held-out). Library means from the corpus-matched library_b.

After training, on held-out subjects compare to the ridge bridge:
  (a) absolute alignment: cos(bridge(EEG), library prompt means)  [ridge: 0.19]
  (b) argmax-prompt readout agreement with the true-audio oracle  [ridge: 16.7%]  <-- the key test
  (c) genre-classifier kappa trained on bridge output             [ridge: 0.551]
  (d) hard-retrieval top-1 (14-way and 500-pool)                  [ridge: 0.88 / 0.39]

Run: python code/alignment/contrastive_bridge.py   (from promptmix-eeg/)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
LIBB = ROOT / "results" / "library_b"
OUT = ROOT / "results" / "contrastive_bridge"
FMA = Path("D:/Projects/CLaBraM/promptmix-eeg/data/fma8s")


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


class Bridge(nn.Module):
    def __init__(self, din=768, dout=512, hid=1024):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Linear(hid, dout))

    def forward(self, x):
        return torch.nn.functional.normalize(self.net(x), dim=-1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "log.txt", "w", encoding="utf-8")

    def say(s):
        print(s, flush=True); log.write(s + "\n"); log.flush()

    t0 = time.time()
    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    # FAIR input: the ridge bridge consumed the aldm-aligned EEG, so train the contrastive
    # bridge from the SAME input (raw labram would be an unfair handicap).
    eeg = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)   # (N,768) aldm-aligned
    aud = z["target"].astype(np.float32)          # (N,512) laion of heard audio
    subj = np.unique(np.asarray(z["subject"]).astype(str), return_inverse=True)[1].astype(np.int64)
    song = z["song"].astype(int)
    n = min(len(eeg), len(aud), len(song)); eeg, aud, subj, song = eeg[:n], aud[:n], subj[:n], song[:n]
    say(f"eeg {eeg.shape} aud {aud.shape} subjects={len(np.unique(subj))} songs={len(np.unique(song))}")

    # library means + oracle (audio-side, bridge-independent)
    lib = np.load(LIBB / "library_emb.npy").astype(np.float32)
    owner = np.load(LIBB / "library_owner.npy").astype(np.int64)
    K = owner.max() + 1
    means = unit(np.stack([unit(lib[owner == k].mean(0)) for k in range(K)]))
    oracle = (unit(aud) @ means.T).argmax(1)

    # subject split
    uniq = np.unique(subj); rng = np.random.default_rng(0)
    vg = set(uniq[rng.permutation(len(uniq))[:max(1, int(round(len(uniq) * 0.1)))]].tolist())
    va = np.array([s in vg for s in subj]); tr = ~va
    say(f"train {tr.sum()} / val {va.sum()} rows; val subjects {len(set(subj[va]))}")

    Xe = torch.from_numpy(eeg[tr]).to(device); Xa = torch.from_numpy(unit(aud[tr])).to(device)
    model = Bridge(eeg.shape[1], aud.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    tau = 0.1; B = 256
    model.train()
    for step in range(1, 4001):
        idx = torch.randint(0, len(Xe), (B,), device=device)
        e = model(Xe[idx]); a = Xa[idx]
        logits = (e @ a.T) / tau
        lab = torch.arange(B, device=device)
        loss = nn.functional.cross_entropy(logits, lab)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0:
            say(f"  step {step} loss={loss.item():.3f} [{time.time()-t0:.0f}s]")

    # evaluate on held-out subjects
    model.eval()
    with torch.no_grad():
        Zv = model(torch.from_numpy(eeg[va]).to(device)).cpu().numpy()
    zv = unit(Zv); audv = unit(aud[va]); ov = oracle[va]; sv = song[va]

    cos_lib = (zv @ means.T)
    abs_align = float(cos_lib.max(1).mean())
    argmax_pred = cos_lib.argmax(1)
    argmax_agree = float((argmax_pred == ov).mean())
    say(f"\n[contrastive bridge, held-out] abs-alignment cos(bridge,libmeans)={abs_align:.4f} (ridge 0.19)")
    say(f"[KEY] argmax-prompt readout agreement w/ oracle = {argmax_agree:.4f} (ridge 0.167, chance {1/K:.3f})")

    # genre classifier on bridge output (held-out), kappa
    from sklearn.metrics import cohen_kappa_score
    trv = ~va
    with torch.no_grad():
        Zt = model(torch.from_numpy(eeg[trv]).to(device)).cpu().numpy()
    clf = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, K)).to(device)
    o2 = torch.optim.AdamW(clf.parameters(), lr=1e-3)
    c2 = nn.CrossEntropyLoss()
    Xt = torch.from_numpy(Zt).to(device); Yt = torch.from_numpy(oracle[trv]).to(device)
    Xv = torch.from_numpy(Zv).to(device); Yv = torch.from_numpy(ov).to(device)
    clf.train()
    for step in range(1, 2001):
        i = torch.randint(0, len(Xt), (256,), device=device)
        l = c2(clf(Xt[i]), Yt[i]); o2.zero_grad(); l.backward(); o2.step()
    clf.eval()
    with torch.no_grad():
        p = clf(Xv).argmax(1).cpu().numpy()
    yv = ov
    bal = float(np.mean([(p[yv == c] == c).mean() for c in range(K) if (yv == c).sum() > 0]))
    kap = float(cohen_kappa_score(yv, p))
    say(f"[genre clf on contrastive bridge] overall={float((p==yv).mean()):.4f} balanced={bal:.4f} kappa={kap:.4f} (ridge kappa 0.551)")

    # hard retrieval: song centroids (from audio) + FMA distractors, bridge(EEG) queries
    cent = unit(np.stack([unit(aud[song == s]).mean(0) for s in range(14)]))
    dist = unit(np.load(FMA / "cond.npy").astype(np.float32))
    def ret(D):
        G = np.vstack([cent, dist[np.random.default_rng(7).permutation(len(dist))[:D]]]) if D else cent
        return float(((zv @ G.T).argmax(1) == sv).mean())
    say(f"[retrieval] 14-way={ret(0):.4f}  500-pool={ret(486):.4f}  4146-pool={ret(len(dist)):.4f}")

    json.dump({"abs_align": abs_align, "argmax_agree": argmax_agree, "genre_kappa": kap,
               "genre_balanced": bal, "retr_14": ret(0), "retr_500": ret(486), "retr_4146": ret(len(dist)),
               "ridge_baseline": {"abs_align": 0.19, "argmax_agree": 0.167, "genre_kappa": 0.551,
                                   "retr_14": 0.882, "retr_500": 0.388, "retr_4146": 0.266}},
              open(OUT / "result.json", "w"), indent=2)
    torch.save({"bridge": model.state_dict()}, OUT / "bridge.pt")
    say(f"\ndone -> {OUT}  [{time.time()-t0:.0f}s]"); log.close()


if __name__ == "__main__":
    main()
