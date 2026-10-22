"""A pilot: calibration-aware head on frozen LaBraM-200 features.

The mainline trained head loses the calibratable subject fiber (calibrated ceiling ~0.32,
vs 0.39 for frozen LaBraM-200 and 0.71 for band-power). This pilot asks whether an explicit
calibration loss during head training preserves/creates that fiber.

Two heads, same data and batches:
  clip_only : InfoNCE between the head output and the audio CLAP target
  calib     : clip_only + per-subject differentiable Procrustes on support rows, then a
              shared 10-way classifier on the calibrated query rows

Evaluation on the held-out subjects (fold 0: 02-05; val 06-07): head output, k-shot
Procrustes to the 10 song prototypes, nearest prototype. Compare zero-shot and k = 10/100/300,
and against the frozen LaBraM-200 reference (zero 0.102, oracle 0.391, k=10 0.253).

Run: python code/information/calib_head_pilot.py
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
GATE = ROOT.parent / "mainv2" / "data" / "gate_nmedh14_ov2_a.npz"
OUT = ROOT / "results" / "calib_head_pilot"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
TEST = {"02", "03", "04", "05"}
VAL = {"06", "07"}
KS = [0, 10, 100, 300]
EPOCHS, STEPS, SUBJ_PER_STEP = 30, 100, 4


def unit(x):
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def procrustes_torch(A, B):
    """Thin differentiable orthogonal Procrustes; A, B are [m, d], m << d."""
    A = A - A.mean(0, keepdim=True)
    B = B - B.mean(0, keepdim=True)
    Ua, Sa, Vha = torch.linalg.svd(A, full_matrices=False)
    Ub, Sb, Vhb = torch.linalg.svd(B, full_matrices=False)
    M = (Sa[:, None] * (Ua.T @ Ub)) * Sb[None, :]
    P, _, Qh = torch.linalg.svd(M)
    return (Vha.T @ P) @ (Vhb.T @ Qh.T).T


def build_batch(sg, subs, rng):
    sup, qry, sup_song, qry_song, sup_sub, qry_sub = [], [], [], [], [], []
    for si, s in enumerate(subs):
        for so in range(10):
            segs = list(sg[s][so].items())
            if len(segs) < 2:
                continue
            i1, i2 = rng.choice(len(segs), 2, replace=False)
            r1 = segs[i1][1][0]; r2 = segs[i2][1][0]
            sup.append(r1); qry.append(r2)
            sup_song.append(so); qry_song.append(so)
            sup_sub.append(si); qry_sub.append(si)
    return (np.array(sup), np.array(qry), np.array(sup_song), np.array(qry_song),
            np.array(sup_sub), np.array(qry_sub))


def train_head(X, T, P, sg, train_subs, use_calib, seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    head = nn.Linear(X.shape[1], 512).to(DEV)
    clf = nn.Linear(512, 10).to(DEV) if use_calib else None
    params = list(head.parameters()) + (list(clf.parameters()) if clf else [])
    opt = torch.optim.Adam(params, lr=1e-3)
    X = X.to(DEV); T = unit(T.to(DEV)); P = unit(P.to(DEV))
    rng = np.random.default_rng(seed)
    for ep in range(EPOCHS):
        for _ in range(STEPS):
            subs = rng.choice(train_subs, SUBJ_PER_STEP, replace=False)
            rs, rq, ys, yq, ss, sq = build_batch(sg, subs, rng)
            if len(rs) < 20:
                continue
            zs = unit(head(X[torch.as_tensor(rs, device=DEV)]))
            zq = unit(head(X[torch.as_tensor(rq, device=DEV)]))
            allz = torch.cat([zs, zq]); allt = T[torch.as_tensor(np.concatenate([rs, rq]), device=DEV)]
            logits = (allz @ allt.T) / 0.07
            clip = F.cross_entropy(logits, torch.arange(len(allz), device=DEV))
            loss = clip
            if use_calib:
                cal = 0.0
                for si in range(SUBJ_PER_STEP):
                    ms = torch.as_tensor(ss == si, device=DEV)
                    mq = torch.as_tensor(sq == si, device=DEV)
                    if ms.sum() < 5 or mq.sum() < 5:
                        continue
                    Cs = torch.stack([zs[ms][torch.as_tensor(ys[ms.cpu().numpy()] == c, device=DEV)].mean(0)
                                      if (ys[ms.cpu().numpy()] == c).any() else torch.zeros_like(zs[0])
                                      for c in range(10)])
                    Cs = unit(Cs)
                    R = procrustes_torch(Cs, P)
                    qcal = (zq[mq] - zs[ms].mean(0)) @ R
                    logits_c = clf(qcal)
                    cal = cal + F.cross_entropy(logits_c, torch.as_tensor(yq[mq.cpu().numpy()], device=DEV))
                loss = loss + cal / SUBJ_PER_STEP
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9:
            print(f"  [{('calib' if use_calib else 'clip')}] ep {ep+1} loss {float(loss):.4f}", flush=True)
    return head


def pick_rows_eval(rows_c, Z, n):
    if n >= len(rows_c):
        return rows_c
    if n <= 2:
        cen = Z[rows_c].mean(0)
        order = np.argsort(-(unit_np(Z[rows_c]) @ unit_np(cen)))
        return rows_c[order[:n]]
    U = unit_np(Z[rows_c])
    sel = [int(np.argmax(U @ unit_np(U.mean(0))))]
    dist = 1.0 - U @ U[sel[0]]
    for _ in range(1, n):
        i = int(np.argmax(dist)); sel.append(i)
        dist = np.minimum(dist, 1.0 - U @ U[i])
    return rows_c[np.array(sel)]


def evaluate(head, X, P, song, subj, seg, al, seed=0):
    head.eval()
    P = unit(P.to(DEV)); rng = np.random.default_rng(seed)
    acc = {k: [] for k in KS}
    with torch.no_grad():
        Z = unit(head(X.to(DEV))).cpu().numpy()
    for s in TEST:
        rows = np.where(al & (subj == s))[0]
        segs = np.unique(seg[rows]); rng.shuffle(segs)
        fg = set(segs[:len(segs) // 2])
        rf = rows[np.isin(seg[rows], list(fg))]
        rt = rows[~np.isin(seg[rows], list(fg))]
        if len(rf) < 100 or len(rt) < 100:
            continue
        yte = song[rt]
        for k in KS:
            if k == 0:
                pred = (unit_np(Z[rt]) @ unit_np(P.cpu().numpy()).T).argmax(1)
            else:
                picks = []
                for c in range(10):
                    rc = rf[song[rf] == c]
                    if len(rc) == 0:
                        continue
                    picks.append(pick_rows_eval(rc, Z, min(k // 10, len(rc))))
                ap = np.concatenate(picks); ay = song[ap]
                present = np.array([c for c in range(10) if (ay == c).any()])
                Cs = unit_np(np.stack([Z[ap][ay == c].mean(0) for c in present]))
                Pp = unit_np(P.cpu().numpy())[present]
                A = torch.tensor(Cs - Cs.mean(0), dtype=torch.float32, device=DEV)
                B = torch.tensor(Pp - Pp.mean(0), dtype=torch.float32, device=DEV)
                R = procrustes_torch(A, B).cpu().numpy()
                test = (Z[rt] - Cs.mean(0)) @ R
                pred = (unit_np(test) @ unit_np(P.cpu().numpy()).T).argmax(1)
            acc[k].append(float((pred == yte).mean()))
    return {k: float(np.mean(v)) for k, v in acc.items()}, acc


def unit_np(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="labram", choices=["labram", "bandpower"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    z = np.load(GATE, allow_pickle=True)
    X = torch.tensor(z[args.features].astype(np.float32))
    T = torch.tensor(z["target"].astype(np.float32))
    P = torch.tensor(z["proto"][:10].astype(np.float32))
    song = z["song"].astype(int); subj = z["subject"].astype(str)
    seg = z["seg2"].astype(int); ds = z["dataset"].astype(int); al = z["aligned8"].astype(bool)
    train_subs = [s for s in np.unique(subj) if s not in TEST and s not in VAL and s in set(subj[ds == 0])]
    train_rows = np.where((ds == 0) & np.isin(subj, train_subs))[0]
    sg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in train_rows:
        sg[subj[r]][song[r]][seg[r]].append(r)
    print(f"fold 0: train subjects {len(train_subs)}, test {sorted(TEST)}")

    res = {"protocol": {"features": f"frozen {args.features}", "head": f"linear {X.shape[1]}->512, L2",
                        "fold": "test 02-05, val 06-07", "epochs": EPOCHS,
                        "batch": f"{SUBJ_PER_STEP} subjects x 10 songs x (1 support + 1 query)"},
           "heads": {}, "reference_frozen": ({"zero": 0.102, "oracle": 0.391, "k10": 0.253} if args.features == "labram" else {"zero": 0.089, "oracle_fold": 0.508, "k10_fold": 0.413})}
    for name, use_calib in (("clip_only", False), ("calib", True)):
        print(f"training {name} ...")
        head = train_head(X, T, P, sg, train_subs, use_calib)
        torch.save(head.state_dict(), OUT / f"{name}_{args.features}_head.pt")
        acc, raw = evaluate(head, X, P, song, subj, seg, al)
        res["heads"][name] = acc
        print(f"{name}: " + " ".join(f"k{k} {acc[k]:.3f}" for k in KS))
    out = OUT / f"{args.features}_pilot.json"
    json.dump(res, open(out, "w", encoding="utf-8"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
