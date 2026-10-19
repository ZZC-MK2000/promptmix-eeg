"""Learned prompt codebook -- selected to cover songs it was NOT selected on.

The existing library was hand-picked on the same corpus it is evaluated on. This asks the
generalization question that matters for a paper: if we select K=12 prompts from the
22-prompt pool (12 existing + 10 new MUSIN-G genres) on a training set of songs, do they
still cover the real-window cloud of held-out songs?

Selection: greedy facility location -- add the prompt whose clip set raises the running
mean max-cosine the most -- optionally with a diversity constraint (pairwise prompt-mean
cosine <= c_max) so paraphrases cannot fill the codebook. Prompts stay native: nothing is
interpolated, so the on-manifold requirement is satisfied by construction.

Evaluation: the same simplex coverage statistic as fig1 / coverage_expanded (fit a convex
combination over the individual clips of the selected prompts, per target window).

Run: python code/information/codebook_select.py     (CPU)
"""
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
OUT = ROOT / "results" / "codebook"
LIBF = ROOT / "results" / "coverage_expanded"
N_TRAIN, N_TEST, ITERS = 2000, 500, 200
K = 12
SEED = 0


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def proj_simplex(v):
    u = np.sort(v)[::-1]; css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    return np.maximum(v - (css[rho] - 1) / (rho + 1), 0)


def fit_simplex(L, t, iters=ITERS):
    m = L.shape[0]; eta = 1.0 / (np.linalg.norm(L, 2) ** 2 + 1e-8)
    w = np.full(m, 1.0 / m)
    for _ in range(iters):
        r = L.T @ w - t
        w = proj_simplex(w - eta * (L @ r))
    return w


def coverage(T, lib, own, k):
    means = unit(np.stack([lib[own == j].mean(0) for j in range(k)]))
    mix, best, bmean = [], [], []
    for t in T:
        w = fit_simplex(lib, t)
        mix.append(float(unit(lib.T @ w) @ t))
        best.append(float((lib @ t).max()))
        bmean.append(float((means @ t).max()))
    return {"convex_mixture": float(np.mean(mix)),
            "best_single_clip": float(np.mean(best)),
            "best_prompt_mean": float(np.mean(bmean)),
            "trivial_target_mean": float((T @ unit(T.mean(0))).mean())}


def select_greedy(T, lib, owner, k_pool, want, c_max=None):
    per = unit(np.stack([lib[owner == j].mean(0) for j in range(k_pool)]))
    chosen, cur = [], np.full(len(T), -2.0)
    for _ in range(want):
        best, bj = None, -1
        for j in range(k_pool):
            if j in chosen:
                continue
            if c_max is not None and chosen and (per[chosen] @ per[j]).max() > c_max:
                continue
            v = float(np.maximum(cur, T @ per[j]).mean())
            if best is None or v > best:
                best, bj = v, j
        if bj < 0:                      # constraint too tight: relax and take the best
            bj = int(np.argmax([-1e9 if j in chosen else float(np.maximum(cur, T @ per[j]).mean())
                                for j in range(k_pool)]))
        chosen.append(bj); cur = np.maximum(cur, T @ per[bj])
    return chosen


def subsample(rows, n, rng):
    if len(rows) <= n:
        return rows
    return rows[np.sort(rng.choice(len(rows), n, replace=False))]


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    lib = unit(np.load(LIBF / "library_emb.npy").astype(np.float32))
    owner = np.load(LIBF / "library_owner.npy").astype(int)
    prompts = json.load(open(LIBF / "prompts.json", encoding="utf-8"))
    T = unit(np.load(ROOT / "results" / "prompt_classifier" / "true_emb_all.npy").astype(np.float32))
    n_pool = int(owner.max()) + 1
    assert lib.shape[0] == n_pool * 6 and len(prompts) == n_pool

    # rows -> gate -> song (the kept rows are byte-identical bridge outputs)
    import torch
    z = np.load(ROOT.parent / "mainv2" / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)
    aldm = np.load(ROOT.parent / "mainv2" / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    w = torch.load(ROOT.parent / "mainv2" / "results" / "eeg_to_clap_proj.pt",
                   map_location="cpu", weights_only=True)
    ca = unit(aldm @ w["weight"].numpy().astype(np.float32).T + w["bias"].numpy().astype(np.float32))
    key = {ca[i].tobytes(): i for i in range(len(ca))}
    idx = np.array([key[cond[j].tobytes()] for j in range(len(cond))])
    song = z["song"][idx].astype(int)
    assert len(T) == len(song)

    rng = np.random.default_rng(SEED)
    songs = np.unique(song)
    folds = np.array_split(rng.permutation(songs), 5)
    arms = ["existing_12", "all_22", "greedy_12", "greedy_12_diverse", "random_12"]
    per_fold = {a: [] for a in arms}
    picks = {a: [] for a in arms}

    print(f"{len(song)} windows, {len(songs)} songs, {n_pool} prompts, K={K}")
    for f, test_songs in enumerate(folds):
        tr_rows = np.where(~np.isin(song, test_songs))[0]
        te_rows = np.where(np.isin(song, test_songs))[0]
        Ttr = T[subsample(tr_rows, N_TRAIN, rng)]
        Tte = T[subsample(te_rows, N_TEST, rng)]
        for a in arms:
            if a == "existing_12":
                pick = list(range(12))
            elif a == "all_22":
                pick = list(range(n_pool))
            elif a == "greedy_12":
                pick = select_greedy(Ttr, lib, owner, n_pool, K)
            elif a == "greedy_12_diverse":
                pick = select_greedy(Ttr, lib, owner, n_pool, K, c_max=0.90)
            else:
                pick = rng.permutation(n_pool)[:K].tolist()
            sel = np.concatenate([np.where(owner == j)[0] for j in pick])
            own = np.repeat(np.arange(len(pick)), 6)
            r = coverage(Tte, lib[sel], own, len(pick))
            per_fold[a].append(r["convex_mixture"])
            picks[a].append(sorted(int(x) for x in pick))
            print(f"  fold {f} test_songs={sorted(int(s) for s in test_songs)} {a:18} "
                  f"convex {r['convex_mixture']:.4f} (trivial {r['trivial_target_mean']:.4f}) "
                  f"[{time.time()-t0:.0f}s]")

    res = {"protocol": {
        "selection": "greedy facility location on training-song windows; prompts stay native",
        "evaluation": "simplex coverage on held-out-song windows",
        "folds": 5, "n_train": N_TRAIN, "n_test": N_TEST, "simplex_iters": ITERS, "K": K,
        "pool": n_pool, "diversity_cmax": 0.90,
    }, "arms": {}}
    for a in arms:
        v = np.array(per_fold[a])
        res["arms"][a] = {"convex_mean": float(v.mean()), "convex_sd": float(v.std(ddof=1)),
                          "convex_per_fold": v.tolist(), "picked": picks[a],
                          "gain_vs_existing": float(v.mean() - np.mean(per_fold["existing_12"]))}
    res["summary"] = {a: round(res["arms"][a]["convex_mean"], 4) for a in arms}
    res["summary"]["gain_greedy_vs_existing"] = round(
        res["arms"]["greedy_12"]["convex_mean"] - res["arms"]["existing_12"]["convex_mean"], 4)
    res["summary"]["gain_diverse_vs_existing"] = round(
        res["arms"]["greedy_12_diverse"]["convex_mean"] - res["arms"]["existing_12"]["convex_mean"], 4)
    json.dump(res, open(OUT / "codebook_select.json", "w", encoding="utf-8"), indent=2)
    print("SUMMARY", json.dumps(res["summary"], indent=1))
    print("wrote", OUT / "codebook_select.json")


if __name__ == "__main__":
    main()
