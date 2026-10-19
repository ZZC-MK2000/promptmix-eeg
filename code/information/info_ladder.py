"""Information ladder: what musical information does EEG carry, and for whom?

Corpus design (this decides whether the numbers mean anything):
  - NMED-T (dataset 0) = 20 subjects x 10 songs, fully crossed -> the clean ladder.
  - NMED-H (dataset 1) = 48 subjects, each listened to exactly 1 of 4 songs, so subject
    and song identity are confounded by design; reported separately, never pooled into a
    "held-out subject" claim.

Representations (frozen; the bridge is refit inside every training fold, so it is
inductive and comparable to the frozen features):
  - bandpower (1804-d), LaBraM (200-d), EEG-ALDM (512-d)
  - bridge: EEG-ALDM -> audio-CLAP ridge, fit on training rows only.

Splits (the split ladder):
  - random rows       : leaks subject AND adjacent segment (reference upper bound)
  - held-out segments : same subjects, new 8-s windows
  - held-out subjects : unseen listeners, all songs (the honest cross-subject number)

Levels:
  - song identity (10-way on NMED-T; 4-way on NMED-H)
  - style family (NMED-T vs NMED-H, 2-way; segment-held-out only, confound noted)
  - continuous audio content: cosine to the true audio-CLAP, cosine to a train-only song
    centroid, within-song segment retrieval (oracle song; only where the true segment is
    in the training set), and pooled R^2.

The 12-class "genre oracle" is NOT used as a ladder level: its labels collapse (one class
holds ~70% of NMED-T), so it cannot measure genre information. Its distribution is written
to the JSON as a label-validity control.

Every discrete number is paired with chance + majority; every continuous number with a
global-mean predictor. 95% CIs bootstrap over the held-out unit (subject / segment / row).

Run: python code/information/info_ladder.py
"""
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
GATE = ROOT.parent / "mainv2" / "data" / "gate_nmedh14_ov2_lfix.npz"
PROJ = ROOT.parent / "mainv2" / "results" / "eeg_to_clap_proj.pt"
OUT = ROOT / "results" / "info_ladder"
N_BOOT = 1000
SEED = 0


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def load_rows():
    """Kept rows of data/eegcond mapped back to their gate rows (bridge is byte-identical)."""
    z = np.load(GATE, allow_pickle=True)  # trusted local artifact from this workspace
    cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)
    aldm = np.load(ROOT.parent / "mainv2" / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    w = torch.load(PROJ, map_location="cpu", weights_only=True)
    ca = unit(aldm @ w["weight"].numpy().astype(np.float32).T + w["bias"].numpy().astype(np.float32))
    key = {ca[i].tobytes(): i for i in range(len(ca))}
    idx = np.array([key.get(cond[j].tobytes(), -1) for j in range(len(cond))])
    assert (idx >= 0).all(), "some kept rows do not map to a gate row"
    assert len(np.unique(idx)) == len(idx), "kept -> gate map is not injective"
    target = z["target"][idx].astype(np.float32)
    return dict(
        song=z["song"][idx].astype(np.int64),
        dataset=z["dataset"][idx].astype(np.int64),
        subject=z["subject"][idx],
        group=np.load(ROOT / "data" / "eegcond" / "group.npy").astype(np.int64),
        oracle=np.load(ROOT / "results" / "library_b" / "oracle_all.npy").astype(np.int64),
        target=target,
        target_unit=unit(target),
        bandpower=z["bandpower"][idx].astype(np.float32),
        labram=z["labram"][idx].astype(np.float32),
        aldm=aldm[idx],
    )


# ----------------------------------------------------------------------------- metrics
def conf_stats(y, pred, units, k):
    """Per-unit confusion matrices: bootstrap resamples units, not rows."""
    uu = {u: i for i, u in enumerate(np.unique(units))}
    C = np.zeros((len(uu), k, k), np.int64)
    for t, p, u in zip(y, pred, units):
        C[uu[u], t, p] += 1
    return C


def pooled_from_conf(c):
    n = c.sum()
    acc = np.trace(c) / n
    present = c.sum(1) > 0
    recall = np.diag(c)[present] / c.sum(1)[present]
    pe = float((c.sum(0) * c.sum(1)).sum()) / (n * n)
    kappa = (acc - pe) / (1 - pe) if pe < 1 else 0.0
    return float(acc), float(recall.mean()), float(kappa)


def boot_discrete(C, rng, n=N_BOOT):
    U = len(C)
    vals = np.zeros((n, 3))
    for b in range(n):
        vals[b] = pooled_from_conf(C[rng.integers(0, U, U)].sum(0))
    return np.percentile(vals, [2.5, 97.5], axis=0).T  # 3 x 2


def boot_mean(x, units, rng, n=N_BOOT):
    valid = ~np.isnan(x)
    x, units = x[valid], units[valid]
    if not len(x):
        return None
    uu = {u: i for i, u in enumerate(np.unique(units))}
    s = np.zeros(len(uu)); c = np.zeros(len(uu))
    for v, u in zip(x, units):
        s[uu[u]] += v; c[uu[u]] += 1
    vals = [s[p].sum() / c[p].sum() for p in
            (rng.integers(0, len(uu), len(uu)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5]).tolist()


def fit_folds(X, y, groups, split, rep, target=None):
    """Out-of-fold predictions. rep='bridge' predicts CLAP inside each fold."""
    if split == "random":
        folds = list(KFold(5, shuffle=True, random_state=SEED).split(X))
    elif split == "segment":
        folds = list(GroupKFold(5).split(X, y, groups["segment"]))
    elif split == "subject":
        folds = list(GroupKFold(5).split(X, y, groups["subject"]))
    else:
        raise ValueError(split)
    oof = np.zeros(len(y), np.int64)
    clap = np.zeros((len(y), target.shape[1]), np.float32) if target is not None else None
    for tr, te in folds:
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]).astype(np.float32), sc.transform(X[te]).astype(np.float32)
        if rep == "bridge":
            rg = Ridge(alpha=1e3).fit(Xtr, target[tr])
            Ftr, Fte = rg.predict(Xtr), rg.predict(Xte)
            if clap is not None:
                clap[te] = Fte
        else:
            Ftr, Fte = Xtr, Xte
            if target is not None:
                clap[te] = Ridge(alpha=1e3).fit(Xtr, target[tr]).predict(Xte)
        oof[te] = LogisticRegression(max_iter=2000, C=1.0).fit(Ftr, y[tr]).predict(Fte)
    return oof, clap, folds


def continuous_metrics(clap_pred, target_unit, song, group_arr, folds, allow_within):
    n = len(song)
    cos_t = np.zeros(n); cos_c = np.zeros(n); cos_g = np.zeros(n); cos_r = np.zeros(n)
    hit = np.full(n, np.nan); chance = np.full(n, np.nan)
    sse = np.zeros(n); sst = np.zeros(n)
    for tr, te in folds:
        mean_tr = target_unit[tr].mean(0)
        g = unit(mean_tr)
        cent = {s: unit(target_unit[tr][song[tr] == s].mean(0)) for s in np.unique(song[tr])}
        P = unit(clap_pred[te])
        cos_t[te] = (P * target_unit[te]).sum(1)
        cos_c[te] = np.array([P[i] @ cent[song[r]] for i, r in enumerate(te)])
        cos_g[te] = target_unit[te] @ g                      # trivial global-mean baseline
        c = np.stack([cent[song[r]] for r in te])            # own-song centroid per test row
        Pr = P - (P * c).sum(1, keepdims=True) * c
        Tr = target_unit[te] - (target_unit[te] * c).sum(1, keepdims=True) * c
        cos_r[te] = (unit(Pr) * unit(Tr)).sum(1)            # within-song residual alignment
        sse[te] = ((target_unit[te] - clap_pred[te]) ** 2).sum(1)
        sst[te] = ((target_unit[te] - mean_tr) ** 2).sum(1)
        if not allow_within:
            continue
        seg_ids = np.unique(group_arr[tr])
        seg_mat = np.stack([unit(target_unit[tr][group_arr[tr] == gid].mean(0)) for gid in seg_ids])
        seg_song = np.array([song[tr][group_arr[tr] == gid][0] for gid in seg_ids])
        for s in np.unique(song[te]):
            ci = np.where(seg_song == s)[0]
            if not len(ci):
                continue
            mask = song[te] == s
            sc = P[mask] @ seg_mat[ci].T
            hit[te[mask]] = (seg_ids[ci[sc.argmax(1)]] == group_arr[te[mask]])
            chance[te[mask]] = 1.0 / len(ci)
    return dict(cos_target=cos_t, cos_centroid=cos_c, cos_global=cos_g, cos_residual=cos_r,
                hit=hit, chance=chance, r2=float(1 - sse.sum() / sst.sum()))


def discrete_block(X, y, groups, split, rep, k, target=None):
    oof, clap, folds = fit_folds(X, y, groups, split, rep, target)
    C = conf_stats(y, oof, groups["unit_key"], k)
    acc, bal, kap = pooled_from_conf(C.sum(0))
    ci = boot_discrete(C, np.random.default_rng(SEED))
    counts = np.bincount(y, minlength=k)
    return dict(acc=acc, bal_acc=bal, kappa=kap,
                acc_ci=ci[0].tolist(), bal_ci=ci[1].tolist(), kappa_ci=ci[2].tolist(),
                chance=1.0 / k, majority=float(counts.max() / counts.sum()), n=int(len(y)),
                clap_oof=clap, folds=folds)


def unit_key(split, subj, grp):
    return subj if split == "subject" else (grp if split == "segment" else np.arange(len(subj)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_rows()
    song, ds, subj, grp = d["song"], d["dataset"], d["subject"], d["group"]
    tgtu = d["target_unit"]
    reps = {"bandpower": d["bandpower"], "labram": d["labram"],
            "aldm": d["aldm"], "bridge": d["aldm"]}

    res = {
        "coverage": {
            "rows": int(len(song)), "subjects": int(len(np.unique(subj))),
            "songs": int(len(np.unique(song))), "segments": int(len(np.unique(grp))),
            "nmedt": {"rows": int((ds == 0).sum()), "subjects": int(len(np.unique(subj[ds == 0]))),
                      "songs": int(len(np.unique(song[ds == 0])))},
            "nmedh": {"rows": int((ds == 1).sum()), "subjects": int(len(np.unique(subj[ds == 1]))),
                      "songs": int(len(np.unique(song[ds == 1])))},
        },
        "label_validity": {
            "oracle_genre_distribution_all": np.bincount(d["oracle"], minlength=12).tolist(),
            "oracle_genre_distribution_nmedt": np.bincount(d["oracle"][ds == 0], minlength=12).tolist(),
            "oracle_genre_majority_all": float(np.bincount(d["oracle"], minlength=12).max() / len(d["oracle"])),
            "oracle_genre_majority_nmedt": float(np.bincount(d["oracle"][ds == 0], minlength=12).max() / (ds == 0).sum()),
            "note": "the 12-class oracle collapses (one class holds ~70% of NMED-T); it is a "
                    "label-validity control, not a genre rung. Raw NMED-T audio is not "
                    "available here, so tempo/energy rungs are not computed.",
        },
        "nmedt": {}, "nmedh": {}, "style_family": {},
    }

    # ---------------- NMED-T ladder (crossed design) ----------------
    m = ds == 0
    song_t, grp_t, subj_t = song[m], grp[m], subj[m]
    k = len(np.unique(song_t))
    print(f"NMED-T: {m.sum()} rows, {len(np.unique(subj_t))} subjects, {k} songs")
    for split in ("random", "segment", "subject"):
        uk = unit_key(split, subj_t, grp_t)
        groups = {"subject": subj_t, "segment": grp_t, "unit_key": uk}
        res["nmedt"][f"split_{split}"] = {}
        for name, X in reps.items():
            b = discrete_block(X[m], song_t, groups, split, name, k, tgtu[m])
            cont = continuous_metrics(b["clap_oof"], tgtu[m], song_t, grp_t, b["folds"],
                                      allow_within=(split != "segment"))
            rng = np.random.default_rng(SEED)
            excess = cont["cos_target"] - cont["cos_global"]
            res["nmedt"][f"split_{split}"][name] = {
                "song": {kk: v for kk, v in b.items() if kk not in ("clap_oof", "folds")},
                "continuous": {
                    "cos_target": float(np.nanmean(cont["cos_target"])),
                    "cos_target_ci": boot_mean(cont["cos_target"], uk, rng),
                    "cos_global_mean_baseline": float(np.nanmean(cont["cos_global"])),
                    "cos_global_mean_baseline_ci": boot_mean(cont["cos_global"], uk, rng),
                    "excess_cos_over_trivial": float(np.nanmean(excess)),
                    "excess_cos_over_trivial_ci": boot_mean(excess, uk, rng),
                    "cos_residual_within_song": float(np.nanmean(cont["cos_residual"])),
                    "cos_residual_within_song_ci": boot_mean(cont["cos_residual"], uk, rng),
                    "cos_song_centroid_shrinkage_inflated": float(np.nanmean(cont["cos_centroid"])),
                    "within_song_top1": (None if not np.isfinite(cont["hit"]).any()
                                         else float(np.nanmean(cont["hit"]))),
                    "within_song_chance": (None if not np.isfinite(cont["chance"]).any()
                                           else float(np.nanmean(cont["chance"]))),
                    "within_song_top1_ci": boot_mean(cont["hit"], uk, rng),
                    "r2": cont["r2"],
                },
            }
            c = res["nmedt"][f"split_{split}"][name]["continuous"]
            print(f"  [{split:7}] {name:9} song acc {b['acc']:.3f} bal {b['bal_acc']:.3f} "
                  f"kappa {b['kappa']:.3f} | cos {c['cos_target']:.3f} "
                  f"(trivial {c['cos_global_mean_baseline']:.3f}, "
                  f"excess {c['excess_cos_over_trivial']:+.3f}) "
                  f"resid {c['cos_residual_within_song']:.3f} "
                  f"within {c['within_song_top1'] if c['within_song_top1'] is not None else float('nan'):.3f} "
                  f"(ch {c['within_song_chance'] if c['within_song_chance'] is not None else float('nan'):.3f})")

    # audio-side ceiling for the song label (validates the label under every split)
    res["nmedt"]["audio_clap_ceiling"] = {}
    for split in ("random", "segment", "subject"):
        groups = {"subject": subj_t, "segment": grp_t,
                  "unit_key": unit_key(split, subj_t, grp_t)}
        b = discrete_block(tgtu[m], song_t, groups, split, "audio", k, None)
        res["nmedt"]["audio_clap_ceiling"][split] = {
            kk: v for kk, v in b.items() if kk not in ("clap_oof", "folds")}
        print(f"  [{split:7}] audio ceiling song acc {b['acc']:.3f}")

    # ---------------- NMED-H (subject/song confounded; separate) ----------------
    h = ds == 1
    song_h, grp_h, subj_h = song[h] - song[h].min(), grp[h], subj[h]
    kh = len(np.unique(song_h))
    for split in ("segment", "subject"):
        groups = {"subject": subj_h, "segment": grp_h,
                  "unit_key": unit_key(split, subj_h, grp_h)}
        res["nmedh"][f"split_{split}"] = {}
        for name, X in reps.items():
            b = discrete_block(X[h], song_h, groups, split, name, kh, tgtu[h])
            res["nmedh"][f"split_{split}"][name] = {
                "song": {kk: v for kk, v in b.items() if kk not in ("clap_oof", "folds")}}
            print(f"  NMED-H [{split:7}] {name:9} song acc {b['acc']:.3f} "
                  f"bal {b['bal_acc']:.3f} kappa {b['kappa']:.3f} (chance 0.25)")

    # ---------------- style family (2-way), segment-held-out only ----------------
    groups_s = {"subject": subj, "segment": grp, "unit_key": grp}
    for name, X in reps.items():
        b = discrete_block(X, ds, groups_s, "segment", name, 2, tgtu)
        res["style_family"][name] = {
            "segment_heldout": {kk: v for kk, v in b.items() if kk not in ("clap_oof", "folds")}}
        print(f"  style family {name:9} acc {b['acc']:.3f} bal {b['bal_acc']:.3f} "
              f"kappa {b['kappa']:.3f} (chance 0.5, majority {b['majority']:.3f})")

    # ---------------- controls and representation caveats ----------------
    sid = json.load(open(ROOT.parent / "mainv2" / "results" / "subject_id_probe_result.json",
                         encoding="utf-8"))
    res["subject_identity_control"] = {
        "source": "mainv2/results/subject_id_probe_result.json (block 70/10/20 split, 2058 test rows)",
        "task": "68-way subject identification; windows are shared across subjects, so the audio-target control is the meaningful zero",
        "bandpower": sid["A_subject_68way"]["bandpower"]["accuracy"],
        "labram_mean_pooled_200": sid["A_subject_68way"]["labram200"]["accuracy"],
        "aligned_embedding": sid["A_subject_68way"]["aligned_eeg_embedding"]["accuracy"],
        "audio_target_control": sid["A_subject_68way"]["audio_target_stimulus_control"]["accuracy"],
        "read": "subject identity is nearly saturated in bandpower/LaBraM and still 0.67 in the aligned "
                "embedding; any split that keeps subjects in train can exploit it.",
    }
    res["representation_notes"] = {
        "bandpower": "frozen 1804-d band-power features.",
        "labram": "frozen LaBraM-Base, 200-d mean-pooled patches (a compressed representation, not the full token grid).",
        "aldm": "stage-1 trained alignment head output (512-d). Trained on ALL rows, so its subject-held-out "
                "numbers are transductive/optimistic; the frozen rows are the clean cross-subject comparison.",
        "bridge": "EEG-ALDM -> audio-CLAP ridge refit inside each training fold; inherits aldm's training.",
        "audio_clap_ceiling": "audio-CLAP -> song label probe; validates the label but is not an EEG result.",
        "style_family": "NMED-T vs NMED-H is a batch/song-set confound (no song is shared between batches), "
                        "so 1.0 is not evidence of style decoding.",
    }
    res["protocol"] = {
        "nmedt": "20 subjects x 10 songs, fully crossed; subject-held-out is a real "
                 "cross-listener test.",
        "nmedh": "48 subjects x 1 song each; subject and song are confounded by design, "
                 "so it is never pooled into a held-out-subject claim.",
        "bridge": "EEG-ALDM -> audio-CLAP ridge refit inside every training fold (inductive).",
        "genre_oracle": "not used as a level (see label_validity).",
        "within_song": "only defined for random/subject splits, where the true segment "
                       "exists in the training centroids; undefined for segment-held-out.",
        "metrics": "discrete: acc/balanced acc/Cohen kappa vs chance+majority; continuous: "
                   "cosine to true CLAP, cosine to train-only song centroid, within-song "
                   "segment top-1, pooled R^2; 95% CI by bootstrapping the held-out unit.",
    }
    json.dump(res, open(OUT / "info_ladder.json", "w", encoding="utf-8"), indent=2)
    print("\nwrote", OUT / "info_ladder.json")


if __name__ == "__main__":
    main()
