"""Qwen-prompt pilot: genre-conditioned generation vs song-level description selection (pre-registered).

Pre-registration: docs/prereg_qwen_two_paths_2026-10-22.md
Same 16 windows and fixed seeds as generation.json; frozen MusicGen-small.
Six arms:
  genre_oracle / genre_clf / genre_majority
  song_oracle / song_calib / song_samegenre

Run via launch_guard.py (tag qwen_two_paths).
"""
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from calib_head_pilot import GATE, OUT as HEAD_OUT, unit_np  # noqa: E402
from calib_head_5fold import FOLDS  # noqa: E402
from calib_head_genre import embed, fit_rotation, load_head  # noqa: E402
from desc_generate import spectral_metrics  # noqa: E402

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = ROOT.parent / "mainv2"
DESC = ROOT / "results" / "descriptions"
GEN = DESC / "gen_qwen_pilot"
SR = 32000
SEEDS = (20261021, 20261022)
ARMS = ["genre_oracle", "genre_clf", "genre_majority",
        "song_oracle", "song_calib", "song_samegenre"]


def main():
    GEN.mkdir(parents=True, exist_ok=True)
    gen = json.load(open(DESC / "generation.json", encoding="utf-8"))
    rows = sorted({r["row"] for r in gen["records"]})
    true_song_of = {r["row"]: r["true_song"] for r in gen["records"]}
    gp = json.load(open(DESC / "genre_prompts_qwen.json", encoding="utf-8"))
    genre_text = [p["qwen_prompt"] for p in gp["prompts"]]
    Dg = unit_np(np.array([p["embedding"] for p in gp["prompts"]], dtype=np.float32))
    cs = json.load(open(DESC / "captions_qwen_song.json", encoding="utf-8"))
    song_text = [s["medoid_caption"] for s in cs["songs"]]
    Ds = unit_np(np.array([s["medoid_embedding"] for s in cs["songs"]], dtype=np.float32))
    # same-genre control: most similar other song caption by text embedding
    sim = Ds @ Ds.T
    samegenre = {}
    for s in range(14):
        sim[s, s] = -9
        samegenre[s] = int(np.argmax(sim[s]))

    z = np.load(GATE, allow_pickle=True)
    cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)
    codes = np.load(ROOT / "data" / "eegcond" / "codes.npy").astype(np.int64)
    aldm = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    w = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu", weights_only=True)
    ca = unit_np(aldm @ w["weight"].numpy().astype(np.float32).T + w["bias"].numpy().astype(np.float32))
    key = {ca[i].tobytes(): i for i in range(len(ca))}
    idx = np.array([key[cond[j].tobytes()] for j in range(len(cond))])
    X = torch.tensor(z["bandpower"][idx].astype(np.float32))
    X = (X - X.mean(0)) / (X.std(0) + 1e-6)
    T = unit_np(np.load(ROOT / "results" / "prompt_classifier" / "true_emb_all.npy").astype(np.float32))
    yg = np.load(ROOT / "results" / "prompt_classifier" / "true_oracle_all.npy").astype(np.int64)
    song = z["song"][idx].astype(int); subj = z["subject"][idx].astype(str)
    seg = z["seg2"][idx].astype(int); ds = z["dataset"][idx].astype(int)
    al = z["aligned8"][idx].astype(bool)

    f = 0
    meta = FOLDS[f]
    train_subs = [s for s in np.unique(subj)
                  if s not in meta["test"] and s not in meta["val"] and s in set(subj[ds == 0])]
    train_rows = np.where((ds == 0) & np.isin(subj, train_subs))[0]
    Pn = unit_np(np.stack([T[train_rows][song[train_rows] == c].mean(0) for c in range(10)]))
    head = load_head(HEAD_OUT / f"calib_bandpower_laion_f{f}.pt", X.shape[1])
    Z = embed(head, X)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(cond[train_rows], yg[train_rows])
    maj_genre = int(np.bincount(yg[train_rows]).argmax())
    print("majority genre", maj_genre, "train rows", len(train_rows), flush=True)

    rng = np.random.default_rng(f)
    q_by_row = {}
    for s in meta["test"]:
        rr = np.where(al & (subj == s))[0]
        segs = np.unique(seg[rr]); rng.shuffle(segs)
        fg = set(segs[:len(segs) // 2])
        rf = rr[np.isin(seg[rr], list(fg))]
        rt = rr[~np.isin(seg[rr], list(fg))]
        if len(rf) < 100 or len(rt) < 100:
            continue
        R, m = fit_rotation(Z, rf, song, Pn, k=100)
        q = (Z[rt] - m) @ R
        for j, r in enumerate(rt):
            q_by_row[int(r)] = q[j]
    missing = [r for r in rows if r not in q_by_row]
    print("rows missing from fold-0 rt:", missing, flush=True)
    if missing:
        rows = [r for r in rows if r in q_by_row]

    sys.path.insert(0, str(M2 / "src" / "generate"))
    sys.path.insert(0, str(ROOT / "code" / "shared"))
    sys.path.insert(0, str(ROOT / "code" / "radii"))
    from musicgen_clap_finetune import MusicGenCLAP, load_clap, clap_embed_audio
    from arms_2x2 import embed_text, gen_from_enc

    base = MusicGenCLAP(device="cuda", use_lora=False); mg = base.mg; mg.eval()
    phase = {}
    for text in set(genre_text + song_text):
        phase[text] = embed_text(mg, text, "cuda")

    records = []
    for row in rows:
        true_song = true_song_of[row]
        true_genre = int(yg[row])
        clf_genre = int(clf.predict(cond[row:row + 1])[0])
        calib_song = int(np.argmax(q_by_row[row] @ Ds.T))
        prompts = {
            "genre_oracle": (genre_text[true_genre], true_genre),
            "genre_clf": (genre_text[clf_genre], clf_genre),
            "genre_majority": (genre_text[maj_genre], maj_genre),
            "song_oracle": (song_text[true_song], true_song),
            "song_calib": (song_text[calib_song], calib_song),
            "song_samegenre": (song_text[samegenre[true_song]], samegenre[true_song]),
        }
        for seed in SEEDS:
            for arm in ARMS:
                text, sel_song_or_genre = prompts[arm]
                wav = np.asarray(gen_from_enc(mg, phase[text].unsqueeze(0), 400, 0.9, None,
                                              seed=seed), np.float32).reshape(-1)
                sf.write(GEN / f"{arm}_row{row}_s{seed}.wav", wav, SR)
                records.append({"row": row, "true_song": true_song, "true_genre": true_genre,
                                "clf_genre": clf_genre, "majority_genre": maj_genre,
                                "calib_song": calib_song, "arm": arm, "seed": seed,
                                "prompt": text, "selected": sel_song_or_genre})
        print(f"row {row} song {true_song} true_genre {true_genre} clf {clf_genre} "
              f"calib_song {calib_song} done", flush=True)
    json.dump({"protocol": {"fold": f, "rows": rows, "seeds": list(SEEDS),
                            "majority_genre": maj_genre,
                            "samegenre_map": {str(k): v for k, v in samegenre.items()}},
               "records": records},
              open(DESC / "generation_qwen_pilot.json", "w", encoding="utf-8"), indent=2)
    print("wrote generation_qwen_pilot.json", flush=True)

    # --- analysis ---
    clap, proc, _ = load_clap("cuda")
    W = unit_np(np.stack([T[r] for r in rows]))
    song_of = np.array([true_song_of[r] for r in rows])
    songs_present = sorted(set(song_of.tolist()))
    S = unit_np(np.stack([W[song_of == s].mean(0) for s in songs_present]))
    sims_song = W @ W.T
    iu = np.triu_indices(len(rows), 1)
    same_mask = song_of[:, None] == song_of[None, :]
    out = []
    for rec in records:
        r = rec["row"]; arm = rec["arm"]; seed = rec["seed"]
        wav, sr = sf.read(GEN / f"{arm}_row{r}_s{seed}.wav", dtype="float32")
        e = unit_np(np.asarray(clap_embed_audio(clap, proc, sr, wav, "cuda"), dtype=np.float32))
        sg = e @ Dg.T
        ss = e @ S.T
        i = rows.index(r)
        own = songs_present.index(rec["true_song"])
        sm = float(ss[own] - np.delete(ss, own).mean())
        is_genre = arm.startswith("genre")
        out.append({**rec, "cos_to_true_genre": float(e @ Dg[rec["true_genre"]]),
                    "cos_to_selected_genre": float(e @ Dg[rec["selected"]]) if is_genre else None,
                    "genre_argmax": int(sg.argmax()),
                    "genre_acc_true": int(int(sg.argmax()) == rec["true_genre"]),
                    "genre_acc_selected": int(int(sg.argmax()) == rec["selected"]) if is_genre else None,
                    "song_cen_own": float(ss[own]), "song_cen_margin": sm,
                    "song_rank": int((ss > ss[own]).sum()) + 1,
                    "clap_to_stimulus": float(e @ T[r])})
        if len(out) % 24 == 0:
            print("analyzed", len(out), flush=True)
    json.dump({"protocol": {"rows": rows, "seeds": list(SEEDS),
                            "song_centroid_pair_cos": float(sims_song[iu][~same_mask[iu]].mean())},
               "records": out},
              open(DESC / "pilot_qwen_two_paths.json", "w", encoding="utf-8"), indent=2)

    # aggregates + paired tests
    def vals(arm, key):
        d = {}
        for rec in out:
            if rec["arm"] == arm:
                d.setdefault(rec["row"], []).append(rec[key])
        return np.array([np.mean(d[r]) for r in rows])
    print("\n=== pilot aggregates (mean over rows, 2 seeds) ===")
    for arm in ARMS:
        print(f"{arm:14} cos_true_genre {vals(arm,'cos_to_true_genre').mean():+.3f} "
              f"acc_true {vals(arm,'genre_acc_true').mean():.3f} "
              f"song_margin {vals(arm,'song_cen_margin').mean():+.3f} "
              f"rank1 {int((vals(arm,'song_rank')==1).sum())}/{len(rows)}")
    pairs = [("genre_oracle", "genre_majority", "cos_to_true_genre"),
             ("genre_clf", "genre_majority", "cos_to_true_genre"),
             ("genre_oracle", "genre_majority", "genre_acc_true"),
             ("genre_clf", "genre_majority", "genre_acc_true"),
             ("song_oracle", "song_samegenre", "song_cen_margin"),
             ("song_calib", "song_samegenre", "song_cen_margin"),
             ("song_oracle", "song_calib", "song_cen_margin")]
    for a, b, key in pairs:
        d = vals(a, key) - vals(b, key)
        se = d.std(ddof=1) / np.sqrt(len(d))
        print(f"{a} - {b} [{key}]: {d.mean():+.3f}  SE {se:.3f}  t {d.mean()/se:+.2f}")
    print("wrote pilot_qwen_two_paths.json")


if __name__ == "__main__":
    main()
