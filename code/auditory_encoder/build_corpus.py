"""Build a unified 19-channel / 200 Hz / 8 s cross-subject auditory corpus.

Sources:
  local75 (existing windows.npy + meta.npz)
  CIRE   (122-ch, 256 Hz, 240 x 5 s epochs per subject, trial order fixed)
  Alice  (64-ch continuous audiobook, optional once extracted)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = ROOT.parent / "mainv2"
STD19 = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
         "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
CIRE_ALIAS = {"Pz": "POz", "O1": "OI1", "O2": "OI2"}


def zscore(x):
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    return np.clip((x - mu) / sd, -10, 10)


def load_local75():
    w = np.load(M2 / "data" / "local75" / "windows.npy").astype(np.float32)
    m = np.load(M2 / "data" / "local75" / "meta.npz", allow_pickle=True)
    out = np.empty_like(w, dtype=np.float16)
    for i in range(len(w)):
        out[i] = zscore(w[i]).astype(np.float16)
    meta = {"subject": np.array([str(x) for x in m["subject"]]),
            "stimulus": np.array([str(x) for x in m["stim"]]),
            "dataset": np.array([str(x) for x in m["dataset"]]),
            "win": m["win"].astype(np.int64)}
    return out, meta


def cire_labels(chanlocs):
    labels = []
    for i in range(chanlocs.shape[1]):
        a = np.asarray(chanlocs[0, i]["labels"])
        s = "".join(a.ravel().tolist()) if a.dtype.kind in "US" else \
            "".join(chr(int(x)) for x in a.ravel() if int(x) > 0)
        labels.append(s)
    return labels


def cire_channel_idx(labels):
    low = {x.lower(): i for i, x in enumerate(labels)}
    idx = []
    for c in STD19:
        key = c.lower()
        if key not in low and c in CIRE_ALIAS:
            key = CIRE_ALIAS[c].lower()
        if key not in low:
            raise KeyError(f"CIRE missing channel {c} (and alias), have {labels}")
        idx.append(low[key])
    return idx


def load_cire(cire_dir, limit=None):
    files = sorted(Path(cire_dir).glob("sub-*_task-CIRE_eeg.set"))
    if limit:
        files = files[:limit]
    wins, subjects, stimuli, ds, winidx = [], [], [], [], []
    for p in files:
        d = sio.loadmat(p, variable_names=["data", "chanlocs", "srate"], chars_as_strings=False)
        data = d["data"].astype(np.float32)          # [ch, time, trials]
        srate = int(np.asarray(d["srate"]).ravel()[0])
        labels = cire_labels(d["chanlocs"])
        idx = cire_channel_idx(labels)
        n_tr = data.shape[2]
        if n_tr != 240:
            print(f"  skip {p.name}: {n_tr} trials (incomplete; order would shift)", flush=True)
            continue
        for t in range(n_tr):
            seg = data[idx, :, t]                    # [19, 1280]
            if srate != 200:
                seg = resample_poly(seg, 200, srate, axis=1)
            if seg.shape[1] < 1600:
                seg = np.pad(seg, ((0, 0), (0, 1600 - seg.shape[1])), mode="reflect")
            else:
                seg = seg[:, :1600]
            wins.append(zscore(seg).astype(np.float16))
            subjects.append(f"CIRE:{p.name[:6]}")
            stimuli.append(f"cire:{t:03d}")
            ds.append("cire")
            winidx.append(t)
        print(f"  {p.name}: {n_tr} trials, {data.shape[0]} ch, {srate} Hz", flush=True)
    return (np.stack(wins), {"subject": np.array(subjects), "stimulus": np.array(stimuli),
                             "dataset": np.array(ds), "win": np.array(winidx)})


def load_alice(alice_dir, max_per_subject=300):
    import mne
    files = sorted(Path(alice_dir).rglob("*.fif"))
    if not files:
        print("no Alice .fif found, skip", flush=True)
        return None, None
    std = mne.channels.make_standard_montage("standard_1005").get_positions()["ch_pos"]
    wins, subjects, stimuli, ds, winidx = [], [], [], [], []
    for p in files:
        raw = mne.io.read_raw_fif(p, preload=True, verbose="ERROR")
        pos = raw.get_montage().get_positions()["ch_pos"]
        picks = []
        for c in STD19:
            target = np.asarray(std[c])
            best = min(pos, key=lambda k: np.linalg.norm(np.asarray(pos[k]) - target))
            picks.append(raw.ch_names.index(best))
        x = raw.get_data(picks=picks)
        if raw.info["sfreq"] != 200:
            x = resample_poly(x.astype(np.float32), 200, int(round(raw.info["sfreq"])), axis=1)
        ann = sorted((float(a["onset"]), str(a["description"]))
                     for a in raw.annotations if str(a["description"]).isdigit())
        n = 0
        for i, (onset, label) in enumerate(ann):
            end = ann[i + 1][0] if i + 1 < len(ann) else onset + 56
            end = min(end, onset + 56)
            a = int(onset * 200); b = int(end * 200)
            for s in range(a, b - 1600 + 1, 400):
                wins.append(zscore(x[:, s:s + 1600]).astype(np.float16))
                subjects.append(f"ALICE:{p.stem.split('_')[0]}")
                stimuli.append(f"alice:{label}:{(s - a) // 200}")
                ds.append("alice")
                winidx.append((s - a) // 200)
                n += 1
                if n >= max_per_subject:
                    break
            if n >= max_per_subject:
                break
        print(f"  {p.name}: {n} windows, {len(picks)} ch, {raw.info['sfreq']} Hz", flush=True)
    return (np.stack(wins), {"subject": np.array(subjects), "stimulus": np.array(stimuli),
                             "dataset": np.array(ds), "win": np.array(winidx)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(M2 / "data" / "auditory_corpus"))
    ap.add_argument("--cire-dir", default="D:/Projects/CLaBraM/data_external/cire")
    ap.add_argument("--alice-dir", default="D:/Projects/CLaBraM/data_external/alice")
    ap.add_argument("--no-local75", action="store_true")
    ap.add_argument("--no-alice", action="store_true")
    ap.add_argument("--cire-limit", type=int, default=0)
    ap.add_argument("--alice-max", type=int, default=300)
    args = ap.parse_args()

    parts = []
    if not args.no_local75:
        w, m = load_local75()
        parts.append((w, m))
        print("local75", w.shape, flush=True)
    w, m = load_cire(args.cire_dir, args.cire_limit or None)
    parts.append((w, m))
    print("cire", w.shape, flush=True)
    if not args.no_alice:
        w, m = load_alice(args.alice_dir, args.alice_max)
        if w is not None:
            parts.append((w, m))
            print("alice", w.shape, flush=True)

    W = np.concatenate([p[0] for p in parts])
    meta = {k: np.concatenate([p[1][k] for p in parts]) for k in ("subject", "stimulus", "dataset", "win")}
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "windows.npy", W)
    np.savez(out / "meta.npz", **meta)
    stats = {d: int((meta["dataset"] == d).sum()) for d in np.unique(meta["dataset"])}
    summary = {"windows": int(len(W)), "subjects": int(len(np.unique(meta["subject"]))),
               "stimuli": int(len(np.unique(meta["stimulus"]))),
               "by_dataset": stats}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
