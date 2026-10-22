"""
【做 1】LaBraM on NMED-T — does a pretrained encoder close the 3.6x gap?

`gate_nmedt_50way.py` gave 50-way 0.1061 (band power + MLP) against the prior art's
LaBraM 0.380 / channel-oriented 0.487.  The remaining explanations are the ENCODER
and the DATA VOLUME.  This script isolates the encoder: same dataset, same windows,
same protocol, same head, same evaluation — only the front-end changes.

Front-ends compared on identical rows:
  band_power : 82 ch x 22 bands (2 Hz bins, 1-45 Hz), log10     [the current best]
  labram     : LaBraM-Base frozen -> 200-d mean-pooled patches

Setup
  channels : 82 EGI electrodes with UNIQUE standard_1020 slots (nmedt_channel_map.py)
  window   : 8 s (= 8 LaBraM patches of 200 samples @ 200 Hz, LaBraM's native input)
  stride   : 8 s (non-overlapping)
  target   : CLAP embedding of the SAME 8 s of audio (deterministic harness)
  split    : random by WINDOW, subjects shared (standard protocol)
  metric   : 50-way in-batch top-1 + shuffled-pairing control

The two front-ends share the same rows, so the comparison is exact.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import spectrogram

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
# ---- mainv2 workspace bootstrap (generated) -------------------------------
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent            # src/<group>/
ROOT = HERE.parents[1]                            # mainv2/
DATA = Path(os.environ.get("MAINV2_DATA", ROOT / "data"))      # read-only caches
RESULTS = Path(os.environ.get("MAINV2_RESULTS", ROOT / "results"))  # result json + prereg json
for _g in ("data_pipeline", "encoder", "audio_target", "align", "measurement",
           "control", "reporting", "evaluation"):
    _p = str(ROOT / "src" / _g)
    if _p not in sys.path:
        sys.path.insert(0, _p)
from clap_eval import ClapEvaluator          # noqa: E402
from encoder_labram import LabramEncoder    # noqa: E402

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NMEDT = Path(os.environ.get("NMEDT_ROOT", "data/NMED-T"))
WIN, STRIDE, FS = 8.0, 8.0, 200.0
BANDS = [(lo, lo + 2) for lo in range(1, 45, 2)]
NPZ = DATA / "gate_nmedt_labram.npz"
OUT = RESULTS / "gate_nmedt_labram_result.json"
SEEDS = [0, 1, 2, 3, 4]
N_DRAWS = 3000


def load_eeg_200(path):
    """NMED-T .mat -> [nch, T] at 200 Hz, band-limited, chunked."""
    import h5py
    with h5py.File(str(path), "r") as h:
        X = h["X"]
        T, C = X.shape
        from scipy.signal import butter, sosfiltfilt, resample_poly
        sos = butter(4, [0.5, 45.0], "band", fs=1000.0, output="sos")
        step = 200_000
        out = []
        for a in range(0, T, step):
            b = min(a + step, T)
            blk = np.asarray(X[a:b, :], dtype=np.float64)
            if blk.shape[0] < 200:
                break
            blk = sosfiltfilt(sos, blk, axis=0)
            blk = resample_poly(blk, 1, 5, axis=0)          # 1000 -> 200 Hz
            out.append(blk.T.astype(np.float32))
        return np.concatenate(out, axis=1)


def band_features(x, fs=FS):
    f, tt, Sxx = spectrogram(x, fs=fs, nperseg=int(fs), noverlap=0,
                             axis=-1, scaling="density", mode="psd")
    out = []
    for lo, hi in BANDS:
        m = (f >= lo) & (f < hi)
        if not m.any():
            raise ValueError(f"band {lo}-{hi} has no bins")
        out.append(np.log10(np.maximum(Sxx[:, m, :].mean(1), 1e-12)))
    return np.concatenate(out, axis=0)          # [C*nband, T_frames @1 Hz]


def build():
    if NPZ.exists():
        d = np.load(NPZ, allow_pickle=True)
        print(f"  cached: rows {len(d['song'])} labram {d['labram'].shape} "
              f"bp {d['bandpower'].shape}")
        return d
    cm = json.loads((RESULTS / "nmedt_egi_to_1020.json").read_text(encoding="utf-8"))
    order = cm["order"]
    egi_idx = [int(e[1:]) - 1 for e in order]           # E1 -> column 0
    print(f"  {len(order)} EGI channels selected")

    de = json.loads((NMEDT / "_din_events.json").read_text(encoding="utf-8"))
    sm = json.loads((NMEDT / "song_mapping.json").read_text(encoding="utf-8"))
    song_file = {int(k): NMEDT / "Music" / v["file"] for k, v in sm.items()}

    ev = ClapEvaluator()
    proto, winemb = {}, {}
    t0 = time.time()
    for idx, p in sorted(song_file.items()):
        try:
            wav = ev.load(str(p))
        except Exception:
            print(f"    song {idx} unreadable, skipped")
            continue
        dur = len(wav) / ev.sr
        proto[idx] = ev.embed(wav, mode="auto")
        for k in range(int(dur // WIN)):
            a, b = int(k * WIN * ev.sr), int((k * WIN + WIN) * ev.sr)
            winemb[(idx, k)] = ev.embed(wav[a:b], mode="window")
        print(f"    song {idx}: {int(dur//WIN)} windows of {WIN:g}s  "
              f"[{time.time()-t0:5.0f}s]")
    print(f"  CLAP: {len(proto)} songs, {len(winemb)} windows  [{time.time()-t0:.0f}s]")

    enc = LabramEncoder(device=str(DEV), freeze=True, strict=True, norm="div100",
                        input_chans=cm["input_chans"])
    print(f"  LaBraM load: missing={len(enc.load_report[0])} "
          f"unexpected={len(enc.load_report[1])}  channels={enc.n_ch}")

    UV, BP, TAR, WEM, SONG, SUB, RUN, SEG = [], [], [], [], [], [], [], []
    t0 = time.time()
    for sub in sorted(set(k.split("_")[0] for k in de)):
        for part in ("1", "2"):
            key = f"{sub}_{part}"
            x_all = load_eeg_200(NMEDT / f"{key}_raw.mat")
            x = x_all[egi_idx]                       # 82 channels, LaBraM order
            bp = band_features(x)                    # [82*nband, 1 Hz frames]
            fr = 1.0                                 # 200-sample hop at 200 Hz
            tt = np.arange(bp.shape[1]) / fr
            n_win = 0
            for e in de[key]["stimulus_triggers"]:
                if not (21 <= e["trigger"] <= 30):
                    continue
                idx = e["trigger"] - 21
                if idx not in proto:
                    continue
                a_s = e["t_sec"]
                for k in sorted(kk for (s_, kk) in winemb if s_ == idx):
                    lo, hi = a_s + k * WIN, a_s + k * WIN + WIN
                    m = (tt >= lo) & (tt < hi)
                    if m.sum() < 1:
                        break
                    a = int(round(lo * FS))
                    b = a + int(WIN * FS)
                    if b > x.shape[1]:
                        break
                    UV.append(x[:, a:b])
                    BP.append(bp[:, m].mean(1))
                    TAR.append(winemb[(idx, k)])
                    WEM.append(winemb[(idx, k)])
                    SONG.append(idx)
                    SUB.append(sub)
                    RUN.append(key)
                    SEG.append(k)
                    n_win += 1
            print(f"    {key}: {n_win} windows (total {len(SONG)})  "
                  f"[{time.time()-t0:5.0f}s]")

    UV = np.stack(UV)
    print(f"  extracted {UV.shape} uV windows; running frozen LaBraM")
    lab = []
    with torch.no_grad():
        t1 = torch.tensor(UV) if False else None
        for i in range(0, len(UV), 128):
            lab.append(enc(torch.tensor(UV[i:i + 128])).cpu().numpy())
    lab = np.concatenate(lab).astype(np.float32)
    print(f"  LaBraM features {lab.shape}")
    order_songs = sorted(proto)
    np.savez_compressed(
        NPZ, labram=lab, bandpower=np.asarray(BP, np.float32),
        target=np.asarray(TAR, np.float32), winemb=np.asarray(WEM, np.float32),
        song=np.asarray(SONG, np.int32), subject=np.asarray(SUB),
        run=np.asarray(RUN), seg=np.asarray(SEG, np.int32),
        proto=np.stack([proto[s] for s in order_songs]).astype(np.float32),
        proto_songs=np.asarray(order_songs, np.int32))
    print(f"  wrote {NPZ.name}")
    return np.load(NPZ, allow_pickle=True)


def l2(x, eps=1e-8):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


def info_nce(z, t, tau=0.07):
    lg = z @ t.T / tau
    lab = torch.arange(z.shape[0], device=z.device)
    return 0.5 * (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab))


class Head(nn.Module):
    def __init__(self, d_in, d_out=512, h=256):
        super().__init__()
        self.n = nn.LayerNorm(d_in)
        self.f = nn.Sequential(nn.Linear(d_in, h), nn.GELU(), nn.Linear(h, d_out))

    def forward(self, x):
        return self.f(self.n(x))


@torch.no_grad()
def top1_inbatch(zq, tk, n_way, n_draws=N_DRAWS, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = zq.shape[0]
    hits = tot = 0
    for _ in range(n_draws):
        idx = torch.randperm(n, generator=g)[:n_way].to(zq.device)
        sims = zq[idx] @ tk[idx].T
        hits += int((sims.argmax(1) == torch.arange(len(idx), device=zq.device)).sum())
        tot += len(idx)
    return hits / tot


def run_front(tag, X, T, n, results):
    X = torch.tensor(X, device=DEV) if not torch.is_tensor(X) else X
    rs, cs = [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        pm = rng.permutation(n)
        n_tr, n_va = int(0.70 * n), int(0.15 * n)
        tr = torch.tensor(pm[:n_tr], device=DEV)
        va = torch.tensor(pm[n_tr:n_tr + n_va], device=DEV)
        te = torch.tensor(pm[n_tr + n_va:], device=DEV)
        torch.manual_seed(seed)
        m = Head(X.shape[1]).to(DEV)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        epochs, bs = 40, 512
        sch = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=1e-3, total_steps=max(epochs * max(len(tr) // bs, 1), 1))
        best, st = -1.0, None
        for ep in range(epochs):
            m.train()
            p = torch.randperm(len(tr), device=DEV)
            for i in range(0, len(tr) - bs + 1, bs):
                j = tr[p[i:i + bs]]
                loss = info_nce(l2(m(X[j])), l2(T[j]))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                opt.step()
                sch.step()
            if (ep + 1) % 10 == 0 or ep == epochs - 1:
                m.eval()
                with torch.no_grad():
                    v = top1_inbatch(l2(m(X[va])), l2(T[va]), 50, n_draws=300)
                if v > best:
                    best, st = v, {k: t.clone() for k, t in m.state_dict().items()}
        m.load_state_dict(st)
        m.eval()
        Tn = l2(T)
        with torch.no_grad():
            zq = l2(m(X[te]))
        Tsh = Tn[torch.randperm(n, device=DEV)][te]
        a = top1_inbatch(zq, Tn[te], 50, seed=seed)
        c = top1_inbatch(zq, Tsh, 50, seed=seed)
        rs.append(a); cs.append(c)
        print(f"    {tag} seed {seed}: val {best:.4f}  test 50-way {a:.4f}  "
              f"control {c:.4f}")
    a, c = np.array(rs), np.array(cs)
    print(f"  {tag:11s} 50-way real {a.mean():.4f} +- {a.std():.4f}   "
          f"control {c.mean():.4f}   chance 0.0200   {a.mean()/0.02:.2f}x")
    results[tag] = {"real_mean": float(a.mean()), "real_std": float(a.std()),
                    "control_mean": float(c.mean()),
                    "times_chance": float(a.mean() / 0.02)}


def main():
    d = build()
    n = len(d["song"])
    T = torch.tensor(d["target"], device=DEV)
    print(f"\n{n} rows | {len(set(d['song']))} songs | "
          f"labram {d['labram'].shape[1]}-d | bandpower {d['bandpower'].shape[1]}-d")
    results = {}
    print("=" * 96)
    run_front("band_power", d["bandpower"], T, n, results)
    run_front("labram", d["labram"], T, n, results)
    OUT.write_text(json.dumps({"window_s": WIN, "n_rows": int(n),
                               "seeds": SEEDS, "results": results}, indent=2),
                   encoding="utf-8")
    print("=" * 96)
    print("PRIOR ART 50-way (NMED-T+NMED-H, 95/5, chance 0.02): "
          "EEG2Mel 0.259 | LaBraM 0.380 | CBraMod 0.402 | channel-oriented 0.487")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
