"""
Duplicate-free 50-way, matching the prior art's sampler.

Prior art (jqin4749/EEG-to-Music, `scripts/eval_embspace.py` + `MusicSegmentSampler`):
    n_way = 50, in-batch CLIP retrieval, acc = (argmax(dim=-1) == arange) repeated 10x
    -- literally identical to `gate_nmedt_labram.top1_inbatch`.

BUT their sampler is not uniform.  Their own docstring says why:

    "The same excerpt is presented to every subject (and, on NMED-H, forward and
     reversed share a song_name), so drawing windows uniformly from the test subset
     puts byte-identical candidates in the same batch, where the diagonal-matching
     rule scores a correct retrieval as an error.  Keying on (song_name, window_idx)
     -- what determines the audio slice -- and emitting one window per key makes
     every candidate set duplicate-free."

On NMED-T the same excerpt really is played to all 20 subjects, so my 6,300 rows
collapse to ~315 distinct audio slices.  Drawing 50 rows uniformly therefore puts
~4 duplicate PAIRS in every candidate set, and each duplicate pair makes one
correct retrieval unwinnable.  Rounds 1-2 used `torch.randperm` -> the numbers were
systematically LOW.

This script re-measures the same cached features with
    (a) the uniform sampler  (what rounds 1-2 did), and
    (b) the duplicate-free sampler (what the prior art does),
so the size of my own bias is visible.

Runs on CPU on purpose: the GPU is busy with the fine-tune.
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
# ---- mainv2 workspace bootstrap (generated) -------------------------------
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent            # src/<group>/
ROOT = HERE.parents[1]                            # mainv2/
DATA = ROOT / "data"                              # read-only caches
RESULTS = ROOT / "results"                        # result json + prereg json
for _g in ("data_pipeline", "encoder", "audio_target", "align", "measurement",
           "control", "reporting", "evaluation"):
    _p = str(ROOT / "src" / _g)
    if _p not in sys.path:
        sys.path.insert(0, _p)

DEV = torch.device("cpu")            # GPU is occupied by the fine-tune
N_WAY, N_DRAWS, SEEDS = 50, 3000, [0, 1, 2, 3, 4]
NPZ = DATA / "gate_nmedt_labram.npz"
NPZ2 = DATA / "gate_nmedt_labram2_tokens.npz"
OUT = RESULTS / "gate_nmedt_nway_result.json"


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
def top1_rows(zq, tk, rows):
    """rows: [n_draws, n_way] row indices; count diagonal wins."""
    hits = 0
    ar = torch.arange(rows.shape[1], device=zq.device)   # follow the caller's device
    for i in range(rows.shape[0]):
        idx = torch.as_tensor(rows[i], device=zq.device)
        sims = zq[idx] @ tk[idx].T
        hits += int((sims.argmax(1) == ar).sum())
    return hits / rows.size


def rows_uniform(n, rng):
    return np.stack([rng.permutation(n)[:N_WAY] for _ in range(N_DRAWS)])


def rows_dup_free(pad, rng):
    """one row per DISTINCT audio key; uniform over keys and over rows inside a key."""
    nk, maxg = pad.shape
    if nk < N_WAY:
        raise ValueError(f"only {nk} distinct slices, need {N_WAY}")
    # numpy refuses a 2-D `size` together with replace=False -> draw one at a time
    sel = np.empty((N_DRAWS, N_WAY), np.int64)
    for d in range(N_DRAWS):
        sel[d] = rng.choice(nk, N_WAY, replace=False)
    off = rng.integers(0, maxg, size=(N_DRAWS, N_WAY))
    rows = pad[sel, off]
    bad = rows < 0
    if bad.any():
        rows[bad] = pad[sel[bad], 0]
    return rows


def build_pad(keys):
    """keys: per-row key for a subset.  Returns [n_keys, max_rows] of POSITIONAL
    indices into that subset (-1 = pad)."""
    uk, inv = np.unique(keys, return_inverse=True)
    cnt = np.bincount(inv, minlength=len(uk))
    pad = np.full((len(uk), int(cnt.max())), -1, np.int64)
    pos = np.zeros(len(uk), np.int64)
    for i, k in enumerate(inv):
        pad[k, pos[k]] = i
        pos[k] += 1
    return pad


def train_head(X, T, tr):
    torch.manual_seed(0)
    m = Head(X.shape[1])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs, bs = 40, 512
    st_total = max(epochs * max(len(tr) // bs, 1), 1)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3,
                                              total_steps=st_total)
    for _ in range(epochs):
        m.train()
        p = torch.randperm(len(tr))
        for i in range(0, len(tr) - bs + 1, bs):
            j = tr[p[i:i + bs]]
            loss = info_nce(l2(m(X[j])), l2(T[j]))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            sch.step()
    m.eval()
    return m


def run(tag, X, T, key, n, results):
    X = torch.tensor(np.ascontiguousarray(X, np.float32))
    T = torch.tensor(np.ascontiguousarray(T, np.float32))
    uni, dup, ctrl = [], [], []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        pm = rng.permutation(n)
        n_tr = int(0.70 * n)
        tr = torch.as_tensor(pm[:n_tr])
        te = pm[n_tr + int(0.15 * n):]
        m = train_head(X, T, tr)
        with torch.no_grad():
            zq = l2(m(X[torch.as_tensor(te)]))          # query = EEG
            tk = l2(T[torch.as_tensor(te)])             # candidates = audio
            tk_sh = tk[torch.randperm(len(te))]
        pad = build_pad(key[te])
        ru = rows_uniform(len(te), rng)
        rd = rows_dup_free(pad, rng)
        uni.append(top1_rows(zq, tk, ru))
        dup.append(top1_rows(zq, tk, rd))
        ctrl.append(top1_rows(zq, tk_sh, rd))
        if seed == 0:
            u0 = ru[0]
            print(f"      uniform draw of 50 -> {len(u0)-len(set(u0))} duplicate "
                  f"rows; {pad.shape[0]} distinct slices in the test split")
    u_, d_, c_ = np.array(uni), np.array(dup), np.array(ctrl)
    print(f"  {tag:18s} uniform {u_.mean():.4f} +- {u_.std():.4f} | "
          f"dup-free {d_.mean():.4f} +- {d_.std():.4f} | ctrl {c_.mean():.4f} | "
          f"{d_.mean()/0.02:5.2f}x | sampler alone x{d_.mean()/max(u_.mean(),1e-9):.2f}")
    results[tag] = {"uniform": float(u_.mean()), "uniform_std": float(u_.std()),
                    "dup_free": float(d_.mean()), "dup_free_std": float(d_.std()),
                    "ctrl": float(c_.mean())}


def main():
    ref = np.load(NPZ)
    tok = np.load(NPZ2, allow_pickle=True)
    n = len(ref["song"])
    key = ref["song"].astype(np.int64) * 1000 + ref["seg"]
    ukeys, inv = np.unique(key, return_inverse=True)
    cnt = np.bincount(inv)
    print(f"{n} rows -> {len(ukeys)} distinct audio slices "
          f"(rows per slice: mean {cnt.mean():.2f}, max {cnt.max()})")
    print("  NMED-T replays every excerpt to all 20 subjects, so one 'row' =") 
    print("  (subject, song, segment); uniform sampling cannot avoid copies\n")

    T = ref["target"]
    bp = ref["bandpower"].astype(np.float32)
    tr0 = np.random.default_rng(0).permutation(n)[:int(0.7 * n)]
    bp_std = (bp - bp[tr0].mean(0)) / (bp[tr0].std(0) + 1e-6)

    results = {"n_rows": int(n), "n_distinct_slices": int(len(ukeys)),
               "n_way": N_WAY, "seeds": SEEDS}
    print("=" * 112)
    run("band_power", bp, T, key, n, results)
    run("band_power + std", bp_std, T, key, n, results)
    run("labram frozen", np.asarray(tok["div100_pool"], np.float32), T, key, n,
        results)
    print("=" * 112)
    print("chance 0.0200; the prior art's 0.380 / 0.487 use the DUPLICATE-FREE "
          "sampler, so that column is the comparable one")
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
