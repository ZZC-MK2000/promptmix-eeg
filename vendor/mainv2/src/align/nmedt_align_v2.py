"""
Alignment ladder trainer (prereg: data/prereg_ladder_20260925.json).

One script, one variable at a time:

    --cache  no10 : gate_nmedt10.npz / *_uv16.npy, 7020 non-overlap rows,
                    window-random split 70/15/15 (verbatim `nmedt_align.splits`)
    --cache  ov2  : gate_nmedt10_ov2.*, ~27.8k rows on a 2 s stride,
                    split BY 60 s BLOCK 70/10/20; val/test evaluate ONLY
                    `aligned8` rows (the original 8 s grid -> comparable numbers)

    --front  enc   : full-encoder InfoNCE (7.47 M trainable), the 0.0310 arm's setup
    --front  bp|bpstd|labram : frozen features + Head (40 ep, bs 512), cheap controls

    --pool   on    : all distinct TRAIN-split audio slices as extra InfoNCE negatives
                     (audio side frozen -> no staleness; MoCo machinery unnecessary).
                     Pool entries whose slice key equals a batch row's key are MASKED
                     (same-slice copies across subjects are positive duplicates, not
                     negatives).  Pool is built from the TRAIN split only (leak guard).
    --dense  LAM   : per-second envelope-feature InfoNCE auxiliary (scipy STFT,
                     16-d/s: 8 log-band energies + 8 band fluxes, z-scored).
                     EEG per-second embedding = channel-mean of LaBraM patch tokens
                     (token order is channel-major: 'B N A T -> B (N A) T').
                     Negatives include the OTHER SECONDS OF THE SAME WINDOW ->
                     same-song hard negatives that force temporal information.
    --lag    D     : integer seconds; dense target[t] = env[song][s0 + t + D].

Metrics (five-piece set per discipline rule 2):
    10-way proto retrieval (chance 0.1, ceiling recomputed per cache, comparison
    ALWAYS through proto_songs mapping §2-H), k-way duplicate-free slice retrieval
    (k = min(50, distinct test slices), chance 1/k), shuffled-pairing control,
    seed recorded, split unit recorded.

Usage
    python nmedt_align_v2.py --tag L0_enc_no10 --cache no10 --front enc --seed 0
    python nmedt_align_v2.py --tag L1_enc_ov2  --cache ov2  --front enc --seed 0
    python nmedt_align_v2.py --tag L2_enc_ov2_pool --cache ov2 --front enc --pool --seed 0
    python nmedt_align_v2.py --tag L3_enc_ov2_pool_dense --cache ov2 --front enc --pool --dense 0.5 --lag 0 --seed 0
"""
import argparse
import json
import sys
import time
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
DATA = Path(os.environ.get("MAINV2_DATA", ROOT / "data"))      # read-only caches
RESULTS = Path(os.environ.get("MAINV2_RESULTS", ROOT / "results"))  # result json + prereg json
for _g in ("data_pipeline", "encoder", "audio_target", "align", "measurement",
           "control", "reporting", "evaluation"):
    _p = str(ROOT / "src" / _g)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from encoder_labram import LabramEncoder                                  # noqa: E402
from gate_nmedt_labram import DEV                                         # noqa: E402
from gate_nmedt_nway import Head, info_nce, l2, build_pad                 # noqa: E402

OUT = RESULTS / "align_ladder_result.json"
ENV_CACHE = DATA / "nmedt_env_secs.npz"
PRETRAIN_CKPT = DATA / "nmedt_pretrain_ckpt.pt"
NMEDT = Path(os.environ.get("NMEDT_ROOT", "data/NMED-T"))
N_DRAWS = 3000
TAU = 0.07

# Cheap-screen gates, verbatim from data/prereg_shape_matching_alignment_20261001.json,
# written before any result existed.  All three must pass or the run stops before
# generation.  The parenthesised numbers are the references the thresholds were derived
# from, not thresholds themselves.
SCREEN_S1_MUTUAL_COS_MIN = 0.40       # audio 0.5917 on the same 84 units; was 0.0332
SCREEN_S2_RETRIEVAL_MIN = 0.60        # baseline 0.6931 = 14-way, mean of seeds 0 and 1
SCREEN_S3_PARTICIPATION_MIN = 8.0     # centred audio participation ratio 19.3


# --------------------------------------------------------------------- data
# h14 plus its two retargeted variants: same EEG rows, same order, same block split,
# only `target`/`proto` differ.  Anything keyed on "the merged 14-way cache" has to cover
# all three, or the new arms silently lose their per-dataset breakdown.
H14_CACHES = ("h14", "h14_lfix", "h14_a")


def load_cache(which):
    if which == "no10":
        npz, uv = DATA / "gate_nmedt10.npz", DATA / "gate_nmedt10_uv16.npy"
    elif which == "ov2":
        npz, uv = DATA / "gate_nmedt10_ov2.npz", DATA / "gate_nmedt10_ov2_uv16.npy"
    elif which == "h14":
        npz, uv = DATA / "gate_nmedh14_ov2.npz", DATA / "gate_nmedh14_ov2_uv16.npy"
    # Both retargeted caches hold the SAME EEG rows in the SAME order and differ only in
    # `target`/`proto`, so they deliberately share h14's uv16 -- that is what makes the
    # arm comparison single-factor.  See nmedh14_retarget_claps.py.
    elif which == "h14_lfix":
        npz, uv = DATA / "gate_nmedh14_ov2_lfix.npz", DATA / "gate_nmedh14_ov2_uv16.npy"
    elif which == "h14_a":
        npz, uv = DATA / "gate_nmedh14_ov2_a.npz", DATA / "gate_nmedh14_ov2_uv16.npy"
    else:
        raise ValueError(which)
    d = np.load(npz, allow_pickle=True)
    n = len(d["song"])
    if uv.exists():
        U = np.load(uv, mmap_mode="r")
        assert U.shape[0] == n, f"uv/npz row mismatch {U.shape[0]} vs {n}"
    else:
        U = None
    return d, U, n


def proto_mapping(d):
    """song id -> prototype ROW through proto_songs (discipline §2-H guard)."""
    ps = np.asarray(d["proto_songs"])
    if not (ps == np.arange(len(ps))).all():
        print(f"  !! proto_songs={ps.tolist()} not identity -> explicit mapping")
    pos = np.full(int(ps.max()) + 1, -1, np.int64)
    pos[ps] = np.arange(len(ps))
    lab = pos[d["song"]]
    assert (lab >= 0).all(), "songs without prototype"
    return lab


def make_splits(d, which, seed, heldout=(), valid=()):
    """Returns tr/va/te row-index arrays + split-unit description."""
    n = len(d["song"])
    rng = np.random.default_rng(seed)
    if heldout:
        # Subject-held-out control: test subjects are never seen in train or val.
        assert "subject" in d, "--heldout-subjects needs a cache with a 'subject' field"
        subj = np.asarray(d["subject"]).astype(str)
        al = np.asarray(d["aligned8"]).astype(bool)
        ho = np.isin(subj, list(heldout))
        vs = np.isin(subj, list(valid))
        tr = np.where(~ho & ~vs)[0]
        va = np.where(vs & al)[0]
        te = np.where(ho & al)[0]
        return tr, va, te, (f"heldout-subject(test={','.join(heldout)}) "
                            f"val-subject({','.join(valid)})")
    if which == "no10":
        pm = rng.permutation(n)                    # verbatim nmedt_align.splits
        n_tr, n_va = int(0.70 * n), int(0.15 * n)
        return pm[:n_tr], pm[n_tr:n_tr + n_va], pm[n_tr + n_va:], "window(70/15/15)"
    # ov2: split by contiguous 60 s blocks; eval rows = aligned8 only
    blocks = np.asarray(d["block"])
    ub = np.unique(blocks)
    pm = rng.permutation(len(ub))
    n_tr, n_va = int(0.70 * len(ub)), int(0.10 * len(ub))
    b_tr, b_va, b_te = ub[pm[:n_tr]], ub[pm[n_tr:n_tr + n_va]], ub[pm[n_tr + n_va:]]
    al = np.asarray(d["aligned8"]).astype(bool)
    tr = np.where(np.isin(blocks, b_tr))[0]
    va = np.where(np.isin(blocks, b_va) & al)[0]
    te = np.where(np.isin(blocks, b_te) & al)[0]
    return tr, va, te, "block60s(70/10/20), eval=aligned8"


# --------------------------------------------------------------------- metrics
def rows_dup_free_k(pad, rng, k):
    nk, maxg = pad.shape
    k = min(k, nk)
    sel = np.stack([rng.choice(nk, k, replace=False) for _ in range(N_DRAWS)])
    off = rng.integers(0, maxg, size=(N_DRAWS, k))
    rows = pad[sel, off]
    bad = rows < 0
    if bad.any():
        rows[bad] = pad[sel[bad], 0]
    return rows, k


@torch.no_grad()
def top1_rows_k(zq, tk, rows):
    hits = 0
    ar = torch.arange(rows.shape[1], device=zq.device)
    for i in range(rows.shape[0]):
        idx = torch.as_tensor(rows[i], device=zq.device)
        sims = zq[idx] @ tk[idx].T
        hits += int((sims.argmax(1) == ar).sum())
    return hits / rows.size


def evaluate(zq, T_all, lab, proto_P, key, te_rows, seed, tag=""):
    """zq: [m,512] normalized query embeddings (torch, DEV) for te_rows."""
    rng = np.random.default_rng(seed)
    te = np.asarray(te_rows)
    sd = np.asarray(lab)[te]
    ten = float((zq @ proto_P.T).argmax(1).cpu().numpy().__eq__(sd).mean())
    rp = rng.permutation(proto_P.shape[0])
    ten_c = float((zq @ proto_P.T).argmax(1).cpu().numpy().__eq__(rp[sd]).mean())
    tk = l2(T_all[te])
    pad = build_pad(np.asarray(key)[te])
    rd, k = rows_dup_free_k(pad, rng, 50)
    kway = top1_rows_k(zq, tk, rd)
    sh = rng.permutation(len(te))
    ctrl = top1_rows_k(zq, tk[torch.as_tensor(sh, device=DEV)], rd)
    return {"ten_way": ten, "ten_ctrl": ten_c, "kway": kway, "k": int(k),
            "kway_ctrl": ctrl, "n_test": int(len(te)), "tag": tag}


# --------------------------------------------------------------------- envelope
def build_env_features():
    """Per-second 16-d envelope features for the 10 songs (scipy, no librosa)."""
    if ENV_CACHE.exists():
        return np.load(ENV_CACHE)
    from scipy.signal import spectrogram as spg
    from clap_eval import ClapEvaluator
    sm = json.loads((NMEDT / "song_mapping.json").read_text(encoding="utf-8"))
    ev = ClapEvaluator()
    bands = np.logspace(np.log10(62.5), np.log10(8000), 9)
    feats, lens = {}, {}
    for idx in sorted(int(k) for k in sm):
        wav = ev.load(str(NMEDT / "Music" / sm[str(idx)]["file"]))
        sr = ev.sr
        f, tt, S = spg(wav, fs=sr, nperseg=2048, noverlap=2048 - 512,
                       mode="psd", scaling="density")
        P = S.mean(0) if S.ndim == 3 else S          # mono -> [freq, frames]
        E = np.stack([np.log10(np.maximum(P[(f >= a) & (f < b)].mean(0), 1e-12))
                      for a, b in zip(bands[:-1], bands[1:])])      # [8, frames]
        frame_t = tt
        n_sec = int(frame_t.max()) + 1
        sec_e = np.full((n_sec, 8), np.nan)
        sec_f = np.full((n_sec, 8), np.nan)
        dE = np.abs(np.diff(E, axis=1))
        for s in range(n_sec):
            m = (frame_t >= s) & (frame_t < s + 1)
            if m.sum():
                sec_e[s] = E[:, m].mean(1)
            mf = (frame_t[:-1] >= s) & (frame_t[:-1] < s + 1)
            if mf.sum():
                sec_f[s] = dE[:, mf].mean(1)
        sec_e = np.nan_to_num(sec_e, nan=sec_e[~np.isnan(sec_e)].mean(0).min())
        sec_f = np.nan_to_num(sec_f, nan=0.0)
        feats[idx] = np.concatenate([sec_e, sec_f], 1).astype(np.float32)
        lens[idx] = n_sec
        print(f"  env song {idx}: {n_sec} s, feat {feats[idx].shape}")
    ids = sorted(feats)
    max_len = max(lens[i] for i in ids)
    X = np.zeros((len(ids), max_len, 16), np.float32)
    for j, i in enumerate(ids):
        X[j, :lens[i]] = feats[i]
    real = np.concatenate([feats[i] for i in ids])          # z-score on real secs
    mu, sd = real.mean(0), real.std(0) + 1e-8
    X = (X - mu) / sd
    for j, i in enumerate(ids):                              # re-zero the padding
        X[j, lens[i]:] = 0.0
    np.savez_compressed(ENV_CACHE, env=X.astype(np.float32),
                        lengths=np.array([lens[i] for i in ids]),
                        mu=mu.astype(np.float32), sd=sd.astype(np.float32))
    print(f"  wrote {ENV_CACHE.name}  {X.shape}")
    return np.load(ENV_CACHE)


def load_dense_source(kind):
    """-> (E [10, max_sec, D], lengths [10]) standardized per-second targets."""
    if kind == "env":
        env = build_env_features()
        return env["env"], env["lengths"]
    if kind == "mimi":
        p = DATA / "nmedt_mimi_secs.npz"
        if not p.exists():
            raise SystemExit(f"{p} missing -- run nmedt_mimi_secs.py first")
        z = np.load(p)
        return z["latents"], z["lengths"]
    raise ValueError(kind)


def load_mert_source(layer, cache="ov2"):
    # every 14-song cache shares one MERT target file; the dense auxiliary is not the
    # variable being manipulated between arms
    fn = {"ov2": "nmedt_mert_secs.npz", "h14": "nmedt14_mert_secs.npz",
          "h14_lfix": "nmedt14_mert_secs.npz",
          "h14_a": "nmedt14_mert_secs.npz"}[cache]
    p = DATA / fn
    if not p.exists():
        raise SystemExit(f"{p} missing -- run the MERT cache builders first")
    z = np.load(p)
    lat = z["latents"]                     # [10, 13, S, 768] fp16, z-scored
    L = lat.shape[1]
    li = layer if layer >= 0 else L + layer
    if not 0 <= li < L:
        raise SystemExit(f"--mert-layer {layer} out of range (0..{L-1})")
    return lat[:, li].astype(np.float32), z["lengths"]


def dense_targets(d, E, lengths, lag):
    """[n, 8, D] per-window per-second dense targets (lag-shifted), + valid mask."""
    songs, seg2 = d["song"], d["seg2"]
    n = len(songs)
    out = np.zeros((n, 8, E.shape[2]), np.float32)
    valid = np.ones(n, bool)
    for i in range(n):
        s0 = int(seg2[i]) * 2 + lag
        L = int(lengths[songs[i]])
        if s0 < 0 or s0 + 8 > L:
            valid[i] = False
            s0 = int(np.clip(s0, 0, max(L - 8, 0)))
        out[i] = E[songs[i], s0:s0 + 8]
    return out, valid


# --------------------------------------------------------------------- trainers
def info_nce_pool(z, t, pool_t, pool_keys, batch_keys):
    """Symmetric InfoNCE; extra negatives = pool slices, same-key entries masked."""
    z, t = l2(z), l2(t)
    lg_in = z @ t.T / TAU
    lab = torch.arange(z.shape[0], device=z.device)
    loss_e2a = F.cross_entropy(lg_in, lab)
    loss_a2e = F.cross_entropy(lg_in.T, lab)
    if pool_t is not None and pool_t.shape[0]:
        lg_p = z @ pool_t.T / TAU                       # [B, P]
        mask = (batch_keys[:, None] == pool_keys[None, :])
        lg_p = lg_p.masked_fill(torch.as_tensor(mask, device=z.device), -1e4)
        loss_e2a = F.cross_entropy(torch.cat([lg_in, lg_p], 1), lab)
    return 0.5 * (loss_e2a + loss_a2e)


def cloud_shape(Q):
    """Mean pairwise cosine (S1) and centred participation ratio (S3).

    Retrieval accuracy is blind to both, which is how the first end-to-end run
    cleared every retrieval gate while feeding the generator a near-orthogonal
    cloud.  The participation ratio is computed on the MEAN-CENTRED cloud -- that is
    the definition the preregistered audio reference 19.3 was measured with; on the
    uncentred cloud the audio itself scores 2.3, so an uncentred reading would make
    the >= 8 anti-collapse gate unreachable by construction.
    """
    Q = np.asarray(Q, np.float64)
    Qn = Q / np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-12)
    n = Qn.shape[0]
    if n < 2:
        return {"mutual_cos": float("nan"), "participation_ratio": float("nan"),
                "n": int(n)}
    s = Qn.sum(0)
    mut = float(((s * s).sum() - n) / (n * (n - 1)))
    C = Qn - Qn.mean(0)
    ev = np.clip(np.linalg.eigvalsh((C.T @ C) / n), 0, None)
    pr = float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-12))
    return {"mutual_cos": mut, "participation_ratio": pr, "n": int(n)}


def train_frozen(a, X, T, key, lab, proto, tr, va, te, pool_keys_t, pool_t):
    Xt = torch.tensor(np.ascontiguousarray(X, np.float32), device=DEV)
    Tt = torch.tensor(np.ascontiguousarray(T, np.float32), device=DEV)
    P = l2(torch.tensor(np.ascontiguousarray(proto, np.float32), device=DEV))
    trt = torch.as_tensor(tr, device=DEV)
    torch.manual_seed(a.seed)
    m = Head(X.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    epochs, bs = 40, 512
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=1e-3, total_steps=max(epochs * max(len(tr) // bs, 1), 1))
    vat = torch.as_tensor(va, device=DEV)
    best, best_state, best_ep = -1, None, -1
    t0 = time.time()
    for ep in range(epochs):
        m.train()
        p = torch.randperm(len(trt), device=DEV)
        for i in range(0, len(trt) - bs + 1, bs):
            j = trt[p[i:i + bs]]
            bk = torch.as_tensor(np.asarray(key)[j.cpu().numpy()], device=DEV)
            loss = info_nce_pool(m(Xt[j]), Tt[j],
                                 pool_t if a.pool else None,
                                 pool_keys_t if a.pool else None, bk)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step()
        if (ep + 1) % 10 == 0 or ep == epochs - 1:
            m.eval()
            with torch.no_grad():
                zq = l2(m(Xt[vat]))
            sd = np.asarray(lab)[va]
            v10 = float((zq @ P.T).argmax(1).cpu().numpy().__eq__(sd).mean())
            if v10 > best:
                best, best_ep = v10, ep + 1
                best_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
            print(f"    ep {ep+1:3d}  val 10-way {v10:.4f}  best {best:.4f} "
                  f"({best_ep})  [{time.time()-t0:.0f}s]")
    # also snapshot at final epoch if it was never evaluated
    if best_state is None:
        best_state = {k: t.detach().clone() for k, t in m.state_dict().items()}
    m.load_state_dict(best_state); m.eval()
    with torch.no_grad():
        zq = l2(m(Xt[torch.as_tensor(te, device=DEV)]))
    return zq, P, best, best_ep, sum(p.numel() for p in m.parameters()), 0


def build_encoder(init):
    cm = json.loads((RESULTS / "nmedt_egi_to_1020.json").read_text(encoding="utf-8"))
    enc = LabramEncoder(device=str(DEV), freeze=False, strict=True, norm="div100",
                        input_chans=cm["input_chans"])
    tag = "stock labram-base (clinical 2500h)"
    if init == "indomain":
        if not PRETRAIN_CKPT.exists():
            raise SystemExit(f"--init indomain needs {PRETRAIN_CKPT}")
        sd = torch.load(PRETRAIN_CKPT, map_location="cpu")
        enc.backbone.load_state_dict(sd["student"], strict=True)
        tag = (f"in-domain NMED-T (step {sd.get('step', '?')}, "
               f"val acc {sd.get('val') and round(sd['val']['val_acc'], 4)})")
    return enc, tag


def train_encoder(a, d, U, T, key, lab, proto, tr, va, te,
                  pool_keys_t, pool_t, dense_t, dense_v, sub=None):
    enc, init_tag = build_encoder(a.init)
    head = Head(200).to(DEV)
    clf = nn.Linear(200, int(np.asarray(lab).max()) + 1).to(DEV) \
        if a.ce > 0 else None
    dhead = nn.Sequential(nn.LayerNorm(200), nn.Linear(200, 64)).to(DEV) \
        if a.dense > 0 else None
    eproj = nn.Linear(int(np.asarray(dense_t).shape[2]), 64).to(DEV) \
        if a.dense > 0 else None
    Tt = torch.tensor(np.asarray(T, np.float32), device=DEV)
    P = l2(torch.tensor(np.ascontiguousarray(proto, np.float32), device=DEV))
    trt = torch.tensor(np.asarray(tr), device=DEV)
    vat = torch.tensor(np.asarray(va), device=DEV)
    if dhead is not None:
        Dt = torch.tensor(np.asarray(dense_t, np.float32), device=DEV)
    if sub is not None:
        mu_t, V_t, Ts_t = sub["mu"], sub["V"], sub["Tsub"]
        pool_ts = sub.get("pool_t")

    def to_target(h):
        """head output -> 512-d L2-normalised vector (reconstructed when sub is on)."""
        if sub is None:
            return l2(h)
        return l2(mu_t + ((h - mu_t) @ V_t) @ V_t.T)

    def uv_of(idx):
        return torch.tensor(np.asarray(U[idx.cpu().numpy()], np.float32), device=DEV)

    params = [{"params": enc.parameters(), "lr": a.lr_enc},
              {"params": head.parameters(), "lr": a.lr_head}]
    if dhead is not None:
        params += [{"params": dhead.parameters(), "lr": a.lr_head},
                   {"params": eproj.parameters(), "lr": a.lr_head}]
    if clf is not None:
        params += [{"params": clf.parameters(), "lr": a.lr_head}]
    opt = torch.optim.AdamW(params, weight_decay=0.01)
    micro, accum = a.micro, a.accum
    spe = max(len(trt) // (micro * accum), 1)
    total = a.epochs * spe
    warm = max(int(0.05 * total), 20)
    sch = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / warm,
                           0.5 * (1 + np.cos(np.pi * min(max(s - warm, 0) /
                                                        max(total - warm, 1), 1.0)))))
    scaler = torch.cuda.amp.GradScaler()
    ntr = sum(p.numel() for g in params for p in g["params"] if p.requires_grad)
    mods = {"enc": enc, "head": head}
    if dhead is not None:
        mods["dhead"], mods["eproj"] = dhead, eproj
    if clf is not None:
        mods["clf"] = clf
    shadow = None
    if a.ema > 0:
        shadow = {f"{nm}.{k}": v.detach().clone().float()
                  for nm, mod in mods.items() for k, v in mod.state_dict().items()}
    print(f"  encoder trainable {ntr/1e6:.2f} M | train rows {len(trt)} | "
          f"{a.epochs} ep x {spe} steps | pool={'on' if a.pool else 'off'} "
          f"dense={a.dense}({a.dense_target}) lag={a.lag} ema={a.ema} "
          f"ce={a.ce} clip_w={a.clip_w} "
          f"subspace={sub['K'] if sub is not None else 0} pos_w={a.pos_w} "
          f"init={a.init if a.front == 'enc' else 'stock'}")

    def forward_batch(j, need_tokens):
        x = uv_of(j)
        if need_tokens:
            tok = enc(x, pool="tokens")                 # [B, C*8, 200]
            b, ct, dd = tok.shape
            zc = tok.reshape(b, 82, ct // 82, dd).mean(1)   # [B, 8, 200] channel-major
            pooled = zc.mean(1)
        else:
            pooled = enc(x)
            zc = None
        return pooled, zc

    def encode_rows(idx, bs=64):
        out = []
        with torch.no_grad():
            for i in range(0, len(idx), bs):
                j = idx[i:i + bs]
                out.append(to_target(head(enc(uv_of(j)))).cpu().numpy())
        return torch.tensor(np.concatenate(out), device=DEV)

    key_np = np.asarray(key)
    lab_np = np.asarray(lab)
    yte = torch.as_tensor(lab_np, device=DEV)
    dense_rows_ok = (torch.tensor(np.asarray(dense_v), device=DEV)
                     if dhead is not None else None)

    def swap_in(state):
        backup = {nm: {k: v.detach().clone() for k, v in mod.state_dict().items()}
                  for nm, mod in mods.items()}
        for nm, mod in mods.items():
            sd0 = mod.state_dict()
            mod.load_state_dict({k: state[f"{nm}.{k}"].to(sd0[k].dtype)
                                 for k in sd0})
        return backup

    @torch.no_grad()
    def val_metrics():
        backup = swap_in(shadow) if shadow is not None else None
        vp = []
        for i in range(0, len(vat), 64):
            j = vat[i:i + 64]
            pooled = enc(uv_of(j))
            vp.append(torch.cat([to_target(head(pooled)).cpu(),
                                 clf(pooled).argmax(1).cpu()[:, None].float()], 1)
                      if clf is not None else to_target(head(pooled)).cpu())
        vp = torch.cat(vp).to(DEV)
        if backup is not None:
            for nm, mod in mods.items():
                mod.load_state_dict(backup[nm])
        v10 = float((vp[:, :512] @ P.T).argmax(1).cpu().numpy()
                    .__eq__(lab_np[va]).mean())
        vclf = (float(vp[:, 512].cpu().numpy().__eq__(lab_np[va]).mean())
                if clf is not None else None)
        return v10, vclf

    t0 = time.time()
    best, best_state, best_ep, n_skip = -1, None, -1, 0
    best_v10 = -1.0
    for ep in range(a.epochs):
        enc.train(); head.train()
        if dhead is not None:
            dhead.train(); eproj.train()
        if clf is not None:
            clf.train()
        p = torch.randperm(len(trt), device=DEV)
        run_loss, run_clip, run_dense, run_ce, run_pos = 0.0, 0.0, 0.0, 0.0, 0.0
        for i in range(0, spe * micro * accum, micro * accum):
            opt.zero_grad(set_to_none=True)
            ok = True
            for k in range(accum):
                j = trt[p[i + k * micro:i + (k + 1) * micro]]
                if len(j) < 2:
                    continue
                if dhead is not None:
                    j = j[dense_rows_ok[j]]
                    if len(j) < 2:
                        continue
                with torch.cuda.amp.autocast():
                    pooled, zc = forward_batch(j, dhead is not None)
                    bk = torch.as_tensor(key_np[j.cpu().numpy()], device=DEV)
                    h = head(pooled)
                    if sub is None:
                        ze, tgt = h, Tt[j]
                        pl = pool_t if a.pool else None
                    else:
                        # fp32 on purpose: the audio cloud is a narrow cone, so the
                        # subspace coordinate is small relative to mu and fp16 rounds
                        # away exactly the part the placement anchor is supposed to fit.
                        with torch.cuda.amp.autocast(enabled=False):
                            ze = (h.float() - mu_t) @ V_t
                            eh = l2(mu_t + ze @ V_t.T)
                        tgt, pl = Ts_t[j], (pool_ts if a.pool else None)
                    clip = info_nce_pool(ze, tgt, pl,
                                         pool_keys_t if a.pool else None, bk)
                    loss = a.clip_w * clip / accum
                    if sub is not None:
                        pos = (1.0 - (eh * l2(Tt[j])).sum(-1)).mean()
                        loss = loss + a.pos_w * pos / accum
                        run_pos += float(pos)
                    if dhead is not None:
                        b = zc.shape[0]
                        ze = l2(dhead(zc).reshape(b * 8, -1))
                        env = l2(eproj(Dt[j]).reshape(b * 8, -1))
                        lg = ze @ env.T / TAU
                        lb = torch.arange(b * 8, device=DEV)
                        dl = 0.5 * (F.cross_entropy(lg, lb)
                                    + F.cross_entropy(lg.T, lb))
                        loss = loss + a.dense * dl / accum
                        run_dense += float(dl)
                    if clf is not None:
                        cl = F.cross_entropy(clf(pooled), yte[j])
                        loss = loss + a.ce * cl / accum
                        run_ce += float(cl)
                if not torch.isfinite(loss):
                    ok = False
                    break
                scaler.scale(loss).backward()
                run_loss += float(loss); run_clip += float(clip)
            if ok:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(
                    [q for g in params for q in g["params"]], 1.0)
                scaler.step(opt); scaler.update()
                if shadow is not None:
                    with torch.no_grad():
                        for nm, mod in mods.items():
                            for pk, pv in mod.named_parameters():
                                if pv.requires_grad:
                                    shadow[f"{nm}.{pk}"].mul_(a.ema).add_(
                                        pv.detach().float(), alpha=1 - a.ema)
            else:
                n_skip += 1
                opt.zero_grad(set_to_none=True)
                if n_skip > 20:
                    raise SystemExit(f"ABORT: {n_skip} non-finite losses")
            sch.step()
        v10, vclf = val_metrics()
        score = vclf if vclf is not None else v10
        if score > best:
            best, best_ep = score, ep + 1
            best_v10 = v10
            if shadow is not None:
                best_state = {k: t.clone() for k, t in shadow.items()}
            else:
                best_state = {f"{nm}.{k}": t.detach().clone()
                              for nm, mod in mods.items()
                              for k, t in mod.state_dict().items()}
        print(f"    ep {ep+1:3d}/{a.epochs}  val 10-way {v10:.4f}  "
              + (f"val clf {vclf:.4f}  " if vclf is not None else "")
              + f"best {best:.4f}({best_ep})  loss {run_loss/max(spe,1):.4f} clip "
              f"{run_clip/max(spe,1):.4f} dense {run_dense/max(spe,1):.4f}"
              + (f" pos {run_pos/max(spe,1):.4f}" if sub is not None else "")
              + (f" ce {run_ce/max(spe,1):.4f}" if clf is not None else "")
              + f"  skip {n_skip}  [{time.time()-t0:6.0f}s, "
                f"{time.time()-t0:.0f}s tot]")
        sys.stdout.flush()
    if best_state:
        swap_in(best_state)
    enc.eval(); head.eval()
    if clf is not None:
        clf.eval()
    zq = encode_rows(torch.as_tensor(te, device=DEV))
    test_clf = None
    if clf is not None:
        te_np = np.asarray(te)
        preds = np.empty(len(te_np), np.int64)
        with torch.no_grad():
            for i in range(0, len(te_np), 64):
                j = torch.as_tensor(te_np[i:i + 64], device=DEV)
                preds[i:i + len(j)] = clf(enc(uv_of(j))).argmax(1).cpu().numpy()
        test_clf = float(preds.__eq__(lab_np[te_np]).mean())
        print(f"  test clf acc (CE head): {test_clf:.4f}")
    else:
        preds = None
    if a.save_emb:
        emb_all = encode_rows(torch.arange(len(d["song"]), device=DEV))
        np.save(a.save_emb, emb_all.cpu().numpy().astype(np.float32))
        print(f"  saved embeddings {emb_all.shape} -> {a.save_emb}")
    return zq, P, best, best_ep, ntr, n_skip, test_clf, init_tag, best_v10, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cache", choices=["no10", "ov2", "h14", "h14_lfix", "h14_a"],
                    required=True)
    ap.add_argument("--front", choices=["enc", "bp", "bpstd", "labram",
                                        "fusion", "fusion_emb",
                                        "fusion_emb_clamp"],
                    default="enc")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--micro", type=int, default=16)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr-enc", type=float, default=1e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--pool", action="store_true")
    ap.add_argument("--dense", type=float, default=0.0)
    ap.add_argument("--dense-target", choices=["env", "mimi", "mert"], default="env")
    ap.add_argument("--mert-layer", type=int, default=-1,
                    help="MERT layer for dense targets (0=feature proj, -1=last)")
    ap.add_argument("--ce", type=float, default=0.0,
                    help="weight of song-ID cross-entropy head (multi-task arm)")
    ap.add_argument("--clip-w", type=float, default=1.0,
                    help="weight of the CLIP-window InfoNCE term (0 = CE-only control)")
    ap.add_argument("--init", choices=["stock", "indomain"], default="stock",
                    help="encoder initialisation (indomain = NMED-T pretrained ckpt)")
    ap.add_argument("--ema", type=float, default=0.0,
                    help="EMA weight-average decay for eval/snapshot (0 = off)")
    ap.add_argument("--lag", type=int, default=0)
    ap.add_argument("--smoke-steps", type=int, default=0,
                    help="timing smoke: run N encoder steps then exit")
    ap.add_argument("--save-emb", default="",
                    help="after encoder training, save l2 head embeddings [n,512]")
    ap.add_argument("--emb-file", default="",
                    help="for --front fusion_emb: npy of [n,512] aligned embeddings")
    ap.add_argument("--align-subspace", type=int, default=0,
                    help="K>0: align in the top-K principal subspace of the TRAIN-row "
                         "audio cloud and reconstruct with its mean, instead of aligning "
                         "in the raw 512-d space (0 = off, previous behaviour)")
    ap.add_argument("--pos-w", type=float, default=0.0,
                    help="weight of the placement anchor 1-cos(reconstruction, own audio "
                         "target).  Needed whenever --align-subspace>0: the contrastive "
                         "term L2-normalises both sides, so on its own it leaves the "
                         "subspace coordinate's magnitude -- and hence the reconstructed "
                         "cloud's shape -- unconstrained.")
    ap.add_argument("--subspace-out", default="",
                    help="npz to freeze mu/V into, so the generation step loads the "
                         "identical constants instead of recomputing them")
    ap.add_argument("--heldout-subjects", default="",
                    help="comma-separated subject ids held out for the test split "
                         "(subject-held-out control; excluded from train and val)")
    ap.add_argument("--val-subjects", default="",
                    help="comma-separated subject ids used for validation and excluded "
                         "from train (needed with --heldout-subjects so val is unseen too)")
    a = ap.parse_args()

    d, U, n = load_cache(a.cache)
    lab = proto_mapping(d)
    key = d["song"].astype(np.int64) * 1000 + d["seg2"].astype(np.int64) \
        if "seg2" in d else d["song"].astype(np.int64) * 1000 + d["seg"].astype(np.int64)
    T, proto = d["target"], d["proto"]
    tr, va, te, split_desc = make_splits(
        d, a.cache, a.seed,
        tuple(s.strip() for s in a.heldout_subjects.split(",") if s.strip()),
        tuple(s.strip() for s in a.val_subjects.split(",") if s.strip()))
    print(f"[{a.tag}] cache={a.cache} n={n} front={a.front} seed={a.seed} "
          f"split={split_desc}")
    print(f"  rows tr/va/te = {len(tr)}/{len(va)}/{len(te)} | "
          f"test distinct slices = {len(np.unique(key[te]))} | "
          f"train distinct slices = {len(np.unique(key[tr]))}")

    # audio-side ceiling (physical check, discipline rule 13)
    q = l2(torch.tensor(np.asarray(T, np.float32), device=DEV))
    P_all = l2(torch.tensor(np.ascontiguousarray(proto, np.float32), device=DEV))
    ceil10 = float((q @ P_all.T).argmax(1).cpu().numpy().__eq__(lab).mean())
    print(f"  audio ceiling {len(proto)}-way: {ceil10:.4f}")
    assert ceil10 > 0.75, f"ceiling {ceil10:.3f} implausible -> label/mapping bug"

    # negative pool: distinct TRAIN-split slices only (leak guard)
    pool_keys_t, pool_t, pool_first = None, None, None
    if a.pool:
        ktr = key[tr]
        uk, fi = np.unique(ktr, return_index=True)     # fi indexes into ktr
        first = np.asarray(tr)[fi]                     # row ids of first copies
        pool_first = first
        assert len(np.unique(key[first])) == len(uk)
        pool_keys_t = torch.as_tensor(uk, device=DEV)
        pool_t = l2(torch.tensor(np.asarray(T, np.float32)[first], device=DEV))
        print(f"  pool: {len(uk)} distinct train slices")

    # shape-matching target space (prereg: prereg_shape_matching_alignment_20261001)
    # Retrieval accuracy constrains only the RANKING of the EEG cloud against the
    # audio prototypes, not its position or spread: the first end-to-end run cleared
    # every retrieval gate while producing a near-orthogonal cloud (mutual cosine
    # 0.033) inside an audio cloud that lives in a narrow cone (0.592, participation
    # ratio 19.3 of 512).  The generator only ever saw the cone.  Aligning in the
    # top-K principal subspace of the audio cloud and reconstructing through its mean
    # makes the EEG cloud occupy the same region, and the placement anchor below is
    # what actually pins the magnitude that InfoNCE normalises away.
    sub = None
    if a.align_subspace > 0:
        if a.front != "enc":
            raise SystemExit("--align-subspace requires --front enc")
        if a.pos_w <= 0:
            raise SystemExit("--align-subspace>0 requires --pos-w>0: InfoNCE "
                             "L2-normalises both sides, so alone it leaves the "
                             "subspace coordinate's magnitude -- and hence the "
                             "reconstructed cloud's shape -- unconstrained")
        K = int(a.align_subspace)
        Tf = np.asarray(T, np.float64)
        if not 0 < K < Tf.shape[1]:
            raise SystemExit(f"--align-subspace {K} must be in 1..{Tf.shape[1] - 1}")
        # TRAIN rows only: the end-to-end conditioning units are all test rows, so
        # fitting mu/V on anything else would leak them into the generator's input.
        mu_np = Tf[tr].mean(0)
        _, sv, vt = np.linalg.svd(Tf[tr] - mu_np, full_matrices=False)
        V_np = np.ascontiguousarray(vt[:K].T)                 # [512, K]
        var = float((sv[:K] ** 2).sum() / (sv ** 2).sum())
        pr_a = float((sv ** 2).sum() ** 2 / (sv ** 4).sum())
        dev_te = Tf[te] - mu_np
        var_te = float(1.0 - ((dev_te - dev_te @ V_np @ V_np.T) ** 2).sum()
                       / (dev_te ** 2).sum())
        Tsub_np = ((Tf - mu_np) @ V_np).astype(np.float32)
        sub = {"K": K, "var_captured": var, "var_captured_test_rows": var_te,
               "pr_audio": pr_a,
               "mu": torch.tensor(mu_np, dtype=torch.float32, device=DEV),
               "V": torch.tensor(V_np, dtype=torch.float32, device=DEV),
               "Tsub": torch.tensor(Tsub_np, device=DEV)}
        if a.pool:
            sub["pool_t"] = l2(torch.tensor(Tsub_np[pool_first], device=DEV))
        print(f"  align subspace: K={K} fit on {len(tr)} TRAIN rows, captures "
              f"{var * 100:.2f}% of audio deviation variance "
              f"({var_te * 100:.2f}% on the {len(te)} held-out test rows) | "
              f"centred audio participation ratio {pr_a:.2f} of {Tf.shape[1]}")
        if a.subspace_out:
            np.savez(a.subspace_out,
                     mu=mu_np.astype(np.float32), V=V_np.astype(np.float32),
                     K=K, var_captured=var, var_captured_test_rows=var_te,
                     pr_audio=pr_a, fit_rows=len(tr),
                     cache=a.cache, seed=a.seed,
                     built_utc=time.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"  froze mu/V -> {a.subspace_out}")

    dense_t = dense_v = None
    if a.ce > 0 and a.front != "enc":
        raise SystemExit("--ce requires --front enc")
    if a.dense > 0:
        if a.front != "enc":
            raise SystemExit("--dense requires --front enc")
        if a.dense_target == "mert":
            E, lengths = load_mert_source(a.mert_layer, a.cache)
        else:
            if a.cache in H14_CACHES:
                raise SystemExit("the 14-song caches support --dense-target mert only "
                                 "(env/mimi caches cover the 10 T songs)")
            E, lengths = load_dense_source(a.dense_target)
        dense_t, dense_v = dense_targets(d, E, lengths, a.lag)
        print(f"  dense targets ({a.dense_target}"
              f"{'' if a.dense_target != 'mert' else f' layer {a.mert_layer}'}"
              f", dim {E.shape[2]}): valid {dense_v.sum()}/{n} (lag {a.lag}s)")

    t0 = time.time()
    test_clf = None
    init_tag = "stock labram-base"
    best_v10 = None
    if a.front == "enc":
        if a.smoke_steps:
            smoke_timing(a, d, U, T, key, tr, pool_keys_t, pool_t, dense_t, dense_v)
            return
        zq, P, best, best_ep, ntr, n_skip, test_clf, init_tag, best_v10, te_preds = \
            train_encoder(a, d, U, T, key, lab, proto, tr, va, te,
                          pool_keys_t, pool_t, dense_t, dense_v, sub)
    else:
        bpx = np.asarray(d["bandpower"], np.float32)
        bpstd = (bpx - bpx[tr].mean(0)) / (bpx[tr].std(0) + 1e-6)
        if a.front == "bp":
            X = bpx
        elif a.front == "bpstd":
            X = bpstd
        elif a.front == "labram":
            X = np.asarray(d["labram"], np.float32)
        elif a.front == "fusion":
            lbx = np.asarray(d["labram"], np.float32)
            lbstd = (lbx - lbx[tr].mean(0)) / (lbx[tr].std(0) + 1e-6)
            X = np.hstack([bpstd, lbstd])
        elif a.front == "fusion_emb":
            if not a.emb_file:
                raise SystemExit("--front fusion_emb needs --emb-file")
            emb = np.load(a.emb_file).astype(np.float32)
            assert emb.shape[0] == n, f"emb rows {emb.shape[0]} != {n}"
            es = (emb - emb[tr].mean(0)) / (emb[tr].std(0) + 1e-6)
            X = np.hstack([bpstd, es])
        elif a.front == "fusion_emb_clamp":
            # early-fusion retry: floor the per-dim std of the l2-normalized
            # embeddings (5% of the median std) so near-constant dims cannot be
            # amplified into outliers that dominate the LayerNorm -- the suspected
            # cause of the first early-fusion failure (0.2373 < bp alone).
            if not a.emb_file:
                raise SystemExit("--front fusion_emb_clamp needs --emb-file")
            emb = np.load(a.emb_file).astype(np.float32)
            assert emb.shape[0] == n, f"emb rows {emb.shape[0]} != {n}"
            sd_e = emb[tr].std(0)
            floor = max(0.05 * float(np.median(sd_e)), 1e-6)
            n_clamped = int((sd_e < floor).sum())
            print(f"  clamp: {n_clamped}/{len(sd_e)} emb dims below std floor "
                  f"{floor:.5f}")
            es = (emb - emb[tr].mean(0)) / np.maximum(sd_e, floor)
            X = np.hstack([bpstd, es])
        zq, P, best, best_ep, ntr, n_skip = train_frozen(
            a, np.asarray(X, np.float32), T, key, lab, proto, tr, va, te,
            pool_keys_t, pool_t)
        te_preds = None
    res = evaluate(zq, torch.tensor(np.asarray(T, np.float32), device=DEV),
                   lab, proto if False else P, key, te, a.seed, tag=a.tag)
    n_way = len(np.asarray(proto))
    chance = 1.0 / n_way
    shape_eeg = cloud_shape(zq.detach().cpu().numpy())
    shape_aud = cloud_shape(np.asarray(T, np.float64)[np.asarray(te)])
    print(f"  shape test rows: EEG mutual_cos {shape_eeg['mutual_cos']:.4f} "
          f"PR {shape_eeg['participation_ratio']:.2f} | audio targets "
          f"mutual_cos {shape_aud['mutual_cos']:.4f} "
          f"PR {shape_aud['participation_ratio']:.2f}")
    screen = None
    if a.align_subspace > 0:
        # Evaluated mechanically against the frozen thresholds; a FAIL means stop here
        # and do not launch generation, whatever the retrieval number looks like.
        g = {"S1_shape": (shape_eeg["mutual_cos"], SCREEN_S1_MUTUAL_COS_MIN),
             "S2_retrieval_floor": (float(res["ten_way"]), SCREEN_S2_RETRIEVAL_MIN),
             "S3_anti_collapse": (shape_eeg["participation_ratio"],
                                  SCREEN_S3_PARTICIPATION_MIN)}
        screen = {nm: {"value": v, "min": m, "pass": bool(v >= m)}
                  for nm, (v, m) in g.items()}
        screen["all_pass"] = all(x["pass"] for x in screen.values())
        screen["audio_reference_same_rows"] = shape_aud
        screen["proceed_to_generation"] = screen["all_pass"]
        for nm in g:
            x = screen[nm]
            print(f"  screen {nm}: {x['value']:.4f} vs min {x['min']} -> "
                  f"{'PASS' if x['pass'] else 'FAIL'}")
        print("  SCREEN " + ("PASS -- generation may be launched"
                             if screen["all_pass"] else
                             "FAIL -- STOP, do not launch generation"))
    # per-dataset breakdown on the merged 14-way cache (T rows vs H rows)
    ds_break = None
    if a.cache in H14_CACHES:
        ds = np.asarray(d["dataset"])[np.asarray(te)]
        proto_acc = (zq @ P.T).argmax(1).cpu().numpy() == np.asarray(lab)[np.asarray(te)]
        ds_break = {"n_test_T": int((ds == 0).sum()), "n_test_H": int((ds == 1).sum())}
        for nm, m in (("T", ds == 0), ("H", ds == 1)):
            if m.sum():
                ds_break[f"retr_{nm}"] = float(proto_acc[m].mean())
                if te_preds is not None:
                    ds_break[f"clf_{nm}"] = float(
                        te_preds[m].__eq__(np.asarray(lab)[np.asarray(te)][m]).mean())
        print(f"  per-dataset: " + "  ".join(f"{k}={v:.4f}" if isinstance(v, float)
                                             else f"{k}={v}" for k, v in ds_break.items()))
    rec = {"tag": a.tag, "cache": a.cache, "front": a.front, "seed": a.seed,
           "epochs": a.epochs, "pool": bool(a.pool), "dense": a.dense,
           "dense_target": a.dense_target, "ema": a.ema,
           "ce": a.ce, "clip_w": a.clip_w,
           "align_subspace": int(a.align_subspace), "pos_w": a.pos_w,
           "subspace_var_captured": sub["var_captured"] if sub else None,
           "subspace_var_captured_test_rows":
               sub["var_captured_test_rows"] if sub else None,
           "subspace_pr_audio_centred": sub["pr_audio"] if sub else None,
           "subspace_out": a.subspace_out or None,
           "screen": screen,
           "shape_eeg": shape_eeg, "shape_audio_targets": shape_aud,
           "init": a.init if a.front == "enc" else "stock",
           "init_tag": init_tag if a.front == "enc" else None,
           "mert_layer": a.mert_layer if a.dense_target == "mert" else None,
           "lag": a.lag, "split": split_desc,
           "n_rows": int(n), "n_train": int(len(tr)), "n_test": int(len(te)),
           "ceiling_ten": ceil10, "chance_ten": chance, "n_way": n_way,
           "chance_kway": 1.0 / res["k"],
           "val_best_ten": best if a.front != "enc" or (a.ce == 0) else best_v10,
           "val_best_score": best,
           "val_best_metric": "clf-acc" if (a.front == "enc" and a.ce > 0) else "retrieval-10way",
           "val_best_ep": best_ep,
           "n_trainable": int(ntr), "skipped_steps": int(n_skip),
           "test_clf_acc": test_clf, "ds_break": ds_break,
           "train_seconds": round(time.time() - t0),
           **{k: v for k, v in res.items() if k != "tag"}}
    prev = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    prev.setdefault(a.tag, []).append(rec)
    OUT.write_text(json.dumps(prev, indent=2), encoding="utf-8")
    print(f"\n== {a.tag} seed {a.seed}: {n_way}-way {rec['ten_way']:.4f} "
          f"(ctrl {rec['ten_ctrl']:.4f}, {rec['ten_way']/chance:.2f}x) | "
          f"{rec['k']}-way {rec['kway']:.4f} (ctrl {rec['kway_ctrl']:.4f}, "
          f"{rec['kway']*rec['k']:.2f}x) | val best {best:.4f} "
          f"({rec['val_best_metric']}) @ep{best_ep}"
          + (f" | test clf acc {test_clf:.4f}" if test_clf is not None else "")
          + f" | shape EEG cos {shape_eeg['mutual_cos']:.3f}/PR "
            f"{shape_eeg['participation_ratio']:.1f} vs audio "
            f"{shape_aud['mutual_cos']:.3f}/{shape_aud['participation_ratio']:.1f}"
          + f" | {rec['train_seconds']}s")
    print(f"   wrote {OUT}")


def smoke_timing(a, d, U, T, key, tr, pool_keys_t, pool_t, dense_t, dense_v):
    """Time N steps of the EXACT training config (discipline rule 9)."""
    cm = json.loads((RESULTS / "nmedt_egi_to_1020.json").read_text(encoding="utf-8"))
    enc = LabramEncoder(device=str(DEV), freeze=False, strict=True,
                        norm="div100", input_chans=cm["input_chans"])
    head = Head(200).to(DEV)
    Tt = torch.tensor(np.asarray(T, np.float32), device=DEV)
    trt = torch.tensor(np.asarray(tr), device=DEV)
    opt = torch.optim.AdamW([{"params": enc.parameters(), "lr": a.lr_enc},
                             {"params": head.parameters(), "lr": a.lr_head}],
                            weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler()
    key_np = np.asarray(key)
    torch.cuda.synchronize()
    t0 = time.time()
    p = torch.randperm(len(trt), device=DEV)
    for s in range(a.smoke_steps):
        j = trt[p[s * a.micro:(s + 1) * a.micro]]
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast():
            bk = torch.as_tensor(key_np[j.cpu().numpy()], device=DEV)
            loss = info_nce_pool(head(enc(uv_of(U, j))), Tt[j],
                                 pool_t if a.pool else None,
                                 pool_keys_t if a.pool else None, bk)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(
            list(enc.parameters()) + list(head.parameters()), 1.0)
        scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        el = time.time() - t0
        mem = torch.cuda.max_memory_allocated() / 2**20
        print(f"  smoke step {s+1}: {el/(s+1):.3f} s/step (micro {a.micro}), "
              f"loss {float(loss):.4f}, peak GPU {mem:.0f} MB")
    spe = max(len(trt) // (a.micro * a.accum), 1)
    per_ep = el / a.smoke_steps * spe * a.accum
    print(f"  => steps/epoch {spe}, est {per_ep/60:.1f} min/epoch, "
          f"{a.epochs} epochs ~ {per_ep*a.epochs/3600:.1f} h")


def uv_of(U, j):
    return torch.tensor(np.asarray(U[j.cpu().numpy()], np.float32), device=DEV)


if __name__ == "__main__":
    main()
