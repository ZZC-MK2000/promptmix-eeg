"""Self-contained experiment: does a CORPUS-MATCHED, class-distinct prompt library break the
chiptune label collapse found with the original 12-prompt library (54.6% of true windows landed
on "chiptune", top1-top2 margin only 0.047)?

We rebuild the library with 12 prompts that match an Indian/film/pop EEG corpus and are maximally
distinct, then measure, in one shot:
  1. coverage radius  (mixture / best-single / uniform / global mean vs the 60 held-out real windows)
  2. oracle label distribution + majority baseline  (argmax over true-audio CLAP, via cache match)
  3. a fresh EEG->oracle classifier held out by subject: overall acc, balanced acc, Cohen's kappa
and print a side-by-side vs the original library's numbers.

Nothing outside results/library_b/ is touched.

Run: python src/train/library_b_experiment.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
LIB = ROOT / "results" / "gen_eegcond" / "library_radius"
OLD = ROOT / "results" / "prompt_classifier"          # original K=12 artifacts
OUT = ROOT / "results" / "library_b"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

SR = 32000
K = 12
PROMPTS_B = [
    "upbeat indian bollywood dance song with dhol and bright synths",
    "melodic indian film ballad with lush strings and bamboo flute",
    "catchy pop song with airy female vocals and glossy production",
    "smooth r&b groove with warm bass and crisp rimshot drums",
    "energetic electronic dance music with a heavy four-on-the-floor kick",
    "mellow acoustic pop with fingerpicked guitar and soft brushes",
    "funky groove with slap bass and punchy brass stabs",
    "latin reggaeton beat with tropical synths and dembow rhythm",
    "dreamy ambient synth pad with slow evolving textures",
    "classic rock with a driving electric guitar riff and live drums",
    "hip hop trap beat with boomy 808 bass and rattling hi-hats",
    "cinematic orchestral action cue with pounding war drums",
]


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def proj_simplex(v):
    u = np.sort(v)[::-1]; css = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, len(v) + 1) > (css - 1))[0][-1]
    return np.maximum(v - (css[rho] - 1) / (rho + 1), 0)


def fit_simplex(L, t, iters=600):
    m = L.shape[0]; eta = 1.0 / (np.linalg.norm(L, 2) ** 2 + 1e-8)
    w = np.full(m, 1.0 / m)
    for _ in range(iters):
        r = L.T @ w - t
        w = proj_simplex(w - eta * (L @ r))
    return w


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    log = open(OUT / "log.txt", "a", encoding="utf-8")

    def say(s, **kw):
        print(s, flush=True); log.write(s + "\n"); log.flush()

    t0 = time.time()
    from musicgen_clap_finetune import MusicGenCLAP, load_clap, clap_embed_audio
    from arms_2x2 import embed_text, gen_from_enc

    base = MusicGenCLAP(device=device, use_lora=False); mg = base.mg; mg.eval()
    clap, proc, csr = load_clap(device)

    # 1) generate the new library (K prompts x 6 clips), embed each clip
    clips = []
    for k, p in enumerate(PROMPTS_B):
        h = embed_text(mg, p, device)
        for j in range(6):
            w = gen_from_enc(mg, h.unsqueeze(0), 400, 0.9, None, seed=1234 + 1000 * k + j)
            clips.append(clap_embed_audio(clap, proc, SR, w, device))
        say(f"  lib prompt{k} '{p[:34]}' [{time.time()-t0:.0f}s]", flush=True)
    lib_emb = unit(np.stack(clips).astype(np.float32))           # (72,512)
    owner = np.repeat(np.arange(K), 6)
    means = unit(np.stack([lib_emb[owner == k].mean(0) for k in range(K)]))   # (K,512)
    np.save(OUT / "library_emb.npy", lib_emb); np.save(OUT / "library_owner.npy", owner)

    # 2) coverage radius vs the 60 held-out real windows from library_radius
    Tembed = np.load(LIB / "true_emb.npy").astype(np.float32)     # (60,512)
    Tembed = unit(Tembed)
    glob = unit(lib_emb.mean(0, keepdims=True))[0]
    uni = unit(means.mean(0, keepdims=True))[0]
    mix, best, bmean = [], [], []
    for t in Tembed:
        w = fit_simplex(lib_emb, t)
        mix.append(float(unit(lib_emb.T @ w) @ t))
        best.append(float((lib_emb @ t).max()))
        bmean.append(float((means @ t).max()))
    say(f"\n[coverage, new lib] mixture={np.mean(mix):.4f} best_single_clip={np.mean(best):.4f} "
        f"best_prompt_mean={np.mean(bmean):.4f} uniform={float((Tembed @ uni).mean()):.4f} "
        f"global={float((Tembed @ glob).mean()):.4f}", flush=True)

    # 3) oracle over ALL rows via cache true-audio CLAP (no decode), aligned to cond.npy
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    group = np.load(DATA / "group.npy").astype(np.int64)
    N = len(cond)
    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    target = z["target"].astype(np.float32)
    eeg_aldm = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    proj = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu", weights_only=True)
    W = proj["weight"].numpy().astype(np.float32); b = proj["bias"].numpy().astype(np.float32)
    cond_all = unit(eeg_aldm @ W.T + b).astype(np.float32)
    key = {cond_all[i].tobytes(): i for i in range(len(cond_all))}
    te = np.zeros((N, 512), np.float32); matched = 0
    for j in range(N):
        i = key.get(cond[j].tobytes(), -1)
        if i >= 0:
            te[j] = target[i]; matched += 1
    cov = matched / N
    cos = unit(te) @ means.T
    oracle = cos.argmax(1).astype(np.int64)
    dist = np.bincount(oracle, minlength=K).tolist()
    say(f"\n[oracle, new lib] cache coverage={cov:.3f} dist={dist} "
        f"majority={max(dist)/matched:.4f} top2margin={(np.sort(cos,axis=1)[:,-1]-np.sort(cos,axis=1)[:,-2]).mean():.4f}",
        flush=True)
    np.save(OUT / "oracle_all.npy", oracle)

    # 4) fresh EEG->oracle classifier, held out by subject; overall/balanced acc + kappa
    from finetune_prompt_classifier import PromptClassifier
    uniq = np.unique(group); rng = np.random.default_rng(0)
    val_g = set(uniq[rng.permutation(len(uniq))[:max(1, int(round(len(uniq) * 0.1)))]].tolist())
    va = np.array([g in val_g for g in group])
    Xtr = torch.from_numpy(cond[~va]); ytr = torch.from_numpy(oracle[~va])
    Xva = torch.from_numpy(cond[va]); yva = torch.from_numpy(oracle[va])
    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
    model = PromptClassifier(512, 256, K, 0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    crit = torch.nn.CrossEntropyLoss()
    model.train(); it = iter(dl)
    for step in range(1, 3001):
        try: xb, yb = next(it)
        except StopIteration: it = iter(dl); xb, yb = next(it)
        loss = crit(model(xb.to(device)), yb.to(device))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xva.to(device)).argmax(1).cpu().numpy()
    yv = yva.numpy()
    overall = float((pred == yv).mean())
    recs = [(pred[yv == c] == c).mean() for c in range(K) if (yv == c).sum() > 0]
    bal = float(np.mean(recs))
    po = overall; pe = sum(((yv == c).mean()) * ((pred == c).mean()) for c in range(K))
    kap = float((po - pe) / (1 - pe + 1e-12))
    maj = float((yv == np.bincount(yv, minlength=K).argmax()).mean())
    say(f"\n[classifier, new lib] val_overall={overall:.4f} balanced={bal:.4f} kappa={kap:.4f} "
        f"majority={maj:.4f} chance={1/K:.4f}", flush=True)

    # side-by-side vs original
    try:
        oold = json.load(open(OLD / "eval_true_oracle.json"))
        say(f"\n=== ORIGINAL (12-prompt) === overall={oold['overall']:.4f} balanced={oold['balanced']:.4f} "
            f"kappa={oold['kappa']:.4f} majority={oold['acc_majority']:.4f}")
    except Exception as e:
        say(f"(original eval json not found: {e})")
    say(f"=== NEW corpus-matched lib === overall={overall:.4f} balanced={bal:.4f} kappa={kap:.4f} "
        f"majority={maj:.4f}")
    json.dump({"coverage": {"mixture": float(np.mean(mix)), "best_single": float(np.mean(best)),
               "best_prompt_mean": float(np.mean(bmean))},
               "oracle_dist": dist, "majority": float(max(dist) / matched),
               "cache_coverage": cov, "top2_margin": float((np.sort(cos,1)[:,-1]-np.sort(cos,1)[:,-2]).mean()),
               "val_overall": overall, "val_balanced": bal, "val_kappa": kap,
               "acc_majority": maj, "prompts": PROMPTS_B},
              open(OUT / "summary.json", "w"), indent=2)
    say(f"\ndone -> {OUT}  [{time.time()-t0:.0f}s]")
    log.close()


if __name__ == "__main__":
    main()
