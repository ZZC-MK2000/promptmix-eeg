"""Decisive identity test for the EEG-conditioned generator (GPU, scaled up).

Question: does bridge(EEG) carry WHICH stimulus this was, once it drives the generator?

Two levels, each against its own chance line:
  window-level: among the N true references, is the clip's own 8s window rank 1?
  song-level:   is the clip's nearest true reference from the SAME SONG?
Arms:
  trained    = ep7 LoRA + trained clap_proj
  randomproj = pretrained decoder + freshly initialised clap_proj (control: learns nothing)

cond.npy has no song label, but its rows are an order-preserving subsequence of the
original 40336 bridge(EEG) rows, so exact float bytes recover the mapping (and song).

Run: python src/train/identity_test.py --n 30 --device cuda
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
OUT = ROOT / "results" / "gen_eegcond" / "identity"
sys.path.insert(0, str(M2 / "src" / "generate"))


def recover_song(cond):
    """Map each cond.npy row back to its original cache row -> song label.

    cond.npy is an order-preserving subsequence of the bridge(EEG) rows, so a single
    monotone walk on exact float bytes recovers the mapping.
    """
    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    song = z["song"].astype(int)
    eeg = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    proj = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu",
                       weights_only=True)
    W = proj["weight"].numpy(); b = proj["bias"].numpy()
    v = eeg @ W.T + b
    orig = (v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)).astype(np.float32)
    if len(orig) != len(song):
        raise RuntimeError(f"bridge rows {len(orig)} != song labels {len(song)}")

    j, mapped = 0, []
    for i in range(len(cond)):
        key = cond[i].tobytes()
        while j < len(orig) and orig[j].tobytes() != key:
            j += 1
        if j >= len(orig):
            return None, None            # could not align -> give up
        mapped.append(j)
        j += 1
    idx = np.array(mapped)
    return song[idx], idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--temp", type=float, default=0.9)
    ap.add_argument("--ckpt", default=str(ROOT / "results" / "gen_eegcond" / "best.pt"))
    ap.add_argument("--k", type=int, default=1, help="condition KV tokens; >1 scores a "
                    "multi-token checkpoint from finetune_gen_k.py")
    ap.add_argument("--outdir", default="", help="default: identity (k=1) or identity_kK")
    args = ap.parse_args()

    global OUT
    if args.outdir:
        OUT = Path(args.outdir)
    elif args.k > 1:
        OUT = ROOT / "results" / "gen_eegcond" / f"identity_k{args.k}"

    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    import soundfile as sf
    from musicgen_clap_finetune import (MusicGenCLAP, load_trained_model, load_clap,
                                        clap_embed_audio, SR)

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int64)
    group = np.load(DATA / "group.npy").astype(np.int64)

    uniq = np.unique(group)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(uniq))
    n_val = max(1, int(round(len(uniq) * 0.1)))
    val_g = set(uniq[order[:n_val]].tolist())
    va = np.where(np.array([g in val_g for g in group]))[0]
    # one row per distinct code-group: two picks sharing a group would decode to the SAME
    # target audio, which makes the retrieval task ill-posed.
    seen, one_per_group = set(), []
    for r in va:
        if group[r] not in seen:
            seen.add(int(group[r]))
            one_per_group.append(int(r))
    cand = np.array(one_per_group)
    pick = cand[np.linspace(0, len(cand) - 1, args.n, dtype=int)]
    print(f"held-out rows {len(va)} -> {len(cand)} distinct target groups; picking {len(pick)}")

    songs, orig_idx = recover_song(cond)
    if songs is None:
        print("!! 歌标签还原失败(字节不唯一) -> 只报窗级")
        sel_song = None
    else:
        sel_song = songs[pick]
        print(f"song labels recovered; picked songs distribution = "
              f"{np.bincount(sel_song, minlength=14).tolist()}")

    clap, cproc, csr = load_clap(dev)
    unit = lambda v: v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-8)

    # ---- true references (decode target codes) ----
    t0 = time.time()
    base = MusicGenCLAP(device=dev, use_lora=False)
    mg = base.mg
    TRUE = []
    for i in pick:
        w = mg.audio_encoder.decode(torch.from_numpy(codes[i]).unsqueeze(0).unsqueeze(0).to(dev),
                                    audio_scales=[None]).audio_values
        TRUE.append(np.asarray(w.cpu()).reshape(-1))
        sf.write(str(OUT / f"TRUE_row{i:05d}.wav"), TRUE[-1], SR)
    Tembed = unit(np.stack([clap_embed_audio(clap, cproc, csr, w, dev) for w in TRUE]))
    print(f"true refs done [{time.time()-t0:.0f}s]")

    results = {}
    for arm in ("trained", "randomproj"):
        t1 = time.time()
        if args.k > 1:
            from pilot_multitoken import CondWidthModel
            from finetune_gen_k import load_state as load_k
            model = CondWidthModel(dev, k_tokens=args.k)
            if arm == "trained":
                ck = load_k(model, args.ckpt)
                print(f"[{arm}] k={args.k} multi-token condition, loaded {Path(args.ckpt).name}")
            else:
                model.m.eval()
                print(f"[{arm}] WIDTH-MATCHED control: k={args.k} freshly initialised "
                      f"clap_proj + zero-effect LoRA (= pretrained decoder)")
        elif arm == "trained":
            model, ck = load_trained_model(args.ckpt, device=dev)
            print(f"[{arm}] epoch={ck.get('epoch')} val_loss={ck.get('val_loss'):.4f}")
        else:
            model = MusicGenCLAP(device=dev, use_lora=False)
            model.eval()
            print(f"[{arm}] control: pretrained decoder + random clap_proj")
        G, Graw = [], []
        for i in pick:
            wav = np.asarray(model.generate(torch.from_numpy(cond[i]),
                                            max_new_tokens=args.tokens,
                                            temperature=args.temp)).reshape(-1)
            Graw.append(wav)
            sf.write(str(OUT / f"gen_{arm}_row{i:05d}.wav"), wav, SR)
            G.append(clap_embed_audio(clap, cproc, csr, wav, dev))
            rms = float(np.sqrt(np.mean(wav ** 2)))
            print(f"  row{i}: len={len(wav)/SR:.2f}s rms={rms:.4f}", flush=True)
        Gem = unit(np.stack(G))
        results[arm] = (Gem, Graw)
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
        print(f"[{arm}] generated [{time.time()-t1:.0f}s]")

    # ---- scoring ----
    n = len(pick)
    for arm in ("trained", "randomproj"):
        Gm = results[arm][0]
        S = Gm @ Tembed.T
        own_rank, same_song_top1 = [], []
        for gi in range(n):
            o = np.argsort(-S[gi])
            own_rank.append(int(np.where(o == gi)[0][0]) + 1)
            if sel_song is not None:
                same_song_top1.append(int(sel_song[o[0]] == sel_song[gi]))
        d = np.array([S[i, i] - np.delete(S[i], i).mean() for i in range(n)])
        margin = float(d.mean())
        se = float(d.std(ddof=1) / np.sqrt(n))
        win_top1 = sum(1 for r in own_rank if r == 1) / n
        print(f"\n### arm={arm}  n={n}")
        print(f"  WINDOW-level: top-1={win_top1:.3f} ({win_top1*n:.0f}/{n})  "
              f"chance={1/n:.3f}   mean rank={np.mean(own_rank):.2f} (chance {(n+1)/2:.2f})")
        print(f"  margin own-others={margin:+.4f}  paired SE={se:.4f}  -> {margin/max(se,1e-9):+.2f} SD")
        if sel_song is not None and same_song_top1:
            # chance for same-song: sum p_k^2 over song distribution
            p = np.bincount(sel_song, minlength=14) / n
            chance_s = float((p * p).sum())
            ss = float(np.mean(same_song_top1))
            print(f"  SONG-level: nearest true ref is SAME SONG = {ss:.3f} "
                  f"({int(ss*n)}/{n})   chance={chance_s:.3f}")
        print(f"  ranks={own_rank}")

    json.dump({"n": n, "pick": [int(x) for x in pick],
               "songs": None if sel_song is None else [int(x) for x in sel_song],
               "temp": args.temp, "tokens": args.tokens,
               "ckpt": str(args.ckpt)},
              open(OUT / "test_spec.json", "w"), indent=2)
    print(f"\nclips + spec in {OUT}")


if __name__ == "__main__":
    main()
