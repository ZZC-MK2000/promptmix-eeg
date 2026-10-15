"""Oracle-condition upper bound: feed the TARGET window's own real-audio laion vector.

Everything about the EEG path is taken out of the loop: the cross-attention condition is the
CLAP embedding of the very window the model is supposed to produce (unit-norm, 512-d). If the
output is still not musical, the generator (or the codec) is the limit, not the condition.

Two interfaces x two decoders, all four fed the same oracle condition, same fixed recipe:
  ep16_*     / mix12000_*  = the interface (512 -> 8 cross-attention tokens) learned by that run
  _frozen    = pristine pretrained decoder (LoRA discarded)
  _finetuned = that run's own LoRA-bent decoder

If even the pristine decoder with a perfect condition buzzes, the interface/condition port is the
problem; if the oracle condition sounds clean but bridge(EEG) doesn't, the loss is in the EEG path.

Reference rows printed automatically: the same interface driven with bridge(EEG) instead (from
mixed_eval's eval.json) and the metric ceiling (score the condition itself as if it were output).

margin = per-target (cos to its own stimulus window minus mean cos to the other 59), averaged;
SE = std/sqrt(n); t = margin/SE; chance top-1 = 1/n; zero-crossing rate is a texture proxy
higher = buzzier, with the codec-round-trip of the real windows (~0.0556) as the fair bar.

Run: python src/train/oracle_cond_probe.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
CKPT = ROOT / "results" / "gen_eegcond_mixed" / "ckpt_s12000.pt"
EVALJSON = ROOT / "results" / "gen_eegcond_mixed" / "eval" / "eval.json"
OUT = ROOT / "results" / "oracle_cond"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

SR = 32000
RECIPE = dict(temperature=1.0, top_k=250)


def pick_rows(group, n, seed=0, val_frac=0.1):
    uniq = np.unique(group)
    rng = np.random.default_rng(seed)
    n_val = max(1, int(round(len(uniq) * val_frac)))
    val_g = set(uniq[rng.permutation(len(uniq))[:n_val]].tolist())
    va = np.where(np.array([g in val_g for g in group]))[0]
    seen, one = set(), []
    for r in va:
        if int(group[r]) not in seen:
            seen.add(int(group[r]))
            one.append(int(r))
    one = np.array(one)
    return one[np.linspace(0, len(one) - 1, n, dtype=int)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=400)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import soundfile as sf
    from musicgen_clap_finetune import MusicGenCLAP, load_clap, clap_embed_audio

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)
    unit = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    codes = np.load(DATA / "codes.npy").astype(np.int64)
    group = np.load(DATA / "group.npy").astype(np.int64)
    cond = np.load(DATA / "cond.npy").astype(np.float32)      # bridge(EEG), for the reference arm
    rows = pick_rows(group, args.n)
    n = len(rows)

    clap, proc, csr = load_clap(device)
    base = MusicGenCLAP(device=device, use_lora=False)
    TRUE, WAVE = [], {}
    for r in rows:
        oc = base.mg.audio_encoder.decode(
            torch.from_numpy(codes[r]).unsqueeze(0).unsqueeze(0).to(device),
            audio_scales=[None]).audio_values
        w = np.asarray(oc.cpu()).reshape(-1)
        TRUE.append(w)
        WAVE[int(r)] = w
    T = unit(np.stack([clap_embed_audio(clap, proc, SR, w, device) for w in TRUE]))
    t5d = base.mg.config.text_encoder.d_model
    del base
    if device == "cuda":
        torch.cuda.empty_cache()

    def zcr(w):
        return float(np.mean(np.abs(np.diff(np.sign(w))))) / 2

    z_true = float(np.mean([zcr(w) for w in TRUE]))
    print(f"{n} held-out targets; 条件=目标窗真音 laion 向量（oracle）; 配方 {RECIPE}; "
          f"随机 top-1={1/n:.3f}; 天花板过零率={z_true:.4f}", flush=True)

    # metric ceiling: score the condition itself as if it were the output
    Sc = T @ T.T
    own_o = np.array([Sc[i, i] for i in range(n)])
    oth_o = np.array([np.delete(Sc[i], i).mean() for i in range(n)])
    d_o = own_o - oth_o
    se_o = float(d_o.std(ddof=1) / np.sqrt(n))
    print(f"  {'条件本身(上界)':<18} margin={d_o.mean():+.4f}±{se_o:.4f} t={d_o.mean()/se_o:+.2f} "
          f"top1=1.000 own={own_o.mean():.4f} others={oth_o.mean():.4f} 过零率={z_true:.4f}",
          flush=True)

    ARMS = [("ep16", ROOT / "results" / "gen_eegcond_k8" / "ckpt_ep16.pt"),
            ("mix12000", CKPT)]

    def build(ck, frozen_lora):
        k = int(ck.get("k", 8))
        m = MusicGenCLAP(device=device, use_lora=not frozen_lora)
        proj = nn.Sequential(nn.Linear(512, t5d), nn.GELU(), nn.Linear(t5d, t5d * k))
        proj.load_state_dict(ck["clap_proj"])
        m.clap_proj = proj.to(device).eval()
        m._k = k
        if not frozen_lora:
            m.mg.decoder.load_state_dict(ck["decoder_lora"], strict=False)
        m.mg.eval()
        return m

    def gen(model, emb):
        from transformers.modeling_outputs import BaseModelOutput
        e = torch.from_numpy(np.ascontiguousarray(emb)).float().unsqueeze(0).to(model.device)
        with torch.no_grad():
            tok = model.clap_proj(e)
            enc = BaseModelOutput(last_hidden_state=tok.view(1, model._k, -1))
            out = model.mg.generate(encoder_outputs=enc, max_new_tokens=args.tokens,
                                    do_sample=True, **RECIPE)
        return np.asarray(out.cpu()).reshape(-1)

    res = {}
    for tag, path in ARMS:
        ck = torch.load(str(path), map_location="cpu", weights_only=True)
        for frozen in (True, False):
            arm = f"{tag}_{'frozen' if frozen else 'finetuned'}"
            t0 = time.time()
            model = build(ck, frozen_lora=frozen)
            acc_emb, zs, pk = [], [], []
            for rep in range(max(1, args.reps)):
                G = []
                for q, r in enumerate(rows):
                    w = gen(model, T[q])
                    G.append(clap_embed_audio(clap, proc, SR, w, device))
                    if rep == 0:
                        zs.append(zcr(w)); pk.append(float(np.abs(w).max()))
                        if q < 8:
                            sf.write(str(OUT / f"{arm}_row{int(r):05d}.wav"), w, SR)
                acc_emb.append(np.stack(G))
            Gm = unit(np.mean(np.stack(acc_emb), axis=0))
            assert Gm.shape == T.shape, f"生成侧 {Gm.shape} 与参照侧 {T.shape} 不一致"
            S = Gm @ T.T
            own = np.array([S[i, i] for i in range(n)])
            others = np.array([np.delete(S[i], i).mean() for i in range(n)])
            d = own - others
            se = float(d.std(ddof=1) / np.sqrt(n))
            ranks = [int(np.where(np.argsort(-S[i]) == i)[0][0]) + 1 for i in range(n)]
            GG = Gm @ Gm.T
            res[arm] = dict(margin=float(d.mean()), se=se, t=float(d.mean() / se),
                            top1=sum(1 for x in ranks if x == 1) / n,
                            mean_rank=float(np.mean(ranks)),
                            own=float(own.mean()), others=float(others.mean()),
                            gg=float(GG[np.triu_indices(n, 1)].mean()),
                            zcr=float(np.mean(zs)), peak=float(np.mean(pk)),
                            per_target_margin=[float(x) for x in d])
            rr = res[arm]
            print(f"  {arm:<20} margin={rr['margin']:+.4f}±{se:.4f} t={rr['t']:+.2f} "
                  f"top1={rr['top1']:.3f} 秩={rr['mean_rank']:.1f} | 过零率={rr['zcr']:.4f} "
                  f"峰值={rr['peak']:.2f} GENvGEN={rr['gg']:.4f} own={rr['own']:.4f} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
            del model
            if device == "cuda":
                torch.cuda.empty_cache()

    ref = None
    if EVALJSON.exists():
        j = json.loads(EVALJSON.read_text(encoding="utf-8"))
        for rec in j["results"]:
            if rec["tag"] == "12000":
                ref = {kk: rec[kk] for kk in ("margin", "se", "t", "top1", "mean_rank",
                                             "song", "zcr", "gg", "own")}
                ref["arm"] = "同一权重、条件换成 bridge(EEG)"
                print(f"  参照 {ref['arm']}: margin={ref['margin']:+.4f}±{ref['se']:.4f} "
                      f"t={ref['t']:+.2f} top1={ref['top1']:.3f} 过零率={ref['zcr']:.4f} "
                      f"own={ref['own']:.4f}", flush=True)

    gap = {}
    for arm in res:
        gap[arm] = dict(own_vs_ceiling=float(own_o.mean() - res[arm]["own"]),
                        zcr_vs_true=float(res[arm]["zcr"] - z_true))

    json.dump({"n": n, "rows": [int(x) for x in rows], "reps": args.reps, "recipe": RECIPE,
               "true_zcr_reference": z_true, "chance_top1": 1 / n,
               "condition_ceiling": {"own": float(own_o.mean()), "others": float(oth_o.mean()),
                                     "margin": float(d_o.mean())},
               "results": res, "eeg_condition_reference": ref, "gap": gap},
              open(OUT / "result.json", "w"), indent=2)
    print(f"\n判读: own 距上界 {own_o.mean():.4f} 还差多少 = 生成器把条件'执行'出来多少; "
          f"过零率距天花板 {z_true:.4f} 还差多少 = 剩下的毛刺有多少来自生成器本身。")
    print(f"音频在 {OUT}（每个 arm 同样 8 个行号）")


if __name__ == "__main__":
    main()
