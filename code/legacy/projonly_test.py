"""Decisive split: does EEG faithfulness need the LoRA, or only the learned interface?

Take ep16's trained clap_proj (the EEG -> 8 cross-attention tokens mapping) and attach it to
a PRISTINE pretrained decoder -- LoRA discarded. Same fixed sampling recipe for both arms.

  rough but faithful now, clean but infaithful here  -> faithfulness requires bending the
                                                        decoder: real trade-off, needs KL-to-pretrained
  clean AND faithful here                            -> the LoRA was the source of the artefacts;
                                                        adapting to EEG only needed the interface

margin = per-target (cos to its own stimulus minus mean cos to the other 59), averaged; t is
margin in units of its own standard error; chance top-1 = 1/60.
Zero-crossing rate = fraction of sign flips per sample, a texture proxy (higher = buzzier).

Run: python src/train/projonly_test.py
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
OUT = ROOT / "results" / "gen_eegcond" / "projonly"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
sys.path.insert(0, str(M2 / "src" / "generate"))

WAV_SR = 32000
RECIPE = dict(temperature=1.0, top_k=250)     # best recipe from sampling_sweep, fixed for both arms


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
    import soundfile as sf
    from pilot_multitoken import CondWidthModel
    from musicgen_clap_finetune import (MusicGenCLAP, load_clap, clap_embed_audio)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    ck = torch.load(str(ROOT / "results" / "gen_eegcond_k8" / "ckpt_ep16.pt"),
                    map_location="cpu", weights_only=True)
    proj_sd, k = ck["clap_proj"], int(ck["k"])
    print(f"ckpt epoch={ck.get('epoch')} val_loss={ck.get('val_loss'):.4f} k={k}")

    cond = np.load(DATA / "cond.npy").astype(np.float32)
    codes = np.load(DATA / "codes.npy").astype(np.int64)
    group = np.load(DATA / "group.npy").astype(np.int64)
    rows = pick_rows(group, 60)
    n = len(rows)

    clap, proc, csr = load_clap(device)
    unit = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

    base = MusicGenCLAP(device=device, use_lora=False)
    T = unit(np.stack([
        clap_embed_audio(clap, proc, WAV_SR,
                         np.asarray(base.mg.audio_encoder.decode(
                             torch.from_numpy(codes[r]).unsqueeze(0).unsqueeze(0).to(device),
                             audio_scales=[None]).audio_values.cpu()).reshape(-1), device)
        for r in rows]))
    t5d = base.mg.config.text_encoder.d_model
    del base
    if device == "cuda":
        torch.cuda.empty_cache()

    def build(arm):
        """arm 'projonly' = pristine decoder + trained interface; 'lora' = ep16 as-is."""
        import torch.nn as nn
        if arm == "projonly":
            m = MusicGenCLAP(device=device, use_lora=False)   # untouched pretrained decoder
            proj = nn.Sequential(nn.Linear(512, t5d), nn.GELU(), nn.Linear(t5d, t5d * k))
            proj.load_state_dict(proj_sd)
            m.clap_proj = proj.to(device).eval()
            m._k = k
        else:
            m = CondWidthModel(device, k_tokens=k)
            m.m.clap_proj.load_state_dict(proj_sd)
            m.m.mg.decoder.load_state_dict(ck["decoder_lora"], strict=False)
            m.m.eval()
            m = m.m
            m._k = k
        return m

    def cond_tokens(model, emb):
        out = model.clap_proj(emb.to(model.device))
        return out.view(out.shape[0], model._k, -1)

    def gen(model, emb):
        from transformers.modeling_outputs import BaseModelOutput
        enc = BaseModelOutput(last_hidden_state=cond_tokens(model, emb.unsqueeze(0)))
        out = model.mg.generate(encoder_outputs=enc, max_new_tokens=400,
                                do_sample=True, **RECIPE)
        return np.asarray(out.cpu()).reshape(-1)

    res = {}
    for arm in ("projonly", "lora"):
        t0 = time.time()
        model = build(arm)
        G, zcr, peak = [], [], []
        for j, r in enumerate(rows):
            w = gen(model, torch.from_numpy(cond[r]))
            G.append(clap_embed_audio(clap, proc, WAV_SR, w, device))
            zcr.append(float(np.mean(np.abs(np.diff(np.sign(w))))) / 2)
            peak.append(float(np.abs(w).max()))
            if j < 10:
                sf.write(str(OUT / f"{arm}_row{r:05d}.wav"), w, WAV_SR)
        Gm = unit(np.stack(G))
        S = Gm @ T.T
        own = np.array([S[j, j] for j in range(n)])
        others = np.array([np.delete(S[j], j).mean() for j in range(n)])
        d = own - others
        se = float(d.std(ddof=1) / np.sqrt(n))
        ranks = [int(np.where(np.argsort(-S[j]) == j)[0][0]) + 1 for j in range(n)]
        GG = Gm @ Gm.T
        res[arm] = dict(margin=float(d.mean()), se=se, t=float(d.mean() / se),
                        top1=sum(1 for x in ranks if x == 1) / n,
                        mean_rank=float(np.mean(ranks)), own=float(own.mean()),
                        others=float(others.mean()),
                        gg=float(GG[np.triu_indices(n, 1)].mean()),
                        zcr=float(np.mean(zcr)), peak=float(np.mean(peak)))
        r_ = res[arm]
        print(f"  {arm:<9} margin={r_['margin']:+.4f}±{r_['se']:.4f} t={r_['t']:+.2f} "
              f"top1={r_['top1']:.3f} rank={r_['mean_rank']:.2f} "
              f"过零率={r_['zcr']:.4f} 峰值均={r_['peak']:.2f} "
              f"GEN-v-GEN={r_['gg']:.4f} [{time.time()-t0:.0f}s]", flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    a, b = res["projonly"], res["lora"]
    print("\n===== 判读（参照: 原始未微调+文本 过零率 0.025-0.047）=====")
    print(f"只保留接口(丢 LoRA): 过零率 {a['zcr']:.4f}  margin {a['margin']:+.4f} (t={a['t']:+.2f})")
    print(f"ep16 原样(接口+LoRA): 过零率 {b['zcr']:.4f}  margin {b['margin']:+.4f} (t={b['t']:+.2f})")
    if a["zcr"] < b["zcr"] * 0.7:
        print("→ 毛刺主要来自 LoRA")
    else:
        print("→ 丢掉 LoRA 并没有让纹理变干净，毛刺另有来源")
    if a["t"] >= 2:
        print("→ 且只用接口的版本仍然显著贴住脑电")
    else:
        print("→ 但只用接口就不贴脑电了：贴合 EEG 确实需要改解码器")
    json.dump(res, open(OUT / "result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
