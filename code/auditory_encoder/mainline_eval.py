"""Mainline 36,036-query / 1,103-corpus bridge evaluation of the new encoders.

Same queries, corpus and held-subject ridge bridge as the bandpower reference
(.382 song hit / .146 val cosine), so this is the fair mainline comparison.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
sys.path.insert(0, str(ROOT / "code" / "auditory_encoder"))
sys.path.insert(0, str(ROOT / "code" / "information"))
import mainline_eval_finetune as M  # noqa: E402
from model import AuditoryEncoder  # noqa: E402

OUT = ROOT / "results" / "auditory_encoder"


def zscore_t(x):
    mu = x.mean(-1, keepdim=True)
    sd = x.std(-1, keepdim=True) + 1e-6
    return torch.clamp((x - mu) / sd, -10, 10)


@torch.no_grad()
def encode(uv, idx, model, bs=64):
    dev = next(model.parameters()).device
    out = []
    for i in range(0, len(uv), bs):
        x = torch.tensor(np.asarray(uv[i:i + bs][:, idx, :], dtype=np.float32), device=dev)
        z, _, _ = model(zscore_t(x))
        out.append(z.cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def load_model(ckpt):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = ck.get("args", {})
    m = AuditoryEncoder(d=a.get("d", 128), depth=a.get("depth", 4), heads=a.get("heads", 4),
                        spectral=a.get("spectral", False), spectral_only=a.get("spectral_only", False))
    sd = {k: v for k, v in ck["model"].items() if not k.startswith("subj_head")}
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--out", default=str(OUT / "mainline_eval.json"))
    args = ap.parse_args()
    cond = np.load(M.DATA / "cond.npy").astype(np.float32)
    group = np.load(M.DATA / "group.npy").astype(np.int64)
    zl = np.load(M.ZL, allow_pickle=True)
    song_all = zl["song"].astype(int); subj_all = np.array([str(x) for x in zl["subject"]])
    run_all = np.array([str(x) for x in zl["run"]]); seg_all = zl["seg2"].astype(int)
    target = zl["target"].astype(np.float32)
    uv = np.load(M.UV, mmap_mode="r")
    idx = M.channel_idx()
    # query -> cache-row mapping, identical to the bandpower / ALDM scripts
    eeg = np.load(M.M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    proj = torch.load(M.M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu", weights_only=True)
    ref = M.unit(eeg @ proj["weight"].numpy().T + proj["bias"].numpy()).astype(np.float32)
    j, mapped = 0, []
    for i in range(len(cond)):
        k = cond[i].tobytes()
        while j < len(ref) and ref[j].tobytes() != k:
            j += 1
        mapped.append(j); j += 1
    mapped = np.array(mapped)
    gvals = np.unique(group)
    first = {int(g): int(np.where(group == g)[0][0]) for g in gvals}
    rows = np.array([first[int(g)] for g in gvals])
    print(f"mapped {len(mapped)} queries, {len(rows)} corpus rows", flush=True)

    res = []
    za = np.load(M.M2 / "data" / "gate_nmedh14_ov2_a.npz", allow_pickle=True)
    res.append(M.bridge_eval(za["bandpower"].astype(np.float32), target, subj_all, song_all,
                             run_all, seg_all, group, mapped, rows, "bandpower"))
    for ck in args.ckpts:
        name = Path(ck).parent.name + "/" + Path(ck).stem
        model = load_model(ck).to(M.DEV)
        E = encode(uv, idx, model)
        print(f"{name}: encoded {E.shape}", flush=True)
        res.append(M.bridge_eval(E, target, subj_all, song_all, run_all, seg_all,
                                 group, mapped, rows, name))
        del model
    Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")
    for r in res:
        print(f"{r['tag']:40s} song {r['song_hit_raw']:.4f}  val_cos {r['val_cos']:.4f}  "
              f"smooth9 {r['smooth']['9']:.4f}  win {r['window_hit_raw']:.4f}  lam {r['lam']:g}",
              flush=True)


if __name__ == "__main__":
    main()
