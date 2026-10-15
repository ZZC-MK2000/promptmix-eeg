"""How far is the vector the generator actually receives from the answer's own vector?

cond.npy = bridge(EEG) (unit-norm laion space). The gate cache's `tgt` holds each row's real
stimulus-window laion embedding -- the oracle the probe fed in. Their cosine is the size of the
hole the fine-tuning is being asked to paper over, computed entirely from files already on disk.

Row order is recovered by the same byte-walk used everywhere else (cond rows were emitted in
gate order), so row i of cond is the same presentation as row i of tgt.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))
from mixed_eval import pick_rows      # noqa: E402

unit = lambda x: x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)

cond = np.load(ROOT / "data" / "eegcond" / "cond.npy").astype(np.float32)   # byte-exact: the walk
codes_group = np.load(ROOT / "data" / "eegcond" / "group.npy").astype(np.int64)

proj = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu", weights_only=True)
eeg = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
o = eeg @ proj["weight"].numpy().T + proj["bias"].numpy()
orig = unit(o)

z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
tgt = unit(z["target"].astype(np.float32))
song = z["song"].astype(int)

j, mapped = 0, []
for i in range(len(cond)):
    kk = cond[i].tobytes()
    while j < len(orig) and orig[j].tobytes() != kk:
        j += 1
    mapped.append(j)
    j += 1
mapped = np.array(mapped)
bad = int((mapped >= len(orig)).sum())
print(f"字节走位（按 cond 与桥输出逐字节配对）失配行数: {bad} / {len(mapped)}")
if bad:
    raise SystemExit("走位失败：cond 与 orig 不是同一份位级数据，别再用这个映射")

rows = pick_rows(codes_group, 60)
gap = np.array([float(cond[r] @ tgt[mapped[r]]) for r in rows])
own_cond = np.array([float(orig[mapped[r]] @ tgt[mapped[r]]) for r in rows])
oth = np.array([float(cond[r] @ np.delete(tgt[mapped[rows]], k, 0).mean(0))
                for k, r in enumerate(rows)])
print(f"60 条留 out 目标上：桥输出向量与'答案向量'（同一窗真音的 laion 向量）的余弦")
print(f"  mean={gap.mean():.4f}  中位={np.median(gap):.4f}  最小={gap.min():.4f}  最大={gap.max():.4f}")
print(f"  （ridge 直连版本 orig 与答案向量的余弦 mean={own_cond.mean():.4f}）")
print(f"  桥输出对'其余 59 个答案向量的平均'余弦 mean={oth.mean():.4f}")
print(f"  ⇒ 身份方向上的可用信号 = {gap.mean():.4f} − {oth.mean():.4f} = {gap.mean()-oth.mean():.4f}", flush=True)
ss = song[mapped[rows]]
p = np.bincount(ss, minlength=14) / len(ss)
print(f"  这 60 条覆盖 {len(np.unique(ss))} 首歌；按歌分布随机同歌概率={float((p*p).sum()):.3f}")
