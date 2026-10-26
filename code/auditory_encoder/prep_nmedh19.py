"""Prepare the NMED-H cache as 19-channel, per-window z-scored 8 s windows."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
sys.path.insert(0, str(ROOT / "code" / "information"))
import mainline_eval_finetune as M  # noqa: E402

uv = np.load(M.UV, mmap_mode="r")
idx = M.channel_idx()
out = np.empty((len(uv), 19, 1600), np.float16)
for i in range(0, len(uv), 256):
    x = np.asarray(uv[i:i + 256][:, idx, :], dtype=np.float32)
    mu = x.mean(-1, keepdims=True); sd = x.std(-1, keepdims=True) + 1e-6
    out[i:i + 256] = np.clip((x - mu) / sd, -10, 10).astype(np.float16)
    if i % 4096 == 0:
        print(i, flush=True)
p = ROOT / "results" / "auditory_encoder" / "nmedh19.npy"
np.save(p, out)
print("saved", out.shape, p, flush=True)
