"""Extract 40,336 NMED-H encoder contents from a pretrained auditory encoder."""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
sys.path.insert(0, str(ROOT / "code" / "auditory_encoder"))
import mainline_eval as ME  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()
uv = np.load(ME.M.UV, mmap_mode="r")
idx = ME.M.channel_idx()
model = ME.load_model(args.ckpt).to(ME.M.DEV)
E = ME.encode(uv, idx, model)
np.save(args.out, E)
print("saved", E.shape, E.dtype, args.out, flush=True)
