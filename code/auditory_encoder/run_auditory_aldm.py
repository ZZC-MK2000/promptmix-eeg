"""Run the mainline ALDM trainer with the pretrained auditory encoder as the EEG branch."""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, "D:/Projects/CLaBraM/main/clabram")
sys.path.insert(0, "D:/Projects/CLaBraM/promptmix-eeg/code/auditory_encoder")
sys.path.insert(0, "D:/Projects/CLaBraM/promptmix-eeg/code/information")
import nmedt_align_v2 as A  # noqa: E402
import mainline_eval_finetune as M  # noqa: E402
from model import AuditoryEncoder  # noqa: E402


class Wrap(nn.Module):
    def __init__(self, ckpt):
        super().__init__()
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        a = ck.get("args", {})
        self.enc = AuditoryEncoder(d=a.get("d", 128), depth=a.get("depth", 4),
                                   heads=a.get("heads", 4),
                                   spectral=a.get("spectral", False),
                                   spectral_only=a.get("spectral_only", False))
        sd = {k: v for k, v in ck["model"].items() if not k.startswith("subj_head")}
        self.enc.load_state_dict(sd, strict=False)
        self.proj = nn.Linear(a.get("d", 128), 200)
        self.idx = M.channel_idx()

    def forward(self, x, pool=None):
        if x.shape[1] != 19:
            x = x[:, self.idx, :]
        mu = x.mean(-1, keepdim=True); sd = x.std(-1, keepdim=True) + 1e-6
        x = torch.clamp((x - mu) / sd, -10, 10)
        if pool == "tokens":
            tok = self.proj(self.enc.forward_tokens(x))       # [B, 19*8, 200]
            b = tok.shape[0]
            t = tok.view(b, 19, 8, 200)
            m = t.mean(1, keepdim=True)
            # 82-channel expansion whose channel mean equals the 19-channel mean
            t82 = torch.cat([t.repeat_interleave(4, dim=1), m.repeat(1, 6, 1, 1)], dim=1)
            return t82.reshape(b, 82 * 8, 200)
        z, _, _ = self.enc(x)
        return self.proj(z)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--emb", required=True)
    ap.add_argument("--tag", default="aud_encoder_aldm_e5")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--rt", action="store_true", help="use the RT_aldm_s0 recipe flags")
    ap.add_argument("--smoke-steps", type=int, default=0)
    args = ap.parse_args()
    A.build_encoder = lambda init: (Wrap(args.ckpt).to(A.DEV), Path(args.ckpt).stem)
    argv = ["nmedt_align_v2.py", "--cache", "h14_a", "--front", "enc",
            "--tag", args.tag, "--epochs", str(args.epochs), "--save-emb", args.emb]
    if args.rt:
        argv += ["--pool", "--ema", "0.999", "--ce", "1", "--clip-w", "1",
                 "--dense", "0.5", "--dense-target", "mert", "--mert-layer", "-1",
                 "--init", "indomain"]
    if args.smoke_steps:
        argv += ["--smoke-steps", str(args.smoke_steps)]
    sys.argv = argv
    A.main()
