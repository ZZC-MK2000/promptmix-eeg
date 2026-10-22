"""
Frozen LaBraM-Base encoder for ds002721 19-channel / 200 Hz EEG.

Interface facts taken from the source, not assumed:
  * `NeuralTransformerForMaskedEEGModeling.forward_features(x, input_chans, bool_masked_pos)`
  * input `x` is 4-D: [batch, n_electrodes, n_time_patches, patch_size]
    (5 s at 200 Hz -> 5 patches of 200 samples)
  * `input_chans = [0] + [standard_1020.index(ch) + 1 for ch in ch_names]`,
    element 0 addresses the CLS slot of `pos_embed` (1, 129, 200)
  * `time_embed` has only 16 slots -> n_time_patches must be <= 16
  * LaBraM's own normalisation is `data / 100` on MICROVOLTS
    (`LaBraM-main/data_processor/data_preprocess.py`)

Config is read from the checkpoint's own `args` / state-dict shapes:
  model=beit_base_patch200_1600_8k_vocab_align, patch_size=200, embed_dim=200,
  depth=12, num_heads=10 (k_norm bias is 20 = 200/10), mlp_ratio=4, qkv_bias=False,
  qk_norm=LayerNorm(1e-6), out_chans=8, layer_scale_init_value=0.1, vocab=8192
"""
import os
import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

LABRAM_DIR = Path(os.environ.get(
    "LABRAM_DIR",
    Path(__file__).resolve().parents[4] / "vendor" / "LaBraM-main"))
CKPT = Path(os.environ.get("LABRAM_CKPT", LABRAM_DIR / "checkpoints" / "labram-base.pth"))
if not LABRAM_DIR.exists():
    raise RuntimeError(f"LaBraM code not found at {LABRAM_DIR}; set LABRAM_DIR (see docs/DATA_AND_MODELS.md)")
if not CKPT.exists():
    raise RuntimeError(f"LaBraM checkpoint not found at {CKPT}; set LABRAM_CKPT")

# ds002721 channel order, as they appear in the EDF headers
CH19 = ["FP1", "FP2", "F7", "F3", "FZ", "F4", "F8", "T3", "C3", "CZ",
        "C4", "T4", "T5", "P3", "PZ", "P4", "T6", "O1", "O2"]

PATCH, FS, WIN_S = 200, 200.0, 5.0
N_PATCH = int(WIN_S * FS) // PATCH          # 5
OUT_DIM = 200


# ---- mainv2 workspace bootstrap (generated) -------------------------------
import os, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent            # src/<group>/
ROOT = HERE.parents[1]                            # mainv2/
DATA = ROOT / "data"                              # read-only caches
RESULTS = ROOT / "results"                        # result json + prereg json
for _g in ("data_pipeline", "encoder", "audio_target", "align", "measurement",
           "control", "reporting", "evaluation"):
    _p = str(ROOT / "src" / _g)
    if _p not in sys.path:
        sys.path.insert(0, _p)
def input_chans_for(ch_names=CH19):
    """[0] + LaBraM pos_embed indices (1-based) for each electrode."""
    sys.path.insert(0, str(Path(__file__).parent))
    from channels import load_standard_1020
    S = load_standard_1020()
    idx = [0]
    for c in ch_names:
        if c not in S:
            raise KeyError(f"{c} not in LaBraM standard_1020")
        idx.append(S.index(c) + 1)
    if max(idx) > 128:
        raise ValueError(f"channel index {max(idx)} exceeds pos_embed's 128 slots")
    return idx, S


class LabramEncoder(nn.Module):
    """Frozen LaBraM-Base -> 200-d mean-pooled patch-token feature."""

    def __init__(self, device="cuda", freeze=True, strict=True, norm="div100",
                 input_chans=None):
        super().__init__()
        if str(LABRAM_DIR) not in sys.path:
            sys.path.insert(0, str(LABRAM_DIR))
        from modeling_pretrain import NeuralTransformerForMaskedEEGModeling

        torch.manual_seed(0)
        self.backbone = NeuralTransformerForMaskedEEGModeling(
            EEG_size=int(WIN_S * FS), patch_size=PATCH, in_chans=1, out_chans=8,
            vocab_size=8192, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4.,
            qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6),
            qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.1,
            attn_head_dim=None, use_abs_pos_emb=True, use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False, init_std=0.02,
        )
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        sd = {k[len("student."):]: v for k, v in ck["model"].items()
              if k.startswith("student.")}
        missing, unexpected = self.backbone.load_state_dict(sd, strict=False)
        self.load_report = (list(missing), list(unexpected))
        if strict and (missing or unexpected):
            raise RuntimeError(f"checkpoint mismatch: missing={missing[:6]} "
                               f"unexpected={unexpected[:6]}")

        idx = list(input_chans) if input_chans is not None else input_chans_for()[0]
        if max(idx) > 128:
            raise ValueError(f"channel slot {max(idx)} exceeds pos_embed's 128")
        self.register_buffer("input_chans", torch.tensor(idx, dtype=torch.long))
        self.n_ch = len(idx) - 1
        self.norm = norm
        self.device_ = torch.device(device)
        self.to(self.device_)
        if freeze:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()
        self.frozen = freeze

    @property
    def out_dim(self):
        return OUT_DIM

    def _prep(self, x_uv):
        """[B, C, T] microvolts -> the 4-D tensor LaBraM expects.
        Patch count is derived from T, so 8 s windows work as well as 5 s."""
        x_uv = x_uv.to(self.device_)
        if x_uv.dim() == 3:
            b, c, t = x_uv.shape
            if t % PATCH:
                raise ValueError(f"{t} samples is not a multiple of {PATCH}")
            n_patch = t // PATCH
            if n_patch > 16:
                raise ValueError(f"{n_patch} patches exceeds time_embed's 16 slots")
            x = x_uv.reshape(b, c, n_patch, PATCH)
        else:
            x = x_uv
        if self.norm == "div100":
            x = x / 100.0
        elif self.norm == "zchan":
            # per-electrode z-score inside the window (dims 2,3 == time)
            m = x.mean(dim=(2, 3), keepdim=True)
            s = x.std(dim=(2, 3), keepdim=True) + 1e-6
            x = (x - m) / s
        elif self.norm == "zwin":
            m = x.mean(dim=(1, 3), keepdim=True)
            s = x.std(dim=(1, 3), keepdim=True) + 1e-6
            x = (x - m) / s
        elif self.norm == "none":
            pass
        else:
            raise ValueError(self.norm)
        return x

    def forward(self, x_uv, pool="mean"):
        x = self._prep(x_uv)
        b = x.shape[0]
        mask = torch.zeros((b, x.shape[1] * x.shape[2]), dtype=torch.bool,
                           device=x.device)
        if self.frozen:
            with torch.no_grad():
                h = self.backbone.forward_features(x, self.input_chans, mask)
        else:
            h = self.backbone.forward_features(x, self.input_chans, mask)
        t = h[:, 1:]                                   # patch tokens, electrode-major
        if pool == "mean":
            return t.mean(1)
        if pool == "cls":
            return h[:, 0]
        if pool == "tokens":
            return t
        raise ValueError(f"unknown pool {pool!r}")
