"""LaBraM-lite: per-channel 1-s raw-patch tokens + masked code/recon targets."""
import torch
import torch.nn as nn


class PatchEncoder(nn.Module):
    def __init__(self, n_ch=64, patch=120, n_time=8, d=128, depth=4, heads=4,
                 codebook=256, drop=0.1):
        super().__init__()
        self.n_ch, self.patch, self.n_time = n_ch, patch, n_time
        self.proj = nn.Linear(patch, d)
        self.ch = nn.Parameter(torch.zeros(n_ch, d))
        self.time = nn.Parameter(torch.zeros(n_time, d))
        self.mask_token = nn.Parameter(torch.zeros(1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=drop,
            activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d)
        self.code_head = nn.Linear(d, codebook)
        self.recon_head = nn.Linear(d, patch)

    def forward(self, x, mask=None):
        # x: [B,C,T]; mask: [B, C*T/patch] bool, True = masked token
        b, c, t = x.shape
        p = t // self.patch
        tok = x.reshape(b, p, c, self.patch)
        tok = self.proj(tok) + self.ch[None, None] + self.time[None, :p, None]
        tok = tok.reshape(b, p * c, -1)
        if mask is not None:
            tok = torch.where(mask[..., None], self.mask_token, tok)
        h = self.blocks(torch.cat([self.cls.expand(b, -1, -1), tok], 1))
        content = self.norm(h[:, 0])
        z = h[:, 1:]
        return content, self.recon_head(z), self.code_head(z)


def patch_targets(x, patch=120):
    b, c, t = x.shape
    p = t // patch
    return x.reshape(b, p, c, patch).reshape(b, p * c, patch)
