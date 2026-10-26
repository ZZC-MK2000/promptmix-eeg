"""Spatial-spectral EEG encoder.

Pipeline:
  raw EEG [B,C,T]
    -> learnable spatial filter bank (PC1-like; K components)
    -> 1-s Hann STFT / log bandpower tokens
    -> temporal Transformer (CLS + time tokens)
    -> content embedding [B,d] + masked spectral reconstruction

The model deliberately keeps the spatial stage explicit: the 2f audit showed that
channel averaging washes out the harmonic/beat response, while a PC1 spatial filter
retains it. Downstream users can inspect `spatial_weights()`.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))


class SpatialFilterBank(nn.Module):
    """K learnable spatial filters; each is centred and L2-normalised."""

    def __init__(self, n_ch: int, n_filters: int = 8):
        super().__init__()
        self.filters = nn.Parameter(torch.randn(n_filters, n_ch) * 0.02)
        self.n_ch = n_ch
        self.n_filters = n_filters

    def weights(self):
        w = self.filters - self.filters.mean(-1, keepdim=True)
        return w / (w.norm(dim=-1, keepdim=True) + 1e-6)

    def forward(self, x):
        # x: [B,C,T] -> [B,K,T]
        return torch.einsum("kc,bct->bkt", self.weights(), x)


class SpectralTokenizer(nn.Module):
    def __init__(self, n_filters: int, patch: int = 200, sfreq: float = 200.0,
                 n_bands: int = len(BANDS), d: int = 128):
        super().__init__()
        self.patch = patch
        self.n_filters = n_filters
        self.n_bands = n_bands
        freqs = torch.fft.rfftfreq(patch, d=1.0 / sfreq)
        self.register_buffer(
            "band_mask",
            torch.stack([(freqs >= lo) & (freqs < hi) for lo, hi in BANDS]),
        )
        self.proj = nn.Linear(n_filters * n_bands, d)

    def forward(self, z):
        # z: [B,K,T] -> tokens [B,N,d], features [B,N,K*n_bands]
        b, k, t = z.shape
        n = t // self.patch
        if n == 0:
            raise ValueError(f"input too short: T={t}, patch={self.patch}")
        p = z[:, :, : n * self.patch].reshape(b, k, n, self.patch)
        power = torch.fft.rfft(p.float(), dim=-1).abs().pow(2)
        feats = torch.stack(
            [torch.log1p(power[..., m].mean(-1)) for m in self.band_mask], dim=-1
        )  # [B,K,N,n_bands]
        feats = feats.permute(0, 2, 1, 3).reshape(b, n, k * self.n_bands)
        return self.proj(feats), feats


class SpatialSpectralEncoder(nn.Module):
    def __init__(self, n_ch: int = 19, d: int = 128, depth: int = 4, heads: int = 4,
                 n_filters: int = 8, patch: int = 200, sfreq: float = 200.0,
                 drop: float = 0.1, audio_dim: int = 768, spectral_skip: bool = False):
        super().__init__()
        self.n_ch, self.d, self.patch, self.sfreq = n_ch, d, patch, sfreq
        self.spectral_skip = spectral_skip
        self.spatial = SpatialFilterBank(n_ch, n_filters)
        self.tokenizer = SpectralTokenizer(n_filters, patch, sfreq, d=d)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, 1024, d))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=heads, dim_feedforward=4 * d, dropout=drop,
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(d)
        self.spectral_dec = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
            nn.Linear(2 * d, n_filters * len(BANDS)),
        )
        self.audio_head = nn.Linear(d, audio_dim)
        self.spec_skip = nn.Linear(n_filters * len(BANDS), d) if spectral_skip else None

    def spatial_weights(self):
        return self.spatial.weights().detach()

    def forward(self, x, mask=None, return_tokens=False):
        # x: [B,C,T]; mask: [B,N] bool, True = masked time token
        z = self.spatial(x)
        tok, feats = self.tokenizer(z)
        b, n, _ = tok.shape
        if n > self.pos.shape[1]:
            raise ValueError(f"too many tokens: {n} > {self.pos.shape[1]}")
        tok = tok + self.pos[:, :n]
        if mask is not None:
            if mask.shape != (b, n):
                raise ValueError(f"mask shape {tuple(mask.shape)} != {(b, n)}")
            tok = torch.where(mask[..., None], self.mask_token, tok)
        h = self.blocks(torch.cat([self.cls.expand(b, -1, -1), tok], dim=1))
        content = h[:, 0]
        if self.spec_skip is not None:
            content = content + self.spec_skip(feats.mean(1))
        content = self.norm(content)
        recon = self.spectral_dec(h[:, 1:])
        if return_tokens:
            return content, recon, feats, h[:, 1:]
        return content, recon, feats


def masked_spectral_loss(recon, feats, mask):
    if mask is None or not mask.any():
        return recon.sum() * 0.0
    return F.mse_loss(recon[mask], feats[mask])


def vicreg_terms(z, var_w=1.0, cov_w=0.1):
    z = F.layer_norm(z, z.shape[-1:])
    std = z.std(0, unbiased=False)
    var = F.relu(1.0 - std).mean()
    zc = z - z.mean(0, keepdim=True)
    cov = zc.T @ zc / max(1, len(zc) - 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    return var_w * var + cov_w * off.pow(2).sum() / z.shape[-1]


if __name__ == "__main__":
    m = SpatialSpectralEncoder()
    n = sum(p.numel() for p in m.parameters())
    x = torch.randn(4, 19, 1600)
    mask = torch.rand(4, 8) < 0.25
    z, rec, feat = m(x, mask)
    loss = masked_spectral_loss(rec, feat, mask)
    loss.backward()
    print("params", n)
    print("content", tuple(z.shape), "recon", tuple(rec.shape), "feat", tuple(feat.shape),
          "loss", float(loss))
