"""v3: per-channel spectral tokens + channel attention + temporal transformer.

Instead of compressing channels into K spatial filters, each EEG channel contributes
a 5-band log-bandpower token per 1-s patch. A channel Transformer learns spatial
attention (a learned, data-dependent PC1-like mixing) independently at each time;
a temporal Transformer then models time. This preserves per-channel spectral detail
(the bandpower baseline's advantage) while adding learned spatial/temporal context.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


BANDS = ((1.0, 4.0), (4.0, 8.0), (8.0, 13.0), (13.0, 30.0), (30.0, 45.0))


class ChannelSpectralTokenizer(nn.Module):
    def __init__(self, n_ch, patch=200, sfreq=200.0, d=128, drop=0.1):
        super().__init__()
        self.n_ch, self.patch, self.n_bands, self.d = n_ch, patch, len(BANDS), d
        freqs = torch.fft.rfftfreq(patch, d=1.0 / sfreq)
        self.register_buffer("band_mask", torch.stack([(freqs >= lo) & (freqs < hi)
                                                       for lo, hi in BANDS]))
        self.proj = nn.Linear(self.n_bands, d)
        self.ch_emb = nn.Parameter(torch.zeros(1, n_ch, 1, d))
        self.time_emb = nn.Parameter(torch.zeros(1, 1, 1024, d))
        self.ch_drop = nn.Dropout(drop)
        nn.init.trunc_normal_(self.ch_emb, std=0.02)
        nn.init.trunc_normal_(self.time_emb, std=0.02)

    def forward(self, x):
        b, c, t = x.shape
        n = t // self.patch
        p = x[:, :, :n * self.patch].reshape(b, c, n, self.patch).float()
        power = torch.fft.rfft(p, dim=-1).abs().pow(2)
        bands = torch.stack([torch.log1p(power[..., m].mean(-1)) for m in self.band_mask],
                            dim=-1)  # [B,C,N,F]
        tok = self.proj(bands) + self.ch_emb + self.time_emb[:, :, :n]
        return self.ch_drop(tok), bands


class SpatialSpectralV3(nn.Module):
    def __init__(self, n_ch=19, d=128, depth=4, heads=4, ch_depth=2, patch=200,
                 sfreq=200.0, drop=0.1, audio_dim=768, spectral_skip=True):
        super().__init__()
        self.n_ch, self.d, self.patch = n_ch, d, patch
        self.tokenizer = ChannelSpectralTokenizer(n_ch, patch, sfreq, d, drop)
        self.ch_cls = nn.Parameter(torch.zeros(1, 1, d))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Parameter(torch.zeros(1, 1024, d))
        for p in (self.ch_cls, self.mask_token, self.cls, self.pos):
            nn.init.trunc_normal_(p, std=0.02)
        ch_layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, "gelu",
                                              batch_first=True, norm_first=True)
        self.ch_blocks = nn.TransformerEncoder(ch_layer, ch_depth)
        self.ch_norm = nn.LayerNorm(d)
        t_layer = nn.TransformerEncoderLayer(d, heads, 4 * d, drop, "gelu",
                                             batch_first=True, norm_first=True)
        self.t_blocks = nn.TransformerEncoder(t_layer, depth)
        self.norm = nn.LayerNorm(d)
        n_feat = n_ch * len(BANDS)
        self.spec_skip = nn.Linear(n_feat, d) if spectral_skip else None
        self.spectral_dec = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, n_feat))
        self.audio_head = nn.Linear(d, audio_dim)
        self.contrast_head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, x, mask=None):
        # x [B,C,T]; mask [B,N]
        tok, bands = self.tokenizer(x)  # [B,C,N,d], [B,C,N,F]
        b, c, n, d = tok.shape
        # channel attention at each time independently
        ch = tok.permute(0, 2, 1, 3).reshape(b * n, c, d)
        ch = torch.cat([self.ch_cls.expand(b * n, -1, -1), ch], 1)
        ch = self.ch_norm(self.ch_blocks(ch))
        tok = ch[:, 0].view(b, n, d)
        tok = tok + self.pos[:, :n]
        if mask is not None:
            tok = torch.where(mask[..., None], self.mask_token, tok)
        h = self.t_blocks(torch.cat([self.cls.expand(b, -1, -1), tok], 1))
        content = h[:, 0]
        if self.spec_skip is not None:
            content = content + self.spec_skip(bands.mean(2).reshape(b, -1))
        content = self.norm(content)
        recon = self.spectral_dec(h[:, 1:])
        target = bands.permute(0, 2, 1, 3).reshape(b, n, c * len(BANDS))
        return content, recon, target


def masked_spectral_loss(recon, target, mask):
    if mask is None or not mask.any():
        return recon.sum() * 0.0
    return F.mse_loss(recon[mask], target[mask])


def vicreg_terms(z, var_w=1.0, cov_w=0.1):
    z = F.layer_norm(z, z.shape[-1:])
    std = z.std(0, unbiased=False)
    var = F.relu(1.0 - std).mean()
    zc = z - z.mean(0, keepdim=True)
    cov = zc.T @ zc / max(1, len(zc) - 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    return var_w * var + cov_w * off.pow(2).sum() / z.shape[-1]


if __name__ == "__main__":
    m = SpatialSpectralV3()
    x = torch.randn(4, 19, 1600)
    mask = torch.rand(4, 8) < 0.25
    z, rec, target = m(x, mask)
    loss = masked_spectral_loss(rec, target, mask)
    loss.backward()
    print("params", sum(p.numel() for p in m.parameters()))
    print("content", tuple(z.shape), "recon", tuple(rec.shape), "target", tuple(target.shape),
          "loss", float(loss))
