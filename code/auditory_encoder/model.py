"""Compact auditory EEG encoder with an explicit per-subject fiber module.

Not derived from LaBraM: 1 s patches, channel/time embeddings, a 4-layer
Transformer, masked-patch reconstruction, and a differentiable orthogonal
Procrustes fiber estimated from labelled support windows.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, heads, drop=0.1):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(4 * d, d), nn.Dropout(drop))

    def forward(self, x):
        h = self.n1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.n2(x))


class AuditoryEncoder(nn.Module):
    def __init__(self, d=128, depth=4, heads=4, n_ch=19, n_time=8, patch=200, drop=0.1,
                 spectral=False, spectral_only=False, sfreq=200):
        super().__init__()
        self.n_ch, self.n_time, self.patch = n_ch, n_time, patch
        self.spectral, self.spectral_only = spectral, spectral_only
        self.bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
        freqs = torch.fft.rfftfreq(patch, d=1 / sfreq)
        self.register_buffer("band_mask", torch.stack(
            [((freqs >= lo) & (freqs < hi)) for lo, hi in self.bands]))
        self.embed = nn.Linear(patch, d)
        self.spec_proj = nn.Linear(len(self.bands), d)
        self.ch_emb = nn.Parameter(torch.zeros(1, n_ch, 1, d))
        self.time_emb = nn.Parameter(torch.zeros(1, 1, n_time, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d))
        nn.init.trunc_normal_(self.ch_emb, std=0.02)
        nn.init.trunc_normal_(self.time_emb, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        # contrastive projection head: used only for the pretraining losses;
        # downstream uses the encoder content z, not this head.
        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(),
                                  nn.Linear(2 * d, d))
        self.audio_head = nn.Linear(d, 768)
        self.dec = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(),
                                 nn.Linear(4 * d, patch))
        self.subj_head = nn.Linear(d, 1)  # replaced by train script for N subjects

    def patchify(self, x):
        b, c, t = x.shape
        assert c == self.n_ch and t == self.n_time * self.patch
        return x.view(b, c, self.n_time, self.patch)

    def bandpower(self, p):
        power = torch.fft.rfft(p.float(), dim=-1).abs().pow(2)
        feats = [torch.log1p(power[..., m].mean(-1)) for m in self.band_mask]
        return torch.stack(feats, -1)  # [B, C, T, n_bands]

    def _tokens(self, x, mask=None):
        b = x.shape[0]
        p = self.patchify(x)
        raw = self.embed(p.reshape(b, self.n_ch * self.n_time, self.patch))
        raw = raw.view(b, self.n_ch, self.n_time, -1)
        tok = raw
        if self.spectral or self.spectral_only:
            spec = self.spec_proj(self.bandpower(p))
            tok = spec if self.spectral_only else tok + spec
        tok = tok + self.ch_emb + self.time_emb
        tok = tok.reshape(b, self.n_ch * self.n_time, -1)
        if mask is not None:
            tok = torch.where(mask[..., None], self.mask_token, tok)
        tok = torch.cat([self.cls.expand(b, -1, -1), tok], 1)
        for blk in self.blocks:
            tok = blk(tok)
        return self.norm(tok)

    def forward_tokens(self, x):
        return self._tokens(x)[:, 1:]

    def forward(self, x, mask=None):
        b = x.shape[0]
        p = self.patchify(x)
        tok = self._tokens(x, mask)
        content = tok[:, 0]
        recon = self.dec(tok[:, 1:])
        return content, recon, p.reshape(b, self.n_ch * self.n_time, self.patch)


def procrustes(a, b):
    """Orthogonal R minimising ||a R - b||, a/b [C, d].

    Thin-SVD form: the d x d SVD of a.T @ b is rank-deficient when C < d and
    often fails to converge.  This uses two C x d SVDs plus one C x C SVD.
    """
    a = a - a.mean(0, keepdim=True)
    b = b - b.mean(0, keepdim=True)
    ua, sa, vta = torch.linalg.svd(a, full_matrices=False)
    ub, sb, vtb = torch.linalg.svd(b, full_matrices=False)
    m = (sa[:, None] * (ua.T @ ub)) * sb[None, :]
    p, _, qt = torch.linalg.svd(m, full_matrices=False)
    return (vta.T @ p) @ (vtb.T @ qt.T).T


def fiber_apply(z_sup, z_q, n_sup, n_q):
    """z_sup [K*n_sup,d], z_q [K*n_q,d] -> aligned queries, per-subject R.

    Every subject has the same n_sup anchor stimuli, in the same order.
    The reference is the cross-subject consensus of anchor centroids (stop-grad).
    """
    k = z_sup.shape[0] // n_sup
    a = z_sup.view(k, n_sup, -1)
    ref = a.mean(0).detach()
    out, rs = [], []
    for i in range(k):
        ai = a[i]
        r = procrustes(ai, ref)
        rs.append(r)
        out.append((z_q.view(k, n_q, -1)[i] - ai.mean(0)) @ r)
    return torch.cat(out, 0), rs


def vicreg(z, var_w=1.0, cov_w=0.1):
    z = F.layer_norm(z, z.shape[-1:])
    std = z.std(0, unbiased=False)
    var = F.relu(1.0 - std).mean()
    zc = z - z.mean(0, keepdim=True)
    cov = zc.T @ zc / max(1, len(zc) - 1)
    off = cov - torch.diag_embed(torch.diagonal(cov))
    return var_w * var, cov_w * off.pow(2).sum() / z.shape[-1], var, off.pow(2).sum() / z.shape[-1]
