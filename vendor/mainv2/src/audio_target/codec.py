"""Frozen audio codec: kyutai/mimi.

Why the standalone model and not pocket-tts's Mimi:
    the released pocket-tts checkpoint ships a ZEROED Mimi ENCODER (keys present,
    load_state_dict reports missing=0, but the values are all zero).  Only its decoder
    is usable.  So training targets must be produced by the standalone `kyutai/mimi`.

Contract
--------
    encode(wav[N], sr)      -> latents [T, 32] at 12.5 Hz   (continuous, pre-RVQ path)
    decode(latents[T, 32])  -> wav[N] at 24 kHz

MUSIN-G audio is 8 kHz, so it is resampled to 24 kHz on the way in; that is a hard
information ceiling (a first-smoke-test limitation, not a conclusion-grade setup).
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torchaudio
import soundfile as sf

MODEL_ID = "kyutai/mimi"
SR = 24000
FRAME_RATE = 12.5
LATENT_DIM = 512          # decoder input width (hidden_size), NOT codebook_dim (256)


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
def _first_tensor(obj):
    """Unwrap the tensor out of either a plain Tensor or a HF ModelOutput."""
    if torch.is_tensor(obj):
        return obj
    if hasattr(obj, "last_hidden_state"):
        return obj.last_hidden_state
    if hasattr(obj, "audio_values"):
        return obj.audio_values
    if isinstance(obj, (tuple, list)) and obj:
        return _first_tensor(obj[0])
    raise TypeError(f"cannot extract a tensor from {type(obj)}")


class Mimi:
    def __init__(self, device: str = "cuda", dtype=torch.float32):
        from transformers import MimiModel
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.model = MimiModel.from_pretrained(MODEL_ID).to(self.device).eval()
        self.model.requires_grad_(False)
        cfg = self.model.config
        self.frame_rate = float(getattr(cfg, "frame_rate", FRAME_RATE))
        self.sampling_rate = int(getattr(cfg, "sampling_rate", SR))
        # NOTE: the DECODER input width is `hidden_size` (512).  `codebook_dim` (256)
        # is the RVQ embedding width and is NOT what the decoder consumes.
        self.hidden_size = int(getattr(cfg, "hidden_size", 512))
        self.codebook_dim = int(getattr(cfg, "codebook_dim", 256))
        self.codebook_size = int(getattr(cfg, "codebook_size", 2048))
        self.num_quantizers = int(getattr(cfg, "num_quantizers", 32))
        self.latent_dim = self.hidden_size

    # ------------------------------------------------------------------ io
    @staticmethod
    def load_audio(path, target_sr=SR):
        """-> mono float32 at target_sr, shape [N]."""
        wav, sr = sf.read(path, always_2d=True, dtype="float32")
        wav = wav.mean(axis=1)                       # stereo -> mono
        t = torch.from_numpy(wav)[None]              # [1, N]
        if sr != target_sr:
            t = torchaudio.functional.resample(t, sr, target_sr)
        return t[0].numpy(), target_sr

    # -------------------------------------------------------------- encode
    @torch.no_grad()
    def encode(self, wav, sr=SR):
        """wav [N] -> continuous encoder latents.

        NOTE: in this transformers version `MimiEncoder.forward` returns a plain Tensor
        (shape [B, D, T']) rather than a dataclass, so do not touch .last_hidden_state.
        Also `codebook_dim` is 256 -- the "32" in my earlier notes is the NUMBER OF
        QUANTIZERS, not the latent width.  The encoder width is what the decoder wants.
        """
        if sr != self.sampling_rate:
            t = torch.from_numpy(np.asarray(wav, dtype=np.float32))[None]
            t = torchaudio.functional.resample(t, sr, self.sampling_rate)
            wav = t[0].numpy()
        x = torch.from_numpy(np.asarray(wav, dtype=np.float32))[None, None]
        x = x.to(self.device)
        enc = self.model.encoder(x)
        if hasattr(enc, "last_hidden_state"):
            enc = enc.last_hidden_state
        return enc[0].transpose(0, 1).cpu().numpy()          # [T', D]

    @torch.no_grad()
    def encode_codes(self, wav, sr=SR):
        """wav [N] -> discrete RVQ codes [num_quantizers, T] at 12.5 Hz."""
        x = self._as_input(wav, sr)
        return self.model.encode(x).audio_codes[0]         # [Q, T]

    @torch.no_grad()
    def encode_latent(self, wav, sr=SR):
        """wav [N] -> CONTINUOUS decoder-ready latent [T, D] at 12.5 Hz.

        This is option 甲: z_q = quantizer.decode(codes).  It is exactly what the
        decoder consumes, so it is a continuous target that the frozen decoder can
        reproduce losslessly, and it keeps CALM's continuous+flow design.
        """
        x = self._as_input(wav, sr)
        codes = self.model.encode(x).audio_codes           # [1, Q, T]
        z = self.model.quantizer.decode(codes)             # [1, D, T]
        if z.dim() == 3 and z.shape[1] == self.hidden_size:
            z = z.transpose(1, 2)                          # -> [1, T, D]
        return z[0].cpu().numpy()                          # [T, D]

    @torch.no_grad()
    def decode(self, latents, target_len=None):
        """CONTINUOUS latent [T, D] (or [B, T, D]) @12.5 Hz -> wav [N] @ sampling_rate.

        `target_len` truncates the tail: the decoder emits whole frames
        (T * 2 * 960 samples) whereas `MimiModel.decode` trims to the padding-mask
        length.  Without the trim the output runs ~50 ms long for a 100 s clip.

        Replicates `MimiModel._decode_frame` exactly:

            embeddings = quantizer.decode(codes)
            embeddings = upsample(embeddings)                 # 12.5 Hz -> 25 Hz
            embeddings = decoder_transformer(embeddings.T)
            outputs    = decoder(embeddings)

        Skipping `upsample` / `decoder_transformer` yields audio at HALF the expected
        sample count -- that was the length-mismatch bug.
        """
        z = torch.as_tensor(np.asarray(latents, dtype=np.float32))
        if z.dim() == 2:
            z = z[None]
        z = z.to(self.device)
        if z.shape[1] != self.latent_dim and z.dim() == 3 and z.shape[2] == self.latent_dim:
            z = z.transpose(1, 2)                          # [B, T, D] -> [B, D, T]
        if z.shape[1] != self.latent_dim:
            raise ValueError(
                f"latent dim mismatch: got {tuple(z.shape)}, expected feature dim "
                f"{self.latent_dim} (decoder input width)")

        emb = self.model.upsample(z)                       # [B, D, 2T]
        dout = self.model.decoder_transformer(emb.transpose(1, 2), return_dict=True)
        emb = _first_tensor(dout).transpose(1, 2)          # [B, D, 2T]
        out = self.model.decoder(emb)
        out = _first_tensor(out).detach().cpu()
        if out.dim() == 3:                                 # [B, C, N]
            wav = out[0, 0].numpy()
        elif out.dim() == 2:                               # [C, N]
            wav = out[0].numpy()
        else:
            wav = out.numpy().reshape(-1)
        if target_len is not None:
            wav = wav[: int(target_len)]
        return wav

    # ------------------------------------------------------------ helpers
    def _as_input(self, wav, sr):
        if sr != self.sampling_rate:
            t = torch.from_numpy(np.asarray(wav, dtype=np.float32))[None]
            t = torchaudio.functional.resample(t, sr, self.sampling_rate)
            wav = t[0].numpy()
        return torch.from_numpy(np.asarray(wav, dtype=np.float32))[None, None].to(self.device)

    @staticmethod
    def _to_BTD(z):
        if torch.is_tensor(z):
            pass
        elif hasattr(z, "last_hidden_state"):
            z = z.last_hidden_state
        else:
            z = z[0]
        if z.dim() == 2:
            z = z[None]
        return z


def _smoke():
    print("=" * 92)
    print("Mimi API probe")
    m = Mimi()
    print(f"  device={m.device}  sampling_rate={m.sampling_rate}  "
          f"frame_rate={m.frame_rate}  latent_dim={m.latent_dim}")
    print(f"  config keys: {[k for k in dir(m.model.config) if not k.startswith('_')][:24]}")

    cfg = m.model.config
    print(f"  codebook_dim={getattr(cfg,'codebook_dim',None)}  "
          f"codebook_size={getattr(cfg,'codebook_size',None)}  "
          f"num_quantizers={getattr(cfg,'num_quantizers',None)}  "
          f"hidden_size={getattr(cfg,'hidden_size',None)}  "
          f"encodec_frame_rate={getattr(cfg,'encodec_frame_rate',None)}")

    p = r"d:\Projects\CLaBraM\dataset\musin-g\Code\ESongs\6.esh.wav"
    wav, sr = m.load_audio(p)
    dur = len(wav) / sr
    print(f"\n  loaded {os.path.basename(p)}: {dur:.3f}s @ {sr} Hz (source 8000 Hz)")

    codes = m.encode_codes(wav, sr)
    print(f"  RVQ codes : {tuple(codes.shape)}  rate = {codes.shape[-1]/dur:.3f} Hz")

    z = m.encode_latent(wav, sr)
    print(f"  CONTINUOUS latent (option 甲): {z.shape}  "
          f"rate = {z.shape[0]/dur:.3f} Hz  dtype={z.dtype}")
    print(f"  latent stats: mean={z.mean():+.4f} std={z.std():.4f} "
          f"min={z.min():+.3f} max={z.max():+.3f}")
    print(f"  distinct frames/values: unique rows "
          f"{len(np.unique(z.round(4), axis=0))}/{z.shape[0]}")

    # decoder-exactness: decode(encode_latent(x)) must equal the full model round trip
    x = torch.from_numpy(wav.astype(np.float32))[None, None].to(m.device)
    ref = m.model(x).audio_values[0, 0].cpu().numpy()
    got = m.decode(z)
    print(f"\n  full round trip len = {len(ref)}   decode(latent) len = {len(got)}")
    n = min(len(ref), len(got))
    print(f"  max |decode(latent) - model(x)| = {np.abs(ref[:n] - got[:n]).max():.3e}  "
          f"=> decoder-exact: {np.allclose(ref[:n], got[:n], atol=1e-6)}")
    a = wav[:n]
    snr = 10 * np.log10(np.sum(a**2) / max(np.sum((a - ref[:n]) ** 2), 1e-30))
    print(f"  quantised round-trip SNR = {snr:.2f} dB")


if __name__ == "__main__":
    _smoke()
