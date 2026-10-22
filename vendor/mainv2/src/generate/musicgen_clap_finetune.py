"""Fine-tune MusicGen-small with CLAP audio-tower embeddings as the sole
conditioning signal.

Pipeline
--------
1. Scan a directory of audio files (FMA or any music collection).
2. For each file, extract one or more 10-second segments.
3. Compute CLAP audio-tower embedding (512-d, L2-normalised) per segment.
4. Fine-tune MusicGen decoder: freeze T5 text encoder + Encodec audio encoder,
   add a small projection Linear(512 → 768) that maps CLAP embedding into the
   T5 output space, and train the decoder's cross-attention to use it.
5. At inference, feed a CLAP embedding → project → generate → decode to audio.

Usage
-----
    # Step 1: prepare data (compute CLAP embeddings for all FMA segments)
    python musicgen_clap_finetune.py --prepare --audio-dir D:/Projects/CLaBraM/data_external/fma_small/fma_small

    # Step 2: train
    python musicgen_clap_finetune.py --train --epochs 10 --batch-size 8

    # Step 3: inference from a CLAP embedding file
    python musicgen_clap_finetune.py --infer --cond-emb path/to/embedding.npy --out out.wav

    # Step 4: inference from EEG embedding (needs projection to CLAP space)
    python musicgen_clap_finetune.py --infer --eeg-emb path/to/eeg_emb.npy --out out.wav
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import glob
import time
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "audio_target"))

DATA = Path(os.environ.get("MAINV2_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("MAINV2_RESULTS", ROOT / "results"))
MODEL_DIR = Path(os.environ.get("MUSICGEN_DIR", ROOT.parent / "models" / "musicgen-small"))
FMA_DIR = Path(os.environ.get("FMA_DIR", ROOT.parent / "data_external" / "fma_small" / "fma_small"))
CLAP_ID = "laion/clap-htsat-unfused"
SEG_SEC = 10.0
SR = 32000
SEG_SAMPLES = int(SEG_SEC * SR)


# ── CLAP embedding ──────────────────────────────────────────────────────────

def load_clap(device="cuda"):
    from transformers import ClapModel, ClapProcessor
    proc = ClapProcessor.from_pretrained(CLAP_ID)
    model = ClapModel.from_pretrained(CLAP_ID).eval().to(device)
    sr = int(proc.feature_extractor.sampling_rate)
    return model, proc, sr


@torch.no_grad()
def clap_embed_audio(model, proc, sr, wav_np, device):
    """wav_np: mono float32 [T], any sample rate. Returns 512-d L2-normalised."""
    import torchaudio
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=0)
    t = torch.from_numpy(wav_np).float().unsqueeze(0).to(device)
    clap_sr = 48000
    if sr != clap_sr:
        t = torchaudio.functional.resample(t, sr, clap_sr)
    if t.shape[-1] > int(clap_sr * SEG_SEC):
        t = t[:, :int(clap_sr * SEG_SEC)]
    inputs = proc(audio=t.squeeze().cpu().numpy(), sampling_rate=clap_sr, return_tensors="pt")
    inputs = {k: v.to(device) if hasattr(v, 'to') else v for k, v in inputs.items()}
    out = model.get_audio_features(**inputs)
    if hasattr(out, 'pooler_output'):
        emb = out.pooler_output
    elif hasattr(out, 'last_hidden_state'):
        emb = out.last_hidden_state.mean(dim=1)
    elif isinstance(out, torch.Tensor):
        emb = out
    else:
        emb = out[0] if isinstance(out, (tuple, list)) else out.last_hidden_state[:, 0]
    if emb.ndim > 2:
        emb = emb.mean(dim=1)
    return F.normalize(emb, dim=-1).cpu().numpy().flatten()


# ── Data preparation ────────────────────────────────────────────────────────

def load_audio(path, target_sr=SR):
    import soundfile as sf
    import torchaudio
    wav, sr = sf.read(path, dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav_t = torch.from_numpy(wav).unsqueeze(0)
    if sr != target_sr:
        wav_t = torchaudio.functional.resample(wav_t, sr, target_sr)
    return wav_t.squeeze(0).numpy()


def prepare_data(audio_dir, out_dir):
    """Scan audio_dir for .mp3/.wav/.flac, split into 10s segments,
    compute CLAP embedding for each, save to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exts = ("*.mp3", "*.wav", "*.flac", "*.ogg")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(audio_dir, "**", ext), recursive=True))
    files.sort()
    print(f"Found {len(files)} audio files in {audio_dir}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, proc, clap_sr = load_clap(device)

    all_embs = []
    all_meta = []
    seg_count = 0

    for i, fpath in enumerate(files):
        try:
            wav = load_audio(fpath, SR)
        except Exception as e:
            print(f"  SKIP {fpath}: {e}")
            continue

        duration = len(wav) / SR
        if duration < SEG_SEC:
            continue

        n_segs = int(duration / SEG_SEC)
        for s in range(n_segs):
            start = int(s * SEG_SAMPLES)
            end = start + SEG_SAMPLES
            seg = wav[start:end].astype(np.float32)

            emb = clap_embed_audio(model, proc, clap_sr, seg, device)
            all_embs.append(emb)
            all_meta.append({
                "file": os.path.relpath(fpath, audio_dir),
                "seg_idx": s,
                "start_sec": s * SEG_SEC,
            })
            seg_count += 1

        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(files)}] {seg_count} segments so far")

    embs = np.stack(all_embs).astype(np.float32)
    np.save(out_dir / "fma_clap_embs.npy", embs)
    with open(out_dir / "fma_clap_meta.json", "w") as f:
        json.dump(all_meta, f, indent=1)
    print(f"Saved {embs.shape[0]} segments, embeddings {embs.shape} to {out_dir}")
    return embs, all_meta


def prepare_codes(data_dir, audio_dir=None):
    """Pre-compute Encodec codes for all segments in a prepared data directory.

    Reads meta.json to find audio segments, runs each through MusicGen's audio
    encoder, and saves the resulting codes as .npy files in data_dir/codes/.
    """
    data_dir = Path(data_dir)
    meta_path = data_dir / "fma_clap_meta.json"
    codes_dir = data_dir / "codes"
    codes_dir.mkdir(parents=True, exist_ok=True)

    with open(meta_path) as f:
        meta = json.load(f)
    print(f"Found {len(meta)} segments in {meta_path}")

    existing = set(p.stem for p in codes_dir.glob("*.npy"))
    if len(existing) == len(meta):
        print(f"All {len(meta)} codes already exist in {codes_dir}, skipping.")
        return

    if audio_dir is None:
        audio_dir = str(FMA_DIR)
    audio_root = Path(audio_dir)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    local = str(MODEL_DIR) if MODEL_DIR.exists() else "facebook/musicgen-small"
    from transformers import MusicgenForConditionalGeneration
    mg = MusicgenForConditionalGeneration.from_pretrained(local).to(device)
    mg.audio_encoder.eval()
    num_codebooks = mg.decoder.num_codebooks
    print(f"MusicGen loaded: {num_codebooks} codebooks, audio_root={audio_root}")

    for i, m in enumerate(meta):
        out_path = codes_dir / f"{i:05d}.npy"
        if str(i) in existing:
            continue
        fpath = str(audio_root / m["file"])
        try:
            wav = load_audio(fpath, SR)
        except Exception as e:
            print(f"  SKIP seg {i} ({fpath}): {e}")
            continue
        start = int(m["start_sec"] * SR)
        seg = wav[start:start + SEG_SAMPLES].astype(np.float32)
        seg_t = torch.from_numpy(seg).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            enc_out = mg.audio_encoder.encode(seg_t)
            codes = enc_out.audio_codes[0].cpu().numpy()
        np.save(out_path, codes.astype(np.int16))
        if (i + 1) % 200 == 0:
            print(f"  [{i+1}/{len(meta)}] codes saved")

    print(f"Done. {len(list(codes_dir.glob('*.npy')))} code files in {codes_dir}")


# ── Dataset ─────────────────────────────────────────────────────────────────

class FMACondDataset(Dataset):
    """Each item: (audio_waveform_10s, clap_embedding_512)."""

    def __init__(self, audio_dir, emb_path, meta_path, split="train", train_frac=0.9):
        self.audio_dir = audio_dir
        self.embs = np.load(emb_path)
        with open(meta_path) as f:
            self.meta = json.load(f)
        assert len(self.embs) == len(self.meta)

        n = len(self.meta)
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        split_idx = int(n * train_frac)
        if split == "train":
            self.indices = perm[:split_idx]
        else:
            self.indices = perm[split_idx:]
        print(f"{split}: {len(self.indices)} segments")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        m = self.meta[real_idx]
        fpath = os.path.join(self.audio_dir, m["file"])
        wav = load_audio(fpath, SR)
        start = int(m["start_sec"] * SR)
        seg = wav[start:start + SEG_SAMPLES].astype(np.float32)
        emb = self.embs[real_idx]
        return torch.from_numpy(seg), torch.from_numpy(emb)


class FMACondCodesDataset(Dataset):
    """Each item: (pre_computed_codes, clap_embedding_512).
    codes shape: [num_codebooks, T'] loaded from .npy files."""

    def __init__(self, codes_dir, emb_path, split="train", train_frac=0.9):
        self.codes_dir = Path(codes_dir)
        self.embs = np.load(emb_path)
        n = len(self.embs)
        self.indices_by_split = {}
        rng = np.random.RandomState(42)
        perm = rng.permutation(n)
        split_idx = int(n * train_frac)
        self.indices_by_split["train"] = perm[:split_idx]
        self.indices_by_split["val"] = perm[split_idx:]
        self.indices = self.indices_by_split[split]
        print(f"{split}: {len(self.indices)} segments (from pre-computed codes)")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        codes = np.load(self.codes_dir / f"{real_idx:05d}.npy")
        codes_t = torch.from_numpy(codes.astype(np.int64)).squeeze(0)
        emb = self.embs[real_idx]
        return codes_t, torch.from_numpy(emb)


# ── Model wrapper ───────────────────────────────────────────────────────────

class MusicGenCLAP(nn.Module):
    """MusicGen decoder with CLAP embedding as cross-attention condition."""

    def __init__(self, device="cuda", use_lora=False, lora_r=16, lora_alpha=32, lora_dropout=0.05):
        super().__init__()
        from transformers import MusicgenForConditionalGeneration
        local = str(MODEL_DIR) if MODEL_DIR.exists() else "facebook/musicgen-small"
        self.mg = MusicgenForConditionalGeneration.from_pretrained(local)

        t5_dim = self.mg.config.text_encoder.d_model
        print(f"T5 d_model = {t5_dim}, CLAP dim = 512")

        self.clap_proj = nn.Sequential(
            nn.Linear(512, t5_dim),
            nn.GELU(),
            nn.Linear(t5_dim, t5_dim),
        )

        for p in self.mg.text_encoder.parameters():
            p.requires_grad = False
        for p in self.mg.audio_encoder.parameters():
            p.requires_grad = False

        self.use_lora = use_lora
        if use_lora:
            # Freeze the whole decoder, then inject low-rank adapters into its
            # attention/FFN projections. Preserves pretrained generation while
            # letting the decoder learn to attend to CLAP conditioning.
            for p in self.mg.decoder.parameters():
                p.requires_grad = False
            from peft import LoraConfig, inject_adapter_in_model
            cfg = LoraConfig(
                r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"],
                bias="none",
            )
            self.mg.decoder = inject_adapter_in_model(cfg, self.mg.decoder)
            n_lora = sum(p.numel() for p in self.mg.decoder.parameters() if p.requires_grad)
            print(f"LoRA on decoder: r={lora_r} alpha={lora_alpha}, trainable LoRA params={n_lora/1e6:.2f}M")
        else:
            for p in self.mg.decoder.parameters():
                p.requires_grad = True

        for p in self.clap_proj.parameters():
            p.requires_grad = True

        self.mg.text_encoder.eval()
        self.mg.audio_encoder.eval()

        self.device = device
        self.to(device)

    def _delayed_inputs(self, audio_codes):
        """Build delay-pattern decoder_input_ids + labels matching what generate() feeds.

        MusicGen's decoder sees codebook c shifted right by c positions (see
        transformers build_delay_pattern_mask). Training must use the SAME layout or
        the model learns a factorization it never sees at inference (-> noise/silence).

        codes [B, C, T] -> (decoder_input_ids [B*C, T-1], labels [B, T-C, C])
        """
        B, C, T = audio_codes.shape
        pad = getattr(self.mg.config.decoder, "pad_token_id", None)
        if pad is None:
            pad = self.mg.config.decoder.vocab_size
        L = T + C - 1
        D = torch.full((B, C, L), pad, dtype=audio_codes.dtype, device=audio_codes.device)
        for c in range(C):
            D[:, c, c:c + T] = audio_codes[:, c, :]
        dec_input = D[:, :, :T - 1].reshape(B * C, -1)          # positions 0..T-2
        # label at position p (=C-1+t), codebook c = codes[c, p-c+1] = codes[c, C+t-c]
        t = torch.arange(T - C, device=audio_codes.device)      # [T-C]
        cc = torch.arange(C, device=audio_codes.device)         # [C]
        cidx = (C + t[:, None] - cc[None, :]).t()               # [C, T-C]
        idx = cidx.unsqueeze(0).expand(B, C, T - C)             # [B, C, T-C]
        labels = torch.gather(audio_codes, 2, idx).permute(0, 2, 1)  # [B, T-C, C]
        return dec_input, labels

    def forward(self, audio_wav, clap_emb):
        """
        audio_wav: [B, T] float32 waveform at 32kHz
        clap_emb:  [B, 512] float32 CLAP embedding
        Returns: loss (scalar)
        """
        audio_wav = audio_wav.to(self.device)
        clap_emb = clap_emb.to(self.device)
        bsz = audio_wav.shape[0]
        num_codebooks = self.mg.decoder.num_codebooks

        with torch.no_grad():
            audio_3d = audio_wav.unsqueeze(1)
            enc_out = self.mg.audio_encoder.encode(audio_3d)
            audio_codes = enc_out.audio_codes[0]

        cross_attn_out = self.clap_proj(clap_emb)
        cross_attn_out = cross_attn_out.unsqueeze(1)

        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=cross_attn_out)

        dec_input, labels = self._delayed_inputs(audio_codes)

        out = self.mg(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=dec_input,
            labels=labels,
            return_dict=True,
        )
        return out.loss

    def forward_from_codes(self, audio_codes, clap_emb):
        """
        audio_codes: [B, num_codebooks, T] int64 pre-computed Encodec codes
        clap_emb:    [B, 512] float32 CLAP embedding
        Returns: loss (scalar)
        """
        audio_codes = audio_codes.to(self.device)
        clap_emb = clap_emb.to(self.device)
        bsz = audio_codes.shape[0]
        num_codebooks = audio_codes.shape[1]

        cross_attn_out = self.clap_proj(clap_emb)
        cross_attn_out = cross_attn_out.unsqueeze(1)

        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=cross_attn_out)

        dec_input, labels = self._delayed_inputs(audio_codes)

        out = self.mg(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=dec_input,
            labels=labels,
            return_dict=True,
        )
        return out.loss

    @torch.no_grad()
    def generate(self, clap_emb, max_new_tokens=512, temperature=1.0):
        """Generate audio conditioned on CLAP embedding.
        clap_emb: [B, 512] or [512]
        Returns: waveform numpy array (generate auto-decodes via Encodec)
        """
        if clap_emb.ndim == 1:
            clap_emb = clap_emb.unsqueeze(0)
        clap_emb = clap_emb.to(self.device)

        cross_attn_out = self.clap_proj(clap_emb).unsqueeze(1)

        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=cross_attn_out)

        output = self.mg.generate(
            encoder_outputs=encoder_outputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
        )
        return output.squeeze().cpu().numpy()


def _save_ckpt(model, path, epoch, val_loss):
    """Save clap_proj always; for LoRA only the trainable adapter tensors (small),
    for full fine-tune the whole decoder state_dict (large)."""
    payload = {"epoch": epoch, "val_loss": val_loss,
               "clap_proj": model.clap_proj.state_dict()}
    if getattr(model, "use_lora", False):
        payload["use_lora"] = True
        sd = model.mg.decoder.state_dict()
        payload["decoder_lora"] = {k: v for k, v in sd.items() if "lora_" in k}
    else:
        payload["use_lora"] = False
        payload["decoder"] = model.mg.decoder.state_dict()
    torch.save(payload, path)


def load_trained_model(ckpt_path, device="cuda"):
    """Reconstruct a MusicGenCLAP from a checkpoint, LoRA-aware. Used by eval/inference."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model = MusicGenCLAP(device=device, use_lora=bool(ckpt.get("use_lora", False)))
    model.clap_proj.load_state_dict(ckpt["clap_proj"])
    if ckpt.get("use_lora") and "decoder_lora" in ckpt:
        missing, unexpected = model.mg.decoder.load_state_dict(ckpt["decoder_lora"], strict=False)
        # missing = all base weights (expected, they stay pretrained); no unexpected is the real check
        if unexpected:
            print(f"WARNING: {len(unexpected)} unexpected LoRA keys on load")
    elif "decoder" in ckpt:
        model.mg.decoder.load_state_dict(ckpt["decoder"])
    model.eval()
    print(f"Loaded checkpoint epoch {ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.4f} "
          f"({'LoRA' if ckpt.get('use_lora') else ('proj+decoder' if 'decoder' in ckpt else 'proj only')})")
    return model, ckpt


# ── Training ────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    data_dir = Path(args.data_dir)
    emb_path = data_dir / "fma_clap_embs.npy"
    meta_path = data_dir / "fma_clap_meta.json"

    if not emb_path.exists():
        print(f"ERROR: {emb_path} not found. Run --prepare first.")
        return

    audio_dir = getattr(args, 'train_audio_dir', None) or str(FMA_DIR)
    codes_dir = data_dir / "codes"
    use_codes = codes_dir.exists() and len(list(codes_dir.glob("*.npy"))) > 0
    if use_codes:
        print(f"Using pre-computed codes from {codes_dir}")
        train_ds = FMACondCodesDataset(str(codes_dir), str(emb_path), split="train")
        val_ds = FMACondCodesDataset(str(codes_dir), str(emb_path), split="val")
    else:
        train_ds = FMACondDataset(audio_dir, str(emb_path), str(meta_path), split="train")
        val_ds = FMACondDataset(audio_dir, str(emb_path), str(meta_path), split="val")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = MusicGenCLAP(device=device, use_lora=getattr(args, "lora", False))
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {n_trainable/1e6:.1f}M")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * args.epochs)

    ckpt_dir = RESULTS / "musicgen_clap_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (codes_or_wav, emb) in enumerate(train_loader):
            if use_codes:
                loss = model.forward_from_codes(codes_or_wav, emb)
            else:
                loss = model(codes_or_wav, emb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1
            loss_val = loss.item()
            del loss

            if global_step % 50 == 0:
                elapsed = time.time() - t0
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step {global_step}  loss {loss_val:.4f}  "
                      f"lr {lr_now:.2e}  elapsed {elapsed/60:.1f} min", flush=True)

        avg_train = epoch_loss / max(n_batches, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for codes_or_wav, emb in val_loader:
                if use_codes:
                    vl = model.forward_from_codes(codes_or_wav, emb)
                else:
                    vl = model(codes_or_wav, emb)
                val_loss += vl.item()
                n_val += 1
        avg_val = val_loss / max(n_val, 1)
        torch.cuda.empty_cache()

        print(f"Epoch {epoch+1}/{args.epochs}: train_loss={avg_train:.4f}  val_loss={avg_val:.4f}")

        if avg_val < best_val:
            best_val = avg_val
            _save_ckpt(model, ckpt_dir / "best_clap_proj.pt", epoch, avg_val)
            print(f"  → saved best checkpoint (val_loss={avg_val:.4f})")

        if (epoch + 1) % 5 == 0:
            _save_ckpt(model, ckpt_dir / f"clap_proj_ep{epoch+1}.pt", epoch, avg_val)

    total_time = time.time() - t0
    print(f"Training done in {total_time/3600:.1f} hours. Best val_loss: {best_val:.4f}")

    result = {
        "epochs": args.epochs,
        "best_val_loss": best_val,
        "total_time_min": total_time / 60,
        "trainable_params": n_trainable,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
    }
    with open(RESULTS / "musicgen_clap_train_result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Result saved to {RESULTS / 'musicgen_clap_train_result.json'}")


# ── Inference ───────────────────────────────────────────────────────────────

def infer(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicGenCLAP(device=device)

    ckpt_path = RESULTS / "musicgen_clap_ckpt" / "best_clap_proj.pt"
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found. Train first.")
        return
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    model.clap_proj.load_state_dict(ckpt["clap_proj"])
    if "decoder" in ckpt:
        model.mg.decoder.load_state_dict(ckpt["decoder"])
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f} (proj+decoder)")
    else:
        print(f"Loaded checkpoint from epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f} (proj only)")

    model.eval()

    if args.cond_emb:
        emb = np.load(args.cond_emb)
        if emb.ndim == 2:
            emb = emb[0]
        emb = torch.from_numpy(emb).float().to(device)
    elif args.eeg_emb:
        eeg_emb = np.load(args.eeg_emb)
        if eeg_emb.ndim == 2:
            eeg_emb = eeg_emb[0]
        eeg_emb = torch.from_numpy(eeg_emb).float().to(device)
        emb = _eeg_to_clap(eeg_emb, device)
    else:
        print("ERROR: specify --cond-emb or --eeg-emb")
        return

    print(f"Generating {args.max_tokens} tokens...")
    wav_np = model.generate(emb, max_new_tokens=args.max_tokens, temperature=args.temperature)
    print(f"Generated waveform shape: {wav_np.shape}")

    import soundfile as sf
    out_path = args.out or str(RESULTS / "musicgen_clap_output.wav")
    sf.write(out_path, wav_np, SR)
    print(f"Saved to {out_path} ({len(wav_np)/SR:.1f}s)")
    return out_path


def _eeg_to_clap(eeg_emb, device):
    """Project EEG embedding (AudioLDM CLAP space) to laion CLAP space.

    Uses a learned linear projection trained on paired audio embeddings.
    Falls back to identity if no projection is available.
    """
    proj_path = RESULTS / "eeg_to_clap_proj.pt"
    if proj_path.exists():
        proj = torch.load(proj_path, map_location=device, weights_only=True)
        W = proj["weight"]
        b = proj.get("bias", torch.zeros(W.shape[0]))
        return F.normalize(eeg_emb @ W.T + b, dim=-1)
    else:
        print("WARNING: No EEG→CLAP projection found. Using raw EEG embedding as proxy.")
        print("  Run train_eeg_to_clap_proj.py first for proper projection.")
        return F.normalize(eeg_emb, dim=-1)


# ── Quick demo: generate from NMED audio CLAP embeddings ────────────────────

def demo_from_nmed(args):
    """Generate audio conditioned on CLAP embeddings extracted from NMED-H audio.
    This tests the pipeline without needing EEG at all."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MusicGenCLAP(device=device)

    ckpt_path = RESULTS / "musicgen_clap_ckpt" / "best_clap_proj.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.clap_proj.load_state_dict(ckpt["clap_proj"])
        if "decoder" in ckpt:
            model.mg.decoder.load_state_dict(ckpt["decoder"])
            print(f"Loaded checkpoint epoch {ckpt['epoch']} (proj+decoder)")
        else:
            print(f"Loaded checkpoint epoch {ckpt['epoch']} (proj only)")
    else:
        print("No checkpoint found, using random projection (untrained)")

    model.eval()

    clap_model, clap_proc, clap_sr = load_clap(device)

    audio_dir = DATA / "nmedh_audio"
    wav_files = sorted(glob.glob(str(audio_dir / "*.wav")))
    if not wav_files:
        print(f"No audio files in {audio_dir}")
        return

    out_dir = RESULTS / "musicgen_clap_demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    for wf in wav_files[:4]:
        name = Path(wf).stem
        print(f"\nProcessing {name}...")
        wav = load_audio(wf, SR)

        n_segs = min(3, int(len(wav) / SR / SEG_SEC))
        for s in range(n_segs):
            start = int(s * SEG_SEC * SR)
            seg = wav[start:start + SEG_SAMPLES].astype(np.float32)
            emb = clap_embed_audio(clap_model, clap_proc, clap_sr, seg, device)
            emb_t = torch.from_numpy(emb).float().to(device)

            gen_np = model.generate(emb_t, max_new_tokens=args.max_tokens, temperature=args.temperature)

            out_path = out_dir / f"{name}_seg{s}_generated.wav"
            import soundfile as sf
            sf.write(str(out_path), gen_np, SR)
            print(f"  seg {s}: emb norm={np.linalg.norm(emb):.4f}, "
                  f"gen rms={np.sqrt(np.mean(gen_np**2)):.4f}, saved {out_path}")

            orig_path = out_dir / f"{name}_seg{s}_original.wav"
            sf.write(str(orig_path), seg, SR)

    print(f"\nDemo outputs in {out_dir}")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prepare-codes", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--lora", action="store_true",
                        help="LoRA fine-tune the decoder instead of full-parameter (preserves pretrained generation)")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--audio-dir", type=str, default=str(FMA_DIR))
    parser.add_argument("--data-dir", type=str, default=str(DATA / "fma_prepared"))
    parser.add_argument("--train-audio-dir", type=str, default=None,
                        help="Audio directory for training (overrides --audio-dir for train)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cond-emb", type=str, default=None)
    parser.add_argument("--eeg-emb", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=1.0)
    args = parser.parse_args()

    if args.prepare:
        prepare_data(args.audio_dir, args.data_dir)
    elif args.prepare_codes:
        prepare_codes(args.data_dir, args.audio_dir)
    elif args.train:
        train(args)
    elif args.infer:
        infer(args)
    elif args.demo:
        demo_from_nmed(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
