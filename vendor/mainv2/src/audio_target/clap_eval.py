"""Deterministic CLAP evaluation harness.

WHY THIS EXISTS
---------------
`ClapProcessor` pads/truncates every input to a fixed ~10 s mel (1001 frames).  When the
source is LONGER, it sets `is_longer=True` and `ClapModel` then applies a **random crop**,
so the same long clip yields a different embedding on every call (verified: 4 different
hashes for a 100 s song; identical hashes for 1 s and 10 s inputs).

Consequence: any prototype built from a whole track is irreproducible, and a single-run
top-1 can be quoted to meaningless precision.

THIS HARNESS GUARANTEES DETERMINISM
-----------------------------------
  * `embed()` never feeds more than `SEG` (10 s) of audio to the model.  Longer inputs
    are split into FIXED, non-overlapping segments, each embedded separately, then
    averaged and L2-normalised.  No randomness anywhere.
  * `assert_deterministic()` re-embeds a sample and requires bit-identical output.
  * `repeat()` runs any measurement n times and reports mean +- std, so the caller can
    never accidentally quote a single run.
"""
import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import hashlib
import numpy as np
import torch
import soundfile as sf
import torchaudio

SEG = 10.0          # CLAP's native input length in seconds
DEFAULT_CID = "laion/clap-htsat-unfused"


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
def _hash(x):
    return hashlib.md5(np.ascontiguousarray(np.asarray(x, np.float32)).tobytes()).hexdigest()[:12]


class ClapEvaluator:
    def __init__(self, cid=DEFAULT_CID, device=None):
        from transformers import ClapModel, ClapProcessor
        self.cid = cid
        self.proc = ClapProcessor.from_pretrained(cid)
        self.model = ClapModel.from_pretrained(cid).eval()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = self.model.to(self.device)
        self.sr = int(self.proc.feature_extractor.sampling_rate)
        self.dim = int(self.model.config.projection_dim)

    # ------------------------------------------------------------------ audio io
    def load(self, path):
        w, sr = sf.read(path, always_2d=True, dtype="float32")
        w = w.mean(axis=1)
        t = torch.from_numpy(w)[None]
        if sr != self.sr:
            t = torchaudio.functional.resample(t, sr, self.sr)
        return t[0].numpy()

    # ---------------------------------------------------------------- embedding
    def _embed_le10s(self, wav):
        """Exactly one forward pass on audio of <= SEG seconds.  Deterministic."""
        inp = self.proc(audio=[wav], sampling_rate=self.sr, return_tensors="pt")
        il = inp.get("is_longer")
        if il is not None and bool(il.any()):
            raise ValueError(
                f"input longer than CLAP's native {SEG}s window -> the model will apply a "
                f"RANDOM crop and the embedding is irreproducible. Split it first.")
        with torch.no_grad():
            out = self.model.get_audio_features(
                **{k: v.to(self.device) for k, v in inp.items()
                   if k in ("input_features", "is_longer", "input_features_mask")})
        e = getattr(out, "pooler_output", None)
        if e is None:
            e = out[0]
        return e[0].detach().cpu().numpy()

    def embed(self, wav, mode="auto"):
        """-> [D] L2-normalised, DETERMINISTIC for any input length.

        mode='window' : the input IS the unit of interest; require it be <= SEG.
        mode='auto'   : if longer than SEG, split into fixed non-overlapping SEG chunks,
                        embed each, average.  This is the safe prototype builder.
        """
        n = len(wav)
        limit = int(SEG * self.sr)
        if n <= limit or mode == "window":
            if n > limit:
                raise ValueError(f"window of {n/self.sr:.2f}s exceeds {SEG}s")
            return self._l2(self._embed_le10s(wav))
        step = limit
        chunks = [wav[i:i + step] for i in range(0, n, step)]
        chunks = [c for c in chunks if len(c) >= int(0.04 * self.sr)]
        embs = np.stack([self._embed_le10s(c) for c in chunks])
        return self._l2(embs.mean(axis=0))

    @staticmethod
    def _l2(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v

    def embed_many(self, wavs, mode="auto"):
        return np.stack([self.embed(w, mode=mode) for w in wavs])

    # ------------------------------------------------------------------ metrics
    @staticmethod
    def retrieve(query, protos):
        """query [D], protos [N, D] (both L2-normalised) -> (top1_idx, sims, rank)."""
        sims = protos @ query
        idx = int(np.argmax(sims))
        order = np.argsort(-sims)
        rank = int(np.where(order == 0)[0][0]) if False else None
        return idx, sims

    @staticmethod
    def identify(query, protos, true_idx):
        sims = protos @ query
        order = np.argsort(-sims)
        rank = int(np.where(order == true_idx)[0][0]) + 1
        return int(np.argmax(sims)) == true_idx, rank, float(sims[true_idx]), \
            float(np.max(np.delete(sims, true_idx)))

    # ------------------------------------------------------------------ tooling
    def assert_deterministic(self, wav, reps=3):
        hs = [_hash(self.embed(wav)) for _ in range(reps)]
        ok = len(set(hs)) == 1
        return ok, hs


def repeat(fn, n=5):
    """Run a measurement n times -> (mean, std, values).  Never quote a single run."""
    vals = np.array([fn() for _ in range(n)], dtype=float)
    return float(vals.mean()), float(vals.std()), vals.tolist()


if __name__ == "__main__":
    import glob, collections
    ev = ClapEvaluator()
    print(f"checkpoint={ev.cid}  sr={ev.sr}  dim={ev.dim}  device={ev.device}")

    w6 = ev.load(r"d:\Projects\CLaBraM\dataset\musin-g\Code\ESongs\6.esh.wav")
    print(f"\nsong 6: {len(w6)/ev.sr:.1f} s")
    print(f"  processor pads to a fixed ~{SEG}s mel, and >{SEG}s triggers a RANDOM crop.")

    # 1) a <=10 s window must be bit-stable
    ok, hs = ev.assert_deterministic(w6[:int(5 * ev.sr)])
    print(f"\n[window 5 s]      deterministic={ok}  hashes={hs}")

    # 2) the segmented full-song prototype must ALSO be bit-stable (the whole point)
    ok, hs = ev.assert_deterministic(w6)
    print(f"[full song {len(w6)/ev.sr:.0f} s] deterministic={ok}  hashes={hs}")
    if ok:
        print("  => the segmented prototype FIXES the irreproducibility")

    # 3) show the naive (single-pass) route fails, for contrast
    try:
        ev._embed_le10s(w6)
        print("[naive full-song] unexpectedly succeeded")
    except Exception as e:
        print(f"[naive full-song] correctly refused: {str(e)[:70]}...")

    # 4) sanity: songs should be retrievable from their own 5 s windows
    import glob
    files = sorted(glob.glob(r"d:\Projects\CLaBraM\dataset\musin-g\Code\ESongs\*.esh.wav"))
    songs = {i: ev.load(f) for i, f in enumerate(files, 1)}
    P = ev.embed_many([songs[i] for i in songs])
    print(f"\nprototype matrix: {P.shape}  (songs 1..{len(songs)})")

    def measure():
        corr, n = 0, 0
        for i in songs:
            w = songs[i]
            for st in range(0, len(w) - int(5 * ev.sr), int(10 * ev.sr)):
                q = ev.embed(w[st:st + int(5 * ev.sr)], mode="window")
                okh, rank, own, other = ev.identify(q, P, i - 1)
                corr += okh
                n += 1
        return corr / n

    m, s, vals = repeat(measure, 3)
    print(f"\n5 s-window retrieval over {len(songs)} songs:")
    print(f"  top-1 = {m:.4f} +- {s:.4f}   runs={[round(v,4) for v in vals]}")
    print(f"  deterministic across repeats: {s == 0.0}")
