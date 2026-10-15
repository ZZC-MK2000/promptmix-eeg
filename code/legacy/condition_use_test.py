"""Is the model actually USING its condition, or has it learned to ignore it?

Compare loss on correctly paired (condition, codes) against SHUFFLED pairs, on both corpora:
  EEG pool (our 14-song stimulus windows) and FMA pool (4132 diverse segments).
If loss(correct) ~= loss(shuffled), the decoder is modelling the marginal and ignoring the
condition -- which is exactly what a plateau at ~6.4 nats on diverse FMA music would mean.

Also reports the unconditional-entropy reference: loss with a fixed constant condition.

Run: python src/train/condition_use_test.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
DATA = ROOT / "data" / "eegcond"
FMA8 = ROOT / "data" / "fma8s"
for _s in ("shared","radii","classifier","delivery","evaluation","alignment","figures","legacy"): sys.path.insert(0, str(ROOT / "code" / _s))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="ep16,mixed_s12000,fma_s4000")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--device", default="")
    args = ap.parse_args()

    from pilot_multitoken import CondWidthModel

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    paths = {
        "ep16": ROOT / "results/gen_eegcond_k8/ckpt_ep16.pt",
        "mixed_s12000": ROOT / "results/gen_eegcond_mixed/ckpt_s12000.pt",
        "fma_s4000": ROOT / "results/gen_fmaonly/ckpt_s4000.pt",
        "fma_s8000": ROOT / "results/gen_fmaonly/ckpt_s8000.pt",
        "fma_s12000": ROOT / "results/gen_fmaonly/ckpt_s12000.pt",
        "fma_s16000": ROOT / "results/gen_fmaonly/ckpt_s16000.pt",
        "fma_s20000": ROOT / "results/gen_fmaonly/ckpt_s20000.pt",
    }

    econ = np.load(DATA / "cond.npy").astype(np.float32)
    ecode = np.load(DATA / "codes.npy")
    grp = np.load(DATA / "group.npy").astype(np.int64)
    fcon = np.load(FMA8 / "cond.npy").astype(np.float32)
    fcode = np.load(FMA8 / "codes.npy")

    rng = np.random.default_rng(0)
    uniq = np.unique(grp)
    n_val = max(1, int(round(len(uniq) * 0.1)))
    val_g = set(uniq[rng.permutation(len(uniq))[:n_val]].tolist())
    va = np.array([g in val_g for g in grp])
    ei = np.where(va)[0][:args.n]
    fi = rng.choice(len(fcon), min(args.n, len(fcon)), replace=False)

    pools = {
        "EEG留out": (torch.from_numpy(econ[ei]), torch.from_numpy(ecode[ei]).long()),
        "FMA": (torch.from_numpy(fcon[fi]), torch.from_numpy(fcode[fi]).long()),
    }
    print(f"每池 {len(ei)}/{len(fi)} 条；打乱 = 把条件与码的配对随机重排（码不变）")

    g = torch.Generator().manual_seed(args.seed)
    for spec in args.ckpts.split(","):
        spec = spec.strip()
        is_path = spec.endswith(".pt") or "/" in spec
        name = Path(spec).stem if is_path else spec
        p = Path(spec) if is_path else paths.get(spec)
        if p is None or not p.exists():
            print(f"  {name}: 无此 ckpt，跳过"); continue
        model = CondWidthModel(device, k_tokens=args.k)
        ck = torch.load(str(p), map_location="cpu", weights_only=True)
        model.m.clap_proj.load_state_dict(ck["clap_proj"])
        model.m.mg.decoder.load_state_dict(ck["decoder_lora"], strict=False)
        model.m.eval()
        line = [f"  {name.strip():<14}"]
        for pname, (con, cd) in pools.items():
            perm = torch.randperm(len(cd), generator=g)
            const = con.mean(0, keepdim=True).repeat(len(cd), 1)
            res = {}
            for tag, cc in (("正确配对", con), ("打乱配对", con[perm]), ("常数条件", const)):
                tot, cnt = 0.0, 0
                with torch.no_grad():
                    for a in range(0, len(cd), args.batch):
                        l = model.loss(cd[a:a + args.batch], cc[a:a + args.batch])
                        tot += float(l.item()) * min(args.batch, len(cd) - a)
                        cnt += min(args.batch, len(cd) - a)
                res[tag] = tot / cnt
            line.append(f"{pname}: 正确={res['正确配对']:.4f} 打乱={res['打乱配对']:.4f} "
                        f"常数={res['常数条件']:.4f} 差={res['正确配对']-res['打乱配对']:+.4f}")
        print("  ".join(line), flush=True)
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print("\n判读: “差”越负说明正确配对损失越低=模型真在用条件；差≈0 说明它在无视条件、")
    print("只建模边际分布（这时 token loss 停在语料的无条件熵上，怎么训都不动）。")


if __name__ == "__main__":
    main()
