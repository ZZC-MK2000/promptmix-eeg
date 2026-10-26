"""Aggregate SparrKULee scaling runs into one table/JSON."""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sparr_scale import list_files  # noqa: E402
SCALE = ROOT / "results" / "sparr_env_scale"
PROBE = ROOT / "results" / "sparr_story_probe"
OUT = ROOT / "results" / "sparr_scaling"


def train_file_count(a):
    recs = list_files(a["root"])
    subs = sorted({r["subject"] for r in recs})
    test = set(subs[:a["test_subjects"]])
    pool = set([s for s in subs if s not in test][:a["subjects"]])
    splits = set(a.get("train_splits", "train").split(","))
    return len([r for r in recs if r["subject"] in pool and r["split"] in splits])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="train")
    args = ap.parse_args()
    rows = []
    for d in sorted(SCALE.iterdir()):
        if not d.name.startswith(args.prefix):
            continue
        logp = d / "train_log.json"
        pp = PROBE / d.name / "eval.json"
        if not logp.exists() or not pp.exists():
            continue
        log = json.load(open(logp, encoding="utf-8"))
        probe = json.load(open(pp, encoding="utf-8"))
        ps = probe["methods"]["z"]["per_subject"]
        subs = sorted(ps)
        ch = np.array([ps[s]["chance"] for s in subs])
        z = np.array([ps[s]["acc"] for s in subs])
        bp = np.array([probe["methods"]["bp"]["per_subject"][s]["acc"] for s in subs])
        p = float(stats.ttest_rel(z, bp).pvalue)
        m = re.match(r"train_s(\d+)_d(\d+)_l(\d+)(?:_10k)?$", d.name)
        rows.append({
            "run": d.name,
            "subjects": log["args"]["subjects"],
            "depth": log["args"]["depth"],
            "d": log["args"]["d"],
            "steps": log["args"]["steps"],
            "params": probe["params"],
            "train_files": train_file_count(log["args"]),
            "mean_chance": float(ch.mean()),
            "z_acc": float(z.mean()),
            "bp_acc": float(bp.mean()),
            "z_excess": float((z - ch).mean()),
            "bp_excess": float((bp - ch).mean()),
            "delta": float((z - bp).mean()),
            "paired_p": p,
        })
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"{'run':28} {'subj':>4} {'params':>9} {'files':>5} {'z':>6} {'bp':>6} {'dz-bp':>7} {'p':>7}")
    for r in rows:
        print(f"{r['run']:28} {r['subjects']:4d} {r['params']:9d} "
              f"{str(r['train_files']):>5} {r['z_acc']:.3f} {r['bp_acc']:.3f} "
              f"{r['delta']:+.3f} {r['paired_p']:.3f}")
    print("wrote", OUT / "summary.json")


if __name__ == "__main__":
    main()
