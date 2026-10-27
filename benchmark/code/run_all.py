"""One-click AEB sanity pass: verify artifacts, audit splits, rebuild leaderboard."""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def check(path):
    p = ROOT / path
    print(("OK  " if p.exists() else "MISS"), p)
    return p.exists()


def main():
    required = [
        "splits/nmedt_folds.json", "splits/leave_one_dataset_out.json",
        "manifests/split_audit.json", "manifests/coverage.json",
        "manifests/cache_manifest.json", "results/reference.json",
    ]
    ok = all(check(p) for p in required)
    if not ok:
        raise SystemExit("missing required artifacts; see README")
    subprocess.run([sys.executable, str(HERE / "audit_splits.py")], check=False)
    subprocess.run([sys.executable, str(HERE / "make_leaderboard.py")], check=False)
    d = json.load(open(ROOT / "manifests/split_audit.json", encoding="utf-8"))
    print(json.dumps({k: len(v) for k, v in d["nmedt"].items()}, indent=2))
    print("AEB sanity pass complete; raw caches and weights are user-supplied.")


if __name__ == "__main__":
    main()
