"""Audit every row-keyed artifact set for SAMPLE COVERAGE, and fail on prefix-sampling damage.

Why this exists (2026-10-16/17 incident): `deliver_library_b.py` picked its showcase rows with
`val_rows[:14]` — a prefix of a subject-ordered array, not a sample. The 14 "heard vs generated"
pairs published on the demo page turned out to be 6 distinct audio windows, all from ONE subject
(sub-02), ONE run, ONE song. Every own-vs-others statistic computed on that set was invalid,
because the "other 13" baseline contained byte-identical copies of the target.

This script is the guard. For each artifact directory whose filenames carry `row<N>`, it resolves
the row to (group, subject, song), deduplicates the underlying stimulus audio by content hash, and
reports coverage. It exits non-zero when a published set is too concentrated, so a future prefix
bug cannot reach the page silently.

Discipline rules this enforces: 31 (展示用样本的选取规则必须落盘并检查与被比较变量无关),
29 (跨批次比较主键必须是底层行 id), 33 (候选集必须与查询无共享样本), 16 (发现 bug 后回查污染过哪些数字).

Run: python code/evaluation/audit_sample_coverage.py
"""
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
M2 = Path("D:/Projects/CLaBraM/mainv2")
DATA = ROOT / "data" / "eegcond"
OUT = ROOT / "results" / "sample_audit"

# artifact sets that are quoted publicly (demo page, figures, outline)
SETS = {
    "library_b/listen (demo 'heard vs generated')": "results/library_b/listen",
    "prompt_classifier_true/listen": "results/prompt_classifier_true/listen",
    "prompt_classifier_true/listen_B": "results/prompt_classifier_true/listen_B",
    "gen_eegcond_mixed/eval (head-to-head fine-tuned arm)": "results/gen_eegcond_mixed/eval",
    "demo (end-to-end CLI)": "results/demo",
}
MIN_SONGS = 4       # a published showcase set must span at least this many distinct songs
MIN_GROUPS = 4      # ...and this many distinct source groups (non-overlapping 8-s blocks)


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def build_row_index():
    """row (index into data/eegcond/*.npy) -> (group, subject, song)."""
    group = np.load(DATA / "group.npy").astype(np.int64)
    cond = np.load(DATA / "cond.npy").astype(np.float32)
    z = np.load(M2 / "data" / "gate_nmedh14_ov2_lfix.npz", allow_pickle=True)
    song = np.array(z["song"]).astype(np.int64)
    subj = np.array(z["subject"]).astype(str)
    run = np.array(z["run"]).astype(str)
    ea = np.load(M2 / "data" / "eeg_emb_aldm_s0.npy").astype(np.float32)
    W = torch.load(M2 / "results" / "eeg_to_clap_proj.pt", map_location="cpu", weights_only=True)
    cond_all = unit(ea @ W["weight"].numpy().astype(np.float32).T + W["bias"].numpy().astype(np.float32))
    key = {}
    dupes = 0
    for i in range(len(cond_all)):
        b = cond_all[i].tobytes()
        if b in key:
            dupes += 1
        key[b] = i
    rows = {}
    unmatched = 0
    for r in range(len(cond)):
        i = key.get(cond[r].tobytes(), -1)
        if i < 0:
            unmatched += 1
            continue
        rows[r] = (int(group[r]), int(song[i]), subj[i], run[i])
    print(f"row index built: {len(rows)} rows resolved, {unmatched} unmatched, "
          f"{dupes} duplicate bridge keys (must be 0 for this map to be trustworthy)")
    return rows


def content_hash(d):
    """Hash the stimulus audio behind a set, so 'distinct' means distinct bytes, not distinct row."""
    h = {}
    for f in sorted(d.rglob("ref__row*.wav")):
        h[f.stem.replace("ref__", "")] = hashlib.md5(f.read_bytes()).hexdigest()[:10]
    return h


def main():
    rows = build_row_index()
    report = {}
    failures = []
    for label, sub in SETS.items():
        d = ROOT / sub
        if not d.exists():
            print(f"\n{label}: MISSING, skipped")
            continue
        found = defaultdict(set)
        for f in d.rglob("*.wav"):
            m = re.search(r"row(\d+)", f.name)
            if m:
                found[int(m.group(1))].add(f.name)
        if not found:
            print(f"\n{label}: no row-keyed wavs")
            continue
        rs = sorted(found)
        known = [r for r in rs if r in rows]
        grp = sorted({rows[r][0] for r in known})
        sng = sorted({rows[r][1] for r in known})
        sbj = sorted({rows[r][2] for r in known})
        rn = sorted({rows[r][3] for r in known})
        chash = content_hash(d)
        distinct_audio = len(set(chash.values())) if chash else None
        cov = {"set": sub, "n_rows": len(rs), "n_groups": len(grp), "n_songs": len(sng),
               "n_subjects": len(sbj), "n_runs": len(rn), "songs": sng,
               "subjects": sbj, "runs": rn,
               "distinct_stimulus_audio": distinct_audio,
               "rows": rs}
        report[label] = cov
        dup = (len(rs) - distinct_audio) if distinct_audio else 0
        print(f"\n{label}")
        print(f"   rows={len(rs)}  distinct groups={len(grp)}  songs={len(sng)}{sng}  "
              f"subjects={len(sbj)}{sbj}  runs={len(rn)}{rn}")
        if distinct_audio is not None:
            print(f"   distinct stimulus audio (md5)={distinct_audio}  duplicate rows={dup}")
        bad = []
        if len(sng) < MIN_SONGS:
            bad.append(f"songs {len(sng)} < {MIN_SONGS}")
        if len(grp) < MIN_GROUPS:
            bad.append(f"groups {len(grp)} < {MIN_GROUPS}")
        if distinct_audio is not None and distinct_audio < MIN_SONGS:
            bad.append(f"distinct audio {distinct_audio} < {MIN_SONGS}")
        if bad:
            print(f"   *** FAIL: {'; '.join(bad)}  -> this set must not be used as a showcase")
            failures.append(label)
        else:
            print("   ok")
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(report, open(OUT / "coverage.json", "w", encoding="utf-8"), indent=2, default=str)
    print(f"\nwrote {OUT/'coverage.json'}")
    if failures:
        print(f"\n{len(failures)} set(s) fail coverage: {failures}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
