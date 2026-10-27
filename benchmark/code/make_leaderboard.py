"""Build the static AEB leaderboard from results/reference.json."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/reference.json")
    ap.add_argument("--out", default="leaderboard/index.html")
    args = ap.parse_args()
    d = json.load(open(args.results))
    rows = []
    for m in d["frozen"]:
        rows.append((m["model"], m["family"], m.get("params", "—"),
                     f"{m['logistic']:.3f}", "frozen probe"))
    for m in d["finetune"]:
        rows.append((m["model"], "supervised", m.get("params", "—"),
                     f"{m['per_subject']:.3f}", "per-subject fine-tune"))
    trs = "\n".join(
        f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td><b>{e}</b></td><td>{f}</td></tr>"
        for a, b, c, e, f in rows)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Auditory EEG Bench — leaderboard v0.1</title>
<style>body{{font:16px/1.6 Georgia,serif;max-width:980px;margin:40px auto;padding:0 20px}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:8px 10px;border-bottom:1px solid #ddd;text-align:left}}
th{{background:#f3f5f7}}code{{background:#eee;padding:1px 5px}}.warn{{color:#a33}}</style></head>
<body><h1>Auditory EEG Bench — leaderboard v0.1</h1>
<p>NMED-T 10-way, per-subject trial-wise 50/50; chance 0.10. Frozen probes and
fine-tuning are separate protocols. <span class="warn">Reference results; unified 3-seed
grid and full LODO are pending.</span></p>
<table><thead><tr><th>model</th><th>family</th><th>params</th><th>10-way</th><th>protocol</th></tr></thead>
<tbody>{trs}</tbody></table>
<p>Controls: random rows .896 / blocks .873 / subjects .106 (band-power); answer-conditioned
zero/shuffled EEG still 1.000 pair accuracy. See <code>README.md</code> and
<code>manifests/split_audit.json</code>. No raw EEG or checkpoints are released.</p>
</body></html>"""
    p = Path(args.out); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    print("wrote", p)


if __name__ == "__main__":
    main()
