"""Regenerate every paper figure from cached results (CPU only, no GPU, no model training).
Order matters only in that fig0-3 come from the radius JSONs already on disk.

Not included: make_fig_stim_vs_gen.py (needs the mainv2 laion-CLAP helper and re-embeds 28 clips)
and code/alignment/song_confusion.py (needs mainv2's gate npz). Run those two directly.

Run: python code/figures/render_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
PRE = [(ROOT / "code" / "radii" / "coverage_unified.py", "coverage table under one convention")]
SCRIPTS = ["make_fig_architecture.py", "make_figures.py",
           "make_figures_spectro.py", "make_fig_genre_confusion.py", "make_fig_compat_map.py",
           "make_fig_retrieval_pool.py", "make_fig_mfcc_fad.py"]

for p, why in PRE:
    print(f"\n===== {p.name} ({why}) =====", flush=True)
    subprocess.run([sys.executable, str(p)], check=True)

for s in SCRIPTS:
    print(f"\n===== {s} =====", flush=True)
    subprocess.run([sys.executable, str(HERE / s)], check=True)
print("\nall figures regenerated")
