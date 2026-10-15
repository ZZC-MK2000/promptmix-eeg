"""Regenerate every paper figure from cached results (CPU only, no GPU, no model training).
Order matters only in that fig0-3 come from the radius JSONs already on disk.

Run: python code/figures/render_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
SCRIPTS = ["make_fig_architecture.py", "make_figures.py", "make_figures_tsne.py",
           "make_figures_spectro.py", "make_fig_genre_confusion.py", "make_fig_compat_map.py",
           "make_fig_retrieval_pool.py", "make_fig_mfcc_fad.py"]

for s in SCRIPTS:
    print(f"\n===== {s} =====", flush=True)
    subprocess.run([sys.executable, str(HERE / s)], check=True)
print("\nall figures regenerated")
