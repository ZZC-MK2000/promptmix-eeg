"""Regenerate the paper figures from cached results (CPU only, no GPU, no model training).
Order matters only in that fig1-3 come from the radius JSONs already on disk.

Not included:
  - fig0 (make_fig_architecture.py): fig0 is a hand-supplied structure diagram; rebuilding it
    would overwrite that file with the older matplotlib version, so it is left untouched.
  - fig12 (make_fig_stim_vs_gen.py): needs the mainv2 laion-CLAP helper and re-embeds 28 clips.
  - fig11 (code/alignment/song_confusion.py): needs mainv2's gate npz.
Run the last two directly if needed.

Run: python code/figures/render_all.py
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
PRE = [(ROOT / "code" / "radii" / "coverage_unified.py", "coverage table under one convention")]
SCRIPTS = ["make_fig_align_structure.py", "make_figures.py",
           "make_figures_spectro.py", "make_fig_genre_confusion.py", "make_fig_compat_map.py",
           "make_fig_retrieval_pool.py", "make_fig_mfcc_fad.py", "make_fig_info_ladder.py",
           "make_fig_k_shot.py", "make_fig_oracle_alignment.py",
           "make_fig_kshot_two_levels.py", "make_fig_three_paths.py"]

for p, why in PRE:
    print(f"\n===== {p.name} ({why}) =====", flush=True)
    subprocess.run([sys.executable, str(p)], check=True)

for s in SCRIPTS:
    print(f"\n===== {s} =====", flush=True)
    subprocess.run([sys.executable, str(HERE / s)], check=True)
print("\nall figures regenerated")
