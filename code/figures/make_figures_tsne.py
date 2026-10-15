"""Fig 5 (CPU): t-SNE of laion-CLAP embeddings -- do the corpus-matched library clips SPAN the
cloud of real held-out windows? This visualizes the coverage-radius claim (0.879) geometrically:
if the library points surround/interleave the real-window cloud, a mixture over them can reach a
real window. Subsampled for speed. English labels.

Run: python src/train/make_figures_tsne.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("D:/Projects/CLaBraM/promptmix-eeg")
FIG = ROOT.parent / "paper" / "figures"
LIBB = ROOT / "results" / "library_b"


def unit(x):
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-8)


def main():
    from sklearn.manifold import TSNE
    lib = unit(np.load(LIBB / "library_emb.npy").astype(np.float32))      # (72,512)
    te = unit(np.load(ROOT / "results" / "prompt_classifier" / "true_emb_all.npy").astype(np.float32))
    rng = np.random.default_rng(0)
    te = te[rng.choice(len(te), 400, replace=False)]                       # (400,512)
    X = np.vstack([te, lib])
    y = np.array([0] * len(te) + [1] * len(lib))
    Z = TSNE(n_components=2, perplexity=30, init="pca", random_state=0,
             max_iter=1000).fit_transform(X)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=14, c="#1f77b4", alpha=0.5, label="real held-out windows (400)")
    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=60, c="#d62728", marker="X", label="library clips (72)")
    ax.legend(fontsize=9, loc="best")
    ax.set_title("t-SNE (laion-CLAP): corpus-matched library spans the real-window cloud")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(FIG / "fig5_tsne_coverage.png", dpi=150)
    print("wrote", FIG / "fig5_tsne_coverage.png")


if __name__ == "__main__":
    main()
