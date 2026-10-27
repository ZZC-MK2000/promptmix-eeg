"""Subject-level paired statistics and bootstrap CIs for AEB."""
import numpy as np


def paired_subject_test(a, b, n_boot=2000, seed=0):
    """Paired difference b - a over subjects; return mean, 95% CI and sign count."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = b[m] - a[m]
    if len(d) < 2:
        return {"n": int(len(d)), "mean_diff": None, "ci95": [None, None], "b_gt_a": None}
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(d, len(d), replace=True).mean() for _ in range(n_boot)])
    return {"n": int(len(d)), "mean_diff": float(d.mean()),
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
            "b_gt_a": int((d > 0).sum())}


def summarise(values):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return {"n": int(len(v)), "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)) if len(v) > 1 else None}
