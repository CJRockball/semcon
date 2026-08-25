import pandas as pd
import numpy as np


def hedges_g(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """|Hedges g| per column vs binary y. NaN-safe. TRAIN-fold data only."""
    x1, x0 = X[y == 1], X[y == 0]
    n1, n0 = len(x1), len(x0)
    m1, m0 = x1.mean(), x0.mean()
    v1, v0 = x1.var(ddof=1), x0.var(ddof=1)
    sp = np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2))
    d = (m1 - m0) / sp.replace(0, np.nan)
    J = 1 - 3 / (4 * (n1 + n0) - 9)          # small-sample correction
    return (d * J).abs().fillna(0.0)

def select_features(Xtr, ytr, g_min=0.25, k_min=30, k_max=100):
    """Univariate filter, recomputed inside every fold (no full-data leakage).
    g >= 0.25 keeps real effects (EDA: noise floor ~0.1, real effects 0.3-0.5);
    k_min/k_max guard against degenerate fold selections."""
    g = hedges_g(Xtr, ytr)
    sel = g[g >= g_min].index.tolist()
    if len(sel) < 10:
        sel = g.nlargest(k_min).index.tolist()
    elif len(sel) > k_max:
        sel = g.nlargest(k_max).index.tolist()
    return sel
