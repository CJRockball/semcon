"""Unit tests for the scorecard panel — synthetic frames, no database."""

from __future__ import annotations

import numpy as np
import pandas as pd

from semcon import schema
from semcon.scorecard import compute_panel


def _scores(ps: list[float]) -> pd.DataFrame:
    n = len(ps)
    rank = pd.Series(ps).rank(ascending=False, method="first").astype(int)
    return pd.DataFrame(
        {
            schema.KEY_COL: np.arange(1, n + 1),
            "score_raw": ps,
            "p_cal": ps,
            "rank": rank,
            "decile": ((rank - 1) * 10 // n) + 1,
        }
    )


def _labels(fail_ids: set[int], n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            schema.KEY_COL: np.arange(1, n + 1),
            schema.TARGET_COL: [1 if i in fail_ids else -1 for i in range(1, n + 1)],
        }
    )


def test_perfect_queue_captures_top():
    ps = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    panel = compute_panel(_scores(ps), labels=_labels({1, 2}, 10))
    assert panel["n_scored"] == 10
    assert panel["n_fails"] == 2
    assert panel["top_decile_fails"] == 1  # decile 1 of 10 rows = rank 1 only
    assert panel["top_decile_capture"] == 0.5
    assert panel["top_decile_lift"] == (1 / 1) / 0.2


def test_reversed_queue_captures_nothing():
    ps = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    panel = compute_panel(_scores(ps), labels=_labels({9, 10}, 10))
    assert panel["top_decile_fails"] == 0
    assert panel["top_decile_capture"] == 0.0


def test_unlabeled_batch_degrades_gracefully():
    panel = compute_panel(_scores([0.5, 0.4, 0.3, 0.2, 0.1]), labels=None)
    assert panel["top_decile_capture"] is None
    assert panel["n_fails"] is None
    assert panel["p_median"] == 0.3  # distribution panel still computes


def test_zero_fails_no_crash():
    panel = compute_panel(_scores([0.9, 0.1, 0.05, 0.02, 0.01]), labels=_labels(set(), 5))
    assert panel["n_fails"] == 0
    assert panel["top_decile_capture"] is None
    assert panel["top_decile_lift"] is None


def test_clique_enrichment_math():
    ps = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    flags = pd.DataFrame(
        {
            schema.KEY_COL: np.arange(1, 11),
            "f_miss_clq14": [1] + [0] * 9,  # only the top-ranked wafer carries the flag
        }
    )
    panel = compute_panel(_scores(ps), flags=flags)
    assert panel["clq14_top_decile_share"] == 1.0
    assert panel["clq14_batch_share"] == 0.1
    assert panel["clq14_enrichment"] == 10.0


def test_pcal_nan_falls_back_to_raw():
    s = _scores([0.5, 0.4, 0.3])
    s["p_cal"] = np.nan
    panel = compute_panel(s)
    assert panel["p_median"] == 0.4
