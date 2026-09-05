"""Tests for semcon.train_xgb.

Scope: the pure and CPU-only parts of the training pipeline. run_cv and
refit_final fit XGBoost with device='cuda' and are deliberately not
unit-tested here. The data contract (three-zone split, boundary integrity,
label home) is covered by tests/test_data_layer.py against the live DB;
what is pinned here is the CLI contract and the evaluation stage.

Note: the old parquet-based load_data test was retired with the flat-file
pipeline — load_data now composes extract + build_features + feature_columns
from the database, and its split behavior is exactly what test_data_layer.py
proves. Nothing here reads data files.
"""

import matplotlib

matplotlib.use("Agg")  # headless: evaluation saves figures

import numpy as np
import pandas as pd
import pytest

from semcon import train_xgb

# ── parse_args ────────────────────────────────────────────────────────────────


def test_parse_args_defaults():
    args = train_xgb.parse_args([])
    assert args.use_selection is True
    assert args.repeats == 3
    assert args.overrides == []
    assert args.run_name is None


def test_parse_args_flags():
    args = train_xgb.parse_args(
        ["--no-selection", "--run-name", "baseline", "--set", "max_depth=5", "--set", "eta=0.1"]
    )
    assert args.use_selection is False
    assert args.run_name == "baseline"
    assert args.overrides == ["max_depth=5", "eta=0.1"]  # repeated, order kept


# ── evaluate ──────────────────────────────────────────────────────────────────


def test_evaluate_writes_artifacts_and_metrics(tmp_path):
    """Separable toy problem: fails score 0.9, passes score 0.1.

    Frames carry is_fail (the validated label) — evaluate reads no other
    column; OOF/holdout scores arrive as arrays from the training stage.
    """
    y_train = np.array([0] * 90 + [1] * 10)
    y_hold = np.array([0] * 18 + [1] * 2)
    oof = np.tile(np.where(y_train == 1, 0.9, 0.1), (3, 1))  # repeats x n
    p_hold = np.where(y_hold == 1, 0.9, 0.1)
    df_train = pd.DataFrame({"is_fail": y_train})
    df_test = pd.DataFrame({"is_fail": y_hold})

    metrics = train_xgb.evaluate(df_train, df_test, oof, p_hold, out=tmp_path)

    expected = {
        "summary_oof.csv",
        "summary_hold.csv",
        "pr_curve_holdout.png",
        "conf_heatmap.png",
        "pred_hold_stats.csv",
        "oof_mean_stats.csv",
        "pr_oof_hold.csv",
    }
    written = {p.name for p in tmp_path.iterdir()}
    assert expected <= written, f"missing artifacts: {expected - written}"
    for name in expected:
        assert (tmp_path / name).stat().st_size > 0

    assert 0.0 < metrics["threshold"] < 1.0
    # separable scores => perfect holdout ranking and recall at the OOF threshold
    assert metrics["holdout_aucpr"] == pytest.approx(1.0)
    assert metrics["holdout_rocauc"] == pytest.approx(1.0)
    assert metrics["holdout_recall"] == pytest.approx(1.0)
