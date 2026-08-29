"""Tests for semcon.train_xgb.

Scope: the pure and CPU-only parts of the training pipeline. run_cv and
refit_final fit XGBoost with device='cuda' and need the real parquet data,
so they are deliberately not unit-tested here; what is pinned down is the
data contract (time-ordered split, regime tail drop) and the evaluation
stage (artifacts written, metrics sane on a separable toy problem).
"""

import json

import matplotlib
matplotlib.use("Agg")  # headless: evaluation saves figures

import numpy as np
import pandas as pd
import pytest

from semcon import train_xgb


# ── parse_args ───────────────────────────────────────────────────────────────

def test_parse_args_defaults():
    args = train_xgb.parse_args([])
    assert args.use_selection is True
    assert args.repeats == 3
    assert args.overrides == []
    assert args.run_name is None


def test_parse_args_flags():
    args = train_xgb.parse_args(
        ["--no-selection", "--run-name", "baseline",
         "--set", "max_depth=5", "--set", "eta=0.1"])
    assert args.use_selection is False
    assert args.run_name == "baseline"
    assert args.overrides == ["max_depth=5", "eta=0.1"]  # repeated, order kept


# ── load_data ────────────────────────────────────────────────────────────────

def _write_fake_dataset(path, n=100):
    rng = np.random.default_rng(0)
    dfX = pd.DataFrame(rng.normal(size=(n, 5)), columns=[f"s{i}" for i in range(5)])
    dfy = pd.DataFrame({
        "target": (rng.random(n) < 0.1).astype(int),
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h"),
    })
    dfX.to_parquet(path / "dfX_v2.parquet")
    dfy.to_parquet(path / "dfy_v1.parquet")
    (path / "dfX_v2.dataset.json").write_text(json.dumps({"sha256_16": "0" * 16}))


def test_load_data_time_ordered_tail_drop(tmp_path, monkeypatch):
    _write_fake_dataset(tmp_path, n=100)
    monkeypatch.setattr(train_xgb, "DATA_PROCESSED", tmp_path)

    df_train, df_test, target, data_info = train_xgb.load_data(
        test_split=15, time_split=10)

    assert target == "target"
    assert data_info["sha256_16"] == "0" * 16
    # 100 rows - 10-row regime tail = 90; the last 15 of those are the holdout
    assert len(df_train) == 75
    assert len(df_test) == 15
    # time-blocked: every train timestamp precedes every test timestamp
    assert df_train["timestamp"].max() < df_test["timestamp"].min()
    # the dropped tail is gone entirely (nothing newer than row 89 survives)
    assert df_test["timestamp"].max() <= \
        pd.date_range("2026-01-01", periods=100, freq="h")[89]
    # both frames come back time-sorted
    assert df_train["timestamp"].is_monotonic_increasing
    assert df_test["timestamp"].is_monotonic_increasing


# ── evaluate ─────────────────────────────────────────────────────────────────

def test_evaluate_writes_artifacts_and_metrics(tmp_path):
    # separable toy problem: fails get 0.9, passes get 0.1
    y_train = np.array([0] * 90 + [1] * 10)
    y_hold = np.array([0] * 18 + [1] * 2)
    oof = np.tile(np.where(y_train == 1, 0.9, 0.1), (3, 1))  # repeats x n
    p_hold = np.where(y_hold == 1, 0.9, 0.1)
    df_train = pd.DataFrame({
        "target": y_train,
        "timestamp": pd.date_range("2026-01-01", periods=len(y_train), freq="h")})
    df_test = pd.DataFrame({
        "target": y_hold,
        "timestamp": pd.date_range("2026-02-01", periods=len(y_hold), freq="h")})

    metrics = train_xgb.evaluate(df_train, df_test, oof, p_hold, out=tmp_path)

    expected = {"summary_oof.csv", "summary_hold.csv", "pr_curve_holdout.png",
                "conf_heatmap.png", "pred_hold_stats.csv", "oof_mean_stats.csv",
                "pr_oof_hold.csv"}
    written = {p.name for p in tmp_path.iterdir()}
    assert expected <= written
    for name in expected:
        assert (tmp_path / name).stat().st_size > 0

    assert 0.0 < metrics["threshold"] < 1.0
    # separable scores => perfect holdout ranking and recall at the OOF threshold
    assert metrics["holdout_aucpr"] == pytest.approx(1.0)
    assert metrics["holdout_rocauc"] == pytest.approx(1.0)
    assert metrics["holdout_recall"] == pytest.approx(1.0)
