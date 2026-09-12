"""Unit tests for semcon.retrain_trigger (Item 9)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from semcon.retrain_trigger import evaluate_retrain_trigger


def test_evaluate_retrain_trigger_missing_file(tmp_path: Path):
    missing_file = tmp_path / "does_not_exist.csv"
    res = evaluate_retrain_trigger(index_path=missing_file)
    assert isinstance(res, dict)
    assert res["retrain_recommended"] is False
    assert res["verdict"] == "NO_DATA"


def test_evaluate_retrain_trigger_empty_file(tmp_path: Path):
    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("", encoding="utf-8")
    res = evaluate_retrain_trigger(index_path=empty_file)
    assert res["retrain_recommended"] is False
    assert res["verdict"] in {"EMPTY", "NO_METRICS"}


def test_evaluate_retrain_trigger_in_control(tmp_path: Path):
    p = tmp_path / "index_monitor.csv"
    df = pd.DataFrame(
        [
            {"run_id": "spc_001", "type": "spc", "max_delta": 0.02},
            {"run_id": "spc_002", "type": "spc", "max_delta": 0.03},
        ]
    )
    df.to_csv(p, index=False)

    res = evaluate_retrain_trigger(index_path=p, delta_threshold=0.05, consecutive_runs=1)
    assert res["retrain_recommended"] is False
    assert res["verdict"] == "IN_CONTROL"
    assert res["recent_values"] == [0.03]


def test_evaluate_retrain_trigger_excursion_single_run(tmp_path: Path):
    p = tmp_path / "index_monitor.csv"
    df = pd.DataFrame(
        [
            {"run_id": "spc_001", "type": "spc", "max_delta": 0.02},
            {"run_id": "spc_002", "type": "spc", "max_delta": 0.08},
        ]
    )
    df.to_csv(p, index=False)

    res = evaluate_retrain_trigger(index_path=p, delta_threshold=0.05, consecutive_runs=1)
    assert res["retrain_recommended"] is True
    assert res["verdict"] == "RETRAIN_RECOMMENDED"
    assert res["recent_values"] == [0.08]


def test_evaluate_retrain_trigger_consecutive_runs_requirement(tmp_path: Path):
    p = tmp_path / "index_monitor.csv"
    # Run 1 is below threshold, run 2 is above, but consecutive_runs requires 2
    df = pd.DataFrame(
        [
            {"run_id": "spc_001", "type": "spc", "max_delta": 0.02},
            {"run_id": "spc_002", "type": "spc", "max_delta": 0.08},
        ]
    )
    df.to_csv(p, index=False)

    # 1 run is not enough when requiring 2 consecutive
    res = evaluate_retrain_trigger(index_path=p, delta_threshold=0.05, consecutive_runs=2)
    assert res["retrain_recommended"] is False

    # Append another breached run -> now 2 consecutive breaches
    df_breach = pd.concat(
        [df, pd.DataFrame([{"run_id": "spc_003", "type": "spc", "max_delta": 0.09}])],
        ignore_index=True,
    )
    df_breach.to_csv(p, index=False)

    res2 = evaluate_retrain_trigger(index_path=p, delta_threshold=0.05, consecutive_runs=2)
    assert res2["retrain_recommended"] is True
    assert res2["verdict"] == "RETRAIN_RECOMMENDED"


def test_evaluate_retrain_trigger_explicit_verdict_column(tmp_path: Path):
    p = tmp_path / "index_monitor.csv"
    df = pd.DataFrame(
        [
            {
                "run_id": "mon_001",
                "type": "monitoring",
                "drift_fraction": 0.02,
                "verdict": "RETRAIN_RECOMMENDED",
            }
        ]
    )
    df.to_csv(p, index=False)

    res = evaluate_retrain_trigger(index_path=p)
    assert res["retrain_recommended"] is True
    assert res["verdict"] == "RETRAIN_RECOMMENDED"
