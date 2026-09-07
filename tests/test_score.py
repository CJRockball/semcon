"""Contract tests for the scorer — no database required."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from semcon import score


def test_check_contract_raises_on_missing():
    frame = pd.DataFrame({"s001": [0.5]})
    with pytest.raises(ValueError, match="missing"):
        score.check_contract(frame, ["s001", "s002"])


def test_check_contract_passes_and_orders():
    frame = pd.DataFrame({"b": [1], "a": [2], "extra": [3]})
    out = score.check_contract(frame, ["a", "b"])
    assert list(out.columns) == ["a", "b"]


def _fake_registry(tmp_path):
    runs = tmp_path / "runs"
    for run_id in ("20260101_000000_xgb_sel", "20260102_000000_xgb_sel"):
        (runs / run_id).mkdir(parents=True)
        (runs / run_id / "model.ubj").touch()
    cal = runs / "20260102_000100_cal-platt__xgb_sel"
    cal.mkdir()
    (cal / score.CALIBRATOR_NAME).touch()
    idx = pd.DataFrame(
        [
            {"run_id": "20260101_000000_xgb_sel", "type": "", "parent_run": ""},
            {"run_id": "20260102_000000_xgb_sel", "type": "", "parent_run": ""},
            {
                "run_id": "20260102_000100_cal-platt__xgb_sel",
                "type": "calibration",
                "parent_run": "20260102_000000_xgb_sel",
            },
        ]
    )
    idx.to_csv(runs / "index.csv", index=False)
    return runs


def test_resolve_latest_and_calibrator(tmp_path, monkeypatch):
    runs = _fake_registry(tmp_path)
    monkeypatch.setattr(score, "RUNS", runs)
    monkeypatch.setattr(score, "INDEX", runs / "index.csv")
    _, _, train_id, cal_id = score.resolve_runs("latest", no_cal=False)
    assert train_id == "20260102_000000_xgb_sel"
    assert cal_id == "20260102_000100_cal-platt__xgb_sel"


def test_resolve_no_calibrator_linkage(tmp_path, monkeypatch):
    runs = _fake_registry(tmp_path)
    monkeypatch.setattr(score, "RUNS", runs)
    monkeypatch.setattr(score, "INDEX", runs / "index.csv")
    _, _, train_id, cal_id = score.resolve_runs("20260101_000000_xgb_sel", no_cal=False)
    assert train_id == "20260101_000000_xgb_sel"
    assert cal_id is None  # the calibrator belongs to the other training run


def test_apply_calibrator_is_monotone():
    raw = np.linspace(-3, 3, 9)
    y = (raw > 0).astype(int)
    cal = LogisticRegression().fit(raw.reshape(-1, 1), y)
    p = score.apply_calibrator(cal, raw)
    assert p.shape == (9,)
    assert np.all(np.diff(p) > 0)
