from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from sklearn.linear_model import LogisticRegression

from semcon import schema
from semcon.doe_run import (
    append_doe_index,
    load_background_frame,
    resolve_surrogate,
    score_design,
    score_design_ood,
    write_run_artifacts,
)


class DummyEngine:
    pass


def _tiny_background() -> pd.DataFrame:
    return pd.DataFrame(
        {
            schema.KEY_COL: [1, 2, 3, 4],
            schema.TIME_COL: pd.to_datetime(
                [
                    "2026-01-01 00:00",
                    "2026-01-01 01:00",
                    "2026-01-01 02:00",
                    "2026-01-01 03:00",
                ]
            ),
            "s060": [10.0, 11.0, 12.0, 13.0],
            "s123": [0.1, 0.2, 0.3, 0.4],
        }
    )


def _tiny_design() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "run_id": ["doe_001", "doe_002"],
            "run_order": [1, 2],
            "s060": [10.0, 13.0],
            "s123": [0.1, 0.4],
        }
    )


def test_resolve_surrogate_and_score_design(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    background = _tiny_background()
    design = _tiny_design()
    features = ["s060", "s123"]

    train_dir = tmp_path / "train"
    train_dir.mkdir()
    (train_dir / "features.json").write_text(json.dumps({"features": features}), encoding="utf-8")

    X = background[features].to_numpy(dtype=float)
    y = np.array([0, 0, 1, 1])
    model = xgb.XGBClassifier(n_estimators=2, max_depth=1, random_state=7)
    model.fit(X, y)
    model.get_booster().save_model(train_dir / "model.ubj")

    cal_dir = tmp_path / "cal"
    cal_dir.mkdir()
    calibrator = LogisticRegression().fit(X[:, [0]], y)
    joblib.dump(calibrator, cal_dir / "calibrator_platt.joblib")

    monkeypatch.setattr(
        "semcon.doe_run.resolve_runs",
        lambda run, no_cal: (train_dir, cal_dir, train_dir.name, cal_dir.name),
    )

    surrogate = resolve_surrogate(run="latest", no_cal=False)

    out = score_design(
        background,
        design,
        surrogate,
        noise_mode="bernoulli",
        gaussian_sigma=None,
        seed=7,
    )

    assert len(out) == len(background) * len(design)
    assert {"run_id", "score_raw", "p_cal", "y_observed"}.issubset(out.columns)
    assert out["run_id"].nunique() == 2


def test_score_design_ood_flags_expected_structure() -> None:
    background = _tiny_background()
    design = _tiny_design()

    out = score_design_ood(design, background, ["s060", "s123"])

    assert len(out) == 2
    assert {"mahalanobis", "knn_distance", "ood_flag"}.issubset(out.columns)
    assert out["mahalanobis"].ge(0).all()
    assert out["knn_distance"].ge(0).all()


def test_write_run_artifacts_and_append_index(tmp_path: Path) -> None:
    predictions = pd.DataFrame({"run_id": ["doe_001"], "score_raw": [0.1], "p_cal": [0.2]})
    ood = pd.DataFrame({"run_id": ["doe_001"], "mahalanobis": [1.0], "knn_distance": [0.5]})
    run_config = {"label": "demo", "note": "test"}

    run_dir, predictions_path, ood_path, config_path = write_run_artifacts(
        predictions, ood, run_config
    )

    assert run_dir.exists()
    assert predictions_path.exists()
    assert ood_path.exists()
    assert config_path.exists()

    append_doe_index(run_dir, {"type": "doe", "parent_run": "train", "note": "test"})

    from semcon.paths import ARTIFACTS

    assert (ARTIFACTS / "index.csv").exists()


def test_load_background_frame_uses_extract_window(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def fake_extract(engine, start, end, cutoff, exclude_after):
        called["args"] = (start, end, cutoff, exclude_after)
        return _tiny_background()

    monkeypatch.setattr("semcon.doe_run.extract", fake_extract)
    monkeypatch.setattr("semcon.doe_run.get_engine", lambda: DummyEngine())

    out = load_background_frame("2026-01-01", "2026-01-02")

    assert not out.empty
    assert called["args"] == ("2026-01-01", "2026-01-02", None, None)
