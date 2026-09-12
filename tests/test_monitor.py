"""Unit and regression tests for semcon.monitor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from semcon.monitor import (
    compute_prediction_entropy,
    evaluate_feature_drift,
    evaluate_monitoring_rules,
    evaluate_output_health,
)


def _make_dummy_scores(n: int = 50, high_risk_frac: float = 0.05) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    scores = rng.uniform(0.01, 0.10, size=n)
    n_high = int(n * high_risk_frac)
    if n_high > 0:
        scores[:n_high] = rng.uniform(0.20, 0.50, size=n_high)
    return pd.DataFrame(
        {
            "wafer_id": np.arange(1, n + 1),
            "score_raw": scores,
            "p_cal": scores,
            "rank": np.arange(1, n + 1),
            "decile": np.repeat(1, n),
        }
    )


def _make_dummy_limits(features: list[str]) -> pd.DataFrame:
    rows = {}
    for f in features:
        rows[f] = {
            "center": 10.0,
            "sigma": 1.0,
            "lcl": 7.0,
            "ucl": 13.0,
            "mr_bar": 1.128,
            "mr_ucl": 3.68,
            "degenerate": False,
        }
    return pd.DataFrame(rows).T


def test_compute_prediction_entropy():
    # Symmetric uncertain probabilities (p=0.5) have max binary entropy of 1.0
    p = np.array([0.5, 0.5, 0.5])
    assert np.isclose(compute_prediction_entropy(p), 1.0, atol=1e-3)

    # Near-zero probabilities have low entropy
    p_low = np.array([0.01, 0.01, 0.01])
    assert compute_prediction_entropy(p_low) < 0.15


def test_evaluate_output_health_in_control():
    scores = _make_dummy_scores(n=100, high_risk_frac=0.04)  # 4% triage rate < 20% UCL
    health = evaluate_output_health(scores, high_risk_threshold=0.15, triage_ucl=0.20)

    assert isinstance(health, dict)
    assert health["n_wafers"] == 100
    assert health["triage_rate"] == 0.04
    assert health["risk_ooc"] is False


def test_evaluate_output_health_excursion():
    scores = _make_dummy_scores(n=100, high_risk_frac=0.35)  # 35% triage rate >= 20% UCL
    health = evaluate_output_health(scores, high_risk_threshold=0.15, triage_ucl=0.20)

    assert isinstance(health, dict)
    assert health["n_wafers"] == 100
    assert health["triage_rate"] == 0.35
    assert health["risk_ooc"] is True


def test_evaluate_feature_drift_nominal():
    features = ["s022", "s060", "s104", "s511"]
    limits = _make_dummy_limits(features)

    # Sensors generated around center=10.0, sigma=1.0
    rng = np.random.default_rng(123)
    data = {f: rng.normal(10.0, 0.8, size=50) for f in features}
    batch = pd.DataFrame(data)

    drift = evaluate_feature_drift(batch, limits, features, drift_feature_fraction=0.25)
    assert isinstance(drift, dict)
    assert drift["n_features_checked"] == 4
    assert drift["n_features_drifted"] == 0
    assert drift["feature_drift_flag"] is False


def test_evaluate_feature_drift_shift_detected():
    features = ["s022", "s060", "s104", "s511"]
    limits = _make_dummy_limits(features)

    rng = np.random.default_rng(123)
    data = {f: rng.normal(10.0, 0.8, size=50) for f in features}
    # Induce strong +4 sigma shift on s060 and s104 (2 of 4 = 50% >= 25% threshold)
    data["s060"] = rng.normal(14.5, 0.5, size=50)
    data["s104"] = rng.normal(14.0, 0.5, size=50)
    batch = pd.DataFrame(data)

    drift = evaluate_feature_drift(batch, limits, features, drift_feature_fraction=0.25)
    assert isinstance(drift, dict)
    assert drift["n_features_drifted"] >= 2
    assert "s060" in drift["drifted_features"]
    assert drift["feature_drift_flag"] is True
    assert drift["max_feature_z"] >= 4.0


def test_evaluate_monitoring_rules_in_control():
    health = {
        "n_wafers": 100,
        "mean_risk": 0.05,
        "std_risk": 0.02,
        "triage_rate": 0.04,
        "entropy": 0.3,
        "risk_ooc": False,
    }
    drift = {
        "n_features_checked": 10,
        "n_features_drifted": 0,
        "drift_fraction": 0.0,
        "drifted_features": [],
        "max_feature_z": 1.2,
        "feature_drift_flag": False,
    }

    verdict = evaluate_monitoring_rules(health, drift)
    assert isinstance(verdict, dict)
    assert verdict["verdict"] == "IN_CONTROL"
    assert "Continue standard" in verdict["recommended_action"]


def test_evaluate_monitoring_rules_investigate_chamber():
    health = {
        "n_wafers": 100,
        "mean_risk": 0.06,
        "std_risk": 0.02,
        "triage_rate": 0.05,
        "entropy": 0.35,
        "risk_ooc": False,
    }
    drift = {
        "n_features_checked": 10,
        "n_features_drifted": 4,
        "drift_fraction": 0.40,
        "drifted_features": ["s060", "s104"],
        "max_feature_z": 3.8,
        "feature_drift_flag": True,
    }

    verdict = evaluate_monitoring_rules(health, drift)
    assert isinstance(verdict, dict)
    assert verdict["verdict"] == "INVESTIGATE_CHAMBER"
    assert "inspect physical chambers" in verdict["recommended_action"]


def test_evaluate_monitoring_rules_retrain_recommended():
    health = {
        "n_wafers": 100,
        "mean_risk": 0.28,
        "std_risk": 0.12,
        "triage_rate": 0.38,
        "entropy": 0.85,
        "risk_ooc": True,
    }
    drift = {
        "n_features_checked": 10,
        "n_features_drifted": 5,
        "drift_fraction": 0.50,
        "drifted_features": ["s060", "s104", "s022"],
        "max_feature_z": 4.5,
        "feature_drift_flag": True,
    }

    verdict = evaluate_monitoring_rules(health, drift)
    assert isinstance(verdict, dict)
    assert verdict["verdict"] == "RETRAIN_RECOMMENDED"
    assert "Trigger model retraining" in verdict["recommended_action"]
