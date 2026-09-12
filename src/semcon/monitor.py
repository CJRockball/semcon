"""Monitoring and drift-to-retrain engine for batch production surveillance.

Surveils scored wafer batches from artifacts/scores/ (or incoming lot streams)
against:
  Stream A (Output Health):
    - Rolling calibrated failure risk (p_cal)
    - Triage excursion rate (fraction of wafers with p_cal >= high_risk_threshold)
    - Prediction entropy / dispersion
  Stream B (Input Feature Drift):
    - Robust Phase-I sensor limits (median +/- k * 1.4826 * MAD from spc.py)
    - Western Electric and out-of-control (OOC) rates across top model features

Produces deterministic fab-actionable verdicts:
  - IN_CONTROL: System is operating within statistical boundaries.
  - INVESTIGATE_CHAMBER: Input feature drift observed without risk distribution shift.
  - RETRAIN_RECOMMENDED: Persistent risk distribution shift or combined feature+risk excursion.

Entry point: semcon-monitor = semcon.monitor:main
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from semcon import tracking
from semcon.paths import ARTIFACTS, LOGS
from semcon.score import load_contract, resolve_runs
from semcon.spc import we_rules
from semcon.utils import setup_logging

logger = logging.getLogger("semcon")

RUNS = ARTIFACTS / "runs"
SCORES = ARTIFACTS / "scores"
MONITOR_DIR = ARTIFACTS / "monitoring"
INDEX_MONITOR = ARTIFACTS / "index_monitor.csv"


def compute_prediction_entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    """Compute normalized Shannon binary entropy of predicted risks."""
    p = np.clip(p, eps, 1.0 - eps)
    h = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return float(np.mean(h))


def evaluate_output_health(
    scores: pd.DataFrame,
    *,
    high_risk_threshold: float = 0.15,
    triage_ucl: float = 0.20,
) -> dict:
    """Evaluate Stream A: Calibrated output risk distributions."""
    col = "p_cal" if "p_cal" in scores.columns and scores["p_cal"].notna().any() else "score_raw"
    p = scores[col].dropna().to_numpy(dtype=float)
    if len(p) == 0:
        raise ValueError("scores table contains no valid prediction values")

    mean_risk = float(np.mean(p))
    std_risk = float(np.std(p))
    triage_rate = float(np.mean(p >= high_risk_threshold))
    entropy = compute_prediction_entropy(p)

    # Risk out of control if batch triage rate exceeds the upper threshold
    risk_ooc = bool(triage_rate >= triage_ucl)

    return {
        "n_wafers": int(len(p)),
        "mean_risk": round(mean_risk, 4),
        "std_risk": round(std_risk, 4),
        "triage_rate": round(triage_rate, 4),
        "entropy": round(entropy, 4),
        "risk_ooc": risk_ooc,
    }


def evaluate_feature_drift(
    batch_features: pd.DataFrame,
    phase1_limits: pd.DataFrame,
    key_features: list[str],
    *,
    drift_feature_fraction: float = 0.25,
) -> dict:
    """Evaluate Stream B: Feature deviations against frozen Phase-I limits."""
    valid_features = [
        f for f in key_features if f in phase1_limits.index and f in batch_features.columns
    ]
    if not valid_features:
        return {
            "n_features_checked": 0,
            "n_features_drifted": 0,
            "drift_fraction": 0.0,
            "drifted_features": [],
            "max_feature_z": 0.0,
            "feature_drift_flag": False,
        }

    drifted = []
    max_z = 0.0

    for f in valid_features:
        series = batch_features[f].dropna()
        if series.empty:
            continue
        center = float(phase1_limits.loc[f, "center"])
        sigma = float(phase1_limits.loc[f, "sigma"])
        if not np.isfinite(sigma) or sigma <= 0:
            continue

        z = np.abs((series.to_numpy(dtype=float) - center) / sigma)
        current_max_z = float(np.max(z))
        if current_max_z > max_z:
            max_z = current_max_z

        # Use spc.we_rules logic: rule 1 (|z| > 3) or >= 40% batch points beyond 2 sigma
        wr = we_rules(series, center, sigma)
        batch_ooc_rate = np.mean(z > 2.0)
        if wr["r1"].any() or batch_ooc_rate >= 0.40:
            drifted.append(f)

    drift_fraction = len(drifted) / len(valid_features)
    feature_drift_flag = bool(drift_fraction >= drift_feature_fraction)

    return {
        "n_features_checked": len(valid_features),
        "n_features_drifted": len(drifted),
        "drift_fraction": round(drift_fraction, 4),
        "drifted_features": drifted,
        "max_feature_z": round(max_z, 2),
        "feature_drift_flag": feature_drift_flag,
    }


def evaluate_monitoring_rules(
    output_health: dict,
    feature_drift: dict,
) -> dict:
    """Deterministic state-machine verdict based on twin-stream signals."""
    if output_health["risk_ooc"]:
        verdict = "RETRAIN_RECOMMENDED"
        rationale = (
            f"Risk excursion: triage rate {output_health['triage_rate']:.2%} exceeded UCL. "
            f"Feature drift: {feature_drift['n_features_drifted']}/{feature_drift['n_features_checked']} channels."
        )
        action = "Halt automated deployment. Trigger model retraining and recalibration pipeline."
    elif feature_drift["feature_drift_flag"]:
        verdict = "INVESTIGATE_CHAMBER"
        rationale = (
            f"Sensor drift detected on {feature_drift['n_features_drifted']}/{feature_drift['n_features_checked']} "
            f"key channels ({feature_drift['drifted_features'][:5]}), but output risk remains in control."
        )
        action = "Notify fab equipment engineer to inspect physical chambers, MFCs, and sensor calibration."
    else:
        verdict = "IN_CONTROL"
        rationale = (
            f"Normal operation: triage rate {output_health['triage_rate']:.2%}, "
            f"feature drift fraction {feature_drift['drift_fraction']:.1%} within limits."
        )
        action = "Continue standard automated wafer scoring."

    return {
        "verdict": verdict,
        "rationale": rationale,
        "output_health": output_health,
        "feature_drift": feature_drift,
        "recommended_action": action,
    }


def load_latest_scores(score_run: str | Path | None = None) -> tuple[pd.DataFrame, Path]:
    """Find and load score results from artifacts/scores."""
    if score_run is not None:
        p = Path(score_run)
        if p.is_file():
            return pd.read_parquet(p), p.parent
        if p.is_dir():
            pq = p / "scores.parquet"
            if pq.exists():
                return pd.read_parquet(pq), p
            raise FileNotFoundError(f"No scores.parquet found in {p}")

    # Fallback to latest directory in SCORES
    if not SCORES.exists():
        raise FileNotFoundError(f"Scores root not found: {SCORES}")
    score_dirs = sorted([d for d in SCORES.iterdir() if d.is_dir()])
    if not score_dirs:
        raise FileNotFoundError(f"No score runs found in {SCORES}")
    latest_dir = score_dirs[-1]
    pq = latest_dir / "scores.parquet"
    if not pq.exists():
        raise FileNotFoundError(f"scores.parquet not found in {latest_dir}")
    return pd.read_parquet(pq), latest_dir


def load_phase1_spc_limits(train_dir: Path) -> pd.DataFrame:
    """Load or generate robust Phase-I SPC limits for the contract features."""
    spc_csv = train_dir / "spc_limits.csv"
    if spc_csv.exists():
        return pd.read_csv(spc_csv, index_col=0)

    # Search in spc runs under ARTIFACTS / runs
    for run in sorted(RUNS.glob("spc_*"), reverse=True):
        candidate = run / "spc_limits.csv"
        if candidate.exists():
            return pd.read_csv(candidate, index_col=0)

    # Empty fallback frame if limits not precalculated
    return pd.DataFrame(columns=["center", "sigma", "lcl", "ucl", "degenerate"])


def run_monitoring(
    *,
    score_run: str | Path | None = None,
    run_id: str = "latest",
    note: str = "",
) -> dict:
    """Main execution entry point for monitoring."""
    train_dir, cal_dir, train_id, cal_id = resolve_runs(run_id, no_cal=False)
    features = load_contract(train_dir)
    scores, score_dir = load_latest_scores(score_run)
    limits = load_phase1_spc_limits(train_dir)

    out_health = evaluate_output_health(scores)
    feat_drift = evaluate_feature_drift(scores, limits, features)
    verdict = evaluate_monitoring_rules(out_health, feat_drift)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"monitor_{verdict['verdict'].lower()}"
    run_config = {
        "script": "monitor",
        "timestamp": ts,
        "train_run": train_id,
        "cal_run": cal_id,
        "score_dir": str(score_dir),
        "verdict": verdict["verdict"],
        "rationale": verdict["rationale"],
        "action": verdict["recommended_action"],
        "metrics": {
            "mean_risk": out_health["mean_risk"],
            "triage_rate": out_health["triage_rate"],
            "entropy": out_health["entropy"],
            "drift_fraction": feat_drift["drift_fraction"],
            "max_feature_z": feat_drift["max_feature_z"],
        },
    }

    run_dir, meta = tracking.make_run(
        config=run_config,
        run_name=run_name,
        note=note,
        runs_root=MONITOR_DIR,
    )

    report_path = run_dir / "monitor_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": meta,
                "config": run_config,
                "verdict": verdict,
            },
            f,
            indent=2,
            default=str,
        )

    # Append to artifacts/index_monitor.csv
    tracking.append_index(
        run_dir,
        {
            "type": "monitoring",
            "parent_run": train_id,
            "score_dir": str(score_dir.name),
            "verdict": verdict["verdict"],
            "triage_rate": out_health["triage_rate"],
            "mean_risk": out_health["mean_risk"],
            "drift_fraction": feat_drift["drift_fraction"],
            "n_drifted": feat_drift["n_features_drifted"],
            "note": note,
        },
        index_file=INDEX_MONITOR,
    )

    logger.info("[monitor] Verdict: %s | %s", verdict["verdict"], verdict["rationale"])
    logger.info("[monitor] Report written to: %s", report_path)
    return verdict


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Semiconductor ML Drift & Output Health Monitor")
    p.add_argument("--score-run", default=None, help="Path or folder of scored batch parquet")
    p.add_argument("--run", default="latest", help="Training run ID or 'latest'")
    p.add_argument("--note", default="", help="Optional audit note")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(logfile=LOGS / "monitor.log")
    run_monitoring(score_run=args.score_run, run_id=args.run, note=args.note)


if __name__ == "__main__":
    main()
