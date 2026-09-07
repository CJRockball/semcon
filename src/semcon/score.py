"""Batch scoring: extract a time window, rebuild features, score, rank, register.

The serving path mirrors evaluation exactly: same extraction SQL, same
build_features, same feature contract (features.json from the training run).
The run registry doubles as the model store: --run <id> or the latest training
run in index.csv; the Platt calibrator is resolved through the
type='calibration' + parent_run linkage.

Entry point: semcon-score = semcon.score:main
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from semcon import schema
from semcon.db import get_engine
from semcon.extract import extract
from semcon.feature_eng import build_features
from semcon.paths import ARTIFACTS, LOGS
from semcon.utils import setup_logging

logger = logging.getLogger("semcon")

RUNS = ARTIFACTS / "runs"
SCORES = ARTIFACTS / "scores"
INDEX = ARTIFACTS / "index.csv"
CALIBRATOR_NAME = "calibrator_platt.joblib"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()[:7]
    except Exception:
        return "unknown"


def load_contract(run_dir: Path) -> list[str]:
    """Training feature list — the exact columns the model was fit on."""
    spec = json.loads((run_dir / "features.json").read_text())
    features = spec["features"] if isinstance(spec, dict) else spec
    if not features:
        raise ValueError(f"empty feature contract in {run_dir / 'features.json'}")
    return list(features)


def check_contract(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Raise on missing columns; return the frame reordered to the contract."""
    missing = [c for c in features if c not in frame.columns]
    if missing:
        raise ValueError(
            f"feature contract violation: {len(missing)} missing columns, first ten: {missing[:10]}"
        )
    return frame[features].astype("float32")


def resolve_runs(run: str, no_cal: bool) -> tuple[Path, Path | None, str, str | None]:
    """Resolve the training run (and its calibrator run) from the registry."""
    idx = pd.read_csv(INDEX, dtype=str).fillna("")
    trained = idx[
        (idx["type"] == "") & idx["run_id"].map(lambda r: (RUNS / r / "model.ubj").exists())
    ]
    if trained.empty:
        raise ValueError(f"no training runs with model.ubj found via {INDEX}")
    if run == "latest":
        train_id = trained["run_id"].max()
    else:
        if run not in set(trained["run_id"]):
            raise ValueError(f"run {run!r} not found among training runs in {INDEX}")
        train_id = run
    cal_id = None
    if not no_cal:
        cals = idx[
            (idx["type"] == "calibration")
            & (idx["parent_run"] == train_id)
            & idx["run_id"].map(lambda r: (RUNS / r / CALIBRATOR_NAME).exists())
        ]
        if not cals.empty:
            cal_id = cals["run_id"].max()
        else:
            logger.warning("no calibrator for %s — scoring uncalibrated", train_id)
    train_dir = RUNS / train_id
    cal_dir = (RUNS / cal_id) if cal_id else None
    return train_dir, cal_dir, train_id, cal_id


def apply_calibrator(calibrator, raw: np.ndarray) -> np.ndarray:
    """Platt-style map from raw score to calibrated probability."""
    x = np.asarray(raw, dtype=float).reshape(-1, 1)
    if hasattr(calibrator, "predict_proba"):
        return calibrator.predict_proba(x)[:, 1]
    if hasattr(calibrator, "predict"):
        return np.asarray(calibrator.predict(x), dtype=float)
    raise TypeError(f"unsupported calibrator type: {type(calibrator)!r}")


def score_frame(
    frame: pd.DataFrame,
    booster: xgb.Booster,
    features: list[str],
    calibrator,
) -> pd.DataFrame:
    engineered, _ = build_features(frame)  # registry rows discarded: scoring never registers
    X = check_contract(engineered, features)
    dm = xgb.DMatrix(X.to_numpy(), feature_names=features)
    raw = booster.predict(dm)
    p = apply_calibrator(calibrator, raw) if calibrator is not None else raw
    out = pd.DataFrame(
        {
            schema.KEY_COL: frame[schema.KEY_COL].to_numpy(),
            schema.TIME_COL: frame[schema.TIME_COL].to_numpy(),
            "score_raw": raw,
            "p_cal": p if calibrator is not None else np.nan,
        }
    )
    out["rank"] = (
        out["p_cal"].fillna(out["score_raw"]).rank(ascending=False, method="first").astype(int)
    )
    out["decile"] = ((out["rank"] - 1) * 10 // len(out)) + 1
    return out.sort_values("rank").reset_index(drop=True)


def reconcile(out: pd.DataFrame, train_dir: Path, cal_dir: Path | None, tol: float = 1e-6) -> dict:
    """Compare scores against stored holdout predictions (evaluation path).

    Positional match is the strong form. Legacy prediction arrays carry no
    wafer_id, so on positional mismatch we fall back to a sorted value-set
    comparison and say so plainly in the report.
    """
    if cal_dir is not None and (cal_dir / "p_hold_cal.npy").exists():
        ref_path, col = cal_dir / "p_hold_cal.npy", "p_cal"
    else:
        ref_path, col = train_dir / "p_hold.npy", "score_raw"
    ref = np.load(ref_path)
    got = out[col].to_numpy()
    report = {
        "reference": ref_path.name,
        "column": col,
        "n_reference": int(len(ref)),
        "n_scored": int(len(got)),
    }
    if len(ref) != len(got):
        report.update(positional_max_abs_diff=None, verdict="fail_length")
        logger.warning("reconcile: length mismatch ref=%d scored=%d", len(ref), len(got))
        return report
    pos = float(np.max(np.abs(ref - got)))
    report["positional_max_abs_diff"] = pos
    if pos < tol:
        report["verdict"] = "pass"
        logger.info("reconcile: positional match, max |diff| = %.3e vs %s", pos, ref_path.name)
        return report
    srt = float(np.max(np.abs(np.sort(ref) - np.sort(got))))
    report["sorted_max_abs_diff"] = srt
    if srt < tol:
        report["verdict"] = "pass_values_only"
        logger.warning(
            "reconcile: value sets match (sorted max |diff| = %.3e) but row order differs "
            "from %s — legacy arrays carry no wafer_id; positional order unverifiable",
            srt,
            ref_path.name,
        )
    else:
        report["verdict"] = "fail_values"
        logger.warning(
            "reconcile: VALUE MISMATCH sorted max |diff| = %.3e vs %s — real divergence",
            srt,
            ref_path.name,
        )
    return report


def append_index(row: dict) -> None:
    """One registry row per scoring batch — no run counts unless it's in the index."""
    with open(INDEX, newline="") as f:
        fields = next(csv.reader(f))
    with open(INDEX, "a", newline="") as f:
        csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore").writerow(row)


def main() -> None:
    global logger
    logger = setup_logging(logfile=LOGS / "score.log")
    ap = argparse.ArgumentParser(description="Batch-score a time window of wafers.")
    ap.add_argument("--start", required=True, help="window start, e.g. '2026-01-05 00:00'")
    ap.add_argument("--end", required=True, help="window end (inclusive)")
    ap.add_argument("--label", required=True, help="batch label, e.g. batch_a_clean")
    ap.add_argument("--run", default="latest", help="training run id or 'latest'")
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument(
        "--reconcile",
        action="store_true",
        help="compare against stored holdout predictions (holdout replay)",
    )
    args = ap.parse_args()

    logger.info("[score] start | window %s..%s label=%s", args.start, args.end, args.label)
    engine = get_engine()
    # cutoff=None + exclude_after=None -> every row 'unassigned'; the frozen
    # cv/holdout/excluded taxonomy is never touched. Both must be None: the
    # SQL checks exclude_after first.
    frame = extract(engine, start=args.start, end=args.end, cutoff=None, exclude_after=None)
    if frame.empty:
        raise ValueError(f"no wafers in window {args.start}..{args.end}")
    train_dir, cal_dir, train_id, cal_id = resolve_runs(args.run, args.no_calibrate)
    features = load_contract(train_dir)
    booster = xgb.Booster()
    booster.load_model(train_dir / "model.ubj")
    calibrator = joblib.load(cal_dir / CALIBRATOR_NAME) if cal_dir else None

    out = score_frame(frame, booster, features, calibrator)
    logger.info("[score] scored n=%d", len(out))

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_score__{args.label}"
    out_dir = SCORES / run_id
    out_dir.mkdir(parents=True, exist_ok=False)
    out.to_parquet(out_dir / "scores.parquet", index=False)
    config = {
        "run_id": run_id,
        "timestamp": ts,
        "git_sha": git_sha(),
        "config": {
            "script": "score",
            "label": args.label,
            "start": args.start,
            "end": args.end,
            "train_run": train_id,
            "cal_run": cal_id,
            "n_scored": int(len(out)),
        },
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    if args.reconcile:
        report = reconcile(out, train_dir, cal_dir)
        (out_dir / "reconciliation.json").write_text(json.dumps(report, indent=2))
    append_index(
        {
            "run_id": run_id,
            "git_sha": config["git_sha"],
            "note": f"window {args.start}..{args.end}; cal={cal_id or 'none'}; n={len(out)}",
            "type": "score",
            "parent_run": train_id,
        }
    )
    logger.info("[score] top of queue:\n%s", out.head(10).to_string(index=False))
    logger.info("[score] done -> %s", out_dir)


if __name__ == "__main__":
    main()
