"""Run a surrogate DOE against a calibrated model."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from semcon import schema, tracking
from semcon.calibrate import predict_calibrated
from semcon.config import load_config
from semcon.db import get_engine
from semcon.extract import extract
from semcon.paths import ARTIFACTS, LOGS
from semcon.score import apply_calibrator, check_contract, load_contract, resolve_runs
from semcon.utils import setup_logging

logger = logging.getLogger("semcon")

RUNS = ARTIFACTS / "runs"
DOE_ROOT = ARTIFACTS / "doe"


def resolve_surrogate(run: str = "latest", no_cal: bool = False):
    """Resolve the training run and optional calibrator into one bundle."""
    train_dir, cal_dir, train_id, cal_id = resolve_runs(run, no_cal)
    features = load_contract(train_dir)

    booster = xgb.Booster()
    booster.load_model(train_dir / "model.ubj")

    calibrator = None
    calibrator_method = None
    if cal_dir is not None:
        calibrator = joblib.load(cal_dir / "calibrator_platt.joblib")
        calibrator_method = "platt"

    return train_dir, cal_dir, train_id, cal_id, features, booster, calibrator, calibrator_method


def load_background_frame(start: str, end: str, *, engine=None) -> pd.DataFrame:
    """Load the background wafers used for ALE-style averaging."""
    engine = engine or get_engine()
    frame = extract(engine, start=start, end=end, cutoff=None, exclude_after=None)
    if frame.empty:
        raise ValueError(f"no background wafers in window {start}..{end}")
    return frame.sort_values([schema.TIME_COL, schema.KEY_COL]).reset_index(drop=True)


def _design_factor_columns(design: pd.DataFrame) -> list[str]:
    return [
        c
        for c in design.columns
        if not c.endswith("_coded")
        and c not in {"run_id", "design_row", "is_center", "is_replicate", "replicate", "run_order"}
    ]


def _validate_design_frame(design: pd.DataFrame, features: list[str]) -> list[str]:
    factors = _design_factor_columns(design)
    missing = [c for c in factors if c not in features]
    if missing:
        raise ValueError(f"design factors missing from model contract: {missing}")
    return factors


def _build_design_matrix(
    background: pd.DataFrame,
    design: pd.DataFrame,
    factors: list[str],
) -> pd.DataFrame:
    expanded = []
    for _, row in design.iterrows():
        block = background.copy()
        for factor in factors:
            block[factor] = row[factor]
        block["run_id"] = row["run_id"]
        block["run_order"] = row["run_order"]
        expanded.append(block)
    return pd.concat(expanded, ignore_index=True)


def _predict_raw_scores(frame: pd.DataFrame, features: list[str], booster: xgb.Booster) -> np.ndarray:
    X = check_contract(frame, features)
    dm = xgb.DMatrix(X.to_numpy(), feature_names=features)
    return booster.predict(dm)


def _predict_probabilities(frame: pd.DataFrame, surrogate) -> tuple[np.ndarray, np.ndarray]:
    _, _, _, _, features, booster, calibrator, calibrator_method = surrogate
    raw = _predict_raw_scores(frame, features, booster)

    if calibrator is None:
        return raw, raw

    if calibrator_method == "platt":
        cal = predict_calibrated(calibrator, "platt", raw)
    else:
        cal = apply_calibrator(calibrator, raw)

    return raw, np.asarray(cal, dtype=float)


def _logit(p: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def _apply_noise(p: np.ndarray, noise_mode: str, gaussian_sigma: float | None, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)

    if noise_mode == "bernoulli":
        return rng.binomial(1, p).astype(float)

    if noise_mode == "gaussian_logit":
        if gaussian_sigma is None:
            raise ValueError("gaussian_sigma is required for gaussian_logit noise")
        noisy_logit = _logit(p) + rng.normal(0.0, gaussian_sigma, size=len(p))
        return 1.0 / (1.0 + np.exp(-noisy_logit))

    raise ValueError(f"unsupported noise mode: {noise_mode}")


def score_design(
    background: pd.DataFrame,
    design: pd.DataFrame,
    surrogate,
    *,
    noise_mode: str,
    gaussian_sigma: float | None,
    seed: int,
) -> pd.DataFrame:
    """Evaluate the design over the background frame and return run-level results."""
    factors = _validate_design_frame(design, surrogate[4])
    expanded = _build_design_matrix(background, design, factors)
    raw, p_cal = _predict_probabilities(expanded, surrogate)
    y_noisy = _apply_noise(p_cal, noise_mode, gaussian_sigma, seed)

    out = expanded[[schema.KEY_COL, schema.TIME_COL, "run_id", "run_order"]].copy()
    for factor in factors:
        out[factor] = expanded[factor].to_numpy()
    out["score_raw"] = raw
    out["p_cal"] = p_cal
    out["y_observed"] = y_noisy
    return out


def _mahalanobis_distance(x: np.ndarray, center: np.ndarray, cov_inv: np.ndarray) -> np.ndarray:
    delta = x - center
    return np.sqrt(np.einsum("ij,jk,ik->i", delta, cov_inv, delta))


def _knn_distance(x: np.ndarray, ref: np.ndarray, k: int = 5) -> np.ndarray:
    if len(ref) == 0:
        return np.full(len(x), np.inf)
    k = min(k, len(ref))
    d = np.sqrt(((x[:, None, :] - ref[None, :, :]) ** 2).sum(axis=2))
    return np.partition(d, kth=k - 1, axis=1)[:, :k].mean(axis=1)


def score_design_ood(
    design: pd.DataFrame,
    background: pd.DataFrame,
    factors: list[str],
) -> pd.DataFrame:
    """Compute simple OOD diagnostics for each design point."""
    if design.empty:
        raise ValueError("design is empty")
    if background.empty:
        raise ValueError("background is empty")

    ref = background[factors].to_numpy(dtype=float)
    pts = design[factors].to_numpy(dtype=float)

    center = np.nanmean(ref, axis=0)
    cov = np.cov(ref, rowvar=False)
    cov_inv = np.linalg.pinv(cov)

    mah = _mahalanobis_distance(pts, center, cov_inv)
    knn = _knn_distance(pts, ref, k=5)

    out = design[["run_id", "run_order"]].copy()
    for factor in factors:
        out[factor] = design[factor].to_numpy()
    out["mahalanobis"] = mah
    out["knn_distance"] = knn
    out["ood_flag"] = (mah > np.nanpercentile(mah, 95)) | (knn > np.nanpercentile(knn, 95))
    return out


def write_run_artifacts(predictions: pd.DataFrame, ood: pd.DataFrame, run_config: dict):
    """Persist the DOE run outputs under artifacts/doe/."""
    run_dir, meta = tracking.make_run(
        config=run_config,
        run_name=f"doe_{run_config['label']}",
        note=run_config.get("note", ""),
        runs_root=DOE_ROOT,
    )

    predictions_path = run_dir / "design_predictions.parquet"
    ood_path = run_dir / "ood_table.parquet"
    config_path = run_dir / "config.json"

    predictions.to_parquet(predictions_path, index=False)
    ood.to_parquet(ood_path, index=False)
    config_path.write_text(json.dumps(meta, indent=2, default=str))

    return run_dir, predictions_path, ood_path, config_path


def append_doe_index(run_dir: Path, metrics: dict) -> None:
    """Append one DOE run row to the registry index."""
    tracking.append_index(run_dir, metrics)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run a surrogate DOE against a calibrated model")
    p.add_argument("--design", required=True)
    p.add_argument("--background-start", required=True)
    p.add_argument("--background-end", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--run", default="latest")
    p.add_argument("--no-calibrate", action="store_true")
    p.add_argument("--noise-mode", default=None)
    p.add_argument("--gaussian-sigma", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--note", default="")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(logfile=LOGS / "doe_run.log")

    cfg = load_config().doe
    noise_mode = args.noise_mode or cfg.noise_mode
    gaussian_sigma = args.gaussian_sigma if args.gaussian_sigma is not None else cfg.gaussian_sigma
    seed = args.seed if args.seed is not None else cfg.seed

    design = pd.read_csv(args.design)
    surrogate = resolve_surrogate(run=args.run, no_cal=args.no_calibrate)
    background = load_background_frame(args.background_start, args.background_end)

    predictions = score_design(
        background,
        design,
        surrogate,
        noise_mode=noise_mode,
        gaussian_sigma=gaussian_sigma,
        seed=seed,
    )
    factors = _validate_design_frame(design, surrogate[4])
    ood = score_design_ood(design, background, factors)

    run_config = {
        "script": "doe_run",
        "label": args.label,
        "background_start": args.background_start,
        "background_end": args.background_end,
        "run": args.run,
        "train_run": surrogate[2],
        "cal_run": surrogate[3],
        "noise_mode": noise_mode,
        "gaussian_sigma": gaussian_sigma,
        "seed": seed,
        "note": args.note,
        "n_design_rows": int(len(design)),
        "n_background_rows": int(len(background)),
    }

    run_dir, predictions_path, ood_path, config_path = write_run_artifacts(predictions, ood, run_config)

    metrics = {
        "type": "doe",
        "parent_run": surrogate[2],
        "note": args.note,
        "n_design_rows": int(len(design)),
        "n_background_rows": int(len(background)),
        "noise_mode": noise_mode,
    }
    append_doe_index(run_dir, metrics)

    logger.info("[doe_run] wrote %s", run_dir)
    return run_dir, predictions_path, ood_path, config_path


if __name__ == "__main__":
    main()