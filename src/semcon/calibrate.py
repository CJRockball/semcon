#%%
"""Probability calibration for a trained XGBoost run.

A monotone remap (Platt by default, isotonic optional) fit on the parent run's
OOF predictions ONLY, then applied to the holdout. Calibration repairs the
score scale distorted by scale_pos_weight: Brier drops, thresholds become
interpretable probabilities. It cannot change the ranking — PR-AUC/ROC-AUC
are invariant by construction (asserted below on every run).

Creates its own run folder with parent_run recorded — a derived-experiment
demo for the tracking registry. Requires the parent run to contain
oof_xgb1.npy and p_hold.npy.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: save figures, never plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from semcon import schema
from semcon.db import get_engine
from semcon.extract import extract
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)

from semcon.paths import ARTIFACTS, LOGS
from semcon.utils import setup_logging
from semcon import tracking
from semcon.validate import ensure_is_fail


logger = logging.getLogger("semcon")


# ---------------------------------------------------------------- cli

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Calibrate a trained run's probabilities")
    p.add_argument("--run-id", default=None,
                   help="parent training run folder; default = latest *_xgb_* run")
    p.add_argument("--method", choices=["platt", "isotonic"], default="platt",
                   help="platt is the stabler choice at ~90 OOF positives")
    p.add_argument("--note", default="")
    return p.parse_args(argv)


# ---------------------------------------------------------------- data

def find_run(run_id: str | None) -> Path:
    runs_root = ARTIFACTS / "runs"
    if run_id is not None:
        run = runs_root / run_id
        if not run.is_dir():
            raise FileNotFoundError(f"no run named {run_id} in {runs_root}")
        return run
    candidates = [r for r in sorted(runs_root.iterdir())
                  if r.is_dir()
                  and (r / "oof_xgb1.npy").exists()
                  and (r / "p_hold.npy").exists()]
    if not candidates:
        raise FileNotFoundError(
            f"no calibratable runs (oof_xgb1.npy + p_hold.npy) under {runs_root}")
    return candidates[-1]


def load_labels_from_run(run_dir: Path) -> tuple[pd.Series, dict]:
    """Full label vector in split-index order, plus the run's splits.json.

    splits.json records WHICH positions (timestamp-sorted frame); the DB
    records WHAT the labels are. Caller slices cv/holdout via .iloc.
    """
    splits_path = run_dir / "splits.json"
    assert splits_path.exists(), f"no splits.json in {run_dir} — rerun train_xgb"
    splits = json.loads(splits_path.read_text())

    engine = get_engine()
    df = extract(engine).sort_values([schema.TIME_COL, schema.KEY_COL]).reset_index(drop=True)

    df = ensure_is_fail(df, engine)
    y = df["is_fail"]

    return y.reset_index(drop=True), splits


def load_parent(run: Path):
    """OOF scores + holdout scores + labels, reconstructed via splits.json."""
    oof = np.load(run / "oof_xgb1.npy")              # (repeats, n_cv)
    p_hold = np.load(run / "p_hold.npy")
    y, splits = load_labels_from_run(run)

    y_cv = y.iloc[np.asarray(splits["train_index"])]
    y_hold = y.iloc[np.asarray(splits["holdout_index"])]

    assert len(y_cv) == oof.shape[-1], (
        f"label/OOF mismatch: {len(y_cv)} vs {oof.shape[-1]}")
    assert len(y_hold) == len(p_hold), (
        f"holdout label/pred mismatch: {len(y_hold)} vs {len(p_hold)}")

    expected = splits.get("n_train_fails")           # present on new runs only
    if expected is not None:
        assert int(y_cv.sum()) == expected, (
            f"CV fails {int(y_cv.sum())} != recorded {expected}")

    return oof.mean(axis=0), y_cv, p_hold, y_hold


# ---------------------------------------------------------------- calibration

def fit_calibrator(method: str, scores: np.ndarray, y: np.ndarray):
    if method == "platt":
        return LogisticRegression().fit(scores.reshape(-1, 1), y)
    return IsotonicRegression(out_of_bounds="clip").fit(scores, y)


def predict_calibrated(cal, method: str, scores) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if method == "platt":
        return cal.predict_proba(s.reshape(-1, 1))[:, 1]
    return cal.predict(s)


def save_reliability(y, p_raw, p_cal, out: Path, label: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    for p, lb, c in [(p_raw, "raw", "tab:red"), (p_cal, "calibrated", "tab:blue")]:
        frac, mean_pred = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac, "o-", ms=4, lw=1.2, color=c, label=lb)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    ax.set(xlabel="mean predicted probability", ylabel="observed fail fraction",
           title=f"Reliability — {label}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main(argv=None):
    args = parse_args(argv)
    setup_logging(logfile=LOGS / "ml.log")

    parent = find_run(args.run_id)
    oof_mean, y_cv, p_hold_raw, y_hold = load_parent(parent)
    logger.info(f"[calibrate] parent={parent.name} method={args.method}")

    cal = fit_calibrator(args.method, oof_mean, y_cv)
    oof_cal = predict_calibrated(cal, args.method, oof_mean)
    p_hold_cal = predict_calibrated(cal, args.method, p_hold_raw)
    
    # calibration is monotone: ranking metrics must be invariant
    assert np.isclose(roc_auc_score(y_hold, p_hold_raw),
                      roc_auc_score(y_hold, p_hold_cal))

    metrics = {
        "type": "calibration",
        "parent_run": parent.name,
        "method": args.method,
        "brier_oof_raw": round(float(brier_score_loss(y_cv, oof_mean)), 4),
        "brier_oof_cal": round(float(brier_score_loss(y_cv, oof_cal)), 4),
        "brier_holdout_raw": round(float(brier_score_loss(y_hold, p_hold_raw)), 4),
        "brier_holdout_cal": round(float(brier_score_loss(y_hold, p_hold_cal)), 4),
        "holdout_aucpr": round(float(average_precision_score(y_hold, p_hold_cal)), 4),
    }

    parent_slug = parent.name.split("_", 2)[-1]        # "sel_g025" from "20260829_085633_sel_g025"
    run_dir, meta = tracking.make_run(
        config={"script": "calibrate", "method": args.method, "parent_run": parent.name},
        run_name=f"cal-{args.method}__{parent_slug}",
        note=args.note)
    logger.info(f"run {run_dir.name} | git {meta['git_sha']}")

    joblib.dump(cal, run_dir / f"calibrator_{args.method}.joblib")
    np.save(run_dir / "oof_cal.npy", oof_cal)
    np.save(run_dir / "p_hold_cal.npy", p_hold_cal)
    pd.DataFrame([metrics]).to_csv(run_dir / "calibration_metrics.csv", index=False)
    save_reliability(y_cv, oof_mean, oof_cal, run_dir / "reliability_oof.png", "OOF")
    save_reliability(y_hold, p_hold_raw, p_hold_cal,
                     run_dir / "reliability_holdout.png", "holdout")
    tracking.append_index(run_dir, metrics)

    logger.info(f"[calibrate] {metrics}")
    logger.info(f"[calibrate] done | artifacts -> {run_dir}")


if __name__ == "__main__":
    main()
