"""Artifact loaders for the semcon monitoring dashboard.

Pure consumer layer: this module reads what the pipeline persisted and
returns plain pandas objects. It never recomputes SPC statistics, never
scores wafers, and never imports Dash or Plotly - that is what makes it
unit-testable in CI without a browser.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from semcon.paths import ARTIFACTS

logger = logging.getLogger("semcon")

PATHS = {
    "monitor_index": ARTIFACTS / "index_monitor.csv",
    "training_index": ARTIFACTS / "index.csv",
    "runs": ARTIFACTS / "runs",
    "scores": ARTIFACTS / "scores",
}

DEFAULT_FILES = {
    "manifest": "files.json",
    "config": "config.json",
    "limits": "spc_limits.csv",
    "screening": "spc_screening.csv",
    "pchart": "pchart_data.csv",
    "protocol_rates": "protocol_rates.csv",
    "protocol_row_missing": "protocol_row_missing.csv",
    "imr_series": "imr_series.csv",
    "scores": "scores.parquet",
    "scorecard": "scorecard.json",
    "reconciliation": "reconciliation.json",
}

SCORE_COL_CANDIDATES = (
    "p_cal",
    "prob_cal",
    "p_calibrated",
    "calibrated",
    "p_hold",
    "prob",
    "score",
    "y_score",
)


def run_label(run: dict) -> str:
    run_id = run["run_id"]
    return run_id.split("__", 1)[1] if "__" in run_id else run_id


def _read_registry(index_path: Path, kind: str) -> pd.DataFrame:
    if not index_path.exists():
        raise FileNotFoundError(f"registry not found: {index_path}")
    df = pd.read_csv(index_path)
    if "run_id" not in df.columns:
        raise ValueError(f"{index_path.name} lacks a run_id column: {list(df.columns)}")
    if "type" in df.columns:
        df = df[df["type"] == kind]
    if df.empty:
        raise ValueError(f"no '{kind}' rows in {index_path.name}")
    return df


def _to_runs(df: pd.DataFrame, base: Path, default_kind: str) -> list[dict]:
    return [
        {
            "run_id": str(r["run_id"]),
            "kind": str(r.get("type", default_kind)),
            "path": base / str(r["run_id"]),
            "meta": dict(r),
        }
        for _, r in df.iterrows()
    ]


def list_monitor_runs(index_path: Path | None = None) -> list[dict]:
    index_path = index_path or PATHS["monitor_index"]
    df = _read_registry(index_path, "spc")
    return _to_runs(df, index_path.parent / PATHS["runs"].name, "spc")


def list_training_runs(index_path: Path | None = None) -> list[dict]:
    index_path = index_path or PATHS["training_index"]
    if not index_path.exists():
        raise FileNotFoundError(f"registry not found: {index_path}")
    df = pd.read_csv(index_path)
    if "run_id" not in df.columns:
        raise ValueError(f"{index_path.name} lacks a run_id column: {list(df.columns)}")
    if df.empty:
        raise ValueError(f"no rows in {index_path.name}")
    return _to_runs(df, index_path.parent / PATHS["runs"].name, "run")


def latest_monitor_run(index_path: Path | None = None) -> dict:
    return list_monitor_runs(index_path)[-1]


def latest_training_run(index_path: Path | None = None) -> dict:
    return list_training_runs(index_path)[-1]


def _manifest(run: dict) -> dict:
    p = run["path"] / DEFAULT_FILES["manifest"]
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def resolve_artifact(run: dict, role: str) -> Path:
    names = {**DEFAULT_FILES, **_manifest(run)}
    if role not in names:
        raise ValueError(f"unknown artifact role '{role}'; known: {sorted(names)}")
    return run["path"] / names[role]


def artifact_path(run: dict, role: str) -> Path:
    p = resolve_artifact(run, role)
    if not p.exists():
        raise FileNotFoundError(f"expected artifact missing for role '{role}': {p}")
    return p


def load_run_config(run: dict) -> dict:
    return json.loads(artifact_path(run, "config").read_text())


def load_screening(run: dict) -> pd.DataFrame:
    df = pd.read_csv(artifact_path(run, "screening"), index_col="feature")
    df["degenerate"] = df["degenerate"].astype(bool)
    num = [c for c in df.columns if c != "degenerate"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    logger.info(f"screening: {len(df)} sensors ({int(df['degenerate'].sum())} degenerate)")
    return df


def load_limits(run: dict) -> pd.DataFrame:
    df = pd.read_csv(artifact_path(run, "limits"), index_col="feature")
    df["degenerate"] = df["degenerate"].astype(bool)
    num = [c for c in df.columns if c != "degenerate"]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return df


def load_drift_table(run: dict, delta_min: float = 0.05) -> pd.DataFrame:
    s = load_screening(run)
    s = s[~s["degenerate"] & s["delta"].notna()].copy()
    s["drift"] = s["delta"] >= delta_min
    return s.sort_values("delta", ascending=False)


def load_pchart(run: dict) -> pd.DataFrame:
    return pd.read_csv(artifact_path(run, "pchart"))


def load_protocol_rates(run: dict) -> pd.DataFrame:
    return pd.read_csv(artifact_path(run, "protocol_rates"))


def load_protocol_row_missing(run: dict) -> pd.DataFrame:
    return pd.read_csv(artifact_path(run, "protocol_row_missing"))


def load_imr_series(run: dict, feature: str | None = None) -> pd.DataFrame:
    df = pd.read_csv(artifact_path(run, "imr_series"))
    if feature is None:
        return df
    available = sorted(df["feature"].unique())
    if feature not in available:
        raise ValueError(f"no IMR series for '{feature}'; showcase features: {available}")
    return df[df["feature"] == feature].reset_index(drop=True)


def list_figures(run: dict) -> dict[str, Path]:
    figs = run["path"] / "figures"
    if not figs.is_dir():
        logger.warning(f"no figures dir for {run['run_id']}: {figs}")
        return {}
    return {p.stem: p for p in sorted(figs.glob("*.png"))}


def list_score_batches(scores_dir: Path | None = None) -> list[dict]:
    scores_dir = scores_dir or PATHS["scores"]
    if not scores_dir.is_dir():
        raise FileNotFoundError(f"scores directory not found: {scores_dir}")
    return [
        {"run_id": p.name, "kind": "score", "path": p, "meta": {}}
        for p in sorted(scores_dir.iterdir())
        if p.is_dir() and resolve_artifact({"path": p}, "scores").exists()
    ]


def latest_score_batch(scores_dir: Path | None = None) -> dict:
    batches = list_score_batches(scores_dir)
    if not batches:
        raise ValueError(f"no scored batches under {scores_dir or PATHS['scores']}")
    return batches[-1]


def _resolve_score_col(df: pd.DataFrame, score_col: str | None) -> str:
    if score_col is not None:
        if score_col not in df.columns:
            raise ValueError(f"score_col '{score_col}' not in {list(df.columns)}")
        return score_col
    for cand in SCORE_COL_CANDIDATES:
        if cand in df.columns and pd.api.types.is_numeric_dtype(df[cand]):
            return cand
    raise ValueError(
        f"no known probability column in scores.parquet; "
        f"columns are {list(df.columns)} - pass score_col explicitly"
    )


def load_score_queue(
    batch: dict, score_col: str | None = None, top_k: int | None = None
) -> pd.DataFrame:
    df = pd.read_parquet(artifact_path(batch, "scores"))
    col = _resolve_score_col(df, score_col)
    if "rank" in df.columns:
        df = df.sort_values("rank").reset_index(drop=True)
    else:
        logger.warning(f"{run_label(batch)}: no persisted rank, deriving from '{col}'")
        df = df.sort_values(col, ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
    if top_k is not None:
        df["in_top_k"] = df["rank"] <= top_k
    logger.info(f"queue: {len(df)} wafers from {run_label(batch)}, score column '{col}'")
    return df


def load_batch_summary(batch: dict) -> dict:
    out = {}
    for key, role in [("scorecard", "scorecard"), ("reconciliation", "reconciliation")]:
        p = resolve_artifact(batch, role)
        out[key] = json.loads(p.read_text()) if p.exists() else {}
    return out
