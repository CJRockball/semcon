"""Design generation for surrogate DOE experiments."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyDOE3 import fracfact, fullfact

from semcon import schema
from semcon.config import load_config
from semcon.paths import ARTIFACTS
from semcon.tracking import make_run

logger = logging.getLogger("semcon")


def load_selected_features(features_path: Path) -> list[str]:
    """Load selected model features from a features.json-style artifact."""
    with open(features_path, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        features = payload
    elif isinstance(payload, dict):
        features = payload.get("features")
    else:
        raise ValueError(f"Unsupported feature artifact format: {features_path}")

    if not features or not isinstance(features, list):
        raise ValueError(f"No feature list found in {features_path}")

    return [str(x) for x in features]


def _is_missingness_feature(name: str) -> bool:
    lowered = name.lower()
    return lowered.startswith("f_miss") or "missing" in lowered or "clique" in lowered


def _is_raw_sensor(name: str) -> bool:
    return (
        name.startswith(schema.SENSOR_PREFIX)
        and len(name) == 4
        and name[1:].isdigit()
    )


def select_doe_factors(
    selected_features: list[str],
    *,
    max_factors: int,
    preferred: list[str] | None = None,
) -> list[str]:
    """Filter selected model features down to valid DOE factor names."""
    if max_factors < 2:
        raise ValueError("DOE needs at least 2 factors")

    candidates = [f for f in selected_features if _is_raw_sensor(f)]
    candidates = [f for f in candidates if not _is_missingness_feature(f)]

    if not candidates:
        raise ValueError("No valid raw sensor factors available for DOE")

    preferred = preferred or []
    ordered = [f for f in preferred if f in candidates]
    ordered.extend([f for f in candidates if f not in ordered])

    return ordered[:max_factors]


def _coded_full_factorial(n_factors: int) -> np.ndarray:
    raw = fullfact([2] * n_factors)
    return np.where(raw == 0, -1.0, 1.0)


def _coded_fractional_factorial(generator: str) -> np.ndarray:
    coded = fracfact(generator)
    return coded.astype(float)


def _expand_replicates(coded: np.ndarray, replicates: int) -> np.ndarray:
    if replicates < 1:
        raise ValueError("replicates must be >= 1")
    return np.tile(coded, (replicates, 1))


def _append_center_points(coded: np.ndarray, center_points: int) -> np.ndarray:
    if center_points < 0:
        raise ValueError("center_points must be >= 0")
    if center_points == 0:
        return coded
    center = np.zeros((center_points, coded.shape[1]), dtype=float)
    return np.vstack([coded, center])


def _coded_to_actual(coded_value: float, factor: dict) -> float:
    if coded_value == -1.0:
        return float(factor["low"])
    if coded_value == 0.0:
        if factor.get("center") is not None:
            return float(factor["center"])
        return float((factor["low"] + factor["high"]) / 2.0)
    if coded_value == 1.0:
        return float(factor["high"])
    raise ValueError(f"Unsupported coded level {coded_value} for factor {factor['name']}")


def build_factorial_design(
    factors: list[dict],
    *,
    fraction: str | None = None,
    run_prefix: str = "doe",
    center_points: int | None = None,
    replicates: int | None = None,
    randomize: bool | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """Build a two-level factorial design table using pyDOE3."""
    cfg = load_config().doe
    center_points = cfg.center_points if center_points is None else center_points
    replicates = cfg.replicates if replicates is None else replicates
    randomize = cfg.randomize if randomize is None else randomize
    seed = cfg.seed if seed is None else seed

    factor_names = [f["name"] for f in factors]
    if len(factor_names) < 2:
        raise ValueError("At least two factors are required")
    if len(set(factor_names)) != len(factor_names):
        raise ValueError("Factor names must be unique")

    if fraction:
        coded = _coded_fractional_factorial(fraction)
        if coded.shape[1] != len(factors):
            raise ValueError("Fractional design generator does not match number of factors")
    else:
        coded = _coded_full_factorial(len(factors))

    coded = _expand_replicates(coded, replicates)
    coded = _append_center_points(coded, center_points)

    rows: list[dict[str, object]] = []
    base_rows = len(coded) // replicates

    for i, row in enumerate(coded, start=1):
        rec: dict[str, object] = {
            "design_row": i,
            "is_center": bool(np.allclose(row, 0.0)),
            "replicate": ((i - 1) // base_rows) + 1,
        }
        for j, factor in enumerate(factors):
            coded_value = float(row[j])
            rec[f"{factor['name']}_coded"] = coded_value
            rec[factor["name"]] = _coded_to_actual(coded_value, factor)
        rows.append(rec)

    design = pd.DataFrame(rows)
    design["is_replicate"] = design["replicate"] > 1

    if randomize:
        rng = np.random.default_rng(seed)
        design["run_order"] = rng.permutation(len(design)) + 1
    else:
        design["run_order"] = np.arange(1, len(design) + 1)

    design.insert(0, "run_id", [f"{run_prefix}_{i:03d}" for i in range(1, len(design) + 1)])
    design = design.sort_values("run_order").reset_index(drop=True)
    return design


def describe_design(
    factors: list[dict],
    *,
    fraction: str | None = None,
    run_prefix: str = "doe",
    center_points: int | None = None,
    replicates: int | None = None,
    randomize: bool | None = None,
    seed: int | None = None,
) -> dict:
    """Return a JSON-serializable description of the design configuration."""
    cfg = load_config().doe
    return {
        "seed": cfg.seed if seed is None else seed,
        "center_points": cfg.center_points if center_points is None else center_points,
        "replicates": cfg.replicates if replicates is None else replicates,
        "randomize": cfg.randomize if randomize is None else randomize,
        "fraction": fraction,
        "run_prefix": run_prefix,
        "factors": factors,
    }


def write_design_artifacts(
    design: pd.DataFrame,
    metadata: dict,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Persist the design matrix and metadata to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    design_csv = output_dir / "design.csv"
    design_json = output_dir / "design_metadata.json"

    design.to_csv(design_csv, index=False)
    with open(design_json, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return design_csv, design_json