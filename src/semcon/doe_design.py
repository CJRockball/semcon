"""Design generation for surrogate DOE experiments."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pyDOE3 import fracfact, fullfact

from semcon import schema, tracking
from semcon.config import load_config
from semcon.db import get_engine
from semcon.extract import extract
from semcon.feature_eng import build_features
from semcon.paths import ARTIFACTS, LOGS
from semcon.score import load_contract, resolve_runs
from semcon.utils import setup_logging

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
    return name.startswith(schema.SENSOR_PREFIX) and len(name) == 4 and name[1:].isdigit()


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
    cfg = load_config()
    center_points = cfg.doe.center_points if center_points is None else center_points
    replicates = cfg.doe.replicates if replicates is None else replicates
    randomize = cfg.doe.randomize if randomize is None else randomize
    seed = cfg.pipeline.seed if seed is None else seed

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
    cfg = load_config()
    return {
        "seed": cfg.pipeline.seed if seed is None else seed,
        "center_points": cfg.doe.center_points if center_points is None else center_points,
        "replicates": cfg.doe.replicates if replicates is None else replicates,
        "randomize": cfg.doe.randomize if randomize is None else randomize,
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


def load_shap_summary(train_dir: Path) -> pd.DataFrame:
    """Load ranked TreeSHAP feature importance from a training run."""
    path = train_dir / "shap" / "shap_feature_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing SHAP summary: {path}. "
            "Run semcon-trainxgb so the training run writes SHAP artifacts."
        )

    summary = pd.read_csv(path, index_col=0)
    if "mean_abs_shap" not in summary.columns:
        raise ValueError(f"Expected mean_abs_shap column in {path}; got {summary.columns.tolist()}")

    summary.index = summary.index.astype(str)
    return summary.sort_values("mean_abs_shap", ascending=False)


def rank_candidate_factors(
    selected_features: list[str],
    shap_summary: pd.DataFrame,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return eligible raw-sensor DOE factors ranked by SHAP importance.

    Missingness and engineered features are deliberately excluded: they are
    model inputs but not controllable physical process knobs.
    """
    eligible = select_doe_factors(
        selected_features,
        max_factors=len(selected_features),
        preferred=["s060"],
    )

    rows = []
    for feature in eligible:
        if feature not in frame.columns:
            continue

        values = pd.to_numeric(frame[feature], errors="coerce").dropna()
        if values.empty:
            continue

        shap_value = (
            float(shap_summary.loc[feature, "mean_abs_shap"])
            if feature in shap_summary.index
            else np.nan
        )

        rows.append(
            {
                "factor": feature,
                "mean_abs_shap": shap_value,
                "n_non_null": int(values.notna().sum()),
                "low_q10": float(values.quantile(0.10)),
                "center_q50": float(values.quantile(0.50)),
                "high_q90": float(values.quantile(0.90)),
            }
        )

    candidates = pd.DataFrame(rows)
    if candidates.empty:
        raise ValueError("No eligible non-null raw-sensor factors found for DOE")

    candidates["preferred_s060"] = candidates["factor"].eq("s060")
    return candidates.sort_values(
        ["preferred_s060", "mean_abs_shap"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)


def factors_from_candidates(
    candidates: pd.DataFrame,
    n_factors: int,
    preferred: list[str] | None = None,
) -> list[dict]:
    """Choose factor definitions from ranked candidates.

    The levels are observed Q10/Q50/Q90 values from the background frame.
    They are not recipe recommendations and are not physical tool limits.
    """
    if n_factors < 2:
        raise ValueError("n_factors must be >= 2")

    preferred = preferred or ["s060"]
    candidate_names = set(candidates["factor"])

    ordered_names = [name for name in preferred if name in candidate_names]
    ordered_names.extend(
        name for name in candidates["factor"].tolist() if name not in ordered_names
    )

    selected_names = ordered_names[:n_factors]
    if len(selected_names) < n_factors:
        raise ValueError(
            f"Requested {n_factors} DOE factors but only "
            f"{len(selected_names)} eligible factors are available"
        )

    selected = candidates.set_index("factor").loc[selected_names].reset_index()

    factors = []
    for _, row in selected.iterrows():
        factors.append(
            {
                "name": str(row["factor"]),
                "low": float(row["low_q10"]),
                "center": float(row["center_q50"]),
                "high": float(row["high_q90"]),
                "units": "SECOM sensor units; physical units unavailable",
                "rationale": (
                    "Selected raw sensor feature. "
                    f"mean_abs_shap={row['mean_abs_shap']:.6g}; "
                    "levels are observed Q10/Q50/Q90."
                ),
            }
        )

    return factors


def parse_args(argv=None):
    """Parse command-line arguments for DOE design generation."""
    parser = argparse.ArgumentParser(
        description="Generate a pyDOE3 surrogate-DOE design from a trained model run."
    )
    parser.add_argument(
        "--run",
        default="latest",
        help="Training run ID or 'latest' (default: latest)",
    )
    parser.add_argument(
        "--label",
        required=True,
        help="Design label used in the timestamped artifact folder",
    )
    parser.add_argument(
        "--n-factors",
        type=int,
        default=None,
        help="Number of raw sensor factors; default comes from config.doe.max_factors",
    )
    parser.add_argument(
        "--preferred",
        action="append",
        default=[],
        help="Preferred raw-sensor factor; repeatable. Default includes s060.",
    )
    parser.add_argument(
        "--fraction",
        default=None,
        help="Optional pyDOE3 fractional-factorial generator, e.g. 'a b c ab'",
    )
    parser.add_argument(
        "--center-points",
        type=int,
        default=None,
        help="Override config.doe.center_points",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=None,
        help="Override config.doe.replicates",
    )
    parser.add_argument(
        "--no-randomize",
        action="store_true",
        help="Keep design rows in generated order instead of seeded randomized order",
    )
    parser.add_argument(
        "--note",
        default="",
        help="Free-text note stored in config.json and artifacts/index.csv",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Generate and register a surrogate DOE design."""
    args = parse_args(argv)
    global logger
    logger = setup_logging(logfile=LOGS / "doe_design.log")

    cfg = load_config()

    n_factors = args.n_factors if args.n_factors is not None else cfg.doe.max_factors
    center_points = args.center_points if args.center_points is not None else cfg.doe.center_points
    replicates = args.replicates if args.replicates is not None else cfg.doe.replicates
    randomize = False if args.no_randomize else cfg.doe.randomize
    seed = cfg.pipeline.seed

    preferred = ["s060", *args.preferred]
    preferred = list(dict.fromkeys(preferred))

    train_dir, _, train_id, _ = resolve_runs(args.run, no_cal=True)
    selected_features = load_contract(train_dir)
    shap_summary = load_shap_summary(train_dir)

    engine = get_engine()
    frame = extract(engine)
    frame, _ = build_features(frame)

    candidates = rank_candidate_factors(selected_features, shap_summary, frame)
    factors = factors_from_candidates(
        candidates,
        n_factors=n_factors,
        preferred=preferred,
    )

    design = build_factorial_design(
        factors,
        fraction=args.fraction,
        run_prefix="doe",
        center_points=center_points,
        replicates=replicates,
        randomize=randomize,
        seed=seed,
    )

    run_config = {
        "script": "doe_design",
        "label": args.label,
        "parent_run": train_id,
        "n_factors": n_factors,
        "preferred": preferred,
        "fraction": args.fraction,
        "center_points": center_points,
        "replicates": replicates,
        "randomize": randomize,
        "seed": seed,
        "factor_level_policy": "observed Q10/Q50/Q90 from extracted SECOM frame",
        "factor_policy": (
            "Raw continuous sensors only; engineered missingness and clique "
            "features excluded because they are not controllable physical knobs."
        ),
        "claim_boundary": (
            "Surrogate DOE design only. Factor levels are observed sensor "
            "quantiles, not recipe settings; causal confirmation requires "
            "controlled physical lots."
        ),
    }

    run_dir, meta = tracking.make_run(
        config=run_config,
        run_name=f"doe-design_{args.label}",
        note=args.note,
        runs_root=ARTIFACTS / "doe",
    )

    metadata = describe_design(
        factors,
        fraction=args.fraction,
        run_prefix="doe",
        center_points=center_points,
        replicates=replicates,
        randomize=randomize,
        seed=seed,
    )
    metadata.update(
        {
            "design_run_id": run_dir.name,
            "parent_run": train_id,
            "note": args.note,
            "factor_level_policy": run_config["factor_level_policy"],
            "factor_policy": run_config["factor_policy"],
            "claim_boundary": run_config["claim_boundary"],
        }
    )

    design_csv, design_json = write_design_artifacts(design, metadata, run_dir)
    candidates.to_csv(run_dir / "factor_candidates.csv", index=False)

    selected_payload = {
        "parent_run": train_id,
        "selected_factors": factors,
        "excluded_feature_policy": (
            "Engineered missingness and clique indicators are excluded from DOE "
            "factors because they are measurement/data-quality signals, not "
            "controllable process settings."
        ),
    }
    (run_dir / "selected_factors.json").write_text(
        json.dumps(selected_payload, indent=2),
        encoding="utf-8",
    )

    tracking.append_index(
        run_dir,
        {
            "type": "doe_design",
            "parent_run": train_id,
            "run_name": f"doe-design_{args.label}",
            "note": args.note,
            "n_design_rows": int(len(design)),
            "n_factors": int(len(factors)),
        },
    )

    latest_pointer = ARTIFACTS / "doe" / "latest_design"
    latest_pointer.parent.mkdir(parents=True, exist_ok=True)
    latest_pointer.write_text(f"{run_dir.relative_to(ARTIFACTS.parent)}\n", encoding="utf-8")
    logger.info("[doe_design] parent=%s", train_id)
    logger.info("[doe_design] selected factors=%s", [f["name"] for f in factors])
    logger.info("[doe_design] wrote design -> %s", design_csv)
    logger.info("[doe_design] wrote metadata -> %s", design_json)
    logger.info("[doe_design] updated latest pointer -> %s", latest_pointer)
    logger.info(
        "[doe_design] next:\n"
        "semcon-doe-run --design %s "
        "--background-start '<start>' --background-end '<end>' "
        "--label %s",
        design_csv,
        args.label,
    )

    return


if __name__ == "__main__":
    main()
