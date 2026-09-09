"""Analyze surrogate DOE runs.

This module consumes design_predictions.parquet from doe_run.py and produces:
- design-cell summaries,
- main-effect and interaction estimates,
- binomial-GLM / OLS model outputs,
- residual diagnostics,
- effect and interaction figures,
- a bounded surrogate recommendation.

Important:
This analyzes a calibrated-model surrogate, not physical fab experiments.
The results express what the current model believes within observed support.
They are hypothesis-generation evidence only; real controlled lots are
required for causal confirmation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import probplot

from semcon.paths import LOGS
from semcon.utils import setup_logging

logger = logging.getLogger("semcon")

REQUIRED_COLUMNS = {
    "run_id",
    "run_order",
    "score_raw",
    "p_cal",
    "y_observed",
}


def load_predictions(path: Path) -> pd.DataFrame:
    """Load and validate DOE surrogate predictions."""
    if not path.exists():
        raise FileNotFoundError(f"DOE predictions not found: {path}")

    df = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"DOE prediction artifact missing columns: {sorted(missing)}")

    if df.empty:
        raise ValueError("DOE prediction artifact is empty")

    if not df["run_id"].notna().all():
        raise ValueError("DOE prediction artifact contains null run_id values")

    return df


def infer_factors(df: pd.DataFrame) -> list[str]:
    """Infer DOE factor columns from the run output."""
    excluded = REQUIRED_COLUMNS | {
        "wafer_id",
        "timestamp",
        "rank",
        "decile",
        "design_row",
        "is_center",
        "is_replicate",
        "replicate",
    }

    factors = [
        col
        for col in df.columns
        if col not in excluded
        and not col.endswith("_coded")
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(factors) < 2:
        raise ValueError(f"Need at least two numeric DOE factor columns; found {factors}")

    return factors


def summarize_design_cells(
    df: pd.DataFrame,
    factors: list[str],
    response: str = "p_cal",
) -> pd.DataFrame:
    """Aggregate wafer-level predictions into one summary per DOE design cell."""
    if response not in df.columns:
        raise ValueError(f"Response column {response!r} not found")

    grouped = (
        df.groupby(["run_id", "run_order", *factors], dropna=False)[response]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": f"{response}_mean",
                "std": f"{response}_std",
                "count": "n_background",
            }
        )
    )

    grouped[f"{response}_std"] = grouped[f"{response}_std"].fillna(0.0)
    return grouped.sort_values("run_order").reset_index(drop=True)


def build_formula(factors: list[str], response: str, include_interactions: bool = True) -> str:
    """Build an OLS formula with main effects and pairwise interactions."""
    main = " + ".join(factors)

    if not include_interactions:
        return f"{response} ~ {main}"

    interactions = [f"{a}:{b}" for i, a in enumerate(factors) for b in factors[i + 1 :]]
    terms = " + ".join([main, *interactions])

    return f"{response} ~ {terms}"


def fit_effect_model(
    cell_summary: pd.DataFrame,
    factors: list[str],
    response: str = "p_cal_mean",
    include_interactions: bool = True,
):
    """Fit OLS model to design-cell means.

    The response is a calibrated-model surrogate mean, not a physical response.
    OLS coefficients therefore quantify controlled contrasts on the surrogate.
    """
    formula = build_formula(factors, response, include_interactions)
    return smf.ols(formula=formula, data=cell_summary).fit()


def effects_table(model) -> pd.DataFrame:
    """Return model coefficients, uncertainty, and p-values as a tidy table."""
    out = pd.DataFrame(
        {
            "term": model.params.index,
            "estimate": model.params.values,
            "std_error": model.bse.values,
            "t_value": model.tvalues.values,
            "p_value": model.pvalues.values,
        }
    )

    out["abs_estimate"] = out["estimate"].abs()
    out["is_interaction"] = out["term"].str.contains(":", regex=False)
    return out.sort_values("abs_estimate", ascending=False).reset_index(drop=True)


def main_effects_table(
    cell_summary: pd.DataFrame,
    factors: list[str],
    response: str = "p_cal_mean",
) -> pd.DataFrame:
    """Calculate simple high-minus-low contrasts for each factor."""
    rows = []

    for factor in factors:
        levels = sorted(cell_summary[factor].dropna().unique())
        if len(levels) < 2:
            continue

        low = levels[0]
        high = levels[-1]

        low_mean = cell_summary.loc[cell_summary[factor] == low, response].mean()
        high_mean = cell_summary.loc[cell_summary[factor] == high, response].mean()

        rows.append(
            {
                "factor": factor,
                "low": low,
                "high": high,
                "response_low": low_mean,
                "response_high": high_mean,
                "effect_high_minus_low": high_mean - low_mean,
                "abs_effect": abs(high_mean - low_mean),
            }
        )

    return pd.DataFrame(rows).sort_values("abs_effect", ascending=False).reset_index(drop=True)


def check_curvature(
    cell_summary: pd.DataFrame,
    factors: list[str],
    response: str = "p_cal_mean",
) -> dict:
    """Compare center-point response to factorial-corner response.

    Center points are inferred when all factor values equal their midpoint.
    If there are no center points, return an explicit unavailable result.
    """
    midpoints = {
        factor: (cell_summary[factor].min() + cell_summary[factor].max()) / 2.0
        for factor in factors
    }

    is_center = np.ones(len(cell_summary), dtype=bool)
    for factor, midpoint in midpoints.items():
        is_center &= np.isclose(cell_summary[factor].to_numpy(), midpoint)

    center = cell_summary.loc[is_center, response]
    corners = cell_summary.loc[~is_center, response]

    if center.empty:
        return {
            "available": False,
            "center_mean": None,
            "corner_mean": float(corners.mean()),
            "difference": None,
        }

    center_mean = float(center.mean())
    corner_mean = float(corners.mean())

    return {
        "available": True,
        "center_mean": center_mean,
        "corner_mean": corner_mean,
        "difference": center_mean - corner_mean,
    }


def residual_diagnostics(model) -> pd.DataFrame:
    """Return fitted values and residuals for diagnostic plots."""
    return pd.DataFrame(
        {
            "fitted": model.fittedvalues,
            "residual": model.resid,
            "studentized_residual": model.get_influence().resid_studentized_internal,
        }
    )


def save_effects_pareto(effects: pd.DataFrame, output_path: Path) -> None:
    """Save an absolute-effect Pareto chart."""
    plot_df = effects.loc[effects["term"] != "Intercept"].sort_values(
        "abs_estimate", ascending=True
    )

    fig, ax = plt.subplots(figsize=(7, max(3, 0.35 * len(plot_df))))
    ax.barh(plot_df["term"], plot_df["abs_estimate"], color="tab:blue")
    ax.set(
        title="Surrogate DOE effect magnitudes",
        xlabel="Absolute coefficient estimate",
        ylabel="Model term",
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_residual_diagnostics(diagnostics: pd.DataFrame, output_path: Path) -> None:
    """Save residual-versus-fitted and normal-probability diagnostic panels."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(diagnostics["fitted"], diagnostics["residual"], alpha=0.75)
    axes[0].axhline(0.0, color="black", linewidth=1)
    axes[0].set(
        title="Residuals vs fitted",
        xlabel="Fitted surrogate response",
        ylabel="Residual",
    )

    probplot(diagnostics["residual"], dist="norm", plot=axes[1])
    axes[1].set_title("Normal probability plot")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_interaction_plot(
    cell_summary: pd.DataFrame,
    factor_a: str,
    factor_b: str,
    output_path: Path,
    response: str = "p_cal_mean",
) -> None:
    """Save one two-factor interaction plot."""
    grouped = (
        cell_summary.groupby([factor_a, factor_b], dropna=False)[response].mean().reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 4))

    for level, sub in grouped.groupby(factor_b, sort=True):
        sub = sub.sort_values(factor_a)
        ax.plot(
            sub[factor_a],
            sub[response],
            marker="o",
            linewidth=1.5,
            label=f"{factor_b}={level:g}",
        )

    ax.set(
        title=f"Interaction: {factor_a} × {factor_b}",
        xlabel=factor_a,
        ylabel="Mean calibrated surrogate risk",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def select_recommendation(
    cell_summary: pd.DataFrame,
    factors: list[str],
    *,
    response: str = "p_cal_mean",
    maximize: bool = False,
) -> dict:
    """Select best observed supported design cell.

    For this project, lower calibrated failure risk is preferable, so
    maximize=False is the expected setting.
    """
    if maximize:
        best = cell_summary.loc[cell_summary[response].idxmax()]
    else:
        best = cell_summary.loc[cell_summary[response].idxmin()]

    settings = {factor: float(best[factor]) for factor in factors}

    return {
        "response": response,
        "objective": "maximize" if maximize else "minimize",
        "recommended_run_id": str(best["run_id"]),
        "recommended_run_order": int(best["run_order"]),
        "predicted_response": float(best[response]),
        "factor_settings": settings,
        "claim_boundary": (
            "Surrogate recommendation only. It describes the calibrated model "
            "within the explored design region and requires confirmation on "
            "controlled physical lots before any process change."
        ),
    }


def write_analysis_artifacts(
    output_dir: Path,
    cell_summary: pd.DataFrame,
    effects: pd.DataFrame,
    main_effects: pd.DataFrame,
    diagnostics: pd.DataFrame,
    curvature: dict,
    recommendation: dict,
    model,
) -> None:
    """Write tabular and structured DOE analysis artifacts."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cell_summary.to_csv(output_dir / "design_cell_summary.csv", index=False)
    effects.to_csv(output_dir / "effects_table.csv", index=False)
    main_effects.to_csv(output_dir / "main_effects.csv", index=False)
    diagnostics.to_csv(output_dir / "residual_diagnostics.csv", index=False)

    with open(output_dir / "curvature.json", "w", encoding="utf-8") as f:
        json.dump(curvature, f, indent=2)

    with open(output_dir / "recommendation.json", "w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2)

    (output_dir / "model_summary.txt").write_text(model.summary().as_text(), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Analyze a surrogate DOE run")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to design_predictions.parquet from doe_run.py",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for DOE analysis artifacts",
    )
    parser.add_argument(
        "--response",
        default="p_cal",
        choices=["p_cal", "score_raw", "y_observed"],
        help="Surrogate response to summarize and analyze",
    )
    parser.add_argument(
        "--no-interactions",
        action="store_true",
        help="Fit main effects only",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(logfile=LOGS / "doe_analyze.log")

    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)

    df = load_predictions(predictions_path)
    factors = infer_factors(df)
    cell_summary = summarize_design_cells(df, factors, response=args.response)

    response_col = f"{args.response}_mean"
    model = fit_effect_model(
        cell_summary,
        factors,
        response=response_col,
        include_interactions=not args.no_interactions,
    )

    effects = effects_table(model)
    main_effects = main_effects_table(cell_summary, factors, response=response_col)
    curvature = check_curvature(cell_summary, factors, response=response_col)
    diagnostics = residual_diagnostics(model)
    recommendation = select_recommendation(
        cell_summary,
        factors,
        response=response_col,
        maximize=False,
    )

    write_analysis_artifacts(
        output_dir,
        cell_summary,
        effects,
        main_effects,
        diagnostics,
        curvature,
        recommendation,
        model,
    )

    save_effects_pareto(effects, output_dir / "effects_pareto.png")
    save_residual_diagnostics(diagnostics, output_dir / "residual_diagnostics.png")

    for i, factor_a in enumerate(factors):
        for factor_b in factors[i + 1 :]:
            filename = f"interaction_{factor_a}_{factor_b}.png"
            save_interaction_plot(
                cell_summary,
                factor_a,
                factor_b,
                output_dir / filename,
                response=response_col,
            )

    logger.info("[doe_analyze] factors=%s", factors)
    logger.info("[doe_analyze] recommendation=%s", recommendation)
    logger.info("[doe_analyze] done -> %s", output_dir)

    return recommendation


if __name__ == "__main__":
    main()
