"""Analyze surrogate DOE runs.

Primary analysis:
- summarize the deterministic calibrated surrogate response, p_cal;
- estimate main effects/interactions on coded factor values;
- produce diagnostics, plots, and a bounded surrogate recommendation.

Secondary analysis:
- group simulated y_observed outcomes per design row;
- fit a binomial GLM to failures/passes;
- report that analysis explicitly as simulated Bernoulli sampling conditioned
  on the calibrated surrogate, not physical process replication.

This analyzes a calibrated-model surrogate, not physical fab experiments.
Real controlled lots are required for causal confirmation.
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
import statsmodels.api as sm
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

DESIGN_METADATA_COLUMNS = {
    "design_row",
    "is_center",
    "is_replicate",
    "replicate",
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
    """Infer actual DOE factor columns from the run output."""
    excluded = REQUIRED_COLUMNS | {
        "wafer_id",
        "timestamp",
        "rank",
        "decile",
        "split",
        "is_fail",
        *DESIGN_METADATA_COLUMNS,
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


def infer_coded_factors(df: pd.DataFrame, factors: list[str]) -> list[str]:
    """Infer coded columns and validate one-to-one alignment with actual factors."""
    coded = [f"{factor}_coded" for factor in factors]
    missing = [col for col in coded if col not in df.columns]
    if missing:
        raise ValueError(
            "Missing coded DOE factor columns in design_predictions.parquet: "
            f"{missing}. Regenerate the run with the updated doe_run.py."
        )
    return coded


def summarize_design_cells(
    df: pd.DataFrame,
    factors: list[str],
    coded_factors: list[str] | None = None,
    response: str = "p_cal",
) -> pd.DataFrame:
    """Aggregate wafer-level predictions into one summary per DOE design row."""
    if response not in df.columns:
        raise ValueError(f"Response column {response!r} not found")

    group_cols = ["run_id", "run_order"]
    for col in ("design_row", "is_center", "is_replicate", "replicate"):
        if col in df.columns:
            group_cols.append(col)
    group_cols += [*factors, *(coded_factors or [])]
    group_cols = list(dict.fromkeys(group_cols))

    group_cols = list(dict.fromkeys(group_cols))

    grouped = (
        df.groupby(group_cols, dropna=False)[response]
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
    """Return OLS model coefficients, uncertainty, and p-values as a tidy table."""
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
    out["analysis_scope"] = "deterministic calibrated surrogate response"
    return out.sort_values("abs_estimate", ascending=False).reset_index(drop=True)


def main_effects_table(
    cell_summary: pd.DataFrame,
    factors: list[str],
    response: str = "p_cal_mean",
) -> pd.DataFrame:
    """Calculate simple high-minus-low contrasts for each actual factor."""
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

    This is a deterministic surrogate curvature check. Duplicate center rows
    do not create physical pure error when the model input is unchanged.
    """
    if "is_center" in cell_summary.columns:
        center = cell_summary.loc[cell_summary["is_center"], response]
        corners = cell_summary.loc[~cell_summary["is_center"], response]
    else:
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
            "corner_mean": float(corners.mean()) if not corners.empty else None,
            "difference": None,
            "interpretation": "No center-point rows found in the design.",
        }

    center_mean = float(center.mean())
    corner_mean = float(corners.mean())

    return {
        "available": True,
        "center_mean": center_mean,
        "corner_mean": corner_mean,
        "difference": center_mean - corner_mean,
        "n_center_rows": int(center.shape[0]),
        "interpretation": (
            "Deterministic surrogate curvature check; duplicate center rows do "
            "not estimate physical process replication error."
        ),
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


def summarize_bernoulli_cells(
    df: pd.DataFrame,
    factors: list[str],
    coded_factors: list[str],
) -> pd.DataFrame:
    """Aggregate simulated Bernoulli outcomes per DOE design row.

    For each design row, all background rows are scored and one y_observed is
    sampled from p_cal for each background context. This function aggregates
    those simulated binary outcomes into failures/passes/trials.
    """
    if "y_observed" not in df.columns:
        raise ValueError("y_observed is required for grouped Bernoulli analysis")

    group_cols = [
        "run_id",
        "run_order",
        "design_row",
        "is_center",
        "is_replicate",
        "replicate",
        *factors,
        *coded_factors,
    ]
    group_cols = [col for col in dict.fromkeys(group_cols) if col in df.columns]

    grouped = (
        df.groupby(group_cols, dropna=False)
        .agg(
            simulated_failures=("y_observed", "sum"),
            n_trials=("y_observed", "size"),
            expected_failures_from_p_cal=("p_cal", "sum"),
            expected_failure_rate_from_p_cal=("p_cal", "mean"),
        )
        .reset_index()
    )

    grouped["simulated_failures"] = grouped["simulated_failures"].astype(int)
    grouped["n_trials"] = grouped["n_trials"].astype(int)
    grouped["simulated_passes"] = grouped["n_trials"] - grouped["simulated_failures"]
    grouped["simulated_failure_rate"] = grouped["simulated_failures"] / grouped["n_trials"]

    return grouped.sort_values("run_order").reset_index(drop=True)


def _build_glm_design_matrix(
    bernoulli_cells: pd.DataFrame,
    coded_factors: list[str],
    include_interactions: bool,
) -> pd.DataFrame:
    """Build coded-factor design matrix for grouped binomial GLM."""
    X = bernoulli_cells[coded_factors].copy()

    if include_interactions:
        for i, factor_a in enumerate(coded_factors):
            for factor_b in coded_factors[i + 1 :]:
                X[f"{factor_a}:{factor_b}"] = bernoulli_cells[factor_a] * bernoulli_cells[factor_b]

    return sm.add_constant(X, has_constant="add")


def fit_bernoulli_glm(
    bernoulli_cells: pd.DataFrame,
    coded_factors: list[str],
    *,
    include_interactions: bool = True,
):
    """Fit a grouped binomial GLM to simulated Bernoulli outcomes.

    The response is [simulated_failures, simulated_passes]. This is a
    secondary simulation analysis conditioned on calibrated surrogate
    probabilities; it does not estimate physical process replication error.
    """
    if bernoulli_cells.empty:
        raise ValueError("bernoulli_cells is empty")
    if len(coded_factors) < 2:
        raise ValueError("Need at least two coded factors for the GLM")

    required = {"simulated_failures", "simulated_passes"}
    missing = required - set(bernoulli_cells.columns)
    if missing:
        raise ValueError(f"bernoulli_cells missing columns: {sorted(missing)}")

    X = _build_glm_design_matrix(
        bernoulli_cells,
        coded_factors,
        include_interactions=include_interactions,
    )
    y = bernoulli_cells[["simulated_failures", "simulated_passes"]].to_numpy(dtype=float)

    return sm.GLM(y, X, family=sm.families.Binomial()).fit()


def bernoulli_effects_table(model) -> pd.DataFrame:
    """Return grouped-binomial GLM coefficients as a tidy table."""
    out = pd.DataFrame(
        {
            "term": model.params.index,
            "estimate_log_odds": model.params.values,
            "std_error": model.bse.values,
            "z_value": model.tvalues.values,
            "p_value": model.pvalues.values,
        }
    )

    out["odds_ratio"] = np.exp(out["estimate_log_odds"])
    out["abs_estimate_log_odds"] = out["estimate_log_odds"].abs()
    out["is_interaction"] = out["term"].str.contains(":", regex=False)
    out["analysis_scope"] = "simulated Bernoulli response conditioned on calibrated surrogate"

    return out.sort_values(
        "abs_estimate_log_odds",
        ascending=False,
    ).reset_index(drop=True)


def bernoulli_scope() -> dict:
    """Describe what the simulated binary-response GLM means."""
    return {
        "analysis_type": "grouped_binomial_glm",
        "response_source": "y_observed sampled from calibrated surrogate p_cal",
        "noise_model": "Bernoulli",
        "causal_claim": False,
        "interpretation": (
            "Coefficients describe how the calibrated surrogate's predicted "
            "failure probability varies across coded DOE settings under "
            "simulated Bernoulli sampling. They are not physical-process "
            "effects and do not establish that changing fab conditions will "
            "change yield."
        ),
        "physical_confirmation": (
            "Requires randomized, blocked confirmation lots at mapped controllable recipe settings."
        ),
    }


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
    bernoulli_cells: pd.DataFrame | None = None,
    bernoulli_effects: pd.DataFrame | None = None,
    bernoulli_model=None,
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

    if bernoulli_cells is not None:
        bernoulli_cells.to_csv(output_dir / "bernoulli_cell_summary.csv", index=False)

    if bernoulli_effects is not None:
        bernoulli_effects.to_csv(output_dir / "bernoulli_effects_table.csv", index=False)

    if bernoulli_model is not None:
        (output_dir / "bernoulli_model_summary.txt").write_text(
            bernoulli_model.summary().as_text(),
            encoding="utf-8",
        )

        with open(output_dir / "bernoulli_scope.json", "w", encoding="utf-8") as f:
            json.dump(bernoulli_scope(), f, indent=2)


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
        choices=["p_cal", "score_raw"],
        help="Primary deterministic surrogate response to analyze",
    )
    parser.add_argument(
        "--no-interactions",
        action="store_true",
        help="Fit main effects only in both surrogate and Bernoulli models",
    )
    parser.add_argument(
        "--no-bernoulli",
        action="store_true",
        help="Skip grouped simulated Bernoulli GLM analysis",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(logfile=LOGS / "doe_analyze.log")

    predictions_path = Path(args.predictions)
    output_dir = Path(args.output_dir)

    df = load_predictions(predictions_path)
    factors = infer_factors(df)
    coded_factors = infer_coded_factors(df, factors)

    cell_summary = summarize_design_cells(
        df,
        factors,
        coded_factors=coded_factors,
        response=args.response,
    )

    response_col = f"{args.response}_mean"
    model = fit_effect_model(
        cell_summary,
        coded_factors,
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

    bernoulli_cells = None
    bernoulli_model = None
    bernoulli_effects = None

    if not args.no_bernoulli:
        bernoulli_cells = summarize_bernoulli_cells(
            df,
            factors,
            coded_factors,
        )
        bernoulli_model = fit_bernoulli_glm(
            bernoulli_cells,
            coded_factors,
            include_interactions=not args.no_interactions,
        )
        bernoulli_effects = bernoulli_effects_table(bernoulli_model)

    write_analysis_artifacts(
        output_dir,
        cell_summary,
        effects,
        main_effects,
        diagnostics,
        curvature,
        recommendation,
        model,
        bernoulli_cells=bernoulli_cells,
        bernoulli_effects=bernoulli_effects,
        bernoulli_model=bernoulli_model,
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
    logger.info("[doe_analyze] coded factors=%s", coded_factors)
    logger.info("[doe_analyze] recommendation=%s", recommendation)
    if bernoulli_effects is not None:
        logger.info(
            "[doe_analyze] grouped Bernoulli GLM is simulation-only; see bernoulli_scope.json"
        )
    logger.info("[doe_analyze] done -> %s", output_dir)


if __name__ == "__main__":
    main()
