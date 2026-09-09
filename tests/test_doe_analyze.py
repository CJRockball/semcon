from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from semcon.doe_analyze import (
    REQUIRED_COLUMNS,
    build_formula,
    check_curvature,
    effects_table,
    fit_effect_model,
    infer_factors,
    load_predictions,
    main_effects_table,
    residual_diagnostics,
    select_recommendation,
    summarize_design_cells,
    write_analysis_artifacts,
)


def _prediction_frame() -> pd.DataFrame:
    """Small deterministic surrogate DOE with a visible s060 effect."""
    rows = []

    for run_id, run_order, s060, s123 in [
        ("doe_001", 1, 0.0, 0.0),
        ("doe_002", 2, 0.0, 1.0),
        ("doe_003", 3, 1.0, 0.0),
        ("doe_004", 4, 1.0, 1.0),
        ("doe_005", 5, 0.5, 0.5),
    ]:
        for wafer_id in range(1, 6):
            p = 0.15 + 0.30 * s060 + 0.10 * s123 + 0.15 * s060 * s123
            p += wafer_id * 0.001

            rows.append(
                {
                    "wafer_id": wafer_id,
                    "timestamp": pd.Timestamp("2026-01-01") + pd.Timedelta(hours=wafer_id),
                    "run_id": run_id,
                    "run_order": run_order,
                    "s060": s060,
                    "s123": s123,
                    "score_raw": p,
                    "p_cal": p,
                    "y_observed": int(p > 0.35),
                }
            )

    return pd.DataFrame(rows)


def test_load_predictions_rejects_missing_required_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.parquet"
    pd.DataFrame({"run_id": ["doe_001"]}).to_parquet(path, index=False)

    with pytest.raises(ValueError, match="missing columns"):
        load_predictions(path)


def test_load_predictions_reads_valid_frame(tmp_path: Path) -> None:
    frame = _prediction_frame()
    path = tmp_path / "predictions.parquet"
    frame.to_parquet(path, index=False)

    loaded = load_predictions(path)

    assert len(loaded) == len(frame)
    assert REQUIRED_COLUMNS.issubset(loaded.columns)


def test_infer_factors_returns_design_inputs_only() -> None:
    frame = _prediction_frame()
    factors = infer_factors(frame)

    assert factors == ["s060", "s123"]
    assert "p_cal" not in factors
    assert "score_raw" not in factors
    assert "y_observed" not in factors


def test_summarize_design_cells_returns_one_row_per_design_cell() -> None:
    frame = _prediction_frame()
    summary = summarize_design_cells(frame, ["s060", "s123"])

    assert len(summary) == frame["run_id"].nunique()
    assert {"p_cal_mean", "p_cal_std", "n_background"}.issubset(summary.columns)
    assert summary["n_background"].eq(5).all()


def test_build_formula_contains_main_effects_and_interaction() -> None:
    formula = build_formula(["s060", "s123"], "p_cal_mean", include_interactions=True)

    assert formula == "p_cal_mean ~ s060 + s123 + s060:s123"


def test_fit_effect_model_recovers_positive_s060_effect() -> None:
    frame = _prediction_frame()
    summary = summarize_design_cells(frame, ["s060", "s123"])

    model = fit_effect_model(
        summary,
        ["s060", "s123"],
        response="p_cal_mean",
        include_interactions=True,
    )

    assert model.params["s060"] > 0
    assert model.params["s123"] > 0
    assert model.params["s060:s123"] > 0


def test_effects_table_and_diagnostics_have_expected_schema() -> None:
    frame = _prediction_frame()
    summary = summarize_design_cells(frame, ["s060", "s123"])
    model = fit_effect_model(summary, ["s060", "s123"])

    effects = effects_table(model)
    diagnostics = residual_diagnostics(model)

    assert {"term", "estimate", "std_error", "p_value", "abs_estimate"}.issubset(effects.columns)
    assert {"fitted", "residual", "studentized_residual"}.issubset(diagnostics.columns)
    assert len(diagnostics) == len(summary)


def test_main_effects_ranks_s060_above_s123() -> None:
    frame = _prediction_frame()
    summary = summarize_design_cells(frame, ["s060", "s123"])

    effects = main_effects_table(summary, ["s060", "s123"])

    assert effects.iloc[0]["factor"] == "s060"
    assert effects.iloc[0]["effect_high_minus_low"] > 0


def test_check_curvature_detects_center_point() -> None:
    frame = _prediction_frame()
    summary = summarize_design_cells(frame, ["s060", "s123"])

    curvature = check_curvature(summary, ["s060", "s123"])

    assert curvature["available"] is True
    assert curvature["center_mean"] is not None
    assert curvature["corner_mean"] is not None


def test_select_recommendation_minimizes_calibrated_risk() -> None:
    frame = _prediction_frame()
    summary = summarize_design_cells(frame, ["s060", "s123"])

    recommendation = select_recommendation(
        summary,
        ["s060", "s123"],
        response="p_cal_mean",
        maximize=False,
    )

    assert recommendation["objective"] == "minimize"
    assert recommendation["recommended_run_id"] == "doe_001"
    assert recommendation["factor_settings"]["s060"] == 0.0
    assert recommendation["factor_settings"]["s123"] == 0.0
    assert "Surrogate recommendation only" in recommendation["claim_boundary"]


def test_write_analysis_artifacts(tmp_path: Path) -> None:
    frame = _prediction_frame()
    summary = summarize_design_cells(frame, ["s060", "s123"])
    model = fit_effect_model(summary, ["s060", "s123"])
    effects = effects_table(model)
    main_effects = main_effects_table(summary, ["s060", "s123"])
    diagnostics = residual_diagnostics(model)
    curvature = check_curvature(summary, ["s060", "s123"])
    recommendation = select_recommendation(summary, ["s060", "s123"])

    write_analysis_artifacts(
        tmp_path,
        summary,
        effects,
        main_effects,
        diagnostics,
        curvature,
        recommendation,
        model,
    )

    expected = {
        "design_cell_summary.csv",
        "effects_table.csv",
        "main_effects.csv",
        "residual_diagnostics.csv",
        "curvature.json",
        "recommendation.json",
        "model_summary.txt",
    }

    assert expected.issubset({path.name for path in tmp_path.iterdir()})
