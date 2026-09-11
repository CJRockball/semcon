"""Tests for dash_data: loaders run against a synthetic artifacts tree."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from semcon import dash_data

SCREENING_COLS = [
    "feature", "degenerate",
    "ooc_p1", "ewma_p1", "miss_p1",
    "ooc_p2", "ewma_p2", "miss_p2",
    "ooc_tail", "ewma_tail", "miss_tail",
    "delta", "ewma_delta",
]

NEWEST = "20260911_135749_spc"
OLDER = "20260911_094237_spc"


def _run(artifacts: Path, run_id: str = NEWEST) -> dict:
    return {"run_id": run_id, "kind": "spc", "path": artifacts / "runs" / run_id, "meta": {}}


@pytest.fixture()
def artifacts(tmp_path: Path) -> Path:
    root = tmp_path / "artifacts"
    runs = root / "runs"
    runs.mkdir(parents=True)

    spc_runs = [(OLDER, 11), (NEWEST, 11)]

    for run_id, n_drift in spc_runs:
        assert n_drift >= 0
        d = runs / run_id
        (d / "figures").mkdir(parents=True)
        (d / "config.json").write_text(json.dumps({"k": 3.0, "limits": "robust"}))
        (d / "figures" / "yield_pchart.png").touch()

        screening_name = "spc_screening.csv"
        if run_id == NEWEST:
            screening_name = "screening_snapshot.csv"
            (d / "files.json").write_text(json.dumps({"screening": screening_name}))
        pd.DataFrame(
            [
                ["s001", False, 0.01, 0.02, 0.0, 0.05, 0.03, 0.0, 0.0, 0.0, 0.0, 0.04, 0.01],
                ["s002", True, None, None, None, None, None, None, None, None, None, None, None],
                ["s003", False, 0.10, 0.20, 0.0, 0.40, 0.50, 0.0, 0.0, 0.0, 0.0, 0.30, 0.30],
            ],
            columns=SCREENING_COLS,
        ).to_csv(d / screening_name, index=False)

        pd.DataFrame(
            {
                "feature": ["s001", "s002", "s003"],
                "degenerate": [False, True, False],
                "center": [1.0, None, 2.0],
                "sigma": [0.1, None, 0.2],
                "lcl": [0.7, None, 1.4],
                "ucl": [1.3, None, 2.6],
                "mr_bar": [0.05, None, 0.1],
                "mr_ucl": [0.16, None, 0.33],
            }
        ).to_csv(d / "spc_limits.csv", index=False)
        pd.DataFrame(
            {
                "x": [49, 99],
                "rate": [0.05, 0.10],
                "n": [50, 50],
                "lcl": [0.0, 0.0],
                "ucl": [0.15, 0.15],
                "p0": [0.06, 0.06],
            }
        ).to_csv(d / "pchart_data.csv", index=False)
        pd.DataFrame(
            {
                "x": [49, 99, 49, 99],
                "rate": [0.1, 0.2, 0.0, 0.1],
                "n": [50, 50, 50, 50],
                "feature": ["f_miss_clq14", "f_miss_clq14", "f_miss_block5", "f_miss_block5"],
                "lcl": [0.0] * 4,
                "ucl": [0.3] * 4,
                "p0": [0.1, 0.1, 0.05, 0.05],
            }
        ).to_csv(d / "protocol_rates.csv", index=False)
        pd.DataFrame(
            {
                "feature": ["f_row_missing_rate"] * 3,
                "t": [0, 1, 2],
                "value": [0.01, 0.02, 0.50],
                "center": [0.02] * 3,
                "ucl": [0.05] * 3,
            }
        ).to_csv(d / "protocol_row_missing.csv", index=False)
        pd.DataFrame(
            {
                "feature": ["s001"] * 3 + ["s003"] * 3,
                "t": [0, 1, 2] * 2,
                "value": [1.0, 1.1, 0.9, 2.0, 2.1, 3.5],
                "mr": [None, 0.1, 0.2, None, 0.1, 1.4],
                "ewma": [1.0, 1.02, 1.0, 2.0, 2.02, 2.3],
                "ewma_lcl": [0.9] * 6,
                "ewma_ucl": [1.1] * 3 + [2.2] * 3,
                "r1": [False] * 5 + [True],
                "r2": [False] * 6,
                "r3": [False] * 6,
                "r4": [False] * 6,
            }
        ).to_csv(d / "imr_series.csv", index=False)

    pd.DataFrame(
        {
            "run_id": [run_id for run_id, _ in spc_runs],
            "type": ["spc"] * len(spc_runs),
            "n_screened": [590] * len(spc_runs),
            "n_drift": [n_drift for _, n_drift in spc_runs],
        }
    ).to_csv(root / "index_monitor.csv", index=False)

    scores = root / "scores"
    for batch in ["20260910_163711_score__batch_a_clean", "20260910_163718_score__holdout_replay"]:
        b = scores / batch
        b.mkdir(parents=True)
        pd.DataFrame(
            {
                "wafer_id": [1003, 1001, 1002],
                "timestamp": ["2026-09-10T16:00:02", "2026-09-10T16:00:00", "2026-09-10T16:00:01"],
                "score_raw": [0.50, 0.70, 0.20],
                "p_cal": [0.20, 0.05, 0.40],
                "rank": [2, 3, 1],
                "decile": [2, 10, 1],
            }
        ).to_parquet(b / "scores.parquet", index=False)
        (b / "scorecard.json").write_text(json.dumps({"pr_auc": 0.098}))
        (b / "reconciliation.json").write_text(json.dumps({"max_abs_diff": 0.0}))

    return root


def test_latest_monitor_run_is_last_registry_row(artifacts: Path) -> None:
    run = dash_data.latest_monitor_run(artifacts / "index_monitor.csv")
    assert run["run_id"] == NEWEST
    assert run["meta"]["n_drift"] == 11
    assert run["path"] == artifacts / "runs" / NEWEST


def test_missing_registry_raises_clear_error(artifacts: Path) -> None:
    with pytest.raises(FileNotFoundError, match="registry not found"):
        dash_data.latest_monitor_run(artifacts / "nope.csv")


def test_pointer_manifest_wins_over_default_name(artifacts: Path) -> None:
    run = _run(artifacts, NEWEST)
    assert not (run["path"] / "spc_screening.csv").exists()
    df = dash_data.load_screening(run)
    assert df.loc["s003", "delta"] == pytest.approx(0.30)


def test_default_name_fallback_without_manifest(artifacts: Path) -> None:
    run = _run(artifacts, OLDER)
    assert not (run["path"] / "files.json").exists()
    df = dash_data.load_screening(run)
    assert len(df) == 3


def test_unknown_artifact_role_raises_with_known_list(artifacts: Path) -> None:
    with pytest.raises(ValueError, match="unknown artifact role"):
        dash_data.resolve_artifact(_run(artifacts), "wafer_map")


def test_load_screening_dtypes_and_degenerate_nan(artifacts: Path) -> None:
    df = dash_data.load_screening(_run(artifacts, OLDER))
    assert df.loc["s002", "degenerate"]
    assert pd.isna(df.loc["s002", "delta"])
    assert df["ooc_p1"].dtype.kind == "f"


def test_load_screening_missing_file_raises(artifacts: Path) -> None:
    orphan = _run(artifacts, "20990101_000000_spc")
    with pytest.raises(FileNotFoundError, match="role 'screening'"):
        dash_data.load_screening(orphan)


def test_drift_table_flags_and_orders(artifacts: Path) -> None:
    df = dash_data.load_drift_table(_run(artifacts), delta_min=0.05)
    assert list(df.index) == ["s003", "s001"]
    assert df["drift"].tolist() == [True, False]
    assert "s002" not in df.index


def test_load_pchart_has_windowed_limits(artifacts: Path) -> None:
    df = dash_data.load_pchart(_run(artifacts))
    assert list(df.columns) == ["x", "rate", "n", "lcl", "ucl", "p0"]
    assert df["rate"].tolist() == [0.05, 0.10]


def test_load_protocol_rates_long_format(artifacts: Path) -> None:
    df = dash_data.load_protocol_rates(_run(artifacts))
    assert set(df["feature"]) == {"f_miss_clq14", "f_miss_block5"}
    assert len(df) == 4


def test_load_protocol_row_missing_series(artifacts: Path) -> None:
    df = dash_data.load_protocol_row_missing(_run(artifacts))
    assert df["value"].max() == 0.50
    assert (df["value"] > df["ucl"]).sum() == 1


def test_load_imr_series_filter_and_unknown_feature(artifacts: Path) -> None:
    run = _run(artifacts)
    full = dash_data.load_imr_series(run)
    assert set(full["feature"]) == {"s001", "s003"}
    one = dash_data.load_imr_series(run, "s003")
    assert one["t"].tolist() == [0, 1, 2]
    assert one["r1"].tolist() == [False, False, True]
    with pytest.raises(ValueError, match="showcase features"):
        dash_data.load_imr_series(run, "s999")


def test_score_queue_respects_persisted_rank(artifacts: Path) -> None:
    batch = dash_data.latest_score_batch(artifacts / "scores")
    assert dash_data.run_label(batch) == "holdout_replay"
    q = dash_data.load_score_queue(batch, top_k=1)
    assert q["wafer_id"].tolist() == [1002, 1003, 1001]
    assert q["rank"].tolist() == [1, 2, 3]
    assert q["in_top_k"].tolist() == [True, False, False]
    assert q["decile"].tolist() == [1, 2, 10]


def test_score_queue_score_col_contract(artifacts: Path) -> None:
    batch = dash_data.latest_score_batch(artifacts / "scores")
    q = dash_data.load_score_queue(batch)
    assert "p_cal" in q.columns
    with pytest.raises(ValueError, match="score_col 'nope'"):
        dash_data.load_score_queue(batch, score_col="nope")


def test_list_figures_and_batch_summary(artifacts: Path) -> None:
    figs = dash_data.list_figures(_run(artifacts))
    assert "yield_pchart" in figs
    batch = dash_data.latest_score_batch(artifacts / "scores")
    summary = dash_data.load_batch_summary(batch)
    assert summary["scorecard"]["pr_auc"] == pytest.approx(0.098)
    assert summary["reconciliation"]["max_abs_diff"] == 0.0
