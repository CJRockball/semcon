"""Unit tests for dash_figs: pure Plotly figures, no Dash or filesystem."""

from __future__ import annotations

import pandas as pd
import pytest
from plotly.graph_objects import Figure

from semcon import dash_figs


@pytest.fixture()
def pchart() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x": [49, 99, 149],
            "rate": [0.05, 0.22, 0.08],
            "n": [50, 50, 50],
            "lcl": [0.0, 0.0, 0.0],
            "ucl": [0.15, 0.15, 0.15],
            "p0": [0.06, 0.06, 0.06],
        }
    )


@pytest.fixture()
def protocol_rates() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["f_miss_clq14", "f_miss_clq14", "f_miss_block5", "f_miss_block5"],
            "x": [49, 99, 49, 99],
            "rate": [0.1, 0.4, 0.0, 0.1],
            "n": [50, 50, 50, 50],
            "lcl": [0.0] * 4,
            "ucl": [0.3] * 4,
            "p0": [0.1, 0.1, 0.05, 0.05],
        }
    )


@pytest.fixture()
def row_missing() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["f_row_missing_rate"] * 3,
            "t": [0, 1, 2],
            "value": [0.01, 0.02, 0.50],
            "center": [0.02] * 3,
            "ucl": [0.05] * 3,
        }
    )


@pytest.fixture()
def imr() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["s003"] * 3,
            "t": [0, 1, 2],
            "value": [2.0, 2.1, 3.5],
            "mr": [None, 0.1, 1.4],
            "ewma": [2.0, 2.02, 2.3],
            "ewma_lcl": [1.8] * 3,
            "ewma_ucl": [2.2] * 3,
            "r1": [False, False, True],
            "r2": [False] * 3,
            "r3": [False] * 3,
            "r4": [False] * 3,
        }
    )


@pytest.fixture()
def limits() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "center": [2.0],
            "lcl": [1.4],
            "ucl": [2.6],
            "mr_bar": [0.2],
            "mr_ucl": [0.65],
            "degenerate": [False],
        },
        index=pd.Index(["s003"], name="feature"),
    )


@pytest.fixture()
def screening() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "degenerate": [False, False, True],
            "ooc_p1": [0.01, 0.10, None],
            "ooc_p2": [0.05, 0.40, None],
            "delta": [0.04, 0.30, None],
        },
        index=pd.Index(["s001", "s003", "s002"], name="feature"),
    )


@pytest.fixture()
def queue() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "wafer_id": [1002, 1003, 1001],
            "p_cal": [0.40, 0.20, 0.05],
            "rank": [1, 2, 3],
            "decile": [1, 2, 10],
        }
    )


def test_pchart_returns_figure_with_limits_and_alarm(pchart: pd.DataFrame) -> None:
    fig = dash_figs.make_pchart_fig(pchart, i_hold=100, i_tail=200)
    assert isinstance(fig, Figure)
    assert {trace.name for trace in fig.data} >= {"Observed fail rate", "UCL", "LCL", "Beyond UCL"}
    assert len(fig.layout.shapes) >= 3  # p-bar + Phase-II + tail markers


def test_pchart_rejects_incomplete_contract(pchart: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="pchart missing columns"):
        dash_figs.make_pchart_fig(pchart.drop(columns="ucl"))


def test_protocol_rates_support_overview_and_one_feature(protocol_rates: pd.DataFrame) -> None:
    overview = dash_figs.make_protocol_rates_fig(protocol_rates)
    assert isinstance(overview, Figure)
    assert {trace.name for trace in overview.data} == {"f_miss_block5", "f_miss_clq14"}
    selected = dash_figs.make_protocol_rates_fig(protocol_rates, feature="f_miss_clq14")
    assert {trace.name for trace in selected.data} >= {"f_miss_clq14", "UCL", "LCL", "Beyond UCL"}


def test_protocol_rates_reject_unknown_feature(protocol_rates: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="unknown protocol feature"):
        dash_figs.make_protocol_rates_fig(protocol_rates, feature="f_nope")


def test_row_missing_marks_beyond_ucl(row_missing: pd.DataFrame) -> None:
    fig = dash_figs.make_row_missing_fig(row_missing)
    assert isinstance(fig, Figure)
    assert {trace.name for trace in fig.data} >= {"f_row_missing_rate", "Beyond UCL"}


def test_imr_returns_three_panel_figure(imr: pd.DataFrame, limits: pd.DataFrame) -> None:
    fig = dash_figs.make_imr_fig(imr, limits, "s003", i_hold=1, i_tail=2)
    assert isinstance(fig, Figure)
    assert {trace.name for trace in fig.data} >= {"Value", "MR", "EWMA", "EWMA UCL", "EWMA LCL", "WE alarm"}
    assert len(fig.layout.annotations) >= 3  # subplot titles


def test_imr_rejects_missing_limits(
    imr: pd.DataFrame,
    limits: pd.DataFrame,
) -> None:
    limits_without_s003 = limits.drop(index="s003")

    with pytest.raises(ValueError, match="no Phase-I limits"):
        dash_figs.make_imr_fig(imr, limits_without_s003, "s003")


def test_imr_rejects_missing_series(
    imr: pd.DataFrame,
    limits: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="no IMR series"):
        dash_figs.make_imr_fig(imr, limits, "s001")
        

def test_drift_overview_excludes_degenerate_rows(screening: pd.DataFrame) -> None:
    fig = dash_figs.make_drift_overview_fig(screening)
    assert isinstance(fig, Figure)
    sensor_trace = fig.data[0]
    assert len(sensor_trace.x) == 2
    assert fig.layout.xaxis.type == "log"
    assert fig.layout.yaxis.type == "log"


def test_queue_orders_by_rank_and_limits_display(queue: pd.DataFrame) -> None:
    fig = dash_figs.make_queue_fig(queue, top_k=2)
    assert isinstance(fig, Figure)
    assert list(fig.data[0].y) == ["1003", "1002"]  # rank 2 then 1: rank 1 at top of horizontal plot
    assert "top 2 wafers" in fig.layout.title.text


def test_queue_rejects_missing_contract(queue: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="score queue missing columns"):
        dash_figs.make_queue_fig(queue.drop(columns="p_cal"))


def test_alert_columns_are_dash_agnostic_data() -> None:
    columns = dash_figs.make_alert_columns()
    assert isinstance(columns, list)
    assert [column["field"] for column in columns[:3]] == ["feature", "drift", "delta"]
    assert all(isinstance(column, dict) for column in columns)
