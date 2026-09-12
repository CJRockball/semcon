"""Pure Plotly figures for the semcon monitoring dashboard.

This is the presentation layer between dash_data.py and dash_app.py:
- accepts already-loaded pandas DataFrames only
- returns plotly.graph_objects.Figure objects only
- never reads files, resolves runs, recomputes SPC, scores wafers, or imports Dash

The upstream SPC job owns all statistics and limits. This module renders
those persisted values faithfully, including Phase-I/II boundaries supplied
by the caller from the run config (i_hold / i_tail).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLORS = {
    "process": "#1f77b4",
    "risk": "#d62728",
    "center": "#2ca02c",
    "limit": "#d62728",
    "mr": "#9467bd",
    "ewma": "#8c564b",
    "alert": "#ff7f0e",
    "neutral": "#6b7280",
    "phase_1": "rgba(59, 130, 246, 0.06)",
    "phase_2": "rgba(249, 115, 22, 0.07)",
    "tail": "rgba(34, 197, 94, 0.07)",
}

LAYOUT = {
    "template": "plotly_white",
    "font": {"family": "Arial, sans-serif", "size": 12},
    "margin": {"l": 60, "r": 25, "t": 55, "b": 50},
    "hovermode": "x unified",
}


def _layout(fig: go.Figure, title: str, height: int = 430) -> go.Figure:
    fig.update_layout(title=title, height=height, **LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    return fig


def _phase_marks(fig: go.Figure, i_hold: int | None, i_tail: int | None) -> None:
    """Mark Phase I, Phase II, and tail boundaries without owning split logic."""
    if i_hold is not None:
        fig.add_vline(
            x=i_hold,
            line_dash="dot",
            line_color="#374151",
            line_width=1,
            annotation_text="Phase II",
            annotation_position="top right",
        )
    if i_tail is not None:
        fig.add_vline(
            x=i_tail,
            line_dash="dot",
            line_color="#374151",
            line_width=1,
            annotation_text="Tail",
            annotation_position="top right",
        )


def make_pchart_fig(
    pchart: pd.DataFrame,
    i_hold: int | None = None,
    i_tail: int | None = None,
) -> go.Figure:
    """Windowed yield p-chart from pchart_data.csv.

    Required columns: x, rate, n, lcl, ucl, p0. Values and limits are
    persisted by spc.py; this function does not calculate binomial limits.
    """
    required = {"x", "rate", "n", "lcl", "ucl", "p0"}
    missing = required - set(pchart.columns)
    if missing:
        raise ValueError(f"pchart missing columns: {sorted(missing)}")

    df = pchart.sort_values("x")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["rate"],
            mode="lines+markers",
            name="Observed fail rate",
            line={"color": COLORS["process"], "width": 2},
            marker={"size": 7},
            customdata=np.column_stack([df["n"], df["lcl"], df["ucl"]]),
            hovertemplate=(
                "wafer index %{x}<br>fail rate %{y:.2%}<br>window n=%{customdata[0]}"
                "<br>LCL %{customdata[1]:.2%}<br>UCL %{customdata[2]:.2%}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["ucl"],
            mode="lines",
            name="UCL",
            line={"color": COLORS["limit"], "dash": "dash"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["lcl"],
            mode="lines",
            name="LCL",
            line={"color": COLORS["limit"], "dash": "dash"},
            fill="tonexty",
            fillcolor="rgba(214, 39, 40, 0.04)",
        )
    )
    fig.add_hline(
        y=float(df["p0"].iloc[0]),
        line_color=COLORS["center"],
        line_width=1.5,
        annotation_text=f"Phase-I p̄ = {float(df['p0'].iloc[0]):.2%}",
        annotation_position="top left",
    )
    alarms = df["rate"] > df["ucl"]
    if alarms.any():
        fig.add_trace(
            go.Scatter(
                x=df.loc[alarms, "x"],
                y=df.loc[alarms, "rate"],
                mode="markers",
                name="Beyond UCL",
                marker={
                    "size": 11,
                    "symbol": "circle-open",
                    "color": COLORS["risk"],
                    "line": {"width": 2},
                },
            )
        )
    _phase_marks(fig, i_hold, i_tail)
    _layout(fig, "Yield p-chart: frozen Phase-I limits applied to later wafers")
    fig.update_xaxes(title="Wafer index (time order)")
    fig.update_yaxes(title="Fail rate", tickformat=".1%", rangemode="tozero")
    return fig


def make_protocol_rates_fig(
    rates: pd.DataFrame,
    feature: str | None = None,
    i_hold: int | None = None,
    i_tail: int | None = None,
) -> go.Figure:
    """Windowed dropout/protocol rate chart from protocol_rates.csv.

    If feature is omitted, one trace per persisted protocol feature is shown.
    With a feature selected, its p0, UCL, LCL, and beyond-UCL marks appear.
    """
    required = {"feature", "x", "rate", "n", "lcl", "ucl", "p0"}
    missing = required - set(rates.columns)
    if missing:
        raise ValueError(f"protocol rates missing columns: {sorted(missing)}")
    df = rates.copy()
    available = sorted(df["feature"].unique())
    if feature is not None:
        if feature not in available:
            raise ValueError(f"unknown protocol feature '{feature}'; available: {available}")
        df = df[df["feature"] == feature].sort_values("x")

    fig = go.Figure()
    if feature is None:
        for name, group in df.groupby("feature", sort=True):
            fig.add_trace(
                go.Scatter(
                    x=group["x"],
                    y=group["rate"],
                    mode="lines+markers",
                    name=name,
                    customdata=np.column_stack([group["n"]]),
                    hovertemplate="%{fullData.name}<br>wafer index %{x}<br>rate %{y:.2%}<br>window n=%{customdata[0]}<extra></extra>",
                )
            )
        title = "Protocol missingness rates by window"
    else:
        fig.add_trace(
            go.Scatter(
                x=df["x"],
                y=df["rate"],
                mode="lines+markers",
                name=feature,
                line={"color": COLORS["process"], "width": 2},
                marker={"size": 7},
            )
        )
        for column, name, dash, color in [
            ("ucl", "UCL", "dash", COLORS["limit"]),
            ("lcl", "LCL", "dash", COLORS["limit"]),
        ]:
            fig.add_trace(
                go.Scatter(
                    x=df["x"],
                    y=df[column],
                    mode="lines",
                    name=name,
                    line={"color": color, "dash": dash},
                )
            )
        fig.add_hline(
            y=float(df["p0"].iloc[0]),
            line_color=COLORS["center"],
            line_width=1.5,
            annotation_text=f"Phase-I rate = {float(df['p0'].iloc[0]):.2%}",
            annotation_position="top left",
        )
        alarms = df["rate"] > df["ucl"]
        if alarms.any():
            fig.add_trace(
                go.Scatter(
                    x=df.loc[alarms, "x"],
                    y=df.loc[alarms, "rate"],
                    mode="markers",
                    name="Beyond UCL",
                    marker={
                        "size": 11,
                        "symbol": "circle-open",
                        "color": COLORS["risk"],
                        "line": {"width": 2},
                    },
                )
            )
        title = f"Protocol rate: {feature}"
    _phase_marks(fig, i_hold, i_tail)
    _layout(fig, title)
    fig.update_xaxes(title="Wafer index (time order)")
    fig.update_yaxes(title="Missingness rate", tickformat=".1%", rangemode="tozero")
    return fig


def make_row_missing_fig(
    row_missing: pd.DataFrame,
    i_hold: int | None = None,
    i_tail: int | None = None,
) -> go.Figure:
    """I-chart for fraction of raw sensor channels unmeasured per wafer."""
    required = {"feature", "t", "value", "center", "ucl"}
    missing = required - set(row_missing.columns)
    if missing:
        raise ValueError(f"row-missing series missing columns: {sorted(missing)}")
    df = row_missing.sort_values("t")
    feature = str(df["feature"].iloc[0])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["t"],
            y=df["value"],
            mode="markers",
            name=feature,
            marker={"size": 6, "color": COLORS["process"]},
            hovertemplate="wafer %{x}<br>row missingness %{y:.2%}<extra></extra>",
        )
    )
    fig.add_hline(
        y=float(df["center"].iloc[0]),
        line_color=COLORS["center"],
        line_width=1.5,
        annotation_text="Phase-I center",
        annotation_position="top left",
    )
    fig.add_hline(
        y=float(df["ucl"].iloc[0]),
        line_color=COLORS["limit"],
        line_dash="dash",
        line_width=1.5,
        annotation_text="UCL",
        annotation_position="top left",
    )
    alarms = df["value"] > df["ucl"]
    if alarms.any():
        fig.add_trace(
            go.Scatter(
                x=df.loc[alarms, "t"],
                y=df.loc[alarms, "value"],
                mode="markers",
                name="Beyond UCL",
                marker={
                    "size": 10,
                    "symbol": "circle-open",
                    "color": COLORS["risk"],
                    "line": {"width": 2},
                },
            )
        )
    _phase_marks(fig, i_hold, i_tail)
    _layout(fig, "Protocol drift: fraction of raw channels unmeasured")
    fig.update_xaxes(title="Wafer index (time order)")
    fig.update_yaxes(title="Row missingness", tickformat=".1%", rangemode="tozero")
    return fig


def make_imr_fig(
    imr: pd.DataFrame,
    limits: pd.DataFrame,
    feature: str,
    i_hold: int | None = None,
    i_tail: int | None = None,
) -> go.Figure:
    """Three-row I / MR / EWMA chart for one persisted showcase feature."""
    required = {
        "feature",
        "t",
        "value",
        "mr",
        "ewma",
        "ewma_lcl",
        "ewma_ucl",
        "r1",
        "r2",
        "r3",
        "r4",
    }
    missing = required - set(imr.columns)
    if missing:
        raise ValueError(f"IMR series missing columns: {sorted(missing)}")
    if feature not in set(imr["feature"]):
        raise ValueError(f"no IMR series for '{feature}'")
    if feature not in limits.index:
        raise ValueError(f"no Phase-I limits for '{feature}'")
    df = imr[imr["feature"] == feature].sort_values("t")
    lim = limits.loc[feature]
    if bool(lim["degenerate"]):
        raise ValueError(f"'{feature}' has degenerate Phase-I limits")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(f"I-chart | {feature}", "Moving range", "EWMA"),
        row_heights=[0.48, 0.25, 0.27],
    )
    fig.add_trace(
        go.Scatter(
            x=df["t"],
            y=df["value"],
            mode="markers",
            name="Value",
            marker={"size": 5, "color": COLORS["process"]},
        ),
        row=1,
        col=1,
    )
    for value, _name, color, dash in [
        (float(lim["ucl"]), "UCL", COLORS["limit"], "dash"),
        (float(lim["center"]), "Center", COLORS["center"], "solid"),
        (float(lim["lcl"]), "LCL", COLORS["limit"], "dash"),
    ]:
        fig.add_hline(y=value, line_color=color, line_dash=dash, line_width=1, row=1, col=1)
    we_alarm = df[["r1", "r2", "r3", "r4"]].fillna(False).any(axis=1)
    if we_alarm.any():
        fig.add_trace(
            go.Scatter(
                x=df.loc[we_alarm, "t"],
                y=df.loc[we_alarm, "value"],
                mode="markers",
                name="WE alarm",
                marker={
                    "size": 10,
                    "symbol": "circle-open",
                    "color": COLORS["alert"],
                    "line": {"width": 2},
                },
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=df["t"],
            y=df["mr"],
            mode="markers",
            name="MR",
            marker={"size": 5, "color": COLORS["mr"]},
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=float(lim["mr_ucl"]),
        line_color=COLORS["limit"],
        line_dash="dash",
        line_width=1,
        row=2,
        col=1,
    )
    fig.add_hline(y=float(lim["mr_bar"]), line_color=COLORS["center"], line_width=1, row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=df["t"],
            y=df["ewma"],
            mode="lines",
            name="EWMA",
            line={"color": COLORS["ewma"], "width": 2},
        ),
        row=3,
        col=1,
    )
    for column, name, dash in [("ewma_ucl", "EWMA UCL", "dot"), ("ewma_lcl", "EWMA LCL", "dot")]:
        fig.add_trace(
            go.Scatter(
                x=df["t"],
                y=df[column],
                mode="lines",
                name=name,
                line={"color": COLORS["limit"], "dash": dash, "width": 1},
            ),
            row=3,
            col=1,
        )
    _phase_marks(fig, i_hold, i_tail)
    _layout(fig, f"SPC showcase: {feature}", height=760)
    fig.update_xaxes(title="Wafer index (time order)", row=3, col=1)
    fig.update_yaxes(title="Sensor value", row=1, col=1)
    fig.update_yaxes(title="Moving range", row=2, col=1)
    fig.update_yaxes(title="EWMA", row=3, col=1)
    return fig


def make_drift_overview_fig(screening: pd.DataFrame) -> go.Figure:
    """Phase-I vs Phase-II OOC alarm-rate scatter from screening data."""
    required = {"ooc_p1", "ooc_p2", "delta", "degenerate"}
    missing = required - set(screening.columns)
    if missing:
        raise ValueError(f"screening missing columns: {sorted(missing)}")
    df = screening[
        ~screening["degenerate"] & screening["ooc_p1"].notna() & screening["ooc_p2"].notna()
    ].copy()
    eps = 1e-4
    df["x"] = df["ooc_p1"] + eps
    df["y"] = df["ooc_p2"] + eps
    names = (
        df.index.astype(str)
        if df.index.name
        else df.get("feature", pd.Series(df.index.astype(str)))
    )
    hover = np.column_stack([names, df["delta"]])
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["x"],
            y=df["y"],
            mode="markers",
            name="Screened sensors",
            marker={
                "size": 7,
                "color": df["delta"],
                "colorscale": "RdYlBu_r",
                "showscale": True,
                "colorbar": {"title": "Δ OOC"},
            },
            customdata=hover,
            hovertemplate="%{customdata[0]}<br>Phase-I OOC %{x:.2%}<br>Phase-II OOC %{y:.2%}<br>Δ %{customdata[1]:.2%}<extra></extra>",
        )
    )
    max_rate = max(float(df["x"].max()), float(df["y"].max()), 0.01)
    fig.add_trace(
        go.Scatter(
            x=[eps, max_rate],
            y=[eps, max_rate],
            mode="lines",
            name="No change",
            line={"color": "#111827", "dash": "dash"},
        )
    )
    _layout(fig, "SPC screening: Phase-II alarm rate versus Phase-I", height=500)
    fig.update_xaxes(title="Phase-I OOC alarm rate", type="log", tickformat=".1%")
    fig.update_yaxes(title="Phase-II OOC alarm rate", type="log", tickformat=".1%")
    return fig


def make_queue_fig(queue: pd.DataFrame, top_k: int = 20) -> go.Figure:
    """Ranked calibrated-risk bar chart from a persisted score queue."""
    required = {"wafer_id", "p_cal", "rank"}
    missing = required - set(queue.columns)
    if missing:
        raise ValueError(f"score queue missing columns: {sorted(missing)}")
    df = queue.nsmallest(top_k, "rank").sort_values("rank", ascending=False)
    colors = [COLORS["risk"] if rank <= top_k else COLORS["process"] for rank in df["rank"]]
    custom = np.column_stack([df["rank"], df.get("decile", pd.Series([math.nan] * len(df)))])
    fig = go.Figure(
        go.Bar(
            x=df["p_cal"],
            y=df["wafer_id"].astype(str),
            orientation="h",
            marker_color=colors,
            customdata=custom,
            hovertemplate="wafer %{y}<br>calibrated risk %{x:.2%}<br>inspection rank %{customdata[0]}<br>risk decile %{customdata[1]}<extra></extra>",
            name="Calibrated risk",
        )
    )
    _layout(
        fig,
        f"Inspection-priority queue: top {min(top_k, len(queue))} wafers",
        height=max(430, 28 * len(df) + 130),
    )
    fig.update_xaxes(title="Calibrated failure risk", tickformat=".1%", rangemode="tozero")
    fig.update_yaxes(title="Wafer ID")
    return fig


def make_alert_columns() -> list[dict]:
    """dash-ag-grid column definitions for the drift/alert table.

    Kept as data, not a Dash component, so dash_app.py owns the UI library.
    """
    return [
        {"field": "feature", "headerName": "Feature", "filter": True, "pinned": "left"},
        {"field": "drift", "headerName": "Drift", "filter": True},
        {
            "field": "delta",
            "headerName": "Δ OOC",
            "valueFormatter": {"function": "d3.format('.2%')(params.value)"},
        },
        {
            "field": "ooc_p1",
            "headerName": "Phase-I OOC",
            "valueFormatter": {"function": "d3.format('.2%')(params.value)"},
        },
        {
            "field": "ooc_p2",
            "headerName": "Phase-II OOC",
            "valueFormatter": {"function": "d3.format('.2%')(params.value)"},
        },
        {
            "field": "ewma_delta",
            "headerName": "Δ EWMA",
            "valueFormatter": {"function": "d3.format('.2%')(params.value)"},
        },
        {
            "field": "miss_p2",
            "headerName": "Phase-II missing",
            "valueFormatter": {"function": "d3.format('.2%')(params.value)"},
        },
    ]
