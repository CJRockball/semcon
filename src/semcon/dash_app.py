"""Dash application for semcon process and model monitoring.

This is the only dashboard module that imports Dash. It owns the browser:
layout, callbacks, CLI, and the optional artifacts-root override.

Architecture:
  dash_data.py  -> registry/pointer resolution and DataFrames
  dash_figs.py  -> DataFrames to pure Plotly Figures
  dash_app.py   -> components and callback wiring

The app resolves the latest SPC run inside every refresh callback. A cron job
can append a new row to index_monitor.csv and write its artifacts; the next
refresh displays that run without an application restart.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import dash_ag_grid as dag
from dash import Dash, Input, Output, dcc, html

from semcon import dash_data, dash_figs
from semcon.paths import ARTIFACTS

APP_TITLE = "semcon | Process & Model Monitoring"
REFRESH_MS = 30_000
TOP_K = 20

IDS = {
    "refresh": "dash-refresh",
    "status": "dash-status",
    "run_id": "dash-run-id",
    "kpi_screened": "dash-kpi-screened",
    "kpi_drift": "dash-kpi-drift",
    "kpi_alarm": "dash-kpi-alarm",
    "kpi_snapshot": "dash-kpi-snapshot",
    "pchart": "dash-pchart",
    "row_missing": "dash-row-missing",
    "protocol_feature": "dash-protocol-feature",
    "protocol_rates": "dash-protocol-rates",
    "overview": "dash-overview",
    "alert_grid": "dash-alert-grid",
    "imr_feature": "dash-imr-feature",
    "imr": "dash-imr",
    "score_batch": "dash-score-batch",
    "queue": "dash-queue",
}

STYLES = {
    "page": {
        "maxWidth": "1440px",
        "margin": "0 auto",
        "padding": "20px 28px 40px",
        "fontFamily": "Arial, sans-serif",
        "backgroundColor": "#f8fafc",
    },
    "header": {"marginBottom": "14px"},
    "subtitle": {"color": "#4b5563", "marginTop": "4px"},
    "status": {"color": "#4b5563", "fontSize": "0.9rem", "marginBottom": "14px"},
    "kpis": {"display": "flex", "flexWrap": "wrap", "gap": "12px", "marginBottom": "16px"},
    "kpi": {
        "background": "white",
        "border": "1px solid #e5e7eb",
        "borderRadius": "8px",
        "padding": "12px 16px",
        "minWidth": "180px",
        "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
    },
    "kpi_label": {"fontSize": "0.78rem", "color": "#6b7280", "textTransform": "uppercase"},
    "kpi_value": {"fontSize": "1.5rem", "fontWeight": "700", "color": "#111827"},
    "panel": {
        "background": "white",
        "border": "1px solid #e5e7eb",
        "borderRadius": "8px",
        "padding": "10px 14px 14px",
        "marginBottom": "16px",
    },
    "controls": {
        "display": "flex",
        "flexWrap": "wrap",
        "alignItems": "center",
        "gap": "12px",
        "margin": "6px 0 10px",
    },
    "label": {"fontWeight": "600", "fontSize": "0.9rem"},
}


def configure_artifacts(artifacts: Path | None) -> Path:
    """Set dashboard loader roots; default is the project artifacts directory.

    All roots are kept together. With --artifacts /mounted/artifacts, the
    dashboard reads /mounted/artifacts/{index*,runs,scores}; no current
    working-directory assumption leaks into callbacks.
    """
    root = Path(artifacts) if artifacts is not None else ARTIFACTS
    dash_data.PATHS.update(
        {
            "monitor_index": root / "index_monitor.csv",
            "training_index": root / "index.csv",
            "runs": root / "runs",
            "scores": root / "scores",
        }
    )
    return root


def _kpi(label: str, component_id: str) -> html.Div:
    return html.Div(
        [
            html.Div(label, style=STYLES["kpi_label"]),
            html.Div("—", id=component_id, style=STYLES["kpi_value"]),
        ],
        style=STYLES["kpi"],
    )


def _panel(title: str, children: list) -> html.Div:
    return html.Div(
        [html.H3(title, style={"fontSize": "1.05rem", "margin": "6px 0 8px"}), *children],
        style=STYLES["panel"],
    )


def _controls(label: str, component) -> html.Div:
    return html.Div([html.Span(label, style=STYLES["label"]), component], style=STYLES["controls"])


def build_layout() -> html.Div:
    """Static component tree. Data-bearing props are populated by callbacks."""
    return html.Div(
        [
            dcc.Interval(id=IDS["refresh"], interval=REFRESH_MS, n_intervals=0),
            html.Header(
                [
                    html.H1(APP_TITLE, style={"marginBottom": "0"}),
                    html.P(
                        "Batch-scored inspection priority and frozen-limit SPC. "
                        "SECOM lacks tool/chamber/lot identifiers, so drill-down is by time phase, channel, and score rank.",
                        style=STYLES["subtitle"],
                    ),
                ],
                style=STYLES["header"],
            ),
            html.Div(id=IDS["status"], style=STYLES["status"]),
            html.Div(
                [
                    _kpi("Latest monitoring run", IDS["run_id"]),
                    _kpi("Sensors screened", IDS["kpi_screened"]),
                    _kpi("Drifted sensors", IDS["kpi_drift"]),
                    _kpi("Median Phase-I OOC", IDS["kpi_alarm"]),
                    _kpi("Gold snapshot", IDS["kpi_snapshot"]),
                ],
                style=STYLES["kpis"],
            ),
            dcc.Tabs(
                [
                    dcc.Tab(
                        label="Process health",
                        children=[
                            _panel(
                                "Yield p-chart",
                                [dcc.Graph(id=IDS["pchart"], config={"displaylogo": False})],
                            ),
                            _panel(
                                "Raw-channel protocol health",
                                [dcc.Graph(id=IDS["row_missing"], config={"displaylogo": False})],
                            ),
                            _panel(
                                "Dropout-indicator rates",
                                [
                                    _controls(
                                        "Protocol feature",
                                        dcc.Dropdown(
                                            id=IDS["protocol_feature"],
                                            clearable=False,
                                            style={"minWidth": "260px"},
                                        ),
                                    ),
                                    dcc.Graph(
                                        id=IDS["protocol_rates"], config={"displaylogo": False}
                                    ),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="SPC screening",
                        children=[
                            _panel(
                                "Phase-I versus Phase-II alarm-rate overview",
                                [dcc.Graph(id=IDS["overview"], config={"displaylogo": False})],
                            ),
                            _panel(
                                "Screening alerts",
                                [
                                    dag.AgGrid(
                                        id=IDS["alert_grid"],
                                        columnDefs=dash_figs.make_alert_columns(),
                                        defaultColDef={
                                            "sortable": True,
                                            "filter": True,
                                            "resizable": True,
                                        },
                                        dashGridOptions={
                                            "pagination": True,
                                            "paginationPageSize": 15,
                                        },
                                        style={"height": "520px"},
                                    )
                                ],
                            ),
                            _panel(
                                "Showcase channel: I / MR / EWMA",
                                [
                                    _controls(
                                        "Feature",
                                        dcc.Dropdown(
                                            id=IDS["imr_feature"],
                                            clearable=False,
                                            style={"minWidth": "220px"},
                                        ),
                                    ),
                                    dcc.Graph(id=IDS["imr"], config={"displaylogo": False}),
                                ],
                            ),
                        ],
                    ),
                    dcc.Tab(
                        label="Inspection priority",
                        children=[
                            _panel(
                                "Batch scoring queue",
                                [
                                    _controls(
                                        "Scored batch",
                                        dcc.Dropdown(
                                            id=IDS["score_batch"],
                                            clearable=False,
                                            style={"minWidth": "360px"},
                                        ),
                                    ),
                                    dcc.Graph(id=IDS["queue"], config={"displaylogo": False}),
                                ],
                            )
                        ],
                    ),
                ]
            ),
        ],
        style=STYLES["page"],
    )


def _split_boundaries(run: dict) -> tuple[int | None, int | None]:
    """Read i_hold/i_tail stored by spc.py; absent values keep figures useful."""
    config = dash_data.load_run_config(run)
    return config.get("i_hold"), config.get("i_tail")


def _latest_run_or_none() -> tuple[dict | None, str | None]:
    try:
        return dash_data.latest_monitor_run(), None
    except (FileNotFoundError, ValueError) as exc:
        return None, str(exc)


def register_callbacks(app: Dash) -> None:
    """Wire component interactions; each refresh resolves the latest registry row."""

    @app.callback(
        Output(IDS["status"], "children"),
        Output(IDS["run_id"], "children"),
        Output(IDS["kpi_screened"], "children"),
        Output(IDS["kpi_drift"], "children"),
        Output(IDS["kpi_alarm"], "children"),
        Output(IDS["kpi_snapshot"], "children"),
        Input(IDS["refresh"], "n_intervals"),
    )
    def refresh_summary(_: int):
        run, error = _latest_run_or_none()
        if error:
            return (f"Waiting for monitoring artifacts: {error}", "—", "—", "—", "—", "—")
        meta = run["meta"]
        return (
            f"Reading the latest registry row on every refresh. Artifact root: {dash_data.PATHS['monitor_index'].parent}",
            run["run_id"],
            str(meta.get("n_screened", "—")),
            str(meta.get("n_drift", "—")),
            f"{float(meta['median_ooc_p1']):.2%}" if pd_notna(meta.get("median_ooc_p1")) else "—",
            str(meta.get("snapshot_id", "—")),
        )

    @app.callback(
        Output(IDS["pchart"], "figure"),
        Output(IDS["row_missing"], "figure"),
        Output(IDS["overview"], "figure"),
        Output(IDS["alert_grid"], "rowData"),
        Output(IDS["protocol_feature"], "options"),
        Output(IDS["protocol_feature"], "value"),
        Output(IDS["imr_feature"], "options"),
        Output(IDS["imr_feature"], "value"),
        Input(IDS["refresh"], "n_intervals"),
    )
    def refresh_monitoring(_: int):
        run, error = _latest_run_or_none()
        if error:
            blank = _message_figure("Monitoring artifacts unavailable", error)
            return blank, blank, blank, [], [], None, [], None
        try:
            i_hold, i_tail = _split_boundaries(run)
            pchart = dash_data.load_pchart(run)
            row_missing = dash_data.load_protocol_row_missing(run)
            protocol = dash_data.load_protocol_rates(run)
            screening = dash_data.load_screening(run)
            drift = dash_data.load_drift_table(run)
            imr = dash_data.load_imr_series(run)
            protocol_features = sorted(protocol["feature"].unique())
            imr_features = sorted(imr["feature"].unique())
            return (
                dash_figs.make_pchart_fig(pchart, i_hold, i_tail),
                dash_figs.make_row_missing_fig(row_missing, i_hold, i_tail),
                dash_figs.make_drift_overview_fig(screening),
                drift.reset_index(names="feature").to_dict("records"),
                [{"label": f, "value": f} for f in protocol_features],
                protocol_features[0] if protocol_features else None,
                [{"label": f, "value": f} for f in imr_features],
                imr_features[0] if imr_features else None,
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            blank = _message_figure("Monitoring artifact contract error", str(exc))
            return blank, blank, blank, [], [], None, [], None

    @app.callback(
        Output(IDS["protocol_rates"], "figure"),
        Input(IDS["protocol_feature"], "value"),
        Input(IDS["refresh"], "n_intervals"),
    )
    def refresh_protocol(feature: str | None, _: int):
        run, error = _latest_run_or_none()
        if error or feature is None:
            return _message_figure("Protocol rates", error or "Choose a protocol feature")
        try:
            i_hold, i_tail = _split_boundaries(run)
            return dash_figs.make_protocol_rates_fig(
                dash_data.load_protocol_rates(run), feature, i_hold, i_tail
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            return _message_figure("Protocol artifact contract error", str(exc))

    @app.callback(
        Output(IDS["imr"], "figure"),
        Input(IDS["imr_feature"], "value"),
        Input(IDS["refresh"], "n_intervals"),
    )
    def refresh_imr(feature: str | None, _: int):
        run, error = _latest_run_or_none()
        if error or feature is None:
            return _message_figure("I / MR / EWMA", error or "Choose a showcase feature")
        try:
            i_hold, i_tail = _split_boundaries(run)
            return dash_figs.make_imr_fig(
                dash_data.load_imr_series(run), dash_data.load_limits(run), feature, i_hold, i_tail
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            return _message_figure("IMR artifact contract error", str(exc))

    @app.callback(
        Output(IDS["score_batch"], "options"),
        Output(IDS["score_batch"], "value"),
        Input(IDS["refresh"], "n_intervals"),
    )
    def refresh_score_batches(_: int):
        try:
            batches = dash_data.list_score_batches()
        except FileNotFoundError:
            return [], None
        options = [
            {"label": dash_data.run_label(batch), "value": batch["run_id"]} for batch in batches
        ]
        return options, options[-1]["value"] if options else None

    @app.callback(
        Output(IDS["queue"], "figure"),
        Input(IDS["score_batch"], "value"),
        Input(IDS["refresh"], "n_intervals"),
    )
    def refresh_queue(run_id: str | None, _: int):
        if run_id is None:
            return _message_figure("Inspection-priority queue", "No scored batches found")
        try:
            batches = {batch["run_id"]: batch for batch in dash_data.list_score_batches()}
            if run_id not in batches:
                raise ValueError(f"score batch is no longer available: {run_id}")
            return dash_figs.make_queue_fig(
                dash_data.load_score_queue(batches[run_id]), top_k=TOP_K
            )
        except (FileNotFoundError, ValueError, KeyError) as exc:
            return _message_figure("Score artifact contract error", str(exc))


def pd_notna(value) -> bool:
    """Avoid importing pandas solely to render a nullable registry scalar."""
    return value is not None and str(value).lower() != "nan"


def _message_figure(title: str, message: str):
    """Safe visual state rather than a callback traceback in the browser."""
    from plotly.graph_objects import Figure

    fig = Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)
    fig.update_layout(title=title, height=300, template="plotly_white")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def create_app(artifacts: Path | None = None) -> Dash:
    """Create the dashboard application; artifact root override supports Docker."""
    configure_artifacts(artifacts)
    app = Dash(__name__, title=APP_TITLE, suppress_callback_exceptions=True)
    app.layout = build_layout()
    register_callbacks(app)
    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch the semcon monitoring dashboard")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=None,
        help="Artifact root containing index_monitor.csv, runs/, and scores/ (default: project artifacts/)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8050, help="TCP port (default: 8050)")
    parser.add_argument("--debug", action="store_true", help="Enable Dash debug mode")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    app = create_app(args.artifacts)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
