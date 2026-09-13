"""Tests for dash_app: app factory, CLI parsing, and artifact-root configuration.

No browser server is started. Data/figure details belong to test_dash_data.py
and test_dash_figs.py; these tests verify only the Dash integration seam.
"""

from __future__ import annotations

from pathlib import Path

from dash import dcc, html

from semcon import dash_app, dash_data


def _walk(component):
    """Yield a Dash component tree without requiring dash.testing/browser tools."""
    yield component
    children = getattr(component, "children", None)
    if children is None:
        return
    if not isinstance(children, (list, tuple)):
        children = [children]
    for child in children:
        if child is not None:
            yield from _walk(child)


def test_configure_artifacts_updates_all_loader_roots(tmp_path: Path) -> None:
    root = tmp_path / "mounted_artifacts"
    configured = dash_app.configure_artifacts(root)
    assert configured == root
    assert {
        "monitor_index": root / "index_monitor.csv",
        "training_index": root / "index.csv",
        "runs": root / "runs",
        "scores": root / "scores",
    } == dash_data.PATHS


def test_create_app_has_expected_title_components_and_callbacks(tmp_path: Path) -> None:
    app = dash_app.create_app(tmp_path / "artifacts")
    assert app.title == dash_app.APP_TITLE
    assert app.layout is not None
    assert len(app.callback_map) == 6
    ids = {getattr(component, "id", None) for component in _walk(app.layout)}
    assert set(dash_app.IDS.values()) <= ids
    assert any(isinstance(component, dcc.Interval) for component in _walk(app.layout))
    assert any(isinstance(component, html.H1) for component in _walk(app.layout))


def test_parse_args_defaults() -> None:
    args = dash_app.parse_args([])
    assert args.artifacts is None
    assert args.host == "127.0.0.1"
    assert args.port == 8050
    assert not args.debug


def test_parse_args_accepts_artifact_root_and_server_options(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    args = dash_app.parse_args(
        ["--artifacts", str(root), "--host", "0.0.0.0", "--port", "9000", "--debug"]
    )
    assert args.artifacts == root
    assert args.host == "0.0.0.0"
    assert args.port == 9000
    assert args.debug


def test_message_figure_is_safe_empty_state() -> None:
    fig = dash_app._message_figure("No data", "index_monitor.csv not found")
    assert fig.layout.title.text == "No data"
    assert fig.layout.annotations[0].text == "index_monitor.csv not found"
    assert fig.layout.xaxis.visible is False
    assert fig.layout.yaxis.visible is False


def test_pd_notna_handles_registry_null_forms() -> None:
    assert dash_app.pd_notna(0.025)
    assert dash_app.pd_notna("20260911_snapshot")
    assert not dash_app.pd_notna(None)
    assert not dash_app.pd_notna(float("nan"))
    assert not dash_app.pd_notna("NaN")
