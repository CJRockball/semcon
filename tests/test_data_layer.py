"""Data-layer contract tests: ingest -> extract -> registry -> features -> labels.

These tests assert on the live database and extraction, so run them after
`make` (or at least `make ingest extract explore features validate`).
One contract per test; the fixtures do the heavy lifting once per session.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import inspect

from semcon import schema
from semcon.config import CUTOFF, EXCLUDE_AFTER
from semcon.db import feature_columns, get_engine, load_registry
from semcon.extract import extract
from semcon.feature_eng import build_features
from semcon.validate import ensure_is_fail

EXPECTED_ZONES = {"cv": 1309, "holdout": 231, "excluded": 27}
EXPECTED_FAILS = {"cv": 90, "holdout": 14, "excluded": 0}
ENGINEERED = ["f_miss_clq14", "f_miss_clq23", "f_miss_block5", "f_row_missing_rate"]
CLIQUE_ANCHORS = {"f_miss_clq14": 794, "f_miss_clq23": 715}  # caught the s112/s113 bug


@pytest.fixture(scope="session")
def engine():
    return get_engine()


@pytest.fixture(scope="session")
def frame(engine):
    """The silver extraction: sensors + key + timestamp + target + split."""
    return extract(engine)


@pytest.fixture(scope="session")
def model_frame(frame):
    """Silver + engineered features (the train_xgb input composition)."""
    out, _registry_rows = build_features(frame)
    return out


@pytest.fixture(scope="session")
def registry(engine):
    return load_registry(engine)


def _names(reg: pd.DataFrame) -> pd.Index:
    return pd.Index(reg["column_name"]) if "column_name" in reg.columns else reg.index


# --- ingest / bronze ------------------------------------------------------------


def test_bronze_tables_exist(engine):
    tables = set(inspect(engine).get_table_names())
    expected = {"sensor_readings", "wafer_labels", "column_registry", "ingestion_log"}
    assert expected <= tables, f"missing: {expected - tables}"


def test_bronze_row_counts(engine):
    for table in ("sensor_readings", "wafer_labels"):
        n = int(pd.read_sql(f"SELECT COUNT(*) AS n FROM {table}", engine)["n"][0])
        assert n == schema.EXPECTED_WAFERS, f"{table}: {n} rows"


def test_primary_key_survives(engine):
    """Canary for the replace-drops-schema bug class."""
    info = pd.read_sql("PRAGMA table_info(sensor_readings)", engine)
    row = info[info["name"] == schema.KEY_COL]
    assert not row.empty and int(row["pk"].iloc[0]) == 1, "wafer_id is not the PK"


def test_ingestion_log_records_hashes(engine):
    log = pd.read_sql("SELECT * FROM ingestion_log", engine)
    assert len(log) >= 2, "one row per source file per load"
    hashes = log["source_sha256"].dropna().astype(str)
    assert not hashes.empty, "source_sha256 never populated"
    assert hashes.str.fullmatch(r"[0-9a-fA-F]{16,64}").all(), \
        f"unexpected hash values: {hashes.head().tolist()}"


# --- extract / silver -----------------------------------------------------------


def test_frame_shape(frame):
    # 590 sensors + wafer_id + timestamp + target + split
    assert frame.shape == (schema.EXPECTED_WAFERS, schema.N_SENSORS + 4)


def test_key_unique_nonnull(frame):
    assert frame[schema.KEY_COL].is_unique
    assert frame[schema.KEY_COL].notna().all()


def test_frame_sorted(frame):
    """The ordering invariant every positional consumer relies on."""
    assert frame[schema.TIME_COL].is_monotonic_increasing


def test_split_zone_counts(frame):
    counts = frame["split"].value_counts().to_dict()
    assert counts == EXPECTED_ZONES, counts


def test_split_zone_fails(frame):
    fails = (
        frame.groupby("split")[schema.TARGET_COL]
        .apply(lambda s: int(s.eq(1).sum()))
        .to_dict()
    )
    assert fails == EXPECTED_FAILS, fails
    assert sum(fails.values()) == 104


def test_split_boundaries_respected(frame):
    """Strict inequalities also prove no timestamp sits exactly on a boundary."""
    cutoff, excl = pd.Timestamp(CUTOFF), pd.Timestamp(EXCLUDE_AFTER)
    zones = frame.groupby("split")[schema.TIME_COL]
    assert zones.get_group("cv").max() < cutoff <= zones.get_group("holdout").min()
    assert zones.get_group("holdout").max() < excl <= zones.get_group("excluded").min()


# --- registry / explore ---------------------------------------------------------


def test_every_frame_column_registered(registry, frame):
    names = _names(registry)
    dupes = names[names.duplicated()]
    assert dupes.empty, f"duplicate registry rows: {dupes.tolist()[:5]}"
    missing = set(frame.columns) - set(names)
    assert not missing, f"unregistered frame columns: {sorted(missing)[:5]}"
    valid = {str(getattr(r, "value", r)).lower() for r in schema.Role}
    bad = set(registry["role"].astype(str).str.lower().unique()) - valid
    assert not bad, f"invalid roles: {bad}"


def test_active_feature_contract(registry):
    """The inclusion contract: 261 features, and metadata can never leak in."""
    active = feature_columns(registry)
    assert len(active) == 261, f"active features: {len(active)} != 261"
    leaked = set(active) & {
        schema.KEY_COL, schema.TIME_COL, schema.TARGET_COL, "split", "is_fail",
    }
    assert not leaked, f"non-features in active set: {leaked}"


# --- feature_eng -----------------------------------------------------------------


def test_engineered_columns_registered_and_present(registry, model_frame):
    names = set(_names(registry))
    assert not [c for c in ENGINEERED if c not in names], "engineered column unregistered"
    assert not [c for c in ENGINEERED if c not in model_frame.columns], "not in model frame"


def test_clique_anchors(model_frame):
    for col, expected in CLIQUE_ANCHORS.items():
        actual = int(model_frame[col].sum())
        assert actual == expected, f"{col}: {actual} != {expected}"


# --- label home (validate) -------------------------------------------------------


def test_is_fail_single_home(frame, engine):
    df = ensure_is_fail(frame, engine)
    assert "is_fail" in df.columns
    assert df["is_fail"].dtype == np.int8
    assert df["is_fail"].eq(frame[schema.TARGET_COL].eq(1).astype("int8")).all()
    assert int(df["is_fail"].sum()) == 104
