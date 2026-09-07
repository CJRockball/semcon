"""Migration-equivalence canaries: the real SECOM snapshot must look exactly
as the pipeline was validated against. Requires the real database.
Run: uv run pytest -m local_data
"""

import pandas as pd
import pytest

from semcon import schema
from semcon.db import feature_columns, get_engine, load_registry
from semcon.extract import extract
from semcon.feature_eng import build_features

pytestmark = pytest.mark.local_data   # every test in this file gets the tag

EXPECTED_ZONES = {"cv": 1309, "holdout": 231, "excluded": 27}
EXPECTED_FAILS = {"cv": 90, "holdout": 14, "excluded": 0}
CLIQUE_ANCHORS = {"f_miss_clq14": 794, "f_miss_clq23": 715}  # caught the s112/s113 bug
EXPECTED_ACTIVE = 261


@pytest.fixture(scope="module")
def real_engine():
    return get_engine()   # the real DB, default path


@pytest.fixture(scope="module")
def real_frame(real_engine):
    return extract(real_engine)   # default CUTOFF/EXCLUDE_AFTER from config


@pytest.fixture(scope="module")
def real_model_frame(real_frame):
    out, _rows = build_features(real_frame)
    return out


@pytest.fixture(scope="module")
def real_registry(real_engine):
    return load_registry(real_engine)


def test_bronze_row_counts(real_engine):
    for table in ("sensor_readings", "wafer_labels"):
        n = int(pd.read_sql(f"SELECT COUNT(*) AS n FROM {table}", real_engine)["n"][0])
        assert n == schema.EXPECTED_WAFERS, f"{table}: {n} rows"


def test_split_zone_counts(real_frame):
    counts = real_frame["split"].value_counts().to_dict()
    assert counts == EXPECTED_ZONES, counts


def test_split_zone_fails(real_frame):
    fails = real_frame.groupby("split")[schema.TARGET_COL].apply(lambda s: int(s.eq(1).sum())).to_dict()
    assert fails == EXPECTED_FAILS, fails
    assert sum(fails.values()) == 104


def test_clique_anchors(real_model_frame):
    for col, expected in CLIQUE_ANCHORS.items():
        actual = int(real_model_frame[col].sum())
        assert actual == expected, f"{col}: {actual} != {expected}"


def test_active_feature_count(real_registry):
    """explore.py's demotion: 590 raw -> 261 active."""
    assert len(feature_columns(real_registry)) == EXPECTED_ACTIVE
