"""Migration-equivalence canaries: the real SECOM snapshot must look exactly
as the pipeline was validated against. Requires the real database.
Run: uv run pytest -m local_data
"""

import pytest

from semcon.db import get_engine, load_registry
from semcon.extract import extract
from semcon.feature_eng import build_features

pytestmark = pytest.mark.local_data   # every test in this file gets the tag

EXPECTED_ZONES = {"cv": 1309, "holdout": 231, "excluded": 27}
EXPECTED_FAILS = {"cv": 90, "holdout": 14, "excluded": 0}
CLIQUE_ANCHORS = {"f_miss_clq14": 794, "f_miss_clq23": 715}  # caught the s112/s113 bug
EXPECTED_ACTIVE_FEATURES = 261


@pytest.fixture(scope="module")
def real_engine():
    return get_engine()   # the real DB, default path


@pytest.fixture(scope="module")
def real_frame(real_engine):
    return extract(real_engine)   # default CUTOFF/EXCLUDE_AFTER from config


@pytest.fixture(scope="module")
def real_model_frame(real_frame, real_engine):
    out, rows = build_features(real_frame)
    return out   # registration already happened in production runs


@pytest.fixture(scope="module")
def real_registry(real_engine):
    return load_registry(real_engine)
