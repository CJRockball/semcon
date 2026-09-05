"""Schema facts — structural constants the pipeline assumes.

Decisions (cutoff, holdout size, thresholds) live in config.py.
Facts about the data's shape live here. Rule of thumb: if changing a
value would break the code's assumptions, it is a fact and belongs here.
"""

import enum

KEY_COL = "wafer_id"
TARGET_COL = "target"
TIME_COL = "timestamp"
SPLIT_COL = "split"

SENSOR_PREFIX = "s"
ENG_PREFIX = "f_"
N_SENSORS = 590

METADATA_COLS = [TIME_COL, SPLIT_COL]
NON_FEATURE_COLS = [KEY_COL, TARGET_COL, TIME_COL, SPLIT_COL]

EXPECTED_WAFERS = 1567  # external data contract for the SECOM snapshot


class Role(enum.StrEnum):
    KEY = "key"
    METADATA = "metadata"
    FEATURE_RAW = "feature_raw"
    FEATURE_ENG = "feature_eng"
    TARGET = "target"


class Status(enum.StrEnum):
    ACTIVE = "active"
    EXCLUDED = "excluded"
