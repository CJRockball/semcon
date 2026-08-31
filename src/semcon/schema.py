KEY_COL = "wafer_id"
TARGET_COL = "target"
TIME_COL = "timestamp"
SENSOR_PREFIX = "s"
N_SENSORS = 590
ENG_PREFIX = "f_"
METADATA_COLS = [TIME_COL, "split"]

class Role:            # or StrEnum
    KEY = "key"
    METADATA = "metadata"
    FEATURE_RAW = "feature_raw"
    FEATURE_ENG = "feature_eng"
    TARGET = "target"