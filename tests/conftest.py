import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine

from semcon import db_ingest, schema
from semcon.db import get_engine, load_registry, register_columns
from semcon.extract import extract
from semcon.feature_eng import build_features

N_WAFERS = 20
FIXTURE_CUTOFF_OFFSET = pd.Timedelta(minutes=30)

def _write_raw(raw):
    rng = np.random.default_rng(7)
    X = rng.normal(0.5, 0.1, size=(N_WAFERS, schema.N_SENSORS))
    X[[3, 4, 5], 0:2] = np.nan          # planted clique: s001+s002 missing together
    pd.DataFrame(X).to_csv(raw / "secom.data", sep=" ", header=False, index=False)
    ts = pd.date_range("2008-07-25 00:00:00", periods=N_WAFERS, freq="h")
    labels = [-1] * N_WAFERS
    labels[2] = labels[6] = labels[13] = 1
    pd.DataFrame({"y": labels, "t": ts.strftime("%d/%m/%Y %H:%M:%S")}).to_csv(
        raw / "secom_labels.data", sep=" ", header=False, index=False)
    return ts


@pytest.fixture(scope="session")
def synthetic_env(tmp_path_factory):
    """Build the fixture DB; return engine + the timestamps used."""
    raw = tmp_path_factory.mktemp("raw")
    ts = _write_raw(raw)
    eng = create_engine(f"sqlite:///{raw / 'test.db'}")
    db_ingest.setup_db(eng)
    dfX, dfy = db_ingest.load_data(raw, expected_wafers=N_WAFERS)
    db_ingest.insert_data(dfX, dfy, db_ingest.build_registry(dfX), eng, data_dir=raw)
    return eng, ts


@pytest.fixture(scope="session")
def engine(synthetic_env):
    eng, _ts = synthetic_env
    return eng


@pytest.fixture(scope="session")
def frame(synthetic_env):
    eng, ts = synthetic_env
    return extract(
        eng,
        cutoff=str(ts[12] - pd.Timedelta(minutes=30)),         # between wafer 12 and 13
        exclude_after=str(ts[17] - pd.Timedelta(minutes=30)),  # between wafer 17 and 18
    )


@pytest.fixture(scope="session")
def monkeypatch_session():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="session")
def model_frame(frame, synthetic_env):
    """Silver + engineered features (the train_xgb input composition)."""
    eng, _ts = synthetic_env
    out, registry_rows = build_features(frame)
    register_columns(registry_rows, eng)    
    return out


@pytest.fixture(scope="session")
def registry(model_frame, synthetic_env):
    eng, _ts = synthetic_env
    return load_registry(eng)

