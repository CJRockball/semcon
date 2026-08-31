"""Ingest raw SECOM files into SQLite (bronze layer).

Raw copy only: type coercion, no value transforms. Labels stay -1/1;
encoding happens downstream. Registers every raw column in
column_registry. Convention: whoever creates a column registers it.
"""
from __future__ import annotations

import hashlib
import logging
import subprocess
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import (
    Column, DateTime, Engine, Float, Integer, MetaData, String, Table, text,
)

from semcon import schema
from semcon.db import get_engine
from semcon.paths import DATA_RAW, LOGS
from semcon.utils import setup_logging

logger = logging.getLogger("semcon")


def file_sha256(path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def setup_db(engine: Engine) -> None:
    metadata = MetaData()
    Table(
        "sensor_readings", metadata,
        Column(schema.KEY_COL, Integer, primary_key=True),
        *[Column(f"{schema.SENSOR_PREFIX}{i:03d}", Float)
          for i in range(1, schema.N_SENSORS + 1)],
    )
    Table(
        "wafer_labels", metadata,
        Column(schema.KEY_COL, Integer, primary_key=True),
        Column(schema.TARGET_COL, Integer),
        Column(schema.TIME_COL, DateTime),
    )
    Table(
        "column_registry", metadata,
        Column("column_name", String, primary_key=True),
        Column("role", String),
        Column("status", String),
        Column("col_index", Integer),
        Column("missing_pct", Float),
        Column("derived_from", String),
        Column("notes", String),
    )
    Table(
        "ingestion_log", metadata,
        Column("ingest_id", Integer, primary_key=True),
        Column("load_ts", DateTime),
        Column("source_file", String),
        Column("source_sha256", String),
        Column("table_name", String),
        Column("rows_inserted", Integer),
        Column("git_sha", String),
    )
    metadata.create_all(engine)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw files. Errors propagate — a failed load must die loudly."""
    dfX = pd.read_csv(
        DATA_RAW / "secom.data",
        sep=r"\s+", header=None, na_values=["NaN"],
    )
    dfX.columns = [
        f"{schema.SENSOR_PREFIX}{i + 1:03d}" for i in range(schema.N_SENSORS)
    ]
    dfX.insert(0, schema.KEY_COL, range(1, len(dfX) + 1))

    dfy = pd.read_csv(
        DATA_RAW / "secom_labels.data",
        sep=r"\s+", header=None,
        names=[schema.TARGET_COL, schema.TIME_COL],
    )
    dfy[schema.TIME_COL] = pd.to_datetime(
        dfy[schema.TIME_COL], format="%d/%m/%Y %H:%M:%S", errors="coerce",
    )
    dfy.insert(0, schema.KEY_COL, range(1, len(dfy) + 1))

    if not len(dfX) == len(dfy) == schema.EXPECTED_WAFERS:
        raise ValueError(
            f"Expected {schema.EXPECTED_WAFERS} wafers, "
            f"got {len(dfX)} readings / {len(dfy)} labels"
        )
    if dfX.shape[1] != schema.N_SENSORS + 1:
        raise ValueError(f"Expected {schema.N_SENSORS} sensors, got {dfX.shape[1] - 1}")
    if dfy[schema.TIME_COL].isna().any():
        raise ValueError("Unparseable timestamps in secom_labels.data")

    logger.info("loaded %d wafers x %d sensors", len(dfX), dfX.shape[1] - 1)
    return dfX, dfy


def build_registry(dfX: pd.DataFrame) -> pd.DataFrame:
    """Register raw columns. Status changes and engineered columns are
    registered later by the modules that create them."""
    sensors = [c for c in dfX.columns if c != schema.KEY_COL]
    rows = [
        {"column_name": schema.KEY_COL, "role": schema.Role.KEY.value,
         "status": schema.Status.ACTIVE.value, "col_index": 0, "missing_pct": 0.0,
         "derived_from": None, "notes": "row order in raw file is chronological"},
        {"column_name": schema.TARGET_COL, "role": schema.Role.TARGET.value,
         "status": schema.Status.ACTIVE.value, "col_index": None, "missing_pct": 0.0,
         "derived_from": None, "notes": "raw -1/1; encode downstream"},
        {"column_name": schema.TIME_COL, "role": schema.Role.METADATA.value,
         "status": schema.Status.ACTIVE.value, "col_index": None, "missing_pct": 0.0,
         "derived_from": None, "notes": None},
    ]
    miss = dfX[sensors].isna().mean()
    rows += [
        {"column_name": s, "role": schema.Role.FEATURE_RAW.value,
         "status": schema.Status.ACTIVE.value, "col_index": int(s[1:]),
         "missing_pct": float(miss[s]), "derived_from": None, "notes": None}
        for s in sensors
    ]
    return pd.DataFrame(rows)


def insert_data(dfX, dfy, registry, engine: Engine) -> None:
    log = pd.DataFrame([
        {"load_ts": datetime.now(timezone.utc), "source_file": "secom.data",
         "source_sha256": file_sha256(DATA_RAW / "secom.data"),
         "table_name": "sensor_readings", "rows_inserted": len(dfX),
         "git_sha": git_sha()},
        {"load_ts": datetime.now(timezone.utc), "source_file": "secom_labels.data",
         "source_sha256": file_sha256(DATA_RAW / "secom_labels.data"),
         "table_name": "wafer_labels", "rows_inserted": len(dfy),
         "git_sha": git_sha()},
    ])
    with engine.begin() as conn:  # one transaction: all-or-nothing
        for table in ("sensor_readings", "wafer_labels", "column_registry"):
            conn.execute(text(f"DELETE FROM {table}"))
        dfX.to_sql("sensor_readings", conn, if_exists="append", index=False)
        dfy.to_sql("wafer_labels", conn, if_exists="append", index=False)
        registry.to_sql("column_registry", conn, if_exists="append", index=False)
        log.to_sql("ingestion_log", conn, if_exists="append", index=False)
    logger.info(
        "ingested %d readings, %d labels, %d registry rows",
        len(dfX), len(dfy), len(registry),
    )


def main() -> None:
    global logger
    logger = setup_logging(logfile=LOGS / "db_ingest.log")
    logger.info("[db_ingest] start")
    engine = get_engine()
    setup_db(engine)
    dfX, dfy = load_data()
    registry = build_registry(dfX)
    insert_data(dfX, dfy, registry, engine)
    logger.info("[db_ingest] done")


if __name__ == "__main__":
    main()