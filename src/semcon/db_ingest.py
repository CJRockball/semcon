import pandas as pd
import logging
from pathlib import Path

from sqlalchemy import (
    Table, 
    Column, 
    Integer, Float, String, Boolean, DateTime, Engine,
    MetaData, 
    create_engine,
    text,
)
from sqlalchemy.exc import SQLAlchemyError
import hashlib, subprocess
from datetime import datetime, timezone

from semcon.paths import DATA_RAW, LOGS, ROOT
from semcon.utils import setup_logging
from semcon.db import get_engine

logger = logging.getLogger("semcon")

def file_sha256(path:Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()

def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def setup_db(engine:Engine):
    """ Setup database with tables
    
    """
    
    metadata = MetaData()
    sensor_readings = Table(
        "sensor_readings", metadata,
        Column("wafer_id", Integer, primary_key=True),
        *[Column(f"s{i:03d}", Float) for i in range(1, 591)],
    )

    wafer_labels = Table(
        'wafer_labels', metadata,
        Column("wafer_id", Integer, primary_key=True),
        Column('target', Integer),
        Column('timestamp', DateTime)
    )

    sensor_catalog = Table(
        'sensor_catalog', metadata,
        Column('sensor_id', String, primary_key=True),
        Column('col_index', Integer),
        Column('missing_pct', Float),
        Column('in_stable_29', Boolean),
        Column('notes', String) 
    )
    
    ingestion_log = Table(
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
    return

def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """ Load original data format return a feature dataframe and a label dataframe
    
    """
    
    try: 
        dfX = pd.read_csv(DATA_RAW / 'secom.data',
                        sep=r"\s+",
                        header=None,
                        na_values=['NaN'],
                        )
        dfX.columns = [f"s{i+1:03d}" for i in range(590)]
        dfX = dfX.copy()
        dfX.insert(0, "wafer_id", range(1, len(dfX) + 1))


        dfy = pd.read_csv(DATA_RAW / 'secom_labels.data',
                        sep=r'\s+',
                        header=None,
                        names=['target', 'timestamp']
                        )

        dfy['timestamp'] = pd.to_datetime(
                            dfy['timestamp'],
                            format="%d/%m/%Y %H:%M:%S",
                            errors='coerce',
        )

        dfy.insert(0, "wafer_id", range(1, len(dfy) + 1))

    except Exception as e:
        logger.exception(f'An unexpected error occurred: {e}')

    print(f'Data shape: {dfX.shape}')
    assert len(dfX) == len(dfy) == 1567
    assert dfX.shape[1] == 591

    return dfX, dfy


def build_catalog(dfX: pd.DataFrame) -> pd.DataFrame:
    sensors = [c for c in dfX.columns if c != "wafer_id"]
    return pd.DataFrame({
        "sensor_id": sensors,
        "col_index": [int(s[1:]) for s in sensors],
        "missing_pct": dfX[sensors].isna().mean().to_numpy(),
        "in_stable_29": None,   # unknown until selection runs; False would lie
        "notes": None,
    })
    

def insert_data(dfX:pd.DataFrame, dfy:pd.DataFrame, engine:Engine):
    catalog = build_catalog(dfX)
    log = pd.DataFrame([
        {"load_ts": datetime.now(timezone.utc), "source_file": "secom.data",
         "source_sha256": file_sha256(DATA_RAW / "secom.data"),
         "table_name": "sensor_readings", "rows_inserted": len(dfX), "git_sha": git_sha()},
        {"load_ts": datetime.now(timezone.utc), "source_file": "secom_labels.data",
         "source_sha256": file_sha256(DATA_RAW / "secom_labels.data"),
         "table_name": "wafer_labels", "rows_inserted": len(dfy), "git_sha": git_sha()},
    ])
    with engine.begin() as conn:  # one transaction: all-or-nothing
        conn.execute(text("DELETE FROM sensor_readings"))
        conn.execute(text("DELETE FROM wafer_labels"))
        conn.execute(text("DELETE FROM sensor_catalog"))
        dfX.to_sql("sensor_readings", conn, if_exists="append", index=False)
        dfy.to_sql("wafer_labels", conn, if_exists="append", index=False)
        catalog.to_sql("sensor_catalog", conn, if_exists="append", index=False)
        log.to_sql("ingestion_log", conn, if_exists="append", index=False)


def main():
        
    logger = setup_logging(logfile=LOGS / "ml.log")
    logger.info('[ingest.py] Start data pipeline')

    engine = get_engine()

    setup_db(engine)    
    dfX, dfy = load_data()
    insert_data(dfX, dfy, engine)


if __name__ == "__main__":
    main()