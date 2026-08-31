"""Extract the wide frame via SQL.

Always the full frame; the split label comes from config.CUTOFF
(None pre-EDA -> every row 'unassigned'). Thin by design: no cleaning,
no label encoding, no feature work — those live downstream.
"""
import logging

import pandas as pd
from sqlalchemy import Engine

from semcon import schema
from semcon.config import CUTOFF
from semcon.db import get_engine, register_columns, run_query
from semcon.paths import LOGS
from semcon.utils import setup_logging

logger = logging.getLogger("semcon")


def extract(
    engine: Engine,
    start: str = "1970-01-01",
    end: str = "2100-01-01",
    cutoff=CUTOFF,
) -> pd.DataFrame:
    
    df = run_query("extract_wafers", engine, start=start, end=end, cutoff=cutoff)
    df[schema.TIME_COL] = pd.to_datetime(df[schema.TIME_COL], format="ISO8601")
    
    if cutoff is not None:
        n_clash = int((df[schema.TIME_COL] == pd.Timestamp(cutoff)).sum())
        if n_clash:
            raise ValueError(
                f"{n_clash} wafers timestamped exactly at cutoff {cutoff}; "
                "BETWEEN is inclusive — move the cutoff between wafers"
            )

    register_columns(
        [{
            "column_name": schema.SPLIT_COL,
            "role": schema.Role.METADATA.value,
            "status": schema.Status.ACTIVE.value,
            "col_index": None,
            "missing_pct": 0.0,
            "derived_from": schema.TIME_COL,
            "notes": "cv/holdout label from config.CUTOFF",
        }],
        engine,
    )
    return df


def main() -> None:
    global logger
    logger = setup_logging(logfile=LOGS / "extract.log")
    logger.info("[extract] start")
    engine = get_engine()
    df = extract(engine)
    logger.info("[extract] frame %s; split counts: %s",
                df.shape, df[schema.SPLIT_COL].value_counts().to_dict())
    logger.info("[extract] done")


if __name__ == "__main__":
    main()