from __future__ import annotations

import logging

import matplotlib

matplotlib.use("Agg")  # artifacts over inline display - files are the product

from semcon import schema
from semcon.db import get_engine
from semcon.extract import extract

logger = logging.getLogger("semcon")


def load_data():
    """Wide frame from the DB, sorted once; split positions are computed here.

    Post-migration: extract() returns raw -1/1 target; recoded to 0/1 once.
    """
    engine = get_engine()
    df = extract(engine).sort_values(schema.TIME_COL).reset_index(drop=True)

    df_date = df[["wafer_id", "timestamp"]]

    print(df_date.iloc[1305:1315])

    return


if __name__ == "__main__":
    load_data()
