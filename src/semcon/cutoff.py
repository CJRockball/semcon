from __future__ import annotations

import argparse
import logging
import warnings

import matplotlib
matplotlib.use("Agg")  # artifacts over inline display - files are the product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from semcon import tracking
from semcon.db import get_engine, data_fingerprint
from semcon.extract import extract
from semcon.paths import ARTIFACTS, LOGS
from semcon.utils import setup_logging
from semcon.config import CLIQUE_14, CLIQUE_23, BLOCK5_FIRST, load_config

from semcon import schema

logger = logging.getLogger("semcon")



def load_data():
    """Wide frame from the DB, sorted once; split positions are computed here.

    Post-migration: extract() returns raw -1/1 target; recoded to 0/1 once.
    """
    engine = get_engine()
    df = extract(engine).sort_values(schema.TIME_COL).reset_index(drop=True)

    df_date = df['timestamp']
    
    print(df_date.iloc[1309])

    return 

if __name__ == "__main__":
    load_data()