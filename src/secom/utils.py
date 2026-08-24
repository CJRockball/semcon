#%%
# build_features.py
import logging
import sys
from pathlib import Path
import pandas as pd
from secom.paths import LOGS

def setup_logging(level: int = logging.INFO, log_file: str = "eda.log") -> logging.Logger:
    """Same logger as the EDA notebook; appends to the shared log file."""
    logger = logging.getLogger("secom")          # same name => same logger
    logger.setLevel(level)
    if logger.handlers:                          # idempotent on re-import
        return logger
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(log_file, mode="a", encoding="utf-8")   # append, not overwrite
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.propagate = False
    return logger

