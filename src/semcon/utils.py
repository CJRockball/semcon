#%%
# build_features.py
import logging
import sys
from pathlib import Path


def setup_logging(level: int = logging.INFO,
                  logfile: str | Path | None = None) -> logging.Logger:
    """Idempotent project logger.

    First call configures console output; every call returns the same logger.
    A new logfile argument is honored on ANY call — its FileHandler is added
    unless a handler for that exact path already exists.
    """
    logger = logging.getLogger('semcon')
    logger.setLevel(level)
    logger.propagate = False          # don't double-print through root

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")

    has_console = any(
        isinstance(h, logging.StreamHandler)
        and not isinstance(h, logging.FileHandler)
        for h in logger.handlers
    )
    if not has_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

    if logfile is not None:
        logfile = Path(logfile)
        logfile.parent.mkdir(parents=True, exist_ok=True)
        existing = {
            Path(h.baseFilename)
            for h in logger.handlers
            if isinstance(h, logging.FileHandler)
        }
        if logfile not in existing:
            fh = logging.FileHandler(logfile, mode="a", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    return logger
