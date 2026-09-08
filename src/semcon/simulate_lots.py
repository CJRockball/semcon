"""Simulate incoming lots: bootstrap-resample real wafers into dated batches.

Batch A clean jitter, B mean shift on one in-model sensor, C clique-14
dropout. Appends to bronze, writes each batch payload + sha256 to data/sim/
for provenance, logs to ingestion_log, and prints the semcon-score command
per batch.

Determinism contract: outputs are a pure function of (seed, source data,
parameters) — `make clean && make && semcon-simulate` restores identical
batches: same wafer_ids, timestamps, and payloads. Wafer ids come from fixed
slots (batch A at 1,000,001, ...), so re-running on a non-cleaned database
fails loudly on the wafer_id PK instead of silently duplicating lots.

True labels ARE stored in wafer_labels — for the evaluation panel only.
The scorer never reads them.

WARNING: `semcon-ingest` wipes and reloads bronze from the raw files; running
it after simulation deletes the simulated lots. Simulation is append-only,
and always the last write to bronze.

Entry point: semcon-simulate = semcon.simulate_lots:main
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import subprocess
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from semcon import schema
from semcon.config import EXCLUDE_AFTER
from semcon.db import get_engine
from semcon.extract import extract
from semcon.feature_eng import CLIQUE_14
from semcon.paths import LOGS
from semcon.utils import setup_logging

logger = logging.getLogger("semcon")

SIM_DIR = LOGS.parent / "data" / "sim"
SENSORS = [f"{schema.SENSOR_PREFIX}{i:03d}" for i in range(1, schema.N_SENSORS + 1)]
BATCH_GAP_DAYS = 7
ID_BASE = 1_000_000  # simulated ids live far above any real wafer_id
ID_SLOT = 10_000  # per-batch id block


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()[:7]
    except Exception:
        return "unknown"


def jitter(X: pd.DataFrame, rng: np.random.Generator, frac: float = 0.01) -> pd.DataFrame:
    """Additive gaussian noise, 1% of per-column std, observed values only."""
    noise = pd.DataFrame(rng.normal(0.0, frac, size=X.shape), columns=X.columns) * X.std()
    return (X + noise).where(X.notna())  # never fill a NaN — missingness is signal


def shift(X: pd.DataFrame, sensor: str, sigma: float) -> pd.DataFrame:
    X = X.copy()
    X[sensor] = X[sensor] + sigma * X[sensor].std()
    return X


def dropout(X: pd.DataFrame, rng: np.random.Generator, frac: float) -> pd.DataFrame:
    X = X.copy()
    rows = rng.random(len(X)) < frac
    X.loc[rows, CLIQUE_14] = np.nan
    return X


def append_lots(
    engine, name: str, slot: int, X: pd.DataFrame, y: np.ndarray, ts: pd.DatetimeIndex
) -> np.ndarray:
    ids = ID_BASE + slot * ID_SLOT + np.arange(len(X))
    dfX = X.copy()
    dfX.insert(0, schema.KEY_COL, ids)
    dfy = pd.DataFrame({schema.KEY_COL: ids, schema.TARGET_COL: y, schema.TIME_COL: ts})
    SIM_DIR.mkdir(parents=True, exist_ok=True)
    payload = SIM_DIR / f"{name}.parquet"
    pd.concat([dfy.reset_index(drop=True), X.reset_index(drop=True)], axis=1).to_parquet(
        payload, index=False
    )
    digest = hashlib.sha256(payload.read_bytes()).hexdigest()
    now = datetime.now()
    log = pd.DataFrame(
        [
            {
                "load_ts": now,
                "source_file": f"sim:{name}",
                "source_sha256": digest,
                "table_name": "sensor_readings",
                "rows_inserted": len(dfX),
                "git_sha": git_sha(),
            },
            {
                "load_ts": now,
                "source_file": f"sim:{name}",
                "source_sha256": digest,
                "table_name": "wafer_labels",
                "rows_inserted": len(dfy),
                "git_sha": git_sha(),
            },
        ]
    )
    try:
        with engine.begin() as conn:  # one transaction, all-or-nothing
            dfX.to_sql("sensor_readings", conn, if_exists="append", index=False)
            dfy.to_sql("wafer_labels", conn, if_exists="append", index=False)
            log.to_sql("ingestion_log", conn, if_exists="append", index=False)
    except Exception as exc:
        raise RuntimeError(
            f"batch {name!r} failed — already ingested? the wafer_id PK is the guard "
            f"(simulated ids start at {ID_BASE})"
        ) from exc
    return ids


def main() -> None:
    global logger
    logger = setup_logging(logfile=LOGS / "simulate.log")
    ap = argparse.ArgumentParser(description="Simulate three incoming lots and append to bronze.")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=100, help="wafers per batch")
    ap.add_argument("--start", default="2026-01-05", help="batch A start date")
    ap.add_argument("--shift-sensor", default="s060", help="in-model sensor for batch B")
    ap.add_argument("--shift-sigma", type=float, default=2.0)
    ap.add_argument("--clique-frac", type=float, default=0.3, help="batch C dropout share")
    args = ap.parse_args()

    base = pd.Timestamp(args.start)
    if base <= pd.Timestamp(EXCLUDE_AFTER):
        raise ValueError(f"batch start {base} must be after EXCLUDE_AFTER ({EXCLUDE_AFTER})")

    logger.info("[simulate] start | seed=%d n=%d start=%s", args.seed, args.n, args.start)
    engine = get_engine()
    top = int(
        pd.read_sql(f"SELECT MAX({schema.KEY_COL}) AS m FROM sensor_readings", engine)["m"].iloc[0]
    )
    if top >= ID_BASE:
        raise ValueError(
            f"sensor_readings already contains wafer_id {top} >= {ID_BASE}; "
            "simulated lots are present (or real data grew) — clean first or abort"
        )
    pool = extract(engine, cutoff=None, exclude_after=None)  # full history, 'unassigned'
    rng = np.random.default_rng(args.seed)

    transforms = {
        "batch_a_clean": lambda X: X,
        "batch_b_shift": lambda X: shift(X, args.shift_sensor, args.shift_sigma),
        "batch_c_dropout": lambda X: dropout(X, rng, args.clique_frac),
    }
    manifest = []
    commands = []
    for slot, (name, fn) in enumerate(transforms.items()):
        idx = rng.integers(0, len(pool), size=args.n)
        X = jitter(pool[SENSORS].iloc[idx].reset_index(drop=True), rng)
        X = fn(X)
        y = pool[schema.TARGET_COL].iloc[idx].to_numpy()
        ts = pd.date_range(
            base + pd.Timedelta(days=slot * BATCH_GAP_DAYS), periods=args.n, freq="h"
        )
        ids = append_lots(engine, name, slot, X, y, ts)
        manifest.append(
            {
                "batch": name,
                "n": args.n,
                "wafer_id_first": int(ids[0]),
                "wafer_id_last": int(ids[-1]),
                "start": f"{ts[0]:%Y-%m-%d %H:%M}",
                "end": f"{ts[-1]:%Y-%m-%d %H:%M}",
            }
        )
        commands.append(
            f'semcon-score --start "{ts[0]:%Y-%m-%d %H:%M}" '
            f'--end "{ts[-1]:%Y-%m-%d %H:%M}" --label {name}'
        )
        logger.info("[simulate] %s ingested: %d wafers, ids %d-%d", name, args.n, ids[0], ids[-1])
    pd.DataFrame(manifest).to_csv(SIM_DIR / "batches_manifest.csv", index=False)
    print("\nscore commands:")
    for cmd in commands:
        print(f"  {cmd}")
    logger.info("[simulate] done -> %s", SIM_DIR)


if __name__ == "__main__":
    main()
