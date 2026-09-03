"""Validation gate for the extracted frame.

Owns three contracts, checked loudly, in one place:
1. Schema: key/metadata/target typed and valid; the s### sensor block
   complete and float64; split labels from the known vocabulary.
2. Expectations: row/column counts (graduated from db_ingest — ingest
   proves bytes arrived; validate proves the frame is analytically usable).
3. Drift: per-column missingness vs the latest gold snapshot's frozen
   registry — report-only; SPC owns the response, this module detects.

Also the single home of is_fail (target recode -1/1 -> 0/1), registered
in the column registry as role=target, derived_from=target. Consumers
never recode labels themselves.
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pandera.pandas as pa

from semcon import schema, tracking
from semcon.db import get_engine, register_columns
from semcon.extract import extract
from semcon.paths import DATA, LOGS
from semcon.utils import setup_logging

logger = logging.getLogger("semcon")

SPLIT_LABELS = frozenset({"cv", "holdout", "unassigned", "excluded"})
DRIFT_FLAG = 0.02  # missingness delta (fraction) worth a report line


# ---------------------------------------------------------------- label


def ensure_is_fail(df: pd.DataFrame, engine) -> pd.DataFrame:
    """Create-or-verify is_fail; register it. The only place this happens."""
    truth = df[schema.TARGET_COL].eq(1).astype("int8")
    if "is_fail" in df.columns:
        if not df["is_fail"].equals(truth):
            raise ValueError("is_fail exists but disagrees with target recode")
    else:
        df = df.assign(is_fail=truth)
    register_columns([{
    "column_name": "is_fail",
    "role": schema.Role.TARGET,
    "status": "active",
    "derived_from": schema.TARGET_COL,
    "notes": "recode -1/1 -> 0/1; single home: validate.py",
}], engine)
    return df


# ---------------------------------------------------------------- schema


def check_schema(df: pd.DataFrame) -> None:
    """Pandera for the structural columns; a sweep for the sensor block.

    590 numeric columns are the wrong tool for pandera — check the block
    by pattern and count, and reserve Column checks for load-bearing ones.
    """
    structural = pa.DataFrameSchema(
        {
            schema.KEY_COL: pa.Column(pa.Int64, nullable=False, unique=True),
            schema.TIME_COL: pa.Column("datetime64[ns]", nullable=False),
            schema.TARGET_COL: pa.Column(
                pa.Int64, pa.Check.isin([-1, 1]), nullable=False),
            "is_fail": pa.Column(pa.Int8, pa.Check.isin([0, 1]), nullable=False),
            "split": pa.Column(str, pa.Check.isin(sorted(SPLIT_LABELS)),
                               nullable=False),
        },
        strict=False,  # sensor/engineered columns pass through
    )
    structural.validate(df, lazy=True)  # collect all failures before raising

    sensors = [c for c in df.columns
               if re.fullmatch(rf"{schema.SENSOR_PREFIX}\d{{3}}", c)]
    if len(sensors) != schema.N_SENSORS:
        raise ValueError(f"sensor block: {len(sensors)} cols, "
                         f"expected {schema.N_SENSORS}")
    non_float = [c for c in sensors if df[c].dtype != np.float64]
    if non_float:
        raise ValueError(f"non-float64 sensors: {non_float[:5]}...")

    if len(df) != schema.EXPECTED_WAFERS:
        raise ValueError(f"rows: {len(df)}, expected {schema.EXPECTED_WAFERS}")


# ---------------------------------------------------------------- drift


def missingness_drift(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None]:
    """Per-column missingness now vs the latest gold snapshot's registry."""
    current = df.isna().mean().rename("missing_now").to_frame()
    snaps = sorted((DATA / "snapshots" / "gold").glob("*/registry.csv"))
    if not snaps:
        return current.assign(missing_snapshot=np.nan, delta=np.nan), None

    frozen = pd.read_csv(snaps[-1], index_col=0)
    frozen.index = frozen.index.astype(str)
    out = current.join(frozen[["missing_pct"]].rename(
        columns={"missing_pct": "missing_snapshot"}))
    out["delta"] = out["missing_now"] - out["missing_snapshot"]
    out["flag"] = out["delta"].abs() > DRIFT_FLAG
    return out.sort_values("delta", key=abs, ascending=False), snaps[-1].parent.name


# ---------------------------------------------------------------- main


def main(argv=None):
    p = argparse.ArgumentParser(description="Validate the extracted frame")
    p.add_argument("--note", default="")
    args = p.parse_args(argv)
    setup_logging(logfile=LOGS / "ml.log")
    logger.info("[validate] start")

    engine = get_engine()
    df = extract(engine)
    df = ensure_is_fail(df, engine)
    check_schema(df)

    drift, snapshot_ref = missingness_drift(df)
    n_flags = int(drift["flag"].sum()) if "flag" in drift else 0

    report = {
        "rows": int(len(df)),
        "cols": int(df.shape[1]),
        "fails": int(df["is_fail"].sum()),
        "split_counts": df["split"].value_counts().to_dict(),
        "snapshot_ref": snapshot_ref,
        "drift_flags": n_flags,
    }

    run_dir, meta = tracking.make_run(
        config={"script": "validate", "drift_flag": DRIFT_FLAG},
        run_name="validate", note=args.note)
    drift.to_csv(run_dir / "missingness_drift.csv")
    (run_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    tracking.append_index(run_dir, {"type": "validate", **report,
                                    "git_sha": meta["git_sha"]})

    logger.info(f"[validate] {report}")
    if n_flags:
        logger.warning(f"[validate] {n_flags} columns drifted past "
                       f"{DRIFT_FLAG} — see missingness_drift.csv")
    logger.info(f"[validate] done | artifacts -> {run_dir}")


if __name__ == "__main__":
    main()