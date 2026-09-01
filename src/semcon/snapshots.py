"""Gold snapshots — the immutable memory of what training saw.

Written by training code immediately before fitting; never by producers.
Each snapshot is a directory under data/snapshots/gold/{snapshot_id}/
holding matrix.parquet, its dataset card, a frozen registry.csv copy, and
the manifest.json sidecar. The run's config.json records snapshot_id,
closing the chain: raw sha -> ingestion_log -> snapshot -> run -> model.

Silver snapshots are skipped by design: bronze + SQL + pinned code
recomputes them deterministically; the fingerprint vouches.
"""
import hashlib
import io
import json
import logging
from datetime import datetime, timezone

import pandas as pd
from sqlalchemy import Engine

from semcon.db import data_fingerprint, load_registry
from semcon.db_ingest import git_sha
from semcon.paths import ROOT
from semcon.tracking import write_dataset_card

logger = logging.getLogger("semcon")

SNAPSHOTS = ROOT / "data" / "snapshots"


def write_gold_snapshot(
    df: pd.DataFrame, engine: Engine, config: dict | None = None
) -> str:
    """Persist the training matrix immutably; return the snapshot_id.

    config: the run's decision dict (cutoff, thresholds, seeds) — recorded
    verbatim, not hashed, so the manifest is readable without a decoder.
    """
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    payload = buf.getvalue()
    sha = hashlib.sha256(payload).hexdigest()

    snapshot_id = f"{datetime.now(timezone.utc):%Y%m%d_%H%M%S}_{sha[:8]}"
    out = SNAPSHOTS / "gold" / snapshot_id
    out.mkdir(parents=True, exist_ok=False)

    (out / "matrix.parquet").write_bytes(payload)
    write_dataset_card(out / "matrix.parquet", df)
    load_registry(engine).to_csv(out / "registry.csv", index=False)

    manifest = {
        "snapshot_id": snapshot_id,
        "layer": "gold",
        "created": ts if (ts := datetime.now(timezone.utc)) else None,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "git_sha": git_sha(),
        "config": config or {},
        "fingerprint": data_fingerprint(engine),
        "parquet_sha256": sha,
    }
    manifest["created"] = ts.isoformat()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    logger.info(
        "[snapshots] gold %s: %d rows x %d cols, sha256 %s",
        snapshot_id, df.shape[0], df.shape[1], sha[:16],
    )
    return snapshot_id