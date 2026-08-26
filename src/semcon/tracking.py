"""Lightweight run tracking: one folder per run, config + git + data state.

Deliberately not MLflow. For tens of runs, a folder per run plus an
append-only index is enough — every artifact stays diffable plain text
and there is no server to maintain.

Design rule: this module must stay project-agnostic (the only semcon
dependency is the ARTIFACTS default). tracking.py + paths.py + utils.py
are the reusable kernel for future projects.

Layout
------
artifacts/runs/
├── index.csv                      # one row per run, appended
└── 20260826_090000_sel_g025/
    ├── config.json                # args, constants, hyperparams, versions, git sha, note
    ├── dataset.json               # input identity: hashes, shape, target stats
    ├── splits.json                # positional train/holdout indices
    ├── features.json              # final feature list + selection stability
    ├── oof.npy
    ├── cv_metrics.parquet
    ├── summary_oof.csv / summary_hold.csv
    ├── pr_curve_holdout.png / conf_heatmap.png
    └── shap/ ...
"""
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from semcon.paths import ARTIFACTS

TRACKED_PKGS = ("numpy", "pandas", "scikit-learn", "xgboost")


def _git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def _pkg_versions() -> dict:
    out = {}
    for pkg in TRACKED_PKGS:
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = "not-installed"
    return out


def _slugify(s: str) -> str:
    """Filesystem-safe run name: lowercase, [a-z0-9._-], dash-separated."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s.strip()).strip("-").lower()
    return s or "run"


def hash_file(path: Path, algo: str = "sha256", chunk: int = 1 << 20) -> str:
    """Content hash of a file, streamed in chunks. Byte-identity: answers
    'did this file change?', not 'did the values change?' (parquet rewrites
    change bytes even for identical data — that is the intended semantics)."""
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def make_run(config: dict, run_name: str | None = None, note: str = "",
             runs_root: Path | None = None) -> tuple[Path, dict]:
    """Create artifacts/runs/<timestamp>_<name>/ and write config.json."""
    runs_root = runs_root or ARTIFACTS / "runs"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = runs_root / f"{ts}_{_slugify(run_name or 'run')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    meta = {
        "run_id": run_dir.name,
        "timestamp": ts,
        "git_sha": _git_sha(),
        "python": sys.version.split()[0],
        "packages": _pkg_versions(),
        "note": note,
        "config": config,
    }
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2, default=str))
    return run_dir, meta


def save_dataset_info(run_dir: Path, *, features_path: Path, target_path: Path,
                      dfX: pd.DataFrame, dfy: pd.DataFrame) -> dict:
    """Write dataset.json: input identity for this run.

    Three layers: content hash (did the file change), structure (shape plus
    a column-list hash, which catches same-count/different-membership), and
    target stats (fail count, prevalence).
    """
    cols = "|".join(map(str, dfX.columns))
    info = {
        "features_file": features_path.name,
        "features_sha256_16": hash_file(features_path)[:16],
        "n_rows": int(len(dfX)),
        "n_features": int(dfX.shape[1]),
        "columns_md5_16": hashlib.md5(cols.encode()).hexdigest()[:16],
        "target_file": target_path.name,
        "target_sha256_16": hash_file(target_path)[:16],
        "n_fails": int(dfy["target"].sum()),
        "prevalence": float(dfy["target"].mean()),
    }
    (run_dir / "dataset.json").write_text(json.dumps(info, indent=2))
    return info


def save_splits(run_dir: Path, df_train: pd.DataFrame,
                df_test: pd.DataFrame) -> None:
    """Persist the exact split. Positional indices are meaningful because
    load_data sorts by timestamp and resets the index."""
    payload = {
        "train_index": df_train.index.tolist(),
        "holdout_index": df_test.index.tolist(),
        "n_train": len(df_train),
        "n_holdout": len(df_test),
    }
    (run_dir / "splits.json").write_text(json.dumps(payload))


def save_features(run_dir: Path, feats: list,
                  stability: pd.Series | None = None) -> None:
    payload = {"n_features": len(feats), "features": [str(f) for f in feats]}
    if stability is not None:
        payload["stability"] = {str(k): float(v) for k, v in stability.items()}
    (run_dir / "features.json").write_text(json.dumps(payload, indent=2))


def append_index(run_dir: Path, metrics: dict,
                 index_file: Path | None = None) -> None:
    """Append one row to runs/index.csv — the cross-run comparison table.
    Deliberately dumb: whatever keys the caller puts in `metrics` become
    columns (run name, note, data hash, headline metrics)."""
    index_file = index_file or run_dir.parent / "index.csv"
    row = {"run_id": run_dir.name, **metrics}
    df = pd.DataFrame([row])
    df.to_csv(index_file, mode="a", header=not index_file.exists(), index=False)


def write_dataset_card(parquet_path: Path, df: pd.DataFrame,
                       target: pd.Series | None = None) -> Path:
    """Sidecar card, written by the PRODUCER at dataset creation."""
    info = {"file": parquet_path.name, "sha256_16": hash_file(parquet_path)[:16],
            "n_rows": len(df), "n_features": int(df.shape[1]),
            "columns_md5_16": hashlib.md5("|".join(map(str, df.columns)).encode()).hexdigest()[:16]}
    if target is not None:
        info.update(n_fails=int(target.sum()), prevalence=float(target.mean()))
    card = parquet_path.with_suffix(".dataset.json")     # dfX_v2.dataset.json
    card.write_text(json.dumps(info, indent=2))
    return card