"""EDA and column retirement — reads from the database, writes no data files.

Roles after the SQL migration:
  1. extract() the wide frame (raw -1/1 target, split labels, all 590 sensors)
  2. assess column quality against configured rules (pure, testable)
  3. apply verdicts to column_registry as status flips, one reason per rule
  4. write EDA evidence (diagnostic table, summary, dataset card) — docs, not data

The frame never loses columns mid-pipeline: retirement is a registry status
change, and downstream consumers build X via feature_columns().

Legacy note: dfX_raw.parquet / dfX_v1.parquet / dfy_v1.parquet are frozen
historical artifacts of the flat-file pipeline. Nothing writes them now.

Leakage note: the NZV enrichment rescue uses the target (deviant fail rate
vs base rate). Statistics are computed on the full frame until
config.CUTOFF is set; then compute on the CV pool and apply globally —
the call site changes, the rules stay.
"""

import logging
from datetime import datetime

import numpy as np
import pandas as pd

from semcon import schema
from semcon.config import (
    CV_LIMIT,
    DEVIANT_FAIL_ENRICHMENT,
    DOMINANT_FRAC,
    FEATURE_CORR,
    FEW_UNIQUE,
    MIN_DEVIANT_N,
    NAN_FRAC_MAX,
)
from semcon.db import (
    assert_schema,
    export_registry_csv,
    feature_columns,
    get_engine,
    load_registry,
    retire_columns,
)
from semcon.extract import extract
from semcon.paths import ARTIFACTS, LOGS
from semcon.tracking import write_dataset_card
from semcon.utils import setup_logging

logger = logging.getLogger("semcon")


def assess_quality(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Assess sensor columns against the quality rules. Pure — no I/O.

    Rule order is load-bearing: it reproduces the legacy drop_basic()
    pipeline exactly (constant -> high-NaN -> NZV -> correlated), so the
    retired set matches the historical 590 -> 257 run.

    Returns (diagnostic table indexed by feature, {rule: [columns]}).
    """
    sensors = [c for c in df.columns if c.startswith(schema.SENSOR_PREFIX) and len(c) == 4]
    fail = df[schema.TARGET_COL].eq(1)
    fail_rate = float(fail.mean())

    # full stats table — EDA evidence for every sensor, verdict filled later
    rows = []
    for col in sensors:
        s = df[col]
        vc = s.value_counts(dropna=True)
        if len(vc):
            dom_val, dom_cnt = vc.index[0], vc.iloc[0]
            dom_frac = dom_cnt / s.notna().sum()
        else:
            dom_val, dom_frac = np.nan, np.nan
        mean, std = s.mean(), s.std()
        cv = abs(std / mean) if pd.notna(mean) and abs(mean) > 1e-12 else np.nan
        deviant = s.notna() & (s != dom_val)
        n_dev = int(deviant.sum())
        dev_fail_rate = fail[deviant].mean() if n_dev > 0 else np.nan
        rows.append(
            {
                "feature": col,
                "n_unique": s.nunique(),
                "dom_frac": dom_frac,
                "cv": cv,
                "missing_rate": s.isna().mean(),
                "n_deviant": n_dev,
                "dev_fail_rate": dev_fail_rate,
                "enrichment": dev_fail_rate / fail_rate if n_dev > 0 else np.nan,
            }
        )
    diag = pd.DataFrame(rows).set_index("feature")

    # rule 1: constants
    n_unique = df[sensors].nunique()
    constants = n_unique[n_unique == 1].index.tolist()

    # rule 2: high missing
    stage2 = [c for c in sensors if c not in set(constants)]
    miss = df[stage2].isna().mean()
    high_nan = miss[miss > NAN_FRAC_MAX].index.tolist()

    # rule 3: near-zero variance, with target-enrichment rescue
    stage3 = [c for c in stage2 if c not in set(high_nan)]
    d3 = diag.loc[stage3]
    flagged = d3[
        (d3["dom_frac"] > DOMINANT_FRAC) | (d3["cv"] < CV_LIMIT) | (d3["n_unique"] < FEW_UNIQUE)
    ]
    rescued = flagged[
        (flagged["n_deviant"] >= MIN_DEVIANT_N) & (flagged["enrichment"] >= DEVIANT_FAIL_ENRICHMENT)
    ]
    nzv = flagged.index.difference(rescued.index).tolist()

    # rule 4: correlated pairs — drop the more-missing member
    stage4 = [c for c in stage3 if c not in set(nzv)]
    corr = df[stage4].corr().unstack()
    pairs = corr[corr.index.get_level_values(0) < corr.index.get_level_values(1)]
    abs_pairs = pairs.abs().sort_values(ascending=False)
    to_drop = set()
    for a, b in abs_pairs[abs_pairs > FEATURE_CORR].index:
        if a not in to_drop and b not in to_drop:
            m = df[[a, b]].isna().mean()
            to_drop.add(m.idxmax())
    correlated = sorted(to_drop)

    verdicts = {
        "constant: n_unique == 1": constants,
        f"missing: NaN fraction > {NAN_FRAC_MAX}": high_nan,
        (
            f"near-zero variance: dom_frac > {DOMINANT_FRAC} or cv < {CV_LIMIT} "
            f"or n_unique < {FEW_UNIQUE}; deviants not enriched "
            f"(rescue needs n_deviant >= {MIN_DEVIANT_N} and "
            f"enrichment >= {DEVIANT_FAIL_ENRICHMENT})"
        ): nzv,
        (
            f"correlated: |r| > {FEATURE_CORR}; dropped the more-missing member of the pair"
        ): correlated,
    }
    return diag, verdicts


def apply_verdicts(verdicts: dict[str, list[str]], engine) -> int:
    """Flip registry status per rule, so every exclusion carries its reason."""
    total = 0
    for reason, cols in verdicts.items():
        if cols:
            retire_columns(cols, reason, engine)
            total += len(cols)
    return total


def write_eda_evidence(
    diag: pd.DataFrame, verdicts: dict[str, list[str]], n_active: int, engine
) -> None:
    out = ARTIFACTS / f"eda_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)

    verdict_col = pd.Series("keep", index=diag.index)
    for rule, cols in reversed(list(verdicts.items())):
        verdict_col.loc[cols] = rule  # earlier rules win (precedence)
    diag = diag.assign(verdict=verdict_col)

    diag.to_parquet(out / "diagnostic.parquet")
    diag.to_csv(out / "diagnostic.csv")
    write_dataset_card(out / "diagnostic.parquet", diag)

    lines = [
        f"# EDA summary — {datetime.now():%Y-%m-%d %H:%M}",
        "",
        "Rules and thresholds live in config.py (decisions, not constants).",
        "",
        "| rule | columns retired |",
        "|---|---|",
    ]
    for rule, cols in verdicts.items():
        lines.append(f"| {rule} | {len(cols)} |")
    lines += [
        "",
        f"Active sensors after retirement: **{n_active}** (legacy flat-file run: 257 — must match)",
        "",
        "Supervised-rule note: the NZV enrichment rescue uses the target. "
        "Computed on the full frame until config.CUTOFF is set; then on the "
        "CV pool only, applied globally.",
        "",
        "Data artifacts written by this run: none. Registry flips applied; "
        "this folder is documentation-grade evidence.",
    ]
    (out / "eda_summary.md").write_text("\n".join(lines) + "\n")
    logger.info("[explore] EDA evidence -> %s", out)


def main() -> None:
    global logger
    logger = setup_logging(logfile=LOGS / "explore.log")
    logger.info("[explore] start")

    engine = get_engine()
    assert_schema(engine)
    df = extract(engine)

    diag, verdicts = assess_quality(df)
    retired = apply_verdicts(verdicts, engine)
    active = feature_columns(load_registry(engine))
    logger.info(f"[explore] retired {retired} sensors, {len(active)} active (legacy run: 257)")

    write_eda_evidence(diag, verdicts, len(active), engine)
    export_registry_csv(engine)
    logger.info("[explore] done")


if __name__ == "__main__":
    main()
