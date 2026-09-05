"""Feature engineering — pure functions over the extracted wide frame.

Reads nothing from disk, writes nothing to disk. Input is the wide frame
from extract(); output is the same frame with f_* columns appended plus
the registry rows describing them. Convention: whoever creates a column
registers it.

The four features are missingness/data-quality signals computed from raw
sensors — including registry-excluded ones (clique-14 members are retired
by the >50% NaN rule). Excluded-from-X does not mean excluded-from-lineage;
the wide frame exists precisely so these features can still be built.
"""

import logging

import pandas as pd

from semcon import schema
from semcon.config import BLOCK5_FIRST, CLIQUE_14, CLIQUE_23
from semcon.db import assert_schema, get_engine, register_columns
from semcon.extract import extract
from semcon.paths import LOGS
from semcon.utils import setup_logging

logger = logging.getLogger("semcon")

# data expectations from the flat-file EDA (Section 3.6 anchors) — contract
# checks, not parameters; if these change, the data changed, investigate
EXPECTED_CLQ14 = 794
EXPECTED_CLQ23 = 715


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """Append the four engineered features; return (frame, registry rows).

    Clique/block membership comes from config as s-names (converted from
    legacy 0-based indices at the config site, with provenance comments).
    """
    sensors = [c for c in df.columns if c.startswith(schema.SENSOR_PREFIX) and len(c) == 4]
    block5 = [f"{schema.SENSOR_PREFIX}{i:03d}" for i in range(BLOCK5_FIRST, schema.N_SENSORS + 1)]

    feats = pd.DataFrame(index=df.index)
    feats["f_miss_clq14"] = df[CLIQUE_14].isna().any(axis=1).astype("int8")
    feats["f_miss_clq23"] = df[CLIQUE_23].isna().any(axis=1).astype("int8")
    feats["f_miss_block5"] = df[block5].isna().any(axis=1).astype("int8")
    feats["f_row_missing_rate"] = df[sensors].isna().mean(axis=1).astype("float32")

    if int(feats["f_miss_clq14"].sum()) != EXPECTED_CLQ14:
        raise ValueError(
            f"f_miss_clq14 sums to {feats['f_miss_clq14'].sum()}, "
            f"expected {EXPECTED_CLQ14} — clique membership or data changed"
        )
    if int(feats["f_miss_clq23"].sum()) != EXPECTED_CLQ23:
        raise ValueError(
            f"f_miss_clq23 sums to {feats['f_miss_clq23'].sum()}, "
            f"expected {EXPECTED_CLQ23} — clique membership or data changed"
        )

    rows = [
        {
            "column_name": "f_miss_clq14",
            "role": schema.Role.FEATURE_ENG.value,
            "status": schema.Status.ACTIVE.value,
            "col_index": None,
            "missing_pct": 0.0,
            "derived_from": ",".join(CLIQUE_14),
            "notes": "any-NaN over clique 14; legacy name miss_clq14 "
            "(legacy idx 72,73,345,346; OR 0.49, p=0.0008)",
        },
        {
            "column_name": "f_miss_clq23",
            "role": schema.Role.FEATURE_ENG.value,
            "status": schema.Status.ACTIVE.value,
            "col_index": None,
            "missing_pct": 0.0,
            "derived_from": ",".join(CLIQUE_23),
            "notes": "any-NaN over clique 23; legacy name miss_clq23 "
            "(legacy idx 112,247,385,519; OR 0.53, p=0.0031)",
        },
        {
            "column_name": "f_miss_block5",
            "role": schema.Role.FEATURE_ENG.value,
            "status": schema.Status.ACTIVE.value,
            "col_index": None,
            "missing_pct": 0.0,
            "derived_from": f"s{BLOCK5_FIRST:03d}-s{schema.N_SENSORS:03d}",
            "notes": "block-5 dropout flag, label-free probe for the "
            "time-blocked holdout; legacy name miss_block5",
        },
        {
            "column_name": "f_row_missing_rate",
            "role": schema.Role.FEATURE_ENG.value,
            "status": schema.Status.ACTIVE.value,
            "col_index": None,
            "missing_pct": 0.0,
            "derived_from": "all 590 sensors",
            "notes": "per-wafer missing share; legacy name row_missing_rate",
        },
    ]

    out = pd.concat([df, feats], axis=1)
    logger.info(
        "[feature_eng] %d in -> %d out | clq14=%d clq23=%d block5=%d",
        df.shape[1],
        out.shape[1],
        feats["f_miss_clq14"].sum(),
        feats["f_miss_clq23"].sum(),
        feats["f_miss_block5"].sum(),
    )
    return out, rows


def main() -> None:
    global logger
    logger = setup_logging(logfile=LOGS / "feature_eng.log")
    logger.info("[feature_eng] start")

    engine = get_engine()
    assert_schema(engine)
    df = extract(engine)
    df_out, rows = build_features(df)
    register_columns(rows, engine)

    logger.info("[feature_eng] done — no files written by design")


if __name__ == "__main__":
    main()
