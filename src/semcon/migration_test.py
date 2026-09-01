#%%

import pandas as pd
import numpy as np
from semcon.paths import DATA_PROCESSED
from semcon.schema import TIME_COL
from semcon.db import get_engine, load_registry, feature_columns
from semcon.extract import extract


#%%

dfX_raw = pd.read_parquet(DATA_PROCESSED / "dfX_raw.parquet")  # explore.py's current CSV-based load, unmodified
dfy = pd.read_parquet(DATA_PROCESSED / "dfy_v1.parquet")

engine = get_engine()
df_full = extract(engine)

#%%

# display(pd.to_datetime(dfy["timestamp"]))
# display(df['timestamp'])

#%%
sensors = [f"s{i:03d}" for i in range(1, 591)]
assert np.array_equal(df_full[sensors].to_numpy(), dfX_raw.to_numpy(), equal_nan=True)
assert np.array_equal(
    df_full["target"].eq(1).astype("int8").to_numpy(),   # raw -1/1 -> fail flag
    dfy["target"].to_numpy(),
)
assert df_full["target"].eq(1).sum() == 104              # the known fail count
old_ts = pd.to_datetime(dfy["timestamp"])
assert old_ts.equals(df_full["timestamp"])

# %%

dfX_v1 = pd.read_parquet(DATA_PROCESSED / 'dfX_v1.parquet')

active = feature_columns(load_registry(engine))       # 257 names — the registry answers
dfX_db_v1 = df_full[active]                                    # the "reduced" frame, built here

# %%

assert np.array_equal(dfX_db_v1.to_numpy(), dfX_v1.to_numpy(), equal_nan=True)


#%%
# === Model equivalence: pre-migration vs post-migration runs ===
# Same data, same splits, same seed -> OOF and holdout probabilities should
# match to float noise. Anything beyond ~1e-6 means real divergence.

import json

import numpy as np
import pandas as pd

from semcon.paths import ARTIFACTS

PAIRS = [
    ("20260829_102922_baseline", "20260901_144433_xgb_base"),  # no selection
    ("20260829_103106_sel_g025", "20260901_143904_xgb_sel"),   # fold-internal selection
]

LEGACY_ENG = {"miss_clq14": "f_miss_clq14", "miss_clq23": "f_miss_clq23",
              "miss_block5": "f_miss_block5", "row_missing_rate": "f_row_missing_rate"}


def to_new_name(n) -> str:
    """Map legacy feature names to s-/f_-names. Legacy sensors were 0-based."""
    n = str(n)
    if n.startswith("s") or n.startswith("f_"):
        return n
    if n in LEGACY_ENG:
        return LEGACY_ENG[n]
    return f"s{int(n) + 1:03d}"


def load_feature_names(run_dir) -> set:
    obj = json.loads((run_dir / "features.json").read_text())
    if isinstance(obj, dict):                       # tolerate {"features": [...]}
        obj = obj.get("features") or obj.get("selected") or next(iter(obj.values()))
    return {to_new_name(f) for f in obj}


reports = []
for old_name, new_name in PAIRS:
    old, new = ARTIFACTS / 'runs' / old_name, ARTIFACTS / 'runs' / new_name
    rep = {"pair": f"{old_name[-9:]} vs {new_name[-9:]}"}

    oof_old, oof_new = np.load(old / "oof_xgb1.npy"), np.load(new / "oof_xgb1.npy")
    assert oof_old.shape == oof_new.shape, f"OOF shape {oof_old.shape} vs {oof_new.shape}"
    rep["oof_max_diff"] = float(np.abs(oof_old - oof_new).max())

    p_old, p_new = np.load(old / "p_hold.npy"), np.load(new / "p_hold.npy")
    assert p_old.shape == p_new.shape, "holdout vector length changed"
    rep["p_hold_max_diff"] = float(np.abs(p_old - p_new).max())

    m_old = pd.read_csv(old / "cv_metrics_xgb1.csv", index_col=0)
    m_new = pd.read_csv(new / "cv_metrics_xgb1.csv", index_col=0)
    for col in ["aucpr", "rocauc", "brier"]:
        rep[f"{col}_mean_diff"] = round(float(m_old[col].mean() - m_new[col].mean()), 6)

    f_old, f_new = load_feature_names(old), load_feature_names(new)
    rep["features_match"] = f_old == f_new
    rep["only_old"] = sorted(f_old - f_new)[:5]
    rep["only_new"] = sorted(f_new - f_old)[:5]
    reports.append(rep)

summary = pd.DataFrame(reports).set_index("pair")
print(summary.to_string())

for rep in reports:
    assert rep["oof_max_diff"] < 1e-6, f"OOF diverged: {rep}"
    assert rep["p_hold_max_diff"] < 1e-6, f"holdout probs diverged: {rep}"
    assert rep["features_match"], f"feature sets differ: {rep['only_old']} / {rep['only_new']}"
print("model equivalence: PASS")

# %%