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


# %%
