#%%

from semcon.db import get_engine, run_query
from semcon.extract import extract
import pandas as pd
import numpy as np
from semcon.paths import DATA_PROCESSED
from semcon.schema import TIME_COL
dfX = pd.read_parquet(DATA_PROCESSED / "dfX_raw.parquet")  # explore.py's current CSV-based load, unmodified
dfy = pd.read_parquet(DATA_PROCESSED / "dfy_v1.parquet")

# df = run_query("extract_wafers", get_engine(),
#                start="1970-01-01", end="2100-01-01", cutoff=None)
# df[TIME_COL] = pd.to_datetime(df[TIME_COL], format="ISO8601")
# df = df.sort_values("wafer_id").reset_index(drop=True)
engine = get_engine()
df = extract(engine)

#%%

# display(pd.to_datetime(dfy["timestamp"]))
# display(df['timestamp'])

#%%
sensors = [f"s{i:03d}" for i in range(1, 591)]
assert np.array_equal(df[sensors].to_numpy(), dfX.to_numpy(), equal_nan=True)
assert np.array_equal(
    df["target"].eq(1).astype("int8").to_numpy(),   # raw -1/1 -> fail flag
    dfy["target"].to_numpy(),
)
assert df["target"].eq(1).sum() == 104              # the known fail count
old_ts = pd.to_datetime(dfy["timestamp"])
assert old_ts.equals(df["timestamp"])

# %%

