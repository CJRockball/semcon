#%%
import pandas as pd
import os

print(os.getcwd())
cols = pd.read_parquet("../../data/processed/dfX_v2.parquet").columns

print([c for c in cols if not c.isdigit()])


# %%
