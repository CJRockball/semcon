#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from secom.paths import DATA_RAW, DATA_PROCESSED, ARTIFACTS
from secom.utils import setup_logging

setup_logging()

#%%

dfX = pd.read_csv(DATA_RAW / 'secom.data',
                  sep=r"\s+",
                  header=None,
                  na_values=['NaN'],
                  dtype="float64",
                  )

dfy = pd.read_csv(DATA_RAW / 'secom_labels.data',
                  sep=r'\s+',
                  header=None,
                  names=['target', 'timestamp']
                  )

dfy['timestamp'] = pd.to_datetime(
                    dfy['timestamp'],
                    format="%d/%m/%Y %H:%M:%S",
                    errors='coerce',
) 

dfy['target'] = dfy['target'].eq(1).astype('int8')


assert len(dfX) == len(dfy) == 1567
assert dfX.shape[1] == 590

dfX.to_parquet(DATA_PROCESSED / 'dfX_raw.parquet')
dfy.to_parquet(DATA_PROCESSED / 'dfy_raw.parquet')

# %% Remove constant

n_values = dfX.nunique()
drop_list_const = n_values[n_values == 1].index.tolist()
dfX = dfX.drop(columns=drop_list_const)
#print(dfX.shape)

# drop features with >50% nan
drop_feature_limit = int(len(dfX)/2)

n_nan = dfX.isnull().sum()
drop_list_nan = n_nan[n_nan > drop_feature_limit].index.tolist()
dfX = dfX.drop(columns=drop_list_nan)
#print(dfX.shape)

print(f'Number of constant: {len(drop_list_const)}, number of nan>50%: {len(drop_list_nan)}')

# %% concentrated-value diagnostics — vectorized, no loop

DOMINANT_FRAC = 0.99   # dominant value in >99% of non-null rows
CV_LIMIT      = 0.01
FEW_UNIQUE    = 5
DEVIANT_FAIL_ENRICHMENT = 2.0   # keep if deviants are >=2x enriched for fails

fail_rate = dfy['target'].mean()          # ~0.066

def dominant_value_stats(s: pd.Series):
    vc = s.value_counts(dropna=True)
    dom_val, dom_cnt = vc.index[0], vc.iloc[0]
    return dom_val, dom_cnt / s.notna().sum()

rows = []
for col in dfX.columns:
    s = dfX[col]
    dom_val, dom_frac = dominant_value_stats(s)
    mean, std = s.mean(), s.std()
    cv = abs(std / mean) if pd.notna(mean) and abs(mean) > 1e-12 else np.nan

    # deviant rows = not the dominant value (and not NaN)
    deviant = s.notna() & (s != dom_val)
    n_dev = deviant.sum()
    dev_fail_rate = dfy.loc[deviant, 'target'].mean() if n_dev > 0 else np.nan

    rows.append({
        'feature':      col,
        'n_unique':     s.nunique(),
        'dom_frac':     dom_frac,
        'cv':           cv,
        'missing_rate': s.isna().mean(),
        'n_deviant':    n_dev,
        'dev_fail_rate': dev_fail_rate,
        'enrichment':   dev_fail_rate / fail_rate if n_dev > 0 else np.nan,
    })

diag = pd.DataFrame(rows).set_index('feature')
display(diag)
diag.to_parquet(ARTIFACTS / 'diagnostic.parquet')

df_check = pd.read_parquet(ARTIFACTS / 'diagnostic.parquet')
display(df_check)

# %% classify

flagged = diag[
    (diag['dom_frac'] > DOMINANT_FRAC) |
    (diag['cv'] < CV_LIMIT) |
    (diag['n_unique'] < FEW_UNIQUE)
]

# keep flagged features whose rare deviants are enriched for fails
keep = flagged[
    (flagged['n_deviant'] >= 5) &
    (flagged['enrichment'] >= DEVIANT_FAIL_ENRICHMENT)
]

drop = flagged.index.difference(keep.index)

print(f"flagged near-constant: {len(flagged)}, keep (enriched): {len(keep)}, drop: {len(drop)}")
display(keep.sort_values('enrichment', ascending=False))

dropped_cols = {'constant': drop_list_const, 'high_nan': drop_list_nan, 'nzv': drop.tolist()}
dfX = dfX.drop(columns=drop)

print(f'Total remaining features: {dfX.shape[1]}')

# %%

corr_matrix = dfX.corr()
pairs = corr_matrix.unstack()

# 3. Filter out self-correlation (1.0) and duplicate mirrored pairs
# We look at the multi-index keys and only keep pairs where the first name is less than the second
unique_pairs = pairs[pairs.index.get_level_values(0) < pairs.index.get_level_values(1)]

# 4. Sort to see the strongest relationships
sorted_pairs = unique_pairs.sort_values(ascending=False)

# Keep pairs with abs correlation > 0.9 and make drop list
abs_sorted_pairs = sorted_pairs.abs()
to_drop = set()
for a, b in abs_sorted_pairs[abs_sorted_pairs > 0.95].index:
    if a not in to_drop and b not in to_drop:
        # drop the one with more missing data
        miss = dfX[[a, b]].isna().mean()
        to_drop.add(miss.idxmax())

dfX = dfX.drop(columns=list(to_drop))

# %% Save cleaned data

print(dfX.shape)
dfX.to_parquet(DATA_PROCESSED / 'dfX_v1.parquet')
print(dfy.shape)
dfy.to_parquet(DATA_PROCESSED / 'dfy_v1.parquet')

df_check = pd.read_parquet(DATA_PROCESSED / 'dfX_v1.parquet')

display(df_check)
print(df_check.shape)

#%%



