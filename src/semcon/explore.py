#%%
import pandas as pd
import numpy as np
import logging

from semcon.paths import DATA_RAW, DATA_PROCESSED, ARTIFACTS, LOGS
from semcon.utils import setup_logging
from semcon.tracking import write_dataset_card

logger = logging.getLogger("semcon")

#%%

def load_data():
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
    return dfX, dfy

# %% Remove constant

def drop_basic(dfX, dfy):
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


    # concentrated-value diagnostics — vectorized, no loop
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
    diag.to_parquet(ARTIFACTS / 'diagnostic.parquet')

    df_check = pd.read_parquet(ARTIFACTS / 'diagnostic.parquet')


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

    dfX = dfX.drop(columns=drop)
    print(f'Total remaining features: {dfX.shape[1]}')

    # Drop correlated 
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

    dropped_cols = {'constant': drop_list_const, 'high_nan': drop_list_nan, 
                    'nzv': drop.tolist(), 'corr':list(to_drop)}
    dfX = dfX.drop(columns=list(to_drop))
    
    return dfX, dfy, dropped_cols

#%%

def main():
    
    logger = setup_logging(logfile=LOGS / "ml.log")
    logger.info('[explore.py] Start data pipeline')
    
    dfX, dfy = load_data()
    dfX, dfy, dropped_cols = drop_basic(dfX, dfy)

    dfX.to_parquet(DATA_PROCESSED / 'dfX_v1.parquet')
    write_dataset_card(DATA_PROCESSED / 'dfX_v1.parquet', dfX, dfy['target'])
    logger.info(f"[explore.py] Saved dfX_v1.parquet {dfX.shape}")

    dfy.to_parquet(DATA_PROCESSED / 'dfy_v1.parquet')
    write_dataset_card(DATA_PROCESSED / 'dfy_v1.parquet', dfy)
    logger.info(f"[explore.py] Saved dfy_v1.parquet {dfy.shape}")    

    return


if __name__ == "__main__":
    main()

