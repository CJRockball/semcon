#%%
# build_features.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

from semcon.paths import DATA_PROCESSED, LOGS
from semcon.utils import setup_logging
from semcon.tracking import write_dataset_card

logger = logging.getLogger("semcon")

# %% missingness indicators — built from RAW features
                 # partial 5th block: columns 542..589

def build_features(df_values: pd.DataFrame, dfX_raw: pd.DataFrame,
                   clique1:list[int], clique2:list[int], block:int, logger) -> pd.DataFrame:
    """Add missingness/data-quality features to the preprocessed value frame."""
    feats = pd.DataFrame(index=df_values.index)

    # clique indicators — from RAW: clique 14 members (50.7% NaN) were dropped by
    # the >50% rule and don't exist in df_values
    feats['miss_clq14'] = dfX_raw[clique1].isna().any(axis=1).astype('int8')
    feats['miss_clq23'] = dfX_raw[clique2].isna().any(axis=1).astype('int8')

    # block-5 dropout flag — label-free probe for the time-blocked holdout
    feats['miss_block5'] = dfX_raw.iloc[:, block:].isna().any(axis=1).astype('int8')

    # per-wafer data quality — computed on the raw 590 for consistency with Section 3.6
    feats['row_missing_rate'] = dfX_raw.isna().mean(axis=1).astype('float32')

    # sanity anchors from the EDA dropout summary
    assert feats['miss_clq14'].sum() == 794
    assert feats['miss_clq23'].sum() == 715

    out = pd.concat([df_values, feats], axis=1)
    logger.info(f"Feature build: {df_values.shape[1]} values + {feats.shape[1]} engineered "
             f"= {out.shape[1]} cols | clq14={feats['miss_clq14'].sum()}, "
             f"clq23={feats['miss_clq23'].sum()}, block5={feats['miss_block5'].sum()}, "
             "row_missing_rate")
    return out


def main():

    logger.info('[feature_eng.py] Start feature engineering pipeline')

    clique1 = [72, 73, 345, 346]      # CLIQUE_14 OR 0.49, p=0.0008
    clique2 = [112, 247, 385, 519]    # CLIQUE_23, OR 0.53, p=0.0031
    block = 542 


    df_values = pd.read_parquet(DATA_PROCESSED / 'dfX_v1.parquet')
    dfX_raw   = pd.read_parquet(DATA_PROCESSED / 'dfX_raw.parquet')
    dfy = pd.read_parquet(DATA_PROCESSED / 'dfy_v1.parquet')

    df_model = build_features(df_values, dfX_raw, clique1, clique2, block, logger)
    df_model.to_parquet(DATA_PROCESSED / 'dfX_v2.parquet')
    write_dataset_card(DATA_PROCESSED / 'dfX_v2.parquet', df_model, dfy['target'])
    
    logger.info(f"Saved dfX_v2.parquet {df_model.shape}")

    


if __name__ == '__main__':
    main()

# %%
