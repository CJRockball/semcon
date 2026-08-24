#%%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score, 
    brier_score_loss,

    average_precision_score,
)

import xgboost as xgb
from xgboost import XGBClassifier

from secom.paths import DATA_PROCESSED, ARTIFACTS
from secom.utils import setup_logging

RANDOM_STATE = 1337
setup_logging()

#%%

dfX = pd.read_parquet(DATA_PROCESSED / 'dfX_v2.parquet')
dfy = pd.read_parquet(DATA_PROCESSED / 'dfy_v1.parquet')

# Sort training data on the 
df_train = pd.concat([dfX, dfy], axis=1).sort_values('timestamp').reset_index(drop=True)

# Drop last 27, from eda, different regime 
df_train = df_train.iloc[:-27, :]
# Use the last 15% (231 rows) as test data
df_test = df_train.iloc[-231:,:]
df_train = df_train.iloc[:-231, :]

target = 'target'

#%%
#%% Fold-internal feature selection

def hedges_g(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """|Hedges g| per column vs binary y. NaN-safe. TRAIN-fold data only."""
    x1, x0 = X[y == 1], X[y == 0]
    n1, n0 = len(x1), len(x0)
    m1, m0 = x1.mean(), x0.mean()
    v1, v0 = x1.var(ddof=1), x0.var(ddof=1)
    sp = np.sqrt(((n1 - 1) * v1 + (n0 - 1) * v0) / (n1 + n0 - 2))
    d = (m1 - m0) / sp.replace(0, np.nan)
    J = 1 - 3 / (4 * (n1 + n0) - 9)          # small-sample correction
    return (d * J).abs().fillna(0.0)

def select_features(Xtr, ytr, g_min=0.25, k_min=30, k_max=100):
    """Univariate filter, recomputed inside every fold (no full-data leakage).
    g >= 0.25 keeps real effects (EDA: noise floor ~0.1, real effects 0.3-0.5);
    k_min/k_max guard against degenerate fold selections."""
    g = hedges_g(Xtr, ytr)
    sel = g[g >= g_min].index.tolist()
    if len(sel) < 10:
        sel = g.nlargest(k_min).index.tolist()
    elif len(sel) > k_max:
        sel = g.nlargest(k_max).index.tolist()
    return sel

#%%
df_y = df_train[target].copy()
df_X = df_train.drop(columns=['timestamp', 'target']).copy(deep=True)

KFOLDS, REPEATS = 5, 3
USE_SELECTION = True   # False -> all-feature baseline; run both, compare

rskf = RepeatedStratifiedKFold(n_splits=KFOLDS, n_repeats=REPEATS,
                               random_state=RANDOM_STATE)

oof = np.zeros((REPEATS, len(df_X)))   # one OOF vector per repeat
rows = []
sel_count = pd.Series(0, index=df_X.columns)

models = []
rows = []
for i,(train_index, valid_index) in enumerate(rskf.split(df_X, df_y)):
    Xtrain = df_X.iloc[train_index].copy()
    ytrain = df_y.iloc[train_index].copy()
    Xvalid = df_X.iloc[valid_index].copy()
    yvalid = df_y.iloc[valid_index].copy()
    
    if USE_SELECTION:
        feats = select_features(Xtrain, ytrain)
        sel_count[feats] += 1
        Xtrain, Xvalid = Xtrain[feats], Xvalid[feats]
                                                                
    # XGB
    # Early stopping call back, use to get best model back
    es = xgb.callback.EarlyStopping(
    rounds=100,
    min_delta=1e-5,
    save_best=True,
    maximize=True,
    data_name="validation_0",
    metric_name="aucpr",)

    # Inside your fold loop, after iloc slicing:
    sample_weights = compute_sample_weight(class_weight="balanced", y=ytrain)
        
    
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric=['logloss', 'aucpr'], # Last one for early-stopping
        #enable_categorical=True,
        #early_stopping_rounds=100,
        callbacks=[es],
        n_estimators=5000,
        
        learning_rate=0.03,
        max_depth=4,
        max_bin=511,
        reg_alpha=3,
        reg_lambda=2,
        device = 'cuda',
    )
    
    model.fit(Xtrain, ytrain,
              sample_weight=sample_weights,   # <-- per-sample weights
              eval_set=[(Xvalid, yvalid)],
              verbose=100
              )
    
    ypred_proba = model.predict_proba(Xvalid)
    y_pred = model.predict(Xvalid)
    oof[i // KFOLDS, valid_index] = ypred_proba[:,1]
      
    rows.append({
        'repeat': i // KFOLDS, 'fold': i % KFOLDS,
        'aucpr': average_precision_score(yvalid, ypred_proba[:,1]),
        'rocauc': roc_auc_score(yvalid, ypred_proba[:,1]),
        'brier': brier_score_loss(yvalid, ypred_proba[:,1]),
        'best_iter': model.best_iteration,
        'n_feats': Xtrain.shape[1],
    }) 

res = pd.DataFrame(rows)
print(res.groupby('repeat')[['aucpr', 'rocauc', 'brier']].mean().round(4))
print(res[['aucpr', 'rocauc', 'brier']].agg(['mean', 'std']).round(4))

#%% Selection stability across the 15 fits
if USE_SELECTION:
    stability = (sel_count / (KFOLDS * REPEATS)).sort_values(ascending=False)
    print(stability.head(30))
    stable = stability[stability >= 0.8].index.tolist()
    print(f'{len(stable)} features selected in >=80% of fits')

#%% Final refit on full CV pool, ONE holdout evaluation
y_hold = df_test['target'].copy()
X_hold = df_test.drop(columns=['timestamp', 'target']).copy()

feats = select_features(df_X, df_y) if USE_SELECTION else df_X.columns.tolist()
best_iter = int(np.median(res['best_iter']))

final = XGBClassifier(
    objective='binary:logistic',
    eval_metric='aucpr',
    n_estimators=best_iter,
    learning_rate=0.03, max_depth=4, max_bin=511,
    reg_alpha=3, reg_lambda=2,
    scale_pos_weight=float((df_y == 0).sum() / (df_y == 1).sum()),
    device='cuda', random_state=RANDOM_STATE,
)
final.fit(df_X[feats], df_y)

p_hold = final.predict_proba(X_hold[feats])[:, 1]
print(f'HOLDOUT  aucpr={average_precision_score(y_hold, p_hold):.4f}  '
      f'rocauc={roc_auc_score(y_hold, p_hold):.4f}  '
      f'brier={brier_score_loss(y_hold, p_hold):.4f}')


np.save(ARTIFACTS / '/oof_xgb2.npy', oof)
res.to_parquet(ARTIFACTS / '/cv_metrics_xgb2.parquet')


# %%

# Get feature importance scores
importance_scores = model.get_booster().get_score(importance_type='total_gain')
df_imp = pd.DataFrame.from_dict(importance_scores, orient='index', columns=['Importance'])

display(df_imp)
df_imp.plot(kind='barh')

#%% SHAP
