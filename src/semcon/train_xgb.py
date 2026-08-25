#%%
import sys
import pandas as pd
import numpy as np
import argparse

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score, 
    brier_score_loss,
    average_precision_score,
)

import xgboost as xgb
from xgboost import XGBClassifier

from semcon.paths import DATA_PROCESSED, ARTIFACTS, LOGS
from semcon.utils import setup_logging
from semcon.selection import select_features
from semcon.explain import save_shap_plots
from semcon.evaluation import (tune_threshold, 
                               classification_summary, 
                               save_pr_curve, 
                               save_confusion_heatmap,
                               recall_at_flagrate
)

RANDOM_STATE = 1337

#%%

def parse_args(argv=None):
    p = argparse.ArgumentParser(description='SECOM XGBoost training')
    p.add_argument('--no-selection', action='store_false', dest='use_selection',
                   help='all-feature baseline instead of fold-internal selection')
    p.add_argument('--repeats', type=int, default=3)
    return p.parse_args(argv)

def load_data(test_split:int, time_split:int):
    dfX = pd.read_parquet(DATA_PROCESSED / 'dfX_v2.parquet')
    dfy = pd.read_parquet(DATA_PROCESSED / 'dfy_v1.parquet')

    # Sort training data on the 
    df_train = pd.concat([dfX, dfy], axis=1).sort_values('timestamp').reset_index(drop=True)

    # Drop last 27, from eda, different regime 
    df_train = df_train.iloc[:-time_split, :]
    # Use the last 15% (231 rows) as test data
    df_test = df_train.iloc[-test_split:,:]
    df_train = df_train.iloc[:-test_split, :]

    target = 'target'
    return df_train, df_test, target
    
    
#%%

def run_cv(df_train:pd.DataFrame, target:str, 
           kfolds:int, repeats:int, use_selection:bool, logger) \
        -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    
    df_y = df_train[target].copy()
    df_X = df_train.drop(columns=['timestamp', 'target']).copy(deep=True)

    rskf = RepeatedStratifiedKFold(n_splits=kfolds, n_repeats=repeats,
                                random_state=RANDOM_STATE)

    oof = np.zeros((repeats, len(df_X)))   # one OOF vector per repeat
    fold_metrics = []
    sel_count = pd.Series(0, index=df_X.columns)
    for i,(train_index, valid_index) in enumerate(rskf.split(df_X, df_y)):
        Xtrain = df_X.iloc[train_index].copy()
        ytrain = df_y.iloc[train_index].copy()
        Xvalid = df_X.iloc[valid_index].copy()
        yvalid = df_y.iloc[valid_index].copy()
        
        if use_selection:
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

      
        model = XGBClassifier(
            objective='binary:logistic',
            eval_metric=['logloss', 'aucpr'], # Last one for early-stopping
            scale_pos_weight=float((ytrain == 0).sum() / (ytrain == 1).sum()),
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
                eval_set=[(Xvalid, yvalid)],
                verbose=100
                )
        
        ypred_proba = model.predict_proba(Xvalid)
        oof[i // kfolds, valid_index] = ypred_proba[:,1]
        
        fold_metrics.append({
            'repeat': i // kfolds, 'fold': i % kfolds,
            'aucpr': average_precision_score(yvalid, ypred_proba[:,1]),
            'rocauc': roc_auc_score(yvalid, ypred_proba[:,1]),
            'brier': brier_score_loss(yvalid, ypred_proba[:,1]),
            'best_iter': model.best_iteration,
            'n_feats': Xtrain.shape[1],
        }) 

    res = pd.DataFrame(fold_metrics)
    logger.info(res.groupby('repeat')[['aucpr', 'rocauc', 'brier']].mean().round(4))
    logger.info(res[['aucpr', 'rocauc', 'brier']].agg(['mean', 'std']).round(4))
    return res, oof, sel_count

#%% Selection stability across the 15 fits
def refit_final(df_train:pd.DataFrame, df_test:pd.DataFrame, oof:np.ndarray,
                res:pd.DataFrame, sel_count:pd.Series, 
                kfolds:int, repeats:int, use_selection:bool, logger):

    if use_selection:
        stability = (sel_count / (kfolds * repeats)).sort_values(ascending=False)
        print(stability.head(30))
        stable = stability[stability >= 0.8].index.tolist()
        print(f'{len(stable)} features selected in >=80% of fits')

    # Final refit on full CV pool, ONE holdout evaluation
    df_y = df_train['target'].copy()
    df_X = df_train.drop(columns=['timestamp', 'target']).copy()

    y_hold = df_test['target'].copy()
    X_hold = df_test.drop(columns=['timestamp', 'target']).copy()

    feats = select_features(df_X, df_y) if use_selection else df_X.columns.tolist()
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
    logger.info(f'HOLDOUT  aucpr={average_precision_score(y_hold, p_hold):.4f}  '
        f'rocauc={roc_auc_score(y_hold, p_hold):.4f}  '
        f'brier={brier_score_loss(y_hold, p_hold):.4f}')

    np.save(ARTIFACTS / 'oof_xgb2.npy', oof)
    res.to_parquet(ARTIFACTS / 'cv_metrics_xgb2.parquet')
    return final, feats, p_hold

def evaluate(
    df_train:pd.DataFrame, 
    df_test:pd.DataFrame, 
    oof:np.ndarray, 
    p_hold:np.ndarray, 
    logger):
    
    df_y = df_train['target'].copy()
    y_hold = df_test['target'].copy()

    oof_mean = oof.mean(axis=0)                       # average the 3 repeats
    thr = tune_threshold(df_y, oof_mean, criterion='mcc')
    summary_oof = classification_summary(df_y, oof_mean, thr)
    summary_oof.to_csv(ARTIFACTS / 'summary_oof.csv')
    summary_hold = classification_summary(y_hold, p_hold, thr)
    summary_hold.to_csv(ARTIFACTS / 'summary_hold.csv')
    logger.info(summary_oof)      # in-regime, honest
    logger.info(summary_hold)      # same threshold, drift regime
    save_pr_curve(y_hold, p_hold, ARTIFACTS / 'pr_curve_holdout.png')
    save_confusion_heatmap(y_hold, p_hold, thr, 
                           ARTIFACTS / 'conf_heatmap.png', 
                           )
    
    q = float(summary_oof[['tp', 'fp']].sum() / len(df_y)) 
    r_value = recall_at_flagrate(y_hold, p_hold, q=q)
    logger.info(f'recall_at_flagrate: {r_value}')
    
    pred_hold_stats = pd.Series(p_hold).describe()
    pred_hold_stats.to_csv(ARTIFACTS / 'pred_hold_stats.csv')
    logger.info(pred_hold_stats)
    oof_mean_stats = pd.Series(oof_mean).describe()
    oof_mean_stats.to_csv(ARTIFACTS / 'oof_mean_stats.csv')
    logger.info(oof_mean_stats)
    
    return


#%% 

def main(argv=None):         # argv param => testable
    if argv is None:
        argv = [] if "ipykernel" in sys.modules else sys.argv[1:]
    args = parse_args(argv)
    
    logger = setup_logging(logfile=LOGS / "ml.log")
    logger.info(f'[train_xgb] start | selection={args.use_selection} '
                f'repeats={args.repeats}')

    KFOLDS = 5
    TEST_SPLIT = 231 # Chose 15% of the data
    TIME_SPLIT = 27 # From EDA different regime
    
    df_train, df_test, target = load_data(TEST_SPLIT, TIME_SPLIT)
    res, oof, sel_count = run_cv(df_train, target, kfolds=KFOLDS,
                                 repeats=args.repeats,
                                 use_selection=args.use_selection, logger=logger,)
    
    final, feats, p_hold = refit_final(df_train, df_test, oof, res, sel_count, 
                                kfolds=KFOLDS,
                                repeats=args.repeats,
                                use_selection=args.use_selection,logger=logger,)

    evaluate(df_train, df_test, oof, p_hold, logger)

    df_X = df_train.drop(columns=["timestamp", "target"])

    shap_summary = save_shap_plots(
        model=final,
        X=df_X[feats],
        output_dir=ARTIFACTS / "shap",
    )

    logger.info(
        "Top SHAP features:\n%s",
        shap_summary.head(15).to_string(),
    )


if __name__ == "__main__":
    main()
    #main(["--no-selection", "--repeats", "3"])

# %%
