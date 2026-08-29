#%%
import sys
import pandas as pd
import numpy as np
import argparse
import logging
import json 
from pathlib import Path 

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
from semcon.config import parse_overrides, load_config
from semcon.evaluation import (tune_threshold, 
                               classification_summary, 
                               save_pr_curve, 
                               save_confusion_heatmap,
                               operating_points,
                               recall_at_flagrate
)
from semcon.tracking import (
    make_run,
    save_splits,
    append_index,
    save_features,
)

logger = logging.getLogger("semcon")


#%%

def parse_args(argv=None):
    p = argparse.ArgumentParser(description='SECOM XGBoost training')
    # Experiment arguments
    p.add_argument('--run-name', type=str, default=None,
               help='experiment slug for the run folder, e.g. baseline')
    p.add_argument('--note', type=str, default='',
               help='free-text note stored in config.json and the runs index')
    # Data feature
    p.add_argument('--no-selection', action='store_false', dest='use_selection',
                   help='all-feature baseline instead of fold-internal selection')
    # CV feature
    p.add_argument('--repeats', type=int, default=3)
    # XGB Model features
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
               dest='overrides',
               help='override a model param, repeatable: --set max_depth=5')
    return p.parse_args(argv)

def load_data(test_split:int, time_split:int):
    X_PATH = DATA_PROCESSED / 'dfX_v2.parquet'
    Y_PATH = DATA_PROCESSED / 'dfy_v1.parquet'
    dfX = pd.read_parquet(X_PATH)
    dfy = pd.read_parquet(Y_PATH)
    data_info = json.loads(X_PATH.with_suffix('.dataset.json').read_text())

    # Sort training data on the 
    df_train = pd.concat([dfX, dfy], axis=1).sort_values('timestamp').reset_index(drop=True)

    # Drop last 27, from eda, different regime 
    df_train = df_train.iloc[:-time_split, :]
    # Use the last 15% (231 rows) as test data
    df_test = df_train.iloc[-test_split:,:]
    df_train = df_train.iloc[:-test_split, :]

    target = 'target'
    return df_train, df_test, target, data_info
    
    
#%%

def run_cv(df_train:pd.DataFrame, target:str, xgb_params:dict, random:int,
           kfolds:int, repeats:int, use_selection:bool) \
        -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    
    df_y = df_train[target].copy()
    df_X = df_train.drop(columns=['timestamp', 'target']).copy(deep=True)

    rskf = RepeatedStratifiedKFold(n_splits=kfolds, n_repeats=repeats,
                                random_state=random)

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
            **xgb_params,
            eval_metric=['logloss', 'aucpr'], # Last one for early-stopping
            scale_pos_weight=float((ytrain == 0).sum() / (ytrain == 1).sum()),
            callbacks=[es],            
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
                res:pd.DataFrame, xgb_params:dict, sel_count:pd.Series, 
                kfolds:int, repeats:int, use_selection:bool, random:int, out:Path) \
                -> tuple[XGBClassifier, list, np.ndarray, pd.Series | None]:
    
    stability = None
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
        **xgb_params,
        scale_pos_weight=float((df_y == 0).sum() / (df_y == 1).sum()),
        device='cuda', random_state=random,
    )
    final.fit(df_X[feats], df_y)

    final.get_booster().save_model(str(out / "model.ubj"))
    p_hold = final.predict_proba(X_hold[feats])[:, 1]
    np.save(out / "p_hold.npy", p_hold)
    logger.info(f'HOLDOUT  aucpr={average_precision_score(y_hold, p_hold):.4f}  '
        f'rocauc={roc_auc_score(y_hold, p_hold):.4f}  '
        f'brier={brier_score_loss(y_hold, p_hold):.4f}')

    np.save(out / 'oof_xgb1.npy', oof)
    res.to_parquet(out / 'cv_metrics_xgb1.csv')
    return final, feats, p_hold, stability

def evaluate(
    df_train:pd.DataFrame, 
    df_test:pd.DataFrame, 
    oof:np.ndarray, 
    p_hold:np.ndarray, 
    out:Path,
    )->dict:
    
    df_y = df_train['target'].copy()
    y_hold = df_test['target'].copy()

    oof_mean = oof.mean(axis=0)                       # average the 3 repeats
    thr = tune_threshold(df_y, oof_mean, criterion='mcc')
    summary_oof = classification_summary(df_y, oof_mean, thr)
    summary_oof.to_csv(out / 'summary_oof.csv')
    summary_hold = classification_summary(y_hold, p_hold, thr)
    summary_hold.to_csv(out / 'summary_hold.csv')
    logger.info(summary_oof)      # in-regime, honest
    logger.info(summary_hold)      # same threshold, drift regime
    save_pr_curve(y_hold, p_hold, out / 'pr_curve_holdout.png')
    save_confusion_heatmap(y_hold, p_hold, thr, 
                           out / 'conf_heatmap.png', 
                           )
    
    q = float(summary_oof[['tp', 'fp']].sum() / len(df_y)) 
    r_value = recall_at_flagrate(y_hold, p_hold, q=q)
    logger.info(f'recall_at_flagrate: {r_value}')
    
    pred_hold_stats = pd.Series(p_hold).describe()
    pred_hold_stats.to_csv(out / 'pred_hold_stats.csv')
    logger.info(pred_hold_stats)
    oof_mean_stats = pd.Series(oof_mean).describe()
    oof_mean_stats.to_csv(out / 'oof_mean_stats.csv')
    logger.info(oof_mean_stats)
    
    pr_oof_hold = operating_points(df_y, oof_mean)
    pr_oof_hold.to_csv(out / 'pr_oof_hold.csv')
    logger.info(pr_oof_hold)
    return {
        "threshold": float(thr),
        "holdout_aucpr": float(average_precision_score(y_hold, p_hold)),
        "holdout_rocauc": float(roc_auc_score(y_hold, p_hold)),
        "holdout_brier": float(brier_score_loss(y_hold, p_hold)),
        "holdout_recall": float(summary_hold["recall"]),
        "holdout_precision": float(summary_hold["precision"]),
        "flagrate_recall": float(r_value[0]),
    }


#%% 

def main(argv=None):         # argv param => testable
    # arg check for notebooks
    if argv is None:
        argv = [] if "ipykernel" in sys.modules else sys.argv[1:]
    args = parse_args(argv)
    # Log start training pipeline and argv 
    logger = setup_logging(logfile=LOGS / "ml.log")
    logger.info(f'[train_xgb] start | selection={args.use_selection} '
                f'repeats={args.repeats}')
    # Load configs and overwrite if new ones are included in argv
    cfg = load_config()
    if args.overrides:
        overrides = parse_overrides(args.overrides)       # CLI over everything
        cfg = cfg.with_model_overrides(overrides)
    
    # Make run folder and first data dump
    xgb_params = cfg.model.model_dump()
    run_dir, run_meta = make_run(
        config={"script": "train_xgb", "use_selection": args.use_selection,
                "repeats": args.repeats, "kfolds": cfg.pipeline.kfolds,
                "tail_n": cfg.pipeline.tail_n, "holdout_n": cfg.pipeline.holdout_n,
                "model": xgb_params, "random_state": cfg.pipeline.seed},
        run_name=args.run_name or ("xgb_sel" if args.use_selection else "xgb_base"),
        note=args.note,
    )
    
    # Load data
    df_train, df_test, target, data_info = load_data(cfg.pipeline.holdout_n, cfg.pipeline.tail_n)
    # Save data from cv splits  
    save_splits(run_dir, df_train, df_test)
    # Run cv training 
    res, oof, sel_count = run_cv(df_train, target, xgb_params, cfg.pipeline.seed,
                                 kfolds=cfg.pipeline.kfolds,
                                 repeats=args.repeats,
                                 use_selection=args.use_selection)
    # Based on training refit full model
    final, feats, p_hold, stability = refit_final(df_train, df_test, oof, res, xgb_params,
                                sel_count, kfolds=cfg.pipeline.kfolds,
                                repeats=args.repeats,use_selection=args.use_selection,
                                random=cfg.pipeline.seed, out=run_dir)
    
    # Save information on selected features
    save_features(run_dir, feats, stability)
    # Evaluate full model on holdout data
    holdout_metrics = evaluate(df_train, df_test, oof, p_hold, out=run_dir)
    
    # Set up shap analysis
    df_X = df_train.drop(columns=["timestamp", "target"])
    # Run and save shap analysis
    shap_summary = save_shap_plots(
        model=final,
        X=df_X[feats],
        output_dir=run_dir / "shap",
    )
    logger.info("Top SHAP features:\n%s", shap_summary.head(15).to_string(),)
    
    # Save experiment data in index.csv
    append_index(run_dir, {
    "run_name": args.run_name, "note": args.note,
    "git_sha": run_meta["git_sha"],
    "data": data_info["sha256_16"],          # the aligned card key, per the earlier fix
    "cv_aucpr_mean": float(res["aucpr"].mean()),
    "cv_aucpr_std": float(res["aucpr"].std()),
    **holdout_metrics,
})
    
    logger.info(f'[train_xgb] end')

if __name__ == "__main__":
    #main()
    main(["--no-selection"])

# %%
