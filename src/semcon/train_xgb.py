"""SECOM XGBoost training — wide-frame edition.

Data flows: extract() -> build_features() -> registry selection -> CV.
The frame carries metadata (wafer_id, timestamp, split); X is built by
inclusion from the registry, never by dropping known non-features.

Deferred by design:
  - split-column adoption is blocked on config.CUTOFF (the split is
    positional tail-drop + holdout until EDA freezes the timestamps);
    the equivalence guard in load_data() activates the day CUTOFF lands
  - is_fail encoding lives here until Phase 4's validate.py owns the
    silver boundary
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from xgboost import XGBClassifier

from semcon import schema
from semcon.config import CUTOFF, load_config, parse_overrides
from semcon.db import (
    assert_schema,
    data_fingerprint,
    feature_columns,
    get_engine,
    load_registry,
    register_columns,
)
from semcon.evaluation import (
    classification_summary,
    operating_points,
    recall_at_flagrate,
    save_confusion_heatmap,
    save_pr_curve,
    tune_threshold,
)
from semcon.explain import save_shap_plots
from semcon.extract import extract
from semcon.feature_eng import build_features
from semcon.paths import ARTIFACTS, LOGS
from semcon.selection import select_features
from semcon.snapshots import write_gold_snapshot
from semcon.tracking import append_index, make_run, save_features, save_splits
from semcon.utils import setup_logging

logger = logging.getLogger("semcon")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description='SECOM XGBoost training')
    p.add_argument('--run-name', type=str, default=None,
               help='experiment slug for the run folder, e.g. baseline')
    p.add_argument('--note', type=str, default='',
               help='free-text note stored in config.json and the runs index')
    p.add_argument('--no-selection', action='store_false', dest='use_selection',
                   help='all-feature baseline instead of fold-internal selection')
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--set', action='append', default=[], metavar='KEY=VALUE',
                   dest='overrides',
                   help='override a model param, repeatable: --set max_depth=5')
    return p.parse_args(argv)


def load_data(engine, test_split: int, time_split: int):
    """Wide frame from the DB; legacy positional tail-drop + holdout split.

    Returns (df_train, df_test, features, fingerprint). Frames carry
    metadata columns; `features` is the registry-active list — build X by
    inclusion, never by dropping.
    """
    assert_schema(engine)
    df = extract(engine)
    df, eng_rows = build_features(df)
    register_columns(eng_rows, engine)  # idempotent upsert — self-healing
                                        # if feature_eng hasn't run on this DB

    # Phase 4: is_fail moves to validate.py (silver boundary owns encoding)
    df["is_fail"] = df[schema.TARGET_COL].eq(1).astype("int8")

    features = feature_columns(load_registry(engine))
    bad = set(schema.NON_FEATURE_COLS) & set(features)
    if bad:
        raise ValueError(f"registry-active list contains non-features: {sorted(bad)}")

    df = df.sort_values(schema.TIME_COL).reset_index(drop=True)
    df_train = df.iloc[:-time_split, :]          # drop regime tail (EDA finding)
    df_test = df_train.iloc[-test_split:, :]     # last holdout_n rows
    df_train = df_train.iloc[:-test_split, :]

    if CUTOFF is not None:
        # equivalence guard: positional split must agree with the SQL labels
        if not (df_train[schema.SPLIT_COL] == "cv").all():
            raise ValueError("train rows not all 'cv' — positional vs SQL split disagree")
        if not (df_test[schema.SPLIT_COL] == "holdout").all():
            raise ValueError("test rows not all 'holdout' — positional vs SQL split disagree")

    return df_train, df_test, features, data_fingerprint(engine)


def run_cv(df_train: pd.DataFrame, features: list[str], xgb_params: dict,
           random: int, kfolds: int, repeats: int, use_selection: bool) \
        -> tuple[pd.DataFrame, np.ndarray, pd.Series]:

    df_y = df_train["is_fail"].copy()
    df_X = df_train[features].copy(deep=True)

    rskf = RepeatedStratifiedKFold(n_splits=kfolds, n_repeats=repeats,
                                   random_state=random)

    oof = np.zeros((repeats, len(df_X)))   # one OOF vector per repeat
    fold_metrics = []
    sel_count = pd.Series(0, index=df_X.columns)
    for i, (train_index, valid_index) in enumerate(rskf.split(df_X, df_y)):
        Xtrain = df_X.iloc[train_index].copy()
        ytrain = df_y.iloc[train_index].copy()
        Xvalid = df_X.iloc[valid_index].copy()
        yvalid = df_y.iloc[valid_index].copy()

        if use_selection:
            feats = select_features(Xtrain, ytrain)
            sel_count[feats] += 1
            Xtrain, Xvalid = Xtrain[feats], Xvalid[feats]

        es = xgb.callback.EarlyStopping(
            rounds=100,
            min_delta=1e-5,
            save_best=True,
            maximize=True,
            data_name="validation_0",
            metric_name="aucpr",
        )

        model = XGBClassifier(
            **xgb_params,
            eval_metric=['logloss', 'aucpr'],
            scale_pos_weight=float((ytrain == 0).sum() / (ytrain == 1).sum()),
            callbacks=[es],
            device='cuda',
        )

        model.fit(Xtrain, ytrain,
                  eval_set=[(Xvalid, yvalid)],
                  verbose=100)

        ypred_proba = model.predict_proba(Xvalid)
        oof[i // kfolds, valid_index] = ypred_proba[:, 1]

        fold_metrics.append({
            'repeat': i // kfolds, 'fold': i % kfolds,
            'aucpr': average_precision_score(yvalid, ypred_proba[:, 1]),
            'rocauc': roc_auc_score(yvalid, ypred_proba[:, 1]),
            'brier': brier_score_loss(yvalid, ypred_proba[:, 1]),
            'best_iter': model.best_iteration,
            'n_feats': Xtrain.shape[1],
        })

    res = pd.DataFrame(fold_metrics)
    logger.info(res.groupby('repeat')[['aucpr', 'rocauc', 'brier']].mean().round(4))
    logger.info(res[['aucpr', 'rocauc', 'brier']].agg(['mean', 'std']).round(4))
    return res, oof, sel_count


def refit_final(df_train: pd.DataFrame, df_test: pd.DataFrame,
                features: list[str], oof: np.ndarray, res: pd.DataFrame,
                xgb_params: dict, sel_count: pd.Series, kfolds: int,
                repeats: int, use_selection: bool, random: int, out: Path) \
        -> tuple[XGBClassifier, list, np.ndarray, pd.Series | None]:

    stability = None
    if use_selection:
        stability = (sel_count / (kfolds * repeats)).sort_values(ascending=False)
        logger.info("selection stability (top 30):\n%s", stability.head(30).to_string())
        stable = stability[stability >= 0.8].index.tolist()
        logger.info("%d features selected in >=80%% of fits", len(stable))

    df_y = df_train["is_fail"].copy()
    df_X = df_train[features].copy()
    y_hold = df_test["is_fail"].copy()
    X_hold = df_test[features].copy()

    feats = select_features(df_X, df_y) if use_selection else list(features)
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
    res.to_csv(out / 'cv_metrics_xgb1.csv')
    return final, feats, p_hold, stability


def evaluate(df_train: pd.DataFrame, df_test: pd.DataFrame, oof: np.ndarray,
             p_hold: np.ndarray, out: Path) -> dict:

    df_y = df_train["is_fail"].copy()
    y_hold = df_test["is_fail"].copy()

    oof_mean = oof.mean(axis=0)
    thr = tune_threshold(df_y, oof_mean, criterion='mcc')
    summary_oof = classification_summary(df_y, oof_mean, thr)
    summary_oof.to_csv(out / 'summary_oof.csv')
    summary_hold = classification_summary(y_hold, p_hold, thr)
    summary_hold.to_csv(out / 'summary_hold.csv')
    logger.info(summary_oof)
    logger.info(summary_hold)
    save_pr_curve(y_hold, p_hold, out / 'pr_curve_holdout.png')
    save_confusion_heatmap(y_hold, p_hold, thr, out / 'conf_heatmap.png')

    q = float(summary_oof[['tp', 'fp']].sum() / len(df_y))
    r_value = recall_at_flagrate(y_hold, p_hold, q=q)
    logger.info(f'recall_at_flagrate: {r_value}')

    pred_hold_stats = pd.Series(p_hold).describe()
    pred_hold_stats.to_csv(out / 'pred_hold_stats.csv')
    oof_mean_stats = pd.Series(oof_mean).describe()
    oof_mean_stats.to_csv(out / 'oof_mean_stats.csv')

    pr_oof_hold = operating_points(df_y, oof_mean)
    pr_oof_hold.to_csv(out / 'pr_oof_hold.csv')
    return {
        "threshold": float(thr),
        "holdout_aucpr": float(average_precision_score(y_hold, p_hold)),
        "holdout_rocauc": float(roc_auc_score(y_hold, p_hold)),
        "holdout_brier": float(brier_score_loss(y_hold, p_hold)),
        "holdout_recall": float(summary_hold["recall"]),
        "holdout_precision": float(summary_hold["precision"]),
        "flagrate_recall": float(r_value[0]),
    }


def main(argv=None):
    if argv is None:
        argv = [] if "ipykernel" in sys.modules else sys.argv[1:]
    args = parse_args(argv)
    global logger
    logger = setup_logging(logfile=LOGS / "ml.log")
    logger.info(f'[train_xgb] start | selection={args.use_selection} '
                f'repeats={args.repeats}')

    cfg = load_config()
    if args.overrides:
        overrides = parse_overrides(args.overrides)
        cfg = cfg.with_model_overrides(overrides)

    engine = get_engine()
    xgb_params = cfg.model.model_dump()
    run_dir, run_meta = make_run(
        config={"script": "train_xgb", "use_selection": args.use_selection,
                "repeats": args.repeats, "kfolds": cfg.pipeline.kfolds,
                "tail_n": cfg.pipeline.tail_n, "holdout_n": cfg.pipeline.holdout_n,
                "model": xgb_params, "random_state": cfg.pipeline.seed},
        run_name=args.run_name or ("xgb_sel" if args.use_selection else "xgb_base"),
        note=args.note,
    )

    df_train, df_test, features, fingerprint = load_data(
        engine, cfg.pipeline.holdout_n, cfg.pipeline.tail_n)
    save_splits(run_dir, df_train, df_test)

    res, oof, sel_count = run_cv(df_train, features, xgb_params, cfg.pipeline.seed,
                                 kfolds=cfg.pipeline.kfolds,
                                 repeats=args.repeats,
                                 use_selection=args.use_selection)

    final, feats, p_hold, stability = refit_final(
        df_train, df_test, features, oof, res, xgb_params, sel_count,
        kfolds=cfg.pipeline.kfolds, repeats=args.repeats,
        use_selection=args.use_selection, random=cfg.pipeline.seed, out=run_dir)

    extra = set(feats) - set(features)
    if extra:
        raise ValueError(f"selected features outside registry-active set: {sorted(extra)}")

    # the memorization point: snapshot exactly what the final model saw
    snapshot_id = write_gold_snapshot(
        df_train[[schema.KEY_COL, *feats, "is_fail"]],
        engine,
        config={"cutoff": str(CUTOFF), "tail_n": cfg.pipeline.tail_n,
                "holdout_n": cfg.pipeline.holdout_n, "seed": cfg.pipeline.seed,
                "use_selection": args.use_selection},
    )

    save_features(run_dir, feats, stability)
    holdout_metrics = evaluate(df_train, df_test, oof, p_hold, out=run_dir)

    shap_summary = save_shap_plots(
        model=final,
        X=df_train[feats],
        output_dir=run_dir / "shap",
    )
    logger.info("Top SHAP features:\n%s", shap_summary.head(15).to_string())

    append_index(run_dir, {
        "run_name": args.run_name, "note": args.note,
        "git_sha": run_meta["git_sha"],
        "data": fingerprint["raw"].get("secom.data", "")[:16],
        "snapshot_id": snapshot_id,
        "cv_aucpr_mean": float(res["aucpr"].mean()),
        "cv_aucpr_std": float(res["aucpr"].std()),
        **holdout_metrics,
    })

    logger.info(f'[train_xgb] end')


if __name__ == "__main__":
    main()