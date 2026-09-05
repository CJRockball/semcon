# %%
"""SARIMAX experiments: daily wafer volume (seasonality), 50-wafer fail
fraction (ARIMA grid + staged exogenous tests), and channel s468 drift
detection (forecast-interval breaches).

Phase discipline: model selection (grids, AICc) and rolling-origin evaluation
happen inside Phase I only; the holdout is forecast ONCE per final model.
The tail stays out of scope: it is a measurement-protocol change (see the
spc.py protocol charts), not something a value-based model can anticipate.

Window grids restart at the phase boundary: Phase-I windows cover rows
[0, i_hold), Phase-II windows [i_hold, i_tail). A global grid would straddle
the boundary.

Runs register in artifacts/index_monitor.csv (surveillance table), like spc.

Post-migration note: the DB's target is raw -1/1; load_data recodes to 0/1
once here, so every downstream consumer sees the same convention.
Channel and engineered-feature identifiers are registry s-names; artifact
filenames derive from them directly (s468_series.png etc.), so the
notebook's section titles and the run folder always agree.
"""

from __future__ import annotations

import argparse
import logging
import warnings

import matplotlib

matplotlib.use("Agg")  # artifacts over inline display - files are the product
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from semcon import schema, tracking
from semcon.config import BLOCK5_FIRST, CLIQUE_14, CLIQUE_23, load_config
from semcon.db import data_fingerprint, get_engine
from semcon.extract import extract
from semcon.paths import ARTIFACTS, LOGS
from semcon.utils import setup_logging

logger = logging.getLogger("semcon")

MONITOR_INDEX = ARTIFACTS / "index_monitor.csv"
ARIMA_ORDERS = [(1, 0, 0), (0, 1, 1), (1, 1, 0)]  # declared grid, ~2 params max
SEASONAL_MODELS = {  # declared candidates for the daily-volume seasonality test
    "ar1": ((1, 0, 0), None),
    "sar7": ((1, 0, 0), (1, 0, 0, 7)),  # weekly loading rhythm (hypothesized)
    "sar14": ((1, 0, 0), (1, 0, 0, 14)),  # biweekly (exploratory)
}
# Pre-registered hypothesis (EDA): missingness HALVES the fail rate, so the
# coefficient on every f_miss_* regressor should be NEGATIVE.
EXOG_CANDIDATES = ["f_miss_clq23", "f_miss_clq14", "f_row_missing_rate"]
EXOG_EXPLORATORY = ["s468"]  # does the strongest value-drifter co-move with fails?
DRIFT_CHANNEL = "s468"  # top delta in spc_screening.csv
# after the other constants; BLOCK5 is derived, not duplicated
BLOCK5 = [f"{schema.SENSOR_PREFIX}{n:03d}" for n in range(BLOCK5_FIRST, 591)]

# ---------------------------------------------------------------- data


def load_data(holdout_n: int, tail_n: int) -> tuple[pd.DataFrame, int, int]:
    """Wide frame from the DB, sorted once; split positions are computed here.

    Post-migration: extract() returns raw -1/1 target; recoded to 0/1 once.
    """
    engine = get_engine()
    df = extract(engine).sort_values(schema.TIME_COL).reset_index(drop=True)
    df["target"] = df["target"].eq(1).astype("int8")
    df = add_protocol_features(df)
    fp = data_fingerprint(engine)
    n = len(df)
    i_hold = n - tail_n - holdout_n
    i_tail = n - tail_n
    logger.info(
        f"[sarimax] data: {n} rows | phase I [0:{i_hold}] "
        f"holdout [{i_hold}:{i_tail}] tail [{i_tail}:{n}] | "
        f"fingerprint {fp}"
    )
    return df, i_hold, i_tail


def add_protocol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Missingness/data-quality features from the raw wide frame.

    Same definitions as spc.add_protocol_features — promote to a shared helper
    (feature_eng) at the next cleanup; duplicated here so this module doesn't
    import spc.py's module-level matplotlib.use("Agg") side effect.
    """
    df = df.copy()

    def _clique_any_nan(members: list[str]) -> pd.Series:
        present = [c for c in members if c in df.columns]
        assert present, f"none of {members} present — registry or schema drifted"
        return df[present].isna().any(axis=1)

    df["f_miss_clq14"] = _clique_any_nan(CLIQUE_14).astype("int8")
    df["f_miss_clq23"] = _clique_any_nan(CLIQUE_23).astype("int8")
    df["f_miss_block5"] = df[BLOCK5].isna().any(axis=1).astype("int8")
    raw = [c for c in df.columns if c.startswith(schema.SENSOR_PREFIX) and c[1:].isdigit()]
    df["f_row_missing_rate"] = df[raw].isna().mean(axis=1).astype("float32")
    return df


# ---------------------------------------------------------------- series builders


def windowed_mean(
    df: pd.DataFrame, cols: list[str], start: int, end: int, window: int
) -> pd.DataFrame:
    """Windowed mean over rows [start, end); the grid restarts at `start`.

    Same construction as spc.windowed_rate (mean over the ACTUAL n per window).
    Promote to a shared helper at the next cleanup; kept local so this module
    does not import spc.py's module-level matplotlib.use("Agg") twice.
    """
    s = df.iloc[start:end][list(cols)].astype(float)
    return s.groupby(np.arange(len(s)) // window).mean()


def fail_fraction(df: pd.DataFrame, i_hold: int, window: int = 50) -> pd.Series:
    """Fail fraction per `window` wafers, Phase I - the p-chart series."""
    return windowed_mean(df, ["target"], 0, i_hold, window)["target"]


def daily_volume(df: pd.DataFrame, i_hold: int) -> pd.DataFrame:
    """Wafers measured per calendar day, Phase I only.

    Timestamps are test/measurement times: this is the test floor's takt
    rhythm (weekday/weekend loading), a proxy for the production rhythm.
    Empty days are real zero-production observations and stay in.
    """
    ph1 = df.iloc[:i_hold][["timestamp", "target"]].copy()
    ph1["timestamp"] = pd.to_datetime(ph1["timestamp"])
    daily = (
        ph1.groupby(pd.Grouper(key="timestamp", freq="D"))
        .agg(
            total_fail=("target", "sum"),
            fail_freq=("target", "mean"),
            sample_size=("timestamp", "count"),
        )
        .reset_index()
    )
    logger.info(
        f"daily volume: {len(daily)} days, "
        f"{daily['sample_size'].min()}-{daily['sample_size'].max()} wafers/day"
    )
    return daily


# ---------------------------------------------------------------- plot helpers


def save_series_fig(x, y, title: str, out_png) -> None:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(x, y, marker="o", linestyle="None", ms=4)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_acf_pacf_fig(data, lag: int, title: str, out_png) -> None:
    data = pd.Series(np.asarray(data).squeeze()).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    plot_acf(data, lags=lag, ax=axes[0])
    plot_pacf(data, lags=lag, ax=axes[1])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def save_forecast_fig(
    train: pd.Series, actual: pd.Series, fc_mean, fc_lo, fc_hi, title: str, out_png
) -> np.ndarray:
    """Phase-I series + Phase-II forecast with 95% interval; returns breach mask."""
    actual = np.asarray(actual, dtype=float)
    fc_mean = np.asarray(fc_mean, dtype=float)
    tr_x = np.arange(len(train))
    fc_x = np.arange(len(train), len(train) + len(fc_mean))
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(tr_x, train, ".", ms=4, label="phase I (fit)")
    ax.plot(fc_x, actual, "o", ms=6, color="tab:green", label="phase II actual")
    ax.plot(fc_x, fc_mean, "x--", color="tab:red", label="forecast")
    ax.fill_between(fc_x, fc_lo, fc_hi, color="tab:red", alpha=0.15, label="95% interval")
    ax.axvline(len(train) - 0.5, color="k", ls=":", lw=1)
    breach = (actual < fc_lo) | (actual > fc_hi)
    if breach.any():
        ax.plot(fc_x[breach], actual[breach], "o", mfc="none", mec="red", ms=12, label="breach")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return breach


# ---------------------------------------------------------------- model helpers


def fit_arima(series, order, exog=None):
    """One ARIMA fit with the repo trend rule, warnings silenced.

    d lives in the order (never difference by hand). Constant only when d == 0;
    with d >= 1 a constant would be a secular drift we have no reason to
    believe in.
    """
    trend = "c" if order[1] == 0 else "n"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return ARIMA(series, exog=exog, order=order, trend=trend).fit()


def fit_orders(series: pd.Series, orders: list, label: str) -> tuple[pd.DataFrame, dict]:
    """Fit the declared grid on the LEVEL series, AICc-ranked (small n)."""
    table, models = [], {}
    for order in orders:
        try:
            res = fit_arima(series, order)
        except Exception as e:
            logger.warning(f"ARIMA{order} failed: {e}")
            continue
        models[order] = res
        table.append({"order": order, "aic": res.aic, "aicc": res.aicc, "bic": res.bic})
    out = pd.DataFrame(table).sort_values("aicc").reset_index(drop=True)
    logger.info(f"ARIMA grid ({label}, phase I, ranked by AICc):\n{out.to_string(index=False)}")
    return out, models


def mase(y_true, y_pred, scale: float) -> float:
    """MAE / in-sample naive scale; < 1 beats the no-model forecaster."""
    return float(
        np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))) / scale
    )


def rolling_origin(
    y: pd.Series, candidates: dict, min_train: int = 16, horizon: int = 2, step: int = 2
) -> pd.DataFrame:
    """Rolling-origin evaluation INSIDE the given series (phase I only).

    candidates: {name: (order, exog_array_or_None)}; persistence and mean
    baselines are always added. The time-series analog of repeated CV:
    repeat where repetition is legitimate, spend the holdout once.
    """
    y = np.asarray(y, dtype=float)
    scale = np.abs(np.diff(y)).mean()
    errs = {}
    for t in range(min_train, len(y) - horizon + 1, step):
        fc = {"persistence": np.repeat(y[t - 1], horizon), "mean": np.repeat(y[:t].mean(), horizon)}
        for name, (order, exog) in candidates.items():
            try:
                ex_tr = exog[:t] if exog is not None else None
                ex_fc = exog[t : t + horizon] if exog is not None else None
                fit = fit_arima(y[:t], order, exog=ex_tr)
                fc[name] = np.asarray(fit.forecast(horizon, exog=ex_fc))
            except Exception as e:
                logger.warning(f"rolling origin t={t} {name} failed: {e}")
        actual = y[t : t + horizon]
        for name, f in fc.items():
            errs.setdefault(name, []).extend(np.abs(actual - f))
    out = pd.DataFrame(
        [
            {
                "model": k,
                "mae": float(np.mean(v)),
                "mase": float(np.mean(v) / scale) if scale > 0 else np.nan,
                "n": len(v),
            }
            for k, v in errs.items()
        ]
    )
    out = out.sort_values("mase").reset_index(drop=True)
    logger.info(f"rolling-origin MASE (phase I):\n{out.to_string(index=False)}")
    return out


def forecast_holdout(
    train: pd.Series, hold: pd.Series, order, exog_tr, exog_hold, label: str, figs
) -> dict:
    """The one-shot holdout evaluation: refit on all of phase I, forecast once."""
    final = fit_arima(train, order, exog=exog_tr)
    fc = final.get_forecast(len(hold), exog=exog_hold)
    mean = np.asarray(fc.predicted_mean, dtype=float)
    ci = np.asarray(fc.conf_int(alpha=0.05))
    breach = save_forecast_fig(
        train,
        hold,
        mean,
        ci[:, 0],
        ci[:, 1],
        f"{label}: phase-I fit, holdout forecast",
        figs / f"{label}_forecast_holdout.png",
    )
    scale = np.abs(np.diff(np.asarray(train, dtype=float))).mean()
    rows = [
        {
            "model": label,
            "mase": mase(hold, mean, scale),
            "coverage": float(np.mean((hold >= ci[:, 0]) & (hold <= ci[:, 1]))),
            "breaches": int(breach.sum()),
        },
        {
            "model": "persistence",
            "mase": mase(hold, np.repeat(train.iloc[-1], len(hold)), scale),
            "coverage": np.nan,
            "breaches": np.nan,
        },
        {
            "model": "mean",
            "mase": mase(hold, np.repeat(train.mean(), len(hold)), scale),
            "coverage": np.nan,
            "breaches": np.nan,
        },
    ]
    out = pd.DataFrame(rows)
    logger.info(f"holdout forecast ({label}):\n{out.to_string(index=False)}")
    return out


# ---------------------------------------------------------------- A: seasonal volume


def seasonal_volume(df: pd.DataFrame, i_hold: int, figs, summaries, lag: int = 20) -> pd.DataFrame:
    """Daily wafer volume: series, ACF, declared seasonal candidates by AICc."""
    daily = daily_volume(df, i_hold)
    vol = daily["sample_size"]
    save_series_fig(
        daily["timestamp"], vol, "wafers measured per day (phase I)", figs / "volume_series.png"
    )
    save_acf_pacf_fig(vol, lag, "daily volume", figs / "volume_acf_pacf.png")
    save_acf_pacf_fig(
        vol.diff(), lag, "daily volume, first difference", figs / "volume_diff_acf_pacf.png"
    )

    rows, models = [], {}
    for name, (order, sorder) in SEASONAL_MODELS.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = SARIMAX(vol, order=order, seasonal_order=sorder, trend="c").fit()
        except Exception as e:
            logger.warning(f"{name} failed: {e}")
            continue
        models[name] = res
        rows.append(
            {
                "model": name,
                "order": str(order),
                "seasonal": str(sorder),
                "aic": res.aic,
                "aicc": res.aicc,
                "bic": res.bic,
            }
        )
        (summaries / f"volume_{name}.txt").write_text(str(res.summary()))
    table = pd.DataFrame(rows).sort_values("aicc").reset_index(drop=True)
    table.to_csv(summaries.parent / "seasonal_candidates.csv", index=False)
    logger.info(f"daily-volume seasonality candidates (AICc):\n{table.to_string(index=False)}")
    best = table.iloc[0]["model"]
    logger.info(f"seasonal winner: {best}")
    save_acf_pacf_fig(
        models[best].resid, lag, f"residuals: {best}", figs / "volume_resid_acf_pacf.png"
    )
    return table


# ---------------------------------------------------------------- B: fail fraction


def exog_tests(frac: pd.Series, exog_df: pd.DataFrame, order) -> pd.DataFrame:
    """One extra parameter at a time on the frozen order (26-point budget).

    Pre-registered sign: EDA says missingness halves the fail rate, so f_miss_*
    coefficients should be NEGATIVE.
    """
    base = fit_arima(frac, order)
    rows = [
        {"exog": "(none)", "aicc": base.aicc, "coef": np.nan, "pvalue": np.nan, "sign_check": ""}
    ]
    for col in exog_df.columns:
        try:
            res = fit_arima(frac, order, exog=exog_df[[col]])
        except Exception as e:
            logger.warning(f"exog {col} failed: {e}")
            continue
        coef = float(res.params.get(col, np.nan))
        sign = ""
        if col.startswith("f_miss"):
            sign = "ok (neg, as predicted)" if coef < 0 else "WRONG SIGN"
        rows.append(
            {
                "exog": col,
                "aicc": res.aicc,
                "coef": coef,
                "pvalue": float(res.pvalues.get(col, np.nan)),
                "sign_check": sign,
            }
        )
    out = pd.DataFrame(rows)
    logger.info(
        f"exog tests on frozen ARIMA{order} (AICc, one regressor each):\n"
        f"{out.to_string(index=False)}"
    )
    return out


def failrate_arima(
    df: pd.DataFrame,
    i_hold: int,
    i_tail: int,
    figs,
    summaries,
    window: int = 50,
    lag: int = 10,
    min_train: int = 16,
    horizon: int = 2,
) -> dict:
    """Fail fraction: grid -> staged exog -> rolling origin -> one-shot holdout."""
    frac = fail_fraction(df, i_hold, window)
    frac_hold = windowed_mean(df, ["target"], i_hold, i_tail, window)["target"]
    save_series_fig(
        frac.index,
        frac,
        f"fail fraction per {window} wafers (phase I)",
        figs / "failfrac_series.png",
    )
    save_acf_pacf_fig(frac, lag, "fail fraction", figs / "failfrac_acf_pacf.png")

    # 1. declared grid on phase I, AICc-ranked
    grid, models = fit_orders(frac, ARIMA_ORDERS, "fail fraction")
    grid.to_csv(summaries.parent / "arima_grid_failfrac.csv", index=False)
    best_order = grid.iloc[0]["order"]
    best = models[best_order]
    (summaries / "failfrac_best.txt").write_text(str(best.summary()))
    save_acf_pacf_fig(
        best.resid, lag, f"residuals: ARIMA{best_order}", figs / "failfrac_resid_acf_pacf.png"
    )

    # 2. staged exog tests on the frozen order (one extra parameter each)
    exog_p1 = windowed_mean(df, EXOG_CANDIDATES + EXOG_EXPLORATORY, 0, i_hold, window)
    assert len(exog_p1) == len(frac), "exog/frac window misalignment"
    exog_table = exog_tests(frac, exog_p1, best_order)
    exog_table.to_csv(summaries.parent / "exog_tests.csv", index=False)

    # an exog model must beat the univariate AICc by > 2 to earn the parameter
    base_aicc = float(exog_table.iloc[0]["aicc"])
    exog_best = exog_table.iloc[1:].sort_values("aicc").iloc[0]
    use_exog = bool(exog_best["aicc"] < base_aicc - 2)
    logger.info(
        f"exog decision: {'use ' + exog_best['exog'] if use_exog else 'stay univariate'} "
        f"(base AICc {base_aicc:.1f}, best exog {exog_best['aicc']:.1f})"
    )

    # 3. rolling-origin evaluation inside phase I, vs baselines
    candidates = {"arima": (best_order, None)}
    if use_exog:
        col = exog_best["exog"]
        candidates[f"arima+{col}"] = (best_order, exog_p1[[col]].to_numpy())
    ro = rolling_origin(frac, candidates, min_train=min_train, horizon=horizon)
    ro.to_csv(summaries.parent / "rolling_origin_mase.csv", index=False)

    # 4. one-shot holdout forecast with the rolling-origin winner
    ro_models = ro[~ro["model"].isin(["persistence", "mean"])]
    winner = ro_models.iloc[0]["model"]
    order_w, exog_w = candidates[winner]
    exog_hold = None
    if exog_w is not None:
        col = winner.split("+", 1)[1]
        exog_hold = windowed_mean(df, [col], i_hold, i_tail, window).to_numpy()
    hold = forecast_holdout(frac, frac_hold, order_w, exog_w, exog_hold, "failfrac", figs)
    hold.to_csv(summaries.parent / "holdout_metrics_failfrac.csv", index=False)

    return {
        "best_order": best_order,
        "exog_winner": exog_best["exog"] if use_exog else "(none)",
        "exog_coef": float(exog_best["coef"]),
        "exog_p": float(exog_best["pvalue"]),
        "ro_winner": winner,
        "ro_mase": float(ro_models.iloc[0]["mase"]),
        "holdout_mase": float(hold.iloc[0]["mase"]),
        "holdout_coverage": float(hold.iloc[0]["coverage"]),
    }


# ---------------------------------------------------------------- C: channel drift


def channel_drift(
    df: pd.DataFrame,
    i_hold: int,
    i_tail: int,
    figs,
    summaries,
    channel: str = DRIFT_CHANNEL,
    window: int = 50,
    lag: int = 10,
) -> dict:
    """Channel s468 as TARGET: forecast-based drift detection.

    SPC answered 'did it move between two frozen windows' (descriptive).
    This asks the prospective question: fit phase I, forecast the holdout
    with intervals, and count sustained breaches. Here forecast FAILURE is
    the detection signal - a high MASE on the holdout is the drift verdict,
    not a modeling embarrassment.
    """
    ch = windowed_mean(df, [channel], 0, i_hold, window)[channel]
    ch_hold = windowed_mean(df, [channel], i_hold, i_tail, window)[channel]
    save_series_fig(
        ch.index,
        ch,
        f"{channel} windowed mean per {window} wafers (phase I)",
        figs / f"{channel}_series.png",
    )
    save_acf_pacf_fig(ch, lag, f"{channel}", figs / f"{channel}_acf_pacf.png")

    grid, models = fit_orders(ch, ARIMA_ORDERS, f"{channel}")
    grid.to_csv(summaries.parent / f"arima_grid_{channel}.csv", index=False)
    best_order = grid.iloc[0]["order"]
    (summaries / f"{channel}_best.txt").write_text(str(models[best_order].summary()))

    hold = forecast_holdout(ch, ch_hold, best_order, None, None, f"{channel}", figs)
    hold.to_csv(summaries.parent / f"holdout_metrics_{channel}.csv", index=False)
    breaches = int(hold.iloc[0]["breaches"])
    result = {
        "channel": channel,
        "order": str(best_order),
        "holdout_mase": float(hold.iloc[0]["mase"]),
        "breaches": breaches,
        "holdout_points": len(ch_hold),
        "breach_share": round(breaches / len(ch_hold), 3),
    }
    logger.info(f"{channel} drift verdict: {result}")
    return result


# ---------------------------------------------------------------- cli / main


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SECOM SARIMAX experiments")
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--lag", type=int, default=20)
    p.add_argument("--min-train", type=int, default=16)
    p.add_argument("--horizon", type=int, default=2)
    p.add_argument("--note", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(logfile=LOGS / "ml.log")
    logger.info(f"[sarimax] start | window={args.window} note={args.note}")
    cfg = load_config()
    df, i_hold, i_tail = load_data(cfg.pipeline.holdout_n, cfg.pipeline.tail_n)

    run, meta = tracking.make_run(
        config={**vars(args), "i_hold": i_hold, "i_tail": i_tail},
        run_name="sarimax",
        note=args.note or "",
    )
    figs = run / "figures"
    summaries = run / "summaries"
    figs.mkdir(exist_ok=True)
    summaries.mkdir(exist_ok=True)
    logger.info(f"run {run.name} | git {meta['git_sha']}")

    seas = seasonal_volume(df, i_hold, figs, summaries, lag=args.lag)
    fail = failrate_arima(
        df,
        i_hold,
        i_tail,
        figs,
        summaries,
        window=args.window,
        min_train=args.min_train,
        horizon=args.horizon,
    )
    drift = channel_drift(df, i_hold, i_tail, figs, summaries, window=args.window)

    tracking.append_index(
        run,
        {
            "type": "sarimax",
            "window": args.window,
            "note": args.note or "",
            "seasonal_winner": seas.iloc[0]["model"],
            "failfrac_order": str(fail["best_order"]),
            "exog_winner": fail["exog_winner"],
            "exog_coef": round(fail["exog_coef"], 4),
            "exog_p": round(fail["exog_p"], 4),
            "ro_winner": fail["ro_winner"],
            "ro_mase": round(fail["ro_mase"], 3),
            "holdout_mase": round(fail["holdout_mase"], 3),
            "holdout_coverage": round(fail["holdout_coverage"], 3),
            "s468_breach_share": drift["breach_share"],
        },
        index_file=MONITOR_INDEX,
    )
    logger.info(f"[sarimax] done | artifacts -> {run}")


def _in_ipykernel() -> bool:
    """True inside a Jupyter kernel; import-free so the script env needs no IPython."""
    return "ipykernel" in __import__("sys").modules


# %%
if __name__ == "__main__" and not _in_ipykernel():
    main([])

# %% Interactive cells:
# df, i_hold, i_tail = load_data(231, 27)
# then call seasonal_volume / failrate_arima / channel_drift with your own
# figs/summaries dirs, or just run main()
