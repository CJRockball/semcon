"""Tests for the SARIMAX module: the design decisions, not the statistics.

Each test pins one contract we would regret breaking: actual-n windowing,
grid restart at the phase boundary, empty days kept in the volume series,
the trend rule, the AICc-ranked grid contract, MASE arithmetic, rolling-origin
baseline behavior, and the pre-registered sign check. Synthetic data only.
"""
import numpy as np
import pandas as pd

from semcon.sarimax import (daily_volume, exog_tests, fit_arima, fit_orders,
                            mase, rolling_origin, save_forecast_fig,
                            windowed_mean)


def test_windowed_mean_actual_n_and_restart():
    df = pd.DataFrame({"a": np.arange(10.0)})
    out = windowed_mean(df, ["a"], start=0, end=10, window=4)
    # last window has 2 wafers, not 4 - the /50 deflation bug class
    assert list(out["a"]) == [1.5, 5.5, 8.5]
    # the grid restarts at `start`, so phase II never straddles the boundary
    out2 = windowed_mean(df, ["a"], start=2, end=10, window=4)
    assert list(out2["a"]) == [3.5, 7.5]


def test_daily_volume_keeps_empty_days():
    ts = ["2024-01-01 08:00", "2024-01-01 09:00", "2024-01-03 10:00"]
    df = pd.DataFrame({"timestamp": ts, "target": [0, 1, 0]})
    daily = daily_volume(df, i_hold=3)
    assert len(daily) == 3                      # Jan 2 is a real zero-wafer day
    assert (daily["sample_size"] == 0).sum() == 1
    assert daily["total_fail"].sum() == 1


def test_fit_arima_trend_rule():
    rng = np.random.default_rng(0)
    s = pd.Series(rng.normal(size=80))
    # constant only when d == 0; with d >= 1 a constant would be a secular drift
    assert "const" in fit_arima(s, (1, 0, 0)).params.index
    assert "const" not in fit_arima(s, (0, 1, 1)).params.index


def test_fit_orders_aicc_sorted():
    rng = np.random.default_rng(2)
    s = pd.Series(rng.normal(size=60))
    table, models = fit_orders(s, [(1, 0, 0), (0, 1, 1), (1, 1, 0)], "test")
    assert set(models) == {(1, 0, 0), (0, 1, 1), (1, 1, 0)}
    assert table["aicc"].is_monotonic_increasing


def test_mase_known_values():
    assert mase([1.0, 2.0, 3.0], [1.0, 2.0, 3.0], scale=2.0) == 0.0
    assert mase([1.0, 2.0], [0.0, 2.0], scale=1.0) == 0.5  # MAE 0.5 over scale 1


def test_rolling_origin_baselines():
    y_const = pd.Series(np.full(30, 0.07))
    out = rolling_origin(y_const, {}, min_train=16, horizon=2, step=2)
    assert set(out["model"]) == {"persistence", "mean"}
    assert (out["mae"] < 1e-12).all()

    # on a unit ramp, persistence lags by one step: |e| = 1 and 2 at h = 1, 2
    y_ramp = pd.Series(np.arange(30.0))
    out = rolling_origin(y_ramp, {}, min_train=16, horizon=2, step=2)
    assert out.loc[out["model"] == "persistence", "mae"].iloc[0] == 1.5


def test_exog_tests_recovers_negative_sign():
    rng = np.random.default_rng(1)
    n = 120
    exog = pd.DataFrame({"f_miss_clq14": rng.normal(size=n)})
    y = pd.Series(0.07 - 0.5 * exog["f_miss_clq14"].to_numpy()
                  + rng.normal(scale=0.01, size=n))
    tab = exog_tests(y, exog, (1, 0, 0))
    assert tab["exog"].head(1).item() == "(none)"   # baseline is row zero
    row = tab.loc[tab["exog"] == "f_miss_clq14"].squeeze()
    assert row["coef"] < 0
    assert row["sign_check"] == "ok (neg, as predicted)"


def test_save_forecast_fig_breach_mask(tmp_path):
    train = pd.Series(np.linspace(5, 6, 27))
    actual = pd.Series([5.5, 5.6, 9.0, 5.4, 5.5])  # index 2 breaches
    out = tmp_path / "fc.png"
    breach = save_forecast_fig(train, actual, np.full(5, 5.5),
                               np.full(5, 5.0), np.full(5, 6.0), "t", out)
    assert list(breach) == [False, False, True, False, False]
    assert out.exists() and out.stat().st_size > 0
