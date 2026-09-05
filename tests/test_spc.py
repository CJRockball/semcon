"""First test module for the repo: the SPC math that must be exactly right."""

import numpy as np
import pandas as pd

from semcon.spc import binomial_limits, compute_limits, ewma, moving_range, screening, we_rules


def test_robust_limits_resist_outliers():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=1000))
    x.iloc[np.arange(0, 1000, 20)] = 50.0  # 5% gross outliers, scattered
    rob = compute_limits(pd.DataFrame({"a": x})).loc["a", "sigma"]
    cls = compute_limits(pd.DataFrame({"a": x}), method="classic").loc["a", "sigma"]
    assert 0.8 < rob < 1.5
    assert cls > 2 * rob


def test_screening_flags_shifted_feature_only():
    rng = np.random.default_rng(1)
    n1, n_hold, n_tail = 1309, 231, 27  # mirror the real three-zone split
    n2 = n_hold + n_tail
    df = pd.DataFrame(
        {
            "calm": rng.normal(size=n1 + n2),
            "drift": np.r_[rng.normal(size=n1), rng.normal(2.0, 1.0, n2)],
        }
    )
    lim = compute_limits(df.iloc[:n1])
    scr = screening(df, ["calm", "drift"], lim, i_hold=n1, i_tail=n1 + n_hold)
    assert scr.loc["drift", "delta"] > 0.08
    assert scr.loc["calm", "delta"] < 0.03
    # EWMA catches the persistent shift that 3-sigma limits dilute
    assert scr.loc["drift", "ewma_delta"] > 0.5


def test_screening_empty_phase_returns_nan_not_crash():
    rng = np.random.default_rng(2)
    df = pd.DataFrame({"a": rng.normal(size=100)})
    lim = compute_limits(df)
    scr = screening(df, ["a"], lim, i_hold=100, i_tail=100)  # no phase II, no tail
    assert np.isnan(scr.loc["a", "ooc_p2"])
    assert np.isnan(scr.loc["a", "miss_tail"])


def test_mr_boundary_continuity():
    mr = moving_range(pd.Series([1.0, 2.0, 4.0, 3.0]))
    assert np.isnan(mr.iloc[0])
    assert list(mr.iloc[1:]) == [1.0, 2.0, 1.0]


def test_we_rules_fire_on_constructed_sequences():
    x = pd.Series([0.0] * 10 + [4.0] + [0.0] * 10 + [0.5] * 8)
    wr = we_rules(x, center=0.0, sigma=1.0)
    assert wr["r1"].iloc[10]
    assert wr["r4"].iloc[-1] and not wr["r4"].iloc[-9]
    wr2 = we_rules(pd.Series([0.0, 2.5, 0.1, 2.6]), center=0.0, sigma=1.0)
    assert wr2["r2"].iloc[3]
    assert not wr2["r2"].iloc[2]  # triggering point only, not every window member


def test_ewma_signals_persistent_shift():
    x = pd.Series(np.r_[np.zeros(50), np.full(50, 2.0)])
    z, lcl, ucl = ewma(x, lam=0.2, center=0.0, sigma=1.0)
    assert z.iloc[-1] > ucl
    assert z.iloc[25] < ucl


def test_binomial_lcl_clips_at_zero():
    lcl, ucl = binomial_limits(p0=0.067, n=np.array([50]))
    assert lcl[0] == 0.0
    assert ucl[0] > 0.067


def test_plot_overview_smoke(tmp_path):
    from semcon.spc import plot_overview

    rng = np.random.default_rng(4)
    screen = pd.DataFrame(
        {
            "degenerate": False,
            "ooc_p1": rng.uniform(0, 0.02, 12),
            "ooc_p2": rng.uniform(0, 0.3, 12),
            "delta": rng.uniform(-0.01, 0.3, 12),
        },
        index=[str(i) for i in range(12)],
    )
    out = tmp_path / "overview.png"
    plot_overview(screen, {"3", "7"}, out)
    assert out.exists() and out.stat().st_size > 0
