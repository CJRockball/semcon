# %%
"""Statistical process control for the SECOM pipeline.

Phase I  = CV pool (rows before the time holdout): limits computed and frozen here.
Phase II = holdout + tail: frozen limits applied unchanged; the alarm-rate jump is drift.
Unlike model training, the tail rows are INCLUDED - a monitoring system exists to
catch exactly that regime change.

Post-migration (2026-09-01): data comes from SQLite via extract() - the raw 590
sensors, not the retired 257-column value view. A monitoring net should cover
channels the model never sees, so the screening denominator widens by design
(degenerate and high-NaN sensors are screened and flagged, not dropped upstream).
extract() returns raw channels only; the four protocol features (row missing
rate, clique/block dropout indicators) are computed here on the raw frame from
the dataset constants in config.py (CLIQUE_14/23, BLOCK5_FIRST). Schema
constants are the only name source; legacy names in input files (old
features.json, old EDA master table) are mapped by _current_name().

Pipeline: screening pass over all sensor columns -> spc_screening.csv
-> overview scatter -> deterministic 2x2 showcase selection -> I-MR/EWMA charts
-> protocol-feature charts -> rolling fail-rate p-chart.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: save figures, never plt.show()
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from semcon import schema, tracking
from semcon.config import BLOCK5_FIRST, CLIQUE_14, CLIQUE_23, load_config
from semcon.db import data_fingerprint, get_engine
from semcon.extract import extract
from semcon.paths import ARTIFACTS, LOGS
from semcon.snapshots import write_gold_snapshot
from semcon.utils import setup_logging

logger = logging.getLogger("semcon")


D2, D4 = 1.128, 3.267  # I-MR constants, subgroup n = 2
MAD_TO_SIGMA = 1.4826  # consistency constant for normal data

# Block 5 membership: raw columns BLOCK5_FIRST..590 (legacy 542-589), the late
# station group. Clique membership lives in config.py (CLIQUE_14/CLIQUE_23).
BLOCK5 = [f"{schema.SENSOR_PREFIX}{n:03d}" for n in range(BLOCK5_FIRST, 591)]

# Engineered protocol features, computed by add_protocol_features(); charted
# by plot_protocol, never screened as values.
RATE_FEATURES = ["f_miss_clq14", "f_miss_clq23", "f_miss_block5"]

# Fallback showcase pool: the 100%-stability sensors from the xgb_sel run
# (legacy ['21', '510', '59', '348', '431', '28', '129', '103']).
# f_miss_clq14 also hit 100% but is charted with the protocol features.
DEFAULT_STABLE = ["s022", "s511", "s060", "s349", "s432", "s029", "s130", "s104"]


def _current_name(name: str) -> str:
    """Accept legacy names (0-based ints, unprefixed miss_*) -> modern names.

    Legacy sensor '59' -> 's060'; 'miss_clq14' -> 'f_miss_clq14'; modern
    s-/f- names pass through. Unknown strings pass through unchanged.
    """
    name = str(name)
    if name.startswith((schema.SENSOR_PREFIX, "f_")):
        return name
    if name.isdigit():
        return f"{schema.SENSOR_PREFIX}{int(name) + 1:03d}"
    if name.startswith("miss_") or name == "row_missing_rate":
        return f"f_{name}"
    return name


def raw_sensor_columns(df: pd.DataFrame) -> list[str]:
    """All raw sensor columns (sNNN) in the frame - the full monitoring net.

    Registry-free by design: screening covers sensors the value pipeline
    retired (constants, high-NaN), because monitoring exists for the channels
    the model never sees.
    """
    return [
        c
        for c in df.columns
        if c.startswith(schema.SENSOR_PREFIX) and c[len(schema.SENSOR_PREFIX) :].isdigit()
    ]


def add_protocol_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the four engineered protocol columns on the raw frame.

    extract() delivers raw channels only. Membership comes from config.py;
    each list is intersected with the frame and asserted non-empty, so a
    schema drift fails loudly instead of charting nonsense.
    """
    sensors = raw_sensor_columns(df)
    df["f_row_missing_rate"] = df[sensors].isna().mean(axis=1)

    for col, members in [
        ("f_miss_clq14", CLIQUE_14),
        ("f_miss_clq23", CLIQUE_23),
        ("f_miss_block5", BLOCK5),
    ]:
        present = [m for m in members if m in df.columns]
        assert present, f"no members of {col} found in the frame"
        df[col] = df[present].isna().any(axis=1).astype("int8")
        logger.info(f"protocol {col}: {len(present)} members, rate {df[col].mean():.3f}")
    return df


# ---------------------------------------------------------------- data


def load_data(
    holdout_n: int, tail_n: int, snap_config: dict
) -> tuple[pd.DataFrame, int, int, dict, str]:
    """Wide frame from SQLite, sorted ONCE, sliced positionally.

    Returns (df, i_hold, i_tail, fingerprint, snapshot_id). The target is
    recoded from raw -1/1 to 0/1 (rates and fail markers assume 0/1), schema
    columns are aliased to timestamp/target, and the protocol features are
    computed on the raw frame. The gold snapshot records exactly the frame
    SPC consumes; data_fingerprint is taken on the database itself.
    """
    engine = get_engine()
    df = extract(engine).sort_values(schema.TIME_COL).reset_index(drop=True)
    fp = data_fingerprint(engine)  # dict: raw table hashes + extract SQL hash

    df = df.rename(columns={schema.TIME_COL: "timestamp", schema.TARGET_COL: "target"})
    df["target"] = df["target"].eq(1).astype("int8")  # raw -1/1 -> 0/1
    df = add_protocol_features(df)

    snapshot_id = write_gold_snapshot(df, engine, config=snap_config)

    n = len(df)
    i_hold = n - tail_n - holdout_n
    i_tail = n - tail_n
    logger.info(
        f"SPC data: {n} rows | phase I [0:{i_hold}] "
        f"holdout [{i_hold}:{i_tail}] tail [{i_tail}:{n}] | "
        f"fingerprint {fp} | snapshot {snapshot_id}"
    )
    return df, i_hold, i_tail, fp, snapshot_id


# ---------------------------------------------------------------- limits & statistics


def moving_range(s: pd.Series) -> pd.Series:
    """MR computed on the full sorted series: continuity across phase boundaries."""
    return s.diff().abs()


def compute_limits(train: pd.DataFrame, method: str = "robust", k: float = 3.0) -> pd.DataFrame:
    """Frozen Phase-I center lines and limits, one row per column.

    robust : median +/- k * 1.4826 * MAD  (survives the outlier-dominated features)
    classic: mean   +/- k * MR-bar / d2   (textbook I-chart, kept for comparison)
    """
    rows = {}
    for col in train.columns:
        x = train[col].dropna()
        mr_bar = float(moving_range(x).mean())
        if len(x) < 25:
            rows[col] = dict(
                center=np.nan,
                sigma=np.nan,
                lcl=np.nan,
                ucl=np.nan,
                mr_bar=mr_bar,
                mr_ucl=np.nan,
                degenerate=True,
            )
            continue
        if method == "robust":
            center = float(x.median())
            sigma = float(MAD_TO_SIGMA * (x - center).abs().median())
        else:
            center = float(x.mean())
            sigma = mr_bar / D2
        degenerate = not np.isfinite(sigma) or sigma <= 0
        rows[col] = dict(
            center=center,
            sigma=sigma,
            lcl=center - k * sigma,
            ucl=center + k * sigma,
            mr_bar=mr_bar,
            mr_ucl=float(D4 * mr_bar),
            degenerate=degenerate,
        )
    out = pd.DataFrame(rows).T
    num = ["center", "sigma", "lcl", "ucl", "mr_bar", "mr_ucl"]
    out[num] = out[num].apply(pd.to_numeric)
    out["degenerate"] = out["degenerate"].astype(bool)
    return out


def ewma(
    x: pd.Series, lam: float, center: float, sigma: float, k: float = 3.0
) -> tuple[pd.Series, float, float]:
    """EWMA with asymptotic limits; NaNs are skipped (state carried)."""
    z = np.empty(len(x))
    prev = center
    for i, v in enumerate(x.to_numpy(dtype=float)):
        if np.isfinite(v):
            prev = lam * v + (1.0 - lam) * prev
        z[i] = prev
    half = k * sigma * np.sqrt(lam / (2.0 - lam))
    return pd.Series(z, index=x.index), center - half, center + half


def screening(
    df: pd.DataFrame,
    cols: list[str],
    limits: pd.DataFrame,
    i_hold: int,
    i_tail: int,
    ewma_lam: float = 0.2,
) -> pd.DataFrame:
    """One row per sensor: frozen-limit alarm rates per phase + missingness rates."""
    phases = {"p1": slice(0, i_hold), "p2": slice(i_hold, len(df)), "tail": slice(i_tail, len(df))}
    rows = {}
    for col in cols:
        lim = limits.loc[col]
        rec = {"degenerate": bool(lim["degenerate"])}
        if lim["degenerate"]:
            rows[col] = rec
            continue
        x = df[col]
        z, e_lcl, e_ucl = ewma(x, ewma_lam, lim["center"], lim["sigma"])
        for tag, sl in phases.items():
            xs, zs = x.iloc[sl], z.iloc[sl]
            if len(xs) == 0:
                rec[f"ooc_{tag}"] = np.nan
                rec[f"ewma_{tag}"] = np.nan
                rec[f"miss_{tag}"] = np.nan
                continue
            n_obs = int(xs.notna().sum())
            alarm = ((xs < lim["lcl"]) | (xs > lim["ucl"])) & xs.notna()
            rec[f"ooc_{tag}"] = float(alarm.sum() / n_obs) if n_obs else np.nan
            rec[f"ewma_{tag}"] = float(((zs < e_lcl) | (zs > e_ucl)).mean())
            rec[f"miss_{tag}"] = 1.0 - n_obs / len(xs)
        rec["delta"] = rec["ooc_p2"] - rec["ooc_p1"]
        rec["ewma_delta"] = rec["ewma_p2"] - rec["ewma_p1"]
        rows[col] = rec
    out = pd.DataFrame(rows).T
    out.index = out.index.map(str)
    num = [c for c in out.columns if c != "degenerate"]
    out[num] = out[num].apply(pd.to_numeric)
    out["degenerate"] = out["degenerate"].astype(bool)
    return out


def we_rules(x: pd.Series, center: float, sigma: float) -> pd.DataFrame:
    """Western Electric rules 1-4. Alarm assigned at the triggering point.

    r1: |z| > 3. r2: 2 of last 3 beyond 2 sigma, same side. r3: 4 of last 5
    beyond 1 sigma, same side. r4: 8 consecutive on the same side of center.
    """
    z = ((x - center) / sigma).to_numpy(dtype=float)
    side = np.sign(z)
    n = len(z)
    r1 = np.isfinite(z) & (np.abs(z) > 3)
    r2 = np.zeros(n, bool)
    r3 = np.zeros(n, bool)
    r4 = np.zeros(n, bool)
    for i in range(n):
        if not np.isfinite(z[i]) or side[i] == 0:
            continue
        w = z[max(0, i - 2) : i + 1]
        s = side[max(0, i - 2) : i + 1]
        m = np.isfinite(w) & (np.abs(w) > 2) & (s == side[i])
        if m.sum() >= 2 and abs(z[i]) > 2:
            r2[i] = True
        w = z[max(0, i - 4) : i + 1]
        s = side[max(0, i - 4) : i + 1]
        m = np.isfinite(w) & (np.abs(w) > 1) & (s == side[i])
        if m.sum() >= 4 and abs(z[i]) > 1:
            r3[i] = True
        j = i
        while j >= 0 and np.isfinite(z[j]) and side[j] == side[i]:
            j -= 1
        if i - j >= 8:
            r4[i] = True
    return pd.DataFrame({"r1": r1, "r2": r2, "r3": r3, "r4": r4}, index=x.index)


def binomial_limits(p0: float, n, k: float = 3.0):
    """P-chart limits; LCL clipped at 0 (cannot signal improvement, only excursions)."""
    sd = np.sqrt(p0 * (1.0 - p0) / np.asarray(n, dtype=float))
    return np.clip(p0 - k * sd, 0.0, None), p0 + k * sd


def windowed_rate(s: pd.Series, window: int) -> pd.DataFrame:
    """Positional windows; x = right edge of each window for plotting."""
    pos = np.arange(len(s))
    grp = s.astype(float).groupby(pos // window)
    return pd.DataFrame(
        {"rate": grp.mean(), "n": grp.size(), "x": grp.apply(lambda v: v.index[-1]).astype(int)}
    )


# ---------------------------------------------------------------- selection


def load_stable_set(path: str | None, min_freq: float) -> set[str]:
    """Stable features from a selection run's features.json (or csv/parquet).

    Accepts {name: freq}, the same nested under 'stability'/'sel_count', a plain
    list, or a table with a frequency column. Values > 1.5 are treated as counts
    and normalized by their max. Legacy names are mapped via _current_name.
    Falls back to DEFAULT_STABLE.
    """
    if path is None:
        return set(DEFAULT_STABLE)
    p = Path(path)
    try:
        if p.suffix == ".json":
            obj = json.loads(p.read_text())
            for key in ("stability", "sel_count", "features"):
                if isinstance(obj, dict) and key in obj:
                    obj = obj[key]
                    break
            if isinstance(obj, list):
                return {_current_name(f) for f in obj}
            vals = {str(k): float(v) for k, v in obj.items()}
        else:
            t = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, index_col=0)
            col = next(c for c in ("stability", "sel_count", "freq") if c in t.columns)
            vals = {str(i): float(v) for i, v in t[col].items()}
        mx = max(vals.values())
        scale = mx if mx > 1.5 else 1.0
        return {_current_name(k) for k, v in vals.items() if v / scale >= min_freq}
    except Exception as e:
        logger.warning(f"could not parse stable set from {p} ({e}); using DEFAULT_STABLE")
        return set(DEFAULT_STABLE)


def select_showcase(
    screen: pd.DataFrame, stable: set[str], delta_min: float, n_showcase: int
) -> tuple[list[str], dict]:
    """Deterministic 2x2: (in stable set) x (Phase-II drift) - the table chooses."""
    s = screen[~screen["degenerate"] & screen["delta"].notna()].copy()
    drift = s["delta"] >= delta_min
    inst = s.index.isin(stable)
    cells = {
        "stable+drift": s[inst & drift].sort_values("delta", ascending=False),
        "stable+calm": s[inst & ~drift].sort_values("delta"),
        "other+drift": s[~inst & drift].sort_values("delta", ascending=False),
    }
    for name, c in cells.items():
        logger.info(f"showcase cell {name}: {len(c)} features -> {list(c.index[:8])}")
    picks = list(cells["stable+drift"].index[:n_showcase])
    picks += [f for f in cells["stable+calm"].index[:2] if f not in picks]
    picks += [f for f in cells["other+drift"].index[:2] if f not in picks]
    if len(picks) < n_showcase:  # top up from the strongest drifters overall
        rest = s.sort_values("delta", ascending=False).index
        picks += [f for f in rest if f not in picks][: n_showcase - len(picks)]
    return picks, cells


def load_master_notes(path: str | None) -> dict[str, str]:
    """Optional EDA master table -> per-feature annotation for chart titles.

    The master table may be indexed by legacy integers (the EDA notebook
    presents the legacy view); keys are mapped via _current_name.
    """
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning(f"master table not found: {p}")
        return {}
    m = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, index_col=0)
    m.index = [_current_name(i) for i in m.index.map(str)]
    notes = {}
    for f in m.index:
        bits = []
        for colname, fmt in [
            ("hedgesg", "g={:.2f}"),
            ("log2vrclip", "log2VR={:.1f}"),
            ("tailenrichment", "tail={:.1f}"),
        ]:
            if colname in m.columns and pd.notna(m.loc[f, colname]):
                bits.append(fmt.format(m.loc[f, colname]))
        if bits:
            notes[f] = "  [" + ", ".join(bits) + "]"
    return notes


# ---------------------------------------------------------------- figures


def plot_overview(screen: pd.DataFrame, stable: set[str], out_png: Path) -> None:
    s = screen[~screen["degenerate"] & screen["delta"].notna()]
    eps = 1e-4
    inst = s.index.isin(stable)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        s.loc[~inst, "ooc_p1"] + eps,
        s.loc[~inst, "ooc_p2"] + eps,
        s=12,
        c="0.65",
        label="screened sensors",
    )
    ax.scatter(
        s.loc[inst, "ooc_p1"] + eps,
        s.loc[inst, "ooc_p2"] + eps,
        s=26,
        c="tab:red",
        label="model-stable set",
    )
    ax.plot([eps, 1], [eps, 1], ls="--", c="k", lw=1)
    for name, r in s.nlargest(3, "delta").iterrows():
        ax.annotate(
            name,
            (r["ooc_p1"] + eps, r["ooc_p2"] + eps),
            fontsize=8,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set(
        xscale="log",
        yscale="log",
        xlabel="Phase-I alarm rate (CV pool)",
        ylabel="Phase-II alarm rate (holdout + tail)",
        title="SPC screening: drift shows as distance above the diagonal",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_imr(
    df: pd.DataFrame,
    col: str,
    lim: pd.Series,
    i_hold: int,
    i_tail: int,
    lam: float,
    note: str,
    out_png: Path,
) -> None:
    x = df[col]
    t = np.arange(len(df))
    xv = x.to_numpy(dtype=float)
    fails = (df["target"] == 1).to_numpy()
    obs = np.isfinite(xv)
    mr = moving_range(x).to_numpy()
    z, e_lcl, e_ucl = ewma(x, lam, lim["center"], lim["sigma"])
    wr = we_rules(x, lim["center"], lim["sigma"])

    fig, axes = plt.subplots(
        3, 1, figsize=(11, 9), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]}
    )
    ax = axes[0]
    ax.plot(t, xv, ".", ms=4, color="tab:blue")
    ax.plot(t[fails & obs], xv[fails & obs], "x", ms=7, color="red", label="fail wafer")
    extra = wr[["r2", "r3", "r4"]].any(axis=1).to_numpy() & ~wr["r1"].to_numpy() & obs
    ax.plot(t[extra], xv[extra], "o", mfc="none", mec="orange", ms=9, label="WE rule 2-4")
    for y, c, lb in [
        (lim["ucl"], "r", f"UCL {lim['ucl']:.3g}"),
        (lim["center"], "g", f"center {lim['center']:.3g}"),
        (lim["lcl"], "r", f"LCL {lim['lcl']:.3g}"),
    ]:
        ax.axhline(y, color=c, ls="--" if c == "r" else "-", lw=1, label=lb)
    ax.set_title(f"I-chart | {col}{note}")
    ax.legend(fontsize=8, loc="upper left")

    axes[1].plot(t, mr, ".", ms=4, color="purple")
    axes[1].axhline(lim["mr_ucl"], color="r", ls="--", lw=1, label=f"MR UCL {lim['mr_ucl']:.3g}")
    axes[1].axhline(lim["mr_bar"], color="g", lw=1)
    axes[1].set_title("Moving range")
    axes[1].legend(fontsize=8, loc="upper left")

    axes[2].plot(t, z, "-", lw=1, color="tab:brown", label=f"EWMA lam={lam}")
    axes[2].axhline(e_ucl, color="r", ls=":", lw=1)
    axes[2].axhline(e_lcl, color="r", ls=":", lw=1)
    axes[2].axhline(lim["center"], color="g", lw=1)
    axes[2].set_title("EWMA")
    axes[2].legend(fontsize=8, loc="upper left")

    for a in axes:
        for b in (i_hold, i_tail):
            a.axvline(b, color="k", ls=":", lw=1)
    axes[2].set_xlabel("wafer index (time order)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def plot_protocol(df: pd.DataFrame, i_hold: int, i_tail: int, window: int, out_dir: Path) -> None:
    # (a) per-wafer missingness across the raw 590: the NaN-explosion exhibit.
    # The values layer looks calm at the tail; this protocol layer should scream.
    col = "f_row_missing_rate"
    lim = compute_limits(df[[col]].iloc[:i_hold], method="robust").loc[col]
    x = df[col]
    t = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(t, x, ".", ms=4, color="tab:blue")
    for y, c, lb in [
        (lim["ucl"], "r", f"UCL {lim['ucl']:.3g}"),
        (lim["center"], "g", f"center {lim['center']:.3g}"),
    ]:
        ax.axhline(y, color=c, ls="--" if c == "r" else "-", lw=1, label=lb)
    for b in (i_hold, i_tail):
        ax.axvline(b, color="k", ls=":", lw=1)
    ax.set_title(
        "f_row_missing_rate (fraction of the raw 590 channels unmeasured)"
        " - protocol drift, not process drift"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "protocol_row_missing_rate.png", dpi=180)
    plt.close(fig)

    # (b) windowed rates of the three missingness indicators, binomial limits
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    for ax, c in zip(axes, RATE_FEATURES, strict=True):
        w = windowed_rate(df[c], window)
        p0 = float(df[c].iloc[:i_hold].mean())
        lcl, ucl = binomial_limits(p0, w["n"].to_numpy())
        ax.plot(w["x"], w["rate"], ".-", ms=4, lw=0.8)
        ax.plot(w["x"], ucl, "r--", lw=1)
        ax.plot(w["x"], lcl, "r--", lw=1)
        ax.axhline(p0, color="g", lw=1)
        beyond = w["rate"].to_numpy() > ucl
        ax.plot(
            w["x"].to_numpy()[beyond],
            w["rate"].to_numpy()[beyond],
            "o",
            mfc="none",
            mec="red",
            ms=9,
        )
        for b in (i_hold, i_tail):
            ax.axvline(b, color="k", ls=":", lw=1)
        ax.set_title(f"{c} rate per {window} wafers (phase-I rate={p0:.3f})", fontsize=10)
    axes[-1].set_xlabel("wafer index (time order)")
    fig.tight_layout()
    fig.savefig(out_dir / "protocol_missing_indicators.png", dpi=180)
    plt.close(fig)


def plot_pchart(df: pd.DataFrame, i_hold: int, i_tail: int, window: int, out_png: Path) -> None:
    w = windowed_rate(df["target"], window)
    p0 = float(df["target"].iloc[:i_hold].mean())
    lcl, ucl = binomial_limits(p0, w["n"].to_numpy())
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(w["x"], w["rate"], ".-", ms=4, lw=0.8, label=f"fail rate / {window} wafers")
    ax.plot(w["x"], ucl, "r--", lw=1, label="UCL")
    ax.plot(w["x"], lcl, "r--", lw=1, label="LCL (clipped at 0)")
    ax.axhline(p0, color="g", lw=1, label=f"phase-I p-bar = {p0:.3f}")
    beyond = w["rate"].to_numpy() > ucl
    ax.plot(
        w["x"].to_numpy()[beyond],
        w["rate"].to_numpy()[beyond],
        "o",
        mfc="none",
        mec="red",
        ms=9,
        label="beyond UCL",
    )
    for b in (i_hold, i_tail):
        ax.axvline(b, color="k", ls=":", lw=1)
    ax.set_title(
        f"Yield p-chart - LCL clips at 0, so improvement (the 0/{i_tail} tail) "
        "cannot alarm; excursions can"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


# ---------------------------------------------------------------- run registry seam


def _start_run(tag: str, config: dict, note: str = "") -> Path:
    """Run folder via tracking.make_run (writes config.json itself)."""
    run_dir, meta = tracking.make_run(config=config, run_name=tag, note=note)
    logger.info(f"run {run_dir.name} | git {meta['git_sha']}")
    return run_dir


# ---------------------------------------------------------------- cli


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="SECOM SPC: Phase-I/II screening + showcase charts")
    p.add_argument("--limits", choices=["robust", "classic"], default="robust")
    p.add_argument("--k", type=float, default=3.0)
    p.add_argument(
        "--stable-from",
        default=None,
        help="features.json/csv/parquet from a selection run; "
        "legacy and modern names both accepted; "
        "default = hardcoded 100%-stable sensors",
    )
    p.add_argument("--stable-min", type=float, default=0.8)
    p.add_argument(
        "--delta-min",
        type=float,
        default=0.05,
        help="Phase-II minus Phase-I alarm rate that counts as drift",
    )
    p.add_argument("--n-showcase", type=int, default=4)
    p.add_argument("--window", type=int, default=50)
    p.add_argument("--ewma-lam", type=float, default=0.2)
    p.add_argument(
        "--master-path",
        default=None,
        help="EDA master table for chart-title annotations (optional)",
    )
    p.add_argument("--note", default=None)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    setup_logging(logfile=LOGS / "ml.log")
    logger.info(
        f"[spc.py] start | limits={args.limits} k={args.k} window={args.window} note={args.note}"
    )
    cfg = load_config()

    snap_config = {
        "holdout_n": cfg.pipeline.holdout_n,
        "tail_n": cfg.pipeline.tail_n,
        "limits": args.limits,
        "k": args.k,
        "ewma_lam": args.ewma_lam,
        "window": args.window,
    }
    df, i_hold, i_tail, fp, snapshot_id = load_data(
        cfg.pipeline.holdout_n, cfg.pipeline.tail_n, snap_config
    )
    cols = raw_sensor_columns(df)  # all raw sensors: the monitoring net

    run = _start_run(
        "spc",
        config={**vars(args), "i_hold": i_hold, "i_tail": i_tail, "n_sensors": len(cols)},
        note=args.note or "",
    )
    figs = run / "figures"
    figs.mkdir(exist_ok=True)

    limits = compute_limits(df[cols].iloc[:i_hold], method=args.limits, k=args.k)
    screen = screening(df, cols, limits, i_hold, i_tail, ewma_lam=args.ewma_lam)
    limits.to_csv(run / "spc_limits.csv", index_label="feature", float_format="%.6g")
    screen.to_csv(run / "spc_screening.csv", index_label="feature", float_format="%.6g")
    logger.info(
        f"screened {len(screen)} sensors "
        f"({int(screen['degenerate'].sum())} degenerate) | median phase-I "
        f"alarm rate {screen['ooc_p1'].median():.4f}"
    )

    stable = load_stable_set(args.stable_from, args.stable_min)
    picks, _ = select_showcase(screen, stable, args.delta_min, args.n_showcase)
    logger.info(f"showcase: {picks}")

    plot_overview(screen, stable, figs / "overview_scatter.png")
    notes = load_master_notes(args.master_path)
    for col in picks:
        plot_imr(
            df,
            col,
            limits.loc[col],
            i_hold,
            i_tail,
            lam=args.ewma_lam,
            note=notes.get(col, ""),
            out_png=figs / f"imr_{col}.png",
        )
    plot_protocol(df, i_hold, i_tail, args.window, figs)
    plot_pchart(df, i_hold, i_tail, args.window, figs / "yield_pchart.png")

    tracking.append_index(
        run,
        {
            "type": "spc",
            "limits": args.limits,
            "note": args.note or "",
            "n_screened": len(screen),
            "median_ooc_p1": round(float(screen["ooc_p1"].median()), 4),
            "max_delta": round(float(screen["delta"].max()), 4),
            "n_drift": int((screen["delta"] >= args.delta_min).sum()),
            "data": fp["raw"]["secom.data"][:16],  # values-matrix hash, 16-char
            "snapshot_id": snapshot_id,
        },
        index_file=ARTIFACTS / "index_monitor.csv",
    )
    logger.info(f"[spc.py] done | artifacts -> {run}")


def _in_ipykernel() -> bool:
    """True under a Jupyter kernel, without importing IPython (which the
    project venv need not carry)."""
    return "ipykernel" in sys.modules


if __name__ == "__main__" and not _in_ipykernel():
    main()
# %%
