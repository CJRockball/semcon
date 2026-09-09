from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semcon.doe_design import (
    build_factorial_design,
    describe_design,
    load_selected_features,
    select_doe_factors,
    write_design_artifacts,
)


def test_load_selected_features_list(tmp_path: Path) -> None:
    path = tmp_path / "features.json"
    path.write_text(json.dumps(["s001", "s060", "f_miss_clq14"]), encoding="utf-8")

    assert load_selected_features(path) == ["s001", "s060", "f_miss_clq14"]


def test_load_selected_features_dict(tmp_path: Path) -> None:
    path = tmp_path / "features.json"
    path.write_text(json.dumps({"features": ["s001", "s060"]}), encoding="utf-8")

    assert load_selected_features(path) == ["s001", "s060"]


def test_select_doe_factors_filters_missingness_and_prefers_order() -> None:
    selected = ["s001", "f_miss_clq14", "s060", "s123", "f_row_missing_rate"]

    got = select_doe_factors(selected, max_factors=3, preferred=["s060"])

    assert got == ["s060", "s001", "s123"]


def test_select_doe_factors_requires_raw_sensors() -> None:
    with pytest.raises(ValueError, match="No valid raw sensor factors"):
        select_doe_factors(["f_miss_clq14", "f_row_missing_rate"], max_factors=3)


def test_build_factorial_design_shape_and_levels() -> None:
    factors = [
        {"name": "s060", "low": 10.0, "high": 20.0},
        {"name": "s123", "low": -1.0, "high": 1.0},
    ]

    design = build_factorial_design(
        factors,
        center_points=2,
        replicates=2,
        randomize=False,
        seed=7,
    )

    assert len(design) == (2**2) * 2 + 2
    assert set(design["s060_coded"].unique()) == {-1.0, 0.0, 1.0}
    assert set(design["s123_coded"].unique()) == {-1.0, 0.0, 1.0}
    assert design["is_center"].sum() == 2
    assert design["replicate"].max() == 2
    assert design["run_id"].is_unique
    assert design["run_order"].is_unique


def test_build_factorial_design_maps_actual_values() -> None:
    factors = [
        {"name": "s060", "low": 10.0, "high": 20.0},
        {"name": "s123", "low": -1.0, "high": 1.0},
    ]

    design = build_factorial_design(
        factors,
        center_points=1,
        replicates=1,
        randomize=False,
        seed=7,
    )

    lo_lo = design[(design["s060_coded"] == -1.0) & (design["s123_coded"] == -1.0)].iloc[0]
    hi_hi = design[(design["s060_coded"] == 1.0) & (design["s123_coded"] == 1.0)].iloc[0]
    center = design[design["is_center"]].iloc[0]

    assert lo_lo["s060"] == 10.0
    assert lo_lo["s123"] == -1.0
    assert hi_hi["s060"] == 20.0
    assert hi_hi["s123"] == 1.0
    assert center["s060"] == 15.0
    assert center["s123"] == 0.0


def test_build_factorial_design_seeded_randomization_is_deterministic() -> None:
    factors = [
        {"name": "s060", "low": 10.0, "high": 20.0},
        {"name": "s123", "low": -1.0, "high": 1.0},
    ]

    d1 = build_factorial_design(factors, center_points=1, replicates=1, randomize=True, seed=42)
    d2 = build_factorial_design(factors, center_points=1, replicates=1, randomize=True, seed=42)

    pd.testing.assert_frame_equal(d1, d2)


def test_describe_design_and_write_artifacts(tmp_path: Path) -> None:
    factors = [
        {"name": "s060", "low": 10.0, "high": 20.0},
        {"name": "s123", "low": -1.0, "high": 1.0},
    ]

    design = build_factorial_design(
        factors,
        center_points=1,
        replicates=1,
        randomize=True,
        seed=11,
    )
    metadata = describe_design(
        factors,
        center_points=1,
        replicates=1,
        randomize=True,
        seed=11,
    )

    design_csv, design_json = write_design_artifacts(design, metadata, tmp_path)

    assert design_csv.exists()
    assert design_json.exists()

    loaded = pd.read_csv(design_csv)
    meta = json.loads(design_json.read_text(encoding="utf-8"))

    assert len(loaded) == len(design)
    assert meta["seed"] == 11
    assert meta["factors"][0]["name"] == "s060"