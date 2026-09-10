# Surrogate DOE

Design-of-experiments analysis **on the calibrated model**, not on the fab.

The DOE module asks: *if we perturb the top raw-sensor features across their
observed ranges, how does the calibrated model's predicted failure risk move?*
It is a structured sensitivity study over the trained surrogate — useful for
hypothesis generation and for demonstrating DOE literacy, bounded by an
explicit claim boundary:

> Factor levels are observed sensor quantiles, not recipe settings. Nothing
> here is causal. Confirmation requires controlled physical lots.

## Quickstart

```bash
make doe
```

Runs all three stages (design → run → analyze) against the latest calibrated
`xgb_sel` model, using the pointer files `artifacts/doe/latest_design` and
`artifacts/doe/latest_run`. Knobs live at the top of the Makefile:
`DOE_LABEL`, `DOE_N_FACTORS`, `DOE_CENTER_POINTS`, `DOE_BG_START`,
`DOE_BG_END`.

## The three stages

### 1. Design — `semcon-doe-design`

```bash
semcon-doe-design --run latest --label s060-factorial --n-factors 3 --center-points 3
```

- Ranks candidate factors by TreeSHAP importance from the parent training run
  (`shap/shap_feature_summary.csv`), restricted to **raw sensor columns**.
  Engineered missingness/clique features are excluded: they are data-quality
  signals, not controllable knobs. `s060` is pinned as the preferred factor.
- Levels are the observed **Q10 / Q50 / Q90** quantiles of each factor in the
  extracted SECOM frame. They are not tool limits and not recipe settings.
- Builds a two-level full factorial (or a `--fraction` generator via pyDOE3),
  appends center points, randomizes run order with a seeded RNG.

Current design (run `20260910_145703_doe-design_s060-factorial`, parent
`20260910_145542_xgb_sel`):

| Factor | Low (Q10) | Center (Q50) | High (Q90) | mean abs SHAP |
|---|---:|---:|---:|---:|
| s060 | -4.150 | 1.174 | 18.879 | 0.803 |
| s022 | -6416.313 | -5525.250 | -5194.993 | 0.435 |
| s461 | 15.275 | 26.152 | 46.191 | 0.393 |

A 2^3 factorial + 3 center points = **11 design rows**.

### 2. Run — `semcon-doe-run`

```bash
semcon-doe-run --design "$(cat artifacts/doe/latest_design)/design.csv" \
  --background-start "2008-07-19 00:00:00" --background-end "2008-10-05 05:29:59" \
  --label s060-factorial
```

- Expands each design row over every wafer in the background window
  (ALE-style averaging): the DOE factor is overwritten on the raw sensor
  column, then `build_features()` rebuilds engineered features exactly as the
  training/scoring paths do, and the calibrated model scores each row.
- Outputs two responses per scored wafer:
  - `p_cal` — calibrated probability. **Deterministic primary response.**
  - `y_observed` — one Bernoulli draw from `p_cal` (config
    `noise_mode="bernoulli"`). **Simulated secondary response**, identical
    seed ⇒ identical draw. It is not physical replication.
- Computes OOD diagnostics (Mahalanobis + kNN distance of each design point
  against complete-case background support, thresholds at the background 99th
  percentile) into `ood_table.parquet`.

First run: 11 design rows × 1,309 background wafers, seed 1337.

### 3. Analyze — `semcon-doe-analyze`

```bash
semcon-doe-analyze \
  --predictions "$(cat artifacts/doe/latest_run)/design_predictions.parquet" \
  --output-dir "$(cat artifacts/doe/latest_run)/analysis"
```

Two analyses are produced side by side. Read them in this order.

## Primary analysis: deterministic surrogate surface

OLS on coded factors (-1/0/+1) with the design-row mean `p_cal_mean` as
response. From the first run (R^2 = 0.993):

| Term | Coefficient (probability) | p-value |
|---|---:|---:|
| s060_coded | +0.01257 | <0.001 |
| s022_coded | +0.00666 | <0.001 |
| s461_coded | +0.00579 | <0.001 |
| s060:s022 | +0.00313 | 0.009 |
| s060:s461 | +0.00260 | 0.017 |
| s022:s461 | +0.00153 | 0.082 |

Interpretation: on the probability scale, moving any of the three factors from
low to high raises modeled failure risk; `s060` dominates (high-minus-low
effect +2.51 percentage points, vs +1.33 for s022 and +1.16 for s461). The
model surface also carries small positive two-factor interactions.

`recommendation.json` picks the lowest-risk supported cell — here
`doe_001` (all factors low), predicted risk 2.71%. This is a *surrogate*
recommendation only; it says where the model is optimistic, not where the
line should run.

### Center points and curvature

`curvature.json` compares center-point response to factorial-corner response.
Note the center level is the **median (Q50)**, not the arithmetic midpoint of
low/high — the check relies on the `is_center` flag carried through the
pipeline, not on midpoint inference.

Important subtlety: for the *deterministic* response, duplicate center rows
return identical values, so they add no pure-error estimate. They exist for
the Bernoulli layer (below) and for the curvature contrast itself.

## Secondary analysis: grouped Bernoulli GLM

`y_observed` is aggregated per design row into failures/passes out of 1,309
trials, then fit with a binomial GLM (logit link) on the coded factors:

| Term | Log-odds coef | Odds ratio | p-value |
|---|---:|---:|---:|
| s060_coded | +0.237 | 1.27 | <0.001 |
| s022_coded | +0.162 | 1.18 | <0.001 |
| s461_coded | +0.106 | 1.11 | 0.031 |
| interactions | |0.96–1.05| | 0.35–0.41 |

Main effects agree with the deterministic surface (same ranking, same
direction). The interactions are **not detected** here — expected, not a
contradiction:

- The GLM works on the log-odds scale; a ~0.3 percentage-point probability
  interaction at ~4–5% baseline risk compresses to a very small log-odds
  effect.
- One Bernoulli realization adds binomial sampling noise; interactions, being
  differences of differences, need more evidence than main effects.
- Rows within a cell share the design point but have heterogeneous `p_cal`
  across background wafers, so grouped counts are not perfectly binomial
  (observed dispersion < 1 in this run).

The Bernoulli layer exists to demonstrate binary-response DOE mechanics and to
make the center points earn their keep (their `y_observed` draws genuinely
differ). It is simulation conditioned on the surrogate — see
`bernoulli_scope.json`, which ships with every analysis and states
`causal_claim: false` in machine-readable form.

## Artifact reference

Per stage, under `artifacts/doe/<timestamp>_<name>/`:

| Stage | Files |
|---|---|
| design | `design.csv`, `design_metadata.json`, `factor_candidates.csv`, `selected_factors.json`, `config.json` |
| run | `design_predictions.parquet`, `ood_table.parquet`, `config.json` |
| analysis | `design_cell_summary.csv`, `effects_table.csv`, `main_effects.csv`, `curvature.json`, `recommendation.json`, `model_summary.txt`, `residual_diagnostics.csv`, `effects_pareto.png`, `residual_diagnostics.png`, `interaction_*.png`, plus Bernoulli: `bernoulli_cell_summary.csv`, `bernoulli_effects_table.csv`, `bernoulli_model_summary.txt`, `bernoulli_scope.json` |

Pointers: `artifacts/doe/latest_design`, `artifacts/doe/latest_run`.

## Configuration

`DOEConfig` in `semcon/config.py`:

| Field | Default | Meaning |
|---|---|---|
| `max_factors` | 4 | Cap on design factors |
| `center_points` | 3 | Center rows appended to the factorial |
| `replicates` | 1 | Full-design replicates |
| `randomize` | true | Seeded run-order shuffle |
| `noise_mode` | `"bernoulli"` | `"bernoulli"` or `"gaussian_logit"` |
| `gaussian_sigma` | null | Required iff `gaussian_logit` |

## Testing

```bash
uv run pytest tests/test_doe_design.py tests/test_doe_run.py tests/test_doe_analyze.py -q
```

Coverage includes: Bernoulli noise is binary and seed-reproducible; coded
columns survive `score_design`; grouped cells satisfy failures + passes =
trials; the GLM recovers a known main-effect direction; all analysis
artifacts (including the four Bernoulli outputs) are written.

## Known limitations and next steps

- **No face-centered CCD yet.** Quadratic response-surface terms would need
  axial points; the current design only screens mains + two-factor
  interactions on the surrogate.
- **Single Bernoulli realization.** A seed-sensitivity sweep (5–10 seeds,
  collect coefficient distributions) would quantify how much of the GLM
  interaction disagreement is sampling noise. Not yet automated.
- **Sensors are not recipes.** Mapping `s060/s022/s461` back to controllable
  process settings is a domain step that happens before any physical
  confirmation lots, and is deliberately out of scope for this repo.
