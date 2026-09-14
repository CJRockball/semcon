# Data — SECOM semiconductor pass/fail data contract

Yield-risk decision support from inline sensor surveillance. Each row is one
production entity (wafer) with a pass/fail result from in-house line testing and
the timestamp of that test point.

This document is the **data-layer contract**: what lives where, what is tracked
in Git, how data lineage is established, and how the governed data layer feeds
training, batch scoring, monitoring, surrogate DOE, and dashboard workflows.
For the reader-facing methodology narrative, see
[Data and preprocessing](../docs/01_data_and_preprocessing.md). For model
validation decisions, see [Model validation record](../validation.md).

## Source and license

- **Source:** [UCI Machine Learning Repository, dataset 179](https://archive.ics.uci.edu/dataset/179/secom)
- **Donors:** Michael McCann, Adrian Johnston; donated 18 November 2008
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — sharing
  and adaptation are permitted with attribution
- **Citation:** McCann, M. & Johnston, A. (2008). *SECOM* [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C54305

The donor framing is that not all monitored signals are equally valuable and
that identifying signals associated with downstream yield risk can improve
process throughput, shorten time to learning, and reduce unit cost. This
repository implements that problem as a reproducible risk-ranking and
monitoring workflow. It does not claim causal process knowledge from
observational data alone.

## Raw data (`data/raw/` — not tracked in Git)

| File | Content |
|---|---|
| `secom.data` | 1,567 × 590 sensor measurements, space-separated text |
| `secom_labels.data` | Two columns: label (−1 = pass, +1 = fail) and test timestamp |
| `secom.names` | UCI metadata |

Facts that matter before touching the data:

- **Class imbalance:** 104 failures out of 1,567 wafers (6.6%, approximately
  14:1). Predicting “pass” everywhere gives 93.4% accuracy, so accuracy is not
  a meaningful primary metric.
- **Missing values:** present throughout and encoded as the literal `NaN`.
  Missingness is structured through sensor dropout blocks rather than assumed
  to be random.
- **Anonymous features:** sensors are integer indices, not physical process
  names or engineering units. Root-cause statements can only refer to feature
  patterns, not identified equipment.
- **Time ordered:** the final 27 wafers sit in a distinct missingness regime
  (“NaN explosion”) and are excluded from modelling by design. Do not shuffle
  blindly.
- **Donor baseline:** kernel-ridge classification with 10-fold CV and 40
  selected features reported balanced error rates of 33.5–40.1%, TPR of about
  48–60%, and TNR of about 72–78%. Results far above this range warrant a
  leakage check rather than immediate celebration.

Fetch via `pip install ucimlrepo` then `fetch_ucirepo(id=179)`, or download
from the UCI page above.

## Layout — governed data and derived layers

| Layer | Object | Tracked in Git? | Purpose |
|---|---|---:|---|
| Raw | `data/raw/*` | No | Original SECOM files; regenerable from source |
| Bronze | `data/secom.db` (SQLite) | No | Untouched local relational landing zone |
| Registry export | `data/column_registry.csv` | Yes | Generated audit trail for column roles, status, and lineage |
| Silver | Extracted frame in memory via SQL | No | Validated, time-bounded analytical input |
| Gold | `data/snapshots/gold/<snapshot_id>/` | Manifests yes; matrices no | Reproducible model-training matrix and metadata |
| Operational replay | `data/sim/` | No | Generated incoming-lot fixtures for batch scoring and monitoring |
| Legacy | `data/processed/*.parquet` | No | Frozen pre-migration regression evidence only |

The SQLite database stands in for the MES/historian layer that SECOM lacks as
flat files: raw data is loaded untouched into bronze, analysis starts from SQL
extraction in silver, and only the model-facing matrix is snapshotted in gold.
One source of truth exists per layer; derived data is never written back into
the database.

## The database (`data/secom.db` — derived, not tracked)

Built by `make ingest` through `src/semcon/db_ingest.py`. The database has four
core tables:

| Table | Content |
|---|---|
| `sensor_readings` | `wafer_id` primary key plus `s001`–`s590` REAL columns; generated, zero-padded sensor names |
| `wafer_labels` | `wafer_id`, raw `target` (−1/+1), and `timestamp`; stored separately to require an explicit join |
| `column_registry` | One row per column: role, status, missing percentage, `derived_from`, and notes |
| `ingestion_log` | One row per load: source file, SHA-256, row count, and Git SHA |

`target` is stored in its source form (−1/+1). The binary encoding `is_fail` is
a derived column with exactly one creator: `validate.py`. Consumers do not
recode labels independently; they call `ensure_is_fail`.

## The column registry

The registry is the feature schema. Every column in the extracted frame has
exactly one row with a **role** (`key`, `metadata`, `feature_raw`,
`feature_eng`, `target`) and a **status** (`active` or `excluded` with a
reason).

- **Whoever creates a column registers it.** `db_ingest` registers the raw 593
  columns: 590 sensors, key, target, and timestamp. `extract` registers
  `split`, `feature_eng` registers each engineered feature with
  `derived_from` lineage, and `validate` registers `is_fail`.
- **Features are retired, never silently dropped.** The screening rules in
  `explore.py` retire constant, high-missingness, near-zero-variance, and
  redundant correlated features while retaining the reason for each decision.
  The fail-enrichment rescue rule prevents rare but potentially informative
  deviations from being discarded mechanically.
- **The committed registry export is generated, not hand edited.** Its Git
  history is an audit trail of column decisions.

The currently documented screening outcome is 333 retired raw features and 257
active raw features. Four engineered missingness features bring the active
model-input count to 261 before any selected-feature modelling decision.

## Extraction and temporal split (silver)

`make extract` executes `sql/extract_wafers.sql` through `extract.py`, producing
a wide in-memory frame of 1,567 rows × 594 columns: 590 sensors plus
`wafer_id`, `timestamp`, `target`, and `split`.

The split has three chronological zones and is driven by two configuration
decisions in `src/semcon/config.py`, frozen after exploratory analysis and
recorded in `validation.md`.

| Zone | Rows | Failures | Boundary |
|---|---:|---:|---|
| `cv` | 1,309 | 90 | Before `CUTOFF` |
| `holdout` | 231 | 14 | `CUTOFF` ≤ timestamp < `EXCLUDE_AFTER` |
| `excluded` | 27 | 0 | From `EXCLUDE_AFTER`; late missingness-regime break |

Rules that keep the split honest:

- Boundaries are explicit decisions, changed only through a deliberate config
  edit with supporting evidence and its own commit.
- No timestamp may equal a boundary exactly because SQL `BETWEEN` is inclusive;
  `extract.py` raises if it detects a boundary collision.
- `train_xgb` asserts that the SQL split reproduces the legacy positional split
  before adopting it. This is the split-equivalence guard.
- The chronological holdout is protected from feature selection, final model
  selection, and calibration fitting until final evaluation.

## Feature engineering (gold inputs)

`make features` appends four EDA-derived columns. They are computed from the raw
590-channel matrix because some source columns may have been retired during
screening and therefore no longer appear among active raw sensors. The build
self-validates against dropout anchors identified during exploration.

| Feature | Meaning | Interpretation boundary |
|---|---|---|
| `f_miss_clq14` | Missingness indicator for a sensor dropout clique | Predictive observation-process signal; not a controllable process setting |
| `f_miss_clq23` | Missingness indicator for a second dropout clique | Predictive observation-process signal; not a controllable process setting |
| `f_miss_block5` | Missingness indicator for the final station-group block | Label-free data-quality probe re-examined on the holdout |
| `f_row_missing_rate` | Fraction of the 590 raw channels missing for a wafer | Overall data-quality covariate; not a direct process measurement |

The clique indicators reflect sensor groups missing together on the same wafers.
For example, the first clique is target-associated but its direction indicates a
measurement-protocol or routing signal rather than sensor degradation. These
features can be valid model inputs while remaining invalid DOE factors. The
surrogate DOE workflow explicitly excludes engineered missingness and clique
features from candidate-factor selection.

## Validation gate

`make validate` runs `validate.py`, which applies a Pandera schema to the
structural columns and checks the raw sensor block. It verifies key uniqueness
and non-nullness, timestamp type, the raw target vocabulary `{−1, +1}`, split
vocabulary, expected row counts, and sensor-block count/dtypes. It also creates
`is_fail` through the single approved path.

The gate writes a missingness-drift report against the latest gold snapshot at:

```text
artifacts/runs/<timestamp>_validate/missingness_drift.csv
```

Schema failures raise. Drift is report-only at this stage because SPC and
monitoring own the operational response.

## Gold snapshots (`data/snapshots/gold/<snapshot_id>/`)

`train_xgb` writes a gold snapshot immediately before final fitting. It records
the exact model-facing matrix and metadata needed to reproduce the training
boundary:

| Snapshot object | Git status | Purpose |
|---|---:|---|
| `matrix.parquet` | Ignored | Exact matrix consumed by the training run |
| `manifest.json` | Tracked | Configuration, cutoffs, row/column counts, Git SHA, and data fingerprint |
| `registry.csv` | Tracked | Frozen copy of the column registry at snapshot time |

The resulting lineage chain is:

```text
raw-file SHA-256
  → ingestion_log
  → column registry
  → snapshot manifest
  → run config.json
  → feature contract
  → model.ubj and calibration artifact
```

From a training run, this chain makes it possible to answer which source files,
transformations, columns, and configuration produced the model. Batch scoring
uses the frozen training-run feature contract rather than directly consuming a
snapshot matrix.

## Downstream consumers

The data layer supports multiple downstream workflows without allowing each to
invent its own data interpretation.

| Consumer | Reads from the governed data layer | Contract enforced |
|---|---|---|
| `train_xgb.py` | SQL extraction, registered feature engineering, and gold snapshot | Chronological split, active registry status, frozen feature contract |
| `calibrate.py` and `evaluation.py` | Registered model outputs from the development workflow | Calibration and evaluation separation from protected holdout evidence |
| `score.py` | Incoming/replayed batch plus model and feature contract | Required feature names/order and recreated engineered features |
| `monitor.py` | Scored batch outputs plus frozen Phase-I references | Input availability, baseline comparability, append-only monitoring records |
| `doe_design.py` / `doe_run.py` | Extracted background frame and selected model contract | Raw-sensor-only factors, observed-support levels, and OOD checks |
| Dash application | Registered score, SPC, forecasting, monitoring, and retraining artifacts | Artifact-based presentation; no independent source-data transformation |

For the full lifecycle, see
[Batch scoring and retraining governance](../docs/06_mlops_batch_scoring_and_governance.md).

## Operational replay data (`data/sim/`)

`data/sim/` contains generated incoming-lot fixtures used to exercise batch
scoring and monitoring. It is derived data, not UCI source data and not a
replacement for MES/FDC ingestion. It is removed by `make clean` and regenerated
by the replay workflow.

Scenario labels such as `batch_a_clean`, `batch_b_shift`, and
`batch_c_dropout` are deterministic fixtures used to test the expected routing
of scoring, monitoring, and retraining policy. They do not represent verified
historical fab incidents. A simulated sensor shift demonstrates that a defined
monitoring rule can detect a defined perturbation; it does not identify a real
chamber fault.

## Legacy naming and migration evidence

The retired flat-file pipeline used zero-based integer feature names. The
SQLite-backed pipeline uses one-based `s001`–`s590` names. The mapping is:

```text
legacy n ↔ s(n + 1)
```

The mapping is verified by positional migration tests and missingness-clique
anchors, not assumed. Frozen Parquet files in `data/processed/` are retained
only as a regression baseline for `migration_test.py`, which checks that
post-migration behavior reproduces the pre-migration pipeline where expected.
No other module should read them.

## Regenerate and verify

Run from the repository root:

```bash
make clean
make
make test
make hygiene
make demo
```

`make clean` removes the database, snapshots, derived artifacts, logs, replay
data, and Python caches while preserving raw source data and the environment.
`make` rebuilds the core pipeline: ingest, extract, explore, features, baseline
and selected-model training, calibration, SPC, and SARIMAX. `make demo` then
creates replay batches, scores them, runs a reconciled holdout replay, and
writes scorecards.

Monitoring, retraining-policy evaluation, Dash startup, and DOE execution are
explicit operational workflows. Run them when validating the complete portfolio
lifecycle; do not imply that `make` alone executes every optional subsystem.

CLI checks after ingestion:

```bash
sqlite3 data/secom.db "PRAGMA table_info(sensor_readings);"
sqlite3 data/secom.db "SELECT COUNT(*) FROM sensor_readings;"
sqlite3 data/secom.db "SELECT role, COUNT(*) FROM column_registry GROUP BY role;"
sqlite3 data/secom.db "SELECT COUNT(*) FROM column_registry WHERE role='feature_eng';"
```

Expected checks include a primary key on `wafer_id`, 1,567 rows in
`sensor_readings`, registry role counts consistent with the generated export,
and four engineered feature rows.

## Gotchas

- **Do not use `:param` tokens in SQL comments.** The SQLAlchemy binder can
  count them and fail with a binding error.
- **The `sqlite3` CLI can silently create an empty database at a wrong path.**
  If `PRAGMA` returns nothing, verify the path before debugging the loader.
- **`is_fail` has one home: `validate.py`.** Writing `target.eq(1)` in another
  consumer creates a second label-encoding path and violates the contract.
- **Notebooks do not build paths from the working directory.** Notebook kernels
  typically run from `notebooks/`; use `semcon.paths` constants.
- **Do not interpret anonymous sensor indices as physical controls.** Raw sensor
  features can be explored as surrogate DOE factors only after observed-support
  and domain review; they are not recipe recommendations.