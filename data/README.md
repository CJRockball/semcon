<!-- Save as: data/README.md -->

# Data — SECOM Semiconductor Pass/Fail, DB-backed layout

Yield prediction from inline sensor surveillance. Each row is one production
entity (wafer) with a pass/fail result from in-house line testing and the
timestamp of that test point.

This document is the data-layer contract: what lives where, what is tracked in
git, and how to verify it without running anything.

## Source and license

- **Source:** [UCI Machine Learning Repository, dataset 179](https://archive.ics.uci.edu/dataset/179/secom)
- **Donors:** Michael McCann, Adrian Johnston (donated 18 Nov 2008)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — sharing
  and adaptation permitted with attribution
- **Citation:** McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C54305

The donors' framing: not all monitored signals are equally valuable; the task is
to identify the signals that drive yield excursions downstream, to "increase
process throughput, decrease time to learning and reduce per unit production
costs." That is exactly the framing of this repo.

## Raw data (`data/raw/` — not tracked in git)

| File | Content |
|---|---|
| `secom.data` | 1,567 × 590 sensor measurements, space-separated text |
| `secom_labels.data` | Two columns: label (−1 = pass, +1 = fail) and test timestamp |
| `secom.names` | UCI metadata |

Facts that matter before touching the data:

- **Class imbalance:** 104 fails out of 1,567 wafers (6.6%, ~14:1). Predicting
  "pass" everywhere scores 93.4% accuracy — accuracy is a meaningless metric here.
- **Missing values:** present throughout, encoded as the literal `NaN`.
  Missingness is *structured* (sensor dropout blocks), not random — see the EDA
  notebook.
- **Anonymous features:** sensors are integer indices, not physical names.
  Root-cause statements can only go as far as feature groups, not equipment.
- **Time-ordered:** the final 27 wafers sit in a different missingness regime
  (the "NaN explosion", EDA §3.5). They are excluded from modeling by design —
  see the split policy below. Do not shuffle blindly.
- **Donor baseline** (kernel-ridge classifier, 10-fold CV, 40 selected
  features): balanced error rate 33.5–40.1%, TPR ~48–60%, TNR ~72–78%.
  Anything claiming far above this should be checked for leakage.

Fetch via `pip install ucimlrepo` then `fetch_ucirepo(id=179)`, or download
from the UCI page above.

## Layout — one writable store, three layers

| Layer | Object | Tracked in git? |
|---|---|---|
| Raw | `data/raw/*` | No — regenerable via fetch |
| Bronze | `data/secom.db` (SQLite) | **No — derived; rebuild freely** |
| Registry export | `data/column_registry.csv` | Yes — the audit trail of column decisions |
| Silver | extracted frame (in memory, via SQL) | No — recomputed deterministically |
| Gold | `data/snapshots/gold/<snapshot_id>/` | Manifests **yes**, matrices **no** |
| Legacy | `data/processed/*.parquet` | No — frozen pre-migration evidence only |

The SQLite database stands in for the MES/historian layer that SECOM lacks as
flat files: raw data is loaded untouched (bronze), all analysis starts from SQL
extraction (silver), and only the model-facing matrix is snapshotted (gold).
One source of truth per layer; derived data is never written back into the DB.

## The database (`data/secom.db` — derived, not tracked)

Built by `make ingest` (`src/semcon/db_ingest.py`). Four tables:

| Table | Content |
|---|---|
| `sensor_readings` | `wafer_id` PK + `s001`–`s590` REAL columns (generated, zero-padded) |
| `wafer_labels` | `wafer_id`, `target` (raw −1/+1), `timestamp` — separate table, forcing a real JOIN |
| `column_registry` | One row per column: role, status, missing_pct, derived_from, notes |
| `ingestion_log` | One row per load: source file, sha256, rows, git SHA — the audit trail |

`target` is stored raw (−1/+1). The 0/1 encoding (`is_fail`) is a derived
column with exactly one creator: `validate.py`. Consumers never recode labels
themselves — they call `ensure_is_fail`.

## The column registry

The registry is the feature schema. Every column in the extracted frame has
exactly one row with a **role** (`key`, `metadata`, `feature_raw`,
`feature_eng`, `target`) and a **status** (`active` / `excluded` + reason).

- **Whoever creates a column registers it**: `db_ingest` registers the raw 593
  (590 sensors + key + target + timestamp), `extract` registers `split`,
  `feature_eng` registers each engineered column with `derived_from` lineage,
  `validate` registers `is_fail`.
- **Features are never dropped, only retired with reasons.** Retirement rules
  (applied by `explore.py`): constant 116, high-missing 28, near-zero-variance
  19 (with a fail-enrichment rescue rule — rare excursions can be the signal),
  correlated 170 → **333 retired, 257 active**. Add the 4 engineered features:
  **261 active model inputs**.
- The committed export `data/column_registry.csv` is generated, never
  hand-edited; its git history is the audit trail of column decisions.

## Extraction and the split (silver)

`make extract` runs `sql/extract_wafers.sql` through `extract.py` → the wide
frame (1,567 × 594): sensors + `wafer_id` + `timestamp` + `target` + `split`.

The split is **three zones, chronological, driven by two config decisions**
(`src/semcon/config.py`, frozen after EDA — see `validation.md`):

| Zone | Rows | Fails | Boundary |
|---|---|---|---|
| `cv` | 1,309 | 90 | before `CUTOFF` |
| `holdout` | 231 | 14 | `CUTOFF` ≤ ts < `EXCLUDE_AFTER` |
| `excluded` | 27 | 0 | from `EXCLUDE_AFTER` (regime break) |

Rules that keep the split honest:

- Boundaries are decisions: changed only by a deliberate config edit, with the
  EDA reference in a comment, as its own commit.
- No timestamp may equal a boundary exactly (SQL `BETWEEN` is inclusive);
  `extract.py` raises if one does.
- `train_xgb` asserts the SQL split reproduces the legacy positional split
  before adopting it (the split-equivalence guard).

## Feature engineering (gold inputs)

`make features` appends four EDA-derived columns, computed from the raw
590-channel matrix because several source columns were retired in Stage 1 and
no longer exist among the active sensors. The build self-validates against the
EDA dropout anchors (794 / 715 wafers):

- **`f_miss_clq14`** (794 wafers), **`f_miss_clq23`** (715 wafers) — dropout-
  clique indicators: sensor groups missing on exactly the same wafers.
  Target-associated (clique 14: OR ≈ 0.49, p ≈ 0.0008), direction: fails are
  missing *less* often — a measurement-protocol/routing signal, not sensor
  degradation (EDA §3.4, §3.7).
- **`f_miss_block5`** — block-5 (final station group) dropout flag. The only
  block where fails edge *ahead* on missingness; a weak effect, included as a
  label-free probe re-examined on the holdout (EDA §3.5, §3.7).
- **`f_row_missing_rate`** — per-wafer fraction of the 590 raw channels
  missing. A data-quality covariate; near-identical across classes
  (pass 4.55% vs fail 4.32%).

## Validation gate

`make validate` (`validate.py`): pandera schema on the structural columns
(key unique/non-null, timestamp typed, `target` ∈ {−1,1}, `split` vocabulary),
a count/dtype sweep over the 590-sensor block, row-count expectations, the
`is_fail` creation above, and a missingness-drift report against the latest
gold snapshot (`artifacts/runs/<ts>_validate/missingness_drift.csv`).
Schema failures raise; drift is report-only — SPC owns the response.

## Snapshots (`data/snapshots/gold/<snapshot_id>/`)

Written by `train_xgb` immediately before the final fit — the exact matrix the
model saw. Each snapshot holds `matrix.parquet` (gitignored), `manifest.json`
(config incl. cutoffs, row/col counts, git SHA, data fingerprint) and a frozen
`registry.csv` copy (both tracked). The lineage chain this closes:

raw-file sha256 → `ingestion_log` → registry → snapshot manifest → run
`config.json` → `model.ubj`. From any run folder you can answer "exactly which
bytes trained this model?" in under 30 seconds.

## Legacy naming

The retired flat-file pipeline used 0-based integer feature names; the DB uses
1-based `s001`–`s590`. Mapping: **legacy `n` ↔ `s(n+1)`** — proven by the
positional migration test and the clique anchors (794/715), not assumed.
The frozen parquets in `data/processed/` (`dfX_raw/v1/v2`, `dfy_raw/v1`) are
kept solely as the regression baseline for `migration_test.py`, which verified
that post-migration runs reproduce pre-migration predictions bit-for-bit.
Nothing else should read them.

## Regenerate and verify

```bash
make clean   # remove DB, snapshots, artifacts, logs (raw data is preserved)
make         # rebuild everything: ingest -> extract -> explore -> features
             # -> train (base + sel) -> calibrate -> spc -> sarimax
make test && make hygiene
```

CLI verification without running the pipeline:

```bash
sqlite3 data/secom.db "PRAGMA table_info(sensor_readings);"   # wafer_id has pk=1
sqlite3 data/secom.db "SELECT COUNT(*) FROM sensor_readings;"  # 1567
sqlite3 data/secom.db "SELECT role, COUNT(*) FROM column_registry GROUP BY role;"
sqlite3 data/secom.db "SELECT COUNT(*) FROM column_registry WHERE role='feature_eng';"  # 4
```

## Gotchas (each learned the hard way)

- **No `:param` tokens in SQL comments** — the SQLAlchemy binder counts them
  and the query fails with a binding error.
- **`sqlite3` CLI silently creates an empty DB at a wrong path** — if `PRAGMA`
  returns nothing, check the path before suspecting the loader.
- **`is_fail` has one home** — `validate.py`; if you find yourself writing
  `target.eq(1)`, you are creating a second home.
- **Notebooks never build paths from CWD** — the kernel's working directory is
  `notebooks/`; use `semcon.paths` constants or nothing.