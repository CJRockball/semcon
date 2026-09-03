<!-- Save as: validation.md (repo root) -->

# Validation — SECOM Pipeline

This document states what can go wrong in this pipeline and where each guard
lives. Every claim in the README traces back to a rule documented here.

## 1. Data extraction contract

The bronze layer (SQLite) stores raw values: target as −1/1, timestamps as
ISO-8601 TEXT. No encoding, no typing at rest.

Typing and semantic encoding happen at the read boundary:

- `extract.py` coerces timestamps via `pd.to_datetime(..., format="ISO8601")`
- `validate.py` is the single home of `is_fail` (0/1), created by
  `ensure_is_fail()` and registered in `column_registry` as
  `role=target, derived_from=target`. Consumers (`train_xgb`, `calibrate`,
  scoring) call `ensure_is_fail` — no consumer recodes labels itself, and the
  recode is never persisted back to bronze.

**Ordering is a pipeline invariant.** All positional logic assumes the frame
sorted by `(timestamp, wafer_id)` — timestamps are not guaranteed unique, so
the tie-break key is mandatory. The sort lives in one place (`extract.py`);
consumers inherit it. `splits.json` indices are meaningful only under this
ordering.

**Guards:** the migration regression test (`src/semcon/migration_test.py`,
pending graduation to `tests/`) compares old flat-file artifacts against DB
extraction and passes bitwise — post-migration runs reproduce pre-migration
predictions exactly. The fingerprint chain (`data_fingerprint()` in `db.py`)
records raw-file SHA-256 + SQL-text hash per run; two runs with the same
fingerprint read the same bytes.

**What it prevents:** silent divergence between "the data the old pipeline
saw" and "the data the new pipeline sees"; label-encoding drift between
consumers; tie-order instability at split boundaries.

## 2. Split discipline

Three time-ordered zones, defined by two config decisions
(`src/semcon/config.py`, each with its EDA provenance in a comment):

| Zone | Rows (0-indexed) | Count | Fails | Role |
|---|---|---|---|---|
| `cv` | 0–1308 | 1,309 | 90 | training + repeated CV |
| `holdout` | 1309–1539 | 231 | 14 | one-shot evaluation, spent exactly once |
| `excluded` | 1540–1566 | 27 | 0 | regime break; monitored by SPC only |

**CUTOFF = `2008-10-05 05:30:00`** sits strictly between the last CV wafer
(04:48:00) and the first holdout wafer (05:31:00). **EXCLUDE_AFTER** marks the
NaN-explosion regime break (EDA §3.5). Both are *decisions*: they change only
by a deliberate config edit, with the EDA reference, as their own commit.

The `split` column in the extracted frame is the single source of split truth
(SQL `CASE` in `sql/extract_wafers.sql`; `NULL` boundaries yield `unassigned`
so a pre-decision full frame never carries a false label). The equivalence
guard in `train_xgb.py` asserted that the SQL split reproduces the legacy
positional split exactly — fired and passed with the first CUTOFF-era runs
(2026-09-02), after which the positional logic was retired.

**Boundary rules:** no timestamp may equal a boundary (SQL `BETWEEN` is
inclusive); `extract.py` raises on an exact clash. Fail counts are
run-recorded (`n_train_fails` / `n_holdout_fails` in `splits.json`), not
folklore — see §7.

**Why the tail is excluded:** the SPC protocol charts show a
measurement-protocol change — clique-sensor missing rates go 36% → 94% → 100%
across the three phases. A value-based model cannot anticipate a protocol
change; including the tail would test the model on data the deployment
process would never produce.

## 3. Fit-inside-fold

Anything with learned parameters is fit inside the CV fold, never on the
full pool. This applies to:

- **Feature selection** (`selection.py`, Hedges' g filter) — fit per fold;
  the stable-29 are the post-CV intersection of fold-level survivors
- **Calibration** (`calibrate.py`, Platt/isotonic) — fit on out-of-fold
  predictions only
- **Threshold tuning** — on OOF predictions only; the holdout is evaluated
  once, at the frozen threshold

**What it prevents:** CV metrics inflated by information leaking from the
validation fold into the feature set, the calibration map, or the operating
point.

## 4. Column-quality rules and the one supervised exception

The retirement rules in `explore.py` (590 → 333 retired → 257 active) are
unsupervised: constant, missingness > 50%, dominant value > 99%, CV < 0.01,
< 5 unique values, correlated pairs at |r| > 0.95. Each retirement is written
to `column_registry` with the rule and threshold as its reason.

**One exception:** the NZV-rescue rule (`dev_fail_rate / fail_rate >= 2.0`)
uses the target. It is supervised and therefore carries mild leakage
exposure. It is applied globally today; now that CUTOFF exists, the queued
fix — compute the rescue on the CV pool only — is unblocked and is the one
open leakage item in this document.

**EDA disclosure:** the retirement rules were derived from EDA run on all
1,567 rows including the future holdout. All rules except the NZV rescue are
unsupervised, so the risk is minimal, but it is stated here rather than
assumed away.

## 5. Registry governance

`column_registry` is a ledger, not a cache. It records who created or retired
a column and why (`derived_from` lineage for engineered columns). It never
learns which features a model selected — the stable-29 live in the run folder
(`features.json`), versioned per run. The registry answers "is this column
usable at all"; the run answers "which usable columns won this time."

**Whoever creates a column registers it:** `db_ingest` (raw set), `extract`
(`split`), `feature_eng` (engineered columns), `validate` (`is_fail`).

**Guards:** the subset assert in `train_xgb.py` (`feats ⊆ active`) — a run
referencing a retired column fails loudly at load time. The inclusion
contract builds X from registry-active feature columns only, so key,
metadata, split, and target columns can never leak into the feature matrix.

## 6. Validation gate and snapshots

`validate.py` is the pipeline's gate, run between extraction and modeling
(`make validate`):

- **Schema** (pandera): key unique/non-null, timestamp typed, `target` ∈
  {−1,1}, `split` vocabulary, 590-sensor block complete and float64
- **Expectations**: row count, sensor count, all three split zones non-empty
  (a wrong boundary timestamp fails loudly here, not in a model metric)
- **Drift report**: per-column missingness vs the latest gold snapshot's
  frozen registry — report-only; SPC owns the response

Gold snapshots (`data/snapshots/gold/<snapshot_id>/`) freeze the exact matrix
the model saw: `matrix.parquet` (gitignored) + `manifest.json` + frozen
`registry.csv` (tracked). The lineage chain this closes:

raw-file sha256 → `ingestion_log` → registry → snapshot manifest → run
`config.json` → `model.ubj`.

**The 30-second test:** from any run folder, answering "exactly which bytes
trained this model?" must take under 30 seconds via snapshot_id → manifest →
fingerprint → `ingestion_log`.

## 7. Run-recorded expectations

Expectations travel with the run that produced them, never as constants in
consumers. `splits.json` records `n_train`, `n_holdout`, `n_train_fails`,
`n_holdout_fails`; `calibrate.py` asserts its reconstructed labels against
these, and asserts artifact pairings (`len(y_cv) == oof.shape[-1]`,
`len(y_hold) == len(p_hold)`).

**Case study (2026-09-02):** a hard-coded "88 CV fails" assert, calibrated to
the flat-file era, fired against the DB-era truth of 90. The run's own record
settled it: 90 + 14 + 0 = 104 — the partition closes; the old quoted numbers
never did (88 + 15 = 103). Constants in consumers go stale when decisions
move; run-recorded expectations cannot.

## 8. Monitoring boundary

SPC screens all non-degenerate raw sensors, including retired ones — the
monitoring net is deliberately wider than the modeling net. Phase-I limits
are frozen on the CV pool (median ± 3 × 1.4826 × MAD); Phase II applies them
unchanged to holdout + excluded tail. The alarm-rate jump is the drift
signal.

The 2026-09-01 run (pre-conversion, 257-feature frame): 457 non-degenerate
sensors screened, 133 degenerate, median Phase-I alarm 2.53% (vs 0.27%
nominal for normal data — the heavy-tailed baseline is measured, not
assumed), 63 drifting. Post-migration runs screen the wide frame; refresh
these numbers from the latest `*_spc` run when citing.

**What it catches:** protocol-layer drift (missing-rate explosions) that the
value layer is blind to — the two instruments measure different features of
the distribution, and both are needed.

## 9. Repository guards

`make hygiene` encodes two policies as checks, run before every merge:

- no `.db` file tracked in git (the DB is derived; `*.db` in `.gitignore`
  plus one-time `git rm --cached`)
- the data store has exactly one reader: only `db_ingest.py` touches the raw
  files and only the frozen `migration_test.py` references the legacy
  parquets — every other module goes through the DB extraction or consumes
  run artifacts