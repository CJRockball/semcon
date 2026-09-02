# Validation — SECOM Pipeline

This document states what can go wrong in this pipeline and where each guard
lives. Every claim in the README traces back to a rule documented here.

## 1. Data extraction contract

The bronze layer (SQLite) stores raw values: target as -1/1, timestamps as
ISO-8601 TEXT. No encoding, no typing at rest.

Typing and semantic encoding happen at the read boundary, in
`src/semcon/extract.py` only:

- timestamps coerced via `pd.to_datetime(..., format="ISO8601")`
- target recoded to 0/1 via `.eq(1)` in each consumer (`train_xgb`, `spc`,
  `sarimax`) at load time — one line, one place, never persisted back

**Guard:** the migration regression test (`src/semcon/migration_test.py`,
pending graduation to `tests/`) compares old flat-file artifacts against DB
extraction — sensor values, label mapping, timestamps — and passes bitwise.
The fingerprint chain (`data_fingerprint()` in `db.py`) records raw-file
SHA-256 + SQL-text hash per run; two runs with the same fingerprint read the
same bytes.

**What it prevents:** silent divergence between "the data the old pipeline
saw" and "the data the new pipeline sees."

## 2. Split discipline

Three time-ordered segments, defined by two boundaries:

| Segment | Rows (0-indexed) | Count | Role |
|---|---|---|---|
| CV pool | 0–1308 | 1309 | training + repeated CV |
| Holdout | 1309–1539 | 231 | one-shot evaluation, spent exactly once |
| Tail | 1540–1566 | 27 | excluded from training; monitored by SPC only |

**CUTOFF** (config.py) is the timestamp of row 1309: `2008-10-05 05:31:00`.
It defines the CV/holdout boundary. Currently the split is computed
positionally (`i_hold = n - tail_n - holdout_n`); setting CUTOFF activates the
equivalence guard in `train_xgb.py` asserting positional and timestamp splits
name the same rows.

**The tail is a separate decision, not part of CUTOFF.** It is excluded from
training because the SPC protocol charts (Section 6 of `spc_results.ipynb`)
show a measurement-protocol change — clique-sensor missing rates go
36% → 94% → 100% across the three phases. A value-based model cannot
anticipate a protocol change; including the tail would test the model on
data the deployment process would never produce.

**Guard:** the tail exclusion is a stated config decision (`tail_n = 27`),
not an emergent property of the data.

## 3. Fit-inside-fold

Anything with learned parameters is fit inside the CV fold, never on the
full pool. This applies to:

- **Feature selection** (`selection.py`, Hedge's g filter) — fit per fold;
  the stable-29 are the post-CV intersection of fold-level survivors
- **Calibration** (`calibrate.py`, Platt/isotonic) — fit on out-of-fold
  predictions only

**What it prevents:** CV metrics inflated by information leaking from the
validation fold into the feature set or the calibration map.

## 4. Column-quality rules and the one supervised exception

The exclusion rules in `explore.py` (590 → 257 active) are unsupervised:
missingness > 50%, dominant value > 99%, CV < 0.01, < 5 unique values,
correlated pairs at |r| > 0.95. Each retirement is written to
`column_registry` with the rule and threshold as its reason.

**One exception:** the NZV-rescue rule (`dev_fail_rate / fail_rate >= 2.0`)
uses the target. It is supervised and therefore carries mild leakage
exposure. It is applied globally today; the fix — compute it on the CV pool
only — is queued behind the CUTOFF decision.

**EDA disclosure:** the exclusion rules were derived from EDA run on all
1,567 rows including the future holdout. All rules except the NZV rescue are
unsupervised, so the risk is minimal, but it is stated here rather than
assumed away.

## 5. Registry governance

`column_registry` is a ledger, not a cache. It records who retired a column
and why. It never learns which features a model selected — the stable-29
live in the run folder (`features.json`), versioned per run. The registry
answers "is this column usable at all"; the run answers "which usable
columns won this time."

**Guard:** the subset assert in `train_xgb.py` (`feats ⊆ active`) — a run
referencing a retired column fails loudly at load time.

## 6. Monitoring boundary

SPC screens all 590 raw sensors, including retired ones — the monitoring net
is deliberately wider than the modeling net. Phase-I limits are frozen
(median ± 3 × 1.4826 × MAD); Phase II applies them unchanged. The alarm-rate
jump is the drift signal.

The 2026-09-01 run: 457 non-degenerate sensors screened, 133 degenerate,
median Phase-I alarm 2.53% (vs 0.27% nominal for normal data — the
heavy-tailed baseline is measured, not assumed), 63 drifting.

**What it catches:** protocol-layer drift (missing-rate explosions) that the
value layer is blind to — the two instruments measure different features of
the distribution, and both are needed.