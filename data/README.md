<!-- Save as: data/README.md -->

# Dataset: SECOM — Semiconductor Manufacturing Pass/Fail

Yield prediction from inline sensor surveillance. Each row is one production
entity (wafer) with a pass/fail result from in-house line testing and the
timestamp of that test point.

## Source and license

- **Source:** [UCI Machine Learning Repository, dataset 179](https://archive.ics.uci.edu/dataset/179/secom)
- **Donors:** Michael McCann, Adrian Johnston (donated 18 Nov 2008)
- **License:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — sharing
  and adaptation permitted with attribution
- **Citation:** McCann, M. & Johnston, A. (2008). SECOM [Dataset]. UCI Machine
  Learning Repository. https://doi.org/10.24432/C54305

The donors' own framing: not all monitored signals are equally valuable; the
task is to identify the signals that drive yield excursions downstream, to
"increase process throughput, decrease time to learning and reduce per unit
production costs." That is exactly the framing of this repo.

## Raw data (`data/raw/` — not tracked in git)

| File | Content |
|---|---|
| `secom.data` | 1,567 × 590 sensor measurements, space-separated text |
| `secom_labels.data` | Two columns: label (−1 = pass, +1 = fail) and test timestamp |
| `secom.names` | UCI metadata |

Facts that matter before touching the data:

- **Class imbalance:** 104 fails out of 1,567 wafers (6.6%, ~14:1). Predicting
  "pass" everywhere scores 93.4% accuracy — accuracy is a meaningless metric here.
- **Missing values:** present throughout, at varying intensity per feature,
  encoded as the MatLab-style literal `NaN`. Missingness is *structured*
  (sensor dropout blocks), not random — see the EDA notebook.
- **Anonymous features:** sensors are integer indices, not physical names.
  Root-cause statements can only go as far as feature groups, not equipment.
- **Time-ordered:** the timestamp orders wafers through the process. The final
  ~27 wafers sit in a different missingness regime (see EDA §3.5) — do not
  shuffle blindly.
- **Donor baseline** (kernel-ridge classifier, 10-fold CV, 40 selected
  features): balanced error rate 33.5–40.1%, true-positive rate ~48–60%,
  true-negative rate ~72–78%, depending on the feature-selection method.
  Anything claiming far above this should be checked for leakage.

Fetch via `pip install ucimlrepo` then `fetch_ucirepo(id=179)`, or download
from the UCI page above.

## Processing pipeline

Two stages, each a script in `src/semcon/`. The target file
(`dfy_v1.parquet`: `target` as 0/1, plus `timestamp`) is written once by
`explore.py` and carried through unchanged.

### Stage 1 — `explore.py` → `dfX_v1.parquet` (590 → 257 features)

Removes features that are inherently uninformative, keeping a record of every
decision in `artifacts/diagnostic.parquet`:

1. **Constant features** — zero variance, dropped.
2. **High-missing features** — more than 50% NaN, dropped as continuous inputs
   (their dropout *patterns* are later mined in Stage 2).
3. **Near-zero-variance / concentrated values** — dominant value in >99% of
   rows, CV < 0.01, or fewer than 5 unique values. With a rescue rule: a flagged
   feature is kept if its rare deviant values are ≥2× enriched for fails
   (≥5 deviant rows) — rare excursions can be the failure signal.
4. **Redundant pairs** — for |r| > 0.95, the member with more missing data is
   dropped.

### Stage 2 — `feature_eng.py` → `dfX_v2.parquet` (257 → 261 features)

Four EDA-derived features capturing *missingness structure* rather than sensor
values. They are computed from the raw 590-feature matrix (`build_features`),
because several source columns were dropped in Stage 1 and no longer exist in
the cleaned frame — clique 14's members (50.7% NaN) were themselves removed by
the >50% rule. The build self-validates with asserts anchored to the EDA
dropout counts (794 / 715 wafers).

- **`miss_clq14`** (794 wafers), **`miss_clq23`** (715 wafers) — binary
  indicators that a wafer belongs to dropout clique 14 / 23: groups of sensors
  that go missing on exactly the same wafers (co-missingness connected
  components, EDA §3.3). Both cliques are target-associated (clique 14:
  OR ≈ 0.49, p ≈ 0.0008) with a notable direction: fails are missing *less*
  often — a measurement-protocol/routing signal, not sensor degradation
  (EDA §3.4, §3.7).
- **`miss_block5`** — indicator that a wafer is missing the block-5 channel
  group, the final station group in the repeated block layout. The only block
  where fails edge *ahead* on missingness — a weak effect, included as a
  label-free probe to be re-examined on the time-blocked holdout
  (EDA §3.5, §3.7).
- **`row_missing_rate`** — per-wafer fraction of the 590 raw channels missing
  (float32), computed on the raw matrix for consistency with EDA §3.6. A
  data-quality covariate, not an expected signal: row missingness is nearly
  identical across classes (pass 4.55% vs fail 4.32%).

`dfX_v2.parquet` + `dfy_v1.parquet` are the modeling inputs.

## Processed files (`data/processed/` — not tracked in git)

| File | Shape | Content |
|---|---|---|
| `dfX_raw.parquet` | 1,567 × 590 | Raw features as parquet (lossless convenience copy) |
| `dfy_raw.parquet` | 1,567 × 2 | Raw labels + timestamp as parquet |
| `dfX_v1.parquet` | 1,567 × 257 | Cleaned sensor features |
| `dfy_v1.parquet` | 1,567 × 2 | `target` (0/1), `timestamp` |
| `dfX_v2.parquet` | 1,567 × 261 | v1 + engineered missingness features |