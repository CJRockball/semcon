# Model validation record

[Home](README.md) · [Modelling and results](docs/02_modelling_and_results.md) · [MLOps and governance](docs/06_mlops_batch_scoring_and_governance.md)

> **Document purpose:** This is the controlled, human-readable validation record for the canonical portfolio model. It complements the machine-readable run ledgers in `artifacts/index.csv` and `artifacts/index_monitor.csv`; it does not replace them.

| Document control | Value |
|---|---|
| System | Semiconductor yield-risk decision support (`semcon`) |
| Current canonical model | `20260914_093559_xgb_sel` |
| Bound calibration run | `20260914_093608_cal_platt` |
| Canonical rebuild date | 2026-09-14 |
| Validation status | Accepted for reproducible portfolio demonstration |
| Document owner | Repository maintainer |
| Scope | Historical SECOM data, deterministic local batch replay, and artifact-based review |
| Prohibited interpretation | This record is not a production-release approval or authorization for autonomous manufacturing control |

## 1. Intended use

The system estimates and ranks the risk of an adverse quality outcome from high-dimensional semiconductor sensor measurements. Its intended role is **decision support**: prioritize observations or replayed lots for inspection and engineering review, summarize batch-level risk, and provide structured evidence when input or output behavior differs from a defined reference state.

The model is not intended to:

- Release, hold, scrap, or disposition wafers or lots.
- Change a process recipe, tool setting, or maintenance plan.
- Attribute an excursion to a particular chamber, tool, supplier, process layer, or operator.
- Establish a causal sensor-to-yield relationship.
- Deploy a replacement model automatically after a monitoring event.

All operational actions remain human-owned. The system’s outputs are risk estimates, monitoring signals, and governed recommendations—not commands.

## 2. Data lineage and scope

The project uses the UCI SECOM dataset as a portfolio proxy for wide, sparse semiconductor manufacturing data. Source files are ingested into SQLite and consumed through a validated extraction path rather than repeatedly read from unconstrained files. The pipeline records derived artifacts under `data/` and `artifacts/`, including run configuration, feature contract, and score outputs.

### Temporal policy

Chronology is part of the evaluation design. The development period is separated from a protected chronological tail holdout. A later regime with a documented missing-data deterioration is excluded according to the configured boundary. Internal validation is performed within the development period; the tail holdout is reserved for final evaluation and is not used to select features, tune the final model, or fit probability calibration.

The configured pipeline uses five internal folds and a fixed tail holdout of 231 observations. Exact cutoff and exclusion parameters are versioned in `src/semcon/config.py` and associated run metadata.

### Missingness and feature policy

Missingness is treated as a potentially informative observation-process signal, not merely a value to fill. The data workflow screens columns for excessive missingness, low information, and redundancy. Selected missingness-cluster indicators may be engineered as model inputs where supported by exploration.

These engineered missingness features are predictive features only. They are not physical process knobs and are deliberately excluded from surrogate DOE factor selection. The model feature list is stored as a run-level contract; scoring and downstream analysis resolve the contract rather than relying on an assumed column order.

### Data limitations

SECOM does not include the manufacturing metadata needed for a live fab deployment. In particular, it lacks validated tool and chamber identity, recipe version, product and layer context, lot and wafer genealogy, maintenance events, metrology context, wafer-map structure, and physical sensor units. The model therefore cannot make chamber-level, recipe-level, or causal manufacturing claims.

## 3. Canonical model specification

| Component | Canonical reference | Validation relevance |
|---|---|---|
| Selected training run | `20260914_093559_xgb_sel` | Current portfolio candidate; selected-feature XGBoost model |
| Baseline training run | `20260914_093551_xgb_base` | Comparator for feature-selection and model-complexity decisions |
| Calibration run | `20260914_093608_cal_platt` | Platt calibration artifact associated with the selected training run |
| Feature contract | Training-run `features.json` artifact | Defines required feature names and order for scoring |
| Model artifact | Training-run `model.ubj` artifact | Frozen fitted XGBoost booster used for canonical scoring |
| Scoring outputs | `artifacts/scores/20260914_093612_score__batch_a_clean` and later score runs | Evidence of batch-scoring behavior under the canonical model bundle |
| Monitoring ledger | `artifacts/index_monitor.csv` | Append-only record of monitoring evidence and verdicts |
| Retraining decision | `artifacts/retrain/latest_decision.json` | Policy output; not an automatic model-promotion event |

The selected model and calibration artifact must be consumed as a bundle. Scoring rebuilds engineered features, validates the feature contract, and emits calibrated risk when the calibration artifact is available. This reduces the risk of incompatible feature ordering, missing engineered columns, or accidental pairing of a model with the wrong calibration object.

## 4. Validation protocol

### Development and holdout separation

The selected pipeline is developed using the historical development window and internal cross-validation. The chronological holdout is preserved for final evaluation. This design is more realistic than an unrestricted random split because a deployed model scores later observations, not randomly sampled replicas of its own training period.

### Evaluation criteria

The validation assessment considers multiple dimensions:

| Dimension | Evidence | Why it matters |
|---|---|---|
| Discrimination | PR-AUC and ROC-AUC from registered evaluation artifacts | Tests whether higher-risk observations are ranked above lower-risk observations |
| Imbalance-aware performance | PR-AUC interpreted against positive-class prevalence | Avoids overstating accuracy or ROC-AUC in an imbalanced problem |
| Operational utility | Precision, recall, and triage volume at the configured threshold | Relates model output to constrained inspection capacity |
| Probability reliability | Calibration outputs and calibrated scoring behavior | Supports risk thresholds, scorecards, and distribution monitoring |
| Temporal generalization | Protected chronological-holdout evaluation | Tests behavior on observations later than the development period |
| Reproducibility | Versioned configuration, run artifacts, feature contracts, tests, and clean rebuild | Demonstrates that conclusions do not depend on hidden local state |

The authoritative metric values are the files written by the canonical run and registered in `artifacts/index.csv`. This record intentionally does not duplicate numerical values that should be read directly from generated evaluation artifacts; duplicating them manually creates an avoidable risk of drift between code, artifacts, and documentation.

### Validation checks

The canonical rebuild should be considered valid only when the following checks are complete:

- The model and calibration artifacts exist and resolve successfully as a compatible bundle.
- The selected feature contract exists and batch scoring passes contract validation.
- The protected-holdout evaluation artifacts exist for the selected run.
- Calibration artifacts exist and the scoring path emits calibrated probabilities.
- `make test` passes from the final source state.
- `make hygiene` passes, confirming repository policy on generated databases and controlled data reads.
- The Dash application loads against the rebuilt artifact state.
- Batch replay, monitoring, and retrain-policy artifacts are generated from the canonical run family.

## 5. Canonical operational evidence

The 2026-09-14 clean rebuild generated a canonical chain of artifacts under the selected model and calibrator:

```text
20260914_093559_xgb_sel
  → 20260914_093608_cal_platt
  → batch scoring and scorecards
  → monitoring records in artifacts/index_monitor.csv
  → retrain-policy output in artifacts/retrain/latest_decision.json
```

### Batch replay evidence

| Batch label | Purpose | Interpretation boundary |
|---|---|---|
| `batch_a_clean` | Nominal replay scenario | Demonstrates reference-like batch scoring and normal-state handling |
| `batch_b_shift` | Controlled input-shift scenario | Tests feature/process-investigation routing; it does not identify a real chamber fault |
| `batch_c_dropout` | Controlled degradation scenario | Tests adverse input/output-health conditions and persistence policy |
| `holdout_replay` | Chronological holdout replay | Connects operational scoring to the protected evaluation partition |

These are deterministic portfolio scenarios. They are not historical production incidents and should not be represented as evidence that a real fabrication facility experienced the named conditions.

### Monitoring and policy evidence

Monitoring evaluates two evidence streams: input-feature health against frozen reference limits and output health from calibrated-risk behavior and triage-rate persistence. The resulting verdicts route a human response:

| Verdict | Meaning | Required response |
|---|---|---|
| `IN_CONTROL` | No configured material feature or output-health condition is active | Continue scheduled scoring and surveillance |
| `INVESTIGATE_CHAMBER` | Input drift is present without the configured persistent output-risk shift | Review process, measurement, and data-lineage context; do not infer a specific chamber from SECOM alone |
| `RETRAIN_RECOMMENDED` | Persistence or combined-evidence criteria meet the retrain policy | Open governed candidate-model review; do not promote automatically |

The canonical policy artifact records a retraining recommendation after the configured persistence condition is met across the replay evaluation window. This is validation of policy routing, not evidence that retraining improves the model or that a replacement has been approved.

## 6. Surrogate DOE validation boundary

The surrogate DOE workflow is evaluated as an analysis and hypothesis-generation tool, not as a causal experiment. It selects eligible raw sensor features from the model context, defines factor levels from observed Q10/Q50/Q90 support, applies factorial contrasts over an observed background frame, and evaluates the calibrated model response.

The workflow writes factor metadata, design matrices, surrogate predictions, effect and interaction summaries, and out-of-distribution diagnostics. OOD checks use observed-background support so that model predictions at unsupported factor combinations are visibly flagged.

A DOE output may generate a testable engineering hypothesis. It cannot validate a recipe adjustment, prove a process effect, or authorize a change without a controlled physical experiment with safe ranges, randomization, blocking, and measured outcomes.

## 7. Residual risks and limitations

The canonical model is suitable for **reproducible portfolio demonstration** under the stated data boundary. Residual risks prevent it from being treated as a live manufacturing system:

- Historical labels and missingness patterns may not represent a future production population.
- The source data does not identify the physical process context necessary for root-cause attribution.
- Probability calibration can deteriorate as process conditions, measurement systems, or class prevalence change.
- A monitored feature drift may represent a process change, logging change, data-quality issue, or changing product mix.
- An elevated risk score is an inspection-priority signal, not a confirmed defect.
- Surrogate DOE results remain conditional on the trained model and observed data support.
- The replay pipeline demonstrates interfaces and policy behavior; it is not a substitute for a MES/FDC-integrated operational deployment.

## 8. Validation decision

**Decision:** `20260914_093559_xgb_sel` with `20260914_093608_cal_platt` is accepted as the canonical model bundle for reproducible portfolio demonstration.

**Decision basis:** The final run family provides a registered selected model, compatible calibration artifact, feature contract, batch-scoring outputs, scorecards, monitoring records, retrain-policy output, surrogate DOE artifacts, and curated dashboard screenshots. The repository is designed to support repeatable pipeline execution and automated tests from a clean derived-artifact state.

**Not approved for:** Live fab deployment, autonomous product disposition, automated recipe/tool change, causal root-cause attribution, or automatic replacement-model promotion.

**Required before any future candidate promotion:** Re-run the controlled validation protocol; compare candidate and incumbent on the protected evaluation design; assess calibration and threshold behavior; review data lineage and label maturity; update this record with a new ledger entry; and obtain explicit human approval.

## Appendix A. Append-only validation ledger

Do not edit a historical ledger row to make it look current. Add a new dated record when model, data, policy, or validation evidence changes materially. The machine-readable run indexes remain the detailed execution record; this table captures the human validation decision.

| Date | Event type | Candidate / active bundle | Data or policy change | Evidence reviewed | Decision | Owner / reviewer | Notes |
|---|---|---|---|---|---|---|---|
| 2026-09-14 | Canonical clean rebuild | `20260914_093559_xgb_sel` + `20260914_093608_cal_platt` | Fresh end-to-end pipeline run; batch replay, monitoring, retrain-policy, and DOE artifacts regenerated | Run registry, model/calibration artifacts, score outputs, monitoring ledger, retrain decision, DOE artifacts, tests and hygiene checks | Accepted for reproducible portfolio demonstration | Repository maintainer | Not a production release; prior run family remains historical evidence |

## Appendix B. Change-control template

Add a row to the ledger and update the relevant sections above when any of the following occurs:

| Change category | Examples | Minimum required review |
|---|---|---|
| Model change | New algorithm, hyperparameters, feature selection, or threshold policy | Candidate-versus-incumbent comparison, protected-holdout evaluation, calibration review |
| Data change | New extraction window, missingness regime, source schema, or labels | Lineage review, schema validation, drift assessment, repeat validation as needed |
| Calibration change | New method or recalibrated model | Reliability assessment and threshold-impact review |
| Monitoring change | New thresholds, features, persistence rules, or verdict logic | Back-test or deterministic scenario verification and policy sign-off |
| Retraining-policy change | New trigger or promotion rule | Governance review, test coverage, and explicit update to intended-use boundaries |