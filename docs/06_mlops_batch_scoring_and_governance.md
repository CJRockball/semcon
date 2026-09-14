# Batch scoring, monitoring, and retraining governance

[Home](../README.md) · [Previous: Dashboard and decision support](05_dashboard_and_decision_support.md)

| Item | Summary |
|---|---|
| Objective | Operate the model as an auditable batch workflow: validate inputs, score with a frozen contract, monitor behavior, and govern retraining recommendations. |
| Main artifacts | Model and calibrator runs, score outputs, scorecards, lot-replay outputs, monitoring reports, monitor index, retrain decision artifacts, tests, and CI. |
| Key claim | Monitoring and retraining are separate, explicit decisions; a retraining recommendation never silently replaces the active model. |
| Boundary | Batch replay is a deterministic portfolio simulation, not a live MES/FDC feed or evidence of production deployment. |

## Lifecycle

The operational design follows a controlled sequence rather than treating inference as a standalone `predict()` call:

```text
New or replayed batch
    → validate schema and feature availability
    → resolve registered training run and calibrator
    → rebuild engineered features and enforce feature contract
    → score and rank observations
    → generate scorecard
    → monitor input and output health
    → apply retraining policy
    → review through dashboard and artifacts
```

Each stage has a distinct responsibility. The scorer produces predictions. The scorecard summarizes those predictions for an operational audience. The monitor evaluates whether inputs and outputs still resemble the reference conditions. The retrain-policy module considers monitoring history and policy criteria. Keeping these roles separate makes the workflow easier to test, audit, and reason about.

## Batch contract and scoring

Batch scoring resolves a selected training run and its model contract instead of relying on mutable global state. The contract defines the expected feature names and ordering. Before inference, the workflow rebuilds engineered features through the same feature-engineering path used during development and checks the resulting frame against the registered contract.

This prevents several common deployment failures:

- An incoming batch contains a differently named or omitted sensor column.
- Feature order changes silently between training and scoring.
- An engineered data-quality feature is missing because scoring bypassed the training transformation.
- A model is paired accidentally with an incompatible calibrator or feature list.

The scoring output includes the raw model score and the calibrated probability when a calibrator is available. The calibrated value is the preferred operational quantity for risk thresholds, scorecards, and monitoring summaries. All score outputs are tied back to the run identifiers used to create them.

## Replay batches and scenarios

The project includes a lot-replay utility to exercise the scoring and monitoring pipeline without claiming access to a live fab stream. It turns held-out or simulated observations into timestamped incoming lots and supports deterministic scenarios that stress the decision logic.

The purpose is integration evidence, not synthetic-data theatre. The replay path should preserve realistic correlation and missingness structure where possible, then add controlled perturbations only when testing a specific response. A useful scenario set is:

| Scenario | Intended behavior | Expected decision |
|---|---|---|
| Nominal / batch A | Reference-like incoming lots | `IN_CONTROL` |
| Sensor drift / batch B | Input-feature change without material risk-distribution deterioration | `INVESTIGATE_CHAMBER` |
| Excursion / batch C | Persistent input and/or output-health deterioration | `RETRAIN_RECOMMENDED` after policy conditions are met |

The labels are test fixtures, not claims about actual fab events. A replayed sensor shift is used to prove that the monitoring logic can detect a defined perturbation; it does not demonstrate that the source dataset contains a verified chamber excursion.

## Monitoring responsibilities

`monitor.py` owns surveillance. It loads the context needed to interpret new scored batches, calculates output-health and feature-drift summaries, evaluates deterministic rules, and writes append-only records and inspection artifacts.

Two evidence streams are combined:

- **Output health:** calibrated-risk distribution, triage rate, rolling behavior, and persistence of out-of-control lots.
- **Feature drift:** deviation of selected monitored features from their frozen Phase-I references, including robust location and dispersion comparisons.

The monitor emits a concise verdict and reason. The verdict does not determine business action by itself; it provides a standardized routing signal.

| Verdict | Meaning | Default human response |
|---|---|---|
| `IN_CONTROL` | No configured material input or output-health condition is active | Continue normal scoring and scheduled surveillance |
| `INVESTIGATE_CHAMBER` | Feature-level drift is present without the configured persistent risk shift | Review measurement and process context; verify data lineage and available tool information |
| `RETRAIN_RECOMMENDED` | Persistent output-health deterioration or combined evidence meets the defined rule | Start governed model-review and retraining evaluation; do not automatically deploy a replacement |

The middle label is intentionally a shorthand. Because SECOM lacks tool and chamber identity, an `INVESTIGATE_CHAMBER` result cannot name or prove a physical chamber fault.

## Retraining is governed, not automatic

A model should not be retrained merely because a monitoring statistic changed. The change may originate in the process, product mix, measurement system, data extraction, or label delay. Blind retraining can normalize corrupted inputs, encode a transient anomaly, or degrade a model whose performance has not yet been measured on matured labels.

`retrain_trigger.py` therefore owns policy evaluation rather than model fitting. It consumes monitoring evidence and produces a retraining recommendation when explicit criteria are met. A recommendation should initiate the following review sequence:

1. Verify data lineage, schema, ingestion behavior, and missingness changes.
2. Confirm that the alert is persistent and not explained by a known operational event.
3. Wait for or obtain sufficiently mature outcome labels for performance evaluation where feasible.
4. Train a candidate model through the same controlled pipeline.
5. Compare candidate and incumbent on the protected evaluation protocol, including discrimination, calibration, and operational triage behavior.
6. Register the decision and promote only through an explicit approval step.

This design is intentionally closer to governed model maintenance than to simplistic continuous training.

## Artifact lineage

The project uses timestamped run directories and index files to make artifacts discoverable. A reader should be able to trace a dashboard conclusion backward through its source artifacts:

```text
Dashboard status
    → monitoring verdict/report
    → scorecard and scored batch
    → calibrator and training run
    → feature contract and configuration
    → extraction/snapshot boundary
```

The key artifact classes are model runs, calibration runs, score outputs, scorecards, monitoring records, DOE runs, and retraining-decision artifacts. Append-only indexes preserve a compact experiment and operations ledger. This does not make the system enterprise-grade governance, but it demonstrates the essential principle: model behavior should be attributable to versioned inputs, configuration, and artifacts.

## Tests, CI, and reproducibility

The test suite covers the data layer, modelling helpers, scoring and scorecards, SPC, SARIMAX, surrogate DOE, Dash data/figure behavior, monitoring, and retraining policy. Tests should focus on contracts and decision invariants: expected columns, deterministic scenario outcomes, correct verdict routing, no accidental mutation of production registries, and artifact creation under controlled temporary paths.

The repository Makefile provides a practical entry point for the pipeline and test commands. Before presenting a command in the main README, it should be verified against the final console-script names in `pyproject.toml`. The core checks should remain straightforward:

```bash
make test
make hygiene
```

A clean, repeatable batch replay is especially valuable portfolio evidence because it exercises the full chain from incoming-lot simulation through scoring, monitoring, and governed recommendation without overstating it as live production infrastructure.

---

**Previous:** [Dashboard and decision support](05_dashboard_and_decision_support.md)