# Modelling and results

[Home](../README.md) · [Previous: Data and preprocessing](01_data_and_preprocessing.md) · [Next: Process monitoring and forecasting](03_process_monitoring_and_forecasting.md)

| Item | Summary |
|---|---|
| Objective | Build a reproducible yield-risk model whose outputs can support ranking, triage, and downstream monitoring. |
| Main artifacts | Training runs, feature contracts, calibrated model artifacts, holdout evaluation outputs, SHAP summaries, and `validation.md`. |
| Key claim | Model quality is assessed with protected temporal evaluation and calibrated probabilities rather than accuracy alone. |
| Boundary | Predictive importance and SHAP attribution are associational; they do not identify a physical root cause or prove a process effect. |

## Decision objective

The modelling objective is to estimate and rank the probability of an adverse quality outcome for each observation. In an operational setting, this supports a constrained decision: inspection, engineering review, and metrology capacity are limited, so the system should help prioritize the observations or lots where extra attention is most justified.

This is not an accuracy-first classification exercise. With an imbalanced outcome, a trivial classifier can appear accurate while missing many of the observations that matter. The project therefore evaluates discrimination, ranking, calibration, and thresholded triage behavior. A useful probability must be both able to separate higher- from lower-risk observations and sufficiently calibrated to support consistent operational thresholds.

## Training pipeline

The modelling pipeline is intentionally modular:

```text
Extracted, time-bounded data
    → feature engineering and registry controls
    → baseline and selected-feature XGBoost training
    → cross-validation on the development period
    → calibration on held-out development predictions
    → protected chronological holdout evaluation
    → artifact registration and explainability outputs
```

The primary model family is gradient-boosted trees using XGBoost with a binary logistic objective. This is a practical choice for sparse, nonlinear, mixed-quality tabular sensor data: it can represent interactions and nonlinear thresholds while remaining compatible with explicit feature contracts, calibrated post-processing, and SHAP-based explanation.

The pipeline trains a baseline configuration and a selected-feature configuration. The comparison is important because dimensionality reduction is not treated as automatically beneficial. A smaller feature set is preferable only if it preserves or improves the metrics that matter, simplifies the scoring contract, and remains stable on the protected holdout.

## Validation discipline

The source data is ordered in time, so validation follows the temporal structure described in the data documentation. Internal folds are used within the development period for model selection. The final time-tail holdout is protected from feature-selection decisions, model selection, and calibration fitting until final evaluation.

This separation avoids a subtle but common source of optimistic bias. If the holdout is used repeatedly to choose features, tune thresholds, or select a calibrator, it gradually becomes a development set even if it retains the label “test.” Here, the holdout is intended to answer a single question: how does the selected pipeline behave on later observations that were not used to shape it?

The exact split policy, canonical run references, and results belong in [`validation.md`](../validation.md). The README should summarize only the final chosen metrics; this page explains how those metrics should be interpreted.

## Metrics that matter

The project should present model results in the order that matches the decision problem.

| Metric or view | Why it is included | Interpretation |
|---|---|---|
| PR-AUC | The positive class is relatively rare | Measures ranking quality where precision and recall matter; compare against prevalence rather than treating the value in isolation |
| ROC-AUC | Provides a familiar global discrimination view | Useful but can be flattering under imbalance, so it is not sufficient on its own |
| Recall / precision at a triage threshold | Represents a capacity-constrained action | Shows how many adverse outcomes are found and what inspection burden the threshold creates |
| Calibration curve and Brier-style evidence | Tests whether probabilities behave like probabilities | Important because thresholds, scorecards, drift metrics, and policy depend on calibrated risk |
| Holdout results | Tests temporal generalization | The most credible evidence of how the selected pipeline behaves on later data |

Accuracy should not lead the story. It is easy to obtain a high accuracy score when the negative class dominates, even if the system provides little value for identifying scarce high-risk cases.

## Calibration and operational ranking

Raw boosted-tree scores are useful for ranking but should not automatically be treated as probabilities. The project therefore supports probability calibration, using a fitted calibration artifact associated with the training run. Scoring resolves both the model and its optional calibrator as a bundle, producing raw and calibrated outputs where applicable.

Calibration changes the question from “which observations rank higher?” to “does a value such as 0.12 represent a comparable level of estimated risk across batches?” This distinction is operationally important. A risk threshold used to prioritize inspection should not drift merely because the score scale has changed. The calibrated score is therefore the appropriate input for scorecards, monitoring summaries, and surrogate-response analysis.

The downstream scorecard layer translates scored observations into a more usable decision surface: ranked observations, risk distribution summaries, threshold counts, and comparisons against a reference batch when available. The system remains decision support. A high risk score is a prompt to inspect evidence and context, not an automatic disposition decision.

## Explainability and engineering interpretation

The project writes TreeSHAP summaries for trained models. SHAP provides a local and global description of how the model’s prediction changes with its inputs under the fitted model. It is useful for identifying features worth investigating, for checking whether the model relies heavily on missingness patterns, and for selecting eligible raw-sensor candidates for the surrogate DOE workflow.

The correct interpretation is careful:

- A feature with high SHAP magnitude is influential to the model, not necessarily physically causal.
- A missingness feature can be strongly predictive while representing a measurement or logging regime rather than a controllable process parameter.
- A sensor’s apparent direction of effect may depend on interactions and on the observed data distribution.
- Any proposed engineering action requires process knowledge and controlled confirmation beyond this dataset.

This is why the DOE module filters candidate factors to raw sensor features and records that its recommendations are surrogate hypotheses rather than recipe changes.

## Reading the results honestly

The project should report one canonical selected run and its protected-holdout results, supported by the registered artifacts. Avoid presenting a long list of trial runs in the main narrative. Instead, show the baseline-versus-selected comparison, the calibration assessment, and the operational effect of the chosen triage threshold. The full run registry preserves the experiment trail for readers who want to audit it.

A credible conclusion has the following form:

> The selected pipeline provides a calibrated risk-ranking signal that is evaluated on later data and can prioritize inspection. It does not establish causal process relationships, identify a particular chamber, or replace disposition authority.

That level of restraint is a strength. It keeps the model aligned with what the data and validation design can actually support.

---

**Previous:** [Data and preprocessing](01_data_and_preprocessing.md)  
**Next:** [Process monitoring and forecasting](03_process_monitoring_and_forecasting.md)