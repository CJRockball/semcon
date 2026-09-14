# Dashboard and decision support

[Home](../README.md) · [Previous: Surrogate DOE](04_surrogate_doe.md) · [Next: Batch scoring and retraining governance](06_mlops_batch_scoring_and_governance.md)

| Item | Summary |
|---|---|
| Objective | Present scored-risk, process-health, and model-health evidence in a form that supports a human operational decision. |
| Main artifacts | Dash application, data-loading layer, Plotly figure builders, scorecards, monitoring index, and registered run artifacts. |
| Key claim | The dashboard consolidates evidence and routes attention; it does not replace engineering judgment or automate product disposition. |
| Boundary | Dashboard signals inherit the limits of the source data, model, and monitoring logic. They are not real-time MES/FDC integration. |

## Intended user and decision

The Dash application is designed as a decision-support surface for a process, yield, quality, or data-science user who needs to answer a small number of operational questions quickly:

- Which incoming observations or lots should be prioritized for inspection?
- How does the current batch compare with the reference or expected risk distribution?
- Are key model inputs showing signs of drift?
- Is the situation consistent with normal operation, an investigation, or a governed model review?
- What model run, calibration artifact, and monitoring record produced the displayed conclusion?

The dashboard does not create a new analytical path separate from the pipeline. It reads registered artifacts from training, calibration, scoring, SPC, forecasting, monitoring, and retraining workflows. This keeps the visual layer traceable to the same reproducible outputs used elsewhere in the repository.

## Operational console

The primary Dash interface consolidates batch-level risk distributions, scorecard indicators, and lot prioritization queues into an integrated engineering dashboard:

![Dashboard Overview](../assets/screenshots/dash_over2.png)
*Figure 5.1: Primary Dash operational view showing active batch selection (`batch_a_clean`), wafer risk score distributions, and triage prioritization queue.*

## Calibration and failure risk analysis

Accurate probability estimation is essential for reliable thresholding. The failure probability analysis panel displays empirical calibration behavior alongside the operational triage boundary:

![Calibration and Failure Probability](../assets/screenshots/dash_cal_fail2.png)
*Figure 5.2: Calibrated probability curve and failure score triage panel, verifying consistent operational mapping from raw boosted scores to actionable risk levels.*

## Decision flow

```text
Registered model and calibrator
    + incoming/replayed batch
    → batch score and scorecard
    → feature and output-health monitoring
    → dashboard panels
    → human review and action
```

A useful dashboard should make the sequence visible rather than presenting a collection of unrelated charts. The reader should be able to move from an elevated risk summary to the affected batch, then to the relevant feature or monitoring evidence, and finally to the recommended next action.

## Scored-risk views

The scoring layer produces calibrated probability estimates and ranking information. The dashboard exposes both the overall distribution and the practical queue of observations requiring attention. High-risk results are shown with enough context to support a decision: score, rank, batch or lot label, timestamp where available, and the relevant model/run identifiers.

Scorecard views are especially important because they translate model output into operational language. Rather than asking a user to interpret a raw prediction file, a scorecard can summarize volume, elevated-risk counts, thresholded triage rate, distribution shift relative to a reference batch, and the highest-priority records.

A threshold is an operating choice, not a universal truth. The dashboard makes clear which threshold is being applied and avoids implying that a probability above the threshold is a confirmed defect. It is an inspection-priority signal derived from the fitted and calibrated model.

## Process and model-health views

The dashboard brings together signals that are easy to confuse when viewed separately:

| View | Question answered | Typical interpretation |
|---|---|---|
| Risk distribution | Has the current scored population shifted? | A shift may reflect a process change, input change, product mix difference, or model/data issue |
| Feature drift | Are monitored input features outside their reference behavior? | Directs review toward measurement and process context, not a confirmed root cause |
| SPC status | Is the monitored series consistent with its frozen baseline? | Distinguishes routine variation from a potential out-of-control signal |
| Forecast view | What trajectory is plausible given the recent sequence? | Supports planning and earlier investigation, not deterministic prediction |
| Monitoring verdict | What does the combined rule set recommend? | Routes the user to normal operation, investigation, or retraining evaluation |

The monitoring verdict is intentionally compact: `IN_CONTROL`, `INVESTIGATE_CHAMBER`, or `RETRAIN_RECOMMENDED`. In this project, `INVESTIGATE_CHAMBER` should be read as an operational shorthand for a process or measurement investigation; the dataset itself does not identify a physical chamber.

## Traceability in the interface

Every decision-facing display preserves provenance. At minimum, the dashboard shows:

- The active training run and calibrator run (e.g., `20260914_093559_xgb_sel` + `20260914_093608_cal_platt`).
- The batch or score-run label being viewed (`batch_a_clean`, `batch_b_shift`, `batch_c_dropout`, `holdout_replay`).
- The timestamp of the monitoring execution.
- The configured threshold or control setting used in the displayed summary.
- The artifact path or identifier behind a retraining recommendation.

Traceability turns a dashboard from a visualization exercise into an operational interface. A user who sees an alert can locate the source scorecard or monitoring artifact, reproduce the calculation, and understand whether the evidence comes from model behavior, feature behavior, or both.

## What the dashboard does not do

The dashboard is deliberately not an autonomous control system. It does not send production commands, hold lots, alter recipes, declare a root cause, or deploy a new model. Those actions require process ownership, data lineage checks, and validation beyond what this portfolio dataset can provide.

Its role is narrower and useful: consolidate evidence, make the status legible, and reduce the time needed for a qualified human to decide what to inspect next.

---

**Previous:** [Surrogate DOE](04_surrogate_doe.md)  
**Next:** [Batch scoring and retraining governance](06_mlops_batch_scoring_and_governance.md)
