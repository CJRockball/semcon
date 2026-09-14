# Process monitoring and forecasting

[Home](../README.md) · [Previous: Modelling and results](02_modelling_and_results.md) · [Next: Surrogate DOE](04_surrogate_doe.md)

| Item | Summary |
|---|---|
| Objective | Detect departures from a reference process and provide time-series forecasts that support earlier investigation. |
| Main artifacts | SPC screening outputs, frozen reference limits, alert records, SARIMAX experiment artifacts, and dashboard figures. |
| Key claim | Monitoring distinguishes evidence of a process or data shift from a stable operating state; it does not establish a physical root cause. |
| Boundary | The dataset has limited tool and lot context, so alerts cannot be attributed directly to a chamber, recipe, maintenance event, or product family. |

## Why risk scoring is not enough

A calibrated risk model describes the estimated risk of the observations currently being scored. It does not, by itself, tell an engineer whether the broader process has changed, whether a measurement channel is behaving unusually, or whether a recent sequence is consistent with baseline variation. Those questions require separate process-monitoring logic.

This project treats statistical process control and forecasting as complementary surveillance tools. SPC asks whether a monitored series has departed from a frozen reference distribution. SARIMAX asks whether the observed historical time pattern supports a forecast of the next period. Neither tool diagnoses mechanism. Together, they help separate normal variation from signals that warrant a closer look.

## Reference baseline and surveillance

The workflow distinguishes between two conceptual phases:

- **Phase I** establishes a reference baseline from the chosen historical window. Robust location and dispersion statistics, screening outputs, and control boundaries are stored as artifacts.
- **Phase II** evaluates later or incoming observations against those frozen references. The point is to detect change relative to the baseline, not to redefine normal behavior every time a batch arrives.

The use of frozen reference limits prevents a monitoring system from normalizing away sustained drift by continually recalculating its own definition of normal. Where robust statistics are appropriate, median and median absolute deviation style calculations reduce sensitivity to isolated extreme values in sparse sensor data.

## SPC signals and interpretation

The SPC workflow screens eligible signals and evaluates out-of-control evidence using control boundaries and rule logic. A single point beyond a limit may be meaningful, but practical monitoring should also consider sequences, persistence, and the breadth of affected variables. The value of an alert depends on whether it is isolated, repeated, or accompanied by a change in the scored risk distribution.

| Signal pattern | What it suggests | Appropriate response |
|---|---|---|
| Stable input features and stable risk distribution | No evidence of a material monitored change | Continue normal scoring and routine surveillance |
| One or more drifting sensor features with stable output risk | Measurement, tool, chamber, or data-quality investigation may be needed | Review feature-level evidence and available operational context |
| Persistent rise in calibrated risk or triage rate | The scored population, process, data pipeline, or model assumptions may have changed | Increase review intensity and evaluate model-health evidence |
| Widespread feature drift plus output shift | A significant operating-regime change is plausible | Investigate process/data lineage and consider governed retraining evaluation |

The labels are deliberately cautious. In this dataset, a feature alert cannot prove a chamber issue because chamber identity is not available. The project can say “investigate the process or measurement context”; it cannot honestly say “chamber X caused the excursion.”

## SARIMAX forecasting

SARIMAX is included as a separate time-series analysis layer. It models a selected temporal series with autoregressive, moving-average, differencing, and optional exogenous structure, then produces forecasts and diagnostic artifacts. The role is not to forecast individual wafer outcomes. It is to forecast an aggregated or monitored process-related series and provide uncertainty-aware evidence about its expected near-term trajectory.

A useful operational interpretation is:

- SPC is primarily a **detection** tool: has the process or monitored series departed from historical control?
- SARIMAX is a **forecasting** tool: given the observed sequence, what is the plausible next trajectory and uncertainty interval?

The two methods should not be forced into a single score. An observed excursion can be important even if it was difficult to forecast; a forecast can indicate a possible deterioration without proving that the process is currently out of control.

## From alert to action

The monitoring design connects statistical evidence to a small set of human decisions. A nominal state continues the normal scoring process. A localized input drift prompts engineering review of the relevant measurement or process context. A persistent output-health shift triggers closer review of data lineage, model performance, and retraining eligibility.

This separation matters because “retrain” is not the default answer to every alert. If a sensor has changed because of a logging problem, retraining on corrupted data would make the model worse. If a process has shifted but labels have not yet matured, model performance cannot yet be evaluated honestly. A retraining decision therefore belongs to a governed policy layer, documented in the MLOps section, rather than inside the SPC calculation itself.

## Limits of surveillance

SPC and SARIMAX are valuable because they make deviations visible and repeatable. They are not substitutes for manufacturing context. The SECOM dataset does not identify product, process layer, chamber, tool, recipe, maintenance state, or wafer genealogy. The analysis therefore cannot separate common-cause from assignable-cause variation with the same confidence as a production FDC or MES-integrated system.

The appropriate claim is narrower and more useful: the project shows how a time-aware data science workflow can create auditable change signals, connect them to scored-risk behavior, and route them to the right kind of investigation.

---

**Previous:** [Modelling and results](02_modelling_and_results.md)  
**Next:** [Surrogate DOE](04_surrogate_doe.md)