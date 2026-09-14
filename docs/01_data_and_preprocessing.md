# Data and preprocessing

[Home](../README.md) · [Next: Modelling and results](02_modelling_and_results.md)

| Item | Summary |
|---|---|
| Objective | Convert the SECOM source files into a governed, time-aware modelling dataset without treating missingness or chronology as afterthoughts. |
| Main artifacts | SQLite store, extraction outputs, column registry, snapshots, EDA artifacts, and the frozen feature contract for each training run. |
| Key claim | The pipeline makes data-quality, feature-retirement, and split decisions explicit and reproducible. |
| Boundary | SECOM does not provide real fab context such as chamber, recipe, product, maintenance, genealogy, or physical sensor units. |

## Dataset and decision context

This project uses the UCI SECOM dataset as a proxy for high-dimensional semiconductor manufacturing data. Each row represents an observation with a binary quality outcome and a wide set of sensor-derived measurements. The project frames the task as **yield-risk decision support**: produce a calibrated, auditable risk signal that can help focus inspection or engineering attention where it is most valuable.

That framing matters. The objective is not to claim that the model can autonomously release wafers, identify a physical root cause, or prescribe a recipe adjustment. It is to turn a difficult tabular dataset into a disciplined workflow for ranking risk, inspecting changes over time, and documenting the limits of the available evidence.

## Storage and extraction boundary

The raw source is ingested into a local SQLite store rather than read repeatedly from ad hoc files throughout the codebase. This creates a clear boundary between source ingestion and downstream analytics:

```text
Raw SECOM files
    → SQLite ingestion
    → validated extraction
    → snapshots and split registration
    → exploration and feature engineering
    → training / scoring / monitoring
```

The boundary is practical as well as architectural. It centralizes schema checks, makes extraction windows explicit, and reduces the risk that later scripts silently consume a differently filtered version of the data. Derived artifacts are written under `data/` and `artifacts/`; a training run records its own configuration and feature contract so later scoring and analysis can resolve the exact model inputs that were used.

## Time structure and split policy

Chronology is treated as part of the data-generating process. The dataset is ordered by its available timestamp field, and the project uses a chronological cutoff to separate the development period from the final holdout period. A later regime with a pronounced missing-data deterioration is excluded according to an EDA-derived boundary rather than blended indiscriminately into model development.

This is more conservative than a random train/test split. In a real manufacturing setting, future lots are not exchangeable with historical lots: tool condition, measurement quality, product mix, maintenance activity, and process drift can change over time. A chronological holdout therefore provides a more credible approximation of the deployment question: how does a model developed on earlier observations behave on later observations?

The configured development setup uses five folds for internal validation and a fixed tail holdout of 231 observations. The exact cutoff and exclusion boundary are versioned in configuration and documented in the validation record. The important principle is not the numerical split alone; it is that the holdout is protected from feature selection, hyperparameter selection, and calibration fitting until final evaluation.

## Missingness as a process signal

Missing values are not handled as a purely mechanical preprocessing nuisance. In manufacturing data, missingness may arise from sensor availability, logging behavior, equipment state, data-collection changes, or structured measurement failures. A missing reading can therefore carry information even when its numeric value is unavailable.

The preprocessing workflow distinguishes several cases:

- Columns with excessive missingness are candidates for retirement because their observed support is too limited.
- Near-zero-variance and low-information columns are screened so they do not create the illusion of model complexity without useful discrimination.
- Highly correlated features are treated as redundant candidates rather than retained automatically.
- Certain missingness patterns are retained as engineered indicators when supported by exploratory evidence.

The feature-engineering stage creates data-quality features, including selected missingness-cluster indicators. These features are legitimate predictive inputs because they preserve information about the observation process. They are not, however, controllable physical process knobs. That distinction is enforced later in the surrogate DOE workflow, where engineered missingness and clique indicators are excluded from candidate factor selection.

## NaN clusters and engineered features

A single missing-value percentage can hide structured behavior. The project therefore examines groups of sensors that tend to be missing together. Such co-missingness can indicate a shared acquisition path, a measurement block, or a regime change in the source data. When a cluster is sufficiently stable and relevant, it can be represented as an engineered feature rather than leaving the model to infer the pattern indirectly from many sparse columns.

This approach has two benefits. First, it creates a more interpretable representation of data quality: an engineer can see that a group of measurements is absent together rather than only seeing a model response to imputed values. Second, it lets the modelling layer distinguish a low reading from an unavailable reading. The price is that these features must be handled carefully: they may be predictive without being causal, and they cannot be interpreted as recipe settings.

## Feature-screening policy

Feature screening occurs before the final model contract is frozen. The policy combines data-quality and redundancy considerations rather than relying on feature importance alone. The configured rules include thresholds for missingness, dominant values, coefficient of variation, low cardinality, and pairwise correlation. The precise thresholds live in `src/semcon/config.py`; the resulting column decisions are recorded in the registry and EDA artifacts.

The final feature set is deliberately a contract. Training writes the selected feature list into the run artifacts. Batch scoring, calibration, surrogate DOE, and monitoring resolve that contract rather than reconstructing columns by convention. This prevents a common operational failure mode in tabular ML: a model is trained with one feature set but scored later with an accidental variant.

## Reproducibility and limits

The pipeline uses explicit configuration, deterministic seeds where stochastic behavior is involved, timestamped run directories, and append-only indexes for experiment tracking. These controls make the project inspectable, but they do not remove the limits of the source data.

SECOM does not include the metadata that would be required for a production-grade fab deployment: chamber and tool identity, recipe version, lot genealogy, wafer-map structure, maintenance history, metrology context, and physical engineering units. Consequently, the project does not claim chamber-level root-cause attribution, causal sensor effects, or operational recipe optimization. It demonstrates how such an analytics system can be structured when the available data is wide, sparse, imbalanced, and time-dependent.

---

**Next:** [Modelling and results](02_modelling_and_results.md)