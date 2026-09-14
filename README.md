# Semiconductor Yield-Risk Decision Support

A reproducible, time-aware machine-learning and process-analytics workflow built on the UCI SECOM dataset. The project turns high-dimensional, sparse manufacturing measurements into calibrated yield-risk rankings, batch scorecards, statistical monitoring, surrogate DOE evidence, and governed retraining recommendations.

> **Scope:** This is a portfolio implementation using historical SECOM data. It is not connected to a fab MES, FDC, APC, or quality-disposition system. It does not autonomously hold lots, change recipes, diagnose a physical root cause, or deploy a replacement model.

## What it does

```text
Raw SECOM files
  → SQLite ingestion and validated extraction
  → time-aware preprocessing and feature engineering
  → XGBoost training, calibration, evaluation, and explanation
  → batch scoring and inspection-priority scorecards
  → SPC / forecasting / model-health monitoring
  → Dash decision-support interface
  → governed retraining recommendation
```

The system is designed around an operational question: given constrained inspection and engineering capacity, which observations or replayed lots deserve attention, and is the evidence consistent with routine variation, a process/data investigation, or a model-review event?

## Repository layout

```text
.
├── src/semcon/       # Pipeline, models, scoring, monitoring, dashboard, DOE
├── tests/            # Unit and integration-style contract tests
├── docs/             # Technical documentation by lifecycle stage
├── notebooks/        # Exploratory analysis notebooks
├── config/           # Versioned runtime configuration
├── data/             # Local source, SQLite, snapshots; raw data is not tracked
├── artifacts/        # Registered derived runs and decision artifacts
├── validation.md     # Validation protocol and canonical run evidence
├── doe.md            # Detailed DOE implementation notes
└── Makefile          # Reproducible pipeline and hygiene commands
```

The Python package separates responsibilities rather than hiding the workflow in one notebook. Data ingestion and extraction, feature engineering, training, calibration, scoring, SPC, forecasting, DOE, monitoring, retraining policy, Dash presentation, and artifact tracking are implemented as explicit modules under `src/semcon/`.

## Quick start

### 1. Install

This repository uses [uv](https://docs.astral.sh/uv/) for dependency and environment management.

```bash
git clone https://github.com/CJRockball/semcon.git
cd semcon
uv sync
```

### 2. Obtain the data

Download the SECOM source data from the UCI Machine Learning Repository and place the required files in the local data location expected by the ingestion command. The raw dataset and the generated SQLite database are deliberately excluded from Git.

### 3. Run checks

```bash
make test
make hygiene
```

`make test` runs the repository test suite. `make hygiene` checks the repository policy that generated databases are not tracked and that direct file reads remain contained to the intended data-layer boundary.

### 4. Run the pipeline

```bash
make
```

The default pipeline performs ingestion, extraction, exploration, feature engineering, baseline and selected-feature model training, calibration, SPC, and SARIMAX stages according to the Makefile dependencies. Training and analysis runs write versioned artifacts under `artifacts/`.

### 5. Exercise the batch workflow

```bash
make demo
```

The demo path replays deterministic incoming-lot scenarios, scores the resulting batches, runs a reconciled holdout replay, and creates scorecards. These scenarios are test and demonstration fixtures—not evidence of a live manufacturing feed.

## Documentation

The documentation is split by the questions a reader is likely to ask when reviewing an applied ML system. Read it in sequence for the complete design narrative, or jump to the topic most relevant to your role.

| Guide | Focus |
|---|---|
| [Data and preprocessing](docs/01_data_and_preprocessing.md) | Dataset structure, time boundary, missingness clusters, feature policy, and leakage controls |
| [Modelling and results](docs/02_modelling_and_results.md) | Decision objective, model contract, temporal validation, calibration, and interpretation of results |
| [Process monitoring and forecasting](docs/03_process_monitoring_and_forecasting.md) | Phase-I/Phase-II SPC, alert interpretation, and SARIMAX forecasting |
| [Surrogate DOE](docs/04_surrogate_doe.md) | Factor eligibility, observed-support design levels, OOD checks, interaction analysis, and causal limits |
| [Dashboard and decision support](docs/05_dashboard_and_decision_support.md) | Dash interface, scorecards, operational questions, and traceability |
| [Batch scoring and retraining governance](docs/06_mlops_batch_scoring_and_governance.md) | Batch contract, scenario replay, monitoring states, artifact lineage, retraining policy, tests, and CI |
| [Validation record](validation.md) | Evaluation protocol, canonical run evidence, and methodology decisions |
| [DOE implementation notes](doe.md) | Command-level and artifact-level details for the surrogate DOE workflow |

## Operational workflow

The project distinguishes prediction, monitoring, and retraining because they answer different questions.

```text
Incoming or replayed batch
  → schema and feature-contract validation
  → calibrated risk scoring
  → scorecard and inspection-priority ranking
  → feature-drift and output-health monitoring
  → policy evaluation

IN_CONTROL
  → continue normal scoring and surveillance

INVESTIGATE_CHAMBER
  → review measurement/process context and data lineage

RETRAIN_RECOMMENDED
  → start governed model review; do not automatically deploy a replacement
```

`INVESTIGATE_CHAMBER` is an operational routing label, not a claim that the system has identified a physical chamber fault. SECOM does not include the tool, chamber, recipe, maintenance, or genealogy metadata needed to make that attribution.

## Technical choices

| Area | Implementation | Why it matters |
|---|---|---|
| Data boundary | SQLite ingestion, validated extraction, snapshots, column registry | Avoids uncontrolled file reads and makes transformations inspectable |
| Temporal validation | Development period plus protected chronological holdout | Better reflects the future-data question than an unrestricted random split |
| Predictive model | XGBoost binary classifier with feature contract | Handles nonlinear, sparse, high-dimensional tabular structure |
| Calibration | Registered probability calibrator | Supports thresholded triage and probability-based monitoring |
| Explainability | TreeSHAP summaries | Identifies model-relevant signals without claiming physical causality |
| Process surveillance | SPC and SARIMAX | Separates change detection from near-term temporal forecasting |
| Experimentation | Surrogate factorial DOE with OOD checks | Generates bounded hypotheses from the model within observed support |
| Operations | Batch scoring, scorecards, monitoring, retraining policy, Dash | Demonstrates an auditable lifecycle beyond offline model fitting |

## What this project does not claim

The project is deliberately explicit about the boundary between a rigorous portfolio workflow and a production fab system.

- **Predictive association is not causal process knowledge.** A feature with high model importance or SHAP contribution is relevant to the fitted model; it is not proof of a root cause.
- **Surrogate DOE is not a physical DOE.** Factor settings use observed data support and model predictions; recipe changes require controlled, safe, physical confirmation.
- **Monitoring is not autonomous control.** Alerts route a human investigation and do not release, hold, or alter product or equipment.
- **A retraining recommendation is not deployment.** Data lineage, label maturity, validation, calibration, incumbent comparison, and approval are required before promotion.
- **SECOM is not a complete fab data model.** It lacks chamber/tool identity, recipe, layer, product, lot genealogy, maintenance events, wafer maps, and physical sensor units.

These constraints are not hidden limitations; they determine the design. The project demonstrates how to build a technically defensible analytics workflow when data is wide, sparse, imbalanced, and time-dependent, while avoiding claims that the available evidence cannot support.

## Reproducibility

Runs are registered with configuration and artifact metadata so that a displayed score, monitoring decision, or DOE result can be traced back through its upstream model and data boundary. The intended lineage is:

```text
Dashboard or decision artifact
  → monitoring report or scorecard
  → scored batch
  → calibrator and training run
  → feature contract and configuration
  → extraction and snapshot boundary
```

The source package, Makefile, test suite, and GitHub Actions workflow provide the operational backbone. Exact CLI entry points and stage dependencies are defined in `pyproject.toml` and the Makefile; check those files as the source of truth when adapting the workflow.

## License and data attribution

Code in this repository is released under the MIT License. The SECOM dataset is external to this repository and remains subject to the terms and attribution requirements of its original source.