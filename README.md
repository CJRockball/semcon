# Semcon — Semiconductor Yield-Risk Decision Support

[![CI](https://github.com/CJRockball/semcon/actions/workflows/ci.yml/badge.svg)](https://github.com/CJRockball/semcon/actions/workflows/ci.yml)
[![End-to-end smoke test](https://github.com/CJRockball/semcon/actions/workflows/e2e.yml/badge.svg)](https://github.com/CJRockball/semcon/actions/workflows/e2e.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Ruff](https://img.shields.io/badge/code%20style-Ruff-D7FF64?logo=ruff&logoColor=261230)](https://docs.astral.sh/ruff/)
[![License](https://img.shields.io/github/license/CJRockball/semcon)](LICENSE)
[![Latest commit](https://img.shields.io/github/last-commit/CJRockball/semcon)](https://github.com/CJRockball/semcon/commits/main)

A reproducible semiconductor-manufacturing analytics project built around the UCI SECOM dataset. Semcon converts high-dimensional, missingness-heavy sensor data into calibrated wafer-risk estimates, ranked inspection priorities, process/model-health monitoring, and governed retraining recommendations.

Semcon is deliberately framed as **decision support**. It demonstrates the analytics, validation, workflow, and software interfaces that support yield and metrology decisions; it does not autonomously change tool recipes, hold or release product, or replace engineering approval.

## Dashboard

The Dash/Plotly application makes the model run, measurement-priority workflow, and monitoring state accessible to yield and process engineers without requiring them to inspect raw artifacts.

![Semcon operational dashboard overview](assets/screenshots/dash_over2.png)

The project also includes calibration/failure analysis, individual-and-moving-range process monitoring, DOE analysis, and monitoring-trigger views:

| Calibration and failure review | I–MR process monitoring |
|---|---|
| ![Calibration and failure dashboard](assets/screenshots/dash_cal_fail2.png) | ![I-MR dashboard](assets/screenshots/dash_imr2.png) |

## What it does

| Operational question | Semcon component | Output |
|---|---|---|
| Is incoming manufacturing data structurally valid? | Ingestion, schema checks, SQLite snapshots | Validated data contract and reproducible snapshot |
| Which wafers merit additional measurement? | Calibrated XGBoost scoring | Per-wafer probability and inspection-priority flag |
| Which lots should be reviewed first? | Scorecard aggregation | Ranked lot and wafer worklist |
| Is process/model behavior still healthy? | SPC, SARIMAX, and monitoring | `IN_CONTROL`, `INVESTIGATE_CHAMBER`, or `RETRAIN_RECOMMENDED` |
| Is retraining justified? | Retrain policy | Governed recommendation rather than automatic model replacement |
| What experiments should be run next? | DOE design, simulation, and analysis | Auditable screening and response-surface workflow |

## System flow

```text
Raw SECOM data
   │
   ▼
Ingest → profile → feature engineering → validated SQLite snapshot
   │
   ▼
Leakage-controlled training → calibration → evaluation → run registry
   │
   ▼
New/replayed lot features → semcon-score → calibrated wafer-risk scores
   │                                           │
   ▼                                           ▼
semcon-scorecard → ranked inspection worklist  semcon-monitor → health verdict
                                                     │
                                                     ▼
                                           semcon-retrain → policy recommendation
                                                     │
                                                     ▼
                                              Dash operational dashboard
```

The scoring workflow is batch-oriented rather than an always-on prediction API: a scheduled job can invoke `semcon-score` when a lot is ready, write score artifacts, and exit. `semcon-scorecard` then turns those scores into a measurement-priority worklist. Dash is a separate, long-running visualization and audit service.

## Results and validation

The pipeline emphasizes imbalanced-class evaluation and decision quality rather than accuracy alone. The repository records baseline-versus-selected model comparison, holdout evaluation, probability calibration, threshold selection, and explainability artifacts in the run registry and validation documentation.

- Start with [validation.md](validation.md) for split discipline, leakage controls, metrics, calibration, and limitations.
- `artifacts/index.csv` is the model-run ledger.
- `artifacts/runs/<run_id>/` holds immutable run-specific metrics, figures, models, and metadata.
- PR-AUC is prioritized because the defective outcome is rare; ROC-AUC is reported as a secondary discrimination measure.

## Quick start

### Requirements

Install Python 3.11+ and [uv](https://docs.astral.sh/uv/). The project uses `uv.lock` to reproduce the pinned dependency environment. `uv sync --frozen` creates a local `.venv` in the cloned repository; the clone itself does not include or activate a virtual environment.

```bash
git clone https://github.com/CJRockball/semcon.git
cd semcon
uv sync --frozen
```

Open the cloned `semcon` directory itself as the VS Code workspace. VS Code usually detects `.venv` after `uv sync`; otherwise select `.venv/bin/python` on macOS/Linux or `.venv\Scripts\python.exe` on Windows.

### Minimum data-enabled verification

The original SECOM source data are not versioned in this repository. Download the required source files into `data/raw/` following [data/README.md](data/README.md), then build the local data layer and run the full suite:

```bash
# Create the local raw SQLite layer.
uv run semcon-ingest

# Apply data-quality profiling and materialize eligible base features.
uv run semcon-explore

# Add derived missingness features to the analytic matrix.
uv run semcon-feature-eng

# Verify the full repository, including local data-contract tests.
uv run pytest
```

The original SECOM input has 590 measurement columns. Profiling retains 257 usable base features; feature engineering adds four derived missingness features, producing the 261-feature analytic matrix protected by the local data-contract test.

Run linting separately:

```bash
uv run ruff check .
```

The default CI workflow runs linting, the Python test suite, and a small CLI smoke check. A separate end-to-end workflow is manually triggered because it downloads the dataset and exercises the full data-dependent path.

### Full portfolio demonstration

After placing the original SECOM source files in `data/raw/`, rebuild the complete data, modeling, monitoring, scoring, governance, and DOE demonstration with:

```bash
make full
```

`make full` is the aggregate clean-rebuild target. It performs the equivalent of:

```bash
make clean
make
make demo
make trigger
make doe
```

The targets are deliberately separated so individual components can also be reproduced or debugged independently:

| Target | Purpose |
|---|---|
| `make clean` | Removes locally generated database, snapshots, caches, and runtime artifacts before a clean rebuild |
| `make` | Builds the core data, feature, model, calibration, evaluation, SPC, and SARIMAX workflow |
| `make demo` | Replays/scores incoming lots and produces ranked measurement priorities plus monitoring evidence |
| `make trigger` | Applies the governed retrain policy to accumulated monitoring history |
| `make doe` | Produces the DOE design and analysis demonstration |

`make full` regenerates pipeline-owned derived artifacts. It does not recreate the original raw SECOM inputs in `data/raw/`, manually curated documentation, Git-tracked static assets, or Docker images.

### Typical local workflow

Use project CLI help for the current argument contract:

```bash
uv run semcon-ingest --help
uv run semcon-explore --help
uv run semcon-feature-eng --help
uv run semcon-train --help
uv run semcon-score --help
uv run semcon-scorecard --help
uv run semcon-monitor --help
uv run semcon-retrain --help
uv run semcon-dash --help
```

A representative operations sequence after a trained model exists is:

```bash
# Score an incoming or replayed lot, then produce measurement priorities.
uv run semcon-score
uv run semcon-scorecard

# Evaluate monitoring and retrain policy on accumulated artifacts.
uv run semcon-monitor
uv run semcon-retrain

# Start the local dashboard.
uv run semcon-dash
```

## Dashboard Docker deployment

The root `Dockerfile` packages the pinned `uv.lock` environment, source, SQL, documentation assets, and committed run artifacts. Its default command launches **only the Dash dashboard** on port 8050; it does not start scoring, scorecard generation, or retraining jobs. The `.dockerignore` excludes local environments, Git metadata, raw input data, and build/test caches from the image build context.

Build and run the dashboard from the repository root:

```bash
docker build -t semcon:latest .
docker run --rm -p 8050:8050 --name semcon-dash semcon:latest
```

Then browse to [http://localhost:8050](http://localhost:8050).

For a batch-oriented deployment, use the same image as a short-lived scheduled workload and mount host storage for incoming data and output artifacts. Do not combine the batch job with the Dash container unless there is a deliberate orchestration reason to do so:

```bash
# Illustrative only: inspect the installed CLI for current scoring flags.
docker run --rm \
  -v "$(pwd)/artifacts:/app/artifacts" \
  semcon:latest \
  uv run semcon-score --help
```

A scheduler such as cron, Airflow, or a Kubernetes CronJob should invoke scoring only after an input lot has been validated and made available atomically. It should invoke scorecard generation after successful scoring. A processed-file archive, database status, or immutable lot identifier is needed to avoid repeatedly scoring the same lot.

## Monitoring and governance

`semcon-monitor` uses the scored stream and frozen reference boundaries to emit an operational state:

| Verdict | Meaning | Human response |
|---|---|---|
| `IN_CONTROL` | No material output-risk or key-feature drift alarm | Continue normal surveillance |
| `INVESTIGATE_CHAMBER` | Feature/process shift without a corresponding sustained risk shift | Review equipment, recipe, and metrology context |
| `RETRAIN_RECOMMENDED` | Persistent model-output degradation and/or policy conditions | Validate data lineage and holdout performance before approving a new model |

`semcon-retrain` is intentionally downstream of monitoring. It converts accumulated evidence into a governed recommendation; it does not silently overwrite the active model. This separation makes decisions auditable and prevents a transient signal from directly changing production analytics.

![Monitoring trigger view](assets/screenshots/monitor_trigger2.png)

## Repository map

```text
src/semcon/
├── db.py, db_ingest.py, extract.py, schema.py, snapshots.py, validate.py
│   └── data acquisition, validation, SQLite snapshots, and data contracts
├── explore.py, feature_eng.py
│   └── profiling, feature eligibility, and derived analytic features
├── train_xgb.py, selection.py, calibrate.py, evaluation.py, explain.py
│   └── training, selection, calibration, validation, and explainability
├── simulate_lots.py, score.py, scorecard.py
│   └── replayed/incoming batch handling, scoring, and inspection prioritization
├── spc.py, sarimax.py, monitor.py, retrain_trigger.py
│   └── process/model surveillance and retrain policy
├── doe_design.py, doe_run.py, doe_analyze.py
│   └── experimental design, response simulation, and analysis
└── dash_app.py, dash_data.py, dash_figs.py
    └── Dash/Plotly operational visualization

tests/                  pytest suite
docs/                   design and analysis documentation
assets/                 README/dashboard visual assets
artifacts/              committed run registry and demo outputs
.github/workflows/      CI and manually triggered end-to-end workflows
```

## Data and limitations

SECOM is a public benchmark dataset, not a complete fab data model. It lacks true lot genealogy, chamber/tool identifiers, recipes, maintenance history, wafer maps, causal interventions, and real metrology/MES integrations. Accordingly:

- Lot replay and injected drift scenarios demonstrate streaming interfaces and monitoring behavior; they are not claims of a live factory deployment.
- DOE modules provide an explicit experimental-analysis workflow, but synthetic response components are labeled as such and must not be interpreted as observed fab experiments.
- Risk scores are inspection-prioritization inputs, not autonomous disposition, recipe-control, or safety decisions.
- Any production implementation would require validated data lineage, access control, model approval/versioning, integration testing against MES/FDC/metrology systems, and engineering sign-off.

## Further documentation

- [Validation and modeling protocol](validation.md)
- [DOE workflow](docs/doe.md)
- [Data setup and acquisition](data/README.md)
- [GitHub Actions CI workflow](.github/workflows/ci.yml)
- [Manually triggered end-to-end workflow](.github/workflows/e2e.yml)

## License

See [LICENSE](LICENSE).
