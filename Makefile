# Makefile - SECOM pipeline, one command end to end.
#
#   make            full pipeline: ingest -> extract -> explore -> features
#                   -> train (base + sel) -> calibrate -> spc -> sarimax
#   make demo       serving demo: simulate 3 lots -> score -> reconciled
#                   holdout replay -> scorecards
#   make doe        surrogate DOE: design -> run -> analyze
#                   (primary p_cal OLS + secondary grouped Bernoulli GLM)
#   make train      both training runs only
#   make <stage>    single stage: ingest, extract, explore, features,
#                   train-base, train-sel, calibrate, spc, sarimax
#   make test       pytest
#   make hygiene    repo policy checks (no tracked .db, no stray data reads)
#
# Console scripts come from pyproject.toml; the two data-layer stages use
# python -m until semcon-ingest / semcon-extract entry points land (Phase 5).

# If a script name differs on your machine, fix it once in this block:

UV       := uv run
INGEST   := $(UV) python -m semcon.db_ingest   # -> $(UV) semcon-ingest
EXTRACT  := $(UV) python -m semcon.extract     # -> $(UV) semcon-extract
VALIDATE := $(UV) semcon-validate
EXPLORE  := $(UV) semcon-explore
FEATURE  := $(UV) semcon-features
TRAIN    := $(UV) semcon-train_xgb
CALIB    := $(UV) semcon-calibrate
SPC      := $(UV) semcon-spc
SARIMAX  := $(UV) semcon-sarimax
SIMULATE := $(UV) semcon-simulate
SCORE    := $(UV) semcon-score
SCORECARD := $(UV) semcon-scorecard
DOE_DESIGN := $(UV) semcon-doe-design
DOE_RUN := $(UV) semcon-doe-run
DOE_ANALYZE := $(UV) semcon-doe-analyze
DASH      := $(UV) semcon-dash

# Variables
DOE_LABEL := s060-factorial
DOE_N_FACTORS := 3
DOE_CENTER_POINTS := 3
# Pointer files written by doe_design.py / doe_run.py
DOE_DESIGN_POINTER := artifacts/doe/latest_design
DOE_RUN_POINTER := artifacts/doe/latest_run
# Background window mirrors the SECOM train/calibration zone used for the
# first DOE pass; change here if the scoring background changes.
DOE_BG_START := 2008-07-19 00:00:00
DOE_BG_END := 2008-10-05 05:29:59

# Run-name slugs, matching the post-migration defaults in train_xgb.
# Selection strength (gamma) comes from semcon config, not the CLI - if
# you change it there, the slug convention is your reminder to note it.
BASE_RUN := xgb_base
SEL_RUN  := xgb_sel

.PHONY: all ingest extract explore features train train-base train-sel \
        calibrate spc sarimax doe test hygiene clean demo dash

all: calibrate spc sarimax
	@echo "==> pipeline complete - ledger: artifacts/index.csv"

ingest:
	@echo "==> ingest (raw -> sqlite bronze)"
	$(INGEST)

extract: ingest
	@echo "==> extract (sql -> wide frame, split registration)"
	$(EXTRACT)

validate: extract
	@echo "==> validate "
	$(VALIDATE)

explore: extract
	@echo "==> explore (registry retirement decisions)"
	$(EXPLORE)

features: explore
	@echo "==> feature engineering (register f* columns)"
	$(FEATURE)

train: train-base train-sel

train-base: features
	@echo "==> train baseline (all features, no selection)"
	$(TRAIN) --no-selection --run-name $(BASE_RUN)

train-sel: features
	@echo "==> train with feature selection ($(SEL_RUN))"
	$(TRAIN) --run-name $(SEL_RUN)

calibrate: train-sel
	@echo "==> calibrate latest $(SEL_RUN) run (platt)"
	run=$$(ls artifacts/runs | grep '_$(SEL_RUN)$$' | tail -1); \
	test -n "$$run" || { echo "no $(SEL_RUN) run found in artifacts/runs"; exit 1; }; \
	echo "    parent run: $$run"; \
	$(CALIB) --run-id $$run --method platt

spc: train-sel
	@echo "==> spc monitoring"
	$(SPC)

sarimax: extract
	@echo "==> sarimax experiments"
	$(SARIMAX)

dash:
	@echo "==> launch monitoring dashboard"
	$(DASH)

# Batch windows mirror simulate_lots.py defaults (seed 7, start 2026-01-05);
# the holdout window mirrors the SECOM snapshot zone boundaries. Change together.
demo: calibrate
	@echo "==> simulate 3 incoming lots (append-only, PK-guarded)"
	$(SIMULATE)
	@echo "==> score batches + reconciled holdout replay"
	$(SCORE) --start "2026-01-05 00:00" --end "2026-01-09 03:01" --label batch_a_clean
	$(SCORE) --start "2026-01-12 00:00" --end "2026-01-16 03:01" --label batch_b_shift
	$(SCORE) --start "2026-01-19 00:00" --end "2026-01-23 03:01" --label batch_c_dropout
	$(SCORE) --start "2008-10-05 05:30:59" --end "2008-10-15 19:24:01" --label holdout_replay --reconcile
	@echo "==> scorecards (evaluation panels, registered)"
	$(SCORECARD) --score-run latest --label batch_a_clean
	$(SCORECARD) --score-run latest --label batch_b_shift --reference-label batch_a_clean
	$(SCORECARD) --score-run latest --label batch_c_dropout --reference-label batch_a_clean
	$(SCORECARD) --score-run latest --label holdout_replay

# Surrogate DOE chain. Design/run/analyze stages stay separate on disk; this
# target just wires them through the latest_design / latest_run pointers.
# Primary response is deterministic p_cal; the grouped Bernoulli GLM is a
# simulation-only secondary analysis (see bernoulli_scope.json in the output).
doe: calibrate
	@echo "==> doe design ($(DOE_N_FACTORS) factors, $(DOE_CENTER_POINTS) center points)"
	$(DOE_DESIGN) --run latest --label $(DOE_LABEL) \
		--n-factors $(DOE_N_FACTORS) --center-points $(DOE_CENTER_POINTS)
	@echo "==> doe run over background $(DOE_BG_START) .. $(DOE_BG_END)"
	$(DOE_RUN) --design "$$(cat $(DOE_DESIGN_POINTER))/design.csv" \
		--background-start "$(DOE_BG_START)" --background-end "$(DOE_BG_END)" \
		--label $(DOE_LABEL)
	@echo "==> doe analyze (surrogate OLS + grouped Bernoulli GLM)"
	$(DOE_ANALYZE) --predictions "$$(cat $(DOE_RUN_POINTER))/design_predictions.parquet" \
		--output-dir "$$(cat $(DOE_RUN_POINTER))/analysis"
	@echo "==> doe complete - analysis in $$(cat $(DOE_RUN_POINTER))/analysis"

test:
	$(UV) pytest -q

hygiene:
	@git ls-files | grep '\.db$$' && { echo "FAIL: .db tracked in git"; exit 1; } \
		|| echo "ok: no .db tracked"
	@grep -rn "read_csv\|read_parquet" src --include="*.py" \
		| grep -v -e db_ingest -e migration_test \
		&& { echo "FAIL: stray data-store reads above"; exit 1; } \
		|| echo "ok: data reads contained to ingest + migration test"

clean:
	@echo "==> removing derived database, snapshots, artifacts, logs, and Python caches"
	rm -f data/secom.db data/secom.db-journal data/secom.db-wal data/secom.db-shm
	rm -rf data/snapshots
	rm -f data/column_registry.csv
	rm -rf artifacts/runs
	rm -f artifacts/index.csv artifacts/index_monitor.csv artifacts/diagnostic.parquet
	rm -rf artifacts/eda_*
	rm -f logs/*
	rm -rf artifacts/scores
	rm -rf data/sim
	rm -rf artifacts/doe
	find src tests -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	@mkdir -p data/snapshots artifacts/runs logs
	@touch logs/.gitignore
	@echo "==> clean complete; raw data and .venv preserved"