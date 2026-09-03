# Makefile - SECOM pipeline, one command end to end.
#
#   make            full pipeline: ingest -> extract -> explore -> features
#                   -> train (base + sel) -> calibrate -> spc -> sarimax
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

# Run-name slugs, matching the post-migration defaults in train_xgb.
# Selection strength (gamma) comes from semcon config, not the CLI - if
# you change it there, the slug convention is your reminder to note it.
BASE_RUN := xgb_base
SEL_RUN  := xgb_sel

.PHONY: all ingest extract explore features train train-base train-sel \
        calibrate spc sarimax test hygiene clean

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

test:
	$(UV) pytest -q

hygiene:
	@git ls-files | grep '\.db$$' && { echo "FAIL: .db tracked in git"; exit 1; } \
		|| echo "ok: no .db tracked"
	@grep -rnE "read_(csv|parquet).*(DATA_RAW|DATA_PROCESSED|secom\.|dfX_v|dfy_v)" \
		src --include="*.py" \
		| grep -v -e db_ingest -e migration_test \
		&& { echo "FAIL: stray data-store reads above"; exit 1; } \
		|| echo "ok: data store has one reader (+ frozen migration test)"
	@$(UV) ruff check src || echo "note: ruff findings above"

clean:
	@echo "==> removing derived database, snapshots, artifacts, logs, and Python caches"
	rm -f data/secom.db data/secom.db-journal data/secom.db-wal data/secom.db-shm
	rm -rf data/snapshots
	rm -f data/column_registry.csv
	rm -rf artifacts/runs
	rm -f artifacts/index.csv artifacts/index_monitor.csv artifacts/diagnostic.parquet
	rm -rf artifacts/eda_*
	rm -f logs/*
	find src tests -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type f -name "*.py[co]" -delete
	@mkdir -p data/snapshots artifacts/runs logs
	@touch logs/.gitignore
	@echo "==> clean complete; raw data and .venv preserved"