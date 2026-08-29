# Makefile - SECOM pipeline, one command end to end.
#
#   make            full pipeline: data -> features -> train (base + sel) -> calibrate -> spc -> sarimax
#   make train      both training runs only
#   make <stage>    single stage: data, features, train-base, train-sel, calibrate, spc, sarimax
#   make test       pytest
#
# Everything runs through the uv console scripts from pyproject.toml.
# If a script name differs on your machine, fix it once in this block:

UV      := uv run
EXPLORE := $(UV) semcon-explore
FEATURE := $(UV) semcon-features
TRAIN   := $(UV) semcon-train_xgb
CALIB   := $(UV) semcon-calibrate
SPC     := $(UV) semcon-spc
SARIMAX := $(UV) semcon-sarimax

# Run-name slug for the feature-selection run. The selection strength
# (gamma = 0.25) comes from the semcon config, not the CLI - if you change
# it there, rename this slug to match.
SEL_RUN := sel_g025

.PHONY: all data features train train-base train-sel calibrate spc sarimax test

all: data features train calibrate spc sarimax
	@echo "==> pipeline complete - ledger: artifacts/index.csv"

data:
	@echo "==> explore (raw -> processed)"
	$(EXPLORE)

features: data
	@echo "==> feature engineering"
	$(FEATURE)

train: train-base train-sel

train-base: features
	@echo "==> train baseline (all features, no selection)"
	$(TRAIN) --no-selection --run-name baseline

train-sel: features
	@echo "==> train with feature selection ($(SEL_RUN))"
	$(TRAIN) --run-name $(SEL_RUN)

calibrate: train-sel
	@echo "==> calibrate latest $(SEL_RUN) run (platt)"
	run=$$(ls artifacts/runs | grep '_$(SEL_RUN)$$' | tail -1); \
	echo "    parent run: $$run"; \
	$(CALIB) --run-id $$run --method platt

spc: train-sel
	@echo "==> spc monitoring"
	$(SPC)

sarimax: data
	@echo "==> sarimax experiments"
	$(SARIMAX)

test:
	$(UV) pytest -q
