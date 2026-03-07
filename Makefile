# UAVBench — Reproducibility Makefile
# Usage: make reproduce
# Requires: Python >=3.10, pip, git

PYTHON      ?= python3
VENV        := .venv
PIP         := $(VENV)/bin/pip
PYTEST      := $(VENV)/bin/python -m pytest
SEEDS       ?= 30
OUTPUT_ROOT ?= results/paper_validation

.PHONY: help venv install lock test lint reproduce clean \
        run-single run-all run-validation artifacts

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ─── Environment ───────────────────────────────────────────────

venv:  ## Create virtual environment
	$(PYTHON) -m venv $(VENV)

install: venv  ## Install all dependencies (locked)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements-lock.txt
	$(PIP) install -e ".[all]"

lock:  ## Regenerate requirements-lock.txt from current env
	$(PIP) freeze | grep -v '^-e ' > requirements-lock.txt
	@echo "Lock file written: requirements-lock.txt"

# ─── Quality ───────────────────────────────────────────────────

test:  ## Run full test suite
	$(PYTEST) tests/ -q --tb=short

lint:  ## Type-check with mypy
	$(VENV)/bin/mypy src/ tests/ --ignore-missing-imports

# ─── Benchmark Runs ────────────────────────────────────────────

run-single:  ## Run single-scenario benchmark (quick check)
	$(VENV)/bin/python -m uavbench run \
		--scenarios osm_penteli_fire_delivery_medium \
		--planners astar,periodic_replan,aggressive_replan,dstar_lite,apf \
		--trials 1 \
		--seed-base 42

run-all:  ## Run all scenarios × all planners (1 trial)
	$(VENV)/bin/python -m uavbench run \
		--scenarios osm_penteli_fire_delivery_medium,osm_piraeus_flood_rescue_medium,osm_downtown_fire_surveillance_medium \
		--planners astar,periodic_replan,aggressive_replan,dstar_lite,apf \
		--trials 1 \
		--seed-base 42

# ─── Paper Artifacts ───────────────────────────────────────────

artifacts:  ## Export publication-ready CSV + LaTeX + manifest
	$(VENV)/bin/python scripts/export_artifacts.py

run-validation:  ## Run full paper validation pipeline (all scenarios x planners x seeds)
	$(VENV)/bin/python scripts/run_paper_experiments.py \
		--seeds $(SEEDS) \
		--output $(OUTPUT_ROOT)/all_episodes.csv

# ─── One-Shot Reproduce ────────────────────────────────────────

reproduce: install test run-validation  ## Full reproducibility: install → test → validate
	@echo ""
	@echo "============================================"
	@echo " Reproduction complete."
	@echo " Tests:    $$($(PYTEST) tests/ -q --tb=no 2>&1 | tail -1)"
	@echo " Outputs:  $(OUTPUT_ROOT)/"
	@echo " Manifest: $(OUTPUT_ROOT)/validation_manifest.json"
	@echo "============================================"

# ─── Housekeeping ──────────────────────────────────────────────

clean:  ## Remove generated artifacts and caches
	rm -rf results/ outputs/ __pycache__ .pytest_cache .mypy_cache
	find . -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
