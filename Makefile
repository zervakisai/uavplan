# UAVBench — Reproducibility Makefile
# Usage: make reproduce
# Requires: Python >=3.10, pip, git

PYTHON      ?= python3
VENV        := .venv
PIP         := $(VENV)/bin/pip
PYTEST      := $(VENV)/bin/python -m pytest
SEEDS       ?= 30
HORIZON     ?= 320
OUTPUT_ROOT ?= results/paper_scientific_validation_full

.PHONY: help venv install lock test lint reproduce clean \
        run-static run-dynamic run-ablation run-validation artifacts

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

test:  ## Run full test suite (245+ tests)
	$(PYTEST) tests/ -q --tb=short

lint:  ## Type-check with mypy
	$(VENV)/bin/mypy src/ tests/ --ignore-missing-imports

# ─── Benchmark Runs ────────────────────────────────────────────

run-static:  ## Run static-track benchmark (control)
	$(VENV)/bin/python -m uavbench.cli.benchmark \
		--track static \
		--planners astar,theta_star,dstar_lite,ad_star,dwa,mppi \
		--trials 1 \
		--paper-protocol \
		--fail-fast

run-dynamic:  ## Run dynamic-track benchmark (stress)
	$(VENV)/bin/python -m uavbench.cli.benchmark \
		--track dynamic \
		--planners astar,theta_star,dstar_lite,ad_star,dwa,mppi \
		--trials 1 \
		--paper-protocol \
		--protocol-variant default \
		--fail-fast

run-ablation:  ## Run ablation variants
	@for variant in no_interactions no_forced_breaks no_guardrail risk_only blocking_only; do \
		echo "=== Ablation: $$variant ==="; \
		$(VENV)/bin/python -m uavbench.cli.benchmark \
			--track dynamic \
			--planners astar,dstar_lite,ad_star,mppi \
			--trials 1 \
			--paper-protocol \
			--protocol-variant $$variant; \
	done

# ─── Paper Artifacts ───────────────────────────────────────────

artifacts:  ## Export publication-ready CSV + LaTeX + manifest
	$(VENV)/bin/python scripts/export_paper_artifacts.py \
		--seeds $(SEEDS) \
		--output-root results/paper

run-validation:  ## Run full best-paper scientific validation pipeline
	MPLCONFIGDIR=/tmp/mpl $(VENV)/bin/python scripts/paper_best_paper_validation.py \
		--seeds $(SEEDS) \
		--episode-horizon $(HORIZON) \
		--output-root $(OUTPUT_ROOT) \
		--strict-fairness

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
